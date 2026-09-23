"""RobotOps AI —— Phase 4 LangGraph 多 Agent 工作流。

流程图（与需求一致）：

```text
START
  → Data Analysis Agent（Phase 1 Pandas：指标 + 异常）
  → Abnormal Check（程序判定，不调用大模型）
      ├─ 无异常 → Report Agent → END
      └─ 有异常 → Diagnosis Agent → RAG Agent → Recommendation Agent → Report Agent → END
```

异常分支由程序结果驱动（``state["has_anomalies"]`` 来自 Phase 1 的异常识别），
大模型不参与"是否异常"的判断。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

try:  # pragma: no cover - 依赖缺失时给出明确安装提示
    from langgraph.graph import END, START, StateGraph
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "Phase 4 多 Agent 工作流需要 langgraph，请执行 python -m pip install -r requirements.txt 安装依赖"
    ) from error

from robotops import config as project_config
from robotops.llm.client import ChatResult, DeepSeekClient
from robotops.llm.config import LLMConfig
from robotops.llm.logger import setup_llm_logger
from robotops.rag import KnowledgeBaseRetriever, RagConfig

from ..agents.base import has_llm_failure
from ..agents.data_analysis import DataAnalysisAgent
from ..agents.diagnosis import DiagnosisAgent
from ..agents.rag_agent import RagAgent
from ..agents.recommendation import RecommendationAgent
from ..agents.report import ReportAgent
from .logging_utils import WorkflowLogger
from .state import (
    WorkflowState,
    create_initial_state,
    serializable_state,
    state_overview,
)

# 节点名称
NODE_DATA_ANALYSIS = "data_analysis"
NODE_ABNORMAL_CHECK = "abnormal_check"
NODE_DIAGNOSIS = "diagnosis"
NODE_RAG = "rag"
NODE_RECOMMENDATION = "recommendation"
NODE_REPORT = "report"

ABNORMAL_CHECK_TAG = "Abnormal Check"
WORKFLOW_TAG = "Workflow"


class RobotOpsWorkflow:
    """机器人运营多 Agent 工作流（LangGraph 编排）。"""

    def __init__(
        self,
        *,
        llm_config: LLMConfig | None = None,
        client: DeepSeekClient | None = None,
        retriever: KnowledgeBaseRetriever | None = None,
        rag_config: RagConfig | None = None,
        logger: WorkflowLogger | None = None,
        verbose: bool = True,
        echo: bool = True,
        log_file: str | Path | None = None,
        llm_log_level: str = "WARNING",
    ) -> None:
        self.logger = logger or WorkflowLogger(verbose=verbose, echo=echo, log_file=log_file)
        self.llm_config = llm_config
        self._client = client
        self._retriever = retriever
        self._rag_config = rag_config
        self.llm_log_level = llm_log_level
        self._compiled: Any = None
        self._agents_ready = False

    # -- 依赖 -------------------------------------------------------------
    @property
    def client(self) -> DeepSeekClient:
        """惰性创建 DeepSeek 客户端（复用 Phase 2 实现：超时/重试/错误映射）。"""

        if self._client is None:
            config = self.llm_config or LLMConfig.from_env()
            self._client = DeepSeekClient(
                config=config,
                logger=setup_llm_logger(level=self.llm_log_level, console=False),
            )
        return self._client

    @property
    def retriever(self) -> KnowledgeBaseRetriever:
        if self._retriever is None:
            self._retriever = KnowledgeBaseRetriever(self._rag_config or RagConfig.from_env())
        return self._retriever

    def _build_agents(self) -> dict[str, Any]:
        if not self._agents_ready:
            self.agents = {
                NODE_DATA_ANALYSIS: DataAnalysisAgent(logger=self.logger),
                NODE_DIAGNOSIS: DiagnosisAgent(logger=self.logger, client=self.client),
                NODE_RAG: RagAgent(
                    logger=self.logger,
                    retriever=self._retriever,
                    rag_config=self._rag_config,
                ),
                NODE_RECOMMENDATION: RecommendationAgent(
                    logger=self.logger, client=self.client
                ),
                NODE_REPORT: ReportAgent(logger=self.logger, client=self.client),
            }
            self._agents_ready = True
        return self.agents

    # -- 图 ---------------------------------------------------------------
    def build(self) -> Any:
        """构建并编译 LangGraph 工作流。"""

        if self._compiled is not None:
            return self._compiled

        agents = self._build_agents()
        graph = StateGraph(WorkflowState)
        graph.add_node(NODE_DATA_ANALYSIS, self._as_node(agents[NODE_DATA_ANALYSIS]))
        graph.add_node(NODE_ABNORMAL_CHECK, self._abnormal_check_node)
        graph.add_node(NODE_DIAGNOSIS, self._as_node(agents[NODE_DIAGNOSIS]))
        graph.add_node(NODE_RAG, self._as_node(agents[NODE_RAG]))
        graph.add_node(NODE_RECOMMENDATION, self._as_node(agents[NODE_RECOMMENDATION]))
        graph.add_node(NODE_REPORT, self._as_node(agents[NODE_REPORT]))

        graph.add_edge(START, NODE_DATA_ANALYSIS)
        graph.add_conditional_edges(
            NODE_DATA_ANALYSIS,
            self._route_after_data_analysis,
            {NODE_ABNORMAL_CHECK: NODE_ABNORMAL_CHECK, NODE_REPORT: NODE_REPORT},
        )
        graph.add_conditional_edges(
            NODE_ABNORMAL_CHECK,
            self._route_after_check,
            {NODE_DIAGNOSIS: NODE_DIAGNOSIS, NODE_REPORT: NODE_REPORT},
        )
        graph.add_edge(NODE_DIAGNOSIS, NODE_RAG)
        graph.add_edge(NODE_RAG, NODE_RECOMMENDATION)
        graph.add_edge(NODE_RECOMMENDATION, NODE_REPORT)
        graph.add_edge(NODE_REPORT, END)

        self._compiled = graph.compile()
        return self._compiled

    @staticmethod
    def _as_node(agent: Any) -> Callable[[WorkflowState], dict[str, Any]]:
        def node(state: WorkflowState) -> dict[str, Any]:
            return agent.run(state)

        node.__name__ = f"node_{agent.stage}"
        return node

    # -- 节点：异常检查（程序判定）----------------------------------------
    def _abnormal_check_node(self, state: WorkflowState) -> dict[str, Any]:
        anomaly_count = int(state.get("anomaly_count") or 0)
        has_anomalies = bool(state.get("has_anomalies")) or anomaly_count > 0
        thresholds = state.get("thresholds") or {}
        threshold_text = "、".join(f"{key} {value:g}" for key, value in thresholds.items())
        if has_anomalies:
            detail = f"程序判定命中异常 {anomaly_count} 条（阈值：{threshold_text}），进入诊断流程"
        else:
            detail = f"程序判定未发现超过阈值的异常（阈值：{threshold_text}），跳过诊断流程"
        self.logger.log(f"[{ABNORMAL_CHECK_TAG}] {detail}")

        return {
            "has_anomalies": has_anomalies,
            "abnormal_check_report": detail,
            "steps": [
                *(state.get("steps") or []),
                {
                    "stage": NODE_ABNORMAL_CHECK,
                    "agent": ABNORMAL_CHECK_TAG,
                    "status": "completed",
                    "detail": detail,
                    "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": 0.0,
                },
            ],
        }

    # -- 路由 -------------------------------------------------------------
    @staticmethod
    def _route_after_data_analysis(state: WorkflowState) -> str:
        """数据分析失败时直接进入报告节点（报告会写明失败原因）。"""

        if NODE_DATA_ANALYSIS in (state.get("failed_stages") or []):
            return NODE_REPORT
        return NODE_ABNORMAL_CHECK

    @staticmethod
    def _route_after_check(state: WorkflowState) -> str:
        """有异常走完整流程，无异常直接出报告。"""

        return NODE_DIAGNOSIS if state.get("has_anomalies") else NODE_REPORT

    # -- 运行 -------------------------------------------------------------
    def run(
        self,
        *,
        data_path: str | Path | None = None,
        thresholds: project_config.AnomalyThresholds | None = None,
        use_knowledge: bool = True,
        use_llm: bool = True,
        rag_top_k: int | None = None,
        rag_min_similarity: float | None = None,
        rag_max_cases: int | None = None,
        export: bool = True,
        output_dir: str | Path | None = None,
        state: WorkflowState | None = None,
    ) -> WorkflowState:
        """执行完整工作流，返回最终 State。"""

        initial = state or create_initial_state(
            data_path=data_path,
            thresholds=thresholds,
            use_knowledge=use_knowledge,
            use_llm=use_llm,
            rag_top_k=rag_top_k,
            rag_min_similarity=rag_min_similarity,
            rag_max_cases=rag_max_cases,
            export=export,
            output_dir=output_dir,
        )
        self.logger.info(
            f"started - 数据源={Path(initial['data_path']).name} "
            f"知识库={'启用' if initial.get('use_knowledge') else '关闭'} "
            f"大模型={'启用' if initial.get('use_llm') else '离线模板'}"
        )

        final_state = self.build().invoke(initial)
        final_state["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_state["workflow_status"] = self._status_of(final_state)

        if final_state["workflow_status"] == "failed":
            self.logger.log(f"[{WORKFLOW_TAG}] failed - 数据分析阶段未成功完成")
        elif final_state["workflow_status"] == "completed_with_errors":
            self.logger.log(
                f"[{WORKFLOW_TAG}] finished with errors - "
                f"{len(final_state.get('errors') or [])} 个阶段出现异常，报告已降级生成"
            )
        else:
            self.logger.log(f"[{WORKFLOW_TAG}] finished")
        self.logger.info(state_overview(final_state))
        return final_state

    @staticmethod
    def _status_of(state: WorkflowState) -> str:
        failed = state.get("failed_stages") or []
        if NODE_DATA_ANALYSIS in failed:
            return "failed"
        if state.get("errors"):
            return "completed_with_errors"
        return "completed"


def run_workflow(
    *,
    data_path: str | Path | None = None,
    thresholds: project_config.AnomalyThresholds | None = None,
    use_knowledge: bool = True,
    use_llm: bool = True,
    rag_top_k: int | None = None,
    rag_min_similarity: float | None = None,
    rag_max_cases: int | None = None,
    export: bool = True,
    output_dir: str | Path | None = None,
    verbose: bool = True,
    echo: bool = True,
    log_file: str | Path | None = None,
) -> WorkflowState:
    """便捷入口：直接运行多 Agent 工作流。"""

    workflow = RobotOpsWorkflow(verbose=verbose, echo=echo, log_file=log_file)
    return workflow.run(
        data_path=data_path,
        thresholds=thresholds,
        use_knowledge=use_knowledge,
        use_llm=use_llm,
        rag_top_k=rag_top_k,
        rag_min_similarity=rag_min_similarity,
        rag_max_cases=rag_max_cases,
        export=export,
        output_dir=output_dir,
    )


def export_workflow_outputs(
    state: WorkflowState,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """导出工作流结果：报告 Markdown + State JSON + 日志。"""

    import json

    target = Path(output_dir) if output_dir is not None else project_config.OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    report_path = target / "workflow_report.md"
    report_path.write_text(state.get("final_report") or "", encoding=project_config.TEXT_ENCODING)
    written.append(report_path)

    state_path = target / "workflow_state.json"
    state_path.write_text(
        json.dumps(serializable_state(state), ensure_ascii=False, indent=2, default=str),
        encoding=project_config.TEXT_ENCODING,
    )
    written.append(state_path)

    steps_path = target / "workflow_steps.json"
    steps_path.write_text(
        json.dumps(
            {
                "workflow_status": state.get("workflow_status"),
                "steps": state.get("steps") or [],
                "errors": state.get("errors") or [],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding=project_config.TEXT_ENCODING,
    )
    written.append(steps_path)
    return written

