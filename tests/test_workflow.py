"""Phase 4：多 Agent 工作流测试（路由分支、State、日志、降级与错误处理）。"""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.data_analysis import DataAnalysisAgent
from app.agents.diagnosis import DiagnosisAgent
from app.agents.rag_agent import RagAgent
from app.graph.logging_utils import WorkflowLogger
from app.graph.state import (
    REQUIRED_STATE_KEYS,
    create_initial_state,
    missing_required_keys,
    serializable_state,
)
from app.graph.workflow import RobotOpsWorkflow
from robotops import config  # noqa: E402
from robotops.llm.client import DeepSeekClient
from robotops.llm.config import LLMConfig
from robotops.rag import KnowledgeBaseRetriever, RagConfig
from scripts.generate_demo_data import generate_demo_file
from tests._helpers import (
    FakeTransport,
    cleanup_dir,
    deepseek_error_response,
    deepseek_response,
    make_temp_chroma_dir,
    make_temp_dir,
    release_chroma_client,
    report_markdown,
    workflow_llm_responses,
    write_no_anomaly_data,
)

REQUIRED_LOG_LINES = (
    "[Data Analysis Agent] started",
    "[Data Analysis Agent] completed",
    "[Diagnosis Agent] started",
    "[Recommendation Agent] completed",
    "[Report Agent] completed",
)


def silent_logger(name: str = "robotops.llm.workflow-tests") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    return logger


class WorkflowTestCase(unittest.TestCase):
    """公共准备：演示数据、临时向量库、假大模型客户端。"""

    @classmethod
    def setUpClass(cls) -> None:
        if not config.DEMO_EXCEL_FILE.exists():
            generate_demo_file(config.DEMO_EXCEL_FILE)
        cls.work_dir = make_temp_chroma_dir()
        cls.retriever = KnowledgeBaseRetriever(RagConfig(persist_dir=cls.work_dir))
        cls.retriever.rebuild()
        cls.temp_dir = make_temp_dir(prefix="robotops_test_workflow_")
        cls.normal_data = write_no_anomaly_data(cls.temp_dir / "normal.xlsx")

    @classmethod
    def tearDownClass(cls) -> None:
        release_chroma_client(cls.retriever)
        cleanup_dir(cls.work_dir)
        cleanup_dir(cls.temp_dir)

    # -- 工具 -------------------------------------------------------------
    def build_workflow(
        self,
        transport: FakeTransport | None = None,
        *,
        retriever: KnowledgeBaseRetriever | None = None,
        log_file: Path | None = None,
    ) -> tuple[RobotOpsWorkflow, FakeTransport]:
        scripted = transport or FakeTransport(
            *(deepseek_response(item) for item in workflow_llm_responses())
        )
        client = DeepSeekClient(
            config=LLMConfig(api_key="sk-test-key", max_retries=0),
            transport=scripted,
            logger=silent_logger(),
            sleep=lambda seconds: None,
        )
        logger = WorkflowLogger(
            verbose=False, echo=False, log_file=log_file or (self.temp_dir / "workflow.log")
        )
        workflow = RobotOpsWorkflow(
            client=client,
            retriever=retriever if retriever is not None else self.retriever,
            logger=logger,
            verbose=False,
            echo=False,
        )
        return workflow, scripted

    def stages(self, state: dict) -> list[str]:
        return [step["stage"] for step in state.get("steps") or []]


class WorkflowRoutingTests(WorkflowTestCase):
    def test_01_normal_data_skips_diagnosis_flow(self) -> None:
        """测试1：正常数据 → 不触发诊断 / RAG / 建议流程（离线模板模式）。"""

        workflow, transport = self.build_workflow()

        state = workflow.run(data_path=self.normal_data, export=False, use_llm=False)

        self.assertEqual(
            self.stages(state), ["data_analysis", "abnormal_check", "report"]
        )
        self.assertFalse(state["has_anomalies"])
        self.assertEqual(state["anomaly_count"], 0)
        self.assertEqual(transport.calls, 0, "无异常时不应调用大模型")
        self.assertEqual(state["diagnosis"], {})
        self.assertEqual(state["retrieved_cases"], [])
        self.assertEqual(state["recommendations"], {})
        self.assertEqual(state["workflow_status"], "completed")
        self.assertIn("未发现超过阈值的异常", state["final_report"])
        self.assertIn("[Abnormal Check]", workflow.logger.text())

    def test_01b_normal_data_still_allows_report_llm(self) -> None:
        """正常数据下只有 Report Agent 会调用大模型（诊断/RAG/建议被跳过）。"""

        workflow, transport = self.build_workflow(
            FakeTransport(deepseek_response(report_markdown()))
        )

        state = workflow.run(data_path=self.normal_data, export=False, use_llm=True)

        self.assertEqual(self.stages(state), ["data_analysis", "abnormal_check", "report"])
        self.assertEqual(transport.calls, 1, "正常数据下只应调用一次大模型（报告）")
        log = workflow.logger.text()
        for tag in ("Diagnosis Agent", "RAG Agent", "Recommendation Agent"):
            self.assertNotIn(f"[{tag}]", log)
        self.assertIn("[Report Agent] completed", log)
        self.assertIn("## 三、异常发现", state["final_report"])

    def test_02_abnormal_data_triggers_full_flow(self) -> None:
        """测试2：异常数据 → 触发完整流程（诊断 → RAG → 建议 → 报告）。"""

        workflow, transport = self.build_workflow()

        state = workflow.run(export=False)

        self.assertEqual(
            self.stages(state),
            ["data_analysis", "abnormal_check", "diagnosis", "rag", "recommendation", "report"],
        )
        self.assertTrue(state["has_anomalies"])
        self.assertGreater(state["anomaly_count"], 0)
        self.assertTrue(state["diagnosis"]["possible_reasons"])
        self.assertTrue(state["retrieved_cases"])
        self.assertTrue(state["recommendations"]["items"])
        self.assertEqual(transport.calls, 3, "应按顺序调用诊断 / 建议 / 报告三次")
        self.assertEqual(state["workflow_status"], "completed")
        self.assertIn("## 四、可能原因（推测）", state["final_report"])
        self.assertIn("CASE-FAULT-001", str(state["retrieved_cases"]))

    def test_03_rag_without_cases_still_completes(self) -> None:
        """测试3：RAG 没有找到相关案例 → 工作流仍然可以完成。"""

        workflow, transport = self.build_workflow()

        state = workflow.run(export=False, rag_min_similarity=0.99)

        self.assertEqual(state["retrieved_cases"], [])
        self.assertEqual(
            self.stages(state),
            ["data_analysis", "abnormal_check", "diagnosis", "rag", "recommendation", "report"],
        )
        self.assertEqual(state["workflow_status"], "completed")
        self.assertEqual(state["errors"], [])
        self.assertIn("no relevant case found", workflow.logger.text())
        self.assertIn(
            "未检索到足够相关的历史案例", str(state["retrieval"].get("reason"))
        )
        # 没有案例时，必须把「未检索到足够相关案例」传给建议 Agent
        recommendation_prompt = transport.requests[1].body.decode("utf-8")
        self.assertIn("未检索到足够相关的历史案例", recommendation_prompt)
        self.assertTrue(state["recommendations"]["items"], "没有案例时仍应给出建议")

    def test_04_thresholds_are_decided_by_program(self) -> None:
        """异常判定由程序完成：放宽阈值后同一份数据不再触发诊断。"""

        workflow, transport = self.build_workflow()
        loose = config.AnomalyThresholds(
            fault_rate_upper=999.0,
            satisfaction_lower=0.0,
            uptime_rate_lower=0.0,
            saving_rate_lower=-999.0,
        )

        state = workflow.run(export=False, thresholds=loose, use_llm=False)

        self.assertFalse(state["has_anomalies"])
        self.assertNotIn("diagnosis", self.stages(state))
        self.assertEqual(transport.calls, 0)

    def test_05_state_contract(self) -> None:
        workflow, _ = self.build_workflow()

        state = workflow.run(export=False)

        self.assertEqual(missing_required_keys(state), [])
        for key in REQUIRED_STATE_KEYS:
            self.assertIn(key, state)
        self.assertIn("项目数量", state["metrics"]["values"])
        self.assertEqual(state["metrics"]["values"]["项目数量"], 6)
        exported = serializable_state(state)
        self.assertNotIn("analysis_result", exported)
        self.assertTrue(exported["final_report_preview"])

    def test_06_workflow_log_lines(self) -> None:
        workflow, _ = self.build_workflow()

        workflow.run(export=False)
        log = workflow.logger.text()

        for line in REQUIRED_LOG_LINES:
            self.assertIn(line, log)
        self.assertIn("[RAG Agent] retrieved", log)
        self.assertIn("[Workflow] finished", log)

    def test_07_agents_have_single_responsibility(self) -> None:
        workflow, _ = self.build_workflow()
        state = workflow.run(export=False)

        # 数据分析 Agent 只负责分析相关字段
        data_agent = DataAnalysisAgent(logger=workflow.logger)
        data_update = data_agent.run(state)
        self.assertIn("metrics", data_update)
        for key in ("diagnosis", "retrieved_cases", "recommendations", "final_report"):
            self.assertNotIn(key, data_update)

        # RAG Agent 只负责检索相关字段
        rag_agent = RagAgent(logger=workflow.logger, retriever=self.retriever)
        rag_update = rag_agent.run(state)
        self.assertIn("retrieved_cases", rag_update)
        for key in ("metrics", "diagnosis", "recommendations", "final_report"):
            self.assertNotIn(key, rag_update)

        # 诊断 Agent 只负责诊断字段
        diagnosis_agent = DiagnosisAgent(logger=workflow.logger, client=workflow.client)
        diagnosis_update = diagnosis_agent.run({**state, "use_llm": False})
        self.assertIn("diagnosis", diagnosis_update)
        for key in ("retrieved_cases", "recommendations", "final_report"):
            self.assertNotIn(key, diagnosis_update)


class WorkflowErrorTests(WorkflowTestCase):
    def test_08_llm_failure_gives_clear_error(self) -> None:
        """测试4：DeepSeek API 失败 → 给出明确错误，且不抛 traceback。"""

        transport = FakeTransport(deepseek_error_response(401, "Authentication Fails"))
        workflow, _ = self.build_workflow(transport)

        state = workflow.run(export=False)

        llm_errors = [item for item in state["errors"] if item.get("is_llm_error")]
        self.assertTrue(llm_errors, "应记录大模型失败")
        message = " ".join(item["message"] for item in llm_errors)
        self.assertIn("认证失败", message)
        self.assertIn("DEEPSEEK_API_KEY", " ".join(item.get("hint", "") for item in llm_errors))
        self.assertEqual(transport.calls, 1, "失败后不应重复调用大模型")

        log = workflow.logger.text()
        self.assertIn("[Diagnosis Agent] failed", log)
        self.assertIn("[Recommendation Agent] skipped", log)
        self.assertIn("[Report Agent] completed", log)

        self.assertEqual(state["workflow_status"], "completed_with_errors")
        self.assertIn("降级说明", state["final_report"])
        self.assertIn("大模型调用失败", state["final_report"])
        self.assertEqual(state["report_meta"]["source"], "offline_template")

    def test_09_data_error_is_reported_without_traceback(self) -> None:
        workflow, _ = self.build_workflow()

        state = workflow.run(
            data_path=self.temp_dir / "not-exists.xlsx", export=False, use_llm=False
        )

        self.assertEqual(state["workflow_status"], "failed")
        self.assertEqual(self.stages(state)[0], "data_analysis")
        self.assertIn("report", self.stages(state))
        record = state["errors"][0]
        self.assertEqual(record["stage"], "data_analysis")
        self.assertIn("未找到数据文件", record["message"])
        self.assertIn("提示", record["message"])
        self.assertIn("数据分析阶段未成功完成", workflow.logger.text())

    def test_10_rag_error_degrades_but_completes(self) -> None:
        broken = KnowledgeBaseRetriever(
            RagConfig(persist_dir=self.temp_dir / "chroma_broken", knowledge_dir=self.temp_dir / "no_knowledge")
        )
        workflow, transport = self.build_workflow(retriever=broken)

        state = workflow.run(export=False)

        rag_errors = [item for item in state["errors"] if item.get("is_rag_error")]
        self.assertTrue(rag_errors, "知识库不可用时应记录 RAG 错误")
        self.assertEqual(state["retrieved_cases"], [])
        self.assertEqual(state["workflow_status"], "completed_with_errors")
        self.assertTrue(state["final_report"])
        self.assertTrue(state["recommendations"]["items"])
        self.assertEqual(transport.calls, 3, "RAG 失败不应阻断大模型流程")


if __name__ == "__main__":
    unittest.main(verbosity=2)
