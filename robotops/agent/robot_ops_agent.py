"""RobotOps AI —— Phase 2「机器人运营数据分析 Agent」。

数据流（不把原始 Excel 交给大模型）：

    Excel/CSV
      → Phase 1 Pandas 分析（读取 → 清洗 → 指标 → 异常）
      → 结构化载荷（payload.py）
      → DeepSeek（llm/client.py）
      → 结构化分析结果（llm/schema.py）

职责边界：
- 指标数值全部由 Phase 1 程序计算，Agent 只做解读；
- 网络/超时/密钥/格式等异常统一由 LLM 模块的错误类型表达，便于上层给出清晰提示。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import config as project_config
from ..exceptions import ConfigurationError
from ..pipeline import AnalysisResult
from ..llm.client import DeepSeekClient
from ..llm.config import LLMConfig
from ..llm.errors import LLMResponseFormatError
from ..llm.logger import setup_llm_logger
from ..llm.prompts import PROMPT_VERSION, RAG_PROMPT_VERSION, build_messages
from ..llm.schema import AIAnalysis, parse_ai_analysis
from ..rag.config import RagConfig
from ..rag.query_builder import DEFAULT_MAX_QUERIES
from ..rag.retriever import KnowledgeBaseRetriever, RetrievalResult
from .payload import PAYLOAD_VERSION, build_analysis_payload, payload_to_json

AGENT_NAME = "RobotOpsAnalysisAgent"
RAW_RESPONSE_FILE = "deepseek_raw_response.txt"
#: AI 分析结果的输出文件（不含扩展名）
AI_OUTPUT_BASENAME = "ai_analysis"
#: 发送给模型的输入载荷备份文件
AI_PAYLOAD_FILE = "ai_input_payload.json"
#: RAG 检索结果备份文件（查询、命中案例、相似度）
AI_RAG_FILE = "ai_rag_retrieval.json"


@dataclass
class AgentRunResult:
    """一次 Agent 运行的结果（含输入载荷与输出分析）。"""

    payload: dict[str, Any]
    analysis: AIAnalysis | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    model: str = ""
    prompt_version: str = PROMPT_VERSION
    payload_version: str = PAYLOAD_VERSION
    usage: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    elapsed_seconds: float = 0.0
    dry_run: bool = False
    retrieval: RetrievalResult | None = None

    @property
    def payload_json(self) -> str:
        return payload_to_json(self.payload)

    @property
    def retrieved_cases(self) -> list[dict[str, Any]]:
        """RAG 检索到的历史案例（供报告与导出使用）。"""

        if self.retrieval is None or self.retrieval.is_empty:
            return []
        return [case.to_dict() for case in self.retrieval.cases]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "agent": AGENT_NAME,
            "payload_version": self.payload_version,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "dry_run": self.dry_run,
            "attempts": self.attempts,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "usage": self.usage,
            "input_payload": self.payload,
        }
        if self.retrieval is not None:
            data["retrieval"] = self.retrieval.to_dict()
        if self.analysis is not None:
            data["analysis"] = self.analysis.to_dict()
        return data


class RobotOpsAnalysisAgent:
    """把 Phase 1 的结构化分析结果交给 DeepSeek 做解读。"""

    def __init__(
        self,
        *,
        config: LLMConfig | None = None,
        client: DeepSeekClient | None = None,
        logger: Any = None,
        env_file: str | Path | None = None,
        retriever: KnowledgeBaseRetriever | None = None,
        rag_config: RagConfig | None = None,
        use_knowledge: bool | None = None,
        rag_top_k: int | None = None,
        rag_min_similarity: float | None = None,
        rag_max_cases: int | None = None,
        rag_max_queries: int = DEFAULT_MAX_QUERIES,
    ) -> None:
        self.config = config or LLMConfig.from_env(env_file=env_file)
        self.logger = logger or setup_llm_logger(
            level=self.config.log_level, log_file=self.config.log_file
        )
        self._client = client
        self._retriever = retriever
        self._rag_config = rag_config
        self._env_file = env_file
        # 默认行为与 Phase 2 保持一致：只有显式提供知识库配置时才启用 RAG
        self.use_knowledge = (
            bool(use_knowledge)
            if use_knowledge is not None
            else (retriever is not None or rag_config is not None)
        )
        self.rag_top_k = rag_top_k
        self.rag_min_similarity = rag_min_similarity
        self.rag_max_cases = rag_max_cases
        self.rag_max_queries = rag_max_queries

    # -- 客户端 -----------------------------------------------------------
    @property
    def client(self) -> DeepSeekClient:
        if self._client is None:
            self._client = DeepSeekClient(config=self.config, logger=self.logger)
        return self._client

    # -- 知识库（RAG）-----------------------------------------------------
    @property
    def retriever(self) -> KnowledgeBaseRetriever:
        """惰性创建知识库检索器（未安装 ChromaDB 时会抛出带提示的异常）。"""

        if self._retriever is None:
            rag_config = self._rag_config or RagConfig.from_env(env_file=self._env_file)
            self._retriever = KnowledgeBaseRetriever(rag_config)
        return self._retriever

    def search_knowledge(self, query_text: str, *, top_k: int | None = None) -> RetrievalResult:
        """按自然语言问题检索历史案例（Phase 3 检索模块的便捷入口）。"""

        retriever = self.retriever
        retriever.ensure_index()
        return retriever.search(query_text, top_k=top_k or self.rag_top_k)

    def _retrieve(self, result: AnalysisResult) -> RetrievalResult:
        """生成异常描述并检索历史案例。"""

        retriever = self.retriever
        self.logger.info(
            "RAG 检索开始：知识库=%s 集合=%s top_k=%s min_similarity=%s",
            retriever.config.knowledge_dir,
            retriever.config.collection_name,
            self.rag_top_k or retriever.config.top_k,
            self.rag_min_similarity if self.rag_min_similarity is not None else retriever.config.min_similarity,
        )
        index_status = retriever.ensure_index()
        self.logger.info(
            "知识库索引就绪：案例 %s 个 / 文本块 %s 个", index_status.case_count, index_status.chunk_count
        )
        retrieval = retriever.retrieve_for_result(
            result,
            max_queries=self.rag_max_queries,
            top_k=self.rag_top_k,
            min_similarity=self.rag_min_similarity,
            max_cases=self.rag_max_cases,
        )
        if retrieval.cases:
            self.logger.info(
                "RAG 检索完成：命中 %s 个案例（最高相关度 %.3f）",
                len(retrieval.cases),
                retrieval.cases[0].similarity,
            )
        else:
            self.logger.warning("RAG 检索未命中相关案例：%s", retrieval.reason)
        return retrieval

    # -- 载荷 -------------------------------------------------------------
    def build_payload(self, result: AnalysisResult) -> dict[str, Any]:
        """把 Phase 1 结果转换为 Agent 输入载荷。"""

        if not isinstance(result, AnalysisResult):
            raise ConfigurationError(
                "Agent 输入必须是 Phase 1 的结构化分析结果（AnalysisResult）。"
                "请先执行 Pandas 分析流程（run_analysis），不要直接把 Excel 原始数据传给 Agent。"
            )
        payload = build_analysis_payload(result)
        self.logger.debug(
            "已构造 Agent 载荷：version=%s 项目=%s 异常机器人=%s",
            payload.get("payload_version"),
            len(payload.get("projects", [])),
            len(payload.get("abnormal_robots", [])),
        )
        return payload

    # -- 干跑（不消耗 API）-------------------------------------------------
    def dry_run(self, result: AnalysisResult) -> AgentRunResult:
        """只构造载荷与提示词，不调用 API（无需 API Key，零消耗）。"""

        payload = self.build_payload(result)
        retrieval = self._retrieve(result) if self.use_knowledge else None
        messages = self._build_messages(payload, retrieval)
        self.logger.info("干跑模式：已生成载荷与提示词，未调用 DeepSeek API")
        return AgentRunResult(
            payload=payload,
            analysis=None,
            messages=messages,
            model=self.config.model,
            dry_run=True,
            retrieval=retrieval,
        )

    # -- 正式调用 ---------------------------------------------------------
    def analyze(self, result: AnalysisResult) -> AgentRunResult:
        """调用 DeepSeek 生成结构化分析结论。"""

        payload = self.build_payload(result)
        retrieval = self._retrieve(result) if self.use_knowledge else None
        messages = self._build_messages(payload, retrieval)
        started = datetime.now()

        prompt_version = RAG_PROMPT_VERSION if (retrieval and retrieval.cases) else PROMPT_VERSION
        self.logger.info(
            "Agent 开始分析：model=%s prompt_version=%s payload_version=%s 历史案例=%s",
            self.config.model,
            prompt_version,
            PAYLOAD_VERSION,
            len(retrieval.cases) if retrieval else 0,
        )

        chat_result = self.client.chat(messages)

        try:
            analysis = parse_ai_analysis(
                chat_result.content,
                model=chat_result.model,
                usage=chat_result.usage,
                payload_meta={
                    "data_source": payload["meta"]["data_source"],
                    "analysis_window": payload["meta"]["analysis_window"],
                    "cleaned_record_count": payload["meta"]["cleaned_record_count"],
                },
                prompt_version=prompt_version,
                retrieved_cases=(
                    [case.to_dict() for case in retrieval.cases] if retrieval else None
                ),
            )
        except LLMResponseFormatError as error:
            saved = self._save_raw_response(chat_result.content)
            hint = error.hint
            if saved is not None:
                hint = f"{hint}（原始返回已保存：{saved}）"
            raise LLMResponseFormatError(str(error), hint=hint, raw=error.raw) from error

        elapsed = (datetime.now() - started).total_seconds()
        self.logger.info(
            "Agent 分析完成：耗时=%.2fs 关键发现=%s 条 建议=%s 条 校验提示=%s 条",
            elapsed,
            len(analysis.key_findings),
            len(analysis.recommendations),
            len(analysis.warnings),
        )

        return AgentRunResult(
            payload=payload,
            analysis=analysis,
            messages=messages,
            model=chat_result.model,
            usage=chat_result.usage,
            attempts=chat_result.attempts,
            elapsed_seconds=elapsed,
            dry_run=False,
            retrieval=retrieval,
        )

    def _build_messages(
        self, payload: dict[str, Any], retrieval: RetrievalResult | None
    ) -> list[dict[str, str]]:
        """按是否命中历史案例构造提示词。"""

        cases = None
        note = ""
        if retrieval is not None:
            if retrieval.cases:
                budget = (self._rag_config or self.retriever.config).case_text_budget
                cases = retrieval.to_prompt_entries(budget=budget)
            else:
                note = retrieval.reason or "未检索到足够相关的历史案例"
        return build_messages(payload, retrieved_cases=cases, retrieval_note=note)

    # -- 工具 -------------------------------------------------------------
    def _save_raw_response(self, content: str) -> Path | None:
        target = project_config.OUTPUT_DIR / RAW_RESPONSE_FILE
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content or "", encoding="utf-8")
        except OSError:
            return None
        return target


def run_agent(
    *,
    data_path: str | Path | None = None,
    env_file: str | Path | None = None,
    sheet_name: str | int | None = None,
    thresholds: project_config.AnomalyThresholds | None = None,
    dry_run: bool = False,
    export_phase1: bool = True,
    verbose: bool = True,
    use_knowledge: bool = False,
) -> AgentRunResult:
    """便捷入口：跑完 Phase 1 分析后直接交给 Agent。"""

    from ..pipeline import run_analysis

    analysis_result = run_analysis(
        data_path,
        sheet_name=sheet_name,
        thresholds=thresholds,
        export=export_phase1,
        verbose=verbose,
    )

    agent = RobotOpsAnalysisAgent(
        env_file=env_file,
        use_knowledge=use_knowledge,
        rag_config=RagConfig.from_env(env_file=env_file) if use_knowledge else None,
    )
    return agent.dry_run(analysis_result) if dry_run else agent.analyze(analysis_result)
