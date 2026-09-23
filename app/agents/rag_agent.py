"""Phase 4 —— RAG Agent（不调用大模型）。

职责（单一）：根据异常描述检索历史案例，并保留案例来源与标题。

复用 Phase 3 的检索模块（``robotops.rag``）：异常描述生成、相关性门控、
分类优先检索、相似度阈值过滤、Top-K 聚合均已实现，本 Agent 只负责调用与整理。
"""

from __future__ import annotations

from typing import Any

from robotops.rag import KnowledgeBaseRetriever, RagConfig

from ..graph.state import WorkflowState
from .base import BaseAgent


class RagAgent(BaseAgent):
    tag = "RAG Agent"
    stage = "rag"

    def __init__(
        self,
        *,
        logger: Any,
        retriever: KnowledgeBaseRetriever | None = None,
        rag_config: RagConfig | None = None,
        case_text_budget: int = 700,
    ) -> None:
        super().__init__(logger=logger, client=None)
        self._retriever = retriever
        self._rag_config = rag_config
        self.case_text_budget = case_text_budget

    @property
    def retriever(self) -> KnowledgeBaseRetriever:
        if self._retriever is None:
            self._retriever = KnowledgeBaseRetriever(self._rag_config or RagConfig.from_env())
        return self._retriever

    def execute(self, state: WorkflowState) -> dict[str, Any]:
        if not state.get("use_knowledge", True):
            self.logger.log(f"[{self.tag}] skipped - 已通过 --no-rag 关闭知识库检索")
            return {
                "retrieved_cases": [],
                "retrieval": {
                    "enabled": False,
                    "reason": "本次运行使用 --no-rag，未检索历史案例",
                    "cases": [],
                    "queries": [],
                },
            }

        analysis_result = state.get("analysis_result")
        if analysis_result is None:
            return {
                "retrieved_cases": [],
                "retrieval": {
                    "enabled": True,
                    "reason": "缺少 Phase 1 分析结果，未执行检索",
                    "cases": [],
                    "queries": [],
                },
            }

        retriever = self.retriever
        retriever.ensure_index()
        retrieval = retriever.retrieve_for_result(
            analysis_result,
            top_k=state.get("rag_top_k"),
            min_similarity=state.get("rag_min_similarity"),
            max_cases=state.get("rag_max_cases"),
        )

        cases = [
            {
                **case.to_dict(),
                "case_summary": case.summary_text(budget=self.case_text_budget),
            }
            for case in retrieval.cases
        ]
        if cases:
            self.logger.log(f"[{self.tag}] retrieved {len(cases)} cases")
        else:
            reason = retrieval.reason or "未检索到足够相关的历史案例"
            self.logger.log(f"[{self.tag}] no relevant case found - {reason}")

        return {
            "retrieved_cases": cases,
            "retrieval": {**retrieval.to_dict(), "enabled": True},
        }

    def summarize(self, update: dict[str, Any], state: WorkflowState) -> str:
        return f"retrieved {len(update.get('retrieved_cases') or [])} cases"

