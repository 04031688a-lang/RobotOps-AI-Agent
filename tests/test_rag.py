"""Phase 11：RAG 检索场景测试（正常检索 / RAG 无结果 / 领域门控）。

覆盖需求要求的「RAG 无结果」等场景：
1. 相关问题 —— 命中预期案例；
2. 无关问题（领域门控）—— 返回空结果并给出统一提示；
3. 相似度阈值过高 —— 同样返回空结果（RAG 无结果路径）；
4. 空查询 / 短查询 —— 不崩溃、返回空或有效结果。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotops.rag import KnowledgeBaseRetriever, RagConfig  # noqa: E402
from robotops.rag.config import NO_RELEVANT_CASE_MESSAGE  # noqa: E402
from robotops.rag.document_loader import load_case_documents  # noqa: E402
from tests._helpers import (  # noqa: E402
    cleanup_dir,
    make_temp_chroma_dir,
    release_chroma_client,
)

RELEVANT_QUERY = "B小区机器人故障率明显升高，同时维修次数增加。"
IRRELEVANT_QUERY = "今天天气怎么样？"


class RagScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.work_dir = make_temp_chroma_dir(prefix="robotops_test_rag11_")
        cls.config = RagConfig(persist_dir=cls.work_dir)
        cls.documents = load_case_documents(cls.config)
        cls.retriever = KnowledgeBaseRetriever(cls.config, documents=cls.documents)
        cls.retriever.rebuild()

    @classmethod
    def tearDownClass(cls) -> None:
        release_chroma_client(cls.retriever)
        cleanup_dir(cls.work_dir)

    # -- 1) 相关问题命中 ---------------------------------------------------
    def test_relevant_query_hits_cases(self) -> None:
        result = self.retriever.search(RELEVANT_QUERY, top_k=3)

        self.assertTrue(result.cases)
        case_ids = {case.case_id for case in result.cases}
        self.assertIn("CASE-FAULT-001", case_ids)
        for case in result.cases:
            self.assertGreaterEqual(case.similarity, self.config.min_similarity)

    # -- 2) 无关问题：RAG 无结果 -------------------------------------------
    def test_irrelevant_query_returns_no_cases(self) -> None:
        result = self.retriever.search(IRRELEVANT_QUERY)

        self.assertTrue(result.is_empty)
        self.assertFalse(result.is_relevant)
        self.assertIn("未包含机器人运营领域词", "；".join(result.notes))
        # 用户看到的统一提示
        self.assertEqual(result.to_dict()["message"], NO_RELEVANT_CASE_MESSAGE)

    # -- 3) 阈值过高：RAG 无结果 -------------------------------------------
    def test_high_similarity_threshold_returns_no_cases(self) -> None:
        result = self.retriever.search(RELEVANT_QUERY, min_similarity=0.99)

        self.assertTrue(result.is_empty)
        self.assertIn("未检索到足够相关的历史案例", result.reason)

    # -- 4) 空查询不崩溃 ---------------------------------------------------
    def test_empty_query_is_safe(self) -> None:
        result = self.retriever.search("")

        self.assertTrue(result.is_empty)
        self.assertFalse(result.is_relevant)

    # -- 5) 检索结果可序列化（供日志 / 前端使用） ---------------------------
    def test_result_serializable(self) -> None:
        result = self.retriever.search(RELEVANT_QUERY, top_k=3)

        payload = result.to_dict()
        self.assertIsInstance(payload["case_count"], int)
        self.assertTrue(payload["cases"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
