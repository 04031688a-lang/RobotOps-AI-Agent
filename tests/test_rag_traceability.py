"""Phase 10：RAG 来源追踪与相关性阈值测试。

覆盖需求场景：
1. 正常 RAG（案例必须带 case_id / title / category / similarity_score / source_file / content）；
2. RAG 无结果（无关问题）；
3. RAG 低相似度（低于阈值时不强行返回案例，给出统一提示）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.rag_agent import RagAgent  # noqa: E402
from app.graph.logging_utils import WorkflowLogger  # noqa: E402
from app.graph.state import create_initial_state  # noqa: E402
from robotops import config  # noqa: E402
from robotops.pipeline import run_analysis  # noqa: E402
from robotops.rag import KnowledgeBaseRetriever, RagConfig  # noqa: E402
from robotops.rag.config import NO_RELEVANT_CASE_MESSAGE  # noqa: E402
from tests._helpers import (  # noqa: E402
    cleanup_dir,
    make_temp_chroma_dir,
    make_temp_dir,
    release_chroma_client,
)

#: 需求要求的 6 个来源字段
REQUIRED_SOURCE_FIELDS = (
    "case_id",
    "title",
    "category",
    "similarity_score",
    "source_file",
    "content",
)

RELEVANT_QUERY = "B小区机器人故障率明显升高，同时维修次数增加。"


class RagTraceabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.work_dir = make_temp_chroma_dir()
        cls.temp_dir = make_temp_dir(prefix="robotops_test_phase10_")
        cls.config = RagConfig(persist_dir=cls.work_dir)
        cls.retriever = KnowledgeBaseRetriever(cls.config)
        cls.retriever.rebuild()

    @classmethod
    def tearDownClass(cls) -> None:
        release_chroma_client(cls.retriever)
        cleanup_dir(cls.work_dir)
        cleanup_dir(cls.temp_dir)

    # -- 1. 正常 RAG -------------------------------------------------------
    def test_normal_retrieval_keeps_all_source_fields(self) -> None:
        result = self.retriever.search(RELEVANT_QUERY)

        self.assertTrue(result.cases, "相关内容应检索到案例")
        for case in result.cases:
            payload = case.to_dict()
            with self.subTest(case=payload.get("case_id")):
                for field in REQUIRED_SOURCE_FIELDS:
                    self.assertIn(field, payload)
                    self.assertNotIn(payload[field], (None, ""), f"{field} 不应为空")
                self.assertGreater(float(payload["similarity_score"]), 0.0)
                self.assertEqual(
                    case.source_info().keys(),
                    {"case_id", "title", "category", "similarity_score", "source_file", "content"},
                )

    def test_case_content_contains_case_body(self) -> None:
        result = self.retriever.search(RELEVANT_QUERY)
        payload = result.cases[0].to_dict()

        # 案例正文必须能追溯到具体内容（而不只是标题）
        self.assertIn("【问题】", payload["content"])
        self.assertTrue(payload["title"])
        self.assertGreater(len(payload["content"]), 50)

    def test_prompt_entries_keep_source_info(self) -> None:
        result = self.retriever.search(RELEVANT_QUERY)
        entry = result.to_prompt_entries()[0]

        for field in REQUIRED_SOURCE_FIELDS:
            self.assertIn(field, entry)
        self.assertIn("case_summary", entry)

    def test_retrieval_result_has_message_field(self) -> None:
        result = self.retriever.search(RELEVANT_QUERY)

        self.assertEqual(result.to_dict()["message"], "")
        self.assertTrue(result.to_dict()["is_relevant"])

    # -- 2. RAG 无结果 -----------------------------------------------------
    def test_irrelevant_query_returns_no_case_with_message(self) -> None:
        result = self.retriever.search("今天天气怎么样？")

        self.assertFalse(result.cases)
        self.assertEqual(result.to_dict()["message"], NO_RELEVANT_CASE_MESSAGE)
        self.assertIn("未检索到足够相关的历史案例", result.reason)
        self.assertIn("当前知识库未找到高相关历史案例", NO_RELEVANT_CASE_MESSAGE)

    # -- 3. RAG 低相似度 ---------------------------------------------------
    def test_low_similarity_threshold_returns_no_case(self) -> None:
        result = self.retriever.search(RELEVANT_QUERY, min_similarity=0.99)

        self.assertFalse(result.cases, "低于阈值不应强行返回案例")
        self.assertEqual(result.to_dict()["message"], NO_RELEVANT_CASE_MESSAGE)
        self.assertIn("0.99", result.reason)

    def test_threshold_is_configurable_and_respected(self) -> None:
        loose = self.retriever.search(RELEVANT_QUERY, min_similarity=0.05)
        strict = self.retriever.search(RELEVANT_QUERY, min_similarity=0.95)

        self.assertTrue(loose.cases)
        self.assertFalse(strict.cases)
        self.assertLessEqual(
            max(case.similarity for case in loose.cases), 1.0
        )

    # -- Agent 层 ----------------------------------------------------------
    def test_rag_agent_passes_message_and_sources(self) -> None:
        analysis = run_analysis(config.DEMO_EXCEL_FILE, export=False, verbose=False)
        state = dict(create_initial_state(data_path=str(config.DEMO_EXCEL_FILE)))
        state["analysis_result"] = analysis

        logger = WorkflowLogger(
            verbose=False, echo=False, log_file=self.temp_dir / "phase10_rag.log"
        )
        agent = RagAgent(logger=logger, retriever=self.retriever)
        update = agent.execute(state)

        self.assertTrue(update["retrieved_cases"])
        for case in update["retrieved_cases"]:
            for field in REQUIRED_SOURCE_FIELDS:
                self.assertIn(field, case)
            self.assertIn("case_summary", case)
        self.assertIn("message", update["retrieval"])
        self.assertTrue(update["retrieval"]["is_relevant"])

    def test_rag_agent_records_no_result_message_when_disabled(self) -> None:
        state = dict(create_initial_state(data_path=str(config.DEMO_EXCEL_FILE)))
        state["use_knowledge"] = False

        logger = WorkflowLogger(
            verbose=False, echo=False, log_file=self.temp_dir / "phase10_rag_disabled.log"
        )
        agent = RagAgent(logger=logger, retriever=self.retriever)
        update = agent.execute(state)

        self.assertEqual(update["retrieved_cases"], [])
        self.assertIn("message", update["retrieval"])
        self.assertTrue(update["retrieval"]["message"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

