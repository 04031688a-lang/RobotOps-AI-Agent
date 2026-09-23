"""Phase 3：RAG 检索测试（ChromaDB 向量库 + Top-K 案例）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotops.rag import KnowledgeBaseRetriever, RagConfig  # noqa: E402
from robotops.rag.document_loader import CaseChunk, load_case_documents  # noqa: E402
from robotops.rag.retriever import expand_query  # noqa: E402
from tests._helpers import (  # noqa: E402
    cleanup_dir,
    make_temp_chroma_dir,
    release_chroma_client,
    release_local_chroma,
)

#: 需求中的示例问题 -> 期望命中的案例
EXPECTED_HITS: dict[str, str] = {
    "B小区机器人故障率明显升高，同时维修次数增加。": "CASE-FAULT-001",
    "机器人运行率下降，设备长时间停在充电位。": "CASE-OPS-001",
    "用户满意度下降，投诉说地面清洁不干净。": "CASE-SAT-001",
    "设备闲置，任务分配不均。": "CASE-OPS-002",
    "节降率下降，运营成本偏高。": "CASE-OPS-004",
    "巡检不到位，出现漏检点位。": "CASE-SAT-004",
}

#: 与机器人运营无关的问题 -> 不应返回案例
IRRELEVANT_QUERIES: tuple[str, ...] = (
    "今天天气怎么样？",
    "帮我写一首诗",
    "机器人多少钱一台",
    "Python 怎么读取 Excel 文件",
)


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.work_dir = make_temp_chroma_dir()
        cls.config = RagConfig(persist_dir=cls.work_dir)
        cls.documents = load_case_documents(cls.config)
        cls.retriever = KnowledgeBaseRetriever(cls.config, documents=cls.documents)
        cls.status = cls.retriever.rebuild()

    @classmethod
    def tearDownClass(cls) -> None:
        release_chroma_client(cls.retriever)
        cleanup_dir(cls.work_dir)

    # -- 索引 -------------------------------------------------------------
    def test_index_status_after_rebuild(self) -> None:
        self.assertTrue(self.status.exists)
        self.assertEqual(self.status.case_count, len(self.documents))
        self.assertGreater(self.status.chunk_count, len(self.documents))
        self.assertFalse(self.status.stale)
        self.assertEqual(self.status.embedding["backend"], "local_hashing")
        self.assertIn("案例数/块数", self.status.describe())

    def test_auto_build_when_index_missing(self) -> None:
        # 复用类级临时目录的子目录：类结束时统一释放句柄并清理
        tmp = self.work_dir / "auto_build"
        retriever = KnowledgeBaseRetriever(RagConfig(persist_dir=tmp))
        try:
            status = retriever.ensure_index()

            self.assertTrue(status.exists)
            self.assertGreater(status.chunk_count, 0)
        finally:
            release_local_chroma(retriever)

    def test_ensure_index_skips_build_when_auto_build_disabled(self) -> None:
        tmp = self.work_dir / "no_auto_build"
        retriever = KnowledgeBaseRetriever(RagConfig(persist_dir=tmp, auto_build=False))
        try:
            status = retriever.ensure_index()

            self.assertFalse(status.exists)
            self.assertEqual(status.chunk_count, 0)
        finally:
            release_local_chroma(retriever)

    def test_stale_detection_on_parameter_change(self) -> None:
        changed = RagConfig(persist_dir=self.config.persist_dir, embedding_dim=256)
        retriever = KnowledgeBaseRetriever(changed, documents=self.documents)

        self.assertTrue(retriever.status().stale)

    # -- 检索 -------------------------------------------------------------
    def test_relevant_queries_hit_expected_cases(self) -> None:
        for query, expected_case in EXPECTED_HITS.items():
            with self.subTest(query=query):
                result = self.retriever.search(query, top_k=3)
                case_ids = [case.case_id for case in result.cases]
                self.assertIn(expected_case, case_ids, f"{query} 未命中 {expected_case}")

    def test_requirement_example_query(self) -> None:
        result = self.retriever.search("B小区机器人故障率明显升高，同时维修次数增加。")

        self.assertTrue(result.cases)
        self.assertEqual(result.cases[0].case_id, "CASE-FAULT-001")
        self.assertGreater(result.cases[0].similarity, self.config.min_similarity)
        self.assertTrue(result.cases[0].matched_sections)
        self.assertTrue(result.cases[0].matched_queries)

    def test_irrelevant_queries_return_nothing(self) -> None:
        for query in IRRELEVANT_QUERIES:
            with self.subTest(query=query):
                result = self.retriever.search(query)
                self.assertTrue(result.is_empty, f"{query} 不应命中案例")
                self.assertIn("未检索到足够相关的历史案例", result.reason)

    def test_domain_gate(self) -> None:
        relevant, _ = self.retriever.is_domain_query("机器人故障率升高")
        irrelevant, note = self.retriever.is_domain_query("今天天气怎么样？")

        self.assertTrue(relevant)
        self.assertFalse(irrelevant)
        self.assertIn("领域词", note)

    def test_top_k_and_similarity_filters(self) -> None:
        limited = self.retriever.search("机器人故障率升高 维修次数增加", top_k=1)
        strict = self.retriever.search("机器人故障率升高", min_similarity=0.99)

        self.assertLessEqual(len(limited.cases), 1)
        self.assertTrue(strict.is_empty)

    def test_retrieved_case_metadata(self) -> None:
        case = self.retriever.search("机器人重复故障，同一部件反复损坏").cases[0]
        payload = case.to_dict()

        self.assertEqual(payload["case_id"], "CASE-FAULT-001")
        self.assertTrue(payload["title"])
        self.assertTrue(payload["case_type"])
        self.assertTrue(payload["keywords"])
        self.assertIn("模拟案例", payload["data_nature"])
        self.assertTrue(case.sections.get("处理措施"))
        self.assertLess(len(case.summary_text(budget=100)), 200)

    def test_retrieve_for_phase1_result(self) -> None:
        from robotops.pipeline import run_analysis

        result = run_analysis(export=False, verbose=False)
        retrieval = self.retriever.retrieve_for_result(result)

        self.assertTrue(retrieval.queries)
        self.assertTrue(all(query.text for query in retrieval.queries))
        self.assertTrue(retrieval.cases, "应能检索到历史案例")
        self.assertTrue(all(case.matched_queries for case in retrieval.cases))
        # 命中的查询应覆盖多个异常类型
        labels = {label for case in retrieval.cases for label in case.matched_queries}
        self.assertGreaterEqual(len(labels), 2)

    def test_query_expansion(self) -> None:
        short_text, short_terms = expand_query("满意度低怎么办")
        long_query = "B小区机器人故障率明显升高，同时维修次数增加，需要排查重复故障与维修频次问题"
        long_text, long_terms = expand_query(long_query)
        generic_text, generic_terms = expand_query("机器人多少钱一台")

        self.assertEqual(short_terms, ["满意度"])
        self.assertGreater(len(short_text), len("满意度低怎么办"))
        self.assertEqual(long_terms, [], "长查询包含多个业务词时无需扩展")
        self.assertEqual(long_text, long_query)
        self.assertEqual(generic_terms, [])
        self.assertEqual(generic_text, "机器人多少钱一台")

    # -- 写入 / 删除 ------------------------------------------------------
    def test_upsert_and_delete_case(self) -> None:
        chunk = CaseChunk(
            chunk_id="CASE-TEMP-001::问题::0",
            case_id="CASE-TEMP-001",
            title="临时测试案例（与运营无关的占位内容）",
            case_type="临时测试",
            category="operation",
            section="问题",
            chunk_index=0,
            text="临时测试专用片段：机器人故障率升高且维修次数增加",
            keywords=["临时测试"],
            data_nature="模拟案例（虚构）",
        )
        self.retriever.store.upsert_chunks([chunk])
        try:
            found = self.retriever.retrieve_chunks(
                "临时测试专用片段 机器人故障率升高且维修次数增加",
                top_k=5,
                min_similarity=0.0,
            )
            self.assertIn("CASE-TEMP-001::问题::0", [item.chunk_id for item in found])
        finally:
            deleted = self.retriever.store.delete_case("CASE-TEMP-001")

        self.assertGreaterEqual(deleted, 1)
        remaining = self.retriever.retrieve_chunks(
            "临时测试专用片段 机器人故障率升高", top_k=5, min_similarity=0.0
        )
        self.assertNotIn("CASE-TEMP-001::问题::0", [item.chunk_id for item in remaining])


if __name__ == "__main__":
    unittest.main(verbosity=2)
