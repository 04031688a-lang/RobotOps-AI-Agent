"""Phase 3：Agent + RAG 集成测试（当前项目数据与历史案例必须区分）。"""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotops import config  # noqa: E402
from robotops.agent.robot_ops_agent import RobotOpsAnalysisAgent  # noqa: E402
from robotops.llm.client import DeepSeekClient  # noqa: E402
from robotops.llm.config import LLMConfig  # noqa: E402
from robotops.pipeline import run_analysis  # noqa: E402
from robotops.rag import KnowledgeBaseRetriever, RagConfig  # noqa: E402
from scripts.generate_demo_data import generate_demo_file  # noqa: E402
from tests._helpers import (  # noqa: E402
    FakeTransport,
    cleanup_dir,
    deepseek_response,
    make_temp_chroma_dir,
    release_chroma_client,
    sample_ai_json,
)

CURRENT_DATA_MARKER = "【结构化分析结果（当前项目数据，JSON）开始】"
HISTORY_MARKER = "【历史相似案例（JSON）开始】"


def silent_logger(name: str = "robotops.llm.rag-tests") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    return logger


class RagAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not config.DEMO_EXCEL_FILE.exists():
            generate_demo_file(config.DEMO_EXCEL_FILE)
        cls.result = run_analysis(export=False, verbose=False)
        cls.work_dir = make_temp_chroma_dir()
        cls.rag_config = RagConfig(persist_dir=cls.work_dir)
        cls.retriever = KnowledgeBaseRetriever(cls.rag_config)
        cls.retriever.rebuild()

    @classmethod
    def tearDownClass(cls) -> None:
        release_chroma_client(cls.retriever)
        cleanup_dir(cls.work_dir)

    def _agent(
        self,
        transport: FakeTransport,
        *,
        with_knowledge: bool = True,
        min_similarity: float | None = None,
    ) -> RobotOpsAnalysisAgent:
        llm_config = LLMConfig(api_key="sk-test-key", max_retries=0)
        client = DeepSeekClient(
            config=llm_config, transport=transport, logger=silent_logger(), sleep=lambda s: None
        )
        return RobotOpsAnalysisAgent(
            config=llm_config,
            client=client,
            logger=silent_logger(),
            retriever=self.retriever if with_knowledge else None,
            use_knowledge=with_knowledge,
            rag_min_similarity=min_similarity,
        )

    def test_rag_prompt_separates_current_data_and_history(self) -> None:
        transport = FakeTransport(deepseek_response(sample_ai_json()))
        agent = self._agent(transport)

        run = agent.analyze(self.result)

        messages = transport.last_request_json["messages"]
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]

        # 系统提示必须包含历史案例使用规则
        for rule in (
            "历史案例使用规则",
            "优先依据「当前项目数据」",
            "历史案例只能作为参考",
            "模拟案例",
            "当前项目事实",
            "历史相似案例",
            "推测原因",
            "建议措施",
            "未检索到足够相关的历史案例",
        ):
            self.assertIn(rule, system_prompt)

        # 用户提示必须把「当前项目数据」与「历史案例」放在两个独立区块
        self.assertIn(CURRENT_DATA_MARKER, user_prompt)
        self.assertIn(HISTORY_MARKER, user_prompt)
        self.assertLess(user_prompt.index(CURRENT_DATA_MARKER), user_prompt.index(HISTORY_MARKER))
        self.assertIn("不是当前项目的数据", user_prompt)

        # 当前数据是真实指标，历史案例是案例编号
        self.assertIn(f'"project_count": {self.result.metrics.project_count}', user_prompt)
        self.assertIn("CASE-", user_prompt)

        self.assertEqual(run.analysis.prompt_version, "phase3-rag-v1")
        self.assertTrue(run.retrieved_cases)
        self.assertNotIn("CASE-", run.payload_json, "历史案例不应混入结构化载荷")

    def test_report_renders_history_section(self) -> None:
        transport = FakeTransport(deepseek_response(sample_ai_json()))
        run = self._agent(transport).analyze(self.result)

        markdown = run.analysis.to_markdown()
        console = run.analysis.to_console_text()
        data = run.analysis.to_dict()

        self.assertIn("历史相似案例（RAG 检索）", markdown)
        self.assertIn("模拟案例", markdown)
        self.assertIn("历史相似案例（RAG 检索）", console)
        self.assertTrue(data["meta"]["retrieved_cases"])
        self.assertIn("retrieval", run.to_dict())

    def test_without_knowledge_keeps_phase2_behaviour(self) -> None:
        transport = FakeTransport(deepseek_response(sample_ai_json()))
        run = self._agent(transport, with_knowledge=False).analyze(self.result)

        user_prompt = transport.last_request_json["messages"][1]["content"]

        self.assertEqual(run.analysis.prompt_version, "phase2-agent-v1")
        self.assertNotIn(HISTORY_MARKER, user_prompt)
        self.assertIsNone(run.retrieval)
        self.assertEqual(run.retrieved_cases, [])

    def test_no_relevant_case_is_reported_explicitly(self) -> None:
        transport = FakeTransport(deepseek_response(sample_ai_json()))
        run = self._agent(transport, min_similarity=0.99).analyze(self.result)

        user_prompt = transport.last_request_json["messages"][1]["content"]

        self.assertTrue(run.retrieval.is_empty)
        self.assertIn("未检索到足够相关的历史案例", user_prompt)
        self.assertNotIn(HISTORY_MARKER, user_prompt)

    def test_dry_run_includes_retrieval_without_api_call(self) -> None:
        transport = FakeTransport(deepseek_response(sample_ai_json()))
        run = self._agent(transport).dry_run(self.result)

        self.assertEqual(transport.calls, 0)
        self.assertIsNone(run.analysis)
        self.assertTrue(run.retrieval.cases)
        self.assertTrue(run.retrieved_cases)
        self.assertIn(HISTORY_MARKER, run.messages[1]["content"])

    def test_search_knowledge_convenience_api(self) -> None:
        transport = FakeTransport(deepseek_response(sample_ai_json()))
        agent = self._agent(transport)

        result = agent.search_knowledge("机器人故障率升高，同时维修次数增加")

        self.assertTrue(result.cases)
        self.assertEqual(result.cases[0].case_id, "CASE-FAULT-001")


if __name__ == "__main__":
    unittest.main(verbosity=2)
