"""Phase 2：机器人运营数据分析 Agent 测试（数据流、Prompt、错误处理）。"""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from robotops import config  # noqa: E402
from robotops.agent.robot_ops_agent import RAW_RESPONSE_FILE, RobotOpsAnalysisAgent  # noqa: E402
from robotops.exceptions import ConfigurationError  # noqa: E402
from robotops.llm.client import DeepSeekClient  # noqa: E402
from robotops.llm.config import LLMConfig  # noqa: E402
from robotops.llm.errors import LLMResponseFormatError  # noqa: E402
from robotops.pipeline import run_analysis  # noqa: E402
from scripts.generate_demo_data import generate_demo_file  # noqa: E402
from tests._helpers import FakeTransport, deepseek_response, sample_ai_json  # noqa: E402


def silent_logger(name: str = "robotops.llm.agent-tests") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    return logger


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not config.DEMO_EXCEL_FILE.exists():
            generate_demo_file(config.DEMO_EXCEL_FILE)
        cls.result = run_analysis(export=False, verbose=False)

    def _agent(self, transport: FakeTransport) -> RobotOpsAnalysisAgent:
        config_ = LLMConfig(api_key="sk-test-key", max_retries=0)
        client = DeepSeekClient(
            config=config_, transport=transport, logger=silent_logger(), sleep=lambda s: None
        )
        return RobotOpsAnalysisAgent(config=config_, client=client, logger=silent_logger())

    def test_analyze_returns_structured_result(self) -> None:
        transport = FakeTransport(deepseek_response(sample_ai_json()))
        agent = self._agent(transport)

        run = agent.analyze(self.result)

        self.assertFalse(run.dry_run)
        self.assertIsNotNone(run.analysis)
        self.assertEqual(run.attempts, 1)
        self.assertEqual(run.usage["total_tokens"], 1500)
        self.assertEqual(len(run.analysis.recommendations), 1)
        self.assertEqual(run.payload["core_metrics"]["project_count"], 6)

    def test_prompt_is_grounded_in_payload(self) -> None:
        transport = FakeTransport(deepseek_response(sample_ai_json()))
        agent = self._agent(transport)

        agent.analyze(self.result)

        messages = transport.last_request_json["messages"]
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]

        self.assertEqual(messages[0]["role"], "system")
        for rule in ("不得虚构数据", "不得修改或重新计算核心指标", "当前数据不足以判断", "只输出一个合法的 JSON"):
            self.assertIn(rule, system_prompt)
        self.assertIn("结构化分析结果", user_prompt)
        self.assertIn('"core_metrics"', user_prompt)
        self.assertIn(f'"project_count": {self.result.metrics.project_count}', user_prompt)
        self.assertIn(config.DEMO_EXCEL_FILE.name, user_prompt)

    def test_invalid_json_response_saves_raw_text(self) -> None:
        transport = FakeTransport(deepseek_response("这不是 JSON，只是普通文本"))
        agent = self._agent(transport)

        with mock.patch.object(config, "OUTPUT_DIR", Path(self._temp_dir())):
            with self.assertRaises(LLMResponseFormatError) as ctx:
                agent.analyze(self.result)

            from robotops import config as config_module

            raw_file = config_module.OUTPUT_DIR / RAW_RESPONSE_FILE
            self.assertTrue(raw_file.exists())
            self.assertIn("这不是 JSON", raw_file.read_text(encoding="utf-8"))
        self.assertIn("deepseek_raw_response.txt", ctx.exception.hint)

    def test_dry_run_does_not_call_api(self) -> None:
        transport = FakeTransport(deepseek_response(sample_ai_json()))
        agent = self._agent(transport)

        run = agent.dry_run(self.result)

        self.assertTrue(run.dry_run)
        self.assertIsNone(run.analysis)
        self.assertEqual(transport.calls, 0)
        self.assertEqual(len(run.messages), 2)

    def test_input_must_be_phase1_result(self) -> None:
        agent = self._agent(FakeTransport(deepseek_response(sample_ai_json())))

        for invalid in (None, pd.DataFrame(), [1, 2, 3]):
            with self.subTest(invalid=type(invalid).__name__):
                with self.assertRaises(ConfigurationError):
                    agent.analyze(invalid)  # type: ignore[arg-type]

    def _temp_dir(self) -> str:
        import tempfile

        if not hasattr(self, "_temp_handle"):
            self._temp_handle = tempfile.TemporaryDirectory()  # type: ignore[attr-defined]
            self.addCleanup(self._temp_handle.cleanup)  # type: ignore[attr-defined]
        return self._temp_handle.name  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main(verbosity=2)

