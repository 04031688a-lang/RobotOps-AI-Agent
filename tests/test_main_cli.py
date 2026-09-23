"""Phase 2：命令行入口测试（干跑、缺少 Key、API 失败、成功输出与导出）。"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as main_module  # noqa: E402
from robotops import config  # noqa: E402
from robotops.agent.robot_ops_agent import AI_OUTPUT_BASENAME, AI_PAYLOAD_FILE, RobotOpsAnalysisAgent  # noqa: E402
from robotops.llm import config as llm_config  # noqa: E402
from robotops.llm.client import DeepSeekClient  # noqa: E402
from robotops.llm.config import LLMConfig  # noqa: E402
from robotops.rag import config as rag_config  # noqa: E402
from scripts.generate_demo_data import generate_demo_file  # noqa: E402
from tests._helpers import (  # noqa: E402
    FakeTransport,
    cleanup_dir,
    deepseek_error_response,
    deepseek_response,
    make_temp_chroma_dir,
    release_chroma_client,
    sample_ai_json,
)


def silent_logger(name: str = "robotops.llm.main-tests") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    return logger


class MainCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not config.DEMO_EXCEL_FILE.exists():
            generate_demo_file(config.DEMO_EXCEL_FILE)
        cls.data_args = ["--data", str(config.DEMO_EXCEL_FILE)]
        cls._temp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._temp.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def setUp(self) -> None:
        self._snapshot = dict(os.environ)
        os.environ.pop(llm_config.ENV_API_KEY, None)
        # RAG 使用临时向量库目录，避免污染项目的 data/chroma
        self._chroma_tmp = make_temp_chroma_dir()
        os.environ[rag_config.ENV_CHROMA_DIR] = str(self._chroma_tmp)
        # 阻断真实 .env，保证测试不会误用真实 API Key（也不会发起真实调用）
        self._patches = [
            mock.patch.object(llm_config, "ensure_env_loaded", return_value=None),
            mock.patch.object(rag_config, "ensure_env_loaded", return_value=None),
        ]
        for patch in self._patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in self._patches:
            patch.stop()
        os.environ.clear()
        os.environ.update(self._snapshot)
        release_chroma_client(None)  # 释放 sqlite 句柄后再删除临时向量库目录
        cleanup_dir(self._chroma_tmp)

    # -- 工具 -------------------------------------------------------------
    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main_module.main(argv)
        return code, out.getvalue(), err.getvalue()

    def _patched_agent(self, transport: FakeTransport):
        config_ = LLMConfig(api_key="sk-test-key", max_retries=0)
        client = DeepSeekClient(
            config=config_, transport=transport, logger=silent_logger(), sleep=lambda s: None
        )
        agent = RobotOpsAnalysisAgent(config=config_, client=client, logger=silent_logger())
        return mock.patch.object(
            main_module, "create_agent", side_effect=lambda args, dry_run=False: agent
        )

    # -- 用例 -------------------------------------------------------------
    def test_dry_run_works_without_api_key(self) -> None:
        with mock.patch.object(llm_config, "ensure_env_loaded", return_value=None):
            code, out, _ = self._run([*self.data_args, "--dry-run", "--quiet", "--no-export"])

        self.assertEqual(code, 0)
        self.assertIn("干跑模式", out)
        self.assertIn("robot-ops-analysis-v1", out)
        self.assertIn('"core_metrics"', out)

    def test_missing_api_key_returns_input_error(self) -> None:
        with mock.patch.object(llm_config, "ensure_env_loaded", return_value=None):
            code, _, err = self._run([*self.data_args, "--quiet", "--no-export"])

        self.assertEqual(code, 2)
        self.assertIn(llm_config.ENV_API_KEY, err)
        self.assertIn(".env", err)
        self.assertIn("--dry-run", err)

    def test_llm_auth_failure_returns_llm_error(self) -> None:
        transport = FakeTransport(deepseek_error_response(401, "Authentication Fails"))
        with self._patched_agent(transport):
            code, _, err = self._run([*self.data_args, "--quiet", "--no-export"])

        self.assertEqual(code, 3)
        self.assertIn("认证失败", err)
        self.assertIn("DEEPSEEK_API_KEY", err)

    def test_timeout_failure_returns_llm_error(self) -> None:
        from robotops.llm.errors import LLMTimeoutError

        transport = FakeTransport(LLMTimeoutError("请求 DeepSeek 超时（60 秒）"))
        with self._patched_agent(transport):
            code, _, err = self._run([*self.data_args, "--quiet", "--no-export"])

        self.assertEqual(code, 3)
        self.assertIn("超时", err)
        self.assertIn("DEEPSEEK_TIMEOUT", err)

    def test_success_path_prints_structured_sections(self) -> None:
        transport = FakeTransport(deepseek_response(sample_ai_json()))
        with self._patched_agent(transport):
            code, out, _ = self._run([*self.data_args, "--quiet", "--no-export"])

        self.assertEqual(code, 0)
        for title in ("运营概览", "关键发现", "异常发现", "可能原因", "优化建议"):
            self.assertIn(title, out)

    def test_json_output_mode(self) -> None:
        transport = FakeTransport(deepseek_response(sample_ai_json()))
        with self._patched_agent(transport):
            code, out, _ = self._run([*self.data_args, "--quiet", "--no-export", "--json"])

        self.assertEqual(code, 0)
        data = json.loads(out[out.index("{") : out.rindex("}") + 1])
        self.assertEqual(
            set(data.keys()),
            {"overview", "key_findings", "abnormal_projects", "possible_reasons", "recommendations", "meta"},
        )

    def test_exports_ai_outputs_and_phase1_files(self) -> None:
        output_dir = self.tmp / "out"
        transport = FakeTransport(deepseek_response(sample_ai_json()))
        with self._patched_agent(transport):
            code, _, _ = self._run(
                [*self.data_args, "--quiet", "--output", str(output_dir)]
            )

        self.assertEqual(code, 0)
        self.assertTrue((output_dir / AI_PAYLOAD_FILE).exists())
        self.assertTrue((output_dir / f"{AI_OUTPUT_BASENAME}.json").exists())
        self.assertTrue((output_dir / f"{AI_OUTPUT_BASENAME}.md").exists())
        # Phase 1 的结果文件仍然正常导出
        self.assertTrue((output_dir / "robot_ops_report.xlsx").exists())
        payload = json.loads((output_dir / AI_PAYLOAD_FILE).read_text(encoding="utf-8"))
        self.assertEqual(payload["payload_version"], "robot-ops-analysis-v1")
        markdown = (output_dir / f"{AI_OUTPUT_BASENAME}.md").read_text(encoding="utf-8")
        self.assertIn("运营概览", markdown)

    def test_show_prompt_prints_rules(self) -> None:
        transport = FakeTransport(deepseek_response(sample_ai_json()))
        with self._patched_agent(transport):
            code, out, _ = self._run(
                [*self.data_args, "--quiet", "--no-export", "--show-prompt"]
            )

        self.assertEqual(code, 0)
        self.assertIn("发送给 DeepSeek 的 Prompt", out)
        self.assertIn("不得虚构数据", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
