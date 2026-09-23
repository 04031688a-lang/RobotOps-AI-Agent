"""Phase 4：工作流命令行入口测试（退出码、导出文件、降级提示）。"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_workflow as workflow_cli  # noqa: E402
from app.graph.workflow import RobotOpsWorkflow  # noqa: E402
from robotops import config  # noqa: E402
from robotops.llm import config as llm_config  # noqa: E402
from robotops.llm.client import DeepSeekClient  # noqa: E402
from robotops.llm.config import LLMConfig  # noqa: E402
from robotops.rag import config as rag_config  # noqa: E402
from scripts.generate_demo_data import generate_demo_file  # noqa: E402
from tests._helpers import (  # noqa: E402
    FakeTransport,
    cleanup_dir,
    deepseek_error_response,
    make_temp_dir,
    release_chroma_client,
)


def silent_logger(name: str = "robotops.llm.workflow-cli-tests") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    return logger


class WorkflowCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not config.DEMO_EXCEL_FILE.exists():
            generate_demo_file(config.DEMO_EXCEL_FILE)

    def setUp(self) -> None:
        self._snapshot = dict(os.environ)
        os.environ.pop(llm_config.ENV_API_KEY, None)
        self.temp_dir = make_temp_dir(prefix="robotops_test_workflow_cli_")
        os.environ[rag_config.ENV_CHROMA_DIR] = str(self.temp_dir / "chroma")
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
        release_chroma_client(None)  # 先释放 sqlite 句柄，再删除临时目录（Windows）
        cleanup_dir(self.temp_dir)

    # -- 工具 -------------------------------------------------------------
    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = workflow_cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def _patch_workflow_with(self, transport: FakeTransport):
        fake_client = DeepSeekClient(
            config=LLMConfig(api_key="sk-test-key", max_retries=0),
            transport=transport,
            logger=silent_logger(),
            sleep=lambda seconds: None,
        )

        def fake_create(args, *, client=None, retriever=None):  # noqa: ARG001
            return RobotOpsWorkflow(
                client=fake_client,  # 注入假客户端（不发真实请求）
                retriever=retriever,
                verbose=False,
                echo=False,
                log_file=Path(args.output_dir or config.OUTPUT_DIR) / "workflow.log",
            )

        return mock.patch.object(workflow_cli, "create_workflow", side_effect=fake_create)

    # -- 用例 -------------------------------------------------------------
    def test_cli_offline_success_exports_files(self) -> None:
        output_dir = self.temp_dir / "out"

        code, out, _ = self._run(
            ["--no-llm", "--quiet", "--output", str(output_dir), "--data", str(config.DEMO_EXCEL_FILE)]
        )

        self.assertEqual(code, 0)
        self.assertIn("多 Agent 工作流结果", out)
        self.assertIn("工作流状态：completed", out)
        for name in ("workflow_report.md", "workflow_state.json", "workflow_steps.json"):
            self.assertTrue((output_dir / name).exists(), f"缺少 {name}")
        report = (output_dir / "workflow_report.md").read_text(encoding="utf-8")
        self.assertIn("## 三、异常发现", report)

    def test_cli_missing_api_key_exits_with_input_error(self) -> None:
        code, _, err = self._run(["--quiet", "--no-export", "--data", str(config.DEMO_EXCEL_FILE)])

        self.assertEqual(code, 2)
        self.assertIn(llm_config.ENV_API_KEY, err)
        self.assertIn("--no-llm", err)

    def test_cli_llm_failure_exits_with_llm_error(self) -> None:
        os.environ[llm_config.ENV_API_KEY] = "sk-test-key"  # 通过预检，随后由假客户端报 401
        transport = FakeTransport(deepseek_error_response(401, "Authentication Fails"))

        with self._patch_workflow_with(transport):
            code, out, err = self._run(
                ["--quiet", "--no-export", "--data", str(config.DEMO_EXCEL_FILE)]
            )

        self.assertEqual(code, 3)
        self.assertIn("认证失败", err)
        self.assertIn("DEEPSEEK_API_KEY", err)
        self.assertIn("错误汇总", err)
        self.assertIn("工作流状态：completed_with_errors", out)

    def test_cli_invalid_data_path_exits_with_input_error(self) -> None:
        code, _, err = self._run(
            ["--no-llm", "--quiet", "--no-export", "--data", str(self.temp_dir / "missing.xlsx")]
        )

        self.assertEqual(code, 2)
        self.assertIn("未找到数据文件", err)
        self.assertNotIn("Traceback", err)

    def test_cli_rag_unavailable_degrades_without_failure(self) -> None:
        empty_knowledge = self.temp_dir / "empty_knowledge"
        empty_knowledge.mkdir()
        os.environ[rag_config.ENV_KNOWLEDGE_DIR] = str(empty_knowledge)

        code, out, err = self._run(
            ["--no-llm", "--quiet", "--no-export", "--data", str(config.DEMO_EXCEL_FILE)]
        )

        self.assertEqual(code, 0, "知识库不可用时工作流应降级完成")
        self.assertIn("已跳过 RAG 知识库", err)
        self.assertIn("工作流状态：completed", out)

    def test_cli_json_mode_outputs_state(self) -> None:
        code, out, _ = self._run(
            ["--no-llm", "--quiet", "--json", "--no-export", "--data", str(config.DEMO_EXCEL_FILE)]
        )

        self.assertEqual(code, 0)
        data = json.loads(out[out.index("{") : out.rindex("}") + 1])
        self.assertEqual(data["workflow_status"], "completed")
        for key in ("raw_data_summary", "metrics", "abnormal_projects", "final_report"):
            self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
