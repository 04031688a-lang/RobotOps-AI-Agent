"""Phase 11：统一异常处理测试。

覆盖需求要求的 11 类故障场景，并验证两条底线：

1. **不崩**：任何一类故障都不会把异常抛给用户（工作流/前端/命令行优雅降级）；
2. **不暴露 traceback**：面向用户的文案只有「错误信息 + 排查建议」，
   详细 traceback 只写入统一日志（output/robotops.log）。

场景清单：
    1. Excel 格式错误        2. 字段缺失           3. 数据类型错误
    4. API Key 不存在        5. API 调用失败       6. API 超时
    7. RAG 失败              8. LLM 返回空结果     9. Agent 节点失败
    10. SQLite 失败          11. 报告生成失败
"""

from __future__ import annotations

import logging
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.base import BaseAgent  # noqa: E402
from app.agents.diagnosis import DiagnosisAgent  # noqa: E402
from app.agents.rag_agent import RagAgent  # noqa: E402
from app.graph.logging_utils import WorkflowLogger  # noqa: E402
from app.graph.state import create_initial_state  # noqa: E402
from frontend import ui_helpers as ui  # noqa: E402
from robotops import config  # noqa: E402
from robotops import logger as unified_logger  # noqa: E402
from robotops.cli import main as cli_main  # noqa: E402
from robotops.data_loader import load_operation_data  # noqa: E402
from robotops.exceptions import (  # noqa: E402
    DataLoadError,
    DataValidationError,
    ReportGenerationError,
    describe_error,
)
from robotops.issues import IssueStorageError, IssueStore  # noqa: E402
from robotops.llm.client import DeepSeekClient  # noqa: E402
from robotops.llm.config import LLMConfig  # noqa: E402
from robotops.llm.errors import (  # noqa: E402
    LLMAuthError,
    LLMEmptyResponseError,
    LLMServerError,
    LLMTimeoutError,
    MissingAPIKeyError,
)
from robotops.llm.logger import setup_llm_logger  # noqa: E402
from robotops.pipeline import run_analysis, run_analysis_with_quality  # noqa: E402
from robotops.quality import check_data_quality  # noqa: E402
from robotops.rag.errors import KnowledgeBaseError  # noqa: E402
from scripts.generate_demo_data import generate_demo_file  # noqa: E402
from tests._helpers import (  # noqa: E402
    FakeTransport,
    base_operation_frame,
    deepseek_error_response,
    deepseek_response,
    make_temp_dir,
    write_excel,
)

TRACEBACK_MARKER = "Traceback"


def silent_llm_logger(name: str = "robotops.llm.phase11-tests") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    return logger


class Phase11ErrorTestCase(unittest.TestCase):
    """公共准备：临时目录 + 独立日志文件（不污染真实 output/robotops.log）。"""

    @classmethod
    def setUpClass(cls) -> None:
        if not config.DEMO_EXCEL_FILE.exists():
            generate_demo_file(config.DEMO_EXCEL_FILE)

    def setUp(self) -> None:
        self.temp_dir = make_temp_dir(prefix="robotops_test_phase11_")
        self.log_file = self.temp_dir / "robotops.log"
        unified_logger.setup_logger(log_file=self.log_file)

    def tearDown(self) -> None:
        unified_logger.setup_logger(log_file=self.temp_dir / "teardown.log")
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -- 工具 -------------------------------------------------------------
    def log_text(self) -> str:
        return unified_logger.read_log_file(self.log_file)

    def corrupt_excel(self) -> Path:
        path = self.temp_dir / "corrupt.xlsx"
        path.write_bytes(b"this is not a real excel file")
        return path

    def demo_result(self):
        return run_analysis(config.DEMO_EXCEL_FILE, export=False, verbose=False)

    def workflow_logger(self, name: str = "workflow.log") -> WorkflowLogger:
        return WorkflowLogger(
            verbose=False, echo=False, log_file=self.temp_dir / name
        )

    def build_client(self, transport: FakeTransport, *, max_retries: int = 0) -> DeepSeekClient:
        return DeepSeekClient(
            config=LLMConfig(api_key="sk-test-key", max_retries=max_retries),
            transport=transport,
            logger=setup_llm_logger(level="WARNING", console=False),
            sleep=lambda seconds: None,
        )

    def assert_friendly(self, error: BaseException) -> str:
        """断言面向用户的文案友好：有中文提示、无 traceback。"""

        text = describe_error(error)
        self.assertTrue(text.strip())
        self.assertNotIn(TRACEBACK_MARKER, text)
        self.assertNotIn("File \"", text)
        return text


# ---------------------------------------------------------------------------
# 1. Excel 格式错误
# ---------------------------------------------------------------------------
class ExcelFormatErrorTests(Phase11ErrorTestCase):
    def test_corrupt_excel_raises_typed_error(self) -> None:
        with self.assertRaises(DataLoadError) as ctx:
            load_operation_data(self.corrupt_excel())

        self.assert_friendly(ctx.exception)

    def test_corrupt_excel_frontend_preview_is_friendly(self) -> None:
        preview = ui.preview_uploaded_file(self.corrupt_excel())

        self.assertFalse(preview.is_readable)
        self.assertTrue(preview.error_message)
        self.assertNotIn(TRACEBACK_MARKER, preview.error_message)

    def test_corrupt_excel_quality_gate_does_not_crash(self) -> None:
        outcome = run_analysis_with_quality(
            self.corrupt_excel(), output_dir=self.temp_dir, export=False, verbose=False
        )

        self.assertFalse(outcome.analyzed)
        self.assertEqual(outcome.quality.status, config.QUALITY_STATUS_ERROR)
        self.assertIn("文件读取失败", "；".join(outcome.quality.errors))
        # 用户不看到 traceback，日志里保留详细错误
        self.assertNotIn(TRACEBACK_MARKER, "；".join(outcome.quality.errors))
        self.assertIn(TRACEBACK_MARKER, self.log_text())


# ---------------------------------------------------------------------------
# 2 / 3. 字段缺失与数据类型错误
# ---------------------------------------------------------------------------
class FieldAndTypeErrorTests(Phase11ErrorTestCase):
    def test_missing_column_raises_data_validation_error(self) -> None:
        frame = base_operation_frame().drop(columns=[config.COL_ROBOT_ID])
        path = write_excel(frame, self.temp_dir / "missing.xlsx")

        with self.assertRaises(DataValidationError) as ctx:
            load_operation_data(path)

        text = self.assert_friendly(ctx.exception)
        self.assertIn(config.COL_ROBOT_ID, text)

    def test_missing_column_blocks_analysis_without_crash(self) -> None:
        frame = base_operation_frame().drop(columns=[config.COL_COST])
        path = write_excel(frame, self.temp_dir / "missing_cost.xlsx")

        outcome = run_analysis_with_quality(
            path, output_dir=self.temp_dir, export=False, verbose=False
        )

        self.assertFalse(outcome.analyzed)
        self.assertIn(config.COL_COST, outcome.quality.missing_required_columns)

    def test_invalid_numeric_type_is_reported_and_cleaned(self) -> None:
        frame = base_operation_frame()
        frame[config.COL_RUNNING_HOURS] = frame[config.COL_RUNNING_HOURS].astype(object)
        frame.loc[0, config.COL_RUNNING_HOURS] = "八小时"
        path = write_excel(frame, self.temp_dir / "bad_type.xlsx")

        # 质量检查能识别非法数值
        raw = load_operation_data(path, validate=False)
        report = check_data_quality(raw.data)
        self.assertGreaterEqual(report.invalid_values, 1)

        # 分析流程不崩溃（清洗阶段按规则置空/记录问题）
        result = run_analysis(path, output_dir=self.temp_dir, export=False, verbose=False)
        self.assertGreater(result.metrics.record_count, 0)
        self.assertGreaterEqual(len(result.cleaned.issues), 1)


# ---------------------------------------------------------------------------
# 4 / 5 / 6 / 8. API Key、API 调用失败、超时、空返回
# ---------------------------------------------------------------------------
class LLMErrorTests(Phase11ErrorTestCase):
    def test_missing_api_key_has_hint(self) -> None:
        with self.assertRaises(MissingAPIKeyError) as ctx:
            LLMConfig(api_key=None).validate(require_api_key=True)

        text = self.assert_friendly(ctx.exception)
        self.assertIn("DEEPSEEK_API_KEY", text)

    def test_api_auth_failure_maps_to_typed_error(self) -> None:
        client = self.build_client(FakeTransport(deepseek_error_response(401, "Authentication Fails")))

        with self.assertRaises(LLMAuthError) as ctx:
            client.chat([{"role": "user", "content": "hi"}])

        text = self.assert_friendly(ctx.exception)
        self.assertIn("排查建议", text)

    def test_api_server_failure_maps_to_typed_error(self) -> None:
        client = self.build_client(FakeTransport(deepseek_error_response(500, "server error")))

        with self.assertRaises(LLMServerError) as ctx:
            client.chat([{"role": "user", "content": "hi"}])

        self.assert_friendly(ctx.exception)
        self.assertIn("DeepSeek API call failed", self.log_text())

    def test_api_timeout_maps_to_typed_error(self) -> None:
        client = self.build_client(FakeTransport(LLMTimeoutError("请求 DeepSeek 超时")))

        with self.assertRaises(LLMTimeoutError) as ctx:
            client.chat([{"role": "user", "content": "hi"}])

        text = self.assert_friendly(ctx.exception)
        self.assertIn("DEEPSEEK_TIMEOUT", text)

    def test_empty_llm_response_maps_to_typed_error(self) -> None:
        client = self.build_client(FakeTransport(deepseek_response("")))

        with self.assertRaises(LLMEmptyResponseError) as ctx:
            client.chat([{"role": "user", "content": "hi"}])

        self.assert_friendly(ctx.exception)

    def test_agent_records_llm_failure_and_degrades(self) -> None:
        state = dict(create_initial_state(data_path=str(config.DEMO_EXCEL_FILE)))
        state["use_llm"] = True
        client = self.build_client(FakeTransport(deepseek_response("")))

        update = DiagnosisAgent(logger=self.workflow_logger("diag.log"), client=client).run(state)

        errors = update.get("errors") or []
        self.assertTrue(errors)
        self.assertTrue(errors[0]["is_llm_error"])
        self.assertNotIn(TRACEBACK_MARKER, errors[0]["message"])
        self.assertIn("diagnosis", update.get("failed_stages") or [])


# ---------------------------------------------------------------------------
# 7. RAG 失败
# ---------------------------------------------------------------------------
class RagErrorTests(Phase11ErrorTestCase):
    def test_rag_failure_is_recorded_and_workflow_continues(self) -> None:
        state = dict(create_initial_state(data_path=str(config.DEMO_EXCEL_FILE)))
        state["analysis_result"] = self.demo_result()

        retriever = mock.MagicMock()
        retriever.ensure_index.side_effect = KnowledgeBaseError("向量库不可用（测试构造）")

        agent = RagAgent(logger=self.workflow_logger("rag.log"), retriever=retriever)
        update = agent.run(state)

        errors = update.get("errors") or []
        self.assertTrue(errors)
        self.assertTrue(errors[0]["is_rag_error"])
        self.assertNotIn(TRACEBACK_MARKER, errors[0]["message"])
        self.assertIn("rag", update.get("failed_stages") or [])

    def test_rag_error_message_is_friendly(self) -> None:
        error = KnowledgeBaseError("知识库目录为空")

        text = self.assert_friendly(error)
        self.assertIn("排查建议", text)
        self.assertIn("knowledge", text)


# ---------------------------------------------------------------------------
# 9. Agent 节点失败
# ---------------------------------------------------------------------------
class _BoomAgent(BaseAgent):
    tag = "Boom Agent"
    stage = "boom"

    def execute(self, state):  # pragma: no cover - 仅用于测试异常路径
        raise ValueError("节点内部错误（测试构造）")


class AgentNodeFailureTests(Phase11ErrorTestCase):
    def test_agent_node_failure_is_recorded_without_raising(self) -> None:
        agent = _BoomAgent(logger=self.workflow_logger("boom.log"))

        update = agent.run({})

        errors = update.get("errors") or []
        self.assertTrue(errors)
        self.assertEqual(errors[0]["kind"], "未预期错误")
        self.assertNotIn(TRACEBACK_MARKER, errors[0]["message"])
        self.assertEqual(update.get("failed_stages"), ["boom"])
        self.assertEqual(update["steps"][-1]["status"], "failed")


# ---------------------------------------------------------------------------
# 10. SQLite 失败
# ---------------------------------------------------------------------------
class SqliteErrorTests(Phase11ErrorTestCase):
    def test_store_open_failure_is_wrapped(self) -> None:
        blocker = self.temp_dir / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")

        with self.assertRaises(IssueStorageError) as ctx:
            IssueStore(blocker / "issues.db")

        text = self.assert_friendly(ctx.exception)
        self.assertIn("issues.db", text)


# ---------------------------------------------------------------------------
# 11. 报告生成失败
# ---------------------------------------------------------------------------
class ReportErrorTests(Phase11ErrorTestCase):
    def test_report_export_failure_is_wrapped_and_logged(self) -> None:
        result = self.demo_result()

        with mock.patch(
            "robotops.report._export_results", side_effect=OSError("磁盘已满（测试构造）")
        ):
            with self.assertRaises(ReportGenerationError) as ctx:
                from robotops.report import export_results

                export_results(result, self.temp_dir / "out")

        text = self.assert_friendly(ctx.exception)
        self.assertIn("排查建议", text)
        self.assertIn(TRACEBACK_MARKER, self.log_text())


# ---------------------------------------------------------------------------
# 用户侧文案：命令行不打印 traceback
# ---------------------------------------------------------------------------
class UserFacingMessageTests(Phase11ErrorTestCase):
    def test_cli_prints_friendly_message_without_traceback(self) -> None:
        import contextlib
        import io

        path = self.corrupt_excel()
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_main(["--data", str(path), "--no-export", "--quiet"])

        self.assertEqual(code, 2)
        stderr_text = err.getvalue()
        self.assertIn("[分析失败]", stderr_text)
        self.assertNotIn(TRACEBACK_MARKER, stderr_text)
        self.assertNotIn("File \"", stderr_text)
        # 详细错误保留在日志里
        self.assertIn(TRACEBACK_MARKER, self.log_text())

    def test_describe_error_never_dumps_traceback(self) -> None:
        for error in (
            DataLoadError("读取 Excel 失败：坏文件"),
            DataValidationError("数据文件缺少必需列：运营成本"),
            ReportGenerationError("报告生成失败"),
            RuntimeError("未知错误"),
        ):
            with self.subTest(error=type(error).__name__):
                text = describe_error(error)
                self.assertNotIn(TRACEBACK_MARKER, text)
                self.assertNotIn("File \"", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

