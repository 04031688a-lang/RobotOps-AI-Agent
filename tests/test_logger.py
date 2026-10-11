"""Phase 11：统一日志模块测试。

覆盖：
- 日志格式与需求示例一致（``2026-09-24 10:21:31 INFO Data analysis started``）；
- 文件写入 / 幂等初始化 / 控制台开关；
- ``log_exception`` 写入 traceback，但消息里同时保留错误摘要；
- pipeline / issue 等接入点真正产出业务事件。
"""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotops import logger as unified_logger  # noqa: E402
from robotops.exceptions import DataValidationError  # noqa: E402
from tests._helpers import base_operation_frame, make_temp_dir  # noqa: E402


class UnifiedLoggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = make_temp_dir(prefix="robotops_test_logger_")
        self.log_file = self.temp_dir / "robotops.log"

    def tearDown(self) -> None:
        # 恢复默认配置，避免影响其他测试
        unified_logger.setup_logger(log_file=self.temp_dir / "teardown.log")
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _read(self) -> str:
        return unified_logger.read_log_file(self.log_file)

    def test_event_format_matches_requirement(self) -> None:
        """格式：``2026-09-24 10:21:31 INFO Data analysis started``。"""

        unified_logger.setup_logger(log_file=self.log_file)
        unified_logger.log_event("Data analysis started")

        content = self._read()
        # 形如 ``2026-10-11 12:34:56 INFO Data analysis started``
        self.assertRegex(content.strip(), r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO Data analysis started$")

    def test_event_with_args(self) -> None:
        unified_logger.setup_logger(log_file=self.log_file)
        unified_logger.log_event("%s robots loaded", 86)

        self.assertIn("86 robots loaded", self._read())

    def test_idempotent_setup(self) -> None:
        """重复初始化不会重复添加处理器（日志不重复写两遍）。"""

        unified_logger.setup_logger(log_file=self.log_file)
        unified_logger.setup_logger(log_file=self.log_file)
        unified_logger.log_event("only once")

        self.assertEqual(self._read().strip().count("only once"), 1)

    def test_console_output_toggle(self) -> None:
        unified_logger.setup_logger(log_file=self.log_file, console=True)
        # console=True 时应存在 StreamHandler
        handlers = logging.getLogger(unified_logger.LOGGER_NAME).handlers
        self.assertTrue(any(isinstance(h, logging.StreamHandler) for h in handlers))

    def test_log_error_and_exception(self) -> None:
        unified_logger.setup_logger(log_file=self.log_file)
        unified_logger.log_error("DeepSeek API call failed: timeout")

        try:
            raise DataValidationError("缺少必填列：机器人ID")
        except DataValidationError as error:
            unified_logger.log_exception(error, context="run_analysis")

        content = self._read()
        self.assertIn("ERROR DeepSeek API call failed: timeout", content)
        self.assertIn("DataValidationError", content)
        self.assertIn("run_analysis", content)
        # traceback 详情（文件名）也应进入日志
        self.assertIn("test_logger.py", content)

    def test_read_log_file_missing_returns_empty(self) -> None:
        self.assertEqual(unified_logger.read_log_file(self.temp_dir / "nope.log"), "")

    def test_unwritable_log_file_does_not_raise(self) -> None:
        """日志文件不可写时不应影响主流程。"""

        # Windows 上不存在目录且无权限创建 -> FileHandler 失败 -> 退化为静默
        unified_logger.setup_logger(log_file="Z:/definitely/not/exist/robotops.log")
        unified_logger.log_event("still alive")
        # 不抛异常即通过


class PipelineLoggingTests(unittest.TestCase):
    """pipeline 接入点应产出需求要求的业务事件。"""

    def setUp(self) -> None:
        self.temp_dir = make_temp_dir(prefix="robotops_test_logger_")
        self.log_file = self.temp_dir / "robotops.log"
        unified_logger.setup_logger(log_file=self.log_file)

        from robotops.pipeline import run_analysis
        from tests._helpers import write_excel

        self.run_analysis = run_analysis
        self.write_excel = write_excel
        self.data_file = self.temp_dir / "data.xlsx"
        self.write_excel(base_operation_frame(), self.data_file)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_emits_required_events(self) -> None:
        result = self.run_analysis(
            self.data_file, output_dir=self.temp_dir, verbose=False, export=False
        )

        content = unified_logger.read_log_file(self.log_file)
        self.assertIn("Data analysis started", content)
        self.assertIn("Data loaded:", content)
        self.assertIn("Data cleaning completed:", content)
        self.assertIn("Metrics calculated:", content)
        self.assertIn("Anomaly detection completed:", content)
        # 异常项目数应为真实计算值
        self.assertIn(f"{len(result.anomaly_project_summary)} abnormal projects detected", content)

    def test_report_generation_event(self) -> None:
        self.run_analysis(self.data_file, output_dir=self.temp_dir, verbose=False)

        content = unified_logger.read_log_file(self.log_file)
        self.assertIn("Report generated:", content)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
