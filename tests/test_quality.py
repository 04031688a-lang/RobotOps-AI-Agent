"""Phase 11：数据质量门测试（正常 / 空数据 / 缺失字段 / 不可读文件）。

覆盖 Phase 8 质量门在异常输入下的稳定性：
1. 正常数据 —— 质量通过并完成分析；
2. 空数据 —— 返回错误级质量报告，不抛异常、不进入分析；
3. 缺失字段 —— 返回严重错误报告，阻止分析；
4. 不可读文件 —— 返回报告而非崩溃。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotops import config  # noqa: E402
from robotops.pipeline import run_analysis_with_quality  # noqa: E402
from tests._helpers import base_operation_frame, make_temp_dir, write_excel  # noqa: E402


class QualityGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = make_temp_dir(prefix="robotops_test_quality_")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -- 1) 正常数据 -------------------------------------------------------
    def test_normal_data_passes_and_analyzes(self) -> None:
        path = write_excel(base_operation_frame(), self.temp_dir / "normal.xlsx")

        outcome = run_analysis_with_quality(
            path, output_dir=self.temp_dir, verbose=False, export=False
        )

        self.assertTrue(outcome.analyzed)
        self.assertNotEqual(outcome.quality.status, config.QUALITY_STATUS_ERROR)
        self.assertIsNone(outcome.skipped_reason)
        self.assertIsNotNone(outcome.indicators)

    # -- 2) 空数据 ---------------------------------------------------------
    def test_empty_data_blocks_analysis_without_crash(self) -> None:
        path = write_excel(base_operation_frame().iloc[0:0], self.temp_dir / "empty.xlsx")

        outcome = run_analysis_with_quality(
            path, output_dir=self.temp_dir, verbose=False, export=False
        )

        self.assertFalse(outcome.analyzed)
        self.assertEqual(outcome.quality.status, config.QUALITY_STATUS_ERROR)
        self.assertIn("数据质量检查未通过", outcome.skipped_reason)

    # -- 3) 缺失字段 -------------------------------------------------------
    def test_missing_column_blocks_analysis(self) -> None:
        frame = base_operation_frame().drop(columns=[config.COL_ROBOT_ID])
        path = write_excel(frame, self.temp_dir / "missing.xlsx")

        outcome = run_analysis_with_quality(
            path, output_dir=self.temp_dir, verbose=False, export=False
        )

        self.assertFalse(outcome.analyzed)
        self.assertEqual(outcome.quality.status, config.QUALITY_STATUS_ERROR)
        self.assertIn(config.COL_ROBOT_ID, "；".join(outcome.quality.errors))

    # -- 4) 不可读文件 -----------------------------------------------------
    def test_unreadable_file_returns_report(self) -> None:
        outcome = run_analysis_with_quality(
            self.temp_dir / "不存在的文件.xlsx",
            output_dir=self.temp_dir,
            verbose=False,
            export=False,
        )

        self.assertFalse(outcome.analyzed)
        self.assertEqual(outcome.quality.status, config.QUALITY_STATUS_ERROR)
        self.assertIn("文件读取失败", "；".join(outcome.quality.errors))

    # -- 5) 轻度脏数据仍可分析 ---------------------------------------------
    def test_minor_dirty_data_still_analyzes(self) -> None:
        frame = base_operation_frame().copy()
        frame.loc[0, config.COL_SATISFACTION] = None  # 个别缺失，不构成严重问题

        path = write_excel(frame, self.temp_dir / "dirty.xlsx")
        outcome = run_analysis_with_quality(
            path, output_dir=self.temp_dir, verbose=False, export=False
        )

        self.assertTrue(outcome.analyzed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
