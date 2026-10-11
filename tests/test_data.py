"""Phase 11：数据输入场景测试（正常 / 异常 / 空数据 / 缺失字段）。

按需求以文件级用例覆盖四类数据场景：
1. 正常数据 —— 完整跑通读取 → 清洗 → 指标 → 异常识别，指标合理；
2. 异常数据 —— 满意度 / 故障率超阈值，能检出异常项目；
3. 空数据 —— 只有表头没有记录，给出明确错误（DataLoadError）；
4. 缺失字段 —— 缺必填列，给出明确错误（DataValidationError）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from robotops import config  # noqa: E402
from robotops.data_loader import load_operation_data  # noqa: E402
from robotops.exceptions import DataLoadError, DataValidationError  # noqa: E402
from robotops.pipeline import run_analysis  # noqa: E402
from tests._helpers import base_operation_frame, make_temp_dir, write_excel  # noqa: E402


def abnormal_frame() -> pd.DataFrame:
    """构造一份包含明显异常的运营数据（满意度 / 运行率 / 节降率均超阈值）。"""

    rows = []
    for day in range(1, 4):
        rows.append(
            {
                config.COL_DATE: f"2026-09-{day:02d}",
                config.COL_PROJECT: "异常项目",
                config.COL_ROBOT_ID: "BAD-01",
                config.COL_ROBOT_TYPE: "清洁机器人",
                config.COL_RUNNING_HOURS: 5.0,      # 运行率 50% < 80%
                config.COL_PLANNED_HOURS: 10.0,
                config.COL_FAULT_COUNT: 2,
                config.COL_INSPECTION_COUNT: 10,    # 周期故障率 20% > 5%
                config.COL_REPAIR_COUNT: 2,
                config.COL_SATISFACTION: 70.0,      # < 85
                config.COL_COST: 300.0,
                config.COL_SAVING_RATE: 2.0,        # < 10
            }
        )
    return pd.DataFrame(rows)


class DataScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = make_temp_dir(prefix="robotops_test_data_")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -- 1) 正常数据 -------------------------------------------------------
    def test_normal_data_full_pipeline(self) -> None:
        path = write_excel(base_operation_frame(), self.temp_dir / "normal.xlsx")

        result = run_analysis(path, output_dir=self.temp_dir, verbose=False, export=False)

        self.assertEqual(result.metrics.project_count, 2)
        self.assertEqual(result.metrics.robot_count, 3)
        # base_operation_frame 中 R-02 满意度 80 < 85、节降率 10 未低于 10：
        # 满意度至少命中 1 条异常
        self.assertGreaterEqual(result.anomaly_count, 1)

    # -- 2) 异常数据 -------------------------------------------------------
    def test_abnormal_data_detected(self) -> None:
        path = write_excel(abnormal_frame(), self.temp_dir / "abnormal.xlsx")

        result = run_analysis(path, output_dir=self.temp_dir, verbose=False, export=False)

        self.assertTrue(result.anomaly_count >= 3)
        abnormal_projects = set(result.anomaly_project_summary["项目名称"].tolist())
        self.assertIn("异常项目", abnormal_projects)
        # 三类异常都应出现
        anomaly_types = set(result.anomaly_summary["异常类型"].tolist())
        self.assertTrue({"满意度偏低", "运行率偏低"} <= anomaly_types)

    # -- 3) 空数据 ---------------------------------------------------------
    def test_empty_data_raises_friendly_error(self) -> None:
        path = write_excel(base_operation_frame().iloc[0:0], self.temp_dir / "empty.xlsx")

        with self.assertRaises(DataLoadError) as ctx:
            load_operation_data(path)

        self.assertIn("没有可读取的记录", str(ctx.exception))

    def test_empty_frame_via_pipeline_raises(self) -> None:
        path = write_excel(base_operation_frame().iloc[0:0], self.temp_dir / "empty2.xlsx")

        with self.assertRaises(DataLoadError):
            run_analysis(path, output_dir=self.temp_dir, verbose=False, export=False)

    # -- 4) 缺失字段 -------------------------------------------------------
    def test_missing_required_column_raises(self) -> None:
        frame = base_operation_frame().drop(columns=[config.COL_ROBOT_ID])
        path = write_excel(frame, self.temp_dir / "missing.xlsx")

        with self.assertRaises(DataValidationError) as ctx:
            load_operation_data(path)

        self.assertIn(config.COL_ROBOT_ID, str(ctx.exception))

    def test_missing_satisfaction_column_raises(self) -> None:
        frame = base_operation_frame().drop(columns=[config.COL_SATISFACTION])
        path = write_excel(frame, self.temp_dir / "missing2.xlsx")

        with self.assertRaises(DataValidationError):
            run_analysis(path, output_dir=self.temp_dir, verbose=False, export=False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
