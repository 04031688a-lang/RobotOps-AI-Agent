"""指标计算模块测试（使用可控的小数据集校验精确数值）。"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from robotops import config  # noqa: E402
from robotops.data_cleaner import clean_operation_data  # noqa: E402
from robotops.metrics import calculate_core_metrics, project_level_summary  # noqa: E402
from tests._helpers import base_operation_frame  # noqa: E402


def metrics_frame() -> pd.DataFrame:
    """3 条记录、2 个项目、3 台机器人，数值便于手算核对。"""

    return pd.DataFrame(
        [
            {
                config.COL_DATE: "2026-08-01",
                config.COL_PROJECT: "项目甲",
                config.COL_ROBOT_ID: "R-01",
                config.COL_ROBOT_TYPE: "清洁机器人",
                config.COL_RUNNING_HOURS: 10.0,
                config.COL_PLANNED_HOURS: 20.0,
                config.COL_FAULT_COUNT: 1,
                config.COL_INSPECTION_COUNT: 10,
                config.COL_REPAIR_COUNT: 0,
                config.COL_SATISFACTION: 90.0,
                config.COL_COST: 100.0,
                config.COL_SAVING_RATE: 20.0,
            },
            {
                config.COL_DATE: "2026-08-01",
                config.COL_PROJECT: "项目甲",
                config.COL_ROBOT_ID: "R-02",
                config.COL_ROBOT_TYPE: "巡检机器人",
                config.COL_RUNNING_HOURS: 20.0,
                config.COL_PLANNED_HOURS: 20.0,
                config.COL_FAULT_COUNT: 2,
                config.COL_INSPECTION_COUNT: 10,
                config.COL_REPAIR_COUNT: 1,
                config.COL_SATISFACTION: 80.0,
                config.COL_COST: 200.0,
                config.COL_SAVING_RATE: 10.0,
            },
            {
                config.COL_DATE: "2026-08-02",
                config.COL_PROJECT: "项目乙",
                config.COL_ROBOT_ID: "R-03",
                config.COL_ROBOT_TYPE: "配送机器人",
                config.COL_RUNNING_HOURS: None,  # 缺失：不计入平均运行时长
                config.COL_PLANNED_HOURS: 20.0,
                config.COL_FAULT_COUNT: 0,
                config.COL_INSPECTION_COUNT: 0,  # 分母为 0：故障率跳过
                config.COL_REPAIR_COUNT: 0,
                config.COL_SATISFACTION: None,  # 缺失：不计入平均满意度
                config.COL_COST: 500.0,
                config.COL_SAVING_RATE: None,  # 缺失：不计入平均节降率
            },
        ]
    )


class MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = clean_operation_data(metrics_frame()).data
        self.summary = calculate_core_metrics(self.data)

    def test_core_metrics_values(self) -> None:
        self.assertEqual(self.summary.project_count, 2)
        self.assertEqual(self.summary.robot_count, 3)
        self.assertAlmostEqual(self.summary.avg_running_hours, 15.0, places=6)
        # 故障率 = (1 + 2) / (10 + 10) × 100 = 15%
        self.assertAlmostEqual(self.summary.fault_rate, 15.0, places=6)
        self.assertAlmostEqual(self.summary.avg_satisfaction, 85.0, places=6)
        self.assertAlmostEqual(self.summary.total_cost, 800.0, places=6)
        self.assertAlmostEqual(self.summary.avg_saving_rate, 15.0, places=6)
        self.assertEqual(self.summary.record_count, 3)
        self.assertEqual(self.summary.total_fault_count, 3)
        self.assertAlmostEqual(self.summary.avg_uptime_rate, (50.0 + 100.0) / 2, places=6)

    def test_metrics_rows_cover_all_required_metrics(self) -> None:
        rows = {row["指标"]: row for row in self.summary.as_rows()}
        required = [
            "项目数量",
            "机器人数量",
            "平均运行时长",
            "故障率",
            "平均满意度",
            "总运营成本",
            "平均节降率",
        ]
        for name in required:
            self.assertIn(name, rows)
            self.assertEqual(rows[name]["指标类型"], "核心指标")
        self.assertEqual(rows["项目数量"]["数值"], 2)
        self.assertEqual(rows["总运营成本"]["数值"], 800.0)

    def test_project_level_summary(self) -> None:
        summary = project_level_summary(self.data)

        self.assertEqual(list(summary["项目名称"]), ["项目乙", "项目甲"])  # 按总成本倒序
        project_a = summary.loc[summary["项目名称"] == "项目甲"].iloc[0]
        self.assertEqual(project_a["机器人数量"], 2)
        self.assertEqual(project_a["记录数"], 2)
        self.assertAlmostEqual(project_a["平均运行时长(小时)"], 15.0, places=2)
        self.assertAlmostEqual(project_a["故障率(%)"], 15.0, places=2)
        self.assertAlmostEqual(project_a["平均满意度(分)"], 85.0, places=2)
        self.assertAlmostEqual(project_a["总运营成本(元)"], 300.0, places=2)

    def test_empty_frame_returns_empty_metrics(self) -> None:
        summary = calculate_core_metrics(pd.DataFrame(columns=list(config.ALL_COLUMNS)))

        self.assertEqual(summary.project_count, 0)
        self.assertTrue(math.isnan(summary.avg_running_hours))
        self.assertEqual(project_level_summary(pd.DataFrame()).shape[0], 0)

    def test_missing_values_are_skipped(self) -> None:
        # 缺失运行时长/满意度/节降率的记录不应拉低平均值
        self.assertNotIn(None, [self.summary.avg_satisfaction, self.summary.avg_saving_rate])
        self.assertEqual(self.data[config.COL_SATISFACTION].isna().sum(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

