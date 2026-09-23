"""数据清洗模块测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from robotops import config  # noqa: E402
from robotops.data_cleaner import ISSUE_COLUMNS, clean_operation_data  # noqa: E402
from robotops.exceptions import DataValidationError  # noqa: E402
from tests._helpers import base_operation_frame  # noqa: E402

NAN = float("nan")


def dirty_frame() -> pd.DataFrame:
    """构造包含典型脏数据的运营数据。"""

    rows = [
        # A：正常记录
        dict(R="R-01", date="2026-08-01", satisfaction=90.0),
        # B：与 A 完全重复
        dict(R="R-01", date="2026-08-01", satisfaction=90.0),
        # C：与 A 同一机器人同一天重复上报（内容不同，应保留这条）
        dict(R="R-01", date="2026-08-01", satisfaction=70.0),
        # D：满意度缺失
        dict(R="R-02", date="2026-08-01", satisfaction=None),
        # E：满意度越界
        dict(R="R-03", date="2026-08-02", satisfaction=120.0),
        # F：项目名称含首尾空格
        dict(R="R-04", date="2026-08-02", satisfaction=88.0, project="  项目甲  "),
        # G：运营成本为负
        dict(R="R-05", date="2026-08-03", satisfaction=90.0, cost=-300.0),
        # H：运行时长缺失
        dict(R="R-06", date="2026-08-03", satisfaction=90.0, running=None),
        # I：巡检次数为 0，故障率无法计算
        dict(R="R-07", date="2026-08-04", satisfaction=90.0, inspections=0, faults=2),
        # J：故障次数缺失，应按 0 处理
        dict(R="R-08", date="2026-08-04", satisfaction=90.0, faults=None),
    ]
    frame = pd.DataFrame(
        [
            {
                config.COL_DATE: row["date"],
                config.COL_PROJECT: row.get("project", "项目甲"),
                config.COL_ROBOT_ID: row["R"],
                config.COL_ROBOT_TYPE: "清洁机器人",
                config.COL_RUNNING_HOURS: row.get("running", 6.0),
                config.COL_PLANNED_HOURS: 10.0,
                config.COL_FAULT_COUNT: row.get("faults", 1),
                config.COL_INSPECTION_COUNT: row.get("inspections", 10),
                config.COL_REPAIR_COUNT: 0,
                config.COL_SATISFACTION: row["satisfaction"],
                config.COL_COST: row.get("cost", 100.0),
                config.COL_SAVING_RATE: 20.0,
            }
            for row in rows
        ]
    )
    # 追加一条全空行
    return pd.concat([frame, pd.DataFrame([{name: NAN for name in frame.columns}])], ignore_index=True)


class DataCleanerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = clean_operation_data(dirty_frame())
        self.data = self.result.data

    def _row(self, robot_id: str) -> pd.Series:
        matched = self.data.loc[self.data[config.COL_ROBOT_ID] == robot_id]
        self.assertEqual(len(matched), 1, f"{robot_id} 应只有一条记录")
        return matched.iloc[0]

    def test_row_count_and_stats(self) -> None:
        # 10 条原始记录 + 1 条空行 = 11；剔除空行、完全重复行、被覆盖的重复上报记录 = 3
        self.assertEqual(self.result.stats["原始记录数"], 11)
        self.assertEqual(len(self.data), 8)
        self.assertEqual(self.result.stats["剔除记录数"], 3)
        self.assertGreater(self.result.stats["问题条目数"], 0)

    def test_deduplicate_keeps_last_record(self) -> None:
        row = self._row("R-01")
        self.assertEqual(row[config.COL_SATISFACTION], 70.0)

    def test_text_whitespace_stripped(self) -> None:
        self.assertEqual(self._row("R-04")[config.COL_PROJECT], "项目甲")
        self.assertEqual(self.data[config.COL_PROJECT].nunique(), 1)

    def test_invalid_values_become_missing(self) -> None:
        self.assertTrue(pd.isna(self._row("R-02")[config.COL_SATISFACTION]))  # 缺失保持缺失
        self.assertTrue(pd.isna(self._row("R-03")[config.COL_SATISFACTION]))  # 越界置空
        self.assertTrue(pd.isna(self._row("R-05")[config.COL_COST]))  # 负成本置空
        self.assertTrue(pd.isna(self._row("R-06")[config.COL_RUNNING_HOURS]))  # 缺失保持缺失

    def test_missing_counts_filled_with_zero(self) -> None:
        self.assertEqual(self._row("R-08")[config.COL_FAULT_COUNT], 0.0)

    def test_derived_columns(self) -> None:
        # 运行率 = 6 / 10 × 100 = 60
        self.assertAlmostEqual(self._row("R-01")[config.COL_UPTIME_RATE], 60.0, places=4)
        # 故障率 = 1 / 10 × 100 = 10
        self.assertAlmostEqual(self._row("R-01")[config.COL_FAULT_RATE], 10.0, places=4)
        # 巡检次数为 0 时故障率无法计算
        self.assertTrue(pd.isna(self._row("R-07")[config.COL_FAULT_RATE]))
        # 运行时长缺失时运行率为空
        self.assertTrue(pd.isna(self._row("R-06")[config.COL_UPTIME_RATE]))

    def test_issue_log_contains_expected_types(self) -> None:
        kinds = set(self.result.issues["问题类型"])
        for expected in ("空行", "完全重复记录", "同一机器人同一天重复上报", "满意度越界", "运营成本为负"):
            self.assertIn(expected, kinds)
        self.assertEqual(list(self.result.issues.columns), list(ISSUE_COLUMNS))

    def test_planned_hours_filled_from_robot_type(self) -> None:
        frame = base_operation_frame().drop(columns=[config.COL_PLANNED_HOURS])
        frame.loc[:, config.COL_RUNNING_HOURS] = 4.0

        result = clean_operation_data(frame)

        self.assertIn(config.COL_PLANNED_HOURS, result.data.columns)
        # 清洁机器人默认计划运行时长 8 小时 -> 运行率 = 4 / 8 × 100 = 50
        cleaning_row = result.data.loc[result.data[config.COL_ROBOT_ID] == "R-01"]
        self.assertAlmostEqual(cleaning_row.iloc[0][config.COL_PLANNED_HOURS], 8.0, places=4)
        self.assertAlmostEqual(cleaning_row.iloc[0][config.COL_UPTIME_RATE], 50.0, places=4)

    def test_all_rows_invalid_raises(self) -> None:
        frame = base_operation_frame()
        frame.loc[:, config.COL_DATE] = "不是日期"

        with self.assertRaises(DataValidationError):
            clean_operation_data(frame)

    def test_sorted_by_date_and_robot(self) -> None:
        ordered = self.data[[config.COL_DATE, config.COL_ROBOT_ID]]
        self.assertTrue(ordered.equals(ordered.sort_values(list(ordered.columns), kind="stable").reset_index(drop=True)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
