"""异常识别模块测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from robotops import config  # noqa: E402
from robotops.anomaly import (  # noqa: E402
    build_rules,
    detect_anomalies,
    empty_anomaly_frame,
    summarize_anomalies,
    summarize_anomalies_by_project,
)
from robotops.data_cleaner import clean_operation_data  # noqa: E402
from robotops.exceptions import ConfigurationError  # noqa: E402
from robotops.models import ANOMALY_COLUMNS  # noqa: E402


def _record(
    *,
    robot_id: str,
    date: str,
    running_hours: float | None,
    planned_hours: float,
    faults: int,
    inspections: int,
    satisfaction: float | None,
    saving_rate: float | None,
    project: str = "项目甲",
) -> dict[str, object]:
    return {
        config.COL_DATE: date,
        config.COL_PROJECT: project,
        config.COL_ROBOT_ID: robot_id,
        config.COL_ROBOT_TYPE: "巡检机器人",
        config.COL_RUNNING_HOURS: running_hours,
        config.COL_PLANNED_HOURS: planned_hours,
        config.COL_FAULT_COUNT: faults,
        config.COL_INSPECTION_COUNT: inspections,
        config.COL_REPAIR_COUNT: 0,
        config.COL_SATISFACTION: satisfaction,
        config.COL_COST: 100.0,
        config.COL_SAVING_RATE: saving_rate,
    }


def anomaly_frame() -> pd.DataFrame:
    """R-01 第一天四条规则全部命中；第二天与 R-02 正常。"""

    return pd.DataFrame(
        [
            _record(
                robot_id="R-01",
                date="2026-08-01",
                running_hours=7.0,  # 运行率 70% < 80%
                planned_hours=10.0,
                faults=3,  # 单日故障率 30% > 5%
                inspections=10,
                satisfaction=80.0,  # < 85
                saving_rate=5.0,  # < 10
            ),
            _record(
                robot_id="R-01",
                date="2026-08-02",
                running_hours=10.0,
                planned_hours=10.0,
                faults=0,
                inspections=400,  # 用于拉低累计故障率
                satisfaction=95.0,
                saving_rate=25.0,
            ),
            _record(
                robot_id="R-02",
                date="2026-08-01",
                running_hours=9.0,
                planned_hours=10.0,
                faults=0,
                inspections=10,
                satisfaction=95.0,
                saving_rate=20.0,
            ),
        ]
    )


class AnomalyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = clean_operation_data(anomaly_frame()).data

    def test_record_scope_flags_all_four_rules(self) -> None:
        anomalies = detect_anomalies(self.data, fault_rate_scope="record")

        self.assertEqual(len(anomalies), 4)
        self.assertEqual(
            set(anomalies["异常类型"]),
            {"故障率偏高", "满意度偏低", "运行率偏低", "节降率偏低"},
        )
        self.assertEqual(set(anomalies["机器人ID"]), {"R-01"})
        self.assertTrue((anomalies["判定粒度"] == "逐日记录").all())

    def test_default_scope_uses_robot_period_for_fault_rate(self) -> None:
        anomalies = detect_anomalies(self.data)

        self.assertEqual(set(anomalies["异常类型"]), {"满意度偏低", "运行率偏低", "节降率偏低"})
        # 累计口径：(3 + 0) / (10 + 400) = 0.73%，低于 5% 阈值 -> 不命中
        self.assertNotIn("故障率偏高", set(anomalies["异常类型"]))

        fault_rate_rows = detect_anomalies(
            clean_operation_data(
                pd.DataFrame(
                    [
                        _record(
                            robot_id="R-09",
                            date="2026-08-01",
                            running_hours=10.0,
                            planned_hours=10.0,
                            faults=9,
                            inspections=30,
                            satisfaction=95.0,
                            saving_rate=20.0,
                        )
                    ]
                )
            ).data
        )
        self.assertEqual(len(fault_rate_rows), 1)
        self.assertEqual(fault_rate_rows.iloc[0]["判定粒度"], "机器人周期累计")
        self.assertAlmostEqual(fault_rate_rows.iloc[0]["实际值"], 30.0, places=2)
        self.assertEqual(fault_rate_rows.iloc[0]["严重程度"], "高")

    def test_missing_metric_is_not_flagged(self) -> None:
        frame = pd.DataFrame(
            [
                _record(
                    robot_id="R-03",
                    date="2026-08-01",
                    running_hours=None,  # 运行率空
                    planned_hours=10.0,
                    faults=2,
                    inspections=0,  # 故障率空
                    satisfaction=None,  # 满意度空
                    saving_rate=20.0,
                )
            ]
        )
        data = clean_operation_data(frame).data

        anomalies = detect_anomalies(data, fault_rate_scope="record")

        self.assertTrue(anomalies.empty)

    def test_severity_levels(self) -> None:
        rule = build_rules()[0]  # 故障率 > 5%
        self.assertEqual(rule.severity_and_deviation(10.0)[0], "高")
        self.assertEqual(rule.severity_and_deviation(6.5)[0], "中")
        self.assertEqual(rule.severity_and_deviation(5.5)[0], "低")

    def test_invalid_scope_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            detect_anomalies(self.data, fault_rate_scope="未知口径")

    def test_empty_frame_returns_typed_empty_result(self) -> None:
        empty = detect_anomalies(pd.DataFrame())

        self.assertTrue(empty.empty)
        self.assertEqual(list(empty.columns), ["异常序号", *ANOMALY_COLUMNS])
        self.assertEqual(list(empty_anomaly_frame().columns), list(empty.columns))

    def test_summaries(self) -> None:
        anomalies = detect_anomalies(self.data, fault_rate_scope="record")
        summary = summarize_anomalies(anomalies)
        by_project = summarize_anomalies_by_project(anomalies)

        self.assertEqual(int(summary["命中记录数"].sum()), len(anomalies))
        self.assertEqual(len(summary), 4)
        self.assertIn("判定条件", summary.columns)
        project_row = by_project.iloc[0]
        self.assertEqual(project_row["项目名称"], "项目甲")
        self.assertEqual(project_row["异常记录数"], 4)
        self.assertEqual(project_row["涉及机器人数"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

