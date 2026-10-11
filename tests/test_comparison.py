"""Phase 9 项目横向对比测试。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from robotops import config  # noqa: E402
from robotops.comparison import (  # noqa: E402
    COMPARISON_METRICS,
    comparison_frame,
    comparison_metric_frame,
    compare_projects,
)


def _row(project: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        config.COL_DATE: "2026-08-01",
        config.COL_PROJECT: project,
        config.COL_ROBOT_ID: project + "-01",
        config.COL_ROBOT_TYPE: "清洁机器人",
        config.COL_RUNNING_HOURS: 9.0,
        config.COL_PLANNED_HOURS: 10.0,
        config.COL_FAULT_COUNT: 1,
        config.COL_INSPECTION_COUNT: 10,
        config.COL_REPAIR_COUNT: 1,
        config.COL_SATISFACTION: 90.0,
        config.COL_COST: 1000.0,
        config.COL_SAVING_RATE: 20.0,
    }
    row.update(overrides)
    return row


def three_project_frame() -> pd.DataFrame:
    """三个项目：A 利用率最高，B 故障率与成本最优，C 居中。"""

    return pd.DataFrame(
        [
            _row("A小区"),
            _row(
                "B小区",
                **{
                    config.COL_RUNNING_HOURS: 5.0,
                    config.COL_FAULT_COUNT: 0,
                    config.COL_REPAIR_COUNT: 0,
                    config.COL_SATISFACTION: 80.0,
                    config.COL_COST: 2000.0,
                    config.COL_SAVING_RATE: 10.0,
                },
            ),
            _row(
                "C小区",
                **{
                    config.COL_RUNNING_HOURS: 8.0,
                    config.COL_FAULT_COUNT: 0.5,
                    config.COL_REPAIR_COUNT: 0,
                    config.COL_SATISFACTION: 85.0,
                    config.COL_COST: 3000.0,
                    config.COL_SAVING_RATE: 15.0,
                },
            ),
        ]
    )


class ProjectComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.comparison = compare_projects(three_project_frame())

    def test_seven_metrics_compared(self) -> None:
        self.assertEqual(len(COMPARISON_METRICS), 7)
        self.assertEqual(
            [item.name for item in COMPARISON_METRICS],
            ["故障率", "设备运行率", "设备利用率", "用户满意度", "运营成本", "节降率", "MTTR"],
        )
        frame = self.comparison.to_frame()
        for name in [item.name for item in COMPARISON_METRICS]:
            self.assertIn(name, frame.columns)

    def test_project_values_are_hand_checkable(self) -> None:
        self.assertAlmostEqual(self.comparison.value_of("A小区", "utilization_rate"), 90.0, places=4)
        self.assertAlmostEqual(self.comparison.value_of("B小区", "utilization_rate"), 50.0, places=4)
        self.assertAlmostEqual(self.comparison.value_of("A小区", "fault_rate"), 10.0, places=4)
        self.assertAlmostEqual(self.comparison.value_of("B小区", "fault_rate"), 0.0, places=4)
        self.assertAlmostEqual(self.comparison.value_of("A小区", "total_cost"), 1000.0, places=4)
        # MTTR：A 停机 (10-9)=1 小时 / 维修 1 次 = 1 小时/次；B、C 无维修记录 -> 空值
        self.assertAlmostEqual(self.comparison.value_of("A小区", "mttr"), 1.0, places=4)
        self.assertIsNone(self.comparison.value_of("B小区", "mttr"))

    def test_ranking_respects_direction(self) -> None:
        fault = self.comparison.ranking("fault_rate")
        self.assertEqual(list(fault["项目名称"]), ["B小区", "C小区", "A小区"])
        self.assertEqual(list(fault["排名"]), [1, 2, 3])

        utilization = self.comparison.ranking("utilization_rate")
        self.assertEqual(list(utilization["项目名称"]), ["A小区", "C小区", "B小区"])

        cost = self.comparison.ranking("total_cost")
        self.assertEqual(list(cost["项目名称"]), ["A小区", "B小区", "C小区"])

    def test_best_and_worst(self) -> None:
        self.assertEqual(self.comparison.best("fault_rate")["project"], "B小区")
        self.assertEqual(self.comparison.worst("fault_rate")["project"], "A小区")
        self.assertEqual(self.comparison.best("satisfaction")["project"], "A小区")

    def test_delta_against_overall_average(self) -> None:
        ranking = self.comparison.ranking("utilization_rate")
        row_a = ranking.loc[ranking["项目名称"] == "A小区"].iloc[0]

        self.assertAlmostEqual(self.comparison.overall["utilization_rate"], (90 + 50 + 80) / 3, places=4)
        self.assertAlmostEqual(row_a["与整体平均差异"], 90.0 - (90 + 50 + 80) / 3, places=2)

    def test_chart_frame(self) -> None:
        chart = self.comparison.chart_frame(["utilization_rate", "fault_rate"])

        self.assertEqual(list(chart.index), ["A小区", "B小区", "C小区"])
        self.assertEqual(list(chart.columns), ["设备利用率", "故障率"])

    def test_highlights_are_factual(self) -> None:
        lines = self.comparison.highlight_lines()

        self.assertTrue(lines)
        self.assertTrue(any("故障率" in line for line in lines))
        self.assertTrue(all("推测" not in line for line in lines))

    def test_as_dict_is_json_serializable(self) -> None:
        payload = self.comparison.as_dict()
        text = json.dumps(payload, ensure_ascii=False)

        self.assertEqual(len(payload["projects"]), 3)
        self.assertEqual(len(payload["metrics"]), 7)
        self.assertIn("rankings", payload)
        self.assertLess(len(text), 50_000)

    def test_summary_dict(self) -> None:
        payload = self.comparison.summary_dict()

        self.assertEqual(payload["project_count"], 3)
        self.assertIn("best", payload)
        self.assertIn("highlights", payload)

    def test_metric_frame(self) -> None:
        frame = comparison_metric_frame()

        self.assertEqual(len(frame), 7)
        self.assertIn("优劣方向", frame.columns)
        self.assertEqual(set(frame["优劣方向"]), {"越高越好", "越低越好"})


class ProjectComparisonEdgeCaseTests(unittest.TestCase):
    def test_single_project_warns(self) -> None:
        comparison = compare_projects(pd.DataFrame([_row("唯一项目")]))

        self.assertEqual(comparison.project_count, 1)
        self.assertTrue(any("只有 1 个项目" in item for item in comparison.warnings))

    def test_empty_and_missing_column(self) -> None:
        for value in (None, pd.DataFrame()):
            comparison = compare_projects(value)
            self.assertFalse(comparison.available)
            self.assertTrue(comparison.warnings)

        without_project = pd.DataFrame([{config.COL_COST: 1.0}])
        comparison = compare_projects(without_project)
        self.assertFalse(comparison.available)
        self.assertTrue(any("项目名称" in item for item in comparison.warnings))

    def test_ranking_of_metric_without_data(self) -> None:
        comparison = compare_projects(pd.DataFrame([_row("A小区", **{config.COL_REPAIR_COUNT: 0})]))
        ranking = comparison.ranking("mttr")

        self.assertTrue(ranking.empty)

    def test_comparison_frame_helper(self) -> None:
        frame = comparison_frame(three_project_frame())

        self.assertEqual(len(frame), 3)
        self.assertIn("设备利用率", frame.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)

