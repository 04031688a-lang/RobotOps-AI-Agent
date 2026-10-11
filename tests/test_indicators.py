"""Phase 8 运营指标体系测试。

覆盖：
1. 九项指标的计算口径（用手算可核对的小数据集）；
2. 阈值判定：故障率 > 5% / 满意度 < 85 / 设备利用率 < 80% / 节降率 < 10%；
3. MTTR 高于项目平均水平 → 预警（含单台设备项目不可比较的情况）；
4. 空数据、缺列等异常输入不崩溃；
5. 指标与阈值的说明表结构。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from robotops import config  # noqa: E402
from robotops.indicators import (  # noqa: E402
    INDICATOR_COLUMNS,
    LEVEL_ANOMALY,
    LEVEL_WARNING,
    calculate_indicators,
    detect_indicator_anomalies,
    indicator_definitions_frame,
    indicator_thresholds_frame,
    project_level_indicators,
    summarize_indicator_anomalies,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        config.COL_DATE: "2026-08-01",
        config.COL_PROJECT: "项目甲",
        config.COL_ROBOT_ID: "R-01",
        config.COL_ROBOT_TYPE: "清洁机器人",
        config.COL_RUNNING_HOURS: 8.0,
        config.COL_PLANNED_HOURS: 10.0,
        config.COL_FAULT_COUNT: 1,
        config.COL_INSPECTION_COUNT: 10,
        config.COL_REPAIR_COUNT: 1,
        config.COL_SATISFACTION: 90.0,
        config.COL_COST: 100.0,
        config.COL_SAVING_RATE: 20.0,
    }
    row.update(overrides)
    return row


def hand_frame() -> pd.DataFrame:
    """3 台机器人 / 2 个项目，数值便于手算核对。"""

    return pd.DataFrame(
        [
            _row(),
            _row(
                **{
                    config.COL_ROBOT_ID: "R-02",
                    config.COL_RUNNING_HOURS: 6.0,
                    config.COL_FAULT_COUNT: 0,
                    config.COL_REPAIR_COUNT: 0,
                    config.COL_SATISFACTION: 80.0,
                    config.COL_COST: 200.0,
                    config.COL_SAVING_RATE: 5.0,
                }
            ),
            _row(
                **{
                    config.COL_PROJECT: "项目乙",
                    config.COL_ROBOT_ID: "R-03",
                    config.COL_RUNNING_HOURS: 9.0,
                    config.COL_FAULT_COUNT: 0,
                    config.COL_REPAIR_COUNT: 0,
                    config.COL_SATISFACTION: 95.0,
                    config.COL_COST: 300.0,
                    config.COL_SAVING_RATE: 12.0,
                }
            ),
        ]
    )


class IndicatorCalculationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = hand_frame()
        self.indicators = calculate_indicators(self.data)

    def test_nine_indicators_exist(self) -> None:
        rows = {row["指标"] for row in self.indicators.as_rows()}

        self.assertEqual(
            rows,
            {
                config.COL_ONLINE_RATE,
                config.COL_UTILIZATION_RATE,
                "故障率",
                config.COL_MTBF,
                config.COL_MTTR,
                "用户满意度",
                "总运营成本",
                config.COL_AVG_COST,
                "节降率",
            },
        )
        self.assertEqual(len(rows), 9)

    def test_indicator_values_are_hand_checkable(self) -> None:
        item = self.indicators

        # 在线率 = 3 个有效机器日 / (3 台 × 1 天) = 100%
        self.assertAlmostEqual(item.online_rate, 100.0, places=6)
        # 利用率 = 23 / 30 = 76.67%
        self.assertAlmostEqual(item.utilization_rate, 23.0 / 30.0 * 100.0, places=6)
        # 故障率 = 1 / 30 = 3.33%
        self.assertAlmostEqual(item.fault_rate, 1.0 / 30.0 * 100.0, places=6)
        # MTBF = 23 / 1 = 23 小时/次
        self.assertAlmostEqual(item.mtbf, 23.0, places=6)
        # 停机时长 = (10-8)+(10-6)+(10-9) = 7；MTTR = 7 / 1 = 7 小时/次
        self.assertAlmostEqual(item.downtime_hours, 7.0, places=6)
        self.assertAlmostEqual(item.mttr, 7.0, places=6)
        self.assertAlmostEqual(item.avg_satisfaction, (90 + 80 + 95) / 3, places=6)
        self.assertAlmostEqual(item.total_cost, 600.0, places=6)
        self.assertAlmostEqual(item.avg_cost, 200.0, places=6)
        self.assertAlmostEqual(item.avg_saving_rate, (20 + 5 + 12) / 3, places=6)

    def test_zero_fault_and_repair_return_blank(self) -> None:
        no_fault = pd.DataFrame([_row(**{config.COL_FAULT_COUNT: 0, config.COL_REPAIR_COUNT: 0})])
        indicators = calculate_indicators(no_fault)

        self.assertTrue(math.isnan(indicators.mtbf))
        self.assertTrue(math.isnan(indicators.mttr))
        self.assertEqual(indicators.as_dict()[config.COL_MTBF], None)
        self.assertEqual(indicators.as_dict()[config.COL_MTTR], None)

    def test_planned_hours_fallback_by_robot_type(self) -> None:
        frame = hand_frame().drop(columns=[config.COL_PLANNED_HOURS])
        indicators = calculate_indicators(frame)

        # 清洁机器人计划 8 小时：8 + 6 + 9 = 23 小时运行 / (8 + 8 + 8) = 95.83%
        self.assertAlmostEqual(indicators.total_planned_hours, 24.0, places=6)

    def test_empty_and_none_input(self) -> None:
        for value in (None, pd.DataFrame()):
            indicators = calculate_indicators(value)
            self.assertEqual(indicators.record_count, 0)
            self.assertTrue(math.isnan(indicators.online_rate))

    def test_project_level_indicators(self) -> None:
        frame = project_level_indicators(self.data)

        self.assertEqual(list(frame.columns), list(INDICATOR_COLUMNS))
        self.assertIn("全部项目", set(frame["项目名称"]))
        project_a = frame.loc[frame["项目名称"] == "项目甲"].iloc[0]
        self.assertEqual(project_a["机器人数"], 2)
        self.assertAlmostEqual(project_a["设备利用率(%)"], 70.0, places=2)


class IndicatorThresholdTests(unittest.TestCase):
    def test_default_threshold_values(self) -> None:
        limits = config.DEFAULT_INDICATOR_THRESHOLDS

        self.assertEqual(limits.fault_rate_upper, 5.0)
        self.assertEqual(limits.satisfaction_lower, 85.0)
        self.assertEqual(limits.utilization_lower, 80.0)
        self.assertEqual(limits.saving_rate_lower, 10.0)
        self.assertEqual(limits.mttr_project_avg_ratio, 1.0)

    def test_thresholds_are_centralized(self) -> None:
        # 阈值必须来自 config，而不是散落在业务代码里
        frame = indicator_thresholds_frame()
        self.assertEqual(len(frame), 5)
        self.assertIn(config.COL_UTILIZATION_RATE, set(frame["指标"]))
        self.assertIn(config.COL_MTTR, set(frame["指标"]))

    def test_utilization_below_threshold_is_anomaly(self) -> None:
        anomalies = detect_indicator_anomalies(self._low_utilization_frame())

        hits = anomalies.loc[
            (anomalies["指标"] == config.COL_UTILIZATION_RATE)
            & (anomalies["级别"] == LEVEL_ANOMALY)
        ]
        self.assertFalse(hits.empty)
        # 阈值取配置默认值 80%
        self.assertTrue((hits["阈值"] == 80.0).all())

    def _low_utilization_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                _row(**{config.COL_RUNNING_HOURS: 4.0, config.COL_FAULT_COUNT: 0,
                        config.COL_REPAIR_COUNT: 0}),
                _row(**{config.COL_ROBOT_ID: "R-02", config.COL_RUNNING_HOURS: 4.0,
                        config.COL_FAULT_COUNT: 0, config.COL_REPAIR_COUNT: 0}),
                _row(**{config.COL_PROJECT: "项目乙", config.COL_ROBOT_ID: "R-03",
                        config.COL_RUNNING_HOURS: 9.0, config.COL_FAULT_COUNT: 0,
                        config.COL_REPAIR_COUNT: 0}),
            ]
        )

    def test_fault_rate_and_satisfaction_and_saving_rate_hits(self) -> None:
        frame = pd.DataFrame(
            [
                _row(**{
                    config.COL_FAULT_COUNT: 2, config.COL_INSPECTION_COUNT: 10,
                    config.COL_SATISFACTION: 70.0, config.COL_SAVING_RATE: 5.0,
                    config.COL_RUNNING_HOURS: 9.5, config.COL_REPAIR_COUNT: 0,
                })
            ]
        )
        anomalies = detect_indicator_anomalies(frame)
        overall = anomalies.loc[anomalies["对象类型"] == "整体"]
        metrics = set(overall["指标"])

        self.assertIn("故障率", metrics)
        self.assertIn("用户满意度", metrics)
        self.assertIn("节降率", metrics)
        self.assertTrue((overall["级别"] == LEVEL_ANOMALY).all())

    def test_threshold_boundary_is_not_anomaly(self) -> None:
        # 故障率正好 5%、满意度正好 85、节降率正好 10% -> 不触发（口径为严格大于 / 小于）
        frame = pd.DataFrame(
            [
                _row(**{
                    config.COL_FAULT_COUNT: 1, config.COL_INSPECTION_COUNT: 20,
                    config.COL_SATISFACTION: 85.0, config.COL_SAVING_RATE: 10.0,
                    config.COL_RUNNING_HOURS: 10.0, config.COL_REPAIR_COUNT: 0,
                })
            ]
        )
        anomalies = detect_indicator_anomalies(frame)
        overall = anomalies.loc[anomalies["对象类型"] == "整体"]

        self.assertTrue(overall.empty, "临界值不应判定为异常")

    def test_custom_thresholds_are_respected(self) -> None:
        frame = hand_frame()
        limits = config.IndicatorThresholds(satisfaction_lower=90.0, utilization_lower=95.0)
        anomalies = detect_indicator_anomalies(frame, thresholds=limits)
        overall = anomalies.loc[anomalies["对象类型"] == "整体"]
        metrics = set(overall["指标"])

        self.assertIn("用户满意度", metrics)
        self.assertIn(config.COL_UTILIZATION_RATE, metrics)

    def test_mttr_above_project_average_is_warning(self) -> None:
        frame = pd.DataFrame(
            [
                _row(**{config.COL_ROBOT_ID: "R-A", config.COL_RUNNING_HOURS: 5.0,
                        config.COL_REPAIR_COUNT: 1, config.COL_FAULT_COUNT: 1}),
                _row(**{config.COL_ROBOT_ID: "R-B", config.COL_RUNNING_HOURS: 9.0,
                        config.COL_REPAIR_COUNT: 1, config.COL_FAULT_COUNT: 1}),
            ]
        )
        anomalies = detect_indicator_anomalies(frame)
        mttr_hits = anomalies.loc[anomalies["指标"] == config.COL_MTTR]

        self.assertFalse(mttr_hits.empty)
        self.assertTrue((mttr_hits["级别"] == LEVEL_WARNING).all())
        self.assertEqual(set(mttr_hits["对象名称"]), {"R-A"})

    def test_single_robot_project_has_no_mttr_warning(self) -> None:
        frame = pd.DataFrame(
            [
                _row(**{config.COL_ROBOT_ID: "R-SOLO", config.COL_RUNNING_HOURS: 3.0,
                        config.COL_REPAIR_COUNT: 1, config.COL_FAULT_COUNT: 1}),
            ]
        )
        anomalies = detect_indicator_anomalies(frame)

        self.assertTrue(anomalies.loc[anomalies["指标"] == config.COL_MTTR].empty)

    def test_summary_and_empty_input(self) -> None:
        anomalies = detect_indicator_anomalies(self._low_utilization_frame())
        summary = summarize_indicator_anomalies(anomalies)

        self.assertFalse(summary.empty)
        self.assertIn("命中对象数", summary.columns)
        self.assertTrue(detect_indicator_anomalies(None).empty)
        self.assertTrue(detect_indicator_anomalies(pd.DataFrame()).empty)
        self.assertTrue(summarize_indicator_anomalies(None).empty)

    def test_indicator_definitions_frame(self) -> None:
        definitions = indicator_definitions_frame()

        self.assertEqual(len(definitions), 9)
        self.assertEqual(set(definitions.columns), {"指标", "单位", "计算口径"})


if __name__ == "__main__":
    unittest.main(verbosity=2)

