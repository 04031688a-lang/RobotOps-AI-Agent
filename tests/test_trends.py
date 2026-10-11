"""Phase 9 趋势分析测试。

至少覆盖需求要求的 5 类场景：正常趋势 / 上升趋势 / 下降趋势 / 数据不足 / 日期异常，
外加：粒度自动选择、指标区间口径、环比与同期变化、连续上升下降、周期完整性处理。
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from robotops import config  # noqa: E402
from robotops.trends import (  # noqa: E402
    TREND_METRICS,
    build_trend_series,
    metric_value,
    metric_values,
    trend_metric_definitions_frame,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        config.COL_DATE: "2026-08-03",
        config.COL_PROJECT: "项目甲",
        config.COL_ROBOT_ID: "R-01",
        config.COL_ROBOT_TYPE: "清洁机器人",
        config.COL_RUNNING_HOURS: 8.0,
        config.COL_PLANNED_HOURS: 10.0,
        config.COL_FAULT_COUNT: 0,
        config.COL_INSPECTION_COUNT: 10,
        config.COL_REPAIR_COUNT: 0,
        config.COL_SATISFACTION: 90.0,
        config.COL_COST: 1000.0,
        config.COL_SAVING_RATE: 15.0,
    }
    row.update(overrides)
    return row


def daily_frame(
    start: str,
    days: int,
    *,
    per_day: Callable[[int], dict[str, Any]] | None = None,
    **base: object,
) -> pd.DataFrame:
    """生成按天一条记录的数据（默认同一台机器人）。"""

    start_ts = pd.Timestamp(start)
    rows: list[dict[str, object]] = []
    for index in range(days):
        row = _row(**base)
        row[config.COL_DATE] = (start_ts + pd.Timedelta(days=index)).strftime("%Y-%m-%d")
        if per_day is not None:
            row.update(per_day(index))
        rows.append(row)
    return pd.DataFrame(rows)


class TrendGranularityTests(unittest.TestCase):
    def test_explicit_granularity(self) -> None:
        frame = daily_frame("2026-08-03", 21)  # 3 个完整自然周

        for granularity, expected in (
            (config.TREND_GRANULARITY_DAY, 21),
            (config.TREND_GRANULARITY_WEEK, 3),
            (config.TREND_GRANULARITY_MONTH, 1),
        ):
            with self.subTest(granularity=granularity):
                analysis = build_trend_series(frame, granularity=granularity)
                self.assertEqual(analysis.granularity, granularity)
                self.assertEqual(analysis.period_count, expected)
                self.assertFalse(analysis.auto_detected)

    def test_auto_granularity_by_span(self) -> None:
        # 跨度 < 21 天 -> 日；>= 21 -> 周；>= 90 -> 月
        short = build_trend_series(daily_frame("2026-08-03", 10))
        medium = build_trend_series(daily_frame("2026-08-03", 35))
        long = build_trend_series(daily_frame("2026-01-01", 120))

        self.assertEqual(short.granularity, config.TREND_GRANULARITY_DAY)
        self.assertEqual(medium.granularity, config.TREND_GRANULARITY_WEEK)
        self.assertEqual(long.granularity, config.TREND_GRANULARITY_MONTH)
        self.assertTrue(medium.auto_detected)

    def test_invalid_granularity_falls_back_to_auto(self) -> None:
        analysis = build_trend_series(daily_frame("2026-08-03", 10), granularity="每小时")

        self.assertIn(
            analysis.granularity, config.TREND_GRANULARITY_CHOICES
        )


class TrendMetricValueTests(unittest.TestCase):
    def test_seven_metrics_defined(self) -> None:
        definitions = trend_metric_definitions_frame()

        self.assertEqual(len(definitions), 7)
        self.assertEqual(
            [item.name for item in TREND_METRICS],
            [
                "故障率",
                "设备运行率",
                "设备利用率",
                "用户满意度",
                "运营成本",
                "节降率",
                "MTTR",
            ],
        )

    def test_metric_values_are_hand_checkable(self) -> None:
        frame = pd.DataFrame(
            [
                _row(**{config.COL_RUNNING_HOURS: 8.0, config.COL_FAULT_COUNT: 1,
                        config.COL_REPAIR_COUNT: 1}),
                _row(**{config.COL_RUNNING_HOURS: 6.0, config.COL_FAULT_COUNT: 0,
                        config.COL_REPAIR_COUNT: 0, config.COL_SATISFACTION: 80.0,
                        config.COL_COST: 3000.0, config.COL_SAVING_RATE: 10.0}),
            ]
        )
        values = metric_values(frame)

        # 故障率 = 1 / 20 = 5%
        self.assertAlmostEqual(values["fault_rate"], 5.0, places=6)
        # 运行率 = (80% + 60%) / 2 = 70%
        self.assertAlmostEqual(values["uptime_rate"], 70.0, places=6)
        # 利用率 = 14 / 20 = 70%
        self.assertAlmostEqual(values["utilization_rate"], 70.0, places=6)
        self.assertAlmostEqual(values["satisfaction"], 85.0, places=6)
        self.assertAlmostEqual(values["total_cost"], 4000.0, places=6)
        self.assertAlmostEqual(values["saving_rate"], 12.5, places=6)
        # MTTR = 停机时长合计 (10-8)+(10-6) = 6 小时 / 维修 1 次 = 6 小时/次
        self.assertAlmostEqual(values["mttr"], 6.0, places=6)

    def test_metric_value_edge_cases(self) -> None:
        empty = pd.DataFrame(columns=list(config.ALL_COLUMNS))

        self.assertTrue(math.isnan(metric_value(empty, "fault_rate")))
        self.assertTrue(math.isnan(metric_value(pd.DataFrame(), "unknown_key")))

        # 巡检次数为 0 时故障率无法计算；维修次数为 0 时 MTTR 无法计算
        no_repair = pd.DataFrame([_row(**{config.COL_INSPECTION_COUNT: 0})])
        values = metric_values(no_repair)
        self.assertTrue(math.isnan(values["fault_rate"]))
        self.assertTrue(math.isnan(values["mttr"]))


class TrendChangeTests(unittest.TestCase):
    def test_rising_trend_with_consecutive_detection(self) -> None:
        # 三周故障率 4% -> 6% -> 8%（每天 10 次巡检）
        faults = {0: 0.4, 1: 0.4, 2: 0.4, 3: 0.6, 4: 0.6, 5: 0.6, 6: 0.6,
                  7: 0.8, 8: 0.8, 9: 0.8, 10: 0.8}
        frame = daily_frame(
            "2026-08-03",
            21,
            per_day=lambda index: {
                config.COL_FAULT_COUNT: faults.get(index // 2, 0.8),
                config.COL_SATISFACTION: 92.0 - (index // 7) * 3.0,
            },
        )
        analysis = build_trend_series(frame, granularity=config.TREND_GRANULARITY_WEEK)
        change = analysis.change_of("fault_rate")

        self.assertIsNotNone(change)
        self.assertAlmostEqual(change.latest_value, 8.0, places=4)
        self.assertAlmostEqual(change.previous_value, 6.0, places=4)
        self.assertAlmostEqual(change.change_ratio, (8.0 - 6.0) / 6.0, places=4)
        self.assertEqual(change.direction, config.TREND_UP)
        self.assertTrue(change.is_consecutive_up)
        self.assertEqual(change.consecutive_periods, 3)
        self.assertIn("连续 3 期上升", change.as_row()["连续变化"])

        satisfaction = analysis.change_of("satisfaction")
        self.assertEqual(satisfaction.direction, config.TREND_DOWN)
        self.assertTrue(satisfaction.is_consecutive_down)

    def test_falling_trend(self) -> None:
        decrement = {0: 1.0, 1: 0.5, 2: 0.25}
        frame = daily_frame(
            "2026-08-03",
            21,
            per_day=lambda index: {config.COL_FAULT_COUNT: decrement[index // 7]},
        )
        analysis = build_trend_series(frame, granularity=config.TREND_GRANULARITY_WEEK)
        change = analysis.change_of("fault_rate")

        self.assertEqual(change.direction, config.TREND_DOWN)
        self.assertTrue(change.is_consecutive_down)
        self.assertLess(change.change_ratio, 0)

    def test_flat_trend_is_not_reported_as_change(self) -> None:
        frame = daily_frame("2026-08-03", 21)  # 全部指标恒定
        analysis = build_trend_series(frame, granularity=config.TREND_GRANULARITY_WEEK)

        for change in analysis.changes:
            with self.subTest(metric=change.metric_name):
                if change.metric_key == "mttr":
                    # 维修次数为 0 时 MTTR 无法计算，应明确标记数据不足而不是编造数值
                    self.assertEqual(change.status, "数据不足")
                    continue
                self.assertEqual(change.status, "正常")
                self.assertEqual(change.direction, config.TREND_FLAT)
        cost = analysis.change_of("total_cost")
        self.assertEqual(cost.direction, config.TREND_FLAT)
        self.assertFalse(cost.is_consecutive_up)
        self.assertFalse(cost.is_consecutive_down)

    def test_change_ratio_formula(self) -> None:
        # 上周 4.2%、本周 6.8% -> 环比 = (6.8 - 4.2) / 4.2
        frame = daily_frame(
            "2026-08-10",
            14,
            per_day=lambda index: {
                config.COL_FAULT_COUNT: 0.42 if index < 7 else 0.68,
                config.COL_INSPECTION_COUNT: 10,
            },
        )
        analysis = build_trend_series(frame, granularity=config.TREND_GRANULARITY_WEEK)
        change = analysis.change_of("fault_rate")

        self.assertAlmostEqual(change.previous_value, 4.2, places=4)
        self.assertAlmostEqual(change.latest_value, 6.8, places=4)
        self.assertAlmostEqual(change.change_ratio, (6.8 - 4.2) / 4.2, places=6)
        self.assertAlmostEqual(change.change_percent, 61.9, places=2)

    def test_same_period_change_for_week(self) -> None:
        frame = daily_frame("2026-08-03", 42)  # 6 个完整周
        analysis = build_trend_series(frame, granularity=config.TREND_GRANULARITY_WEEK)
        change = analysis.change_of("satisfaction")

        self.assertIsNotNone(change.same_period_value)
        self.assertEqual(change.same_period_label, "上月同期变化")
        self.assertAlmostEqual(change.same_period_ratio, 0.0, places=6)

    def test_same_period_change_for_month(self) -> None:
        # 覆盖 14 个月（2025-01-01 ~ 2026-02-03）：最新完整月 2026-01，去年同期 2025-01 在数据内
        frame = daily_frame("2025-01-01", 400)
        analysis = build_trend_series(frame, granularity=config.TREND_GRANULARITY_MONTH)
        change = analysis.change_of("satisfaction")

        self.assertIsNotNone(change.same_period_value)
        self.assertEqual(change.same_period_label, "去年同期变化")
        self.assertEqual(change.same_period_note, "")

    def test_same_period_change_unavailable_when_span_too_short(self) -> None:
        # 只有 7 个月数据时无法计算去年同期
        frame = daily_frame("2025-01-01", 200)
        analysis = build_trend_series(frame, granularity=config.TREND_GRANULARITY_MONTH)
        change = analysis.change_of("satisfaction")

        self.assertIsNone(change.same_period_value)
        self.assertIsNone(change.same_period_ratio)
        self.assertTrue(change.same_period_note)

    def test_changes_dict_uses_metric_keys(self) -> None:
        frame = daily_frame("2026-08-03", 21)
        payload = build_trend_series(frame, granularity=config.TREND_GRANULARITY_WEEK).summary_dict()

        self.assertIn("changes", payload)
        for key in payload["changes"]:
            self.assertTrue(key.endswith("_change"))
        self.assertIn("granularity", payload)
        self.assertIn("period_count", payload)
        self.assertIn("highlights", payload)


class TrendDataProblemTests(unittest.TestCase):
    def test_insufficient_data_single_period(self) -> None:
        frame = daily_frame("2026-08-03", 7)
        analysis = build_trend_series(frame, granularity=config.TREND_GRANULARITY_WEEK)

        self.assertEqual(analysis.period_count, 1)
        self.assertTrue(analysis.warnings)
        change = analysis.change_of("fault_rate")
        self.assertEqual(change.status, "数据不足")
        self.assertIsNone(change.change_ratio)

    def test_empty_and_missing_column(self) -> None:
        for value in (None, pd.DataFrame()):
            analysis = build_trend_series(value)
            self.assertFalse(analysis.available)
            self.assertTrue(analysis.warnings)

        no_date = pd.DataFrame([{config.COL_COST: 100.0}])
        analysis = build_trend_series(no_date)
        self.assertFalse(analysis.available)
        self.assertTrue(any("日期" in item for item in analysis.warnings))

    def test_invalid_dates_are_skipped_with_warning(self) -> None:
        frame = daily_frame("2026-08-03", 14)
        broken = frame.copy()
        broken.loc[0, config.COL_DATE] = "不是日期"
        broken.loc[1, config.COL_DATE] = None

        analysis = build_trend_series(broken, granularity=config.TREND_GRANULARITY_DAY)

        self.assertEqual(analysis.invalid_date_rows, 2)
        self.assertTrue(any("日期异常" in item for item in analysis.warnings))
        self.assertEqual(analysis.valid_rows, 12)

    def test_all_dates_invalid(self) -> None:
        frame = daily_frame("2026-08-03", 5)
        frame[config.COL_DATE] = "坏日期"

        analysis = build_trend_series(frame)

        self.assertFalse(analysis.available)
        self.assertEqual(analysis.period_count, 0)
        self.assertEqual(analysis.invalid_date_rows, 5)
        self.assertTrue(any("日期异常" in item for item in analysis.warnings))

    def test_partial_periods_are_excluded(self) -> None:
        # 数据从周三开始、到次周周二结束：首尾周都不完整
        frame = daily_frame("2026-08-05", 13)  # 2026-08-05(周三) ~ 2026-08-17(周一)
        analysis = build_trend_series(frame, granularity=config.TREND_GRANULARITY_WEEK)

        self.assertTrue(analysis.partial_periods)
        self.assertTrue(any("周期不完整" in item for item in analysis.warnings))

    def test_period_frame_marks_completeness(self) -> None:
        frame = daily_frame("2026-08-03", 14)
        analysis = build_trend_series(frame, granularity=config.TREND_GRANULARITY_WEEK)
        period_frame = analysis.series_frame()

        self.assertIn("完整", period_frame.columns)
        self.assertTrue(bool(period_frame["完整"].all()))


class TrendOutputTests(unittest.TestCase):
    def test_to_frame_columns(self) -> None:
        analysis = build_trend_series(
            daily_frame("2026-08-03", 21), granularity=config.TREND_GRANULARITY_WEEK
        )
        frame = analysis.to_frame()

        self.assertEqual(len(frame), 7)
        for column in ("指标", "最新期间", "最新值", "环比(%)", "方向", "连续变化", "同期变化(%)"):
            self.assertIn(column, frame.columns)

    def test_chart_frame(self) -> None:
        analysis = build_trend_series(
            daily_frame("2026-08-03", 21), granularity=config.TREND_GRANULARITY_WEEK
        )
        chart = analysis.chart_frame("fault_rate")

        self.assertEqual(list(chart.columns), ["期间", "数值"])
        self.assertEqual(len(chart), analysis.period_count)
        self.assertTrue(chart["数值"].notna().all())

    def test_as_dict_is_json_serializable(self) -> None:
        analysis = build_trend_series(
            daily_frame("2026-08-03", 21), granularity=config.TREND_GRANULARITY_WEEK
        )
        payload = analysis.as_dict()
        text = json.dumps(payload, ensure_ascii=False)

        self.assertIn("fault_rate", payload["metrics"])
        self.assertIn("granularity", payload)
        self.assertIn("periods", payload)
        self.assertLess(len(text), 50_000)

    def test_highlight_lines_describe_trends_without_interpretation(self) -> None:
        frame = daily_frame(
            "2026-08-03",
            21,
            per_day=lambda index: {config.COL_FAULT_COUNT: 0.4 + (index // 7) * 0.2},
        )
        analysis = build_trend_series(frame, granularity=config.TREND_GRANULARITY_WEEK)
        lines = analysis.highlight_lines()

        self.assertTrue(lines)
        self.assertTrue(any("故障率" in line for line in lines))
        self.assertTrue(all("推测" not in line for line in lines))

    def test_summary_text_contains_window(self) -> None:
        analysis = build_trend_series(
            daily_frame("2026-08-03", 21), granularity=config.TREND_GRANULARITY_WEEK
        )
        text = analysis.summary_text()

        self.assertIn("运营趋势", text)
        self.assertIn("分析窗口", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

