"""Phase 9 前端测试：趋势图表、趋势变化表与项目对比区域。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from frontend import trend_helpers as uit  # noqa: E402
from robotops import config  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")


def _row(project: str, date: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        config.COL_DATE: date,
        config.COL_PROJECT: project,
        config.COL_ROBOT_ID: project + "-01",
        config.COL_ROBOT_TYPE: "清洁机器人",
        config.COL_RUNNING_HOURS: 8.0,
        config.COL_PLANNED_HOURS: 10.0,
        config.COL_FAULT_COUNT: 0.5,
        config.COL_INSPECTION_COUNT: 10,
        config.COL_REPAIR_COUNT: 1,
        config.COL_SATISFACTION: 90.0,
        config.COL_COST: 1000.0,
        config.COL_SAVING_RATE: 15.0,
    }
    row.update(overrides)
    return row


def three_week_frame() -> pd.DataFrame:
    """3 个完整自然周 x 2 个项目，故障率上升、成本上升。"""

    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2026-08-03")
    for index in range(21):
        day = (start + pd.Timedelta(days=index)).strftime("%Y-%m-%d")
        week = index // 7
        for project in ("A小区", "B小区"):
            rows.append(
                _row(
                    project,
                    day,
                    **{
                        config.COL_FAULT_COUNT: 0.4 + week * 0.2,
                        config.COL_COST: 1000.0 + week * 200.0,
                        config.COL_SATISFACTION: 92.0 - week * 2.0,
                    },
                )
            )
    return pd.DataFrame(rows)


class TrendPanelDataTests(unittest.TestCase):
    def test_granularity_choice_mapping(self) -> None:
        self.assertIsNone(uit.granularity_choice("自动"))
        self.assertEqual(uit.granularity_choice("周"), config.TREND_GRANULARITY_WEEK)
        self.assertEqual(uit.granularity_choice("月"), config.TREND_GRANULARITY_MONTH)
        self.assertEqual(uit.granularity_choice("未知"), None)

    def test_trend_panel_structure(self) -> None:
        panel = uit.trend_panel_data(three_week_frame(), granularity=config.TREND_GRANULARITY_WEEK)

        self.assertTrue(panel["available"])
        self.assertEqual(panel["granularity_label"], "周")
        self.assertEqual(panel["period_count"], 3)
        self.assertEqual(len(panel["metric_options"]), 7)
        self.assertIn("自动", panel["granularity_options"])
        for key in ("fault_rate", "satisfaction", "total_cost"):
            self.assertIn(key, panel["charts"])
            self.assertEqual(list(panel["charts"][key].columns), ["期间", "数值"])
        self.assertIn("环比(%)", panel["change_frame"].columns)

    def test_trend_panel_reports_rising_and_falling(self) -> None:
        panel = uit.trend_panel_data(three_week_frame(), granularity=config.TREND_GRANULARITY_WEEK)
        changes = panel["change_frame"].set_index("指标")

        self.assertEqual(changes.loc["故障率", "方向"], config.TREND_UP)
        self.assertEqual(changes.loc["用户满意度", "方向"], config.TREND_DOWN)
        self.assertEqual(changes.loc["运营成本", "方向"], config.TREND_UP)
        self.assertTrue(panel["highlights"])

    def test_trend_panel_without_data(self) -> None:
        for value in (None, pd.DataFrame()):
            panel = uit.trend_panel_data(value)
            self.assertFalse(panel["available"])
            self.assertTrue(panel["headline"])

    def test_comparison_panel_structure(self) -> None:
        panel = uit.comparison_panel_data(three_week_frame())

        self.assertTrue(panel["available"])
        self.assertEqual(panel["project_count"], 2)
        self.assertEqual(list(panel["chart"].index), ["A小区", "B小区"])
        self.assertEqual(list(panel["chart"].columns), ["设备运行率", "设备利用率", "故障率"])
        self.assertTrue(panel["highlights"])

    def test_comparison_panel_without_data(self) -> None:
        panel = uit.comparison_panel_data(None)

        self.assertFalse(panel["available"])
        self.assertTrue(panel["headline"])

    def test_ranking_table(self) -> None:
        frame = uit.ranking_table(three_week_frame(), "fault_rate")

        self.assertIn("排名", frame.columns)
        self.assertEqual(len(frame), 2)


class Phase9PageTests(unittest.TestCase):
    """无头渲染：分析完成后必须出现趋势与对比区域。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = AppTest.from_file(APP_PATH, default_timeout=180)
        cls.app.run()
        for toggle in cls.app.toggle:
            if toggle.label == "启用 AI 分析（DeepSeek）":
                toggle.set_value(False)
        cls.app.run()
        for button in cls.app.button:
            if button.label == "使用项目演示数据试跑":
                button.click()
        cls.app.run()
        for button in cls.app.button:
            if button.label == "开始分析":
                button.click()
        cls.app.run()

    def test_no_exception(self) -> None:
        self.assertEqual([str(item.value) for item in self.app.exception], [])

    def test_phase9_section_rendered(self) -> None:
        subheaders = [item.value for item in self.app.subheader]

        self.assertIn("运营趋势与项目对比（Phase 9）", subheaders)

    def test_granularity_and_comparison_controls(self) -> None:
        radios = {item.label: item for item in self.app.radio}
        self.assertIn("趋势粒度", radios)
        self.assertEqual(radios["趋势粒度"].value, "自动")

        multiselects = {item.label: item for item in self.app.multiselect}
        self.assertIn("选择对比指标", multiselects)
        self.assertTrue(multiselects["选择对比指标"].value)

    def test_charts_rendered(self) -> None:
        charts = self.app.get("vega_lite_chart")

        # 故障率 / 满意度 / 成本 / 其他指标 + 项目对比
        self.assertGreaterEqual(len(charts), 4, "应渲染趋势折线图与项目对比图")

    def test_trend_and_comparison_tables(self) -> None:
        frames = [item.value for item in self.app.dataframe]
        columns = [list(getattr(frame, "columns", [])) for frame in frames]

        self.assertTrue(
            any("环比(%)" in cols and "方向" in cols for cols in columns),
            "应展示趋势变化表（含环比与方向）",
        )
        self.assertTrue(
            any("设备利用率" in cols and "项目名称" in cols for cols in columns),
            "应展示项目横向对比表",
        )

    def test_state_carries_phase9_data(self) -> None:
        state = self.app.session_state["analysis_state"]

        self.assertIn("trend_analysis", state)
        self.assertIn("project_comparison", state)
        self.assertTrue(state["trend_analysis"].get("period_count"))
        self.assertTrue(state["project_comparison"].get("project_count"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

