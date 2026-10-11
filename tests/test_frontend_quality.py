"""Phase 8 前端展示测试：数据质量检查区域与运营指标体系区域。

分两层：
1. 纯函数层（frontend/quality_helpers.py）：不依赖 Streamlit，直接断言界面数据结构；
2. AppTest 无头渲染：验证页面上确实出现「数据质量检查」与「运营指标体系（Phase 8）」。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from frontend import quality_helpers as uiq  # noqa: E402
from robotops import config  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        config.COL_DATE: "2026-08-01",
        config.COL_PROJECT: "项目甲",
        config.COL_ROBOT_ID: "R-01",
        config.COL_ROBOT_TYPE: "清洁机器人",
        config.COL_RUNNING_HOURS: 9.0,
        config.COL_PLANNED_HOURS: 10.0,
        config.COL_FAULT_COUNT: 0,
        config.COL_INSPECTION_COUNT: 10,
        config.COL_REPAIR_COUNT: 0,
        config.COL_SATISFACTION: 92.0,
        config.COL_COST: 1000.0,
        config.COL_SAVING_RATE: 15.0,
    }
    row.update(overrides)
    return row


def clean_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(),
            _row(**{config.COL_ROBOT_ID: "R-02", config.COL_DATE: "2026-08-02"}),
        ]
    )


class QualityPanelDataTests(unittest.TestCase):
    def test_missing_report(self) -> None:
        panel = uiq.quality_panel_data(None)

        self.assertFalse(panel["available"])
        self.assertTrue(panel["can_analyze"])
        self.assertEqual(panel["block_level"], "info")

    def test_pass_report(self) -> None:
        panel = uiq.quality_panel_data(uiq.check_quality(clean_frame(), frame_name="ok.xlsx"))

        self.assertTrue(panel["available"])
        self.assertEqual(panel["status"], config.QUALITY_STATUS_PASS)
        self.assertEqual(panel["block_level"], "success")
        self.assertTrue(panel["can_analyze"])
        self.assertTrue(any(line == "✓ 字段完整" for line in panel["lines"]))
        self.assertFalse(panel["checks"].empty)
        self.assertEqual(panel["block_message"], "")

    def test_warning_report(self) -> None:
        frame = clean_frame()
        frame.loc[0, config.COL_COST] = -50.0
        panel = uiq.quality_panel_data(uiq.check_quality(frame))

        self.assertEqual(panel["status"], config.QUALITY_STATUS_WARNING)
        self.assertEqual(panel["block_level"], "warning")
        self.assertTrue(panel["can_analyze"])
        self.assertEqual(panel["counts"]["invalid_values"], 1)
        self.assertTrue(any(line.startswith("⚠") for line in panel["lines"]))
        self.assertFalse(panel["problems"].empty)

    def test_error_report_blocks_analysis(self) -> None:
        frame = clean_frame().drop(columns=[config.COL_RUNNING_HOURS])
        panel = uiq.quality_panel_data(uiq.check_quality(frame))

        self.assertEqual(panel["status"], config.QUALITY_STATUS_ERROR)
        self.assertEqual(panel["block_level"], "error")
        self.assertFalse(panel["can_analyze"])
        self.assertIn("已禁止进入 Agent 分析", panel["block_message"])
        self.assertTrue(any("禁止" in line for line in panel["lines"]))

    def test_panel_lines_cover_required_ui_items(self) -> None:
        frame = clean_frame()
        frame.loc[0, config.COL_SATISFACTION] = None
        panel = uiq.quality_panel_data(uiq.check_quality(frame))
        text = " ".join(panel["lines"])

        self.assertIn("字段完整", text)
        self.assertIn("缺失数据", text)
        self.assertIn("无重复记录", text)


class IndicatorPanelDataTests(unittest.TestCase):
    def test_missing_data(self) -> None:
        panel = uiq.indicator_panel_data(None)

        self.assertFalse(panel["available"])
        self.assertEqual(panel["anomaly_count"], 0)
        self.assertEqual(len(panel["thresholds_table"]), 5)

    def test_indicator_metrics_and_thresholds(self) -> None:
        panel = uiq.indicator_panel_data(clean_frame())

        self.assertTrue(panel["available"])
        self.assertEqual(len(panel["metrics"]), 9)
        self.assertIn(config.COL_UTILIZATION_RATE, set(panel["metrics"]["指标"]))
        self.assertIn("全部项目", set(panel["project_table"]["项目名称"]))
        self.assertGreaterEqual(panel["anomaly_count"] + panel["warning_count"], 0)

    def test_low_utilization_reports_anomaly(self) -> None:
        frame = pd.DataFrame(
            [
                _row(**{config.COL_RUNNING_HOURS: 4.0}),
                _row(**{config.COL_ROBOT_ID: "R-02", config.COL_DATE: "2026-08-02",
                        config.COL_RUNNING_HOURS: 4.0}),
            ]
        )
        panel = uiq.indicator_panel_data(frame)

        self.assertGreaterEqual(panel["anomaly_count"], 1)
        self.assertIn("异常", panel["headline"])
        hits = panel["anomalies"]
        self.assertIn(config.COL_UTILIZATION_RATE, set(hits["指标"]))

    def test_custom_thresholds_change_result(self) -> None:
        frame = pd.DataFrame([_row(**{config.COL_RUNNING_HOURS: 8.0})])
        strict = config.IndicatorThresholds(utilization_lower=95.0)
        panel = uiq.indicator_panel_data(frame, thresholds=strict)

        self.assertGreaterEqual(panel["anomaly_count"], 1)


class Phase8PageTests(unittest.TestCase):
    """无头渲染：页面上必须出现数据质量检查与指标体系区域。"""

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

    def test_quality_section_rendered(self) -> None:
        subheaders = [item.value for item in self.app.subheader]
        self.assertIn("数据质量检查", subheaders)
        self.assertIn("二、开始分析", subheaders)  # 既有章节编号保持不变

        captions = " ".join(item.value for item in self.app.caption)
        self.assertIn("数据质量报告", captions)

    def test_quality_lines_and_status(self) -> None:
        markdown_text = " ".join(item.value for item in self.app.markdown)
        captions = " ".join(item.value for item in self.app.caption)

        self.assertIn("✓ 字段完整", markdown_text)
        self.assertIn("缺失数据", markdown_text + captions)
        self.assertIn("状态：", captions)

    def test_analyze_allowed_for_warning_data(self) -> None:
        buttons = [button for button in self.app.button if button.label == "开始分析"]
        self.assertTrue(buttons)
        self.assertFalse(buttons[0].disabled, "演示数据为警告级别，应允许继续分析")

    def test_indicator_section_after_analysis(self) -> None:
        for button in self.app.button:
            if button.label == "开始分析":
                button.click()
        self.app.run()

        self.assertEqual([str(item.value) for item in self.app.exception], [])
        subheaders = [item.value for item in self.app.subheader]
        self.assertIn("运营指标体系（Phase 8）", subheaders)

        frames = [item.value for item in self.app.dataframe]
        indicator_frames = [frame for frame in frames if "指标" in getattr(frame, "columns", [])]
        self.assertTrue(indicator_frames, "页面上应展示运营指标表")
        metrics = set()
        for frame in indicator_frames:
            metrics.update(str(name) for name in frame["指标"].tolist())
        self.assertIn(config.COL_UTILIZATION_RATE, metrics)
        self.assertIn(config.COL_MTBF, metrics)


if __name__ == "__main__":
    unittest.main(verbosity=2)

