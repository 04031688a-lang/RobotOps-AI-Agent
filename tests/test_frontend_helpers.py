"""Phase 5：前端展示辅助函数测试（纯逻辑，不依赖 Streamlit 运行时）。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from frontend import ui_helpers as ui  # noqa: E402
from robotops import config  # noqa: E402
from robotops.data_cleaner import clean_operation_data  # noqa: E402
from robotops.llm import config as llm_config  # noqa: E402
from robotops.models import COLUMN_SPECS  # noqa: E402
from tests._helpers import base_operation_frame, make_temp_dir, write_excel  # noqa: E402


def sample_state() -> dict:
    """构造一份最小的工作流 State（与 Phase 4 的输出结构一致）。"""

    return {
        "has_anomalies": True,
        "anomaly_count": 3,
        "thresholds": config.DEFAULT_THRESHOLDS.as_dict(),
        "metrics": {
            "values": {
                "项目数量": 1,
                "机器人数量": 2,
                "平均运行时长": 15.0,
                "故障率": 15.0,
                "平均满意度": 85.0,
                "总运营成本": 800.0,
                "平均节降率": 15.0,
            },
            "units": {"故障率": "%", "平均节降率": "%"},
        },
        "raw_data_summary": {"数据源": "demo.xlsx", "清洗后记录数": 3, "原始记录数": 3},
        "anomaly_summary": [
            {
                "anomaly_type": "故障率偏高",
                "condition": "> 5%",
                "hits": 1,
                "projects_involved": 1,
                "robots_involved": 1,
                "high": 1,
                "medium": 0,
                "low": 0,
            }
        ],
        "abnormal_projects": [
            {
                "project": "项目甲",
                "anomaly_records": 3,
                "affected_robots": 2,
                "high_severity": 1,
                "medium_severity": 0,
                "low_severity": 2,
                "main_anomaly_types": "故障率偏高(2)、满意度偏低(1)",
            }
        ],
        "diagnosis": {
            "source": "llm",
            "summary": "推测：与部件劣化有关。",
            "possible_reasons": [
                {
                    "reason": "推测：该设备可能存在部件劣化",
                    "confidence": "中",
                    "based_on": "故障率高于同项目其他设备",
                    "data_gap": "缺少故障类型字段",
                    "anomaly_type": "故障率偏高",
                }
            ],
        },
        "retrieved_cases": [
            {
                "case_id": "CASE-FAULT-001",
                "title": "清洁机器人重复故障（同一部件反复损坏）",
                "case_type": "机器人重复故障",
                "similarity": 0.472,
                "source_file": "CASE-FAULT-001.md",
                "data_nature": "模拟案例（虚构）",
                "matched_sections": ["问题", "处理措施"],
                "case_summary": (
                    "案例编号：CASE-FAULT-001\n案例标题：清洁机器人重复故障\n"
                    "【问题】该台机器人两周内出现 6 次同类故障。\n"
                    "【可能原因】只更换部件、未排查根因。\n"
                    "【处理措施】下线检修并调换作业区域。"
                ),
            }
        ],
        "recommendations": {
            "source": "llm",
            "items": [
                {
                    "action": "对该设备做单机专项复盘",
                    "priority": "高",
                    "target": "高故障机器人",
                    "expected_effect": "定位根因",
                    "verification": "补齐故障类型后重新计算故障率",
                    "reference_case": "CASE-FAULT-001",
                }
            ],
        },
        "final_report": "# RobotOps AI 运营分析报告\n\n## 一、运营概览\n\n示例内容。\n",
        "steps": [
            {
                "agent": "Data Analysis Agent",
                "status": "completed",
                "duration_seconds": 0.5,
                "detail": "项目 1 个 / 机器人 2 台",
            }
        ],
        "errors": [
            {
                "agent": "RAG Agent",
                "kind": "业务处理失败",
                "message": "知识库目录不存在",
                "hint": "执行 rebuild",
            }
        ],
    }


class UploadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = make_temp_dir(prefix="robotops_test_frontend_")

    def tearDown(self) -> None:
        ui.clear_upload_dir(self.tmp)

    def test_save_uploaded_file_rejects_unsupported_suffix(self) -> None:
        with self.assertRaises(ValueError):
            ui.save_uploaded_file("data.txt", b"hello", upload_dir=self.tmp)

    def test_save_and_preview_valid_file(self) -> None:
        source = write_excel(base_operation_frame(), self.tmp / "demo.xlsx")

        path = ui.save_uploaded_file("demo.xlsx", source.read_bytes(), upload_dir=self.tmp / "uploads")
        preview = ui.preview_uploaded_file(path)

        self.assertTrue(preview.is_valid)
        self.assertEqual(preview.row_count, 3)
        self.assertEqual(preview.missing_columns, [])
        self.assertEqual(preview.sheet_name, "运营数据")

    def test_preview_reports_missing_columns(self) -> None:
        frame = base_operation_frame().drop(columns=[config.COL_SATISFACTION])
        source = write_excel(frame, self.tmp / "missing.xlsx")

        preview = ui.preview_uploaded_file(source)

        self.assertFalse(preview.is_valid)
        self.assertTrue(preview.is_readable)
        self.assertEqual(preview.missing_columns, [config.COL_SATISFACTION])

    def test_preview_reports_unreadable_file(self) -> None:
        broken = self.tmp / "broken.xlsx"
        broken.write_bytes(b"this is not an excel file")

        preview = ui.preview_uploaded_file(broken)

        self.assertFalse(preview.is_readable)
        self.assertTrue(preview.error_message)
        self.assertTrue(preview.frame.empty)

    def test_field_info_frame(self) -> None:
        info = ui.field_info_frame(base_operation_frame())

        self.assertIn("字段名", info.columns)
        self.assertIn("是否必需", info.columns)
        required = info.loc[info["字段名"] == config.COL_PROJECT, "是否必需"].iloc[0]
        self.assertEqual(required, "必需")
        optional = info.loc[info["字段名"] == config.COL_PLANNED_HOURS, "是否必需"].iloc[0]
        self.assertEqual(optional, "可选")
        self.assertEqual(ui.required_columns_frame().shape[0], len(COLUMN_SPECS))


class OverviewTests(unittest.TestCase):
    def test_overview_cards_cover_required_metrics(self) -> None:
        cards = ui.overview_cards(sample_state()["metrics"]["values"], {"故障率": "%"})

        labels = [card["label"] for card in cards]
        self.assertEqual(
            labels,
            [
                "项目数量",
                "机器人数量",
                "平均运行时长",
                "故障率",
                "平均满意度",
                "总运营成本",
                "平均节降率",
            ],
        )
        self.assertEqual(cards[0]["value"], "1")
        self.assertEqual(cards[3]["value"], "15%")
        self.assertTrue(cards[0]["help"])

    def test_format_metric_value(self) -> None:
        self.assertEqual(ui.format_metric_value(None), "-")
        self.assertEqual(ui.format_metric_value(float("nan")), "-")
        self.assertEqual(ui.format_metric_value(6), "6")
        self.assertEqual(ui.format_metric_value(1111739.6), "1,111,739.60")
        self.assertEqual(ui.format_metric_value(2.06, "%"), "2.06%")


class TrendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cleaned = clean_operation_data(base_operation_frame()).data
        cls.cleaned = pd.concat(
            [
                cls.cleaned,
                cls.cleaned.assign(**{config.COL_DATE: pd.Timestamp("2026-08-02")}),
            ],
            ignore_index=True,
        )

    def test_daily_trend(self) -> None:
        trend = ui.daily_trend(self.cleaned, config.COL_FAULT_RATE)

        self.assertIn(config.COL_DATE, trend.columns)
        self.assertIn(config.COL_FAULT_RATE, trend.columns)
        self.assertEqual(len(trend), 2)

    def test_trend_values_and_series_mapping(self) -> None:
        values = ui.trend_values(self.cleaned, "故障率")

        self.assertEqual(len(values), 2)
        self.assertTrue(all(isinstance(value, float) for value in values))
        self.assertEqual(ui.trend_values(self.cleaned, "项目数量"), [])
        self.assertEqual(
            ui.trend_series_for_anomalies("故障率偏高(2)、满意度偏低(1)"), ["故障率", "满意度"]
        )
        self.assertEqual(ui.trend_series_for_anomalies(""), [])

    def test_project_daily_metrics(self) -> None:
        trend = ui.project_daily_metrics(self.cleaned, "项目甲")

        self.assertIn("故障率", trend.columns)
        self.assertIn("满意度", trend.columns)
        self.assertEqual(len(trend), 2)
        self.assertTrue(ui.project_daily_metrics(self.cleaned, "不存在的项目").empty)


class StateRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = sample_state()

    def test_abnormal_project_cards(self) -> None:
        cards = ui.abnormal_project_cards(self.state)

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["project"], "项目甲")
        self.assertIn("故障率偏高", cards[0]["main_anomaly_types"])
        self.assertTrue(cards[0]["reasons"])
        self.assertIn("推测", cards[0]["reasons"][0]["reason"])

    def test_case_cards_include_disclaimer_and_sections(self) -> None:
        cards = ui.case_cards(self.state)

        self.assertEqual(len(cards), 1)
        card = cards[0]
        self.assertEqual(card["title"], "清洁机器人重复故障（同一部件反复损坏）")
        self.assertEqual(card["case_type"], "机器人重复故障")
        self.assertIn("未排查根因", card["possible_reason"])
        self.assertIn("下线检修", card["actions"])
        self.assertEqual(card["disclaimer"], ui.CASE_DISCLAIMER)
        self.assertIn("不代表当前项目实际情况", card["disclaimer"])

    def test_parse_case_sections(self) -> None:
        sections = ui.parse_case_sections(self.state["retrieved_cases"][0]["case_summary"])

        self.assertEqual(set(sections), {"问题", "可能原因", "处理措施"})
        self.assertIn("6 次同类故障", sections["问题"])
        self.assertEqual(ui.parse_case_sections(""), {})

    def test_errors_and_trace(self) -> None:
        errors = ui.normalize_errors(self.state)
        trace = ui.workflow_trace(self.state)

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["agent"], "RAG Agent")
        self.assertIn("知识库目录不存在", errors[0]["message"])
        self.assertEqual(list(trace.columns), ["节点", "状态", "耗时(秒)", "说明"])
        self.assertEqual(trace.iloc[0]["节点"], "Data Analysis Agent")
        self.assertTrue(ui.workflow_trace({}).empty)

    def test_report_download_payload(self) -> None:
        payload = ui.report_download_payload(self.state)

        self.assertEqual(payload["file_name"], "robotops_operation_report.md")
        self.assertEqual(payload["mime"], "text/markdown")
        self.assertIn("RobotOps AI 运营分析报告", payload["markdown"])
        self.assertIn("附：工作流执行轨迹", payload["markdown"])
        self.assertIn("RAG Agent", payload["markdown"])
        self.assertTrue(payload["markdown"].endswith("\n"))

    def test_report_download_payload_without_report(self) -> None:
        payload = ui.report_download_payload({})

        self.assertIn("本次运行未生成报告内容", payload["markdown"])

    def test_api_key_state_and_describe_error(self) -> None:
        snapshot = dict(os.environ)
        try:
            os.environ[llm_config.ENV_API_KEY] = "sk-abcdefghijklmnop"
            has_key, masked = ui.api_key_state()
            self.assertTrue(has_key)
            self.assertNotIn("abcdefghijklmnop", masked)
            self.assertIn("****", masked)
        finally:
            os.environ.clear()
            os.environ.update(snapshot)

        self.assertEqual(ui.describe_error(RuntimeError("boom\nboom2")), "boom boom2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
