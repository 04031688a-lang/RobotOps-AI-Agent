"""Phase 2：Agent 输入载荷测试（必须来自 Pandas 结构化结果，而不是原始 Excel）。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from robotops import config  # noqa: E402
from robotops.agent.payload import (  # noqa: E402
    METRIC_UNITS,
    PAYLOAD_VERSION,
    build_analysis_payload,
    payload_to_json,
)
from robotops.agent.robot_ops_agent import RobotOpsAnalysisAgent  # noqa: E402
from robotops.exceptions import ConfigurationError  # noqa: E402
from robotops.llm.config import LLMConfig  # noqa: E402
from robotops.pipeline import run_analysis  # noqa: E402
from scripts.generate_demo_data import generate_demo_file  # noqa: E402


class AgentPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not config.DEMO_EXCEL_FILE.exists():
            generate_demo_file(config.DEMO_EXCEL_FILE)
        cls.result = run_analysis(export=False, verbose=False)
        cls.payload = build_analysis_payload(cls.result)

    def test_top_level_structure(self) -> None:
        self.assertEqual(
            list(self.payload.keys()),
            [
                "payload_version",
                "meta",
                "units",
                "core_metrics",
                "metric_definitions",
                "projects",
                "abnormal_projects",
                "abnormal_robots",
                "anomaly_summary",
                "data_quality",
                "notes",
            ],
        )
        self.assertEqual(self.payload["payload_version"], PAYLOAD_VERSION)

    def test_core_metrics_are_copied_from_phase1(self) -> None:
        metrics = self.result.metrics
        core = self.payload["core_metrics"]

        self.assertEqual(core["project_count"], metrics.project_count)
        self.assertEqual(core["robot_count"], metrics.robot_count)
        self.assertEqual(core["record_count"], metrics.record_count)
        self.assertAlmostEqual(core["avg_runtime"], round(metrics.avg_running_hours, 2), places=2)
        self.assertAlmostEqual(core["avg_fault_rate"], round(metrics.fault_rate, 2), places=2)
        self.assertAlmostEqual(core["avg_satisfaction"], round(metrics.avg_satisfaction, 2), places=2)
        self.assertAlmostEqual(core["total_cost"], round(metrics.total_cost, 2), places=2)
        self.assertAlmostEqual(
            core["avg_cost_reduction_rate"], round(metrics.avg_saving_rate, 2), places=2
        )
        self.assertEqual(core["total_fault_count"], metrics.total_fault_count)
        self.assertAlmostEqual(
            core["maintenance_to_fault_ratio"],
            round(
                self.result.cleaned_data[config.COL_REPAIR_COUNT].sum()
                / metrics.total_fault_count
                * 100,
                2,
            ),
            places=2,
        )

    def test_units_cover_every_core_metric(self) -> None:
        for key in self.payload["core_metrics"]:
            with self.subTest(metric=key):
                self.assertIn(key, self.payload["units"])
                self.assertTrue(self.payload["units"][key])
                self.assertEqual(self.payload["units"][key], METRIC_UNITS[key])

    def test_projects_and_rollups_are_consistent(self) -> None:
        projects = self.payload["projects"]
        summary = self.result.project_summary

        self.assertEqual(len(projects), self.result.metrics.project_count)
        self.assertEqual([item["project"] for item in projects], list(summary["项目名称"]))
        self.assertEqual(
            sum(item["robot_count"] for item in projects), self.result.metrics.robot_count
        )

    def test_abnormal_projects_are_subset_and_complete(self) -> None:
        project_names = {item["project"] for item in self.payload["projects"]}
        abnormal = self.payload["abnormal_projects"]
        anomalies = self.result.anomalies

        self.assertTrue(abnormal)
        for item in abnormal:
            self.assertIn(item["project"], project_names)
        self.assertEqual(
            sum(item["anomaly_records"] for item in abnormal), int(len(anomalies))
        )

    def test_abnormal_robots_match_anomaly_records(self) -> None:
        expected = set(anomalies_robots(self.result))
        actual = {item["robot_id"] for item in self.payload["abnormal_robots"]}

        self.assertEqual(actual, expected)
        for item in self.payload["abnormal_robots"]:
            self.assertGreater(item["anomaly_records"], 0)
            self.assertTrue(item["anomaly_types"])

    def test_payload_contains_no_raw_export_data(self) -> None:
        text = payload_to_json(self.payload)

        self.assertLess(len(text), 200_000, "载荷过大，可能混入了逐条明细")
        self.assertNotIn("records", self.payload)
        self.assertNotIn("cleaned_data", self.payload)
        for key in ("projects", "abnormal_robots"):
            for item in self.payload[key]:
                self.assertLessEqual(len(item), 15, f"{key} 元素字段过多，疑似包含明细")
        json.loads(text)  # 可序列化

    def test_notes_state_the_guardrails(self) -> None:
        notes = " ".join(self.payload["notes"])

        self.assertIn("不得重新计算", notes)
        self.assertIn("累计口径", notes)
        self.assertIn("数据中不含", notes)

    def test_meta_carries_context(self) -> None:
        meta = self.payload["meta"]

        self.assertEqual(meta["data_source"], config.DEMO_EXCEL_FILE.name)
        self.assertEqual(meta["sheet_name"], "运营数据")
        self.assertEqual(meta["cleaned_record_count"], self.result.metrics.record_count)
        self.assertIn("~", meta["analysis_window"])
        self.assertEqual(meta["thresholds"], self.result.thresholds.as_dict())

    def test_data_quality_section(self) -> None:
        quality = self.payload["data_quality"]

        self.assertEqual(quality["cleaning_stats"]["清洗后记录数"], self.result.metrics.record_count)
        self.assertEqual(quality["issues_total"], len(self.result.cleaned.issues))
        self.assertEqual(len(quality["issues"]), quality["issues_total"])

    def test_raw_dataframe_is_rejected(self) -> None:
        frame = pd.DataFrame([{config.COL_DATE: "2026-08-01"}])

        with self.assertRaises(TypeError):
            build_analysis_payload(frame)  # type: ignore[arg-type]

        agent = RobotOpsAnalysisAgent(config=LLMConfig(api_key="sk-test"))
        with self.assertRaises(ConfigurationError):
            agent.build_payload(frame)  # type: ignore[arg-type]


def anomalies_robots(result) -> list[str]:
    return sorted(set(result.anomalies["机器人ID"].tolist()))


if __name__ == "__main__":
    unittest.main(verbosity=2)

