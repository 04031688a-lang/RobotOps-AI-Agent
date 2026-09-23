"""端到端测试：演示数据生成 -> 分析 -> 结果导出 -> 命令行入口。"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from robotops import config  # noqa: E402
from robotops.cli import main  # noqa: E402
from robotops.pipeline import run_analysis  # noqa: E402
from scripts.generate_demo_data import DEFAULT_DAYS, build_demo_dataframe, generate_demo_file  # noqa: E402

CORE_METRIC_NAMES = {
    "项目数量",
    "机器人数量",
    "平均运行时长",
    "故障率",
    "平均满意度",
    "总运营成本",
    "平均节降率",
}


class PipelineEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._temp.name)
        cls.data_path = cls.tmp / "robot_operation_demo.xlsx"
        generate_demo_file(cls.data_path)
        cls.output_dir = cls.tmp / "output"
        cls.result = run_analysis(
            cls.data_path, output_dir=cls.output_dir, verbose=False
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_cleaning_handles_injected_dirty_rows(self) -> None:
        stats = self.result.cleaned.stats
        self.assertEqual(stats["原始记录数"], 23 * DEFAULT_DAYS + 7)
        # 7 条脏数据中：1 条完全重复 + 1 条同机器人同日重复上报被剔除，
        # 其余 5 条属于字段级问题，记录本身保留
        self.assertEqual(stats["剔除记录数"], 2)
        self.assertEqual(stats["清洗后记录数"], 23 * DEFAULT_DAYS + 5)
        self.assertEqual(stats["机器人数"], 23)
        self.assertEqual(stats["项目数"], 6)
        self.assertEqual(len(self.result.anomalies), self.result.anomaly_count)
        self._assert_issues_cover_injected_problems()

    def _assert_issues_cover_injected_problems(self) -> None:
        """清洗问题清单应覆盖演示数据注入的各类脏数据。"""

        kinds = set(self.result.cleaned.issues["问题类型"])
        expected = {
            "空白字符",
            "完全重复记录",
            "同一机器人同一天重复上报",
            "满意度越界",
            "运营成本为负",
            "关键指标缺失",
        }
        self.assertTrue(expected.issubset(kinds), f"缺少问题类型：{expected - kinds}")

    def test_core_metrics_are_reasonable(self) -> None:
        metrics = self.result.metrics
        self.assertEqual(metrics.project_count, 6)
        self.assertEqual(metrics.robot_count, 23)
        self.assertGreater(metrics.avg_running_hours, 0)
        self.assertGreater(metrics.fault_rate, 0)
        self.assertTrue(80 <= metrics.avg_satisfaction <= 100)
        self.assertGreater(metrics.total_cost, 0)
        self.assertGreater(metrics.avg_saving_rate, 0)
        self.assertEqual(set(self.result.metrics_dict()) & CORE_METRIC_NAMES, CORE_METRIC_NAMES)

    def test_anomalies_cover_all_four_rules(self) -> None:
        anomalies = self.result.anomalies
        self.assertFalse(anomalies.empty)
        self.assertEqual(
            set(anomalies["异常类型"]),
            {"故障率偏高", "满意度偏低", "运行率偏低", "节降率偏低"},
        )
        fault_rows = anomalies.loc[anomalies["异常类型"] == "故障率偏高"]
        self.assertTrue((fault_rows["判定粒度"] == "机器人周期累计").all())
        self.assertTrue(
            (anomalies.loc[anomalies["异常类型"] != "故障率偏高", "判定粒度"] == "逐日记录").all()
        )

    def test_exported_files(self) -> None:
        files = self.result.exported_files
        self.assertEqual(len(files), 7)
        for path in files:
            self.assertTrue(path.exists(), f"{path} 未生成")

        excel_path = next(path for path in files if path.suffix == ".xlsx")
        with pd.ExcelFile(excel_path, engine="openpyxl") as workbook:
            for sheet in ("核心指标", "项目汇总", "异常明细", "清洗问题", "指标口径", "异常规则"):
                self.assertIn(sheet, workbook.sheet_names)

        json_path = next(path for path in files if path.name.endswith("_metrics.json"))
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["阶段"], "Phase 1 - 机器人运营数据分析基础模块")
        self.assertIn("故障率", payload["核心指标"])
        self.assertEqual(payload["核心指标"]["项目数量"], 6)

    def test_csv_input_supported(self) -> None:
        csv_path = self.tmp / "robot_operation_demo.csv"
        build_demo_dataframe(days=3).to_csv(csv_path, index=False, encoding="utf-8-sig")

        result = run_analysis(csv_path, export=False, verbose=False)

        self.assertEqual(result.metrics.project_count, 6)
        self.assertEqual(result.metrics.robot_count, 23)
        self.assertFalse(result.exported_files)

    def test_cli_success_and_failure_exit_codes(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            success_code = main(["--data", str(self.data_path), "--no-export", "--quiet"])
            failure_code = main(["--data", str(self.tmp / "missing.xlsx"), "--quiet"])

        self.assertEqual(success_code, 0)
        self.assertEqual(failure_code, 2)

    def test_cli_threshold_override_changes_hits(self) -> None:
        strict_file = self.tmp / "robot_operation_demo.xlsx"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(
                [
                    "--data",
                    str(strict_file),
                    "--satisfaction-lower",
                    "95",
                    "--no-export",
                    "--quiet",
                ]
            )
        self.assertEqual(code, 0)

        result = run_analysis(strict_file, export=False, verbose=False,
                              thresholds=config.AnomalyThresholds(satisfaction_lower=95.0))
        self.assertGreater(
            len(result.anomalies.loc[result.anomalies["异常类型"] == "满意度偏低"]),
            len(self.result.anomalies.loc[self.result.anomalies["异常类型"] == "满意度偏低"]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
