"""Phase 8 数据质量检查测试。

覆盖需求要求的 7 类测试数据：
1. 正常数据  2. 缺失数据  3. 重复数据  4. 非法数值
5. 缺少字段  6. 超范围满意度  7. 负数成本

外加健壮性测试：任何脏数据（包括真实 xlsx / csv 文件）都不能让程序崩溃。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from robotops import config  # noqa: E402
from robotops.data_loader import load_operation_data  # noqa: E402
from robotops.pipeline import run_analysis_with_quality  # noqa: E402
from robotops.quality import check_data_quality, quality_rules_frame  # noqa: E402
from scripts.generate_demo_data import generate_demo_file  # noqa: E402


def _row(**overrides: object) -> dict[str, object]:
    """构造一条合法记录，可用关键字覆盖任意字段。"""

    row: dict[str, object] = {
        config.COL_DATE: "2026-08-01",
        config.COL_PROJECT: "项目甲",
        config.COL_ROBOT_ID: "R-01",
        config.COL_ROBOT_TYPE: "清洁机器人",
        config.COL_RUNNING_HOURS: 8.0,
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


def build_frame(*overrides: dict[str, object]) -> pd.DataFrame:
    """按记录构造 DataFrame（混合类型会得到 object 列，贴近真实上传文件）。

    默认给每条记录不同的「日期 + 机器人ID」，避免无意中制造重复记录。
    """

    items = overrides or ({},)
    rows: list[dict[str, object]] = []
    for index, item in enumerate(items):
        row = _row()
        row[config.COL_DATE] = "2026-08-{:02d}".format(index + 1)
        row[config.COL_ROBOT_ID] = "R-{:02d}".format(index + 1)
        row.update(item)
        rows.append(row)
    return pd.DataFrame(rows)


def normal_frame(rows: int = 3) -> pd.DataFrame:
    """正常数据：无缺失、无重复、无越界。"""

    return build_frame(
        *[
            {
                config.COL_ROBOT_ID: "R-0" + str(index + 1),
                config.COL_DATE: "2026-08-0" + str(index + 1),
            }
            for index in range(rows)
        ]
    )


class DataQualityCheckTests(unittest.TestCase):
    # -- 1. 正常数据 -------------------------------------------------------
    def test_normal_data_passes(self) -> None:
        report = check_data_quality(normal_frame())

        self.assertEqual(report.status, config.QUALITY_STATUS_PASS)
        self.assertTrue(report.can_analyze)
        self.assertEqual(report.missing_values, 0)
        self.assertEqual(report.duplicate_rows, 0)
        self.assertEqual(report.invalid_values, 0)
        self.assertEqual(report.warnings, [])
        self.assertEqual(report.errors, [])
        self.assertTrue(any(line.startswith("✓ 字段完整") for line in report.summary_lines()))

    # -- 2. 缺失数据 -------------------------------------------------------
    def test_missing_values_detected(self) -> None:
        frame = build_frame(
            {config.COL_SATISFACTION: None},
            {config.COL_RUNNING_HOURS: None},
            {},
        )
        report = check_data_quality(frame)

        self.assertEqual(report.status, config.QUALITY_STATUS_WARNING)
        self.assertTrue(report.can_analyze)
        self.assertEqual(report.missing_values, 2)
        self.assertTrue(any("缺失" in item for item in report.warnings))
        self.assertTrue(any("发现 2 条缺失数据" in line for line in report.summary_lines()))

    # -- 3. 重复数据 -------------------------------------------------------
    def test_duplicate_rows_detected(self) -> None:
        base = normal_frame(rows=6)
        frame = pd.concat([base, base.iloc[[0]]], ignore_index=True)  # 1/7 条完全重复
        report = check_data_quality(frame)

        self.assertEqual(report.duplicate_rows, 1)
        self.assertEqual(report.status, config.QUALITY_STATUS_WARNING)
        self.assertTrue(any("重复记录" in item for item in report.warnings))

        # 同一机器人同一天重复上报（内容不同）同样要能识别
        dup_row = _row(
            **{
                config.COL_DATE: base.loc[0, config.COL_DATE],
                config.COL_ROBOT_ID: base.loc[0, config.COL_ROBOT_ID],
                config.COL_COST: 2000.0,
            }
        )
        key_dup = pd.concat([base, pd.DataFrame([dup_row])], ignore_index=True)
        key_report = check_data_quality(key_dup)
        self.assertEqual(key_report.duplicate_rows, 1)
        self.assertEqual(key_report.status, config.QUALITY_STATUS_WARNING)

    def test_duplicate_ratio_over_threshold_is_severe(self) -> None:
        base = normal_frame(rows=6)
        frame = pd.concat([base, base.iloc[[0, 1, 2]]], ignore_index=True)  # 3/9 = 33%
        report = check_data_quality(frame)

        self.assertEqual(report.status, config.QUALITY_STATUS_ERROR)
        self.assertFalse(report.can_analyze)

    # -- 4. 非法数值 -------------------------------------------------------
    def test_invalid_numeric_values_detected(self) -> None:
        frame = build_frame(
            {config.COL_RUNNING_HOURS: "八小时"},
            {config.COL_COST: "待确认"},
            {},
        )
        report = check_data_quality(frame)

        self.assertEqual(report.status, config.QUALITY_STATUS_WARNING)
        self.assertEqual(report.invalid_values, 2)
        numeric_check = next(item for item in report.checks if item.key == "numeric_parseable")
        self.assertEqual(numeric_check.affected_rows, 2)
        self.assertTrue(any("无法解析" in item for item in report.warnings))

    def test_tolerated_numeric_text_is_not_invalid(self) -> None:
        frame = build_frame(
            {config.COL_COST: "1,000.50", config.COL_SAVING_RATE: "18%"},
            {config.COL_COST: "2,000.00", config.COL_SAVING_RATE: "12%"},
        )
        report = check_data_quality(frame)

        self.assertEqual(report.invalid_values, 0)
        self.assertEqual(report.status, config.QUALITY_STATUS_PASS)

    # -- 5. 缺少字段 -------------------------------------------------------
    def test_missing_required_column_is_severe(self) -> None:
        frame = normal_frame().drop(columns=[config.COL_COST])
        report = check_data_quality(frame)

        self.assertEqual(report.status, config.QUALITY_STATUS_ERROR)
        self.assertFalse(report.can_analyze)
        self.assertIn(config.COL_COST, report.missing_required_columns)
        self.assertTrue(report.errors)
        self.assertTrue(any("禁止" in line for line in report.summary_lines()))

    # -- 6. 超范围满意度 ---------------------------------------------------
    def test_out_of_range_satisfaction_detected(self) -> None:
        frame = build_frame({config.COL_SATISFACTION: 120.0}, {config.COL_SATISFACTION: -5.0}, {})
        report = check_data_quality(frame)

        self.assertEqual(report.status, config.QUALITY_STATUS_WARNING)
        self.assertEqual(report.invalid_values, 2)
        satisfaction_check = next(item for item in report.checks if item.key == "satisfaction_range")
        self.assertEqual(satisfaction_check.affected_rows, 2)
        self.assertEqual(satisfaction_check.level, config.QUALITY_STATUS_WARNING)

    # -- 7. 负数成本 -------------------------------------------------------
    def test_negative_cost_detected(self) -> None:
        frame = build_frame({config.COL_COST: -100.0}, {}, {})
        report = check_data_quality(frame)

        self.assertEqual(report.status, config.QUALITY_STATUS_WARNING)
        self.assertEqual(report.invalid_values, 1)
        cost_check = next(item for item in report.checks if item.key == "cost_negative")
        self.assertEqual(cost_check.affected_rows, 1)
        self.assertTrue(any("运营成本" in item for item in report.warnings))

    # -- 其它字段的负数与越界 ---------------------------------------------
    def test_other_negative_and_range_checks(self) -> None:
        frame = build_frame(
            {
                config.COL_RUNNING_HOURS: -1.0,
                config.COL_FAULT_COUNT: -2,
                config.COL_REPAIR_COUNT: -1,
                config.COL_SAVING_RATE: 500.0,
            },
            {},
            {},
        )
        report = check_data_quality(frame)

        self.assertEqual(report.status, config.QUALITY_STATUS_WARNING)
        self.assertEqual(report.invalid_values, 4)
        keys = {item.key for item in report.problem_checks}
        self.assertLessEqual(
            {
                "running_hours_negative",
                "fault_count_negative",
                "repair_count_negative",
                "saving_rate_range",
            },
            keys,
        )

    def test_invalid_date_and_blank_key_fields(self) -> None:
        frame = build_frame(
            {config.COL_DATE: "不是日期"},
            {config.COL_ROBOT_ID: "   "},
            {config.COL_PROJECT: None},
        )
        report = check_data_quality(frame)

        self.assertEqual(report.status, config.QUALITY_STATUS_WARNING)
        date_check = next(item for item in report.checks if item.key == "date_valid")
        self.assertEqual(date_check.affected_rows, 1)
        robot_check = next(item for item in report.checks if item.key == "robot_id_not_blank")
        self.assertEqual(robot_check.affected_rows, 1)
        project_check = next(item for item in report.checks if item.key == "project_not_blank")
        self.assertEqual(project_check.affected_rows, 1)

    def test_key_field_fully_blank_is_severe(self) -> None:
        frame = normal_frame()
        frame[config.COL_PROJECT] = None
        report = check_data_quality(frame)

        self.assertEqual(report.status, config.QUALITY_STATUS_ERROR)
        self.assertFalse(report.can_analyze)

    def test_massive_missing_is_severe(self) -> None:
        frame = build_frame(
            {},
            {config.COL_COST: None},
            {config.COL_COST: None},
            {config.COL_COST: None},
            {config.COL_COST: None},
            {config.COL_COST: None},
        )
        report = check_data_quality(frame)

        self.assertEqual(report.status, config.QUALITY_STATUS_ERROR)
        self.assertEqual(report.missing_values, 5)
        self.assertFalse(report.can_analyze)

    def test_small_sample_missing_is_warning_not_error(self) -> None:
        # 2 行里缺 1 行 = 50%，比例很高但样本太小，按警告处理（不禁止分析）
        frame = build_frame({}, {config.COL_COST: None})
        report = check_data_quality(frame)

        self.assertEqual(report.status, config.QUALITY_STATUS_WARNING)
        self.assertTrue(report.can_analyze)


class DataQualityRobustnessTests(unittest.TestCase):
    """异常输入不能让程序崩溃。"""

    def test_none_and_empty_frame(self) -> None:
        for value in (None, pd.DataFrame(), pd.DataFrame(columns=list(config.ALL_COLUMNS))):
            report = check_data_quality(value)
            self.assertEqual(report.status, config.QUALITY_STATUS_ERROR)
            self.assertFalse(report.can_analyze)
            self.assertTrue(report.as_dict()["errors"])

    def test_non_dataframe_input(self) -> None:
        report = check_data_quality(["不是 DataFrame"])  # type: ignore[arg-type]
        self.assertEqual(report.status, config.QUALITY_STATUS_ERROR)

    def test_all_text_garbage_frame(self) -> None:
        frame = pd.DataFrame([{"任意列": "x", "另一列": "y"} for _ in range(3)])
        report = check_data_quality(frame)

        self.assertEqual(report.status, config.QUALITY_STATUS_ERROR)
        self.assertEqual(len(report.missing_required_columns), len(config.REQUIRED_COLUMNS))

    def test_unparseable_objects_do_not_raise(self) -> None:
        frame = build_frame(
            {config.COL_COST: {"异常": "对象"}},
            {config.COL_DATE: object()},
            {},
        )
        report = check_data_quality(frame)

        self.assertIn(report.status, {config.QUALITY_STATUS_WARNING, config.QUALITY_STATUS_ERROR})
        self.assertIsInstance(report.as_dict(), dict)

    def test_report_frames_and_rules(self) -> None:
        report = check_data_quality(normal_frame())
        detail = report.to_frame()

        self.assertFalse(detail.empty)
        self.assertIn("检查项", detail.columns)
        self.assertTrue(report.problem_frame().empty)
        self.assertFalse(quality_rules_frame().empty)

    def test_as_dict_structure(self) -> None:
        payload = check_data_quality(build_frame({config.COL_SATISFACTION: None}, {})).as_dict()

        for key in ("status", "missing_values", "duplicate_rows", "invalid_values", "warnings"):
            self.assertIn(key, payload)
        self.assertIsInstance(payload["warnings"], list)
        self.assertIsInstance(payload["checks"], list)


class DirtyFileDoesNotCrashTests(unittest.TestCase):
    """真实文件（xlsx / csv）层面的兜底：脏文件不能让程序崩溃。"""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._temp.name)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_dirty_excel_and_csv_return_reports(self) -> None:
        rows = [
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {config.COL_SATISFACTION: 150.0},
            {config.COL_COST: -100.0},
        ]
        frame = build_frame(*rows)
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)  # 1/12 条重复

        xlsx = self.tmp / "dirty.xlsx"
        csv = self.tmp / "dirty.csv"
        frame.to_excel(xlsx, index=False)
        frame.to_csv(csv, index=False, encoding="utf-8-sig")

        for path in (xlsx, csv):
            loaded = load_operation_data(path, validate=False)
            report = check_data_quality(loaded.data, frame_name=path.name)
            self.assertEqual(report.status, config.QUALITY_STATUS_WARNING)
            self.assertGreater(report.duplicate_rows, 0)
            self.assertGreater(report.invalid_values, 0)

    def test_excel_without_required_column_is_reported_not_raised(self) -> None:
        frame = normal_frame().drop(columns=[config.COL_ROBOT_ID])
        path = self.tmp / "missing_column.xlsx"
        frame.to_excel(path, index=False)

        loaded = load_operation_data(path, validate=False)
        report = check_data_quality(loaded.data, frame_name=path.name)

        self.assertEqual(report.status, config.QUALITY_STATUS_ERROR)
        self.assertIn(config.COL_ROBOT_ID, report.missing_required_columns)


class QualityGatedPipelineTests(unittest.TestCase):
    """质量检查 -> 通过后才分析：整条 Phase 8 流程的端到端测试。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._temp.name)
        cls.demo_path = cls.tmp / "robot_operation_demo.xlsx"
        generate_demo_file(cls.demo_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_demo_data_is_analyzed_and_exports_phase8_outputs(self) -> None:
        output_dir = self.tmp / "out_demo"
        outcome = run_analysis_with_quality(
            self.demo_path, output_dir=output_dir, verbose=False
        )

        self.assertTrue(outcome.analyzed)
        self.assertIsNotNone(outcome.result)
        self.assertTrue(outcome.quality.can_analyze)
        # 演示数据里注入了脏数据，因此是警告而不是通过
        self.assertEqual(outcome.quality.status, config.QUALITY_STATUS_WARNING)

        self.assertIsNotNone(outcome.indicators)
        self.assertEqual(len(outcome.indicators.as_rows()), 9)
        self.assertFalse(outcome.project_indicators.empty)

        names = {path.name for path in outcome.phase8_files}
        self.assertEqual(
            names,
            {
                "data_quality_report.json",
                "data_quality_checks.csv",
                "robot_ops_indicators.csv",
                "robot_ops_indicator_anomalies.csv",
            },
        )
        for path in outcome.phase8_files:
            self.assertTrue(path.exists(), str(path) + " 未生成")

        payload = json.loads(
            (output_dir / "data_quality_report.json").read_text(encoding="utf-8")
        )
        for key in ("status", "missing_values", "duplicate_rows", "invalid_values", "warnings"):
            self.assertIn(key, payload)
        self.assertGreater(payload["duplicate_rows"], 0)

        # Phase 1 的既有产物仍然照常导出，并额外包含 Phase 8 产物
        self.assertGreaterEqual(len(outcome.result.exported_files), 11)

    def test_missing_required_column_blocks_analysis_without_crash(self) -> None:
        frame = normal_frame().drop(columns=[config.COL_COST])
        path = self.tmp / "missing_cost.xlsx"
        frame.to_excel(path, index=False)

        outcome = run_analysis_with_quality(path, export=False, verbose=False)

        self.assertFalse(outcome.analyzed)
        self.assertIsNone(outcome.result)
        self.assertEqual(outcome.quality.status, config.QUALITY_STATUS_ERROR)
        self.assertIn(config.COL_COST, outcome.quality.missing_required_columns)
        self.assertTrue(outcome.skipped_reason)

    def test_non_severe_problems_still_analyze(self) -> None:
        frame = build_frame(
            *[{} for _ in range(8)] + [{config.COL_COST: -100.0}, {config.COL_SATISFACTION: 150.0}]
        )
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)  # 1/11 条重复
        path = self.tmp / "dirty_but_analyzable.xlsx"
        frame.to_excel(path, index=False)

        outcome = run_analysis_with_quality(path, export=False, verbose=False)

        self.assertEqual(outcome.quality.status, config.QUALITY_STATUS_WARNING)
        self.assertTrue(outcome.analyzed)
        self.assertIsNotNone(outcome.indicators)
        self.assertGreater(outcome.quality.invalid_values, 0)

    def test_unreadable_file_returns_report_instead_of_raising(self) -> None:
        outcome = run_analysis_with_quality(
            self.tmp / "not_exists.xlsx", export=False, verbose=False
        )

        self.assertFalse(outcome.analyzed)
        self.assertEqual(outcome.quality.status, config.QUALITY_STATUS_ERROR)
        self.assertTrue(outcome.quality.errors)
        self.assertIn("不可读", outcome.skipped_reason)
        self.assertIn("quality", outcome.as_dict())

    def test_outcome_as_dict_is_serializable(self) -> None:
        outcome = run_analysis_with_quality(self.demo_path, export=False, verbose=False)
        payload = outcome.as_dict()

        self.assertTrue(payload["analyzed"])
        self.assertIn("indicators", payload)
        json.dumps(payload, ensure_ascii=False)  # 必须可 JSON 序列化


if __name__ == "__main__":
    unittest.main(verbosity=2)

