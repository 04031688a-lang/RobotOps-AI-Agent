"""数据读取模块测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from robotops import config  # noqa: E402
from robotops.data_loader import list_excel_sheets, load_operation_data  # noqa: E402
from robotops.exceptions import DataLoadError, DataValidationError  # noqa: E402
from tests._helpers import base_operation_frame, write_excel  # noqa: E402


class DataLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._temp.name)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_load_excel_success(self) -> None:
        path = write_excel(base_operation_frame(), self.tmp_dir / "demo.xlsx")

        result = load_operation_data(path)

        self.assertEqual(result.row_count, 3)
        self.assertEqual(result.sheet_name, "运营数据")
        # 读取阶段保持源文件的列顺序，只校验必需列是否齐全
        self.assertEqual(sorted(result.data.columns), sorted(config.ALL_COLUMNS))
        self.assertEqual(list_excel_sheets(path), ["运营数据"])

    def test_load_csv_with_gbk_encoding(self) -> None:
        path = self.tmp_dir / "demo_gbk.csv"
        base_operation_frame().to_csv(path, index=False, encoding="gbk")

        result = load_operation_data(path)

        self.assertEqual(result.row_count, 3)
        self.assertEqual(result.encoding, "gbk")

    def test_column_alias_mapping(self) -> None:
        frame = base_operation_frame().rename(
            columns={
                "项目名称": "项目",
                "机器人ID": "机器人编号",
                "机器人类型": "设备类型",
                "运行时长": "运行小时数",
                "故障次数": "故障数",
                "巡检次数": "巡检数",
                "维修次数": "维修数",
                "用户满意度": "满意度",
                "运营成本": "成本",
                "节降率": "节能率",
            }
        )
        path = write_excel(frame, self.tmp_dir / "alias.xlsx")

        result = load_operation_data(path)

        self.assertIn(config.COL_PROJECT, result.data.columns)
        self.assertIn(config.COL_SATISFACTION, result.data.columns)
        self.assertIn("项目", result.renamed_columns)

    def test_missing_required_column_raises(self) -> None:
        frame = base_operation_frame().drop(columns=[config.COL_SATISFACTION])
        path = write_excel(frame, self.tmp_dir / "missing.xlsx")

        with self.assertRaises(DataValidationError) as ctx:
            load_operation_data(path)

        self.assertIn(config.COL_SATISFACTION, str(ctx.exception))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(DataLoadError):
            load_operation_data(self.tmp_dir / "not_exists.xlsx")

    def test_unsupported_suffix_raises(self) -> None:
        path = self.tmp_dir / "demo.txt"
        path.write_text("hello", encoding="utf-8")

        with self.assertRaises(DataLoadError):
            load_operation_data(path)

    def test_optional_column_registered_as_missing(self) -> None:
        frame = base_operation_frame().drop(columns=[config.COL_PLANNED_HOURS])
        path = write_excel(frame, self.tmp_dir / "no_planned.xlsx")

        result = load_operation_data(path)

        self.assertEqual(result.missing_optional_columns, [config.COL_PLANNED_HOURS])


if __name__ == "__main__":
    unittest.main(verbosity=2)
