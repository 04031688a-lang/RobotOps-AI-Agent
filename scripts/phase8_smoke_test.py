"""RobotOps AI —— Phase 8 离线自检脚本（数据质量检查 + 运营指标体系）。

作用：不依赖 API Key、不发起任何网络请求，直接把 Phase 8 的两块能力跑一遍并逐项校验：

    [1] 数据质量检查（演示数据）
    [2] 七类测试数据场景（正常 / 缺失 / 重复 / 非法数值 / 缺少字段 / 越界满意度 / 负数成本）
    [3] 严重问题禁止分析（缺少核心字段时不允许进入 Agent 分析）
    [4] 九项运营指标（由 Python / Pandas 计算）
    [5] 阈值判定（设备利用率、故障率、满意度、节降率、MTTR 预警）
    [6] Phase 8 产物导出（数据质量报告 JSON / 检查明细 CSV / 指标 CSV）

用法：
    python scripts/phase8_smoke_test.py
    python scripts/phase8_smoke_test.py --data data/raw/robot_operation_demo.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from robotops import config  # noqa: E402
from robotops.console_io import setup_console_encoding  # noqa: E402
from robotops.data_loader import load_operation_data  # noqa: E402
from robotops.indicators import (  # noqa: E402
    LEVEL_ANOMALY,
    LEVEL_WARNING,
    detect_indicator_anomalies,
)
from robotops.pipeline import run_analysis_with_quality  # noqa: E402
from robotops.quality import check_data_quality  # noqa: E402


def _check(name: str, ok: bool, detail: str = "") -> int:
    mark = "PASS" if ok else "FAIL"
    text = f"  [{mark}] {name}"
    if detail:
        text += f" —— {detail}"
    print(text)
    return 1 if ok else 0


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


def _frame(*overrides: dict[str, object]) -> pd.DataFrame:
    items = overrides or ({},)
    rows: list[dict[str, object]] = []
    for index, item in enumerate(items):
        row = _row()
        row[config.COL_DATE] = "2026-08-{:02d}".format(index + 1)
        row[config.COL_ROBOT_ID] = "R-{:02d}".format(index + 1)
        row.update(item)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    setup_console_encoding()
    parser = argparse.ArgumentParser(description="RobotOps AI Phase 8 自检")
    parser.add_argument("--data", default=None, help="数据文件路径（默认使用项目演示数据）")
    args = parser.parse_args(argv)

    passed = total = 0
    temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    tmp = Path(temp.name)

    print("=" * 78)
    print("RobotOps AI —— Phase 8 自检（数据质量检查 / 运营指标体系）")
    print("=" * 78)

    # -- [1] 演示数据的数据质量检查 -----------------------------------------
    print("\n[1] 数据质量检查（演示数据）")
    loaded = load_operation_data(args.data, validate=False)
    report = check_data_quality(loaded.data, frame_name=loaded.path.name)
    print("      状态：" + report.status_label)
    for line in report.summary_lines():
        print("      " + line)
    total += 1
    passed += _check(
        "演示数据质量报告生成",
        report.row_count > 0 and report.status in {
            config.QUALITY_STATUS_PASS, config.QUALITY_STATUS_WARNING, config.QUALITY_STATUS_ERROR
        },
        "记录 " + str(report.row_count) + " 条",
    )
    total += 1
    passed += _check(
        "演示数据注入的脏数据被识别",
        report.duplicate_rows > 0 or report.invalid_values > 0 or report.missing_values > 0,
        "缺失 " + str(report.missing_values) + " / 重复 " + str(report.duplicate_rows)
        + " / 非法 " + str(report.invalid_values),
    )

    # -- [2] 七类测试数据场景 ----------------------------------------------
    print("\n[2] 七类测试数据场景")
    base = _frame(*[{} for _ in range(6)])
    duplicate_frame = pd.concat([base, base.iloc[[0]]], ignore_index=True)
    scenarios: list[tuple[str, pd.DataFrame, str]] = [
        ("正常数据 -> 通过", base, config.QUALITY_STATUS_PASS),
        ("缺失数据 -> 警告", _frame({}, {config.COL_SATISFACTION: None}), config.QUALITY_STATUS_WARNING),
        ("重复数据 -> 警告", duplicate_frame, config.QUALITY_STATUS_WARNING),
        ("非法数值 -> 警告", _frame({config.COL_COST: "待确认"}, {}), config.QUALITY_STATUS_WARNING),
        ("缺少字段 -> 严重", base.drop(columns=[config.COL_FAULT_COUNT]), config.QUALITY_STATUS_ERROR),
        ("超范围满意度 -> 警告", _frame({config.COL_SATISFACTION: 150.0}, {}), config.QUALITY_STATUS_WARNING),
        ("负数成本 -> 警告", _frame({config.COL_COST: -100.0}, {}), config.QUALITY_STATUS_WARNING),
    ]
    for name, frame, expected in scenarios:
        result = check_data_quality(frame)
        total += 1
        passed += _check(name, result.status == expected, "实际 " + result.status)

    # -- [3] 严重问题禁止分析 ----------------------------------------------
    print("\n[3] 严重问题禁止分析（缺少核心字段）")
    severe_path = tmp / "missing_column.xlsx"
    base.drop(columns=[config.COL_ROBOT_ID]).to_excel(severe_path, index=False)
    severe = run_analysis_with_quality(severe_path, output_dir=tmp / "severe_out", verbose=False)
    total += 1
    passed += _check(
        "缺少核心字段时不进入分析",
        (not severe.analyzed) and severe.quality.status == config.QUALITY_STATUS_ERROR,
        severe.skipped_reason[:48],
    )
    total += 1
    unreadable = run_analysis_with_quality(tmp / "not_exists.xlsx", export=False, verbose=False)
    passed += _check(
        "文件不可读时返回报告而不抛异常",
        not unreadable.analyzed and bool(unreadable.quality.errors),
    )

    # -- [4] 九项运营指标 --------------------------------------------------
    print("\n[4] 九项运营指标（Python / Pandas 计算）")
    outcome = run_analysis_with_quality(
        args.data, output_dir=tmp / "out", export=True, verbose=False
    )
    total += 1
    passed += _check("通过质量检查后完成分析", outcome.analyzed, outcome.quality.status_label)
    indicators = outcome.indicators
    for row in indicators.as_rows() if indicators is not None else []:
        value = "-" if row["数值"] is None else row["数值"]
        print("      " + str(row["指标"]) + "：" + str(value) + " " + str(row["单位"]))
    total += 1
    passed += _check(
        "九项指标全部产出",
        indicators is not None and len(indicators.as_rows()) == 9,
    )
    total += 1
    passed += _check(
        "指标由程序计算（关键指标均非空）",
        indicators is not None
        and all(v is not None for v in indicators.as_dict().values()),
    )

    # -- [5] 阈值判定 ------------------------------------------------------
    print("\n[5] 阈值判定（集中配置在 config.IndicatorThresholds）")
    low_utilization = _frame(
        {config.COL_RUNNING_HOURS: 4.0},
        {config.COL_RUNNING_HOURS: 4.0},
    )
    low_hits = detect_indicator_anomalies(low_utilization)
    total += 1
    passed += _check(
        "设备利用率 < 80% 判定异常",
        (not low_hits.empty)
        and bool((low_hits["指标"] == config.COL_UTILIZATION_RATE).any())
        and bool((low_hits["级别"] == LEVEL_ANOMALY).any()),
    )

    bad_quality = _frame(
        {
            config.COL_FAULT_COUNT: 2,
            config.COL_SATISFACTION: 70.0,
            config.COL_SAVING_RATE: 5.0,
        }
    )
    bad_hits = detect_indicator_anomalies(bad_quality)
    metrics = set(bad_hits["指标"]) if not bad_hits.empty else set()
    total += 1
    passed += _check(
        "故障率 / 满意度 / 节降率阈值命中",
        {"故障率", "用户满意度", "节降率"}.issubset(metrics),
        "命中：" + "、".join(sorted(metrics)),
    )

    mttr_frame = _frame(
        {config.COL_RUNNING_HOURS: 5.0, config.COL_FAULT_COUNT: 1, config.COL_REPAIR_COUNT: 1},
        {config.COL_RUNNING_HOURS: 9.0, config.COL_FAULT_COUNT: 1, config.COL_REPAIR_COUNT: 1},
    )
    mttr_hits = detect_indicator_anomalies(mttr_frame)
    total += 1
    passed += _check(
        "MTTR 高于项目平均 -> 预警",
        (not mttr_hits.empty)
        and bool((mttr_hits["指标"] == config.COL_MTTR).any())
        and bool((mttr_hits["级别"] == LEVEL_WARNING).any()),
    )

    # -- [6] 产物导出 ------------------------------------------------------
    print("\n[6] Phase 8 产物导出")
    expected_files = {
        "data_quality_report.json",
        "data_quality_checks.csv",
        "robot_ops_indicators.csv",
        "robot_ops_indicator_anomalies.csv",
    }
    written = {path.name for path in outcome.phase8_files}
    total += 1
    passed += _check("Phase 8 四类产物全部导出", expected_files == written, "、".join(sorted(written)))

    payload_path = tmp / "out" / "data_quality_report.json"
    total += 1
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        ok = all(
            key in payload
            for key in ("status", "missing_values", "duplicate_rows", "invalid_values", "warnings")
        )
    except Exception as error:  # noqa: BLE001
        payload, ok = {}, False
        print("      JSON 解析失败：" + str(error))
    passed += _check("数据质量报告 JSON 结构正确", bool(ok))

    temp.cleanup()

    print("\n" + "=" * 78)
    print("自检结果：" + str(passed) + "/" + str(total) + " 项通过")
    print("=" * 78)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

