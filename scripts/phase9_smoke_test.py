"""RobotOps AI —— Phase 9 离线自检脚本（运营趋势分析 + 项目横向对比）。

不依赖 API Key、不发起任何网络请求，直接跑通 Phase 9 的两块能力并逐项校验：

    [1] 趋势生成（日 / 周 / 月 + 自动粒度）
    [2] 趋势变化（环比、方向、连续上升/下降）
    [3] 同期变化（数据支持时计算）
    [4] 边界情况（数据不足、日期异常、周期不完整）
    [5] 项目横向对比（7 项指标、排名、与整体平均差异）
    [6] Agent 接入（趋势与对比作为结构化数据进入提示词）

用法：
    python scripts/phase9_smoke_test.py
    python scripts/phase9_smoke_test.py --data data/raw/robot_operation_demo.xlsx
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from app.graph.prompts import build_diagnosis_prompt  # noqa: E402
from robotops import config  # noqa: E402
from robotops.comparison import COMPARISON_METRICS, compare_projects  # noqa: E402
from robotops.console_io import setup_console_encoding  # noqa: E402
from robotops.data_cleaner import clean_operation_data  # noqa: E402
from robotops.data_loader import load_operation_data  # noqa: E402
from robotops.trends import TREND_METRICS, build_trend_series  # noqa: E402


def _check(name: str, ok: bool, detail: str = "") -> int:
    mark = "PASS" if ok else "FAIL"
    text = "  [" + mark + "] " + name
    if detail:
        text += " —— " + detail
    print(text)
    return 1 if ok else 0


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        config.COL_DATE: "2026-08-03",
        config.COL_PROJECT: "项目甲",
        config.COL_ROBOT_ID: "R-01",
        config.COL_ROBOT_TYPE: "清洁机器人",
        config.COL_RUNNING_HOURS: 8.0,
        config.COL_PLANNED_HOURS: 10.0,
        config.COL_FAULT_COUNT: 0.4,
        config.COL_INSPECTION_COUNT: 10,
        config.COL_REPAIR_COUNT: 1,
        config.COL_SATISFACTION: 92.0,
        config.COL_COST: 1000.0,
        config.COL_SAVING_RATE: 15.0,
    }
    row.update(overrides)
    return row


def daily_frame(
    start: str, days: int, *, projects: tuple[str, ...] = ("项目甲",), **base: object
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    rows: list[dict[str, object]] = []
    for index in range(days):
        day = (start_ts + pd.Timedelta(days=index)).strftime("%Y-%m-%d")
        for project in projects:
            row = _row(project=project, **base)
            row[config.COL_PROJECT] = project
            row[config.COL_ROBOT_ID] = project + "-01"
            row[config.COL_DATE] = day
            row[config.COL_FAULT_COUNT] = 0.4 + (index // 7) * 0.2
            row[config.COL_SATISFACTION] = 92.0 - (index // 7) * 2.0
            row[config.COL_COST] = 1000.0 + (index // 7) * 300.0
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    setup_console_encoding()
    parser = argparse.ArgumentParser(description="RobotOps AI Phase 9 自检")
    parser.add_argument("--data", default=None, help="数据文件路径（默认使用项目演示数据）")
    args = parser.parse_args(argv)

    passed = total = 0
    temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    print("=" * 78)
    print("RobotOps AI —— Phase 9 自检（运营趋势分析 / 项目横向对比）")
    print("=" * 78)

    # -- [1] 趋势生成 ------------------------------------------------------
    print("\n[1] 趋势生成（日 / 周 / 月）")
    sample = daily_frame("2026-08-03", 21, projects=("项目甲", "项目乙"))
    for granularity, expected_periods in (
        (config.TREND_GRANULARITY_DAY, 21),
        (config.TREND_GRANULARITY_WEEK, 3),
        (config.TREND_GRANULARITY_MONTH, 1),
    ):
        analysis = build_trend_series(sample, granularity=granularity)
        total += 1
        passed += _check(
            "粒度「" + config.TREND_GRANULARITY_LABELS[granularity] + "」生成趋势",
            analysis.period_count == expected_periods,
            "周期数 " + str(analysis.period_count),
        )

    auto = build_trend_series(sample)
    total += 1
    passed += _check(
        "自动粒度选择（21 天 -> 周）",
        auto.granularity == config.TREND_GRANULARITY_WEEK and auto.auto_detected,
        "实际 " + auto.granularity_label,
    )

    total += 1
    passed += _check("七项趋势指标全部产出", len(auto.changes) == len(TREND_METRICS) == 7)

    # -- [2] 趋势变化 ------------------------------------------------------
    print("\n[2] 趋势变化（环比 / 方向 / 连续升降）")
    weekly = build_trend_series(sample, granularity=config.TREND_GRANULARITY_WEEK)
    fault = weekly.change_of("fault_rate")
    total += 1
    passed += _check(
        "故障率环比与方向",
        fault is not None and fault.direction == config.TREND_UP and fault.change_ratio is not None,
        "环比 " + str(fault.change_percent) + "%",
    )
    total += 1
    passed += _check(
        "连续上升识别（3 期）",
        bool(fault.is_consecutive_up) and fault.consecutive_periods == 3,
        "连续 " + str(fault.consecutive_periods) + " 期",
    )
    satisfaction = weekly.change_of("satisfaction")
    total += 1
    passed += _check(
        "连续下降识别（满意度）",
        bool(satisfaction.is_consecutive_down),
        "连续 " + str(satisfaction.consecutive_periods) + " 期",
    )

    # -- [3] 同期变化 ------------------------------------------------------
    print("\n[3] 同期变化")
    six_weeks = build_trend_series(
        daily_frame("2026-08-03", 42), granularity=config.TREND_GRANULARITY_WEEK
    )
    change = six_weeks.change_of("fault_rate")
    total += 1
    passed += _check(
        "周粒度同期（上月同期）",
        change.same_period_value is not None,
        change.same_period_label or change.same_period_note,
    )
    monthly = build_trend_series(
        daily_frame("2025-01-01", 400), granularity=config.TREND_GRANULARITY_MONTH
    )
    monthly_change = monthly.change_of("total_cost")
    total += 1
    passed += _check(
        "月粒度同期（去年同期）",
        monthly_change.same_period_value is not None,
        monthly_change.same_period_label or monthly_change.same_period_note,
    )

    # -- [4] 边界情况 ------------------------------------------------------
    print("\n[4] 边界情况（数据不足 / 日期异常 / 周期不完整）")
    single = build_trend_series(daily_frame("2026-08-03", 7), granularity=config.TREND_GRANULARITY_WEEK)
    total += 1
    passed += _check(
        "数据不足时给出明确状态",
        bool(single.warnings) and single.change_of("fault_rate").status == "数据不足",
    )
    broken = daily_frame("2026-08-03", 14)
    broken.loc[0, config.COL_DATE] = "不是日期"
    broken_analysis = build_trend_series(broken)
    total += 1
    passed += _check(
        "日期异常被排除并提示",
        broken_analysis.invalid_date_rows == 1
        and any("日期异常" in item for item in broken_analysis.warnings),
    )
    partial = build_trend_series(
        daily_frame("2026-08-05", 13), granularity=config.TREND_GRANULARITY_WEEK
    )
    total += 1
    passed += _check(
        "不完整周期被识别",
        bool(partial.partial_periods)
        and any("周期不完整" in item for item in partial.warnings),
    )

    # -- [5] 项目横向对比 --------------------------------------------------
    print("\n[5] 项目横向对比（7 项指标）")
    comparison = compare_projects(sample)
    total += 1
    passed += _check(
        "多项目对比表生成",
        comparison.project_count == 2 and len(COMPARISON_METRICS) == 7,
        "项目数 " + str(comparison.project_count),
    )
    ranking = comparison.ranking("fault_rate")
    total += 1
    passed += _check(
        "排名按指标方向计算（故障率越低越好）",
        not ranking.empty and list(ranking["排名"]) == [1, 2],
    )
    total += 1
    passed += _check("对比要点生成", bool(comparison.highlight_lines()))

    # -- [6] Agent 接入 ----------------------------------------------------
    print("\n[6] Agent 接入（趋势与对比作为结构化数据）")
    prompt = build_diagnosis_prompt(
        raw_data_summary={},
        metrics={},
        abnormal_projects=[],
        anomaly_summary=[],
        abnormal_robots=[],
        thresholds={},
        trend_analysis=weekly.summary_dict(),
        project_comparison=comparison.summary_dict(),
    )
    total += 1
    passed += _check("诊断提示词包含运营趋势段落", "【运营趋势" in prompt)
    total += 1
    passed += _check("诊断提示词包含项目对比段落", "【项目横向对比" in prompt)
    total += 1
    passed += _check(
        "趋势数据带有禁止外推的约束",
        "禁止修改或外推" in prompt,
    )

    # -- 真实数据（演示数据）-----------------------------------------------
    print("\n[7] 真实数据（演示数据）趋势与对比")
    loaded = load_operation_data(args.data, validate=False)
    cleaned = clean_operation_data(loaded.data).data
    real_trend = build_trend_series(cleaned)
    real_comparison = compare_projects(cleaned)
    print("      " + real_trend.summary_text().replace("\n", "\n      "))
    total += 1
    passed += _check(
        "演示数据趋势生成",
        real_trend.available and real_trend.period_count >= 2,
        "粒度 " + real_trend.granularity_label + " / 周期 " + str(real_trend.period_count),
    )
    total += 1
    passed += _check(
        "演示数据项目对比生成",
        real_comparison.project_count >= 2,
        "项目数 " + str(real_comparison.project_count),
    )

    temp.cleanup()

    print("\n" + "=" * 78)
    print("自检结果：" + str(passed) + "/" + str(total) + " 项通过")
    print("=" * 78)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

