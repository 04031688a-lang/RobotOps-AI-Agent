"""RobotOps AI —— Phase 9 前端展示辅助（纯函数，不依赖 Streamlit）。

趋势与项目对比的数值全部来自 robotops.trends / robotops.comparison，
这里只负责整理成图表与表格需要的结构，方便界面直接渲染。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from robotops import config as project_config
from robotops.comparison import COMPARISON_METRICS, compare_projects
from robotops.trends import TREND_METRICS, build_trend_series

#: 趋势粒度选项（界面展示文本 -> 传给后端的取值）
GRANULARITY_OPTIONS: tuple[tuple[str, str | None], ...] = (
    ("自动", None),
    ("日", project_config.TREND_GRANULARITY_DAY),
    ("周", project_config.TREND_GRANULARITY_WEEK),
    ("月", project_config.TREND_GRANULARITY_MONTH),
)

#: 需求的四张图中前三张对应的指标
REQUIRED_TREND_METRICS: tuple[str, ...] = ("fault_rate", "satisfaction", "total_cost")

#: 项目对比图默认展示的指标
DEFAULT_COMPARISON_METRICS: tuple[str, ...] = ("uptime_rate", "utilization_rate", "fault_rate")


def granularity_choice(label: str) -> str | None:
    """把界面选项文本转换成后端粒度参数。"""

    for text, value in GRANULARITY_OPTIONS:
        if text == label:
            return value
    return None


def trend_panel_data(
    cleaned_data: pd.DataFrame | None,
    *,
    granularity: str | None = None,
    settings: project_config.TrendSettings | None = None,
) -> dict[str, Any]:
    """把趋势分析整理成界面数据结构。"""

    limits = settings or project_config.DEFAULT_TREND_SETTINGS
    analysis = build_trend_series(cleaned_data, granularity=granularity, settings=limits)

    charts: dict[str, pd.DataFrame] = {}
    for item in TREND_METRICS:
        charts[item.key] = analysis.chart_frame(item.key)

    options = [
        {"key": item.key, "label": item.name, "unit": item.unit} for item in TREND_METRICS
    ]

    return {
        "available": analysis.available,
        "granularity": analysis.granularity,
        "granularity_label": analysis.granularity_label,
        "auto_detected": analysis.auto_detected,
        "period_count": analysis.period_count,
        "window_text": analysis.window_text,
        "granularity_options": [label for label, _ in GRANULARITY_OPTIONS],
        "current_option": _option_label(analysis.granularity),
        "change_frame": analysis.to_frame(),
        "series_frame": analysis.series_frame(),
        "charts": charts,
        "metric_options": options,
        "highlights": analysis.highlight_lines(),
        "warnings": list(analysis.warnings),
        "partial_periods": list(analysis.partial_periods),
        "headline": _trend_headline(analysis),
    }


def comparison_panel_data(
    cleaned_data: pd.DataFrame | None,
    *,
    metric_keys: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """把项目横向对比整理成界面数据结构。"""

    comparison = compare_projects(cleaned_data)
    keys = list(metric_keys) if metric_keys else list(DEFAULT_COMPARISON_METRICS)
    chart = comparison.chart_frame(keys)

    return {
        "available": comparison.available,
        "project_count": comparison.project_count,
        "frame": comparison.to_frame(),
        "chart": chart,
        "metric_options": [
            {"key": item.key, "label": item.name, "unit": item.unit}
            for item in COMPARISON_METRICS
        ],
        "selected_metric_keys": keys,
        "highlights": comparison.highlight_lines(),
        "warnings": list(comparison.warnings),
        "headline": (
            "共 " + str(comparison.project_count) + " 个项目参与对比"
            if comparison.available
            else "暂无可用于对比的项目"
        ),
    }


def ranking_table(cleaned_data: pd.DataFrame | None, metric_key: str) -> pd.DataFrame:
    """单个指标的排名表（界面下拉选择指标时使用）。"""

    comparison = compare_projects(cleaned_data)
    return comparison.ranking(metric_key)


def _option_label(granularity: str) -> str:
    for text, value in GRANULARITY_OPTIONS:
        if value == granularity:
            return text
    return "自动"


def _trend_headline(analysis: Any) -> str:
    if not analysis.available:
        return "暂无可用数据生成趋势"
    if analysis.warnings:
        return (
            "按「" + analysis.granularity_label + "」聚合，共 "
            + str(analysis.period_count) + " 个周期（部分周期不完整，详见提示）"
        )
    return (
        "按「" + analysis.granularity_label + "」聚合，共 "
        + str(analysis.period_count) + " 个周期"
    )

