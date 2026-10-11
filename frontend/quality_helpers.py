"""RobotOps AI —— Phase 8 前端展示辅助函数（纯函数，不依赖 Streamlit）。

前端只负责展示与调用：数据质量检查、运营指标与阈值判定全部复用
robotops.quality / robotops.indicators 的实现，这里只把结果整理成界面需要的结构。

因此本模块在没有 Streamlit 的环境下也能被单元测试直接覆盖。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from robotops import config as project_config
from robotops.indicators import (
    INDICATOR_COLUMNS,
    LEVEL_ANOMALY,
    LEVEL_WARNING,
    calculate_indicators,
    detect_indicator_anomalies,
    indicator_thresholds_frame,
    project_level_indicators,
    summarize_indicator_anomalies,
)
from robotops.quality import DataQualityReport, check_data_quality

#: 质量状态 -> Streamlit 提示级别
QUALITY_BLOCK_LEVEL: dict[str, str] = {
    project_config.QUALITY_STATUS_PASS: "success",
    project_config.QUALITY_STATUS_WARNING: "warning",
    project_config.QUALITY_STATUS_ERROR: "error",
}

#: 质量状态 -> 图标
QUALITY_STATUS_ICON: dict[str, str] = {
    project_config.QUALITY_STATUS_PASS: "✓",
    project_config.QUALITY_STATUS_WARNING: "⚠",
    project_config.QUALITY_STATUS_ERROR: "✗",
}

#: 指标 -> 趋势图列（与 Phase 1 清洗后数据列对应）
INDICATOR_TREND_COLUMNS: dict[str, str] = {
    project_config.COL_UTILIZATION_RATE: project_config.COL_UPTIME_RATE,
    "故障率": project_config.COL_FAULT_RATE,
    "用户满意度": project_config.COL_SATISFACTION,
    "节降率": project_config.COL_SAVING_RATE,
}

_EMPTY_CHECKS_COLUMNS = ["检查项", "结果", "问题行数", "说明", "判定规则", "建议动作"]


# ---------------------------------------------------------------------------
# 数据质量检查
# ---------------------------------------------------------------------------
def check_quality(
    frame: pd.DataFrame | None,
    *,
    frame_name: str = "",
    thresholds: project_config.QualityThresholds | None = None,
) -> DataQualityReport:
    """对上传数据做数据质量检查（薄封装，便于前端统一调用）。"""

    return check_data_quality(frame, thresholds=thresholds, frame_name=frame_name)


def quality_panel_data(report: DataQualityReport | None) -> dict[str, Any]:
    """把数据质量报告整理成界面所需的结构。"""

    if report is None:
        return {
            "available": False,
            "status": "",
            "label": "尚未检查",
            "icon": "•",
            "block_level": "info",
            "can_analyze": True,
            "lines": ["请先上传数据文件后再进行质量检查"],
            "checks": pd.DataFrame(columns=_EMPTY_CHECKS_COLUMNS),
            "problems": pd.DataFrame(columns=_EMPTY_CHECKS_COLUMNS),
            "counts": {"missing_values": 0, "duplicate_rows": 0, "invalid_values": 0},
            "headline": "尚未执行数据质量检查",
            "block_message": "",
            "status_line": "",
        }

    counts = {
        "missing_values": int(report.missing_values),
        "duplicate_rows": int(report.duplicate_rows),
        "invalid_values": int(report.invalid_values),
    }
    return {
        "available": True,
        "status": report.status,
        "label": report.status_label,
        "icon": QUALITY_STATUS_ICON.get(report.status, "•"),
        "block_level": QUALITY_BLOCK_LEVEL.get(report.status, "info"),
        "can_analyze": report.can_analyze,
        "lines": report.summary_lines(),
        "checks": report.to_frame(),
        "problems": report.problem_frame(),
        "counts": counts,
        "headline": _quality_headline(report, counts),
        "block_message": _quality_block_message(report),
        "status_line": (
            "状态：" + report.status_label
            + " · 缺失值 " + str(counts["missing_values"])
            + " 条 · 重复记录 " + str(counts["duplicate_rows"])
            + " 条 · 非法/越界值 " + str(counts["invalid_values"]) + " 条"
        ),
    }


def _quality_headline(report: DataQualityReport, counts: dict[str, int]) -> str:
    if report.status == project_config.QUALITY_STATUS_PASS:
        return "数据质量检查通过，可以进入 Agent 分析。"
    if report.status == project_config.QUALITY_STATUS_ERROR:
        return "数据存在严重问题，已禁止进入 Agent 分析。"
    problems: list[str] = []
    if counts["missing_values"]:
        problems.append("缺失值 " + str(counts["missing_values"]) + " 条")
    if counts["duplicate_rows"]:
        problems.append("重复记录 " + str(counts["duplicate_rows"]) + " 条")
    if counts["invalid_values"]:
        problems.append("非法/越界值 " + str(counts["invalid_values"]) + " 条")
    return "发现数据质量问题：" + "、".join(problems) + "。可以继续分析，建议先修复。"


def _quality_block_message(report: DataQualityReport) -> str:
    if report.can_analyze:
        return ""
    details = report.errors or ["数据质量检查未通过"]
    lines = ["**已禁止进入 Agent 分析**，原因如下："]
    lines.extend("- " + str(item) for item in details[:5])
    if report.missing_required_columns:
        lines.append("- 缺少必要字段：" + "、".join(report.missing_required_columns))
    lines.append("")
    lines.append("请补齐/修正数据后重新上传。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 运营指标体系
# ---------------------------------------------------------------------------
def indicator_panel_data(
    cleaned_data: pd.DataFrame | None,
    *,
    thresholds: project_config.IndicatorThresholds | None = None,
) -> dict[str, Any]:
    """把 Phase 8 九项运营指标整理成界面所需的结构。"""

    limits = thresholds or project_config.DEFAULT_INDICATOR_THRESHOLDS
    empty_metrics = pd.DataFrame(columns=["指标", "数值", "单位", "计算口径"])

    if cleaned_data is None or not isinstance(cleaned_data, pd.DataFrame) or cleaned_data.empty:
        return {
            "available": False,
            "metrics": empty_metrics,
            "project_table": pd.DataFrame(columns=list(INDICATOR_COLUMNS)),
            "anomalies": pd.DataFrame(),
            "anomaly_summary": pd.DataFrame(),
            "thresholds_table": indicator_thresholds_frame(limits),
            "values": {},
            "anomaly_count": 0,
            "warning_count": 0,
            "headline": "暂无可用数据用于计算运营指标",
        }

    indicators = calculate_indicators(cleaned_data)
    anomalies = detect_indicator_anomalies(cleaned_data, thresholds=limits)
    projects = project_level_indicators(cleaned_data)
    summary = summarize_indicator_anomalies(anomalies)

    anomaly_count = int((anomalies["级别"] == LEVEL_ANOMALY).sum()) if not anomalies.empty else 0
    warning_count = int((anomalies["级别"] == LEVEL_WARNING).sum()) if not anomalies.empty else 0

    return {
        "available": True,
        "metrics": pd.DataFrame(indicators.as_rows()),
        "project_table": projects,
        "anomalies": anomalies,
        "anomaly_summary": summary,
        "thresholds_table": indicator_thresholds_frame(limits),
        "values": indicators.as_dict(),
        "anomaly_count": anomaly_count,
        "warning_count": warning_count,
        "headline": _indicator_headline(anomaly_count, warning_count),
    }


def _indicator_headline(anomaly_count: int, warning_count: int) -> str:
    if not anomaly_count and not warning_count:
        return "九项运营指标均在阈值范围内。"
    parts: list[str] = []
    if anomaly_count:
        parts.append("指标级异常 " + str(anomaly_count) + " 项")
    if warning_count:
        parts.append("预警 " + str(warning_count) + " 项")
    return "、".join(parts) + "（判定由程序按集中配置的阈值完成）。"

