"""RobotOps AI —— Phase 2 Agent 输入载荷构造。

核心原则：**Agent 的输入不是原始 Excel**。
先把 Phase 1（Pandas）算好的结构化分析结果压缩成一份小而准的 JSON，
再交给 DeepSeek 解读；载荷中不包含逐条明细，避免噪声与超长请求。

载荷结构（``robot-ops-analysis-v1``）：

- meta              ：数据源、分析窗口、阈值、口径开关等上下文；
- units             ：每个指标的**单位**，避免模型误读数量级；
- core_metrics      ：Phase 1 计算的 7 项核心指标 + 3 项补充统计；
- metric_definitions：指标计算口径说明；
- projects          ：项目维度汇总（全部项目）；
- abnormal_projects ：异常项目汇总（仅命中异常的项目）；
- abnormal_robots   ：异常机器人明细（仅命中异常的机器人，取安全字段）；
- anomaly_summary   ：异常类型汇总；
- data_quality      ：清洗统计与清洗问题清单；
- notes             ：口径与解读注意事项。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .. import config
from ..metrics import calculate_core_metrics, metric_definitions_frame
from ..models import MetricsSummary
from ..pipeline import AnalysisResult

PAYLOAD_VERSION = "robot-ops-analysis-v1"
MAX_CLEANING_ISSUES = 20
MAX_ABNORMAL_ROBOTS = 20

#: 指标单位（与 Phase 1 报告完全一致，避免模型自行换算数量级）
METRIC_UNITS: dict[str, str] = {
    "project_count": "个",
    "robot_count": "台",
    "record_count": "条",
    "avg_runtime": "小时/天",
    "avg_uptime_rate": "%",
    "avg_fault_rate": "%（累计：故障次数/巡检次数）",
    "total_fault_count": "次",
    "total_maintenance_count": "次",
    "maintenance_to_fault_ratio": "%（维修次数/故障次数）",
    "avg_satisfaction": "分（0-100）",
    "total_cost": "元",
    "avg_cost_reduction_rate": "%",
}


def build_analysis_payload(result: AnalysisResult) -> dict[str, Any]:
    """把 Phase 1 的分析结果转换为 Agent 输入载荷。"""

    if not isinstance(result, AnalysisResult):
        raise TypeError(
            "Agent 输入必须是 Phase 1 的分析结果（AnalysisResult）；"
            "不接受原始 Excel 或 DataFrame，请先运行 Phase 1 的 Pandas 分析流程。"
        )

    data = result.cleaned_data
    metrics = result.metrics

    payload: dict[str, Any] = {
        "payload_version": PAYLOAD_VERSION,
        "meta": _build_meta(result),
        "units": dict(METRIC_UNITS),
        "core_metrics": _core_metrics(metrics, data),
        "metric_definitions": _metric_definitions(),
        "projects": _projects(result),
        "abnormal_projects": _abnormal_projects(result),
        "abnormal_robots": _abnormal_robots(result),
        "anomaly_summary": _anomaly_summary(result),
        "data_quality": _data_quality(result),
        "notes": _notes(result),
    }
    return payload


def payload_to_json(payload: dict[str, Any], *, indent: int | None = 2) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def payload_to_text(payload: dict[str, Any], *, indent: int = 2) -> str:
    """控制台展示用的载荷文本（与发送给模型的内容一致）。"""

    return payload_to_json(payload, indent=indent)


# ---------------------------------------------------------------------------
# 各分块构造
# ---------------------------------------------------------------------------
def _build_meta(result: AnalysisResult) -> dict[str, Any]:
    data = result.cleaned_data
    window_start = data[config.COL_DATE].min()
    window_end = data[config.COL_DATE].max()
    window = (
        f"{window_start:%Y-%m-%d} ~ {window_end:%Y-%m-%d}"
        if pd.notna(window_start) and pd.notna(window_end)
        else "-"
    )

    return {
        "generated_at": result.generated_at,
        "data_source": Path(result.data_path).name,
        "sheet_name": result.sheet_name,
        "analysis_window": window,
        "cleaned_record_count": int(len(data)),
        "raw_record_count": int(result.cleaned.stats.get("原始记录数", 0)),
        "robot_types": sorted(data[config.COL_ROBOT_TYPE].dropna().unique().tolist()),
        "thresholds": result.thresholds.as_dict(),
        "fault_rate_anomaly_scope": config.FAULT_RATE_ANOMALY_SCOPE,
        "severity_rule": (
            f"偏差幅度=|实际值-阈值|/|阈值|，>={config.SEVERITY_HIGH_RATIO:.0%} 为高，"
            f">={config.SEVERITY_MEDIUM_RATIO:.0%} 为中，其余为低"
        ),
    }


def _core_metrics(metrics: MetricsSummary, data: pd.DataFrame) -> dict[str, Any]:
    total_maintenance = (
        float(pd.to_numeric(data[config.COL_REPAIR_COUNT], errors="coerce").sum())
        if config.COL_REPAIR_COUNT in data.columns
        else 0.0
    )
    ratio = (
        round(total_maintenance / metrics.total_fault_count * 100, 2)
        if metrics.total_fault_count
        else None
    )
    return {
        "project_count": int(metrics.project_count),
        "robot_count": int(metrics.robot_count),
        "record_count": int(metrics.record_count),
        "avg_runtime": _round(metrics.avg_running_hours),
        "avg_uptime_rate": _round(metrics.avg_uptime_rate),
        "avg_fault_rate": _round(metrics.fault_rate),
        "total_fault_count": int(metrics.total_fault_count),
        "total_maintenance_count": int(round(total_maintenance)),
        "maintenance_to_fault_ratio": ratio,
        "avg_satisfaction": _round(metrics.avg_satisfaction),
        "total_cost": _round(metrics.total_cost),
        "avg_cost_reduction_rate": _round(metrics.avg_saving_rate),
    }


def _metric_definitions() -> list[dict[str, Any]]:
    frame = metric_definitions_frame()
    return [
        {
            "metric": str(row["指标"]),
            "unit": str(row["单位"]),
            "type": str(row["指标类型"]),
            "formula": str(row["计算口径"]),
        }
        for _, row in frame.iterrows()
    ]


def _projects(result: AnalysisResult) -> list[dict[str, Any]]:
    summary = result.project_summary
    projects: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        projects.append(
            {
                "project": str(row["项目名称"]),
                "robot_count": int(row["机器人数量"]),
                "record_count": int(row["记录数"]),
                "avg_runtime": _round(row["平均运行时长(小时)"]),
                "avg_uptime_rate": _round(row["平均运行率(%)"]),
                "avg_fault_rate": _round(row["故障率(%)"]),
                "total_fault_count": int(row["总故障次数"]),
                "avg_satisfaction": _round(row["平均满意度(分)"]),
                "total_cost": _round(row["总运营成本(元)"]),
                "avg_cost_reduction_rate": _round(row["平均节降率(%)"]),
            }
        )
    return projects


def _abnormal_projects(result: AnalysisResult) -> list[dict[str, Any]]:
    summary = result.anomaly_project_summary
    if summary is None or summary.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "project": str(row["项目名称"]),
                "anomaly_records": int(row["异常记录数"]),
                "affected_robots": int(row["涉及机器人数"]),
                "high_severity": int(row["高危"]),
                "medium_severity": int(row["中级"]),
                "low_severity": int(row["低级"]),
                "main_anomaly_types": str(row["主要异常类型"]),
            }
        )
    return rows


def _abnormal_robots(result: AnalysisResult) -> list[dict[str, Any]]:
    anomalies = result.anomalies
    if anomalies is None or anomalies.empty:
        return []

    data = result.cleaned_data
    grouped = anomalies.groupby("机器人ID", sort=False)
    rows: list[dict[str, Any]] = []

    for robot_id, group in grouped:
        subset = data.loc[data[config.COL_ROBOT_ID] == robot_id]
        if subset.empty:
            continue
        robot_metrics = calculate_core_metrics(subset)
        types = sorted({str(item) for item in group["异常类型"].tolist()})
        rows.append(
            {
                "robot_id": str(robot_id),
                "project": str(subset[config.COL_PROJECT].iloc[0]),
                "robot_type": str(subset[config.COL_ROBOT_TYPE].iloc[0]),
                "fault_rate": _round(robot_metrics.fault_rate),
                "avg_uptime_rate": _round(robot_metrics.avg_uptime_rate),
                "avg_satisfaction": _round(robot_metrics.avg_satisfaction),
                "avg_cost_reduction_rate": _round(robot_metrics.avg_saving_rate),
                "total_cost": _round(robot_metrics.total_cost),
                "record_count": int(robot_metrics.record_count),
                "anomaly_records": int(len(group)),
                "high_severity_records": int((group["严重程度"] == "高").sum()),
                "anomaly_types": types,
            }
        )

    rows.sort(
        key=lambda item: (
            item["anomaly_records"],
            item["high_severity_records"],
            item["fault_rate"] or 0,
        ),
        reverse=True,
    )
    return rows[:MAX_ABNORMAL_ROBOTS]


def _anomaly_summary(result: AnalysisResult) -> list[dict[str, Any]]:
    summary = result.anomaly_summary
    if summary is None or summary.empty:
        return []
    return [
        {
            "anomaly_type": str(row["异常类型"]),
            "condition": str(row["判定条件"]),
            "hits": int(row["命中记录数"]),
            "share_of_anomalies": _round(row["占比(%)"]),
            "projects_involved": int(row["涉及项目数"]),
            "robots_involved": int(row["涉及机器人数"]),
            "high": int(row["高危"]),
            "medium": int(row["中级"]),
            "low": int(row["低级"]),
        }
        for _, row in summary.iterrows()
    ]


def _data_quality(result: AnalysisResult) -> dict[str, Any]:
    issues = result.cleaned.issues
    issue_rows: list[dict[str, Any]] = []
    if issues is not None and not issues.empty:
        for _, row in issues.head(MAX_CLEANING_ISSUES).iterrows():
            issue_rows.append(
                {
                    "level": str(row["级别"]),
                    "stage": str(row["环节"]),
                    "issue": str(row["问题类型"]),
                    "description": str(row["问题描述"]),
                    "affected_rows": int(row["涉及行数"]),
                    "action": str(row["处理方式"]),
                }
            )
    return {
        "cleaning_stats": result.cleaned.stats,
        "issues": issue_rows,
        "issues_total": int(len(issues)) if issues is not None else 0,
    }


def _notes(result: AnalysisResult) -> list[str]:
    thresholds = result.thresholds
    return [
        "core_metrics 与 projects 中的所有数值均由 Phase 1 的 Pandas 模块计算完成，模型不得重新计算或修改。",
        "故障率为累计口径：机器人周期累计故障次数 ÷ 累计巡检次数 × 100%。",
        f"异常判定阈值：故障率 > {thresholds.fault_rate_upper:g}%、满意度 < {thresholds.satisfaction_lower:g} 分、"
        f"运行率 < {thresholds.uptime_rate_lower:g}%、节降率 < {thresholds.saving_rate_lower:g}%。",
        "故障率按「机器人 + 分析周期」累计判定；满意度、运行率、节降率按每日记录判定。",
        "运行率 = 运行时长 ÷ 计划运行时长 × 100%；维修/故障比 = 维修次数 ÷ 故障次数 × 100%。",
        "数据中不含故障类型、停机原因、排班与充电、单台成本明细等字段，涉及这类原因的判断必须标注为推测或数据不足。",
        f"清洗阶段共处理 {int(result.cleaned.stats.get('原始记录数', 0))} 条原始记录，"
        f"其中剔除 {int(result.cleaned.stats.get('剔除记录数', 0))} 条重复记录，"
        f"最终 {int(result.cleaned.stats.get('清洗后记录数', 0))} 条参与分析。",
    ]


def _round(value: Any, digits: int = 2) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return round(number, digits)

