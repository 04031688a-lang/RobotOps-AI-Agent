"""RobotOps AI —— Phase 1 指标计算模块。

核心指标口径（与 README 保持一致）：

============ ==================================================
指标          口径
============ ==================================================
项目数量      对「项目名称」去重计数
机器人数量    对「机器人ID」去重计数
平均运行时长  清洗后记录的「运行时长」算术平均值（小时）
故障率        总故障次数 / 总巡检次数 × 100%
平均满意度    有效「用户满意度」记录的算术平均值（分）
总运营成本    有效「运营成本」记录的求和（元）
平均节降率    有效「节降率」记录的算术平均值（%）
============ ==================================================

说明：
- 均值与求和都会自动跳过空值（例如满意度越界被置空的记录不会拉低平均值）；
- 项目维度汇总复用同一套口径，保证与总体指标一致。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config
from .models import CORE_METRIC_DEFINITIONS, EXTRA_METRIC_DEFINITIONS, MetricsSummary


def calculate_core_metrics(frame: pd.DataFrame) -> MetricsSummary:
    """计算 Phase 1 核心指标。"""

    if frame is None or frame.empty:
        return MetricsSummary(
            project_count=0,
            robot_count=0,
            avg_running_hours=math.nan,
            fault_rate=math.nan,
            avg_satisfaction=math.nan,
            total_cost=0.0,
            avg_saving_rate=math.nan,
        )

    total_faults = _sum(frame[config.COL_FAULT_COUNT])
    total_inspections = _sum(frame[config.COL_INSPECTION_COUNT])
    fault_rate = total_faults / total_inspections * 100.0 if total_inspections > 0 else math.nan

    return MetricsSummary(
        project_count=int(frame[config.COL_PROJECT].nunique()),
        robot_count=int(frame[config.COL_ROBOT_ID].nunique()),
        avg_running_hours=_mean(frame[config.COL_RUNNING_HOURS]),
        fault_rate=fault_rate,
        avg_satisfaction=_mean(frame[config.COL_SATISFACTION]),
        total_cost=_sum(frame[config.COL_COST]),
        avg_saving_rate=_mean(frame[config.COL_SAVING_RATE]),
        record_count=int(len(frame)),
        avg_uptime_rate=_mean(frame[config.COL_UPTIME_RATE]),
        total_fault_count=int(round(total_faults)),
    )


def project_level_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """按项目维度汇总核心指标。"""

    columns = [
        "项目名称",
        "机器人数量",
        "记录数",
        "平均运行时长(小时)",
        "平均运行率(%)",
        "故障率(%)",
        "总故障次数",
        "平均满意度(分)",
        "总运营成本(元)",
        "平均节降率(%)",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for project, subset in frame.groupby(config.COL_PROJECT, sort=True):
        summary = calculate_core_metrics(subset)
        rows.append(
            {
                "项目名称": project,
                "机器人数量": summary.robot_count,
                "记录数": summary.record_count,
                "平均运行时长(小时)": _round(summary.avg_running_hours),
                "平均运行率(%)": _round(summary.avg_uptime_rate),
                "故障率(%)": _round(summary.fault_rate),
                "总故障次数": summary.total_fault_count,
                "平均满意度(分)": _round(summary.avg_satisfaction),
                "总运营成本(元)": _round(summary.total_cost),
                "平均节降率(%)": _round(summary.avg_saving_rate),
            }
        )

    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values("总运营成本(元)", ascending=False, kind="stable").reset_index(drop=True)


def metric_definitions_frame() -> pd.DataFrame:
    """返回指标口径表，用于报告与文档。"""

    rows = []
    for definition in CORE_METRIC_DEFINITIONS + EXTRA_METRIC_DEFINITIONS:
        rows.append(
            {
                "指标": definition.name,
                "单位": definition.unit,
                "指标类型": "核心指标" if definition.is_core else "补充统计",
                "计算口径": definition.formula,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _mean(series: pd.Series) -> float:
    values = _finite_values(series)
    if values.size == 0:
        return math.nan
    return float(values.mean())


def _sum(series: pd.Series) -> float:
    values = _finite_values(series)
    if values.size == 0:
        return 0.0
    return float(values.sum())


def _finite_values(series: pd.Series | None) -> np.ndarray:
    if series is None or len(series) == 0:
        return np.array([], dtype="float64")
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    return values[np.isfinite(values)]


def _round(value: float, digits: int = 2) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)

