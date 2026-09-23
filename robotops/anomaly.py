"""RobotOps AI —— Phase 1 异常识别模块。

四条异常规则（阈值集中定义在 ``config.AnomalyThresholds``）：

1. 故障率 > 5%（故障率 = 累计故障次数 / 累计巡检次数 × 100%）
2. 用户满意度 < 85 分
3. 运行率 < 80%（运行率 = 运行时长 / 计划运行时长 × 100%）
4. 节降率 < 10%

判定粒度：
- 故障率是比率型指标，默认按「机器人 + 分析周期」累计判定
  （``config.FAULT_RATE_ANOMALY_SCOPE = "robot_period"``），
  避免单日样本过小（1 次故障 / 8 次巡检 = 12.5%）造成误报；
- 满意度、运行率、节降率是每日指标，按记录逐条判定；
- 也可通过参数把故障率切换为逐条判定（``fault_rate_scope="record"``）。

严重程度按“偏差幅度”分级：
偏差幅度 = |实际值 - 阈值| / |阈值|，
>= 50% 记为「高」，>= 20% 记为「中」，其余记为「低」。

判定指标为空值的记录（例如巡检次数为 0 导致故障率无法计算）不会命中规则，
但会在数据清洗问题清单中体现，避免被静默忽略。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .exceptions import ConfigurationError
from .models import ANOMALY_COLUMNS, AnomalyRecord

SEVERITY_ORDER: tuple[str, ...] = ("高", "中", "低")


@dataclass(frozen=True)
class AnomalyRule:
    """单条异常判定规则。"""

    key: str
    anomaly_type: str
    metric_name: str
    column: str
    operator: str  # ">" 或 "<"
    threshold: float
    unit: str
    scope: str = "record"  # "record"（逐条记录）或 "robot_period"（机器人周期累计）

    @property
    def condition_text(self) -> str:
        return f"{self.metric_name} {self.operator} {self.threshold:g}{self.unit}"

    @property
    def granularity_label(self) -> str:
        return "机器人周期累计" if self.scope == "robot_period" else "逐日记录"

    def severity_and_deviation(self, actual: float) -> tuple[str, float]:
        """返回 (严重程度, 偏差幅度)。"""

        base = abs(self.threshold)
        if base == 0:
            return "高", float("inf")
        if self.operator == ">":
            deviation = (actual - self.threshold) / base
        else:
            deviation = (self.threshold - actual) / base
        deviation = max(deviation, 0.0)
        if deviation >= config.SEVERITY_HIGH_RATIO:
            return "高", deviation
        if deviation >= config.SEVERITY_MEDIUM_RATIO:
            return "中", deviation
        return "低", deviation


def build_rules(
    thresholds: config.AnomalyThresholds | None = None,
    *,
    fault_rate_scope: str | None = None,
) -> tuple[AnomalyRule, ...]:
    """根据阈值构造异常规则集合。"""

    limits = thresholds or config.DEFAULT_THRESHOLDS
    scope = fault_rate_scope or config.FAULT_RATE_ANOMALY_SCOPE
    if scope not in config.FAULT_RATE_SCOPE_CHOICES:
        raise ConfigurationError(
            f"fault_rate_scope 取值必须是 {config.FAULT_RATE_SCOPE_CHOICES} 之一，当前为：{scope}"
        )
    return (
        AnomalyRule(
            key="fault_rate_high",
            anomaly_type="故障率偏高",
            metric_name=config.COL_FAULT_RATE,
            column=config.COL_FAULT_RATE,
            operator=">",
            threshold=limits.fault_rate_upper,
            unit="%",
            scope=scope,
        ),
        AnomalyRule(
            key="satisfaction_low",
            anomaly_type="满意度偏低",
            metric_name=config.COL_SATISFACTION,
            column=config.COL_SATISFACTION,
            operator="<",
            threshold=limits.satisfaction_lower,
            unit="分",
        ),
        AnomalyRule(
            key="uptime_rate_low",
            anomaly_type="运行率偏低",
            metric_name=config.COL_UPTIME_RATE,
            column=config.COL_UPTIME_RATE,
            operator="<",
            threshold=limits.uptime_rate_lower,
            unit="%",
        ),
        AnomalyRule(
            key="saving_rate_low",
            anomaly_type="节降率偏低",
            metric_name=config.COL_SAVING_RATE,
            column=config.COL_SAVING_RATE,
            operator="<",
            threshold=limits.saving_rate_lower,
            unit="%",
        ),
    )


def empty_anomaly_frame() -> pd.DataFrame:
    """返回空的异常明细表（列结构完整）。"""

    return pd.DataFrame(columns=["异常序号", *ANOMALY_COLUMNS])


def detect_anomalies(
    frame: pd.DataFrame,
    thresholds: config.AnomalyThresholds | None = None,
    *,
    fault_rate_scope: str | None = None,
) -> pd.DataFrame:
    """逐条记录判定异常，返回异常明细表。"""

    records: list[AnomalyRecord] = []
    if frame is not None and not frame.empty:
        for rule in build_rules(thresholds, fault_rate_scope=fault_rate_scope):
            if rule.scope == "robot_period":
                records.extend(_apply_aggregate_rule(frame, rule))
            else:
                records.extend(_apply_rule(frame, rule))

    if not records:
        return empty_anomaly_frame()

    result = pd.DataFrame([record.to_dict() for record in records])
    severity_rank = {name: index for index, name in enumerate(SEVERITY_ORDER)}
    result["_严重程度排序"] = result["严重程度"].map(severity_rank)
    result = result.sort_values(
        ["_严重程度排序", "日期", "项目名称", "机器人ID"], kind="stable"
    ).drop(columns="_严重程度排序")
    result = result.reset_index(drop=True)
    result.insert(0, "异常序号", range(1, len(result) + 1))
    return result[["异常序号", *ANOMALY_COLUMNS]]


def summarize_anomalies(anomalies: pd.DataFrame) -> pd.DataFrame:
    """按异常类型汇总命中情况。"""

    columns = [
        "异常类型",
        "判定条件",
        "命中记录数",
        "占比(%)",
        "涉及项目数",
        "涉及机器人数",
        "高危",
        "中级",
        "低级",
    ]
    if anomalies is None or anomalies.empty:
        return pd.DataFrame(columns=columns)

    total = len(anomalies)
    rows: list[dict[str, object]] = []
    for anomaly_type, subset in anomalies.groupby("异常类型", sort=False):
        severity_counts = subset["严重程度"].value_counts()
        rows.append(
            {
                "异常类型": anomaly_type,
                "判定条件": str(subset["判定条件"].iloc[0]),
                "命中记录数": int(len(subset)),
                "占比(%)": round(len(subset) / total * 100, 2),
                "涉及项目数": int(subset["项目名称"].nunique()),
                "涉及机器人数": int(subset["机器人ID"].nunique()),
                "高危": int(severity_counts.get("高", 0)),
                "中级": int(severity_counts.get("中", 0)),
                "低级": int(severity_counts.get("低", 0)),
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("命中记录数", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def summarize_anomalies_by_project(anomalies: pd.DataFrame) -> pd.DataFrame:
    """按项目汇总异常分布，便于定位问题项目。"""

    columns = ["项目名称", "异常记录数", "涉及机器人数", "高危", "中级", "低级", "主要异常类型"]
    if anomalies is None or anomalies.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for project, subset in anomalies.groupby("项目名称", sort=False):
        severity_counts = subset["严重程度"].value_counts()
        main_types = subset["异常类型"].value_counts()
        main_text = "、".join(f"{name}({count})" for name, count in main_types.head(3).items())
        rows.append(
            {
                "项目名称": project,
                "异常记录数": int(len(subset)),
                "涉及机器人数": int(subset["机器人ID"].nunique()),
                "高危": int(severity_counts.get("高", 0)),
                "中级": int(severity_counts.get("中", 0)),
                "低级": int(severity_counts.get("低", 0)),
                "主要异常类型": main_text,
            }
        )

    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("异常记录数", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def describe_rules(thresholds: config.AnomalyThresholds | None = None) -> pd.DataFrame:
    """返回异常规则说明表，用于报告与文档。"""

    rows = []
    for rule in build_rules(thresholds):
        rows.append(
            {
                "异常类型": rule.anomaly_type,
                "判定指标": rule.metric_name,
                "判定条件": rule.condition_text,
                "判定粒度": rule.granularity_label,
                "严重程度分级": (
                    f"偏差幅度 >= {config.SEVERITY_HIGH_RATIO:.0%} 为高；"
                    f">= {config.SEVERITY_MEDIUM_RATIO:.0%} 为中；其余为低"
                ),
                "备注": "判定指标为空值的记录不参与该规则",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 内部实现
# ---------------------------------------------------------------------------
def _apply_rule(frame: pd.DataFrame, rule: AnomalyRule) -> list[AnomalyRecord]:
    if rule.column not in frame.columns:
        return []

    values = pd.to_numeric(frame[rule.column], errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(values)
    hit = valid & ((values > rule.threshold) if rule.operator == ">" else (values < rule.threshold))
    if not bool(hit.any()):
        return []

    records: list[AnomalyRecord] = []
    for position in np.flatnonzero(hit):
        row = frame.iloc[position]
        actual = float(values[position])
        severity, deviation = rule.severity_and_deviation(actual)
        records.append(
            AnomalyRecord(
                operation_date=_format_date(row[config.COL_DATE]),
                project_name=str(row[config.COL_PROJECT]),
                robot_id=str(row[config.COL_ROBOT_ID]),
                robot_type=str(row[config.COL_ROBOT_TYPE]),
                anomaly_type=rule.anomaly_type,
                metric_name=rule.metric_name,
                granularity=rule.granularity_label,
                actual_value=round(actual, 2),
                condition=f"{rule.operator} {rule.threshold:g}{rule.unit}",
                threshold=rule.threshold,
                deviation_ratio=deviation,
                severity=severity,
                description=_describe(rule, actual, severity, deviation),
            )
        )
    return records


def _apply_aggregate_rule(frame: pd.DataFrame, rule: AnomalyRule) -> list[AnomalyRecord]:
    """按「机器人 + 分析周期」累计判定比率型指标（当前用于故障率）。"""

    faults = pd.to_numeric(frame[config.COL_FAULT_COUNT], errors="coerce").to_numpy(
        dtype="float64"
    )
    denominators = pd.to_numeric(frame[config.COL_INSPECTION_COUNT], errors="coerce").to_numpy(
        dtype="float64"
    )

    records: list[AnomalyRecord] = []
    for _, index_positions in frame.groupby(config.COL_ROBOT_ID, sort=True).indices.items():
        positions = np.asarray(index_positions, dtype="int64")
        total_faults = float(np.nansum(faults[positions]))
        total_denominator = float(np.nansum(denominators[positions]))
        if not np.isfinite(total_denominator) or total_denominator <= 0:
            continue

        actual = total_faults / total_denominator * 100.0
        if not actual > rule.threshold:
            continue

        severity, deviation = rule.severity_and_deviation(actual)
        subset = frame.iloc[positions]
        period_start = subset[config.COL_DATE].min()
        period_end = subset[config.COL_DATE].max()
        period_text = f"{_format_date(period_start)} ~ {_format_date(period_end)}"
        records.append(
            AnomalyRecord(
                operation_date=period_text,
                project_name=str(subset[config.COL_PROJECT].iloc[0]),
                robot_id=str(subset[config.COL_ROBOT_ID].iloc[0]),
                robot_type=str(subset[config.COL_ROBOT_TYPE].iloc[0]),
                anomaly_type=rule.anomaly_type,
                metric_name=rule.metric_name,
                granularity=rule.granularity_label,
                actual_value=round(actual, 2),
                condition=f"{rule.operator} {rule.threshold:g}{rule.unit}",
                threshold=rule.threshold,
                deviation_ratio=deviation,
                severity=severity,
                description=(
                    f"{period_text} 累计{rule.metric_name} {actual:.2f}{rule.unit}"
                    f"（{total_faults:.0f} 次故障 / {total_denominator:.0f} 次巡检）"
                    f"{'高于' if rule.operator == '>' else '低于'}阈值 {rule.threshold:g}{rule.unit}，"
                    f"偏差 {deviation * 100:.1f}%，严重程度：{severity}"
                ),
            )
        )
    return records


def _describe(rule: AnomalyRule, actual: float, severity: str, deviation: float) -> str:
    relation = "高于" if rule.operator == ">" else "低于"
    return (
        f"{rule.metric_name} {actual:.2f}{rule.unit} {relation}阈值 {rule.threshold:g}{rule.unit}，"
        f"偏差 {deviation * 100:.1f}%，严重程度：{severity}"
    )


def _format_date(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)
