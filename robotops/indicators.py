"""RobotOps AI —— Phase 8 机器人运营指标体系。

本模块提供 9 项确定性指标，全部由 Python / Pandas 计算，大模型不参与数值计算：

============ ==========================================================
指标          计算口径
============ ==========================================================
设备在线率    有有效运行记录的机器日 / 应运行机器日 × 100%
              （机器日 = 机器人数 × 分析周期天数）
设备利用率    累计运行时长 / 累计计划运行时长 × 100%（周期累计口径）
故障率        累计故障次数 / 累计巡检次数 × 100%
MTBF         累计运行时长 / 累计故障次数（小时/次）
MTTR         累计停机时长 / 累计维修次数（小时/次）
用户满意度    有效「用户满意度」记录的算术平均值（分）
总运营成本    有效「运营成本」记录的求和（元）
平均运营成本  总运营成本 / 有效成本记录数（元/条）
节降率        有效「节降率」记录的算术平均值（%）
============ ==========================================================

口径说明：
- 设备利用率是「周期累计」口径（先求和再相除），与 Phase 1 的「运行率」
  （逐日比值再平均）互补，两者分母相同但聚合方式不同；
- MTTR 的停机时长口径由 config.MTTR_DOWNTIME_PROXY 指定，当前为
  planned_gap = Σmax(计划运行时长 - 运行时长, 0)，即「未达计划的时长」；
  数据源没有维修工单时长字段时，这是可复现的代理口径；
- MTBF / MTTR 在故障次数或维修次数为 0 时返回空值，而不是 0，避免误读。

阈值判定（阈值集中定义在 config.IndicatorThresholds）：
- 故障率 > 5%          -> 异常
- 用户满意度 < 85 分    -> 异常
- 设备利用率 < 80%      -> 异常
- 节降率 < 10%          -> 异常
- MTTR 高于项目平均     -> 预警（按「同一项目内的机器人」比较）
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import config

#: 指标级别
LEVEL_ANOMALY = "异常"
LEVEL_WARNING = "预警"

INDICATOR_COLUMNS: tuple[str, ...] = (
    "项目名称",
    "机器人数",
    "记录数",
    "分析周期天数",
    "设备在线率(%)",
    "设备利用率(%)",
    "故障率(%)",
    "MTBF(小时/次)",
    "MTTR(小时/次)",
    "平均满意度(分)",
    "总运营成本(元)",
    "平均运营成本(元/条)",
    "平均节降率(%)",
)

ANOMALY_COLUMNS_PHASE8: tuple[str, ...] = (
    "对象类型",
    "对象名称",
    "项目名称",
    "指标",
    "实际值",
    "判定条件",
    "阈值",
    "级别",
    "说明",
)


@dataclass(frozen=True)
class IndicatorDefinition:
    """单项指标定义。"""

    name: str
    unit: str
    formula: str

    def as_dict(self) -> dict[str, str]:
        return {"指标": self.name, "单位": self.unit, "计算口径": self.formula}


INDICATOR_DEFINITIONS: tuple[IndicatorDefinition, ...] = (
    IndicatorDefinition(
        config.COL_ONLINE_RATE, "%", "有有效运行记录的机器日 / (机器人数 × 分析周期天数) × 100%"
    ),
    IndicatorDefinition(
        config.COL_UTILIZATION_RATE, "%", "累计运行时长 / 累计计划运行时长 × 100%（周期累计）"
    ),
    IndicatorDefinition("故障率", "%", "累计故障次数 / 累计巡检次数 × 100%"),
    IndicatorDefinition(config.COL_MTBF, "小时/次", "累计运行时长 / 累计故障次数"),
    IndicatorDefinition(config.COL_MTTR, "小时/次", "累计停机时长 / 累计维修次数（停机时长按计划缺口口径）"),
    IndicatorDefinition("用户满意度", "分", "有效「用户满意度」记录的算术平均值"),
    IndicatorDefinition("总运营成本", "元", "有效「运营成本」记录的求和"),
    IndicatorDefinition(config.COL_AVG_COST, "元/条", "总运营成本 / 有效成本记录数"),
    IndicatorDefinition("节降率", "%", "有效「节降率」记录的算术平均值"),
)


def indicator_definitions_frame() -> pd.DataFrame:
    """指标口径表（用于报告与文档）。"""

    return pd.DataFrame([definition.as_dict() for definition in INDICATOR_DEFINITIONS])


# ---------------------------------------------------------------------------
# 指标结果
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RobotOpsIndicators:
    """Phase 8 九项运营指标（含支撑数据）。"""

    online_rate: float = math.nan
    utilization_rate: float = math.nan
    fault_rate: float = math.nan
    mtbf: float = math.nan
    mttr: float = math.nan
    avg_satisfaction: float = math.nan
    total_cost: float = 0.0
    avg_cost: float = math.nan
    avg_saving_rate: float = math.nan
    # 支撑数据（用于解释指标，不参与阈值判定）
    robot_count: int = 0
    record_count: int = 0
    period_days: int = 0
    total_running_hours: float = 0.0
    total_planned_hours: float = 0.0
    total_fault_count: float = 0.0
    total_repair_count: float = 0.0
    downtime_hours: float = 0.0

    def value(self, name: str) -> float:
        mapping = {
            config.COL_ONLINE_RATE: self.online_rate,
            config.COL_UTILIZATION_RATE: self.utilization_rate,
            "故障率": self.fault_rate,
            config.COL_MTBF: self.mtbf,
            config.COL_MTTR: self.mttr,
            "用户满意度": self.avg_satisfaction,
            "总运营成本": self.total_cost,
            config.COL_AVG_COST: self.avg_cost,
            "节降率": self.avg_saving_rate,
        }
        return mapping.get(name, math.nan)

    def as_rows(self, *, digits: int = 2) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for definition in INDICATOR_DEFINITIONS:
            value = self.value(definition.name)
            if isinstance(value, float) and math.isnan(value):
                display: Any = None
            else:
                display = round(float(value), digits)
            rows.append(
                {
                    "指标": definition.name,
                    "数值": display,
                    "单位": definition.unit,
                    "计算口径": definition.formula,
                }
            )
        return rows

    def as_dict(self, *, digits: int = 2) -> dict[str, Any]:
        return {row["指标"]: row["数值"] for row in self.as_rows(digits=digits)}


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def calculate_indicators(
    data: pd.DataFrame | None,
    *,
    planned_hours_default: float | None = None,
) -> RobotOpsIndicators:
    """计算 Phase 8 九项指标。

    Args:
        data: 分析用数据（通常是清洗后的数据；缺少派生列时本函数自行计算）。
        planned_hours_default: 计划运行时长兜底值，默认取 config.DEFAULT_PLANNED_HOURS。

    Returns:
        RobotOpsIndicators：空表返回全空指标（不抛异常）。
    """

    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return RobotOpsIndicators()

    frame = data
    default_planned = (
        float(planned_hours_default)
        if planned_hours_default is not None
        else float(config.DEFAULT_PLANNED_HOURS)
    )

    running = _numeric(frame, config.COL_RUNNING_HOURS)
    planned = _planned_hours(frame, default_planned=default_planned)
    faults = _numeric(frame, config.COL_FAULT_COUNT)
    inspections = _numeric(frame, config.COL_INSPECTION_COUNT)
    repairs = _numeric(frame, config.COL_REPAIR_COUNT)
    satisfaction = _numeric(frame, config.COL_SATISFACTION)
    cost = _numeric(frame, config.COL_COST)
    saving = _numeric(frame, config.COL_SAVING_RATE)

    robot_count = _nunique(frame, config.COL_ROBOT_ID)
    record_count = int(len(frame))
    period_days = _period_days(frame)

    # 设备在线率：有有效运行记录的机器日 / 应运行机器日
    valid_running = int(np.isfinite(running).sum())
    expected_robot_days = robot_count * max(period_days, 1)
    online_rate = (
        min(valid_running / expected_robot_days, 1.0) * 100.0 if expected_robot_days > 0
        else math.nan
    )

    # 设备利用率：累计运行 / 累计计划
    running_total = _sum(running)
    planned_total = _sum(planned)
    utilization_rate = running_total / planned_total * 100.0 if planned_total > 0 else math.nan

    # 故障率：累计故障 / 累计巡检
    fault_total = _sum(faults)
    inspection_total = _sum(inspections)
    fault_rate = fault_total / inspection_total * 100.0 if inspection_total > 0 else math.nan

    # MTBF = 运行时长 / 故障次数；MTTR = 停机时长 / 维修次数
    mtbf = running_total / fault_total if fault_total > 0 else math.nan
    downtime = float(
        np.nansum(np.clip(np.where(np.isfinite(running), planned - running, 0.0), 0.0, None))
    )
    repair_total = _sum(repairs)
    mttr = downtime / repair_total if repair_total > 0 else math.nan

    cost_total = _sum(cost)
    valid_costs = int(np.isfinite(cost).sum())
    avg_cost = cost_total / valid_costs if valid_costs else math.nan

    return RobotOpsIndicators(
        online_rate=online_rate,
        utilization_rate=utilization_rate,
        fault_rate=fault_rate,
        mtbf=mtbf,
        mttr=mttr,
        avg_satisfaction=_mean(satisfaction),
        total_cost=cost_total,
        avg_cost=avg_cost,
        avg_saving_rate=_mean(saving),
        robot_count=robot_count,
        record_count=record_count,
        period_days=period_days,
        total_running_hours=running_total,
        total_planned_hours=planned_total,
        total_fault_count=fault_total,
        total_repair_count=repair_total,
        downtime_hours=downtime,
    )


def project_level_indicators(data: pd.DataFrame | None) -> pd.DataFrame:
    """按项目维度输出九项指标（含整体行）。"""

    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame(columns=list(INDICATOR_COLUMNS))

    rows: list[dict[str, Any]] = []
    rows.append(_indicator_row("全部项目（整体）", data))
    if config.COL_PROJECT in data.columns:
        for project, subset in data.groupby(config.COL_PROJECT, sort=True):
            rows.append(_indicator_row(str(project), subset))

    frame = pd.DataFrame(rows, columns=list(INDICATOR_COLUMNS))
    if len(frame) > 1:
        frame = pd.concat(
            [frame.iloc[[0]], frame.iloc[1:].sort_values("总运营成本(元)", ascending=False, kind="stable")],
            ignore_index=True,
        )
    return frame


def indicator_rows(indicators: RobotOpsIndicators) -> pd.DataFrame:
    """把指标结果转成展示表。"""

    return pd.DataFrame(indicators.as_rows())


# ---------------------------------------------------------------------------
# 阈值判定（聚合口径）
# ---------------------------------------------------------------------------
def detect_indicator_anomalies(
    data: pd.DataFrame | None,
    *,
    thresholds: config.IndicatorThresholds | None = None,
) -> pd.DataFrame:
    """按 Phase 8 阈值对聚合指标做异常 / 预警判定。

    判定对象：
    - 整体指标：故障率、满意度、设备利用率、节降率；
    - 每个项目的同四项指标；
    - MTTR：同一项目内逐台机器人与其「项目平均 MTTR」比较，高于平均值记预警。

    说明：本函数不修改 Phase 1 的逐条异常判定（robotops.anomaly），
    Phase 1 判定的是「记录级异常」，这里是「指标级异常」，两者互补。
    """

    limits = thresholds or config.DEFAULT_INDICATOR_THRESHOLDS
    records: list[dict[str, Any]] = []

    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame(columns=list(ANOMALY_COLUMNS_PHASE8))

    # -- 整体 -------------------------------------------------------------
    overall = calculate_indicators(data)
    records.extend(_scope_records("整体", "全部数据", "", overall, limits))

    # -- 按项目 -----------------------------------------------------------
    project_indicators: dict[str, RobotOpsIndicators] = {}
    if config.COL_PROJECT in data.columns:
        for project, subset in data.groupby(config.COL_PROJECT, sort=True):
            indicators = calculate_indicators(subset)
            project_indicators[str(project)] = indicators
            records.extend(_scope_records("项目", str(project), str(project), indicators, limits))

    # -- MTTR：机器人 vs 项目平均 -----------------------------------------
    records.extend(_mttr_warning_records(data, limits))

    if not records:
        return pd.DataFrame(columns=list(ANOMALY_COLUMNS_PHASE8))

    frame = pd.DataFrame(records, columns=list(ANOMALY_COLUMNS_PHASE8))
    frame["_级别排序"] = frame["级别"].map({LEVEL_ANOMALY: 0, LEVEL_WARNING: 1}).fillna(2)
    frame = frame.sort_values(["_级别排序", "指标"], kind="stable")
    return frame.drop(columns="_级别排序").reset_index(drop=True)


def summarize_indicator_anomalies(anomalies: pd.DataFrame | None) -> pd.DataFrame:
    """按指标汇总异常 / 预警命中情况。"""

    columns = ["指标", "级别", "命中对象数", "判定条件", "阈值"]
    if anomalies is None or anomalies.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for (metric, level), subset in anomalies.groupby(["指标", "级别"], sort=False):
        thresholds = sorted({round(float(value), 2) for value in subset["阈值"].tolist()})
        threshold_text: Any = (
            thresholds[0] if len(thresholds) == 1
            else _text(thresholds[0]) + " ～ " + _text(thresholds[-1])
        )
        rows.append(
            {
                "指标": metric,
                "级别": level,
                "命中对象数": int(len(subset)),
                "判定条件": str(subset["判定条件"].iloc[0]),
                "阈值": threshold_text,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def indicator_thresholds_frame(thresholds: config.IndicatorThresholds | None = None) -> pd.DataFrame:
    """阈值说明表（用于报告与界面）。"""

    limits = thresholds or config.DEFAULT_INDICATOR_THRESHOLDS
    return pd.DataFrame(
        [
            {
                "指标": "故障率",
                "判定条件": "> " + _text(limits.fault_rate_upper) + "%",
                "级别": LEVEL_ANOMALY,
            },
            {
                "指标": "用户满意度",
                "判定条件": "< " + _text(limits.satisfaction_lower) + " 分",
                "级别": LEVEL_ANOMALY,
            },
            {
                "指标": config.COL_UTILIZATION_RATE,
                "判定条件": "< " + _text(limits.utilization_lower) + "%",
                "级别": LEVEL_ANOMALY,
            },
            {
                "指标": "节降率",
                "判定条件": "< " + _text(limits.saving_rate_lower) + "%",
                "级别": LEVEL_ANOMALY,
            },
            {
                "指标": config.COL_MTTR,
                "判定条件": "> 项目平均 MTTR × " + _text(limits.mttr_project_avg_ratio),
                "级别": LEVEL_WARNING,
            },
        ]
    )


# ---------------------------------------------------------------------------
# 内部实现
# ---------------------------------------------------------------------------
def _scope_records(
    scope: str,
    name: str,
    project: str,
    indicators: RobotOpsIndicators,
    limits: config.IndicatorThresholds,
) -> list[dict[str, Any]]:
    """对单个对象（整体或项目）做四项阈值判定。"""

    records: list[dict[str, Any]] = []
    checks: tuple[tuple[str, float, str, float, str], ...] = (
        ("故障率", indicators.fault_rate, ">", limits.fault_rate_upper, "%"),
        ("用户满意度", indicators.avg_satisfaction, "<", limits.satisfaction_lower, "分"),
        (
            config.COL_UTILIZATION_RATE,
            indicators.utilization_rate,
            "<",
            limits.utilization_lower,
            "%",
        ),
        ("节降率", indicators.avg_saving_rate, "<", limits.saving_rate_lower, "%"),
    )
    for metric, value, operator, threshold, unit in checks:
        if value is None or not np.isfinite(value):
            continue
        hit = value > threshold if operator == ">" else value < threshold
        if not hit:
            continue
        relation = "高于" if operator == ">" else "低于"
        records.append(
            {
                "对象类型": scope,
                "对象名称": name,
                "项目名称": project,
                "指标": metric,
                "实际值": round(float(value), 2),
                "判定条件": operator + " " + _text(threshold) + unit,
                "阈值": threshold,
                "级别": LEVEL_ANOMALY,
                "说明": (
                    scope + "「" + name + "」" + metric + " " + _fmt(value)
                    + unit + "，" + relation + "阈值 " + _text(threshold) + unit
                ),
            }
        )
    return records


def _mttr_warning_records(
    data: pd.DataFrame, limits: config.IndicatorThresholds
) -> list[dict[str, Any]]:
    """MTTR 高于同项目平均水平 -> 预警。"""

    if config.COL_ROBOT_ID not in data.columns:
        return []

    robot_mttr: list[dict[str, Any]] = []
    project_column = config.COL_PROJECT if config.COL_PROJECT in data.columns else None
    groups = data.groupby(project_column, sort=True) if project_column else [("", data)]
    for project, subset in groups:
        if config.COL_ROBOT_ID not in subset.columns:
            continue
        per_robot: list[tuple[str, float]] = []
        for robot, robot_subset in subset.groupby(config.COL_ROBOT_ID, sort=True):
            mttr = calculate_indicators(robot_subset).mttr
            if np.isfinite(mttr):
                per_robot.append((str(robot), float(mttr)))
        if len(per_robot) < 2:
            # 只有一台设备时没有「项目平均」可比，跳过（不产出预警）
            continue
        average = float(np.mean([value for _, value in per_robot]))
        for robot, value in per_robot:
            if value > average * limits.mttr_project_avg_ratio:
                robot_mttr.append(
                    {
                        "对象类型": "机器人",
                        "对象名称": robot,
                        "项目名称": str(project),
                        "指标": config.COL_MTTR,
                        "实际值": round(value, 2),
                        "判定条件": "> 项目平均 MTTR × " + _text(limits.mttr_project_avg_ratio),
                        "阈值": round(average * limits.mttr_project_avg_ratio, 2),
                        "级别": LEVEL_WARNING,
                        "说明": (
                            "机器人「" + robot + "」MTTR " + _fmt(value) + " 小时/次，"
                            "高于项目「" + str(project) + "」平均水平 " + _fmt(average) + " 小时/次"
                        ),
                    }
                )
    return robot_mttr


def _indicator_row(name: str, subset: pd.DataFrame) -> dict[str, Any]:
    indicators = calculate_indicators(subset)
    if name == "全部项目（整体）":
        project_text = "全部项目"
    else:
        project_text = name
    return {
        "项目名称": project_text,
        "机器人数": indicators.robot_count,
        "记录数": indicators.record_count,
        "分析周期天数": indicators.period_days,
        "设备在线率(%)": _round(indicators.online_rate),
        "设备利用率(%)": _round(indicators.utilization_rate),
        "故障率(%)": _round(indicators.fault_rate),
        "MTBF(小时/次)": _round(indicators.mtbf),
        "MTTR(小时/次)": _round(indicators.mttr),
        "平均满意度(分)": _round(indicators.avg_satisfaction),
        "总运营成本(元)": _round(indicators.total_cost),
        "平均运营成本(元/条)": _round(indicators.avg_cost),
        "平均节降率(%)": _round(indicators.avg_saving_rate),
    }


def _planned_hours(frame: pd.DataFrame, *, default_planned: float) -> np.ndarray:
    """取计划运行时长（缺失/非正数时按机器人类型默认值兜底）。"""

    planned = _numeric(frame, config.COL_PLANNED_HOURS)
    invalid = ~np.isfinite(planned) | (planned <= 0)
    if bool(np.any(invalid)):
        fallback = _type_defaults(frame, default_planned)
        planned = np.where(invalid, fallback, planned)
    return planned


def _type_defaults(frame: pd.DataFrame, default_planned: float) -> np.ndarray:
    if config.COL_ROBOT_TYPE not in frame.columns:
        return np.full(len(frame), float(default_planned))
    return np.array(
        [
            float(config.PLANNED_HOURS_BY_TYPE.get(str(value).strip(), default_planned))
            for value in frame[config.COL_ROBOT_TYPE].tolist()
        ],
        dtype="float64",
    )


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), math.nan)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
    return values


def _nunique(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    try:
        return int(frame[column].nunique(dropna=True))
    except Exception:  # noqa: BLE001
        return 0


def _period_days(frame: pd.DataFrame) -> int:
    if config.COL_DATE not in frame.columns:
        return 1
    try:
        parsed = pd.to_datetime(frame[config.COL_DATE], errors="coerce")
        days = int(parsed.dt.normalize().nunique(dropna=True))
    except Exception:  # noqa: BLE001
        days = 0
    return max(days, 1)


def _sum(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.sum()) if finite.size else 0.0


def _mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else math.nan


def _round(value: float, digits: int = 2) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def _fmt(value: float) -> str:
    return _text(round(float(value), 2))


def _text(value: float) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(round(number, 2))

