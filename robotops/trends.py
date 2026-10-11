"""RobotOps AI —— Phase 9 运营趋势分析。

把清洗后的运营数据按「日 / 周 / 月」聚合，生成 7 项指标的确定性趋势，并计算：

- 环比（相邻周期变化率）
- 同期变化（日=7 天前、周=4 周前、月=去年同期，数据支持时计算）
- 上升 / 下降 / 持平
- 连续上升 / 连续下降（连续同向变动的周期数）

七项趋势指标：

============ ==================================================
指标          区间口径
============ ==================================================
故障率        区间故障次数合计 / 巡检次数合计 × 100%
设备运行率    区间内逐日运行率（运行时长/计划运行时长）的平均
设备利用率    区间运行时长合计 / 计划运行时长合计 × 100%
用户满意度    区间满意度算术平均值
运营成本      区间运营成本合计（元）
节降率        区间节降率算术平均值
MTTR         区间停机时长合计 / 维修次数合计
============ ==================================================

设计原则：
1. 全部数值都由 Python / Pandas 计算，大模型只负责解释；
2. 粒度可显式指定（day/week/month），也可按数据跨度自动选择；
3. 数据不足（周期数不够）或日期异常时不抛异常，而是给出明确的 warnings；
4. 参数集中在 config.TrendSettings。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# 指标定义
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrendMetricDefinition:
    """单项趋势指标的定义。"""

    key: str
    name: str
    unit: str
    aggregation: str  # ratio_of_sums / mean / sum / mean_ratio
    formula: str

    def as_dict(self) -> dict[str, str]:
        return {
            "指标": self.name,
            "单位": self.unit,
            "聚合方式": AGGREGATION_LABELS.get(self.aggregation, self.aggregation),
            "区间口径": self.formula,
        }


AGGREGATION_LABELS: dict[str, str] = {
    "ratio_of_sums": "先求和再相除",
    "mean": "算术平均",
    "sum": "求和",
    "mean_ratio": "逐条比值再平均",
    "ratio_downtime": "停机时长合计 ÷ 维修次数合计",
}

TREND_METRICS: tuple[TrendMetricDefinition, ...] = (
    TrendMetricDefinition(
        "fault_rate", "故障率", "%", "ratio_of_sums",
        "区间故障次数合计 ÷ 巡检次数合计 × 100%",
    ),
    TrendMetricDefinition(
        "uptime_rate", "设备运行率", "%", "mean_ratio",
        "区间内逐日运行率（运行时长 ÷ 计划运行时长 × 100%）的平均",
    ),
    TrendMetricDefinition(
        "utilization_rate", "设备利用率", "%", "ratio_of_sums",
        "区间运行时长合计 ÷ 计划运行时长合计 × 100%",
    ),
    TrendMetricDefinition(
        "satisfaction", "用户满意度", "分", "mean", "区间满意度算术平均值",
    ),
    TrendMetricDefinition(
        "total_cost", "运营成本", "元", "sum", "区间运营成本合计",
    ),
    TrendMetricDefinition(
        "saving_rate", "节降率", "%", "mean", "区间节降率算术平均值",
    ),
    TrendMetricDefinition(
        "mttr", "MTTR", "小时/次", "ratio_downtime", "区间停机时长合计 ÷ 维修次数合计",
    ),
)

METRIC_BY_KEY: dict[str, TrendMetricDefinition] = {item.key: item for item in TREND_METRICS}


def trend_metric_definitions_frame() -> pd.DataFrame:
    """趋势指标口径表（用于报告与界面）。"""

    return pd.DataFrame([item.as_dict() for item in TREND_METRICS])


# ---------------------------------------------------------------------------
# 变化结果模型
# ---------------------------------------------------------------------------
@dataclass
class TrendChange:
    """单项指标的环比 / 同期 / 连续变化结果。"""

    metric_key: str
    metric_name: str
    unit: str
    latest_period: str = ""
    latest_value: float | None = None
    previous_period: str = ""
    previous_value: float | None = None
    change_ratio: float | None = None
    change_absolute: float | None = None
    direction: str = ""
    streak_direction: str = ""
    streak_moves: int = 0
    is_consecutive_up: bool = False
    is_consecutive_down: bool = False
    same_period_label: str = ""
    same_period_value: float | None = None
    same_period_ratio: float | None = None
    same_period_note: str = "数据不足以计算同期变化"
    status: str = "正常"

    @property
    def consecutive_periods(self) -> int:
        """连续同向变化覆盖的周期数（1 次变动 = 2 个周期）。"""

        return self.streak_moves + 1 if self.streak_moves else 0

    @property
    def change_percent(self) -> float | None:
        return None if self.change_ratio is None else round(self.change_ratio * 100.0, 2)

    @property
    def same_period_percent(self) -> float | None:
        return None if self.same_period_ratio is None else round(self.same_period_ratio * 100.0, 2)

    def as_row(self) -> dict[str, Any]:
        """展示行（趋势变化表）。"""

        if self.status != "正常":
            streak_text = "-"
        elif self.is_consecutive_up:
            streak_text = "连续 " + str(self.consecutive_periods) + " 期上升"
        elif self.is_consecutive_down:
            streak_text = "连续 " + str(self.consecutive_periods) + " 期下降"
        else:
            streak_text = "未连续"

        return {
            "指标": self.metric_name,
            "最新期间": self.latest_period,
            "最新值": _round(self.latest_value),
            "上期值": _round(self.previous_value),
            "环比(%)": self.change_percent,
            "方向": self.direction or "-",
            "连续变化": streak_text,
            "同期变化(%)": self.same_period_percent,
            "同期说明": self.same_period_label or self.same_period_note,
            "单位": self.unit,
            "状态": self.status,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.metric_name,
            "unit": self.unit,
            "latest_period": self.latest_period,
            "latest_value": _round(self.latest_value),
            "previous_period": self.previous_period,
            "previous_value": _round(self.previous_value),
            "change_ratio": _round(self.change_ratio, 4),
            "change_percent": self.change_percent,
            "change_absolute": _round(self.change_absolute),
            "direction": self.direction,
            "streak": {
                "direction": self.streak_direction,
                "moves": self.streak_moves,
                "periods": self.consecutive_periods,
                "is_consecutive_up": self.is_consecutive_up,
                "is_consecutive_down": self.is_consecutive_down,
            },
            "same_period": {
                "label": self.same_period_label,
                "value": _round(self.same_period_value),
                "change_ratio": _round(self.same_period_ratio, 4),
                "note": self.same_period_note,
            },
            "status": self.status,
        }

    def highlight(self) -> str:
        """一句话趋势事实（供报告与大模型解释使用）。"""

        if self.status != "正常":
            return self.metric_name + "：" + self.status
        text = (
            self.metric_name + "最新（" + self.latest_period + "）为 "
            + _text(self.latest_value) + self.unit
        )
        if self.previous_value is not None and self.change_ratio is not None:
            text += (
                "，上期 " + _text(self.previous_value) + self.unit
                + "，环比" + (self.direction or "持平")
                + " " + _text(self.change_percent) + "%"
            )
        if self.is_consecutive_up or self.is_consecutive_down:
            text += "，已连续 " + str(self.consecutive_periods) + " 期" + (self.streak_direction or "")
        if self.same_period_ratio is not None:
            text += (
                "；" + self.same_period_label + " " + _text(self.same_period_percent) + "%"
            )
        return text


@dataclass
class TrendAnalysis:
    """一次趋势分析的全部结果。"""

    granularity: str
    granularity_label: str
    auto_detected: bool = False
    period_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    changes: list[TrendChange] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    date_start: pd.Timestamp | None = None
    date_end: pd.Timestamp | None = None
    valid_rows: int = 0
    invalid_date_rows: int = 0
    partial_periods: list[str] = field(default_factory=list)
    settings: config.TrendSettings = field(default_factory=lambda: config.DEFAULT_TREND_SETTINGS)

    # -- 基本信息 ---------------------------------------------------------
    @property
    def period_count(self) -> int:
        return int(len(self.period_frame))

    @property
    def available(self) -> bool:
        return self.period_count > 0 and not self.period_frame.empty

    @property
    def metric_names(self) -> list[str]:
        return [item.name for item in TREND_METRICS]

    @property
    def window_text(self) -> str:
        if self.date_start is None or self.date_end is None:
            return "-"
        return self.date_start.strftime("%Y-%m-%d") + " ~ " + self.date_end.strftime("%Y-%m-%d")

    def change_of(self, metric_key: str) -> TrendChange | None:
        for item in self.changes:
            if item.metric_key == metric_key:
                return item
        return None

    # -- 输出 -------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """趋势变化表（每项指标一行）。"""

        columns = [
            "指标", "最新期间", "最新值", "上期值", "环比(%)", "方向",
            "连续变化", "同期变化(%)", "同期说明", "单位", "状态",
        ]
        if not self.changes:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([item.as_row() for item in self.changes], columns=columns)

    def series_frame(self) -> pd.DataFrame:
        """周期明细表（期间 + 7 项指标）。"""

        return self.period_frame.copy()

    def chart_frame(self, metric_key: str) -> pd.DataFrame:
        """折线图数据：期间 + 指标值。"""

        definition = METRIC_BY_KEY.get(metric_key)
        if definition is None or self.period_frame.empty:
            return pd.DataFrame(columns=["期间", "数值"])
        frame = self.period_frame[["期间", definition.name]].copy()
        return frame.rename(columns={definition.name: "数值"})

    def highlight_lines(self, *, limit: int = 7) -> list[str]:
        """趋势事实要点（不含解读，解读交给大模型）。"""

        lines: list[str] = []
        for change in self.changes:
            if change.status != "正常":
                continue
            if change.is_consecutive_up or change.is_consecutive_down:
                lines.append(change.highlight())
        for change in self.changes:
            if change.status != "正常":
                continue
            if not (change.is_consecutive_up or change.is_consecutive_down):
                line = change.highlight()
                if line not in lines:
                    lines.append(line)
        return lines[:limit]

    def changes_dict(self) -> dict[str, float]:
        """紧凑变化字典（例如 fault_rate_change: 0.32）。"""

        values: dict[str, float] = {}
        for change in self.changes:
            if change.change_ratio is not None:
                values[change.metric_key + "_change"] = round(float(change.change_ratio), 4)
        return values

    def as_dict(self) -> dict[str, Any]:
        """完整结构化结果（可 JSON 序列化）。"""

        return {
            "granularity": self.granularity,
            "granularity_label": self.granularity_label,
            "auto_detected": self.auto_detected,
            "window": {
                "start": None if self.date_start is None else self.date_start.strftime("%Y-%m-%d"),
                "end": None if self.date_end is None else self.date_end.strftime("%Y-%m-%d"),
            },
            "period_count": self.period_count,
            "periods": self.period_frame["期间"].tolist() if self.available else [],
            "valid_rows": int(self.valid_rows),
            "invalid_date_rows": int(self.invalid_date_rows),
            "partial_periods": list(self.partial_periods),
            "metrics": {item.metric_key: item.as_dict() for item in self.changes},
            "changes": self.changes_dict(),
            "highlights": self.highlight_lines(),
            "warnings": list(self.warnings),
            "settings": self.settings.as_dict(),
        }

    def summary_dict(self) -> dict[str, Any]:
        """提供给 Agent / 前端的最小结构化摘要。"""

        return {
            "granularity": self.granularity,
            "granularity_label": self.granularity_label,
            "period_count": self.period_count,
            "window": {
                "start": None if self.date_start is None else self.date_start.strftime("%Y-%m-%d"),
                "end": None if self.date_end is None else self.date_end.strftime("%Y-%m-%d"),
            },
            "changes": self.changes_dict(),
            "highlights": self.highlight_lines(),
            "warnings": list(self.warnings),
            "partial_periods": list(self.partial_periods),
        }

    def summary_text(self) -> str:
        lines = [
            "【运营趋势（粒度：" + self.granularity_label + "）】",
            "  分析窗口：" + self.window_text + "，周期数：" + str(self.period_count),
        ]
        for line in self.highlight_lines(limit=10):
            lines.append("  · " + line)
        for warning in self.warnings:
            lines.append("  ! " + warning)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 指标计算（供趋势与项目对比共用）
# ---------------------------------------------------------------------------
def metric_value(frame: pd.DataFrame, definition: TrendMetricDefinition | str) -> float:
    """按指标定义计算某个数据子集（例如一个周期、一个项目）的指标值。"""

    item = METRIC_BY_KEY.get(definition) if isinstance(definition, str) else definition
    if item is None or frame is None or len(frame) == 0:
        return math.nan

    if item.aggregation == "ratio_of_sums":
        if item.key == "fault_rate":
            numerator = _total(frame, config.COL_FAULT_COUNT)
            denominator = _total(frame, config.COL_INSPECTION_COUNT)
        else:  # utilization_rate
            numerator = _total(frame, config.COL_RUNNING_HOURS)
            denominator = _total_planned(frame)
        return numerator / denominator * 100.0 if denominator > 0 else math.nan

    if item.aggregation == "ratio_downtime":
        downtime = _downtime_total(frame)
        repairs = _total(frame, config.COL_REPAIR_COUNT)
        return downtime / repairs if repairs > 0 else math.nan

    if item.aggregation == "mean_ratio":
        rates = _uptime_rates(frame)
        finite = rates[np.isfinite(rates)]
        return float(finite.mean()) if finite.size else math.nan

    if item.aggregation == "sum":
        return _total(frame, config.COL_COST)

    column = {
        "satisfaction": config.COL_SATISFACTION,
        "saving_rate": config.COL_SAVING_RATE,
    }.get(item.key)
    return _mean(frame, column) if column else math.nan


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    """一次算出全部 7 项指标。"""

    return {item.key: metric_value(frame, item) for item in TREND_METRICS}


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def build_trend_series(
    data: pd.DataFrame | None,
    *,
    granularity: str | None = None,
    settings: config.TrendSettings | None = None,
) -> TrendAnalysis:
    """按指定粒度（或自动粒度）生成运营趋势。

    自动粒度规则：数据天数 >= 90 用月、>= 21 用周、其余用日。

    Args:
        data: 清洗后的运营数据（需包含日期列）。
        granularity: day / week / month / auto，默认取 config.TrendSettings.default_granularity。
        settings: 趋势参数，默认 config.DEFAULT_TREND_SETTINGS。

    Returns:
        TrendAnalysis：数据不足或日期异常时返回带 warnings 的结果，不抛异常。
    """

    limits = settings or config.DEFAULT_TREND_SETTINGS
    requested = (granularity or limits.default_granularity or config.TREND_GRANULARITY_AUTO).lower()
    if requested not in (
        *config.TREND_GRANULARITY_CHOICES,
        config.TREND_GRANULARITY_AUTO,
    ):
        requested = config.TREND_GRANULARITY_AUTO

    empty = TrendAnalysis(
        granularity=config.TREND_GRANULARITY_DAY,
        granularity_label=config.TREND_GRANULARITY_LABELS[config.TREND_GRANULARITY_DAY],
        auto_detected=requested == config.TREND_GRANULARITY_AUTO,
        warnings=[],
        settings=limits,
    )

    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        empty.warnings.append("数据为空，无法生成趋势")
        return empty

    if config.COL_DATE not in data.columns:
        empty.warnings.append("数据缺少「" + config.COL_DATE + "」字段，无法生成趋势")
        return empty

    frame = data.copy()
    parsed = _parse_dates(frame[config.COL_DATE])
    invalid_mask = parsed.isna()
    invalid_rows = int(invalid_mask.sum())
    if invalid_rows:
        frame = frame.loc[~invalid_mask].copy()
        parsed = parsed.loc[~invalid_mask]
    if frame.empty:
        empty.invalid_date_rows = invalid_rows
        empty.warnings.append(
            "日期异常：没有可用日期（非法/为空 " + str(invalid_rows) + " 条），无法生成趋势"
        )
        return empty

    frame["_日期"] = parsed.dt.normalize()

    resolved, auto_detected = _resolve_granularity(frame["_日期"], requested, limits)
    periods = _period_columns(frame["_日期"], resolved)
    frame = pd.concat([frame, periods], axis=1)

    period_keys = list(dict.fromkeys(periods["_期间键"].tolist()))
    rows: list[dict[str, Any]] = []
    for key in period_keys:
        subset = frame.loc[frame["_期间键"] == key]
        row: dict[str, Any] = {
            "期间": str(subset["_期间"].iloc[0]),
            "开始日期": subset["_开始日期"].iloc[0].strftime("%Y-%m-%d"),
            "结束日期": subset["_结束日期"].iloc[0].strftime("%Y-%m-%d"),
            "记录数": int(len(subset)),
        }
        for definition in TREND_METRICS:
            row[definition.name] = metric_value(subset, definition)
        rows.append(row)

    period_frame = pd.DataFrame(rows)
    if not period_frame.empty:
        period_frame = period_frame.sort_values("开始日期", kind="stable").reset_index(drop=True)

    warnings: list[str] = []
    if invalid_rows:
        warnings.append(
            "日期异常：" + str(invalid_rows) + " 条记录的日期无法解析，已在趋势分析中排除"
        )

    # 标记不完整周期：首尾周期可能只覆盖部分天数（例如最后一周只有 1 天数据）
    if not period_frame.empty:
        period_start = pd.to_datetime(period_frame["开始日期"])
        period_end = pd.to_datetime(period_frame["结束日期"])
        period_frame["完整"] = (period_start >= frame["_日期"].min()) & (
            period_end <= frame["_日期"].max()
        )
    partial_labels = (
        period_frame.loc[~period_frame["完整"], "期间"].tolist() if not period_frame.empty else []
    )

    usable = period_frame
    if partial_labels and limits.exclude_partial_periods:
        complete = period_frame.loc[period_frame["完整"]].reset_index(drop=True)
        if len(complete) >= limits.min_periods_for_change:
            usable = complete
            warnings.append(
                "周期不完整："
                + "、".join(partial_labels[:3])
                + "（共 " + str(len(partial_labels)) + " 个）数据未覆盖整个周期，"
                "已排除在环比与连续变化判定之外"
            )
        else:
            warnings.append(
                "周期不完整："
                + "、".join(partial_labels[:3])
                + " 数据未覆盖整个周期，但完整周期不足，仍按全部周期计算（变化幅度可能失真）"
            )

    if len(usable) < limits.min_periods_for_change:
        warnings.append(
            "数据不足：当前只有 " + str(len(usable)) + " 个可用周期，"
            "至少需要 " + str(limits.min_periods_for_change) + " 个周期才能计算环比"
        )

    changes = _build_changes(usable, resolved, limits)

    return TrendAnalysis(
        granularity=resolved,
        granularity_label=config.TREND_GRANULARITY_LABELS.get(resolved, resolved),
        auto_detected=auto_detected,
        period_frame=period_frame,
        changes=changes,
        warnings=warnings,
        date_start=frame["_日期"].min(),
        date_end=frame["_日期"].max(),
        valid_rows=int(len(frame)),
        invalid_date_rows=invalid_rows,
        partial_periods=partial_labels,
        settings=limits,
    )


def trend_summary_dict(
    data: pd.DataFrame | None,
    *,
    granularity: str | None = None,
    settings: config.TrendSettings | None = None,
) -> dict[str, Any]:
    """一步得到 Agent / 接口用的趋势摘要字典。"""

    return build_trend_series(data, granularity=granularity, settings=settings).summary_dict()


# ---------------------------------------------------------------------------
# 内部实现：粒度与期间
# ---------------------------------------------------------------------------
def _resolve_granularity(
    dates: pd.Series, requested: str, limits: config.TrendSettings
) -> tuple[str, bool]:
    if requested in config.TREND_GRANULARITY_CHOICES:
        return requested, False

    # 用「去重后的自然日数量」而不是日期差：21 天的数据应判定为 3 周
    span = int(dates.dt.normalize().nunique()) if len(dates) else 0
    if span >= limits.auto_month_span_days:
        return config.TREND_GRANULARITY_MONTH, True
    if span >= limits.auto_week_span_days:
        return config.TREND_GRANULARITY_WEEK, True
    return config.TREND_GRANULARITY_DAY, True


def _period_columns(dates: pd.Series, granularity: str) -> pd.DataFrame:
    """返回与输入对齐的期间列：_期间键 / _期间 / _开始日期 / _结束日期。"""

    if granularity == config.TREND_GRANULARITY_MONTH:
        start = dates.dt.to_period("M").dt.start_time
        end = dates.dt.to_period("M").dt.end_time.dt.normalize()
        label = start.dt.strftime("%Y-%m")
    elif granularity == config.TREND_GRANULARITY_WEEK:
        start = (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.normalize()
        end = start + pd.Timedelta(days=6)
        label = start.dt.strftime("%Y-%m-%d") + " ~ " + end.dt.strftime("%m-%d")
    else:
        start = dates.dt.normalize()
        end = start
        label = start.dt.strftime("%Y-%m-%d")

    key = start.dt.strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "_期间键": key,
            "_期间": label,
            "_开始日期": start,
            "_结束日期": end,
        },
        index=dates.index,
    )


# ---------------------------------------------------------------------------
# 内部实现：变化计算
# ---------------------------------------------------------------------------
def _build_changes(
    period_frame: pd.DataFrame, granularity: str, limits: config.TrendSettings
) -> list[TrendChange]:
    changes: list[TrendChange] = []
    if period_frame.empty:
        return changes

    for definition in TREND_METRICS:
        name = definition.name
        if name not in period_frame.columns:
            continue
        series = period_frame[["期间", "开始日期", name]].copy()
        series = series.loc[series[name].notna()]
        change = TrendChange(
            metric_key=definition.key,
            metric_name=name,
            unit=definition.unit,
        )
        if len(series) < limits.min_periods_for_change:
            change.status = "数据不足"
            changes.append(change)
            continue

        values = [float(item) for item in series[name].tolist()]
        labels = [str(item) for item in series["期间"].tolist()]
        start_dates = [pd.Timestamp(item) for item in series["开始日期"].tolist()]

        change.latest_period = labels[-1]
        change.latest_value = values[-1]
        change.previous_period = labels[-2]
        change.previous_value = values[-2]
        change.change_absolute = values[-1] - values[-2]
        change.change_ratio = _ratio(values[-1], values[-2])
        change.direction = _direction_of_pair(values[-2], values[-1], limits)

        streak_direction, moves = _streak(values, limits)
        change.streak_direction = streak_direction
        change.streak_moves = moves
        minimum_moves = max(limits.min_periods_for_consecutive - 1, 1)
        change.is_consecutive_up = streak_direction == config.TREND_UP and moves >= minimum_moves
        change.is_consecutive_down = streak_direction == config.TREND_DOWN and moves >= minimum_moves

        # 同期变化（数据支持时）
        lagged = _same_period_start(start_dates[-1], granularity, limits)
        lookup = {
            pd.Timestamp(item): value for item, value in zip(start_dates, values)
        }
        if lagged is not None and pd.Timestamp(lagged) in lookup:
            change.same_period_value = lookup[pd.Timestamp(lagged)]
            change.same_period_ratio = _ratio(values[-1], change.same_period_value)
            change.same_period_label = SAME_PERIOD_LABELS[granularity] + "同期变化"
            change.same_period_note = ""
        else:
            change.same_period_note = (
                "数据不足以计算同期变化（缺少" + SAME_PERIOD_LABELS[granularity] + "同期数据）"
            )
        changes.append(change)
    return changes


SAME_PERIOD_LABELS: dict[str, str] = {
    config.TREND_GRANULARITY_DAY: "上周",
    config.TREND_GRANULARITY_WEEK: "上月",
    config.TREND_GRANULARITY_MONTH: "去年",
}


def _same_period_start(
    latest_start: pd.Timestamp, granularity: str, limits: config.TrendSettings
) -> pd.Timestamp | None:
    if granularity == config.TREND_GRANULARITY_MONTH:
        return (latest_start - pd.DateOffset(months=limits.same_period_lag_months)).normalize()
    if granularity == config.TREND_GRANULARITY_WEEK:
        return (latest_start - pd.Timedelta(weeks=limits.same_period_lag_weeks)).normalize()
    return (latest_start - pd.Timedelta(days=limits.same_period_lag_days)).normalize()


def _streak(values: Sequence[float], limits: config.TrendSettings) -> tuple[str, int]:
    """返回 (连续方向, 连续同向变动次数)。"""

    if len(values) < 2:
        return "", 0
    signs = [
        _sign(values[index] - values[index - 1], limits, base=values[index - 1])
        for index in range(1, len(values))
    ]
    last = signs[-1]
    if last == 0:
        return config.TREND_FLAT, 0
    count = 0
    for sign in reversed(signs):
        if sign == last:
            count += 1
        else:
            break
    return (config.TREND_UP if last > 0 else config.TREND_DOWN), count


def _sign(value: float, limits: config.TrendSettings, *, base: float | None = None) -> int:
    """带相对容差的符号判定：小幅波动（默认 < 0.01%）按「持平」处理。"""

    tolerance = abs(limits.flat_epsilon)
    if base is not None and np.isfinite(base):
        tolerance = max(tolerance, abs(float(base)) * abs(limits.flat_ratio_epsilon))
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _direction_of_pair(previous: float, latest: float, limits: config.TrendSettings) -> str:
    sign = _sign(latest - previous, limits, base=previous)
    if sign > 0:
        return config.TREND_UP
    if sign < 0:
        return config.TREND_DOWN
    return config.TREND_FLAT


def _ratio(latest: float, previous: float | None) -> float | None:
    if previous is None:
        return None
    if previous == 0:
        return None
    return (latest - previous) / abs(previous)


# ---------------------------------------------------------------------------
# 内部实现：取值工具
# ---------------------------------------------------------------------------
def _parse_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="D", origin="1899-12-30", errors="coerce")
    text = series.astype("string").str.strip()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            return pd.to_datetime(text, errors="coerce")
        except (ValueError, TypeError):
            return pd.to_datetime(text, errors="coerce", format="mixed")


def _numeric(frame: pd.DataFrame, column: str | None) -> np.ndarray:
    if column is None or column not in frame.columns:
        return np.full(len(frame), math.nan)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def _total(frame: pd.DataFrame, column: str | None) -> float:
    values = _numeric(frame, column)
    finite = values[np.isfinite(values)]
    return float(finite.sum()) if finite.size else 0.0


def _mean(frame: pd.DataFrame, column: str | None) -> float:
    values = _numeric(frame, column)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else math.nan


def _planned_values(frame: pd.DataFrame) -> np.ndarray:
    """计划运行时长（缺失或非正数时按机器人类型默认值兜底）。"""

    planned = _numeric(frame, config.COL_PLANNED_HOURS)
    invalid = ~np.isfinite(planned) | (planned <= 0)
    if bool(np.any(invalid)):
        if config.COL_ROBOT_TYPE in frame.columns:
            fallback = np.array(
                [
                    float(
                        config.PLANNED_HOURS_BY_TYPE.get(
                            str(value).strip(), config.DEFAULT_PLANNED_HOURS
                        )
                    )
                    for value in frame[config.COL_ROBOT_TYPE].tolist()
                ],
                dtype="float64",
            )
        else:
            fallback = np.full(len(frame), float(config.DEFAULT_PLANNED_HOURS))
        planned = np.where(invalid, fallback, planned)
    return planned


def _total_planned(frame: pd.DataFrame) -> float:
    values = _planned_values(frame)
    finite = values[np.isfinite(values)]
    return float(finite.sum()) if finite.size else 0.0


def _uptime_rates(frame: pd.DataFrame) -> np.ndarray:
    """逐条运行率（%）：优先使用清洗阶段的派生列，否则按计划时长兜底计算。"""

    if config.COL_UPTIME_RATE in frame.columns:
        values = _numeric(frame, config.COL_UPTIME_RATE)
        if bool(np.any(np.isfinite(values))):
            return values
    running = _numeric(frame, config.COL_RUNNING_HOURS)
    planned = _planned_values(frame)
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(planned > 0, running / planned * 100.0, math.nan)
    return rates.astype("float64")


def _downtime_total(frame: pd.DataFrame) -> float:
    """停机时长合计（计划缺口口径）：Σmax(计划运行时长 - 运行时长, 0)。"""

    running = _numeric(frame, config.COL_RUNNING_HOURS)
    planned = _planned_values(frame)
    valid = np.isfinite(running)
    if not bool(np.any(valid)):
        return 0.0
    gaps = np.where(valid, np.clip(planned - running, 0.0, None), 0.0)
    return float(np.nansum(gaps))


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    rounded = round(number, digits)
    return rounded + 0.0 if rounded == 0 else rounded


def _text(value: Any) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "-"
    if number.is_integer():
        return str(int(number))
    rounded = round(number, 2)
    return "0" if rounded == 0 else str(rounded)

