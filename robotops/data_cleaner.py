"""RobotOps AI —— Phase 1 数据清洗模块。

清洗流程（顺序固定，便于复现与排错）：

1. 规范表头并校验必需列；
2. 删除全空行 / 全空列；
3. 文本列去首尾空白、合并连续空格；
4. 日期解析（支持日期文本、Excel 序列号），无法解析的记录剔除；
5. 数值列类型转换（支持带千分位逗号、百分号、全角符号的文本）；
6. 业务边界校验（负运行时长、负成本、满意度越界等）；
7. 计数类缺失值填充为 0，并记录问题；
8. 去重（完全重复 + 同一机器人同一天重复上报，保留最后一条）；
9. 计算派生列：运行率、故障率；
10. 按日期、机器人ID 排序。

每一步产生的问题都会进入 ``CleaningResult.issues``，便于人工复核。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .data_loader import normalize_headers
from .exceptions import DataValidationError

ISSUE_COLUMNS: tuple[str, ...] = (
    "序号",
    "级别",
    "环节",
    "问题类型",
    "问题描述",
    "涉及行数",
    "处理方式",
)

LEVEL_ERROR = "错误"
LEVEL_WARNING = "警告"
LEVEL_INFO = "提示"


@dataclass
class CleaningResult:
    """清洗结果。"""

    data: pd.DataFrame
    issues: pd.DataFrame
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return int(len(self.data))

    @property
    def issue_count(self) -> int:
        return int(len(self.issues))

    def summary_text(self) -> str:
        lines = ["【数据清洗摘要】"]
        for key, value in self.stats.items():
            lines.append(f"  - {key}：{value}")
        return "\n".join(lines)


@dataclass
class IssueCollector:
    """清洗问题收集器。"""

    items: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        level: str,
        stage: str,
        kind: str,
        description: str,
        *,
        rows: int = 0,
        action: str = "已保留",
    ) -> None:
        self.items.append(
            {
                "级别": level,
                "环节": stage,
                "问题类型": kind,
                "问题描述": description,
                "涉及行数": int(rows),
                "处理方式": action,
            }
        )

    def level_count(self, level: str) -> int:
        return sum(1 for item in self.items if item["级别"] == level)

    def to_frame(self) -> pd.DataFrame:
        if not self.items:
            return pd.DataFrame(columns=list(ISSUE_COLUMNS))
        frame = pd.DataFrame(self.items)
        frame.insert(0, "序号", range(1, len(frame) + 1))
        return frame[list(ISSUE_COLUMNS)]


def clean_operation_data(
    raw: pd.DataFrame,
    *,
    planned_hours_default: float | None = None,
) -> CleaningResult:
    """清洗机器人运营原始数据。

    Args:
        raw: 原始数据（中文列名，可包含脏数据）。
        planned_hours_default: 计划运行时长兜底值，默认取配置值（24 小时）。

    Returns:
        ``CleaningResult``：``data`` 为清洗后的数据，``issues`` 为问题明细。
    """

    default_planned = (
        float(planned_hours_default)
        if planned_hours_default is not None
        else float(config.DEFAULT_PLANNED_HOURS)
    )
    issues = IssueCollector()

    frame = raw.copy(deep=True)
    raw_count = len(frame)

    frame = _normalize_columns(frame, issues)
    frame = _drop_empty_rows_and_columns(frame, issues)
    frame = _clean_text_columns(frame, issues)
    frame = _parse_date_column(frame, issues)
    frame = _coerce_numeric_columns(frame, issues)
    frame = _apply_business_bounds(frame, issues)
    frame = _fill_planned_hours(frame, issues, default_planned=default_planned)
    frame = _fill_missing_counts(frame, issues)
    frame = _deduplicate(frame, issues)
    _log_missing_metric_values(frame, issues)

    frame = frame.loc[:, [name for name in config.ALL_COLUMNS if name in frame.columns]]
    if frame.empty:
        raise DataValidationError(
            "清洗后没有可用记录（记录可能因日期无法解析、关键字段为空或重复上报被剔除），请检查数据源。"
        )
    frame = _add_derived_columns(frame, issues)
    frame = frame.sort_values([config.COL_DATE, config.COL_ROBOT_ID], kind="stable")
    frame = frame.reset_index(drop=True)

    stats = _build_stats(frame, raw_count, issues)
    return CleaningResult(data=frame, issues=issues.to_frame(), stats=stats)


# ---------------------------------------------------------------------------
# 各清洗步骤
# ---------------------------------------------------------------------------
def _normalize_columns(frame: pd.DataFrame, issues: IssueCollector) -> pd.DataFrame:
    frame, renamed = normalize_headers(frame)
    if renamed:
        issues.add(
            LEVEL_INFO,
            "表头规范",
            "列名规范化",
            "；".join(f"{old} -> {new}" for old, new in renamed.items()),
            rows=0,
            action="已按标准列名/别名映射",
        )

    missing = [name for name in config.REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise DataValidationError(
            f"数据缺少必需列：{'、'.join(missing)}；实际列名：{'、'.join(map(str, frame.columns))}"
        )
    return frame


def _drop_empty_rows_and_columns(frame: pd.DataFrame, issues: IssueCollector) -> pd.DataFrame:
    before_rows, before_cols = frame.shape

    empty_row_mask = frame.isna().all(axis=1)
    if bool(empty_row_mask.any()):
        issues.add(
            LEVEL_WARNING,
            "结构清理",
            "空行",
            "存在完全为空的数据行",
            rows=int(empty_row_mask.sum()),
            action="已删除",
        )
        frame = frame.loc[~empty_row_mask].copy()

    blank_columns: list[str] = []
    empty_model_columns: list[str] = []
    for name in frame.columns:
        column = frame[name]
        is_blank = bool(column.notna().sum() == 0)
        if not is_blank and not pd.api.types.is_numeric_dtype(column):
            is_blank = bool(column.astype("string").fillna("").str.strip().eq("").all())
        if not is_blank:
            continue
        if name in config.ALL_COLUMNS:
            empty_model_columns.append(name)
        else:
            blank_columns.append(name)

    if blank_columns:
        issues.add(
            LEVEL_INFO,
            "结构清理",
            "空列",
            f"存在全空列：{'、'.join(map(str, blank_columns))}",
            rows=0,
            action="已删除该列",
        )
        frame = frame.drop(columns=blank_columns)

    if empty_model_columns:
        issues.add(
            LEVEL_ERROR,
            "结构清理",
            "标准字段整列为空",
            f"以下字段没有任何有效值：{'、'.join(empty_model_columns)}，"
            f"依赖这些字段的指标将返回空值，请补充数据源",
            rows=len(frame),
            action="已保留该列，相关指标按空值处理",
        )

    if frame.shape != (before_rows, before_cols):
        frame = frame.reset_index(drop=True)
    return frame


def _clean_text_columns(frame: pd.DataFrame, issues: IssueCollector) -> pd.DataFrame:
    for name in config.TEXT_COLUMNS:
        if name not in frame.columns:
            continue
        original = frame[name].astype("string")
        cleaned = original.str.replace("\u3000", " ", regex=False).str.replace(
            r"\s+", " ", regex=True
        ).str.strip()
        changed = int((cleaned.fillna("") != original.fillna("")).sum())
        if changed:
            issues.add(
                LEVEL_INFO,
                "文本清洗",
                "空白字符",
                f"「{name}」存在首尾空格或多余空白字符",
                rows=changed,
                action="已去除首尾及重复空白",
            )
        frame[name] = cleaned

    for name in (config.COL_PROJECT, config.COL_ROBOT_ID):
        if name not in frame.columns:
            continue
        invalid = frame[name].isna() | frame[name].eq("")
        if bool(invalid.any()):
            issues.add(
                LEVEL_ERROR,
                "关键字段",
                "关键字段缺失",
                f"「{name}」为空，无法归属项目/机器人",
                rows=int(invalid.sum()),
                action="已剔除该记录",
            )
            frame = frame.loc[~invalid].copy()
    return frame.reset_index(drop=True)


def _parse_date_column(frame: pd.DataFrame, issues: IssueCollector) -> pd.DataFrame:
    source = frame[config.COL_DATE]

    if pd.api.types.is_datetime64_any_dtype(source):
        parsed = source
    elif pd.api.types.is_numeric_dtype(source):
        issues.add(
            LEVEL_WARNING,
            "日期解析",
            "日期格式",
            "「日期」列为数字，按 Excel 序列号（1900 日期系统）解析",
            rows=int(source.notna().sum()),
            action="已转换为日期",
        )
        parsed = pd.to_datetime(source, unit="D", origin="1899-12-30", errors="coerce")
    else:
        text = source.astype("string").str.strip()
        with warnings.catch_warnings():
            # 允许逐元素推断日期格式，同时避免 pandas 逐格解析的告警刷屏
            warnings.simplefilter("ignore", UserWarning)
            try:
                parsed = pd.to_datetime(text, errors="coerce")
            except (ValueError, TypeError):
                parsed = pd.to_datetime(text, errors="coerce", format="mixed")

    invalid = parsed.isna()
    if bool(invalid.any()):
        samples = source[invalid].astype("string").head(3).tolist()
        issues.add(
            LEVEL_ERROR,
            "日期解析",
            "日期无法解析",
            f"示例值：{'、'.join(map(str, samples))}",
            rows=int(invalid.sum()),
            action="已剔除该记录",
        )
        frame = frame.loc[~invalid].copy()
        parsed = parsed.loc[~invalid]

    frame[config.COL_DATE] = parsed.dt.normalize()
    return frame.reset_index(drop=True)


def _to_numeric(series: pd.Series) -> pd.Series:
    """把文本数值（含千分位、百分号、全角符号）转换为 float。"""

    text = series.astype("string")
    text = text.str.replace(",", "", regex=False).str.replace("，", "", regex=False)
    text = text.str.replace("%", "", regex=False)
    text = text.str.replace("－", "-", regex=False).str.replace("．", ".", regex=False)
    return pd.to_numeric(text.str.strip(), errors="coerce").astype("float64")


def _coerce_numeric_columns(frame: pd.DataFrame, issues: IssueCollector) -> pd.DataFrame:
    for name in config.NUMERIC_COLUMNS:
        if name not in frame.columns:
            continue
        original = frame[name]
        converted = _to_numeric(original)
        failed = converted.isna() & original.notna() & original.astype("string").str.strip().ne("")
        if bool(failed.any()):
            samples = original[failed].astype("string").head(3).tolist()
            issues.add(
                LEVEL_WARNING,
                "类型转换",
                "非数值内容",
                f"「{name}」存在无法解析的值，示例：{'、'.join(map(str, samples))}",
                rows=int(failed.sum()),
                action="已置为空值",
            )
        frame[name] = converted
    return frame


def _apply_business_bounds(frame: pd.DataFrame, issues: IssueCollector) -> pd.DataFrame:
    # 运行时长：负数视为无效；超过 24 小时记录提示但保留
    negative_hours = frame[config.COL_RUNNING_HOURS] < 0
    if bool(negative_hours.any()):
        issues.add(
            LEVEL_ERROR,
            "业务校验",
            "运行时长为负",
            "运行时长小于 0 小时，不符合业务逻辑",
            rows=int(negative_hours.sum()),
            action="已置为空值",
        )
        frame.loc[negative_hours, config.COL_RUNNING_HOURS] = np.nan

    over_hours = frame[config.COL_RUNNING_HOURS] > config.MAX_RUNNING_HOURS_PER_DAY
    if bool(over_hours.any()):
        issues.add(
            LEVEL_WARNING,
            "业务校验",
            "运行时长超 24 小时",
            "单日运行时长超过 24 小时，请确认是否为累计口径",
            rows=int(over_hours.sum()),
            action="已保留，请人工复核",
        )

    # 计数类字段：负数无效
    for name in config.COUNT_COLUMNS:
        negative = frame[name] < 0
        if bool(negative.any()):
            issues.add(
                LEVEL_ERROR,
                "业务校验",
                f"{name}为负",
                f"「{name}」出现负数，不符合业务逻辑",
                rows=int(negative.sum()),
                action="已置为空值",
            )
            frame.loc[negative, name] = np.nan

    # 满意度区间
    low, high = config.SATISFACTION_RANGE
    invalid_satisfaction = frame[config.COL_SATISFACTION].notna() & (
        (frame[config.COL_SATISFACTION] < low) | (frame[config.COL_SATISFACTION] > high)
    )
    if bool(invalid_satisfaction.any()):
        samples = frame.loc[invalid_satisfaction, config.COL_SATISFACTION].head(3).tolist()
        issues.add(
            LEVEL_WARNING,
            "业务校验",
            "满意度越界",
            f"「用户满意度」应在 {low:g}-{high:g} 之间，越界示例：{'、'.join(map(str, samples))}",
            rows=int(invalid_satisfaction.sum()),
            action="已置为空值，不计入平均满意度",
        )
        frame.loc[invalid_satisfaction, config.COL_SATISFACTION] = np.nan

    # 运营成本
    negative_cost = frame[config.COL_COST] < 0
    if bool(negative_cost.any()):
        issues.add(
            LEVEL_WARNING,
            "业务校验",
            "运营成本为负",
            "「运营成本」出现负数，已置为空值",
            rows=int(negative_cost.sum()),
            action="已置为空值，不计入总成本",
        )
        frame.loc[negative_cost, config.COL_COST] = np.nan

    # 节降率区间
    low, high = config.COST_SAVING_RATE_RANGE
    invalid_saving = frame[config.COL_SAVING_RATE].notna() & (
        (frame[config.COL_SAVING_RATE] < low) | (frame[config.COL_SAVING_RATE] > high)
    )
    if bool(invalid_saving.any()):
        samples = frame.loc[invalid_saving, config.COL_SAVING_RATE].head(3).tolist()
        issues.add(
            LEVEL_WARNING,
            "业务校验",
            "节降率越界",
            f"「节降率」应在 {low:g}%-{high:g}% 之间，越界示例：{'、'.join(map(str, samples))}",
            rows=int(invalid_saving.sum()),
            action="已置为空值，不计入平均节降率",
        )
        frame.loc[invalid_saving, config.COL_SAVING_RATE] = np.nan

    return frame


def _fill_planned_hours(
    frame: pd.DataFrame, issues: IssueCollector, *, default_planned: float
) -> pd.DataFrame:
    """补齐「计划运行时长」，保证运行率可计算。"""

    type_defaults = frame[config.COL_ROBOT_TYPE].map(
        lambda value: config.PLANNED_HOURS_BY_TYPE.get(
            str(value).strip(), default_planned
        )
    )

    if config.COL_PLANNED_HOURS not in frame.columns:
        issues.add(
            LEVEL_INFO,
            "派生字段",
            "缺少计划运行时长",
            "原始数据未提供「计划运行时长」，按机器人类型默认值兜底",
            rows=len(frame),
            action="已按机器人类型默认值补齐",
        )
        frame[config.COL_PLANNED_HOURS] = type_defaults
        return frame

    invalid = frame[config.COL_PLANNED_HOURS].isna() | (frame[config.COL_PLANNED_HOURS] <= 0)
    if bool(invalid.any()):
        issues.add(
            LEVEL_WARNING,
            "业务校验",
            "计划运行时长缺失或非正数",
            "计划运行时长缺失/小于等于 0，无法直接计算运行率",
            rows=int(invalid.sum()),
            action="已按机器人类型默认值补齐",
        )
        frame.loc[invalid, config.COL_PLANNED_HOURS] = type_defaults[invalid]
    return frame


def _fill_missing_counts(frame: pd.DataFrame, issues: IssueCollector) -> pd.DataFrame:
    for name in config.COUNT_COLUMNS:
        missing = frame[name].isna()
        if bool(missing.any()):
            issues.add(
                LEVEL_WARNING,
                "缺失值处理",
                "计数字段缺失",
                f"「{name}」存在缺失值，按 0 次处理",
                rows=int(missing.sum()),
                action="已填充为 0",
            )
            frame.loc[missing, name] = 0.0
        frame[name] = frame[name].astype("float64")
    return frame


def _deduplicate(frame: pd.DataFrame, issues: IssueCollector) -> pd.DataFrame:
    exact_duplicates = int(frame.duplicated().sum())
    if exact_duplicates:
        issues.add(
            LEVEL_WARNING,
            "去重",
            "完全重复记录",
            "存在所有字段完全相同的记录",
            rows=exact_duplicates,
            action="已删除重复行",
        )
        frame = frame.drop_duplicates()

    key_columns = [config.COL_DATE, config.COL_ROBOT_ID]
    key_duplicates = int(frame.duplicated(subset=key_columns).sum())
    if key_duplicates:
        issues.add(
            LEVEL_WARNING,
            "去重",
            "同一机器人同一天重复上报",
            f"按「{' + '.join(key_columns)}」判重，一天只保留一条记录",
            rows=key_duplicates,
            action="已保留最后一条记录",
        )
        frame = frame.drop_duplicates(subset=key_columns, keep="last")

    return frame.reset_index(drop=True)


def _log_missing_metric_values(frame: pd.DataFrame, issues: IssueCollector) -> None:
    """记录参与指标计算的关键字段缺失情况（缺失值不会被填充，只跳过计算）。"""

    tracked = (
        config.COL_SATISFACTION,
        config.COL_RUNNING_HOURS,
        config.COL_COST,
        config.COL_SAVING_RATE,
    )
    missing = {
        name: int(frame[name].isna().sum())
        for name in tracked
        if name in frame.columns and int(frame[name].isna().sum()) > 0
    }
    if not missing:
        return

    issues.add(
        LEVEL_WARNING,
        "缺失值处理",
        "关键指标缺失",
        "以下字段存在缺失值：" + "；".join(f"{name} {count} 条" for name, count in missing.items()),
        rows=sum(missing.values()),
        action="保留记录，相关指标跳过空值",
    )


def _add_derived_columns(frame: pd.DataFrame, issues: IssueCollector) -> pd.DataFrame:
    running = frame[config.COL_RUNNING_HOURS].to_numpy(dtype="float64")
    planned = frame[config.COL_PLANNED_HOURS].to_numpy(dtype="float64")
    faults = frame[config.COL_FAULT_COUNT].to_numpy(dtype="float64")
    inspections = frame[config.COL_INSPECTION_COUNT].to_numpy(dtype="float64")

    uptime = np.full(len(frame), np.nan, dtype="float64")
    valid_uptime = np.isfinite(running) & np.isfinite(planned) & (planned > 0)
    uptime[valid_uptime] = running[valid_uptime] / planned[valid_uptime] * 100.0

    fault_rate = np.full(len(frame), np.nan, dtype="float64")
    valid_fault = np.isfinite(faults) & np.isfinite(inspections) & (inspections > 0)
    fault_rate[valid_fault] = faults[valid_fault] / inspections[valid_fault] * 100.0

    frame[config.COL_UPTIME_RATE] = np.round(uptime, 4)
    frame[config.COL_FAULT_RATE] = np.round(fault_rate, 4)

    missing_uptime = int(np.isnan(uptime).sum())
    if missing_uptime:
        issues.add(
            LEVEL_WARNING,
            "派生字段",
            "运行率无法计算",
            "运行时长或计划运行时长缺失，运行率为空",
            rows=missing_uptime,
            action="保留空值，不参与均值计算",
        )

    zero_inspection = int((np.isfinite(inspections) & (inspections <= 0)).sum())
    if zero_inspection:
        issues.add(
            LEVEL_WARNING,
            "派生字段",
            "故障率无法计算",
            "巡检次数为 0，故障率无分母",
            rows=zero_inspection,
            action="保留空值，不参与故障率计算",
        )
    return frame


def _build_stats(
    frame: pd.DataFrame, raw_count: int, issues: IssueCollector
) -> dict[str, Any]:
    cleaned = len(frame)
    removed = raw_count - cleaned
    date_range = "-"
    if cleaned:
        start = frame[config.COL_DATE].min()
        end = frame[config.COL_DATE].max()
        date_range = f"{start:%Y-%m-%d} 至 {end:%Y-%m-%d}"

    return {
        "原始记录数": raw_count,
        "清洗后记录数": cleaned,
        "剔除记录数": removed,
        "剔除占比(%)": round(removed / raw_count * 100, 2) if raw_count else 0.0,
        "问题条目数": len(issues.items),
        f"其中{LEVEL_ERROR}": issues.level_count(LEVEL_ERROR),
        f"其中{LEVEL_WARNING}": issues.level_count(LEVEL_WARNING),
        f"其中{LEVEL_INFO}": issues.level_count(LEVEL_INFO),
        "日期范围": date_range,
        "项目数": int(frame[config.COL_PROJECT].nunique()) if cleaned else 0,
        "机器人数": int(frame[config.COL_ROBOT_ID].nunique()) if cleaned else 0,
        "满意度缺失记录数": int(frame[config.COL_SATISFACTION].isna().sum()) if cleaned else 0,
        "运行时长缺失记录数": int(frame[config.COL_RUNNING_HOURS].isna().sum()) if cleaned else 0,
        "运营成本缺失记录数": int(frame[config.COL_COST].isna().sum()) if cleaned else 0,
        "节降率缺失记录数": int(frame[config.COL_SAVING_RATE].isna().sum()) if cleaned else 0,
        "运行率为空记录数": int(frame[config.COL_UPTIME_RATE].isna().sum()) if cleaned else 0,
        "故障率为空记录数": int(frame[config.COL_FAULT_RATE].isna().sum()) if cleaned else 0,
    }
