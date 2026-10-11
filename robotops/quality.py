"""RobotOps AI —— Phase 8 数据质量检查模块。

在正式分析之前，先对上传的 Excel / CSV 做一次数据质量体检：

    上传文件 -> 字段检查 -> 数据类型检查 -> 缺失值检查 -> 重复数据检查
             -> 异常值检查 -> 数据质量报告 -> 通过后进入数据分析

设计要点：
1. 本模块只做检查、不改数据；真正的清洗仍然由 data_cleaner 负责，
   两者的判定口径保持一致（数值容错、日期解析、去重键、业务边界都与清洗模块对齐）；
2. 任何异常输入都不会让程序崩溃：统一返回结构化的 DataQualityReport，
   由调用方决定是继续分析还是禁止分析；
3. 严重问题（缺少必需字段、关键字段整列为空、缺失/重复/非法值超过阈值）
   会把 status 记为 error，调用方据此禁止进入 Agent 分析；
4. 所有阈值集中定义在 config.QualityThresholds，不在本模块里硬编码。

报告结构（DataQualityReport.as_dict()）：

    {
        "status": "warning",
        "missing_values": 3,
        "duplicate_rows": 2,
        "invalid_values": 1,
        "warnings": [...],
        "errors": [...]
    }
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import pandas as pd

from . import config

# 检查项分类：决定问题计入报告的哪个计数器
CATEGORY_STRUCTURE = "structure"
CATEGORY_MISSING = "missing"
CATEGORY_DUPLICATE = "duplicate"
CATEGORY_INVALID = "invalid"

#: 状态 -> 界面图标
STATUS_ICON: dict[str, str] = {
    config.QUALITY_STATUS_PASS: "✓",
    config.QUALITY_STATUS_WARNING: "⚠",
    config.QUALITY_STATUS_ERROR: "✗",
}


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QualityCheck:
    """单项数据质量检查结果。"""

    key: str
    name: str
    level: str
    message: str
    affected_rows: int = 0
    category: str = CATEGORY_STRUCTURE
    rule: str = ""
    action: str = ""

    @property
    def is_problem(self) -> bool:
        return self.level != config.QUALITY_STATUS_PASS

    @property
    def icon(self) -> str:
        return STATUS_ICON.get(self.level, "•")

    def as_dict(self) -> dict[str, Any]:
        return {
            "检查项": self.name,
            "结果": config.QUALITY_STATUS_LABELS.get(self.level, self.level),
            "问题行数": self.affected_rows,
            "说明": self.message,
            "判定规则": self.rule,
            "建议动作": self.action,
        }


@dataclass
class DataQualityReport:
    """一次数据质量检查的完整结果。"""

    status: str
    checks: list[QualityCheck] = field(default_factory=list)
    missing_values: int = 0
    duplicate_rows: int = 0
    invalid_values: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    missing_required_columns: list[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    frame_name: str = ""
    thresholds: dict[str, float] = field(default_factory=dict)
    checked_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # -- 状态 ---------------------------------------------------------------
    @property
    def can_analyze(self) -> bool:
        """是否允许继续进入数据分析（只有 error 会禁止）。"""

        return self.status != config.QUALITY_STATUS_ERROR

    @property
    def status_label(self) -> str:
        return config.QUALITY_STATUS_LABELS.get(self.status, self.status)

    @property
    def is_pass(self) -> bool:
        return self.status == config.QUALITY_STATUS_PASS

    @property
    def problem_checks(self) -> list[QualityCheck]:
        return [check for check in self.checks if check.is_problem]

    @property
    def problem_count(self) -> int:
        return len(self.problem_checks)

    # -- 输出 ---------------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        """结构化报告（前端接口 / JSON 导出使用）。"""

        return {
            "status": self.status,
            "status_label": self.status_label,
            "can_analyze": self.can_analyze,
            "missing_values": int(self.missing_values),
            "duplicate_rows": int(self.duplicate_rows),
            "invalid_values": int(self.invalid_values),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "missing_required_columns": list(self.missing_required_columns),
            "row_count": int(self.row_count),
            "column_count": int(self.column_count),
            "frame_name": self.frame_name,
            "checked_at": self.checked_at,
            "checks": [check.as_dict() for check in self.checks],
        }

    def summary_lines(self) -> list[str]:
        """界面用的简短结论行（✓ / ⚠ / ✗ 开头）。"""

        lines: list[str] = []
        by_key = {check.key: check for check in self.checks}

        required = by_key.get("required_columns")
        if required is not None:
            if required.level == config.QUALITY_STATUS_PASS:
                lines.append("✓ 字段完整")
            else:
                lines.append("✗ " + required.message)
        elif self.missing_required_columns:
            lines.append("✗ 缺少必要字段：" + "、".join(self.missing_required_columns))

        dtype_check = by_key.get("dtype_summary")
        if dtype_check is not None:
            lines.append(
                ("✓ 数据类型正常" if dtype_check.level == config.QUALITY_STATUS_PASS
                 else "⚠ " + dtype_check.message)
            )

        date_check = by_key.get("date_valid")
        if date_check is not None and date_check.is_problem:
            lines.append("⚠ " + date_check.message)

        if self.missing_values:
            lines.append("⚠ 发现 " + str(self.missing_values) + " 条缺失数据")
        else:
            lines.append("✓ 无缺失数据")

        if self.duplicate_rows:
            lines.append("⚠ 发现 " + str(self.duplicate_rows) + " 条重复记录")
        else:
            lines.append("✓ 无重复记录")

        if self.invalid_values:
            lines.append("⚠ 发现 " + str(self.invalid_values) + " 条非法/越界值")
        else:
            lines.append("✓ 无非法数值")

        if self.status == config.QUALITY_STATUS_ERROR:
            lines.append("✗ 数据质量未通过，已禁止进入 Agent 分析")
        elif self.status == config.QUALITY_STATUS_WARNING:
            lines.append("⚠ 数据质量存在风险，可继续分析，但建议先修复上述问题")
        else:
            lines.append("✓ 数据质量检查通过，可以进入数据分析")

        return lines

    def summary_text(self) -> str:
        lines = ["【数据质量检查报告】", "  状态：" + self.status_label]
        lines.append(
            "  缺失值 " + str(self.missing_values)
            + " / 重复记录 " + str(self.duplicate_rows)
            + " / 非法值 " + str(self.invalid_values)
        )
        for check in self.checks:
            lines.append("  " + check.icon + " " + check.name + "：" + check.message)
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """检查明细表（报告导出 / 界面表格使用）。"""

        columns = ["检查项", "结果", "问题行数", "说明", "判定规则", "建议动作"]
        return pd.DataFrame([check.as_dict() for check in self.checks], columns=columns)

    def problem_frame(self) -> pd.DataFrame:
        """只包含存在问题的检查项。"""

        frame = self.to_frame()
        if frame.empty:
            return frame
        return frame.loc[frame["结果"] != config.QUALITY_STATUS_LABELS[config.QUALITY_STATUS_PASS]]


# ---------------------------------------------------------------------------
# 检查项清单（用于文档与界面）
# ---------------------------------------------------------------------------
QUALITY_CHECK_SPECS: tuple[dict[str, str], ...] = (
    {"key": "required_columns", "name": "必要字段是否存在", "severity": "严重问题",
     "rule": "必须包含 config.REQUIRED_COLUMNS 中的全部字段，缺失即禁止分析"},
    {"key": "date_valid", "name": "日期字段是否合法", "severity": "警告",
     "rule": "日期可解析为日期或 Excel 序列号，非法日期计入非法值"},
    {"key": "robot_id_not_blank", "name": "机器人ID是否为空", "severity": "警告",
     "rule": "机器人ID 为空或空白字符即计入问题"},
    {"key": "project_not_blank", "name": "项目名称是否为空", "severity": "警告",
     "rule": "项目名称为空或空白字符即计入问题"},
    {"key": "numeric_parseable", "name": "数值字段是否存在非法字符", "severity": "警告",
     "rule": "数值列含无法解析的文本（允许千分位、百分号、全角符号）"},
    {"key": "satisfaction_range", "name": "满意度是否超出 0～100", "severity": "警告",
     "rule": "用户满意度有效区间 0-100 分"},
    {"key": "running_hours_negative", "name": "运行时长是否小于0", "severity": "警告",
     "rule": "运行时长小于 0 小时不符合业务逻辑"},
    {"key": "fault_count_negative", "name": "故障次数是否小于0", "severity": "警告",
     "rule": "故障次数小于 0 次不符合业务逻辑"},
    {"key": "repair_count_negative", "name": "维修次数是否小于0", "severity": "警告",
     "rule": "维修次数小于 0 次不符合业务逻辑"},
    {"key": "cost_negative", "name": "成本是否小于0", "severity": "警告",
     "rule": "运营成本小于 0 元不符合业务逻辑"},
    {"key": "saving_rate_range", "name": "节降率是否处于合理范围", "severity": "警告",
     "rule": "节降率有效区间 -100% ～ 100%"},
    {"key": "duplicate_rows", "name": "是否存在重复记录", "severity": "警告 / 占比过高为严重问题",
     "rule": "完全重复记录 + 同一机器人同一天重复上报"},
    {"key": "missing_values", "name": "是否存在大量缺失值", "severity": "警告 / 占比过高为严重问题",
     "rule": "单列缺失率达到 QualityThresholds 阈值即升级"},
)


def quality_rules_frame() -> pd.DataFrame:
    """检查项清单表（写入文档或界面展示）。"""

    return pd.DataFrame(
        [
            {"检查项": spec["name"], "严重程度": spec["severity"], "判定规则": spec["rule"]}
            for spec in QUALITY_CHECK_SPECS
        ]
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def check_data_quality(
    frame: pd.DataFrame | None,
    *,
    thresholds: config.QualityThresholds | None = None,
    required_columns: Sequence[str] | None = None,
    frame_name: str = "",
) -> DataQualityReport:
    """对原始数据（未清洗）做数据质量检查。

    Args:
        frame: 原始数据（中文列名，可含脏数据）；允许为 None 或空表。
        thresholds: 质量阈值，默认 config.DEFAULT_QUALITY_THRESHOLDS。
        required_columns: 必需字段清单，默认 config.REQUIRED_COLUMNS。
        frame_name: 数据来源名称（用于报告展示）。

    Returns:
        DataQualityReport：无论数据多脏都会返回报告，不会抛异常。
    """

    limits = thresholds or config.DEFAULT_QUALITY_THRESHOLDS
    required = tuple(required_columns) if required_columns is not None else config.REQUIRED_COLUMNS
    checks: list[QualityCheck] = []
    missing_required: list[str] = []

    if frame is None or not isinstance(frame, pd.DataFrame) or frame.shape[0] == 0:
        checks.append(
            QualityCheck(
                key="required_columns",
                name="必要字段是否存在",
                level=config.QUALITY_STATUS_ERROR,
                message="数据表为空：没有任何可分析的数据行",
                category=CATEGORY_STRUCTURE,
                rule="数据表至少需要 1 行记录",
                action="请重新上传包含数据的文件",
            )
        )
        return _build_report(checks, missing_required, 0, 0, frame_name, limits)

    row_count = int(frame.shape[0])
    column_count = int(frame.shape[1])
    columns = [str(name) for name in frame.columns]

    # -- 1. 字段检查 -------------------------------------------------------
    missing_required = [name for name in required if name not in columns]
    if missing_required:
        checks.append(
            QualityCheck(
                key="required_columns",
                name="必要字段是否存在",
                level=config.QUALITY_STATUS_ERROR,
                message="缺少必要字段：" + "、".join(missing_required),
                affected_rows=row_count,
                category=CATEGORY_STRUCTURE,
                rule="必须包含：" + "、".join(required),
                action="请补齐字段后重新上传，本次禁止进入分析",
            )
        )
    else:
        checks.append(
            QualityCheck(
                key="required_columns",
                name="必要字段是否存在",
                level=config.QUALITY_STATUS_PASS,
                message="必要字段齐全（共 " + str(len(required)) + " 个）",
                rule="必须包含：" + "、".join(required),
            )
        )

    # -- 2. 数据类型检查（日期 + 数值）--------------------------------------
    date_invalid: dict[str, int] = {"blank": 0, "unparsable": 0}
    if config.COL_DATE in columns:
        date_invalid = _check_date_column(frame[config.COL_DATE])

    numeric_problems = _check_numeric_columns(frame, columns)
    unparsable_total = int(numeric_problems["total"])

    if not date_invalid["blank"] and not date_invalid["unparsable"] and not unparsable_total:
        dtype_level = config.QUALITY_STATUS_PASS
        dtype_message = "日期与数值字段类型正常"
    else:
        dtype_level = config.QUALITY_STATUS_WARNING
        parts: list[str] = []
        if date_invalid["unparsable"]:
            parts.append("日期无法解析 " + str(date_invalid["unparsable"]) + " 条")
        if date_invalid["blank"]:
            parts.append("日期为空 " + str(date_invalid["blank"]) + " 条")
        if unparsable_total:
            parts.append("数值无法解析 " + str(unparsable_total) + " 条")
        dtype_message = "类型检查发现问题：" + "、".join(parts)
    checks.append(
        QualityCheck(
            key="dtype_summary",
            name="数据类型检查",
            level=dtype_level,
            message=dtype_message,
            category=CATEGORY_STRUCTURE,
            rule="日期可解析；数值列允许千分位、百分号与全角符号",
            action="存在类型问题的值在清洗阶段会被置空或按兜底值处理",
        )
    )

    # -- 3. 日期字段是否合法 -----------------------------------------------
    if config.COL_DATE in columns:
        invalid_date = int(date_invalid["unparsable"])
        blank_date = int(date_invalid["blank"])
        level = config.QUALITY_STATUS_WARNING if (invalid_date or blank_date) else config.QUALITY_STATUS_PASS
        checks.append(
            QualityCheck(
                key="date_valid",
                name="日期字段是否合法",
                level=level,
                message=(
                    "日期合法（" + str(row_count) + " 条记录）" if level == config.QUALITY_STATUS_PASS
                    else "日期存在非法或空值：" + str(invalid_date) + " 条非法 / " + str(blank_date) + " 条为空"
                ),
                affected_rows=invalid_date + blank_date,
                category=CATEGORY_INVALID,
                rule="支持日期文本与 Excel 序列号（1900 日期系统）",
                action="非法日期记录会在清洗阶段被剔除",
            )
        )

    # -- 4 / 5. 关键字段是否为空 -------------------------------------------
    checks.append(
        _key_field_check(frame, config.COL_ROBOT_ID, "robot_id_not_blank", "机器人ID是否为空")
    )
    checks.append(
        _key_field_check(frame, config.COL_PROJECT, "project_not_blank", "项目名称是否为空")
    )

    # -- 6. 数值字段是否存在非法字符 ---------------------------------------
    checks.append(
        QualityCheck(
            key="numeric_parseable",
            name="数值字段是否存在非法字符",
            level=(
                config.QUALITY_STATUS_PASS if not unparsable_total else config.QUALITY_STATUS_WARNING
            ),
            message=(
                "数值字段均可解析"
                if not unparsable_total
                else "发现 " + str(unparsable_total) + " 处无法解析的数值内容："
                + "、".join(numeric_problems["samples"])
            ),
            affected_rows=unparsable_total,
            category=CATEGORY_INVALID,
            rule="允许千分位逗号、百分号、全角负号与小数点",
            action="无法解析的值在清洗阶段会被置为空值",
        )
    )

    # -- 7. 满意度区间 -----------------------------------------------------
    checks.append(
        _range_check(
            frame,
            column=config.COL_SATISFACTION,
            key="satisfaction_range",
            name="满意度是否超出 0～100",
            low=config.SATISFACTION_RANGE[0],
            high=config.SATISFACTION_RANGE[1],
            unit="分",
            action="越界值在清洗阶段会被置为空值，不计入平均满意度",
        )
    )

    # -- 8 / 9 / 10. 负数检查（运行时长 / 故障次数 / 维修次数）------------
    checks.append(
        _negative_check(
            frame, column=config.COL_RUNNING_HOURS, key="running_hours_negative",
            name="运行时长是否小于0", unit="小时",
        )
    )
    checks.append(
        _negative_check(
            frame, column=config.COL_FAULT_COUNT, key="fault_count_negative",
            name="故障次数是否小于0", unit="次",
        )
    )
    checks.append(
        _negative_check(
            frame, column=config.COL_REPAIR_COUNT, key="repair_count_negative",
            name="维修次数是否小于0", unit="次",
        )
    )

    # -- 11. 成本负数 ------------------------------------------------------
    checks.append(
        _negative_check(
            frame, column=config.COL_COST, key="cost_negative", name="成本是否小于0", unit="元",
        )
    )

    # -- 12. 节降率合理范围 ------------------------------------------------
    checks.append(
        _range_check(
            frame,
            column=config.COL_SAVING_RATE,
            key="saving_rate_range",
            name="节降率是否处于合理范围",
            low=config.COST_SAVING_RATE_RANGE[0],
            high=config.COST_SAVING_RATE_RANGE[1],
            unit="%",
            action="越界值在清洗阶段会被置为空值，不计入平均节降率",
        )
    )

    # -- 13. 重复记录 ------------------------------------------------------
    checks.append(_duplicate_check(frame, columns, limits, row_count))

    # -- 14. 大量缺失值 ----------------------------------------------------
    checks.append(_missing_check(frame, row_count, limits))

    return _build_report(checks, missing_required, row_count, column_count, frame_name, limits)


# ---------------------------------------------------------------------------
# 单项检查实现
# ---------------------------------------------------------------------------
def _blank_mask(series: pd.Series) -> pd.Series:
    """空值掩码：None / NaN / 空字符串 / 纯空白字符串。"""

    stripped = series.astype("string").str.strip()
    return series.isna() | stripped.fillna("").eq("")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """把文本数值转换为 float（与 data_cleaner 的容错规则保持一致）。"""

    text = series.astype("string")
    text = text.str.replace(",", "", regex=False).str.replace("，", "", regex=False)
    text = text.str.replace("%", "", regex=False)
    text = text.str.replace("－", "-", regex=False).str.replace("．", ".", regex=False)
    return pd.to_numeric(text.str.strip(), errors="coerce").astype("float64")


def _parse_dates(series: pd.Series) -> pd.Series:
    """解析日期列（与 data_cleaner 的容错规则保持一致）。"""

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


def _check_date_column(series: pd.Series) -> dict[str, int]:
    blank = int(_blank_mask(series).sum())
    try:
        parsed = _parse_dates(series)
        unparsable = int((parsed.isna() & ~_blank_mask(series)).sum())
    except Exception:  # noqa: BLE001 - 任何解析异常都按“无法解析”处理，不向外抛
        unparsable = int((~_blank_mask(series)).sum())
    return {"blank": blank, "unparsable": unparsable}


def _check_numeric_columns(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, dict[str, Any]]:
    """统计数值列的非法内容（含无法解析与全空）。"""

    total = 0
    samples: list[str] = []
    by_column: dict[str, int] = {}
    for name in config.NUMERIC_COLUMNS:
        if name not in columns:
            continue
        original = frame[name]
        blank = _blank_mask(original)
        try:
            converted = _coerce_numeric(original)
            failed = converted.isna() & ~blank
        except Exception:  # noqa: BLE001
            failed = ~blank
        count = int(failed.sum())
        if count:
            by_column[name] = count
            total += count
            for value in original[failed].astype("string").head(3).tolist():
                samples.append(str(value))
    return {"total": total, "by_column": by_column, "samples": samples[:5]}


def _key_field_check(frame: pd.DataFrame, column: str, key: str, name: str) -> QualityCheck:
    if column not in frame.columns:
        return QualityCheck(
            key=key,
            name=name,
            level=config.QUALITY_STATUS_ERROR,
            message="字段「" + column + "」不存在，无法校验",
            category=CATEGORY_STRUCTURE,
            rule="字段必须存在且非空",
            action="请补齐该字段后重新上传",
        )
    blank = int(_blank_mask(frame[column]).sum())
    total = int(len(frame))
    if blank == 0:
        return QualityCheck(
            key=key, name=name, level=config.QUALITY_STATUS_PASS,
            message="无空值（" + str(total) + " 条记录）", rule="字段必须存在且非空",
        )
    # 关键字段整列为空属于严重问题，禁止分析
    level = (
        config.QUALITY_STATUS_ERROR
        if blank == total
        else config.QUALITY_STATUS_WARNING
    )
    return QualityCheck(
        key=key,
        name=name,
        level=level,
        message="发现 " + str(blank) + " 条空值" + ("（整列为空）" if blank == total else ""),
        affected_rows=blank,
        category=CATEGORY_STRUCTURE,
        rule="字段必须存在且非空",
        action="空值记录在清洗阶段会被剔除；建议先补齐数据",
    )


def _range_check(
    frame: pd.DataFrame,
    *,
    column: str,
    key: str,
    name: str,
    low: float,
    high: float,
    unit: str,
    action: str,
) -> QualityCheck:
    rule = "有效区间 " + _number_text(low) + " ～ " + _number_text(high) + unit
    if column not in frame.columns:
        return QualityCheck(
            key=key, name=name, level=config.QUALITY_STATUS_ERROR,
            message="字段「" + column + "」不存在，无法校验", category=CATEGORY_INVALID,
            rule=rule, action="请补齐该字段后重新上传",
        )
    try:
        values = _coerce_numeric(frame[column])
        invalid = values.notna() & ((values < low) | (values > high))
        count = int(invalid.sum())
    except Exception:  # noqa: BLE001
        count = 0
    if count == 0:
        return QualityCheck(
            key=key, name=name, level=config.QUALITY_STATUS_PASS,
            message="全部在有效区间内", rule=rule,
        )
    samples = frame.loc[invalid, column].astype("string").head(3).tolist()
    return QualityCheck(
        key=key,
        name=name,
        level=config.QUALITY_STATUS_WARNING,
        message="「" + column + "」发现 " + str(count) + " 条越界值，示例：" + "、".join(map(str, samples)),
        affected_rows=count,
        category=CATEGORY_INVALID,
        rule=rule,
        action=action,
    )


def _negative_check(frame: pd.DataFrame, *, column: str, key: str, name: str, unit: str) -> QualityCheck:
    rule = "不允许小于 0" + unit
    if column not in frame.columns:
        return QualityCheck(
            key=key, name=name, level=config.QUALITY_STATUS_ERROR,
            message="字段「" + column + "」不存在，无法校验", category=CATEGORY_INVALID,
            rule=rule, action="请补齐该字段后重新上传",
        )
    try:
        values = _coerce_numeric(frame[column])
        negative = values.notna() & (values < 0)
        count = int(negative.sum())
    except Exception:  # noqa: BLE001
        count = 0
    if count == 0:
        return QualityCheck(
            key=key, name=name, level=config.QUALITY_STATUS_PASS,
            message="无负数", rule=rule,
        )
    samples = frame.loc[negative, column].astype("string").head(3).tolist()
    return QualityCheck(
        key=key,
        name=name,
        level=config.QUALITY_STATUS_WARNING,
        message="「" + column + "」发现 " + str(count) + " 条负数，示例：" + "、".join(map(str, samples)),
        affected_rows=count,
        category=CATEGORY_INVALID,
        rule=rule,
        action="负数在清洗阶段会被置为空值，不计入相关指标",
    )


def _duplicate_check(
    frame: pd.DataFrame,
    columns: Sequence[str],
    limits: config.QualityThresholds,
    row_count: int,
) -> QualityCheck:
    rule = "完全重复记录 + 同一机器人同一天重复上报（判定键：日期 + 机器人ID）"
    try:
        exact = int(frame.duplicated().sum())
        deduped = frame.loc[~frame.duplicated()]
        key_duplicates = 0
        if config.COL_DATE in columns and config.COL_ROBOT_ID in columns:
            key_duplicates = int(deduped.duplicated(subset=[config.COL_DATE, config.COL_ROBOT_ID]).sum())
    except Exception:  # noqa: BLE001 - 单元格含不可哈希值等极端情况，降级为警告而不是崩溃
        return QualityCheck(
            key="duplicate_rows",
            name="是否存在重复记录",
            level=config.QUALITY_STATUS_WARNING,
            message="重复记录检查未能完成：数据中存在无法比较的内容，请人工确认",
            category=CATEGORY_DUPLICATE,
            rule=rule,
            action="建议检查数据来源是否混入了非表格内容",
        )

    total_duplicates = exact + key_duplicates
    if total_duplicates == 0:
        return QualityCheck(
            key="duplicate_rows", name="是否存在重复记录", level=config.QUALITY_STATUS_PASS,
            message="无重复记录（" + str(row_count) + " 条记录）", rule=rule,
        )

    ratio = total_duplicates / row_count if row_count else 0.0
    level = (
        config.QUALITY_STATUS_ERROR if ratio >= limits.duplicate_ratio_error
        else config.QUALITY_STATUS_WARNING
    )
    message = (
        "发现 " + str(total_duplicates) + " 条重复记录（完全重复 "
        + str(exact) + " 条 / 同机器人同日重复 " + str(key_duplicates) + " 条），占比 "
        + _percent_text(ratio)
    )
    if level == config.QUALITY_STATUS_ERROR:
        message += "，已超过阈值 " + _percent_text(limits.duplicate_ratio_error) + "（禁止分析）"
    return QualityCheck(
        key="duplicate_rows",
        name="是否存在重复记录",
        level=level,
        message=message,
        affected_rows=total_duplicates,
        category=CATEGORY_DUPLICATE,
        rule=rule + "；占比 >= " + _percent_text(limits.duplicate_ratio_error) + " 记严重问题",
        action="清洗阶段会删除完全重复行，同机器人同一天只保留最后一条",
    )


def _missing_check(
    frame: pd.DataFrame, row_count: int, limits: config.QualityThresholds
) -> QualityCheck:
    rule = (
        "单列缺失率 >= " + _percent_text(limits.missing_ratio_warning) + " 记警告，>= "
        + _percent_text(limits.missing_ratio_error) + " 记严重问题"
    )
    counts: dict[str, int] = {}
    for name in config.ALL_COLUMNS:
        if name in frame.columns:
            try:
                counts[name] = int(_blank_mask(frame[name]).sum())
            except Exception:  # noqa: BLE001
                counts[name] = 0

    total_missing = int(sum(counts.values()))
    if not counts:
        return QualityCheck(
            key="missing_values", name="是否存在大量缺失值", level=config.QUALITY_STATUS_PASS,
            message="数据中没有可识别的标准字段，缺失值检查已跳过",
            affected_rows=0, category=CATEGORY_MISSING, rule=rule,
            action="请确认上传文件使用了标准字段名或受支持的别名",
        )
    if total_missing == 0:
        return QualityCheck(
            key="missing_values", name="是否存在大量缺失值", level=config.QUALITY_STATUS_PASS,
            message="关键字段无缺失值（" + str(row_count) + " 条记录）",
            affected_rows=0, category=CATEGORY_MISSING, rule=rule,
        )

    worst_column = max(counts, key=lambda name: counts[name])
    worst_count = counts[worst_column]
    worst_ratio = worst_count / row_count if row_count else 0.0
    over_ratio = worst_ratio >= limits.missing_ratio_error
    enough_rows = worst_count >= limits.missing_min_rows_for_error
    if over_ratio and enough_rows:
        level = config.QUALITY_STATUS_ERROR
    elif worst_ratio >= limits.missing_ratio_warning:
        level = config.QUALITY_STATUS_WARNING
    else:
        level = config.QUALITY_STATUS_PASS
    message = (
        "发现 " + str(total_missing) + " 条缺失数据；缺失最多的字段「" + worst_column + "」缺 "
        + str(worst_count) + " 条（" + _percent_text(worst_ratio) + "）"
    )
    if level == config.QUALITY_STATUS_ERROR:
        message += "，已超过严重阈值 " + _percent_text(limits.missing_ratio_error) + "（禁止分析）"
    elif over_ratio:
        message += "，比例虽高但缺失条数少于 " + str(limits.missing_min_rows_for_error) + " 条，按警告处理"
    return QualityCheck(
        key="missing_values",
        name="是否存在大量缺失值",
        level=level,
        message=message,
        affected_rows=total_missing,
        category=CATEGORY_MISSING,
        rule=rule,
        action="缺失值不会让程序崩溃：清洗阶段按字段规则置空或填 0，并在清洗问题清单中留痕",
    )


# ---------------------------------------------------------------------------
# 报告组装
# ---------------------------------------------------------------------------
def _build_report(
    checks: list[QualityCheck],
    missing_required: list[str],
    row_count: int,
    column_count: int,
    frame_name: str,
    limits: config.QualityThresholds,
) -> DataQualityReport:
    if any(check.level == config.QUALITY_STATUS_ERROR for check in checks):
        status = config.QUALITY_STATUS_ERROR
    elif any(check.level == config.QUALITY_STATUS_WARNING for check in checks):
        status = config.QUALITY_STATUS_WARNING
    else:
        status = config.QUALITY_STATUS_PASS

    def _sum(category: str) -> int:
        return int(sum(check.affected_rows for check in checks if check.category == category))

    return DataQualityReport(
        status=status,
        checks=checks,
        missing_values=_sum(CATEGORY_MISSING),
        duplicate_rows=_sum(CATEGORY_DUPLICATE),
        invalid_values=_sum(CATEGORY_INVALID),
        warnings=[check.message for check in checks if check.level == config.QUALITY_STATUS_WARNING],
        errors=[check.message for check in checks if check.level == config.QUALITY_STATUS_ERROR],
        missing_required_columns=list(missing_required),
        row_count=row_count,
        column_count=column_count,
        frame_name=frame_name,
        thresholds=limits.as_dict(),
    )


def _number_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _percent_text(ratio: float) -> str:
    return str(round(float(ratio) * 100.0, 2)) + "%"

