"""RobotOps AI —— Phase 1 数据模型。

本模块定义机器人运营数据的“单一事实来源”：
1. ``ColumnSpec`` / ``COLUMN_SPECS``：字段字典（列名、类型、单位、说明）；
2. ``RobotOperationRecord``：单条机器人运营记录（一台机器人一天一条）；
3. ``MetricsSummary``：核心指标结果；
4. ``AnomalyRecord``：异常明细记录。

说明：
- 数据表使用中文列名，便于与业务方直接交换文件；
- Python 内部对象使用英文属性名，通过 ``FIELD_TO_COLUMN`` 做映射；
- ``运行率``、``故障率`` 是清洗阶段计算出的派生列，不属于原始数据。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Iterable, Mapping

import pandas as pd

from . import config
from .config import (
    COL_COST,
    COL_DATE,
    COL_FAULT_COUNT,
    COL_FAULT_RATE,
    COL_INSPECTION_COUNT,
    COL_PLANNED_HOURS,
    COL_PROJECT,
    COL_REPAIR_COUNT,
    COL_ROBOT_ID,
    COL_ROBOT_TYPE,
    COL_RUNNING_HOURS,
    COL_SATISFACTION,
    COL_SAVING_RATE,
    COL_UPTIME_RATE,
)


# ---------------------------------------------------------------------------
# 1. 机器人类型枚举
# ---------------------------------------------------------------------------
class RobotType(str, Enum):
    """Phase 1 支持的机器人类型（未知类型按“其他”处理）。"""

    CLEANING = "清洁机器人"
    INSPECTION = "巡检机器人"
    SECURITY_PATROL = "安防巡逻机器人"
    DELIVERY = "配送机器人"
    DISINFECTION = "消杀机器人"
    AGV = "AGV搬运机器人"
    OTHER = "其他"

    @classmethod
    def values(cls) -> list[str]:
        return [item.value for item in cls]

    @classmethod
    def coerce(cls, value: Any) -> "RobotType":
        """把任意输入值转换为枚举成员，无法识别时返回 ``OTHER``。"""

        if isinstance(value, RobotType):
            return value
        text = "" if value is None else str(value).strip()
        for item in cls:
            if item.value == text:
                return item
        return cls.OTHER

    @property
    def planned_hours(self) -> float:
        """该类型机器人的每日计划运行时长（小时）。"""

        return config.PLANNED_HOURS_BY_TYPE.get(self.value, config.DEFAULT_PLANNED_HOURS)


# ---------------------------------------------------------------------------
# 2. 字段字典
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnSpec:
    """单个数据列的定义。"""

    name: str
    dtype: str  # date / text / int / float
    unit: str
    required: bool
    description: str
    example: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "字段名": self.name,
            "数据类型": {
                "date": "日期",
                "text": "文本",
                "int": "整数",
                "float": "数值",
            }.get(self.dtype, self.dtype),
            "单位": self.unit,
            "是否必需": "必需" if self.required else "可选（缺失自动兜底）",
            "字段说明": self.description,
            "示例值": self.example,
        }


COLUMN_SPECS: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        name=COL_DATE,
        dtype="date",
        unit="",
        required=True,
        description="运营数据日期，每台机器人每天一条记录",
        example="2026-08-01",
    ),
    ColumnSpec(
        name=COL_PROJECT,
        dtype="text",
        unit="",
        required=True,
        description="项目名称（指标中的“项目数量”按其去重统计）",
        example="华东-苏州工业园区清洁项目",
    ),
    ColumnSpec(
        name=COL_ROBOT_ID,
        dtype="text",
        unit="",
        required=True,
        description="机器人唯一编号（指标中的“机器人数量”按其去重统计）",
        example="SZ-CLN-01",
    ),
    ColumnSpec(
        name=COL_ROBOT_TYPE,
        dtype="text",
        unit="",
        required=True,
        description=f"机器人类型，取值参考：{'、'.join(config.PLANNED_HOURS_BY_TYPE)}",
        example="清洁机器人",
    ),
    ColumnSpec(
        name=COL_RUNNING_HOURS,
        dtype="float",
        unit="小时",
        required=True,
        description="当日实际运行时长",
        example="7.6",
    ),
    ColumnSpec(
        name=COL_PLANNED_HOURS,
        dtype="float",
        unit="小时",
        required=False,
        description=(
            "当日计划运行时长，用于计算运行率 = 运行时长 / 计划运行时长 × 100%；"
            f"缺失时按 {config.DEFAULT_PLANNED_HOURS:g} 小时兜底"
        ),
        example="8.0",
    ),
    ColumnSpec(
        name=COL_FAULT_COUNT,
        dtype="int",
        unit="次",
        required=True,
        description="当日故障发生次数",
        example="1",
    ),
    ColumnSpec(
        name=COL_INSPECTION_COUNT,
        dtype="int",
        unit="次",
        required=True,
        description="当日巡检（作业任务）完成次数，作为故障率分母",
        example="6",
    ),
    ColumnSpec(
        name=COL_REPAIR_COUNT,
        dtype="int",
        unit="次",
        required=True,
        description="当日维修次数",
        example="0",
    ),
    ColumnSpec(
        name=COL_SATISFACTION,
        dtype="float",
        unit="分",
        required=True,
        description=f"用户满意度，有效区间 {config.SATISFACTION_RANGE[0]:g}-{config.SATISFACTION_RANGE[1]:g}",
        example="93.5",
    ),
    ColumnSpec(
        name=COL_COST,
        dtype="float",
        unit="元",
        required=True,
        description="当日运营成本",
        example="1180.50",
    ),
    ColumnSpec(
        name=COL_SAVING_RATE,
        dtype="float",
        unit="%",
        required=True,
        description="节降率（成本节降比例）",
        example="18.2",
    ),
)


def data_dictionary_frame() -> pd.DataFrame:
    """返回字段字典 DataFrame，用于写入演示数据文件或文档。"""

    return pd.DataFrame([spec.as_dict() for spec in COLUMN_SPECS])


# ---------------------------------------------------------------------------
# 3. 单条运营记录模型
# ---------------------------------------------------------------------------
FIELD_TO_COLUMN: dict[str, str] = {
    "operation_date": COL_DATE,
    "project_name": COL_PROJECT,
    "robot_id": COL_ROBOT_ID,
    "robot_type": COL_ROBOT_TYPE,
    "running_hours": COL_RUNNING_HOURS,
    "planned_hours": COL_PLANNED_HOURS,
    "fault_count": COL_FAULT_COUNT,
    "inspection_count": COL_INSPECTION_COUNT,
    "repair_count": COL_REPAIR_COUNT,
    "satisfaction": COL_SATISFACTION,
    "operation_cost": COL_COST,
    "cost_saving_rate": COL_SAVING_RATE,
    "uptime_rate": COL_UPTIME_RATE,
    "fault_rate": COL_FAULT_RATE,
}
COLUMN_TO_FIELD: dict[str, str] = {v: k for k, v in FIELD_TO_COLUMN.items()}


def _is_missing(value: Any) -> bool:
    """判断标量是否为空（None / NaN / NaT / 空字符串）。"""

    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _optional_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _optional_int(value: Any) -> int:
    number = _optional_float(value)
    return 0 if number is None else int(round(number))


@dataclass(slots=True)
class RobotOperationRecord:
    """机器人运营数据模型：一台机器人 + 一天 = 一条记录。"""

    operation_date: date | datetime | pd.Timestamp | None = None
    project_name: str = ""
    robot_id: str = ""
    robot_type: str = ""
    running_hours: float | None = None
    planned_hours: float | None = None
    fault_count: int = 0
    inspection_count: int = 0
    repair_count: int = 0
    satisfaction: float | None = None
    operation_cost: float | None = None
    cost_saving_rate: float | None = None

    # -- 派生指标 ---------------------------------------------------------
    @property
    def uptime_rate(self) -> float | None:
        """运行率（%）= 运行时长 / 计划运行时长 × 100%。"""

        if self.running_hours is None or not self.planned_hours:
            return None
        return self.running_hours / self.planned_hours * 100.0

    @property
    def fault_rate(self) -> float | None:
        """故障率（%）= 故障次数 / 巡检次数 × 100%。"""

        if self.inspection_count <= 0:
            return None
        return self.fault_count / self.inspection_count * 100.0

    @property
    def robot_type_enum(self) -> RobotType:
        return RobotType.coerce(self.robot_type)

    # -- 构造方法 ---------------------------------------------------------
    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RobotOperationRecord":
        """支持英文属性名或中文列名两种键。"""

        normalized: dict[str, Any] = {}
        for key, value in data.items():
            if key in FIELD_TO_COLUMN:
                normalized[key] = value
            elif key in COLUMN_TO_FIELD:
                normalized[COLUMN_TO_FIELD[key]] = value

        return cls(
            operation_date=normalized.get("operation_date"),
            project_name=str(normalized.get("project_name") or "").strip(),
            robot_id=str(normalized.get("robot_id") or "").strip(),
            robot_type=str(normalized.get("robot_type") or "").strip(),
            running_hours=_optional_float(normalized.get("running_hours")),
            planned_hours=_optional_float(normalized.get("planned_hours")),
            fault_count=_optional_int(normalized.get("fault_count")),
            inspection_count=_optional_int(normalized.get("inspection_count")),
            repair_count=_optional_int(normalized.get("repair_count")),
            satisfaction=_optional_float(normalized.get("satisfaction")),
            operation_cost=_optional_float(normalized.get("operation_cost")),
            cost_saving_rate=_optional_float(normalized.get("cost_saving_rate")),
        )

    @classmethod
    def from_series(cls, row: pd.Series) -> "RobotOperationRecord":
        """由 DataFrame 的一行构造记录。"""

        return cls.from_mapping(row.to_dict())

    # -- 输出 -------------------------------------------------------------
    def to_dict(self, *, use_chinese_columns: bool = True) -> dict[str, Any]:
        payload = {
            "operation_date": self.operation_date,
            "project_name": self.project_name,
            "robot_id": self.robot_id,
            "robot_type": self.robot_type,
            "running_hours": self.running_hours,
            "planned_hours": self.planned_hours,
            "fault_count": self.fault_count,
            "inspection_count": self.inspection_count,
            "repair_count": self.repair_count,
            "satisfaction": self.satisfaction,
            "operation_cost": self.operation_cost,
            "cost_saving_rate": self.cost_saving_rate,
            "uptime_rate": self.uptime_rate,
            "fault_rate": self.fault_rate,
        }
        if not use_chinese_columns:
            return payload
        return {FIELD_TO_COLUMN[key]: value for key, value in payload.items()}

    def validate(self) -> list[str]:
        """校验单条记录，返回问题描述列表（空列表表示通过）。"""

        problems: list[str] = []
        if _is_missing(self.operation_date):
            problems.append(f"{COL_DATE} 为空")
        if not self.project_name:
            problems.append(f"{COL_PROJECT} 为空")
        if not self.robot_id:
            problems.append(f"{COL_ROBOT_ID} 为空")
        if not self.robot_type:
            problems.append(f"{COL_ROBOT_TYPE} 为空")
        if self.running_hours is None or self.running_hours < 0:
            problems.append(f"{COL_RUNNING_HOURS} 缺失或为负数")
        if self.planned_hours is None or self.planned_hours <= 0:
            problems.append(f"{COL_PLANNED_HOURS} 缺失或非正数")
        if self.inspection_count < 0:
            problems.append(f"{COL_INSPECTION_COUNT} 为负数")
        if self.repair_count < 0:
            problems.append(f"{COL_REPAIR_COUNT} 为负数")
        if self.satisfaction is None or not (
            config.SATISFACTION_RANGE[0] <= self.satisfaction <= config.SATISFACTION_RANGE[1]
        ):
            problems.append(f"{COL_SATISFACTION} 缺失或超出有效区间")
        if self.operation_cost is None or self.operation_cost < 0:
            problems.append(f"{COL_COST} 缺失或为负数")
        if self.cost_saving_rate is None:
            problems.append(f"{COL_SAVING_RATE} 缺失")
        return problems


# ---------------------------------------------------------------------------
# 4. 核心指标模型
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricDefinition:
    """指标定义（名称、单位、口径说明、是否 Phase 1 核心指标）。"""

    name: str
    unit: str
    formula: str
    is_core: bool = True


CORE_METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition("项目数量", "个", "对「项目名称」去重计数"),
    MetricDefinition("机器人数量", "台", "对「机器人ID」去重计数"),
    MetricDefinition("平均运行时长", "小时", "清洗后记录的「运行时长」算术平均值"),
    MetricDefinition("故障率", "%", "总故障次数 / 总巡检次数 × 100%"),
    MetricDefinition("平均满意度", "分", "有效「用户满意度」记录的算术平均值"),
    MetricDefinition("总运营成本", "元", "有效「运营成本」记录的求和"),
    MetricDefinition("平均节降率", "%", "有效「节降率」记录的算术平均值"),
)

EXTRA_METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition("平均运行率", "%", "运行时长 / 计划运行时长 × 100%，再取平均", is_core=False),
    MetricDefinition("记录数", "条", "清洗后保留的记录条数", is_core=False),
    MetricDefinition("总故障次数", "次", "有效「故障次数」记录的求和", is_core=False),
)


@dataclass(frozen=True)
class MetricsSummary:
    """Phase 1 核心指标汇总。"""

    project_count: int
    robot_count: int
    avg_running_hours: float
    fault_rate: float
    avg_satisfaction: float
    total_cost: float
    avg_saving_rate: float
    # 补充统计（不是验收指标，仅用于说明数据规模与运行健康度）
    record_count: int = 0
    avg_uptime_rate: float = math.nan
    total_fault_count: int = 0

    def as_rows(self, *, digits: int = 2) -> list[dict[str, Any]]:
        """按指标定义顺序输出展示行。"""

        values: dict[str, Any] = {
            "项目数量": self.project_count,
            "机器人数量": self.robot_count,
            "平均运行时长": self.avg_running_hours,
            "故障率": self.fault_rate,
            "平均满意度": self.avg_satisfaction,
            "总运营成本": self.total_cost,
            "平均节降率": self.avg_saving_rate,
            "平均运行率": self.avg_uptime_rate,
            "记录数": self.record_count,
            "总故障次数": self.total_fault_count,
        }
        rows: list[dict[str, Any]] = []
        for definition in CORE_METRIC_DEFINITIONS + EXTRA_METRIC_DEFINITIONS:
            value = values[definition.name]
            if isinstance(value, float):
                value = None if math.isnan(value) else round(value, digits)
            rows.append(
                {
                    "指标": definition.name,
                    "数值": value,
                    "单位": definition.unit,
                    "指标类型": "核心指标" if definition.is_core else "补充统计",
                    "计算口径": definition.formula,
                }
            )
        return rows

    def as_dict(self, *, digits: int = 2) -> dict[str, Any]:
        rows = self.as_rows(digits=digits)
        return {row["指标"]: row["数值"] for row in rows}


# ---------------------------------------------------------------------------
# 5. 异常记录模型
# ---------------------------------------------------------------------------
ANOMALY_COLUMNS: tuple[str, ...] = (
    "日期",
    "项目名称",
    "机器人ID",
    "机器人类型",
    "异常类型",
    "判定指标",
    "判定粒度",
    "实际值",
    "判定条件",
    "阈值",
    "偏差幅度(%)",
    "严重程度",
    "异常说明",
)


@dataclass(frozen=True)
class AnomalyRecord:
    """单条异常命中记录。"""

    operation_date: Any
    project_name: str
    robot_id: str
    robot_type: str
    anomaly_type: str
    metric_name: str
    granularity: str
    actual_value: float
    condition: str
    threshold: float
    deviation_ratio: float
    severity: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "日期": self.operation_date,
            "项目名称": self.project_name,
            "机器人ID": self.robot_id,
            "机器人类型": self.robot_type,
            "异常类型": self.anomaly_type,
            "判定指标": self.metric_name,
            "判定粒度": self.granularity,
            "实际值": self.actual_value,
            "判定条件": self.condition,
            "阈值": self.threshold,
            "偏差幅度(%)": round(self.deviation_ratio * 100.0, 2),
            "严重程度": self.severity,
            "异常说明": self.description,
        }


def records_to_frame(records: Iterable[RobotOperationRecord]) -> pd.DataFrame:
    """把记录对象列表转换为中文列名的 DataFrame。"""

    rows = [record.to_dict(use_chinese_columns=True) for record in records]
    frame = pd.DataFrame(rows)
    ordered = [name for name in config.ALL_COLUMNS + config.DERIVED_COLUMNS if name in frame.columns]
    return frame[ordered] if ordered else frame
