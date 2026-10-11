"""RobotOps AI —— Phase 1 全局配置。

本模块集中管理：
1. 项目路径（数据目录、输出目录等）；
2. 数据列名常量与列别名（兼容不同来源的表头写法）；
3. 数据质量校验边界；
4. 异常识别阈值。

其他模块统一从这里读取配置，避免在业务代码里硬编码魔法数字。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. 路径配置
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

DEMO_EXCEL_FILE: Path = RAW_DATA_DIR / "robot_operation_demo.xlsx"
DEFAULT_DATA_FILE: Path = DEMO_EXCEL_FILE

# 支持的数据文件后缀
SUPPORTED_SUFFIXES: tuple[str, ...] = (".xlsx", ".xlsm", ".xls", ".csv")

# ---------------------------------------------------------------------------
# 2. 数据列定义
# ---------------------------------------------------------------------------
COL_DATE = "日期"
COL_PROJECT = "项目名称"
COL_ROBOT_ID = "机器人ID"
COL_ROBOT_TYPE = "机器人类型"
COL_RUNNING_HOURS = "运行时长"
COL_PLANNED_HOURS = "计划运行时长"
COL_FAULT_COUNT = "故障次数"
COL_INSPECTION_COUNT = "巡检次数"
COL_REPAIR_COUNT = "维修次数"
COL_SATISFACTION = "用户满意度"
COL_COST = "运营成本"
COL_SAVING_RATE = "节降率"

# 派生列：由数据清洗模块计算，原始数据中不需要提供
COL_UPTIME_RATE = "运行率"
COL_FAULT_RATE = "故障率"

# 原始数据必须具备的列
REQUIRED_COLUMNS: tuple[str, ...] = (
    COL_DATE,
    COL_PROJECT,
    COL_ROBOT_ID,
    COL_ROBOT_TYPE,
    COL_RUNNING_HOURS,
    COL_FAULT_COUNT,
    COL_INSPECTION_COUNT,
    COL_REPAIR_COUNT,
    COL_SATISFACTION,
    COL_COST,
    COL_SAVING_RATE,
)

# 推荐提供、缺失时可自动兜底的列
OPTIONAL_COLUMNS: tuple[str, ...] = (COL_PLANNED_HOURS,)

ALL_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
DERIVED_COLUMNS: tuple[str, ...] = (COL_UPTIME_RATE, COL_FAULT_RATE)

TEXT_COLUMNS: tuple[str, ...] = (COL_PROJECT, COL_ROBOT_ID, COL_ROBOT_TYPE)
COUNT_COLUMNS: tuple[str, ...] = (COL_FAULT_COUNT, COL_INSPECTION_COUNT, COL_REPAIR_COUNT)
MEASURE_COLUMNS: tuple[str, ...] = (
    COL_RUNNING_HOURS,
    COL_PLANNED_HOURS,
    COL_SATISFACTION,
    COL_COST,
    COL_SAVING_RATE,
)
NUMERIC_COLUMNS: tuple[str, ...] = COUNT_COLUMNS + MEASURE_COLUMNS

# 表头别名：兼容其他系统导出的同义列名，统一映射到标准列名
COLUMN_ALIASES: dict[str, str] = {
    "项目": COL_PROJECT,
    "项目名": COL_PROJECT,
    "项目名称": COL_PROJECT,
    "project": COL_PROJECT,
    "project_name": COL_PROJECT,
    "机器人编号": COL_ROBOT_ID,
    "机器人id": COL_ROBOT_ID,
    "机器人编号id": COL_ROBOT_ID,
    "设备id": COL_ROBOT_ID,
    "robot_id": COL_ROBOT_ID,
    "robotid": COL_ROBOT_ID,
    "类型": COL_ROBOT_TYPE,
    "设备类型": COL_ROBOT_TYPE,
    "robot_type": COL_ROBOT_TYPE,
    "运行时长(小时)": COL_RUNNING_HOURS,
    "运行小时数": COL_RUNNING_HOURS,
    "运行时间": COL_RUNNING_HOURS,
    "planned_hours": COL_PLANNED_HOURS,
    "计划时长": COL_PLANNED_HOURS,
    "计划运行时长(小时)": COL_PLANNED_HOURS,
    "故障数": COL_FAULT_COUNT,
    "故障次数(次)": COL_FAULT_COUNT,
    "fault_count": COL_FAULT_COUNT,
    "巡检数": COL_INSPECTION_COUNT,
    "巡检次数(次)": COL_INSPECTION_COUNT,
    "inspections": COL_INSPECTION_COUNT,
    "维修数": COL_REPAIR_COUNT,
    "维修次数(次)": COL_REPAIR_COUNT,
    "满意度": COL_SATISFACTION,
    "用户满意度(分)": COL_SATISFACTION,
    "satisfaction": COL_SATISFACTION,
    "成本": COL_COST,
    "运营成本(元)": COL_COST,
    "运维成本": COL_COST,
    "cost": COL_COST,
    "节能率": COL_SAVING_RATE,
    "节降率(%)": COL_SAVING_RATE,
    "saving_rate": COL_SAVING_RATE,
}

# ---------------------------------------------------------------------------
# 3. 业务边界与兜底值
# ---------------------------------------------------------------------------
SATISFACTION_RANGE: tuple[float, float] = (0.0, 100.0)
COST_SAVING_RATE_RANGE: tuple[float, float] = (-100.0, 100.0)
MAX_RUNNING_HOURS_PER_DAY: float = 24.0

# 原始数据缺少“计划运行时长”时使用的兜底值（按 24 小时连续作业口径）
DEFAULT_PLANNED_HOURS: float = 24.0

# 各机器人类型的每日计划运行时长（小时），用于生成演示数据与兜底推断
PLANNED_HOURS_BY_TYPE: dict[str, float] = {
    "清洁机器人": 8.0,
    "巡检机器人": 12.0,
    "安防巡逻机器人": 24.0,
    "配送机器人": 10.0,
    "消杀机器人": 6.0,
    "AGV搬运机器人": 16.0,
}


# ---------------------------------------------------------------------------
# 4. 异常识别阈值
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnomalyThresholds:
    """Phase 1 异常识别阈值（可按业务需要调整）。"""

    fault_rate_upper: float = 5.0  # 故障率 > 5% 判定异常
    satisfaction_lower: float = 85.0  # 用户满意度 < 85 分判定异常
    uptime_rate_lower: float = 80.0  # 运行率 < 80% 判定异常
    saving_rate_lower: float = 10.0  # 节降率 < 10% 判定异常

    def as_dict(self) -> dict[str, float]:
        return {
            "故障率上限(%)": self.fault_rate_upper,
            "满意度下限(分)": self.satisfaction_lower,
            "运行率下限(%)": self.uptime_rate_lower,
            "节降率下限(%)": self.saving_rate_lower,
        }


DEFAULT_THRESHOLDS: AnomalyThresholds = AnomalyThresholds()

# 故障率的异常判定口径：
# - "robot_period"：按“机器人 + 分析周期”累计判定（累计故障次数 / 累计巡检次数），默认值。
#   原因：故障率是比率型指标，单日样本过小（例如 1 次故障 / 8 次巡检 = 12.5%），
#   逐日判定会让 5% 阈值失去区分度。
# - "record"：按每日记录逐条判定（更严格，异常数量会明显增加）。
FAULT_RATE_ANOMALY_SCOPE: str = "robot_period"
FAULT_RATE_SCOPE_CHOICES: tuple[str, ...] = ("robot_period", "record")

# 严重程度分级：偏差幅度 = |实际值 - 阈值| / 阈值
SEVERITY_HIGH_RATIO: float = 0.5  # 偏差 >= 50% 记为“高”
SEVERITY_MEDIUM_RATIO: float = 0.2  # 偏差 >= 20% 记为“中”，其余记为“低”

# ---------------------------------------------------------------------------
# 5. 输出配置
# ---------------------------------------------------------------------------
CSV_ENCODING: str = "utf-8-sig"  # 带 BOM，保证 Excel 直接双击打开不乱码
TEXT_ENCODING: str = "utf-8"
REPORT_BASENAME: str = "robot_ops_report"


# ---------------------------------------------------------------------------
# 6. Phase 8：数据质量检查
# ---------------------------------------------------------------------------
# 数据质量状态：
#   pass    通过（可以正常进入分析）
#   warning 可继续分析，但存在需要关注的问题（缺失、重复、越界等）
#   error   严重问题（缺少核心字段等），禁止继续分析
QUALITY_STATUS_PASS: str = "pass"
QUALITY_STATUS_WARNING: str = "warning"
QUALITY_STATUS_ERROR: str = "error"

QUALITY_STATUS_LABELS: dict[str, str] = {
    QUALITY_STATUS_PASS: "通过",
    QUALITY_STATUS_WARNING: "警告",
    QUALITY_STATUS_ERROR: "严重",
}


@dataclass(frozen=True)
class QualityThresholds:
    """Phase 8 数据质量检查阈值（集中配置，业务侧只需改这里）。"""

    # 单列缺失率：达到 warning 记警告，达到 error 记严重问题
    missing_ratio_warning: float = 0.10
    missing_ratio_error: float = 0.50
    # 缺失值升级为"严重问题"的最小缺失条数：
    # 小样本（例如只有 2 行）里缺失 1 行就是 50%，不应直接禁止分析，故加此护栏
    missing_min_rows_for_error: int = 5
    # 重复记录占比达到该值记严重问题
    duplicate_ratio_error: float = 0.20
    # 非法/越界值占比达到该值记严重问题
    invalid_ratio_error: float = 0.20
    # 空行占比达到该值记严重问题
    empty_row_ratio_error: float = 0.50

    def as_dict(self) -> dict[str, float]:
        return {
            "单列缺失率警告阈值": self.missing_ratio_warning,
            "单列缺失率严重阈值": self.missing_ratio_error,
            "缺失值严重问题最小条数": self.missing_min_rows_for_error,
            "重复记录占比严重阈值": self.duplicate_ratio_error,
            "非法值占比严重阈值": self.invalid_ratio_error,
            "空行占比严重阈值": self.empty_row_ratio_error,
        }


DEFAULT_QUALITY_THRESHOLDS: QualityThresholds = QualityThresholds()


# ---------------------------------------------------------------------------
# 7. Phase 8：运营指标体系与阈值
# ---------------------------------------------------------------------------
# Phase 8 新增指标名（指标结果列名，不是原始数据列）
COL_ONLINE_RATE = "设备在线率"
COL_UTILIZATION_RATE = "设备利用率"
COL_MTBF = "MTBF"
COL_MTTR = "MTTR"
COL_AVG_COST = "平均运营成本"

# MTTR 的停机时长口径（当前数据源没有维修工单时长字段时使用）：
#   planned_gap = Σmax(计划运行时长 - 运行时长, 0)
MTTR_DOWNTIME_PROXY: str = "planned_gap"


@dataclass(frozen=True)
class IndicatorThresholds:
    """Phase 8 运营指标阈值。

    说明：
    - 故障率 / 满意度 / 节降率三项与 Phase 1 的 AnomalyThresholds 保持同一默认值，
      区别只在判定粒度：Phase 1 按记录（或机器人周期）判定，Phase 8 按指标聚合结果判定；
    - 设备利用率下限与 MTTR 倍数为 Phase 8 新增。
    """

    fault_rate_upper: float = 5.0  # 故障率 > 5% 判定异常
    satisfaction_lower: float = 85.0  # 用户满意度 < 85 分判定异常
    utilization_lower: float = 80.0  # 设备利用率 < 80% 判定异常
    saving_rate_lower: float = 10.0  # 节降率 < 10% 判定异常
    mttr_project_avg_ratio: float = 1.0  # MTTR 高于项目平均值的该倍数 → 预警

    def as_dict(self) -> dict[str, float]:
        return {
            "故障率上限(%)": self.fault_rate_upper,
            "满意度下限(分)": self.satisfaction_lower,
            "设备利用率下限(%)": self.utilization_lower,
            "节降率下限(%)": self.saving_rate_lower,
            "MTTR高于项目平均倍数": self.mttr_project_avg_ratio,
        }


DEFAULT_INDICATOR_THRESHOLDS: IndicatorThresholds = IndicatorThresholds()


# ---------------------------------------------------------------------------
# 8. Phase 9：趋势分析与项目横向对比
# ---------------------------------------------------------------------------
TREND_GRANULARITY_DAY: str = "day"
TREND_GRANULARITY_WEEK: str = "week"
TREND_GRANULARITY_MONTH: str = "month"
TREND_GRANULARITY_AUTO: str = "auto"
TREND_GRANULARITY_CHOICES: tuple[str, ...] = (
    TREND_GRANULARITY_DAY,
    TREND_GRANULARITY_WEEK,
    TREND_GRANULARITY_MONTH,
)
TREND_GRANULARITY_LABELS: dict[str, str] = {
    TREND_GRANULARITY_DAY: "日",
    TREND_GRANULARITY_WEEK: "周",
    TREND_GRANULARITY_MONTH: "月",
}

#: 趋势方向
TREND_UP: str = "上升"
TREND_DOWN: str = "下降"
TREND_FLAT: str = "持平"


@dataclass(frozen=True)
class TrendSettings:
    """Phase 9 趋势分析参数（集中配置）。"""

    default_granularity: str = TREND_GRANULARITY_AUTO
    # 自动粒度：数据天数（去重后的自然日数量）>= auto_week_span_days 用「周」，
    # >= auto_month_span_days 用「月」，否则用「日」
    auto_week_span_days: int = 21
    auto_month_span_days: int = 90
    # 计算环比至少需要多少个周期
    min_periods_for_change: int = 2
    # 判定「连续上升/连续下降」至少需要多少个周期（3 个周期 = 2 次同向变动）
    min_periods_for_consecutive: int = 3
    # 变化绝对值小于该值视为「持平」
    flat_epsilon: float = 1e-9
    # 相对容差：变化幅度小于上期值的该比例（默认 0.01%）视为「持平」，避免浮点噪声被判成涨跌
    flat_ratio_epsilon: float = 0.0001
    # 是否把不完整的首尾周期（例如最后一周只覆盖 1 天）排除在环比/连续判定之外
    exclude_partial_periods: bool = True
    # 同期变化（同比）的滞后周期数：日=7 天前、周=4 周前、月=12 个月前
    same_period_lag_days: int = 7
    same_period_lag_weeks: int = 4
    same_period_lag_months: int = 12

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "默认粒度": self.default_granularity,
            "自动切周跨度(天)": self.auto_week_span_days,
            "自动切月跨度(天)": self.auto_month_span_days,
            "环比最少周期数": self.min_periods_for_change,
            "连续变化最少周期数": self.min_periods_for_consecutive,
            "排除不完整周期": "是" if self.exclude_partial_periods else "否",
            "持平相对容差": self.flat_ratio_epsilon,
            "同期滞后(日/周/月)": (
                str(self.same_period_lag_days) + "/"
                + str(self.same_period_lag_weeks) + "/"
                + str(self.same_period_lag_months)
            ),
        }


DEFAULT_TREND_SETTINGS: TrendSettings = TrendSettings()
