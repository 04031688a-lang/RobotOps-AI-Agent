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
