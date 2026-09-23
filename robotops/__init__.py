"""RobotOps AI —— 机器人运营数据分析智能 Agent。

当前版本：Phase 1「机器人运营数据分析基础模块」

Phase 1 已实现能力：
- Excel / CSV 数据读取与表头规范化；
- 数据清洗（类型转换、业务边界校验、去重、缺失值处理）；
- 核心指标计算（项目数量、机器人数量、平均运行时长、故障率、
  平均满意度、总运营成本、平均节降率）；
- 异常识别（故障率 > 5%、满意度 < 85、运行率 < 80%、节降率 < 10%）；
- 结果导出（JSON / CSV / Excel / Markdown）。

Phase 2 已实现能力（详见 ``robotops.llm`` 与 ``robotops.agent``）：
- 接入 DeepSeek API（.env 配置 API Key，模型名称集中管理）；
- 机器人运营数据分析 Agent：先由 Pandas 产出结构化结果，再交给大模型解读；
- 结构化输出解析、超时/重试/错误处理与基础日志。

尚未实现（后续阶段）：RAG / 向量库、多 Agent 协作、Streamlit 页面、Docker 部署、数据库。
"""

from __future__ import annotations

from .anomaly import detect_anomalies, describe_rules, summarize_anomalies
from .config import DEFAULT_DATA_FILE, DEFAULT_THRESHOLDS, PROJECT_ROOT, AnomalyThresholds
from .data_cleaner import CleaningResult, clean_operation_data
from .data_loader import LoadResult, load_operation_data
from .exceptions import (
    ConfigurationError,
    DataLoadError,
    DataValidationError,
    RobotOpsError,
)
from .metrics import calculate_core_metrics, project_level_summary
from .models import MetricsSummary, RobotOperationRecord, RobotType
from .pipeline import AnalysisResult, run_analysis

__version__ = "0.1.0"
__phase__ = "Phase 1 - 机器人运营数据分析基础模块"

__all__ = [
    "AnalysisResult",
    "AnomalyThresholds",
    "CleaningResult",
    "ConfigurationError",
    "DEFAULT_DATA_FILE",
    "DEFAULT_THRESHOLDS",
    "DataLoadError",
    "DataValidationError",
    "LoadResult",
    "MetricsSummary",
    "PROJECT_ROOT",
    "RobotOperationRecord",
    "RobotOpsError",
    "RobotType",
    "calculate_core_metrics",
    "clean_operation_data",
    "describe_rules",
    "detect_anomalies",
    "load_operation_data",
    "project_level_summary",
    "run_analysis",
    "summarize_anomalies",
    "__version__",
    "__phase__",
]
