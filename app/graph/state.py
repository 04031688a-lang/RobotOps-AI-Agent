"""RobotOps AI —— Phase 4 工作流统一 State。

需求要求 State 至少包含：
``raw_data_summary`` / ``metrics`` / ``abnormal_projects`` / ``diagnosis`` /
``retrieved_cases`` / ``recommendations`` / ``final_report``；
在此基础上补充了流程控制与可观测性字段（异常标记、步骤轨迹、错误列表等）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict

from robotops import config as project_config

WORKFLOW_VERSION = "phase4-multi-agent-v1"


class WorkflowState(TypedDict, total=False):
    """LangGraph 工作流状态（各节点只更新自己负责的字段）。"""

    # ---- 输入参数 -------------------------------------------------------
    data_path: str
    thresholds: dict[str, float]
    use_knowledge: bool
    use_llm: bool
    rag_top_k: int | None
    rag_min_similarity: float | None
    rag_max_cases: int | None
    export: bool
    output_dir: str

    # ---- 需求要求的核心字段 ---------------------------------------------
    raw_data_summary: dict[str, Any]
    metrics: dict[str, Any]
    abnormal_projects: list[dict[str, Any]]
    diagnosis: dict[str, Any]
    retrieved_cases: list[dict[str, Any]]
    recommendations: dict[str, Any]
    final_report: str
    report_meta: dict[str, Any]

    # ---- 辅助字段（供各 Agent 使用，不参与对外契约）---------------------
    analysis_payload: dict[str, Any]
    anomaly_summary: list[dict[str, Any]]
    abnormal_robots: list[dict[str, Any]]
    data_quality: dict[str, Any]
    retrieval: dict[str, Any]
    analysis_result: Any  # Phase 1 的 AnalysisResult（进程内传递，不序列化）
    analysis_result_path: str

    # ---- 流程控制 -------------------------------------------------------
    has_anomalies: bool
    anomaly_count: int
    abnormal_check_report: str
    workflow_status: str  # running / completed / completed_with_errors / failed
    workflow_version: str
    started_at: str
    finished_at: str
    steps: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    failed_stages: list[str]
    skipped_stages: list[str]
    is_offline: bool


def create_initial_state(
    *,
    data_path: str | None = None,
    thresholds: project_config.AnomalyThresholds | None = None,
    use_knowledge: bool = True,
    use_llm: bool = True,
    rag_top_k: int | None = None,
    rag_min_similarity: float | None = None,
    rag_max_cases: int | None = None,
    export: bool = True,
    output_dir: str | None = None,
) -> WorkflowState:
    """构造工作流初始状态。"""

    limits = thresholds or project_config.DEFAULT_THRESHOLDS
    return WorkflowState(
        data_path=str(data_path) if data_path else str(project_config.DEFAULT_DATA_FILE),
        thresholds=limits.as_dict(),
        use_knowledge=bool(use_knowledge),
        use_llm=bool(use_llm),
        rag_top_k=rag_top_k,
        rag_min_similarity=rag_min_similarity,
        rag_max_cases=rag_max_cases,
        export=bool(export),
        output_dir=str(output_dir) if output_dir else str(project_config.OUTPUT_DIR),
        raw_data_summary={},
        metrics={},
        abnormal_projects=[],
        diagnosis={},
        retrieved_cases=[],
        recommendations={},
        final_report="",
        analysis_payload={},
        anomaly_summary=[],
        abnormal_robots=[],
        data_quality={},
        retrieval={},
        has_anomalies=False,
        anomaly_count=0,
        workflow_status="running",
        workflow_version=WORKFLOW_VERSION,
        started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        steps=[],
        errors=[],
        failed_stages=[],
        skipped_stages=[],
        is_offline=not bool(use_llm),
    )


REQUIRED_STATE_KEYS: tuple[str, ...] = (
    "raw_data_summary",
    "metrics",
    "abnormal_projects",
    "diagnosis",
    "retrieved_cases",
    "recommendations",
    "final_report",
)


_THRESHOLD_FIELDS: dict[str, str] = {
    "故障率上限(%)": "fault_rate_upper",
    "满意度下限(分)": "satisfaction_lower",
    "运行率下限(%)": "uptime_rate_lower",
    "节降率下限(%)": "saving_rate_lower",
}


def thresholds_from_state(
    state: "WorkflowState",
) -> project_config.AnomalyThresholds:
    """从 State 还原异常阈值对象（兼容直接传入 AnomalyThresholds 的情况）。"""

    value = state.get("thresholds")
    if isinstance(value, project_config.AnomalyThresholds):
        return value
    if not isinstance(value, dict):
        return project_config.DEFAULT_THRESHOLDS

    kwargs: dict[str, float] = {}
    for cn_key, field in _THRESHOLD_FIELDS.items():
        if cn_key in value:
            kwargs[field] = float(value[cn_key])
    if not kwargs:
        return project_config.DEFAULT_THRESHOLDS
    defaults = project_config.DEFAULT_THRESHOLDS
    return project_config.AnomalyThresholds(
        fault_rate_upper=kwargs.get("fault_rate_upper", defaults.fault_rate_upper),
        satisfaction_lower=kwargs.get("satisfaction_lower", defaults.satisfaction_lower),
        uptime_rate_lower=kwargs.get("uptime_rate_lower", defaults.uptime_rate_lower),
        saving_rate_lower=kwargs.get("saving_rate_lower", defaults.saving_rate_lower),
    )


def missing_required_keys(state: WorkflowState) -> list[str]:
    """返回缺失的必需字段（用于自检）。"""

    return [key for key in REQUIRED_STATE_KEYS if key not in state]


def serializable_state(state: WorkflowState) -> dict[str, Any]:
    """去掉不可序列化的对象（例如 AnalysisResult），用于导出 JSON。"""

    data = {key: value for key, value in state.items() if key != "analysis_result"}
    data["final_report_preview"] = (state.get("final_report") or "")[:200]
    return data


def state_overview(state: WorkflowState) -> str:
    """一行式状态摘要（CLI 输出与日志使用）。"""

    raw_metrics = state.get("metrics") or {}
    metrics = raw_metrics.get("values", raw_metrics) if isinstance(raw_metrics, dict) else {}
    return (
        f"项目 {metrics.get('项目数量', '-')} 个 / 机器人 {metrics.get('机器人数量', '-')} 台 / "
        f"异常 {state.get('anomaly_count', 0)} 条 / "
        f"历史案例 {len(state.get('retrieved_cases') or [])} 个 / "
        f"状态 {state.get('workflow_status', '-')}"
    )
