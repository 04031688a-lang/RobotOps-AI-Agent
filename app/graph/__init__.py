"""RobotOps AI —— Phase 4 LangGraph 工作流层（State / 日志 / 提示词 / 编排）。

说明：工作流模块（``workflow.py``）依赖 ``app.agents``，为避免与 ``app.agents``
互相导入形成循环，这里通过 ``__getattr__`` 惰性导出工作流对象。
"""

from __future__ import annotations

from .logging_utils import WorkflowLogger
from .prompts import PROMPT_VERSION
from .state import (
    REQUIRED_STATE_KEYS,
    WORKFLOW_VERSION,
    WorkflowState,
    create_initial_state,
    missing_required_keys,
    serializable_state,
    state_overview,
    thresholds_from_state,
)

#: 由 workflow.py 提供、按需加载的成员
_LAZY_WORKFLOW_EXPORTS = frozenset(
    {
        "NODE_ABNORMAL_CHECK",
        "NODE_DATA_ANALYSIS",
        "NODE_DIAGNOSIS",
        "NODE_RAG",
        "NODE_RECOMMENDATION",
        "NODE_REPORT",
        "RobotOpsWorkflow",
        "export_workflow_outputs",
        "run_workflow",
    }
)


def __getattr__(name: str) -> object:
    if name in _LAZY_WORKFLOW_EXPORTS:
        from . import workflow

        return getattr(workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "NODE_ABNORMAL_CHECK",
    "NODE_DATA_ANALYSIS",
    "NODE_DIAGNOSIS",
    "NODE_RAG",
    "NODE_RECOMMENDATION",
    "NODE_REPORT",
    "REQUIRED_STATE_KEYS",
    "RobotOpsWorkflow",
    "WORKFLOW_VERSION",
    "WorkflowLogger",
    "WorkflowState",
    "create_initial_state",
    "export_workflow_outputs",
    "missing_required_keys",
    "run_workflow",
    "serializable_state",
    "state_overview",
    "thresholds_from_state",
]
