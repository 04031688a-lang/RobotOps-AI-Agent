"""RobotOps AI —— Phase 6 运营问题整改闭环模块。

组成：
- ``models``：问题数据模型、状态机与整改效果计算（纯程序计算，不用大模型推断）；
- ``store``：SQLite 持久化（标准库 sqlite3，默认 data/issues.db）；
- ``service``：问题创建、状态流转、处理记录、整改验证与关闭。
"""

from __future__ import annotations

from .errors import (
    IssueError,
    IssueNotFoundError,
    IssueStatusError,
    IssueStorageError,
)
from .models import (
    ALLOWED_TRANSITIONS,
    ANOMALY_METRIC_MAP,
    ISSUE_LIST_COLUMNS,
    Issue,
    IssueNote,
    IssueStatus,
    MetricChange,
    VerificationResult,
    allowed_next_statuses,
    build_verification_result,
    can_transition,
    compute_metric_change,
)
from .service import DEFAULT_AUTHOR, DEFAULT_OWNER, IssueService
from .store import DEFAULT_DB_PATH, SCHEMA_VERSION, IssueStore

__all__ = [
    "ALLOWED_TRANSITIONS",
    "ANOMALY_METRIC_MAP",
    "DEFAULT_AUTHOR",
    "DEFAULT_DB_PATH",
    "DEFAULT_OWNER",
    "ISSUE_LIST_COLUMNS",
    "Issue",
    "IssueError",
    "IssueNote",
    "IssueNotFoundError",
    "IssueService",
    "IssueStatus",
    "IssueStatusError",
    "IssueStorageError",
    "IssueStore",
    "MetricChange",
    "SCHEMA_VERSION",
    "VerificationResult",
    "allowed_next_statuses",
    "build_verification_result",
    "can_transition",
    "compute_metric_change",
]

