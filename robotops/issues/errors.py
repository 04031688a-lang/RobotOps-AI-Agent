"""RobotOps AI —— Phase 6 运营问题闭环异常类型。"""

from __future__ import annotations

from ..exceptions import RobotOpsError


class IssueError(RobotOpsError):
    """运营问题模块错误的基础类型。"""

    default_hint = ""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint or self.default_hint

    def describe(self) -> str:
        text = str(self)
        if self.hint:
            text = f"{text}\n建议：{self.hint}"
        return text


class IssueNotFoundError(IssueError):
    """问题编号不存在。"""

    default_hint = "请在「运营问题」页面刷新列表后重新选择问题编号。"


class IssueStatusError(IssueError):
    """状态流转不合法或缺少必要信息。"""

    default_hint = "运营问题的状态必须按「待处理 → 处理中 → 待验证 → 已完成 → 已关闭」流转。"


class IssueStorageError(IssueError):
    """SQLite 读写失败。"""

    default_hint = "确认 data/issues.db 所在目录可写（关闭占用该文件的程序后重试）。"

