"""RobotOps AI —— Phase 4 工作流日志。

控制台输出格式与需求一致（便于测试与演示）：

    [Data Analysis Agent] started
    [Data Analysis Agent] completed
    [Diagnosis Agent] started
    [RAG Agent] retrieved 3 cases
    [Recommendation Agent] completed
    [Report Agent] completed

同时写入 ``output/workflow.log``（带时间戳），便于事后排查。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from robotops import config as project_config

LOGGER_NAME = "robotops.workflow"
LOG_FILE_NAME = "workflow.log"


class WorkflowLogger:
    """工作流日志器：控制台 + 文件双写。"""

    def __init__(
        self,
        *,
        verbose: bool = True,
        log_file: str | Path | None = None,
        echo: bool = True,
    ) -> None:
        self.verbose = verbose
        self.echo = echo
        self.messages: list[str] = []
        self.log_file = Path(log_file) if log_file is not None else project_config.OUTPUT_DIR / LOG_FILE_NAME

    # -- 主接口 -----------------------------------------------------------
    def log(self, message: str) -> None:
        """记录一行日志（控制台显示原文，文件带时间戳）。"""

        self.messages.append(message)
        if self.verbose and self.echo:
            print(message)
        self._write_file(message)

    def agent_started(self, tag: str) -> None:
        self.log(f"[{tag}] started")

    def agent_completed(self, tag: str, *, detail: str = "") -> None:
        self.log(f"[{tag}] completed" + (f" - {detail}" if detail else ""))

    def agent_failed(self, tag: str, error: str) -> None:
        self.log(f"[{tag}] failed - {error}")

    def agent_skipped(self, tag: str, reason: str) -> None:
        self.log(f"[{tag}] skipped - {reason}")

    def info(self, message: str) -> None:
        self.log(f"[Workflow] {message}")

    # -- 文件 -------------------------------------------------------------
    def _write_file(self, message: str) -> None:
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with self.log_file.open("a", encoding="utf-8") as handle:
                handle.write(f"{stamp} | {message}\n")
        except OSError:
            # 日志文件不可写不应影响工作流执行
            pass

    def text(self) -> str:
        """返回本次运行的全部日志行。"""

        return "\n".join(self.messages)

    def contains(self, message: str) -> bool:
        return any(message in line for line in self.messages)

