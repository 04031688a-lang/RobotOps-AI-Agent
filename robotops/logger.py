"""RobotOps AI —— Phase 11 统一日志模块。

所有业务事件通过本模块写入统一日志文件（默认 ``output/robotops.log``），
格式与需求一致：

.. code-block:: text

    2026-09-24 10:21:31 INFO Data analysis started
    2026-09-24 10:21:32 INFO 86 robots loaded
    2026-09-24 10:21:32 INFO 4 abnormal projects detected
    2026-09-24 10:21:33 INFO RAG retrieved 3 cases
    2026-09-24 10:21:36 INFO Diagnosis completed

记录的事件（详见各接入点）：

- 数据上传 / 读取（data_loader、frontend 上传）
- 数据清洗、指标计算、异常检测（pipeline）
- Agent 开始 / 结束（app/graph 工作流日志桥接）
- RAG 检索（rag/retriever）
- DeepSeek API 调用（llm/client）
- 报告生成（report）
- Issue 创建 / 状态变更 / 验证 / 关闭（issues/service）

设计要点：

- **幂等初始化**：可重复调用 ``setup_logger``，不会重复添加处理器；
- **不阻断主流程**：日志文件不可写时自动退化为静默（仅保留已注册的处理器）；
- **错误分级**：ERROR 记录错误摘要，详细 traceback 只通过 ``log_exception``
  写入日志文件，普通用户不会看到（前端/命令行只展示 ``describe()`` 的友好提示）。
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from types import TracebackType
from typing import Any

from . import config as project_config

LOGGER_NAME = "robotops"
LOG_FILE_NAME = "robotops.log"
DEFAULT_LOG_LEVEL = "INFO"
_HANDLER_MARKER = "_robotops_unified_handler"

#: 统一日志格式：时间 级别 事件（与需求示例一致）
_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    *,
    level: str = DEFAULT_LOG_LEVEL,
    log_file: str | Path | None = None,
    console: bool = False,
) -> logging.Logger:
    """初始化统一日志器（可重复调用，不会重复添加处理器）。

    Args:
        level: 日志级别（默认 INFO）。
        log_file: 日志文件路径，默认 ``output/robotops.log``。
        console: 是否同时输出到 stderr（默认关闭，避免与各入口
            已有的 verbose 控制台输出重复）。
    """

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_resolve_level(level))
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setFormatter(formatter)
        setattr(stream_handler, _HANDLER_MARKER, True)
        logger.addHandler(stream_handler)

    target = Path(log_file) if log_file is not None else project_config.OUTPUT_DIR / LOG_FILE_NAME
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_MARKER, True)
        logger.addHandler(file_handler)
    except OSError:
        # 日志文件不可写不应影响主流程
        logger.debug("统一日志文件不可写，已退化为仅控制台输出：%s", target)

    return logger


def get_logger() -> logging.Logger:
    """获取统一日志器（未初始化时自动按默认配置初始化）。"""

    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        return setup_logger()
    return logger


def log_event(message: str, *args: Any) -> None:
    """记录一条业务事件（INFO 级别）。支持 ``%`` 风格的惰性格式化。"""

    get_logger().info(message, *args)


def log_warning(message: str, *args: Any) -> None:
    """记录一条警告（WARNING 级别）。"""

    get_logger().warning(message, *args)


def log_error(message: str, *args: Any) -> None:
    """记录一条错误摘要（ERROR 级别，不含 traceback）。"""

    get_logger().error(message, *args)


def log_exception(
    error: BaseException,
    *,
    context: str = "",
    tb: TracebackType | None = None,
) -> None:
    """把异常连同详细 traceback 写入日志（用户侧只展示友好提示）。

    Args:
        error: 捕获到的异常对象。
        context: 出错场景描述（例如 ``"run_analysis"``）。
        tb: traceback 对象；缺省时取 ``error.__traceback__``。
    """

    logger = get_logger()
    prefix = f"[{context}] " if context else ""
    detail = "".join(
        traceback.format_exception(type(error), error, tb if tb is not None else error.__traceback__)
    ).rstrip()
    logger.error("%s%s: %s\n%s", prefix, type(error).__name__, error, detail)


def read_log_file(log_file: str | Path | None = None) -> str:
    """读取统一日志文件内容（测试与排查用）。文件不存在时返回空串。"""

    target = Path(log_file) if log_file is not None else project_config.OUTPUT_DIR / LOG_FILE_NAME
    try:
        return target.read_text(encoding="utf-8")
    except OSError:
        return ""


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)
