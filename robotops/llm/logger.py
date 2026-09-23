"""RobotOps AI —— Phase 2 LLM 基础日志。

日志同时输出到控制台（stderr）与文件（默认 output/deepseek_llm.log）。
所有日志都会对 API Key 做脱敏，避免密钥泄漏到日志或终端。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .. import config as project_config

LOGGER_NAME = "robotops.llm"
LOG_FILE_NAME = "deepseek_llm.log"
DEFAULT_LOG_LEVEL = "INFO"
_HANDLER_MARKER = "_robotops_llm_handler"


def mask_secret(value: str | None, *, keep: int = 4) -> str:
    """对密钥类字符串脱敏，例如 ``sk-abcd****wxyz``。"""

    if not value:
        return "(未设置)"

    text = str(value)
    if len(text) <= keep * 2:
        return "*" * len(text)
    return f"{text[:keep]}****{text[-keep:]}"


def setup_llm_logger(
    *,
    level: str = DEFAULT_LOG_LEVEL,
    log_file: str | Path | None = None,
    console: bool = True,
) -> logging.Logger:
    """初始化 LLM 日志器（可重复调用，不会重复添加处理器）。"""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_resolve_level(level))
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
        # 日志文件不可写不应影响主流程，仅退化为控制台输出
        logger.debug("日志文件不可写，已退化为仅控制台输出：%s", target)

    return logger


def get_llm_logger() -> logging.Logger:
    """获取 LLM 日志器（未初始化时返回默认配置）。"""

    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        return setup_llm_logger()
    return logger


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)

