"""RobotOps AI —— Phase 2 LLM 模块。

包含：
- ``config``：模型名称、地址、超时、重试等配置的唯一来源；
- ``env_loader``：读取 .env（内置解析器，可选 python-dotenv）；
- ``client``：DeepSeek（OpenAI 兼容）客户端，含超时、重试、错误映射与日志；
- ``prompts``：Agent 系统提示词与用户提示词构造；
- ``schema``：Agent 结构化输出模型与解析校验；
- ``errors`` / ``logger``：错误类型与基础日志。

本阶段（Phase 2）不包含 RAG、向量库、多 Agent 等能力。
"""

from __future__ import annotations

from .client import ChatResult, DeepSeekClient, HttpRequest, HttpResponse, urllib_transport
from .config import LLMConfig
from .env_loader import ensure_env_loaded, load_env_file
from .errors import (
    LLMAuthError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMError,
    LLMInsufficientBalanceError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseFormatError,
    LLMServerError,
    LLMTimeoutError,
    MissingAPIKeyError,
)
from .logger import get_llm_logger, mask_secret, setup_llm_logger
from .schema import AIAnalysis, parse_ai_analysis

__all__ = [
    "AIAnalysis",
    "ChatResult",
    "DeepSeekClient",
    "HttpRequest",
    "HttpResponse",
    "LLMAuthError",
    "LLMConfig",
    "LLMConfigurationError",
    "LLMEmptyResponseError",
    "LLMError",
    "LLMInsufficientBalanceError",
    "LLMNetworkError",
    "LLMRateLimitError",
    "LLMRequestError",
    "LLMResponseFormatError",
    "LLMServerError",
    "LLMTimeoutError",
    "MissingAPIKeyError",
    "ensure_env_loaded",
    "get_llm_logger",
    "load_env_file",
    "mask_secret",
    "parse_ai_analysis",
    "setup_llm_logger",
    "urllib_transport",
]

