"""RobotOps AI —— Phase 2 LLM 配置（模型名称与接口参数的唯一来源）。

设计约定：
- 模型名称、接口地址、超时、重试次数等**只在本文件定义默认值**，
  业务代码中不允许再出现模型名字面量；
- 所有值都可以通过 .env / 环境变量覆盖（例如 DEEPSEEK_MODEL）；
- API Key 只从环境变量 ``DEEPSEEK_API_KEY`` 读取，绝不写入代码或日志。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .env_loader import ensure_env_loaded, get_env
from .errors import LLMConfigurationError, MissingAPIKeyError

# ---------------------------------------------------------------------------
# 环境变量名称（统一在这里定义，避免散落）
# ---------------------------------------------------------------------------
ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_BASE_URL = "DEEPSEEK_BASE_URL"
ENV_MODEL = "DEEPSEEK_MODEL"
ENV_TIMEOUT = "DEEPSEEK_TIMEOUT"
ENV_MAX_RETRIES = "DEEPSEEK_MAX_RETRIES"
ENV_TEMPERATURE = "DEEPSEEK_TEMPERATURE"
ENV_MAX_TOKENS = "DEEPSEEK_MAX_TOKENS"
ENV_LOG_LEVEL = "DEEPSEEK_LOG_LEVEL"

# ---------------------------------------------------------------------------
# 默认值（唯一来源）
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://api.deepseek.com"
CHAT_COMPLETIONS_PATH = "/chat/completions"

#: 默认模型（可在 .env 中通过 DEEPSEEK_MODEL 覆盖）
DEFAULT_MODEL = "deepseek-chat"

#: 已知可用模型（仅用于给出提示，不强制限制）
KNOWN_MODELS: tuple[str, ...] = ("deepseek-chat", "deepseek-reasoner")

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 4096
DEFAULT_RESPONSE_FORMAT = "json_object"
DEFAULT_LOG_LEVEL = "INFO"

#: 重试退避参数（指数退避 + 上限）
RETRY_BACKOFF_BASE_SECONDS = 1.0
RETRY_BACKOFF_MAX_SECONDS = 8.0

USER_AGENT = "RobotOps-AI/0.2 (Phase 2)"


@dataclass(frozen=True)
class LLMConfig:
    """DeepSeek 调用配置。"""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    response_format: str | None = DEFAULT_RESPONSE_FORMAT
    log_level: str = DEFAULT_LOG_LEVEL
    log_file: Path | None = None
    env_file: Path | None = field(default=None, compare=False)

    # -- 构造 -------------------------------------------------------------
    @classmethod
    def from_env(
        cls,
        *,
        env_file: str | Path | None = None,
        **overrides: Any,
    ) -> "LLMConfig":
        """从 .env / 环境变量构造配置，``overrides`` 中的值优先级最高。"""

        used_env_file = ensure_env_loaded(env_file)

        values: dict[str, Any] = {
            "base_url": get_env(ENV_BASE_URL, DEFAULT_BASE_URL),
            "model": get_env(ENV_MODEL, DEFAULT_MODEL),
            "api_key": get_env(ENV_API_KEY),
            "timeout": _to_float(get_env(ENV_TIMEOUT), DEFAULT_TIMEOUT_SECONDS, ENV_TIMEOUT),
            "max_retries": _to_int(
                get_env(ENV_MAX_RETRIES), DEFAULT_MAX_RETRIES, ENV_MAX_RETRIES
            ),
            "temperature": _to_float(
                get_env(ENV_TEMPERATURE), DEFAULT_TEMPERATURE, ENV_TEMPERATURE
            ),
            "max_tokens": _to_int(get_env(ENV_MAX_TOKENS), DEFAULT_MAX_TOKENS, ENV_MAX_TOKENS),
            "log_level": get_env(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL),
            "env_file": used_env_file,
        }
        values.update(overrides)
        return cls(**values)

    # -- 属性 -------------------------------------------------------------
    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{CHAT_COMPLETIONS_PATH}"

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key and str(self.api_key).strip())

    @property
    def masked_api_key(self) -> str:
        from .logger import mask_secret

        return mask_secret(self.api_key)

    def to_public_dict(self) -> dict[str, Any]:
        """用于日志与报告的安全配置快照（不含密钥明文）。"""

        return {
            "base_url": self.base_url,
            "model": self.model,
            "timeout_seconds": self.timeout,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": self.response_format,
            "api_key": self.masked_api_key,
            "env_file": str(self.env_file) if self.env_file else None,
        }

    # -- 校验 -------------------------------------------------------------
    def validate(self, *, require_api_key: bool = True) -> None:
        """校验配置，非法时抛出带排查建议的异常。"""

        if not self.base_url or not str(self.base_url).startswith(("http://", "https://")):
            raise LLMConfigurationError(
                f"接口地址不合法：{self.base_url!r}，应为 https://api.deepseek.com"
            )
        if not self.model or not str(self.model).strip():
            raise LLMConfigurationError(
                f"模型名称为空，请检查 .env 中的 {ENV_MODEL} 或 robotops/llm/config.py 的默认值"
            )
        if not (self.model in KNOWN_MODELS):
            # 不阻断执行：平台可能新增模型，仅提示
            pass
        if self.timeout <= 0:
            raise LLMConfigurationError(
                f"超时时间必须为正数，当前为 {self.timeout}（环境变量 {ENV_TIMEOUT}）"
            )
        if self.max_retries < 0:
            raise LLMConfigurationError(
                f"重试次数不能为负数，当前为 {self.max_retries}（环境变量 {ENV_MAX_RETRIES}）"
            )
        if not (0.0 <= self.temperature <= 2.0):
            raise LLMConfigurationError(
                f"temperature 应在 0~2 之间，当前为 {self.temperature}"
            )
        if self.max_tokens <= 0:
            raise LLMConfigurationError(
                f"max_tokens 必须为正整数，当前为 {self.max_tokens}"
            )
        if require_api_key and not self.has_api_key:
            raise MissingAPIKeyError(
                f"未检测到环境变量 {ENV_API_KEY}"
                + (f"（已读取配置文件：{self.env_file}）" if self.env_file else "（未找到 .env 文件）")
            )


def _to_float(value: str | None, default: float, env_name: str) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise LLMConfigurationError(
            f"环境变量 {env_name} 应为数字，当前值为 {value!r}"
        ) from exc


def _to_int(value: str | None, default: int, env_name: str) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise LLMConfigurationError(
            f"环境变量 {env_name} 应为整数，当前值为 {value!r}"
        ) from exc


def default_model_name() -> str:
    """返回默认模型名（供文档/测试引用，避免在别处硬编码）。"""

    return DEFAULT_MODEL


def describe_supported_env() -> list[str]:
    """返回本模块识别的环境变量清单，用于帮助信息与文档。"""

    return [
        ENV_API_KEY,
        ENV_BASE_URL,
        ENV_MODEL,
        ENV_TIMEOUT,
        ENV_MAX_RETRIES,
        ENV_TEMPERATURE,
        ENV_MAX_TOKENS,
        ENV_LOG_LEVEL,
    ]

