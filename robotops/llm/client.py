"""RobotOps AI —— Phase 2 DeepSeek API 客户端。

DeepSeek 提供 OpenAI 兼容的 Chat Completions 接口，本模块使用 Python 标准库
``urllib`` 实现调用，因此 **Phase 2 不需要新增任何第三方依赖**。

能力：
- 请求封装（messages、temperature、max_tokens、JSON 输出格式）；
- 超时控制（连接与读取共用同一 timeout，可通过 DEEPSEEK_TIMEOUT 调整）；
- 自动重试（限流 / 服务端错误 / 超时 / 网络错误，指数退避）；
- 错误映射（401/402/403/404/429/5xx 分别给出可执行的排查建议）；
- 基础日志（请求参数、耗时、Token 用量、重试次数，密钥脱敏）。

测试友好：``transport`` 参数可注入假的 HTTP 传输层，从而在无网络、无 API Key
的情况下完整测试调用逻辑。
"""

from __future__ import annotations

import json
import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .config import (
    RETRY_BACKOFF_BASE_SECONDS,
    RETRY_BACKOFF_MAX_SECONDS,
    USER_AGENT,
    LLMConfig,
)
from .errors import (
    LLMAuthError,
    LLMEmptyResponseError,
    LLMError,
    LLMInsufficientBalanceError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseFormatError,
    LLMServerError,
    LLMTimeoutError,
)
from .logger import get_llm_logger, mask_secret

_UNSET: Any = object()


@dataclass(frozen=True)
class HttpRequest:
    """与具体 HTTP 库无关的请求对象（便于测试替身）。"""

    url: str
    method: str = "POST"
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""
    timeout: float = 60.0


@dataclass(frozen=True)
class HttpResponse:
    """与具体 HTTP 库无关的响应对象。"""

    status: int
    body: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)

    def json(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.body) if self.body else None
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None


Transport = Callable[[HttpRequest], HttpResponse]


def urllib_transport(request: HttpRequest) -> HttpResponse:
    """默认传输层：使用标准库 urllib 发起 HTTPS 请求。

    Raises:
        LLMTimeoutError: 请求超时。
        LLMNetworkError: DNS / 连接 / TLS / 代理等网络错误。
    """

    http_request = urllib.request.Request(
        url=request.url,
        data=request.body or None,
        headers=dict(request.headers),
        method=request.method,
    )
    try:
        with urllib.request.urlopen(http_request, timeout=request.timeout) as response:
            return HttpResponse(
                status=int(getattr(response, "status", 200)),
                body=response.read().decode("utf-8", errors="replace"),
                headers={key: value for key, value in dict(response.headers).items()},
            )
    except urllib.error.HTTPError as error:
        body = ""
        try:
            body = error.read().decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - 读取失败时保持空 body
            body = ""
        return HttpResponse(
            status=int(getattr(error, "code", 0) or 0),
            body=body,
            headers={key: value for key, value in dict(error.headers or {}).items()},
        )
    except socket.timeout as error:
        raise LLMTimeoutError(
            f"请求 DeepSeek 超时（{request.timeout:g} 秒）"
        ) from error
    except urllib.error.URLError as error:
        reason = getattr(error, "reason", error)
        if isinstance(reason, socket.timeout):
            raise LLMTimeoutError(
                f"连接 DeepSeek 超时（{request.timeout:g} 秒）"
            ) from error
        raise LLMNetworkError(f"无法连接 DeepSeek：{reason}") from error
    except ssl.SSLError as error:
        raise LLMNetworkError(f"TLS/证书校验失败：{error}") from error
    except OSError as error:
        raise LLMNetworkError(f"网络异常：{error}") from error


@dataclass
class ChatResult:
    """一次成功的对话补全结果。"""

    content: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    attempts: int = 1
    elapsed_seconds: float = 0.0
    request_id: str | None = None


class DeepSeekClient:
    """DeepSeek Chat Completions 客户端。"""

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        transport: Transport | None = None,
        logger: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or LLMConfig.from_env()
        self._transport = transport or urllib_transport
        self._logger = logger or get_llm_logger()
        self._sleep = sleep

    # -- 属性 -------------------------------------------------------------
    @property
    def model(self) -> str:
        return self.config.model

    # -- 主流程 -----------------------------------------------------------
    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        response_format: Any = _UNSET,
        temperature: float | None = None,
        max_tokens: int | None = None,
        require_api_key: bool = True,
    ) -> ChatResult:
        """调用 Chat Completions 接口，返回模型回复。

        Args:
            messages: OpenAI 兼容的 messages 列表。
            response_format: 覆盖默认的返回格式（``None`` 表示不使用 JSON 模式）。
            temperature: 覆盖默认温度。
            max_tokens: 覆盖默认最大输出长度。
            require_api_key: 是否强制要求配置 API Key。

        Raises:
            MissingAPIKeyError / LLMConfigurationError: 配置问题。
            LLMAuthError / LLMRequestError / LLMRateLimitError / LLMServerError /
            LLMTimeoutError / LLMNetworkError / LLMEmptyResponseError /
            LLMResponseFormatError: 调用与解析问题。
        """

        self.config.validate(require_api_key=require_api_key)

        if not messages:
            raise LLMRequestError("messages 不能为空")

        payload = self._build_payload(
            messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        attempts_allowed = self.config.max_retries + 1
        last_error: LLMError | None = None

        self._logger.debug(
            "DeepSeek 配置：model=%s base_url=%s timeout=%ss retries=%s api_key=%s",
            self.config.model,
            self.config.base_url,
            self.config.timeout,
            self.config.max_retries,
            mask_secret(self.config.api_key),
        )

        for attempt in range(1, attempts_allowed + 1):
            started = time.perf_counter()
            self._logger.info(
                "调用 DeepSeek：model=%s attempt=%s/%s messages=%s timeout=%ss",
                self.config.model,
                attempt,
                attempts_allowed,
                len(messages),
                self.config.timeout,
            )
            try:
                result = self._request_once(payload, attempt=attempt, started=started)
                return result
            except (LLMTimeoutError, LLMNetworkError, LLMRateLimitError, LLMServerError) as error:
                last_error = error
                elapsed = time.perf_counter() - started
                if attempt >= attempts_allowed:
                    self._logger.error(
                        "调用失败（已重试 %s 次）：%s | 耗时=%.2fs",
                        attempt - 1,
                        error,
                        elapsed,
                    )
                    break
                delay = _backoff_delay(attempt)
                self._logger.warning(
                    "调用失败将重试：%s | 第 %s 次重试，等待 %.1fs",
                    error,
                    attempt,
                    delay,
                )
                self._sleep(delay)
            except LLMError as error:
                # 不可重试的错误（密钥、参数、余额、解析等）直接抛出
                self._logger.error("调用失败（不重试）：%s", error)
                raise

        assert last_error is not None  # 循环结束时必然有错误
        raise last_error

    # -- 内部实现 ---------------------------------------------------------
    def _build_payload(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        response_format: Any,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        fmt = self.config.response_format if response_format is _UNSET else response_format
        if fmt:
            payload["response_format"] = {"type": fmt}
        return payload

    def _request_once(
        self, payload: dict[str, Any], *, attempt: int, started: float
    ) -> ChatResult:
        request = HttpRequest(
            url=self.config.chat_url,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
            body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=self.config.timeout,
        )

        response = self._transport(request)
        elapsed = time.perf_counter() - started
        self._ensure_success(response)

        data = response.json()
        if data is None:
            raise LLMResponseFormatError(
                "DeepSeek 返回内容不是合法 JSON", raw=response.body, status_code=response.status
            )

        content, model_name, usage, finish_reason = _extract_choice(data)
        if not content or not str(content).strip():
            if finish_reason == "length":
                raise LLMEmptyResponseError(
                    "DeepSeek 返回内容为空（输出长度达到 max_tokens 上限被截断）",
                    hint="请调大 DEEPSEEK_MAX_TOKENS 或减小传入的分析数据规模。",
                )
            raise LLMEmptyResponseError("DeepSeek 返回内容为空")

        self._logger.info(
            "调用成功：model=%s 耗时=%.2fs tokens=%s finish_reason=%s",
            model_name or self.config.model,
            elapsed,
            usage.get("total_tokens", "-"),
            finish_reason or "-",
        )
        return ChatResult(
            content=str(content),
            model=model_name or self.config.model,
            usage=usage,
            finish_reason=finish_reason,
            attempts=attempt,
            elapsed_seconds=elapsed,
            request_id=response.headers.get("x-request-id") if response.headers else None,
        )

    def _ensure_success(self, response: HttpResponse) -> None:
        status = int(response.status)
        if 200 <= status < 300:
            return

        detail = _extract_error_message(response)
        if status in (401, 403):
            raise LLMAuthError(f"认证失败（HTTP {status}）：{detail}", status_code=status, raw=response.body)
        if status == 402:
            raise LLMInsufficientBalanceError(
                f"账户余额不足（HTTP {status}）：{detail}", status_code=status, raw=response.body
            )
        if status == 429:
            raise LLMRateLimitError(
                f"触发限流（HTTP {status}）：{detail}", status_code=status, raw=response.body
            )
        if status in (400, 404, 405, 422):
            raise LLMRequestError(
                f"请求被拒绝（HTTP {status}）：{detail}", status_code=status, raw=response.body
            )
        if 500 <= status < 600:
            raise LLMServerError(
                f"DeepSeek 服务端错误（HTTP {status}）：{detail}",
                status_code=status,
                raw=response.body,
            )
        raise LLMError(
            f"调用失败（HTTP {status}）：{detail}", status_code=status, raw=response.body
        )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _backoff_delay(attempt: int) -> float:
    """指数退避：1s、2s、4s…… 上限 8s。"""

    return min(RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX_SECONDS)


def _extract_choice(
    data: Mapping[str, Any],
) -> tuple[str | None, str | None, dict[str, Any], str | None]:
    """从返回体中取出内容、模型名、Token 用量与结束原因。"""

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, data.get("model"), _usage_of(data), None

    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    if content is None:
        content = first.get("text")
    return (
        content if isinstance(content, str) else None,
        data.get("model") if isinstance(data.get("model"), str) else None,
        _usage_of(data),
        first.get("finish_reason") if isinstance(first.get("finish_reason"), str) else None,
    )


def _usage_of(data: Mapping[str, Any]) -> dict[str, Any]:
    usage = data.get("usage")
    return dict(usage) if isinstance(usage, Mapping) else {}


def _extract_error_message(response: HttpResponse) -> str:
    data = response.json()
    if isinstance(data, Mapping):
        error = data.get("error")
        if isinstance(error, Mapping):
            for key in ("message", "type", "code"):
                value = error.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        message = data.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

    text = (response.body or "").strip().replace("\n", " ")
    return text[:300] if text else "（无返回内容）"

