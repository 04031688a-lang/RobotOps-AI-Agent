"""Phase 2：DeepSeek 客户端测试（超时 / 重试 / 错误映射 / 空返回 / 日志脱敏）。

全部使用假传输层，不需要网络与真实 API Key。
"""

from __future__ import annotations

import json
import logging
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotops.llm.client import DeepSeekClient, HttpRequest, HttpResponse  # noqa: E402
from robotops.llm.config import LLMConfig  # noqa: E402
from robotops.llm.errors import (  # noqa: E402
    LLMAuthError,
    LLMEmptyResponseError,
    LLMInsufficientBalanceError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseFormatError,
    LLMServerError,
    LLMTimeoutError,
    MissingAPIKeyError,
)
from tests._helpers import FakeTransport, deepseek_error_response, deepseek_response  # noqa: E402

MESSAGES = [{"role": "user", "content": "分析这些数据"}]


def silent_logger(name: str = "robotops.llm.tests") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    return logger


class TransportTests(unittest.TestCase):
    def test_fake_transport_records_requests(self) -> None:
        transport = FakeTransport(deepseek_response("ok"))

        response = transport(HttpRequest(url="https://x", body=b"{}"))

        self.assertEqual(response.status, 200)
        self.assertEqual(transport.calls, 1)


class DeepSeekClientTests(unittest.TestCase):
    def _client(self, transport: FakeTransport, **overrides) -> DeepSeekClient:
        overrides.setdefault("api_key", "sk-test-key")
        overrides.setdefault("max_retries", 0)
        config = LLMConfig(**overrides)
        return DeepSeekClient(
            config=config,
            transport=transport,
            logger=silent_logger(),
            sleep=lambda seconds: None,
        )

    def test_successful_call_builds_openai_compatible_payload(self) -> None:
        transport = FakeTransport(deepseek_response("内容", model="deepseek-chat"))
        client = self._client(transport, model="deepseek-chat")

        result = client.chat(MESSAGES)

        self.assertEqual(result.content, "内容")
        self.assertEqual(result.model, "deepseek-chat")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.usage["total_tokens"], 1500)
        self.assertEqual(result.request_id, "req-test")

        body = transport.last_request_json
        self.assertEqual(body["model"], "deepseek-chat")
        self.assertEqual(body["messages"], MESSAGES)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertFalse(body["stream"])
        self.assertEqual(body["temperature"], LLMConfig().temperature)
        self.assertEqual(body["max_tokens"], LLMConfig().max_tokens)
        self.assertEqual(transport.requests[0].headers["Authorization"], "Bearer sk-test-key")
        self.assertEqual(transport.requests[0].url, client.config.chat_url)

    def test_model_comes_from_configuration(self) -> None:
        transport = FakeTransport(deepseek_response("x", model="custom-model"))
        client = self._client(transport, model="custom-model")

        client.chat(MESSAGES)

        self.assertEqual(transport.last_request_json["model"], "custom-model")

    def test_response_format_can_be_disabled(self) -> None:
        transport = FakeTransport(deepseek_response("x"))
        client = self._client(transport)

        client.chat(MESSAGES, response_format=None)

        self.assertNotIn("response_format", transport.last_request_json)

    def test_missing_api_key_blocks_request(self) -> None:
        transport = FakeTransport(deepseek_response("x"))
        client = DeepSeekClient(config=LLMConfig(api_key=None), transport=transport, logger=silent_logger())

        with self.assertRaises(MissingAPIKeyError):
            client.chat(MESSAGES)

        self.assertEqual(transport.calls, 0)

    def test_auth_error_is_not_retried(self) -> None:
        transport = FakeTransport(deepseek_error_response(401, "invalid api key"))
        client = self._client(transport, max_retries=3)

        with self.assertRaises(LLMAuthError) as ctx:
            client.chat(MESSAGES)

        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("invalid api key", str(ctx.exception))
        self.assertIn("DEEPSEEK_API_KEY", ctx.exception.hint)
        self.assertEqual(transport.calls, 1)

    def test_insufficient_balance_error(self) -> None:
        transport = FakeTransport(deepseek_error_response(402, "Insufficient Balance"))
        client = self._client(transport)

        with self.assertRaises(LLMInsufficientBalanceError):
            client.chat(MESSAGES)

    def test_bad_request_error_mentions_model(self) -> None:
        transport = FakeTransport(deepseek_error_response(400, "Model Not Exist"))
        client = self._client(transport)

        with self.assertRaises(LLMRequestError) as ctx:
            client.chat(MESSAGES)

        self.assertIn("DEEPSEEK_MODEL", ctx.exception.hint)

    def test_rate_limit_is_retried_then_succeeds(self) -> None:
        transport = FakeTransport(deepseek_error_response(429, "rate limit"), deepseek_response("ok"))
        client = self._client(transport, max_retries=2)

        result = client.chat(MESSAGES)

        self.assertEqual(result.attempts, 2)
        self.assertEqual(transport.calls, 2)

    def test_server_error_retries_until_exhausted(self) -> None:
        transport = FakeTransport(deepseek_error_response(503, "server busy"))
        client = self._client(transport, max_retries=2)

        with self.assertRaises(LLMServerError):
            client.chat(MESSAGES)

        self.assertEqual(transport.calls, 3)

    def test_timeout_is_retried_and_reported(self) -> None:
        transport = FakeTransport(LLMTimeoutError("请求 DeepSeek 超时（1 秒）"))
        client = self._client(transport, max_retries=1, timeout=1.0)

        with self.assertRaises(LLMTimeoutError) as ctx:
            client.chat(MESSAGES)

        self.assertEqual(transport.calls, 2)
        self.assertIn("DEEPSEEK_TIMEOUT", ctx.exception.hint)

    def test_network_error_is_reported(self) -> None:
        transport = FakeTransport(LLMNetworkError("无法连接 DeepSeek：getaddrinfo failed"))
        client = self._client(transport)

        with self.assertRaises(LLMNetworkError):
            client.chat(MESSAGES)

    def test_empty_content_raises(self) -> None:
        transport = FakeTransport(deepseek_response(""))
        client = self._client(transport)

        with self.assertRaises(LLMEmptyResponseError):
            client.chat(MESSAGES)

    def test_truncated_content_mentions_max_tokens(self) -> None:
        transport = FakeTransport(deepseek_response("", finish_reason="length"))
        client = self._client(transport)

        with self.assertRaises(LLMEmptyResponseError) as ctx:
            client.chat(MESSAGES)

        self.assertIn("DEEPSEEK_MAX_TOKENS", ctx.exception.hint)

    def test_non_json_body_raises_format_error(self) -> None:
        transport = FakeTransport(HttpResponse(status=200, body="<html>gateway</html>"))
        client = self._client(transport)

        with self.assertRaises(LLMResponseFormatError):
            client.chat(MESSAGES)

    def test_missing_choices_raises_empty_error(self) -> None:
        transport = FakeTransport(HttpResponse(status=200, body=json.dumps({"model": "deepseek-chat"})))
        client = self._client(transport)

        with self.assertRaises(LLMEmptyResponseError):
            client.chat(MESSAGES)

    def test_api_key_never_appears_in_logs(self) -> None:
        messages: list[str] = []
        handler = logging.Handler()
        handler.emit = lambda record: messages.append(record.getMessage())  # type: ignore[method-assign]
        logger = logging.getLogger("robotops.llm.mask")
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        secret = "sk-secret-key-value"
        transport = FakeTransport(deepseek_response("ok"))
        client = DeepSeekClient(
            config=LLMConfig(api_key=secret, max_retries=0),
            transport=transport,
            logger=logger,
            sleep=lambda seconds: None,
        )

        client.chat(MESSAGES)

        joined = " | ".join(messages)
        self.assertNotIn(secret, joined)
        self.assertIn("sk-s****alue", joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)

