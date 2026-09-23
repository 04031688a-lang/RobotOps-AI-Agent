"""Phase 2：LLM 配置测试（模型名称集中管理、API Key 不硬编码）。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotops.llm import config as llm_config  # noqa: E402
from robotops.llm.config import LLMConfig, default_model_name, describe_supported_env  # noqa: E402
from robotops.llm.errors import LLMConfigurationError, MissingAPIKeyError  # noqa: E402

ENV_KEYS = (
    llm_config.ENV_API_KEY,
    llm_config.ENV_BASE_URL,
    llm_config.ENV_MODEL,
    llm_config.ENV_TIMEOUT,
    llm_config.ENV_MAX_RETRIES,
    llm_config.ENV_TEMPERATURE,
    llm_config.ENV_MAX_TOKENS,
    llm_config.ENV_LOG_LEVEL,
)


class LLMConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = dict(os.environ)
        for key in ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._snapshot)

    def test_defaults_come_from_config_module(self) -> None:
        config = LLMConfig()

        self.assertEqual(config.model, llm_config.DEFAULT_MODEL)
        self.assertEqual(config.model, default_model_name())
        self.assertEqual(config.base_url, llm_config.DEFAULT_BASE_URL)
        self.assertEqual(config.timeout, llm_config.DEFAULT_TIMEOUT_SECONDS)
        self.assertEqual(config.max_retries, llm_config.DEFAULT_MAX_RETRIES)
        self.assertEqual(config.response_format, "json_object")
        self.assertFalse(config.has_api_key)

    def test_model_name_is_not_hardcoded_outside_config(self) -> None:
        """模型名只允许出现在 config.py 中（提示词与业务代码不应包含模型名字面量）。"""

        project_root = Path(__file__).resolve().parents[1]
        offenders: list[str] = []
        for folder in ("robotops", "scripts", "tests"):
            for path in (project_root / folder).rglob("*.py"):
                if path.name in {"config.py"} and path.parent.name == "llm":
                    continue
                text = path.read_text(encoding="utf-8")
                if llm_config.DEFAULT_MODEL in text or "deepseek-reasoner" in text:
                    offenders.append(str(path.relative_to(project_root)))

        # 测试文件本身需要引用模型名做校验，这里只要求业务代码干净
        offenders = [item for item in offenders if not item.startswith("tests")]
        self.assertEqual(offenders, [], f"模型名散落在：{offenders}")

    def test_from_env_reads_all_supported_variables(self) -> None:
        os.environ.update(
            {
                llm_config.ENV_API_KEY: "sk-env-key",
                llm_config.ENV_BASE_URL: "https://api.deepseek.com",
                llm_config.ENV_MODEL: "deepseek-reasoner",
                llm_config.ENV_TIMEOUT: "90",
                llm_config.ENV_MAX_RETRIES: "4",
                llm_config.ENV_TEMPERATURE: "0.1",
                llm_config.ENV_MAX_TOKENS: "2048",
                llm_config.ENV_LOG_LEVEL: "DEBUG",
            }
        )
        with mock.patch.object(llm_config, "ensure_env_loaded", return_value=None):
            config = LLMConfig.from_env()

        self.assertEqual(config.api_key, "sk-env-key")
        self.assertEqual(config.model, "deepseek-reasoner")
        self.assertEqual(config.timeout, 90.0)
        self.assertEqual(config.max_retries, 4)
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.max_tokens, 2048)
        self.assertEqual(config.log_level, "DEBUG")
        config.validate()

    def test_from_env_rejects_invalid_number(self) -> None:
        os.environ[llm_config.ENV_TIMEOUT] = "不是数字"

        with mock.patch.object(llm_config, "ensure_env_loaded", return_value=None):
            with self.assertRaises(LLMConfigurationError) as ctx:
                LLMConfig.from_env()

        self.assertIn(llm_config.ENV_TIMEOUT, str(ctx.exception))

    def test_validate_requires_api_key(self) -> None:
        with self.assertRaises(MissingAPIKeyError) as ctx:
            LLMConfig().validate(require_api_key=True)

        self.assertIn(llm_config.ENV_API_KEY, str(ctx.exception))
        self.assertIn(".env", ctx.exception.hint)

    def test_validate_skips_api_key_when_not_required(self) -> None:
        LLMConfig().validate(require_api_key=False)  # 不应抛出

    def test_validate_rejects_bad_values(self) -> None:
        cases = [
            {"base_url": "ftp://api.deepseek.com"},
            {"model": "  "},
            {"timeout": 0},
            {"max_retries": -1},
            {"temperature": 3.0},
            {"max_tokens": 0},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(LLMConfigurationError):
                    LLMConfig(api_key="sk-x", **overrides).validate()

    def test_masked_key_and_public_dict(self) -> None:
        config = LLMConfig(api_key="sk-1234567890abcdef")

        self.assertTrue(config.has_api_key)
        self.assertNotIn("1234567890", config.masked_api_key)
        public = config.to_public_dict()
        self.assertNotIn("api_key_raw", public)
        self.assertNotIn("1234567890", str(public))
        self.assertEqual(public["model"], llm_config.DEFAULT_MODEL)

    def test_chat_url_normalises_trailing_slash(self) -> None:
        config = LLMConfig(base_url="https://api.deepseek.com/")
        self.assertEqual(config.chat_url, "https://api.deepseek.com/chat/completions")

    def test_describe_supported_env_lists_required_key(self) -> None:
        self.assertIn(llm_config.ENV_API_KEY, describe_supported_env())
        self.assertIn(llm_config.ENV_MODEL, describe_supported_env())


if __name__ == "__main__":
    unittest.main(verbosity=2)

