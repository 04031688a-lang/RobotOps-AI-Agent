"""Phase 2：.env 读取测试（API Key 只能通过 .env/环境变量提供）。"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotops.llm import env_loader  # noqa: E402
from robotops.llm.env_loader import ENV_FILE_OVERRIDE_VAR, find_env_file, load_env_file, parse_env_text  # noqa: E402


class EnvParseTests(unittest.TestCase):
    def test_parse_basic_lines(self) -> None:
        values = parse_env_text(
            "\n".join(
                [
                    "# 注释行",
                    "",
                    "DEEPSEEK_API_KEY=sk-abc123",
                    "DEEPSEEK_MODEL = deepseek-chat ",
                    "  DEEPSEEK_TIMEOUT=90  ",
                ]
            )
        )

        self.assertEqual(values["DEEPSEEK_API_KEY"], "sk-abc123")
        self.assertEqual(values["DEEPSEEK_MODEL"], "deepseek-chat")
        self.assertEqual(values["DEEPSEEK_TIMEOUT"], "90")

    def test_parse_export_quotes_and_inline_comment(self) -> None:
        values = parse_env_text(
            "\n".join(
                [
                    'export DEEPSEEK_API_KEY="sk-quoted"',
                    "DEEPSEEK_MODEL='deepseek-reasoner'",
                    "DEEPSEEK_LOG_LEVEL=DEBUG  # 行尾注释",
                    "ROBOTOPS_ENV_FILE=D:\\config\\robotops.env",
                ]
            )
        )

        self.assertEqual(values["DEEPSEEK_API_KEY"], "sk-quoted")
        self.assertEqual(values["DEEPSEEK_MODEL"], "deepseek-reasoner")
        self.assertEqual(values["DEEPSEEK_LOG_LEVEL"], "DEBUG")
        self.assertEqual(values["ROBOTOPS_ENV_FILE"], "D:\\config\\robotops.env")

    def test_parse_skips_invalid_lines(self) -> None:
        values = parse_env_text("NO_EQUALS_SIGN\n=empty_key\nVALID=1")

        self.assertEqual(values, {"VALID": "1"})


class EnvFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._temp.name)
        self._snapshot = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._snapshot)
        self._temp.cleanup()

    def _write(self, name: str, text: str) -> Path:
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_load_env_file_sets_variables(self) -> None:
        path = self._write(".env", "DEEPSEEK_API_KEY=sk-test-123\nDEEPSEEK_MODEL=deepseek-chat\n")
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ.pop("DEEPSEEK_MODEL", None)

        loaded = load_env_file(path)

        self.assertEqual(loaded["DEEPSEEK_API_KEY"], "sk-test-123")
        self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "sk-test-123")

    def test_existing_environment_wins_by_default(self) -> None:
        path = self._write(".env", "DEEPSEEK_API_KEY=sk-from-file\n")
        os.environ["DEEPSEEK_API_KEY"] = "sk-from-system"

        load_env_file(path)

        self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "sk-from-system")

    def test_override_flag_replaces_existing(self) -> None:
        path = self._write(".env", "DEEPSEEK_API_KEY=sk-from-file\n")
        os.environ["DEEPSEEK_API_KEY"] = "sk-from-system"

        load_env_file(path, override=True)

        self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "sk-from-file")

    def test_find_env_file_uses_override_variable(self) -> None:
        path = self._write("custom.env", "DEEPSEEK_API_KEY=sk-custom\n")
        os.environ[ENV_FILE_OVERRIDE_VAR] = str(path)

        self.assertEqual(find_env_file(), path)
        os.environ.pop("DEEPSEEK_API_KEY", None)
        self.assertEqual(env_loader.ensure_env_loaded(), path)
        self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "sk-custom")

    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(load_env_file(self.tmp / "not-exists.env"), {})

    def test_ensure_env_loaded_returns_none_without_file(self) -> None:
        os.environ[ENV_FILE_OVERRIDE_VAR] = str(self.tmp / "not-exists.env")

        self.assertIsNone(env_loader.ensure_env_loaded(self.tmp / "also-not-exists.env"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
