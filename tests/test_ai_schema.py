"""Phase 2：Agent 结构化输出解析与校验测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotops.llm.errors import LLMEmptyResponseError, LLMResponseFormatError  # noqa: E402
from robotops.llm.schema import AIAnalysis, parse_ai_analysis  # noqa: E402
from tests._helpers import sample_ai_json, sample_ai_payload  # noqa: E402


class ParseAIAnalysisTests(unittest.TestCase):
    def test_parse_plain_json(self) -> None:
        analysis = parse_ai_analysis(sample_ai_json(), model="deepseek-chat")

        self.assertIn("6 个项目", analysis.overview)
        self.assertEqual(len(analysis.key_findings), 1)
        self.assertEqual(len(analysis.abnormal_projects), 1)
        self.assertEqual(len(analysis.possible_reasons), 1)
        self.assertEqual(len(analysis.recommendations), 1)
        self.assertEqual(analysis.model, "deepseek-chat")
        self.assertEqual(analysis.warnings, [])

    def test_parse_json_in_code_fence(self) -> None:
        analysis = parse_ai_analysis(sample_ai_json(fenced=True))

        self.assertTrue(analysis.overview)
        self.assertEqual(len(analysis.recommendations), 1)

    def test_parse_json_with_surrounding_text(self) -> None:
        text = f"好的，以下是分析结果：\n{sample_ai_json()}\n希望有帮助。"

        analysis = parse_ai_analysis(text)

        self.assertTrue(analysis.overview)

    def test_empty_response_raises(self) -> None:
        for value in ("", "   ", None):
            with self.subTest(value=value):
                with self.assertRaises(LLMEmptyResponseError):
                    parse_ai_analysis(value)  # type: ignore[arg-type]

    def test_invalid_json_raises_with_raw_text(self) -> None:
        with self.assertRaises(LLMResponseFormatError) as ctx:
            parse_ai_analysis("这不是 JSON")

        self.assertEqual(ctx.exception.raw, "这不是 JSON")
        self.assertIn("deepseek_raw_response.txt", ctx.exception.hint)

    def test_non_object_json_raises(self) -> None:
        with self.assertRaises(LLMResponseFormatError):
            parse_ai_analysis("[1, 2, 3]")


class AIAnalysisModelTests(unittest.TestCase):
    def test_missing_fields_produce_warnings(self) -> None:
        analysis = AIAnalysis.from_payload({"overview": "概览"})

        self.assertEqual(analysis.overview, "概览")
        self.assertEqual(analysis.key_findings, [])
        self.assertTrue(any("缺少字段" in item for item in analysis.warnings))
        self.assertTrue(any("key_findings 为空数组" in item for item in analysis.warnings))

    def test_string_sections_are_wrapped(self) -> None:
        payload = sample_ai_payload()
        payload["key_findings"] = "只有一句话的结论"

        analysis = AIAnalysis.from_payload(payload)

        self.assertEqual(analysis.key_findings, ["只有一句话的结论"])
        self.assertTrue(any("不是数组" in item for item in analysis.warnings))

    def test_extra_keys_are_ignored_with_warning(self) -> None:
        payload = sample_ai_payload()
        payload["extra"] = "无用字段"

        analysis = AIAnalysis.from_payload(payload)

        self.assertTrue(any("额外字段" in item for item in analysis.warnings))
        self.assertNotIn("extra", analysis.to_dict())

    def test_non_string_overview_is_converted(self) -> None:
        analysis = AIAnalysis.from_payload({"overview": {"a": 1}})

        self.assertIsInstance(analysis.overview, str)
        self.assertTrue(any("overview 不是字符串" in item for item in analysis.warnings))

    def test_to_dict_structure(self) -> None:
        analysis = parse_ai_analysis(sample_ai_json(), model="deepseek-chat", usage={"total_tokens": 10})
        data = analysis.to_dict()

        self.assertEqual(
            set(data.keys()),
            {
                "overview",
                "key_findings",
                "abnormal_projects",
                "possible_reasons",
                "recommendations",
                "meta",
            },
        )
        self.assertEqual(data["meta"]["model"], "deepseek-chat")
        self.assertEqual(data["meta"]["usage"]["total_tokens"], 10)
        self.assertNotIn("raw_text", data["meta"])
        self.assertIn("raw_text", analysis.to_dict(include_raw=True)["meta"])

    def test_markdown_and_console_render_all_sections(self) -> None:
        analysis = parse_ai_analysis(sample_ai_json())

        markdown = analysis.to_markdown()
        console = analysis.to_console_text()

        for title in ("运营概览", "关键发现", "异常发现", "可能原因", "优化建议"):
            self.assertIn(title, markdown)
            self.assertIn(title, console)
        self.assertIn("证据", console)  # 字段中文标签
        self.assertIn("当前数据不足以判断", console)

    def test_is_empty(self) -> None:
        self.assertTrue(AIAnalysis().is_empty)
        self.assertFalse(parse_ai_analysis(sample_ai_json()).is_empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)

