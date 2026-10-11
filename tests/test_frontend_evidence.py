"""Phase 10 前端测试：「分析依据」区域（无头渲染）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")


class EvidencePanelPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = AppTest.from_file(APP_PATH, default_timeout=180)
        cls.app.run()
        for toggle in cls.app.toggle:
            if toggle.label == "启用 AI 分析（DeepSeek）":
                toggle.set_value(False)
        cls.app.run()
        for button in cls.app.button:
            if button.label == "使用项目演示数据试跑":
                button.click()
        cls.app.run()
        for button in cls.app.button:
            if button.label == "开始分析":
                button.click()
        cls.app.run()

    def test_no_exception(self) -> None:
        self.assertEqual([str(item.value) for item in self.app.exception], [])
        self.assertEqual([str(item.value) for item in self.app.error], [])

    def test_evidence_section_rendered(self) -> None:
        self.assertIn("分析依据", [item.value for item in self.app.subheader])

    def test_three_sections_visible(self) -> None:
        markdown = " ".join(item.value for item in self.app.markdown)

        self.assertIn("当前数据 / 程序计算结果", markdown)
        self.assertIn("历史案例（RAG）", markdown)
        self.assertIn("可能原因（推测）", markdown)

    def test_current_data_and_program_results_listed(self) -> None:
        markdown = " ".join(item.value for item in self.app.markdown)

        self.assertIn("当前故障率", markdown)
        self.assertIn("程序按阈值判定", markdown)

    def test_possible_reasons_marked_as_inference(self) -> None:
        frames = [item.value for item in self.app.dataframe]
        reason_frames = [
            frame for frame in frames
            if "可能原因（推测）" in list(getattr(frame, "columns", []))
        ]

        self.assertTrue(reason_frames, "应展示「可能原因（推测）」表格")
        captions = " ".join(item.value for item in self.app.caption)
        self.assertIn("推测", captions)
        self.assertIn("不代表已确认的事实", captions)

    def test_case_sources_shown(self) -> None:
        markdown = " ".join(item.value for item in self.app.markdown)
        frames = [item.value for item in self.app.dataframe]
        case_frames = [
            frame for frame in frames
            if {"案例编号", "相似度", "来源文件"}.issubset(set(getattr(frame, "columns", [])))
        ]

        self.assertTrue(case_frames or "历史案例", markdown)

    def test_state_carries_evidence(self) -> None:
        state = self.app.session_state["analysis_state"]
        diagnosis = state.get("diagnosis") or {}

        self.assertIn("findings", diagnosis)
        self.assertTrue(diagnosis["findings"])
        self.assertTrue((diagnosis.get("possible_reasons") or [{}])[0].get("evidence"))
        self.assertTrue((diagnosis.get("possible_reasons") or [{}])[0].get("basis"))
        for case in state.get("retrieved_cases") or []:
            for field in ("case_id", "title", "category", "similarity_score", "source_file", "content"):
                self.assertIn(field, case)


if __name__ == "__main__":
    unittest.main(verbosity=2)

