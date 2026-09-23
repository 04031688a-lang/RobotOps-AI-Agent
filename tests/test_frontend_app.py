"""Phase 5：Streamlit 应用测试（AppTest 无头运行，覆盖上传/预览/分析/展示/下载）。

注意：测试中会关闭「启用 AI 分析」，使用离线模板模式运行工作流，
不消耗任何 API 额度，也不发起真实网络请求。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest  # noqa: E402

from app.graph.state import REQUIRED_STATE_KEYS  # noqa: E402
from robotops import config  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")


def run_app() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=180)
    app.run()
    return app


def exception_texts(app: AppTest) -> list[str]:
    return [str(item.value) for item in app.exception]


class FrontendSmokeTests(unittest.TestCase):
    def test_page_renders_without_exception(self) -> None:
        app = run_app()

        self.assertEqual(exception_texts(app), [])
        self.assertEqual(app.title[0].value, "RobotOps AI 机器人运营智能分析平台")
        self.assertIn("一、上传运营数据", [item.value for item in app.subheader])
        self.assertTrue(app.get("file_uploader"))

    def test_prompts_to_upload_when_no_file(self) -> None:
        app = run_app()

        messages = [item.value for item in app.info]
        self.assertTrue(any("请先上传运营数据文件" in message for message in messages))
        analyze_buttons = [button for button in app.button if button.label == "开始分析"]
        self.assertTrue(analyze_buttons)
        self.assertTrue(analyze_buttons[0].disabled, "未上传数据时「开始分析」应处于禁用状态")


class FrontendWorkflowTests(unittest.TestCase):
    """端到端：关闭 AI → 使用演示数据 → 开始分析 → 校验各区域展示与下载。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = run_app()
        # 关闭大模型（离线模板），避免测试消耗 API 额度
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

    def test_no_exception_after_full_flow(self) -> None:
        self.assertEqual(exception_texts(self.app), [])

    def test_preview_section_rendered(self) -> None:
        captions = " ".join(item.value for item in self.app.caption)
        self.assertIn("上表展示前", captions)
        subheaders = [item.value for item in self.app.subheader]
        self.assertIn("二、开始分析", subheaders)

    def test_result_sections_rendered(self) -> None:
        subheaders = [item.value for item in self.app.subheader]
        for title in ("三、运营概览", "四、异常项目", "五、AI 分析", "六、RAG 历史参考案例", "七、最终运营报告"):
            self.assertIn(title, subheaders)
        self.assertIn("历史参考案例，不代表当前项目实际情况", " ".join(item.value for item in self.app.caption))

    def test_overview_metrics_rendered(self) -> None:
        labels = [metric.label for metric in self.app.metric]
        for label in ("项目数量", "机器人数量", "平均运行时长", "故障率", "平均满意度", "总运营成本", "平均节降率"):
            self.assertIn(label, labels)

    def test_report_download_button_available(self) -> None:
        downloads = self.app.get("download_button")
        self.assertTrue(downloads, "应提供 Markdown 报告下载按钮")

    def test_workflow_state_saved_in_session(self) -> None:
        try:
            state = self.app.session_state["analysis_state"]
        except KeyError:  # pragma: no cover - 仅用于失败提示
            self.fail("session_state 中缺少 analysis_state，工作流结果未保存")
        self.assertEqual(state.get("workflow_status"), "completed")
        self.assertIn("final_report", state)
        self.assertTrue(state["final_report"])
        # 数据事实来自程序计算
        self.assertEqual(state["metrics"]["values"].get("项目数量"), 6)
        for key in REQUIRED_STATE_KEYS:  # 前端展示依赖的字段必须齐全
            self.assertIn(key, state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
