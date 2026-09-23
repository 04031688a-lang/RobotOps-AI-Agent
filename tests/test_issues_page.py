"""Phase 6：运营问题页面测试（AppTest 无头运行，覆盖列表 / 详情 / 处理 / 验证入口）。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from frontend import issues_helpers as ih  # noqa: E402
from robotops.issues import IssueService, IssueStatus, IssueStore  # noqa: E402
from tests._helpers import cleanup_dir, make_temp_dir  # noqa: E402
from tests.test_issues import build_state, with_after_metrics  # noqa: E402

ISSUES_VIEW = str(Path(__file__).resolve().parents[1] / "frontend" / "views" / "issues.py")
#: 真实入口（单入口 + 侧边栏模块切换）
FRONTEND_ENTRY = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")


class IssuesPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = make_temp_dir(prefix="robotops_test_issues_page_")
        os.environ["ROBOTOPS_ISSUES_DB"] = str(cls.temp_dir / "issues.db")
        store = IssueStore(cls.temp_dir / "issues.db")
        service = IssueService(store)
        cls.pending = service.create_from_state(build_state(), owner="售后运维")[0]
        cls.verifying = service.create_from_state(build_state(), owner="售后运维")[0]
        service.update_status(cls.verifying.issue_id, IssueStatus.IN_PROGRESS.value)
        service.set_resolution(cls.verifying.issue_id, "更换部件并调整作业区域")
        service.update_status(cls.verifying.issue_id, IssueStatus.PENDING_VERIFICATION.value)
        store.close()

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop("ROBOTOPS_ISSUES_DB", None)
        # 释放页面缓存的 SQLite 连接后再清理临时目录（Windows 下文件被占用会删除失败）
        st.cache_resource.clear()
        st.cache_data.clear()
        cleanup_dir(cls.temp_dir)

    def _open(self) -> AppTest:
        app = AppTest.from_file(ISSUES_VIEW, default_timeout=120)
        app.run()
        return app

    def _exceptions(self, app: AppTest) -> list[str]:
        return [str(item.value) for item in app.exception]

    @staticmethod
    def _option_for(selectbox, issue_id: str) -> str:
        """从控件自身的选项列表中选出目标问题（避免手写标签不一致）。"""

        return next(option for option in selectbox.options if issue_id in str(option))

    def test_page_renders_with_issues(self) -> None:
        app = self._open()

        self.assertEqual(self._exceptions(app), [])
        self.assertEqual(app.title[0].value, "运营问题")
        subheaders = [item.value for item in app.subheader]
        for title in ("一、问题统计", "二、问题列表", "三、问题详情", "四、问题处理"):
            self.assertTrue(
                any(item.startswith(title) for item in subheaders), f"缺少 {title}"
            )

    def test_statistics_cards(self) -> None:
        app = self._open()

        labels = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(labels.get("总问题数"), "2")
        self.assertEqual(labels.get("待处理"), "1")
        self.assertEqual(labels.get("问题关闭率"), "0.0%")
        self.assertIn("平均处理周期(天)", labels)

    def test_issue_list_contains_issue_ids(self) -> None:
        app = self._open()

        frames = app.dataframe
        self.assertTrue(frames, "应渲染问题列表")
        first = frames[0].value
        self.assertIn("问题编号", list(first.columns))
        ids = set(first["问题编号"])
        self.assertEqual(ids, {self.pending.issue_id, self.verifying.issue_id})

    def test_detail_sections_rendered(self) -> None:
        app = self._open()

        markdown_text = "\n".join(item.value for item in app.markdown)
        for label in ("异常指标", "当前数据（项目维度）", "AI 诊断结果（推测）", "AI 建议", "处理结果"):
            self.assertIn(label, markdown_text)
        self.assertIn("RAG 历史案例", markdown_text)
        self.assertIn(
            "历史参考案例，不代表当前项目实际情况",
            "\n".join(item.value for item in app.caption),
        )
        self.assertTrue(any("B小区" in item.value for item in app.markdown))

    def test_status_form_updates_issue(self) -> None:
        app = self._open()

        # 先选中「待处理」的问题，再修改状态
        selected = next(item for item in app.selectbox if item.label == "选择问题编号查看详情")
        selected.set_value(self._option_for(selected, self.pending.issue_id))
        app.run()
        status_box = next(item for item in app.selectbox if item.label == "状态")
        self.assertEqual(list(status_box.options), ["待处理", "处理中", "已关闭"])
        status_box.set_value("处理中")
        next(button for button in app.button if button.label == "保存更新").click()
        app.run()

        self.assertEqual(self._exceptions(app), [])
        store = IssueStore(self.temp_dir / "issues.db")
        try:
            status = store.get_required(self.pending.issue_id).status
        finally:
            store.close()
        self.assertEqual(status, IssueStatus.IN_PROGRESS.value)

    def test_verification_upload_visible_for_pending_verification(self) -> None:
        app = self._open()

        selected = next(item for item in app.selectbox if item.label == "选择问题编号查看详情")
        selected.set_value(self._option_for(selected, self.verifying.issue_id))
        app.run()

        subheaders = [item.value for item in app.subheader]
        self.assertIn("五、整改效果验证", subheaders)
        self.assertTrue(app.get("file_uploader"), "待验证状态应显示整改后数据上传控件")
        self.assertTrue(any("验证整改效果" == button.label for button in app.button))
        self.assertIn(
            "改善幅度 = |整改前 - 整改后| ÷ |整改前|",
            "\n".join(item.value for item in app.caption),
        )

    def test_verification_not_shown_for_pending_issue(self) -> None:
        app = self._open()

        selected = next(item for item in app.selectbox if item.label == "选择问题编号查看详情")
        selected.set_value(self._option_for(selected, self.pending.issue_id))
        app.run()

        self.assertNotIn("五、整改效果验证", [item.value for item in app.subheader])
        self.assertFalse(app.get("file_uploader"))

    def test_helper_frames(self) -> None:
        store = IssueStore(self.temp_dir / "issues.db")
        try:
            issue = IssueService(store).get(self.verifying.issue_id)
        finally:
            store.close()

        metrics = ih.metrics_frame(issue)
        current = ih.current_data_frame(issue)
        notes = ih.note_rows(issue)
        cases = ih.case_rows(issue)

        self.assertIn("异常类型", metrics.columns)
        self.assertIn("最差设备", metrics.columns)
        self.assertIn("项目", set(current["数据项"]))
        self.assertIn("B小区", set(current["数值"]))
        self.assertTrue(all(isinstance(value, str) for value in current["数值"]))
        self.assertIn("处理结果", set(notes["类型"]))
        self.assertEqual(set(cases["案例编号"]), {"CASE-FAULT-001", "CASE-SAT-001"})
        self.assertEqual(ih.verification_frame({}).columns.tolist()[0], "指标")
        cards = {card["label"]: card["value"] for card in ih.stats_cards({"问题关闭率": 0.5})}
        self.assertEqual(cards["问题关闭率"], "50.0%")
        self.assertEqual(cards["平均处理周期(天)"], "—")


class AnalysisPageIssueEntryTests(unittest.TestCase):
    """分析工作台必须提供「创建运营问题」入口（Phase 6 需求三）。

    做法：直接把一份已完成的分析结果写入会话状态，再渲染页面并点击创建按钮，
    这样测试聚焦在「入口 + 创建逻辑」，不需要在 UI 里重跑一遍完整工作流。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = make_temp_dir(prefix="robotops_test_issues_entry_")
        os.environ["ROBOTOPS_ISSUES_DB"] = str(cls.temp_dir / "issues.db")

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop("ROBOTOPS_ISSUES_DB", None)
        st.cache_resource.clear()
        st.cache_data.clear()
        cleanup_dir(cls.temp_dir)

    def test_create_issue_entry_writes_issue(self) -> None:
        state = build_state() | {
            "has_anomalies": True,
            "anomaly_count": 3,
            "metrics": {
                "values": {
                    "项目数量": 1,
                    "机器人数量": 2,
                    "平均运行时长": 7.5,
                    "故障率": 8.2,
                    "平均满意度": 81.0,
                    "总运营成本": 30000.0,
                    "平均节降率": 8.0,
                },
                "units": {"故障率": "%", "平均节降率": "%", "平均满意度": "分"},
            },
            "anomaly_summary": build_state()["anomaly_summary"],
            "steps": [],
            "errors": [],
            "final_report": "# 测试报告\n\n## 三、异常发现\n\n示例。\n",
            "workflow_status": "completed",
        }
        app = AppTest.from_file(FRONTEND_ENTRY, default_timeout=300)
        app.session_state["analysis_state"] = state
        app.session_state["analysis_logs"] = ["[Data Analysis Agent] started"]
        app.session_state["analysis_source"] = "B小区-运营数据.xlsx"
        app.run()

        self.assertEqual([str(item.value) for item in app.exception], [])
        self.assertIn("八、运营问题", [item.value for item in app.subheader])
        self.assertTrue(any(button.label == "创建运营问题" for button in app.button))

        # 点击创建 → 数据库中出现运营问题
        next(button for button in app.button if button.label == "创建运营问题").click()
        app.run()
        self.assertEqual([str(item.value) for item in app.exception], [])

        store = IssueStore(self.temp_dir / "issues.db")
        try:
            issues = IssueService(store).list_issues()
        finally:
            store.close()
        self.assertTrue(issues, "点击「创建运营问题」后应写入运营问题")
        self.assertEqual(issues[0].status, IssueStatus.PENDING.value)
        self.assertTrue(issues[0].project)

        # 清理本次创建的问题，避免影响其他用例
        store = IssueStore(self.temp_dir / "issues.db")
        try:
            for issue in IssueService(store).list_issues():
                store.delete_issue(issue.issue_id)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
