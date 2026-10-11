"""Phase 10：AI 分析依据（evidence）、信息类型区分与容错测试。

覆盖需求场景：
1. 正常路径：findings / evidence / 建议依据完整；
2. LLM 返回异常（缺字段、字段类型错误、非法结构）时优雅降级；
3. 数据不足时明确说明，不编造；
4. 报告与前端对「当前数据 / 程序计算 / 历史案例 / 可能原因 / 建议」的区分；
5. RAG 无高相关案例时的统一提示。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.diagnosis import (  # noqa: E402
    _normalize_findings,
    _normalize_reasons,
    build_findings,
    build_offline_diagnosis,
    program_facts,
)
from app.agents.recommendation import _normalize_items, build_offline_recommendations  # noqa: E402
from app.agents.report import ReportAgent  # noqa: E402
from app.graph.logging_utils import WorkflowLogger  # noqa: E402
from app.graph.prompts import (  # noqa: E402
    COMMON_RULES,
    DIAGNOSIS_SYSTEM_PROMPT,
    RECOMMENDATION_SYSTEM_PROMPT,
    REPORT_SYSTEM_PROMPT,
    build_diagnosis_prompt,
)
from frontend import evidence_helpers as uie  # noqa: E402
from robotops.rag.config import NO_RELEVANT_CASE_MESSAGE  # noqa: E402
from tests._helpers import cleanup_dir, make_temp_dir  # noqa: E402


def rich_state() -> dict:
    """一个包含异常、趋势、项目对比与历史案例的完整 state。"""

    return {
        "metrics": {
            "values": {"故障率": 8.2, "平均满意度": 81.5, "平均节降率": 6.4, "总运营成本": 120000.0},
            "units": {"故障率": "%", "平均满意度": "分"},
        },
        "anomaly_count": 3,
        "anomaly_summary": [
            {"anomaly_type": "故障率偏高", "hits": 3, "condition": "> 5%"},
            {"anomaly_type": "满意度偏低", "hits": 2, "condition": "< 85分"},
        ],
        "trend_analysis": {"highlights": ["故障率最新（2026-W35）为 8.2%，环比上升 72.3%"]},
        "project_comparison": {"highlights": ["故障率：最低 A小区 4.3%；最高 B小区 8.2%"]},
        "retrieved_cases": [
            {
                "case_id": "CASE-FAULT-001",
                "title": "清洁机器人重复故障（同一部件反复损坏）",
                "category": "equipment_fault",
                "similarity_score": 0.87,
                "source_file": "CASE-FAULT-001.md",
                "content": "【问题】某园区清洁机器人两周内出现 6 次同类故障……",
                "data_nature": "模拟案例（虚构）",
            }
        ],
        "retrieval": {"message": "", "is_relevant": True},
        "has_anomalies": True,
    }


def rich_state_with_diagnosis() -> dict:
    """在 rich_state 基础上补充诊断结论与优化建议（用于报告与前端展示测试）。"""

    state = rich_state()
    state["diagnosis"] = {
        "source": "offline_template",
        "summary": "以下原因均为推测。",
        "findings": [
            {
                "finding": "B项目故障率异常",
                "basis": "当前数据",
                "evidence": ["当前故障率 8.2", "项目平均故障率 4.3", "较上周期上升 72.3%"],
            }
        ],
        "possible_reasons": [
            {
                "reason": "设备维护周期过长",
                "confidence": "中",
                "basis": "当前数据",
                "evidence": ["当前故障率 8.2", "项目平均故障率 4.3"],
                "based_on": "故障率与维修次数同步上升",
                "data_gap": "缺少维修工单数据",
            }
        ],
    }
    state["recommendations"] = {
        "items": [
            {
                "action": "缩短维保周期并复核备件批次",
                "priority": "高",
                "target": "高故障项目",
                "expected_effect": "减少重复故障",
                "verification": "跟踪故障率变化",
                "evidence": ["当前故障率 8.2"],
                "reference_case": "CASE-FAULT-001",
            }
        ]
    }
    return state


class ProgramFactsTests(unittest.TestCase):
    def test_facts_are_grouped_by_source(self) -> None:
        facts = program_facts(rich_state())

        self.assertTrue(facts["current"])
        self.assertTrue(facts["program"])
        self.assertTrue(facts["trend"])
        self.assertTrue(facts["comparison"])
        self.assertTrue(facts["cases"])
        self.assertIn("当前故障率 8.2", facts["current"][0])

    def test_findings_carry_evidence(self) -> None:
        findings = build_findings(rich_state())

        self.assertTrue(findings)
        for item in findings:
            self.assertTrue(item["finding"])
            self.assertTrue(item["basis"])
            self.assertTrue(item["evidence"])

    def test_facts_without_data_do_not_crash(self) -> None:
        facts = program_facts({})

        self.assertIn("未发现超过阈值的异常", facts["program"][0])
        self.assertTrue(build_findings({}))


class DiagnosisEvidenceTests(unittest.TestCase):
    def test_offline_diagnosis_has_findings_and_evidence(self) -> None:
        state = rich_state()
        state["use_llm"] = False

        diagnosis = build_offline_diagnosis(state, note="测试")

        self.assertTrue(diagnosis["findings"])
        self.assertTrue(diagnosis["possible_reasons"])
        for reason in diagnosis["possible_reasons"]:
            self.assertTrue(reason["evidence"], "每条可能原因都应带 evidence")
            self.assertEqual(reason["basis"], "程序判定")
        self.assertIn("evidence", diagnosis)

    def test_normalize_reasons_keeps_evidence_and_fills_missing(self) -> None:
        # 正常返回
        value = [
            {
                "reason": "维护周期过长",
                "confidence": "中",
                "basis": "当前数据",
                "evidence": ["当前故障率 8.2%", 123],
                "based_on": "故障率高于阈值",
                "data_gap": "缺少工单",
            }
        ]
        reasons = _normalize_reasons(value, state=rich_state())

        self.assertEqual(reasons[0]["evidence"], ["当前故障率 8.2%", "123"])
        self.assertEqual(reasons[0]["basis"], "当前数据")

        # 模型没给 evidence -> 用程序事实兜底（LLM 返回异常场景）
        fallback = _normalize_reasons([{"reason": "充电策略不当"}], state=rich_state())
        self.assertTrue(fallback[0]["evidence"])

        # 字符串形式的 evidence 也能接受
        text_evidence = _normalize_reasons([{"reason": "x", "evidence": "只有一条依据"}])
        self.assertEqual(text_evidence[0]["evidence"], ["只有一条依据"])

        # 非法结构不应崩溃
        self.assertEqual(_normalize_reasons(None), [])
        self.assertEqual(_normalize_reasons({"reason": "不是数组"}), [])
        wrapped = _normalize_reasons(["纯字符串原因"])
        self.assertEqual(wrapped[0]["reason"], "纯字符串原因")
        self.assertTrue(wrapped[0]["evidence"] is not None)

    def test_normalize_findings_handles_bad_input(self) -> None:
        self.assertEqual(_normalize_findings(None), [])
        self.assertEqual(_normalize_findings("不是数组"), [])
        self.assertEqual(_normalize_findings([{"basis": "当前数据"}]), [])  # 缺 finding

        findings = _normalize_findings(
            [{"finding": "B项目故障率异常", "basis": "当前数据", "evidence": ["当前故障率 8.2%"]}]
        )
        self.assertEqual(findings[0]["finding"], "B项目故障率异常")
        self.assertEqual(findings[0]["evidence"], ["当前故障率 8.2%"])


class RecommendationEvidenceTests(unittest.TestCase):
    def test_offline_recommendations_have_evidence(self) -> None:
        state = rich_state()
        state["use_llm"] = False

        recommendations = build_offline_recommendations(state, note="测试")

        self.assertTrue(recommendations["items"])
        for item in recommendations["items"]:
            self.assertTrue(item["evidence"])
            self.assertEqual(item["basis"], "程序判定")

    def test_normalize_items_keeps_and_fills_evidence(self) -> None:
        state = rich_state()
        items = _normalize_items([{"action": "缩短维保周期", "evidence": ["当前故障率 8.2%"]}], state=state)
        self.assertEqual(items[0]["evidence"], ["当前故障率 8.2%"])

        fallback = _normalize_items([{"action": "补充工单数据"}], state=state)
        self.assertTrue(fallback[0]["evidence"])

        self.assertEqual(_normalize_items(None), [])
        self.assertEqual(_normalize_items("不是数组"), [])


class PromptGuardrailTests(unittest.TestCase):
    def test_common_rules_cover_phase10_requirements(self) -> None:
        rules = COMMON_RULES

        self.assertIn("不得虚构数据", rules)
        self.assertIn("不得虚构历史案例", rules)
        self.assertIn("不得把历史案例中的数据当成当前项目数据", rules)
        self.assertIn("当前数据不足以判断", rules)
        self.assertIn("可能原因（推测）", rules)
        self.assertIn("建议必须与输入中的问题一一相关", rules)
        self.assertIn("evidence", rules)

    def test_agent_prompts_require_evidence_structures(self) -> None:
        self.assertIn("findings", DIAGNOSIS_SYSTEM_PROMPT)
        self.assertIn("evidence", DIAGNOSIS_SYSTEM_PROMPT)
        self.assertIn("basis", DIAGNOSIS_SYSTEM_PROMPT)
        self.assertIn("evidence", RECOMMENDATION_SYSTEM_PROMPT)

    def test_report_prompt_separates_five_information_types(self) -> None:
        for keyword in (
            "当前项目数据事实",
            "程序计算结果",
            "RAG 历史案例",
            "AI 推测原因",
            "AI 运营建议",
        ):
            self.assertIn(keyword, REPORT_SYSTEM_PROMPT)

    def test_diagnosis_prompt_includes_trend_and_comparison(self) -> None:
        prompt = build_diagnosis_prompt(
            raw_data_summary={},
            metrics={},
            abnormal_projects=[],
            anomaly_summary=[],
            abnormal_robots=[],
            thresholds={},
            trend_analysis={"highlights": ["故障率上升"]},
            project_comparison={"highlights": ["A小区最优"]},
        )

        self.assertIn("【运营趋势（程序计算，禁止修改或外推）】", prompt)
        self.assertIn("【项目横向对比（程序计算，禁止修改或外推）】", prompt)


class ReportSeparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = make_temp_dir(prefix="robotops_test_phase10_report_")

    @classmethod
    def tearDownClass(cls) -> None:
        cleanup_dir(cls.temp_dir)

    def _report(self, state: dict) -> str:
        state.setdefault("use_llm", False)
        logger = WorkflowLogger(
            verbose=False, echo=False, log_file=self.temp_dir / "report.log"
        )
        agent = ReportAgent(logger=logger)
        update = agent.execute(state)
        return update["final_report"]

    def test_report_distinguishes_information_types(self) -> None:
        report = self._report(rich_state_with_diagnosis())

        self.assertIn("## 二、关键指标", report)
        self.assertIn("程序计算结果", report)
        self.assertIn("### 分析依据（程序计算）", report)
        self.assertIn("## 四、可能原因（AI 推测，非事实）", report)
        self.assertIn("可能原因（推测）", report)
        self.assertIn("## 五、历史相似案例（仅供参考）", report)
        self.assertIn("## 六、优化建议", report)
        self.assertIn("以上可能原因均为 AI 推测", report)

    def test_report_contains_case_traceability(self) -> None:
        report = self._report(rich_state_with_diagnosis())

        self.assertIn("CASE-FAULT-001", report)
        self.assertIn("相似度：0.87", report)
        self.assertIn("CASE-FAULT-001.md", report)
        self.assertIn("案例内容", report)

    def test_report_without_cases_uses_standard_message(self) -> None:
        state = rich_state_with_diagnosis()
        state["retrieved_cases"] = []
        state["retrieval"] = {"message": NO_RELEVANT_CASE_MESSAGE}

        report = self._report(state)

        self.assertIn(NO_RELEVANT_CASE_MESSAGE, report)

    def test_insufficient_data_is_stated_explicitly(self) -> None:
        state = {
            "use_llm": False,
            "metrics": {"values": {}, "units": {}},
            "raw_data_summary": {},
            "anomaly_summary": [],
            "abnormal_projects": [],
            "diagnosis": {},
            "retrieved_cases": [],
            "recommendations": {},
            "has_anomalies": False,
            "retrieval": {},
            "trend_analysis": {},
            "project_comparison": {},
        }
        report = self._report(state)

        self.assertIn("数据与口径说明", report)
        self.assertIn(NO_RELEVANT_CASE_MESSAGE, report)


class FrontendEvidenceTests(unittest.TestCase):
    def test_panel_sections(self) -> None:
        panel = uie.evidence_panel_data(rich_state_with_diagnosis())

        self.assertEqual(len(panel["reasons_frame"]), 1)
        self.assertIn("设备维护周期过长", panel["reasons_frame"].iloc[0]["可能原因（推测）"])
        self.assertEqual(len(panel["recommendations_frame"]), 1)

    def test_panel_without_diagnosis(self) -> None:
        panel = uie.evidence_panel_data(rich_state())

        self.assertTrue(panel["available"])
        self.assertTrue(panel["current"])
        self.assertEqual(len(panel["cases_frame"]), 1)
        self.assertEqual(len(panel["reasons_frame"]), 0)  # 该 state 未包含诊断结果
        self.assertIn("可能原因（推测）", panel["reasons_frame"].columns)
        self.assertIn("推测", panel["disclaimer"])

    def test_panel_with_diagnosis_and_recommendations(self) -> None:
        state = rich_state()
        state["diagnosis"] = {
            "findings": [{"finding": "故障率异常", "basis": "当前数据", "evidence": ["当前故障率 8.2"]}],
            "possible_reasons": [
                {
                    "reason": "设备维护周期过长",
                    "confidence": "中",
                    "evidence": ["当前故障率 8.2", "项目平均故障率 4.3"],
                    "data_gap": "缺少工单数据",
                }
            ],
        }
        state["recommendations"] = {
            "items": [{"action": "缩短维保周期", "priority": "高", "evidence": ["当前故障率 8.2"]}]
        }

        panel = uie.evidence_panel_data(state)

        self.assertEqual(len(panel["reasons_frame"]), 1)
        reason_row = panel["reasons_frame"].iloc[0]
        self.assertEqual(reason_row["可能原因（推测）"], "设备维护周期过长")
        self.assertIn("当前故障率 8.2", reason_row["分析依据（程序计算）"])
        self.assertEqual(len(panel["recommendations_frame"]), 1)
        self.assertTrue(panel["findings"])

    def test_panel_without_cases_shows_message(self) -> None:
        panel = uie.evidence_panel_data({"retrieved_cases": [], "retrieval": {}})

        self.assertEqual(panel["case_message"], NO_RELEVANT_CASE_MESSAGE)

    def test_panel_without_any_data(self) -> None:
        for value in (None, {}):
            panel = uie.evidence_panel_data(value)
            self.assertIn("case_message", panel)
            self.assertIn("disclaimer", panel)

    def test_case_source_frame_columns(self) -> None:
        frame = uie.case_source_frame(rich_state())

        self.assertEqual(len(frame), 1)
        for column in ("案例编号", "案例标题", "类别", "相似度", "来源文件", "内容摘要"):
            self.assertIn(column, frame.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)

