"""Phase 6：运营问题闭环测试（创建 / 状态流转 / 处理 / 整改验证 / 关闭 / 统计）。

需求要求的 6 个场景分别由以下用例覆盖：
1. 创建问题            -> test_create_issue_from_state
2. 修改状态            -> test_status_transitions / test_close_and_reopen
3. 添加处理结果        -> test_add_note_and_resolution
4. 整改后重新分析      -> test_verify_improvement_flow（含真实工作流的集成用例）
5. 自动计算改善幅度    -> test_compute_metric_change_matches_requirement_example
6. 完成问题关闭        -> test_close_issue_and_stats
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from robotops import config  # noqa: E402
from robotops.issues import (  # noqa: E402
    IssueError,
    IssueService,
    IssueStatus,
    IssueStatusError,
    IssueStore,
    allowed_next_statuses,
    compute_metric_change,
)
from scripts.generate_demo_data import generate_demo_file  # noqa: E402
from tests._helpers import cleanup_dir, make_temp_dir  # noqa: E402


def build_state(*, anomaly_types: str = "故障率偏高(2)、满意度偏低(1)") -> dict:
    """构造一份最小但结构完整的工作流 State（与 Phase 4 输出一致）。"""

    return {
        "has_anomalies": True,
        "anomaly_count": 2,
        "thresholds": {
            "故障率上限(%)": 5.0,
            "满意度下限(分)": 85.0,
            "运行率下限(%)": 80.0,
            "节降率下限(%)": 10.0,
        },
        "anomaly_summary": [
            {
                "anomaly_type": "故障率偏高",
                "condition": "> 5%",
                "hits": 2,
                "projects_involved": 1,
                "robots_involved": 1,
                "high": 1,
                "medium": 0,
                "low": 1,
            },
            {
                "anomaly_type": "满意度偏低",
                "condition": "< 85分",
                "hits": 1,
                "projects_involved": 1,
                "robots_involved": 1,
                "high": 0,
                "medium": 0,
                "low": 1,
            },
        ],
        "abnormal_projects": [
            {
                "project": "B小区",
                "anomaly_records": 3,
                "affected_robots": 1,
                "high_severity": 1,
                "medium_severity": 0,
                "low_severity": 2,
                "main_anomaly_types": anomaly_types,
            }
        ],
        "abnormal_robots": [
            {
                "robot_id": "B-01",
                "project": "B小区",
                "robot_type": "清洁机器人",
                "fault_rate": 12.5,
                "avg_uptime_rate": 78.0,
                "avg_satisfaction": 81.0,
                "avg_cost_reduction_rate": 8.0,
                "record_count": 30,
                "anomaly_records": 3,
                "anomaly_types": ["故障率偏高", "满意度偏低"],
            }
        ],
        "diagnosis": {
            "source": "llm",
            "model": "deepseek-chat",
            "summary": "推测：异常集中在单台设备，可能与部件劣化有关。",
            "possible_reasons": [
                {
                    "reason": "推测：部件劣化或作业强度偏高",
                    "confidence": "中",
                    "based_on": "该设备故障率显著高于同项目其他设备",
                    "data_gap": "缺少故障类型与部件明细",
                    "anomaly_type": "故障率偏高",
                },
                {
                    "reason": "推测：作业质量或服务响应不足",
                    "confidence": "低",
                    "based_on": "满意度低于阈值而设备指标基本正常",
                    "data_gap": "缺少投诉与工单数据",
                    "anomaly_type": "满意度偏低",
                },
            ],
        },
        "retrieved_cases": [
            {
                "case_id": "CASE-FAULT-001",
                "title": "清洁机器人重复故障（同一部件反复损坏）",
                "case_type": "机器人重复故障",
                "similarity": 0.472,
                "source_file": "CASE-FAULT-001.md",
                "data_nature": "模拟案例（虚构）",
                "matched_queries": ["故障率偏高（B小区）"],
            },
            {
                "case_id": "CASE-SAT-001",
                "title": "清洁效果下降引发用户满意度下滑",
                "case_type": "清洁效果下降",
                "similarity": 0.41,
                "source_file": "CASE-SAT-001.md",
                "data_nature": "模拟案例（虚构）",
                "matched_queries": ["满意度偏低（B小区）"],
            },
        ],
        "recommendations": {
            "source": "llm",
            "items": [
                {
                    "action": "对该设备做单机专项复盘",
                    "priority": "高",
                    "target": "B-01",
                    "expected_effect": "定位重复故障根因",
                    "verification": "补齐故障类型后重新统计故障率",
                    "reference_case": "CASE-FAULT-001",
                }
            ],
        },
        "analysis_payload": {
            "projects": [
                {
                    "project": "B小区",
                    "robot_count": 2,
                    "record_count": 30,
                    "avg_runtime": 7.5,
                    "avg_uptime_rate": 78.0,
                    "avg_fault_rate": 8.2,
                    "total_fault_count": 12,
                    "avg_satisfaction": 81.0,
                    "total_cost": 30000.0,
                    "avg_cost_reduction_rate": 8.0,
                }
            ]
        },
        "raw_data_summary": {"数据源": "B小区-运营数据.xlsx"},
        "workflow_status": "completed",
    }


def with_after_metrics(state: dict, **overrides: float) -> dict:
    """基于整改前 State 派生「整改后」State（只替换项目指标）。"""

    after = copy.deepcopy(state)
    for row in after["analysis_payload"]["projects"]:
        if row["project"] == "B小区":
            row.update(overrides)
    return after


class MetricChangeTests(unittest.TestCase):
    def test_compute_metric_change_matches_requirement_example(self) -> None:
        """需求示例：故障率 8.2% → 3.7%，改善幅度 =(8.2-3.7)/8.2 ≈ 54.9%。"""

        change = compute_metric_change(
            metric="故障率", unit="%", before=8.2, after=3.7, lower_is_better=True
        )

        self.assertEqual(change.direction, "下降")
        self.assertAlmostEqual(change.improvement_ratio, 0.5488, places=4)
        self.assertTrue(change.is_improved)
        self.assertIn("下降约 54.9%", change.statement)
        self.assertIn("明显改善", change.statement)

    def test_compute_metric_change_higher_is_better(self) -> None:
        change = compute_metric_change(
            metric="满意度", unit="分", before=81.0, after=92.0, lower_is_better=False
        )

        self.assertEqual(change.direction, "上升")
        self.assertTrue(change.is_improved)
        self.assertAlmostEqual(change.improvement_ratio, 0.1358, places=4)

    def test_no_improvement_and_flat(self) -> None:
        worse = compute_metric_change(
            metric="故障率", unit="%", before=5.0, after=8.0, lower_is_better=True
        )
        flat = compute_metric_change(
            metric="故障率", unit="%", before=5.0, after=5.0, lower_is_better=True
        )

        self.assertFalse(worse.is_improved)
        self.assertIn("未体现改善", worse.statement)
        self.assertEqual(flat.direction, "持平")
        self.assertEqual(flat.improvement_ratio, 0.0)

    def test_zero_before_value_is_safe(self) -> None:
        change = compute_metric_change(
            metric="故障率", unit="%", before=0.0, after=1.0, lower_is_better=True
        )

        self.assertEqual(change.improvement_ratio, 0.0)


class IssueServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = make_temp_dir(prefix="robotops_test_issues_")
        self.store = IssueStore(self.temp_dir / "issues.db")
        self.service = IssueService(self.store)
        self.state = build_state()

    def tearDown(self) -> None:
        self.store.close()
        cleanup_dir(self.temp_dir)

    # -- 1) 创建问题 -------------------------------------------------------
    def test_create_issue_from_state(self) -> None:
        issues = self.service.create_from_state(self.state, owner="售后运维")

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertTrue(issue.issue_id.startswith(f"ISSUE-{issue.created_at[:4]}{issue.created_at[5:7]}-"))
        self.assertEqual(issue.project, "B小区")
        self.assertEqual(issue.title, "机器人故障率异常升高")
        self.assertEqual(issue.priority, "高")
        self.assertEqual(issue.owner, "售后运维")
        self.assertEqual(issue.status, IssueStatus.PENDING.value)
        self.assertEqual(len(issue.metrics), 2)
        self.assertEqual(issue.metrics[0]["actual_value"], 8.2)
        self.assertEqual(issue.metrics[0]["threshold"], 5.0)
        self.assertEqual(issue.metrics[0]["robot_id"], "B-01")
        self.assertEqual(issue.metrics[0]["robot_value"], 12.5)
        self.assertEqual(issue.current_data["项目"], "B小区")
        self.assertEqual(len(issue.diagnosis["possible_reasons"]), 2)
        self.assertEqual([case["case_id"] for case in issue.cases], ["CASE-FAULT-001", "CASE-SAT-001"])
        self.assertTrue(issue.recommendations)
        self.assertEqual(len(issue.notes), 1)
        self.assertIn("故障率 8.2%", issue.metric_text())
        self.assertIn("最差设备 B-01", issue.metric_text())

    def test_issue_ids_increment_within_month(self) -> None:
        first = self.service.create_from_state(self.state)[0]
        second = self.service.create_from_state(self.state)[0]

        self.assertEqual(int(second.issue_id.rsplit("-", 1)[-1]), int(first.issue_id.rsplit("-", 1)[-1]) + 1)

    def test_create_requires_anomalies(self) -> None:
        state = build_state()
        state["has_anomalies"] = False
        state["abnormal_projects"] = []

        with self.assertRaises(IssueError):
            self.service.create_from_state(state)

    def test_create_for_selected_project_only(self) -> None:
        state = build_state()
        state["abnormal_projects"].append(
            {
                "project": "A小区",
                "anomaly_records": 1,
                "affected_robots": 1,
                "high_severity": 0,
                "medium_severity": 0,
                "low_severity": 1,
                "main_anomaly_types": "故障率偏高(1)",
            }
        )
        state["analysis_payload"]["projects"].append(
            {"project": "A小区", "avg_fault_rate": 6.0, "avg_satisfaction": 90.0}
        )

        issues = self.service.create_from_state(state, projects=["A小区"])

        self.assertEqual([issue.project for issue in issues], ["A小区"])
        self.assertEqual(issues[0].priority, "低")

    # -- 2) 状态流转 -------------------------------------------------------
    def test_status_transitions(self) -> None:
        issue = self.service.create_from_state(self.state)[0]

        in_progress = self.service.update_status(issue.issue_id, IssueStatus.IN_PROGRESS.value, note="已派单")
        self.assertEqual(in_progress.status, IssueStatus.IN_PROGRESS.value)

        pending_verification = self.service.update_status(
            issue.issue_id, IssueStatus.PENDING_VERIFICATION.value
        )
        self.assertEqual(pending_verification.status, IssueStatus.PENDING_VERIFICATION.value)
        self.assertTrue(pending_verification.requires_verification)

        completed = self.service.update_status(issue.issue_id, IssueStatus.COMPLETED.value)
        self.assertEqual(completed.status, IssueStatus.COMPLETED.value)

        closed = self.service.close(issue.issue_id, note="验证通过，关闭")
        self.assertEqual(closed.status, IssueStatus.CLOSED.value)
        self.assertTrue(closed.closed_at)
        # 状态变更都会留下记录
        kinds = [note.kind for note in closed.notes]
        self.assertIn("状态变更", kinds)

    def test_illegal_transition_is_rejected(self) -> None:
        issue = self.service.create_from_state(self.state)[0]

        with self.assertRaises(IssueStatusError) as ctx:
            self.service.update_status(issue.issue_id, IssueStatus.COMPLETED.value)

        self.assertIn("不允许", str(ctx.exception))
        self.assertEqual(allowed_next_statuses(IssueStatus.PENDING.value), ("处理中", "已关闭"))

    def test_close_and_reopen(self) -> None:
        issue = self.service.create_from_state(self.state)[0]
        self.service.close(issue.issue_id, note="误报，直接关闭")
        closed = self.service.get(issue.issue_id)
        self.assertEqual(closed.status, IssueStatus.CLOSED.value)
        self.assertTrue(closed.closed_at)

        reopened = self.service.update_status(issue.issue_id, IssueStatus.IN_PROGRESS.value)
        self.assertEqual(reopened.status, IssueStatus.IN_PROGRESS.value)
        self.assertEqual(reopened.closed_at, "")

    def test_update_owner_only(self) -> None:
        issue = self.service.create_from_state(self.state)[0]

        updated = self.service.update_status(issue.issue_id, issue.status, owner="运维-李工")

        self.assertEqual(updated.owner, "运维-李工")
        self.assertEqual(updated.status, issue.status)

    # -- 3) 处理结果与备注 -------------------------------------------------
    def test_add_note_and_resolution(self) -> None:
        issue = self.service.create_from_state(self.state)[0]

        self.service.add_note(issue.issue_id, "已联系厂家", author="张工")
        with_resolution = self.service.set_resolution(issue.issue_id, "更换部件并调整作业区域", author="张工")

        self.assertEqual(with_resolution.resolution, "更换部件并调整作业区域")
        kinds = [note.kind for note in with_resolution.notes]
        self.assertIn("备注", kinds)
        self.assertIn("处理结果", kinds)
        self.assertGreaterEqual(len(with_resolution.notes), 3)
        with self.assertRaises(IssueError):
            self.service.set_resolution(issue.issue_id, "   ")

    # -- 4/5) 整改效果验证 -------------------------------------------------
    def test_verify_improvement_flow(self) -> None:
        issue = self.service.create_from_state(self.state)[0]
        self.service.update_status(issue.issue_id, IssueStatus.IN_PROGRESS.value)
        self.service.set_resolution(issue.issue_id, "更换部件并增加点检频次")
        self.service.update_status(issue.issue_id, IssueStatus.PENDING_VERIFICATION.value)

        after_state = with_after_metrics(
            self.state, avg_fault_rate=3.7, avg_satisfaction=92.0
        )
        issue_after, result = self.service.verify_improvement(
            issue.issue_id, after_state, after_file="B小区-整改后.xlsx"
        )

        changes = {change.metric: change for change in result.changes}
        self.assertAlmostEqual(changes["故障率"].improvement_ratio, 0.5488, places=4)
        self.assertEqual(changes["故障率"].before, 8.2)
        self.assertEqual(changes["故障率"].after, 3.7)
        self.assertIn("下降约 54.9%", result.statement)
        self.assertTrue(result.improved)
        self.assertEqual(issue_after.status, IssueStatus.COMPLETED.value)
        self.assertTrue(issue_after.verification["changes"])
        self.assertEqual(issue_after.verification["after_file"], "B小区-整改后.xlsx")
        self.assertIn("整改验证", [note.kind for note in issue_after.notes])
        self.assertIn("未使用大模型推断", result.note)

    def test_verify_requires_pending_verification(self) -> None:
        issue = self.service.create_from_state(self.state)[0]

        with self.assertRaises(IssueStatusError):
            self.service.verify_improvement(issue.issue_id, self.state)

    def test_verify_without_improvement_keeps_status(self) -> None:
        issue = self.service.create_from_state(self.state)[0]
        self.service.update_status(issue.issue_id, IssueStatus.IN_PROGRESS.value)
        self.service.update_status(issue.issue_id, IssueStatus.PENDING_VERIFICATION.value)

        worse = with_after_metrics(self.state, avg_fault_rate=12.0, avg_satisfaction=70.0)
        issue_after, result = self.service.verify_improvement(issue.issue_id, worse)

        self.assertFalse(result.improved)
        self.assertIn("未体现", result.conclusion)
        self.assertEqual(issue_after.status, IssueStatus.PENDING_VERIFICATION.value)

    def test_verify_unknown_project_raises(self) -> None:
        issue = self.service.create_from_state(self.state)[0]
        self.service.update_status(issue.issue_id, IssueStatus.IN_PROGRESS.value)
        self.service.update_status(issue.issue_id, IssueStatus.PENDING_VERIFICATION.value)

        with self.assertRaises(IssueError):
            self.service.verify_improvement(issue.issue_id, build_state() | {"analysis_payload": {"projects": []}})

    # -- 6) 关闭与统计 -----------------------------------------------------
    def test_close_issue_and_stats(self) -> None:
        issue = self.service.create_from_state(self.state)[0]
        self.service.update_status(issue.issue_id, IssueStatus.IN_PROGRESS.value)
        self.service.update_status(issue.issue_id, IssueStatus.PENDING_VERIFICATION.value)
        self.service.verify_improvement(issue.issue_id, with_after_metrics(self.state, avg_fault_rate=3.7))
        closed = self.service.close(issue.issue_id, note="整改效果验证通过，问题关闭")

        self.assertEqual(closed.status, IssueStatus.CLOSED.value)
        stats = self.service.stats()
        self.assertEqual(stats["总问题数"], 1)
        self.assertEqual(stats["已关闭"], 1)
        self.assertEqual(stats["未关闭问题数"], 0)
        self.assertEqual(stats["问题关闭率"], 1.0)
        self.assertIsNotNone(stats["平均处理周期(天)"])
        self.assertEqual(stats["平均处理周期(天)"], 0.0)

    def test_stats_with_mixed_statuses(self) -> None:
        first = self.service.create_from_state(self.state)[0]
        second = self.service.create_from_state(self.state)[0]
        self.service.update_status(first.issue_id, IssueStatus.IN_PROGRESS.value)
        self.service.close(second.issue_id)

        stats = self.service.stats()

        self.assertEqual(stats["总问题数"], 2)
        self.assertEqual(stats["处理中"], 1)
        self.assertEqual(stats["已关闭"], 1)
        self.assertEqual(stats["问题关闭率"], 0.5)

    # -- 存储 -------------------------------------------------------------
    def test_list_filters_and_persistence(self) -> None:
        issue = self.service.create_from_state(self.state)[0]
        self.service.update_status(issue.issue_id, IssueStatus.IN_PROGRESS.value)

        self.assertEqual(len(self.service.list_issues(status="处理中")), 1)
        self.assertEqual(len(self.service.list_issues(status="待处理")), 0)
        self.assertEqual(len(self.service.list_issues(project="B小区")), 1)
        self.assertEqual(self.service.projects(), ["B小区"])

        # 重新打开数据库，数据仍在
        self.store.close()
        reopened = IssueStore(self.temp_dir / "issues.db")
        try:
            reloaded = IssueService(reopened).get(issue.issue_id)
            self.assertEqual(reloaded.status, IssueStatus.IN_PROGRESS.value)
            self.assertEqual(reloaded.metrics[0]["actual_value"], 8.2)
            self.assertTrue(reloaded.notes)
        finally:
            reopened.close()
        self.store = IssueStore(self.temp_dir / "issues.db")
        self.service = IssueService(self.store)

    def test_delete_issue(self) -> None:
        issue = self.service.create_from_state(self.state)[0]

        self.assertTrue(self.store.delete_issue(issue.issue_id))
        self.assertEqual(self.service.stats()["总问题数"], 0)
        self.assertFalse(self.store.delete_issue(issue.issue_id))

    def test_store_does_not_contain_api_keys(self) -> None:
        """数据安全：问题库中不应出现任何 API Key。"""

        self.service.create_from_state(self.state)
        raw = (self.temp_dir / "issues.db").read_bytes()

        self.assertNotIn(b"sk-", raw)
        self.assertNotIn(b"DEEPSEEK_API_KEY", raw)


class IssueIntegrationTests(unittest.TestCase):
    """真实数据集成：Phase 1~4 工作流 → 创建问题 → 整改后重新分析 → 验证 → 关闭。"""

    @classmethod
    def setUpClass(cls) -> None:
        if not config.DEMO_EXCEL_FILE.exists():
            generate_demo_file(config.DEMO_EXCEL_FILE)
        cls.temp_dir = make_temp_dir(prefix="robotops_test_issues_integration_")
        cls.store = IssueStore(cls.temp_dir / "issues.db")
        cls.service = IssueService(cls.store)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.store.close()
        cleanup_dir(cls.temp_dir)

    def test_end_to_end_closed_loop_with_real_data(self) -> None:
        from app.graph.workflow import RobotOpsWorkflow

        workflow = RobotOpsWorkflow(verbose=False, echo=False, log_file=self.temp_dir / "wf.log")
        before = workflow.run(data_path=config.DEMO_EXCEL_FILE, use_llm=False, export=False)
        project = "华北-天津港智能巡检项目"

        issues = self.service.create_from_state(before, owner="售后运维", projects=[project])
        issue = issues[0]
        self.assertEqual(issue.project, project)
        self.assertEqual(issue.status, IssueStatus.PENDING.value)
        self.assertEqual(issue.priority, "高")
        self.assertTrue(issue.cases)

        # 售后处理 → 待验证
        self.service.update_status(issue.issue_id, IssueStatus.IN_PROGRESS.value, owner="张工", note="已派单")
        self.service.set_resolution(issue.issue_id, "更换部件 + 调整作业区域")
        self.service.update_status(issue.issue_id, IssueStatus.PENDING_VERIFICATION.value)

        # 整改后数据：该项目的故障次数归零、满意度与节降率提升
        after_path = self.temp_dir / "after.xlsx"
        raw = pd.read_excel(config.DEMO_EXCEL_FILE, sheet_name="运营数据")
        mask = raw[config.COL_PROJECT] == project
        raw.loc[mask, config.COL_FAULT_COUNT] = 0
        raw.loc[mask, config.COL_SATISFACTION] = 95.0
        raw.loc[mask, config.COL_SAVING_RATE] = 24.0
        raw.to_excel(after_path, index=False)
        after = workflow.run(data_path=after_path, use_llm=False, export=False)

        issue_after, result = self.service.verify_improvement(
            issue.issue_id, after, after_file=after_path.name
        )

        fault_change = next(change for change in result.changes if change.metric == "故障率")
        self.assertGreater(fault_change.before, fault_change.after)
        self.assertTrue(fault_change.is_improved)
        self.assertGreater(fault_change.improvement_ratio, 0.5)
        self.assertIn("下降约", result.statement)
        self.assertEqual(issue_after.status, IssueStatus.COMPLETED.value)

        closed = self.service.close(issue.issue_id, note="验证通过，关闭问题")
        self.assertEqual(closed.status, IssueStatus.CLOSED.value)
        self.assertTrue(closed.closed_at)
        self.assertEqual(self.service.stats()["问题关闭率"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
