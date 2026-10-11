"""Phase 11：运营问题场景测试（Issue 创建 / 状态更新 / 整改前后数据比较）。

覆盖需求要求的闭环场景（聚焦三件事，不重复 test_issues.py 的全量用例）：
1. Issue 创建 —— 从工作流 State 批量创建，指标为真实计算值；
2. Issue 状态更新 —— 合法流转通过、非法流转被拒绝；
3. 整改前后数据比较 —— 故障率 8.2% → 3.7% 判定为明显改善（约 54.9%）。
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotops.issues import (  # noqa: E402
    IssueError,
    IssueService,
    IssueStatus,
    IssueStatusError,
    IssueStore,
    compute_metric_change,
)
from tests._helpers import make_temp_dir  # noqa: E402
from tests.test_issues import build_state, with_after_metrics  # noqa: E402


class IssueScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = make_temp_dir(prefix="robotops_test_issue11_")
        self.store = IssueStore(self.temp_dir / "issues.db")
        self.service = IssueService(self.store)
        self.state = build_state()

    def tearDown(self) -> None:
        self.store.close()
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -- 1) Issue 创建 -----------------------------------------------------
    def test_create_issue_from_state(self) -> None:
        issues = self.service.create_from_state(self.state, owner="售后运维")

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.project, "B小区")
        self.assertEqual(issue.status, IssueStatus.PENDING.value)
        self.assertEqual(issue.owner, "售后运维")
        # 指标为程序按 State 数据取的真实值
        metrics = {item["metric"]: item for item in issue.metrics}
        self.assertIn("故障率", metrics)
        self.assertEqual(metrics["故障率"]["actual_value"], 8.2)

    def test_create_requires_anomaly_state(self) -> None:
        normal_state = copy.deepcopy(self.state)
        normal_state["has_anomalies"] = False

        with self.assertRaises(IssueError):
            self.service.create_from_state(normal_state)

    # -- 2) Issue 状态更新 -------------------------------------------------
    def test_legal_status_transitions(self) -> None:
        issue = self.service.create_from_state(self.state)[0]

        issue = self.service.update_status(issue.issue_id, IssueStatus.IN_PROGRESS.value)
        self.assertEqual(issue.status, IssueStatus.IN_PROGRESS.value)

        issue = self.service.update_status(issue.issue_id, IssueStatus.PENDING_VERIFICATION.value)
        self.assertEqual(issue.status, IssueStatus.PENDING_VERIFICATION.value)

    def test_illegal_transition_rejected(self) -> None:
        issue = self.service.create_from_state(self.state)[0]

        # 状态流转规则（ALLOWED_TRANSITIONS）：
        #   待处理 -> 处理中 / 已关闭（误报可直接关闭）
        #   待处理 -> 已完成 属于非法跳步，必须被拒绝
        with self.assertRaises(IssueStatusError):
            self.service.update_status(issue.issue_id, IssueStatus.COMPLETED.value)

        # 合法的直接关闭（误报场景）应当通过
        closed = self.service.update_status(issue.issue_id, IssueStatus.CLOSED.value)
        self.assertEqual(closed.status, IssueStatus.CLOSED.value)

    # -- 3) 整改前后数据比较 -----------------------------------------------
    def test_before_after_comparison_improvement(self) -> None:
        """故障率 8.2% → 3.7%：改善幅度约 54.9%，判定明显改善。"""

        change = compute_metric_change(
            metric="故障率", unit="%", before=8.2, after=3.7, lower_is_better=True
        )

        self.assertAlmostEqual(change.improvement_ratio, 0.5488, places=4)
        self.assertTrue(change.is_improved)
        self.assertIn("明显改善", change.statement)

    def test_verify_improvement_end_to_end(self) -> None:
        """完整链路：创建 → 处理 → 待验证 → 上传整改后数据 → 验证通过。"""

        issue = self.service.create_from_state(self.state)[0]
        self.service.update_status(issue.issue_id, IssueStatus.IN_PROGRESS.value)
        self.service.update_status(issue.issue_id, IssueStatus.PENDING_VERIFICATION.value)

        # 整改后：故障率 8.2 -> 3.7、满意度 81 -> 92
        after_state = with_after_metrics(
            self.state, avg_fault_rate=3.7, avg_satisfaction=92.0
        )
        issue, verification = self.service.verify_improvement(issue.issue_id, after_state)

        self.assertTrue(verification.improved)
        self.assertEqual(issue.status, IssueStatus.COMPLETED.value)

        # 关闭
        closed = self.service.close(issue.issue_id)
        self.assertEqual(closed.status, IssueStatus.CLOSED.value)
        self.assertTrue(closed.closed_at)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
