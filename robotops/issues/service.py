"""RobotOps AI —— Phase 6 运营问题闭环业务逻辑。

闭环流程：

    发现问题 → AI 分析 → RAG 历史案例 → AI 建议 → 创建问题 →
    售后/运维处理 → 重新上传数据 → 整改效果验证 → 问题关闭

重要约束：
- 问题的指标值、改善幅度**全部由程序按实际数据计算**，不使用大模型推断；
- 状态流转受 ``ALLOWED_TRANSITIONS`` 限制，非法流转会被拒绝并给出提示。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from .errors import IssueError, IssueNotFoundError, IssueStatusError
from .models import (
    ANOMALY_METRIC_MAP,
    Issue,
    IssueNote,
    IssueStatus,
    VerificationResult,
    build_verification_result,
    can_transition,
    allowed_next_statuses,
)
from .store import IssueStore

DEFAULT_OWNER = "售后运维"
DEFAULT_AUTHOR = "运营管理后台"
MAX_CASES_PER_ISSUE = 3
MAX_RECOMMENDATIONS_PER_ISSUE = 3


class IssueService:
    """运营问题的创建、流转、处理与验证。"""

    def __init__(self, store: IssueStore | None = None, *, default_owner: str = DEFAULT_OWNER) -> None:
        self.store = store or IssueStore()
        self.default_owner = default_owner

    # -- 创建 -------------------------------------------------------------
    def create_from_state(
        self,
        state: dict[str, Any],
        *,
        owner: str | None = None,
        projects: Iterable[str] | None = None,
        source_file: str = "",
    ) -> list[Issue]:
        """按分析结果中的异常项目批量创建运营问题。"""

        abnormal_projects = state.get("abnormal_projects") or []
        if not state.get("has_anomalies") or not abnormal_projects:
            raise IssueError(
                "本次分析未发现异常，无需创建运营问题",
                hint="请先在上传的数据中确认存在超过阈值的异常（故障率/满意度/运行率/节降率）。",
            )

        wanted = {str(item) for item in projects} if projects else None
        payload_projects = {
            str(row.get("project")): row
            for row in ((state.get("analysis_payload") or {}).get("projects") or [])
        }
        anomaly_rows = {
            str(row.get("anomaly_type")): row for row in (state.get("anomaly_summary") or [])
        }
        diagnosis = state.get("diagnosis") or {}
        cases = state.get("retrieved_cases") or []
        recommendations = (state.get("recommendations") or {}).get("items") or []
        thresholds = state.get("thresholds") or {}

        created: list[Issue] = []
        for item in abnormal_projects:
            project = str(item.get("project") or "")
            if not project or (wanted is not None and project not in wanted):
                continue
            types = _parse_anomaly_types(item.get("main_anomaly_types"))
            if not types:
                continue
            project_row = payload_projects.get(project, {})
            metrics = _build_metrics(
                types,
                project_row,
                anomaly_rows,
                thresholds,
                _project_robots(state.get("abnormal_robots") or [], project),
            )
            if not metrics:
                continue

            priority = _priority_of(item)
            issue = Issue(
                issue_id=self.store.next_issue_id(),
                project=project,
                title=_build_title(types),
                anomaly_type="、".join(types),
                priority=priority,
                owner=owner or self.default_owner,
                status=IssueStatus.PENDING.value,
                created_at=_now(),
                updated_at=_now(),
                metrics=metrics,
                current_data=_project_snapshot(project, project_row),
                diagnosis=_build_diagnosis(diagnosis, types),
                cases=_select_cases(cases, types),
                recommendations=recommendations[:MAX_RECOMMENDATIONS_PER_ISSUE],
                source_file=source_file or str((state.get("raw_data_summary") or {}).get("数据源", "")),
                notes=[
                    IssueNote(
                        created_at=_now(),
                        author=DEFAULT_AUTHOR,
                        kind="创建",
                        text=(
                            f"由运营分析自动创建：{project} 命中异常 "
                            f"{item.get('anomaly_records')} 条（{item.get('main_anomaly_types')}），"
                            f"异常指标：{_metrics_text(metrics)}"
                        ),
                    )
                ],
            )
            created.append(self.store.create_issue(issue))

        if not created:
            raise IssueError(
                "未创建任何运营问题",
                hint="请检查所选项目是否存在可映射为问题的异常类型（故障率/满意度/运行率/节降率）。",
            )
        return created

    # -- 处理 -------------------------------------------------------------
    def update_status(
        self,
        issue_id: str,
        status: str,
        *,
        owner: str | None = None,
        note: str = "",
        author: str = DEFAULT_AUTHOR,
    ) -> Issue:
        """修改问题状态（受状态流转规则约束）。"""

        issue = self.store.get_required(issue_id)
        target = IssueStatus.coerce(status).value
        if target == issue.status:
            if owner:
                return self.store.update_fields(issue_id, owner=owner)
            return issue
        if not can_transition(issue.status, target):
            allowed = allowed_next_statuses(issue.status)
            raise IssueStatusError(
                f"不允许从「{issue.status}」直接变更为「{target}」",
                hint=(
                    f"当前状态允许变更为：{'、'.join(allowed)}"
                    if allowed
                    else "该状态为终态，如需继续处理请先重新打开问题。"
                ),
            )

        fields: dict[str, Any] = {"status": target}
        if owner is not None:
            fields["owner"] = owner
        if target == IssueStatus.CLOSED.value:
            fields["closed_at"] = _now()
        elif issue.closed_at:
            fields["closed_at"] = ""

        updated = self.store.update_fields(issue_id, **fields)
        detail = f"状态变更：{issue.status} → {target}"
        if note:
            detail += f"；说明：{note}"
        return self.store.add_note(
            issue_id,
            IssueNote(created_at=_now(), author=author, kind="状态变更", text=detail),
        ) if updated else updated

    def add_note(self, issue_id: str, text: str, *, author: str = DEFAULT_AUTHOR) -> Issue:
        """添加备注。"""

        if not str(text).strip():
            raise IssueError("备注内容不能为空")
        return self.store.add_note(
            issue_id,
            IssueNote(created_at=_now(), author=author, kind="备注", text=str(text).strip()),
        )

    def set_resolution(self, issue_id: str, text: str, *, author: str = DEFAULT_AUTHOR) -> Issue:
        """填写处理结果。"""

        if not str(text).strip():
            raise IssueError("处理结果不能为空")
        issue = self.store.update_fields(issue_id, resolution=str(text).strip())
        self.store.add_note(
            issue_id,
            IssueNote(
                created_at=_now(), author=author, kind="处理结果", text=str(text).strip()
            ),
        )
        return self.store.get_required(issue.issue_id)

    # -- 验证 -------------------------------------------------------------
    def verify_improvement(
        self,
        issue_id: str,
        after_state: dict[str, Any],
        *,
        after_file: str = "",
        auto_advance: bool = True,
        author: str = DEFAULT_AUTHOR,
    ) -> tuple[Issue, VerificationResult]:
        """对比整改前后数据并生成验证结论（改善幅度由程序计算）。"""

        issue = self.store.get_required(issue_id)
        if not issue.requires_verification:
            raise IssueStatusError(
                f"只有「{IssueStatus.PENDING_VERIFICATION.value}」状态的问题可以执行整改效果验证"
                f"（当前状态：{issue.status}）",
                hint="请先把问题状态改为「待验证」，再上传整改后的数据。",
            )
        if not after_state:
            raise IssueError("缺少整改后的分析结果，请重新上传数据并完成分析")

        payload_projects = {
            str(row.get("project")): row
            for row in ((after_state.get("analysis_payload") or {}).get("projects") or [])
        }
        after_row = payload_projects.get(issue.project)
        if after_row is None:
            raise IssueError(
                f"整改后的数据中没有找到项目：{issue.project}",
                hint="请确认上传的整改后数据包含同一项目名称，或检查项目名称是否被修改。",
            )

        before_metrics = {
            str(item.get("metric")): float(item.get("actual_value"))
            for item in issue.metrics
            if item.get("actual_value") is not None
        }
        after_metrics = {
            str(item.get("metric")): float(after_row[str(item.get("payload_key"))])
            for item in issue.metrics
            if item.get("payload_key") and after_row.get(str(item.get("payload_key"))) is not None
        }
        definitions = [
            {
                "metric": str(item.get("metric")),
                "unit": str(item.get("unit") or ""),
                "lower_is_better": bool(item.get("lower_is_better", True)),
            }
            for item in issue.metrics
        ]
        result = build_verification_result(
            issue_id=issue.issue_id,
            project=issue.project,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            metric_definitions=definitions,
            before_file=issue.source_file,
            after_file=after_file or str((after_state.get("raw_data_summary") or {}).get("数据源", "")),
        )

        fields: dict[str, Any] = {"verification": result.to_dict()}
        if result.improved and auto_advance:
            fields["status"] = IssueStatus.COMPLETED.value
        self.store.update_fields(issue.issue_id, **fields)

        lines = [result.statement]
        for change in result.changes:
            lines.append(
                f"{change.metric}：整改前 {change.before:g}{change.unit} → "
                f"整改后 {change.after:g}{change.unit}（{change.direction} {change.improvement_ratio * 100:.1f}%）"
            )
        lines.append(result.conclusion)
        lines.append(result.note)
        self.store.add_note(
            issue.issue_id,
            IssueNote(
                created_at=_now(),
                author=author,
                kind="整改验证",
                text="；".join(lines),
            ),
        )
        return self.store.get_required(issue.issue_id), result

    # -- 完成 / 关闭 -------------------------------------------------------
    def complete(self, issue_id: str, *, note: str = "", author: str = DEFAULT_AUTHOR) -> Issue:
        """标记为已完成（待验证 → 已完成）。"""

        return self.update_status(
            issue_id, IssueStatus.COMPLETED.value, note=note, author=author
        )

    def close(self, issue_id: str, *, note: str = "", author: str = DEFAULT_AUTHOR) -> Issue:
        """关闭问题（已完成 → 已关闭）。"""

        return self.update_status(issue_id, IssueStatus.CLOSED.value, note=note, author=author)

    # -- 查询 -------------------------------------------------------------
    def get(self, issue_id: str) -> Issue:
        return self.store.get_required(issue_id)

    def list_issues(
        self, *, status: str | None = None, project: str | None = None, limit: int | None = None
    ) -> list[Issue]:
        return self.store.list_issues(status=status, project=project, limit=limit)

    def stats(self) -> dict[str, Any]:
        return self.store.stats()

    def projects(self) -> list[str]:
        return self.store.projects()


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_anomaly_types(text: Any) -> list[str]:
    if not text:
        return []
    types: list[str] = []
    for part in str(text).split("、"):
        name = part.split("(")[0].strip()
        if name and name in ANOMALY_METRIC_MAP and name not in types:
            types.append(name)
    return types


def _build_metrics(
    types: Sequence[str],
    project_row: dict[str, Any],
    anomaly_rows: dict[str, dict[str, Any]],
    thresholds: dict[str, Any],
    robots: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for anomaly_type in types:
        definition = ANOMALY_METRIC_MAP[anomaly_type]
        summary = anomaly_rows.get(anomaly_type, {})
        worst = _worst_robot(
            robots, anomaly_type, str(definition.get("robot_key") or definition["payload_key"])
        )
        metrics.append(
            {
                "anomaly_type": anomaly_type,
                "metric": definition["metric"],
                "payload_key": definition["payload_key"],
                "unit": definition["unit"],
                "lower_is_better": definition["lower_is_better"],
                # actual_value 取项目维度平均值（两次分析都存在该字段，便于整改前后对比）
                "actual_value": project_row.get(str(definition["payload_key"])),
                "threshold": thresholds.get(str(definition.get("threshold_key", ""))),
                "condition": summary.get("condition", ""),
                "hits": summary.get("hits", 0),
                "robot_id": worst.get("robot_id") if worst else "",
                "robot_value": worst.get("value") if worst else None,
                "severity": (
                    "高"
                    if summary.get("high")
                    else ("中" if summary.get("medium") else ("低" if summary.get("low") else "-"))
                ),
            }
        )
    return metrics


def _project_robots(robots: Iterable[dict[str, Any]], project: str) -> list[dict[str, Any]]:
    return [robot for robot in robots if str(robot.get("project")) == project]


def _worst_robot(
    robots: Sequence[dict[str, Any]], anomaly_type: str, payload_key: str
) -> dict[str, Any] | None:
    """取该异常类型下命中记录最多的机器人（作为“最差设备”展示）。"""

    candidates = [
        robot
        for robot in robots
        if anomaly_type in (robot.get("anomaly_types") or [])
        and robot.get(payload_key) is not None
    ]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda robot: (int(robot.get("anomaly_records") or 0), float(robot.get(payload_key) or 0)),
    )
    return {"robot_id": best.get("robot_id"), "value": best.get(payload_key)}


def _metrics_text(metrics: Sequence[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in metrics:
        value = item.get("actual_value")
        unit = item.get("unit") or ""
        condition = item.get("condition") or ""
        text = f"{item.get('metric')} {value}{unit}"
        if condition:
            text += f"（{condition}）"
        if item.get("robot_id"):
            text += f"；最差设备 {item['robot_id']} {item.get('robot_value')}{unit}"
        parts.append(text)
    return "；".join(parts)


def _priority_of(project_row: dict[str, Any]) -> str:
    if int(project_row.get("high_severity") or 0) > 0:
        return "高"
    if int(project_row.get("medium_severity") or 0) > 0:
        return "中"
    return "低"


def _build_title(types: Sequence[str]) -> str:
    """问题标题：以主要异常类型命名（其余异常类型在「问题类型/异常指标」中体现）。"""

    primary = types[0]
    return str(ANOMALY_METRIC_MAP[primary].get("title") or f"机器人{primary}")


def _project_snapshot(project: str, project_row: dict[str, Any]) -> dict[str, Any]:
    """问题详情里的「当前数据」（项目维度快照）。"""

    return {
        "项目": project,
        "机器人数量": project_row.get("robot_count"),
        "记录数": project_row.get("record_count"),
        "平均运行时长(小时)": project_row.get("avg_runtime"),
        "平均运行率(%)": project_row.get("avg_uptime_rate"),
        "故障率(%)": project_row.get("avg_fault_rate"),
        "总故障次数": project_row.get("total_fault_count"),
        "平均满意度(分)": project_row.get("avg_satisfaction"),
        "总运营成本(元)": project_row.get("total_cost"),
        "平均节降率(%)": project_row.get("avg_cost_reduction_rate"),
    }


def _build_diagnosis(diagnosis: dict[str, Any], types: Sequence[str]) -> dict[str, Any]:
    reasons = [
        reason
        for reason in (diagnosis.get("possible_reasons") or [])
        if not reason.get("anomaly_type") or reason.get("anomaly_type") in types
    ]
    return {
        "source": diagnosis.get("source", ""),
        "model": diagnosis.get("model", ""),
        "summary": diagnosis.get("summary", ""),
        "possible_reasons": reasons,
    }


def _select_cases(cases: Sequence[dict[str, Any]], types: Sequence[str]) -> list[dict[str, Any]]:
    matched = [
        case
        for case in cases
        if any(str(label).startswith(anomaly_type) for label in (case.get("matched_queries") or []) for anomaly_type in types)
    ]
    selected = matched or list(cases)
    return [
        {
            "case_id": case.get("case_id"),
            "title": case.get("title"),
            "case_type": case.get("case_type"),
            "similarity": case.get("similarity"),
            "source_file": case.get("source_file"),
            "data_nature": case.get("data_nature"),
        }
        for case in selected[:MAX_CASES_PER_ISSUE]
    ]
