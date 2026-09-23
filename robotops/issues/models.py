"""RobotOps AI —— Phase 6 运营问题数据模型。

问题编号格式：``ISSUE-YYYYMM-NNN``（例如 ISSUE-202609-001）。
状态流转：待处理 → 处理中 → 待验证 → 已完成 → 已关闭（允许必要时回退，见 ``ALLOWED_TRANSITIONS``）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterable

#: 异常类型 -> 用于整改效果对比的指标定义
ANOMALY_METRIC_MAP: dict[str, dict[str, Any]] = {
    "故障率偏高": {
        "metric": "故障率",
        "payload_key": "avg_fault_rate",
        "robot_key": "fault_rate",
        "unit": "%",
        "lower_is_better": True,
        "threshold_key": "故障率上限(%)",
        "title": "机器人故障率异常升高",
    },
    "满意度偏低": {
        "metric": "满意度",
        "payload_key": "avg_satisfaction",
        "robot_key": "avg_satisfaction",
        "unit": "分",
        "lower_is_better": False,
        "threshold_key": "满意度下限(分)",
        "title": "用户满意度下降",
    },
    "运行率偏低": {
        "metric": "运行率",
        "payload_key": "avg_uptime_rate",
        "robot_key": "avg_uptime_rate",
        "unit": "%",
        "lower_is_better": False,
        "threshold_key": "运行率下限(%)",
        "title": "机器人运行率偏低",
    },
    "节降率偏低": {
        "metric": "节降率",
        "payload_key": "avg_cost_reduction_rate",
        "robot_key": "avg_cost_reduction_rate",
        "unit": "%",
        "lower_is_better": False,
        "threshold_key": "节降率下限(%)",
        "title": "节降率偏低",
    },
}

#: 问题列表展示字段（前端表格）
ISSUE_LIST_COLUMNS: tuple[str, ...] = (
    "问题编号",
    "项目",
    "问题类型",
    "优先级",
    "负责人",
    "状态",
    "创建时间",
)


class IssueStatus(str, Enum):
    """运营问题状态。"""

    PENDING = "待处理"
    IN_PROGRESS = "处理中"
    PENDING_VERIFICATION = "待验证"
    COMPLETED = "已完成"
    CLOSED = "已关闭"

    @classmethod
    def values(cls) -> list[str]:
        return [item.value for item in cls]

    @classmethod
    def coerce(cls, value: Any) -> "IssueStatus":
        if isinstance(value, IssueStatus):
            return value
        text = str(value or "").strip()
        for item in cls:
            if item.value == text:
                return item
        raise ValueError(f"未知的问题状态：{value!r}；可选值：{'、'.join(cls.values())}")


#: 允许的状态流转（含必要的回退路径）
ALLOWED_TRANSITIONS: dict[str, tuple[str, ...]] = {
    IssueStatus.PENDING.value: (
        IssueStatus.IN_PROGRESS.value,
        IssueStatus.CLOSED.value,  # 误报可直接关闭
    ),
    IssueStatus.IN_PROGRESS.value: (
        IssueStatus.PENDING_VERIFICATION.value,
        IssueStatus.PENDING.value,
        IssueStatus.CLOSED.value,
    ),
    IssueStatus.PENDING_VERIFICATION.value: (
        IssueStatus.COMPLETED.value,
        IssueStatus.IN_PROGRESS.value,  # 验证不通过打回处理
    ),
    IssueStatus.COMPLETED.value: (
        IssueStatus.CLOSED.value,
        IssueStatus.PENDING_VERIFICATION.value,
    ),
    IssueStatus.CLOSED.value: (IssueStatus.IN_PROGRESS.value,),  # 重新打开
}


def allowed_next_statuses(status: str) -> tuple[str, ...]:
    """返回某状态允许流转到的下一状态。"""

    return ALLOWED_TRANSITIONS.get(str(status), ())


def can_transition(current: str, target: str) -> bool:
    return str(target) in allowed_next_statuses(current)


@dataclass
class IssueNote:
    """问题备注 / 处理记录。"""

    created_at: str
    text: str
    author: str = ""
    kind: str = "备注"  # 备注 / 状态变更 / 处理结果 / 整改验证

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Issue:
    """运营问题记录（对应一个项目的异常整改任务）。"""

    issue_id: str
    project: str
    title: str
    anomaly_type: str
    priority: str
    owner: str
    status: str
    created_at: str
    updated_at: str
    metrics: list[dict[str, Any]] = field(default_factory=list)
    current_data: dict[str, Any] = field(default_factory=dict)
    diagnosis: dict[str, Any] = field(default_factory=dict)
    cases: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    resolution: str = ""
    verification: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    closed_at: str = ""
    notes: list[IssueNote] = field(default_factory=list)

    # -- 展示 -------------------------------------------------------------
    def list_row(self) -> dict[str, Any]:
        """问题列表一行。"""

        return {
            "问题编号": self.issue_id,
            "项目": self.project,
            "问题类型": self.anomaly_type,
            "优先级": self.priority,
            "负责人": self.owner or "未指派",
            "状态": self.status,
            "创建时间": self.created_at,
        }

    def metric_text(self) -> str:
        """异常指标文本，例如「故障率 2.91%（> 5%，项目平均；最差设备 TJ-INS-05 10.96%）」。"""

        parts: list[str] = []
        for item in self.metrics:
            value = item.get("actual_value")
            unit = item.get("unit") or ""
            condition = item.get("condition") or ""
            extra: list[str] = []
            if condition:
                extra.append(str(condition))
            if item.get("robot_id"):
                extra.append(
                    f"最差设备 {item['robot_id']} {item.get('robot_value')}{unit}"
                )
            text = f"{item.get('metric')} {value}{unit}"
            if extra:
                text += f"（{'，'.join(extra)}）"
            parts.append(text)
        return "；".join(parts)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["notes"] = [note.to_dict() for note in self.notes]
        data["metric_text"] = self.metric_text()
        return data

    @property
    def is_closed(self) -> bool:
        return self.status == IssueStatus.CLOSED.value

    @property
    def requires_verification(self) -> bool:
        return self.status == IssueStatus.PENDING_VERIFICATION.value


@dataclass
class MetricChange:
    """单个指标在整改前后的变化。"""

    metric: str
    unit: str
    before: float
    after: float
    improvement_ratio: float
    is_improved: bool
    direction: str  # 下降 / 上升 / 持平
    statement: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """整改效果验证结果（全部由程序按实际数据计算）。"""

    issue_id: str
    project: str
    verified_at: str
    before_file: str = ""
    after_file: str = ""
    changes: list[MetricChange] = field(default_factory=list)
    statement: str = ""
    conclusion: str = ""
    improved: bool = False
    note: str = "以上结果由程序按整改前后的实际数据计算，未使用大模型推断。"

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "project": self.project,
            "verified_at": self.verified_at,
            "before_file": self.before_file,
            "after_file": self.after_file,
            "changes": [change.to_dict() for change in self.changes],
            "statement": self.statement,
            "conclusion": self.conclusion,
            "improved": self.improved,
            "note": self.note,
        }


def compute_metric_change(
    *,
    metric: str,
    unit: str,
    before: float,
    after: float,
    lower_is_better: bool,
) -> MetricChange:
    """按实际数据计算改善幅度（不使用任何大模型推断）。

    改善幅度 = |整改前 - 整改后| ÷ |整改前|，
    对「越低越好」的指标（如故障率）取下降幅度，对「越高越好」的指标（如满意度）取上升幅度。
    """

    base = abs(float(before))
    delta = float(after) - float(before)
    if base == 0:
        ratio = 0.0
    else:
        ratio = abs(delta) / base

    if delta == 0:
        direction = "持平"
    elif delta < 0:
        direction = "下降"
    else:
        direction = "上升"

    is_improved = (direction == "下降") if lower_is_better else (direction == "上升")
    ratio = round(ratio, 4)

    if not is_improved:
        statement = (
            f"{metric}由整改前 {before:g}{unit} 变为 {after:g}{unit}（{direction} {ratio * 100:.1f}%），"
            f"未体现改善。"
        )
    elif ratio >= 0.10:
        statement = (
            f"指标较整改前{direction}约 {ratio * 100:.1f}%，当前数据表现出明显改善。"
        )
    else:
        statement = (
            f"指标较整改前{direction}约 {ratio * 100:.1f}%，有轻微改善，建议继续观察。"
        )

    return MetricChange(
        metric=metric,
        unit=unit,
        before=float(before),
        after=float(after),
        improvement_ratio=ratio,
        is_improved=is_improved,
        direction=direction,
        statement=statement,
    )


def build_verification_result(
    *,
    issue_id: str,
    project: str,
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
    metric_definitions: Iterable[dict[str, Any]],
    before_file: str = "",
    after_file: str = "",
    verified_at: str | None = None,
) -> VerificationResult:
    """对比整改前后的指标并生成验证结论。"""

    changes: list[MetricChange] = []
    statements: list[str] = []
    for definition in metric_definitions:
        metric = str(definition["metric"])
        before = before_metrics.get(metric)
        after = after_metrics.get(metric)
        if before is None or after is None:
            continue
        change = compute_metric_change(
            metric=metric,
            unit=str(definition.get("unit") or ""),
            before=float(before),
            after=float(after),
            lower_is_better=bool(definition.get("lower_is_better", True)),
        )
        changes.append(change)
        statements.append(change.statement)

    improved = any(change.is_improved for change in changes)
    if not changes:
        conclusion = "当前数据不足以判断：对比所需指标缺失，请确认两次上传的数据包含相同指标。"
    elif improved:
        conclusion = "整改效果验证通过：至少一项异常指标出现改善。"
    else:
        conclusion = "整改效果未体现：所有对比指标均未见改善，建议继续处理并再次验证。"

    statement = " ".join(statements) if statements else conclusion
    return VerificationResult(
        issue_id=issue_id,
        project=project,
        verified_at=verified_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        before_file=before_file,
        after_file=after_file,
        changes=changes,
        statement=statement,
        conclusion=conclusion,
        improved=improved,
    )
