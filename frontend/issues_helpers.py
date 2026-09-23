"""RobotOps AI —— Phase 6 运营问题展示辅助函数（纯函数，便于单测）。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from robotops.issues import (
    ISSUE_LIST_COLUMNS,
    Issue,
    IssueService,
    IssueStatus,
    IssueStore,
)

#: 统计卡片（需求要求：待处理 / 处理中 / 已完成 / 关闭率 / 平均处理周期）
STAT_CARDS: tuple[tuple[str, str], ...] = (
    (IssueStatus.PENDING.value, "个"),
    (IssueStatus.IN_PROGRESS.value, "个"),
    (IssueStatus.COMPLETED.value, "个"),
    ("问题关闭率", ""),
    ("平均处理周期(天)", "天"),
)


def default_db_path() -> str:
    """问题数据库路径（读取时解析环境变量，便于测试与多环境部署）。"""

    from robotops.issues import store as store_module

    return os.environ.get(store_module.ENV_ISSUES_DB) or str(store_module.DEFAULT_DB_PATH)


def default_service(db_path: str | Path | None = None) -> IssueService:
    """构造问题服务（默认使用 data/issues.db）。"""

    return IssueService(IssueStore(Path(db_path) if db_path else Path(default_db_path())))


# ---------------------------------------------------------------------------
# 列表与统计
# ---------------------------------------------------------------------------
def issues_to_frame(issues: Sequence[Issue]) -> pd.DataFrame:
    """问题列表表格。"""

    if not issues:
        return pd.DataFrame(columns=list(ISSUE_LIST_COLUMNS))
    return pd.DataFrame([issue.list_row() for issue in issues], columns=list(ISSUE_LIST_COLUMNS))


def stats_cards(stats: dict[str, Any]) -> list[dict[str, str]]:
    """统计卡片数据。"""

    cards: list[dict[str, str]] = [
        {"label": "总问题数", "value": str(stats.get("总问题数", 0)), "unit": "个"}
    ]
    for label, unit in STAT_CARDS:
        value = stats.get(label)
        if label == "问题关闭率":
            ratio = float(value or 0.0)
            display = f"{ratio * 100:.1f}%"
        elif value is None:
            display = "—"
        else:
            display = str(value)
        cards.append({"label": label, "value": display, "unit": unit})
    cards.append(
        {
            "label": "未关闭问题数",
            "value": str(stats.get("未关闭问题数", 0)),
            "unit": "个",
        }
    )
    return cards


# ---------------------------------------------------------------------------
# 详情
# ---------------------------------------------------------------------------
def issue_header(issue: Issue) -> dict[str, str]:
    """问题基本信息（需求四要求的关键字段）。"""

    return {
        "问题编号": issue.issue_id,
        "项目": issue.project,
        "问题": issue.title,
        "问题类型": issue.anomaly_type,
        "优先级": issue.priority,
        "负责人": issue.owner or "未指派",
        "当前状态": issue.status,
        "创建时间": issue.created_at,
        "更新时间": issue.updated_at,
        "关闭时间": issue.closed_at or "—",
        "数据来源": issue.source_file or "—",
    }


def metrics_frame(issue: Issue) -> pd.DataFrame:
    """异常指标表（含最差设备，便于定位具体对象）。"""

    rows: list[dict[str, Any]] = []
    for item in issue.metrics:
        robot_value = item.get("robot_value")
        rows.append(
            {
                "异常类型": item.get("anomaly_type"),
                "指标": f"{item.get('metric')}（{item.get('unit')}）",
                "当前值（项目平均）": _text(item.get("actual_value")),
                "阈值": _text(item.get("threshold")),
                "判定条件": str(item.get("condition") or "—"),
                "命中记录": _text(item.get("hits")),
                "严重程度": str(item.get("severity") or "—"),
                "最差设备": item.get("robot_id") or "—",
                "最差设备取值": "—" if robot_value is None else _text(robot_value),
                "改善方向": "越低越好" if item.get("lower_is_better") else "越高越好",
            }
        )
    return pd.DataFrame(rows)


def current_data_frame(issue: Issue) -> pd.DataFrame:
    """当前数据（项目维度快照）。"""

    return pd.DataFrame(
        [
            {"数据项": str(key), "数值": _text(value)}
            for key, value in (issue.current_data or {}).items()
        ]
    )


def diagnosis_rows(issue: Issue) -> list[dict[str, Any]]:
    diagnosis = issue.diagnosis or {}
    return [
        {
            "anomaly_type": reason.get("anomaly_type", ""),
            "reason": reason.get("reason", ""),
            "confidence": reason.get("confidence", ""),
            "based_on": reason.get("based_on", ""),
            "data_gap": reason.get("data_gap", ""),
        }
        for reason in (diagnosis.get("possible_reasons") or [])
    ]


def diagnosis_summary(issue: Issue) -> str:
    diagnosis = issue.diagnosis or {}
    parts = [str(diagnosis.get("summary") or "")]
    if diagnosis.get("source"):
        parts.append(f"（诊断来源：{diagnosis['source']}）")
    return " ".join(part for part in parts if part)


def case_rows(issue: Issue) -> pd.DataFrame:
    """RAG 历史案例表。"""

    if not issue.cases:
        return pd.DataFrame(columns=["案例编号", "案例标题", "案例类型", "相关度", "来源文件", "数据性质"])
    return pd.DataFrame(
        [
            {
                "案例编号": case.get("case_id"),
                "案例标题": case.get("title"),
                "案例类型": case.get("case_type"),
                "相关度": case.get("similarity"),
                "来源文件": case.get("source_file"),
                "数据性质": case.get("data_nature"),
            }
            for case in issue.cases
        ]
    )


def recommendation_rows(issue: Issue) -> list[dict[str, Any]]:
    return [
        {
            "priority": item.get("priority", ""),
            "action": item.get("action", ""),
            "target": item.get("target", ""),
            "expected_effect": item.get("expected_effect", ""),
            "verification": item.get("verification", ""),
            "reference_case": item.get("reference_case", ""),
        }
        for item in (issue.recommendations or [])
    ]


def note_rows(issue: Issue) -> pd.DataFrame:
    """处理记录 / 备注表。"""

    if not issue.notes:
        return pd.DataFrame(columns=["时间", "类型", "操作人", "内容"])
    return pd.DataFrame(
        [
            {
                "时间": note.created_at,
                "类型": note.kind,
                "操作人": note.author or "—",
                "内容": note.text,
            }
            for note in issue.notes
        ]
    )


def verification_frame(verification: dict[str, Any]) -> pd.DataFrame:
    """整改前后对比表。"""

    changes = (verification or {}).get("changes") or []
    if not changes:
        return pd.DataFrame(columns=["指标", "整改前", "整改后", "变化方向", "改善幅度", "是否改善"])
    return pd.DataFrame(
        [
            {
                "指标": f"{item.get('metric')}（{item.get('unit')}）",
                "整改前": item.get("before"),
                "整改后": item.get("after"),
                "变化方向": item.get("direction"),
                "改善幅度": f"{float(item.get('improvement_ratio') or 0) * 100:.1f}%",
                "是否改善": "是" if item.get("is_improved") else "否",
            }
            for item in changes
        ]
    )


def verification_text(verification: dict[str, Any]) -> dict[str, str]:
    """验证结论文本（statement / conclusion / note）。"""

    return {
        "statement": str((verification or {}).get("statement") or ""),
        "conclusion": str((verification or {}).get("conclusion") or ""),
        "note": str((verification or {}).get("note") or ""),
    }


def issue_options(issues: Iterable[Issue]) -> list[str]:
    """详情下拉框选项（问题编号 + 项目）。"""

    return [f"{issue.issue_id}｜{issue.project}｜{issue.status}" for issue in issues]


def _text(value: Any) -> str:
    """把混合类型的展示值统一转成字符串，避免 Streamlit/Arrow 类型告警。"""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
