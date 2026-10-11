"""RobotOps AI —— Phase 10 前端展示辅助：分析依据（纯函数，不依赖 Streamlit）。

把工作流 state 中**已经由程序计算好**的内容整理成「分析依据」面板所需的三个部分：

1. 当前数据 / 程序计算结果（指标、异常判定、趋势、项目对比）；
2. 历史案例（含 case_id、标题、类别、相似度、来源文件、内容）；
3. 可能原因与优化建议（AI 输出，必须标注为推测，并带 evidence）。

所有数值都来自程序计算结果，本模块不做任何推断。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from robotops.rag.config import NO_RELEVANT_CASE_MESSAGE

#: 可能原因必须标注的措辞（前端统一使用，避免被读成事实）
INFERENCE_LABEL = "可能原因（推测）"
INFERENCE_DISCLAIMER = "以上「可能原因」均为 AI 推测，不代表已确认的事实；请结合数据核查后再处置。"


def evidence_panel_data(state: dict[str, Any] | None) -> dict[str, Any]:
    """整理「分析依据」面板数据。"""

    # 惰性导入：避免在页面脚本导入阶段就去解析 app 包（见 frontend/app.py 的 sys.path 说明）
    from app.agents.diagnosis import build_findings, program_facts

    state = state or {}
    facts = program_facts(state)
    diagnosis = state.get("diagnosis") or {}
    recommendations = state.get("recommendations") or {}
    cases = list(state.get("retrieved_cases") or [])
    retrieval = state.get("retrieval") or {}

    reasons = [item for item in (diagnosis.get("possible_reasons") or []) if isinstance(item, dict)]
    items = [item for item in (recommendations.get("items") or []) if isinstance(item, dict)]

    case_rows = [
        {
            "案例编号": case.get("case_id"),
            "案例标题": case.get("title"),
            "类别": case.get("category"),
            "相似度": case.get("similarity_score", case.get("similarity")),
            "来源文件": case.get("source_file"),
        }
        for case in cases
    ]
    reason_rows = [
        {
            INFERENCE_LABEL: reason.get("reason", ""),
            "置信度": reason.get("confidence", "-"),
            "分析依据（程序计算）": "；".join(_as_list(reason.get("evidence"))),
            "缺失数据": reason.get("data_gap") or "当前数据不足以判断",
        }
        for reason in reasons
    ]
    recommendation_rows = [
        {
            "建议动作": item.get("action", ""),
            "优先级": item.get("priority", "-"),
            "分析依据（程序计算）": "；".join(_as_list(item.get("evidence"))),
        }
        for item in items
    ]

    case_message = "" if cases else str(retrieval.get("message") or NO_RELEVANT_CASE_MESSAGE)

    return {
        "available": bool(
            facts["current"] or facts["program"] or cases or reasons or items
        ),
        "findings": diagnosis.get("findings") or build_findings(state),
        "current": list(facts["current"]),
        "program": list(facts["program"]),
        "trend": list(facts["trend"]),
        "comparison": list(facts["comparison"]),
        "cases": cases,
        "cases_frame": pd.DataFrame(
            case_rows, columns=["案例编号", "案例标题", "类别", "相似度", "来源文件"]
        ),
        "case_message": case_message,
        "possible_reasons": reasons,
        "reasons_frame": pd.DataFrame(
            reason_rows, columns=[INFERENCE_LABEL, "置信度", "分析依据（程序计算）", "缺失数据"]
        ),
        "recommendation_rows": recommendation_rows,
        "recommendations_frame": pd.DataFrame(
            recommendation_rows, columns=["建议动作", "优先级", "分析依据（程序计算）"]
        ),
        "inference_label": INFERENCE_LABEL,
        "disclaimer": INFERENCE_DISCLAIMER,
        "case_disclaimer": "历史案例为模拟案例（虚构内容），仅供参考，不代表当前项目数据。",
    }


def current_data_frame(state: dict[str, Any] | None) -> pd.DataFrame:
    """当前数据事实表（指标 + 数值 + 单位）。"""

    state = state or {}
    metrics = state.get("metrics") or {}
    values = metrics.get("values") or {}
    units = metrics.get("units") or {}
    rows = [
        {"指标": name, "数值": value, "单位": units.get(name, "")}
        for name, value in values.items()
    ]
    return pd.DataFrame(rows, columns=["指标", "数值", "单位"])


def case_source_frame(state: dict[str, Any] | None) -> pd.DataFrame:
    """历史案例来源表（case_id / 标题 / 类别 / 相似度 / 来源文件 / 内容摘要）。"""

    state = state or {}
    rows: list[dict[str, Any]] = []
    for case in state.get("retrieved_cases") or []:
        content = str(case.get("content") or "").replace("\n", " ")
        if len(content) > 160:
            content = content[:160].rstrip() + "…"
        rows.append(
            {
                "案例编号": case.get("case_id"),
                "案例标题": case.get("title"),
                "类别": case.get("category"),
                "相似度": case.get("similarity_score", case.get("similarity")),
                "来源文件": case.get("source_file"),
                "内容摘要": content,
                "数据性质": case.get("data_nature", "模拟案例（虚构）"),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["案例编号", "案例标题", "类别", "相似度", "来源文件", "内容摘要", "数据性质"],
    )


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]

