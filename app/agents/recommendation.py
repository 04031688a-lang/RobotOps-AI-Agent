"""Phase 4 —— Recommendation Agent（大模型）。

职责（单一）：综合当前数据 + 异常 + 诊断结论 + 历史案例，生成可执行的运营优化建议。

历史案例只作为参考；引用时必须写明案例编号；不得把案例数据当作当前项目事实。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..graph.prompts import (
    PROMPT_VERSION,
    RECOMMENDATION_SYSTEM_PROMPT,
    build_recommendation_prompt,
)
from ..graph.state import WorkflowState
from .base import BaseAgent

#: 异常类型 -> 模板建议（离线或降级时使用）
OFFLINE_ACTIONS: dict[str, dict[str, str]] = {
    "故障率偏高": {
        "action": "对该机器人做单机专项复盘：核对故障与维修工单，按部件与故障类型归类后再决定更换或调整作业区域",
        "target": "故障率超过阈值的机器人",
        "expected_effect": "定位重复故障根因，使故障率向同项目平均水平收敛",
        "verification": "补齐故障类型与部件字段后，重新计算该设备累计故障率",
    },
    "满意度偏低": {
        "action": "先补齐投诉与工单响应数据，再针对作业质量（清洁效果、巡检到位率）做现场抽查与整改",
        "target": "满意度低于阈值的项目或机器人",
        "expected_effect": "定位满意度下降的真实驱动因素，避免误判为设备问题",
        "verification": "跟踪投诉闭环率与响应时长，观察后续满意度评分变化",
    },
    "运行率偏低": {
        "action": "核查充电策略与排班安排，确认是否存在长时间待机或停机未处置，必要时调整计划运行时长口径",
        "target": "运行率低于阈值的机器人",
        "expected_effect": "减少无效待机，提高运行时长利用率",
        "verification": "补充停机时段与充电日志后，重新统计运行率与欠运行小时数",
    },
    "节降率偏低": {
        "action": "拆分成本结构（能耗、人工、耗材、折旧、维保）并与同类项目对比，确认节降率基线口径是否变化",
        "target": "节降率低于阈值的项目",
        "expected_effect": "识别主要成本项，恢复节降率水平",
        "verification": "按周跟踪单位运行小时成本与节降率变化",
    },
    "default": {
        "action": "补充异常对象的明细数据并安排现场核查，确认原因后再制定针对性措施",
        "target": "存在程序判定异常的对象",
        "expected_effect": "在数据充分的前提下定位问题",
        "verification": "跟踪对应异常指标的变化趋势",
    },
}


class RecommendationAgent(BaseAgent):
    tag = "Recommendation Agent"
    stage = "recommendation"
    requires_llm = True

    def execute(self, state: WorkflowState) -> dict[str, Any]:
        if not state.get("use_llm", True):
            self.logger.info("离线模式：建议 Agent 使用确定性模板（未调用大模型）")
            return {
                "recommendations": build_offline_recommendations(
                    state, note="离线模式（--no-llm）"
                )
            }

        retrieval = state.get("retrieval") or {}
        prompt = build_recommendation_prompt(
            metrics=state.get("metrics") or {},
            abnormal_projects=state.get("abnormal_projects") or [],
            diagnosis=state.get("diagnosis") or {},
            retrieved_cases=state.get("retrieved_cases") or [],
            retrieval_note=str(retrieval.get("reason") or ""),
            thresholds=state.get("thresholds") or {},
        )
        payload, result = self.chat_json(RECOMMENDATION_SYSTEM_PROMPT, prompt)

        recommendations = {
            "source": "llm",
            "model": result.model,
            "prompt_version": PROMPT_VERSION,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": str(payload.get("summary") or "").strip(),
            "items": _normalize_items(payload.get("recommendations")),
            "usage": result.usage,
        }
        if not recommendations["items"]:
            recommendations["note"] = "模型未返回可用的建议，已按空列表记录"
        return {"recommendations": recommendations}

    def summarize(self, update: dict[str, Any], state: WorkflowState) -> str:
        items = (update.get("recommendations") or {}).get("items") or []
        return f"输出 {len(items)} 条优化建议"


def build_offline_recommendations(state: WorkflowState, *, note: str) -> dict[str, Any]:
    """确定性建议模板：按命中的异常类型给出保守措施。"""

    anomaly_types = [
        str(item.get("anomaly_type")) for item in (state.get("anomaly_summary") or [])
    ]
    items: list[dict[str, Any]] = []
    for anomaly_type in dict.fromkeys(anomaly_types):
        template = OFFLINE_ACTIONS.get(anomaly_type, OFFLINE_ACTIONS["default"])
        items.append(
            {
                **template,
                "priority": "高" if anomaly_type in {"故障率偏高", "运行率偏低"} else "中",
                "anomaly_type": anomaly_type,
                "reference_case": "无",
            }
        )

    return {
        "source": "offline_template",
        "model": "",
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": note,
        "summary": "以下建议由程序按异常类型生成，未经过大模型分析；实施前建议先补齐缺失数据。",
        "items": items,
        "usage": {},
    }


def _normalize_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, Any]] = []
    for item in value[:6]:
        if isinstance(item, dict):
            items.append(
                {
                    "action": str(item.get("action") or "").strip(),
                    "priority": str(item.get("priority") or "中").strip(),
                    "target": str(item.get("target") or "").strip(),
                    "expected_effect": str(item.get("expected_effect") or "").strip(),
                    "verification": str(item.get("verification") or "").strip(),
                    "reference_case": str(item.get("reference_case") or "无").strip(),
                }
            )
        elif isinstance(item, str) and item.strip():
            items.append(
                {
                    "action": item.strip(),
                    "priority": "中",
                    "target": "",
                    "expected_effect": "",
                    "verification": "",
                    "reference_case": "无",
                }
            )
    return [item for item in items if item["action"]]

