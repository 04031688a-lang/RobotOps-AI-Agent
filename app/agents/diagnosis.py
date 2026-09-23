"""Phase 4 —— Diagnosis Agent（大模型）。

职责（单一）：只解释异常的可能原因（推测），不做建议、不写报告。

- 输入：当前项目数据（核心指标 + 程序判定的异常）；
- 输出：``state["diagnosis"]``（summary + possible_reasons + 元信息）；
- 不允许虚构数据，不允许重新判定异常（阈值与异常由程序给出）。

当 ``use_llm=False`` 或大模型不可用时使用**确定性模板**给出保守结论，
并明确标注「当前数据不足以判断」，保证工作流仍可完成。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..graph.prompts import PROMPT_VERSION, DIAGNOSIS_SYSTEM_PROMPT, build_diagnosis_prompt
from ..graph.state import WorkflowState
from .base import BaseAgent

#: 异常类型 -> 模板推测原因（离线或降级时使用，均为保守表述）
OFFLINE_REASONS: dict[str, dict[str, str]] = {
    "故障率偏高": {
        "reason": "推测：该设备可能存在部件劣化、安装工艺或作业强度偏高等问题，导致同类故障重复发生",
        "based_on": "该机器人的故障率由程序判定超过阈值，且故障在统计周期内集中出现",
        "data_gap": "当前数据不足以判断，缺少故障类型、故障部件、维修工单与作业区域数据",
    },
    "满意度偏低": {
        "reason": "推测：满意度下降可能与作业质量（清洁效果、巡检到位率）或服务响应时效有关",
        "based_on": "满意度由程序判定低于阈值，而同期设备运行与故障指标基本正常",
        "data_gap": "当前数据不足以判断，缺少投诉内容、工单响应时长与服务质量抽查记录",
    },
    "运行率偏低": {
        "reason": "推测：运行率偏低可能与充电策略、排班安排、停机处置效率或任务不足有关",
        "based_on": "运行率由程序判定低于阈值，而故障次数与维修次数并未同步升高",
        "data_gap": "当前数据不足以判断，缺少停机时段记录、排班计划、充电日志与任务派发量",
    },
    "节降率偏低": {
        "reason": "推测：节降率偏低可能与成本结构（能耗、人工、耗材）或作业方案未优化有关",
        "based_on": "节降率由程序判定低于阈值，且单位运行小时成本高于同类项目",
        "data_gap": "当前数据不足以判断，缺少成本明细拆分与节降率计算基线说明",
    },
    "default": {
        "reason": "推测：异常可能与设备状态、作业安排或服务流程有关",
        "based_on": "程序在该维度判定出超过阈值的异常记录",
        "data_gap": "当前数据不足以判断，缺少支撑根因分析的明细字段",
    },
}


class DiagnosisAgent(BaseAgent):
    tag = "Diagnosis Agent"
    stage = "diagnosis"
    requires_llm = True

    def execute(self, state: WorkflowState) -> dict[str, Any]:
        if not state.get("use_llm", True):
            self.logger.info("离线模式：诊断 Agent 使用确定性模板（未调用大模型）")
            return {"diagnosis": build_offline_diagnosis(state, note="离线模式（--no-llm）")}

        prompt = build_diagnosis_prompt(
            raw_data_summary=state.get("raw_data_summary") or {},
            metrics=state.get("metrics") or {},
            abnormal_projects=state.get("abnormal_projects") or [],
            anomaly_summary=state.get("anomaly_summary") or [],
            abnormal_robots=state.get("abnormal_robots") or [],
            thresholds=state.get("thresholds") or {},
        )
        payload, result = self.chat_json(DIAGNOSIS_SYSTEM_PROMPT, prompt)

        diagnosis = {
            "source": "llm",
            "model": result.model,
            "prompt_version": PROMPT_VERSION,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": str(payload.get("summary") or "").strip(),
            "possible_reasons": _normalize_reasons(payload.get("possible_reasons")),
            "usage": result.usage,
        }
        if not diagnosis["possible_reasons"]:
            diagnosis["note"] = "模型未返回可用的推测原因，已按空列表记录"
        return {"diagnosis": diagnosis}

    def summarize(self, update: dict[str, Any], state: WorkflowState) -> str:
        reasons = (update.get("diagnosis") or {}).get("possible_reasons") or []
        return f"输出 {len(reasons)} 条推测原因"


def build_offline_diagnosis(state: WorkflowState, *, note: str) -> dict[str, Any]:
    """确定性诊断模板：按命中的异常类型逐条给出保守推测。"""

    anomaly_types = [
        str(item.get("anomaly_type")) for item in (state.get("anomaly_summary") or [])
    ]
    reasons: list[dict[str, Any]] = []
    for anomaly_type in dict.fromkeys(anomaly_types):
        template = OFFLINE_REASONS.get(anomaly_type, OFFLINE_REASONS["default"])
        reasons.append(
            {**template, "anomaly_type": anomaly_type, "confidence": "低"},
        )

    return {
        "source": "offline_template",
        "model": "",
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": note,
        "summary": (
            "以下原因由程序按异常类型生成的保守推测，未经过大模型分析；"
            "当前数据不足以判断根因，需补充明细数据后确认。"
        ),
        "possible_reasons": reasons,
        "usage": {},
    }


def _normalize_reasons(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    reasons: list[dict[str, Any]] = []
    for item in value[:6]:
        if isinstance(item, dict):
            reasons.append(
                {
                    "reason": str(item.get("reason") or "").strip(),
                    "confidence": str(item.get("confidence") or "中").strip(),
                    "based_on": str(item.get("based_on") or "").strip(),
                    "data_gap": str(item.get("data_gap") or "").strip(),
                }
            )
        elif isinstance(item, str) and item.strip():
            reasons.append(
                {
                    "reason": item.strip(),
                    "confidence": "中",
                    "based_on": "",
                    "data_gap": "",
                }
            )
    return [item for item in reasons if item["reason"]]

