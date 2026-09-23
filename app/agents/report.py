"""Phase 4 —— Report Agent（大模型 + 程序模板兜底）。

职责（单一）：汇总整个工作流的产出，生成完整的机器人运营分析报告。

- 正常情况下由大模型按固定小节结构撰写报告；
- 大模型不可用或调用失败时，**改用程序模板生成报告**（报告中明确标注降级原因），
  保证工作流始终能产出 ``state["final_report"]``，同时把错误记录到 ``state["errors"]``。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from robotops.llm.errors import LLMError

from ..graph.prompts import PROMPT_VERSION, REPORT_SYSTEM_PROMPT, build_report_prompt
from ..graph.state import WorkflowState
from .base import BaseAgent, has_llm_failure


class ReportAgent(BaseAgent):
    tag = "Report Agent"
    stage = "report"

    def execute(self, state: WorkflowState) -> dict[str, Any]:
        notes = workflow_notes(state)
        errors = list(state.get("errors") or [])
        failed_stages = list(state.get("failed_stages") or [])
        source = "llm"
        model = ""
        usage: dict[str, Any] = {}
        degradation_note = ""

        llm_available = bool(state.get("use_llm", True)) and not has_llm_failure(state)
        if llm_available:
            prompt = build_report_prompt(
                raw_data_summary=state.get("raw_data_summary") or {},
                metrics=state.get("metrics") or {},
                abnormal_projects=state.get("abnormal_projects") or [],
                anomaly_summary=state.get("anomaly_summary") or [],
                diagnosis=state.get("diagnosis") or {},
                retrieved_cases=state.get("retrieved_cases") or [],
                recommendations=state.get("recommendations") or {},
                thresholds=state.get("thresholds") or {},
                workflow_notes=notes,
            )
            try:
                content, result = self.chat_text(REPORT_SYSTEM_PROMPT, prompt)
                report = content.strip()
                model = result.model
                usage = result.usage
            except LLMError as error:
                message = self._message_of(error)
                self.logger.agent_failed(self.tag, f"大模型汇总失败，改用程序模板：{message}")
                errors.append(
                    {
                        "stage": self.stage,
                        "agent": self.tag,
                        "kind": "报告汇总降级",
                        "error_type": type(error).__name__,
                        "message": message,
                        "hint": getattr(error, "hint", "") or "",
                        "is_llm_error": True,
                        "is_rag_error": False,
                        "occurred_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                failed_stages.append(self.stage)
                source = "offline_template"
                degradation_note = f"大模型汇总失败，已改用程序模板生成报告：{message}"
        elif not state.get("use_llm", True):
            source = "offline_template"
            degradation_note = "本次运行使用 --no-llm（离线模板模式），报告由程序生成"
        else:
            source = "offline_template"
            degradation_note = "前序大模型调用失败，报告已由程序模板生成"

        if source != "llm":
            report = self._template_report(state, degradation_note=degradation_note)

        return {
            "final_report": report,
            "report_meta": {
                "source": source,
                "model": model,
                "usage": usage,
                "prompt_version": PROMPT_VERSION,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "characters": len(report),
                "degradation_note": degradation_note,
            },
            "errors": errors,
            "failed_stages": failed_stages,
        }

    def summarize(self, update: dict[str, Any], state: WorkflowState) -> str:
        meta = update.get("report_meta") or {}
        return f"生成报告 {meta.get('characters', len(update.get('final_report') or ''))} 字符（{meta.get('source')}）"

    # -- 程序模板 ---------------------------------------------------------
    def _template_report(self, state: WorkflowState, *, degradation_note: str) -> str:
        metrics = (state.get("metrics") or {}).get("values", {})
        units = (state.get("metrics") or {}).get("units", {})
        summary = state.get("raw_data_summary") or {}
        anomaly_summary = state.get("anomaly_summary") or []
        abnormal_projects = state.get("abnormal_projects") or []
        diagnosis = state.get("diagnosis") or {}
        cases = state.get("retrieved_cases") or []
        recommendations = state.get("recommendations") or {}

        lines: list[str] = ["# RobotOps AI 机器人运营分析报告（多 Agent 工作流）", ""]
        lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- 数据源：{summary.get('数据源', '-')}")
        lines.append(f"- 分析窗口：{summary.get('分析窗口', '-')}")
        lines.append(f"- 工作流版本：{state.get('workflow_version', '-')}")
        lines.append("- 报告来源：程序模板（未使用大模型撰写）")
        if degradation_note:
            lines.append(f"- 降级说明：{degradation_note}")
        lines.append("")

        lines.append("## 一、运营概览")
        lines.append("")
        lines.append(
            f"本期覆盖 {metrics.get('项目数量', '-')} 个项目、{metrics.get('机器人数量', '-')} 台机器人，"
            f"有效记录 {summary.get('清洗后记录数', '-')} 条（原始 {summary.get('原始记录数', '-')} 条，"
            f"剔除 {summary.get('剔除记录数', '-')} 条）。"
        )
        if state.get("has_anomalies"):
            lines.append(
                f"程序按阈值判定共命中异常 {state.get('anomaly_count', 0)} 条，"
                f"涉及 {len(abnormal_projects)} 个项目，需要进一步诊断与处置。"
            )
        else:
            lines.append("程序按阈值判定：本期未发现超过阈值的异常，无需进入诊断与建议环节。")
        lines.append("")

        lines.append("## 二、关键指标")
        lines.append("")
        lines.append("| 指标 | 数值 | 单位 |")
        lines.append("| --- | --- | --- |")
        for name, value in metrics.items():
            lines.append(f"| {name} | {value} | {units.get(name, '-')} |")
        lines.append("")

        lines.append("## 三、异常发现")
        lines.append("")
        if not anomaly_summary:
            lines.append("未发现超过阈值的异常（异常判定由程序按固定阈值完成）。")
        else:
            lines.append("以下异常均由程序按固定阈值判定，非大模型判断：")
            lines.append("")
            lines.append("| 异常类型 | 判定条件 | 命中记录 | 涉及项目 | 涉及机器人 | 高/中/低 |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for item in anomaly_summary:
                lines.append(
                    f"| {item.get('anomaly_type')} | {item.get('condition')} | {item.get('hits')} | "
                    f"{item.get('projects_involved')} | {item.get('robots_involved')} | "
                    f"{item.get('high')}/{item.get('medium')}/{item.get('low')} |"
                )
            lines.append("")
            lines.append("异常集中在以下项目：")
            lines.append("")
            lines.append("| 项目 | 异常记录 | 涉及机器人 | 主要异常类型 |")
            lines.append("| --- | --- | --- | --- |")
            for item in abnormal_projects[:8]:
                lines.append(
                    f"| {item.get('project')} | {item.get('anomaly_records')} | "
                    f"{item.get('affected_robots')} | {item.get('main_anomaly_types')} |"
                )
        lines.append("")

        lines.append("## 四、可能原因（推测）")
        lines.append("")
        reasons = diagnosis.get("possible_reasons") or []
        if not state.get("has_anomalies"):
            lines.append("本期无异常，不需要原因推测。")
        elif not reasons:
            lines.append("当前数据不足以判断，且未获得可用的诊断结论。")
        else:
            lines.append(f"诊断来源：{diagnosis.get('source', '-')}；以下内容均为**推测**，需补充数据后确认。")
            lines.append("")
            for index, item in enumerate(reasons, start=1):
                lines.append(f"{index}. **{item.get('reason', '')}**（置信度：{item.get('confidence', '-')}）")
                if item.get("based_on"):
                    lines.append(f"   - 判断依据：{item['based_on']}")
                lines.append(f"   - 缺失数据：{item.get('data_gap') or '当前数据不足以判断'}")
            lines.append("")

        lines.append("## 五、历史相似案例（仅供参考）")
        lines.append("")
        if not cases:
            lines.append("未检索到足够相关的历史案例（知识库中无相似案例或相似度低于阈值）。")
        else:
            lines.append("以下案例为**模拟案例（虚构内容）**，仅作参考，不代表当前项目数据。")
            lines.append("")
            for index, case in enumerate(cases, start=1):
                lines.append(
                    f"{index}. **[{case.get('case_id')}] {case.get('title')}**"
                    f"｜类型：{case.get('case_type')}｜相关度：{case.get('similarity')}"
                    f"｜来源文件：{case.get('source_file')}"
                )
                if case.get("matched_sections"):
                    lines.append(f"   - 命中片段：{'、'.join(case['matched_sections'])}")
            lines.append("")

        lines.append("## 六、优化建议")
        lines.append("")
        items = recommendations.get("items") or []
        if not state.get("has_anomalies"):
            lines.append("本期无异常，暂不需要针对性优化建议，建议保持现有运维节奏。")
        elif not items:
            lines.append("当前数据不足以判断，未生成可执行建议。")
        else:
            lines.append(f"建议来源：{recommendations.get('source', '-')}。")
            lines.append("")
            for index, item in enumerate(items, start=1):
                lines.append(f"{index}. **[{item.get('priority', '-')}] {item.get('action', '')}**")
                if item.get("target"):
                    lines.append(f"   - 对象：{item['target']}")
                if item.get("expected_effect"):
                    lines.append(f"   - 预期效果：{item['expected_effect']}")
                if item.get("verification"):
                    lines.append(f"   - 验证方式：{item['verification']}")
                if item.get("reference_case") and item["reference_case"] != "无":
                    lines.append(f"   - 参考案例：{item['reference_case']}")
            lines.append("")

        lines.append("## 七、数据与口径说明")
        lines.append("")
        lines.append(f"- 异常判定阈值：{summary.get('阈值', {})}")
        lines.append(f"- 故障率判定粒度：{summary.get('故障率判定粒度', '-')}")
        lines.append(f"- 清洗问题条目：{summary.get('清洗问题条目数', '-')} 条")
        lines.append("- 关键指标均由 Phase 1 的 Pandas 模块计算，大模型未参与计算。")
        errors = state.get("errors") or []
        if errors:
            lines.append("")
            lines.append("### 工作流异常与降级说明")
            lines.append("")
            for record in errors:
                lines.append(
                    f"- [{record.get('agent')}] {record.get('kind')}：{record.get('message')}"
                    + (f"（建议：{record['hint']}）" if record.get("hint") else "")
                )
        lines.append("")
        lines.append("### 工作流执行轨迹")
        lines.append("")
        for step in state.get("steps") or []:
            lines.append(
                f"- [{step.get('agent')}] {step.get('status')}"
                f"（{step.get('duration_seconds')}s）{step.get('detail') or ''}"
            )
        lines.append("")
        return "\n".join(lines)


def workflow_notes(state: WorkflowState) -> list[str]:
    """把工作流执行情况整理成给报告模型看的说明。"""

    notes = [
        f"异常判定完全由程序完成（阈值见数据），大模型不得重新判断异常。",
        f"核心指标由 Phase 1 的 Pandas 模块计算，禁止模型修改或重算。",
    ]
    if not state.get("has_anomalies"):
        notes.append("本次未发现超过阈值的异常，未执行诊断、RAG 与建议环节。")
    if not state.get("use_knowledge", True):
        notes.append("本次运行使用 --no-rag，未检索历史案例。")
    elif not (state.get("retrieved_cases") or []):
        notes.append("本次未检索到足够相关的历史案例，报告中必须明确说明。")
    for record in state.get("errors") or []:
        notes.append(
            f"工作流异常：{record.get('agent')} 阶段 {record.get('kind')} - {record.get('message')}"
        )
    if state.get("is_offline"):
        notes.append("本次运行处于离线/降级模式（--no-llm 或大模型不可用），诊断与建议为程序模板结论。")
    return notes
