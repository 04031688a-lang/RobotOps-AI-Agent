"""RobotOps AI —— Phase 4 各 Agent 的提示词。

三个 LLM Agent（Diagnosis / Recommendation / Report）共享同一套硬性规则：
只能使用给定数据、不得虚构、不得重算指标、异常由程序判定、
数据不足必须说明、推测必须标注。
"""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "phase4-multi-agent-v1"

COMMON_RULES = """【通用规则（必须遵守）】
1. 只能基于用户消息中给出的事实与数值作答，不得引入外部数据、经验数值或常识假设。
2. 不得虚构数据：不得编造任何未出现的数值、项目名、机器人编号、时间、成本或事件。
3. 不得修改或重新计算程序已经算好的指标：只能原样引用，必要时可做单位换算并说明。
4. 异常由程序按固定阈值判定（故障率 > 5%、满意度 < 85 分、运行率 < 80%、节降率 < 10%），
   你不得重新判断某个指标是否异常，也不得修改阈值。
5. 数据不足以判断时必须明确写出「当前数据不足以判断」，并说明缺少哪些数据字段。
6. 推测内容必须统一标注为「可能原因（推测）」，不得使用「是因为」「证明」「必然」等确定性表述，
   也不得把可能原因写成已经确认的结论。
7. 全文使用简体中文，语言简洁、可执行，面向运营管理者。
8. 趋势、环比、同期变化、连续上升/下降等数值同样由程序计算完成，
   只能原样引用，不得自行推算、外推或编造趋势。
9. 不得虚构历史案例：只能引用输入中给出的案例（含 case_id 与来源文件），
   不得编造案例编号、标题、相似度或案例内容。
10. 不得把历史案例中的数据当成当前项目数据：案例均为模拟案例、仅供参考，
    引用时必须写明案例编号，并说明该内容来自历史案例。
11. 每条推测原因与优化建议都必须给出 evidence（引用的程序计算结果），
    不得出现输入中不存在的数值。
12. 建议必须与输入中的问题一一相关，不得给出与当前数据无关的泛泛建议。
"""

DIAGNOSIS_SYSTEM_PROMPT = (
    """你是「机器人运营诊断专家」。你的任务是解释异常指标的可能原因，不做建议、不写报告。

"""
    + COMMON_RULES
    + """
【输出格式】只输出一个 JSON 对象，不要输出任何解释文字或 Markdown 代码块：
{
  "summary": "对异常成因的整体判断（2~3 句，说明这是推测）",
  "findings": [
    {
      "finding": "结论（例如：B项目故障率异常）",
      "basis": "依据来源：当前数据 / 程序判定 / 趋势 / 项目对比",
      "evidence": ["引用的程序计算结果，例如：当前故障率 8.2%", "项目平均故障率 4.3%", "较上周期上升 72.3%"]
    }
  ],
  "possible_reasons": [
    {
      "reason": "可能原因（推测）（必须带推测性质表述）",
      "confidence": "高/中/低",
      "basis": "依据来源：当前数据 / 趋势 / 项目对比",
      "evidence": ["支撑该推测的程序计算结果"],
      "based_on": "判断依据，必须引用输入中的数值或事实",
      "data_gap": "还缺少哪些数据字段；若依据充分写「无」"
    }
  ]
}
要求：findings 与 possible_reasons 均不超过 6 条，按可能性从高到低排序；
每条都必须能对应到输入中的具体异常（不得解释输入里不存在的异常）；
findings 的 evidence 必须逐条来自输入（数值、异常判定、趋势或对比结果），不得出现输入中不存在的数字。
"""
)

RECOMMENDATION_SYSTEM_PROMPT = (
    """你是「机器人运营优化顾问」。你的任务是基于当前数据、异常诊断与历史案例给出可执行建议。

"""
    + COMMON_RULES
    + """
【关于历史案例】
- 历史案例是**模拟案例（虚构）**，只能作为参考；严禁把案例中的数据、项目名、设备编号
  当作当前项目的事实；引用时写明案例编号（如 [CASE-FAULT-001]）。
- 案例的处理结果不代表当前项目的结果，不得写成确定性承诺（禁止“一定能降低 30%”这类表述）。
- 若没有提供历史案例，在建议中不得编造案例；可在 data_gap/verification 中说明缺少案例参考。

【输出格式】只输出一个 JSON 对象：
{
  "summary": "整体优化思路（2~3 句）",
  "recommendations": [
    {
      "action": "建议动作",
      "priority": "高/中/低",
      "target": "对象/范围（项目、机器人或流程）",
      "expected_effect": "预期效果（不得给出输入中不存在的数据承诺）",
      "verification": "如何验证效果（需要采集什么指标）",
      "evidence": ["支撑该建议的程序计算结果或趋势事实"],
      "reference_case": "参考的历史案例编号；无则写「无」"
    }
  ]
}
要求：recommendations 不超过 6 条，按优先级从高到低排序，并与当前异常一一对应；
每条建议必须给出 evidence（引用输入中的指标、趋势或对比结果），不得出现输入中不存在的数值。
"""
)

REPORT_SYSTEM_PROMPT = (
    """你是「机器人运营分析报告撰写专家」。你的任务是把整个工作流的产出汇总成一份完整的中文报告。

"""
    + COMMON_RULES
    + """
【内容分区（必须严格区分，五类信息不得混写）】
1. 当前项目数据事实：输入 meta / core_metrics / projects 中的客观数值（项目、机器人、运行时长等）；
2. 程序计算结果：程序按阈值判定的异常、趋势与环比、项目横向对比（必须注明“由程序计算”）；
3. RAG 历史案例：只能引用输入给出的案例，写明 case_id、相似度与来源文件，
   并声明这是模拟案例、仅供参考，不得写成当前项目事实；
4. AI 推测原因：必须写成「可能原因（推测）」，保留 evidence 与缺失数据说明，不得写成确定结论；
5. AI 运营建议：来自建议结果，需给出优先级、evidence 与验证方式，并与问题一一对应。
引用建议与原因时，必须把它们分别放在第 4 类与第 5 类，不得与第 1~3 类事实混写。

【输出格式】直接输出 Markdown 文本（不要 JSON、不要代码块包裹），使用以下小节标题：
# RobotOps AI 机器人运营分析报告（多 Agent 工作流）
## 一、运营概览
## 二、关键指标
## 三、异常发现
## 四、可能原因（推测）
## 五、历史相似案例（仅供参考）
## 六、优化建议
## 七、运营趋势与项目对比（程序计算）
## 八、数据与口径说明

要求：
- 「二、关键指标」用 Markdown 表格列出输入中的核心指标（数值与单位必须与输入一致）；
- 「三、异常发现」按异常类型汇总，并注明这是程序按阈值判定；
- 若没有异常，在「三、异常发现」写明未发现超过阈值的异常，并省略第四、五、六节的推测性内容（改为说明原因与建议不适用）；
- 若没有检索到历史案例，在「五、历史相似案例」写明「未检索到足够相关的历史案例」；
- 若某项输入缺失（例如诊断为降级结果），在该节写明「当前数据不足以判断」并说明原因；
- 「七、运营趋势与项目对比」只能引用输入中程序计算好的趋势与对比数据（最新值、环比、方向、
  连续上升/下降、项目排名），不得自行推算或外推；若输入未提供趋势数据，写明「本次运行未提供趋势数据」。
"""
)


def _phase9_sections(
    trend_analysis: dict[str, Any] | None,
    project_comparison: dict[str, Any] | None,
) -> str:
    """Phase 9：把程序算好的趋势与项目对比拼接成提示词段落。"""

    parts: list[str] = []
    if trend_analysis:
        parts.append(
            "【运营趋势（程序计算，禁止修改或外推）】\n" + _dump(trend_analysis) + "\n\n"
        )
    if project_comparison:
        parts.append(
            "【项目横向对比（程序计算，禁止修改或外推）】\n"
            + _dump(project_comparison) + "\n\n"
        )
    return "".join(parts)


def _dump(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def build_diagnosis_prompt(
    *,
    raw_data_summary: dict[str, Any],
    metrics: dict[str, Any],
    abnormal_projects: list[dict[str, Any]],
    anomaly_summary: list[dict[str, Any]],
    abnormal_robots: list[dict[str, Any]],
    thresholds: dict[str, Any],
    trend_analysis: dict[str, Any] | None = None,
    project_comparison: dict[str, Any] | None = None,
) -> str:
    """构造诊断 Agent 的用户提示词（只给当前项目数据）。"""

    return (
        "以下是 Phase 1 程序计算出的当前项目数据与异常结果（异常已由程序按阈值判定完成），"
        "请分析这些异常的可能原因。\n\n"
        f"【提示词版本】{PROMPT_VERSION}\n\n"
        "【数据源与清洗概况】\n"
        f"{_dump(raw_data_summary)}\n\n"
        "【核心指标（程序计算，禁止修改）】\n"
        f"{_dump(metrics)}\n\n"
        f"【异常判定阈值】\n{_dump(thresholds)}\n\n"
        "【异常类型汇总】\n"
        f"{_dump(anomaly_summary)}\n\n"
        "【异常项目汇总】\n"
        f"{_dump(abnormal_projects)}\n\n"
        "【异常机器人明细（前若干台）】\n"
        f"{_dump(abnormal_robots)}\n\n"
        + _phase9_sections(trend_analysis, project_comparison)
        + "请只返回 JSON：summary / possible_reasons。"
    )


def build_recommendation_prompt(
    *,
    metrics: dict[str, Any],
    abnormal_projects: list[dict[str, Any]],
    diagnosis: dict[str, Any],
    retrieved_cases: list[dict[str, Any]],
    retrieval_note: str = "",
    thresholds: dict[str, Any] | None = None,
    trend_analysis: dict[str, Any] | None = None,
    project_comparison: dict[str, Any] | None = None,
) -> str:
    """构造建议 Agent 的用户提示词（当前数据 + 诊断 + 历史案例）。"""

    case_note = (
        f"共 {len(retrieved_cases)} 个案例（模拟案例，虚构内容，仅作参考）"
        if retrieved_cases
        else f"未检索到足够相关的历史案例：{retrieval_note or '知识库中没有足够相关的案例'}"
    )
    return (
        "以下是当前项目数据、程序判定出的异常、诊断结论（推测）与历史案例参考，"
        "请给出运营优化建议。\n\n"
        f"【提示词版本】{PROMPT_VERSION}\n\n"
        "【核心指标（程序计算，禁止修改）】\n"
        f"{_dump(metrics)}\n\n"
        f"【异常判定阈值】\n{_dump(thresholds or {})}\n\n"
        "【异常项目汇总】\n"
        f"{_dump(abnormal_projects)}\n\n"
        "【诊断结论（推测，来自诊断 Agent）】\n"
        f"{_dump(diagnosis)}\n\n"
        f"【历史相似案例】{case_note}\n"
        f"{_dump(retrieved_cases)}\n\n"
        + _phase9_sections(trend_analysis, project_comparison)
        + "请只返回 JSON：summary / recommendations。"
    )


def build_report_prompt(
    *,
    raw_data_summary: dict[str, Any],
    metrics: dict[str, Any],
    abnormal_projects: list[dict[str, Any]],
    anomaly_summary: list[dict[str, Any]],
    diagnosis: dict[str, Any],
    retrieved_cases: list[dict[str, Any]],
    recommendations: dict[str, Any],
    thresholds: dict[str, Any],
    workflow_notes: list[str] | None = None,
    trend_analysis: dict[str, Any] | None = None,
    project_comparison: dict[str, Any] | None = None,
) -> str:
    """构造报告 Agent 的用户提示词（汇总全部工作流产出）。"""

    notes = workflow_notes or []
    notes_text = "\n".join(f"- {note}" for note in notes) if notes else "- 无"
    return (
        "以下是本次多 Agent 工作流的全部产出，请汇总成一份完整报告。\n\n"
        f"【提示词版本】{PROMPT_VERSION}\n\n"
        "【数据源与清洗概况】\n"
        f"{_dump(raw_data_summary)}\n\n"
        "【核心指标（程序计算，禁止修改）】\n"
        f"{_dump(metrics)}\n\n"
        f"【异常判定阈值】\n{_dump(thresholds)}\n\n"
        "【异常类型汇总】\n"
        f"{_dump(anomaly_summary)}\n\n"
        "【异常项目汇总】\n"
        f"{_dump(abnormal_projects)}\n\n"
        "【诊断结论（推测）】\n"
        f"{_dump(diagnosis)}\n\n"
        "【历史相似案例（模拟案例，仅供参考）】\n"
        f"{_dump(retrieved_cases)}\n\n"
        "【优化建议】\n"
        f"{_dump(recommendations)}\n\n"
        "【工作流执行说明】\n"
        f"{notes_text}\n\n"
        + _phase9_sections(trend_analysis, project_comparison)
        + "请按系统提示中的小节结构直接输出 Markdown 报告。"
    )

