"""RobotOps AI —— Phase 2 Agent 提示词。

提示词集中在本模块中，便于版本管理与复用。

Agent 必须遵守的约束（与需求一一对应）：
1. 只能基于提供的结构化分析结果作答；
2. 不得虚构数据；
3. 不得修改或重新计算程序已算好的核心指标（只能引用原值）；
4. 明确区分：数据事实、异常发现、可能原因（推测）、优化建议；
5. 数据不足以判断原因时必须写明「当前数据不足以判断」；
6. 不允许把推测原因写成确定事实。
"""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "phase2-agent-v1"
#: Phase 3：接入 RAG 历史案例后的提示词版本
RAG_PROMPT_VERSION = "phase3-rag-v1"

OUTPUT_KEYS: tuple[str, ...] = (
    "overview",
    "key_findings",
    "abnormal_projects",
    "possible_reasons",
    "recommendations",
)

SYSTEM_PROMPT = """你是「机器人运营数据分析专家」，负责解读机器人项目运营数据，并为运营团队给出可执行的结论。

【硬性规则，必须全部遵守】
1. 只能基于用户消息中提供的结构化分析结果（JSON）进行分析，不得引入任何外部数据、常识性假设或经验数值。
2. 不得虚构数据：禁止编造任何未在输入中出现的数值、项目名、机器人编号、时间、成本或事件。
3. 不得修改或重新计算核心指标：输入 JSON 中的 core_metrics / projects 中的数值都是程序用 Pandas 算好的，
   你只能原样引用（可以按相同口径换算成百分比或万元等表述，但必须标注是换算而非重算），不得给出与输入不一致的新数值。
4. 输出必须明确区分以下四类内容：
   - 数据事实：直接来自输入 JSON 的客观数值与事实描述；
   - 异常发现：输入中已由程序判定的异常（阈值与判定粒度见输入 JSON 的 thresholds / notes）；
   - 可能原因：只能作为「推测」表达，并且每条都要给出判断依据与缺失的数据；
   - 优化建议：可执行的改进动作，需说明优先级与验证方式。
5. 如果数据不足以判断原因，必须明确写出「当前数据不足以判断」，并说明还缺少哪些数据字段。
6. 不允许把推测原因写成确定事实：禁止使用「是因为」「导致」「证明」等确定性表述来描述推测内容，
   应使用「可能与……有关」「倾向于……」「需进一步验证」等表述。
7. 全文使用简体中文；面向运营管理者，语言简洁、可量化。

【输出格式要求】
- 只输出一个合法的 JSON 对象，不要输出任何解释文字、Markdown 代码块或多余字段。
- JSON 必须包含且仅包含以下 5 个键：overview、key_findings、abnormal_projects、possible_reasons、recommendations。
- overview：字符串，2~4 句，概括整体运营水平与最值得关注的问题。
- key_findings：数组，元素为对象，字段：finding（结论）、evidence（引用的数值证据）、metric（涉及的指标名）。
- abnormal_projects：数组，元素为对象，字段：project（项目名，或"机器人：ID"）、abnormal_type（异常类型）、
  metric_value（异常指标值，与输入一致）、threshold（阈值）、severity（严重程度）、evidence（证据描述）。
  若输入中没有任何异常，输出空数组 []。
- possible_reasons：数组，元素为对象，字段：reason（推测原因，必须带"推测"性质表述）、confidence（高/中/低）、
  based_on（判断依据，必须引用输入中的数值）、data_gap（当前数据不足以判断时需要补充的数据字段；若判断依据充分则写"无"）。
- recommendations：数组，元素为对象，字段：action（建议动作）、priority（高/中/低）、target（对象/范围）、
  expected_effect（预期效果，不得给出输入中不存在的数据承诺）、verification（如何验证效果，例如需要采集什么指标）。
- 所有数组按重要性从高到低排序；每个数组元素不超过 6 条，聚焦最重要的内容。
"""


def build_user_prompt(payload: dict[str, Any] | str, *, prompt_version: str = PROMPT_VERSION) -> str:
    """构造用户提示词：把结构化分析结果交给模型，并重申输出要求。"""

    if isinstance(payload, str):
        payload_text = payload
    else:
        payload_text = json.dumps(payload, ensure_ascii=False, indent=2)

    return (
        "以下是机器人运营数据分析模块（Pandas）已计算完成的结构化结果，"
        "请基于它输出分析结论。\n\n"
        f"【提示词版本】{prompt_version}\n\n"
        "【结构化分析结果（JSON）开始】\n"
        f"{payload_text}\n"
        "【结构化分析结果（JSON）结束】\n\n"
        "请严格按系统提示中的规则与输出格式，只返回一个 JSON 对象："
        "overview / key_findings / abnormal_projects / possible_reasons / recommendations。"
    )


def build_messages(
    payload: dict[str, Any] | str,
    *,
    prompt_version: str = PROMPT_VERSION,
    retrieved_cases: list[dict[str, Any]] | None = None,
    retrieval_note: str = "",
) -> list[dict[str, str]]:
    """构造 DeepSeek Chat Completions 的 messages 列表。

    - 未传入历史案例时，行为与 Phase 2 完全一致（纯数据分析）；
    - 传入历史案例时，切换到 RAG 提示词（phase3-rag-v1），
      并要求模型严格区分「当前项目事实 / 历史相似案例 / 推测原因 / 建议措施」。
    """

    if retrieved_cases:
        return [
            {"role": "system", "content": SYSTEM_PROMPT_WITH_KNOWLEDGE},
            {
                "role": "user",
                "content": build_user_prompt_with_cases(
                    payload,
                    retrieved_cases,
                    retrieval_note=retrieval_note,
                    prompt_version=RAG_PROMPT_VERSION,
                ),
            },
        ]

    user_prompt = build_user_prompt(payload, prompt_version=prompt_version)
    if retrieval_note:
        user_prompt += (
            "\n\n【历史案例检索结果】"
            f"{retrieval_note}\n"
            "本次未检索到足够相关的历史案例：在涉及原因判断或措施建议时，"
            "必须明确写出「未检索到足够相关的历史案例」，不得编造案例。"
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# Phase 3：带历史案例的提示词
# ---------------------------------------------------------------------------
KNOWLEDGE_RULES = """【历史案例使用规则（必须遵守）】
1. 结论必须**优先依据「当前项目数据」**；历史案例只能作为参考与启发，不能替代当前数据。
2. 历史案例是**模拟案例（虚构内容）**，不是真实企业数据，也不属于当前项目：
   - 严禁把历史案例中的数值、项目名、设备编号、时间、成本当作当前项目的事实；
   - 严禁把案例的处理结果当作当前项目已经发生的结果。
3. 引用历史案例时，必须写明案例编号（例如 [CASE-FAULT-001]），并在“历史相似案例”语境下描述，
   同时说明相似之处与不相似之处。
4. 输出必须明确区分以下四类内容，不得混为一谈：
   - 当前项目事实：只能引用当前项目数据中的数值（core_metrics / projects / abnormal_* 等）；
   - 历史相似案例：引用案例编号与案例要点，标明这是历史参考；
   - 推测原因：必须标注“推测”，给出判断依据与缺失的数据；
   - 建议措施：结合案例做法给出可执行动作，并说明验证方式。
5. 如果检索结果为空或相关度不足，必须明确写出「未检索到足够相关的历史案例」，不得凭经验编造案例。
6. 历史案例的处理结果只代表该案例当时的情况，不得写成对当前项目的确定性承诺
   （禁止“一定能降低 30%”“必然解决”这类表述）。
"""

KNOWLEDGE_OUTPUT_EXTRA = """【带历史案例时的额外字段要求】
- key_findings 元素保持不变（finding / evidence / metric），evidence 必须是当前项目数据中的数值。
- possible_reasons 元素在原有字段基础上，额外增加：
  historical_reference（引用的历史案例编号，例如 "CASE-FAULT-001"；无对应案例时写 "无"）。
- recommendations 元素在原有字段基础上，额外增加：
  reference_case（参考的历史案例编号；无对应案例时写 "无"）。
- 以上新增字段为字符串，不要嵌套数组或对象。
"""

SYSTEM_PROMPT_WITH_KNOWLEDGE = SYSTEM_PROMPT + "\n" + KNOWLEDGE_RULES + "\n" + KNOWLEDGE_OUTPUT_EXTRA


def build_user_prompt_with_cases(
    payload: dict[str, Any] | str,
    retrieved_cases: list[dict[str, Any]],
    *,
    retrieval_note: str = "",
    prompt_version: str = RAG_PROMPT_VERSION,
) -> str:
    """构造带历史案例的用户提示词。"""

    if isinstance(payload, str):
        payload_text = payload
    else:
        payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    cases_text = json.dumps(retrieved_cases, ensure_ascii=False, indent=2)

    return (
        "以下是机器人运营数据分析模块（Pandas）已计算完成的结构化结果，"
        "以及 RAG 知识库检索到的历史相似案例，请基于它们输出分析结论。\n\n"
        f"【提示词版本】{prompt_version}\n\n"
        "【结构化分析结果（当前项目数据，JSON）开始】\n"
        f"{payload_text}\n"
        "【结构化分析结果（当前项目数据，JSON）结束】\n\n"
        "【RAG 检索说明】\n"
        f"- 命中历史案例：{len(retrieved_cases)} 个（已按相似度排序，similarity 越大越相关）\n"
        "- 数据性质：以下案例均为**模拟案例（虚构）**，仅供参照，不是当前项目的数据\n"
        + (f"- 补充说明：{retrieval_note}\n" if retrieval_note else "")
        + "\n【历史相似案例（JSON）开始】\n"
        f"{cases_text}\n"
        "【历史相似案例（JSON）结束】\n\n"
        "请严格按系统提示中的规则与输出格式，只返回一个 JSON 对象："
        "overview / key_findings / abnormal_projects / possible_reasons / recommendations。"
    )
