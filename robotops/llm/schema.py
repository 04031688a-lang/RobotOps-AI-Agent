"""RobotOps AI —— Phase 2 Agent 结构化输出模型。

要求 Agent 输出结构化 JSON；本模块负责：
1. 解析模型返回的 JSON（兼容 ```json 代码块与前后多余文字）；
2. 校验 5 个必需字段并给出校验提示（不因个别字段缺失而崩溃）；
3. 统一转换为 dict / Markdown，便于命令行输出与文件落盘。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .errors import LLMEmptyResponseError, LLMResponseFormatError
from .prompts import OUTPUT_KEYS, PROMPT_VERSION

#: 结构化字段的中文标签（用于 Markdown 与终端展示）
FIELD_LABELS: dict[str, str] = {
    "finding": "结论",
    "evidence": "证据",
    "metric": "指标",
    "project": "对象",
    "abnormal_type": "异常类型",
    "metric_value": "指标值",
    "threshold": "阈值",
    "severity": "严重程度",
    "reason": "推测原因",
    "confidence": "置信度",
    "based_on": "判断依据",
    "data_gap": "缺失数据",
    "action": "建议动作",
    "priority": "优先级",
    "target": "对象/范围",
    "expected_effect": "预期效果",
    "verification": "验证方式",
    "historical_reference": "历史案例参考",
    "reference_case": "参考案例",
    "case_summary": "案例要点",
    "case_id": "案例编号",
    "similarity": "相关度",
    "matched_sections": "命中片段",
    "current_project_fact": "当前项目事实",
}


@dataclass
class AIAnalysis:
    """Agent 结构化分析结果。"""

    overview: str = ""
    key_findings: list[Any] = field(default_factory=list)
    abnormal_projects: list[Any] = field(default_factory=list)
    possible_reasons: list[Any] = field(default_factory=list)
    recommendations: list[Any] = field(default_factory=list)
    model: str = ""
    prompt_version: str = PROMPT_VERSION
    generated_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    usage: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    raw_text: str = ""
    payload_meta: dict[str, Any] = field(default_factory=dict)
    retrieved_cases: list[dict[str, Any]] = field(default_factory=list)

    # -- 构造 -------------------------------------------------------------
    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        model: str = "",
        usage: dict[str, Any] | None = None,
        raw_text: str = "",
        payload_meta: dict[str, Any] | None = None,
        prompt_version: str = PROMPT_VERSION,
        retrieved_cases: list[dict[str, Any]] | None = None,
    ) -> "AIAnalysis":
        if not isinstance(payload, dict):
            raise LLMResponseFormatError(
                f"Agent 返回的 JSON 顶层应为对象，实际为 {type(payload).__name__}"
            )

        warnings: list[str] = []
        missing = [key for key in OUTPUT_KEYS if key not in payload]
        if missing:
            warnings.append(f"模型返回缺少字段：{'、'.join(missing)}（已按空值处理）")

        overview = payload.get("overview") or ""
        if not isinstance(overview, str):
            overview = json.dumps(overview, ensure_ascii=False)
            warnings.append("overview 不是字符串，已转换为文本")
        if not overview.strip():
            warnings.append("overview 为空")

        sections: dict[str, list[Any]] = {}
        for key in OUTPUT_KEYS[1:]:
            value = payload.get(key)
            normalized = _normalize_items(value)
            if value is not None and not isinstance(value, (list, tuple)):
                warnings.append(f"{key} 不是数组，已包装为单元素数组")
            if not normalized:
                warnings.append(f"{key} 为空数组")
            sections[key] = normalized

        extra_keys = [key for key in payload if key not in OUTPUT_KEYS]
        if extra_keys:
            warnings.append(f"模型返回了额外字段：{'、'.join(extra_keys)}（已忽略）")

        return cls(
            overview=overview.strip(),
            key_findings=sections["key_findings"],
            abnormal_projects=sections["abnormal_projects"],
            possible_reasons=sections["possible_reasons"],
            recommendations=sections["recommendations"],
            model=model,
            prompt_version=prompt_version,
            usage=usage or {},
            warnings=warnings,
            raw_text=raw_text,
            payload_meta=payload_meta or {},
            retrieved_cases=list(retrieved_cases or []),
        )

    # -- 输出 -------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return not (
            self.overview
            or self.key_findings
            or self.abnormal_projects
            or self.possible_reasons
            or self.recommendations
        )

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "overview": self.overview,
            "key_findings": self.key_findings,
            "abnormal_projects": self.abnormal_projects,
            "possible_reasons": self.possible_reasons,
            "recommendations": self.recommendations,
            "meta": {
                "model": self.model,
                "prompt_version": self.prompt_version,
                "generated_at": self.generated_at,
                "usage": self.usage,
                "payload_meta": self.payload_meta,
                "retrieved_cases": self.retrieved_cases,
                "warnings": self.warnings,
            },
        }
        if include_raw:
            data["meta"]["raw_text"] = self.raw_text
        return data

    def to_markdown(self) -> str:
        lines: list[str] = ["# RobotOps AI —— DeepSeek 运营数据分析（Phase 2）", ""]
        lines.append(f"- 生成时间：{self.generated_at}")
        if self.model:
            lines.append(f"- 模型：`{self.model}`")
        lines.append(f"- 提示词版本：`{self.prompt_version}`")
        if self.usage:
            lines.append(f"- Token 用量：{_format_usage(self.usage)}")
        if self.payload_meta:
            source = self.payload_meta.get("data_source", "")
            window = self.payload_meta.get("analysis_window", "")
            if source:
                lines.append(f"- 数据源：{source}")
            if window:
                lines.append(f"- 分析窗口：{window}")
        lines.append("")

        lines.append("## 运营概览")
        lines.append("")
        lines.append(self.overview or "_（模型未返回概览内容）_")
        lines.append("")

        lines.extend(_render_section("关键发现", self.key_findings))
        lines.extend(_render_section("异常发现", self.abnormal_projects))
        lines.extend(_render_section("可能原因（推测）", self.possible_reasons))
        lines.extend(_render_section("优化建议", self.recommendations))

        if self.retrieved_cases:
            lines.append("## 历史相似案例（RAG 检索）")
            lines.append("")
            lines.append(
                "> 以下案例为**模拟案例（虚构）**，由本地知识库检索得到，仅作参考，"
                "不代表当前项目数据。"
            )
            lines.append("")
            for index, case in enumerate(self.retrieved_cases, start=1):
                lines.append(
                    f"**{index}. [{case.get('case_id', '-')}] {case.get('title', '')}**"
                )
                lines.append("")
                lines.append(f"- 案例类型：{case.get('case_type', '-')}")
                lines.append(f"- 相关度：{case.get('similarity', '-')}")
                if case.get("matched_sections"):
                    lines.append(f"- 命中片段：{'、'.join(case['matched_sections'])}")
                if case.get("matched_queries"):
                    lines.append(f"- 命中查询：{'；'.join(case['matched_queries'])}")
                if case.get("keywords"):
                    lines.append(f"- 关键词：{'、'.join(case['keywords'])}")
                lines.append(f"- 数据性质：{case.get('data_nature', '模拟案例（虚构）')}")
                lines.append("")
            lines.append("")

        if self.warnings:
            lines.append("## 输出校验提示")
            lines.append("")
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        lines.append("> 本报告由 DeepSeek 基于 Phase 1 程序计算的结构化指标生成；")
        lines.append("> 核心指标数值均由程序计算，模型仅做解读，未参与计算。")
        lines.append("")
        return "\n".join(lines)

    def to_console_text(self) -> str:
        """生成命令行展示文本（中文分区 + 字段标签）。"""

        width = 78
        lines: list[str] = ["=" * width]
        lines.append("DeepSeek 运营数据分析结果（Phase 2 Agent）")
        lines.append("=" * width)
        if self.model:
            lines.append(f"模型           : {self.model}")
        lines.append(f"提示词版本     : {self.prompt_version}")
        if self.usage:
            lines.append(f"Token 用量     : {_format_usage(self.usage)}")
        if self.payload_meta:
            if self.payload_meta.get("data_source"):
                lines.append(f"数据源         : {self.payload_meta['data_source']}")
            if self.payload_meta.get("analysis_window"):
                lines.append(f"分析窗口       : {self.payload_meta['analysis_window']}")

        lines.append("")
        lines.append("【运营概览】")
        lines.append("-" * width)
        lines.append(self.overview or "（模型未返回概览内容）")

        for title, items in (
            ("关键发现", self.key_findings),
            ("异常发现", self.abnormal_projects),
            ("可能原因（推测）", self.possible_reasons),
            ("优化建议", self.recommendations),
        ):
            lines.append("")
            lines.append(f"【{title}】共 {len(items)} 条")
            lines.append("-" * width)
            if not items:
                lines.append("  （无）")
                continue
            for index, item in enumerate(items, start=1):
                if isinstance(item, dict):
                    lines.append(f"  {index}. {_headline(item)}")
                    headline_key = _headline_key(item)
                    for key, value in item.items():
                        if key == headline_key:
                            continue
                        lines.append(f"     - {FIELD_LABELS.get(key, key)}：{_stringify(value)}")
                else:
                    lines.append(f"  {index}. {_stringify(item)}")

        if self.retrieved_cases:
            lines.append("")
            lines.append(
                f"【历史相似案例（RAG 检索）】共 {len(self.retrieved_cases)} 个"
                "（模拟案例，仅供参考）"
            )
            lines.append("-" * width)
            for index, case in enumerate(self.retrieved_cases, start=1):
                lines.append(
                    f"  {index}. [{case.get('case_id', '-')}] {case.get('title', '')}"
                    f"｜类型：{case.get('case_type', '-')}"
                    f"｜相关度：{case.get('similarity', '-')}"
                )
                if case.get("matched_sections"):
                    lines.append(f"     - 命中片段：{'、'.join(case['matched_sections'])}")
                if case.get("matched_queries"):
                    lines.append(f"     - 命中查询：{'；'.join(case['matched_queries'])}")

        if self.warnings:
            lines.append("")
            lines.append("【输出校验提示】")
            lines.append("-" * width)
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        lines.append("=" * width)
        return "\n".join(lines)


def parse_ai_analysis(
    text: str,
    *,
    model: str = "",
    usage: dict[str, Any] | None = None,
    payload_meta: dict[str, Any] | None = None,
    prompt_version: str = PROMPT_VERSION,
    retrieved_cases: list[dict[str, Any]] | None = None,
) -> AIAnalysis:
    """把模型返回文本解析为 ``AIAnalysis``。

    Raises:
        LLMEmptyResponseError: 返回内容为空。
        LLMResponseFormatError: 无法解析为 JSON 对象。
    """

    if text is None or not str(text).strip():
        raise LLMEmptyResponseError("DeepSeek 返回内容为空")

    raw = str(text).strip()
    payload = _try_load_json(raw)
    if payload is None:
        payload = _try_extract_json_object(raw)
    if payload is None:
        raise LLMResponseFormatError(
            "DeepSeek 返回内容不是合法 JSON，无法解析为结构化结果",
            raw=raw,
        )

    return AIAnalysis.from_payload(
        payload,
        model=model,
        usage=usage,
        raw_text=raw,
        payload_meta=payload_meta,
        prompt_version=prompt_version,
        retrieved_cases=retrieved_cases,
    )


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _try_load_json(text: str) -> dict[str, Any] | None:
    try:
        candidate = json.loads(text)
    except json.JSONDecodeError:
        return None
    return candidate if isinstance(candidate, dict) else None


def _try_extract_json_object(text: str) -> dict[str, Any] | None:
    """兼容 ```json 代码块与前后带说明文字的情况。"""

    stripped = text.strip()
    if stripped.startswith("```"):
        body = stripped.split("```")
        for block in body:
            candidate = block.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                parsed = _try_load_json(candidate)
                if parsed is not None:
                    return parsed

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return _try_load_json(stripped[start : end + 1])
    return None


def _normalize_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [item for item in value if item not in (None, "", [], {})]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return [value]


def _format_usage(usage: dict[str, Any]) -> str:
    parts = []
    for key, label in (
        ("prompt_tokens", "输入"),
        ("completion_tokens", "输出"),
        ("total_tokens", "合计"),
    ):
        if key in usage and usage[key] is not None:
            parts.append(f"{label} {usage[key]}")
    return " / ".join(parts) if parts else json.dumps(usage, ensure_ascii=False)


def _render_section(title: str, items: list[Any]) -> list[str]:
    lines = [f"## {title}", ""]
    if not items:
        lines.append("_（无）_")
        lines.append("")
        return lines

    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            headline = _headline(item)
            lines.append(f"**{index}. {headline}**")
            lines.append("")
            for key, value in item.items():
                if key == _headline_key(item):
                    continue
                label = FIELD_LABELS.get(key, key)
                lines.append(f"- {label}：{_stringify(value)}")
            lines.append("")
        else:
            lines.append(f"- {_stringify(item)}")
    if not isinstance(items[0], dict):
        lines.append("")
    return lines


def _headline_key(item: dict[str, Any]) -> str | None:
    for key in (
        "finding",
        "project",
        "reason",
        "action",
        "异常类型",
        "结论",
        "建议动作",
        "推测原因",
        "对象",
    ):
        if key in item:
            return key
    return next(iter(item), None)


def _headline(item: dict[str, Any]) -> str:
    key = _headline_key(item)
    if key is None:
        return "（无标题）"
    return _stringify(item.get(key))


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "；".join(_stringify(item) for item in value)
    if isinstance(value, dict):
        return "；".join(f"{FIELD_LABELS.get(k, k)}={_stringify(v)}" for k, v in value.items())
    return str(value)
