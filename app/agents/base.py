"""RobotOps AI —— Phase 4 Agent 基类。

统一负责：
- ``[Agent 名称] started / completed / failed`` 日志；
- 异常捕获与结构化错误记录（不把 traceback 当作唯一提示）；
- 需要 LLM 的 Agent 在前序 LLM 失败时自动跳过（降级，避免重复调用与重复报错）；
- 复用 Phase 2 的 ``DeepSeekClient``（超时/重试/错误映射均已实现）。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from robotops.exceptions import RobotOpsError
from robotops.llm.client import ChatResult, DeepSeekClient
from robotops.llm.errors import LLMError, LLMResponseFormatError

from ..graph.logging_utils import WorkflowLogger
from ..graph.state import WorkflowState


class BaseAgent(ABC):
    """所有 Agent 的基类：模板方法负责日志、异常与跳过逻辑。"""

    #: 日志标签（与需求中的示例保持一致，例如 "Data Analysis Agent"）
    tag: str = "Agent"
    #: 阶段名（用于 state.failed_stages / skipped_stages）
    stage: str = "agent"
    #: 是否依赖大模型（依赖则在前序 LLM 失败时跳过）
    requires_llm: bool = False

    def __init__(
        self,
        *,
        logger: WorkflowLogger,
        client: DeepSeekClient | None = None,
    ) -> None:
        self.logger = logger
        self.client = client

    # -- 模板方法 ---------------------------------------------------------
    def run(self, state: WorkflowState) -> dict[str, Any]:
        """执行 Agent：记录日志、捕获异常、返回状态更新。"""

        if self.requires_llm and has_llm_failure(state):
            reason = "前序大模型调用已失败，跳过该阶段以避免重复报错"
            self.logger.agent_skipped(self.tag, reason)
            return {
                "skipped_stages": [*state.get("skipped_stages", []), self.stage],
                "steps": [
                    *_steps_of(state),
                    self._step(status="skipped", detail=reason),
                ],
            }

        self.logger.agent_started(self.tag)
        started = datetime.now()
        try:
            update = self.execute(state) or {}
        except LLMError as error:
            return self._handle_error(state, error, started, kind="LLM 调用失败")
        except RobotOpsError as error:
            return self._handle_error(state, error, started, kind="业务处理失败")
        except Exception as error:  # noqa: BLE001 - 兜底：任何异常都要转成明确提示
            return self._handle_error(state, error, started, kind="未预期错误")

        detail = self.summarize(update, state)
        self.logger.agent_completed(self.tag, detail=detail)
        return {
            **update,
            "steps": [*_steps_of(state), self._step(status="completed", detail=detail, started=started)],
        }

    @abstractmethod
    def execute(self, state: WorkflowState) -> dict[str, Any]:
        """Agent 的具体实现（职责单一，只更新自己负责的字段）。"""

    def summarize(self, update: dict[str, Any], state: WorkflowState) -> str:
        """返回一句完成说明（可被子类覆盖）。"""

        return ""

    # -- 错误处理 ---------------------------------------------------------
    def _handle_error(
        self,
        state: WorkflowState,
        error: Exception,
        started: datetime,
        *,
        kind: str,
    ) -> dict[str, Any]:
        message = self._message_of(error)
        self.logger.agent_failed(self.tag, f"{kind}：{message}")
        record = {
            "stage": self.stage,
            "agent": self.tag,
            "kind": kind,
            "error_type": type(error).__name__,
            "message": message,
            "hint": getattr(error, "hint", "") or "",
            "is_llm_error": isinstance(error, LLMError),
            "is_rag_error": _is_rag_error(error),
            "occurred_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return {
            "errors": [*state.get("errors", []), record],
            "failed_stages": [*state.get("failed_stages", []), self.stage],
            "steps": [
                *_steps_of(state),
                self._step(status="failed", detail=f"{kind}：{message}", started=started),
            ],
        }

    @staticmethod
    def _message_of(error: Exception) -> str:
        # 只取错误本身，排查建议（hint）单独记录，避免在日志与报告中重复出现
        return str(error).strip().replace("\n", " ") or type(error).__name__

    def _step(
        self,
        *,
        status: str,
        detail: str = "",
        started: datetime | None = None,
    ) -> dict[str, Any]:
        now = datetime.now()
        return {
            "stage": self.stage,
            "agent": self.tag,
            "status": status,
            "detail": detail,
            "started_at": (started or now).strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round((now - (started or now)).total_seconds(), 3),
        }

    # -- LLM 便捷方法 -----------------------------------------------------
    def chat_json(self, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], ChatResult]:
        """调用大模型并解析 JSON 结果（兼容 ```json 代码块与前后说明文字）。"""

        content, result = self.chat_text(system_prompt, user_prompt)
        payload = extract_json_object(content)
        if payload is None:
            raise LLMResponseFormatError(
                f"{self.tag} 未能从模型返回中解析出 JSON（返回片段：{content[:120]}）",
                raw=content,
            )
        return payload, result

    def chat_text(self, system_prompt: str, user_prompt: str) -> tuple[str, ChatResult]:
        """调用大模型返回文本。"""

        if self.client is None:
            raise LLMError("未配置 DeepSeek 客户端")
        result = self.client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return result.content, result


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def extract_json_object(text: str) -> dict[str, Any] | None:
    """从模型返回中提取 JSON 对象（容忍代码块与前后说明文字）。"""

    if not text or not text.strip():
        return None
    stripped = text.strip()
    candidates: list[str] = [stripped]

    if stripped.startswith("```"):
        for block in stripped.split("```"):
            body = block.strip()
            if body.lower().startswith("json"):
                body = body[4:].strip()
            if body.startswith("{"):
                candidates.append(body)

    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _steps_of(state: WorkflowState) -> list[dict[str, Any]]:
    return list(state.get("steps") or [])


def has_llm_failure(state: WorkflowState) -> bool:
    """判断工作流中是否已经出现大模型调用失败（用于降级跳过）。"""

    return any(record.get("is_llm_error") for record in state.get("errors") or [])


def _is_rag_error(error: Exception) -> bool:
    try:
        from robotops.rag.errors import RagError
    except ImportError:  # pragma: no cover
        return False
    return isinstance(error, RagError)
