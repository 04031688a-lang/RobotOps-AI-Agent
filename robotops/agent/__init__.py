"""RobotOps AI —— Phase 2 机器人运营数据分析 Agent。

对外暴露：
- ``RobotOpsAnalysisAgent``：Agent 主体（载荷构造 → DeepSeek 调用 → 结构化解析）；
- ``AgentRunResult``：一次运行的完整结果；
- ``build_analysis_payload``：Phase 1 结果 → Agent 输入载荷；
- ``run_agent``：便捷入口（Phase 1 分析 + Agent 调用）。
"""

from __future__ import annotations

from .payload import (
    METRIC_UNITS,
    PAYLOAD_VERSION,
    build_analysis_payload,
    payload_to_json,
    payload_to_text,
)
from .robot_ops_agent import (
    AGENT_NAME,
    AI_OUTPUT_BASENAME,
    AI_PAYLOAD_FILE,
    AI_RAG_FILE,
    AgentRunResult,
    RobotOpsAnalysisAgent,
    run_agent,
)

__all__ = [
    "AGENT_NAME",
    "AI_OUTPUT_BASENAME",
    "AI_PAYLOAD_FILE",
    "AI_RAG_FILE",
    "AgentRunResult",
    "METRIC_UNITS",
    "PAYLOAD_VERSION",
    "RobotOpsAnalysisAgent",
    "build_analysis_payload",
    "payload_to_json",
    "payload_to_text",
    "run_agent",
]
