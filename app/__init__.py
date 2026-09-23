"""RobotOps AI —— Phase 4 多 Agent 工作流（LangGraph）。

目录职责：
- ``app/agents/``：5 个单一职责 Agent（数据分析 / 诊断 / RAG / 建议 / 报告）；
- ``app/graph/``：LangGraph 工作流（统一 State、路由分支、日志、提示词）。

设计原则：**本层只做编排与提示词，不重复实现业务逻辑**。
指标计算、异常判定、清洗、RAG 检索、DeepSeek 调用仍然复用
Phase 1（``robotops``）/ Phase 2（``robotops.llm``）/ Phase 3（``robotops.rag``）的既有代码。

本阶段不包含 Streamlit 与运营问题闭环。
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.4.0"

