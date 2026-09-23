"""RobotOps AI —— Phase 4 的 5 个单一职责 Agent。"""

from __future__ import annotations

from .base import BaseAgent, has_llm_failure
from .data_analysis import DataAnalysisAgent
from .diagnosis import DiagnosisAgent, build_offline_diagnosis
from .rag_agent import RagAgent
from .recommendation import RecommendationAgent, build_offline_recommendations
from .report import ReportAgent

__all__ = [
    "BaseAgent",
    "DataAnalysisAgent",
    "DiagnosisAgent",
    "RagAgent",
    "RecommendationAgent",
    "ReportAgent",
    "build_offline_diagnosis",
    "build_offline_recommendations",
    "has_llm_failure",
]

