"""Phase 4 —— Data Analysis Agent（不调用大模型）。

职责（单一）：
- 调用 Phase 1 的 Pandas 分析流程（读取 → 清洗 → 指标 → 异常）；
- 汇总关键指标、清洗概况与异常结果；
- 输出结构化分析结果（复用 Phase 2 的载荷格式），供后续 Agent 使用。

**异常判定完全由程序完成**，本 Agent 不做任何主观判断。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robotops.agent.payload import build_analysis_payload
from robotops.metrics import metric_definitions_frame
from robotops.pipeline import run_analysis

from ..graph.state import WorkflowState, thresholds_from_state
from .base import BaseAgent


class DataAnalysisAgent(BaseAgent):
    tag = "Data Analysis Agent"
    stage = "data_analysis"

    def execute(self, state: WorkflowState) -> dict[str, Any]:
        result = run_analysis(
            state.get("data_path"),
            output_dir=state.get("output_dir"),
            thresholds=thresholds_from_state(state),
            export=bool(state.get("export", True)),
            verbose=False,
        )
        payload = build_analysis_payload(result)

        return {
            "analysis_result": result,
            "analysis_payload": payload,
            "raw_data_summary": {
                "数据源": Path(result.data_path).name,
                "工作表": result.sheet_name,
                "分析窗口": payload["meta"]["analysis_window"],
                "原始记录数": result.cleaned.stats.get("原始记录数"),
                "清洗后记录数": result.cleaned.stats.get("清洗后记录数"),
                "剔除记录数": result.cleaned.stats.get("剔除记录数"),
                "清洗问题条目数": result.cleaned.stats.get("问题条目数"),
                "项目数": result.cleaned.stats.get("项目数"),
                "机器人数": result.cleaned.stats.get("机器人数"),
                "阈值": result.thresholds.as_dict(),
                "故障率判定粒度": payload["meta"]["fault_rate_anomaly_scope"],
            },
            "metrics": {
                "values": result.metrics.as_dict(),
                "units": _metric_units(),
            },
            "abnormal_projects": payload["abnormal_projects"],
            "anomaly_summary": payload["anomaly_summary"],
            "abnormal_robots": payload["abnormal_robots"],
            "data_quality": payload["data_quality"],
            "anomaly_count": result.anomaly_count,
            "has_anomalies": result.anomaly_count > 0,
        }

    def summarize(self, update: dict[str, Any], state: WorkflowState) -> str:
        metrics = (update.get("metrics") or {}).get("values", {})
        return (
            f"项目 {metrics.get('项目数量')} 个 / 机器人 {metrics.get('机器人数量')} 台 / "
            f"异常 {update.get('anomaly_count', 0)} 条"
        )


def _metric_units() -> dict[str, str]:
    frame = metric_definitions_frame()
    return {str(row["指标"]): str(row["单位"]) for _, row in frame.iterrows()}

