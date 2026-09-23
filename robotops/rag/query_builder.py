"""RobotOps AI —— Phase 3 异常描述生成（RAG 检索用的查询语句）。

数据流中的「生成异常描述」一步：
把 Phase 1 的指标与异常明细，转换成接近运营人员口吻的自然语言查询，例如：

    「华中-武汉光谷消杀项目 节降率偏低：WH-DIS-02 平均节降率 9.58% 低于阈值 10%，
      命中 18 天；项目平均节降率 10.51%；相关特征：运营成本过高 能耗 成本结构」

这样既保留了量化事实（供模型核对），又包含领域词（提升检索命中率）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .. import config as project_config
from ..pipeline import AnalysisResult

#: 异常类型 -> 检索补充词（补充同类案例中的常见表述，提升召回）
ANOMALY_QUERY_HINTS: dict[str, str] = {
    "故障率偏高": "机器人重复故障 故障率升高 维修次数增加 部件损坏 备件更换 设备维护周期",
    "满意度偏低": "用户满意度下降 投诉 清洁效果不达标 服务响应慢 作业质量",
    "运行率偏低": "运行率下降 欠运行 停机时间 充电策略 排班冲突 设备闲置 任务调度",
    "节降率偏低": "节降率下降 运营成本过高 能耗偏高 成本结构 人力配置",
}

#: 异常类型 -> 优先检索的知识库分类（命中不足时自动退回全库检索）
ANOMALY_CATEGORY_HINTS: dict[str, tuple[str, ...]] = {
    "故障率偏高": ("equipment_fault", "maintenance"),
    "满意度偏低": ("satisfaction", "maintenance"),
    "运行率偏低": ("operation", "maintenance"),
    "节降率偏低": ("operation",),
}

DEFAULT_MAX_QUERIES = 4


@dataclass
class AnomalyQuery:
    """一条检索查询及其来源说明。"""

    label: str
    text: str
    anomaly_type: str = ""
    scope: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    preferred_categories: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "anomaly_type": self.anomaly_type,
            "scope": self.scope,
            "query": self.text,
            "metrics": self.metrics,
        }


def build_anomaly_queries(
    result: AnalysisResult,
    *,
    max_queries: int = DEFAULT_MAX_QUERIES,
) -> list[AnomalyQuery]:
    """把 Phase 1 的异常识别结果转换为检索查询列表。"""

    if not isinstance(result, AnalysisResult):
        raise TypeError("build_anomaly_queries 需要 Phase 1 的 AnalysisResult")

    queries: list[AnomalyQuery] = []
    anomalies = result.anomalies

    if anomalies is not None and not anomalies.empty:
        for anomaly_type, group in anomalies.groupby("异常类型", sort=False):
            queries.append(_query_for_anomaly_type(result, str(anomaly_type), group))

    if not queries:
        queries.append(_query_for_overall(result))

    queries.sort(key=lambda item: item.metrics.get("hits", 0), reverse=True)
    return queries[: max(int(max_queries), 1)]


def _query_for_anomaly_type(result: AnalysisResult, anomaly_type: str, group: Any) -> AnomalyQuery:
    hints = ANOMALY_QUERY_HINTS.get(anomaly_type, "")
    project_counts = group["项目名称"].value_counts()
    top_project = str(project_counts.index[0]) if len(project_counts) else ""
    robot_counts = group["机器人ID"].value_counts()
    top_robot = str(robot_counts.index[0]) if len(robot_counts) else ""

    parts: list[str] = []
    scope = top_project or "全部项目"
    lead = f"{scope} 机器人{anomaly_type}"
    if top_robot:
        lead += f"（重点设备 {top_robot}）"
    parts.append(lead + "：")

    robot_rows = group.loc[group["机器人ID"] == top_robot] if top_robot else group
    if not robot_rows.empty:
        sample = robot_rows.iloc[0]
        parts.append(
            f"实际值 {sample.get('实际值')}，判定条件 {sample.get('判定条件')}，"
            f"严重程度 {sample.get('严重程度')}，该机器人命中 {len(robot_rows)} 次。"
        )

    project_row = _project_row(result, top_project)
    if project_row:
        parts.append(
            f"项目层面：故障率 {project_row.get('故障率(%)')}%、运行率 "
            f"{project_row.get('平均运行率(%)')}%、满意度 {project_row.get('平均满意度(分)')} 分、"
            f"单位成本 {project_row.get('总运营成本(元)')} 元、节降率 "
            f"{project_row.get('平均节降率(%)')}%。"
        )

    parts.append(
        f"全项目共命中 {int(len(group))} 条，涉及 {int(group['项目名称'].nunique())} 个项目、"
        f"{int(group['机器人ID'].nunique())} 台机器人。"
    )
    if hints:
        parts.append(f"相关特征：{hints}")

    return AnomalyQuery(
        label=f"{anomaly_type}（{scope}）",
        text="".join(parts),
        anomaly_type=anomaly_type,
        scope=scope,
        metrics={
            "hits": int(len(group)),
            "projects": int(group["项目名称"].nunique()),
            "robots": int(group["机器人ID"].nunique()),
            "top_project": top_project,
            "top_robot": top_robot,
            "sample_value": float(robot_rows.iloc[0]["实际值"]) if not robot_rows.empty else None,
            "condition": str(robot_rows.iloc[0]["判定条件"]) if not robot_rows.empty else "",
        },
        preferred_categories=ANOMALY_CATEGORY_HINTS.get(anomaly_type, ()),
    )


def _query_for_overall(result: AnalysisResult) -> AnomalyQuery:
    metrics = result.metrics
    thresholds = result.thresholds
    text = (
        "机器人运营数据分析："
        f"项目 {metrics.project_count} 个、机器人 {metrics.robot_count} 台，"
        f"平均运行时长 {_round(metrics.avg_running_hours)} 小时，平均运行率 "
        f"{_round(metrics.avg_uptime_rate)}%，累计故障率 {_round(metrics.fault_rate)}%"
        f"（阈值 {thresholds.fault_rate_upper:g}%），平均满意度 {_round(metrics.avg_satisfaction)} 分"
        f"（阈值 {thresholds.satisfaction_lower:g} 分），平均节降率 "
        f"{_round(metrics.avg_saving_rate)}%（阈值 {thresholds.saving_rate_lower:g}%）。"
        "请检索相关的机器人运维、故障处理、运行率与成本优化历史案例。"
    )
    return AnomalyQuery(
        label="整体运营情况",
        text=text,
        anomaly_type="",
        scope="全部项目",
        metrics={"hits": 0},
    )


def _project_row(result: AnalysisResult, project_name: str) -> dict[str, Any] | None:
    summary = result.project_summary
    if summary is None or summary.empty or not project_name:
        return None
    matched = summary.loc[summary["项目名称"] == project_name]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()


def _round(value: Any, digits: int = 2) -> Any:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value

