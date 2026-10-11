"""RobotOps AI —— Phase 9 项目横向对比。

把多个项目放在同一张表里比较 7 项指标（口径与趋势分析完全一致，复用 robotops.trends）：

============ ========================== ==============
指标          单位                       优劣方向
============ ========================== ==============
设备运行率    %                          越高越好
设备利用率    %                          越高越好
故障率        %                          越低越好
用户满意度    分                         越高越好
运营成本      元                         越低越好
节降率        %                          越高越好
MTTR         小时/次                    越低越好
============ ========================== ==============

输出包含：
- 项目 × 指标 明细表；
- 每个项目的指标排名（1 = 最优）、与整体平均的差异；
- 结构化字典（供 Agent / 接口使用）与事实型要点（不含解读）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import config
from .trends import TREND_METRICS, TrendMetricDefinition, metric_value

#: 指标优劣方向
BETTER_HIGHER = "higher"
BETTER_LOWER = "lower"

#: 对比指标（沿用趋势指标口径；成本用总运营成本）
COMPARISON_METRICS: tuple[TrendMetricDefinition, ...] = TREND_METRICS

METRIC_DIRECTIONS: dict[str, str] = {
    "uptime_rate": BETTER_HIGHER,
    "utilization_rate": BETTER_HIGHER,
    "fault_rate": BETTER_LOWER,
    "satisfaction": BETTER_HIGHER,
    "total_cost": BETTER_LOWER,
    "saving_rate": BETTER_HIGHER,
    "mttr": BETTER_LOWER,
}

DIRECTION_LABELS: dict[str, str] = {
    BETTER_HIGHER: "越高越好",
    BETTER_LOWER: "越低越好",
}


@dataclass
class ProjectComparison:
    """项目横向对比结果。"""

    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    overall: dict[str, float] = field(default_factory=dict)
    metrics: tuple[TrendMetricDefinition, ...] = COMPARISON_METRICS
    warnings: list[str] = field(default_factory=list)

    # -- 基本信息 ---------------------------------------------------------
    @property
    def project_count(self) -> int:
        return int(len(self.frame))

    @property
    def available(self) -> bool:
        return self.project_count > 0

    @property
    def metric_names(self) -> list[str]:
        return [item.name for item in self.metrics]

    def metric(self, key: str) -> TrendMetricDefinition | None:
        for item in self.metrics:
            if item.key == key:
                return item
        return None

    def metric_column(self, key: str) -> str:
        item = self.metric(key)
        return item.name if item else key

    # -- 对比结果 ---------------------------------------------------------
    def value_of(self, project: str, key: str) -> float | None:
        column = self.metric_column(key)
        if self.frame.empty or column not in self.frame.columns:
            return None
        subset = self.frame.loc[self.frame["项目名称"] == project]
        if subset.empty:
            return None
        value = subset[column].iloc[0]
        return None if pd.isna(value) else float(value)

    def ranking(self, key: str) -> pd.DataFrame:
        """按指标排名（1 = 最优，方向由 METRIC_DIRECTIONS 决定）。"""

        item = self.metric(key)
        column = self.metric_column(key)
        columns = ["项目名称", column, "排名", "与整体平均差异", "相对整体(%)"]
        if item is None or self.frame.empty or column not in self.frame.columns:
            return pd.DataFrame(columns=columns)

        frame = self.frame[["项目名称", column]].copy()
        frame = frame.loc[frame[column].notna()]
        if frame.empty:
            return pd.DataFrame(columns=columns)

        ascending = METRIC_DIRECTIONS.get(key, BETTER_HIGHER) == BETTER_LOWER
        frame = frame.sort_values(column, ascending=ascending, kind="stable").reset_index(drop=True)
        overall_value = self.overall.get(key)
        frame["排名"] = range(1, len(frame) + 1)
        if overall_value is not None and not (isinstance(overall_value, float) and math.isnan(overall_value)):
            frame["与整体平均差异"] = frame[column].map(lambda value: round(float(value) - overall_value, 2))
            frame["相对整体(%)"] = frame[column].map(
                lambda value: (
                    None if not overall_value
                    else round((float(value) - overall_value) / abs(overall_value) * 100.0, 2)
                )
            )
        else:
            frame["与整体平均差异"] = None
            frame["相对整体(%)"] = None
        return frame[columns]

    def best(self, key: str) -> dict[str, Any] | None:
        ranking = self.ranking(key)
        if ranking.empty:
            return None
        row = ranking.iloc[0]
        return {
            "metric": self.metric_column(key),
            "project": str(row["项目名称"]),
            "value": float(row[self.metric_column(key)]),
        }

    def worst(self, key: str) -> dict[str, Any] | None:
        ranking = self.ranking(key)
        if ranking.empty:
            return None
        row = ranking.iloc[-1]
        return {
            "metric": self.metric_column(key),
            "project": str(row["项目名称"]),
            "value": float(row[self.metric_column(key)]),
        }

    # -- 输出 -------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()

    def chart_frame(self, keys: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
        """对比图数据：以项目名为索引，列为各指标。"""

        selected = list(keys) if keys else [item.key for item in self.metrics]
        columns = [self.metric_column(key) for key in selected]
        columns = [name for name in columns if name in self.frame.columns]
        if self.frame.empty or not columns:
            return pd.DataFrame()
        return self.frame.set_index("项目名称")[columns]

    def highlight_lines(self, *, limit: int = 7) -> list[str]:
        """事实型要点（不含推测，解读由大模型完成）。"""

        lines: list[str] = []
        if not self.available:
            return lines
        for item in self.metrics:
            best = self.best(item.key)
            worst = self.worst(item.key)
            if best is None or worst is None:
                continue
            if best["project"] == worst["project"]:
                continue
            label = "最优" if METRIC_DIRECTIONS.get(item.key) == BETTER_HIGHER else "最低"
            lines.append(
                item.name + "：" + label + " " + best["project"] + " "
                + _fmt(best["value"]) + item.unit
                + "；最高 " + worst["project"] + " " + _fmt(worst["value"]) + item.unit
            )
        return lines[:limit]

    def as_dict(self) -> dict[str, Any]:
        """结构化结果（可 JSON 序列化）。"""

        projects: list[dict[str, Any]] = []
        for _, row in self.frame.iterrows():
            item: dict[str, Any] = {
                "project": str(row["项目名称"]),
                "robot_count": int(row["机器人数"]) if "机器人数" in self.frame.columns else None,
                "record_count": int(row["记录数"]) if "记录数" in self.frame.columns else None,
            }
            for metric in self.metrics:
                value = row.get(metric.name)
                item[metric.key] = None if pd.isna(value) else round(float(value), 2)
            projects.append(item)

        return {
            "metrics": [
                {
                    "key": item.key,
                    "name": item.name,
                    "unit": item.unit,
                    "direction": DIRECTION_LABELS.get(
                        METRIC_DIRECTIONS.get(item.key, BETTER_HIGHER), ""
                    ),
                }
                for item in self.metrics
            ],
            "overall": {
                key: (None if value is None or (isinstance(value, float) and math.isnan(value))
                      else round(float(value), 2))
                for key, value in self.overall.items()
            },
            "projects": projects,
            "rankings": {
                item.key: self.ranking(item.key).to_dict("records") for item in self.metrics
            },
            "highlights": self.highlight_lines(),
            "warnings": list(self.warnings),
        }

    def summary_dict(self) -> dict[str, Any]:
        """提供给 Agent / 前端的最小结构化摘要。"""

        return {
            "project_count": self.project_count,
            "metrics": [item.name for item in self.metrics],
            "overall": {
                key: (None if value is None or (isinstance(value, float) and math.isnan(value))
                      else round(float(value), 2))
                for key, value in self.overall.items()
            },
            "best": {
                item.key: self.best(item.key) for item in self.metrics if self.best(item.key)
            },
            "highlights": self.highlight_lines(),
            "warnings": list(self.warnings),
        }

    def summary_text(self) -> str:
        lines = ["【项目横向对比（共 " + str(self.project_count) + " 个项目）】"]
        for line in self.highlight_lines(limit=10):
            lines.append("  · " + line)
        for warning in self.warnings:
            lines.append("  ! " + warning)
        return "\n".join(lines)


def compare_projects(data: pd.DataFrame | None) -> ProjectComparison:
    """按项目维度计算 7 项指标并横向对比。"""

    warnings: list[str] = []
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return ProjectComparison(warnings=["数据为空，无法进行项目对比"])
    if config.COL_PROJECT not in data.columns:
        return ProjectComparison(
            warnings=["数据缺少「" + config.COL_PROJECT + "」字段，无法进行项目对比"]
        )

    rows: list[dict[str, Any]] = []
    for project, subset in data.groupby(config.COL_PROJECT, sort=True):
        row: dict[str, Any] = {
            "项目名称": str(project),
            "机器人数": int(subset[config.COL_ROBOT_ID].nunique()) if config.COL_ROBOT_ID in subset.columns else 0,
            "记录数": int(len(subset)),
        }
        for item in COMPARISON_METRICS:
            row[item.name] = metric_value(subset, item)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return ProjectComparison(warnings=["没有可用于对比的项目"])

    columns = ["项目名称", "机器人数", "记录数"] + [item.name for item in COMPARISON_METRICS]
    frame = frame[[name for name in columns if name in frame.columns]]
    frame = frame.sort_values("项目名称", kind="stable").reset_index(drop=True)

    if len(frame) == 1:
        warnings.append("当前只有 1 个项目，横向对比意义有限")

    overall = {item.key: metric_value(data, item) for item in COMPARISON_METRICS}

    return ProjectComparison(
        frame=frame,
        overall=overall,
        metrics=COMPARISON_METRICS,
        warnings=warnings,
    )


def comparison_frame(data: pd.DataFrame | None) -> pd.DataFrame:
    """便捷函数：直接返回项目 × 指标明细表。"""

    return compare_projects(data).to_frame()


def comparison_metric_frame() -> pd.DataFrame:
    """对比指标口径表（用于界面与报告）。"""

    return pd.DataFrame(
        [
            {
                "指标": item.name,
                "单位": item.unit,
                "优劣方向": DIRECTION_LABELS.get(
                    METRIC_DIRECTIONS.get(item.key, BETTER_HIGHER), ""
                ),
                "计算口径": item.formula,
            }
            for item in COMPARISON_METRICS
        ]
    )


def _fmt(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    rounded = round(number, 2)
    return "0" if rounded == 0 else str(rounded)

