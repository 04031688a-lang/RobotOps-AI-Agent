"""RobotOps AI —— Phase 1 分析流水线。

把各模块串成一条可复用的流程：

读取数据 -> 数据清洗 -> 指标计算 -> 异常识别 -> 结果输出

命令行入口 ``run_analysis.py`` 与本模块的 ``run_analysis()`` 使用同一实现，
方便后续阶段（如接入大模型问答）直接复用 ``AnalysisResult``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import config
from .anomaly import detect_anomalies, summarize_anomalies, summarize_anomalies_by_project
from .data_cleaner import CleaningResult, clean_operation_data
from .data_loader import load_operation_data
from .metrics import calculate_core_metrics, project_level_summary
from .models import MetricsSummary


@dataclass
class AnalysisResult:
    """一次完整分析的全部产物。"""

    data_path: Path
    sheet_name: str | None
    load_summary: dict[str, Any]
    cleaned: CleaningResult
    metrics: MetricsSummary
    project_summary: pd.DataFrame
    anomalies: pd.DataFrame
    anomaly_summary: pd.DataFrame
    anomaly_project_summary: pd.DataFrame
    thresholds: config.AnomalyThresholds = field(default_factory=lambda: config.DEFAULT_THRESHOLDS)
    generated_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    exported_files: list[Path] = field(default_factory=list)

    @property
    def cleaned_data(self) -> pd.DataFrame:
        """清洗后的明细数据。"""

        return self.cleaned.data

    @property
    def anomaly_count(self) -> int:
        return int(len(self.anomalies))

    def metrics_dict(self) -> dict[str, Any]:
        return self.metrics.as_dict()


def run_analysis(
    data_path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    sheet_name: str | int | None = None,
    encoding: str | None = None,
    thresholds: config.AnomalyThresholds | None = None,
    fault_rate_scope: str | None = None,
    export: bool = True,
    write_excel: bool = True,
    verbose: bool = True,
) -> AnalysisResult:
    """执行完整的 Phase 1 分析流程。

    Args:
        data_path: 数据文件路径，默认使用 ``data/raw/robot_operation_demo.xlsx``。
        output_dir: 结果输出目录，默认 ``output/``。
        sheet_name: Excel 工作表名或索引。
        encoding: CSV 编码（默认自动识别）。
        thresholds: 异常识别阈值，默认使用 ``config.DEFAULT_THRESHOLDS``。
        fault_rate_scope: 故障率判定口径，``robot_period``（默认，按机器人累计）
            或 ``record``（逐条记录判定）。
        export: 是否导出结果文件。
        write_excel: 导出时是否生成 Excel 汇总工作簿。
        verbose: 是否在控制台输出进度。

    Returns:
        ``AnalysisResult``：包含清洗结果、核心指标、项目汇总、异常明细等。
    """

    limits = thresholds or config.DEFAULT_THRESHOLDS

    if verbose:
        print("[1/5] 读取数据 ...")
    loaded = load_operation_data(data_path, sheet_name=sheet_name, encoding=encoding)
    if verbose:
        print(f"      数据文件：{loaded.path}")
        print(f"      读取记录：{loaded.row_count} 条 / {loaded.column_count} 列")

    if verbose:
        print("[2/5] 清洗数据 ...")
    cleaned = clean_operation_data(loaded.data)
    if verbose:
        print(
            f"      清洗后记录：{cleaned.stats['清洗后记录数']} 条"
            f"（剔除 {cleaned.stats['剔除记录数']} 条，问题条目 {cleaned.stats['问题条目数']} 项）"
        )

    if verbose:
        print("[3/5] 计算核心指标 ...")
    metrics = calculate_core_metrics(cleaned.data)
    projects = project_level_summary(cleaned.data)

    if verbose:
        print("[4/5] 识别异常 ...")
    anomalies = detect_anomalies(
        cleaned.data, thresholds=limits, fault_rate_scope=fault_rate_scope
    )
    anomaly_summary = summarize_anomalies(anomalies)
    anomaly_project_summary = summarize_anomalies_by_project(anomalies)
    if verbose:
        print(f"      命中异常记录：{len(anomalies)} 条")

    result = AnalysisResult(
        data_path=loaded.path,
        sheet_name=loaded.sheet_name,
        load_summary=loaded.summary(),
        cleaned=cleaned,
        metrics=metrics,
        project_summary=projects,
        anomalies=anomalies,
        anomaly_summary=anomaly_summary,
        anomaly_project_summary=anomaly_project_summary,
        thresholds=limits,
    )

    if export:
        if verbose:
            print("[5/5] 导出结果文件 ...")
        from .report import export_results  # 局部导入，避免模块循环依赖

        result.exported_files = export_results(result, output_dir, write_excel=write_excel)
        if verbose:
            for path in result.exported_files:
                print(f"      已生成：{path}")

    return result
