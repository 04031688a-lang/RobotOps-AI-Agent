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
from .logger import log_event, log_exception
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

    log_event("Data analysis started: %s", Path(data_path or config.DEFAULT_DATA_FILE).name)
    if verbose:
        print("[1/5] 读取数据 ...")
    loaded = load_operation_data(data_path, sheet_name=sheet_name, encoding=encoding)
    log_event(
        "Data loaded: %s rows, %s columns from %s",
        loaded.row_count,
        loaded.column_count,
        loaded.path.name,
    )
    if verbose:
        print(f"      数据文件：{loaded.path}")
        print(f"      读取记录：{loaded.row_count} 条 / {loaded.column_count} 列")

    if verbose:
        print("[2/5] 清洗数据 ...")
    cleaned = clean_operation_data(loaded.data)
    log_event(
        "Data cleaning completed: %s records kept, %s removed, %s issues",
        cleaned.stats.get("清洗后记录数"),
        cleaned.stats.get("剔除记录数"),
        cleaned.stats.get("问题条目数"),
    )
    if verbose:
        print(
            f"      清洗后记录：{cleaned.stats['清洗后记录数']} 条"
            f"（剔除 {cleaned.stats['剔除记录数']} 条，问题条目 {cleaned.stats['问题条目数']} 项）"
        )

    if verbose:
        print("[3/5] 计算核心指标 ...")
    metrics = calculate_core_metrics(cleaned.data)
    projects = project_level_summary(cleaned.data)
    log_event(
        "Metrics calculated: %s robots, %s projects loaded",
        metrics.robot_count,
        metrics.project_count,
    )

    if verbose:
        print("[4/5] 识别异常 ...")
    anomalies = detect_anomalies(
        cleaned.data, thresholds=limits, fault_rate_scope=fault_rate_scope
    )
    anomaly_summary = summarize_anomalies(anomalies)
    anomaly_project_summary = summarize_anomalies_by_project(anomalies)
    log_event(
        "Anomaly detection completed: %s anomaly records, %s abnormal projects detected",
        len(anomalies),
        len(anomaly_project_summary),
    )
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
        log_event(
            "Report generated: %s files written to %s",
            len(result.exported_files),
            result.exported_files[0].parent if result.exported_files else "-",
        )
        if verbose:
            for path in result.exported_files:
                print(f"      已生成：{path}")

    return result


# ---------------------------------------------------------------------------
# Phase 8：数据质量检查 -> 通过后进入数据分析
# ---------------------------------------------------------------------------
@dataclass
class QualityGatedAnalysis:
    """Phase 8 分析结果：数据质量报告 +（通过后的）分析结果与运营指标。"""

    quality: Any
    result: AnalysisResult | None = None
    indicators: Any = None
    project_indicators: pd.DataFrame = field(default_factory=pd.DataFrame)
    indicator_anomalies: pd.DataFrame = field(default_factory=pd.DataFrame)
    indicator_anomaly_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: 未进入分析的原因；正常完成分析时为 None
    skipped_reason: str | None = None
    phase8_files: list[Path] = field(default_factory=list)

    @property
    def analyzed(self) -> bool:
        """是否真正进入了数据分析（质量未通过时为 False）。"""

        return self.result is not None

    def as_dict(self) -> dict[str, Any]:
        """结构化输出，便于接口 / 日志 / 脚本使用。"""

        payload: dict[str, Any] = {
            "quality": self.quality.as_dict() if self.quality is not None else {},
            "analyzed": self.analyzed,
            "skipped_reason": self.skipped_reason,
        }
        if self.indicators is not None:
            payload["indicators"] = self.indicators.as_dict()
        if self.project_indicators is not None and not self.project_indicators.empty:
            payload["project_indicators"] = self.project_indicators.to_dict("records")
        if self.indicator_anomalies is not None and not self.indicator_anomalies.empty:
            payload["indicator_anomalies"] = self.indicator_anomalies.to_dict("records")
        return payload


def run_analysis_with_quality(
    data_path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    sheet_name: str | int | None = None,
    encoding: str | None = None,
    thresholds: config.AnomalyThresholds | None = None,
    indicator_thresholds: config.IndicatorThresholds | None = None,
    quality_thresholds: config.QualityThresholds | None = None,
    fault_rate_scope: str | None = None,
    export: bool = True,
    write_excel: bool = True,
    verbose: bool = True,
) -> QualityGatedAnalysis:
    """Phase 8 完整流程：读取 -> 数据质量检查 -> 通过后分析 -> 计算运营指标。

    与 run_analysis 的区别：
    - 读取时不做必需列硬校验（validate=False），缺字段不再抛异常，
      而是返回 status=error 的数据质量报告；
    - 只有质量检查通过（status != error）才会继续执行分析；
    - 通过后额外计算 Phase 8 九项运营指标与阈值预警。

    注意：本函数不改变既有 run_analysis / 多 Agent 工作流的行为。
    """

    from .indicators import (
        calculate_indicators,
        detect_indicator_anomalies,
        project_level_indicators,
        summarize_indicator_anomalies,
    )
    from .quality import check_data_quality

    limits_quality = quality_thresholds or config.DEFAULT_QUALITY_THRESHOLDS
    limits_indicator = indicator_thresholds or config.DEFAULT_INDICATOR_THRESHOLDS
    frame_name = Path(data_path).name if data_path is not None else config.DEFAULT_DATA_FILE.name

    if verbose:
        print("[Phase 8] 数据质量检查 ...")

    try:
        loaded = load_operation_data(
            data_path,
            sheet_name=sheet_name,
            encoding=encoding,
            validate=False,
            # 空数据文件（只有表头）交给数据质量检查统一判定，而不是直接报文件错误
            allow_empty=True,
        )
    except Exception as error:  # noqa: BLE001 - 文件不可读时同样返回报告，不抛异常
        # 用户侧只看结构化报告；详细 traceback 仅写入统一日志
        log_exception(error, context="run_analysis_with_quality")
        report = check_data_quality(
            None, thresholds=limits_quality, frame_name=frame_name
        )
        report.errors.append("文件读取失败：" + str(error).splitlines()[0])
        report.status = config.QUALITY_STATUS_ERROR
        if verbose:
            print("      文件不可读，已终止分析：" + str(error).splitlines()[0])
        return QualityGatedAnalysis(quality=report, skipped_reason="数据文件不可读")

    report = check_data_quality(
        loaded.data, thresholds=limits_quality, frame_name=str(loaded.path.name)
    )
    if verbose:
        print(
            "      质量状态：" + report.status_label
            + "（缺失值 " + str(report.missing_values)
            + " / 重复 " + str(report.duplicate_rows)
            + " / 非法值 " + str(report.invalid_values) + "）"
        )

    if not report.can_analyze:
        if verbose:
            print("      数据质量未通过，已禁止进入分析：" + "；".join(report.errors[:2]))
        outcome = QualityGatedAnalysis(
            quality=report,
            skipped_reason="数据质量检查未通过：" + "；".join(report.errors) if report.errors else "数据质量检查未通过",
        )
        if export:
            outcome.phase8_files = _export_phase8_outputs(outcome, output_dir, verbose=verbose)
        return outcome

    result = run_analysis(
        data_path,
        output_dir=output_dir,
        sheet_name=sheet_name,
        encoding=encoding,
        thresholds=thresholds,
        fault_rate_scope=fault_rate_scope,
        export=export,
        write_excel=write_excel,
        verbose=verbose,
    )

    if verbose:
        print("[Phase 8] 计算运营指标（9 项）...")
    indicators = calculate_indicators(result.cleaned_data)
    projects = project_level_indicators(result.cleaned_data)
    anomalies = detect_indicator_anomalies(result.cleaned_data, thresholds=limits_indicator)
    anomaly_summary = summarize_indicator_anomalies(anomalies)
    if verbose:
        print(
            "      指标级异常/预警：" + str(len(anomalies)) + " 条"
            + "（设备利用率 " + _indicator_text(indicators.utilization_rate) + "%）"
        )

    outcome = QualityGatedAnalysis(
        quality=report,
        result=result,
        indicators=indicators,
        project_indicators=projects,
        indicator_anomalies=anomalies,
        indicator_anomaly_summary=anomaly_summary,
    )
    if export:
        outcome.phase8_files = _export_phase8_outputs(outcome, output_dir, verbose=verbose)
        result.exported_files.extend(outcome.phase8_files)
    return outcome


def _export_phase8_outputs(
    outcome: QualityGatedAnalysis, output_dir: str | Path | None, *, verbose: bool = True
) -> list[Path]:
    """导出 Phase 8 产物：数据质量报告 + 运营指标 + 指标级异常。"""

    import json

    target = Path(output_dir) if output_dir is not None else config.OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    report_path = target / "data_quality_report.json"
    report_path.write_text(
        json.dumps(outcome.quality.as_dict(), ensure_ascii=False, indent=2),
        encoding=config.TEXT_ENCODING,
    )
    written.append(report_path)

    checks_path = target / "data_quality_checks.csv"
    outcome.quality.to_frame().to_csv(checks_path, index=False, encoding=config.CSV_ENCODING)
    written.append(checks_path)

    if outcome.analyzed:
        indicators_path = target / "robot_ops_indicators.csv"
        outcome.project_indicators.to_csv(
            indicators_path, index=False, encoding=config.CSV_ENCODING
        )
        written.append(indicators_path)

        anomalies_path = target / "robot_ops_indicator_anomalies.csv"
        outcome.indicator_anomalies.to_csv(
            anomalies_path, index=False, encoding=config.CSV_ENCODING
        )
        written.append(anomalies_path)

    if verbose:
        for path in written:
            print("      已生成：" + str(path))
    return written


def _indicator_text(value: float, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if number != number:  # NaN
        return "-"
    return str(round(number, digits))
