"""RobotOps AI —— Phase 1 结果输出模块。

提供两类输出：
1. 控制台报告：清洗摘要 + 核心指标 + 项目汇总 + 异常汇总/明细；
2. 结果文件：JSON / CSV / Excel（多工作表）/ Markdown 报告。

CSV 统一使用 ``utf-8-sig`` 编码，保证 Windows 下双击用 Excel 打开不乱码。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from . import config
from .anomaly import describe_rules
from .metrics import metric_definitions_frame

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型标注，避免循环导入
    from .pipeline import AnalysisResult

CONSOLE_ANOMALY_PREVIEW_ROWS: int = 20


# ---------------------------------------------------------------------------
# 控制台报告
# ---------------------------------------------------------------------------
def render_console_report(result: "AnalysisResult") -> str:
    """生成控制台文本报告。"""

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("RobotOps AI   机器人运营数据分析基础模块（Phase 1）")
    lines.append("=" * 78)
    lines.append(f"数据源文件   : {result.data_path}")
    if result.sheet_name:
        lines.append(f"工作表       : {result.sheet_name}")
    lines.append(f"运行时间     : {result.generated_at}")

    lines.append("")
    lines.append("[1] 数据清洗结果")
    lines.append("-" * 78)
    for key, value in result.cleaned.stats.items():
        lines.append(f"  {key:<16}: {value}")

    lines.append("")
    lines.append("[2] 核心指标")
    lines.append("-" * 78)
    lines.append(_frame_to_text(_metrics_display_frame(result)))

    lines.append("")
    lines.append("[3] 项目维度汇总")
    lines.append("-" * 78)
    lines.append(_frame_to_text(result.project_summary))

    lines.append("")
    lines.append("[4] 异常识别")
    lines.append("-" * 78)
    lines.append(
        "  阈值："
        + "；".join(f"{key} {value:g}" for key, value in result.thresholds.as_dict().items())
    )
    if result.anomalies.empty:
        lines.append("  未发现异常记录。")
    else:
        lines.append(f"  命中异常记录 {len(result.anomalies)} 条，按类型汇总：")
        lines.append(_frame_to_text(result.anomaly_summary))
        lines.append("")
        lines.append(f"  异常明细（按严重程度排序，此处展示前 {CONSOLE_ANOMALY_PREVIEW_ROWS} 条）：")
        lines.append(_frame_to_text(result.anomalies.head(CONSOLE_ANOMALY_PREVIEW_ROWS)))

    if result.exported_files:
        lines.append("")
        lines.append("[5] 已导出结果文件")
        lines.append("-" * 78)
        for path in result.exported_files:
            lines.append(f"  - {path}")

    lines.append("=" * 78)
    return "\n".join(lines)


def print_report(result: "AnalysisResult") -> None:
    """打印控制台报告。"""

    print(render_console_report(result))


def _frame_to_text(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame is None or frame.empty:
        return "  （无数据）"
    target = frame if max_rows is None else frame.head(max_rows)
    text = target.to_string(index=False, justify="left")
    return "\n".join(f"  {line}" for line in text.splitlines())


# ---------------------------------------------------------------------------
# 结果文件导出
# ---------------------------------------------------------------------------
def export_results(
    result: "AnalysisResult",
    output_dir: str | Path | None = None,
    *,
    write_excel: bool = True,
) -> list[Path]:
    """导出分析结果文件，返回已生成的文件路径列表。"""

    target_dir = Path(output_dir) if output_dir is not None else config.OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    basename = config.REPORT_BASENAME
    written: list[Path] = []

    metrics_frame = pd.DataFrame(result.metrics.as_rows())
    definitions_frame = metric_definitions_frame()
    rules_frame = describe_rules(result.thresholds)
    stats_frame = pd.DataFrame(
        [{"统计项": key, "数值": value} for key, value in result.cleaned.stats.items()]
    )

    # 1) 核心指标 CSV
    metrics_path = target_dir / f"{basename}_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False, encoding=config.CSV_ENCODING)
    written.append(metrics_path)

    # 2) 项目维度汇总 CSV
    project_path = target_dir / f"{basename}_project_summary.csv"
    result.project_summary.to_csv(project_path, index=False, encoding=config.CSV_ENCODING)
    written.append(project_path)

    # 3) 异常明细 CSV
    anomaly_path = target_dir / f"{basename}_anomalies.csv"
    result.anomalies.to_csv(anomaly_path, index=False, encoding=config.CSV_ENCODING)
    written.append(anomaly_path)

    # 4) 清洗问题 CSV
    issue_path = target_dir / f"{basename}_cleaning_issues.csv"
    result.cleaned.issues.to_csv(issue_path, index=False, encoding=config.CSV_ENCODING)
    written.append(issue_path)

    # 5) 指标 JSON（便于后续阶段或其他程序消费）
    json_path = target_dir / f"{basename}_metrics.json"
    payload: dict[str, Any] = {
        "项目": "RobotOps AI",
        "阶段": "Phase 1 - 机器人运营数据分析基础模块",
        "生成时间": result.generated_at,
        "数据源文件": str(result.data_path),
        "工作表": result.sheet_name,
        "数据清洗统计": result.cleaned.stats,
        "核心指标": result.metrics.as_dict(),
        "异常识别阈值": result.thresholds.as_dict(),
        "异常命中记录数": int(len(result.anomalies)),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding=config.TEXT_ENCODING,
    )
    written.append(json_path)

    # 6) Excel 汇总工作簿
    if write_excel:
        excel_path = target_dir / f"{basename}.xlsx"
        sheets: dict[str, pd.DataFrame] = {
            "核心指标": metrics_frame,
            "指标口径": definitions_frame,
            "项目汇总": result.project_summary,
            "异常规则": rules_frame,
            "异常汇总": result.anomaly_summary,
            "异常项目汇总": result.anomaly_project_summary,
            "异常明细": result.anomalies,
            "清洗统计": stats_frame,
            "清洗问题": result.cleaned.issues,
        }
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for sheet_name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=sheet_name, index=False, na_rep="")
            autofit_columns(writer)
        written.append(excel_path)

    # 7) Markdown 报告
    markdown_path = target_dir / f"{basename}.md"
    markdown_path.write_text(_build_markdown(result, payload), encoding=config.TEXT_ENCODING)
    written.append(markdown_path)

    return written


def autofit_columns(writer: pd.ExcelWriter, *, max_width: int = 42) -> None:
    """按内容长度调整 Excel 列宽，方便直接查看。"""

    for worksheet in writer.book.worksheets:
        for column_cells in worksheet.columns:
            length = 0
            for cell in column_cells:
                value = cell.value
                if value is None:
                    continue
                length = max(length, max(len(line) for line in str(value).splitlines() or [""]))
            column_letter = column_cells[0].column_letter
            worksheet.column_dimensions[column_letter].width = min(max(length + 2, 10), max_width)


def _build_markdown(result: "AnalysisResult", payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# RobotOps AI 运营数据分析报告（Phase 1）")
    lines.append("")
    lines.append(f"- 生成时间：{result.generated_at}")
    lines.append(f"- 数据源文件：`{result.data_path}`")
    if result.sheet_name:
        lines.append(f"- 工作表：{result.sheet_name}")
    lines.append(f"- 清洗后记录数：{result.cleaned.stats.get('清洗后记录数', 0)}")
    lines.append("")

    lines.append("## 1. 核心指标")
    lines.append("")
    lines.append(_frame_to_markdown(_metrics_display_frame(result)))
    lines.append("")

    lines.append("## 2. 项目维度汇总")
    lines.append("")
    lines.append(_frame_to_markdown(result.project_summary))
    lines.append("")

    lines.append("## 3. 异常识别")
    lines.append("")
    lines.append(
        "阈值："
        + "；".join(f"{key} `{value:g}`" for key, value in result.thresholds.as_dict().items())
    )
    lines.append("")
    if result.anomalies.empty:
        lines.append("未发现异常记录。")
    else:
        lines.append(f"共命中异常记录 **{len(result.anomalies)}** 条。")
        lines.append("")
        lines.append("### 3.1 按异常类型汇总")
        lines.append("")
        lines.append(_frame_to_markdown(result.anomaly_summary))
        lines.append("")
        lines.append("### 3.2 按项目汇总")
        lines.append("")
        lines.append(_frame_to_markdown(result.anomaly_project_summary))
        lines.append("")
        lines.append("### 3.3 异常明细（前 50 条）")
        lines.append("")
        lines.append(_frame_to_markdown(result.anomalies.head(50)))
    lines.append("")

    lines.append("## 4. 数据清洗统计")
    lines.append("")
    for key, value in result.cleaned.stats.items():
        lines.append(f"- {key}：{value}")
    lines.append("")
    lines.append("## 5. 数据清洗问题明细")
    lines.append("")
    lines.append(_frame_to_markdown(result.cleaned.issues))
    lines.append("")

    lines.append("## 6. 异常规则说明")
    lines.append("")
    lines.append(_frame_to_markdown(describe_rules(result.thresholds)))
    lines.append("")
    lines.append("> 说明：本报告由 Phase 1 基础模块生成，未接入任何大模型 / RAG / 多 Agent 能力。")
    lines.append("")
    return "\n".join(lines)


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    """把 DataFrame 渲染为 Markdown 表格（避免额外依赖 tabulate）。"""

    if frame is None or frame.empty:
        return "_（无数据）_"

    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in frame.iterrows():
        cells = []
        for value in row.tolist():
            text = "" if value is None else str(value)
            if isinstance(value, float) and pd.isna(value):
                text = ""
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _metrics_display_frame(result: "AnalysisResult") -> pd.DataFrame:
    """把核心指标渲染为人类可读的表格（整数值不显示小数点）。"""

    frame = pd.DataFrame(result.metrics.as_rows())
    if frame.empty:
        return frame
    frame = frame.copy()
    rendered: list[str] = []
    for value in frame["数值"]:
        if value is None:
            rendered.append("")
        elif float(value).is_integer():
            rendered.append(f"{int(float(value)):,}")
        else:
            rendered.append(f"{float(value):,.2f}")
    frame["数值"] = rendered
    return frame
