"""RobotOps AI —— Phase 5 前端展示辅助函数（纯函数，不依赖 Streamlit）。

前端只负责：用户操作、文件上传、展示结果、调用已有工作流。
因此这里只做「读文件 / 整理展示数据 / 格式化」这类无状态工作，
指标计算、异常判定、RAG 检索、多 Agent 编排全部复用 Phase 1~4 的既有代码。
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from robotops import config as project_config
from robotops.data_loader import load_operation_data
from robotops.exceptions import RobotOpsError
from robotops.llm.config import LLMConfig
from robotops.metrics import metric_definitions_frame
from robotops.models import COLUMN_SPECS

UPLOAD_DIR = project_config.OUTPUT_DIR / "ui_uploads"
SUPPORTED_SUFFIXES: tuple[str, ...] = (".xlsx", ".csv")
PREVIEW_ROWS = 10

#: RAG 案例必须展示的免责声明
CASE_DISCLAIMER = "历史参考案例，不代表当前项目实际情况"

#: 运营概览展示的 7 项核心指标（顺序固定）
OVERVIEW_METRICS: tuple[tuple[str, str], ...] = (
    ("项目数量", "个"),
    ("机器人数量", "台"),
    ("平均运行时长", "小时"),
    ("故障率", "%"),
    ("平均满意度", "分"),
    ("总运营成本", "元"),
    ("平均节降率", "%"),
)

#: 指标 -> 用于趋势图的清洗后数据列（前端做趋势展示，不重新定义指标）
METRIC_TREND_COLUMNS: dict[str, str] = {
    "平均运行时长": project_config.COL_RUNNING_HOURS,
    "故障率": project_config.COL_FAULT_RATE,
    "平均满意度": project_config.COL_SATISFACTION,
    "平均节降率": project_config.COL_SAVING_RATE,
    "运行率": project_config.COL_UPTIME_RATE,
}

#: 异常类型 -> 异常项目趋势图展示的指标（变化趋势）
ANOMALY_TYPE_TO_SERIES: dict[str, str] = {
    "故障率偏高": "故障率",
    "满意度偏低": "满意度",
    "运行率偏低": "运行率",
    "节降率偏低": "节降率",
}


@dataclass
class UploadPreview:
    """上传文件的预览信息。"""

    path: Path
    frame: pd.DataFrame
    sheet_name: str | None = None
    encoding: str | None = None
    missing_columns: list[str] = field(default_factory=list)
    error_message: str = ""

    @property
    def is_readable(self) -> bool:
        return not self.error_message

    @property
    def is_valid(self) -> bool:
        return self.is_readable and not self.missing_columns

    @property
    def row_count(self) -> int:
        return int(len(self.frame))

    @property
    def column_count(self) -> int:
        return int(self.frame.shape[1])


# ---------------------------------------------------------------------------
# 文件上传与预览
# ---------------------------------------------------------------------------
def save_uploaded_file(file_name: str, data: bytes, *, upload_dir: Path | None = None) -> Path:
    """把上传的字节流写入本地临时目录，返回文件路径。"""

    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"暂不支持的文件类型：{suffix or '（无后缀）'}；请上传 "
            + " 或 ".join(SUPPORTED_SUFFIXES)
            + " 文件"
        )

    target_dir = upload_dir or UPLOAD_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", Path(file_name).name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = target_dir / f"{stamp}_{safe_name}"
    path.write_bytes(data)
    return path


def clear_upload_dir(upload_dir: Path | None = None) -> None:
    """清理上传临时目录（best effort）。"""

    shutil.rmtree(upload_dir or UPLOAD_DIR, ignore_errors=True)


def preview_uploaded_file(path: str | Path) -> UploadPreview:
    """读取上传文件用于预览：容忍字段缺失，仅在文件不可读时报错。"""

    target = Path(path)
    try:
        result = load_operation_data(target, validate=False)
    except RobotOpsError as error:
        return UploadPreview(path=target, frame=pd.DataFrame(), error_message=str(error))
    except Exception as error:  # noqa: BLE001 - 任何解析异常都要转成用户可读提示
        return UploadPreview(
            path=target,
            frame=pd.DataFrame(),
            error_message=f"文件解析失败：{error}",
        )

    missing = [name for name in project_config.REQUIRED_COLUMNS if name not in result.data.columns]
    return UploadPreview(
        path=target,
        frame=result.data,
        sheet_name=result.sheet_name,
        encoding=result.encoding,
        missing_columns=missing,
    )


def field_info_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """字段信息表：名称 / 类型 / 非空 / 缺失 / 唯一值 / 示例 / 是否必需。"""

    spec_map = {spec.name: spec for spec in COLUMN_SPECS}
    rows: list[dict[str, Any]] = []
    for name in frame.columns:
        column = frame[name]
        spec = spec_map.get(str(name))
        example = column.dropna()
        rows.append(
            {
                "字段名": str(name),
                "数据类型": _dtype_label(column),
                "非空数量": int(column.notna().sum()),
                "缺失数量": int(column.isna().sum()),
                "唯一值": int(column.nunique(dropna=True)),
                "示例值": "" if example.empty else str(example.iloc[0])[:30],
                "是否必需": (
                    "必需"
                    if spec is not None and spec.required
                    else ("可选" if spec is not None else "扩展字段")
                ),
                "说明": spec.description if spec is not None else "数据源扩展字段，不参与核心指标计算",
            }
        )
    return pd.DataFrame(rows)


def required_columns_frame() -> pd.DataFrame:
    """必需字段清单（用于字段缺失时的提示）。"""

    return pd.DataFrame(
        [
            {
                "字段名": spec.name,
                "是否必需": "必需" if spec.required else "可选",
                "单位": spec.unit or "-",
                "说明": spec.description,
                "示例值": spec.example,
            }
            for spec in COLUMN_SPECS
        ]
    )


# ---------------------------------------------------------------------------
# 运营概览
# ---------------------------------------------------------------------------
def overview_cards(
    metrics_values: dict[str, Any],
    metrics_units: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """构造运营概览的指标卡片（需求要求的 7 项核心指标）。"""

    units = metrics_units or {}
    definitions = {
        str(row["指标"]): str(row["计算口径"]) for _, row in metric_definitions_frame().iterrows()
    }
    cards: list[dict[str, str]] = []
    for name, default_unit in OVERVIEW_METRICS:
        value = metrics_values.get(name)
        cards.append(
            {
                "label": name,
                "value": format_metric_value(value, units.get(name, default_unit)),
                "unit": units.get(name, default_unit),
                "help": definitions.get(name, ""),
            }
        )
    return cards


def format_metric_value(value: Any, unit: str = "") -> str:
    """把指标数值格式化为界面展示文本。"""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        text = f"{int(number):,}"
    elif abs(number) >= 1000:
        text = f"{number:,.2f}"
    else:
        text = f"{number:.2f}"
    if unit == "%":
        return f"{text}%"
    return text


def metric_help_text(metric_name: str) -> str:
    """返回指标计算口径说明。"""

    definitions = {
        str(row["指标"]): str(row["计算口径"]) for _, row in metric_definitions_frame().iterrows()
    }
    return definitions.get(metric_name, "")


# ---------------------------------------------------------------------------
# 趋势（基于 Phase 1 清洗后的数据）
# ---------------------------------------------------------------------------
def daily_trend(
    cleaned: pd.DataFrame,
    column: str,
    *,
    project: str | None = None,
) -> pd.DataFrame:
    """按日期聚合某个指标，返回 ``日期 + 指标`` 两列（用于 sparkline / 折线图）。"""

    if cleaned is None or cleaned.empty or column not in cleaned.columns:
        return pd.DataFrame(columns=["日期", column])

    frame = cleaned
    if project and project_config.COL_PROJECT in frame.columns:
        frame = frame.loc[frame[project_config.COL_PROJECT] == project]
    if frame.empty:
        return pd.DataFrame(columns=["日期", column])

    grouped = (
        frame.groupby(project_config.COL_DATE, as_index=False)[column]
        .mean()
        .sort_values(project_config.COL_DATE)
    )
    return grouped.rename(columns={column: column})


def trend_values(cleaned: pd.DataFrame, metric_name: str, *, project: str | None = None) -> list[float]:
    """取某指标按日的均值序列（用于指标卡片的 sparkline）。"""

    column = METRIC_TREND_COLUMNS.get(metric_name)
    if not column:
        return []
    trend = daily_trend(cleaned, column, project=project)
    if trend.empty:
        return []
    return [round(float(value), 4) for value in trend[column].tolist() if pd.notna(value)]


def trend_series_for_anomalies(main_anomaly_types: Any) -> list[str]:
    """按异常类型决定趋势图展示哪些指标（例如「故障率偏高」→ 故障率）。"""

    text = str(main_anomaly_types or "")
    series: list[str] = []
    for anomaly_type, label in ANOMALY_TYPE_TO_SERIES.items():
        if anomaly_type in text and label not in series:
            series.append(label)
    return series


def project_daily_metrics(cleaned: pd.DataFrame, project: str) -> pd.DataFrame:
    """某项目的按日关键指标（用于异常项目的变化趋势）。"""

    if cleaned is None or cleaned.empty:
        return pd.DataFrame()
    frame = cleaned.loc[cleaned[project_config.COL_PROJECT] == project]
    if frame.empty:
        return pd.DataFrame()

    wanted = {
        project_config.COL_RUNNING_HOURS: "运行时长",
        project_config.COL_UPTIME_RATE: "运行率",
        project_config.COL_FAULT_RATE: "故障率",
        project_config.COL_SATISFACTION: "满意度",
        project_config.COL_SAVING_RATE: "节降率",
    }
    available = {column: label for column, label in wanted.items() if column in frame.columns}
    if not available:
        return pd.DataFrame()

    grouped = frame.groupby(project_config.COL_DATE, as_index=False)[list(available)].mean()
    grouped = grouped.rename(columns=available).sort_values(project_config.COL_DATE)
    return grouped


# ---------------------------------------------------------------------------
# 异常项目 / RAG 案例 / 错误信息
# ---------------------------------------------------------------------------
def abnormal_project_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    """把异常项目、异常指标与诊断内容整理成卡片数据。"""

    projects = state.get("abnormal_projects") or []
    thresholds = state.get("thresholds") or {}
    diagnosis = state.get("diagnosis") or {}
    reasons = diagnosis.get("possible_reasons") or []

    cards: list[dict[str, Any]] = []
    for item in projects:
        types = _split_types(item.get("main_anomaly_types"))
        matched = [reason for reason in reasons if reason.get("anomaly_type") in types] or reasons
        cards.append(
            {
                "project": item.get("project", "-"),
                "anomaly_records": item.get("anomaly_records", 0),
                "affected_robots": item.get("affected_robots", 0),
                "main_anomaly_types": item.get("main_anomaly_types", ""),
                "severity": (
                    f"高 {item.get('high_severity', 0)} / 中 {item.get('medium_severity', 0)} / "
                    f"低 {item.get('low_severity', 0)}"
                ),
                "reasons": [
                    {
                        "anomaly_type": reason.get("anomaly_type", ""),
                        "reason": reason.get("reason", ""),
                        "confidence": reason.get("confidence", ""),
                        "based_on": reason.get("based_on", ""),
                        "data_gap": reason.get("data_gap", ""),
                    }
                    for reason in matched[:3]
                ],
                "thresholds": thresholds,
            }
        )
    return cards


def case_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    """把 RAG 检索到的历史案例整理成卡片数据（含免责声明）。"""

    cases = state.get("retrieved_cases") or []
    cards: list[dict[str, Any]] = []
    for case in cases:
        sections = parse_case_sections(str(case.get("case_summary") or ""))
        cards.append(
            {
                "case_id": case.get("case_id", "-"),
                "title": case.get("title", "-"),
                "case_type": case.get("case_type", "-"),
                "similarity": case.get("similarity"),
                "source_file": case.get("source_file", "-"),
                "data_nature": case.get("data_nature", "模拟案例（虚构）"),
                "possible_reason": _truncate(sections.get("可能原因", ""), 180),
                "actions": _truncate(sections.get("处理措施", ""), 220),
                "matched_sections": case.get("matched_sections") or [],
                "matched_queries": case.get("matched_queries") or [],
                "disclaimer": CASE_DISCLAIMER,
            }
        )
    return cards


def parse_case_sections(case_summary: str) -> dict[str, str]:
    """解析案例文本中的「【小节】内容」结构。"""

    text = case_summary or ""
    matches = list(re.finditer(r"【([^】]+)】", text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        name = match.group(1).strip()
        content = text[start:end].strip()
        if name and content:
            sections[name] = content
    return sections


def normalize_errors(state: dict[str, Any]) -> list[dict[str, str]]:
    """把工作流错误记录整理成用户可读列表。"""

    records = state.get("errors") or []
    normalized: list[dict[str, str]] = []
    for record in records:
        normalized.append(
            {
                "agent": str(record.get("agent") or "-"),
                "kind": str(record.get("kind") or "执行异常"),
                "message": _truncate(str(record.get("message") or "").replace("\n", " "), 300),
                "hint": _truncate(str(record.get("hint") or "").replace("\n", " "), 300),
            }
        )
    return normalized


def workflow_trace(state: dict[str, Any]) -> pd.DataFrame:
    """工作流节点轨迹表。"""

    steps = state.get("steps") or []
    if not steps:
        return pd.DataFrame(columns=["节点", "状态", "耗时(秒)", "说明"])
    return pd.DataFrame(
        [
            {
                "节点": step.get("agent", "-"),
                "状态": step.get("status", "-"),
                "耗时(秒)": step.get("duration_seconds", 0),
                "说明": _truncate(str(step.get("detail") or ""), 160),
            }
            for step in steps
        ]
    )


# ---------------------------------------------------------------------------
# 配置状态与下载
# ---------------------------------------------------------------------------
def api_key_state() -> tuple[bool, str]:
    """返回 (是否已配置 Key, 脱敏 Key)。"""

    try:
        config = LLMConfig.from_env()
    except RobotOpsError:
        return False, "(配置读取失败)"
    return config.has_api_key, config.masked_api_key


def report_download_payload(state: dict[str, Any]) -> dict[str, str]:
    """构造 Markdown 报告下载内容。"""

    report = str(state.get("final_report") or "").strip()
    trace = workflow_trace(state)
    errors = normalize_errors(state)

    parts: list[str] = [report or "# RobotOps AI 运营分析报告\n\n（本次运行未生成报告内容）"]
    if not trace.empty:
        parts.append("\n\n---\n\n## 附：工作流执行轨迹（系统生成）\n")
        for row in trace.to_dict("records"):
            parts.append(f"- [{row['节点']}] {row['状态']}（{row['耗时(秒)']}s）{row['说明']}")
    if errors:
        parts.append("\n\n### 工作流异常与降级说明\n")
        for item in errors:
            line = f"- [{item['agent']}] {item['kind']}：{item['message']}"
            if item["hint"]:
                line += f"（建议：{item['hint']}）"
            parts.append(line)

    return {
        "markdown": "\n".join(parts).strip() + "\n",
        "file_name": "robotops_operation_report.md",
        "mime": "text/markdown",
    }


# ---------------------------------------------------------------------------
# 调用 Phase 4 工作流（前端唯一的业务入口）
# ---------------------------------------------------------------------------
def run_analysis_workflow(
    *,
    data_path: str | Path,
    thresholds: project_config.AnomalyThresholds,
    use_knowledge: bool,
    use_llm: bool,
    top_k: int | None = None,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """调用 Phase 4 的多 Agent 工作流，返回 (最终 State, 工作流日志行)。

    这里不重新实现任何分析逻辑，只负责注入日志器并转发参数。
    """

    from app.graph.logging_utils import WorkflowLogger
    from app.graph.workflow import RobotOpsWorkflow

    target_dir = Path(output_dir) if output_dir is not None else project_config.OUTPUT_DIR
    logger = WorkflowLogger(verbose=False, echo=False, log_file=target_dir / "workflow.log")
    workflow = RobotOpsWorkflow(logger=logger, verbose=False, echo=False)
    state = workflow.run(
        data_path=data_path,
        thresholds=thresholds,
        use_knowledge=use_knowledge,
        use_llm=use_llm,
        rag_top_k=top_k,
        rag_max_cases=top_k,
        export=True,
        output_dir=target_dir,
    )
    return state, list(logger.messages)


def describe_error(error: Exception) -> str:
    """把异常转换为用户可读的一行提示（不暴露 traceback）。"""

    describe = getattr(error, "describe", None)
    text = str(describe()) if callable(describe) else str(error)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _dtype_label(column: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(column):
        return "日期"
    if pd.api.types.is_integer_dtype(column):
        return "整数"
    if pd.api.types.is_float_dtype(column):
        return "数值"
    if pd.api.types.is_bool_dtype(column):
        return "布尔"
    return "文本"


def _split_types(text: Any) -> list[str]:
    if not text:
        return []
    return [part.split("(")[0].strip() for part in str(text).split("、") if part.strip()]


def _truncate(text: str, limit: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "…"
