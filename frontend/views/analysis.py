"""RobotOps AI —— Phase 5 分析工作台页面（Streamlit）。

页面名称：RobotOps AI 机器人运营智能分析平台

本文件只负责：用户操作、文件上传、结果展示、调用已有工作流。
指标计算 / 异常判定 / RAG 检索 / 多 Agent 编排全部复用 Phase 1~4 的既有代码。
Phase 6 增加「创建运营问题」，把异常转成可跟踪的整改任务。

启动：streamlit run frontend/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # 保证可以从项目根目录导入 robotops / app / frontend
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from frontend import ui_helpers as ui
from robotops import config as project_config
from robotops.exceptions import RobotOpsError
from robotops.issues import DEFAULT_OWNER, IssueError, IssueService, IssueStore
from robotops.models import COLUMN_SPECS


@st.cache_data(show_spinner=False, max_entries=8)
def parse_upload(file_bytes: bytes, file_name: str) -> dict:
    """解析上传文件（按内容缓存，避免每次交互重复解析）。"""

    path = ui.save_uploaded_file(file_name, file_bytes)
    preview = ui.preview_uploaded_file(path)
    return {
        "path": str(preview.path),
        "frame": preview.frame,
        "sheet_name": preview.sheet_name,
        "encoding": preview.encoding,
        "missing_columns": preview.missing_columns,
        "error_message": preview.error_message,
        "row_count": preview.row_count,
        "column_count": preview.column_count,
    }


# ---------------------------------------------------------------------------
# 页面标题
# ---------------------------------------------------------------------------
st.title("RobotOps AI 机器人运营智能分析平台")
st.caption(
    "上传机器人运营数据 → Pandas 指标计算与异常识别 → RAG 历史案例检索 → "
    "多 Agent 工作流生成运营分析报告"
)

# ---------------------------------------------------------------------------
# 侧边栏：分析设置
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("分析设置")
    has_api_key, masked_key = ui.api_key_state()
    use_ai = st.toggle(
        "启用 AI 分析（DeepSeek）",
        value=has_api_key,
        help="关闭后使用离线模板结论，不调用大模型、不消耗 API 额度",
        key="use_ai_toggle",
    )
    use_rag = st.toggle(
        "参考历史案例（RAG）",
        value=True,
        help="从 knowledge/ 知识库检索相似历史案例",
        key="use_rag_toggle",
    )
    top_k = st.slider(
        "历史案例检索数量", min_value=1, max_value=10, value=5, step=1, key="rag_top_k_slider"
    )

    with st.expander("异常判定阈值（由程序执行）"):
        defaults = project_config.DEFAULT_THRESHOLDS
        fault_rate_upper = st.number_input(
            "故障率上限（%）",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults.fault_rate_upper),
            step=0.5,
            key="threshold_fault_rate",
        )
        satisfaction_lower = st.number_input(
            "满意度下限（分）",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults.satisfaction_lower),
            step=1.0,
            key="threshold_satisfaction",
        )
        uptime_rate_lower = st.number_input(
            "运行率下限（%）",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults.uptime_rate_lower),
            step=1.0,
            key="threshold_uptime_rate",
        )
        saving_rate_lower = st.number_input(
            "节降率下限（%）",
            min_value=-100.0,
            max_value=100.0,
            value=float(defaults.saving_rate_lower),
            step=1.0,
            key="threshold_saving_rate",
        )

    st.divider()
    st.caption(f"API Key：{masked_key if has_api_key else '未配置'}")
    st.caption(f"运行模式：{'AI 分析' if use_ai else '离线模板'} · {'含历史案例' if use_rag else '不含历史案例'}")
    if not has_api_key:
        st.warning(
            "未检测到 DeepSeek API Key。请在项目根目录创建 `.env` 并配置 "
            "`DEEPSEEK_API_KEY`（可复制 `.env.example`），或关闭「启用 AI 分析」以离线模板方式运行。",
            icon=":material/key_off:",
        )

thresholds = project_config.AnomalyThresholds(
    fault_rate_upper=fault_rate_upper,
    satisfaction_lower=satisfaction_lower,
    uptime_rate_lower=uptime_rate_lower,
    saving_rate_lower=saving_rate_lower,
)

# ---------------------------------------------------------------------------
# 一、上传数据 + 数据预览
# ---------------------------------------------------------------------------
st.subheader("一、上传运营数据")
st.caption(
    "支持 .xlsx 与 .csv。必需字段："
    + "、".join(spec.name for spec in COLUMN_SPECS if spec.required)
)

uploaded_file = st.file_uploader(
    "上传机器人运营数据文件",
    type=["xlsx", "csv"],
    key="uploader",
    help="可先用 scripts/generate_demo_data.py 生成的演示数据试跑（data/raw/robot_operation_demo.xlsx）",
)

# 上传/演示数据只解析一次，把结果放进 session_state 以便跨 rerun 复用
if uploaded_file is not None:
    upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("upload_key") != upload_key:
        st.session_state["upload_info"] = parse_upload(uploaded_file.getvalue(), uploaded_file.name)
        st.session_state["upload_key"] = upload_key
else:
    st.info("请先上传运营数据文件（.xlsx / .csv）。", icon=":material/upload_file:")
    if project_config.DEMO_EXCEL_FILE.exists():
        if st.button("使用项目演示数据试跑", icon=":material/science:"):
            demo_key = f"demo:{project_config.DEMO_EXCEL_FILE.name}"
            if st.session_state.get("upload_key") != demo_key:
                st.session_state["upload_info"] = parse_upload(
                    project_config.DEMO_EXCEL_FILE.read_bytes(),
                    project_config.DEMO_EXCEL_FILE.name,
                )
                st.session_state["upload_key"] = demo_key

upload_info: dict | None = st.session_state.get("upload_info")
data_path: str | None = None

if upload_info and upload_info["error_message"]:
    st.error(f"文件读取失败：{upload_info['error_message']}", icon=":material/error:")
elif upload_info:
    data_path = upload_info["path"]

if upload_info and not upload_info["error_message"]:
    preview_frame: pd.DataFrame = upload_info["frame"]
    st.markdown("**数据预览**")
    metric_cols = st.container(horizontal=True)
    with metric_cols:
        st.metric("数据总量", f"{upload_info['row_count']:,} 行", border=True)
        st.metric("字段数量", f"{upload_info['column_count']} 列", border=True)
        if upload_info.get("sheet_name"):
            st.metric("工作表", str(upload_info["sheet_name"]), border=True)
        if upload_info.get("encoding"):
            st.metric("文件编码", str(upload_info["encoding"]), border=True)

    st.dataframe(preview_frame.head(ui.PREVIEW_ROWS), hide_index=True, height=320)
    st.caption(f"上表展示前 {ui.PREVIEW_ROWS} 行数据。")

    with st.expander("字段信息"):
        st.dataframe(ui.field_info_frame(preview_frame), hide_index=True)

    if upload_info["missing_columns"]:
        st.error(
            "数据字段缺失，无法进行指标计算：缺少 "
            + "、".join(upload_info["missing_columns"])
            + "。请按下方字段清单补齐后重新上传。",
            icon=":material/rule:",
        )
        with st.expander("必需字段清单", expanded=True):
            st.dataframe(ui.required_columns_frame(), hide_index=True)

# ---------------------------------------------------------------------------
# 二、开始分析
# ---------------------------------------------------------------------------
st.subheader("二、开始分析")
data_ready = bool(data_path) and bool(upload_info) and not upload_info["missing_columns"]
ai_blocked = use_ai and not has_api_key

if ai_blocked:
    st.warning(
        "当前未配置 DeepSeek API Key，无法执行 AI 分析。请在 `.env` 中配置 `DEEPSEEK_API_KEY`，"
        "或关闭左侧「启用 AI 分析」改用离线模板结论。",
        icon=":material/key_off:",
    )

analyze_clicked = st.button(
    "开始分析",
    type="primary",
    icon=":material/play_circle:",
    disabled=not data_ready or ai_blocked,
    help="点击后调用 Phase 4 的多 Agent 工作流（分析 → 诊断 → RAG → 建议 → 报告）",
)

if analyze_clicked and data_ready:
    with st.spinner("正在运行多 Agent 工作流，请稍候…"):
        try:
            state, logs = ui.run_analysis_workflow(
                data_path=data_path,
                thresholds=thresholds,
                use_knowledge=use_rag,
                use_llm=use_ai,
                top_k=top_k,
                output_dir=project_config.OUTPUT_DIR,
            )
            st.session_state["analysis_state"] = state
            st.session_state["analysis_logs"] = logs
            st.session_state["analysis_source"] = Path(str(data_path)).name
            st.session_state["analysis_error"] = ""
        except RobotOpsError as error:
            st.session_state["analysis_state"] = None
            st.session_state["analysis_logs"] = []
            st.session_state["analysis_error"] = ui.describe_error(error)
        except Exception as error:  # noqa: BLE001 - 兜底：转换成用户可读提示，不显示 traceback
            st.session_state["analysis_state"] = None
            st.session_state["analysis_logs"] = []
            st.session_state["analysis_error"] = f"分析过程出现未预期错误：{error}"

analysis_state: dict | None = st.session_state.get("analysis_state")
analysis_error: str = st.session_state.get("analysis_error", "")
analysis_logs: list[str] = st.session_state.get("analysis_logs", [])

if analysis_error:
    st.error(f"分析失败：{analysis_error}", icon=":material/error:")
    with st.expander("排查建议"):
        st.markdown(
            "- 确认数据文件字段完整（参考上方「必需字段清单」）；\n"
            "- 确认 `.env` 中的 `DEEPSEEK_API_KEY` 有效，或关闭「启用 AI 分析」；\n"
            "- 知识库异常时可用 `python scripts\\build_knowledge_base.py rebuild` 重建索引；\n"
            "- 命令行方式排查：`python run_workflow.py --quiet`。"
        )

if analysis_state:
    workflow_status = analysis_state.get("workflow_status", "-")
    st.divider()
    st.caption(
        f"分析文件：{st.session_state.get('analysis_source', '-')} · "
        f"工作流状态：{workflow_status} · "
        f"数据源：{(analysis_state.get('raw_data_summary') or {}).get('数据源', '-')}"
    )

    # 工作流错误（含降级说明）
    workflow_errors = ui.normalize_errors(analysis_state)
    if workflow_errors:
        st.warning(
            f"工作流中有 {len(workflow_errors)} 个阶段出现异常，报告已降级生成：",
            icon=":material/warning:",
        )
        for item in workflow_errors:
            text = f"**[{item['agent']}] {item['kind']}**：{item['message']}"
            if item["hint"]:
                text += f"\n\n处理建议：{item['hint']}"
            st.markdown(text)

    cleaned_data = None
    analysis_result = analysis_state.get("analysis_result")
    if analysis_result is not None:
        cleaned_data = getattr(analysis_result, "cleaned_data", None)

    # ---------------- 三、运营概览 ----------------
    st.subheader("三、运营概览")
    metrics = analysis_state.get("metrics") or {}
    cards = ui.overview_cards(metrics.get("values", {}), metrics.get("units", {}))
    with st.container(horizontal=True):
        for card in cards:
            sparkline = ui.trend_values(cleaned_data, card["label"]) if cleaned_data is not None else []
            if sparkline:
                st.metric(
                    card["label"],
                    card["value"],
                    border=True,
                    help=card["help"],
                    chart_data=sparkline,
                    chart_type="line",
                )
            else:
                st.metric(card["label"], card["value"], border=True, help=card["help"])

    # ---------------- 四、异常项目 ----------------
    st.subheader("四、异常项目")
    project_cards = ui.abnormal_project_cards(analysis_state)
    if not analysis_state.get("has_anomalies"):
        st.success("本期未发现超过阈值的异常项，无需进入诊断与建议环节。", icon=":material/check_circle:")
    elif not project_cards:
        st.info("程序判定存在异常记录，但没有可汇总到项目维度的异常。")
    else:
        for card in project_cards:
            with st.container(border=True):
                st.markdown(f"**项目名称：{card['project']}**")
                st.caption(
                    f"异常记录 {card['anomaly_records']} 条 · 涉及机器人 {card['affected_robots']} 台 · "
                    f"严重程度（{card['severity']}）"
                )
                st.markdown(f"**异常指标**：{card['main_anomaly_types'] or '-'}")
                for reason in card["reasons"]:
                    st.markdown(
                        f"- 推测原因（置信度 {reason['confidence'] or '-'}）：{reason['reason']}"
                    )
                    if reason["based_on"]:
                        st.caption(f"判断依据：{reason['based_on']}")
                    if reason["data_gap"]:
                        st.caption(f"缺失数据：{reason['data_gap']}")

                series = ui.trend_series_for_anomalies(card["main_anomaly_types"])
                trend = (
                    ui.project_daily_metrics(cleaned_data, card["project"])
                    if cleaned_data is not None
                    else pd.DataFrame()
                )
                if not trend.empty and series:
                    available = [name for name in series if name in trend.columns]
                    if available:
                        st.markdown("**变化趋势（按日）**")
                        st.line_chart(trend, x="日期", y=available, height=260)
                else:
                    st.caption("（当前数据不足以绘制趋势，请补充按日明细）")

    # ---------------- 五、AI 分析 ----------------
    st.subheader("五、AI 分析")
    diagnosis = analysis_state.get("diagnosis") or {}
    recommendations = analysis_state.get("recommendations") or {}
    tabs = st.tabs(["数据事实", "异常发现", "可能原因", "历史相似案例", "优化建议"])

    with tabs[0]:
        st.markdown("**核心指标（由 Phase 1 Pandas 模块计算）**")
        metric_rows = [
            {
                "指标": card["label"],
                "数值": card["value"],
                "单位": card["unit"],
                "计算口径": card["help"],
            }
            for card in cards
        ]
        st.dataframe(pd.DataFrame(metric_rows), hide_index=True)
        st.markdown("**数据源与清洗统计**")
        summary = analysis_state.get("raw_data_summary") or {}
        st.dataframe(
            pd.DataFrame(
                [{"统计项": key, "数值": str(value)} for key, value in summary.items()]
            ),
            hide_index=True,
        )

    with tabs[1]:
        anomaly_summary = analysis_state.get("anomaly_summary") or []
        if not anomaly_summary:
            st.info("未发现超过阈值的异常（判定由程序按固定阈值完成）。")
        else:
            st.caption("异常由程序按固定阈值判定，大模型不参与异常判断。")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "异常类型": item.get("anomaly_type"),
                            "判定条件": item.get("condition"),
                            "命中记录": item.get("hits"),
                            "涉及项目": item.get("projects_involved"),
                            "涉及机器人": item.get("robots_involved"),
                            "高/中/低": f"{item.get('high')}/{item.get('medium')}/{item.get('low')}",
                        }
                        for item in anomaly_summary
                    ]
                ),
                hide_index=True,
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "项目名称": item.get("project"),
                            "异常记录": item.get("anomaly_records"),
                            "涉及机器人": item.get("affected_robots"),
                            "主要异常类型": item.get("main_anomaly_types"),
                        }
                        for item in analysis_state.get("abnormal_projects") or []
                    ]
                ),
                hide_index=True,
            )

    with tabs[2]:
        reasons = diagnosis.get("possible_reasons") or []
        if not analysis_state.get("has_anomalies"):
            st.info("本期无异常，不需要原因推测。")
        elif not reasons:
            st.info("当前数据不足以判断：未获得可用的诊断结论。")
        else:
            st.caption(
                f"诊断来源：{diagnosis.get('source', '-')}"
                + (f"（模型：{diagnosis.get('model')}）" if diagnosis.get("model") else "")
                + "；以下内容均为推测，需补充数据后确认。"
            )
            if diagnosis.get("summary"):
                st.markdown(diagnosis["summary"])
            for index, reason in enumerate(reasons, start=1):
                with st.container(border=True):
                    st.markdown(f"**{index}. {reason.get('reason', '')}**")
                    st.caption(f"置信度：{reason.get('confidence', '-')}")
                    if reason.get("based_on"):
                        st.markdown(f"判断依据：{reason['based_on']}")
                    st.markdown(f"缺失数据：{reason.get('data_gap') or '当前数据不足以判断'}")

    with tabs[3]:
        cases = analysis_state.get("retrieved_cases") or []
        if not cases:
            st.info("未检索到足够相关的历史案例（知识库中没有相似案例，或相似度低于阈值）。")
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "案例编号": case.get("case_id"),
                            "案例标题": case.get("title"),
                            "案例类型": case.get("case_type"),
                            "相关度": case.get("similarity"),
                            "来源文件": case.get("source_file"),
                        }
                        for case in cases
                    ]
                ),
                hide_index=True,
            )
            st.caption(f"共 {len(cases)} 个历史案例，{ui.CASE_DISCLAIMER}。")

    with tabs[4]:
        items = recommendations.get("items") or []
        if not analysis_state.get("has_anomalies"):
            st.info("本期无异常，暂不需要针对性优化建议。")
        elif not items:
            st.info("当前数据不足以判断：未生成可执行的优化建议。")
        else:
            st.caption(f"建议来源：{recommendations.get('source', '-')}")
            if recommendations.get("summary"):
                st.markdown(recommendations["summary"])
            for index, item in enumerate(items, start=1):
                with st.container(border=True):
                    st.markdown(f"**{index}. [{item.get('priority', '-')}] {item.get('action', '')}**")
                    if item.get("target"):
                        st.markdown(f"对象：{item['target']}")
                    if item.get("expected_effect"):
                        st.markdown(f"预期效果：{item['expected_effect']}")
                    if item.get("verification"):
                        st.markdown(f"验证方式：{item['verification']}")
                    if item.get("reference_case") and item["reference_case"] != "无":
                        st.markdown(f"参考案例：{item['reference_case']}")

    # ---------------- 六、RAG 历史参考案例 ----------------
    st.subheader("六、RAG 历史参考案例")
    case_cards = ui.case_cards(analysis_state)
    if not case_cards:
        st.info("未检索到足够相关的历史案例（本次未参考知识库案例）。")
    else:
        st.caption(f"共 {len(case_cards)} 个案例，全部为{ui.CASE_DISCLAIMER}。")
        for card in case_cards:
            with st.container(border=True):
                st.markdown(f"**{card['title']}**　`{card['case_id']}`")
                similarity = card["similarity"]
                st.caption(
                    f"案例类型：{card['case_type']} · 相关度：{similarity} · 来源：{card['source_file']}"
                )
                st.markdown(f"**相似原因**：{card['possible_reason'] or '（案例未提供）'}")
                st.markdown(f"**处理措施**：{card['actions'] or '（案例未提供）'}")
                st.warning(card["disclaimer"], icon=":material/priority_high:")

    # ---------------- 七、最终运营报告 ----------------
    st.subheader("七、最终运营报告")
    report_payload = ui.report_download_payload(analysis_state)
    with st.container(border=True):
        st.markdown(analysis_state.get("final_report") or "（本次运行未生成报告内容）")
    st.download_button(
        "下载 Markdown 报告",
        data=report_payload["markdown"],
        file_name=report_payload["file_name"],
        mime=report_payload["mime"],
        icon=":material/download:",
        type="primary",
    )

    # ---------------- 八、运营问题（Phase 6 闭环入口）----------------
    st.subheader("八、运营问题")
    with st.container(border=True):
        if not analysis_state.get("has_anomalies"):
            st.info("本次分析未发现超过阈值的异常，无需创建运营问题。")
        else:
            st.markdown("**创建运营问题**")
            st.caption("把本次分析出的异常转成可跟踪的整改任务，后续在「运营问题」页面处理与验证。")
            project_options = [
                str(item.get("project")) for item in analysis_state.get("abnormal_projects") or []
            ]
            selected_projects = st.multiselect(
                "选择要创建问题的项目",
                project_options,
                default=project_options,
                key="issue_projects",
            )
            owner = st.text_input("负责人", value=DEFAULT_OWNER, key="issue_owner")
            if st.button(
                "创建运营问题",
                icon=":material/add_task:",
                type="secondary",
                key="create_issues",
            ):
                try:
                    service = IssueService(IssueStore())
                    created = service.create_from_state(
                        analysis_state,
                        owner=owner or DEFAULT_OWNER,
                        projects=selected_projects,
                        source_file=st.session_state.get("analysis_source", ""),
                    )
                    st.success(
                        f"已创建 {len(created)} 个运营问题："
                        + "、".join(issue.issue_id for issue in created)
                    )
                    st.dataframe(
                        pd.DataFrame([issue.list_row() for issue in created]),
                        hide_index=True,
                    )
                    st.caption("请在左侧导航切换到「运营问题」页面查看详情、填写处理结果并验证整改效果。")
                except IssueError as error:
                    st.error(f"创建运营问题失败：{error.describe()}", icon=":material/error:")
                except Exception as error:  # noqa: BLE001
                    st.error(f"创建运营问题失败：{error}", icon=":material/error:")

    with st.expander("工作流执行轨迹与日志"):
        st.dataframe(ui.workflow_trace(analysis_state), hide_index=True)
        if analysis_logs:
            st.code("\n".join(analysis_logs), language="text")

st.divider()
st.caption(
    "RobotOps AI · Phase 5 可视化界面：本页仅负责操作、上传、展示与调用已有工作流，"
    "指标与异常均由程序计算，AI 结论仅供参考。"
)
