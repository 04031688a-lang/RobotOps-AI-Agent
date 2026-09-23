"""RobotOps AI —— Phase 6「运营问题」页面（Streamlit）。

覆盖整改闭环：创建 → 处理 → 待验证 → 整改效果验证 → 关闭。
所有指标值与改善幅度均由程序按实际上传数据计算，不使用大模型推断。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from frontend import issues_helpers as ih
from frontend.ui_helpers import CASE_DISCLAIMER, describe_error, run_analysis_workflow, save_uploaded_file
from robotops import config as project_config
from robotops.exceptions import RobotOpsError
from robotops.issues import IssueError, IssueService, IssueStatus, allowed_next_statuses


@st.cache_resource(show_spinner=False)
def get_service(db_path: str) -> IssueService:
    """缓存问题服务（复用同一条 SQLite 连接）。"""

    return ih.default_service(db_path)


st.title("运营问题")
st.caption(
    "运营问题整改闭环：发现问题 → AI 分析 → RAG 历史案例 → AI 建议 → 创建问题 → "
    "售后/运维处理 → 重新上传数据 → 整改效果验证 → 问题关闭。"
)
st.caption(
    "数据安全说明：本页面仅保存业务数据（项目、指标、处理记录），**不保存任何 API Key**；"
    "密钥仍只保存在项目根目录的 .env 中。"
)

try:
    service = get_service(ih.default_db_path())
    stats = service.stats()
except RobotOpsError as error:
    st.error(f"无法打开问题数据库：{describe_error(error)}", icon=":material/error:")
    st.stop()

# ---------------------------------------------------------------------------
# 一、统计
# ---------------------------------------------------------------------------
st.subheader("一、问题统计")
with st.container(horizontal=True):
    for card in ih.stats_cards(stats):
        st.metric(card["label"], card["value"], border=True)

if stats.get("平均处理周期(天)") is None:
    st.caption("平均处理周期：暂无已关闭问题，无法计算。")
else:
    st.caption(
        f"平均处理周期基于 {stats.get('已关闭问题数(计入周期)')} 个已关闭问题计算（创建时间 → 关闭时间）。"
    )

# ---------------------------------------------------------------------------
# 二、问题列表
# ---------------------------------------------------------------------------
st.subheader("二、问题列表")
filter_cols = st.columns([2, 1])
with filter_cols[0]:
    status_filter = st.segmented_control(
        "状态筛选",
        options=["全部", *IssueStatus.values()],
        default="全部",
        key="issue_status_filter",
    )
with filter_cols[1]:
    project_filter = st.selectbox(
        "项目筛选", options=["全部", *service.projects()], key="issue_project_filter"
    )

issues = service.list_issues(
    status=None if status_filter in (None, "全部") else str(status_filter),
    project=None if project_filter == "全部" else str(project_filter),
)
issue_frame = ih.issues_to_frame(issues)

if issue_frame.empty:
    st.info(
        "当前没有符合条件的运营问题。请在「分析工作台」完成分析后点击「创建运营问题」。",
        icon=":material/info:",
    )
    st.stop()

st.dataframe(issue_frame, hide_index=True)
st.caption("在下拉框中选择问题编号即可查看完整详情、填写处理结果并执行整改效果验证。")

# ---------------------------------------------------------------------------
# 三、问题详情
# ---------------------------------------------------------------------------
selected_label = st.selectbox(
    "选择问题编号查看详情", options=ih.issue_options(issues), key="issue_selected"
)
issue_id = str(selected_label).split("｜")[0]
try:
    issue = service.get(issue_id)
except IssueError as error:
    st.error(f"读取问题失败：{describe_error(error)}", icon=":material/error:")
    st.stop()

st.divider()
st.subheader(f"三、问题详情：{issue.issue_id}")

header_items = list(ih.issue_header(issue).items())
with st.container(border=True):
    for start in range(0, len(header_items), 4):
        columns = st.columns(4)
        for column, (label, value) in zip(columns, header_items[start : start + 4]):
            with column:
                st.markdown(f"**{label}**")
                st.markdown(str(value))

st.markdown("**异常指标**")
st.dataframe(ih.metrics_frame(issue), hide_index=True)

detail_left, detail_right = st.columns(2)
with detail_left:
    st.markdown("**当前数据（项目维度）**")
    st.dataframe(ih.current_data_frame(issue), hide_index=True)
with detail_right:
    st.markdown("**AI 诊断结果（推测）**")
    summary = ih.diagnosis_summary(issue)
    st.markdown(summary or "（暂无诊断结论：当前数据不足以判断）")
    for index, reason in enumerate(ih.diagnosis_rows(issue), start=1):
        st.markdown(f"{index}. {reason['reason']}（置信度 {reason['confidence'] or '-'}）")
        if reason["based_on"]:
            st.caption(f"判断依据：{reason['based_on']}")
        if reason["data_gap"]:
            st.caption(f"缺失数据：{reason['data_gap']}")

st.markdown("**RAG 历史案例（仅供参考）**")
case_frame = ih.case_rows(issue)
if case_frame.empty:
    st.info("该问题创建时未检索到足够相关的历史案例。")
else:
    st.dataframe(case_frame, hide_index=True)
    st.caption(CASE_DISCLAIMER + "。")

st.markdown("**AI 建议**")
recommendations = ih.recommendation_rows(issue)
if not recommendations:
    st.info("该问题创建时未生成优化建议。")
else:
    for index, item in enumerate(recommendations, start=1):
        with st.container(border=True):
            st.markdown(f"**{index}. [{item['priority'] or '-'}] {item['action']}**")
            if item["target"]:
                st.markdown(f"对象：{item['target']}")
            if item["expected_effect"]:
                st.markdown(f"预期效果：{item['expected_effect']}")
            if item["verification"]:
                st.markdown(f"验证方式：{item['verification']}")
            if item["reference_case"] and item["reference_case"] != "无":
                st.markdown(f"参考案例：{item['reference_case']}")

st.markdown("**处理结果**")
st.markdown(issue.resolution or "（尚未填写处理结果）")

st.markdown("**处理记录与备注**")
st.dataframe(ih.note_rows(issue), hide_index=True)

if issue.verification:
    verification_text = ih.verification_text(issue.verification)
    st.markdown("**整改效果验证**")
    if verification_text["statement"]:
        st.success(verification_text["statement"], icon=":material/verified:")
    st.dataframe(ih.verification_frame(issue.verification), hide_index=True)
    st.caption(
        f"{verification_text['conclusion']} {verification_text['note']}".strip()
        + f"（验证时间：{issue.verification.get('verified_at', '-')}）"
    )

# ---------------------------------------------------------------------------
# 四、问题处理
# ---------------------------------------------------------------------------
st.subheader("四、问题处理")
next_statuses = [issue.status, *allowed_next_statuses(issue.status)]
with st.form("issue_update_form", border=True):
    st.markdown("**修改状态 / 填写处理结果 / 添加备注**")
    form_cols = st.columns([1, 1])
    with form_cols[0]:
        new_status = st.selectbox("状态", options=next_statuses, index=0, key="issue_new_status")
    with form_cols[1]:
        new_owner = st.text_input("负责人", value=issue.owner, key="issue_owner_input")
    new_note = st.text_area(
        "添加备注",
        key="issue_note_input",
        placeholder="例如：已派单给售后运维，预计 2 个工作日内完成现场处理",
    )
    new_resolution = st.text_area(
        "处理结果（整改完成后填写）",
        value=issue.resolution,
        key="issue_resolution_input",
        placeholder="例如：更换故障部件并调整作业区域，增加点检频次",
    )
    submitted = st.form_submit_button("保存更新", type="primary", icon=":material/save:")

if submitted:
    try:
        if new_status != issue.status:
            service.update_status(issue_id, new_status, owner=new_owner, note=new_note)
            st.success(f"状态已更新为「{new_status}」。", icon=":material/check_circle:")
        else:
            if new_owner != issue.owner:
                service.update_status(issue_id, issue.status, owner=new_owner)
            if new_note.strip():
                service.add_note(issue_id, new_note)
            st.success("已保存。", icon=":material/check_circle:")
        if new_resolution.strip() and new_resolution.strip() != issue.resolution:
            service.set_resolution(issue_id, new_resolution)
            st.success("处理结果已保存。", icon=":material/check_circle:")
        st.rerun()
    except IssueError as error:
        st.error(f"保存失败：{describe_error(error)}", icon=":material/error:")
    except RobotOpsError as error:
        st.error(f"保存失败：{describe_error(error)}", icon=":material/error:")

# ---------------------------------------------------------------------------
# 五、整改效果验证（待验证状态）
# ---------------------------------------------------------------------------
if issue.requires_verification:
    st.subheader("五、整改效果验证")
    st.caption(
        "上传整改后的运营数据，系统将按实际数据自动计算改善幅度："
        "改善幅度 = |整改前 - 整改后| ÷ |整改前|；对比仅使用程序计算的指标，不使用大模型推断。"
    )
    uploaded_after = st.file_uploader(
        "上传整改后的运营数据（.xlsx / .csv）",
        type=["xlsx", "csv"],
        key="verify_upload",
        help="建议使用与首次分析相同的字段格式，项目名称保持不变",
    )
    verify_clicked = st.button(
        "验证整改效果",
        icon=":material/verified:",
        type="primary",
        disabled=uploaded_after is None,
        key="verify_button",
    )
    if verify_clicked and uploaded_after is not None:
        try:
            after_path = save_uploaded_file(uploaded_after.name, uploaded_after.getvalue())
            with st.spinner("正在分析整改后的数据并对比指标…"):
                after_state, _logs = run_analysis_workflow(
                    data_path=after_path,
                    thresholds=project_config.DEFAULT_THRESHOLDS,
                    use_knowledge=False,
                    use_llm=False,  # 对比只用程序计算的指标，无需调用大模型
                    top_k=1,
                    output_dir=project_config.OUTPUT_DIR,
                )
                issue_after, result = service.verify_improvement(
                    issue_id, after_state, after_file=uploaded_after.name
                )
            st.success(result.statement, icon=":material/trending_down:")
            st.dataframe(ih.verification_frame(result.to_dict()), hide_index=True)
            st.caption(f"{result.conclusion} {result.note}")
            st.info(
                f"当前状态：{issue_after.status}"
                + (
                    "（验证通过已自动标记为「已完成」，可点击下方按钮关闭问题）"
                    if issue_after.status == IssueStatus.COMPLETED.value
                    else "（验证未通过，可继续处理并再次验证）"
                ),
                icon=":material/flag:",
            )
        except IssueError as error:
            st.error(f"整改效果验证失败：{describe_error(error)}", icon=":material/error:")
        except RobotOpsError as error:
            st.error(f"整改效果验证失败：{describe_error(error)}", icon=":material/error:")
        except Exception as error:  # noqa: BLE001 - 兜底为可读提示
            st.error(f"整改效果验证失败：{error}", icon=":material/error:")
else:
    st.caption(
        "提示：把问题状态改为「待验证」后，即可上传整改后的数据并验证整改效果。"
    )

# ---------------------------------------------------------------------------
# 六、关闭问题
# ---------------------------------------------------------------------------
if issue.status == IssueStatus.COMPLETED.value:
    st.subheader("六、关闭问题")
    close_note = st.text_input(
        "关闭说明（可选）", key="close_note", placeholder="例如：整改效果验证通过，问题关闭"
    )
    if st.button("关闭问题", icon=":material/task_alt:", type="primary", key="close_issue"):
        try:
            service.close(issue_id, note=close_note or "整改效果验证通过，问题关闭")
            st.success(f"问题 {issue_id} 已关闭。", icon=":material/check_circle:")
            st.rerun()
        except IssueError as error:
            st.error(f"关闭失败：{describe_error(error)}", icon=":material/error:")

st.divider()
st.caption(
    f"问题库位置：{ih.default_db_path()}（SQLite，可用环境变量 ROBOTOPS_ISSUES_DB 覆盖）· "
    f"共 {stats.get('总问题数', 0)} 个问题"
)

