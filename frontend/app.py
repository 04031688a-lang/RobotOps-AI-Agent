"""RobotOps AI —— 平台入口（Streamlit，Phase 6 最终形态）。

模块（侧边栏切换，视图文件按脚本直接执行）：
- 分析工作台：上传数据 → 多 Agent 分析 → 创建运营问题（Phase 5 + Phase 6 入口）
- 运营问题：问题列表 / 详情 / 处理 / 整改效果验证 / 关闭（Phase 6）

启动：streamlit run frontend/app.py

说明：这里不用 st.navigation/st.Page，而是「单入口 + 侧边栏模块切换 + 直接执行视图脚本」。
原因是多页路由在无头测试环境（AppTest）中会对页面重复执行，导致控件重复创建；
当前方案保证每次 rerun 只执行一个模块，行为可预期、便于自动化测试。
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="RobotOps AI 机器人运营智能分析平台",
    page_icon=":material/smart_toy:",
    layout="wide",
)

#: 模块名 -> 视图脚本（相对 frontend/ 目录）
VIEW_FILES: dict[str, str] = {
    "分析工作台": "views/analysis.py",
    "运营问题": "views/issues.py",
}

with st.sidebar:
    st.caption("RobotOps AI 机器人运营智能分析平台")
    selected_view = st.segmented_control(
        "功能模块",
        options=list(VIEW_FILES),
        default="分析工作台",
        key="view_switcher",
    )

view_name = selected_view or "分析工作台"
view_path = Path(__file__).resolve().parent / VIEW_FILES[view_name]
runpy.run_path(str(view_path), run_name="__view__")
