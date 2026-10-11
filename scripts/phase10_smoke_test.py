"""RobotOps AI —— Phase 10 离线自检脚本（RAG 可追溯性 + Agent 输出可靠性 + 分析依据）。

不依赖 API Key、不发起网络请求，直接校验：

    [1] RAG 来源追踪（case_id / title / category / similarity_score / source_file / content）
    [2] RAG 相关性阈值（低相似度 / 无关问题不强行返回案例）
    [3] Agent 输出结构含 findings 与 evidence（离线模板同样具备）
    [4] 报告区分五类信息，并展示 evidence 与案例来源
    [5] 前端「分析依据」数据结构
    [6] 端到端：演示数据 → 离线报告关键内容自检

用法：
    python scripts/phase10_smoke_test.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.diagnosis import build_offline_diagnosis, program_facts  # noqa: E402
from app.agents.rag_agent import RagAgent  # noqa: E402
from app.agents.recommendation import build_offline_recommendations  # noqa: E402
from app.graph.logging_utils import WorkflowLogger  # noqa: E402
from app.graph.prompts import COMMON_RULES  # noqa: E402
from app.graph.state import create_initial_state  # noqa: E402
from frontend import evidence_helpers as uie  # noqa: E402
from robotops import config  # noqa: E402
from robotops.console_io import setup_console_encoding  # noqa: E402
from robotops.pipeline import run_analysis  # noqa: E402
from robotops.rag import KnowledgeBaseRetriever, RagConfig  # noqa: E402
from robotops.rag.config import NO_RELEVANT_CASE_MESSAGE  # noqa: E402
from tests._helpers import cleanup_dir, make_temp_chroma_dir, release_chroma_client  # noqa: E402

REQUIRED_SOURCE_FIELDS = ("case_id", "title", "category", "similarity_score", "source_file", "content")
RELEVANT_QUERY = "B小区机器人故障率明显升高，同时维修次数增加。"


def _check(name: str, ok: bool, detail: str = "") -> int:
    mark = "PASS" if ok else "FAIL"
    text = "  [" + mark + "] " + name
    if detail:
        text += " —— " + detail
    print(text)
    return 1 if ok else 0


def main(argv: list[str] | None = None) -> int:
    setup_console_encoding()
    parser = argparse.ArgumentParser(description="RobotOps AI Phase 10 自检")
    parser.parse_args(argv)

    passed = total = 0
    work_dir = make_temp_chroma_dir()
    retriever = KnowledgeBaseRetriever(RagConfig(persist_dir=work_dir))
    temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    tmp = Path(temp.name)

    try:
        retriever.rebuild()
        analysis = run_analysis(config.DEMO_EXCEL_FILE, export=False, verbose=False)

        print("=" * 78)
        print("RobotOps AI —— Phase 10 自检（RAG 可追溯性 / Agent 输出可靠性 / 分析依据）")
        print("=" * 78)

        # -- [1] RAG 来源追踪 ----------------------------------------------
        print("\n[1] RAG 来源追踪")
        result = retriever.search(RELEVANT_QUERY)
        print("      命中案例：" + "、".join(case.case_id for case in result.cases))
        total += 1
        passed += _check("正常检索返回案例", bool(result.cases), str(len(result.cases)) + " 个")
        payload = result.cases[0].to_dict() if result.cases else {}
        missing = [field for field in REQUIRED_SOURCE_FIELDS if not payload.get(field)]
        total += 1
        passed += _check(
            "案例保留 6 个来源字段",
            not missing,
            "缺少：" + "、".join(missing) if missing else "case_id/标题/类别/相似度/来源文件/内容齐全",
        )
        total += 1
        passed += _check(
            "案例内容可追溯（正文而非仅标题）",
            "【问题】" in str(payload.get("content", "")),
            str(len(str(payload.get("content", "")))) + " 字",
        )

        # -- [2] 相关性阈值 ------------------------------------------------
        print("\n[2] RAG 相关性阈值")
        strict = retriever.search(RELEVANT_QUERY, min_similarity=0.99)
        total += 1
        passed += _check(
            "低于阈值不强行返回案例",
            not strict.cases and strict.to_dict()["message"] == NO_RELEVANT_CASE_MESSAGE,
        )
        irrelevant = retriever.search("今天天气怎么样？")
        total += 1
        passed += _check(
            "无关问题返回统一提示",
            not irrelevant.cases and "当前知识库未找到高相关历史案例" in irrelevant.to_dict()["message"],
        )

        # -- [3] Agent 输出结构（findings + evidence）----------------------
        print("\n[3] Agent 输出结构与分析依据")
        state = dict(create_initial_state(data_path=str(config.DEMO_EXCEL_FILE)))
        state["analysis_result"] = analysis
        logger = WorkflowLogger(verbose=False, echo=False, log_file=tmp / "phase10.log")
        rag_update = RagAgent(logger=logger, retriever=retriever).execute(state)
        state.update(rag_update)
        state.update(
            {
                "metrics": {"values": analysis.metrics.as_dict(), "units": {}},
                "anomaly_summary": [
                    {
                        "anomaly_type": str(row["异常类型"]),
                        "hits": int(row["命中记录数"]),
                        "condition": str(row["判定条件"]),
                    }
                    for _, row in analysis.anomaly_summary.iterrows()
                ],
                "anomaly_count": analysis.anomaly_count,
                "has_anomalies": analysis.anomaly_count > 0,
            }
        )
        state["trend_analysis"] = {"highlights": []}
        state["project_comparison"] = {"highlights": []}

        diagnosis = build_offline_diagnosis(state, note="Phase 10 自检")
        total += 1
        passed += _check("诊断输出包含 findings", bool(diagnosis.get("findings")))
        total += 1
        passed += _check(
            "每条可能原因都带 evidence",
            bool(diagnosis.get("possible_reasons"))
            and all(item.get("evidence") for item in diagnosis["possible_reasons"]),
        )
        total += 1
        passed += _check(
            "可能原因标注为推测",
            all("推测" in str(item.get("reason", "")) for item in diagnosis["possible_reasons"]),
        )
        recommendations = build_offline_recommendations(state, note="Phase 10 自检")
        total += 1
        passed += _check(
            "每条建议都带 evidence",
            bool(recommendations.get("items"))
            and all(item.get("evidence") for item in recommendations["items"]),
        )
        facts = program_facts(state)
        total += 1
        passed += _check(
            "程序事实按来源分组",
            all(facts.get(key) is not None for key in ("current", "program", "trend", "comparison", "cases")),
        )

        # -- [4] 提示词规则与报告区分 --------------------------------------
        print("\n[4] 禁止虚构规则与报告信息类型区分")
        total += 1
        passed += _check(
            "提示词包含 Phase 10 六条规则要点",
            all(
                keyword in COMMON_RULES
                for keyword in (
                    "不得虚构数据",
                    "不得虚构历史案例",
                    "不得把历史案例中的数据当成当前项目数据",
                    "当前数据不足以判断",
                    "可能原因（推测）",
                    "建议必须与输入中的问题一一相关",
                )
            ),
        )

        from app.agents.report import ReportAgent

        report_state = dict(state)
        report_state["use_llm"] = False
        report_state["diagnosis"] = diagnosis
        report_state["recommendations"] = recommendations
        report_state["raw_data_summary"] = {"数据源": config.DEMO_EXCEL_FILE.name}
        report_state["abnormal_projects"] = [
            {"project": str(row["项目名称"]), "anomaly_records": int(row["异常记录数"]),
             "affected_robots": int(row["涉及机器人数"]), "main_anomaly_types": str(row["主要异常类型"])}
            for _, row in analysis.anomaly_project_summary.iterrows()
        ]
        report_state["retrieval"] = rag_update["retrieval"]
        report = ReportAgent(logger=logger).execute(report_state)["final_report"]
        total += 1
        passed += _check(
            "报告区分五类信息",
            all(
                keyword in report
                for keyword in ("程序计算结果", "分析依据（程序计算）", "可能原因（AI 推测，非事实）",
                                "历史相似案例（仅供参考）", "优化建议")
            ),
        )
        total += 1
        passed += _check(
            "报告展示案例来源",
            "来源文件" in report and "相似度" in report,
        )

        # -- [5] 前端数据结构 ----------------------------------------------
        print("\n[5] 前端「分析依据」数据结构")
        panel = uie.evidence_panel_data(report_state)
        total += 1
        passed += _check("面板数据可用", bool(panel["available"]))
        total += 1
        passed += _check(
            "可能原因列标注为推测",
            "可能原因（推测）" in list(panel["reasons_frame"].columns)
            and "推测" in panel["disclaimer"],
        )
        total += 1
        passed += _check(
            "案例来源表包含 6 列",
            {"案例编号", "案例标题", "类别", "相似度", "来源文件", "内容摘要"}
            <= set(uie.case_source_frame(report_state).columns),
        )

        # -- [6] 端到端 ----------------------------------------------------
        print("\n[6] 端到端（演示数据）")
        print("      报告长度：" + str(len(report)) + " 字符")
        total += 1
        passed += _check(
            "报告包含推测声明",
            "均为 AI 推测" in report or "可能原因（推测）" in report,
        )
        total += 1
        passed += _check(
            "报告包含当前数据事实",
            "当前故障率" in report or "## 二、关键指标" in report,
        )
    finally:
        release_chroma_client(retriever)
        cleanup_dir(work_dir)
        temp.cleanup()

    print("\n" + "=" * 78)
    print("自检结果：" + str(passed) + "/" + str(total) + " 项通过")
    print("=" * 78)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

