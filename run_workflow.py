"""RobotOps AI —— Phase 4 命令行入口：多 Agent 工作流（LangGraph）。

用法：
    python run_workflow.py                 # 完整工作流（需要 .env 中的 DeepSeek Key）
    python run_workflow.py --no-llm        # 离线模板模式：不调用大模型，零消耗
    python run_workflow.py --no-rag        # 关闭知识库检索
    python run_workflow.py --json          # 以 JSON 输出工作流状态
    python run_workflow.py --help

退出码：0 成功；2 输入/配置问题（数据、知识库、缺少 API Key）；3 大模型调用失败。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from app.graph.state import serializable_state, state_overview
from app.graph.workflow import RobotOpsWorkflow, export_workflow_outputs
from robotops import config as project_config
from robotops.console_io import setup_console_encoding
from robotops.exceptions import RobotOpsError
from robotops.llm.config import LLMConfig
from robotops.llm.errors import LLMError, MissingAPIKeyError
from robotops.rag import KnowledgeBaseRetriever, RagConfig, RagError

EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_LLM_ERROR = 3

WORKFLOW_LOG_FILE = "workflow.log"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_workflow.py",
        description="RobotOps AI Phase 4 —— LangGraph 多 Agent 机器人运营分析工作流",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python run_workflow.py\n"
            "  python run_workflow.py --no-llm --json\n"
            "  python run_workflow.py --data data/raw/robot_operation_demo.xlsx --no-rag\n"
        ),
    )
    parser.add_argument("-d", "--data", dest="data_path", default=None, help="数据文件路径，默认演示数据")
    parser.add_argument("-o", "--output", dest="output_dir", default=None, help="结果输出目录，默认 output/")
    parser.add_argument("--sheet", dest="sheet_name", default=None, help="Excel 工作表名称")
    parser.add_argument("--env-file", dest="env_file", default=None, help="指定 .env 文件路径")
    parser.add_argument("--no-llm", action="store_true", help="离线模板模式：不调用大模型")
    parser.add_argument("--no-rag", action="store_true", help="关闭知识库检索")
    parser.add_argument("--rag-top-k", type=int, default=None, help="RAG 检索 Top-K")
    parser.add_argument("--rag-min-similarity", type=float, default=None, help="RAG 最低相似度阈值")
    parser.add_argument("--rag-max-cases", type=int, default=None, help="最多注入提示词的案例数")
    parser.add_argument("--json", dest="as_json", action="store_true", help="以 JSON 输出工作流状态")
    parser.add_argument("--no-export", action="store_true", help="不导出报告与状态文件")
    parser.add_argument("-q", "--quiet", action="store_true", help="不打印工作流日志（只输出结果）")
    parser.add_argument(
        "--fault-rate-upper", type=float, default=project_config.DEFAULT_THRESHOLDS.fault_rate_upper,
        help="故障率上限阈值(%%)，默认 %(default)s",
    )
    parser.add_argument(
        "--satisfaction-lower", type=float, default=project_config.DEFAULT_THRESHOLDS.satisfaction_lower,
        help="满意度下限阈值(分)，默认 %(default)s",
    )
    parser.add_argument(
        "--uptime-rate-lower", type=float, default=project_config.DEFAULT_THRESHOLDS.uptime_rate_lower,
        help="运行率下限阈值(%%)，默认 %(default)s",
    )
    parser.add_argument(
        "--saving-rate-lower", type=float, default=project_config.DEFAULT_THRESHOLDS.saving_rate_lower,
        help="节降率下限阈值(%%)，默认 %(default)s",
    )
    return parser


def build_thresholds(args: argparse.Namespace) -> project_config.AnomalyThresholds:
    return project_config.AnomalyThresholds(
        fault_rate_upper=args.fault_rate_upper,
        satisfaction_lower=args.satisfaction_lower,
        uptime_rate_lower=args.uptime_rate_lower,
        saving_rate_lower=args.saving_rate_lower,
    )


def create_workflow(
    args: argparse.Namespace,
    *,
    client: object | None = None,
    retriever: KnowledgeBaseRetriever | None = None,
) -> RobotOpsWorkflow:
    """构造工作流（测试可替换本函数注入假客户端/假检索器）。"""

    return RobotOpsWorkflow(
        client=client,  # type: ignore[arg-type]
        retriever=retriever,
        verbose=not args.quiet,
        log_file=Path(args.output_dir or project_config.OUTPUT_DIR) / WORKFLOW_LOG_FILE,
        llm_log_level="WARNING",
    )


def prepare_retriever(args: argparse.Namespace) -> KnowledgeBaseRetriever | None:
    """准备知识库（失败时给出提示并降级为不使用知识库）。"""

    if args.no_rag:
        return None
    try:
        retriever = KnowledgeBaseRetriever(RagConfig.from_env(env_file=args.env_file))
        retriever.ensure_index()
        return retriever
    except RagError as error:
        print(f"[提示] 已跳过 RAG 知识库（{error.describe()}）", file=sys.stderr)
        print("[提示] 工作流将继续执行，但不会参考历史案例。", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    setup_console_encoding()
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])

    # 1) 大模型配置预检（离线模式跳过）
    if not args.no_llm:
        try:
            LLMConfig.from_env(env_file=args.env_file).validate(require_api_key=True)
        except MissingAPIKeyError as error:
            print(f"[配置错误] {error.describe()}", file=sys.stderr)
            print(
                "提示：可以先用 `python run_workflow.py --no-llm` 离线跑通工作流（不消耗 API）。",
                file=sys.stderr,
            )
            return EXIT_INPUT_ERROR
        except LLMError as error:
            print(f"[配置错误] {error.describe()}", file=sys.stderr)
            return EXIT_INPUT_ERROR

    # 2) 知识库准备（失败降级）
    retriever = prepare_retriever(args)
    use_knowledge = not args.no_rag and retriever is not None

    # 3) 运行工作流
    try:
        workflow = create_workflow(args, retriever=retriever)
        state = workflow.run(
            data_path=args.data_path,
            thresholds=build_thresholds(args),
            use_knowledge=use_knowledge,
            use_llm=not args.no_llm,
            rag_top_k=args.rag_top_k,
            rag_min_similarity=args.rag_min_similarity,
            rag_max_cases=args.rag_max_cases,
            export=not args.no_export,
            output_dir=args.output_dir,
        )
    except BrokenPipeError:
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except Exception:  # pragma: no cover
            pass
        return EXIT_OK
    except RobotOpsError as error:
        print(f"[执行失败] {_describe(error)}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except PermissionError as error:
        print(
            f"[执行失败] 文件被占用或无写入权限：{error}\n"
            f"提示：请先关闭正在打开该文件的 Excel 窗口后重试。",
            file=sys.stderr,
        )
        return EXIT_INPUT_ERROR

    # 4) 输出结果（管道被提前关闭时优雅退出）
    try:
        print()
        print("=" * 78)
        print("RobotOps AI Phase 4 —— 多 Agent 工作流结果")
        print("=" * 78)
        print(state_overview(state))
        print(f"工作流状态：{state.get('workflow_status')}")
        if state.get("abnormal_check_report"):
            print(f"异常判定：{state['abnormal_check_report']}")
        print()

        if args.as_json:
            print(json.dumps(serializable_state(state), ensure_ascii=False, indent=2, default=str))
        else:
            print(state.get("final_report") or "（未生成报告）")

        # 5) 导出（阶段内部已导出 Phase 1 文件，这里导出工作流产物）
        if not args.no_export:
            try:
                written = export_workflow_outputs(state, args.output_dir)
            except OSError as error:
                print(f"[提示] 工作流结果导出失败：{error}", file=sys.stderr)
                written = []
            for path in written:
                print(f"已生成：{path}")
    except BrokenPipeError:
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except Exception:  # pragma: no cover
            pass
        return EXIT_OK

    # 6) 退出码
    errors = state.get("errors") or []
    if any(record.get("is_llm_error") for record in errors):
        print()
        print("【错误汇总】以下阶段发生大模型调用失败，报告已降级生成：", file=sys.stderr)
        for record in errors:
            if record.get("is_llm_error"):
                print(
                    f"  - [{record.get('agent')}] {record.get('message')}"
                    + (f"\n    建议：{record['hint']}" if record.get("hint") else ""),
                    file=sys.stderr,
                )
        return EXIT_LLM_ERROR
    if errors or state.get("workflow_status") == "failed":
        print()
        print("【错误汇总】工作流出现以下异常：", file=sys.stderr)
        for record in errors:
            print(
                f"  - [{record.get('agent')}] {record.get('kind')}：{record.get('message')}"
                + (f"\n    建议：{record['hint']}" if record.get("hint") else ""),
                file=sys.stderr,
            )
        return EXIT_INPUT_ERROR
    return EXIT_OK


def _describe(error: Exception) -> str:
    describe = getattr(error, "describe", None)
    return str(describe()) if callable(describe) else str(error)


if __name__ == "__main__":
    raise SystemExit(main())
