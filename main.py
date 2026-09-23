"""RobotOps AI —— Phase 2 命令行入口：机器人运营数据分析 Agent。

流程（原始 Excel 不会直接交给大模型）：

    1. 读取 Excel / CSV
    2. Phase 1 Pandas 模块计算核心指标并识别异常
    3. 把结构化分析结果构造成 Agent 输入载荷
    4. 调用 DeepSeek 生成结构化分析结论
    5. 输出 AI 分析结果（控制台 + JSON / Markdown 文件）

用法：
    python main.py                 # 完整流程（需要 .env 中配置 DEEPSEEK_API_KEY）
    python main.py --dry-run       # 只打印将发送给模型的结构化载荷，不调用 API
    python main.py --help

说明：Phase 1 的入口 ``run_analysis.py`` 保持不变，本文件是 Phase 2 新增入口。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from robotops import config as project_config
from robotops.agent.robot_ops_agent import (
    AI_OUTPUT_BASENAME,
    AI_PAYLOAD_FILE,
    AI_RAG_FILE,
    AgentRunResult,
    RobotOpsAnalysisAgent,
)
from robotops.console_io import setup_console_encoding
from robotops.exceptions import RobotOpsError
from robotops.llm.config import LLMConfig, describe_supported_env
from robotops.llm.config import ENV_LOG_LEVEL
from robotops.llm.errors import LLMError, MissingAPIKeyError
from robotops.llm.env_loader import get_env
from robotops.pipeline import run_analysis
from robotops.rag import (
    KnowledgeBaseRetriever,
    RagConfig,
    RagError,
    describe_embedding_plan,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_LLM_ERROR = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="RobotOps AI Phase 2 —— DeepSeek 机器人运营数据分析 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python main.py\n"
            "  python main.py --dry-run\n"
            "  python main.py --data data/raw/robot_operation_demo.xlsx --show-prompt\n"
            "\n"
            "需要在项目根目录的 .env 中配置：\n  "
            + "\n  ".join(describe_supported_env())
            + "\n"
        ),
    )
    parser.add_argument("-d", "--data", dest="data_path", default=None, help="数据文件路径，默认使用演示数据")
    parser.add_argument("-o", "--output", dest="output_dir", default=None, help="结果输出目录，默认 output/")
    parser.add_argument("--sheet", dest="sheet_name", default=None, help="Excel 工作表名称")
    parser.add_argument("--encoding", dest="encoding", default=None, help="CSV 文件编码，如 gbk")
    parser.add_argument("--env-file", dest="env_file", default=None, help="指定 .env 文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只生成并打印结构化载荷，不调用 DeepSeek API")
    parser.add_argument("--show-prompt", action="store_true", help="打印实际发送的 Prompt（system + user）")
    parser.add_argument("--json", dest="as_json", action="store_true", help="以 JSON 形式输出 AI 分析结果")
    parser.add_argument("--no-export", action="store_true", help="不导出任何结果文件")
    parser.add_argument("-q", "--quiet", action="store_true", help="隐藏过程日志，只输出最终结果")
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="禁用 Phase 3 知识库检索（退化为纯 Phase 2 流程）",
    )
    parser.add_argument("--rag-top-k", type=int, default=None, help="RAG 检索 Top-K（默认取配置值）")
    parser.add_argument(
        "--rag-min-similarity", type=float, default=None, help="RAG 最低相似度阈值（默认取配置值）"
    )
    parser.add_argument("--rag-max-cases", type=int, default=None, help="最多注入提示词的历史案例数")
    parser.add_argument("--rebuild-knowledge", action="store_true", help="重建知识库索引后退出")
    parser.add_argument("--rag-status", action="store_true", help="查看知识库索引状态后退出")
    parser.add_argument("--search", dest="search_query", default=None, help="仅检索知识库 Top-K 案例（不调用大模型）")
    parser.add_argument(
        "--no-rag-export",
        action="store_true",
        help="导出结果时不生成 ai_rag_retrieval.json",
    )
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


def create_agent(args: argparse.Namespace, *, dry_run: bool = False) -> RobotOpsAnalysisAgent:
    """构造 Agent（在测试中可替换本函数以注入假客户端）。"""

    overrides: dict[str, object] = {}
    # --quiet 且用户未显式配置日志级别时，只保留 WARNING 以上日志，保证输出干净
    if getattr(args, "quiet", False) and not get_env(ENV_LOG_LEVEL):
        overrides["log_level"] = "WARNING"

    llm_config = LLMConfig.from_env(env_file=args.env_file, **overrides)
    if not dry_run:
        llm_config.validate(require_api_key=True)
    retriever = getattr(args, "resolved_retriever", None)
    return RobotOpsAnalysisAgent(
        config=llm_config,
        env_file=args.env_file,
        retriever=retriever,
        use_knowledge=bool(retriever),
        rag_top_k=getattr(args, "rag_top_k", None),
        rag_min_similarity=getattr(args, "rag_min_similarity", None),
        rag_max_cases=getattr(args, "rag_max_cases", None),
    )


def prepare_retriever(args: argparse.Namespace) -> KnowledgeBaseRetriever | None:
    """准备知识库检索器：失败时给出提示并降级为纯 Phase 2 流程。"""

    if getattr(args, "no_rag", False):
        return None

    verbose = not getattr(args, "quiet", False)
    try:
        rag_config = RagConfig.from_env(env_file=args.env_file)
        retriever = KnowledgeBaseRetriever(rag_config)
        status = retriever.ensure_index()
        if verbose:
            print(
                f"[Phase 3] 知识库就绪：{rag_config.knowledge_dir}"
                f"（案例 {status.case_count} 个 / 文本块 {status.chunk_count} 个）"
            )
            print(
                f"          Embedding：{describe_embedding_plan(rag_config)['name']}"
                f"（dim={rag_config.embedding_dim}）"
            )
        return retriever
    except RagError as error:
        print(f"[提示] 已跳过 RAG 知识库（{error.describe()}）", file=sys.stderr)
        print("[提示] 将按纯 Phase 2 流程继续分析。", file=sys.stderr)
        return None


def rebuild_knowledge(args: argparse.Namespace) -> int:
    """重建知识库索引（不调用大模型）。"""

    try:
        rag_config = RagConfig.from_env(env_file=args.env_file)
        retriever = KnowledgeBaseRetriever(rag_config)
        status = retriever.rebuild()
    except RagError as error:
        print(f"[知识库错误] {error.describe()}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    print("知识库索引已重建：")
    print(status.describe())
    print(f"Embedding 方案：{describe_embedding_plan(rag_config)}")
    return EXIT_OK


def show_rag_status(args: argparse.Namespace) -> int:
    """查看知识库索引状态（不调用大模型）。"""

    try:
        rag_config = RagConfig.from_env(env_file=args.env_file)
        retriever = KnowledgeBaseRetriever(rag_config)
        status = retriever.status()
    except RagError as error:
        print(f"[知识库错误] {error.describe()}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    print("知识库索引状态：")
    print(status.describe())
    if not status.exists or status.chunk_count == 0:
        print("提示：索引为空，执行 python main.py --rebuild-knowledge 初始化。")
    return EXIT_OK


def search_knowledge(args: argparse.Namespace) -> int:
    """仅检索知识库（不调用大模型），输出 Top-K 相关历史案例。"""

    try:
        rag_config = RagConfig.from_env(env_file=args.env_file)
        retriever = KnowledgeBaseRetriever(rag_config)
        retriever.ensure_index()
        result = retriever.search(
            args.search_query,
            top_k=args.rag_top_k,
            min_similarity=args.rag_min_similarity,
            max_cases=args.rag_max_cases,
        )
    except RagError as error:
        print(f"[知识库错误] {error.describe()}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    print("=" * 78)
    print("RobotOps AI Phase 3 —— 知识库检索（不调用大模型）")
    print("=" * 78)
    print(f"问题：{args.search_query}")
    print()
    print(result.describe(max_cases=args.rag_max_cases or 5))
    if result.cases:
        print()
        print("命中案例要点：")
        for index, case in enumerate(result.cases, start=1):
            print()
            print(f"--- {index}. [{case.case_id}] {case.title}（相关度 {case.similarity:.3f}）---")
            print(case.summary_text(budget=500))
    print("=" * 78)
    return EXIT_OK


def run(args: argparse.Namespace) -> AgentRunResult:
    """执行 Phase 1 分析 + Phase 2 Agent，返回运行结果。"""

    verbose = not args.quiet
    retriever = prepare_retriever(args)
    args.resolved_retriever = retriever

    if verbose:
        print("[Phase 1] 使用 Pandas 分析数据（读取 → 清洗 → 指标 → 异常）...")
    analysis_result = run_analysis(
        data_path=args.data_path,
        output_dir=args.output_dir,
        sheet_name=args.sheet_name,
        encoding=args.encoding,
        thresholds=project_config.AnomalyThresholds(
            fault_rate_upper=args.fault_rate_upper,
            satisfaction_lower=args.satisfaction_lower,
            uptime_rate_lower=args.uptime_rate_lower,
            saving_rate_lower=args.saving_rate_lower,
        ),
        export=not args.no_export,
        verbose=verbose,
    )
    metrics = analysis_result.metrics_dict()
    if verbose:
        print(
            f"          结构化结果：项目 {metrics['项目数量']} 个 / 机器人 {metrics['机器人数量']} 台 / "
            f"异常 {analysis_result.anomaly_count} 条"
        )
        print("[Phase 2] 构造 Agent 输入载荷并调用 DeepSeek ...")

    agent = create_agent(args, dry_run=args.dry_run)
    result = agent.dry_run(analysis_result) if args.dry_run else agent.analyze(analysis_result)

    if args.show_prompt:
        print()
        print("=" * 78)
        print("发送给 DeepSeek 的 Prompt")
        print("=" * 78)
        for message in result.messages:
            print(f"--- role: {message['role']} ---")
            print(message["content"])
            print()

    if verbose and result.retrieval is not None:
        print()
        print(result.retrieval.describe())

    if not args.no_export:
        export_ai_outputs(
            result,
            args.output_dir,
            verbose=verbose,
            write_rag=not getattr(args, "no_rag_export", False),
        )
    return result


def export_ai_outputs(
    result: AgentRunResult,
    output_dir: str | Path | None,
    *,
    verbose: bool = True,
    write_rag: bool = True,
) -> list[Path]:
    """导出 AI 分析结果与输入载荷。"""

    target_dir = Path(output_dir) if output_dir is not None else project_config.OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    payload_path = target_dir / AI_PAYLOAD_FILE
    payload_path.write_text(result.payload_json, encoding=project_config.TEXT_ENCODING)
    written.append(payload_path)

    if write_rag and result.retrieval is not None:
        rag_path = target_dir / AI_RAG_FILE
        rag_path.write_text(
            json.dumps(result.retrieval.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding=project_config.TEXT_ENCODING,
        )
        written.append(rag_path)

    if result.analysis is not None:
        json_path = target_dir / f"{AI_OUTPUT_BASENAME}.json"
        json_path.write_text(
            json.dumps(result.analysis.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding=project_config.TEXT_ENCODING,
        )
        written.append(json_path)

        markdown_path = target_dir / f"{AI_OUTPUT_BASENAME}.md"
        markdown_path.write_text(result.analysis.to_markdown(), encoding=project_config.TEXT_ENCODING)
        written.append(markdown_path)

    if verbose:
        for path in written:
            print(f"      已生成：{path}")
    return written


def print_result(result: AgentRunResult, *, as_json: bool) -> None:
    """输出最终结果。"""

    if result.dry_run:
        print()
        print("=" * 78)
        print("干跑模式（--dry-run）：以下是将发送给 DeepSeek 的结构化载荷，未调用 API")
        print("=" * 78)
        print(result.payload_json)
        print("=" * 78)
        metrics = result.payload.get("core_metrics", {})
        print(
            f"核心指标：项目 {metrics.get('project_count')} 个 / 机器人 {metrics.get('robot_count')} 台 / "
            f"故障率 {metrics.get('avg_fault_rate')}% / 平均满意度 {metrics.get('avg_satisfaction')} 分 / "
            f"总成本 {metrics.get('total_cost')} 元 / 平均节降率 {metrics.get('avg_cost_reduction_rate')}%"
        )
        print(f"异常项目 {len(result.payload.get('abnormal_projects', []))} 个 / "
              f"异常机器人 {len(result.payload.get('abnormal_robots', []))} 台")
        if result.retrieval is not None:
            print()
            print(result.retrieval.describe())
            if result.analysis is None:
                print("（干跑模式未调用 DeepSeek，以上为将随载荷一起发送的历史案例）")
        return

    assert result.analysis is not None
    if as_json:
        print(json.dumps(result.analysis.to_dict(), ensure_ascii=False, indent=2, default=str))
    else:
        print()
        print(result.analysis.to_console_text())


def main(argv: list[str] | None = None) -> int:
    setup_console_encoding()
    args = build_parser().parse_args(argv)

    try:
        if args.rebuild_knowledge:
            return rebuild_knowledge(args)
        if args.rag_status:
            return show_rag_status(args)
        if args.search_query:
            return search_knowledge(args)
        result = run(args)
    except BrokenPipeError:
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except Exception:  # pragma: no cover
            pass
        return EXIT_OK
    except MissingAPIKeyError as exc:
        print(f"[配置错误] {exc.describe()}", file=sys.stderr)
        print(
            "提示：可以先执行 `python main.py --dry-run` 查看将发送给模型的结构化数据（无需 API Key）。",
            file=sys.stderr,
        )
        return EXIT_INPUT_ERROR
    except LLMError as exc:
        print(f"[AI 调用失败] {exc.describe()}", file=sys.stderr)
        return EXIT_LLM_ERROR
    except RobotOpsError as exc:
        print(f"[执行失败] {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except PermissionError as exc:
        print(
            f"[执行失败] 文件被占用或无写入权限：{exc}\n"
            f"提示：请先关闭正在打开该文件的 Excel 窗口后重试。",
            file=sys.stderr,
        )
        return EXIT_INPUT_ERROR

    try:
        print_result(result, as_json=args.as_json)
    except UnicodeEncodeError as exc:
        print(f"[提示] 当前终端无法显示部分字符：{exc}", file=sys.stderr)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
