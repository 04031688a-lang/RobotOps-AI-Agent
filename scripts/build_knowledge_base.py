"""RobotOps AI —— Phase 3 知识库（ChromaDB 向量库）管理脚本。

子命令：
    rebuild   清空并重建索引（知识库全量初始化）
    sync      增量同步（新增案例写入、已删除案例清理）
    status    查看索引状态（集合、案例数、块数、Embedding 方案、是否需要重建）
    list      列出知识库中的案例
    search    输入一个运营问题，输出 Top-K 相关历史案例
    queries   用演示数据的异常描述自动生成查询并检索（对应 Agent 的实际流程）
    demo      离线自检：验证「相关问题能命中」与「无关问题不误命中」

示例：
    python scripts/build_knowledge_base.py rebuild
    python scripts/build_knowledge_base.py search "B小区机器人故障率明显升高，同时维修次数增加。"
    python scripts/build_knowledge_base.py demo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotops.console_io import setup_console_encoding  # noqa: E402
from robotops.rag import (  # noqa: E402
    KnowledgeBaseRetriever,
    RagConfig,
    RagError,
    build_anomaly_queries,
    describe_embedding_plan,
)

RELEVANT_SAMPLES: tuple[str, ...] = (
    "B小区机器人故障率明显升高，同时维修次数增加。",
    "机器人运行率下降，设备长时间停在充电位。",
    "用户满意度下降，投诉说地面清洁不干净。",
    "运营成本偏高，节降率下降。",
    "设备闲置，任务分配不均。",
    "巡检不到位，出现漏检点位。",
)

IRRELEVANT_SAMPLES: tuple[str, ...] = (
    "今天天气怎么样？",
    "帮我写一首诗",
    "机器人多少钱一台",
    "Python 怎么读取 Excel 文件",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_knowledge_base.py",
        description="RobotOps AI Phase 3 知识库（ChromaDB）管理",
    )
    parser.add_argument(
        "command",
        choices=["rebuild", "sync", "status", "list", "search", "queries", "demo"],
        help="要执行的操作",
    )
    parser.add_argument("query", nargs="?", default=None, help="search 命令的问题文本")
    parser.add_argument("--top-k", type=int, default=None, help="检索 Top-K")
    parser.add_argument("--min-similarity", type=float, default=None, help="最低相似度阈值")
    parser.add_argument("--env-file", dest="env_file", default=None, help="指定 .env 路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_console_encoding()
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    config = RagConfig.from_env(env_file=args.env_file)

    try:
        retriever = KnowledgeBaseRetriever(config)
        return _dispatch(args, config, retriever)
    except BrokenPipeError:
        # 输出被管道提前关闭（例如 PowerShell 的 | Select-Object -First N）
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except Exception:  # pragma: no cover
            pass
        return 0
    except RagError as error:
        print(f"[知识库错误] {error.describe()}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace, config: RagConfig, retriever: KnowledgeBaseRetriever) -> int:
    if args.command == "rebuild":
        status = retriever.rebuild()
        print("知识库索引已重建：")
        print(status.describe())
        print(f"Embedding 方案：{describe_embedding_plan(config)}")
        return 0

    if args.command == "sync":
        summary = retriever.sync()
        print(
            f"增量同步完成：新增文本块 {summary['added_chunks']} 个，"
            f"清理文本块 {summary['deleted_chunks']} 个，"
            f"未变化案例 {summary['unchanged_cases']} 个"
        )
        print(retriever.status().describe())
        return 0

    if args.command == "status":
        print(retriever.status().describe())
        return 0

    if args.command == "list":
        documents = retriever.documents
        print(f"知识库案例清单（共 {len(documents)} 个）：")
        for index, document in enumerate(documents, start=1):
            print(
                f"  {index:>2}. [{document.case_id}] {document.title}"
                f"｜类型：{document.case_type}｜分类：{document.category}"
                f"｜关键词语数：{len(document.keywords)}"
            )
        return 0

    if args.command == "search":
        if not args.query:
            print("search 命令需要提供问题文本，例如：\n"
                  '  python scripts\\build_knowledge_base.py search "机器人故障率升高，维修次数增加"',
                  file=sys.stderr)
            return 2
        retriever.ensure_index()
        result = retriever.search(
            args.query, top_k=args.top_k, min_similarity=args.min_similarity
        )
        print(f"问题：{args.query}")
        print()
        print(result.describe())
        for index, case in enumerate(result.cases, start=1):
            print()
            print(f"--- {index}. [{case.case_id}] {case.title}（相关度 {case.similarity:.3f}）---")
            print(case.summary_text(budget=500))
        return 0

    if args.command == "queries":
        return _run_queries(config, retriever, top_k=args.top_k, min_similarity=args.min_similarity)

    return _run_demo(config, retriever, top_k=args.top_k, min_similarity=args.min_similarity)


def _run_queries(
    config: RagConfig,
    retriever: KnowledgeBaseRetriever,
    *,
    top_k: int | None,
    min_similarity: float | None,
) -> int:
    """用演示数据的异常识别结果生成查询并检索（Agent 的真实流程）。"""

    from robotops.pipeline import run_analysis

    print("步骤 1/3：运行 Phase 1 Pandas 分析（导出关闭）")
    result = run_analysis(export=False, verbose=False)
    print(
        f"          记录 {result.metrics.record_count} 条 / 异常 {result.anomaly_count} 条 / "
        f"项目 {result.metrics.project_count} 个"
    )

    print("步骤 2/3：生成异常描述（检索查询）")
    queries = build_anomaly_queries(result)
    for index, query in enumerate(queries, start=1):
        print(f"  {index}. 【{query.label}】{query.text}")

    print("步骤 3/3：RAG 检索 Top-K 历史案例")
    retrieval = retriever.retrieve_for_result(
        result, top_k=top_k, min_similarity=min_similarity
    )
    print()
    print(retrieval.describe(max_cases=config.max_cases))
    print()
    print("命中案例要点：")
    for index, case in enumerate(retrieval.cases, start=1):
        print()
        print(f"--- {index}. [{case.case_id}] {case.title}（相关度 {case.similarity:.3f}）---")
        print(f"    命中查询：{'；'.join(case.matched_queries)}")
        print(f"    命中片段：{'、'.join(case.matched_sections)}")
    return 0


def _run_demo(
    config: RagConfig,
    retriever: KnowledgeBaseRetriever,
    *,
    top_k: int | None,
    min_similarity: float | None,
) -> int:
    """离线自检：相关问题命中 + 无关问题不误命中。"""

    retriever.ensure_index()
    status = retriever.status()
    print("=" * 78)
    print("RobotOps AI Phase 3 —— RAG 离线自检（不调用任何大模型）")
    print("=" * 78)
    print(status.describe())
    print(f"检索参数：top_k={top_k or config.top_k}，min_similarity="
          f"{min_similarity if min_similarity is not None else config.min_similarity}")
    print()

    passed = 0
    total = 0

    print("[1/2] 相关问题应命中案例")
    for query in RELEVANT_SAMPLES:
        total += 1
        result = retriever.search(query, top_k=top_k, min_similarity=min_similarity)
        ok = bool(result.cases)
        passed += int(ok)
        top = result.cases[0] if result.cases else None
        detail = (
            f"[{top.case_id}] {top.title}（相关度 {top.similarity:.3f}）"
            if top
            else "未命中"
        )
        print(f"  [{'通过' if ok else '失败'}] {query} -> {detail}")

    print()
    print("[2/2] 无关问题不应命中案例")
    for query in IRRELEVANT_SAMPLES:
        total += 1
        result = retriever.search(query, top_k=top_k, min_similarity=min_similarity)
        ok = not result.cases
        passed += int(ok)
        detail = "未命中（符合预期）" if ok else f"误命中 {result.cases[0].case_id}"
        print(f"  [{'通过' if ok else '失败'}] {query} -> {detail}")

    print()
    print("=" * 78)
    print(f"自检结果：{passed}/{total} 项通过")
    print("=" * 78)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
