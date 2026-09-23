"""RobotOps AI —— Phase 3 RAG 检索模块。

输入：异常描述（自然语言查询，例如「B小区机器人故障率明显升高，同时维修次数增加。」）
输出：Top-K 相关历史案例（含相似度、命中片段、案例完整要点）

检索流程：
1. **相关性门控**：查询中必须出现机器人运营领域词，否则直接判定「无相关案例」，
   避免无关问题被强行匹配到案例（例如「今天天气怎么样」）。
2. **分类优先检索**：按异常类型优先在对应知识库分类中检索，命中不足时自动退回全库检索。
3. **相似度阈值过滤**：低于 ``min_similarity`` 的命中被丢弃。
4. **按案例聚合**：同一案例的多个文本块合并，保留最高相似度、命中片段与完整要点。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..pipeline import AnalysisResult
from .config import DOMAIN_TERMS, RagConfig
from .document_loader import CaseChunk, CaseDocument, build_chunks, load_case_documents
from .query_builder import AnomalyQuery, DEFAULT_MAX_QUERIES, build_anomaly_queries
from .vector_store import IndexStatus, KnowledgeVectorStore, RetrievedChunk

#: 具体业务词 -> 扩展描述（用于短查询扩展，提升短问题与长案例的匹配度）
#: 只扩展“具体业务词”，不扩展「机器人 / 设备 / 任务」这类泛化词，
#: 这样「机器人多少钱一台」这类无关问题不会被扩展，仍会被阈值过滤掉。
TERM_EXPANSIONS: dict[str, str] = {
    "故障": "机器人重复故障 故障率升高 维修次数增加 部件损坏 根因排查",
    "维修": "维修频繁 备件更换 返修 维修质量 验收标准",
    "售后": "售后响应慢 到场超时 停机时间 服务时限",
    "运行率": "运行率下降 欠运行 停机时间 充电策略 排班冲突",
    "运行时长": "运行时长下降 有效作业时间 等待时间",
    "满意度": "用户满意度下降 投诉 清洁效果不达标 服务响应慢",
    "清洁": "清洁效果下降 地面残留 刷盘磨损 水量设置",
    "巡检": "巡检不到位 漏检 点位遗漏 巡检频次不足",
    "成本": "运营成本过高 成本结构 能耗偏高 人力配置",
    "节降": "节降率下降 成本节降 能耗上升 基线口径",
    "能耗": "能耗偏高 成本结构 充电时段 峰谷电价",
    "闲置": "设备闲置 任务分配不均 单机利用率 产能过剩",
    "调度": "任务调度异常 路径冲突 任务排队 交通管制",
    "备件": "备件库存不足 等待备件 采购周期 安全库存",
    "维保": "维护周期过长 保养延期 维保计划 计划冲突",
    "保养": "维护周期过长 超期保养 保养清单",
    "充电": "充电策略 机会充电 充电时段 回充失败",
    "电池": "电池老化 续航下降 中途停机 容量衰减",
    "噪音": "噪音投诉 夜间作业 敏感区域",
    "停机": "停机时间 运行率下降 待备件 售后响应",
    "投诉": "投诉响应慢 满意度下降 闭环回访",
    "清洁效果": "清洁效果下降 地面残留 用户投诉",
}

#: 短查询判定长度（字符数）
SHORT_QUERY_CHARS = 20


@dataclass
class RetrievedCase:
    """检索命中的一个历史案例（聚合后的结果）。"""

    case_id: str
    title: str
    case_type: str
    category: str
    similarity: float
    matched_sections: list[str] = field(default_factory=list)
    matched_queries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    source_file: str = ""
    data_nature: str = ""
    excerpt: str = ""
    sections: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "case_type": self.case_type,
            "category": self.category,
            "similarity": round(self.similarity, 4),
            "matched_sections": self.matched_sections,
            "matched_queries": self.matched_queries,
            "keywords": self.keywords,
            "source_file": self.source_file,
            "data_nature": self.data_nature,
        }

    def summary_text(self, *, budget: int = 700) -> str:
        lines = [
            f"案例编号：{self.case_id}",
            f"案例标题：{self.title}",
            f"案例类型：{self.case_type}",
        ]
        if self.keywords:
            lines.append(f"关键词：{'、'.join(self.keywords)}")
        if self.data_nature:
            lines.append(f"数据性质：{self.data_nature}")
        for name, body in self.sections.items():
            lines.append(f"【{name}】{body}")
        text = "\n".join(lines)
        if len(text) > budget:
            text = text[:budget].rstrip() + "…（已截断）"
        return text

    def to_prompt_entry(self, *, budget: int = 700) -> dict[str, Any]:
        """转成注入提示词的紧凑结构。"""

        payload = self.to_dict()
        payload["case_summary"] = self.summary_text(budget=budget)
        return payload


@dataclass
class RetrievalResult:
    """一次 RAG 检索的完整结果。"""

    queries: list[AnomalyQuery] = field(default_factory=list)
    cases: list[RetrievedCase] = field(default_factory=list)
    reason: str = ""
    notes: list[str] = field(default_factory=list)
    min_similarity: float = 0.0
    top_k: int = 0
    skipped_queries: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.cases

    @property
    def is_relevant(self) -> bool:
        return bool(self.cases)

    def to_prompt_entries(self, *, budget: int = 700) -> list[dict[str, Any]]:
        return [case.to_prompt_entry(budget=budget) for case in self.cases]

    def to_dict(self) -> dict[str, Any]:
        return {
            "queries": [query.to_dict() for query in self.queries],
            "skipped_queries": self.skipped_queries,
            "case_count": len(self.cases),
            "min_similarity": self.min_similarity,
            "top_k": self.top_k,
            "is_relevant": self.is_relevant,
            "reason": self.reason,
            "notes": self.notes,
            "cases": [case.to_dict() for case in self.cases],
        }

    def describe(self, *, max_cases: int = 5) -> str:
        lines = ["【RAG 检索结果】"]
        for query in self.queries:
            lines.append(f"  查询：{query.label} -> {query.text[:80]}…")
        if self.skipped_queries:
            lines.append(f"  跳过（与机器人运营无关）：{'；'.join(self.skipped_queries)}")
        if self.cases:
            lines.append(f"  命中案例 {len(self.cases)} 个（最低相似度 {self.min_similarity:g}）：")
            for index, case in enumerate(self.cases[:max_cases], start=1):
                lines.append(
                    f"    {index}. [{case.case_id}] {case.title}"
                    f"｜类型：{case.case_type}｜相关度：{case.similarity:.3f}"
                    f"｜命中：{'、'.join(case.matched_sections)}"
                )
        else:
            lines.append(f"  未命中足够相关的案例：{self.reason}")
        return "\n".join(lines)


class KnowledgeBaseRetriever:
    """知识库检索器：负责索引维护与 Top-K 案例检索。"""

    def __init__(
        self,
        config: RagConfig | None = None,
        *,
        store: KnowledgeVectorStore | None = None,
        documents: Sequence[CaseDocument] | None = None,
    ) -> None:
        self.config = config or RagConfig()
        self.config.validate()
        self.store = store or KnowledgeVectorStore(self.config)
        self._documents: list[CaseDocument] = list(documents) if documents else []
        self._chunks: list[CaseChunk] = []

    # -- 文档与索引 -------------------------------------------------------
    @property
    def documents(self) -> list[CaseDocument]:
        if not self._documents:
            self._documents = load_case_documents(self.config)
        return self._documents

    @property
    def chunks(self) -> list[CaseChunk]:
        if not self._chunks:
            self._chunks = build_chunks(self.documents, self.config)
        return self._chunks

    def case_index(self) -> dict[str, CaseDocument]:
        return {document.case_id: document for document in self.documents}

    def status(self) -> IndexStatus:
        return self.store.status()

    def ensure_index(self, *, force: bool = False) -> IndexStatus:
        """确保索引可用：缺失 / 为空 / 参数变化时自动重建。"""

        status = self.store.status()
        needs_build = force or not status.exists or status.chunk_count == 0 or status.stale
        if not needs_build:
            return status
        if not (force or self.config.auto_build):
            return status
        return self.rebuild()

    def rebuild(self) -> IndexStatus:
        """清空并重建索引（知识库重建）。"""

        return self.store.rebuild(self.chunks)

    def sync(self) -> dict[str, int]:
        """增量同步：写入新增案例，删除已从目录移除的案例。"""

        status = self.store.status()
        if not status.exists or status.chunk_count == 0 or status.stale:
            self.rebuild()
            return {"added_chunks": len(self.chunks), "deleted_chunks": 0, "unchanged_cases": 0}

        collection = self.store._get_collection(create=False)
        existing_metadatas = collection.get(include=["metadatas"]).get("metadatas") or []
        indexed_cases = {str(item.get("case_id")) for item in existing_metadatas if item}
        local_cases = set(self.case_index())

        to_add = [chunk for chunk in self.chunks if chunk.case_id not in indexed_cases]
        deleted = 0
        for case_id in indexed_cases - local_cases:
            deleted += self.store.delete_case(case_id)
        if to_add:
            self.store.upsert_chunks(to_add)
        return {
            "added_chunks": len(to_add),
            "deleted_chunks": deleted,
            "unchanged_cases": len(local_cases & indexed_cases),
        }

    # -- 检索 -------------------------------------------------------------
    def retrieve_chunks(
        self,
        query_text: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        return self.store.query(
            query_text,
            top_k=top_k or self.config.top_k,
            where=where,
            min_similarity=self.config.min_similarity if min_similarity is None else min_similarity,
        )

    def is_domain_query(self, query_text: str) -> tuple[bool, str]:
        """相关性门控：判断问题是否属于机器人运营领域。"""

        hits = [term for term in DOMAIN_TERMS if term in (query_text or "")]
        if hits:
            return True, f"命中领域词：{'、'.join(hits[:6])}"
        return False, "查询中未包含机器人运营领域词（如故障、运行率、满意度、成本等）"

    def search(
        self,
        query_text: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
        max_cases: int | None = None,
    ) -> RetrievalResult:
        """单查询检索（供命令行 search 子命令与测试使用）。"""

        query = AnomalyQuery(label="用户查询", text=query_text)
        return self.retrieve_cases(
            [query], top_k=top_k, min_similarity=min_similarity, max_cases=max_cases
        )

    def retrieve_cases(
        self,
        queries: Sequence[AnomalyQuery],
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
        max_cases: int | None = None,
    ) -> RetrievalResult:
        """按多条查询检索并聚合为 Top-K 案例。"""

        limit_k = int(top_k or self.config.top_k)
        limit_cases = int(max_cases or self.config.max_cases)
        threshold = self.config.min_similarity if min_similarity is None else float(min_similarity)

        result = RetrievalResult(queries=list(queries), min_similarity=threshold, top_k=limit_k)
        aggregated: dict[str, RetrievedCase] = {}
        cases = self.case_index()

        for query in queries:
            relevant, note = self.is_domain_query(query.text)
            if not relevant:
                result.skipped_queries.append(query.label)
                result.notes.append(f"{query.label}：{note}")
                continue

            search_text, expanded_terms = expand_query(query.text)
            if expanded_terms:
                result.notes.append(
                    f"{query.label}：已应用查询扩展（{'、'.join(expanded_terms)}）"
                )
            chunks = self._search_with_category_fallback(
                query, limit_k, threshold, search_text=search_text
            )
            if not chunks:
                result.notes.append(f"{query.label}：未找到相似度 >= {threshold:g} 的案例片段")
                continue

            for chunk in chunks:
                _merge_case(aggregated, chunk, query, cases, self.config.case_text_budget)

        result.cases = sorted(
            aggregated.values(), key=lambda item: item.similarity, reverse=True
        )[:limit_cases]

        if not result.cases:
            result.reason = (
                "未检索到足够相关的历史案例"
                f"（最低相似度阈值 {threshold:g}；"
                f"可调低 ROBOTOPS_RAG_MIN_SIMILARITY 或补充知识库文档后重试）"
            )
        return result

    def retrieve_for_result(
        self,
        result: AnalysisResult,
        *,
        max_queries: int = DEFAULT_MAX_QUERIES,
        top_k: int | None = None,
        min_similarity: float | None = None,
        max_cases: int | None = None,
    ) -> RetrievalResult:
        """从 Phase 1 分析结果生成异常描述并检索历史案例。"""

        queries = build_anomaly_queries(result, max_queries=max_queries)
        return self.retrieve_cases(
            queries, top_k=top_k, min_similarity=min_similarity, max_cases=max_cases
        )

    # -- 内部 -------------------------------------------------------------
    def _search_with_category_fallback(
        self,
        query: AnomalyQuery,
        top_k: int,
        threshold: float,
        *,
        search_text: str | None = None,
    ) -> list[RetrievedChunk]:
        text = search_text or query.text
        categories = tuple(query.preferred_categories or ())
        if categories:
            where: dict[str, Any] = (
                {"category": categories[0]}
                if len(categories) == 1
                else {"category": {"$in": list(categories)}}
            )
            chunks = self.retrieve_chunks(
                text, top_k=top_k, min_similarity=threshold, where=where
            )
            if chunks:
                return chunks
        return self.retrieve_chunks(text, top_k=top_k, min_similarity=threshold, where=None)


def expand_query(
    text: str,
    *,
    short_query_chars: int = SHORT_QUERY_CHARS,
    max_terms: int = 3,
) -> tuple[str, list[str]]:
    """短查询扩展：把业务词补成完整表述，避免短问题被长案例“稀释”。

    Returns:
        (用于检索的文本, 本次扩展使用的业务词列表)
    """

    source = text or ""
    matched = [key for key in TERM_EXPANSIONS if key in source]
    if not matched:
        return source, []

    is_short = len(source.strip()) < short_query_chars
    if not is_short and len(matched) >= 2:
        return source, []

    terms = matched[:max_terms]
    extra = " ".join(TERM_EXPANSIONS[term] for term in terms)
    return f"{source} {extra}", terms


def _merge_case(
    aggregated: dict[str, RetrievedCase],
    chunk: RetrievedChunk,
    query: AnomalyQuery,
    cases: dict[str, CaseDocument],
    budget: int,
) -> None:
    existing = aggregated.get(chunk.case_id)
    if existing is None:
        document = cases.get(chunk.case_id)
        existing = RetrievedCase(
            case_id=chunk.case_id,
            title=chunk.title,
            case_type=chunk.case_type,
            category=chunk.category,
            similarity=chunk.similarity,
            keywords=chunk.keywords,
            source_file=chunk.source_file,
            data_nature=chunk.data_nature,
            sections=dict(document.sections) if document else {},
            excerpt=chunk.text,
        )
        aggregated[chunk.case_id] = existing
    else:
        existing.similarity = max(existing.similarity, chunk.similarity)

    if chunk.section not in existing.matched_sections:
        existing.matched_sections.append(chunk.section)
    if query.label not in existing.matched_queries:
        existing.matched_queries.append(query.label)
    if not existing.excerpt:
        existing.excerpt = chunk.text
