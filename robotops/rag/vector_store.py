"""RobotOps AI —— Phase 3 ChromaDB 本地向量库封装。

能力：
- 创建 / 获取本地持久化集合（默认 ``data/chroma/``）；
- 知识库初始化与重建（清空 + 写入）；
- 增量写入（upsert）与按案例删除；
- 相似度检索（返回带相似度与元数据的文本块）；
- 索引状态查询与「索引指纹」校验（方案或切分参数变化时提示重建）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import RagConfig
from .document_loader import CaseChunk
from .embedding import LocalHashingEmbeddingFunction, build_embedding_function
from .errors import IndexVersionMismatchError, KnowledgeBaseError, RagDependencyError

SPACE_METADATA_KEY = "hnsw:space"
DEFAULT_SPACE = "cosine"


@dataclass(frozen=True)
class RetrievedChunk:
    """检索命中的一个文本块。"""

    chunk_id: str
    text: str
    case_id: str
    title: str
    case_type: str
    category: str
    section: str
    source_file: str
    data_nature: str
    keywords: list[str] = field(default_factory=list)
    robot_types: list[str] = field(default_factory=list)
    similarity: float = 0.0
    distance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "case_id": self.case_id,
            "title": self.title,
            "case_type": self.case_type,
            "category": self.category,
            "section": self.section,
            "similarity": round(self.similarity, 4),
            "keywords": self.keywords,
            "source_file": self.source_file,
        }


@dataclass
class IndexStatus:
    """向量库状态。"""

    collection_name: str
    chunk_count: int
    case_count: int
    persist_dir: str
    knowledge_dir: str
    embedding: dict[str, Any] = field(default_factory=dict)
    fingerprint: dict[str, Any] = field(default_factory=dict)
    stored_fingerprint: dict[str, Any] = field(default_factory=dict)
    last_built_at: str | None = None
    stale: bool = False
    exists: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "chunk_count": self.chunk_count,
            "case_count": self.case_count,
            "persist_dir": self.persist_dir,
            "knowledge_dir": self.knowledge_dir,
            "embedding": self.embedding,
            "last_built_at": self.last_built_at,
            "stale": self.stale,
            "exists": self.exists,
        }

    def describe(self) -> str:
        lines = [
            f"集合名称    : {self.collection_name}",
            f"持久化目录  : {self.persist_dir}",
            f"知识库目录  : {self.knowledge_dir}",
            f"案例数/块数 : {self.case_count} / {self.chunk_count}",
            f"Embedding   : {self.embedding.get('name')}（dim={self.embedding.get('dim')}，space=cosine）",
            f"构建时间    : {self.last_built_at or '（未构建）'}",
        ]
        if self.stale:
            lines.append("索引状态    : 需要重建（Embedding 方案或切分参数已变化）")
        return "\n".join(lines)


def import_chromadb() -> Any:
    """导入 chromadb，未安装时给出明确安装提示。"""

    try:
        import chromadb  # type: ignore
    except ImportError as error:  # pragma: no cover - 依赖环境
        raise RagDependencyError(
            "未安装 ChromaDB，无法使用 RAG 知识库功能"
        ) from error
    return chromadb


class KnowledgeVectorStore:
    """ChromaDB 本地向量库封装。"""

    def __init__(
        self,
        config: RagConfig | None = None,
        *,
        embedding_function: LocalHashingEmbeddingFunction | None = None,
        client: Any = None,
    ) -> None:
        self.config = config or RagConfig()
        self.config.validate()
        self._embedding = embedding_function or build_embedding_function(self.config)
        self._client = client

    # -- 基础对象 ---------------------------------------------------------
    @property
    def embedding(self) -> LocalHashingEmbeddingFunction:
        return self._embedding

    @property
    def client(self) -> Any:
        if self._client is None:
            chromadb = import_chromadb()
            self.config.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.config.persist_dir))
        return self._client

    def collection_names(self) -> list[str]:
        collections = self.client.list_collections()
        names: list[str] = []
        for item in collections:
            names.append(item if isinstance(item, str) else getattr(item, "name", str(item)))
        return names

    def collection_exists(self) -> bool:
        return self.config.collection_name in self.collection_names()

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        return int(self._get_collection().count())

    def _get_collection(self, *, create: bool = True) -> Any:
        if not create and not self.collection_exists():
            raise KnowledgeBaseError(
                f"向量库集合不存在：{self.config.collection_name}，请先构建知识库索引"
            )
        if create and not self.collection_exists():
            return self.client.create_collection(
                name=self.config.collection_name,
                embedding_function=self._embedding,
                metadata=self._collection_metadata(),
            )
        return self.client.get_collection(
            name=self.config.collection_name, embedding_function=self._embedding
        )

    def _collection_metadata(self) -> dict[str, Any]:
        fingerprint = self.config.index_fingerprint()
        metadata: dict[str, Any] = {
            SPACE_METADATA_KEY: DEFAULT_SPACE,
            "embedding_name": self._embedding.name(),
            "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        metadata.update({key: value for key, value in fingerprint.items()})
        return metadata

    # -- 写入 -------------------------------------------------------------
    def rebuild(self, chunks: Sequence[CaseChunk]) -> IndexStatus:
        """清空并重建索引。"""

        if not chunks:
            raise KnowledgeBaseError("没有可写入的文本块，请检查知识库文档内容")
        if self.collection_exists():
            self.client.delete_collection(self.config.collection_name)
        collection = self._get_collection(create=True)
        self._upsert(collection, chunks)
        return self.status()

    def upsert_chunks(self, chunks: Sequence[CaseChunk]) -> int:
        """增量写入（同一 chunk_id 覆盖更新）。"""

        if not chunks:
            return 0
        collection = self._get_collection(create=True)
        self._ensure_fingerprint(collection)
        self._upsert(collection, chunks)
        return len(chunks)

    def _upsert(self, collection: Any, chunks: Sequence[CaseChunk]) -> None:
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata() for chunk in chunks],
            embeddings=[self._embedding.embed_one(chunk.embedding_text()) for chunk in chunks],
        )

    def delete_case(self, case_id: str) -> int:
        """删除某个案例的全部文本块，返回删除数量。"""

        if not self.collection_exists():
            return 0
        collection = self._get_collection(create=False)
        existing = collection.get(where={"case_id": case_id}, include=[])
        ids = list(existing.get("ids") or [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)

    # -- 检索 -------------------------------------------------------------
    def query(
        self,
        query_text: str | None = None,
        *,
        query_embedding: Sequence[float] | None = None,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        min_similarity: float = 0.0,
    ) -> list[RetrievedChunk]:
        """相似度检索，返回按相似度降序排列的文本块。"""

        if not self.collection_exists():
            raise KnowledgeBaseError(
                f"向量库集合不存在：{self.config.collection_name}，请先构建知识库索引"
            )
        if query_text is None and query_embedding is None:
            raise KnowledgeBaseError("query_text 与 query_embedding 至少提供一个")

        limit = int(top_k or self.config.top_k)
        if limit <= 0:
            return []

        collection = self._get_collection(create=False)
        space = self._space_of(collection)
        result = collection.query(
            query_texts=[query_text] if query_text is not None else None,
            query_embeddings=[list(query_embedding)] if query_embedding is not None else None,
            n_results=limit,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for index, chunk_id in enumerate(ids):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            similarity = _distance_to_similarity(distance, space)
            if similarity < min_similarity:
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    text=documents[index] if index < len(documents) else "",
                    case_id=str(metadata.get("case_id", "")),
                    title=str(metadata.get("title", "")),
                    case_type=str(metadata.get("case_type", "")),
                    category=str(metadata.get("category", "")),
                    section=str(metadata.get("section", "")),
                    source_file=str(metadata.get("source_file", "")),
                    data_nature=str(metadata.get("data_nature", "")),
                    keywords=_split_list(metadata.get("keywords")),
                    robot_types=_split_list(metadata.get("robot_types")),
                    similarity=similarity,
                    distance=distance,
                )
            )
        chunks.sort(key=lambda item: item.similarity, reverse=True)
        return chunks

    # -- 状态 -------------------------------------------------------------
    def status(self) -> IndexStatus:
        from .embedding import describe_embedding_plan

        exists = self.collection_exists()
        stored: dict[str, Any] = {}
        last_built: str | None = None
        chunk_count = 0
        case_count = 0
        if exists:
            collection = self._get_collection(create=False)
            stored = dict(collection.metadata or {})
            last_built = str(stored.get("built_at")) if stored.get("built_at") else None
            chunk_count = int(collection.count())
            if chunk_count:
                metas = collection.get(include=["metadatas"]).get("metadatas") or []
                case_count = len({str(item.get("case_id")) for item in metas if item})

        fingerprint = self.config.index_fingerprint()
        stale = exists and any(
            stored.get(key) != value for key, value in fingerprint.items()
        )
        return IndexStatus(
            collection_name=self.config.collection_name,
            chunk_count=chunk_count,
            case_count=case_count,
            persist_dir=str(self.config.persist_dir),
            knowledge_dir=str(self.config.knowledge_dir),
            embedding=describe_embedding_plan(self.config),
            fingerprint=fingerprint,
            stored_fingerprint=stored,
            last_built_at=last_built,
            stale=stale,
            exists=exists,
        )

    def _space_of(self, collection: Any) -> str:
        metadata = getattr(collection, "metadata", None) or {}
        return str(metadata.get(SPACE_METADATA_KEY) or DEFAULT_SPACE)

    def _ensure_fingerprint(self, collection: Any) -> None:
        stored = dict(getattr(collection, "metadata", None) or {})
        for key, value in self.config.index_fingerprint().items():
            if key in stored and stored[key] != value:
                raise IndexVersionMismatchError(
                    f"向量库索引与当前配置不一致（{key}: {stored[key]} -> {value}）"
                )


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _distance_to_similarity(distance: float, space: str) -> float:
    """把 Chroma 返回的距离转成相似度（越大越相似）。"""

    if space == "l2":
        return 1.0 / (1.0 + max(distance, 0.0))
    if space == "ip":
        return -distance
    # cosine 距离 = 1 - 余弦相似度
    return max(0.0, 1.0 - distance)


def _split_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [item.strip() for item in str(value).split(",") if item.strip()]

