"""RobotOps AI —— Phase 3 RAG 知识库模块。

组成：
- ``config``：知识库路径、切分参数、检索参数、Embedding 方案（唯一配置来源）；
- ``document_loader``：Markdown 案例加载、元数据解析与文本切分；
- ``embedding``：本地哈希 Embedding（无需下载模型，离线可跑）；
- ``vector_store``：ChromaDB 本地向量库（初始化 / 重建 / 写入 / 检索 / 状态）；
- ``query_builder``：把 Phase 1 异常结果转换成检索用的异常描述；
- ``retriever``：相关性门控 + 分类优先检索 + Top-K 案例聚合。

本阶段不包含多 Agent、LangGraph、Streamlit、数据库、Docker。
"""

from __future__ import annotations

from .config import RagConfig, describe_supported_env
from .document_loader import (
    CaseChunk,
    CaseDocument,
    build_chunks,
    load_case_documents,
    parse_case_document,
)
from .embedding import (
    LocalHashingEmbeddingFunction,
    build_embedding_function,
    cosine_similarity,
    describe_embedding_plan,
)
from .errors import (
    DocumentFormatError,
    IndexVersionMismatchError,
    KnowledgeBaseError,
    RagDependencyError,
    RagError,
)
from .query_builder import AnomalyQuery, build_anomaly_queries
from .retriever import KnowledgeBaseRetriever, RetrievalResult, RetrievedCase
from .vector_store import IndexStatus, KnowledgeVectorStore, RetrievedChunk

__all__ = [
    "AnomalyQuery",
    "CaseChunk",
    "CaseDocument",
    "DocumentFormatError",
    "IndexStatus",
    "IndexVersionMismatchError",
    "KnowledgeBaseError",
    "KnowledgeBaseRetriever",
    "KnowledgeVectorStore",
    "LocalHashingEmbeddingFunction",
    "RagConfig",
    "RagDependencyError",
    "RagError",
    "RetrievalResult",
    "RetrievedCase",
    "RetrievedChunk",
    "build_anomaly_queries",
    "build_chunks",
    "build_embedding_function",
    "cosine_similarity",
    "describe_embedding_plan",
    "describe_supported_env",
    "load_case_documents",
    "parse_case_document",
]

