"""RobotOps AI —— Phase 3 Embedding 方案。

为什么不用 DeepSeek 做 Embedding：
    DeepSeek 开放平台提供的是 Chat Completions（对话补全）接口，
    并未提供 embedding（文本向量化）接口，因此无法用它生成向量。
    详见 README「Embedding 方案说明」。

本模块提供**本地哈希向量**方案：
- 纯 Python + NumPy 实现，不下载任何模型，离线可稳定运行；
- 面向中文场景：同时使用「字符 2-gram / 3-gram」与「领域词 / 英文单词」特征，
  通过哈希映射到固定维度并做 L2 归一化；
- 同一段文本永远得到同一向量（确定性），便于索引重建与结果复现；
- 余弦相似度可直接用于 ChromaDB 的 cosine 空间。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Iterable, Sequence

import numpy as np

from .config import EMBEDDING_BACKEND, RagConfig

try:  # 已安装 ChromaDB 时继承其 EmbeddingFunction 协议基类，避免版本兼容告警
    from chromadb.api.types import EmbeddingFunction as _EmbeddingFunctionBase
except ImportError:  # pragma: no cover - 未安装 chromadb 时仍可单独使用 Embedding

    class _EmbeddingFunctionBase:  # type: ignore[no-redef]
        """chromadb 缺失时的占位基类。"""


EMBEDDING_FUNCTION_NAME = "robotops-local-hashing-v1"
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
DOMAIN_HINTS: tuple[str, ...] = (
    "故障",
    "维修",
    "维保",
    "保养",
    "备件",
    "返修",
    "运行率",
    "运行时长",
    "停机",
    "满意度",
    "投诉",
    "清洁",
    "巡检",
    "漏检",
    "成本",
    "节降",
    "能耗",
    "闲置",
    "调度",
    "充电",
    "电池",
    "续航",
    "售后",
    "响应",
    "噪音",
    "机器人",
    "设备",
    "任务",
)


def normalize_text(text: str) -> str:
    """统一大小写与空白，提升不同书写方式之间的匹配率。"""

    lowered = (text or "").lower()
    return re.sub(r"\s+", " ", lowered).strip()


def _hash_index(token: str, dim: int, salt: str) -> int:
    digest = hashlib.blake2b(f"{salt}:{token}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def text_to_features(
    text: str,
    *,
    dim: int,
    ngram_min: int = 2,
    ngram_max: int = 3,
) -> np.ndarray:
    """把文本转换为固定维度的稀疏哈希向量（已做 L2 归一化）。"""

    vector = np.zeros(dim, dtype="float32")
    normalized = normalize_text(text)
    if not normalized:
        return vector

    # 1) 字符 n-gram 特征（中文场景主要依赖这部分）
    dense = re.sub(r"\s+", "", normalized)
    for size in range(max(ngram_min, 1), max(ngram_max, ngram_min) + 1):
        for start in range(0, max(len(dense) - size + 1, 0)):
            gram = dense[start : start + size]
            vector[_hash_index(f"g{size}:{gram}", dim, "ngram")] += 1.0

    # 2) 英文/数字单词特征
    for word in TOKEN_PATTERN.findall(normalized):
        vector[_hash_index(f"w:{word}", dim, "word")] += 1.6

    # 3) 领域词特征（给业务术语更高权重，提升异常描述与案例的匹配）
    for hint in DOMAIN_HINTS:
        if hint in normalized:
            vector[_hash_index(f"d:{hint}", dim, "domain")] += 3.0

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """余弦相似度（输入向量已归一化时等价于点积）。"""

    a = np.asarray(left, dtype="float64")
    b = np.asarray(right, dtype="float64")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


class LocalHashingEmbeddingFunction(_EmbeddingFunctionBase):
    """ChromaDB 兼容的本地 Embedding 函数。

    同时兼容 ChromaDB 1.x 的 EmbeddingFunction 协议（``__call__`` / ``name`` /
    ``get_config`` / ``build_from_config``），但不强制继承其基类，
    以便在未安装 ChromaDB 时也能单独用于测试。
    """

    def __init__(
        self,
        dim: int = 1024,
        *,
        ngram_min: int = 2,
        ngram_max: int = 3,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正整数")
        self.dim = int(dim)
        self.ngram_min = int(ngram_min)
        self.ngram_max = int(ngram_max)

    # -- ChromaDB 协议 ---------------------------------------------------
    def __call__(self, input: Iterable[str]) -> list[list[float]]:  # noqa: A002 - 与 Chroma 协议保持一致
        return [self.embed_one(text) for text in input]

    def embed_query(self, input: Iterable[str]) -> list[list[float]]:  # noqa: A002
        return self.__call__(input)

    def embed_documents(self, input: Iterable[str]) -> list[list[float]]:  # noqa: A002
        return self.__call__(input)

    @staticmethod
    def name() -> str:
        return EMBEDDING_FUNCTION_NAME

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim, "ngram_min": self.ngram_min, "ngram_max": self.ngram_max}

    @classmethod
    def build_from_config(cls, config: dict[str, Any]) -> "LocalHashingEmbeddingFunction":
        return cls(
            dim=int(config.get("dim", 1024)),
            ngram_min=int(config.get("ngram_min", 2)),
            ngram_max=int(config.get("ngram_max", 3)),
        )

    # -- 便捷方法 --------------------------------------------------------
    def embed_one(self, text: str) -> list[float]:
        return text_to_features(
            text, dim=self.dim, ngram_min=self.ngram_min, ngram_max=self.ngram_max
        ).tolist()


def build_embedding_function(config: RagConfig | None = None) -> LocalHashingEmbeddingFunction:
    """按配置构造 Embedding 函数（当前仅支持本地哈希向量）。"""

    cfg = config or RagConfig()
    cfg.validate()
    return LocalHashingEmbeddingFunction(
        dim=cfg.embedding_dim, ngram_min=cfg.ngram_min, ngram_max=cfg.ngram_max
    )


def describe_embedding_plan(config: RagConfig | None = None) -> dict[str, Any]:
    """返回 Embedding 方案说明，用于 README/报告/日志。"""

    cfg = config or RagConfig()
    return {
        "backend": EMBEDDING_BACKEND,
        "name": EMBEDDING_FUNCTION_NAME,
        "dim": cfg.embedding_dim,
        "ngram_range": f"{cfg.ngram_min}-{cfg.ngram_max}",
        "space": "cosine",
        "reason_not_deepseek": (
            "DeepSeek 开放平台只提供 Chat Completions（对话补全）接口，"
            "没有 embedding 接口，无法生成文本向量"
        ),
    }
