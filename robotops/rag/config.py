"""RobotOps AI —— Phase 3 RAG 配置（知识库路径、切分参数、检索参数的唯一来源）。

复用 Phase 2 的 ``robotops.llm.env_loader`` 读取 .env / 环境变量，
所有参数都可以通过 .env 覆盖，代码中不散落魔法数字。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import config as project_config
from ..llm.env_loader import ensure_env_loaded, get_env

# ---------------------------------------------------------------------------
# 环境变量名称
# ---------------------------------------------------------------------------
ENV_KNOWLEDGE_DIR = "ROBOTOPS_KNOWLEDGE_DIR"
ENV_CHROMA_DIR = "ROBOTOPS_CHROMA_DIR"
ENV_COLLECTION = "ROBOTOPS_RAG_COLLECTION"
ENV_TOP_K = "ROBOTOPS_RAG_TOP_K"
ENV_MAX_CASES = "ROBOTOPS_RAG_MAX_CASES"
ENV_MIN_SIMILARITY = "ROBOTOPS_RAG_MIN_SIMILARITY"
ENV_EMBEDDING_BACKEND = "ROBOTOPS_EMBEDDING_BACKEND"
ENV_EMBEDDING_DIM = "ROBOTOPS_EMBEDDING_DIM"
ENV_AUTO_BUILD = "ROBOTOPS_RAG_AUTO_BUILD"

# ---------------------------------------------------------------------------
# 默认值
# ---------------------------------------------------------------------------
KNOWLEDGE_DIR: Path = project_config.PROJECT_ROOT / "knowledge"
CHROMA_DIR: Path = project_config.DATA_DIR / "chroma"
COLLECTION_NAME = "robot_ops_cases"
INDEX_SCHEMA_VERSION = "phase3-rag-v1"

#: Embedding 方案：本地哈希向量（无需下载模型、可离线稳定运行）
EMBEDDING_BACKEND = "local_hashing"
EMBEDDING_DIM = 1024
NGRAM_MIN = 2
NGRAM_MAX = 3

#: 文本切分参数（按 Markdown 小节切块，超长小节二次切分）
CHUNK_MAX_CHARS = 480
CHUNK_OVERLAP_CHARS = 120

#: 检索参数
DEFAULT_TOP_K = 5
DEFAULT_MAX_CASES = 6
DEFAULT_MIN_SIMILARITY = 0.35
DEFAULT_AUTO_BUILD = True

#: 单个案例注入提示词的最大字符数
CASE_TEXT_BUDGET = 700

#: 判断“问题是否与机器人运营相关”的领域词表（无关问题直接判定为无相关案例）
DOMAIN_TERMS: tuple[str, ...] = (
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
    "清扫",
    "巡检",
    "漏检",
    "成本",
    "节降",
    "能耗",
    "闲置",
    "利用率",
    "调度",
    "排队",
    "路径",
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

REQUIRED_SECTIONS: tuple[str, ...] = ("问题", "现象", "可能原因", "处理措施", "处理结果")


@dataclass(frozen=True)
class RagConfig:
    """RAG 运行配置。"""

    knowledge_dir: Path = KNOWLEDGE_DIR
    persist_dir: Path = CHROMA_DIR
    collection_name: str = COLLECTION_NAME
    top_k: int = DEFAULT_TOP_K
    max_cases: int = DEFAULT_MAX_CASES
    min_similarity: float = DEFAULT_MIN_SIMILARITY
    embedding_backend: str = EMBEDDING_BACKEND
    embedding_dim: int = EMBEDDING_DIM
    ngram_min: int = NGRAM_MIN
    ngram_max: int = NGRAM_MAX
    chunk_max_chars: int = CHUNK_MAX_CHARS
    chunk_overlap_chars: int = CHUNK_OVERLAP_CHARS
    auto_build: bool = DEFAULT_AUTO_BUILD
    index_schema_version: str = INDEX_SCHEMA_VERSION
    case_text_budget: int = CASE_TEXT_BUDGET
    env_file: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        # 允许传入字符串路径（例如 RagConfig(persist_dir="data/chroma")）
        object.__setattr__(self, "knowledge_dir", Path(self.knowledge_dir))
        object.__setattr__(self, "persist_dir", Path(self.persist_dir))
        if self.env_file is not None:
            object.__setattr__(self, "env_file", Path(self.env_file))

    @classmethod
    def from_env(cls, *, env_file: str | Path | None = None, **overrides: Any) -> "RagConfig":
        used_env_file = ensure_env_loaded(env_file)
        values: dict[str, Any] = {
            "knowledge_dir": _to_path(get_env(ENV_KNOWLEDGE_DIR), KNOWLEDGE_DIR),
            "persist_dir": _to_path(get_env(ENV_CHROMA_DIR), CHROMA_DIR),
            "collection_name": get_env(ENV_COLLECTION, COLLECTION_NAME),
            "top_k": _to_int(get_env(ENV_TOP_K), DEFAULT_TOP_K),
            "max_cases": _to_int(get_env(ENV_MAX_CASES), DEFAULT_MAX_CASES),
            "min_similarity": _to_float(get_env(ENV_MIN_SIMILARITY), DEFAULT_MIN_SIMILARITY),
            "embedding_backend": get_env(ENV_EMBEDDING_BACKEND, EMBEDDING_BACKEND),
            "embedding_dim": _to_int(get_env(ENV_EMBEDDING_DIM), EMBEDDING_DIM),
            "auto_build": _to_bool(get_env(ENV_AUTO_BUILD), DEFAULT_AUTO_BUILD),
            "env_file": used_env_file,
        }
        values.update(overrides)
        return cls(**values)

    def to_public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["knowledge_dir"] = str(self.knowledge_dir)
        data["persist_dir"] = str(self.persist_dir)
        data["env_file"] = str(self.env_file) if self.env_file else None
        return data

    def validate(self) -> None:
        if self.embedding_backend != EMBEDDING_BACKEND:
            from .errors import KnowledgeBaseError

            raise KnowledgeBaseError(
                f"暂不支持的 Embedding 方案：{self.embedding_backend}；"
                f"当前版本提供 {EMBEDDING_BACKEND}（本地哈希向量，无需下载模型）"
            )
        if self.embedding_dim <= 0:
            from .errors import KnowledgeBaseError

            raise KnowledgeBaseError(f"embedding_dim 必须为正整数，当前为 {self.embedding_dim}")
        if self.top_k <= 0 or self.max_cases <= 0:
            from .errors import KnowledgeBaseError

            raise KnowledgeBaseError("top_k / max_cases 必须为正整数")
        if not (0.0 <= self.min_similarity <= 1.0):
            from .errors import KnowledgeBaseError

            raise KnowledgeBaseError(
                f"min_similarity 应在 0~1 之间，当前为 {self.min_similarity}"
            )
        if self.ngram_min < 1 or self.ngram_max < self.ngram_min:
            from .errors import KnowledgeBaseError

            raise KnowledgeBaseError(
                f"n-gram 范围不合法：{self.ngram_min}~{self.ngram_max}"
            )

    def index_fingerprint(self) -> dict[str, Any]:
        """索引指纹：方案或参数变化时用于判断是否需要重建。"""

        return {
            "index_schema_version": self.index_schema_version,
            "embedding_backend": self.embedding_backend,
            "embedding_dim": self.embedding_dim,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "chunk_max_chars": self.chunk_max_chars,
            "chunk_overlap_chars": self.chunk_overlap_chars,
        }


def describe_supported_env() -> list[str]:
    """返回本模块识别的环境变量，用于帮助信息与 README。"""

    return [
        ENV_KNOWLEDGE_DIR,
        ENV_CHROMA_DIR,
        ENV_COLLECTION,
        ENV_TOP_K,
        ENV_MAX_CASES,
        ENV_MIN_SIMILARITY,
        ENV_EMBEDDING_BACKEND,
        ENV_EMBEDDING_DIM,
        ENV_AUTO_BUILD,
    ]


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _to_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_config.PROJECT_ROOT / path)


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
