"""RobotOps AI —— Phase 3 知识库文档加载与切分。

职责：
1. 读取 ``knowledge/`` 下的 Markdown 案例；
2. 解析元数据（front matter）与标题；
3. 按 Markdown 小节切分成适合检索的文本块（超长小节二次切分、带重叠）；
4. 保留案例标题、案例类型、分类等元数据，便于检索结果溯源。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import REQUIRED_SECTIONS, RagConfig
from .errors import DocumentFormatError, KnowledgeBaseError

FRONT_MATTER_DELIMITER = "---"
META_KEYS: tuple[str, ...] = (
    "case_id",
    "title",
    "case_type",
    "category",
    "severity_hint",
    "applicable_anomaly",
    "robot_types",
    "keywords",
    "data_nature",
)


@dataclass
class CaseDocument:
    """一个完整的案例文档（含全部小节）。"""

    case_id: str
    title: str
    case_type: str
    category: str
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)

    @property
    def keywords(self) -> list[str]:
        return list(self.meta.get("keywords") or [])

    @property
    def data_nature(self) -> str:
        return str(self.meta.get("data_nature") or "")

    @property
    def robot_types(self) -> list[str]:
        return list(self.meta.get("robot_types") or [])

    def section_names(self) -> list[str]:
        return list(self.sections.keys())

    def summary_text(self, *, budget: int | None = None) -> str:
        """把案例整理成便于注入提示词的紧凑文本。"""

        lines = [f"案例编号：{self.case_id}", f"案例标题：{self.title}"]
        if self.case_type:
            lines.append(f"案例类型：{self.case_type}")
        if self.keywords:
            lines.append(f"关键词：{'、'.join(self.keywords)}")
        for name, body in self.sections.items():
            lines.append(f"【{name}】{body}")
        text = "\n".join(lines)
        if budget is not None and len(text) > budget:
            text = text[:budget].rstrip() + "…（已截断）"
        return text


@dataclass
class CaseChunk:
    """切分后的检索单元。"""

    chunk_id: str
    case_id: str
    title: str
    case_type: str
    category: str
    section: str
    chunk_index: int
    text: str
    keywords: list[str] = field(default_factory=list)
    source_file: str = ""
    data_nature: str = ""
    robot_types: list[str] = field(default_factory=list)

    def embedding_text(self) -> str:
        """用于向量化的文本：在小节内容前补充案例类型与标题，提升中文匹配效果。"""

        header = f"【案例类型】{self.case_type}\n【案例标题】{self.title}\n"
        if self.keywords:
            header += f"【关键词】{'、'.join(self.keywords)}\n"
        return f"{header}{self.text}"

    def metadata(self) -> dict[str, Any]:
        """Chroma 元数据（值必须是标量，因此列表统一转为逗号分隔字符串）。"""

        return {
            "chunk_id": self.chunk_id,
            "case_id": self.case_id,
            "title": self.title,
            "case_type": self.case_type,
            "category": self.category,
            "section": self.section,
            "chunk_index": int(self.chunk_index),
            "keywords": ",".join(self.keywords),
            "source_file": self.source_file,
            "data_nature": self.data_nature,
            "robot_types": ",".join(self.robot_types),
        }


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
def load_case_documents(
    config: RagConfig | None = None,
    *,
    knowledge_dir: str | Path | None = None,
) -> list[CaseDocument]:
    """读取知识库目录下的全部 Markdown 案例。"""

    cfg = config or RagConfig()
    root = Path(knowledge_dir) if knowledge_dir is not None else cfg.knowledge_dir
    if not root.exists():
        raise KnowledgeBaseError(f"知识库目录不存在：{root}")

    files = sorted(path for path in root.rglob("*.md") if path.name.lower() != "readme.md")
    if not files:
        raise KnowledgeBaseError(f"知识库目录下没有案例文件（.md）：{root}")

    documents: list[CaseDocument] = []
    seen_ids: dict[str, Path] = {}
    for path in files:
        document = parse_case_document(path, root=root)
        if document.case_id in seen_ids:
            raise DocumentFormatError(
                f"案例编号重复：{document.case_id}"
                f"（{seen_ids[document.case_id]} 与 {path}）"
            )
        seen_ids[document.case_id] = path
        documents.append(document)
    return documents


def parse_case_document(path: Path, *, root: Path | None = None) -> CaseDocument:
    """解析单个 Markdown 案例文件。"""

    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise KnowledgeBaseError(f"读取案例文件失败：{path}（{error}）") from error

    meta, body = _split_front_matter(text)
    title, sections = _parse_sections(body)

    case_id = str(meta.get("case_id") or "").strip()
    if not case_id:
        raise DocumentFormatError(f"案例缺少 case_id：{path}")

    title = str(meta.get("title") or title or case_id).strip()
    case_type = str(meta.get("case_type") or "").strip()
    if not case_type:
        raise DocumentFormatError(f"案例缺少 case_type（案例类型）：{path}")

    missing = [name for name in REQUIRED_SECTIONS if name not in sections]
    if missing:
        raise DocumentFormatError(
            f"案例 {case_id} 缺少必需小节：{'、'.join(missing)}（文件：{path}）"
        )

    category = str(meta.get("category") or (path.parent.name if root else "")).strip()
    return CaseDocument(
        case_id=case_id,
        title=title,
        case_type=case_type,
        category=category,
        path=path,
        meta=meta,
        sections=sections,
    )


def build_chunks(
    documents: Iterable[CaseDocument],
    config: RagConfig | None = None,
) -> list[CaseChunk]:
    """把案例切分成检索文本块。"""

    cfg = config or RagConfig()
    chunks: list[CaseChunk] = []

    for document in documents:
        for section_name, body in document.sections.items():
            pieces = split_long_text(
                body, max_chars=cfg.chunk_max_chars, overlap=cfg.chunk_overlap_chars
            )
            for index, piece in enumerate(pieces):
                chunks.append(
                    CaseChunk(
                        chunk_id=f"{document.case_id}::{section_name}::{index}",
                        case_id=document.case_id,
                        title=document.title,
                        case_type=document.case_type,
                        category=document.category,
                        section=section_name,
                        chunk_index=index,
                        text=piece,
                        keywords=document.keywords,
                        source_file=document.path.name,
                        data_nature=document.data_nature,
                        robot_types=document.robot_types,
                    )
                )
    return chunks


def split_long_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """把长文本切成带重叠的片段（按行边界尽量不切断句子）。"""

    cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(cleaned_lines)
    if len(cleaned) <= max_chars:
        return [cleaned] if cleaned else []

    pieces: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        if end < len(cleaned):
            newline = cleaned.rfind("\n", start, end)
            if newline > start + max_chars // 2:
                end = newline
        piece = cleaned[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
        while start < len(cleaned) and cleaned[start] in "\n ":
            start += 1
    return pieces


# ---------------------------------------------------------------------------
# 内部解析工具
# ---------------------------------------------------------------------------
def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONT_MATTER_DELIMITER:
        return {}, text

    meta_lines: list[str] = []
    end_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == FRONT_MATTER_DELIMITER:
            end_index = index
            break
        meta_lines.append(lines[index])

    if end_index is None:
        raise DocumentFormatError("front matter 缺少结束分隔符 ---")
    return _parse_meta(meta_lines), "\n".join(lines[end_index + 1 :])


def _parse_meta(lines: list[str]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, _, raw_value = stripped.partition(":")
        meta[key.strip()] = _parse_meta_value(raw_value.strip())
    return meta


def _parse_meta_value(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
    return value.strip("'\"")


def _parse_sections(body: str) -> tuple[str, dict[str, str]]:
    title = ""
    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is not None:
            sections[current] = "\n".join(buffer).strip()

    for line in body.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            flush()
            current = line[3:].strip()
            buffer = []
            continue
        if current is not None:
            buffer.append(line.rstrip())
    flush()
    return title, {name: text for name, text in sections.items() if name}

