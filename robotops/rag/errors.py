"""RobotOps AI —— Phase 3 RAG 模块异常。

继承 Phase 1 的 ``RobotOpsError``，命令行入口可用同一套逻辑捕获，
并给出可执行的排查建议（``hint``）。
"""

from __future__ import annotations

from ..exceptions import RobotOpsError


class RagError(RobotOpsError):
    """RAG 相关错误的基础类型。"""

    default_hint = ""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint or self.default_hint

    def describe(self) -> str:
        text = str(self)
        if self.hint:
            text = f"{text}\n排查建议：{self.hint}"
        return text


class RagDependencyError(RagError):
    """缺少 RAG 依赖（ChromaDB 未安装）。"""

    default_hint = (
        "执行 python -m pip install -r requirements.txt 安装 chromadb；"
        "若暂时不想使用知识库，可加 --no-rag 参数运行纯 Phase 2 流程。"
    )


class KnowledgeBaseError(RagError):
    """知识库目录为空、文档格式错误或索引不可用。"""

    default_hint = (
        "确认 knowledge/ 目录下存在 .md 案例文件（可参考 knowledge/README.md 的格式），"
        "然后执行 python scripts\\build_knowledge_base.py rebuild 重建索引。"
    )


class DocumentFormatError(RagError):
    """Markdown 案例缺少必需小节或元数据。"""

    default_hint = "案例必须包含元数据（case_id/title/case_type）与 5 个必填小节：问题、现象、可能原因、处理措施、处理结果。"


class IndexVersionMismatchError(RagError):
    """向量库由不同的 Embedding 方案或切分参数构建。"""

    default_hint = "执行 python scripts\\build_knowledge_base.py rebuild 重建索引。"

