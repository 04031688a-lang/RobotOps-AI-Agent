"""Phase 3：知识库文档加载与切分测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotops import config  # noqa: E402
from robotops.rag.config import REQUIRED_SECTIONS, RagConfig  # noqa: E402
from robotops.rag.document_loader import (  # noqa: E402
    build_chunks,
    load_case_documents,
    parse_case_document,
    split_long_text,
)
from robotops.rag.errors import DocumentFormatError, KnowledgeBaseError  # noqa: E402

#: 需求中要求覆盖的 12 类案例类型
REQUIRED_CASE_TYPES = {
    "机器人重复故障",
    "机器人运行率下降",
    "巡检不到位",
    "用户满意度下降",
    "清洁效果下降",
    "设备维修频繁",
    "售后响应慢",
    "运营成本过高",
    "节降率下降",
    "设备闲置",
    "任务调度异常",
    "设备维护周期过长",
}

VALID_CASE = """---
case_id: CASE-TEST-001
title: 测试案例标题
case_type: 机器人重复故障
category: equipment_fault
keywords: [重复故障, 故障率升高]
data_nature: 模拟案例（虚构）
---

# CASE-TEST-001 测试案例标题

## 问题
测试问题

## 现象
测试现象

## 可能原因
测试原因

## 处理措施
测试措施

## 处理结果
测试结果
"""


class KnowledgeBaseContentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = load_case_documents()
        cls.chunks = build_chunks(cls.documents)

    def test_case_count_and_directories(self) -> None:
        self.assertGreaterEqual(len(self.documents), 15, "案例数量应不少于 15 个")
        categories = {document.category for document in self.documents}
        self.assertEqual(
            categories, {"equipment_fault", "maintenance", "satisfaction", "operation"}
        )
        # 案例文件必须位于 knowledge/ 下的分类目录中
        for document in self.documents:
            self.assertEqual(document.path.parent.parent, config.PROJECT_ROOT / "knowledge")

    def test_required_case_types_are_covered(self) -> None:
        covered = {document.case_type for document in self.documents}
        missing = REQUIRED_CASE_TYPES - covered
        self.assertEqual(missing, set(), f"缺少案例类型：{missing}")

    def test_every_case_has_required_sections_and_metadata(self) -> None:
        for document in self.documents:
            with self.subTest(case=document.case_id):
                for section in REQUIRED_SECTIONS:
                    self.assertIn(section, document.sections, f"{document.case_id} 缺少「{section}」")
                    self.assertTrue(document.sections[section].strip())
                self.assertTrue(document.case_id)
                self.assertTrue(document.title)
                self.assertTrue(document.case_type)
                self.assertTrue(document.keywords)
                # 必须声明是模拟案例，避免被当成真实企业数据
                self.assertIn("模拟案例", document.data_nature)

    def test_chunks_keep_metadata(self) -> None:
        self.assertTrue(self.chunks)
        for chunk in self.chunks:
            with self.subTest(chunk=chunk.chunk_id):
                self.assertTrue(chunk.text.strip())
                self.assertTrue(chunk.case_id)
                self.assertTrue(chunk.title)
                self.assertTrue(chunk.case_type)
                self.assertIn(chunk.section, REQUIRED_SECTIONS + ("经验总结",))
                metadata = chunk.metadata()
                self.assertEqual(metadata["case_id"], chunk.case_id)
                self.assertEqual(metadata["case_type"], chunk.case_type)
                # Chroma 元数据必须是标量，列表要转成字符串
                for value in metadata.values():
                    self.assertIsInstance(value, (str, int, float, bool))

    def test_chunk_ids_are_unique_and_embedding_text_contains_context(self) -> None:
        ids = [chunk.chunk_id for chunk in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))
        sample = self.chunks[0]
        embedding_text = sample.embedding_text()
        self.assertIn(sample.case_type, embedding_text)
        self.assertIn(sample.title, embedding_text)


class DocumentParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._temp.name)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _write(self, text: str, name: str = "CASE-TEST-001.md") -> Path:
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_parse_valid_case(self) -> None:
        document = parse_case_document(self._write(VALID_CASE))

        self.assertEqual(document.case_id, "CASE-TEST-001")
        self.assertEqual(document.case_type, "机器人重复故障")
        self.assertEqual(document.category, "equipment_fault")
        self.assertEqual(document.keywords, ["重复故障", "故障率升高"])
        self.assertEqual(document.sections["问题"], "测试问题")

    def test_missing_case_id_raises(self) -> None:
        with self.assertRaises(DocumentFormatError):
            parse_case_document(self._write(VALID_CASE.replace("case_id: CASE-TEST-001\n", "")))

    def test_missing_case_type_raises(self) -> None:
        with self.assertRaises(DocumentFormatError):
            parse_case_document(self._write(VALID_CASE.replace("case_type: 机器人重复故障\n", "")))

    def test_missing_section_raises(self) -> None:
        broken = VALID_CASE.replace("## 处理结果\n测试结果\n", "")
        with self.assertRaises(DocumentFormatError) as ctx:
            parse_case_document(self._write(broken))
        self.assertIn("处理结果", str(ctx.exception))

    def test_empty_directory_raises(self) -> None:
        with self.assertRaises(KnowledgeBaseError):
            load_case_documents(RagConfig(knowledge_dir=self.tmp))

    def test_duplicate_case_id_raises(self) -> None:
        self._write(VALID_CASE, "a.md")
        self._write(VALID_CASE, "b.md")

        with self.assertRaises(DocumentFormatError):
            load_case_documents(RagConfig(knowledge_dir=self.tmp))

    def test_split_long_text(self) -> None:
        text = "\n".join(f"第{i}行内容，用于验证长文本切分与重叠逻辑。" for i in range(60))

        pieces = split_long_text(text, max_chars=200, overlap=50)

        self.assertGreater(len(pieces), 1)
        for piece in pieces:
            self.assertLessEqual(len(piece), 220)
        self.assertEqual(split_long_text("短文本", max_chars=200, overlap=50), ["短文本"])
        self.assertEqual(split_long_text("   ", max_chars=10, overlap=0), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

