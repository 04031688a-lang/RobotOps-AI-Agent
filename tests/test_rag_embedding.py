"""Phase 3：Embedding 方案测试（本地哈希向量，离线可跑）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from robotops.rag.config import RagConfig  # noqa: E402
from robotops.rag.embedding import (  # noqa: E402
    LocalHashingEmbeddingFunction,
    build_embedding_function,
    cosine_similarity,
    describe_embedding_plan,
    text_to_features,
)


class EmbeddingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding = LocalHashingEmbeddingFunction(dim=256)

    def test_vector_is_deterministic_and_normalized(self) -> None:
        first = self.embedding.embed_one("机器人故障率升高，维修次数增加")
        second = self.embedding.embed_one("机器人故障率升高，维修次数增加")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 256)
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=5)

    def test_related_text_scores_higher_than_unrelated(self) -> None:
        query = "B小区机器人故障率明显升高，同时维修次数增加。"
        related = "机器人重复故障 故障率升高 维修次数增加 部件反复损坏"
        unrelated = "夜间作业噪音投诉 满意度下降"

        related_score = cosine_similarity(
            self.embedding.embed_one(query), self.embedding.embed_one(related)
        )
        unrelated_score = cosine_similarity(
            self.embedding.embed_one(query), self.embedding.embed_one(unrelated)
        )

        self.assertGreater(related_score, unrelated_score)
        self.assertGreater(related_score, 0.3)

    def test_empty_text_and_zero_vector(self) -> None:
        vector = text_to_features("", dim=64, ngram_min=2, ngram_max=3)

        self.assertEqual(float(np.linalg.norm(vector)), 0.0)
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)
        self.assertEqual(self.embedding.embed_one("   "), [0.0] * 256)

    def test_chroma_protocol_methods(self) -> None:
        self.assertEqual(LocalHashingEmbeddingFunction.name(), "robotops-local-hashing-v1")
        self.assertEqual(self.embedding.default_space(), "cosine")
        self.assertIn("cosine", self.embedding.supported_spaces())
        config = self.embedding.get_config()
        rebuilt = LocalHashingEmbeddingFunction.build_from_config(config)
        self.assertEqual(rebuilt.dim, self.embedding.dim)
        self.assertEqual(
            rebuilt.embed_one("机器人"), self.embedding.embed_one("机器人")
        )
        batch = self.embedding(["第一段", "第二段"])
        self.assertEqual(len(batch), 2)

    def test_build_embedding_function_uses_config(self) -> None:
        function = build_embedding_function(RagConfig(embedding_dim=128))

        self.assertEqual(function.dim, 128)
        self.assertEqual(len(function.embed_one("测试")), 128)

    def test_invalid_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            LocalHashingEmbeddingFunction(dim=0)

    def test_embedding_plan_explains_why_not_deepseek(self) -> None:
        plan = describe_embedding_plan()

        self.assertEqual(plan["backend"], "local_hashing")
        self.assertEqual(plan["space"], "cosine")
        self.assertIn("DeepSeek", plan["reason_not_deepseek"])
        self.assertIn("embedding", plan["reason_not_deepseek"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

