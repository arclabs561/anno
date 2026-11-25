from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from prepare_preco_replay import conll

from entity_memory_replay import (
    POLICIES, Document, Memory, Mention, cluster_scores, compare_wrong_attachments,
    encode, fixed_coverage, load_conll, load_split, replay,
)


class EntityMemoryReplayTests(TestCase):
    def test_conll_preserves_repeated_and_nested_occurrences(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "example.conll"
            annotations = ["(0", "(1)|0)", "(1)", "(2)"]
            tokens = ["the", "guard", "guard", "left"]
            rows = [f"doc\t0\t{i}\t{token}\t{annotation}" for i, (token, annotation)
                    in enumerate(zip(tokens, annotations, strict=True))]
            path.write_text("\n".join(["#begin document (doc); part 0", *rows, "#end document"]),
                            encoding="utf-8")
            self.assertEqual(load_conll(path).mentions, [
                Mention(0, 2, "0"), Mention(1, 2, "1"), Mention(2, 3, "1"), Mention(3, 4, "2")
            ])

    def test_future_suffix_and_gold_labels_cannot_change_earlier_vectors(self):
        first = Document("a", ["Ann", "said", "she", "left"], [Mention(2, 3, "ann")])
        second = Document("b", ["Ann", "said", "she", "stayed"], [Mention(2, 3, "bea")])
        np.testing.assert_array_equal(encode(first, 16), encode(second, 16))
        changed = Document("c", ["Bea", "said", "she", "left"], [Mention(2, 3, "ann")])
        self.assertFalse(np.array_equal(encode(first, 16), encode(changed, 16)))

    def test_array_budget_and_deterministic_eviction(self):
        query = np.array([1., 0.], dtype=np.float32)
        for policy in POLICIES:
            with self.subTest(policy=policy):
                memory = Memory(policy, 2, 40)
                self.assertLessEqual(memory.bytes, 40)
                self.assertEqual(memory.capacity, 1)
                memory.update(1, query, 0)
                memory.update(2, query, 1)
                self.assertEqual(memory.rank(query, 1)[0][0], 2)

    def test_ranking_and_eviction_are_distinct_misses(self):
        document = Document("a", ["a", "b", "a"],
                            [Mention(0, 1, "a"), Mention(1, 2, "b"), Mention(2, 3, "a")])
        features = np.array([[1., 0.], [0., 1.], [0., 1.]], dtype=np.float32)
        evicted = replay(document, features, "centroid", 32, 1, 2)
        retained = replay(document, features, "centroid", 64, 1, 2)
        self.assertEqual(evicted.get("long_retained", 0), 0)
        self.assertEqual(retained["long_retained"], 1)
        self.assertEqual(retained["long_hits"], 0)

    def test_cluster_metrics_and_merge_damage(self):
        perfect = cluster_scores([0, 1, 0, 1], [0, 1, 0, 1])
        merged = cluster_scores([0, 1, 0, 1], [0, 0, 0, 0])
        self.assertEqual(perfect["conll_document_f1"], 1.0)
        self.assertEqual(merged["false_merge_pairs"], 4)
        self.assertLess(merged["conll_document_f1"], 1.0)

    def test_external_features_are_checked_against_occurrence_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "example.conll"
            source.write_text(
                "#begin document (doc); part 0\n"
                "doc\t0\t0\tAnn\t(0)\n"
                "doc\t0\t1\tshe\t(0)\n#end document\n",
                encoding="utf-8",
            )
            vectors = np.eye(2, dtype=np.float32)
            features = root / "example.npz"
            np.savez(features, spans=[[0, 1], [1, 2]], vectors=vectors)
            np.testing.assert_array_equal(load_split(root, 2, root)[0][1], vectors)
            np.savez(features, spans=[[1, 2], [0, 1]], vectors=vectors)
            with self.assertRaisesRegex(ValueError, "spans differ"):
                load_split(root, 2, root)

            np.savez(features, spans=[[0, 1], [1, 2]], vectors=vectors / 2)
            with self.assertRaisesRegex(ValueError, "unit-normalized"):
                load_split(root, 2, root)

    def test_bootstrap_resamples_paired_documents(self):
        box = [{"document": "a", "links": 2, "wrong_attachments": 2}]
        comparator = [{"document": "a", "links": 2, "wrong_attachments": 0}]
        receipt = compare_wrong_attachments(box, comparator)
        self.assertEqual(receipt["baseline_minus_box_95_interval"], [-1.0, -1.0])
        self.assertEqual(receipt["bootstrap_valid"], 2_000)

    def test_fixed_coverage_does_not_use_correctness_to_break_ties(self):
        trace = [(1.0, False), (1.0, True), (0.7, True)]
        self.assertEqual(fixed_coverage(trace, 4, 0.25),
                         {"links": 1, "wrong_attachments": 1})

    def test_preco_adapter_preserves_sentence_offsets_and_distinct_pronouns(self):
        record = {
            "id": "sample",
            "sentences": [["Ann", "said", "she"], ["Bea", "said", "she"]],
            "mention_clusters": [[[0, 0, 1], [1, 2, 3]], [[1, 0, 1], [0, 2, 3]]],
        }
        with TemporaryDirectory() as directory:
            name, contents = conll(record)
            path = Path(directory) / f"{name}.conll"
            path.write_text(contents, encoding="utf-8")
            self.assertEqual(load_conll(path).mentions, [
                Mention(0, 1, "0"), Mention(2, 3, "1"),
                Mention(3, 4, "1"), Mention(5, 6, "0"),
            ])
