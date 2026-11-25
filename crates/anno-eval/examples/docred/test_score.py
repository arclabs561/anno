"""Regression coverage for document-scoped cluster and edge alignment."""

import unittest
from copy import deepcopy

from score import score
from score_jerex import convert, stage_counts


class DocumentRelationTest(unittest.TestCase):
    def setUp(self):
        self.gold = [{
            "title": "One",
            "sents": [["Maya", "founded", "Acme"], ["She", "sold", "it"]],
            "vertexSet": [
                [{"sent_id": 0, "pos": [0, 1]}, {"sent_id": 1, "pos": [0, 1]}],
                [{"sent_id": 0, "pos": [2, 3]}, {"sent_id": 1, "pos": [2, 3]}],
            ],
            "labels": [{"h": 0, "r": "founded", "t": 1}],
        }]
        self.pred = deepcopy(self.gold)

    def test_reordered_ids_and_duplicate_edges(self):
        self.pred[0]["vertexSet"].reverse()
        self.pred[0]["labels"] = [{"h": 1, "r": "founded", "t": 0}] * 2
        self.assertEqual(score(self.gold, self.pred, "supplied")["micro_f1"], 1.0)

    def test_split_track_changes_endpoint_identity(self):
        self.pred[0]["vertexSet"][0].pop()
        self.assertEqual(score(self.gold, self.pred, "predicted")["micro_f1"], 0.0)
        with self.assertRaisesRegex(ValueError, "supplied endpoints"):
            score(self.gold, self.pred, "supplied")

    def test_direction_matters_and_document_sets_must_agree(self):
        self.pred[0]["labels"] = [{"h": 1, "r": "founded", "t": 0}]
        self.assertEqual(score(self.gold, self.pred, "predicted")["matched_relations"], 0)
        with self.assertRaisesRegex(ValueError, "titles"):
            score(self.gold, [], "predicted")

    def test_identical_vertex_spans_do_not_collapse_edges(self):
        self.gold[0]["vertexSet"].append(deepcopy(self.gold[0]["vertexSet"][0]))
        self.gold[0]["labels"].append({"h": 2, "r": "founded", "t": 1})
        self.pred = deepcopy(self.gold)
        supplied = score(self.gold, self.pred, "supplied")
        self.assertEqual(supplied["matched_relations"], 2)
        predicted = score(self.gold, self.pred, "predicted")
        self.assertEqual(predicted["matched_relations"], 0)
        self.assertEqual(predicted["exact_clusters"], 1)
        self.assertEqual(predicted["ambiguous_gold_relations"], 2)
        self.assertEqual(predicted["matched_relations_upper_bound"], 2)
        self.assertEqual(predicted["micro_f1_upper_bound"], 1.0)

    def test_jerex_cross_sentence_entity_and_edge_direction(self):
        joint = [{
            "tokens": ["Maya", "founded", "Acme", "She", "sold", "it"],
            "mentions": [{"start": i, "end": i+1} for i in (0, 2, 3, 5)],
            "entities": [{"mentions": [1, 3]}, {"mentions": [0, 2]}],
            "relations": [{"head": 1, "tail": 0, "type": "founded"}],
        }]
        expected, actual = convert(self.gold, joint)
        self.assertEqual(score(expected, actual, "predicted")["matched_relations"], 1)
        joint[0]["tokens"].reverse()
        with self.assertRaisesRegex(ValueError, "tokens differ"):
            convert(self.gold, joint)

    def test_jerex_stage_counts_keep_proposals_partition_and_relation_labels_distinct(self):
        prediction = [{
            "tokens": ["Maya", "founded", "Acme", "She", "sold", "it"],
            "mentions": [{"start": i, "end": i+1} for i in (0, 1, 2, 3, 5)],
            "entities": [{"mentions": [0, 3]}, {"mentions": [2, 4]}],
            "relations": [{"head": 0, "tail": 1, "type": label} for label in ["owns", "founded"]],
        }]
        gold, actual = convert(self.gold, prediction)
        stages = stage_counts(gold, actual, prediction)
        self.assertEqual(stages["mention_spans"], {"gold": 4, "predicted": 5, "matched": 4})
        self.assertEqual(stages["within_cluster_mention_pairs"]["matched"], 2)
        self.assertEqual(stages["directed_cluster_pairs"]["predicted"], 1)
        self.assertEqual(stages["labeled_relations"]["predicted"], 2)
        self.assertIsNone(stages["exact_typed_clusters"])
        self.assertIsNone(stages["labeled_relations_typed_endpoints"])

    def test_jerex_repeated_tokens_need_document_ids(self):
        duplicate = deepcopy(self.gold[0])
        duplicate["title"] = "Another edition"
        docs = self.gold + [duplicate]
        pred = {"tokens": sum(duplicate["sents"], []), "mentions": [], "entities": [], "relations": []}
        with self.assertRaisesRegex(ValueError, "explicit prediction title"):
            convert(docs, [pred, pred])
        _, actual = convert(docs, [dict(pred, title=doc["title"]) for doc in docs])
        self.assertEqual([doc["title"] for doc in actual], ["One", "Another edition"])

    def test_endpoint_typing_is_an_explicit_axis(self):
        for docs in [self.gold, self.pred]:
            for cluster in docs[0]["vertexSet"]:
                for mention in cluster:
                    mention["type"] = "PER"
        self.pred[0]["vertexSet"][0][0]["type"] = "ORG"
        self.assertEqual(score(self.gold, self.pred, "predicted")["matched_relations"], 1)
        self.assertEqual(
            score(self.gold, self.pred, "predicted", require_entity_type=True)["matched_relations"], 0
        )

    def test_missing_annotations_are_not_empty_gold(self):
        del self.gold[0]["labels"]
        with self.assertRaises(KeyError):
            score(self.gold, self.pred, "supplied")


if __name__ == "__main__":
    unittest.main()
