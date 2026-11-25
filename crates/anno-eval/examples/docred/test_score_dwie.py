import unittest
from copy import deepcopy

from score import score
from score_dwie import convert, score_concepts


class DWIEProjection(unittest.TestCase):
    def setUp(self):
        self.doc = {
            "id": "news", "content": "李 hired Zoë. She left.",
            "concepts": [{"concept": 8}, {"concept": 2}],
            "mentions": [
                {"begin": 0, "end": 1, "text": "李", "concept": 8},
                {"begin": 8, "end": 11, "text": "Zoë", "concept": 2},
                {"begin": 13, "end": 16, "text": "She", "concept": 2},
            ],
            "relations": [{"s": 8, "p": "hired", "o": 2}],
        }

    def test_concept_order_and_surface_are_independent_of_identity(self):
        gold = convert(self.doc)
        pred = deepcopy(self.doc)
        pred["concepts"].reverse()
        self.assertEqual(score([gold], [convert(pred)], "supplied")["micro_f1"], 1)
        pred["mentions"].pop()
        self.assertEqual(score([gold], [convert(pred)], "predicted")["micro_f1"], 0)

    def test_supplied_concept_ids_must_survive_reordering(self):
        pred = deepcopy(self.doc)
        pred["concepts"][0]["concept"] = 88
        pred["mentions"][0]["concept"] = 88
        pred["relations"][0]["s"] = 88
        with self.assertRaisesRegex(ValueError, "concept IDs"):
            score_concepts([convert(self.doc)], [convert(pred)], "supplied")
        self.assertEqual(score_concepts([convert(self.doc)], [convert(pred)], "predicted")["micro_f1"], 1)

    def test_relation_to_unmentioned_concept(self):
        self.doc["concepts"].append({"concept": 10})
        self.doc["relations"].append({"s": 8, "p": "resides_in", "o": 10})
        documents = [convert(self.doc)]
        supplied = score_concepts(documents, documents, "supplied")
        self.assertEqual(supplied["all_concept_matched_relations"], 2)
        predicted = score_concepts(documents, documents, "predicted")
        self.assertEqual(predicted["unanchored_gold_relations"], 1)
        self.assertEqual(predicted["matched_relations"], 1)

    def test_reject_misaligned_source(self):
        self.doc["mentions"][1]["begin"] = 7
        with self.assertRaisesRegex(ValueError, "disagrees with source"):
            convert(self.doc)


if __name__ == "__main__":
    unittest.main()
