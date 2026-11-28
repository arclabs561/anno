"""Ground GLiDRE relations in source text and pre-tokenized DocRED documents."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crates/anno-cli/src/cli/commands"))
import glidre


def docred_document():
    return {
        "title": "Geneva",
        "sents": [["Léa", "arrived", "."], ["She", "saw", "New", "Geneva", "."]],
        "vertexSet": [
            [{"type": "PER", "sent_id": 0, "pos": [0, 1]},
             {"type": "PER", "sent_id": 1, "pos": [0, 1]}],
            [{"type": "LOC", "sent_id": 1, "pos": [2, 4]}],
            [{"type": "ORG", "sent_id": 1, "pos": [2, 4]}],
        ],
        "labels": [{"h": 0, "t": 2, "r": "HELD_OUT"}],
    }


class GroundDocumentRelations(unittest.TestCase):
    def test_native_text_preserves_source_and_shared_ids(self):
        doc = {
            "id": "memo", "text": "𝄞 Léa.\nLéa saw Ana.",
            "entities": [
                {"id": "L", "mentions": [{"start": 2, "end": 5}, {"start": 7, "end": 10}]},
                {"id": 42, "mentions": [{"start": 15, "end": 18, "text": "Ana"}]},
            ],
        }
        split = [("𝄞", 0, 1), ("Léa", 2, 5), (".", 5, 6), ("Léa", 7, 10),
                 ("saw", 11, 14), ("Ana", 15, 18), (".", 18, 19)]
        model = SimpleNamespace(data_processor=SimpleNamespace(words_splitter=lambda text: split))
        edges = [{"h": 0, "t": 1, "r": "saw", "score": 0.9}]
        with patch.object(glidre, "classify", return_value=edges) as classifier:
            result = glidre.predict(model, doc, {"SAW": "saw"}, 0.3)
        self.assertEqual(result["text"], doc["text"])
        indexed = classifier.call_args.args[3]
        self.assertEqual([(m["start"], m["end"]) for m in indexed[0]["mentions"]],
                         [(1, 1), (3, 3)])
        self.assertEqual(result["entities"][0]["mentions"], [
            {"text": "Léa", "start": 2, "end": 5}, {"text": "Léa", "start": 7, "end": 10}])
        self.assertEqual(result["relations"], [
            {"head": "L", "tail": 42, "predicate": "saw", "score": 0.9}])

    def test_native_mentions_reject_partial_tokens_bad_offsets_and_wrong_text(self):
        for mention, error in [({"start": 2, "end": 5}, "token boundaries"),
                               ({"start": 0, "end": 6}, "invalid character span"),
                               ({"start": 0, "end": 5, "text": "other"}, "text mismatch")]:
            doc = {"id": 1, "text": "Café!", "entities": [{"id": 2, "mentions": [mention]}]}
            with self.subTest(mention=mention), self.assertRaisesRegex(ValueError, error):
                glidre.prepare_text(doc, lambda text: [("Café", 0, 4), ("!", 4, 5)])

    def test_global_inclusive_token_offsets_and_ordered_pair_slots(self):
        tokens, vertices = glidre.prepare(docred_document())
        self.assertEqual(tokens[3], "She")
        self.assertEqual(vertices[0]["mentions"][1], {"start": 3, "end": 3, "value": "She"})
        self.assertEqual(vertices[1]["mentions"], [{"start": 5, "end": 6, "value": "New Geneva"}])
        empty = [[] for _ in range(9)]
        # Identical source spans on vertices 1 and 2 still produce different directed edges.
        for head, tail in [(2, 0), (0, 1)]:
            empty[3 * head + tail] = [{
                "entity_1": [(m["start"], m["end"]) for m in vertices[head]["mentions"]],
                "entity_2": [(m["start"], m["end"]) for m in vertices[tail]["mentions"]],
                "score": 0.7, "relation_type": "VISITED",
            }]
        self.assertEqual(glidre.project(empty, vertices, {"VISITED": "P1"}), [
            {"h": 0, "t": 1, "r": "P1", "score": 0.7},
            {"h": 2, "t": 0, "r": "P1", "score": 0.7},
        ])

    def test_invalid_spans_and_prompt_collisions(self):
        for span in ([-1, 1], [0, 0], [2, 4], [False, 1]):
            with self.subTest(span=span):
                doc = docred_document()
                doc["vertexSet"][0][0]["pos"] = span
                with self.assertRaisesRegex(ValueError, "invalid mention span"):
                    glidre.prepare(doc)
        with self.assertRaisesRegex(ValueError, "distinct text prompt"):
            glidre.schema_labels({"P1": "LOCATED_IN", "P2": "LOCATED_IN"})

    def test_decode_must_match_original_directed_endpoint(self):
        _, vertices = glidre.prepare(docred_document())
        predictions = [[] for _ in range(9)]
        predictions[2] = [{"entity_1": [(5, 6)], "entity_2": [(0, 0), (3, 3)]}]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            glidre.project(predictions, vertices, {})


if __name__ == "__main__":
    unittest.main()
