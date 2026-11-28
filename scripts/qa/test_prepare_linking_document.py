import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_linking_document import prepare


class LinkingDocument(unittest.TestCase):
    def setUp(self):
        self.text = "📰 Ada met Ada. She left."
        self.doc = dict(id=23, text=self.text, language="en")
        digest = hashlib.sha256(self.text.encode()).hexdigest()
        def mention(start, end):
            return dict(start=start, end=end, text=self.text[start:end])
        self.ner = {"source_map": {"source_sha256": digest}, "entities": [
            {**mention(2, 5), "id": "first", "type": "PER", "confidence": .99},
            {**mention(10, 13), "id": "second", "type": "PER", "confidence": .98}]}
        self.coref = {"source_sha256": digest, "offset_unit": "unicode_scalar",
                      "source_length": len(self.text), "clusters": [
            {"id": "c1", "mentions": [mention(2, 5), mention(15, 18), {"tokens": []}]},
            {"id": "c2", "mentions": [mention(6, 9)]}]}

    def test_name_collision_and_unanchored_mentions(self):
        doc = prepare(self.doc, self.ner, self.coref)
        self.assertEqual(doc["id"], 23)
        self.assertEqual(doc["text"], self.text)
        cluster, singleton = doc["entities"]
        self.assertEqual(cluster["corpipe_id"], "c1")
        self.assertEqual([(m["text"], m["start"]) for m in cluster["mentions"]], [("Ada", 2), ("She", 15)])
        self.assertEqual(cluster["mentions"][0]["ner_ids"], ["first"])
        self.assertEqual(singleton["id"], "ner:second")
        self.assertEqual(singleton["mentions"][0]["start"], 10)

    def test_stale_results_are_rejected(self):
        self.ner["source_map"]["source_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "NER source"):
            prepare(self.doc, self.ner, self.coref)
        self.ner["source_map"]["source_sha256"] = self.coref["source_sha256"]
        self.coref["clusters"][0]["mentions"][0]["start"] += 1
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            prepare(self.doc, self.ner, self.coref)


if __name__ == "__main__":
    unittest.main()
