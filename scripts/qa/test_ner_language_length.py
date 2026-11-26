"""Offline checks of the multilingual NER panel's sampling and scoring contracts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ner_language_length import MULTINERD_LENGTHS, counts, digest, paired_delta, select, source_rows, spans
from multinerd_spans import text_and_gold


def row(topic, words):
    return {"uri": topic, "docid": topic, "lan": "de", "text": " ".join(["Änne"] * words),
            "entities": []}


class PanelTests(unittest.TestCase):
    def test_minimum_for_each_cell_and_no_reused_topics(self):
        rows = [row(f"Q{i}", 20 if i < 5 else 80) for i in range(10)]
        used = set()
        dev = select(rows, "dev", "de", used, 2)
        test = select(list(reversed(rows)), "test", "de", used, 2)
        self.assertEqual(len({case.topic for case in dev + test}), 8)
        self.assertEqual({case.cell for case in test}, {"de/short", "de/long"})
        self.assertTrue(all(not case.gold for case in test))
        with self.assertRaisesRegex(ValueError, "unique topics"):
            select(rows, "test", "de", used, 2)
        rows[0]["lan"] = "en"
        with self.assertRaisesRegex(ValueError, "language"):
            select(rows, "dev", "de", set(), 1)

    def test_source_verification_and_external_cache(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "dev.de.jsonl"
            path.write_text(json.dumps(row("Q1", 20)) + "\n")
            with patch.dict("ner_language_length.HASHES", {"dev.de": digest(path)}):
                self.assertEqual(len(source_rows(folder, "dev.de", False)), 1)
                path.write_text("changed")
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    source_rows(folder, "dev.de", False)
            (folder / ".git").touch()
            with self.assertRaisesRegex(ValueError, "external"):
                source_rows(folder / "raw", "dev.de", True)

    def test_multinerd_uses_source_language_and_bio_coordinates(self):
        raw = {"lang": "ru", "tokens": ["Анна", "де", "Ли", "здесь", "и", "там", "не", "везде"],
               "ner_tags": [1, 2, 2, 0, 0, 0, 0, 0]}
        text, annotated = text_and_gold([raw])
        self.assertEqual(annotated, [[0, 10, 0]])
        rows = [{"uri": "a", "docid": 1, "lan": "ru", "text": text, "original": raw}]
        long = dict(raw, tokens=raw["tokens"] * 5, ner_tags=raw["ner_tags"] * 5)
        rows.append({"uri": "b", "docid": 2, "lan": "ru", "text": " ".join(long["tokens"]), "original": long})
        cases = select(rows, "test", "ru", set(), 1, corpus="multinerd", lengths=MULTINERD_LENGTHS)
        self.assertEqual(cases[0].gold, {(0, 10, "PER")})
        self.assertEqual(len(cases[1].gold), 5)

    def test_unicode_span_and_unmatched_predictions(self):
        text = "Éva à Liège"
        source = [{"boundaries": [6, 11], "surfaceform": "Liège", "type": "LOC"}]
        valid = {"start": 6, "end": 11, "text": "Liège", "label": "location"}
        wrong_type = dict(valid, label="person")
        source.append({"boundaries": [3, 3], "surfaceform": "", "type": "ORG"})
        gold = spans(text, source, "boundaries")
        self.assertEqual(len(gold), 1)
        self.assertEqual(counts(gold, spans(text, [valid], "label")), (1, 0, 0))
        self.assertEqual(counts(gold, spans(text, [wrong_type], "label")), (0, 1, 1))
        with self.assertRaisesRegex(ValueError, "slice"):
            spans(text, [dict(valid, end=10)], "label")
        self.assertIsNone(paired_delta([((0, 0, 0), (0, 0, 0))])["points"])


if __name__ == "__main__":
    unittest.main()
