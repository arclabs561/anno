import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_domain_fixtures as panel
from check_span_assertions import load_cases


class DomainFixturesTest(unittest.TestCase):
    def test_bio_and_slot_inputs_keep_character_offsets_and_label_scope(self):
        row = panel.sentence("science", "seed", ["🦊", "São", "Paulo", "ESA"],
                             ["O", "I-location", "I-location", "B-university"])
        self.assertEqual((row.text[row.start:row.end], row.start, row.end, row.label),
                         ("São Paulo", 2, 11, "LOC"))
        with self.assertRaises(ValueError):
            panel.sentence("science", "seed", ["orphan"], [])
        record, = panel.records("atis", "seed", ["to new york", "O B-toloc.city_name I-toloc.city_name"])
        self.assertEqual((record.text[record.start:record.end], record.label), ("new york", "LOC"))
        self.assertIsNone(panel.sentence("snips", "s", ["band"], ["B-artist"]))

    def test_contract_preserves_source_window_and_rejects_misaligned_gold(self):
        text = "🦊" * 301 + "Acme Inc." + "x" * 301
        answer = {"answer_start": 301, "text": "Acme Inc."}
        doc = {"title": "agreement", "paragraphs": [{"context": text, "qas": [
            {"id": "a__Parties", "answers": [answer]}]}]}
        row = panel.contract(doc, "s")
        self.assertEqual((row.start, row.end, row.label, len(row.text)), (250, 259, "ORG", 509))
        self.assertEqual(row.text[row.start:row.end], answer["text"])
        answer["answer_start"] += 1
        with self.assertRaisesRegex(ValueError, "disagrees"):
            panel.contract(doc, "s")

    def test_budget_duplicates_shortfall_and_verified_local_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Both copies from AI and the copy from music have exactly the same model input.
            corpora = {"ai.bio": "Ada I-person\n\nAda I-person\n\nZoë B-person\n\none O\n",
                       "music.bio": "Ada B-person\n\nJazz B-misc\n\nBand B-organisation\n"}
            pins = {}
            for name, text in corpora.items():
                raw = text.encode()
                (root / name).write_bytes(raw)
                pins[name] = {"sha256": panel.digest(raw)}
            kwargs = dict(seed="fixed", per_split=1, domains=["ai", "music"], pins=pins)
            a, b = root / "a", root / "b"
            with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
                manifest = panel.prepare(root, a, **kwargs)
                panel.prepare(root, b, **kwargs)
            self.assertEqual(manifest["groups"]["ai"]["source_records"], 4)
            self.assertEqual(manifest["groups"]["ai"]["eligible_records"], 3)
            self.assertEqual(manifest["groups"]["music"]["shortfall"], 1)
            self.assertEqual(manifest["groups"]["music"]["unique_eligible"], 1)
            unique = set()
            for name in ("discovery", "reserved"):
                data = (a / (name + ".json")).read_bytes()
                self.assertEqual(data, (b / (name + ".json")).read_bytes())
                self.assertEqual(panel.digest(data), manifest[name + "_sha256"])
                cases = load_cases(json.loads(data))
                for row in cases.values():
                    hashed = panel.digest(row["text"].encode())
                    self.assertEqual(row["source_sha256"], hashed)
                    self.assertNotIn(hashed, unique)
                    unique.add(hashed)
            self.assertEqual(len(unique), 3)
            with self.assertRaises(FileExistsError):
                panel.prepare(root, a, **kwargs)
            (root / "music.bio").write_text("changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                panel.prepare(root, root / "bad", **kwargs)
            self.assertFalse((root / "bad").exists())

    def test_download_rejects_wrong_content_before_publishing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "origin"
            source.write_bytes(b"correct")
            pins = {"test.bio": {"url": source.as_uri(), "download_bytes": 7,
                                 "sha256": panel.digest(b"correct")}}
            panel.fetch(root / "cache", pins)
            self.assertEqual((root / "cache/test.bio").read_bytes(), b"correct")
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                panel.fetch(root / "other", pins)
            self.assertFalse((root / "other/test.bio").exists())
