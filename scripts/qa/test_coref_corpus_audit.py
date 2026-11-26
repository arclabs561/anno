"""Check the boundary between facilitator judgments, private model inputs and diagnostics."""

import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import coref_corpus
from coref_corpora import corefud, gap, gum, litbank, phrase_detectives, winogender


class CrossCorpusAudit(unittest.TestCase):
    def setUp(self):
        self.case = {
            "id": "opaque", "group": "private-source-identifier", "kind": "choice",
            "text": "Mina saw Jo. She waved at Mina.", "spans": {"A": [0, 4], "B": [9, 11], "target": [13, 16]},
            "gold": "A", "tags": ["correct_nearer"], "aliases": [[0, 4]],
            "boundaries": {"head": {"A": [0, 4]}},
        }

    def test_model_projection_ignores_facilitator_fields_and_future_metadata(self):
        enriched = dict(self.case, new_private_metadata="publisher locator")
        self.assertEqual(coref_corpus.inputs([enriched]), [{"id": "opaque", "text": self.case["text"]}])

    def test_offered_gold_link_is_not_recounted_as_alternate(self):
        text = self.case["text"]
        bounds = [self.case["spans"][key] for key in ("target", "A")]
        row = {"id": self.case["id"], "source_sha256": coref_corpus.source_sha256(text),
               "mentions": [{"start": a, "end": b, "surface": text[a:b]} for a, b in bounds],
               "clusters": [bounds]}
        result = coref_corpus.score_case(self.case, row)["boundaries"]["default"]
        self.assertEqual(result["observed"], "A")
        self.assertEqual(result["alternate_gold_in_window"], 0)
        self.assertIsNone(result["linked_alternate_gold"])
        other = [text.rindex("Mina"), text.rindex("Mina") + 4]
        case = dict(self.case, aliases=[bounds[1], other])
        counted = coref_corpus.score_case(case, row)["boundaries"]["default"]
        self.assertEqual(counted["alternate_gold_in_window"], 1)
        self.assertFalse(counted["linked_alternate_gold"])
        row["mentions"].append({"start": other[0], "end": other[1], "surface": "Mina"})
        row["clusters"][0].append(other)
        self.assertTrue(coref_corpus.score_case(case, row)["boundaries"]["default"]["linked_alternate_gold"])

    def test_registry_matches_enforced_source_pins(self):
        sources = json.loads(coref_corpus.REGISTRY.read_bytes())
        self.assertEqual(sources.keys(), coref_corpus.READERS.keys())
        self.assertEqual(sources["litbank"]["selected_brat_manifest_sha256"], litbank.SOURCE_SHA256)
        self.assertEqual(sources["phrase-detectives"]["archive_sha256"], phrase_detectives.ARCHIVE_SHA256)
        self.assertEqual(sources["gap"]["input_sha256"], gap.SOURCE_SHA256)
        self.assertEqual(sources["gum"]["input_sha256"], gum.SOURCE_SHA256)
        for name, spec in [("dutch-openboek", corefud.DUTCH), ("french-litbankfr", corefud.FRENCH)]:
            self.assertEqual(sources[name]["input_sha256"], spec.sha256)
        self.assertEqual(sources["winogender"]["member_sha256"],
                         {"data/" + name: digest for name, digest in winogender.PINS.items()})

    def test_score_cli_errors_do_not_publish_untrusted_model_fields(self):
        canary = self.case["text"]
        normal = {"id": "opaque", "source_sha256": coref_corpus.source_sha256(canary), "clusters": []}
        malformed = [
            [dict(normal, mentions=[{"start": canary, "end": 4, "surface": "Mina"}])],
            [{"id": canary}, {"id": canary}],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.json"
            args = ["coref_corpus.py", "score", "gap", "--source", tmp, "--predictions", str(path)]
            for rows in malformed:
                with self.subTest(rows=rows), patch.object(sys, "argv", args), \
                        patch.object(gap, "cases", return_value=[self.case]):
                    path.write_text(json.dumps({"results": rows}), encoding="utf-8")
                    with self.assertRaises(SystemExit) as error:
                        runpy.run_path(coref_corpus.__file__, run_name="__main__")
                    self.assertNotIn(canary, str(error.exception))


if __name__ == "__main__":
    unittest.main()
