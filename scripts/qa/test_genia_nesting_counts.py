"""Check that the offline GENIA figure stays bound to its measured source splits."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


class GeniaAggregateReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        try:
            import genia_grounded_overlap
            import plot_genia_nesting
        finally:
            sys.path.pop(0)
        cls.generator = plot_genia_nesting
        cls.overlap = genia_grounded_overlap
        cls.selected = plot_genia_nesting.release_files()
        cls.snapshot = plot_genia_nesting.LEDGER.with_name("genia-nesting-counts.json")

    def test_aggregate_roundtrip_preserves_all_measurements(self):
        splits = self.generator.read_counts(self.snapshot, self.selected)
        with tempfile.TemporaryDirectory() as out:
            restored = Path(out) / "restored.json"
            self.generator.write_counts(restored, splits, self.selected)
            self.assertEqual(
                json.loads(restored.read_text()), json.loads(self.snapshot.read_text())
            )

    def test_source_and_totals_must_match(self):
        for field in ("source", "tokens_by_active_layer_count", "hidden_mentions"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as out:
                doc = json.loads(self.snapshot.read_text())
                split = doc["splits"]["test"]
                if field == "source":
                    split[field] = doc["splits"]["dev"][field]
                elif field == "tokens_by_active_layer_count":
                    split[field][0] += 1
                else:
                    split[field] += 1
                altered = Path(out) / "altered.json"
                altered.write_text(json.dumps(doc))
                with self.assertRaises(ValueError):
                    self.generator.read_counts(altered, self.selected)

    def test_native_receipt_agrees_with_the_figure_for_each_publisher_split(self):
        figure = self.generator.read_counts(self.snapshot, self.selected)
        native = json.loads(
            self.snapshot.with_name("genia-native-exports.json").read_text()
        )
        self.assertEqual(set(native["splits"]), set(figure))
        for split, counts in figure.items():
            with self.subTest(split=split):
                actual = native["splits"][split]
                self.assertEqual(
                    actual["source"],
                    self.generator.source_identity(self.selected[split]),
                )
                self.assertEqual(actual["gold_mentions"], counts.mentions)
                self.assertEqual(actual["jsonl_rows"], counts.mentions)
                self.assertEqual(actual["html_marks"], counts.highlights)
                self.assertEqual(actual["hidden_mentions"], counts.hidden)
                self.assertEqual(actual["sentences_verified"], counts.sentences)
                self.assertEqual(
                    actual["sentences_with_hidden_mentions"],
                    counts.sentences_with_hidden,
                )
                self.assertLessEqual(
                    actual["sentences_with_hidden_mentions"], counts.hidden
                )
                self.assertLessEqual(
                    actual["sentences_with_hidden_mentions"]
                    + actual["sentences_without_gold_mentions"],
                    actual["sentences_verified"],
                )

    def test_native_comparison_checks_unicode_offsets_escaping_and_occurrences(self):
        tokens = ["🐈", "Café", "&", "Café", "<"]
        spans = [(1, 3, "DNA"), (1, 2, "protein"), (3, 4, "protein")]
        html = '🐈 <mark class="entity entity-DNA">Café &amp;<sup>DNA</sup></mark> <mark class="entity entity-protein">Café<sup>protein</sup></mark> &lt;'
        rows = [
            {"start": 2, "end": 8, "type": "DNA", "text": "Café &", "source": "1"},
            {"start": 2, "end": 6, "type": "protein", "text": "Café", "source": "1"},
            {"start": 9, "end": 13, "type": "protein", "text": "Café", "source": "1"},
        ]
        record = {
            "sentence": 1,
            "html": html,
            "jsonl": "\n".join(map(json.dumps, rows)),
        }
        self.assertEqual(
            self.generator.verify_export(1, tokens, spans, record), (3, 3, 2)
        )
        with self.assertRaisesRegex(ValueError, "JSONL"):
            wrong_occurrence = "\n".join(map(json.dumps, [rows[0], rows[1], rows[1]]))
            self.generator.verify_export(
                1, tokens, spans, {**record, "jsonl": wrong_occurrence}
            )
        with self.assertRaisesRegex(ValueError, "highlighted different"):
            self.generator.verify_export(
                1, tokens, spans, {**record, "html": html.replace("DNA", "protein")}
            )
        with self.assertRaisesRegex(ValueError, "source text"):
            self.generator.verify_export(
                1, tokens, spans, {**record, "html": html.replace("🐈", "dog")}
            )
        empty = {"sentence": 2, "html": "&lt;&amp;&gt;", "jsonl": ""}
        self.assertEqual(self.generator.verify_export(2, ["<&>"], [], empty), (0, 0, 0))

    def test_nested_example_reads_the_offsets_and_each_exporters_records(self):
        rows = [
            {"start": 2, "end": 13, "type": "protein", "text": "cyclophilin"},
            {"start": 2, "end": 18, "type": "DNA", "text": "cyclophilin gene"},
        ]
        sample = self.generator.export_example(
            {
                "sentence": 12,
                "html": '🐈 <mark class="entity entity-DNA">cyclophilin gene<sup>DNA</sup></mark>',
                "jsonl": "\n".join(map(json.dumps, rows)),
            }
        )
        self.assertEqual(
            sample, {"sentence": 12, "html": [rows[1]], "jsonl": [rows[1], rows[0]]}
        )

    def test_grounded_adapter_retains_unicode_occurrences_and_negative_sentences(self):
        doc = self.overlap.grounded_sentence(
            "test-1",
            ["🐈", "Café", "&", "Café"],
            [(1, 3, "DNA"), (1, 2, "protein"), (3, 4, "protein")],
        )
        rows = [
            (s["location"]["Text"], s["surface"], s["label"]) for s in doc["signals"]
        ]
        self.assertEqual(
            rows,
            [
                ({"start": 2, "end": 8}, "Café &", "DNA"),
                ({"start": 2, "end": 6}, "Café", "protein"),
                ({"start": 9, "end": 13}, "Café", "protein"),
            ],
        )
        self.assertEqual([s["id"] for s in doc["signals"]], [0, 1, 2])
        no_gold = self.overlap.grounded_sentence("test-2", ["unmarked", "text"], [])
        self.assertEqual((no_gold["text"], no_gold["signals"]), ("unmarked text", []))

    def test_overlap_chart_agrees_with_the_independent_native_export_receipt(self):
        report = json.loads(
            self.snapshot.with_name("genia-span-overlap.json").read_text()
        )
        native = json.loads(self.overlap.REFERENCE.read_text())
        self.overlap.verify_totals(report, native)
        for field in ("documents", "signals", "documents_with_overlap", "characters"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.overlap.verify_totals({**report, field: report[field] - 1}, native)


if __name__ == "__main__":
    unittest.main()
