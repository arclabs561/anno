"""Check that the offline GENIA figure stays bound to its measured source splits."""

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    importlib.util.find_spec("matplotlib"), "GENIA figure requires Matplotlib"
)
class GeniaAggregateReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        try:
            import plot_genia_nesting
        finally:
            sys.path.pop(0)
        cls.generator = plot_genia_nesting
        with plot_genia_nesting.LEDGER.open(newline="") as source:
            cls.selected = {
                row["split"]: row
                for row in csv.DictReader(source)
                if row["id"] == "GENIANested"
            }
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


if __name__ == "__main__":
    unittest.main()
