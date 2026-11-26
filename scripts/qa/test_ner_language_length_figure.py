"""Check that the paired figure uses compatible test cohorts and mention counts."""

import importlib.util
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("matplotlib"), "figure requires Matplotlib"
)
class PairedFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = Path(__file__).resolve().parents[1] / "plot_ner_language_length.py"
        module = runpy.run_path(str(script))
        cls.load_panels = staticmethod(module["read_panels"])
        cls.receipt = module["RECEIPT"]

    def test_frozen_twelve_input_panels_have_separate_denominators(self):
        totals = {
            name: (
                len(rows),
                sum(row.gold for row in rows),
                sum(row.gained for row in rows),
                sum(row.lost for row in rows),
                sum(row.lower > 0 for row in rows),
            )
            for name, _, _, _, rows in self.load_panels(self.receipt)
        }
        self.assertEqual(
            totals, {"RED-FM": (10, 436, 175, 13, 10), "MultiNERD": (4, 68, 30, 8, 1)}
        )

    def test_inconsistent_paired_gain_fails(self):
        receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
        corpus = next(
            data
            for revision, data in receipt["corpora"].items()
            if revision.startswith("Babelscape/multinerd@")
        )
        report = corpus["panels"]["12"]["reports"]["sauerkraut/fp32"]
        report["paired_vs_heuristic"]["ru/long"]["correct_gained"] += 1
        with TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "paired match counts"):
                self.load_panels(path)

    def test_missing_panel_identity_or_fallback_runtime_fails(self):
        receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
        panel = next(iter(receipt["corpora"].values()))["panels"]["12"]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            digest = panel.pop("panel_sha256")
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "panel digest"):
                self.load_panels(path)
            panel["panel_sha256"] = digest
            panel["reports"]["sauerkraut/fp32"]["observed_runtimes_by_call"][
                "fallback"
            ] = 1
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scored calls"):
                self.load_panels(path)


if __name__ == "__main__":
    unittest.main()
