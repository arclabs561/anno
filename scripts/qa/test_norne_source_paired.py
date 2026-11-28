"""Keep the NorNE figures tied to the same two complete publisher tests."""

import json
import unittest
from collections import Counter

from norne_source_paired import EVAL, paired


class NorneFigureTest(unittest.TestCase):
    def test_aggregate_errors_reconcile_with_both_independent_source_reads(self):
        study = json.loads((EVAL / "norne-source-replay.json").read_bytes())
        chart = json.loads((EVAL / "norne-test-errors.json").read_bytes())
        panels = [study["splits"][f"{code}-test"] for code in ("nob", "nno")]
        self.assertEqual(chart["reports"], sum(p["cases"] for p in panels))
        self.assertEqual(chart["prediction_source"], {"kind": "model", "name": "heuristic"})
        for field, native in (("exact", "correct"), ("gold_count", "gold"), ("predicted_count", "predicted")):
            self.assertEqual(chart["entities"][field], sum(p["after"][native] for p in panels))
        confusions = Counter()
        for panel in panels:
            for gold, counts in panel["after_gold_by_type"].items():
                for predicted, n in counts["types_at_exact_boundary"].items():
                    if predicted != gold:
                        confusions[gold, predicted] += n
        self.assertEqual(
            confusions,
            {(c["gold"], c["predicted"]): c["pairs"] for c in chart["same_span_label_confusions"]},
        )

    def test_paired_chart_rejects_gains_that_do_not_reconcile(self):
        study = json.loads((EVAL / "norne-source-replay.json").read_bytes())
        self.assertEqual(paired(study), json.loads((EVAL / "norne-source-paired.json").read_bytes()))
        study["splits"]["nob-test"]["paired_gold"]["gained"] += 1
        with self.assertRaisesRegex(ValueError, "Bokmål"):
            paired(study)


if __name__ == "__main__":
    unittest.main()
