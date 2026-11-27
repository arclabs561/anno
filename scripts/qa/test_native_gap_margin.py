import csv
import json
from pathlib import Path
import tempfile
import unittest

import native_gap_margin as trial


def row():
    return dict(ID="original", Text="Paz Ren he", Pronoun="he", A="Paz", B="Ren", URL="fiction",
                **{"Pronoun-offset": "8", "A-offset": "0", "B-offset": "4", "A-coref": "TRUE", "B-coref": "FALSE"})


def trace(scored, selected, groups):
    return {"id": "original", "max_distance": 100,
            "members": [[0, 3, groups[0]], [4, 7, groups[1]], [8, 10, groups[2]]],
            "decisions": [{}, {}, {"candidates": scored, "selected": selected}]}


class NativeMarginTests(unittest.TestCase):
    def test_rival_can_veto_above_threshold(self):
        observed = trial.lookup(row(), trace([(1, .42, .225), (0, .55, .225)], 0, [0, 1, 0]))
        self.assertEqual(observed.issued, 0)
        self.assertIsNone(observed.gated(.15))
        self.assertGreater(observed.above_threshold, .15)
        self.assertEqual(observed.gated(.125), 0)

    def test_strict_threshold_ties_and_ambiguous_membership(self):
        rejected = trial.lookup(row(), trace([(1, .20, .225), (0, .225, .225)], None, [0, 1, 2]))
        self.assertEqual(rejected.gold_stage, "gold_first_below_or_at_threshold")
        tied = trial.lookup(row(), trace([(1, .5, .225), (0, .5, .225)], 1, [1, 2, 2]))
        self.assertEqual((tied.direct, tied.gold_stage, tied.gated(.025)), (1, "gold_lost_tie", None))
        both = trial.lookup(row(), trace([(1, .5, .225), (0, .25, .225)], 1, [0, 0, 0]))
        self.assertEqual((both.group_arity, both.direct, both.issued, both.gated(0)), (2, 1, None, None))

    def test_policy_uses_unchanged_development_comparison(self):
        root = trial.COMPARISON.parent
        receipt = json.loads((root / "native-gap-margin-development.json").read_bytes())
        commitment = json.loads(trial.COMPARISON.read_bytes())
        pin = trial.file_hash(root / "native-gap-margin-policy.json")
        self.assertEqual(trial.comparison_margin(receipt, pin, commitment), .15)
        receipt["development_grid"][0]["wrong"] = 0
        with self.assertRaisesRegex(ValueError, "commitment"):
            trial.comparison_margin(receipt, pin, commitment)

    def test_panel_requires_unique_rows_and_complete_columns(self):
        example = row()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "source.tsv"
            for fields, count, message in ((list(example), 2, "duplicate ID"),
                                           (list(example)[:-1], 1, "GAP column")):
                with path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fields, delimiter="\t", extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows([example] * count)
                with self.assertRaisesRegex(ValueError, message):
                    trial.read_panel(path)


if __name__ == "__main__":
    unittest.main()
