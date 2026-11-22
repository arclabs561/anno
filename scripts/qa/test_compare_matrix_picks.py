import unittest
from copy import deepcopy

from compare_matrix_picks import DATASETS, validate_pairs


class MatchedPanelContract(unittest.TestCase):
    def test_counterbalanced_picks_reject_history_leakage_and_missing_cells(self):
        pairs = [{"prior_low": ds, "seed": 0, "picks": {"random": DATASETS[0], "worst-first": ds}}
                 for ds in DATASETS]
        validate_pairs(pairs, [0])
        for invalid in (pairs[:-1], pairs + pairs[:1]):
            with self.assertRaisesRegex(ValueError, "incomplete or duplicate"):
                validate_pairs(invalid, [0])
        leaked = deepcopy(pairs)
        leaked[1]["picks"]["random"] = DATASETS[1]
        with self.assertRaisesRegex(ValueError, "random control changed"):
            validate_pairs(leaked, [0])
        inactive = deepcopy(pairs)
        inactive[1]["picks"]["worst-first"] = DATASETS[0]
        with self.assertRaisesRegex(ValueError, "adaptive first pick"):
            validate_pairs(inactive, [0])
