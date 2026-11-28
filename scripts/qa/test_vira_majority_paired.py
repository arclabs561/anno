import json
import unittest

from vira_majority_paired import EVAL, paired


class ViraMajorityPairedTests(unittest.TestCase):
    def setUp(self):
        self.receipt = json.loads((EVAL / "vira-speech-act.json").read_bytes())

    def test_published_held_out_flips_and_abstentions(self):
        rendered = paired(self.receipt)
        self.assertEqual(
            rendered, json.loads((EVAL / "vira-majority-paired.json").read_bytes())
        )
        test, validation = rendered["rows"]
        self.assertEqual((test["gained"], test["lost"]), (16, 3))
        self.assertEqual((validation["gained"], validation["lost"]), (18, 7))
        self.assertEqual(test["observations"] - sum(test["candidate"].values()), 8)

    def test_rejects_a_baseline_chosen_on_test(self):
        self.receipt["splits"]["test"]["training_majority"]["label"] = "query"
        with self.assertRaisesRegex(ValueError, "differs from training majority"):
            paired(self.receipt)


if __name__ == "__main__":
    unittest.main()
