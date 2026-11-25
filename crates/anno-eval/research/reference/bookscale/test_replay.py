from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from run import replay


class BookReplayTests(unittest.TestCase):
    def test_other_annotations_cannot_claim_the_pinned_release(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "test.jsonl"
            data.write_text('{"doc_key":"different","clusters":[]}\n', encoding="utf-8")
            result = subprocess.run(
                [sys.executable, Path(__file__).with_name("run.py"), data],
                capture_output=True, text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected the pinned BookCoref test annotations", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_long_return_can_be_retained_but_ranked_outside_top_four(self):
        # A is frequent; four intervening people make A fifth by recency.
        doc = {"clusters": [
            [[0, 0], [1, 1], [2, 2], [2000, 2000]],
            [[3, 3]], [[4, 4]], [[5, 5]], [[6, 6]],
        ]}
        retained = replay(doc, 5)["gap_ge_1024"]
        self.assertEqual(retained["returning"], 1)
        self.assertEqual(retained["retained"], 1)
        self.assertEqual(retained["recency_at_4"], 0)
        self.assertEqual(retained["frequency_at_4"], 1)

        evicted = replay(doc, 4)["gap_ge_1024"]
        self.assertEqual(evicted["returning"], 1)
        self.assertEqual(evicted["retained"], 0)
        self.assertEqual(evicted["frequency_at_4"], 0)

    def test_mentions_ending_together_share_one_prefix(self):
        doc = {"clusters": [[[0, 2], [1, 2], [4, 4]]]}
        counts = replay(doc, 1)["all"]
        self.assertEqual(counts["new"], 2)
        self.assertEqual(counts["returning"], 1)


if __name__ == "__main__":
    unittest.main()
