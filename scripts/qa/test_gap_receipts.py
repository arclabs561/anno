"""The public receipt must never contain the ordered source records used to calculate it."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import gap_receipts


ROOT = Path(__file__).resolve().parents[2]


def fixture(name):
    return json.loads((ROOT / "docs/eval" / f"{name}.json").read_text())


class GapReceipts(unittest.TestCase):
    def test_stage_rows_and_paths_only_affect_digests(self):
        detail = fixture("native-gap-loss")
        row = {"id": "local-marker", "selected_path_chars": [[0, 1], [5, 7]]}
        detail.update(rows=[row], false_links=[row])
        result = gap_receipts.stages(detail)
        self.assertNotIn("local-marker", json.dumps(result))
        self.assertEqual(result["pair_rows_sha256"], gap_receipts.digest([row]))
        self.assertEqual(result["false_link_paths_sha256"], gap_receipts.digest([row]))

    def test_candidate_rows_stay_local(self):
        detail = fixture("native-gap-possessives")
        detail.update(changed_target_rows=[{"id": "local-marker"}])
        detail["candidate"]["per_id"] = ["local-marker"]
        result = gap_receipts.possessives(detail)
        self.assertNotIn("local-marker", json.dumps(result))
        self.assertEqual(result["changed_target_count"], 1)
        self.assertEqual(result["changed_target_rows_sha256"], gap_receipts.digest(detail["changed_target_rows"]))

    def test_decision_receipt_omits_labels_refusal_ids_and_changes(self):
        detail = fixture("gap-decision-audit")
        snapshot = ["local-marker", "00000"]
        for panel in detail["panels"].values():
            panel.update(ids=["local-marker"], gold_pairs=["10", "00"])
            for arm in panel["arms"].values():
                arm["per_id"] = snapshot
                arm["refusals"] = {"local-marker": "supplied_span_splits_token"}
                arm["empty_successes"] = 1
        for arm in detail["article_holdout_candidate"]["arms"].values():
            arm["changed_observations"] = [{"id": "local-marker"}]
        result = gap_receipts.decisions(detail)
        self.assertNotIn("local-marker", json.dumps(result))
        panel = result["panels"]["published"]
        self.assertNotIn("gold_pairs", panel)
        self.assertEqual(panel["gold_pairs_sha256"], gap_receipts.digest(["10", "00"]))
        self.assertNotEqual(gap_receipts.digest(snapshot), gap_receipts.digest(list(reversed(snapshot))))

    def test_details_require_a_separate_external_file(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "summary.json"
            detail = Path(tmp) / "private.json"
            for destination in (output, ROOT / "docs/row-details.json"):
                with self.assertRaises(ValueError):
                    gap_receipts.write(output, {"count": 1}, {"id": "local-marker"}, destination)
            gap_receipts.write(output, {"count": 1}, {"id": "local-marker"}, detail)
            self.assertEqual(json.loads(output.read_text()), {"count": 1})
            self.assertEqual(json.loads(detail.read_text()), {"id": "local-marker"})
