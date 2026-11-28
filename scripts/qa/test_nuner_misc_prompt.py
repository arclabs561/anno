import json
import unittest
from pathlib import Path

from nuner_fallback_routing import text_sha
from nuner_misc_prompt import paired, predictions, score


def worker(text, spans, ran=True):
    return {"input_sha256": text_sha(text), "ran": ran, "spans": spans}


class NuNERMiscPromptTest(unittest.TestCase):
    def test_exact_boundary_recovery_and_displaced_types_use_paired_mentions(self):
        text = "Ada saw Zoë"
        cases = [(text, [(0, 3, "PER"), (8, 11, "MISC")])]
        old = worker(text, [(0, 3, "PER"), (8, 11, "ORG")])
        new = worker(text, [(0, 3, "miscellaneous"), (8, 11, "miscellaneous")])
        result = score(cases, [old], [new], {"PER": "PER", "MISC": "MISC"})
        self.assertEqual(result["stock_mislabeled_misc"]["corrected"], 1)
        self.assertEqual(result["by_gold_label"]["PER"]["lost"], 1)
        self.assertEqual(result["by_gold_label"]["MISC"]["gained"], 1)
        self.assertEqual(
            result["new_exact_misc_by_stock_overlap"]["same_span_other_type"], 1
        )
        self.assertEqual(
            result["candidate_misc_reference_overlap"]["same_span_other_type"], 1
        )
        self.assertEqual(result["candidate_mapped_unmatched"], 1)
        without_misc = score(
            [(text, [(0, 3, "person")])], [old], [new], {"person": "PER"}
        )
        self.assertNotIn("stock_mislabeled_misc", without_misc)
        self.assertEqual(without_misc["candidate_misc_predictions"], 2)
        self.assertEqual(without_misc["candidate_mapped_unmatched"], 0)

    def test_input_identity_collision_and_unchanged_route(self):
        with self.assertRaisesRegex(ValueError, "text or order"):
            predictions(worker("changed", []), "Ada")
        with self.assertRaisesRegex(ValueError, "collide"):
            predictions(worker("Ada", [(0, 3, "misc"), (0, 3, "miscellaneous")]), "Ada")
        with self.assertRaisesRegex(ValueError, "skipped-sentence"):
            score(
                [("Ada", [])],
                [worker("Ada", [], False)],
                [worker("Ada", [(0, 3, "PER")], False)],
                {"PER": "PER"},
            )

    def test_paired_chart_separates_misc_gains_from_displaced_mapped_labels(self):
        text = "Ada saw Zoë"
        cases = [(text, [(0, 3, "PER"), (8, 11, "MISC")])]
        old = worker(text, [(0, 3, "PER"), (8, 11, "ORG")])
        new = worker(text, [(0, 3, "miscellaneous"), (8, 11, "miscellaneous")])
        projection = {t: t for t in ["MISC", "PER", "ORG", "LOC"]}
        wiki = score(cases, [old], [new], projection)
        wnut = score([(text, [(0, 3, "person")])], [old], [new], {"person": "PER"})
        all_types, misc, other, heldout = paired({"wikigold": wiki, "wnut_test": wnut})[
            "rows"
        ]
        self.assertEqual((all_types["gained"], all_types["lost"]), (1, 1))
        self.assertEqual((misc["gained"], misc["lost"]), (1, 0))
        self.assertEqual(misc["candidate"], {"correct": 1, "unmatched": 1, "calls": 1})
        self.assertEqual((other["gained"], other["lost"]), (0, 1))
        self.assertEqual(
            heldout["candidate"], {"correct": 0, "unmatched": 0, "calls": 1}
        )

    def test_cluster_bootstrap_preserves_repeated_occurrences_in_one_source_document(
        self,
    ):
        cases = [("A", [(0, 1, "PER")]), ("A", [(0, 1, "PER")]), ("B", [(0, 1, "PER")])]
        stock = [worker("A", []), worker("A", []), worker("B", [(0, 1, "PER")])]
        candidate = [
            worker("A", [(0, 1, "PER")]),
            worker("A", [(0, 1, "PER")]),
            worker("B", []),
        ]
        result = score(
            cases,
            stock,
            candidate,
            {"PER": "PER"},
            group_ids=[1, 1, 2],
            unit="document",
        )
        resampling = result["resampling"]
        self.assertEqual((resampling["groups"], resampling["sentences"]), (2, 3))
        self.assertEqual(
            resampling["histogram"],
            [
                {"sentences": 1, "net_exact": -1, "groups": 1},
                {"sentences": 2, "net_exact": 2, "groups": 1},
            ],
        )
        interval = resampling["intervals"]["all"]
        self.assertEqual((interval["lower"], interval["upper"]), (-3, 3))

    def test_published_receipt_regenerates_chart_and_keeps_its_sampling_coverage(self):
        root = Path(__file__).resolve().parents[2] / "docs/eval"
        receipt = json.loads((root / "nuner-misc-prompt.json").read_text())
        portable = json.loads((root / "nuner-misc-prompt-paired.json").read_text())
        self.assertEqual(paired(receipt), portable)
        self.assertNotEqual(
            receipt["worker_sha256"]["stock"], receipt["worker_sha256"]["candidate"]
        )
        for key, n, gold, calls, clusters in [
            ("wikigold", 1696, 3558, 1383, 145),
            ("wnut_test", 1287, 1079, 992, 1287),
        ]:
            result = receipt[key]
            self.assertEqual(
                (result["sentences"], result["gold"], result["nuner_calls"]),
                (n, gold, calls),
            )
            sample = result["resampling"]
            self.assertEqual(sample["groups"], clusters)
            self.assertEqual(sum(c["groups"] for c in sample["histogram"]), clusters)
            self.assertEqual(
                sum(c["groups"] * c["sentences"] for c in sample["histogram"]), n
            )
            self.assertEqual(
                sum(c["groups"] * c["net_exact"] for c in sample["histogram"]),
                result["exact_gained"] - result["exact_lost"],
            )
            for arm in ["stock", "candidate"]:
                self.assertEqual(
                    result[f"{arm}_mapped_suggestions"],
                    result[f"{arm}_mapped_unmatched"] + result[f"{arm}_exact"],
                )
                self.assertLessEqual(
                    result[f"{arm}_mapped_suggestions"], result[f"{arm}_suggestions"]
                )
        target = receipt["wikigold"]["stock_mislabeled_misc"]
        self.assertEqual(target["stock_same_bounds_wrong_type"], 263)
        self.assertEqual(
            sum(
                target[k]
                for k in (
                    "corrected",
                    "same_bounds_wrong_type_after",
                    "boundaries_missing_after",
                )
            ),
            263,
        )
        self.assertNotIn("stock_mislabeled_misc", receipt["wnut_test"])
