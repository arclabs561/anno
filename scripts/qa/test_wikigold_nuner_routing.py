"""Contract tests for WikiGold input coverage, paired policy scores and chart data."""

from collections import Counter
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import wikigold_nuner_routing as study


class WikiGoldRoutingTest(unittest.TestCase):
    def test_markers_starts_unicode_and_repeated_empty_reference_sentences(self):
        data = "Zé I-PER\nHo I-PER\n🧭 I-MISC\n\n-DOCSTART- O\n\nx O\n\nx O\n\n-DOCSTART- O\n"
        cases, info = study.parse_release(data)
        self.assertEqual(cases[0], ("Zé Ho 🧭", [(0, 5, "PER"), (6, 7, "MISC")]))
        self.assertEqual(info["documents_with_sentences"], 2)
        self.assertEqual(info["document_markers_excluded"], 2)
        self.assertEqual(info["sentence_occurrences"], 3)
        self.assertEqual(info["distinct_sentence_texts"], 2)
        self.assertEqual(info["gold_empty_sentences"], 2)
        self.assertEqual(
            (info["tokens"], info["tagged_tokens"], info["gold_spans"]), (5, 3, 2)
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "modified"
            path.write_text(data)
            with self.assertRaisesRegex(ValueError, "pinned corpus"):
                study.load_release(path)

    def test_misc_score_is_in_scope_and_unfamiliarity_requires_training_names(self):
        counts = Counter()
        by_label = {label: Counter() for label in study.PROJECTION}
        study.routing.add_case(
            counts,
            "A B",
            {(0, 1, "MISC")},
            [(0, 1, "ORG"), (2, 3, "URL")],
            [(0, 1, "MISC"), (2, 3, "PER")],
            None,
            study.PROJECTION,
            by_label,
        )
        self.assertEqual(counts["exact_gained"], 1)
        self.assertEqual(counts["stock_outside_schema"], 1)
        self.assertEqual(by_label["MISC"]["exact_gained"], 1)
        self.assertEqual(by_label["ORG"]["unmatched_removed"], 1)
        self.assertEqual(by_label["PER"]["unmatched_added"], 1)
        self.assertNotIn("unfamiliar_gold", counts)

    def test_repeated_timing_rejects_changed_stock_output(self):
        args = SimpleNamespace(stock="s", candidate="c", checkpoint="w")
        empty = {"ran": False, "spans": [], "seconds": 0.001}
        hit = {"ran": True, "spans": [[0, 3, "PER"]], "seconds": 0.12}
        with redirect_stderr(StringIO()), patch.object(
            study.routing,
            "run_worker",
            side_effect=[[empty], [hit], [{**empty, "spans": hit["spans"]}]],
        ):
            with self.assertRaisesRegex(ValueError, "repeated stock timing"):
                study.routing.measure(
                    "control",
                    [("Ada", {(0, 3, "PER")})],
                    None,
                    args,
                    [empty],
                    None,
                    projection=study.PROJECTION,
                    repeat_skipped_timing=True,
                )

    def test_lowercase_bert_misc_scores_in_schema_and_alias_collisions_fail(self):
        gold = {(0, 1, "MISC")}
        counts = {"misc": {"emitted": 0, "exact_after_mapping": 0}}
        normalized = study.routing.map_prediction_labels(
            [(0, 1, "misc"), (2, 3, "misc"), (4, 5, "URL")],
            study.PREDICTION_ALIASES,
            gold,
            counts,
        )
        self.assertEqual(normalized, [(0, 1, "MISC"), (2, 3, "MISC"), (4, 5, "URL")])
        self.assertEqual(counts["misc"], {"emitted": 2, "exact_after_mapping": 1})
        with self.assertRaisesRegex(ValueError, "Projected prediction collision"):
            study.routing.map_prediction_labels(
                [(0, 1, "misc"), (0, 1, "MISC")], study.PREDICTION_ALIASES, gold, counts
            )

    def test_published_chart_and_label_changes_match_the_measured_arms(self):
        wiki = json.loads((study.EVAL / "wikigold-nuner-routing.json").read_text())
        wnut = json.loads(study.STUDY.read_text())
        chart = json.loads((study.EVAL / "wikigold-routing-tradeoff.json").read_text())
        self.assertEqual(study.tradeoff(wiki, wnut), chart)
        self.assertEqual(len(chart["series"]), 4)
        self.assertEqual(
            (chart["schema"], chart["unit"]), ("anno.tradeoff/v1", "sentences")
        )
        for item, measured in zip(
            chart["series"],
            (
                trial[stack]
                for trial in (wnut["splits"]["Test"], wiki["arms"])
                for stack in ("without_bert", "with_bert")
            ),
            strict=True,
        ):
            skipped = measured["skipped_subset"]
            self.assertEqual(
                (item["observations"], item["gold"]),
                (measured["stock_skipped"], skipped["gold"]),
            )
            self.assertEqual(
                [(p["correct"], p["unmatched"]) for p in item["points"]],
                [
                    (skipped["stock_exact"], skipped["stock_in_schema_unmatched"]),
                    (
                        skipped["candidate_exact"],
                        skipped["candidate_in_schema_unmatched"],
                    ),
                ],
            )
        self.assertEqual(wiki["source"]["sha256"], study.replay_wikigold.WIKIGOLD)
        self.assertEqual(
            wiki["reference_wnut_receipt_sha256"], study.routing.file_sha(study.STUDY)
        )
        self.assertEqual(wiki["prediction_aliases"], {"misc": "MISC"})
        self.assertGreater(
            wiki["arms"]["with_bert"]["full_split_prediction_alias_counts"]["stock"][
                "misc"
            ]["exact_after_mapping"],
            0,
        )
        for arm in wiki["arms"].values():
            self.assertEqual(arm["all_sentences"], 1696)
            self.assertEqual(arm["stock_skipped"] + arm["stock_routed"], 1696)
            self.assertEqual(arm["full_split"]["gold"], 3558)
            outcomes = arm["skipped_sentence_outcomes"]
            self.assertEqual(
                sum(
                    count
                    for name, count in outcomes.items()
                    if name != "any_exact_loss"
                ),
                arm["stock_skipped"],
            )
            self.assertLessEqual(
                outcomes["any_exact_loss"], arm["skipped_subset"]["exact_lost"]
            )
            for label_key, total_key in (
                ("gold", "gold"),
                ("exact_gained", "exact_gained"),
                ("exact_lost", "exact_lost"),
                ("unmatched_added", "in_schema_unmatched_added"),
                ("unmatched_removed", "in_schema_unmatched_removed"),
            ):
                self.assertEqual(
                    sum(v[label_key] for v in arm["skipped_by_label"].values()),
                    arm["skipped_subset"][total_key],
                )
            self.assertEqual(
                arm["stock_skipped"],
                arm["warm_skipped_subset_latency"]["stock"]["calls"],
            )
        with self.assertRaisesRegex(ValueError, "nuner_threshold"):
            study.tradeoff({**wiki, "nuner_threshold": 0.5}, wnut)


if __name__ == "__main__":
    unittest.main()
