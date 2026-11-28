"""Check NorNE coordinate projection, paired identities and published denominators."""

import json
import unittest

import norne_source_replay as replay


def block():
    rows = [
        ("Å", "SpaceAfter=No|name=B-PER"),
        (",", "name=O"),
        ("𐐀", "name=B-GPE_LOC|SpaceAfter=No"),
        (".", "name=O"),
    ]
    return "\n".join(
        [
            "# sent_id = invented",
            "# text = Å, 𐐀.",
            *(
                "\t".join([str(i), word, *(["_"] * 7), attrs])
                for i, (word, attrs) in enumerate(rows, 1)
            ),
        ]
    )


class NorneReplayTest(unittest.TestCase):
    def test_authored_unicode_spans_and_inserted_separators(self):
        case = replay.sentence(block())
        self.assertEqual(case.source_gold, [(0, 1, "PER"), (3, 4, "GPE_LOC")])
        self.assertEqual(case.joined_gold, [(0, 1, "PER"), (4, 5, "GPE_LOC")])
        self.assertEqual(case.added_spaces, [1, 5])
        self.assertEqual(
            [case.on_source(span) for span in case.joined_gold], case.source_gold
        )
        with self.assertRaisesRegex(ValueError, "space between"):
            replay.sentence(block().replace("SpaceAfter=No|name=B-PER", "name=B-PER"))
        with self.assertRaisesRegex(ValueError, "Orphan"):
            replay.sentence(block().replace("B-GPE_LOC", "I-GPE_LOC"))

    def test_pairing_counts_multiplicity_and_distinguishes_actual_type_changes(self):
        case = replay.sentence(block())
        before = [(0, 1, "PER"), (4, 5, "PER"), (4, 5, "PER")]
        after = [(0, 1, "LOC"), (3, 4, "PER")]
        self.assertEqual(
            replay.compare_predictions(case, before, after),
            {
                "retained_same_offsets": 0,
                "retained_shifted_offsets": 1,
                "added": 1,
                "removed": 2,
            },
        )
        totals = replay.summarize(
            [case],
            [replay.Observation(before, {0})],
            [replay.Observation(after, set())],
        )
        self.assertEqual(totals["paired_gold"], {"retained": 0, "gained": 0, "lost": 1})
        self.assertEqual(
            totals["paired_gold_loss_on_source"], {"wrong_type_at_exact_boundary": 1}
        )
        self.assertEqual(
            totals["after_gold_by_type"]["GPE_LOC"]["types_at_exact_boundary"],
            {"PER": 1},
        )

    def test_miss_diagnosis_counts_gold_once_and_uses_unicode_punctuation(self):
        gold = (1, 2, "PER")
        self.assertEqual(
            replay.miss_kind("«Å»", gold, [(0, 3, "PER")]),
            "same_type_extended_by_punctuation_only",
        )
        self.assertEqual(
            replay.miss_kind("«Å»", gold, [(0, 3, "PER"), (1, 2, "ORG")]),
            "wrong_type_at_exact_boundary",
        )
        self.assertEqual(
            replay.miss_kind("«Åx", gold, [(0, 3, "PER")]), "other_overlap"
        )
        self.assertEqual(replay.miss_kind("«Å»", gold, [(2, 3, "LOC")]), "no_overlap")

    def test_complete_split_receipts_reconcile_with_native_errors(self):
        docs = replay.ROOT / "docs/eval"
        splits = json.loads((docs / "norne-source-replay.json").read_bytes())["splits"]
        self.assertEqual(sum(p["cases"] for p in splits.values()), 37_620)
        self.assertEqual(sum(p["gold"] for p in splits.values()), 28_406)
        for p in splits.values():
            self.assertEqual(
                (p["source_text_exact_after"], p["source_gold_exact_after"]),
                (p["cases"], p["gold"]),
            )
            gold = p["paired_gold"]
            self.assertEqual(gold["retained"] + gold["lost"], p["before"]["correct"])
            self.assertEqual(gold["retained"] + gold["gained"], p["after"]["correct"])
            self.assertEqual(
                gold["lost"], sum(p["paired_gold_loss_on_source"].values())
            )
            pred = p["paired_predictions"]
            retained = pred["retained_shifted_offsets"] + pred["retained_same_offsets"]
            self.assertEqual(retained + pred["removed"], p["before"]["predicted"])
            self.assertEqual(retained + pred["added"], p["after"]["predicted"])
            for kind in p["after_gold_by_type"].values():
                self.assertEqual(
                    kind["gold"], kind["correct"] + sum(kind["missed"].values())
                )
        summary = json.loads((docs / "norne-test-errors.json").read_bytes())
        self.assertEqual(summary["reports"], 3450)
        self.assertEqual(summary["entities"]["exact"], 669)
        counts = {
            (r["gold"], r["predicted"]): r["pairs"]
            for r in summary["same_span_label_confusions"]
        }
        self.assertEqual(counts["GPE_LOC", "PER"], 319)
        self.assertEqual(counts["LOC", "PER"], 128)


if __name__ == "__main__":
    unittest.main()
