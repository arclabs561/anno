"""The HistNERo study's independently reconstructed gold and native costs reconcile."""

import json
import unittest

import histnero_ner_scoring as study


class HistneroScoringTest(unittest.TestCase):
    def test_visible_bounds_exclude_only_empty_edges_and_keep_source_ordinals(self):
        text, source, aligned = study.source_case(
            {
                "tokens": ["", "Ștefan", "", "cel", "", "și", "", "."],
                "ner_tags": [1, 2, 2, 2, 2, 0, 7, 0],
            }
        )
        self.assertEqual(text, " Ștefan  cel  și  .")
        self.assertEqual(source[0], (0, 13, "PERS"))
        self.assertEqual(aligned, {0: (1, 12, "PER")})
        self.assertEqual(source[1], (17, 17, "PROD"))
        with self.assertRaisesRegex(ValueError, "Orphan"):
            study.source_case({"tokens": ["A"], "ner_tags": [2]})

    def test_person_workload_accounts_for_duplicates_confusions_partial_and_absent(
        self,
    ):
        segments = {(0, 3, "PER"), (5, 8, "LOC")}
        counts, types = study.person_workload(
            segments,
            [
                (0, 3, "PER"),
                (0, 3, "PER"),
                (5, 8, "PER"),
                (6, 10, "PER"),
                (20, 21, "PER"),
                (6, 10, "ORG"),
            ],
        )
        self.assertEqual(set(counts.values()), {1})
        self.assertEqual(types, {"LOC": 1})

    def test_published_rollup_restores_source_and_score_denominators(self):
        receipt = json.loads(
            (study.SCRIPTS.parent / "docs/eval/histnero-ner-scoring.json").read_text()
        )
        for s in receipt["splits"].values():
            raw, aligned = (s["scoring"][n] for n in study.MODES)
            effect = s["effects"]
            self.assertEqual(s["source_gold"], raw["gold"])
            self.assertEqual(aligned["gold"] + s["zero_length_gold_spans"], raw["gold"])
            self.assertEqual(raw["predicted"], aligned["predicted"])
            self.assertEqual(
                aligned["correct"] - raw["correct"],
                effect["labels_gained"]
                - effect["labels_lost"]
                + effect["extents_gained"]
                - effect["extents_lost"],
            )
            self.assertEqual(
                sum(s["person_suggestions"].values()),
                s["per_type"]["dataset"]["PER"]["predicted"],
            )
            self.assertEqual(
                s["person_suggestions"]["same_boundary_other_type"],
                sum(s["person_same_boundary_gold_types"].values()),
            )
        dev = receipt["splits"]["dev"]
        self.assertEqual(
            (
                dev["cases"],
                dev["source_gold"],
                dev["trimmed_gold"],
                dev["zero_length_gold_spans"],
            ),
            (1003, 675, 1, 2),
        )
        self.assertEqual(dev["per_type"]["dataset"]["PER"]["correct"], 66)


if __name__ == "__main__":
    unittest.main()
