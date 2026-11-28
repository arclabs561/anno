# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regression controls for the fixed-budget, paired CrossRE endpoint comparison."""

from collections import Counter
from dataclasses import replace
from math import ceil
from pathlib import Path
import json
from random import Random
import unittest

import relex_endpoint_budget as run


def observation(index, domain, edges, gold):
    entities = {(0, 1), (2, 3)}
    return run.Observation(domain, "test", index, str(index), edges, Counter(gold),
                           entities, Counter())


class EndpointBudgetTest(unittest.TestCase):
    def test_paired_interval_refills_both_queues_at_the_same_global_budget(self):
        edge = (0, 1, "a", 2, 3)
        cases = [
            observation(0, "ai", [(edge, .49, .51)], []),
            observation(1, "ai", [(edge, .48, .52)], [edge]),
            observation(2, "music", [(edge, .47, .99)], [edge]),
            observation(3, "music", [(edge, .46, .97)], []),
            observation(4, "music", [], []),
        ]
        baseline, challenger = run.nominees(cases, 0), run.nominees(cases, 1)
        samples, seed, slots = 200, 42, 2
        interval = run.queue_interval(cases, baseline, challenger, slots, replicates=samples, seed=seed)

        # Independently materialize each sample, recompute model-only rankings from
        # scratch, and score the actual admitted sentences and chosen relations.
        rng = Random(seed)
        differences = []
        for _ in range(samples):
            drawn = rng.choices(cases[:2], k=2) + rng.choices(cases[2:4], k=2)
            sample = [replace(case, index=i) for i, case in enumerate(drawn)]
            result = run.compare(sample, run.nominees(sample, 0), run.nominees(sample, 1), 50)
            self.assertEqual(result["review_slots"], slots)
            counts = result["counts"]
            differences.append(counts.get("endpoint_correct", 0) - counts.get("confidence_correct", 0))
        differences.sort()
        self.assertEqual((interval["lower"], interval["upper"]),
                         (differences[ceil(.025 * samples) - 1], differences[ceil(.975 * samples) - 1]))
        self.assertEqual(interval["eligible_sentences_by_domain"]["music"], 2)
        self.assertEqual(run.resampled_hits([(0, False), (1, True), (2, False)], [0, 2, 1], 1), 1)
        with self.assertRaisesRegex(ValueError, "same eligible"):
            run.queue_interval(cases, baseline[:-1], challenger, slots)

    def test_reviews_are_selected_across_domains_before_scoring(self):
        first = (0, 1, "a", 2, 3)
        alternate = (0, 1, "b", 2, 3)
        cases = [
            observation(0, "ai", [(first, 0.49, 0.51), (alternate, 0.48, 1.0)], [first, alternate]),
            observation(1, "ai", [(first, 0.485, 0.51)], []),
            observation(2, "music", [(first, 0.47, 0.99)], [first]),
            observation(3, "music", [(first, 0.46, 0.98)], []),
            observation(4, "ai", [], []),
        ]
        left = run.nominees(cases, 0)
        right = run.nominees(cases, 1)
        result = run.compare(cases, left, right, 50)
        self.assertEqual(result["review_slots"], 2)
        self.assertEqual(result["counts"]["confidence_correct"], 1)
        self.assertEqual(result["counts"]["endpoint_correct"], 2)
        self.assertEqual(result["counts"]["gained_correct_relations"], 2)
        self.assertEqual(result["counts"]["lost_correct_relations"], 1)
        self.assertEqual(result["counts"]["both_correct_changed_relation"], 1)
        self.assertEqual(result["domains"]["music"]["endpoint_suggestions"], 1)
        self.assertEqual(result["domains"]["music"].get("confidence_suggestions", 0), 0)
        self.assertEqual(result["sentence_allocation"]["candidate_only_sentence"], 1)
        self.assertEqual(result["sentence_allocation"]["baseline_only_sentence"], 1)
        # Reference changes cannot change inference choices.
        for case in cases:
            case.reference.clear()
        self.assertEqual(right, run.nominees(cases, 1))

    def test_condensed_features_come_only_from_retained_endpoint_scores(self):
        head = {"start": 0, "end": 1, "score": .8}
        tail = {"start": 2, "end": 3, "score": .6}
        pairs = [{"head": head, "tail": tail, "relation": "cold", "score": .5},
                 {"head": tail, "tail": head, "relation": "warm", "score": .7}]
        result = run.condensed([head, tail], pairs)
        self.assertEqual(result["candidates"], [(0, 1, "cold", 2, 3, .5, .6)])
        self.assertEqual(result["stock"], [(2, 3, "warm", 0, 1)])
        with self.assertRaisesRegex(ValueError, "Duplicate relation"):
            run.condensed([head, tail], pairs + pairs[:1])

    def test_aggregate_workers_must_agree_on_development_and_cover_test(self):
        selected = {"selected_endpoint_weight": .5}
        confirmation = {"part": "confirmation", "selection": selected, "panels": {
            "calibration": {"same": True}, "confirmation": {}}}
        test = {"part": "test", "selection": selected, "panels": {
            "calibration": {"same": True}, "balanced_test": {}, "additional_test": {},
            "outside_content_holdout": {}, "full_official_test": {}}}
        self.assertEqual(len(run.combine(confirmation, test)["panels"]), 6)
        test["panels"]["calibration"] = {"same": False}
        with self.assertRaisesRegex(ValueError, "development"):
            run.combine(confirmation, test)
