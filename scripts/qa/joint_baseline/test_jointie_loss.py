"""Regression for threshold/cap ordering and separately observed optimizer rejection."""

import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

from aggregate_receipt import external_output, project
from check_jointie_loss import check
from jerex_stages import SPLIT
from jointie_coverage import attribute_schema
from jointie_loss import DecoderTrace


class ReceiptMutationTests(unittest.TestCase):
    def setUp(self):
        source = {
            "doc_key": "synthetic", "sentence": ["A", "B", "C"],
            "ner": [[i, i, "person"] for i in range(3)],
            "relations": [[0, 0, 1, 1, "named"], [0, 0, 2, 2, "named"], [1, 1, 2, 2, "named"]],
        }
        a, b, c = (["person", start, start + 1] for start in (0, 2, 4))

        def trace(h, t, reason):
            return {"head": h, "predicate": "named", "tail": t, "first_loss": reason,
                    "co_windows": [0], "joint_mention_windows": [0]}

        hit = trace(a, b, "matched")
        hit.update(probability=0.8, candidate_threshold=0.5, rank_after_gate=1,
                   per_predicate_cap=5, optimizer_endpoint_count=2, optimizer_edge_utility=1.4)
        weak = trace(b, c, "below_edge_candidate_threshold")
        weak.update(probability=0.2, candidate_threshold=0.5, rank_after_gate=None,
                    per_predicate_cap=5, optimizer_endpoint_count=2)
        counts = {"gold_edges": 3, "decoded_edges": 2, "gold_edges_decoded": 1,
                  "gold_mentions_decoded": 3, "gold_mentions": 3, "decoded_mentions": 3,
                  "first_loss_matched": 1, "first_loss_not_in_sparse_edge_output": 1,
                  "first_loss_below_edge_candidate_threshold": 1}
        self.source = [source]
        self.row = {
            "document": "synthetic", "source_sha256": hashlib.sha256(b"A B C").hexdigest(),
            "windows": [{"start": 0, "end": 5}], "counts": counts,
            "decoded_mention_keys": [a, b, c], "decoded_edge_keys": [[a, "named", b], [b, "named", a]],
            "edge_trace": [hit, trace(a, c, "not_in_sparse_edge_output"), weak],
        }
        self.report = {"enrolled_documents": 1, "scored_documents": 1,
                       "totals": counts.copy(), "documents": [self.row]}
        self.previous = deepcopy(self.report)

    def test_valid_named_replay(self):
        check(self.source, self.report, self.previous)

    def test_replacing_one_correct_edge_with_another_keeps_totals(self):
        self.row["decoded_edge_keys"][0][2] = self.row["decoded_mention_keys"][2]
        with self.assertRaisesRegex(ValueError, "named decoded edge"):
            check(self.source, self.report)

    def test_replacing_a_false_positive_preserves_metrics_but_changes_frozen_graph(self):
        self.row["decoded_edge_keys"][1][0] = self.row["decoded_mention_keys"][2]
        check(self.source, self.report)  # Both versions really have one false positive.
        with self.assertRaisesRegex(ValueError, "historical decoded_edge_keys"):
            check(self.source, self.report, self.previous)

    def test_swapping_two_missed_edge_stages_keeps_totals(self):
        _, sparse, weak = self.row["edge_trace"]
        sparse["first_loss"], weak["first_loss"] = weak["first_loss"], sparse["first_loss"]
        with self.assertRaisesRegex(ValueError, "edge stage disagrees"):
            check(self.source, self.report)


class TraceTests(unittest.TestCase):
    def test_public_reports_have_only_counts_settings_and_hashes(self):
        receipts = Path(__file__).parent / "receipts"
        for pattern in ("jointie_ai64*.json", "jerex*32.json"):
            for path in receipts.glob(pattern):
                summary = json.loads(path.read_bytes())
                keys = {"study", "documents", "full_receipt_sha256", "source_sha256"}
                keys.update({
                    "jerex-stages": {"first_loss", "stages", "model_sha256"},
                    "jerex-predictions": {"outcomes"},
                    "jointie": {"window_words", "overlap_words", "model_sha256", "totals"},
                }[summary["study"]])
                if path.name == "jointie_ai64_training_routes.json":
                    keys.add("training_sha256")
                self.assertEqual(set(summary), keys)
                self.assertIs(type(summary["documents"]), int)
                self.assertRegex(summary["full_receipt_sha256"], r"^[a-f0-9]{64}$")

    def test_projection_excludes_annotation_values_and_commits_to_their_identity(self):
        row = {"document": "private title", "predictions": [{"head": 1, "predicate": "private relation",
               "tail": 2, "outcome": "matched"}], "outcomes": {"matched": 1}}
        report = {"documents": [row], "source_sha256": SPLIT, "outcomes": {"matched": 1}}
        expected = project(report, "jerex-predictions")
        self.assertNotIn("private", json.dumps(expected))
        other = deepcopy(report)
        other["documents"][0]["predictions"][0]["tail"] += 1
        self.assertNotEqual(expected["full_receipt_sha256"], project(other, "jerex-predictions")["full_receipt_sha256"])
        report["outcomes"]["matched"] += 1
        with self.assertRaisesRegex(ValueError, "totals differ"):
            project(report, "jerex-predictions")
        with self.assertRaisesRegex(ValueError, "outside"):
            external_output(Path(__file__).parent / "receipts/raw.json")

    def test_cap_and_solver_are_distinct_and_exhaustive(self):
        h, t = ("a", 0, 1), ("b", 1, 2)

        def edge(head, tail, logit):
            return NS(
                head=head,
                tail=tail,
                relation_type="r",
                candidate_threshold=None,
                probability=0.2,
                logit=logit,
                score=logit,
            )

        weak, strong = edge(h, t, -2), edge(t, h, -1)
        scores = NS(
            start_mappings=[0, 2],
            end_mappings=[1, 3],
            edges=[weak, strong],
            mentions=[NS(key=h), NS(key=t)],
        )
        trace = DecoderTrace()
        trace.windows = [scores]
        trace.problem = NS(
            nodes=[NS(candidate_id=h), NS(candidate_id=t)], edges=[strong]
        )
        trace.solution = NS(edges=[])
        config = NS(
            relation_role_threshold=0.05, relation_pair_cap=1, max_edges_per_type=4
        )
        a, b = ("a", 0, 1), ("b", 2, 3)
        gold = [(a, "r", b), (b, "r", a)]
        outcome = trace.losses(gold, scores, config, [{"start": 0, "end": 3}], set())
        self.assertEqual(
            [(e["first_loss"], e["rank_after_gate"]) for e in outcome],
            [("edge_cap", 2), ("optimizer_did_not_select", 1)],
        )
        row = {"edge_trace": outcome, "counts": {"gold_edges": 2}}
        attribute_schema([row], {"r": (["a"], ["b"])})
        self.assertEqual(
            row["counts"],
            {
                "gold_edges": 2,
                "first_loss_edge_cap": 1,
                "first_loss_endpoint_types_outside_schema": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
