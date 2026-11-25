"""Regression for threshold/cap ordering and separately observed optimizer rejection."""

import unittest
from types import SimpleNamespace as NS

from jointie_coverage import attribute_schema
from jointie_loss import DecoderTrace


class TraceTests(unittest.TestCase):
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
