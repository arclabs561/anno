"""Regression for threshold/cap ordering and separately observed optimizer rejection."""

import json
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

from jointie_coverage import attribute_schema
from jointie_loss import DecoderTrace


class TraceTests(unittest.TestCase):
    def test_published_jerex_prefix_and_edge_outcomes(self):
        receipts = Path(__file__).parent / "receipts"
        earlier = json.loads((receipts / "jerex.json").read_bytes())
        expanded = json.loads((receipts / "jerex_docred32.json").read_bytes())
        docs = expanded["documents"]
        self.assertEqual(len(docs), expanded["enrolled_documents"])
        self.assertEqual(len(docs), expanded["scored_documents"])
        prefix = docs[: len(earlier["titles"])]
        self.assertEqual([doc["document"] for doc in prefix], earlier["titles"])
        for metric, expected in earlier["stages"].items():
            aggregate = Counter()
            for doc in prefix:
                aggregate.update(doc["stages"][metric])
            self.assertEqual(aggregate, expected)
        for doc in docs:
            with self.subTest(document=doc["document"]):
                trace = doc["edge_trace"]
                labeled = doc["stages"]["labeled_relations"]
                self.assertEqual(
                    len({(e["head"], e["predicate"], e["tail"]) for e in trace}),
                    labeled["gold"],
                )
                losses = Counter(e["first_loss"] for e in trace)
                self.assertEqual(losses, doc["first_loss"])
                self.assertEqual(losses["matched"], labeled["matched"])
                for e in trace:
                    if "predicate_probability" in e:
                        self.assertEqual(
                            e["predicate_probability"] >= doc["relation_threshold"],
                            e["first_loss"] == "matched",
                        )

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
