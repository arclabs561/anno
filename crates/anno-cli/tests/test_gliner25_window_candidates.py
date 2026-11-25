"""Run with the GLiNER2.5 Python environment; no checkpoint required."""
import runpy
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from gliner2.joint_ie import JointSchema, compile_schema
from gliner2.joint_ie.candidate_scores import (
    CandidateScoreSet, MentionScore, ScoredRelationEdge, candidate_score_set_to_problem,
)
from gliner2.joint_ie.optimizers import BeamOptimizer

worker = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/cli/commands/joint.py"))
merge = worker["merge_window_candidates"]


def mention(label, start):
    return MentionScore(0, label, start, start + 1, 4.0, 0.98)


def fragment(text, tokens, nodes, edges):
    return CandidateScoreSet(
        text=text, mentions=tuple(nodes), edges=tuple(edges), text_tokens=tuple(t[0] for t in tokens),
        start_mappings=tuple(t[1] for t in tokens), end_mappings=tuple(t[2] for t in tokens)
    )


class DocumentCandidates(unittest.TestCase):
    def test_competing_window_edges_obey_one_document_cardinality_limit(self):
        # The same occurrence of Éva is visible in both windows. The two
        # organizations never share a window, even though the schema permits
        # an organization-to-organization edge.
        text = "Acme Éva Beta"
        words = [("Acme", 0, 4), ("Éva", 5, 8), ("Beta", 9, 13)]
        acme, eva = mention("org", 0), mention("person", 1)
        left = ScoredRelationEdge("employer", eva.key, acme.key, 3.0, 0.95)
        window1 = fragment(text[:8], words[:2], [acme, eva], [left])
        eva, beta = mention("person", 0), mention("org", 1)
        right = ScoredRelationEdge("employer", eva.key, beta.key, 5.0, 0.99)
        window2 = fragment(text[5:], [("Éva", 0, 3), ("Beta", 4, 8)], [eva, beta], [right])

        candidates, windows = merge(text, words, [(0, 2, 0, 8, window1), (1, 3, 5, 13, window2)])
        self.assertEqual(windows, [{"start": 0, "end": 8}, {"start": 5, "end": 13}])
        self.assertEqual(len(candidates.mentions), 3)
        self.assertEqual(len(candidates.edges), 2)
        schema = compile_schema(JointSchema().entities(["person", "org"])
                                .relation("employer", "person", "org", unique_head=True)
                                .relation("owns", "org", "org"))
        problem = candidate_score_set_to_problem(candidates, constraints=schema.constraints)
        solution = BeamOptimizer().optimize(problem)
        self.assertTrue(solution.feasible)
        self.assertEqual([(e.relation_type, e.head, e.tail) for e in solution.edges],
                         [("employer", ("person", 1, 2), ("org", 2, 3))])

    def test_overlap_keeps_strongest_typed_directed_candidate_by_occurrence(self):
        text = "Ana Ana"
        words = [("Ana", 0, 3), ("Ana", 4, 7)]
        a, b = mention("person", 0), mention("person", 1)
        link = ScoredRelationEdge("knows", a.key, b.key, 1.0, 0.73)
        weak = fragment(text, words, [a, b], [link])
        strong = replace(link, logit=2.0, probability=0.88)
        reverse = replace(link, head=b.key, tail=a.key)
        distinct = replace(link, relation_type="supervises")
        repeated = fragment(text, words, [a, b], [strong, reverse, distinct])
        scores, _ = merge(text, words, [(0, 2, 0, 7, weak), (0, 2, 0, 7, repeated)])
        self.assertEqual(len(scores.mentions), 2)
        self.assertEqual(set(scores.edges), {strong, reverse, distinct})

    def test_schema_and_subwords_share_window_budget(self):
        # Two prompt tokens and two subwords per source word.
        processor = SimpleNamespace(transform_record=lambda text, schema: SimpleNamespace(
            input_ids=range(2 + 2 * len(text.split()))
        ))
        model = SimpleNamespace(scorer=SimpleNamespace(processor=processor),
                                score=lambda text, schema: text)
        text = "a b c d e"
        words = [(ch, index, index + 1) for index, ch in enumerate(text) if ch != " "]
        windows = list(worker["model_windows"](model, None, text, words, 4, 1, 6))
        self.assertEqual([w[-1] for w in windows], ["a b", "b c", "c d", "d e"])
        with self.assertRaisesRegex(ValueError, "exceed the encoder budget"):
            list(worker["model_windows"](model, None, text, words, 4, 1, 3))

    def test_missing_endpoint_and_misaligned_windows_fail(self):
        words = [("Éva", 0, 3), ("Acme", 4, 8)]
        eva = mention("person", 0)
        dangling = ScoredRelationEdge("works_for", eva.key, ("org", 1, 2), 3.0, 0.95)
        scores = fragment("Éva Acme", words, [eva], [dangling])
        with self.assertRaisesRegex(ValueError, "missing window endpoint"):
            merge(scores.text, words, [(0, 2, 0, 8, scores)])
        with self.assertRaisesRegex(ValueError, "offsets disagree"):
            merge(scores.text, words, [(0, 1, 0, 3, scores)])


if __name__ == "__main__":
    unittest.main()
