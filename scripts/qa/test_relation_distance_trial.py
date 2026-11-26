import copy
import unittest

from relation_distance_trial import cases, score


class GroundedRelationScoring(unittest.TestCase):
    def test_only_exact_directed_typed_occurrences_match(self):
        text, _, second = next(cases())
        head = text.index("Marie")
        tail = text.index("Acme Inc")
        reference = {((head, head + 5, "PER"), "WORKS_FOR", (tail, tail + 8, "ORG"))}
        edge = {"head": {"text": "Marie", "start": head, "end": head + 5, "type": "PER"},
                "type": "WORKS_FOR",
                "tail": {"text": "Acme Inc", "start": tail, "end": tail + 8, "type": "ORG"}}
        short = copy.deepcopy(edge)
        short["tail"].update(text="Acme", end=tail + 4)
        wrong_type = copy.deepcopy(edge)
        wrong_type["head"]["type"] = "ORG"
        for incorrect in [short, {**edge, "type": "FOUNDED"}, wrong_type,
                          {**edge, "head": edge["tail"], "tail": edge["head"]}]:
            with self.subTest(incorrect=incorrect):
                counts = score(text, [incorrect], reference, second)
                self.assertEqual((counts["matched"], counts["extra"], counts["missed"]), (0, 1, 1))
        duplicate = score(text, [edge, edge], reference, second)
        self.assertEqual((duplicate["matched"], duplicate["extra"], duplicate["missed"]), (1, 1, 0))
