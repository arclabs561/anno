import unittest

from native_pronoun_person_evidence import Answer, predictions, select


class PersonEvidence(unittest.TestCase):
    def test_unique_support_requires_a_unique_native_link_to_that_name(self):
        rows = [Answer(None, 1, target, frozenset(supported)) for target, supported in
                [(0, [0]), (1, [0]), (0, [0, 1]), (None, [1]), (1, []), (1, [1])]]
        self.assertEqual(predictions(rows, "unique_support"), [0, None, None, None, None, 1])

    def test_development_budget_and_ties(self):
        counts = {"correct": 9, "wrong": 8}
        arms = {"stock": counts, "typed": {"correct": 100, "wrong": 9}, "unique_support": counts}
        self.assertEqual(select(arms), "stock")
        arms["unique_support"] = {"correct": 10, "wrong": 8}
        self.assertEqual(select(arms), "unique_support")
