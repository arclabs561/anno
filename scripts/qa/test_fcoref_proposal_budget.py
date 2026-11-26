import unittest

from fcoref_proposal_budget import compare


class BudgetTrialTest(unittest.TestCase):
    def test_equal_totals_keep_gains_losses_and_false_links(self):
        docs = [(None, None, None, [True, False])] * 2
        a = {"per_id": ["11110", "11101"]}
        b = {"per_id": ["11101", "11110"]}
        changes = compare(docs, a, b)
        for key in ("gold_links_recovered", "gold_links_lost", "unmatched_links_added",
                    "unmatched_links_removed", "exact_document_wins", "exact_document_losses"):
            self.assertEqual(changes[key], 1, key)
