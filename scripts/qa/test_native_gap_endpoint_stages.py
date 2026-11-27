"""Project-authored controls for attributing raw native endpoint losses."""

from copy import deepcopy
import unittest

from native_gap_endpoint_stages import checked_partition
from native_gap_loss import stage_details


def decision(selected, candidates, limit):
    return {"selected": selected, "candidates": candidates, "max_antecedents": limit}


class EndpointTraceTests(unittest.TestCase):
    def test_overlap_winner_can_be_hidden_by_singleton_suppression(self):
        shorter, wider, crossing = (0, 2), (0, 4), (2, 5)
        trace = {
            "proposed": [crossing, shorter, wider],
            "retained": [wider],
            "decisions": [decision(None, [], 1)],
            "clusters": [],
            "max_distance": 100,
        }
        self.assertEqual(checked_partition(trace), [])
        trace["retained"] = [shorter]
        with self.assertRaisesRegex(ValueError, "overlap deduplication"):
            checked_partition(trace)

    def test_identical_empty_partitions_can_lose_a_name_at_different_stages(self):
        name, pronoun = (0, 3), (8, 11)
        docs = [("example", "Ada saw she left", (pronoun, name, (12, 16)), (True, False))]
        missing = {
            "id": "example",
            "proposed": [pronoun],
            "retained": [pronoun],
            "decisions": [decision(None, [], 10)],
            "clusters": [],
            "max_distance": 100,
        }
        detected = dict(
            missing,
            proposed=[name, pronoun],
            retained=[name, pronoun],
            decisions=[decision(None, [], 10), decision(None, [(0, 0.1, 0.2)], 10)],
        )
        self.assertEqual(checked_partition(missing), checked_partition(detected))
        self.assertEqual(stage_details(docs, [missing])["stages"], {"proposal": 1})
        self.assertEqual(stage_details(docs, [detected])["stages"], {"cluster_emission": 1})

    def test_edge_reconstruction_checks_unmarked_mentions_and_incoming_links(self):
        spans = [(0, 1), (2, 4), (8, 10), (11, 12)]
        trace = {
            "proposed": spans,
            "retained": spans,
            "max_distance": 100,
            "decisions": [
                decision(None, [], 1),
                decision(0, [(0, 0.7, 0.2)], 1),
                decision(None, [(1, 0.2, 0.2)], 1),
                decision(2, [(2, 0.7, 0.2)], 1),
            ],
            "clusters": [[spans[1], spans[0]], [spans[2], spans[3]]],
        }
        self.assertEqual(checked_partition(trace), [spans[:2], spans[2:]])
        changed = deepcopy(trace)
        changed["clusters"][0].append(changed["clusters"][1].pop())
        with self.assertRaisesRegex(ValueError, "full emitted partition"):
            checked_partition(changed)

    def test_same_chain_can_hide_the_wrong_direct_edge(self):
        spans = [(0, 1), (2, 3), (4, 5)]
        trace = {
            "proposed": spans,
            "retained": spans,
            "max_distance": 100,
            "decisions": [
                decision(None, [], 2),
                decision(0, [(0, 0.8, 0.2)], 2),
                decision(1, [(1, 0.8, 0.2), (0, 0.8, 0.2)], 2),
            ],
            "clusters": [spans],
        }
        self.assertEqual(checked_partition(trace), [spans])
        trace["decisions"][2]["selected"] = 0
        with self.assertRaisesRegex(ValueError, "recorded scores"):
            checked_partition(trace)


if __name__ == "__main__":
    unittest.main()
