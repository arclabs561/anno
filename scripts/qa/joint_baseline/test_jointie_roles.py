# SPDX-License-Identifier: MIT OR Apache-2.0
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from jointie_roles import RoleTrace, ranked_roles
from jointie_ranking import candidates


class RoleTests(unittest.TestCase):
    def test_destinations_checked_before_loading_models(self):
        runner = Path(__file__).resolve().with_name("jointie_coverage.py")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            existing = root / "gold.json"
            existing.write_text("untouched")
            for options in ([], ["--training-routes", "train"], ["--loss-trace"], ["--sparse-roles"],
                            ["--argument-threshold", ".1"], ["--rank-typed-alternatives"]):
                run = subprocess.run([sys.executable, "-O", str(runner), "source", "weights", str(existing), *options],
                                     capture_output=True, text=True)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("output already exists", run.stderr)
                self.assertEqual(existing.read_text(), "untouched")
            for kind in ("repository", "linked_worktree"):
                repo = root / kind
                repo.mkdir()
                if kind == "repository":
                    (repo / ".git").mkdir()
                    (repo / ".git/HEAD").write_text("ref: refs/heads/main")
                else:
                    (repo / ".git").write_text("gitdir: ../administrative/worktrees/tree")
                dest = repo / "private.json"
                run = subprocess.run([sys.executable, "-O", str(runner), "source", "weights", str(dest),
                                      "--training-routes", "train"], capture_output=True, text=True)
                self.assertIn("outside Git checkouts", run.stderr)
                self.assertFalse(dest.exists())

    def test_stable_rank_threshold_and_duplicate_occurrences(self):
        raw, eligible, top = ranked_roles([0, 1], [[.2, .6], [.6, .1]], [[True, True], [True, True]],
                                          [[(4, 5), (2, 3)], [(0, 1), (4, 5)]], .2, 1)
        self.assertEqual(raw, {(4, 5), (2, 3), (0, 1)})
        self.assertEqual(eligible, raw)
        self.assertEqual(top, {(0, 1)})

    def test_overlapping_windows_credit_furthest_observed_stage(self):
        trace = RoleTrace()
        hs, ts = {(1, 2)}, {(3, 4)}
        trace.windows = [{"r": ((hs, hs, hs), (ts, set(), set()), False, set())},
                         {"r": ((hs, hs, hs), (ts, ts, ts), False, set())}]
        edge = {"head": ["a", 1, 2], "tail": ["b", 3, 4], "predicate": "r", "joint_mention_windows": [0, 1]}
        self.assertEqual(trace.classify(edge, [{"start": 0}, {"start": 0}]), ["pair_cap_before_relation_scoring"])

    def test_diversity_preserves_fixed_cap_and_original_logits(self):
        a, z, b, c = ("a", 0, 1), ("z", 0, 1), ("b", 1, 2), ("c", 2, 3)
        def edge(h, t, value):
            return NS(head=h, tail=t, candidate_threshold=None, probability=.6, logit=value, relation_type="r")
        edges = [edge(a, b, 2), edge(z, b, 2), edge(c, z, 1)]
        mentions = [NS(key=k, logit=v) for k, v in ((a, 0), (z, 1), (b, 1), (c, 0))]
        config = NS(relation_role_threshold=.25, relation_pair_cap=2, max_edges_per_type=4)
        stock, alt = candidates(NS(mentions=mentions, edges=edges), config)
        self.assertEqual(stock, edges[:2])
        self.assertEqual(alt, edges[1:])
