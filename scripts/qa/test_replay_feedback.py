import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import replay_feedback as rf
from run_matrix_slot import run_slot


class Oracle:
    SCOPE = "synthetic_contract"

    def __init__(self):
        self.outcomes = {}
        self.calls = []

    def describe(self, root, case):
        return {"family": case.split("--")[0], "source_sha256": rf.key(case),
                "oracle_sha256": rf.key("oracle-v1")}

    def replay(self, root, case):
        self.calls.append(case)
        outcome = self.outcomes.get(case, "pass")
        return {"outcome": outcome, "signature": "assertion-A" if outcome == "fail" else "",
                "producer_sha256": rf.key("executable")}


class Feedback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root / "state"
        self.oracle = Oracle()
        self.enterContext(patch.dict(rf.ADAPTERS, {"fixture":self.oracle}))
        self.enterContext(patch.object(rf, "stamp", return_value="f"*40))

    def record(self, case, outcome):
        self.oracle.outcomes[case] = outcome
        return rf.observe(self.root, self.cache, "fixture", case)

    def pick(self):
        return rf.next_case(self.root, self.cache)

    def test_case_resolution_and_dedup_do_not_erase_related_witnesses(self):
        self.record("a", "fail")
        self.record("a", "fail")
        self.record("a--variant", "fail")
        self.record("z", "fail")
        with sqlite3.connect(self.cache / rf.DB_NAME) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM observations").fetchone()[0], 3)
        pick = self.pick()
        rf.observe(self.root,self.cache,"fixture",pick["case"],pick)
        self.assertEqual(self.pick()["case"],"z")  # additional variants do not buy another turn
        self.record("a","pass")
        self.record("z","pass")
        self.assertEqual(self.pick()["case"],"a--variant")

    def test_unconfirmed_flap_no_entry(self):
        def flap(*args):
            n = len(self.oracle.calls)
            self.oracle.calls.append(n)
            return {"outcome":"fail" if not n else "pass", "signature":"x", "producer_sha256":"1"*64}
        with patch.object(self.oracle,"replay",side_effect=flap), self.assertRaises(ValueError):
            self.record("a","fail")
        self.assertIsNone(self.pick())

    def test_stale_selection_and_aba_rejected(self):
        self.record("a","fail")
        stale = self.pick()
        self.record("a","pass")
        self.record("a","fail")
        with self.assertRaisesRegex(ValueError,"stale"):
            rf.observe(self.root,self.cache,"fixture","a",stale)

    def test_replay_slot_restores_and_retires_only_selected_case(self):
        self.record("a","fail")
        snap = self.root/"snapshot.db"
        rf.backup(self.cache/rf.DB_NAME,snap)
        cold = self.root/"cold"
        rf.backup(snap,cold/rf.DB_NAME)
        with self.assertRaises(FileExistsError):
            rf.backup(snap,cold/rf.DB_NAME)
        self.oracle.outcomes["a"]="pass"
        out=self.root/"slot"
        run_slot(None,cold,out,"mixed",1,4,2,19,replay_history=True)
        self.assertIsNone(rf.next_case(self.root,cold))
        self.assertTrue((out/"replay.json").exists())
        self.assertFalse((cold/"eval-results.jsonl").exists())
        self.assertNotEqual(self.pick(),None)

    def test_random_and_reserved_slots_ignore_open_witness(self):
        self.record("a","fail")
        for policy,slot in [("random",1),("mixed",2),("mixed",3)]:
            with self.subTest(policy=policy,slot=slot), patch.object(rf,"next_case",side_effect=AssertionError), patch("run_matrix_slot.subprocess.run"), self.assertRaisesRegex(ValueError,"no outcome"):
                run_slot(self.root/"native",self.cache,self.root/f"{policy}{slot}",policy,slot,4,2,99,replay_history=True)

    def test_inconsistent_observation_rejected(self):
        self.record("a","fail")
        with sqlite3.connect(self.cache / rf.DB_NAME) as db:
            db.execute("UPDATE witnesses SET case_name='other'")
        with self.assertRaisesRegex(ValueError,"inconsistent"):
            self.pick()
