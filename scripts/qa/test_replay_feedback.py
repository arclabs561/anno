import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import replay_feedback as rf
from run_matrix_slot import run_slot
from brief_round import evidence


class Oracle:
    SCOPE = "synthetic_contract"
    PASS_SIGNATURE = ""
    FAIL_SIGNATURE = "assertion-A"

    def family_for_case(self, root, case):
        return case.split("--")[0]

    def __init__(self):
        self.outcomes = {}
        self.calls = []

    def describe(self, root, case):
        return {
            "family": case.split("--")[0],
            "source_sha256": rf.key(case),
            "oracle_sha256": rf.key("oracle-v1"),
        }

    def replay(self, root, case):
        self.calls.append(case)
        outcome = self.outcomes.get(case, "pass")
        return {
            "outcome": outcome,
            "signature": "assertion-A" if outcome == "fail" else "",
            "producer_sha256": rf.key("executable"),
        }


class Feedback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root / "state"
        self.oracle = Oracle()
        self.enterContext(patch.dict(rf.ADAPTERS, {"fixture": self.oracle}))
        self.enterContext(patch.object(rf, "stamp", return_value="f" * 40))

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
        rf.observe(self.root, self.cache, "fixture", pick["case"], pick)
        self.assertEqual(self.pick()["case"], "z")  # additional variants do not buy another turn
        self.record("a", "pass")
        self.record("z", "pass")
        self.assertEqual(self.pick()["case"], "a--variant")

    def test_extra_families_cannot_buy_a_corpus_another_turn(self):
        self.enterContext(patch.dict(rf.ADAPTERS, {"other": self.oracle}))
        self.record("a", "fail")
        self.record("b", "fail")
        self.oracle.outcomes["z"] = "fail"
        rf.observe(self.root, self.cache, "other", "z")
        selected = self.pick()
        self.assertEqual(selected["corpus"], "fixture")
        rf.observe(self.root, self.cache, "fixture", selected["case"], selected)
        self.assertEqual(self.pick()["corpus"], "other")

    def test_unconfirmed_flap_no_entry(self):
        def flap(*args):
            n = len(self.oracle.calls)
            self.oracle.calls.append(n)
            return {
                "outcome": "fail" if not n else "pass",
                "signature": "x",
                "producer_sha256": "1" * 64,
            }

        with patch.object(self.oracle, "replay", side_effect=flap), self.assertRaises(ValueError):
            self.record("a", "fail")
        self.assertIsNone(self.pick())

    def test_stale_selection_and_aba_rejected(self):
        self.record("a", "fail")
        stale = self.pick()
        self.record("a", "pass")
        self.record("a", "fail")
        with self.assertRaisesRegex(ValueError, "stale"):
            rf.observe(self.root, self.cache, "fixture", "a", stale)

    def test_replay_slot_restores_and_retires_only_selected_case(self):
        self.record("a", "fail")
        snap = self.root / "snapshot.db"
        rf.backup(self.cache / rf.DB_NAME, snap)
        cold = self.root / "cold"
        rf.backup(snap, cold / rf.DB_NAME)
        with self.assertRaises(FileExistsError):
            rf.backup(snap, cold / rf.DB_NAME)
        self.oracle.outcomes["a"] = "pass"
        out = self.root / "slot"
        run_slot(None, cold, out, "mixed", 1, 4, 2, 19, replay_history=True)
        self.assertIsNone(rf.next_case(self.root, cold))
        self.assertTrue((out / "replay.json").exists())
        self.assertFalse((cold / "eval-results.jsonl").exists())
        self.assertNotEqual(self.pick(), None)

    def test_transfer_contains_latest_receipts_and_no_private_storage(self):
        old = self.record("a", "fail")
        current = self.record("a", "pass")
        private = "PRIVATE-CANARY-92a1"
        with sqlite3.connect(self.cache / rf.DB_NAME) as db:
            db.execute("CREATE TABLE auxiliary (raw TEXT)")
            db.execute("INSERT INTO auxiliary VALUES (?)", (private,))
            db.execute("ALTER TABLE witnesses ADD COLUMN debug TEXT")
            db.execute("UPDATE witnesses SET debug=?", (private,))
            db.execute("UPDATE observations SET receipt=? WHERE id=?", (private, old["observation_id"]))
        snap = self.root / "snapshot.db"
        rf.backup(self.cache / rf.DB_NAME, snap)
        self.assertNotIn(private.encode(), snap.read_bytes())
        with sqlite3.connect(snap) as db:
            self.assertEqual(db.execute("SELECT id FROM observations").fetchall(), [(current["observation_id"],)])
            self.assertEqual({r[1] for r in db.execute("PRAGMA table_info(witnesses)")}, set(rf.WITNESS_FIELDS))
            self.assertIsNone(db.execute("SELECT name FROM sqlite_schema WHERE name='auxiliary'").fetchone())

    def test_legacy_wikigold_span_lists_never_enter_snapshot(self):
        self.record("a", "pass")
        legacy = dict(corpus="wikigold", scope="pinned_source_contract", family="wikigold-stacked-exact-spans",
                      source_sha256="1" * 64, oracle_sha256="2" * 64)
        raw = "PRIVATE-GOLD-DIFFERENCES" * 200
        with sqlite3.connect(self.cache / rf.DB_NAME) as db:
            db.execute("INSERT INTO observations VALUES (?,?)", (rf.key(raw), raw))
            db.execute("INSERT INTO witnesses VALUES (?,?,?,?,?,?,?,?,?,?)", (
                rf.key(legacy), legacy["corpus"], legacy["family"], "sentence-0", legacy["source_sha256"],
                legacy["oracle_sha256"], rf.key(raw), 1, 1, 0))
        snap = self.root / "snapshot.db"
        rf.backup(self.cache / rf.DB_NAME, snap)
        self.assertNotIn(b"PRIVATE-GOLD", snap.read_bytes())
        with sqlite3.connect(snap) as db:
            self.assertEqual(db.execute("SELECT corpus FROM witnesses").fetchall(), [("fixture",)])

    def test_transfer_rejects_current_free_text_and_oversized_history_atomically(self):
        original = self.record("a", "fail")
        self.record("b", "pass")
        source = self.cache / rf.DB_NAME
        target = self.root / "out.db"
        with patch.object(rf, "MAX_WITNESSES", 1), self.assertRaisesRegex(ValueError, "exceeds"):
            rf.backup(source, target)
        self.assertFalse(target.exists())
        event = {k: v for k, v in original.items() if k != "observation_id"}
        event["signature"] = "private source passage"
        with sqlite3.connect(source) as db:
            db.execute("UPDATE observations SET id=?,receipt=? WHERE id=?",
                       (rf.key(event), json.dumps(event), original["observation_id"]))
            db.execute("UPDATE witnesses SET latest=? WHERE case_name='a'", (rf.key(event),))
        with self.assertRaisesRegex(ValueError, "signature"):
            rf.backup(source, target)
        self.assertFalse(target.exists())
        with self.assertRaisesRegex(ValueError, "signature"):
            self.pick()

    def test_tracked_unreviewed_catalog_seed_is_ineligible(self):
        with self.assertRaisesRegex(ValueError, "unknown reviewed KB seed"):
            rf.replay_reviewed_kb.seed_path(self.root, "changed-and-restored--unreviewed")
        for case in rf.replay_reviewed_kb.REVIEWED:
            self.assertTrue(rf.replay_reviewed_kb.seed_path(rf.ROOT, case).is_file())

    def test_plan_does_not_consume_or_relabel_evidence(self):
        self.record("a", "fail")
        before = self.pick()
        count = len(self.oracle.calls)
        out = self.root / "plan"
        run_slot(None, self.cache, out, "mixed", 1, 4, 2, 19, plan_only=True, replay_history=True)
        self.assertEqual(self.pick(), before)
        self.assertEqual(len(self.oracle.calls), count)
        self.assertIn("oracle replay pending", evidence(out))

    def test_handoff_rejects_mismatched_oracle_result(self):
        self.record("a", "fail")
        self.oracle.outcomes["a"] = "pass"
        out = self.root / "run"
        run_slot(None, self.cache, out, "mixed", 1, 4, 2, 19, replay_history=True)
        self.assertIn("Oracle result: pass", evidence(out))
        path = out / "replay.json"
        receipt = json.loads(path.read_bytes())
        receipt["producer_sha256"] = "0" * 64
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "does not match"):
            evidence(out)

    def test_random_and_reserved_slots_ignore_open_witness(self):
        self.record("a", "fail")
        for policy, slot in [("random", 1), ("mixed", 2), ("mixed", 3)]:
            with (
                self.subTest(policy=policy, slot=slot),
                patch.object(rf, "next_case", side_effect=AssertionError),
                patch("run_matrix_slot.subprocess.run"),
                self.assertRaisesRegex(ValueError, "no outcome"),
            ):
                run_slot(
                    self.root / "native",
                    self.cache,
                    self.root / f"{policy}{slot}",
                    policy,
                    slot,
                    4,
                    2,
                    99,
                    replay_history=True,
                )

    def test_inconsistent_observation_rejected(self):
        self.record("a", "fail")
        with sqlite3.connect(self.cache / rf.DB_NAME) as db:
            db.execute("UPDATE witnesses SET case_name='other'")
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.pick()
