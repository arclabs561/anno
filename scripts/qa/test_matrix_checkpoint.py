import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from zipfile import ZipFile

from matrix_checkpoint import journal_size, select_checkpoints, snapshot, unpack


class MatrixCheckpointTest(unittest.TestCase):
    def test_restores_latest_eligible_default_branch_run_even_if_other_jobs_failed(self):
        artifacts = [
            {"id": n, "expired": n == 5, "workflow_run": {"head_branch": "main", "id": n}}
            for n in range(1, 6)
        ]
        def run(n):
            return {
                "path": ".github/workflows/other.yml" if n == 3 else ".github/workflows/ci.yml",
                "event": "pull_request" if n == 4 else "push",
                "head_branch": "main", "conclusion": "failure",
                "head_repository": {"full_name": "fork/anno" if n == 2 else "owner/anno"},
            }
        previous, published = select_checkpoints(artifacts, "main", "owner/anno", 6, run)
        self.assertEqual(previous["id"], 1)
        self.assertIsNone(published)
        self.assertEqual(select_checkpoints(artifacts, "other", "owner/anno", 6, run), (None, None))

    def test_same_run_retries_and_late_uploads_do_not_override_a_newer_producer(self):
        artifacts = [
            {"id": artifact, "expired": False, "workflow_run": {"head_branch": "main", "id": run}}
            for artifact, run in [(99, 1), (20, 2), (30, 3), (40, 4)]
        ]
        def trusted(_run):
            return {"path": ".github/workflows/ci.yml", "event": "schedule",
                    "head_branch": "main", "head_repository": {"full_name": "owner/anno"}}
        previous, published = select_checkpoints(artifacts, "main", "owner/anno", 3, trusted)
        self.assertEqual(previous["id"], 20)
        self.assertEqual(published, 4)
        previous, published = select_checkpoints(artifacts[:3], "main", "owner/anno", 3, trusted)
        self.assertEqual(previous["id"], 20)
        self.assertEqual(published, 3)

    def test_no_outcome_does_not_republish_stale_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "muxer_history.ner.json").write_text("{}")
            log = root / "log"
            log.write_text('{"strategy": "worst-first", "chosen": "heuristic"}\n')
            self.assertEqual(snapshot(root, root / "snapshot", [log]), 0)
            self.assertFalse((root / "snapshot").exists())

    def test_snapshot_restores_windows_and_committed_wal_without_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "muxer_history.ner.json").write_text('{"window": 1}')
            (root / "unrelated").write_text("excluded")
            (root / "eval-results.jsonl").write_text('{"observation_id": "current", "score": 0}\n')
            log = root / "current.jsonl"
            log.write_text(json.dumps({"record_type": "outcome", "observation_id": "current"}) + "\n")
            with sqlite3.connect(root / "eval-history.db") as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("CREATE TABLE observations(n)")
                conn.execute("INSERT INTO observations VALUES (7)")
                conn.commit()
                target = root / "saved"
                self.assertEqual(snapshot(root, target, [log]), 1)
            archive = io.BytesIO()
            with ZipFile(archive, "w") as bundle:
                for file in target.iterdir():
                    bundle.write(file, file.name)
            restored = root / "restored"
            unpack(archive, restored)
            self.assertEqual({p.name for p in restored.iterdir()}, {
                "eval-history.db", "eval-results.jsonl", "muxer_history.ner.json"})
            self.assertEqual((restored / "muxer_history.ner.json").read_text(), '{"window": 1}')
            with sqlite3.connect(restored / "eval-history.db") as conn:
                self.assertEqual(conn.execute("SELECT n FROM observations").fetchone(), (7,))

    def test_current_receipts_must_join_the_appended_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "muxer_history.ner.json").write_text("{}")
            journal = root / "eval-results.jsonl"
            journal.write_text('{"observation_id":"old"}\n')
            start = journal_size(root)
            log = root / "current.jsonl"
            target = root / "saved"
            log.write_text('{"record_type":"outcome","observation_id":"old"}\n')
            with self.assertRaisesRegex(ValueError, "missing.*old"):
                snapshot(root, target, [log], start)
            log.write_text('{"record_type":"outcome","observation_id":"current"}\n')
            with self.assertRaisesRegex(ValueError, "missing.*current"):
                snapshot(root, target, [log], start)
            self.assertFalse(target.exists())
            with journal.open("a") as stream:
                stream.write('{"observation_id":"current"}\n')
            self.assertEqual(snapshot(root, target, [log, log], start), 1)

    def test_archive_rejects_paths_outside_the_state_directory(self):
        archive = io.BytesIO()
        with ZipFile(archive, "w") as bundle:
            bundle.writestr("muxer_history.ner.json", "{}")
            bundle.writestr("../other", "unexpected")
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            unpack(archive, Path(directory))
