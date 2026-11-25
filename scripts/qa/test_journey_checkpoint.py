import unittest

from journey_checkpoint import MAX_ARCHIVE_BYTES, WORKFLOW, select


class TrustedArtifactTests(unittest.TestCase):
    def setUp(self):
        self.producer = dict(id=41, repository_id=3, head_repository_id=3, head_branch="main", head_sha="abc")
        self.artifact = dict(id=51, name="expected", expired=False, workflow_run=self.producer,
                             created_at="2026-11-25T02:00:00Z", size_in_bytes=300)
        self.run = dict(id=41, workflow_id=7, path=WORKFLOW, event="schedule", status="completed",
                        conclusion="success", repository={"full_name": "owner/repo"},
                        head_repository={"id": 3}, head_branch="main", head_sha="abc", run_attempt=1,
                        run_started_at="2026-11-25T01:50:00Z", updated_at="2026-11-25T02:01:00Z")

    def choose(self, artifacts, lookup=None):
        return select(artifacts, "expected", "owner/repo", 3, "main", 7, lookup or (lambda *_: self.run))

    def test_accept_only_compatible_successful_producer(self):
        self.assertEqual(self.choose([self.artifact]), (41, 51))
        for override in ({"workflow_id": 4}, {"path": ".github/workflows/ci.yml"}, {"conclusion": "failure"},
                         {"event": "pull_request"}, {"head_repository": {"id": 4}},
                         {"repository": {"full_name": "someone/repo"}}, {"head_branch": "pr/2"},
                         {"head_sha": "changed"}, {"status": "in_progress"}):
            self.assertIsNone(self.choose([self.artifact], lambda *_: {**self.run, **override}))
        for override in ({"expired": True}, {"name": "different-cohort"},
                         {"workflow_run": {**self.producer, "head_repository_id": 42}}):
            self.assertIsNone(self.choose([{**self.artifact, **override}]))
        with self.assertRaises(ValueError):
            self.choose([{**self.artifact, "size_in_bytes": MAX_ARCHIVE_BYTES + 1}])

    def test_retry_follows_the_attempt_that_published_before_it(self):
        retry = {**self.run, "run_attempt": 2, "status": "in_progress", "conclusion": None,
                 "run_started_at": "2026-11-26T01:00:00Z"}
        self.assertEqual(self.choose([self.artifact],
            lambda _id, attempt: self.run if attempt == 1 else retry), (41, 51))
        inconclusive = {**self.run, "conclusion": "failure"}
        self.assertIsNone(self.choose([self.artifact],
            lambda _id, attempt: inconclusive if attempt == 1 else retry))

    def test_absence_and_lookup_failure_are_distinct(self):
        self.assertIsNone(self.choose([]))
        def unavailable(*_):
            raise OSError("api unavailable")
        with self.assertRaises(OSError):
            self.choose([self.artifact], unavailable)
