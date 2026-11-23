import copy
import json
import multiprocessing
import sys
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import replay

SUFFIX = "Alice reviewed the proposal. She approved it. Bob objected. She overruled him."


class CgroupReplayTests(unittest.TestCase):
    def setUp(self):
        self.daemon = {k: True for k in ("CpuCfsQuota", "CPUSet", "MemoryLimit", "SwapLimit")}
        self.daemon["CgroupVersion"] = "2"
        self.row = {"wall_seconds": 3, "timed_out": False, "state": {"OOMKilled": False, "ExitCode": 0},
                    "tokens": 522, "budget": 521, "threads": 1,
                    "measurement": {"thread_affinities": ["3"], "encoded_tokens": 522,
                                    "budget": 521, "threads_configured": 1, "warm_ms": [1] * 5,
                                    "load_ms": 100, "first_call_ms": 1,
                                    "text": "x " * 503 + SUFFIX,
                                    "predictions": {"error": "Invalid input: coreference document has 522 tokens, exceeding max_doc_tokens 521"},
                                    "cgroup": {"cpu.max": "100000 100000",
                                               "memory.max": "2147483648",
                                               "memory.swap.max": "0", "memory.events": "oom 0\noom_kill 0",
                                               "cpuset.cpus.effective": "3"}}}

    def test_delegated_daemon_and_matching_observed_controls(self):
        replay.require_controls(self.daemon)
        replay.verify(self.row, 3)

    def test_each_missing_daemon_control_aborts(self):
        for key in self.daemon:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                replay.require_controls({**self.daemon, key: False})

    def test_inner_limits_must_match_even_when_docker_exits_successfully(self):
        for key, value in (("cpu.max", "400000 100000"), ("cpuset.cpus.effective", "0-4"),
                           ("memory.max", "17179869184"), ("memory.swap.max", "max"),
                           ("memory.events", "oom 1\noom_kill 1")):
            row = copy.deepcopy(self.row)
            row["measurement"]["cgroup"][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                replay.verify(row, 3)

    def test_missing_samples_worker_escape_or_wrong_result_fail(self):
        for key,value in (("warm_ms",[]),("thread_affinities",["3","4"]),("encoded_tokens",521)):
            row = copy.deepcopy(self.row)
            row["measurement"][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                replay.verify(row, 3)

    def test_timeout_and_oom_retain_failure_status(self):
        for key in ("timed_out", "OOMKilled"):
            row = copy.deepcopy(self.row)
            if key == "timed_out":
                row[key] = True
            else:
                row["state"][key] = True
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                replay.verify(row, 3)

    def test_budget_refusals_and_accepted_boundaries(self):
        for n, budget in replay.CASES:
            with self.subTest(n=n, budget=budget):
                row = copy.deepcopy(self.row)
                row.update(tokens=n, budget=budget)
                m = row["measurement"]
                m.update(encoded_tokens=n, budget=budget)
                m["text"] = "x " * (n - 19) + SUFFIX
                if n > budget:
                    for predictions in ({}, {"mentions": [], "clusters": []}, {"error": "backend unavailable"}):
                        m["predictions"] = predictions
                        with self.assertRaises(RuntimeError):
                            replay.verify(row, 3)
                    m["predictions"] = {"error": f"Invalid input: coreference document has {n} tokens, exceeding max_doc_tokens {budget}"}
                    replay.verify(row, 3)
                    continue
                m["predictions"] = {"error": "budget exceeded"}
                with self.assertRaises(RuntimeError):
                    replay.verify(row, 3)
                # Build expected clusters from the words in the fixture,
                # independently of the verifier's integer-offset reference.
                text = m["text"]
                def span(word, last=False):
                    start = text.rindex(word) if last else text.index(word)
                    return [start, start + len(word)]
                clusters = [[span("Alice"), span("She"), span("She", last=True)],
                            [span("the proposal"), span("it")],
                            [span("Bob"), span("him")]]
                if n == 521:
                    reference = Path(__file__).resolve().parents[3] / "crates/anno/tests/fixtures/fcoref_word_budget.json"
                    mentions = [[s, e, text[s:e], score] for s, e, score in json.loads(reference.read_text())]
                else:
                    mentions = [[s, e, text[s:e], 1.0] for cluster in clusters for s, e in cluster]
                m["predictions"] = {"clusters": clusters, "mentions": mentions}
                replay.verify(row, 3)
                m["predictions"]["clusters"] = []
                with self.assertRaises(RuntimeError):
                    replay.verify(row, 3)

    def test_nonfinite_or_nonpositive_timing_is_rejected(self):
        for key in ("warm_ms", "first_call_ms", "load_ms", "wall_seconds"):
            for value in (float("nan"), float("inf"), 0, -1, True):
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                    row = copy.deepcopy(self.row)
                    target = row if key == "wall_seconds" else row["measurement"]
                    target[key] = [value] * 5 if key == "warm_ms" else value
                    replay.verify(row, 3)

    def run_fixture(self, output, callback):
        argv = ["replay", "--output", str(output), "--binary", "/unused/bin",
                "--image", "fixture", "--model", "/unused/model", "--ort", "/unused/ort",
                "--cpu", "3", "--repeats", "1"]
        def digest(path):
            return replay.MODEL_SHA.get(path.name, "2d518e11c7767f9fdc4413bdbeb45acf30db504ff76a82fc19d9238a3f9f6c24")
        with patch.object(sys, "argv", argv), patch.object(replay, "sha", side_effect=digest), \
                patch.object(replay, "docker", side_effect=[json.dumps(self.daemon), "image-id"]), \
                patch.object(replay, "CASES", [(522, 521)]), patch.object(replay, "run_case", side_effect=callback):
            replay.main()

    def test_receipt_lifecycle_and_exclusive_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            def measured(args, image, threads, tokens, budget):
                snapshot = json.loads(path.read_text())
                self.assertEqual(snapshot["status"], "in_progress")
                self.assertEqual(snapshot["active_case"]["threads"], threads)
                row = copy.deepcopy(self.row)
                row["threads"] = row["measurement"]["threads_configured"] = threads
                return row

            self.run_fixture(path, measured)
            complete = json.loads(path.read_text())
            self.assertEqual((complete["status"], len(complete["runs"])), ("completed", 2))
            self.assertNotIn("active_case", complete)
            with self.assertRaises(FileExistsError):
                self.run_fixture(path, measured)
            self.assertEqual(json.loads(path.read_text()), complete)

            failed_path = Path(directory) / "failed.json"
            def invalid(args, image, threads, tokens, budget):
                row = copy.deepcopy(self.row)
                row["threads"] = row["measurement"]["threads_configured"] = threads
                if threads == 2:
                    row["measurement"]["warm_ms"][0] = float("nan")
                return row
            with self.assertRaisesRegex(RuntimeError, "finite positive"):
                self.run_fixture(failed_path, invalid)
            failure = json.loads(failed_path.read_text())
            self.assertEqual((failure["status"], len(failure["runs"])), ("failed", 1))
            self.assertEqual(failure["active_case"]["threads"], 2)
            self.assertIn("failed_run", failure)

    def test_previous_valid_row_cannot_satisfy_the_next_case(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            with self.assertRaisesRegex(RuntimeError, "different scheduled case"):
                self.run_fixture(path, lambda *args: copy.deepcopy(self.row))
            failure = json.loads(path.read_text())
            self.assertEqual([row["threads"] for row in failure["runs"]], [1])
            self.assertEqual((failure["status"], failure["active_case"]["threads"]), ("failed", 2))

    def test_preflight_refusal_preserves_a_failed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            self.daemon["CPUSet"] = False
            with self.assertRaisesRegex(RuntimeError, "Docker must support"):
                self.run_fixture(path, None)
            failure = json.loads(path.read_text())
            self.assertEqual((failure["status"], failure["runs"]), ("failed", []))
            self.assertFalse(failure["daemon"]["CPUSet"])

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "Linux Docker replay")
    def test_terminating_during_next_case_keeps_previous_verified_run(self):
        ctx = multiprocessing.get_context("fork")
        entered = ctx.Event()
        def block_second(args, image, threads, tokens, budget):
            if threads == 2:
                entered.set()
                time.sleep(30)
            return copy.deepcopy(self.row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            process = ctx.Process(target=self.run_fixture, args=(path, block_second))
            process.start()
            try:
                self.assertTrue(entered.wait(10))
            finally:
                process.terminate()
                process.join(10)
            self.assertFalse(process.is_alive())
            snapshot = json.loads(path.read_text())
            self.assertEqual(snapshot["status"], "in_progress")
            self.assertEqual([r["threads"] for r in snapshot["runs"]], [1])
            self.assertEqual(snapshot["active_case"]["threads"], 2)
