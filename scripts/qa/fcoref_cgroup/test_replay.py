import copy
import unittest

import replay


class CgroupReplayTests(unittest.TestCase):
    def setUp(self):
        self.daemon = {k: True for k in ("CpuCfsQuota", "CPUSet", "MemoryLimit", "SwapLimit")}
        self.daemon["CgroupVersion"] = "2"
        self.row = {"timed_out": False, "state": {"OOMKilled": False, "ExitCode": 0},
                    "tokens": 522, "budget": 521, "threads": 1,
                    "measurement": {"thread_affinities": ["3"], "encoded_tokens": 522,
                                    "budget": 521, "threads_configured": 1, "warm_ms": [1] * 5,
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
