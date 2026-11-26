# SPDX-License-Identifier: MIT OR Apache-2.0
"""Offline controls using invented relation rows."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class ResidualLedgers(unittest.TestCase):
    def test_crossre_sample_and_duplicate_multiplicity(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work)
            sample = {"unit_count": 1, "sample_sha256": "a" * 64}
            metrics = {"num_gold_relations": 3, "num_predicted_relations": 1, "strict_relation_matches": 1}
            edge = [[0, 1], [2, 3], "r"]
            second_edge = [edge[0], edge[1], "also-r"]
            row = {"index": 0, "text_sha256": "b" * 64, "sample_sha256": sample["sample_sha256"],
                   "gold": [edge, edge, second_edge], "predicted": [edge]}
            domains = {}
            for domain in ("ai", "literature"):
                arms = []
                domains[domain] = {"arms": arms}
                for backend in ("tplinker", "gliner_relex"):
                    for policy in ("fixed", "oracle"):
                        name = f"{domain}-{backend}-{policy}"
                        arms.append({"backend": backend, "prompt": policy, "sample": sample, "metrics": metrics})
                        result = {"success": True, "metrics": metrics, "provenance": {"evaluated_sample": sample}}
                        (path / (name + ".json")).write_text(json.dumps({"results": [result]}))
                        (path / (name + ".rows")).write_text(json.dumps({**row, "gold": list(reversed(row["gold"]))} if policy == "oracle" else row))
            baseline = path / "baseline.json"
            baseline.write_text(json.dumps({"domains": domains}))
            out = path / "errors.json"
            cmd = [sys.executable, "-O", HERE / "classify_crossre.py", path, out, "--baseline", baseline]
            subprocess.run(cmd, check=True, capture_output=True)
            result = json.loads(out.read_bytes())
            self.assertEqual(len(result), 8)
            self.assertTrue(all(arm["missed"] == {"duplicate_beyond_opposite_multiplicity": 1,
                                                   "forward_exact_pair_other_predicate": 1}
                                for arm in result.values()))
            # Keep the sample fingerprint, matched count, FP and FN unchanged.
            # Both altered inventories retain the original aggregate scores.
            arm_path = path / "ai-gliner_relex-fixed.rows"
            reversed_edge = [edge[1], edge[0], edge[2]]
            for replacement in (
                {"text_sha256": "c" * 64},
                {"gold": [reversed_edge, reversed_edge, second_edge], "predicted": [reversed_edge]},
            ):
                with self.subTest(replacement=replacement):
                    arm_path.write_text(json.dumps({**row, **replacement}))
                    out.unlink(missing_ok=True)
                    rejected = subprocess.run(
                        cmd,
                        capture_output=True, text=True,
                    )
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertIn("source or gold relations differ across arms", rejected.stderr)
                    self.assertFalse(out.exists())
            arm_path.write_text(json.dumps(row))
            row["sample_sha256"] = "c" * 64
            (path / "ai-tplinker-fixed.rows").write_text(json.dumps(row))
            self.assertIn("wrong row order or source identity", subprocess.run(cmd, capture_output=True, text=True).stderr)

    def test_jointie_requires_verified_local_trace_and_fresh_external_output(self):
        with tempfile.TemporaryDirectory() as work:
            tmp = Path(work)
            docs = [{"source_sha256": "a" * 64, "edge_trace": [], "decoded_edge_keys": []}]
            cmd = [sys.executable, "-O", HERE / "classify_jointie_extras.py"]
            for index, arm in enumerate(("w16", "w384", "training_routes")):
                data = {"documents": docs, "arm": index}
                raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
                full = tmp / f"{arm}.json"
                full.write_bytes(raw)
                cmd.append(full)
                (tmp / f"jointie_ai64_{arm}.json").write_text(json.dumps({
                    "full_receipt_sha256": hashlib.sha256(raw).hexdigest(),
                    "totals": {"gold_edges_decoded": 0, "decoded_edges": 0}}))
            out = tmp / "output"
            cmd += [out, "--summaries", tmp]
            subprocess.run(cmd, check=True, capture_output=True)
            self.assertIn("File exists", subprocess.run(cmd, capture_output=True, text=True).stderr)
            cmd[3], cmd[4] = cmd[4], cmd[3]
            self.assertIn("differs from the published receipt", subprocess.run(cmd, capture_output=True, text=True).stderr)
            cmd[6] = HERE / "no-output.json"
            self.assertIn("outside Git checkouts", subprocess.run(cmd, capture_output=True, text=True).stderr)


if __name__ == "__main__":
    unittest.main()
