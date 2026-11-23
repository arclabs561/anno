import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from brief_round import evidence
from domain_coverage import coverage_summary, render_coverage


def site(topic, host, digest="0", final_host="common.example"):
    url = f"https://{host}/doc"
    return {
        "source_url": url, "selected_url": url, "domain": topic,
        "checks": [{"backend": model, "status": "ok", "final_url": f"https://{final_host}/doc",
                    "text_hash": "xxh3:" + digest * 16} for model in ["pattern", "stacked"]],
    }


class CoverageTest(unittest.TestCase):
    def setUp(self):
        self.rows = [site("art", "art.example"), site("law", "law.example")]
        self.data = {"kind": "anno.qa.cross-domain", "schema_version": 1, "seed": 42,
                     "backends": ["pattern", "stacked"], "sources": self.rows,
                     "eligible_sources": {"art": 2, "law": 1, "science": 1}}

    def test_two_catalog_topics_can_share_one_successful_response(self):
        q = coverage_summary(self.data)
        self.assertEqual((q["matched"], q["final_hosts"], q["distinct_texts"]), (2, 1, 1))
        self.assertEqual(q["shared_across_domains"], 1)
        self.assertEqual(q["domains"]["art"]["exclusive_texts"], 0)
        with TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "publishers.json"
            receipt.write_text(json.dumps(self.data))
            handoff = evidence(receipt)
        self.assertIn("topics sampled: 2/3", handoff)
        self.assertIn("| science | 1 | 0 | 0 | 0 | 0 |", handoff)
        self.assertNotIn("art.example", handoff)

    def test_distinct_inputs_on_one_host_and_input_drift(self):
        self.rows[1] = site("law", "law.example", "1")
        q = coverage_summary(self.data)
        self.assertEqual((q["matched"], q["final_hosts"], q["distinct_texts"]), (2, 1, 2))
        self.rows[0]["checks"][1]["final_url"] = "https://another.example/other"
        self.rows[1]["checks"][0]["text_hash"] = "xxh3:" + "2" * 16
        q = coverage_summary(self.data)
        self.assertEqual((q["matched"], q["failures"]), (0, {"input_mismatch": 2}))

    def test_discovery_and_backend_failures_keep_denominator(self):
        self.rows[0].update(selected_url=None, checks=[])
        self.rows[1]["checks"][0].update(status="timeout", final_url=None, text_hash=None)
        q = coverage_summary(self.data)
        self.assertEqual((q["selected"], q["matched"]), (2, 0))
        self.assertEqual(q["failures"], {"discovery": 1, "backend": 1})
        self.data["replay"] = {"reference_sha256": "a" * 64,
                               "outcomes": {s["source_url"]: "unverified" for s in self.rows}}
        self.assertIn("2 unverified", render_coverage(self.data))
        self.data["replay"]["outcomes"][self.rows[0]["source_url"]] = "matched"
        with self.assertRaisesRegex(ValueError, "verified current input"):
            coverage_summary(self.data)

    def test_incomplete_or_duplicate_checks_do_not_establish_coverage(self):
        self.rows[0]["checks"].pop()
        with self.assertRaisesRegex(ValueError, "missing requested backend"):
            coverage_summary(self.data)
        self.rows[0]["checks"].append(self.rows[0]["checks"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            coverage_summary(self.data)
