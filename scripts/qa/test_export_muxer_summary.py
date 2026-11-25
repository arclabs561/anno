import json
import unittest

from export_muxer_summary import BACKENDS, SEEDS, SCORER, summarize

REVISION = "a1" * 20


def receipts():
    rows, outcomes = [], []
    for backend in BACKENDS:
        for seed in SEEDS:
            key = f"private-run-id/{backend}/{seed}"
            policy = json.dumps({"schema":"anno-eval-policy-cohort-v1", "max_examples":32, "scorer":SCORER})
            meta = {
                "success":True, "input_text":"sensitive fixture",
                "provenance":{"schema_version":1, "observation_id":key,
                              "backend":{"effective":backend},
                              "runtime":{"scheduling":{"strategy":"sequential"},"seed":seed,"cached_only":True},
                              "dataset":{"source_url":"https://private.invalid"}},
            }
            rows.append(dict(observation_id=key, task="NER", dataset="WikiGold", backend=backend,
                             seed=seed,n=32,f1=0.5,error=None,git_commit=REVISION[:8],
                             policy_cohort_key=policy,metadata=json.dumps(meta)))
            outcomes.append(dict(schema_version=3, record_type="outcome",strategy="fixed",ok=True,
                                 observation_id=key, dataset="WikiGold", backend=backend,
                                 primary_f1=0.5,policy_cohort_key=policy))
    return rows,outcomes


class PublicSummaryTest(unittest.TestCase):
    def test_aggregate_is_order_independent_bounded_and_safe_to_publish(self):
        rows, outcomes = receipts()
        expected = summarize(rows, outcomes, REVISION)
        self.assertEqual([c["evaluated_trials"] for c in expected["cells"]],[3,3])
        self.assertEqual(expected, summarize(list(reversed(rows))+rows, outcomes+outcomes, REVISION))
        encoded = json.dumps(expected)
        self.assertLess(len(encoded.encode()), 2048)
        for private in ["private-run-id","sensitive fixture","private.invalid","observation_id","source_url"]:
            self.assertNotIn(private, encoded)

    def test_completed_trials_accept_explicit_skip_schema(self):
        rows, outcomes = receipts()
        original = summarize(rows, outcomes, REVISION)
        meta = json.loads(rows[0]["metadata"])
        for version in (2, 3, True):
            meta["provenance"]["schema_version"] = version
            sample = [dict(rows[0], metadata=json.dumps(meta)), *rows[1:]]
            if version == 2:
                self.assertEqual(summarize(sample, outcomes, REVISION), original)
            else:
                with self.assertRaises(ValueError):
                    summarize(sample, outcomes, REVISION)

    def test_unattempted_missing_and_conflicting_evidence_is_rejected(self):
        rows, outcomes = receipts()
        with self.assertRaises(ValueError):
            summarize(rows, outcomes[1:], REVISION)
        with self.assertRaises(ValueError):
            summarize(rows + [dict(rows[0], f1=0.7)], outcomes, REVISION)
        meta = json.loads(rows[0]["metadata"])
        meta["provenance"]["runtime"]["scheduling"]["strategy"] = "not_run"
        with self.assertRaises(ValueError):
            summarize([dict(rows[0],metadata=json.dumps(meta))]+rows[1:],outcomes,REVISION)
