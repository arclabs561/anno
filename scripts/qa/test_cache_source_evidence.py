"""Test the live evidence selector without AWS credentials or publisher data."""

import hashlib
import json
import unittest

import cache_source_evidence as evidence


def digest(data):
    return hashlib.sha256(data).hexdigest()


def listed(key, data):
    return {"Key": "datasets/" + key, "Size": len(data), "ETag": '"etag"'}


class CacheSourceEvidenceTest(unittest.TestCase):
    def test_scoped_reads_deduplication_and_pointer_generation(self):
        old, new, unknown = b"published old", b"published new", b"unidentified payload"
        rows = {
            "A.cache": new,
            f"by-sha256/{digest(new)}/A.cache": new,
            f"by-sha256/{digest(old)}/A.cache": old,
            "A.cache.latest.json": json.dumps(
                {"sha256": digest(new), "dataset": "A.cache"}
            ).encode(),
            "A.cache.manifest.json": b"manifest",
            "B.cache": new,
            "C.cache": unknown,
            "Protected.cache": b"never read",
            "Unmeasured.cache": b"never read",
            "Manual.cache": b"never read",
            "unknown-secret/filename.json": b"never read",
            "C.cache*": b"never read",
        }
        reasons = {
            "A": "eligible",
            "B": "eligible",
            "C": "eligible",
            "D": "eligible",
            "Protected": "shared_read_exclusion",
            "Unmeasured": "no_independent_corpus_measurement",
            "Manual": "requires_separate_source_access",
        }
        refs = {
            digest(data): [
                dict(
                    publisher_id="A",
                    ledger="file",
                    source="https://example.org/release",
                    publisher_path="source/test.json",
                    split_or_subset="test",
                    unit="questions",
                    records="12",
                    tokens="",
                    tagged="",
                )
            ]
            for data in (new, old)
        }

        def read(items):
            self.assertEqual({o.dataset for o in items}, {"A", "B", "C"})
            self.assertEqual(len(items), 6)
            for obj in items:
                yield obj, evidence.object_value(obj, [rows[obj.key.removeprefix("datasets/")]])

        summary, matches = evidence.audit(
            reasons, refs, [listed(k, v) for k, v in rows.items()], read
        )
        self.assertEqual(summary["objects_listed"], len(rows))
        self.assertEqual(sum(summary["objects_by_kind"].values()), len(rows))
        self.assertEqual(sum(summary["payload_objects_by_outcome"].values()), 8)
        self.assertEqual(summary["eligible_snapshot_positions"], {"current": 1, "superseded": 1})
        self.assertEqual(summary["distinct_matched_publisher_digests"], 2)
        self.assertEqual(summary["digests_shared_by_multiple_cache_ids"], 1)
        self.assertEqual(summary["registry_ids_with_own_publisher_match"], 1)
        self.assertEqual(
            summary["registry_ids_by_eligibility"]["eligible"],
            {"with_payload": 3, "without_payload": 1},
        )
        self.assertEqual(
            [
                (r["publisher_id"], r["records"], r["match"])
                for r in matches
                if r["cache_id"] == "B"
            ],
            [("A", "12", "different registry ID")],
        )
        serialized = json.dumps([summary, matches])
        self.assertNotIn("unknown-secret", serialized)
        self.assertNotIn(digest(unknown), serialized)

    def test_integrity_and_missing_pointer(self):
        other = b"old"
        sha = digest(other)
        row = listed(f"by-sha256/{sha}/A.cache", other)
        obj = evidence.classify(row, {"A": "eligible"})
        self.assertEqual(evidence.object_value(obj, [b"new"]), ("snapshot_digest_mismatch", None))
        self.assertEqual(evidence.object_value(obj, [b""]), ("size_changed", None))
        pointer = evidence.classify(listed("A.cache.latest.json", b"[]"), {"A": "eligible"})
        self.assertEqual(evidence.object_value(pointer, [b"[]"]), ("invalid_pointer", None))
        for value in [None, digest(b"not listed")]:
            objects = [row]
            values = {obj: ("read", sha)}
            if value:
                objects.append(listed("A.cache.latest.json", b"[]"))
                values[pointer] = ("read", value)
            report, _ = evidence.audit({"A": "eligible"}, {}, objects, lambda _: values.items())
            self.assertEqual(report["eligible_snapshot_positions"], {"unresolved": 1})

    def test_exact_key_session_limit_and_real_ledger_policy(self):
        reasons, refs = evidence.catalog(evidence.ROOT)
        self.assertEqual(reasons["LitBank"], "shared_read_exclusion")
        self.assertEqual(reasons["SciERCNER"], "shared_read_exclusion")
        for controlled in (
            "THYME",
            "DELICATE",
            "RadCoref",
            "FrenchClinicalNER",
            "I2B22010",
            "OntoNotesSample",
        ):
            self.assertEqual(reasons[controlled], "requires_separate_source_access")
        self.assertEqual(reasons["MATRES"], "public_release_covers_only_part_of_source")
        self.assertTrue(
            all(
                "/tree/" not in r["source"]
                for records in refs.values()
                for r in records
                if r["ledger"] == "dataset-hf-splits.csv"
            )
        )
        file = refs["c797a64d0cf73ed058363f77671486bcfd413e70fda1726ad4a6ba624455225f"][0]
        self.assertEqual(
            (file["publisher_id"], file["records"], file["tokens"], file["tagged"]),
            ("WikiGold", "1696", "39007", "6431"),
        )
        objects = [
            evidence.classify(listed(f'by-sha256/{"a"*64}/{i}.cache', b""), reasons)
            for i in reasons
            if reasons[i] == "eligible"
        ]
        groups = list(evidence.batches("arc-anno-data", objects))
        self.assertGreater(len(groups), 1)
        self.assertEqual([o.key for batch in groups for o in batch], [o.key for o in objects])
        for batch in groups:
            policy = evidence.policy_for("arc-anno-data", [o.key for o in batch])
            self.assertLessEqual(len(policy), 2048)
            self.assertNotIn("*", policy)
            (statement,) = json.loads(policy)["Statement"]
            self.assertEqual(statement["Action"], "s3:GetObject")


if __name__ == "__main__":
    unittest.main()
