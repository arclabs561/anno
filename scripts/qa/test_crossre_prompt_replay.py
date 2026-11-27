"""Verify source identity when several CrossRE domains share one native dataset ID."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import crossre_prompt_replay as replay
finally:
    sys.path.pop(0)


class CrossREPromptReplayTest(unittest.TestCase):
    def test_generic_dataset_caches_keep_each_domain_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "files.csv"
            fields = (
                "id",
                "repository",
                "revision",
                "status",
                "subset",
                "split",
                "rows",
                "sha256",
            )
            with manifest.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                for domain in ("music", "news"):
                    for split in ("train", "test"):
                        record = {
                            "sentence": [domain, split],
                            "ner": [],
                            "relations": [],
                        }
                        data = (json.dumps(record) + "\n").encode()
                        (root / f"{domain}-{split}.json").write_bytes(data)
                        writer.writerow(
                            dict(
                                id="CrossRE",
                                repository="mainlp/CrossRE",
                                revision=replay.REVISION,
                                status="ok",
                                subset=domain,
                                split=split,
                                rows=1,
                                sha256=replay.digest(data),
                            )
                        )
            with patch.object(replay, "MANIFEST", manifest):
                for domain in ("music", "news"):
                    self.assertEqual(
                        replay.prepare(root, root, domain)["dataset"], "CrossRE"
                    )
                for domain in ("music", "news"):
                    cached = root / "cache" / domain / "datasets" / "CrossRE.cache"
                    self.assertEqual(
                        cached.read_bytes(), (root / f"{domain}-test.json").read_bytes()
                    )
                (root / "news-test.json").write_bytes(
                    (root / "music-test.json").read_bytes()
                )
                with self.assertRaisesRegex(ValueError, "file digest"):
                    replay.prepare(root, root, "news")

    def test_all_published_arms_share_the_pinned_input_and_gold(self):
        study = replay.MANIFEST.parent
        receipt = json.loads(
            (study / "crossre-full-prompt-comparison.json").read_text()
        )
        with (study / "crossre-prompt-scope.csv").open(newline="") as source:
            tests = {
                row["domain"]: row
                for row in csv.DictReader(source)
                if row["split"] == "test"
            }
        self.assertEqual(receipt["domains"].keys(), tests.keys())
        self.assertEqual(tests.keys(), replay.DOMAINS.keys())
        self.assertEqual(receipt["relex_artifacts_sha256"], replay.RELEX_ARTIFACTS)
        for domain, test in tests.items():
            with self.subTest(domain=domain):
                run = receipt["domains"][domain]
                self.assertEqual(test["sha256"], run["corpus"]["test_sha256"])
                self.assertCountEqual(
                    ((arm["backend"], arm["prompt"]) for arm in run["arms"]),
                    (
                        (b, p)
                        for b in ("gliner_relex", "tplinker")
                        for p in ("fixed", "oracle")
                    ),
                )
                self.assertEqual(
                    len({arm["sample"]["sample_sha256"] for arm in run["arms"]}), 1
                )
                for arm in run["arms"]:
                    metrics, sample = arm["metrics"], arm["sample"]
                    self.assertEqual(sample["unit_count"], int(test["sentences"]))
                    self.assertEqual(
                        sample["gold_item_count"], int(test["gold_relations"])
                    )
                    self.assertEqual(
                        metrics["num_gold_relations"], sample["gold_item_count"]
                    )
                    self.assertEqual(
                        metrics["predicted_endpoint_docs"], sample["unit_count"]
                    )
                    producer = (
                        replay.RELEX
                        if arm["backend"] == "gliner_relex"
                        else "tplinker-heuristic"
                    )
                    self.assertEqual(
                        sample["relation_producer_documents"],
                        {producer: sample["unit_count"]},
                    )
                    if replay.DOMAINS[domain] == "CrossRE":
                        self.assertEqual(arm["input_sha256"], test["sha256"])
                        self.assertEqual(arm["data_source"], "local cache")


if __name__ == "__main__":
    unittest.main()
