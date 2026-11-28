"""Check the scope of the optional, dated S3 comparison for a local collection."""

import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import index_offline_datasets as index


class LocalIndexTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "collection"
        self.evidence = Path(temp.name) / "registry"
        self.root.mkdir()
        self.evidence.mkdir()
        (self.evidence / "dataset-coverage.csv").write_text(
            "id,name,domain,language,outstanding_reason,same_release_as\n"
            "Example,Example,general,en,Release unavailable,\n"
        )

    def run_index(self, snapshot=None):
        with redirect_stdout(io.StringIO()):
            index.build(self.root, self.evidence, [], snapshot)
        return json.loads((self.root / "archives.json").read_text())

    def write_snapshot(self, path, run):
        path.parent.mkdir(exist_ok=True)
        with ZipFile(path, "w") as archive:
            archive.writestr(
                "cache-source-coverage.json",
                json.dumps(
                    {
                        "listing_started_utc": "2026-11-28T00:00:00Z",
                        "workflow_run_id": run,
                        "repository_commit": "1" * 40,
                        "payload_objects_by_outcome": {"matches_own_publisher_file": 1},
                    }
                ),
            )
            archive.writestr(
                "publisher-file-matches.csv",
                "cache_id,publisher_id,match,object_kind,sha256\n"
                f"Example,Example,same registry ID,legacy,{'a' * 64}\n",
            )

    def test_ordinary_local_index_needs_only_the_registry(self):
        result = self.run_index()
        self.assertEqual((result["registry_ids"], result["included_ids"]), (1, 0))
        for key in (
            "s3_snapshot",
            "matched_s3_ids",
            "matched_s3_objects",
            "publisher_matched_s3_objects_in_snapshot",
        ):
            self.assertIsNone(result[key])
        with (self.root / "datasets.csv").open() as stream:
            (row,) = csv.DictReader(stream)
        self.assertEqual(row["matched_dated_s3_object"], "")
        self.assertEqual(row["status"], "absent")
        with (self.root / "s3-file-matches.csv").open() as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])

    def test_a_bundled_snapshot_is_measured_and_multiple_runs_require_selection(self):
        bundled = self.root / "evidence" / "cache-source-evidence-first.zip"
        self.write_snapshot(bundled, "123")
        first = self.run_index()
        self.assertEqual(first["s3_snapshot"]["workflow_run_id"], "123")
        self.assertEqual(first["publisher_matched_s3_objects_in_snapshot"], 1)
        self.assertEqual(first["matched_s3_objects"], 0)
        other = bundled.with_name("cache-source-evidence-second.zip")
        self.write_snapshot(other, "456")
        with self.assertRaisesRegex(ValueError, "--s3-evidence"):
            self.run_index()
        self.assertEqual(self.run_index(other)["s3_snapshot"]["workflow_run_id"], "456")

    def test_an_explicit_external_snapshot_is_not_counted_as_a_local_archive(self):
        path = self.evidence / "selected.zip"
        self.write_snapshot(path, "777")
        native = self.root / "native"
        native.mkdir()
        manifest = json.dumps(
            {"inputs": [{"id": "Example", "variant": "test", "sha256": "a" * 64}]}
        )
        (native / "example.sources.json").write_text(manifest)
        with ZipFile(native / "example.zip", "w") as pack:
            pack.writestr("sources.json", manifest)
        result = self.run_index(path)
        self.assertEqual(result["s3_snapshot"]["workflow_run_id"], "777")
        self.assertEqual(
            (result["matched_s3_objects"], result["matched_s3_ids"]), (1, 1)
        )
        self.assertEqual(
            [a["path"] for a in result["archives"]], ["native/example.zip"]
        )


if __name__ == "__main__":
    unittest.main()
