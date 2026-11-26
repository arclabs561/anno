"""Exercise the dataset coverage CLI with small, offline evidence ledgers."""

import csv
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

COMPILER = Path(__file__).resolve().parents[1] / "plot_measurement_coverage.py"


@unittest.skipUnless(
    importlib.util.find_spec("matplotlib"), "coverage figures require Matplotlib"
)
class DatasetFigureCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / "inventory.csv"
            ids = ["Overlap", "CI", "Provider", "Archive", "Unlabelled", "Records"]
            with inventory.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "id",
                        "name",
                        "format",
                        "language",
                        "domain",
                        "access",
                        "url",
                        "result",
                    ]
                )
                for ident in ids:
                    writer.writerow(
                        [
                            ident,
                            ident,
                            "BIO",
                            "en",
                            "example",
                            "Public",
                            "",
                            "Registry metadata only",
                        ]
                    )
            ledgers = {
                "direct-files": (
                    "id,status,records,tokens,tagged,unit\n"
                    "Overlap,tagged tokens measured,,8,0,tokens\n"
                ),
                "hf-splits": (
                    "id,status,rows,tokens,tagged,provider_rows\n"
                    "Overlap,scanned,2,8,0,2\n"
                    "Provider,provider_only,,,,20\n"
                    "Unlabelled,scanned_unlabelled,4,12,0,4\n"
                    "Records,scanned_records,6,,,6\n"
                ),
                "github-sources": (
                    "id,status,metric,unit,value,sha256,revision,path\n"
                    f"Archive,ok,count,archive members,9,{'a' * 64},rev,data.zip\n"
                ),
                "ci-prechecks": "dataset,run_id,source\nCI,123,S3 cache\n",
            }
            command = [sys.executable, str(COMPILER), str(inventory), str(root)]
            for option, contents in ledgers.items():
                path = root / (option + ".csv")
                path.write_text(contents)
                command.extend(["--" + option, str(path)])
            subprocess.run(
                command,
                check=True,
                env={
                    **os.environ,
                    "MPLCONFIGDIR": str(root / "mpl"),
                    "XDG_CACHE_HOME": str(root / "cache"),
                },
            )
            with (root / "dataset-coverage.csv").open(newline="") as file:
                cls.rows = list(csv.DictReader(file))
            cls.by_id = {row["id"]: row for row in cls.rows}

    def test_overlapping_routes_contribute_one_registry_entry(self):
        self.assertCountEqual(
            [row["id"] for row in self.rows],
            ["Overlap", "CI", "Provider", "Archive", "Unlabelled", "Records"],
        )
        self.assertEqual(sum(int(row["measured"]) for row in self.rows), 3)
        self.assertEqual(
            set(self.by_id["Overlap"]["measured_routes"].split(" | ")), {"direct", "hf"}
        )

    def test_noncontent_evidence_does_not_qualify(self):
        self.assertEqual(self.by_id["CI"]["ci_s3_precheck_runs"], "1")
        self.assertEqual(self.by_id["Provider"]["hf_provider_counts_only"], "1")
        self.assertEqual(self.by_id["Archive"]["github_measured_files"], "0")
        for ident in ("CI", "Provider", "Archive"):
            with self.subTest(ident=ident):
                self.assertEqual(self.by_id[ident]["measured"], "0")
                self.assertEqual(self.by_id[ident]["tagged_tokens_measured"], "0")

    def test_verified_zero_is_a_valid_tag_count(self):
        row = self.by_id["Overlap"]
        self.assertEqual(row["tagged_tokens_measured"], "1")
        self.assertEqual(set(row["tagged_routes"].split(" | ")), {"direct", "hf"})

    def test_unlabelled_scan_only_qualifies_for_record_coverage(self):
        for ident in ("Unlabelled", "Records"):
            with self.subTest(ident=ident):
                row = self.by_id[ident]
                self.assertEqual(row["measured"], "1")
                self.assertEqual(row["hf_measured_splits"], "1")
                self.assertEqual(row["hf_tagged_splits"], "0")
                self.assertEqual(row["tagged_tokens_measured"], "0")


class PublishedCoverageTest(unittest.TestCase):
    """The published coverage and density CSVs must describe the same survey."""

    def read_rows(self, name):
        path = COMPILER.parents[1] / "docs" / "eval" / name
        with path.open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))

    def test_every_inventory_entry_has_one_outcome(self):
        inventory = self.read_rows("dataset-figure-inventory.csv")
        coverage = self.read_rows("dataset-coverage.csv")
        self.assertCountEqual(
            [row["id"] for row in coverage], [row["id"] for row in inventory]
        )
        self.assertEqual(len(coverage), len({row["id"] for row in coverage}))
        for row in coverage:
            with self.subTest(dataset=row["id"]):
                self.assertEqual(row["measured"] == "1", bool(row["measured_routes"]))
                if row["tagged_tokens_measured"] != "1":
                    self.assertTrue(row["outstanding_reason"])

    def test_density_endpoints_cover_each_token_measured_entry(self):
        coverage = self.read_rows("dataset-coverage.csv")
        density = self.read_rows("dataset-density-ranges.csv")
        self.assertCountEqual(
            [row["id"] for row in density],
            [row["id"] for row in coverage if row["tagged_tokens_measured"] == "1"],
        )
        for row in density:
            with self.subTest(dataset=row["id"]):
                for end in ("min", "max"):
                    tagged, tokens = int(row[end + "_tagged"]), int(
                        row[end + "_tokens"]
                    )
                    self.assertGreater(tokens, 0)
                    self.assertGreaterEqual(tagged, 0)
                    self.assertLessEqual(tagged, tokens)
                    self.assertAlmostEqual(
                        100 * tagged / tokens, float(row[end + "_percent"])
                    )
                    self.assertTrue(row[end + "_source"])
                self.assertLessEqual(
                    float(row["min_percent"]), float(row["max_percent"])
                )


if __name__ == "__main__":
    unittest.main()
