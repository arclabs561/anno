"""Keep corpus measurements, scored outcomes and CI prechecks independent."""

import csv
import importlib.util
import json
from io import BytesIO
from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

SCRIPTS = Path(__file__).resolve().parents[1]
EVAL = SCRIPTS.parent / "docs" / "eval"
sys.path.insert(0, str(SCRIPTS))
from survey_dataset_ci_outcomes import collect, discover, matrix_record, matrix_rows


def read(name):
    with (EVAL / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


class MatrixReceiptsTest(unittest.TestCase):
    def test_nonzero_cost_and_zero_f1_do_not_decide_the_execution_state(self):
        row = dict(
            record_type="outcome",
            dataset="X",
            task="ner",
            backend="stacked",
            seed=42,
            cost_units=5,
            primary_f1=0,
            hard_junk=False,
        )
        path = "_temp/muxer-decisions.jsonl"
        self.assertEqual(matrix_record(row, path)["status"], "Scored")
        failure = matrix_record(dict(row, hard_junk=True), path)
        self.assertEqual(
            (failure["status"], failure["examples"]), ("Failed evaluation", "")
        )
        self.assertEqual(
            matrix_record(dict(row, cost_units=0), path)["status"], "Empty evaluation"
        )
        self.assertIsNone(matrix_record({"record_type": "decision"}, path))
        with self.assertRaisesRegex(ValueError, "recorded score"):
            matrix_record(dict(row, primary_f1=None), path)

    def test_named_load_pass_and_rejection_carry_no_backend_score(self):
        pool = {
            "slice": "re",
            "seed": 7,
            "pass": "initial",
            "pool": {
                "selected": ["Good"],
                "selected_sources": {"Good": "S3Cache"},
                "rejection_examples": [{"dataset": "Missing", "reason": "unavailable"}],
            },
        }
        with ZipFile(BytesIO(), "w") as archive:
            archive.writestr("_temp/muxer-decisions.jsonl", b"")
            archive.writestr("_temp/matrix-dataset-pools.jsonl", json.dumps(pool))
            records = list(matrix_rows(archive))
        self.assertEqual(
            [(r["id"], r["status"]) for r in records],
            [("Good", "Passed load check"), ("Missing", "Load rejected")],
        )
        self.assertTrue(
            all("backend" not in r and "metric_value" not in r for r in records)
        )

    def test_discovery_keeps_cancellations_and_requires_artifact_hashes(self):
        stamp = "2026-11-27T01:00:00Z"
        push = dict(
            id=3,
            created_at=stamp,
            status="completed",
            head_sha="abc",
            conclusion="cancelled",
        )
        with patch(
            "survey_dataset_ci_outcomes.github_page_items", side_effect=[[push], []]
        ):
            [run] = discover(stamp[:10])["runs"]
        self.assertEqual((run["conclusion"], run["artifacts"]), ("cancelled", []))
        with patch(
            "survey_dataset_ci_outcomes.github_page_items",
            side_effect=[[push], [dict(id=4, name="eval-sanity-report", digest=None)]],
        ):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                discover(stamp[:10])

    def test_archive_hash_precedes_parsing(self):
        manifest = {"runs": [{"run": 1, "artifacts": [{"id": 2, "sha256": "0" * 64}]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "2.zip").write_bytes(b"wrong archive")
            with self.assertRaisesRegex(ValueError, "recorded digest"):
                list(collect(manifest, path))

    def test_each_published_ci_row_identifies_a_manifest_archive(self):
        survey = json.loads((EVAL / "dataset-ci-survey.json").read_text())
        archives = {
            f"{r['run']}/artifacts/{a['id']}"
            for r in survey["runs"]
            for a in r["artifacts"]
        }
        for row in read("dataset-ci-outcomes.csv"):
            self.assertIn(row["source"].split("/runs/")[-1], archives)
            self.assertTrue(row["member"] and row["record"])
            if row["status"] == "Scored":
                self.assertGreater(int(row["examples"]), 0)
            elif row["status"] == "Load rejected":
                self.assertFalse(row["backend"])


@unittest.skipUnless(
    importlib.util.find_spec("matplotlib"), "explorer uses the figure generator"
)
class EvidenceIndexTest(unittest.TestCase):
    def test_registry_addition_cannot_disappear_from_a_frozen_inventory(self):
        from dataset_evidence_index import assemble

        with self.assertRaisesRegex(ValueError, "NewDataset"):
            assemble([{"id": "Old"}], {"Old": {}, "NewDataset": {}}, [], {})

    def test_only_unambiguous_own_file_bytes_supply_an_unreported_split(self):
        from dataset_evidence_index import assemble, matched_splits

        ref = dict(
            cache_id="X",
            publisher_id="X",
            source="https://example.org/tags.tsv",
            sha256="a",
            split_or_subset="test",
        )
        matches = matched_splits(
            [
                ref,
                dict(ref, cache_id="Y"),
                dict(ref, sha256="b", source="https://example.org/tags.zip#one"),
                dict(ref, sha256="c", split_or_subset="train"),
                dict(ref, sha256="c"),
            ]
        )
        coverage = [
            dict(id=i, domain="one", measured=1, tagged_tokens_measured=0)
            for i in ("X", "Y")
        ]
        records = [
            dict(id="X", source_digest=d, status="Scored", scope="sample")
            for d in ("a", "b", "c")
        ]
        index = assemble(
            coverage,
            {"X": {"declared_splits": "train|test"}, "Y": {}},
            records,
            matches,
        )
        self.assertEqual(
            {r["source_digest"]: r["split"] for r in index if r["id"] == "X"},
            {"a": "test", "b": "unreported", "c": "unreported"},
        )
        self.assertEqual(
            [r["status"] for r in index if r["id"] == "Y"], ["No indexed evaluation"]
        )

    def test_chinese_original_coordinates_and_romanian_label_policies_are_distinct(
        self,
    ):
        from dataset_evidence_index import study_outcomes

        outcomes = list(study_outcomes(EVAL))
        chinese = [
            row for row in outcomes if row["source"] == "agcner-source-coordinates.json"
        ]
        self.assertEqual({row["split"] for row in chinese}, {"train", "dev", "test"})
        self.assertEqual(sum(row["examples"] for row in chinese), 59776)
        self.assertTrue(all(row["metric_value"] == 0 for row in chinese))
        romanian = [
            row
            for row in outcomes
            if row["source"] == "histnero-ner-scoring.json" and row["split"] == "dev"
        ]
        self.assertEqual(len({row["scope"] for row in romanian}), 2)
        self.assertEqual(
            {row["detail"] for row in romanian},
            {
                "17 exact, 1222 predicted, 675 gold",
                "83 exact, 1222 predicted, 673 gold",
            },
        )

    def test_all_registry_ids_and_deduplicated_annotation_units_remain_available(self):
        coverage = {r["id"] for r in read("dataset-coverage.csv")}
        outcomes = read("dataset-evaluation-coverage.csv")
        self.assertEqual(coverage, {r["id"] for r in outcomes})
        unknown = {r["id"] for r in outcomes if r["status"] == "No indexed evaluation"}
        self.assertFalse(unknown & {r["id"] for r in outcomes if r["source"]})
        self.assertFalse(
            [
                r
                for r in outcomes
                if r["status"] == "Scored"
                and (not r["backend"] or not r["split"] or not r["source"])
            ]
        )
        units = read("dataset-density-units.csv")
        expected = {
            r["id"]: int(r["n_observation_units"])
            for r in read("dataset-density-ranges.csv")
        }
        self.assertEqual(Counter(u["id"] for u in units), expected)
        for u in units:
            self.assertGreater(int(u["tokens"]), 0)
            self.assertAlmostEqual(
                100 * int(u["tagged"]) / int(u["tokens"]), float(u["percent"]), places=5
            )
            self.assertTrue(u["source"] and u["annotation"] and u["route"])


if __name__ == "__main__":
    unittest.main()
