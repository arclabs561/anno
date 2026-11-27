"""Regression checks for equal-partition summaries and training-only estimates."""

import importlib.util
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("matplotlib"), "figures require Matplotlib"
)
class DensityDistributionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        try:
            import plot_density_ranges
            import plot_wikiann_split_density
        finally:
            sys.path.pop(0)
        cls.ranges = plot_density_ranges
        cls.wikiann = plot_wikiann_split_density

    def test_each_distinct_unit_counts_once_in_quantiles(self):
        observation = self.ranges.Observation
        units = [
            observation("X", "publisher", f"file{i}", str(i), (i,), marked, total, "")
            for i, (marked, total) in enumerate(
                ((0, 10000), (2, 10), (8, 10), (10, 10))
            )
        ]
        # A second route for the first file and the large token denominator
        # must both leave its single vote in the unit distribution intact.
        (row,) = self.ranges.summarize([*units, units[0]], {"X": "Example"})
        self.assertEqual(row["n_route_observations"], 5)
        self.assertEqual(row["n_observation_units"], 4)
        self.assertEqual(
            (row["p25_percent"], row["median_percent"], row["p75_percent"]),
            (15, 50, 85),
        )
        self.assertEqual((row["min_tagged"], row["min_tokens"]), (0, 10000))

    def test_small_samples_have_a_median_and_observed_extremes(self):
        units = [
            self.ranges.Observation("X", "file", str(i), str(i), (i,), i, 10, "")
            for i in (1, 4)
        ]
        (row,) = self.ranges.summarize(units, {"X": "Example"})
        self.assertEqual(
            (row["min_percent"], row["median_percent"], row["max_percent"]),
            (10, 25, 40),
        )
        self.assertEqual((row["p25_percent"], row["p75_percent"]), ("", ""))

    def test_direct_archive_members_stay_separate_when_counts_coincide(self):
        direct = [
            dict(
                id="Example",
                status="tagged tokens measured",
                sha256="a" * 64,
                source=f"https://publisher.org/archive.zip/content#{member}",
                split_or_subset="test",
                scope="publisher release file",
                tagged="3",
                tokens="10",
            )
            for member in ("one.bio", "two.bio")
        ]
        observations = self.ranges.collect(direct + direct[:1], [], [], [])
        [summary] = self.ranges.summarize(observations, {"Example": "Example"})
        self.assertEqual(summary["n_observation_units"], 2)
        self.assertEqual(summary["n_route_observations"], 3)

    def test_direct_and_repository_measurements_of_the_same_plain_file_still_join(self):
        source = "https://publisher.org/train.bio"
        direct = [
            dict(
                id="Example",
                status="tagged tokens measured",
                sha256="a" * 64,
                source=source,
                split_or_subset="train",
                scope="publisher file",
                tagged="3",
                tokens="10",
            )
        ]
        repository = [
            dict(
                id="Example",
                status="ok",
                sha256="a" * 64,
                url=source,
                subset="",
                split="train",
                tagged="3",
                tokens="10",
            )
        ]
        [summary] = self.ranges.summarize(
            self.ranges.collect(direct, repository, [], []), {"Example": "Example"}
        )
        self.assertEqual(summary["n_observation_units"], 1)
        self.assertEqual(summary["n_route_observations"], 2)

    def test_anem_verified_training_conversion_keeps_the_distinct_test_split(self):
        archive, converted = sorted(self.ranges.ANEM_TRAIN_SOURCES, reverse=True)
        rows = [
            dict(
                id="AnEM",
                status="tagged tokens measured",
                sha256=archive,
                source="https://www.nactem.ac.uk/anatomy/data/AnEM-1.0.4.tar.gz",
                split_or_subset=split,
                scope="publisher release file",
                tagged="3",
                tokens="10",
            )
            for split in ("train", "test")
        ]
        github = [
            dict(
                id="AnEM",
                status="ok",
                sha256=converted,
                source_url="https://example.org/AnEM_train.conll",
                split="train",
                path="AnEM_train.conll",
                unit=unit,
                value=value,
            )
            for unit, value in (("tagged tokens", "3"), ("tokens", "10"))
        ]
        [summary] = self.ranges.summarize(
            self.ranges.collect(rows, [], [], github), {"AnEM": "AnEM"}
        )
        self.assertEqual(summary["n_observation_units"], 2)
        self.assertEqual(summary["n_route_observations"], 3)

    def test_two_sections_of_one_archive_keep_their_own_denominators(self):
        rows = [
            dict(
                id="NNE",
                status="ok",
                source_url="https://example.org/files.zip",
                split="train",
                path=path,
                sha256="a" * 64,
                unit=unit,
                value=value,
            )
            for path in ("train/01/", "train/02/")
            for unit, value in (("tagged tokens", "3"), ("tokens", "10"))
        ]
        observations = self.ranges.collect([], [], [], rows)
        (summary,) = self.ranges.summarize(observations, {"NNE": "Nested NER"})
        self.assertEqual(summary["n_observation_units"], 2)

    @staticmethod
    def split_rows():
        rows = []
        for lang, rates in {"a": (1, 3, 5), "b": (5, 4, 7), "c": (8, 1, 3)}.items():
            for split, tagged in zip(("train", "validation", "test"), rates):
                rows.append(
                    dict(
                        id="WikiANN",
                        config=lang,
                        split=split,
                        tagged=str(tagged),
                        tokens="10",
                        status="scanned",
                        repository="publisher/corpus",
                        source_revision="source",
                        parquet_revision="conversion",
                    )
                )
        return rows

    def test_other_language_estimate_uses_only_other_training_splits(self):
        rows = self.wikiann.comparison_rows(self.split_rows())
        target = next(row for row in rows if row["config"] == "a")
        # Same language trains at 10%, others train at 50% and 80%.
        self.assertEqual(target["other_languages_train_median_percent"], 65)
        self.assertEqual(target["test_same_language_error_pp"], 40)
        self.assertEqual(target["test_other_languages_error_pp"], 15)
        self.assertEqual(
            (target["validation_tagged"], target["validation_tokens"]), (3, 10)
        )
        self.assertEqual(target["validation_other_languages_error_pp"], 35)

    def test_each_language_must_have_one_complete_pinned_triplet(self):
        rows = self.split_rows()
        for invalid in (
            rows[:-1],
            [*rows, dict(rows[0])],
            [dict(rows[0], status="scanned_unlabelled"), *rows[1:]],
            [dict(rows[0], parquet_revision="different"), *rows[1:]],
        ):
            with self.subTest(invalid=invalid[-1]):
                with self.assertRaises(ValueError):
                    self.wikiann.comparison_rows(invalid)

    def test_multiconer_excludes_pooled_and_requires_complete_pinned_sources(self):
        revision = "a" * 40
        ledger = [
            dict(
                id="MultiCoNERv2",
                config=config,
                split=split,
                status="scanned",
                repository="MultiCoNER/multiconer_v2",
                source_revision=revision,
                parquet_revision="",
                payload_sha256="b" * 64,
                payload_url=f"https://example.org/resolve/{revision}/{config}/{split}.conll",
                tokens="100",
                tagged=str(tagged),
            )
            for config, tagged in [
                ("AA-Language", 10),
                ("BB-Language", 30),
                ("MULTI-Multilingual", 95),
            ]
            for split in ["train", "dev", "test"]
        ]
        rows = self.wikiann.comparison_rows(ledger, "MultiCoNERv2")
        self.assertEqual(
            [row["config"] for row in rows], ["AA-Language", "BB-Language"]
        )
        self.assertEqual(rows[0]["validation_same_language_error_pp"], 0.0)
        self.assertEqual(rows[0]["test_other_languages_error_pp"], 20.0)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.wikiann.comparison_rows(ledger[1:], "MultiCoNERv2")
        ledger[0]["payload_sha256"] = ""
        with self.assertRaisesRegex(ValueError, "Missing pinned"):
            self.wikiann.comparison_rows(ledger, "MultiCoNERv2")


if __name__ == "__main__":
    unittest.main()
