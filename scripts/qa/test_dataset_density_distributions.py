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


if __name__ == "__main__":
    unittest.main()
