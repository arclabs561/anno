import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import annotation_archive_survey as survey


class AnnotationArchiveSurveyTest(unittest.TestCase):
    def test_germeval_counts_union_including_tokens_tagged_only_on_inner_layer(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "nested.tsv"
            path.write_text("1\tX\tB-ORG\tB-LOC\n2\tY\tO\tB-PER\t\n3\tZ\tO\tO\n")
            (row,) = survey.germeval(path, "test")
            self.assertEqual((row["records"], row["tokens"], row["tagged"]), (1, 3, 2))
            self.assertIn("Embedded layer: 2 spans on 2 tokens", row["reason"])

    def test_elgold_counts_linked_and_nil_in_one_genre(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "release.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("data/1_1.txt", "{{A|PERSON|Alice}} and {{A|PERSON|}}")
            (row,) = survey.elgold(path)
            self.assertEqual((row["split_or_subset"], row["records"]), ("News", 1))
            self.assertIn("2 mentions, 1 Wikipedia-linked, 1 NIL", row["reason"])
            with self.assertRaisesRegex(ValueError, "pinned release"):
                survey.measure("ELGold", survey.ELGOLD, path)

    def test_semeval_requires_exact_unicode_character_offsets(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "trial.xml"
            text = '<sentences><sentence id="1"><text>Le café</text><aspectTerms><aspectTerm term="café" from="3" to="7" polarity="positive"/></aspectTerms></sentence></sentences>'
            path.write_text(text, encoding="utf-8")
            (row,) = survey.semeval(path, "restaurants")
            self.assertEqual(row["records"], 1)
            path.write_text(text.replace('to="7"', 'to="8"'), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "offsets"):
                survey.semeval(path, "restaurants")

    def test_saraga_compares_recording_ids_and_timestamps_across_two_indexes(self):
        import json

        first = "00000000-0000-0000-0000-000000000001"
        second = "00000000-0000-0000-0000-000000000002"
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "release.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for tradition, raag, taal in [
                    ("carnatic", "raaga", "taala"),
                    ("hindustani", "raags", "taals"),
                ]:
                    prefix = f"saraga-{survey.SARAGA_COMMIT}/dataset/{tradition}/"
                    archive.writestr(
                        prefix + "file_paths.csv", f",filepath,mbid\n0,Concert/Song/Other,{first}\n"
                    )
                    record = dict(mbid=first, title="Song", length=1000, artists=[])
                    record.update({raag: [], taal: []})
                    archive.writestr(prefix + "Concert/Song/Song.json", json.dumps(record))
                    archive.writestr(prefix + "Concert/Song/Song.sama-manual.txt", "-1\n0.5\n1.5\n")
                    archive.writestr(
                        prefix + "Concert/Extra/Extra.json", json.dumps(dict(record, mbid=second))
                    )
                    archive.writestr(prefix + "Concert/Extra/Extra.sama-manual.txt", "")
            rows = survey.measure("Saraga", survey.SARAGA_ZIP, path)
        self.assertEqual(len(rows), 8)
        self.assertEqual({r["records"] for r in rows}, {1})
        self.assertTrue(all(r["bytes"] and r["source"].startswith(survey.SARAGA_RAW) for r in rows))
        index = next(r for r in rows if r["unit"] == "distinct indexed recording UUIDs")
        self.assertIn("1 additional UUIDs and 1 index path differences", index["reason"])
        self.assertIn("1 negative and 1 after-duration", index["reason"])
        self.assertIn("1 empty files", index["reason"])
