import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset_file_survey as survey
import ncerb_survey
import publisher_release_survey as publisher


class PublisherReleaseSurveyTests(unittest.TestCase):
    def test_benchmark_offset_and_label_filters(self):
        text = "Pain pain pain drug"
        mentions = [
            ("Drug", "drug", 0, 4),
            ("Condition", "pain", 0, 4),
            ("Device", "Pain", 0, 4),
            ("Procedure", "Pain", 0, 4),
        ]
        self.assertEqual(
            ncerb_survey.retained("CHIA", text, mentions), {"drug": 1, "procedure": 1}
        )
        ncbi = [("Modifier", "pain", 5, 9), ("DiseaseClass", "drug", 0, 4)]
        self.assertEqual(ncerb_survey.retained("NCBI", text, ncbi), {"condition": 1})
        self.assertEqual(
            ncerb_survey.retained("BC5CDR", "NA", [("Disease", "NA", 0, 2)]), {}
        )
        result = ncerb_survey.split_result(
            "BioRED",
            "test",
            [
                ("AB", [("OrganismTaxon", "A", 0, 1)]),
                (
                    "AB",
                    [("ChemicalEntity", "A", 0, 1), ("GeneOrGeneProduct", "B", 1, 2)],
                ),
            ],
        )
        self.assertEqual(result["records"], 1)
        self.assertIn("2 retained mentions", result["reason"])
        self.assertIn("2 source documents", result["reason"])

    def test_ami_uses_only_published_layer_for_density(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ami.zip"
            words = (
                '<root xmlns:n="http://nite.sourceforge.net/">'
                '<w n:id="w0"/><gap n:id="g"/><w n:id="w1"/><w n:id="w2"/></root>'
            )
            links = (
                '<root xmlns:n="http://nite.sourceforge.net/">'
                '<named-entity><n:child href="A.words.xml#id(w0)..id(w1)"/></named-entity>'
                '<named-entity><n:child href="A.words.xml#id(w1)"/></named-entity></root>'
            )
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("words/A.words.xml", words)
                z.writestr("words/B.words.xml", words)
                z.writestr("namedEntities/A.ne.xml", links)
            with_layer, without_layer = publisher.ami(path)
            self.assertEqual((with_layer["tokens"], with_layer["tagged"]), (3, 2))
            self.assertEqual(without_layer["tokens"], 3)
            self.assertNotIn("tagged", without_layer)

    def test_review_covers_clinical_entries_and_only_allows_public_data(self):
        entries = {
            r["id"]: r for r in survey.registry_rows()[0] if r["domain"] == "clinical"
        }
        reviews = survey.clinical_access()
        self.assertEqual(set(reviews), set(entries))
        for ident, row in entries.items():
            files, reason = survey.direct_sources(row, survey.local_source_ids())
            if ident in {"THYME", "NCERB"}:
                self.assertEqual(files, publisher.SOURCES[ident])
            else:
                self.assertEqual(
                    (files, reason), ([], reviews[ident]["scope_and_access"])
                )
        # A local provision rule on the clinical notes still admits exactly the text-free release.
        self.assertEqual(
            survey.direct_sources(entries["THYME"], {"THYME"})[0],
            publisher.SOURCES["THYME"],
        )

    def test_bio_validates_labels_and_counts_last_sequence(self):
        rows = io.StringIO("a B-CROP\nb I-CROP\n\nc O\n")
        self.assertEqual(
            publisher.bio(rows, 2),
            dict(unit="nonempty BIO sequences", records=2, tokens=3, tagged=2),
        )
        with self.assertRaises(ValueError):
            publisher.bio(["a not-a-tag"], 2)

    def test_platinum_excludes_training_and_separates_body_times(self):
        entry = next(r for r in survey.registry_rows()[0] if r["id"] == "TempEval3")
        files, _ = survey.direct_sources(entry, {"TempEval3"})
        self.assertEqual(files, publisher.SOURCES["TempEval3"])
        self.assertTrue(
            all(url.startswith(publisher.TE3_PLATINUM + "/") for url in files)
        )
        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / "file.tml"
            xml.write_text(
                '<TimeML><DOCID>doc</DOCID><DCT><TIMEX3 tid="t0"/></DCT>'
                '<TEXT><EVENT eid="e1"/><TIMEX3 tid="t1"/></TEXT>'
                '<TLINK lid="l1"/></TimeML>'
            )
            observation = publisher.platinum(xml, xml.name)
            self.assertEqual(observation["records"], 1)
            self.assertIn(
                "body time expressions 1, document creation times 1",
                observation["reason"],
            )


if __name__ == "__main__":
    unittest.main()
