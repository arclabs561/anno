"""Verify distinctions between archive members, missing gold and projected fields."""

import hashlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Public split replay requires pyarrow")
class ReleaseSurveyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import survey_hf_releases as releases
        from hf_parquet_projection import RemoteFile

        cls.pa = pa
        cls.pq = pq
        cls.hub = releases
        cls.remote = RemoteFile

    def test_blind_rows_do_not_contribute_a_zero_numerator(self):
        blind = b"# id one\n# _ _\nword _ _\n\n# id two\nword _ _\n"
        self.assertEqual(self.hub.conll_counts(io.BytesIO(blind)), (2, 3, None))
        tagged = b"# id two\n# _ _ O\nword _ _ B-X\n"
        self.assertEqual(self.hub.conll_counts(io.BytesIO(tagged)), (1, 2, 1))
        with self.assertRaisesRegex(ValueError, "Partially annotated"):
            self.hub.conll_counts(io.BytesIO(blind + tagged))

    def test_complete_files_and_archive_members_use_their_own_digests(self):
        release = self.hub.BY_ID_REPO["ACORD", "theatticusproject/acord"]
        records = {"queries": b'{"_id":"a","text":"a"}\n', "corpus": b'{"_id":"b","text":"b"}\n'}
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            for split, payload in records.items():
                archive.writestr(self.hub.archive_member(release, "published", split), payload)
        blob = data.getvalue()
        entry = {
            "rfilename": "ACORD Dataset & ReadMe.zip",
            "size": len(blob),
            "lfs": {"sha256": hashlib.sha256(blob).hexdigest()},
        }
        revision = "a" * 40
        url = self.hub.file_url(release.repo, revision, entry["rfilename"])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ANNO_HF_SURVEY_CACHE=tmp):
            path = Path(tmp) / hashlib.sha256(url.encode()).hexdigest()
            path.write_bytes(blob)
            for split, payload in records.items():
                [row] = self.hub.measure((release, revision, "", "published", split, [entry]))
                self.assertEqual(row["rows"], 1)
                self.assertEqual(row["payload_sha256"], hashlib.sha256(payload).hexdigest())
                self.assertIn("#member=", row["payload_url"])
            path.write_bytes(blob[:-1])
            with self.assertRaisesRegex(ValueError, "differs from pinned"):
                self.hub.measure((release, revision, "", "published", "queries", [entry]))

    def test_projected_byte_ranges_reject_wrong_files_or_partial_reads(self):
        def bad_response(request, check):
            self.assertEqual(request.get_header("Range"), "bytes=3-5")
            response = io.BytesIO(b"xyz")
            response.status = 206
            response.headers = {"Content-Range": "bytes 3-5/999"}
            return check(response)

        stream = self.remote("https://example.invalid/a.parquet", 10, bad_response)
        stream.seek(3)
        with self.assertRaisesRegex(OSError, "requested byte range"):
            stream.read(3)

    def test_native_classification_validates_alignment_and_blind_labels(self):
        release = self.hub.BY_ID_REPO["LinCE", "lince-benchmark/lince"]

        def parquet(words, tags):
            file = io.BytesIO()
            self.pq.write_table(self.pa.table({"words": [words], "ner": [tags]}), file)
            return self.pq.ParquetFile(file)

        self.assertEqual(
            self.hub.bio_counts(parquet(["a", "b"], ["", ""]), release, "words", "ner", blind=True),
            (1, 2, None),
        )
        with self.assertRaisesRegex(ValueError, "nonempty tag"):
            self.hub.bio_counts(parquet(["a"], ["O"]), release, "words", "ner", blind=True)
        with self.assertRaisesRegex(ValueError, "differ"):
            self.hub.bio_counts(parquet(["a", "b"], ["O"]), release, "words", "ner")
        with self.assertRaisesRegex(ValueError, "Missing observation column"):
            self.hub.text_rows(parquet(["a"], ["O"]), "nonexistent")

    def test_survey_excludes_partial_conversion_and_uses_named_configs(self):
        release = self.hub.BY_ID_REPO["CALCS2018", "lince-benchmark/lince"]
        files = [
            "ner_spaeng/train/0000.parquet",
            "ner_spaeng/partial-test/0000.parquet",
            "ner_hineng/train/0000.parquet",
            "ner_spaeng/train/notes.txt",
        ]
        groups = self.hub.partitions(release, {}, {"siblings": [{"rfilename": f} for f in files]})
        self.assertEqual(list(groups), [("ner_spaeng", "train")])
        self.assertEqual(groups["ner_spaeng", "train"], [{"rfilename": files[0]}])
