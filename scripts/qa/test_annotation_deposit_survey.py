import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import annotation_deposit_survey as deposits
    import dataset_file_survey as survey
finally:
    sys.path.pop(0)


class AnnotationDepositTest(unittest.TestCase):
    def test_publisher_records_and_archiver_metadata_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gold.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for article in range(50):
                    records = [{"labels": ["fixture"]}] * (1 + article % 2)
                    archive.writestr(f"gold/{article}.json", json.dumps(records))
                archive.writestr("__MACOSX/gold/._0.json", b"metadata")
            self.assertEqual(deposits.journal_gold(path)["records"], 75)
            with self.assertRaisesRegex(ValueError, "pinned release"):
                deposits.measure("ChemDataExtractor", deposits.CHEM_GOLD, path)

    def test_only_the_pinned_http_archive_is_downloaded(self):
        import io

        with tempfile.TemporaryDirectory() as temp, patch.object(
            survey, "urlopen"
        ) as fetch:
            cache = Path(temp)
            survey.cached_file("http://example.org/data.zip", cache)
            fetch.assert_not_called()
            body = io.BytesIO(b"changed archive")
            body.status, body.headers = 200, {}
            fetch.return_value = body
            with self.assertRaisesRegex(ValueError, "pinned journal gold"):
                survey.cached_file(deposits.CHEM_GOLD, cache)
            self.assertEqual(list(cache.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
