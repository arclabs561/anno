import unittest
from pathlib import Path

from check_nasa_dates import catalog_input, check_spans


class CatalogDateContract(unittest.TestCase):
    def test_full_date_with_year_is_required(self):
        fixture = Path(__file__).resolve().parents[2] / "crates/anno-cli/tests/fixtures/public_sources/nasa-eclipse.txt"
        row = fixture.read_text().rstrip("\n")
        html = '<a href="../5MCLEmap/2001-2100/LE2001-01-09T.gif">09651</a>' + row[5:]
        text, gold = catalog_input(html)
        self.assertEqual(text, row)
        self.assertEqual(gold, [(7, 18, "2001 Jan 09")])
        date = {"type": "DATE", "text": "2001 Jan 09", "start": 7, "end": 18}
        check_spans({"entities": [date]}, gold)
        for predictions in ([], [dict(date, text="Jan 09", start=12)],
                            [dict(date, start=8)], [date, date]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                check_spans({"entities": predictions}, gold)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            catalog_input(html.replace("LE2001-01-09T", "LE2001-01-10T"))


if __name__ == "__main__":
    unittest.main()
