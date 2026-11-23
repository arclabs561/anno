import unittest
from pathlib import Path

from check_nasa_dates import catalog_input, check_rendered_catalog, check_spans


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


    def test_web_dates_must_match_the_right_occurrence_and_full_cell(self):
        date = "2001 Jan 09"
        row = f"09651  {date}  20:21:40"
        table_gold = [(7, 18, date)]
        text = f"🦊 Title: {date}\n[09651][19]  {date}  20:21:40"
        first, cell = text.index(date), text.rindex(date)

        def signal(start, end):
            return {"label": "DATE", "surface": text[start:end],
                    "location": {"Text": {"start": start, "end": end}}}

        good = signal(cell, cell + len(date))
        heading = signal(first, first + len(date))
        check_rendered_catalog({"text": text, "signals": [heading, good]}, row, table_gold)
        for errors in ([heading], [signal(cell + 5, cell + len(date))],
                       [good, good], [good, signal(cell, cell + len(date) + 1)]):
            with self.subTest(errors=errors), self.assertRaises(ValueError):
                check_rendered_catalog({"text": text, "signals": errors}, row, table_gold)
        with self.assertRaisesRegex(ValueError, "one rendered calendar cell"):
            check_rendered_catalog({"text": text.replace("09651", "09650"), "signals": [good]}, row, table_gold)


if __name__ == "__main__":
    unittest.main()
