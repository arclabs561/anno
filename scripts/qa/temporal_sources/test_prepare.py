"""Synthetic offset checks for the two offline temporal source adapters."""
import unittest
from xml.etree import ElementTree

from prepare import case_gold, news_row


class SourceOffsets(unittest.TestCase):
    def test_inclusive_utf16_offsets_become_half_open_scalar_spans(self):
        item = {"Text": "may", "Start": 3, "End": 5, "TypeName": "datetimeV2.date"}
        self.assertEqual(case_gold("🌙 May", [item]), [
            {"text": "May", "start_char": 2, "end_char": 5, "type": "date"},
        ])

    def test_virtual_timex_does_not_turn_annotated_sentence_into_a_positive(self):
        xml = ElementTree.fromstring("""
            <Document>
              <token t_id="1" sentence="2">May</token>
              <token t_id="2" sentence="2">I?</token>
              <Markables>
                <EVENT_MENTION><token_anchor t_id="2"/></EVENT_MENTION>
                <TIMEX3 type="DATE" m_id="9"/>
              </Markables>
            </Document>
        """)
        row = news_row(xml, "42", "test", "2")
        self.assertEqual((row["text"], row["time_expressions"]), ("May I?", []))


if __name__ == "__main__":
    unittest.main()
