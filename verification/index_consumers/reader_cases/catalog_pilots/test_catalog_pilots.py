"""Fixed reader expectations from ONS change rows and MusicBrainz entity records."""

import hashlib
import json
import unittest

from reader import SAMPLE, area, music


PANEL = json.loads(SAMPLE.read_text(encoding="utf-8"))
CENDRE = "release/ad6c808f-2948-478a-aad5-3e02a9692e67"
LP = "release/0e97456e-9471-4869-bf43-10ea495f666a"
GAME = "recording/b040d89e-190f-47a0-af7f-20534101056c"
SONG = "work/dee610f0-e425-311f-be39-554747afbddf"


class CatalogPilots(unittest.TestCase):
    def test_sample(self):
        self.assertEqual(hashlib.sha256(SAMPLE.read_bytes()).hexdigest(), "46e4848813cd61a1a33f031880c8267ed60ad14e61f0c1e6f10f2112ebfe1308")
        self.assertEqual(PANEL["ons"]["2023-12"], PANEL["ons"]["2025-12"])

    def test_district_history_in_each_release(self):
        for edition in PANEL["ons"]:
            for district, name in [("E07000028", "Carlisle"), ("E07000029", "Copeland")]:
                with self.subTest(edition=edition, district=name):
                    before = area(PANEL, edition, district, "2023-03-31")
                    self.assertEqual((before["status"], before["cited_area"]["class"], before["changes"]), ("active", "E07", []))
                    after = area(PANEL, edition, district, "2023-04-01")
                    self.assertEqual((after["status"], after["cited_code"], after["cited_area"]["name"]), ("one_target", district, name))
                    self.assertEqual([(e["code"], e["area"]["class"], e["effective"], e["order"]) for e in after["changes"]],
                                     [("E06000063", "E06", "2023-04-01", "331/2022")])

    def test_ambiguous_and_incomplete(self):
        for edition in PANEL["ons"]:
            split = area(PANEL, edition, "E04006085", "2023-04-01")
            self.assertEqual(split["status"], "multiple_targets")
            self.assertEqual([(e["code"], e["area"]["name"]) for e in split["changes"]],
                             [("E04013160", "Attleborough"), ("E04013161", "Great Ellingham")])
            county = area(PANEL, edition, "E10000006", "2023-04-01")
            self.assertEqual((county["status"], county["cited_area"]["terminated"]), ("incomplete_change", "2023-03-31"))
            self.assertEqual(county["changes"], [{"code": None, "area": None, "effective": None, "order": "331/2022"}])
            self.assertEqual(area(PANEL, edition, "E10000020", "2023-04-01")["status"], "active")

    def test_source_coverage(self):
        self.assertEqual(area(PANEL, "2023-12", "E07000028", "2008-12-31")["status"], "before_code")
        self.assertEqual(area(PANEL, "2023-12", "E07000028", "2024-01-01")["status"], "outside_edition")
        self.assertEqual(area(PANEL, "2023-12", "other-local-code", "2023-04-01")["status"], "outside_panel")

    def test_release_credit(self):
        result = music(PANEL, CENDRE, "billing")
        self.assertEqual(result["text"], "Fennesz + Sakamoto")
        self.assertEqual([a["artist"] for a in result["artists"]], [
            "a6f8d42d-d3ff-4268-8d39-ee01341733bb", "a7f7df4a-77d8-4f12-8acd-5c60c93f4de8",
        ])
        self.assertEqual(result["artists"][1]["catalog_name"], "坂本龍一")
        self.assertEqual(music(PANEL, CENDRE, "role", "producer")["status"], "unasserted")
        self.assertEqual(music(PANEL, LP, "billing")["text"], "5Loops")
        self.assertEqual([a["catalog_name"] for a in music(PANEL, LP, "role", "producer")["artists"]], ["s'Poom"])
        self.assertEqual(PANEL["musicbrainz"][LP]["date"], "2010")

    def test_recording_and_composition(self):
        billed = music(PANEL, GAME, "billing")
        self.assertEqual(billed["text"], "Tinchy Stryder featuring Giggs, Professor Green, Tinie Tempah, Devlin, Example, Chipmunk")
        self.assertEqual(billed["artists"][-1]["catalog_name"], "Chip")
        guests = music(PANEL, GAME, "role", "vocal", "guest")
        self.assertEqual({a["catalog_name"] for a in guests["artists"]},
                         {"Giggs", "Professor Green", "Tinie Tempah", "Devlin", "Example", "Chip"})
        self.assertEqual(music(PANEL, GAME, "works")["works"], [SONG.split("/")[1]])
        writers = music(PANEL, SONG, "role", "writer")
        self.assertEqual({a["catalog_name"] for a in writers["artists"]}, {
            "Chip", "James Devlin", "Elliot Gleave", "Charlie Bernardo Kågell", "Eshraque Mughal",
            "Professor Green", "Tinie Tempah",
        })
        self.assertEqual(music(PANEL, GAME, "role", "writer")["status"], "unasserted")
        self.assertEqual(music(PANEL, SONG, "role", "lyricist")["status"], "unasserted")
        self.assertEqual(music(PANEL, SONG, "billing")["status"], "outside_scope")
        self.assertEqual(music(PANEL, "work/outside-panel", "role", "writer")["status"], "outside_panel")


if __name__ == "__main__":
    unittest.main()
