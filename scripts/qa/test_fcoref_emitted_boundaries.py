# SPDX-License-Identifier: MIT OR Apache-2.0
import unittest

from fcoref_emitted_boundaries import reshape, score


def place(text, surface, start=0):
    begin = text.index(surface, start)
    return begin, begin + len(surface)


class BoundaryReplacementTests(unittest.TestCase):
    def test_character_offsets_and_marked_pair(self):
        text = "Éva Wu left. Later Éva Wu the author spoke to Kim. She smiled."
        name = place(text, "Éva Wu")
        long = place(text, "Éva Wu the author")
        she = place(text, "She")
        clusters = [[name, long, she]]
        row = dict(ID="fixture", Text=text, Pronoun="She", A="Éva Wu", B="Kim")
        row.update({"Pronoun-offset": she[0], "A-offset": long[0], "B-offset": text.index("Kim"),
                    "A-coref": "TRUE", "B-coref": "FALSE"})
        result = score([row], [dict(id="fixture", producer="fcoref", clusters=clusters)])
        self.assertEqual(result["paired"]["gold_links_recovered"], 1)
        self.assertEqual(result["added_marked_spans"], 1)
        self.assertEqual(result["removed_spans_outside_marked"], 1)
        self.assertEqual(result["stock"]["all_output_spans"], result["candidate"]["all_output_spans"])

    def test_exact_larger_gold_can_be_lost(self):
        text = "Kim Yu. Dr Kim Yu said she saw Mo."
        title = place(text, "Dr Kim Yu")
        pronoun = place(text, "she")
        row = dict(ID="fixture", Text=text, Pronoun="she", A="Dr Kim Yu", B="Mo")
        row.update({"Pronoun-offset": pronoun[0], "A-offset": title[0], "B-offset": text.index("Mo"),
                    "A-coref": "TRUE", "B-coref": "FALSE"})
        pred = dict(id="fixture", producer="fcoref", clusters=[[place(text, "Kim Yu"), title, pronoun]])
        result = score([row], [pred])
        self.assertEqual(result["paired"]["gold_links_lost"], 1)
        self.assertEqual(result["removed_marked_spans"], 1)
        self.assertEqual(result["added_spans_outside_marked"], 1)

    def test_unique_target_and_global_collisions(self):
        text = "John John. John John John. dr John John left."
        name = place(text, "John John")
        repeated = place(text, "John John John")
        long = place(text, "dr John John left")
        overlapping = place(text, "dr John John")
        groups = [{name, repeated, long, overlapping}]
        self.assertEqual(reshape(text, groups)[1], 0)
        target = place(text, "John John", long[0])
        self.assertEqual(reshape(text, [{name, long}, {target}])[1], 0)
        self.assertEqual(reshape(text, [{name}, {long}])[1], 0)
        self.assertEqual(reshape(text, [{name, long}])[1], 1)

    def test_slice_start_inside_word_is_not_a_name_boundary(self):
        text = "Ann Yu. JoAnn Yu of the council."
        group = {place(text, "Ann Yu"), place(text, "Ann Yu of")}
        self.assertEqual(reshape(text, [group])[1], 0)

    def test_two_different_names_keep_original_boundary(self):
        text = "Éva Wu. Kofi Day. Éva Wu met Kofi Day."
        group = {place(text, n) for n in ("Éva Wu", "Kofi Day", "Éva Wu met Kofi Day")}
        self.assertEqual(reshape(text, [group])[1], 0)


if __name__ == "__main__":
    unittest.main()
