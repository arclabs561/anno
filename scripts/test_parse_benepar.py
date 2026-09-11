"""Adapter contract tests using real NLTK trees and spaCy token offsets."""

import unittest
from unittest.mock import patch

import spacy
from nltk.tree import Tree
from parse_benepar import encode_sentence, parse_text


class AdapterTests(unittest.TestCase):
    def test_download_conflict_rejected_even_for_empty_input(self):
        with (
            patch.dict("os.environ", {"ANNO_NO_DOWNLOADS": "1"}),
            self.assertRaisesRegex(ValueError, "conflicts"),
        ):
            parse_text("  ", "missing-model", "en", True)

    def test_source_offsets_survive_model_normalization(self):
        text = "  Zoë\n  (猫)"
        tokens = [t for t in spacy.blank("en")(text) if not t.is_space]
        tree = Tree.fromstring(
            "(TOP (S (NP (NNP Zoe)) (-LRB- -LRB-) (NN cat) (-RRB- -RRB-)))"
        )
        sentence = encode_sentence(tree, tokens)
        self.assertEqual(
            [t["text"] for t in sentence["tokens"]], ["Zoë", "(", "猫", ")"]
        )
        self.assertEqual(
            [(t["start"], t["end"]) for t in sentence["tokens"]],
            [(2, 5), (8, 9), (9, 10), (10, 11)],
        )
        for token in sentence["tokens"]:
            self.assertEqual(text[token["start"] : token["end"]], token["text"])
        self.assertEqual(sentence["constituents"][0]["labels"], ["TOP", "S"])
        self.assertEqual(sentence["constituents"][2]["labels"], [])

    def test_single_token_unary_chain_keeps_pos_separate(self):
        sentence = encode_sentence(
            Tree.fromstring("(TOP (S (NP (NN robot))))"),
            list(spacy.blank("en")("robot")),
        )
        self.assertEqual(
            sentence["constituents"],
            [{"start": 0, "end": 1, "labels": ["TOP", "S", "NP"]}],
        )
        self.assertEqual(sentence["tokens"][0]["tag"], "NN")

    def test_leaf_count_mismatch_is_an_error(self):
        tokens = list(spacy.blank("en")("robots work"))
        with self.assertRaisesRegex(ValueError, "leaf count"):
            encode_sentence(Tree.fromstring("(TOP (NN robots))"), tokens)
        with self.assertRaisesRegex(ValueError, "more leaves"):
            encode_sentence(Tree.fromstring("(TOP (NN robots) (VB work))"), tokens[:1])

    def test_empty_source_needs_no_model(self):
        self.assertEqual(
            parse_text("\n  ", "missing-model", "en", False),
            {"text": "\n  ", "sentences": []},
        )


if __name__ == "__main__":
    unittest.main()
