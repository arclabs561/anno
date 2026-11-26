"""Regression checks for CrossRE predicate inventories and role-conditioned coverage."""

from pathlib import Path
import sys
import unittest


class CrossREPromptScopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        try:
            import survey_crossre_prompts
        finally:
            sys.path.pop(0)
        cls.survey = survey_crossre_prompts

    def test_sentence_types_and_annotation_occurrences_use_separate_units(self):
        # The training edges are (author, book) and (artist, song). The
        # marginal role rule permits (artist, book) in the test sentence.
        artist_book = ("artist", "created", "book")
        examples = [
            [],
            [artist_book, artist_book],
            [("book", "created", "artist"), ("novel", "inspired", "book")],
        ]
        counts = self.survey.split_counts(
            {"artist", "book", "novel"},
            examples,
            {"artist", "author", "book", "song"},
            {"created": {"author", "artist"}},
            {"created": {"book", "song"}},
        )
        self.assertEqual(
            (
                counts["no_predicate_sentences"],
                counts["one_predicate_type_sentences"],
                counts["multiple_predicate_type_sentences"],
            ),
            (1, 1, 1),
        )
        self.assertEqual(counts["gold_relations"], 4)
        self.assertEqual(counts["relations_with_unseen_predicate"], 1)
        self.assertEqual(counts["relations_with_unseen_endpoint_type"], 1)
        self.assertEqual(counts["relations_outside_train_predicate_roles"], 2)

    def test_every_endpoint_resolves_to_one_annotated_type(self):
        record = {
            "ner": [[0, 0, "person"], [2, 3, "place"]],
            "relations": [[0, 0, 2, 3, "visits"]],
        }
        self.assertEqual(
            self.survey.typed_examples([record]),
            ({"person", "place"}, [[("person", "visits", "place")]]),
        )
        for ner in (record["ner"][:1], [*record["ner"], [2, 3, "name"]]):
            with self.subTest(ner=ner), self.assertRaises(ValueError):
                self.survey.typed_examples([dict(record, ner=ner)])


if __name__ == "__main__":
    unittest.main()
