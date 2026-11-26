"""Synthetic panels for the GUM reader. No upstream text is required."""
from pathlib import Path
import tempfile
import unittest

if __package__:
    from . import gum
else:
    import gum


def row(n, form, pos, entity=None, lemma=None):
    return [str(n), form, lemma or form.lower(), pos, "_", "_", "0", "root", "_",
            "Entity=" + entity if entity else "_"]


def open_mark(chain, kind="person", head=1):
    return f"({chain}-{kind}-new-x-x-{head}-coref"


class GumTests(unittest.TestCase):
    document = gum.Document("test", "Example Author", "https://example.org/source")

    def test_nesting_is_chain_identity_even_for_a_proper_name(self):
        sent = gum.parse_sentence("s", "😀 Éna Team", [
            row(1, "😀", "SYM"), row(2, "Éna", "PROPN",
                open_mark(1, "organization", 2) + open_mark(2) + ")"),
            row(3, "Team", "NOUN", "1)")])
        result, = gum.document_cases(self.document, [sent])
        self.assertEqual(result["spans"], {"inner": [2, 5], "outer": [2, 10]})
        self.assertEqual(result["gold"], "different")
        self.assertEqual(result["provenance"]["sentence_ids"], ["s"])

        same = gum.parse_sentence("s", "Captain Éna", [
            row(1, "Captain", "NOUN", open_mark(2, head=2)),
            row(2, "Éna", "PROPN", open_mark(2) + ")2)")])
        self.assertEqual([r["gold"] for r in gum.document_cases(self.document, [same])], ["same"])

    def test_three_sentence_pronoun_uses_nearest_valid_decoy(self):
        named = lambda chain: open_mark(chain) + ")"
        sentences = [
            gum.parse_sentence("a", "🌙 Éna", [row(1, "🌙", "SYM"), row(2, "Éna", "PROPN", named(1))]),
            gum.parse_sentence("b", "Io Nix", [row(1, "Io", "PROPN", named(2)), row(2, "Nix", "PROPN", named(3))]),
            gum.parse_sentence("c", "She sat", [row(1, "She", "PRON", named(1)), row(2, "sat", "VERB")]),
        ]
        # Exclude the leading sentence from the selected window. Its non-BMP character
        # occupies one Unicode scalar in the reconstructed source document.
        lead = gum.parse_sentence("lead", "😀.", [row(1, "😀", "SYM"), row(2, ".", "PUNCT")])
        case, = gum.document_cases(self.document, [lead, *sentences])
        self.assertEqual(case["provenance"], {
            "document": "test", "author": "Example Author", "sourceURL": "https://example.org/source",
            "sentence_ids": ["a", "b", "c"], "window_start": 3,
        })
        marks = {k: case["text"][a:b] for k, (a, b) in case["spans"].items()}
        self.assertEqual(marks["target"], "She")
        self.assertEqual(marks[case["gold"]], "Éna")
        self.assertEqual({marks["A"], marks["B"]}, {"Éna", "Nix"})
        self.assertEqual(len(list(gum.document_cases(self.document, sentences[1:]))), 0)

    def test_headers_follow_document_boundaries(self):
        blocks = []
        for number in (1, 2):
            blocks.append("\n".join([
                f"# newdoc id = d{number}", f"# meta::author = A & B {number}",
                f"# meta::sourceURL = https://example.org/query?a={number}",
                f"# sent_id = s{number}", "# text = Word", "\t".join(row(1, "Word", "NOUN")),
            ]))
        docs = list(gum.documents("\n\n".join(blocks)))
        self.assertEqual([(d.name, d.author, d.source_url, s[0].sid) for d, s in docs], [
            ("d1", "A & B 1", "https://example.org/query?a=1", "s1"),
            ("d2", "A & B 2", "https://example.org/query?a=2", "s2"),
        ])

    def test_opaque_multiword_token_preserves_only_attested_edges(self):
        sentence = gum.parse_sentence("s", "Zá", [
            row("1-2", "Zá", "_"),
            row(1, "Zo", "PROPN", open_mark(1) + open_mark(2) + ")"),
            row(2, "a", "NOUN", "1)")])
        self.assertEqual([(m.chain, m.start, m.end) for m in sentence.mentions], [(1, 0, 2)])

    def test_source_alignment_and_pin_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "align"):
            gum.parse_sentence("s", "Éna", [row(1, "Ena", "PROPN")])
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / gum.SOURCE_FILE
            source.write_bytes(b"# synthetic split\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                list(gum.cases(source))


if __name__ == "__main__":
    unittest.main()
