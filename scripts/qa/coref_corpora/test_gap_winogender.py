"""Synthetic protocol and span regressions. Publisher data is not needed."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

if __package__:
    from . import gap, winogender
else:
    import gap
    import winogender


class GapTests(unittest.TestCase):
    def test_choice_and_spans_follow_display_order(self):
        text = "😀 Ona met Éli and she waved."
        row = {
            "ID": "authored", "URL": "http://en.wikipedia.org/wiki/An_%C3%A9pisode_(1888)", "Text": text,
            "Pronoun": "she", "A": "Ona", "B": "Éli", "A-coref": "FALSE", "B-coref": "TRUE",
        }
        for label in ("Pronoun", "A", "B"):
            row[label + "-offset"] = str(text.index(row[label]))
        case = gap._case(row, "ordinary")
        start, end = case["spans"][case["gold"]]
        self.assertEqual(text[start:end], "Éli")
        self.assertEqual(case["provenance"], {
            "source_row_id": "authored", "original_article_url": row["URL"],
            "article_history_url": "https://en.wikipedia.org/w/index.php?title=An_%C3%A9pisode_%281888%29&action=history",
        })
        self.assertEqual(case["spans"]["target"], [18, 21])
        self.assertEqual(gap._choice(row, "B", "A"), "A")
        self.assertEqual(gap._choice({**row, "B-coref": "FALSE"}, "A", "B"), "neither")
        with self.assertRaises(ValueError):
            gap._case({**row, "Pronoun-offset": "17"}, "ordinary")
        with self.assertRaises(ValueError):
            gap._case({**row, "B-coref": ""}, "ordinary")

    def test_ordinary_article_draw_cannot_depend_on_gold(self):
        # These two source URL spellings resolve to the same normalized article.
        urls = ["https://example.org/wiki/Ona%20Vale", "https://example.org/wiki/Ona_Vale",
                "https://example.org/wiki/Eli", "https://example.org/wiki/Nix"]
        rows = [{"ID": str(i), "URL": url} for i, url in enumerate(urls)]
        selected = gap._draw(rows, "ordinary", 4)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({gap._article(r) for r in selected}), 3)
        changed = [{**r, "A-coref": "TRUE", "B-coref": "FALSE"} for r in reversed(rows)]
        self.assertEqual([r["ID"] for r in selected],
                         [r["ID"] for r in gap._draw(changed, "ordinary", 4)])

    def test_prior_decision_holdout_is_also_excluded(self):
        # The decision audit uses the original ranking's next distinct articles.
        # Its selections must join the independently salted replication/proposal panels.
        rows = [{"ID": str(i), "URL": f"https://example.org/{i//2}", "Text": f"case {i}"}
                for i in range(1500)]
        ranked = sorted(rows, key=lambda r: gap._rank("anno-gap-v1", r["ID"]))
        excluded = {r["URL"] for r in ranked[:128]}
        holdout = []
        for row in ranked[128:]:
            if row["URL"] not in excluded:
                excluded.add(row["URL"])
                holdout.append(row["ID"])
                if len(holdout) == 128:
                    break
        self.assertLessEqual(set(holdout), {r["ID"] for r in gap._historical_rows(rows)})


class WinogenderTests(unittest.TestCase):
    def test_role_spans_across_gender_and_generic_description(self):
        template = "The $OCCUPATION met the $PARTICIPANT because $NOM_PRONOUN was early."
        for role, form, expected, head, agreed in [
            ("poet", "female", "the poet", "poet", "she was"),
            ("someone", "neutral", "someone", "someone", "they were"),
        ]:
            with self.subTest(role=role, form=form):
                text, spans, heads = winogender._render(template, "artiste", role, form)
                self.assertEqual(text[slice(*spans["A"])], "The artiste")
                self.assertEqual(text[slice(*heads["A"])], "artiste")
                self.assertEqual(text[slice(*spans["B"])], expected)
                self.assertEqual(text[slice(*heads["B"])], head)
                self.assertEqual(text[slice(*spans["target"])], agreed.split()[0])
                self.assertIn(agreed, text)

    def test_sentence_initial_generic_and_invalid_slot(self):
        text, spans, _ = winogender._render(
            "A $PARTICIPANT saw a $OCCUPATION beside $POSS_PRONOUN stall.",
            "painter", "someone", "female",
        )
        self.assertEqual(text[slice(*spans["B"])], "Someone")
        self.assertEqual(text[slice(*spans["target"])], "her")
        with self.assertRaises(ValueError):
            winogender._render("$PARTICIPANT waved.", "poet", "someone", "female")

    def test_modified_sources_rejected_before_producing_cases(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder/"all_sentences.tsv"
            path.write_text("authored\n")
            with self.assertRaises(ValueError):
                list(gap.cases(path))
            with self.assertRaises(ValueError):
                list(winogender.cases(folder))


if __name__ == "__main__":
    unittest.main()
