"""Full-document projection and alignment on authored CoNLL-U examples."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if __package__:
    from . import system_output as scorer
else:
    import system_output as scorer


def sentence(sid, text, forms, annotations, *, work=None, source_spacing=True):
    lines = []
    if work is not None:
        lines.extend([f"# newdoc id = {work}", "# global.Entity = eid-etype-head-other"])
    lines.extend([f"# sent_id = {sid}", f"# text = {text}"])
    for i, (form, annotation) in enumerate(zip(forms, annotations, strict=True), 1):
        misc = []
        if source_spacing and i < len(forms) and forms[i] == ".":
            misc.append("SpaceAfter=No")
        if annotation:
            misc.append("Entity=" + annotation)
        lines.append("\t".join([str(i), form] + ["_"] * 7 + ["|".join(misc) or "_"]))
    return "\n".join(lines)


def sample(first, second, *, spacing=True):
    return "\n\n".join((
        sentence("s1", "Zoë Li waited.", ["Zoë", "Li", "waited", "."], first,
                 work="work", source_spacing=spacing),
        sentence("s2", "She thanked Ari.", ["She", "thanked", "Ari", "."], second, source_spacing=spacing),
        sentence("s3", "Done.", ["Done", "."], ["", ""], work="other", source_spacing=spacing),
    )) + "\n"


class DocumentPredictionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name)
        self.source = path / "reference.conllu"
        self.pred = path / "predicted.conllu"
        self.source.write_text(sample([""] * 4, [""] * 4).replace("# sent_id = s1", "# date = 2000\n# sent_id = s1"))
        self.component = scorer.corefud.Component("Example", "en", "dev.conllu",
                                                  hashlib.sha256(self.source.read_bytes()).hexdigest())
        context = patch.dict(scorer.COMPONENTS, {"example": self.component})
        context.start()
        self.addCleanup(context.stop)
        text = "Zoë Li waited.\nShe thanked Ari."
        lineage = {"work_id": "work", "sentence_ids": ["s1", "s2"], "window_scalar_span": [0, len(text)],
                   "source_sha256": self.component.sha256,
                   "source_member": "CorefUD-1.4-public/data/CorefUD_Example/dev.conllu"}
        common = {"text": text, "provenance": lineage, "group": "work"}
        self.choice = dict(common, id="candidate", kind="choice", gold="A",
                           spans={"target": [15, 18], "A": [0, 6], "B": [27, 30]})
        self.nested = dict(common, id="nested", kind="nested", gold="different",
                           spans={"inner": [4, 6], "outer": [0, 6]})

    def score(self, data, selected=None, include_cases=True):
        self.pred.write_text(data)
        return scorer.score_document_predictions([self.choice, self.nested] if selected is None else selected,
                                                 self.source, "example", self.pred, include_cases=include_cases)

    def test_singleton_publication_and_missing_distractor(self):
        hypothesis = sample(["(d1.e1--1", "d1.e1)(0--1)", "", ""], ["(d1.e1--1)", "", "", ""], spacing=False)
        hypothesis = hypothesis.replace("# sent_id = s1", "# generator = synthetic\n# sent_id = s1")
        report = self.score(hypothesis)
        self.assertEqual(report["scope"], "full_document")
        self.assertEqual(report["coverage"], {
            "documents": 2, "sentences": 3, "tokens": 10, "output_clusters": 2, "output_singleton_clusters": 1,
            "distinct_output_spans": 3, "multiply_owned_spans": 0, "omitted_no_space_flags": 3})
        choice, nested = report["cases"]
        self.assertEqual(choice["observed"], "A")
        self.assertTrue(choice["both_required_marks_seen"])
        self.assertEqual(choice["missing"], ["B"])
        self.assertEqual(nested["observed"], "different")
        self.assertEqual(report["counts"]["choice"]["all_marks_exact"], 0)
        self.assertNotIn("cases", self.score(hypothesis, include_cases=False))
        self.assertNotIn("proposed", choice)

    def test_missing_nested_is_unresolved_and_separate_singletons_do_not_link(self):
        case = self.score(sample(["(e1--1", "e1)", "", ""], ["(e2--1)", "", "", ""]))
        self.assertEqual([r["observed"] for r in case["cases"]], ["neither", "unresolved"])
        self.assertEqual(case["coverage"]["output_singleton_clusters"], 2)

    def test_coextensive_owners_preserve_invariant_and_ambiguous_decisions(self):
        hyp = sample(["(e1-person-1", "e1)(e1--1)(c2--1)", "", ""], ["(e1--1)(c2--1)", "", "", ""])
        hyp = hyp.replace("eid-etype-head-other", "eid-etype-head")
        case = self.score(hyp)
        self.assertEqual(case["coverage"]["multiply_owned_spans"], 2)
        self.assertEqual([r["observed"] for r in case["cases"]], ["ambiguous", "ambiguous"])
        no_match = dict(self.choice, spans={"target": [15, 18], "A": [7, 13], "B": [27, 30]})
        invariant = self.score(hyp, [no_match])["cases"][0]
        self.assertEqual((invariant["observed"], invariant["ambiguous_memberships"]), ("neither", ["target"]))

    def test_declared_other_field_is_optional_and_does_not_change_identity(self):
        plain = sample(["(e1-person-1", "e1)(c2--1)", "", ""], ["(e1--1)", "", "", ""])
        expected = self.score(plain)["cases"]
        for other in ("", "wd:Q42"):
            encoded = plain.replace("(e1-person-1", "(e1-person-1-" + other)
            encoded = encoded.replace("(c2--1)", "(c2--1-" + other + ")")
            self.assertEqual(self.score(encoded)["cases"], expected)
            with self.assertRaisesRegex(ValueError, "bracket"):
                self.score(encoded.replace("eid-etype-head-other", "eid-etype-head"))
        with self.assertRaisesRegex(ValueError, "bracket"):
            self.score(plain.replace("(e1-person-1", "(e1-person-1-x-undeclared"))

    def test_invalid_annotations_fail_closed(self):
        for first in [
            ["(e1--1)(e1--1)", "", "", ""], ["e1)", "", "", ""], ["(e1--1", "", "", ""],
            ["(e1--2)", "", "", ""], ["(e1[1/2]--1)", "", "", ""], ["(e1--1)|Entity=(e2--1)", "", "", ""],
        ]:
            with self.subTest(first=first), self.assertRaises(ValueError):
                self.score(sample(first, [""] * 4))

    def test_all_source_sentences_are_validated_even_for_partial_case_selection(self):
        base = sample([""] * 4, [""] * 4)
        for bad in [
            base.split("\n\n# newdoc id = other")[0], base.replace("s3", "s2"), base.replace("id = other", "id = work"),
            base.replace("# text = Done.", "# text = Changed."), base.replace("\tDone\t", "\tchanged\t"),
            base.replace("# sent_id = s2", "# sent_id = missing"), base.replace("1\tZoë\t", "1-2\tZoë\t"),
            base.replace("1\tZoë\t_\t_\t_\t_\t_\t_\t_\t_", "1\tZoë\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No"),
            base.replace("eid-etype-head-other", "eid-custom"), base.replace("# text = Done.", "# text = Done.\n# text = Done."),
        ]:
            with self.subTest(suffix=bad[-40:]), self.assertRaises(ValueError):
                self.score(bad, [self.choice])

    def test_source_receipt_and_case_identity(self):
        base = sample([""] * 4, [""] * 4)
        for altered in [
            self.choice | {"text": self.choice["text"] + " "},
            self.choice | {"provenance": self.choice["provenance"] | {"window_scalar_span": [1, 32]}},
            self.choice | {"provenance": self.choice["provenance"] | {"sentence_ids": ["s2", "s1"]}},
            self.choice | {"provenance": self.choice["provenance"] | {"source_member": "different"}},
        ]:
            with self.subTest(), self.assertRaises(ValueError):
                self.score(base, [altered])
        with self.assertRaises(ValueError):
            self.score(base, [self.choice, self.choice])
        self.source.write_text("mismatching reference")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.score(base)

    def test_missing_or_non_utf8_prediction_is_rejected(self):
        for binary in (None, b"\xff"):
            if binary is not None:
                self.pred.write_bytes(binary)
            with self.subTest(binary=binary), self.assertRaisesRegex(ValueError, "UTF-8"):
                scorer.score_document_predictions([self.choice], self.source, "example", self.pred)


if __name__ == "__main__":
    unittest.main()
