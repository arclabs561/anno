"""Exercise the Phrase Detectives adapter with authored markables and source text."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET
from zipfile import ZipFile

if __package__:
    from . import phrase_detectives as pd
else:
    import phrase_detectives as pd


def sample_document():
    # Four sentences exercise the window boundary and Unicode offsets. The two names in
    # the second sentence are aliases for one entity and can both be offered in later cases.
    sentences = [
        [("Outside", "NNP", "1", "dn")],
        [("Éva", "NNP", "2", "dn"), ("Sun", "NNP", "2", "do")],
        [("She", "PRP", "2", "do"), ("box", "NN", "3", "dn"), ("map", "NN", "4", "dn")],
        [("her", "PRP$", "2", "do"), ("there", "EX", "7-Pseudo", "non_referring"),
         ("they", "PRP", "8", "do"), ("you", "PRP", "9", "dn")],
    ]
    root = ET.Element("gnomedoc")
    file_desc = ET.SubElement(ET.SubElement(root, "header"), "fileDesc")
    statement = ET.SubElement(file_desc, "publicationStmt")
    ET.SubElement(statement, "availability").text = "Synthetic fixture notice."
    body = ET.SubElement(root, "body")
    ann = ET.SubElement(root, "annotations")
    lines, number = [], 0
    for i, sentence in enumerate(sentences):
        lines.extend([f"# sent_id = s{i}", "# text = " + " ".join(t[0] for t in sentence)])
        for idx, (word, pos, entity, kind) in enumerate(sentence, 1):
            number += 1
            ref = f"ne{number}"
            mark = ET.SubElement(body, "ne", id=ref)
            ET.SubElement(mark, "W", Lpos=pos).text = word
            record = ET.SubElement(ann, "PDante", id=ref)
            ET.SubElement(record, "anchor", mode="e", favoured="y", type=pd.TYPES[kind])
            annotation = f"(EntityID={entity}|MarkableID=markable_{number}|SemType={kind})"
            lines.append(" ".join([str(idx), word] + ["_"] * 8 + [annotation] + ["_"] * 4))
        lines.append("")
    return root, "\n".join(lines).encode()


def surface(case, span):
    a, b = span
    return case["text"][a:b]


class PhraseDetectivesTests(unittest.TestCase):
    def test_public_entrypoint_does_not_accept_unpinned_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.zip"
            path.write_bytes(b"not a pinned ZIP")
            with self.assertRaisesRegex(ValueError, "digest"):
                list(pd.cases(path))

    def test_holdout_entrypoint_windows_and_choice_meanings(self):
        root, conllu = sample_document()
        # Include the expected 45-member release structure without downloading the corpus.
        fixture_names = [f"synthetic{i}" for i in range(45)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.zip"
            with ZipFile(path, "w") as archive:
                for name in fixture_names:
                    prefix = pd.GOLD_ROOT + "wikipedia/"
                    archive.writestr(prefix + "masxml/" + name + "-game.xml", ET.tostring(root))
                    archive.writestr(prefix + "CONLLUA/" + name + ".CONLLUA", conllu)
            with patch.object(pd, "ARCHIVE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()):
                rows = list(pd.cases(path))
        groups = {r["group"] for r in rows}
        self.assertGreater(len(groups), 0)
        self.assertLess(len(groups), len(fixture_names))
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        examples = {surface(r, r["spans"]["target"]): r for r in rows if r["group"] == min(groups)}
        expected = {"She": "both", "her": "neither", "there": "neither",
                    "they": "unscored", "you": "unscored"}
        self.assertEqual({w: r["gold"] for w, r in examples.items()}, expected)
        omitted = examples["her"]
        self.assertEqual(omitted["tags"], ["known_prior_outside_candidates"])
        self.assertNotIn("Outside", omitted["text"])
        self.assertEqual([surface(omitted, s) for s in omitted["aliases"]], ["Éva", "Sun", "She"])
        self.assertEqual([surface(omitted, omitted["spans"][k]) for k in ("A", "B")], ["map", "box"])
        self.assertEqual(examples["there"]["tags"], ["explicit_nonreferential"])
        self.assertEqual(examples["there"]["aliases"], [])
        lineage = omitted["provenance"]
        self.assertEqual(lineage["rights_notice"], "Synthetic fixture notice.")
        self.assertEqual(root.find(lineage["markable_xpath"]).findtext("W"), "her")
        self.assertEqual(root.find(lineage["publication_xpath"]).findtext("availability"),
                         lineage["rights_notice"])
        stem = min(groups).split("/")[1]
        self.assertEqual(lineage["xml_member"], pd.GOLD_ROOT + f"wikipedia/masxml/{stem}-game.xml")
        self.assertEqual(lineage["text_member"], pd.GOLD_ROOT + f"wikipedia/CONLLUA/{stem}.CONLLUA")
        self.assertEqual(examples["they"]["tags"], ["no_earlier_chain_mention"])
        self.assertEqual(examples["you"]["tags"], ["discourse_new"])

    def test_mismatched_offsets_and_surface_are_rejected(self):
        root, conllu = sample_document()
        next(root.iter("W")).text = "elsewhere"
        with self.assertRaisesRegex(ValueError, "token sequences differ"):
            list(pd._document_cases(root, conllu, "g", "d", "doc.xml", "doc.CONLLUA"))

    def test_disagreement_is_separate_from_the_release_label(self):
        record = ET.Element("PDante")
        ET.SubElement(record, "anchor", mode="e", favoured="y", type="NR")
        for i, kind in enumerate(("NR", "NR", "DO", "DO")):
            ET.SubElement(record, "anchor", mode="a", agree="y", type=kind, user_id=str(i))
        favoured, disagrees = pd._judgments(record)
        self.assertEqual(favoured, {"NR"})
        self.assertTrue(disagrees)
        ET.SubElement(record, "anchor", mode="a", agree="y", type="NR", user_id="2")
        ET.SubElement(record, "anchor", mode="v", agree="y", type="DO", user_id="4")
        self.assertFalse(pd._judgments(record)[1])


if __name__ == "__main__":
    unittest.main()
