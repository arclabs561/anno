"""Unit tests; set ANNO_COREFUD_DEV_ZIP to run both original labeled splits."""
from collections import Counter
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

if __package__:
    from . import corefud
else:
    import corefud


def token(i, word, pos, misc, feats="_"):
    return "\t".join([str(i), word, "_", pos, "_", feats, "_", "_", "_", misc])


class CorefUDTest(unittest.TestCase):
    def test_nested_unicode_boundaries_and_heads(self):
        # Synthetic text with a multibyte first token and punctuation attached.
        rows = [
            token(1, "Éliane", "PROPN", "Entity=(e1-person-2(e2-person-1)"),
            token(2, "Dupont", "PROPN", "Entity=e1)|SpaceAfter=No"),
            token(3, ".", "PUNCT", "_"),
        ]
        mentions = corefud._sentence(rows, "Éliane Dupont.", 0, 0)
        self.assertEqual([(m.start, m.end, m.entity) for m in mentions],
                         [(0, 6, "e2"), (0, 13, "e1")])
        with self.assertRaisesRegex(ValueError, "reconstruction"):
            corefud._sentence(rows, "Éliane Dupont .", 0, 0)
        with self.assertRaisesRegex(ValueError, "closing bracket"):
            corefud._sentence([token(1, "A", "PROPN", "Entity=e3)")], "A", 0, 0)

    def test_selects_disjoint_mentions_of_distinct_people(self):
        def mention(start, end, entity):
            return corefud.Mention(0, start, end, entity, "person", "NOUN", False)

        target = mention(20, 21, "e1")
        enclosing = mention(1, 10, "e1")
        nested = mention(3, 5, "e2")
        earlier = mention(0, 1, "e1")
        self.assertEqual(corefud._choice(target, [enclosing, nested, earlier]), (earlier, nested))
        self.assertIsNone(corefud._choice(target, [enclosing, nested]))

    def test_rejects_wrong_source(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)
            split = path / corefud.DUTCH.filename
            split.write_text("unrelated data")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                list(corefud.dutch(split))
            archive = path / "wrong.zip"
            with ZipFile(archive, "w") as z:
                z.writestr("irrelevant", "other")
            with self.assertRaisesRegex(ValueError, "development member"):
                list(corefud.french(archive))

    @unittest.skipUnless(os.environ.get("ANNO_COREFUD_DEV_ZIP"), "local CorefUD archive required")
    def test_official_development_splits(self):
        source = Path(os.environ["ANNO_COREFUD_DEV_ZIP"])
        for component, load, expected in [
            (corefud.DUTCH, corefud.dutch, (2, 3, 792, 601, 293)),
            (corefud.FRENCH, corefud.french, (4, 3, 568, 1037, 552)),
        ]:
            with self.subTest(component=component.name):
                cases = list(load(source))
                self.assertEqual(len({c["id"] for c in cases}), len(cases))
                # Index the publisher's literal comment rows without the adapter parser.
                member = f"CorefUD-1.4-public/data/CorefUD_{component.name}/{component.filename}"
                books = {}
                with ZipFile(source) as archive:
                    raw = archive.read(member)
                    notices = set(archive.namelist())
                for line in raw.decode().splitlines():
                    if line.startswith("# newdoc id = "):
                        work = line.removeprefix("# newdoc id = ")
                        books[work] = {}
                    elif line.startswith("# sent_id = "):
                        sentence_id = line.removeprefix("# sent_id = ")
                    elif line.startswith("# text = "):
                        books[work][sentence_id] = line.removeprefix("# text = ")
                work_text = {k: "\n".join(v.values()) for k, v in books.items()}
                source_digest = hashlib.sha256(raw).hexdigest()
                counts = Counter()
                for item in cases:
                    self.assertEqual(set(item), {"id", "group", "text", "kind", "spans", "gold", "tags", "provenance"})
                    self.assertTrue(all(0 <= start < end <= len(item["text"]) for start, end in item["spans"].values()))
                    locator = item["provenance"]
                    self.assertEqual(locator["source_member"], member)
                    self.assertEqual(locator["source_sha256"], source_digest)
                    self.assertEqual(locator["work_id"], item["group"])
                    self.assertIn(locator["component_readme"], notices)
                    self.assertIn(locator["component_license"], notices)
                    work_sentences = books[locator["work_id"]]
                    self.assertEqual("\n".join(work_sentences[s] for s in locator["sentence_ids"]), item["text"])
                    self.assertEqual(work_text[item["group"]][slice(*locator["window_scalar_span"])], item["text"])
                    spans = item["spans"]
                    if item["kind"] == "nested":
                        outer, inner = spans["outer"], spans["inner"]
                        self.assertTrue(outer[0] <= inner[0] and inner[1] <= outer[1] and inner != outer)
                        counts[item["gold"]] += 1
                    else:
                        self.assertEqual(item["kind"], "choice")
                        self.assertIn(item["gold"], ("A", "B"))
                        wrong = "A" if item["gold"] == "B" else "B"
                        self.assertLessEqual(max(spans["A"][1], spans["B"][1]), spans["target"][0])
                        counts["choice"] += 1
                        counts["nearer_wrong"] += spans[wrong][1] > spans[item["gold"]][1]
                self.assertEqual((len({c["group"] for c in cases}), counts["same"], counts["different"],
                                  counts["choice"], counts["nearer_wrong"]), expected)
                with TemporaryDirectory() as folder, ZipFile(source) as archive:
                    member = f"CorefUD-1.4-public/data/CorefUD_{component.name}/{component.filename}"
                    split = Path(folder) / component.filename
                    split.write_bytes(archive.read(member))
                    self.assertEqual(list(load(split)), cases)


if __name__ == "__main__":
    unittest.main()
