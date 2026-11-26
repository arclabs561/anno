"""Offline LitBank controls. ANNO_LITBANK_ROOT enables the official panel replay."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if __package__:
    from . import litbank
else:
    import litbank


def authored_work(root):
    text = "amber " * 40 + "🦉 Zoë met Ada. Ada's older sister saw her friend. Later she smiled."
    first_ada = text.index("Ada")
    second_ada = text.index("Ada", first_ada + 1)
    annotations = [
        (text.index("Zoë"), "Zoë", "PROP_PER", "zoe-1"),
        (first_ada, "Ada", "PROP_PER", "ada-2"),
        (second_ada, "Ada", "PROP_PER", "ada-2"),
        (second_ada, "Ada's older sister", "NOM_PER", "sister-3"),
        (text.index("her friend"), "her friend", "NOM_PER", "zoe-1"),
        (text.index("she"), "she", "PRON_PER", "zoe-1"),
    ]
    coref, entities = [], []
    for index, (a, surface, tag, chain) in enumerate(annotations):
        suffix = f" {a} {a + len(surface)}\t{surface}\n"
        coref.append(f"T{2*index}\t{tag}{suffix}T{2*index+1}\t{chain}{suffix}")
        if tag != "PRON_PER":
            entities.append(f"T{index}\tPER{suffix}")
    entries = {}
    for layer, rows in (("coref", coref), ("entities", entities)):
        folder = root / layer / "brat"
        folder.mkdir(parents=True)
        for extension, data in (("txt", text), ("ann", "".join(rows))):
            path = folder / f"1_authored_brat.{extension}"
            path.write_bytes(data.encode())
            entries[path.relative_to(root).as_posix()] = hashlib.sha256(data.encode()).hexdigest()
    manifest = "".join(f"{entries[k]}  {k}\n" for k in sorted(entries))
    return hashlib.sha256(manifest.encode()).hexdigest()


class LitBankAdapter(unittest.TestCase):
    def test_choice_alias_nesting_and_exact_unicode_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "litbank"
            checksum = authored_work(source)
            with patch.object(litbank, "WORKS", {1}), patch.object(litbank, "SOURCE_SHA256", checksum):
                loaded = list(litbank.cases(source))
                self.assertEqual(loaded, list(litbank.cases(source.parent)))
        self.assertEqual(len(loaded), 2)
        self.assertEqual({r["group"] for r in loaded}, {"1"})
        choice, = (r for r in loaded if r["kind"] == "choice")
        nested, = (r for r in loaded if r["kind"] == "nested")
        self.assertEqual(nested["gold"], "different")
        self.assertEqual(choice["tags"], ["foil_nearer"])
        self.assertEqual(choice["text"][slice(*choice["spans"][choice["gold"]])], "Zoë")
        self.assertEqual(choice["text"][slice(*choice["spans"]["target"])], "she")
        self.assertEqual([choice["text"][slice(*s)] for s in choice["aliases"]], ["her friend"])
        self.assertEqual(nested["text"][slice(*nested["spans"]["inner"])], "Ada")
        self.assertEqual(nested["text"][slice(*nested["spans"]["outer"])], "Ada's older sister")

    def test_source_identity_and_missing_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            checksum = authored_work(source)
            with patch.object(litbank, "WORKS", {1, 2}), self.assertRaisesRegex(ValueError, "missing"):
                list(litbank.cases(source))
            (source / "coref/brat/1_authored_brat.ann").write_text("")
            with patch.object(litbank, "WORKS", {1}), patch.object(litbank, "SOURCE_SHA256", checksum):
                with self.assertRaisesRegex(ValueError, "pinned"):
                    list(litbank.cases(source))

    def test_ambiguous_chain_suffixes_and_misaligned_spans(self):
        text = "Kay Rex"
        coref = "T1\tPROP_PER 0 3\tKay\nT2\tKay-7 0 3\tKay\nT3\tPROP_PER 4 7\tRex\nT4\tRex-7 4 7\tRex"
        entities = "T1\tPER 0 3\tKay\nT3\tPER 4 7\tRex"
        records, chains = litbank._mentions(text, coref, entities)
        self.assertFalse(records)
        self.assertFalse(chains)
        with self.assertRaisesRegex(ValueError, "source span"):
            litbank._mentions(text, coref.replace("0 3", "0 2"), entities)

    @unittest.skipUnless(os.environ.get("ANNO_LITBANK_ROOT"), "local LitBank source required")
    def test_pinned_work_and_decision_panel(self):
        source = Path(os.environ["ANNO_LITBANK_ROOT"])
        rows = list(litbank.cases(source))
        if not (source / "coref/brat").is_dir():
            source = source / "litbank"
        self.assertEqual(len(rows), 47)
        self.assertEqual(len({r["id"] for r in rows}), 47)
        self.assertEqual(len({r["group"] for r in rows}), 21)
        self.assertEqual(sum(r["gold"] == "same" for r in rows), 4)
        self.assertEqual(sum(r["gold"] == "different" for r in rows), 20)
        self.assertEqual(sum(r["kind"] == "choice" for r in rows), 23)
        # Bind window text, local offsets, alias sets and option ordering.
        for row in rows:
            locator = row["provenance"]
            raw = (source / locator["source_file"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), locator["source_sha256"])
            self.assertEqual(raw.decode()[slice(*locator["window_scalar_span"])], row["text"])
            self.assertEqual(locator["work_id"], row["group"])
            self.assertEqual(locator["gutenberg_url"], f"https://www.gutenberg.org/ebooks/{int(row['group'])}")
            self.assertTrue((source / locator["coref_annotations"]).is_file())
            self.assertTrue((source / locator["entity_annotations"]).is_file())
        scored = [{k: v for k, v in row.items() if k != "provenance"} for row in rows]
        canonical = json.dumps(scored, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(
            hashlib.sha256(canonical.encode()).hexdigest(),
            "659f59173d9d1ccf9b760b26e0824a83c4443bb2bc975c672bc3c94390ef254b",
        )


if __name__ == "__main__":
    unittest.main()
