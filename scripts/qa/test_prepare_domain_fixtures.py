import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.request
import zipfile

import prepare_domain_fixtures as panel
from check_span_assertions import load_cases


class DomainFixturesTest(unittest.TestCase):
    def test_bio_and_slot_inputs_keep_character_offsets_and_label_scope(self):
        row = panel.sentence("science", "seed", ["🦊", "São", "Paulo", "ESA"],
                             ["O", "I-location", "I-location", "B-university"])
        self.assertEqual((row.text[row.start:row.end], row.start, row.end, row.label),
                         ("São Paulo", 2, 11, "LOC"))
        with self.assertRaises(ValueError):
            panel.sentence("science", "seed", ["orphan"], [])
        record, = panel.records("atis", "seed", ["to new york", "O B-toloc.city_name I-toloc.city_name"])
        self.assertEqual((record.text[record.start:record.end], record.label), ("new york", "LOC"))
        self.assertIsNone(panel.sentence("snips", "s", ["band"], ["B-artist"]))

    def test_fetch_and_prepare_select_the_same_source_group(self):
        selections = [([], panel.ENGLISH), (["--group", "multilingual"], panel.MULTILINGUAL),
                      (["--group", "all"], panel.FAMILIES), (["--domain", "weibo-zh"], ["weibo-zh"])]
        for flags, domains in selections:
            with self.subTest(flags=flags):
                with patch.object(sys, "argv", ["qa", "fetch", "data", *flags]), patch.object(panel, "fetch") as fetch:
                    panel.main()
                files = set(name for domain in domains for name in panel.domain_files(domain))
                self.assertEqual(set(fetch.call_args.args[1]), files)
                with patch.object(sys, "argv", ["qa", "prepare", "data", "out", *flags]), patch.object(
                    panel, "prepare", return_value={"groups": {}}
                ) as prepare:
                    panel.main()
                self.assertEqual(list(prepare.call_args.kwargs["domains"]), list(domains))

    def test_weibo_uses_source_scalars_and_projects_named_entities(self):
        data = ("🛰0\tO\n��0\tO\n50\tO\n 0\tO\n北10\tB-GPE.NAM\n京11\tI-GPE.NAM\n"
                "人0\tB-PER.NOM\n\n部0\tB-ORG.NAM\n\n她0\tB-PER.NOM\n")
        place, organization, nominal = panel.records("weibo-zh", "s", [data])
        self.assertEqual((place.text, place.start, place.end, place.label), ("🛰��5 北京人", 5, 7, "LOC"))
        self.assertEqual(organization, panel.Record("部", "部", 0, 1, "ORG"))
        self.assertIsNone(nominal)
        with self.assertRaisesRegex(ValueError, "Weibo"):
            list(panel.records("weibo-zh", "s", ["missing-position\tO"]))

    def test_conll_members_share_verified_archive_and_prepare_offline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "source.zip"
            data = {"conll2002-nl": ("ned", "-DOCSTART- -DOCSTART- O\nZoë N B-PER\n"),
                    "conll2002-es": ("esp", "La DA B-LOC\nCoruña NC I-LOC\n")}
            with zipfile.ZipFile(archive, "w") as z:
                for member, source in data.values():
                    z.writestr(f"conll2002/{member}.testb", source.encode())
            pins = {domain + ".bio": {"url": archive.as_uri(), "member": f"conll2002/{member}.testb",
                                       "download_bytes": archive.stat().st_size,
                                       "archive_sha256": panel.digest(archive.read_bytes()),
                                       "sha256": panel.digest(source.encode())}
                    for domain, (member, source) in data.items()}
            cache = root / "cache"
            with patch("urllib.request.urlopen", wraps=urllib.request.urlopen) as opening:
                panel.fetch(cache, pins)
                opening.assert_called_once()
            with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
                panel.fetch(cache, pins)
                panel.prepare(cache, root / "panel", seed="s", per_split=1,
                              domains=list(data), pins=pins)
            cases = json.loads((root / "panel/discovery.json").read_bytes())
            pairs = [(c["text"], c["assertions"][0]["types"]) for c in cases]
            self.assertEqual(pairs, [("Zoë", ["PER"]), ("La Coruña", ["LOC"])])
            pins["conll2002-nl.bio"]["archive_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "archive differs"):
                panel.fetch(root / "bad", pins)
            self.assertFalse((root / "bad/conll2002-nl.bio").exists())

    def test_contract_preserves_source_window_and_rejects_misaligned_gold(self):
        text = "🦊" * 301 + "Acme Inc." + "x" * 301
        answer = {"answer_start": 301, "text": "Acme Inc."}
        doc = {"title": "agreement", "paragraphs": [{"context": text, "qas": [
            {"id": "a__Parties", "answers": [answer]}]}]}
        row = panel.contract(doc, "s")
        self.assertEqual((row.start, row.end, row.label, len(row.text)), (250, 259, "ORG", 509))
        self.assertEqual(row.text[row.start:row.end], answer["text"])
        answer["answer_start"] += 1
        with self.assertRaisesRegex(ValueError, "disagrees"):
            panel.contract(doc, "s")

    def test_budget_duplicates_shortfall_and_verified_local_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Both copies from AI and the copy from music have exactly the same model input.
            corpora = {"ai.bio": "Ada I-person\n\nAda I-person\n\nZoë B-person\n\none O\n",
                       "music.bio": "Ada B-person\n\nJazz B-misc\n\nBand B-organisation\n"}
            pins = {}
            for name, text in corpora.items():
                raw = text.encode()
                (root / name).write_bytes(raw)
                pins[name] = {"sha256": panel.digest(raw)}
            kwargs = dict(seed="fixed", per_split=1, domains=["ai", "music"], pins=pins)
            a, b = root / "a", root / "b"
            with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
                manifest = panel.prepare(root, a, **kwargs)
                panel.prepare(root, b, **kwargs)
            self.assertEqual(manifest["groups"]["ai"]["source_records"], 4)
            self.assertEqual(manifest["groups"]["ai"]["eligible_records"], 3)
            self.assertEqual(manifest["groups"]["music"]["shortfall"], 1)
            self.assertEqual(manifest["groups"]["music"]["unique_eligible"], 1)
            unique = set()
            for name in ("discovery", "reserved"):
                data = (a / (name + ".json")).read_bytes()
                self.assertEqual(data, (b / (name + ".json")).read_bytes())
                self.assertEqual(panel.digest(data), manifest[name + "_sha256"])
                cases = load_cases(json.loads(data))
                for row in cases.values():
                    hashed = panel.digest(row["text"].encode())
                    self.assertEqual(row["source_sha256"], hashed)
                    self.assertNotIn(hashed, unique)
                    unique.add(hashed)
            self.assertEqual(len(unique), 3)
            with self.assertRaises(FileExistsError):
                panel.prepare(root, a, **kwargs)
            (root / "music.bio").write_text("changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                panel.prepare(root, root / "bad", **kwargs)
            self.assertFalse((root / "bad").exists())

    def test_sparse_domains_balance_splits_after_deduplication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for available in (0, 1, 3, 4, 31):
                with self.subTest(available=available):
                    names = [f"Name{i}" for i in range(available)]
                    # Give every eligible source a duplicate to exercise allocation from unique texts.
                    raw = ("Empty O\n\n" + "".join(f"{name} B-person\n\n" for name in names * 2)).encode()
                    (root / "ai.bio").write_bytes(raw)
                    out = root / str(available)
                    manifest = panel.prepare(root, out, seed="s", per_split=32, domains=["ai"],
                                             pins={"ai.bio": {"sha256": panel.digest(raw)}})
                    row = manifest["groups"]["ai"]
                    discovery = json.loads((out / "discovery.json").read_bytes())
                    reserved = json.loads((out / "reserved.json").read_bytes())
                    self.assertEqual((len(discovery), row["discovery"]), ((available + 1) // 2,) * 2)
                    self.assertEqual((len(reserved), row["reserved"]), (available // 2,) * 2)
                    self.assertEqual(row["shortfall"], 64 - available)
                    self.assertCountEqual([case["text"] for case in discovery + reserved], names)

    def test_prepared_fixture_reports_each_domains_paired_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pins = {}
            for domain, token in [("ai", "Ada"), ("music", "Zoë")]:
                raw = f"{token} B-person\n".encode()
                name = domain + ".bio"
                (root / name).write_bytes(raw)
                pins[name] = {"sha256": panel.digest(raw)}
            prepared = root / "panel"
            panel.prepare(root, prepared, seed="fixed", per_split=1, domains=["ai", "music"], pins=pins)
            fixture = prepared / "discovery.json"
            cases = json.loads(fixture.read_bytes())
            self.assertEqual([c["domain"] for c in cases], ["ai", "music"])
            predictions = {"ai": root / "candidate.jsonl", "music": root / "baseline.jsonl"}
            for hit_domain, path in predictions.items():
                with path.open("w", encoding="utf-8") as stream:
                    for case in cases:
                        entities = []
                        if case["domain"] == hit_domain:
                            rule, = case["assertions"]
                            entities.append({"start": rule["start"], "end": rule["end"],
                                             "type": rule["types"][0], "text": rule["text"]})
                        stream.write(json.dumps({"id": case["id"], "source_sha256": case["source_sha256"],
                                                 "entities": entities}) + "\n")
            checker = Path(__file__).with_name("check_span_assertions.py")
            result = subprocess.run([sys.executable, checker, "--fixture", fixture,
                                     "--predictions", predictions["ai"],
                                     "--baseline-predictions", predictions["music"]],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)  # One lost assertion.
            report = json.loads(result.stdout)
            domains = {row["domain"]: row for row in report["domains"]}
            self.assertEqual(set(domains), {"ai", "music"})
            for domain, baseline, candidate in [("ai", 0, 1), ("music", 1, 0)]:
                row = domains[domain]
                self.assertEqual(row["documents"], 1)
                self.assertEqual(row["exact_span"], {"passed": candidate, "total": 1})
                self.assertEqual(row["comparison"], {"baseline_passed": baseline,
                                                     "gained": candidate, "lost": baseline})

    def test_download_rejects_wrong_content_before_publishing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "origin"
            source.write_bytes(b"correct")
            pins = {"test.bio": {"url": source.as_uri(), "download_bytes": 7,
                                 "sha256": panel.digest(b"correct")}}
            panel.fetch(root / "cache", pins)
            self.assertEqual((root / "cache/test.bio").read_bytes(), b"correct")
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                panel.fetch(root / "other", pins)
            self.assertFalse((root / "other/test.bio").exists())
