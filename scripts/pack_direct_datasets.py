#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Build or independently verify the Anno direct-source offline dataset packs."""

import argparse
import bz2
from collections import Counter
import csv
import fnmatch
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET
import zipfile


RECIPE = Path(__file__).parent / "qa/direct_pack_sources.json"
SCHEMA = "anno.offline-direct/v1"
INDEX_FIELDS = (
    "registry_id",
    "role",
    "split",
    "path",
    "parser",
    "records",
    "unit",
    "bytes",
    "sha256",
    "source_url",
    "source_sha256",
    "source_member",
    "metrics",
)


def encoded(obj):
    return (
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def text_reader(data):
    return io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig", newline="")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def measure(kind, data):
    """Read the packed payload, including tokens, labels, boundaries or relational IDs when present."""
    for schema, separator, required in (
        ("americasnli", "\t", ("language", "premise", "hypothesis", "label")),
        ("qxo", "\t", ("index", "quechua", "tag", "coref")),
        ("winoqueer", ",", ("sent_x", "sent_y", "Gender_ID_x", "Gender_ID_y")),
        ("harmonies", "\t", ("corpus", "piece", "mc", "mn", "staff", "label")),
        ("score-catalog", "\t", ("corpus", "piece", "label_count")),
    ):
        if kind == schema:
            rows = csv.DictReader(text_reader(data), delimiter=separator)
            require(
                set(required) <= set(rows.fieldnames or []),
                f"{kind}: unexpected columns",
            )
            count = 0
            groups = Counter()
            for row in rows:
                require(
                    None not in row and all(row[key] is not None for key in required),
                    f"{kind}: misaligned row",
                )
                if kind == "americasnli":
                    require(
                        row["label"] in ("entailment", "neutral", "contradiction")
                        and row["premise"]
                        and row["hypothesis"],
                        "NLI label or text missing",
                    )
                    groups[row["language"]] += 1
                elif kind == "winoqueer":
                    require(
                        row["sent_x"] and row["sent_y"], "Empty counterfactual pair"
                    )
                count += 1
            result = {
                "unit": {
                    "qxo": "annotated token rows",
                    "winoqueer": "paired prompts",
                    "harmonies": "harmonic labels",
                    "score-catalog": "scores",
                    "americasnli": "inference pairs",
                }[kind],
                "records": count,
            }
            if groups:
                result["language_counts"] = dict(groups)
            return result
    if kind in ("bio", "charter", "conllu", "gicoref"):
        sequences = tokens = tagged = 0
        documents = set()
        in_sequence = False
        for line in text_reader(data):
            raw = line.rstrip("\r\n")
            if not raw.strip():
                in_sequence = False
                continue
            if kind == "gicoref":
                if raw.startswith("#begin document"):
                    documents.add(raw)
                if raw.startswith("#"):
                    continue
                fields = raw.split("\t")
                require(len(fields) == 2 and fields[0], "Invalid GICoref token")
                tagged += fields[-1] != "-"
            elif kind == "conllu":
                if raw.startswith("#"):
                    continue
                fields = raw.split("\t")
                require(len(fields) == 10, "Expected ten CoNLL-U columns")
                if not fields[0].isdigit():
                    continue
            elif kind == "charter":
                fields = raw.rstrip("\t").split("\t")
                require(
                    len(fields) == 5 and fields[-1],
                    "Invalid charter token/layers/document",
                )
                require(
                    all(re.fullmatch(r"O|[BI]-.+", t) for t in fields[1:3]),
                    "Invalid charter BIO layer",
                )
                if not fields[0]:
                    require(fields[1:3] == ["O", "O"], "Annotated empty charter token")
                    continue
                documents.add(fields[-1])
                tagged += any(t != "O" for t in fields[1:3])
            else:
                fields = raw.split()
                require(
                    len(fields) >= 2 and re.fullmatch(r"O|[BIO]-.+", fields[-1]),
                    "Invalid BIO token/label",
                )
                tagged += fields[-1] != "O"
            sequences += not in_sequence
            in_sequence = True
            tokens += 1
        require(tokens > 0, "No annotated tokens")
        if kind in ("charter", "gicoref"):
            return {
                "records": len(documents),
                "unit": "documents",
                "tokens": tokens,
                "tokens_with_markers": tagged,
            }
        result = {"records": sequences, "unit": "sentence blocks", "tokens": tokens}
        if kind == "bio":
            result["tagged_tokens"] = tagged
        return result
    if kind in ("bbq", "rams", "wiki-articles"):
        count = mentions = 0
        for line in text_reader(data):
            if not line.strip():
                continue
            r = json.loads(line)
            if kind == "bbq":
                require(
                    r["context"]
                    and r["question"]
                    and r["label"] in (0, 1, 2)
                    and all(r[f"ans{i}"] for i in range(3)),
                    "Invalid three-choice question",
                )
            elif kind == "rams":
                size = sum(map(len, r["sentences"]))
                require(size and r["evt_triggers"], "Missing event context")
                for head, tail, _ in r["gold_evt_links"]:
                    require(
                        all(0 <= s <= e < size for s, e in (head, tail)),
                        "Out-of-window event annotation",
                    )
                mentions += len(r["gold_evt_links"])
            elif kind == "wiki-articles":
                require(
                    r["title"]
                    and r["page_id"]
                    and r["revision_id"]
                    and r["history_url"].startswith("https://ie.wikipedia.org/"),
                    "Missing article attribution",
                )
            count += 1
        result = {
            "records": count,
            "unit": {
                "bbq": "questions",
                "rams": "event windows",
                "wiki-articles": "main-namespace nonredirect articles",
            }[kind],
        }
        if kind == "rams":
            result["event_argument_links"] = mentions
        return result
    if kind in ("mudoco", "fewrel", "quoref", "coqa", "json-map"):
        root = json.loads(data)
        if kind == "mudoco":
            dialogs = root["dialogs"]
            splits = Counter()
            turns = 0
            for d in dialogs.values():
                require(isinstance(d["turns"], list) and d["turns"], "Empty dialogue")
                splits[d["split"]] += 1
                turns += len(d["turns"])
            return {
                "unit": "dialogues",
                "records": len(dialogs),
                "turns": turns,
                "split_counts": dict(splits),
            }
        if kind == "fewrel":
            count = 0
            for label, rows in root.items():
                require(label.startswith("P") and rows, "Missing relation examples")
                for row in rows:
                    words = row["tokens"]
                    require(
                        words
                        and all(
                            0 <= i < len(words)
                            for k in ("h", "t")
                            for position in row[k][2]
                            for i in position
                        ),
                        "Invalid relation entity position",
                    )
                count += len(rows)
            return {
                "unit": "relation examples",
                "records": count,
                "relations": len(root),
            }
        if kind == "quoref":
            count = answers = 0
            mismatches = 0
            affected = set()
            for article in root["data"]:
                for paragraph in article["paragraphs"]:
                    for q in paragraph["qas"]:
                        require(
                            q["question"] and q["id"] and q["answers"],
                            "Unanswered Quoref question",
                        )
                        for a in q["answers"]:
                            start, answer = a["answer_start"], a["text"]
                            context = paragraph["context"]
                            require(
                                0 <= start <= len(context) - len(answer)
                                and answer in context,
                                "Missing Quoref answer text or out-of-range offset",
                            )
                            if context[start : start + len(answer)] != answer:
                                mismatches += 1
                                affected.add(q["id"])
                            answers += 1
                        count += 1
            return {
                "unit": "questions",
                "records": count,
                "answer_spans": answers,
                "source_offset_mismatches": mismatches,
                "questions_with_offset_mismatches": len(affected),
            }
        if kind == "coqa":
            count = 0
            for story in root["data"]:
                require(
                    story["source"] in ("wikipedia", "gutenberg") and story["story"],
                    "Unexpected CoQA source",
                )
                require(
                    [q["turn_id"] for q in story["questions"]]
                    == [a["turn_id"] for a in story["answers"]],
                    "Unaligned CoQA turns",
                )
                count += len(story["questions"])
            return {
                "unit": "question-answer turns",
                "records": count,
                "passages": len(root["data"]),
            }
        return {"unit": "mapping entries", "records": len(root)}
    if kind in ("ami-words", "ami-entities", "ami-types", "ddi"):
        root = ET.fromstring(data)
        if kind == "ddi":
            sentences = root.findall(".//sentence")
            entities = pairs = 0
            for s in sentences:
                ids = {e.get("id") for e in s.findall("entity")}
                entities += len(ids)
                for p in s.findall("pair"):
                    require(
                        p.get("e1") in ids and p.get("e2") in ids,
                        "DDI pair refers to absent entity",
                    )
                    pairs += 1
            return {
                "unit": "sentences",
                "records": len(sentences),
                "entities": entities,
                "candidate_pairs": pairs,
            }
        tag, unit = {
            "ami-words": ("w", "transcript words"),
            "ami-entities": ("named-entity", "linked named entities"),
            "ami-types": ("ne-type", "entity ontology nodes"),
        }[kind]
        return {
            "unit": unit,
            "records": sum(v.tag.split("}")[-1] == tag for v in root.iter()),
        }
    if kind == "tweetnerd":
        named = no_mention = 0
        ids = set()
        for row in csv.reader(text_reader(data), delimiter="\t"):
            if row == ["tweet_id", "phrase", "start", "end", "entityId", "score"]:
                continue
            require(len(row) == 6 and row[0].isdigit(), "Invalid TweetNERD annotation")
            ids.add(row[0])
            start, end = int(row[2]), int(row[3])
            float(row[5])
            if row[1:] == ["NO_PHRASE", "-1", "-1", "NO_ENTITY", "-1"]:
                no_mention += 1
            else:
                require(0 <= start <= end, "Invalid TweetNERD mention bounds")
                named += 1
        return {
            "unit": "annotation rows",
            "records": named + no_mention,
            "unique_tweet_ids": len(ids),
            "no_mention_rows": no_mention,
        }
    if kind == "elgold":
        text = data.decode("utf-8-sig")
        spans = re.findall(r"\{\{([^{}]*?)\}\}", text)
        require(all(len(s.split("|")) >= 3 for s in spans), "Invalid ELGold marker")
        return {"unit": "documents", "records": 1, "inline_annotations": len(spans)}
    if kind in ("bc-sentences", "bc-spans", "drug-entities", "drug-relations", "ener"):
        n = unusual = 0
        for line in text_reader(data):
            if not line.strip():
                continue
            if kind == "bc-sentences":
                require(
                    len(line.strip().split(maxsplit=1)) == 2, "Missing BC2GM sentence"
                )
            elif kind == "bc-spans":
                require(re.match(r"^[^|]+\|\d+ \d+\|", line), "Invalid BC2GM span")
            elif kind.startswith("drug-"):
                f = line.rstrip("\r\n").split("\t")
                require(
                    len(f) >= (6 if kind == "drug-entities" else 4) and f[0].isdigit(),
                    "Invalid DrugProt table",
                )
                if kind == "drug-entities":
                    require(int(f[3]) <= int(f[4]), "Invalid DrugProt offsets")
            elif kind == "ener":
                f = next(csv.reader([line]))
                require(len(f) >= 2, "Invalid E-NER token")
                unusual += not bool(re.fullmatch(r"O|[BI]-.+", f[-1]))
            n += 1
        unit = {
            "bc-sentences": "sentences",
            "bc-spans": "span annotations",
            "drug-entities": "entity spans",
            "drug-relations": "directed relations",
            "ener": "token rows",
        }[kind]
        result = {"unit": unit, "records": n}
        if kind == "ener":
            result["rows_outside_BIO"] = unusual
        return result
    if kind == "lines":
        return {
            "unit": "nonempty lines",
            "records": sum(bool(l.strip()) for l in text_reader(data)),
        }
    if kind == "notice":
        return {"unit": "notice", "records": 1}
    raise ValueError(f"Unknown parser {kind}")


def acquire(src, caches, fetch):
    for key in (sha(src["url"].encode()), src["sha256"]):
        for cache in caches:
            path = cache / key
            if path.is_file():
                require(
                    path.stat().st_size == src["bytes"]
                    and file_sha(path) == src["sha256"],
                    f"Data changed at {path}",
                )
                return path
    if not fetch:
        raise FileNotFoundError(
            f"Missing verified cache for {src['url']}. Supply --fetch to download the pinned bytes."
        )
    cache = caches[-1]
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temp:
        partial = Path(temp.name)
        try:
            with urlopen(
                Request(src["url"], headers={"User-Agent": "Anno offline corpus pack"}),
                timeout=120,
            ) as incoming:
                shutil.copyfileobj(incoming, temp)
            temp.flush()
            require(
                partial.stat().st_size == src["bytes"]
                and file_sha(partial) == src["sha256"],
                f"Source changed: {src['url']}",
            )
            path = cache / sha(src["url"].encode())
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    return path


def additions(src, caches, fetch):
    form = src.get("container", "file")
    if form == "charter":
        from medieval_charter_survey import ARCHIVE_BYTES, DIRECTORY, PARTS, charter_zip

        cache = next(
            (p for p in caches if (p / "medieval-charter-6463699").is_dir()), caches[-1]
        )
        base = src["url"].split("#")[0]
        if not fetch:
            complete = cache / sha(base.encode())
            required = {DIRECTORY[-1], *(part[-1] for part in PARTS.values())}
            range_cache = cache / "medieval-charter-6463699"
            require(
                (complete.is_file() and complete.stat().st_size == ARCHIVE_BYTES)
                or all((range_cache / name).is_file() for name in required),
                "Missing verified charter ZIP ranges",
            )
        with charter_zip(base, cache) as z:
            name = src["url"].split("#")[1]
            b = z.read(name)
            require(
                sha(b) == src["sha256"] and len(b) == src["bytes"],
                "Charter member changed",
            )
            yield src["files"][0], name, b
        return
    path = acquire(src, caches, fetch)
    if form == "wiki":
        output = io.BytesIO()
        for _, page in ET.iterparse(bz2.BZ2File(path), events=("end",)):
            if page.tag.split("}")[-1] != "page":
                continue
            fields = {e.tag.split("}")[-1]: e for e in page}
            if fields["ns"].text == "0" and "redirect" not in fields:
                revision = {e.tag.split("}")[-1]: e for e in fields["revision"]}
                pid = fields["id"].text
                rid = revision["id"].text
                article = {
                    "title": fields["title"].text,
                    "page_id": pid,
                    "revision_id": rid,
                    "text": revision["text"].text or "",
                    "revision_url": f"https://ie.wikipedia.org/w/index.php?oldid={rid}",
                    "history_url": f"https://ie.wikipedia.org/w/index.php?curid={pid}&action=history",
                }
                output.write(
                    (
                        json.dumps(article, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    ).encode()
                )
            page.clear()
        yield src["files"][0], "nonredirect articles, namespace 0", output.getvalue()
        return
    if form == "file":
        yield src["files"][0], "", path.read_bytes()
        return
    if form == "coqa":
        data = json.loads(path.read_bytes())
        selected = [
            r for r in data["data"] if r["source"] in ("wikipedia", "gutenberg")
        ]
        yield src["files"][0], "Wikipedia and Gutenberg domain records", encoded(
            {**data, "data": selected}
        )
        return
    with zipfile.ZipFile(path) if form == "zip" else tarfile.open(path) as archive:
        names = (
            [i.filename for i in archive.infolist() if not i.is_dir()]
            if form == "zip"
            else [m.name for m in archive.getmembers() if m.isfile()]
        )
        for rule in src["files"]:
            matches = sorted(n for n in names if fnmatch.fnmatchcase(n, rule["match"]))
            require(
                len(matches) == rule["count"],
                f"Archive subset changed: {rule['match']} ({len(matches)} vs {rule['count']})",
            )
            for member in matches:
                data = (
                    archive.read(member)
                    if form == "zip"
                    else archive.extractfile(member).read()
                )
                yield rule, member, data


def safe_path(entry):
    p = PurePosixPath(entry)
    require(
        not p.is_absolute() and ".." not in p.parts and "\\" not in entry,
        "Unsafe package member",
    )
    return entry


def write_member(archive, name, data):
    info = zipfile.ZipInfo(safe_path(name), date_time=(2026, 11, 28, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data, compresslevel=5)


def index_bytes(rows):
    b = io.StringIO(newline="")
    w = csv.DictWriter(b, fieldnames=INDEX_FIELDS, delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return b.getvalue().encode()


def build(recipe, name, caches, fetch, out):
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"anno-direct-{name}.zip"
    tmp = final.with_suffix(".partial")
    datasets = {k: v for k, v in recipe["datasets"].items() if v["pack"] == name}
    rows = []
    seen = set()
    manifests = {}
    try:
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as archive:
            for ident, dataset in datasets.items():
                start = len(rows)
                for src in dataset["sources"]:
                    for rule, member, b in additions(src, caches, fetch):
                        role = rule.get("role", "data")
                        dest = rule.get("dest") or member
                        path = safe_path(
                            f'{"data" if role=="data" else "notices"}/{ident}/{dest}'
                        )
                        require(path not in seen, f"Duplicate {path}")
                        seen.add(path)
                        try:
                            stats = measure(rule["parser"], b)
                        except Exception as e:
                            raise ValueError(
                                f"Offline parser failed for {path}: {e}"
                            ) from e
                        write_member(archive, path, b)
                        rows.append(
                            dict(
                                registry_id=ident,
                                role=role,
                                split=rule["split"],
                                path=path,
                                parser=rule["parser"],
                                records=stats["records"],
                                unit=stats["unit"],
                                bytes=len(b),
                                sha256=sha(b),
                                source_url=src["url"],
                                source_sha256=src["sha256"],
                                source_member=member,
                                metrics=json.dumps(
                                    stats, sort_keys=True, ensure_ascii=False
                                ),
                            )
                        )
                data_rows = [row for row in rows[start:] if row["role"] == "data"]
                require(data_rows, f"No files for {ident}")
                summary = {}
                for row in data_rows:
                    summary.setdefault(row["split"], Counter())[row["unit"]] += row[
                        "records"
                    ]
                manifests[ident] = {
                    k: v for k, v in dataset.items() if k not in ("sources", "pack")
                }
                manifests[ident].update(
                    data_files=len(data_rows),
                    data_bytes=sum(r["bytes"] for r in data_rows),
                    splits=summary,
                )
                print(f"{ident}: {len(data_rows)} data files", flush=True)
            manifest = {
                "offline_cross_file_checks": linked_evidence(archive, rows),
                "schema": SCHEMA,
                "archive": final.name,
                "registry_ids": list(manifests),
                "datasets": manifests,
                "data_files": sum(d["data_files"] for d in manifests.values()),
                "data_bytes": sum(d["data_bytes"] for d in manifests.values()),
            }
            idx = index_bytes(rows)
            write_member(archive, "file-index.tsv", idx)
            write_member(archive, "manifest.json", encoded(manifest))
            write_member(archive, "sources.json", encoded({"datasets": datasets}))
            write_member(archive, "verify.py", Path(__file__).read_bytes())
            write_member(
                archive,
                "README.md",
                f"# Anno offline {name} datasets\n\nOpen `manifest.json` for the registry IDs, coverage, source terms and exclusions. `data/` contains the payloads. `file-index.tsv` provides each file’s source, SHA-256, split and parser results. Dataset terms apply separately, including noncommercial and research-only material. Check each entry before sharing.\n\nTo verify without a network connection or third-party package, extract `verify.py` and run `python3 verify.py --verify {final.name}` (Python 3.11+). The verifier opens every file, rechecks its digest, and reruns its format parser.\n".encode(),
            )
        verify(tmp)
        tmp.replace(final)
        receipt = manifest | {
            "archive_bytes": final.stat().st_size,
            "archive_sha256": file_sha(final),
        }
        (out / f"anno-direct-{name}.manifest.json").write_bytes(encoded(receipt))
        (out / f"anno-direct-{name}.files.tsv").write_bytes(idx)
        return receipt
    finally:
        tmp.unlink(missing_ok=True)


def linked_evidence(archive, rows):
    """Verify relationships spanning multiple files in the package itself."""
    groups = {}
    for row in rows:
        if row["role"] == "data":
            groups.setdefault(row["registry_id"], []).append(row)
    results = {}
    if "LojbanTatoeba" in groups:
        left, right = groups["LojbanTatoeba"]
        with archive.open(left["path"]) as one, archive.open(right["path"]) as two:
            matched = sum(
                1 for a, b in zip(one, two, strict=True) if a.strip() and b.strip()
            )
        require(
            matched == int(left["records"]) == int(right["records"]),
            "Unaligned Tatoeba translations",
        )
        results["LojbanTatoeba"] = {"aligned_nonempty_sentence_pairs": matched}
    if "AMIMeeting" in groups:
        tracks = {}
        labels = []
        types = set()
        nite = "{http://nite.sourceforge.net/}"
        for row in groups["AMIMeeting"]:
            file = PurePosixPath(row["path"]).name
            if row["parser"] == "ami-words":
                tracks[file] = row["path"]
            elif row["parser"] == "ami-types":
                types = {
                    x.attrib[nite + "id"]
                    for x in ET.fromstring(archive.read(row["path"])).iter()
                }
            elif row["parser"] == "ami-entities":
                labels.append(row["path"])
        mentions = pointers = 0
        for name in labels:
            wanted = PurePosixPath(name).name.removesuffix(".ne.xml") + ".words.xml"
            words = ET.fromstring(archive.read(tracks[wanted]))
            positions = {e.attrib[nite + "id"]: i for i, e in enumerate(words)}
            for entity in ET.fromstring(archive.read(name)):
                children = entity.findall(nite + "child")
                require(children, "AMI annotation without word links")
                for child in children:
                    ref = re.fullmatch(
                        r"([^#]+)#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?",
                        child.attrib["href"],
                    )
                    require(
                        ref
                        and ref[1] == wanted
                        and 0 <= positions[ref[2]] <= positions[ref[3] or ref[2]],
                        "Broken AMI transcript link",
                    )
                for pointer in entity.findall(nite + "pointer"):
                    ref = re.fullmatch(
                        r"ne-types\.xml#id\(([^)]+)\)", pointer.attrib["href"]
                    )
                    require(ref and ref[1] in types, "Broken AMI entity type link")
                    pointers += 1
                mentions += 1
        results["AMIMeeting"] = {
            "speaker_transcripts": len(tracks),
            "speaker_tracks_with_entity_layer": len(labels),
            "entities_with_valid_transcript_links": mentions,
            "valid_type_links": pointers,
        }
    if "DistantListeningCorpus" in groups:
        rows_by_type = {r["parser"]: r for r in groups["DistantListeningCorpus"]}
        with archive.open(rows_by_type["score-catalog"]["path"]) as f:
            expected = {
                (r["corpus"], r["piece"]): int(r["label_count"])
                for r in csv.DictReader(
                    io.TextIOWrapper(f, encoding="utf-8-sig", newline=""),
                    delimiter="\t",
                )
            }
        with archive.open(rows_by_type["harmonies"]["path"]) as f:
            actual = Counter(
                (r["corpus"], r["piece"])
                for r in csv.DictReader(
                    io.TextIOWrapper(f, encoding="utf-8-sig", newline=""),
                    delimiter="\t",
                )
            )
        require(
            not actual.keys() - expected.keys()
            and all(actual[k] == n for k, n in expected.items()),
            "Harmonic label count differs from the source catalogue",
        )
        results["DistantListeningCorpus"] = {
            "catalogued_scores": len(expected),
            "annotations_matched_to_catalogue": sum(actual.values()),
        }
    if "DrugProtBioCreative" in groups:
        reported = {}
        for split in ("training", "development"):
            tables = {
                r["parser"]: r["path"]
                for r in groups["DrugProtBioCreative"]
                if r["split"].startswith(split + "/")
            }
            with archive.open(tables["drug-entities"]) as f:
                entities = {
                    (r[0], r[1])
                    for r in csv.reader(
                        io.TextIOWrapper(f, encoding="utf-8-sig", newline=""),
                        delimiter="\t",
                    )
                    if r
                }
            count = 0
            with archive.open(tables["drug-relations"]) as f:
                for row in csv.reader(
                    io.TextIOWrapper(f, encoding="utf-8-sig", newline=""),
                    delimiter="\t",
                ):
                    if not row:
                        continue
                    require(
                        all(
                            (row[0], arg.split(":", 1)[1]) in entities
                            for arg in row[2:4]
                        ),
                        "DrugProt relation refers to absent entity",
                    )
                    count += 1
            reported[split] = count
        results["DrugProtBioCreative"] = {"relations_with_both_endpoint_ids": reported}
    return results


def verify(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP members")
        manifest = json.loads(z.read("manifest.json"))
        require(manifest["schema"] == SCHEMA, "Unsupported manifest")
        rows = list(
            csv.DictReader(
                io.TextIOWrapper(
                    z.open("file-index.tsv"), encoding="utf-8", newline=""
                ),
                delimiter="\t",
            )
        )
        datafiles = {}
        counts = {}
        byte_counts = Counter()
        file_counts = Counter()
        for row in rows:
            name = safe_path(row["path"])
            require(name not in datafiles, "Duplicate indexed payload")
            b = z.read(name)
            datafiles[name] = True
            require(
                len(b) == int(row["bytes"]) and sha(b) == row["sha256"],
                f"Changed bytes in {name}",
            )
            stats = measure(row["parser"], b)
            require(
                stats == json.loads(row["metrics"])
                and stats["records"] == int(row["records"])
                and stats["unit"] == row["unit"],
                f"Changed parse result in {name}",
            )
            if row["role"] == "data":
                ident = row["registry_id"]
                counts.setdefault(ident, {}).setdefault(row["split"], Counter())[
                    stats["unit"]
                ] += stats["records"]
                byte_counts[ident] += len(b)
                file_counts[ident] += 1
        require(
            set(names)
            == set(datafiles)
            | {
                "manifest.json",
                "file-index.tsv",
                "sources.json",
                "README.md",
                "verify.py",
            },
            "Unindexed ZIP members",
        )
        require(
            set(counts) == set(manifest["registry_ids"]) == set(manifest["datasets"]),
            "Dataset ID mismatch",
        )
        for ident, scope in manifest["datasets"].items():
            require(
                counts[ident] == scope["splits"]
                and file_counts[ident] == scope["data_files"]
                and byte_counts[ident] == scope["data_bytes"],
                f"Scope differs for {ident}",
            )
        require(
            sum(file_counts.values()) == manifest["data_files"]
            and sum(byte_counts.values()) == manifest["data_bytes"],
            "Pack totals differ",
        )
        require(
            linked_evidence(z, rows) == manifest["offline_cross_file_checks"],
            "Cross-file validation has changed",
        )
    print(
        f'Validated {manifest["data_files"]} data files and {len(counts)} registry IDs',
        flush=True,
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    g = cli.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", type=Path)
    g.add_argument("--pack", choices=("annotations", "language", "all"))
    cli.add_argument("--recipe", type=Path, default=RECIPE)
    cli.add_argument("--cache", action="append", type=Path)
    cli.add_argument("--fetch", action="store_true")
    cli.add_argument("--out", type=Path)
    args = cli.parse_args()
    if args.verify:
        verify(args.verify)
        return
    if not args.out or not args.cache:
        cli.error("--pack requires --out and at least one --cache")
    recipe = json.loads(args.recipe.read_bytes())
    for name in ("annotations", "language") if args.pack == "all" else (args.pack,):
        build(recipe, name, args.cache, args.fetch, args.out)


if __name__ == "__main__":
    main()
