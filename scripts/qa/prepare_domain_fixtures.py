"""Prepare source-bound, equally budgeted discovery and reserved NER fixtures."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
from typing import NamedTuple
import urllib.request
import zipfile

DEFAULT_SEED = "anno-domain-transfer-20261123-v1"
SOURCES = Path(__file__).with_name("domain-sources.json")
ENGLISH = {
    "ai": "CrossNER Wikipedia", "literature": "CrossNER Wikipedia",
    "music": "CrossNER Wikipedia", "politics": "CrossNER Wikipedia", "science": "CrossNER Wikipedia",
    "conll2003": "Reuters news", "wnut": "WNUT social posts", "atis": "ATIS flight queries",
    "snips": "SNIPS assistant queries", "cuad": "CUAD contracts",
}
MULTILINGUAL = {
    "conll2002-nl": "CoNLL-2002 Dutch news", "conll2002-es": "CoNLL-2002 Spanish news",
    "weibo-zh": "WeiboNER Chinese social posts",
}
# Only these directly licensed publisher inputs have a hosted QA acquisition review.
HOSTED = {"snips-weather": "Snips weather validation", "snips-restaurant": "Snips restaurant validation"}
LOCAL = {**ENGLISH, **MULTILINGUAL}
FAMILIES = {**LOCAL, **HOSTED}
LABELS = {"person": "PER", "per": "PER", "organisation": "ORG", "organization": "ORG",
          "org": "ORG", "corporation": "ORG", "location": "LOC", "loc": "LOC"}
COMPANY_SUFFIX = re.compile(r"\b(incorporated|inc|llc|ltd|limited|plc|corp|corporation)\.?$", re.I)


class Record(NamedTuple):
    key: str
    text: str
    start: int
    end: int
    label: str


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rank(seed, *values):
    data = json.dumps([seed, *values], ensure_ascii=False, separators=(",", ":")).encode()
    return digest(data)


def coarse_label(domain, role):
    if domain == "atis":
        return "LOC" if role.split(".")[-1] in {"city_name", "state_name", "country_name"} else None
    if domain == "snips":
        return "LOC" if role in {"city", "state", "country"} else None
    if domain == "weibo-zh":
        return {"PER.NAM": "PER", "ORG.NAM": "ORG", "LOC.NAM": "LOC", "GPE.NAM": "LOC"}.get(role)
    return LABELS.get(role.lower())


def sentence(domain, seed, tokens, tags):
    separator = "" if domain == "weibo-zh" else " "
    text = separator.join(tokens)
    offset = 0
    spans = []
    current = None
    for token, tag in zip(tokens, tags, strict=True):
        end = offset + len(token)
        if tag == "O":
            current = None
        else:
            prefix, role = tag.split("-", 1)
            if prefix not in {"B", "I"} or not role:
                raise ValueError(f"invalid IOB tag {tag!r}")
            # IOB1 permits I at the beginning of an entity.
            if prefix == "B" or current is None or current[2] != role:
                current = [offset, end, role]
                spans.append(current)
            else:
                current[1] = end
        offset = end + len(separator)
    anchors = [(s, e, label) for s, e, role in spans if (label := coarse_label(domain, role))]
    if anchors:
        best = min(anchors, key=lambda a: rank(seed, text, *a))
        return Record(text, text, *best)
    return None


def contract(doc, seed):
    candidates = []
    for paragraph in doc["paragraphs"]:
        text = paragraph["context"]
        for question in paragraph["qas"]:
            if not question["id"].endswith("__Parties"):
                continue
            for answer in question["answers"]:
                name, start = answer["text"], answer["answer_start"]
                if not (COMPANY_SUFFIX.search(name) and 2 <= len(name.split()) <= 12
                        and name[0].isupper() and not any(char in name for char in ':\n;"')):
                    continue
                end = start + len(name)
                if text[start:end] != name or start < 0:
                    raise ValueError("CUAD party annotation disagrees with source")
                left = max(0, start - 250)
                record = Record(doc["title"], text[left:end + 250], start - left, end - left, "ORG")
                candidates.append((rank(seed, text, start, end, "ORG"), record))
    return min(candidates)[1] if candidates else None


def read_source(data_dir, name, pins):
    raw = (data_dir / name).read_bytes()
    if digest(raw) != pins[name]["sha256"]:
        raise ValueError(f"{name}: source SHA-256 differs from the pinned revision")
    return raw.decode("utf-8")


def domain_files(domain):
    if domain == "cuad" or domain in HOSTED:
        return [domain + ".json"]
    if domain in {"atis", "snips"}:
        return [domain + ".in", domain + ".out"]
    return [domain + ".bio"]


def tagged_token(domain, line):
    if domain == "weibo-zh":
        annotated, tag = line.split("\t")
        # One source character precedes the word-position digits. The pin also
        # contains 16 rows with two literal replacement characters as their text.
        match = re.fullmatch(r"(.|\ufffd\ufffd)[0-9]+", annotated)
        if not match:
            raise ValueError("invalid Weibo source character or position")
        return match[1], tag
    if domain in {"conll2002-nl", "conll2002-es"}:
        token, _pos, tag = line.split()
        return token, tag
    return line.rsplit(maxsplit=1)


def records(domain, seed, inputs):
    if domain in HOSTED:
        utterances, = json.loads(inputs[0]).values()
        for utterance in utterances:
            parts = utterance["data"]
            text = "".join(part["text"] for part in parts)
            offset = 0
            anchors = []
            for part in parts:
                end = offset + len(part["text"])
                if part.get("entity") in {"city", "state", "country"} and end > offset:
                    anchors.append((offset, end, "LOC"))
                offset = end
            if anchors:
                best = min(anchors, key=lambda a: rank(seed, text, *a))
                yield Record(text, text, *best)
            else:
                yield None
    elif domain == "cuad":
        for doc in json.loads(inputs[0])["data"]:
            yield contract(doc, seed)
    elif domain in {"atis", "snips"}:
        texts, tags = (source.splitlines() for source in inputs)
        for text, labels in zip(texts, tags, strict=True):
            yield sentence(domain, seed, text.split(), labels.split())
    else:
        tokens, tags = [], []
        lines = inputs[0].split("\n") if domain == "weibo-zh" else inputs[0].splitlines()
        for line in lines + [""]:
            if domain in {"conll2002-nl", "conll2002-es"} and line.startswith("-DOCSTART- "):
                line = ""
            if line.strip():
                token, tag = tagged_token(domain, line)
                tokens.append(token)
                tags.append(tag)
            elif tokens:
                yield sentence(domain, seed, tokens, tags)
                tokens, tags = [], []


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def prepare(data_dir, output, *, seed, per_split, domains, pins):
    if per_split < 1 or not seed:
        raise ValueError("per-split must be positive and seed must be nonempty")
    if output.exists():
        raise FileExistsError(f"{output}: choose a new output directory")
    groups = {}
    parts = {"discovery": [], "reserved": []}
    selected_texts = set()
    source_names = set()
    for domain in domains:
        files = domain_files(domain)
        source_names.update(files)
        inputs = [read_source(data_dir, name, pins) for name in files]
        rows = list(records(domain, seed, inputs))
        eligible = [(i, row) for i, row in enumerate(rows) if row is not None]
        eligible.sort(key=lambda item: rank(seed, domain, item[1].key))
        unique = set()
        picked = []
        for index, row in eligible:
            text_hash = digest(row.text.encode("utf-8"))
            if text_hash in selected_texts or text_hash in unique:
                continue
            unique.add(text_hash)
            if len(picked) == 2 * per_split:
                continue
            rule = {"kind": "exact_span", "start": row.start, "end": row.end, "types": [row.label],
                    "text": row.text[row.start:row.end]}
            picked.append({"id": f"{domain}/{text_hash[:20]}", "domain": domain, "text": row.text,
                           "source_record": index, "source_sha256": text_hash, "assertions": [rule]})
            selected_texts.add(text_hash)
        boundary = (len(picked) + 1) // 2
        parts["discovery"].extend(picked[:boundary])
        parts["reserved"].extend(picked[boundary:])
        groups[domain] = {"source_family": FAMILIES[domain], "files": files,
                          "source_records": len(rows), "eligible_records": len(eligible),
                          "unique_eligible": len(unique), "discovery": boundary,
                          "reserved": len(picked) - boundary, "shortfall": 2 * per_split - len(picked)}
    manifest = {"seed": seed, "per_split": per_split, "groups": groups,
                "sources": {name: pins[name]["sha256"] for name in sorted(source_names)}}
    credits = {name: pins[name]["credits"] for name in sorted(source_names) if "credits" in pins[name]}
    if credits:
        manifest["source_credits"] = credits
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "panel"
        stage.mkdir()
        for part, cases in parts.items():
            raw = json_bytes(cases)
            (stage / f"{part}.json").write_bytes(raw)
            manifest[part + "_sha256"] = digest(raw)
        (stage / "manifest.json").write_bytes(json_bytes(manifest))
        stage.rename(output)
    return manifest


def fetch(data_dir, pins):
    data_dir.mkdir(parents=True, exist_ok=True)
    pending = {}
    for name, source in pins.items():
        target = data_dir / name
        if target.exists():
            read_source(data_dir, name, pins)
            continue
        pending.setdefault(source["url"], []).append(name)
    for url, names in pending.items():
        with urllib.request.urlopen(url, timeout=60) as response:
            download = response.read(pins[names[0]]["download_bytes"] + 1)
        for name in names:
            source = pins[name]
            raw = download
            if len(raw) != source["download_bytes"]:
                raise ValueError(f"{name}: unexpected download size")
            if "member" in source:
                if digest(raw) != source["archive_sha256"]:
                    raise ValueError(f"{name}: downloaded archive differs from the pin")
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    raw = archive.read(source["member"])
            if digest(raw) != source["sha256"]:
                raise ValueError(f"{name}: downloaded source SHA-256 differs from the pin")
            # Only publish a complete, verified snapshot to concurrent readers.
            with tempfile.TemporaryDirectory(dir=data_dir) as temp:
                staged = Path(temp) / name
                staged.write_bytes(raw)
                staged.replace(data_dir / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("fetch", help="download and verify publisher sources reviewed for automated acquisition")
    download.add_argument("data", type=Path)
    make = commands.add_parser("prepare", help="use verified local files; no network access")
    make.add_argument("data", type=Path)
    make.add_argument("output", type=Path)
    make.add_argument("--per-split", type=int, default=32, help="budget per domain per split")
    make.add_argument("--seed", default=DEFAULT_SEED)
    groups = {"english": ENGLISH, "multilingual": MULTILINGUAL, "all": LOCAL, "hosted": HOSTED}
    for cmd in (download, make):
        selection = cmd.add_mutually_exclusive_group()
        selection.add_argument("--group", choices=groups, default="hosted",
                               help="source group (default: hosted)")
        selection.add_argument("--domain", action="append", choices=FAMILIES,
                               help="select individual domains (repeatable)")
    args = parser.parse_args()
    pins = json.loads(SOURCES.read_bytes())
    domains = groups[args.group]
    if args.domain:
        domains = [d for d in FAMILIES if d in args.domain]
    try:
        if args.command == "fetch":
            local_only = [domain for domain in domains if domain not in HOSTED]
            if local_only:
                raise ValueError(f"{', '.join(local_only)}: automatic acquisition has no recorded clearance; "
                                 "stage lawfully obtained local files and use prepare")
            files = dict.fromkeys(name for d in domains for name in domain_files(d))
            fetch(args.data, {name: pins[name] for name in files})
        else:
            manifest = prepare(args.data, args.output, seed=args.seed, per_split=args.per_split,
                               domains=domains, pins=pins)
            for domain, row in manifest["groups"].items():
                print(f"{domain}: discovery={row['discovery']} reserved={row['reserved']} "
                      f"shortfall={row['shortfall']}")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Cannot {args.command} domain fixtures: {exc}\n")


if __name__ == "__main__":
    main()
