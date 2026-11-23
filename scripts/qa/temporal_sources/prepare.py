"""Prepare the two frozen temporal QA panels from verified local source files."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree
from zipfile import ZipFile

MEANTIME = "7696266b571217c335e501124e06db858c1b5d34ed392b4c447f731a7d18d5e5"
MS_SPEC = "7a9ff209927f3c5d9b26e28935706b46ff8433bde4b49e2135dee186752938a4"
MS_LICENSE = "dea9265341829002e2c23a7372393eb2ed6e26085fb623f38a4ba0af833f30a6"
PANEL_SHA256 = {
    "meantime": "caedf88859b6cce230f4d588d480b0f2f2bd7cf7abe406203434bf7e04ff6da5",
    "msrt": "344785824e1031608d122c61c5b53f2446b528e7d80ededa7ad90831b7bb01cb",
}

# Source-empty full-model regression cases with unequivocally non-temporal
# readings. Several other empty cases express real dates in optional modes.
EMPTY_CASES = {
    94, 100, 102, 103, 104, 106, 107, 108, 291, 369, 370, 371, 372,
    404, 405, 406, 445, 446, 532, 533, 534, 541, 542, 552, 555, 608,
    611, 614, 706, 732, 733, 753, 769, 789, 820, 821, 829, 930, 953,
}
MS_TYPES = {"date", "daterange", "datetime", "datetimerange", "duration", "set", "time", "timerange"}


def checked(data, expected):
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"Unexpected SHA-256 (expected {expected})")
    return data


def sample(group, salt, count):
    return sorted(group, key=lambda key: sha256(f"{salt}/{key}".encode()).digest())[:count]


def news_row(xml, doc, topic, sentence):
    tokens = {
        t.attrib["t_id"]: t.text for t in xml.findall("token") if t.attrib["sentence"] == sentence
    }
    if not tokens:
        return None
    text = " ".join(tokens.values())
    offsets = {}
    pos = 0
    for tid, token in tokens.items():
        offsets[tid] = (pos, pos + len(token))
        pos += len(token) + 1

    gold = []
    has_markable = False
    for mark in xml.find("Markables"):
        anchors = {t.attrib["t_id"] for t in mark.findall("token_anchor")}
        if not anchors.intersection(tokens):
            continue
        has_markable = True
        if mark.tag != "TIMEX3":
            continue
        if not anchors.issubset(tokens):
            raise ValueError(f"TIMEX crosses sentence in {doc} sentence {sentence}")
        start = min(offsets[t][0] for t in anchors)
        end = max(offsets[t][1] for t in anchors)
        if {t for t, (a, _) in offsets.items() if start <= a < end} != anchors:
            raise ValueError(f"Discontinuous TIMEX in {doc} sentence {sentence}")
        gold.append({"text": text[start:end], "start_char": start, "end_char": end,
                     "type": mark.attrib["type"], "tid": mark.attrib["m_id"]})
    if not has_markable:
        raise ValueError(f"No document-level annotation in {doc} sentence {sentence}")
    return {"id": f"meantime-{doc}-s{sentence}", "source": topic, "doc_id": doc, "text": text,
            "time_expressions": sorted(gold, key=lambda g: (g["start_char"], g["end_char"]))}


def news_panel(path):
    # Only read the known archive. Never extract paths supplied by a zip member.
    verified = checked(path.read_bytes(), MEANTIME)
    by_topic = defaultdict(dict)
    unique = {}
    documents = {}
    with ZipFile(BytesIO(verified)) as archive:
        for name in archive.namelist():
            if "/intra-doc_annotation/" in name and name.endswith(".xml"):
                topic, filename = name.split("/")[-2:]
                by_topic[topic][filename.split("_")[0]] = name
        for topic, files in sorted(by_topic.items()):
            for doc in sample(files, f"anno-next-quality-meantime-v1/{topic}", 6):
                xml = ElementTree.fromstring(archive.read(files[doc]))
                documents[doc] = {"member": files[doc], "url": xml.attrib["url"]}
                for sentence in ("2", "3", "4"):
                    row = news_row(xml, doc, topic, sentence)
                    if row:
                        unique.setdefault(row["text"].casefold(), row)
        license_text = archive.read("meantime_newsreader_english_oct15/LICENSE.txt")
    return list(unique.values()), documents, license_text


def case_gold(text, results):
    encoded = text.encode("utf-16-le")
    gold = []
    for item in results:
        if not {"Text", "Start", "End", "TypeName"} <= item.keys():
            return None
        start = item["Start"] * 2
        end = (item["End"] + 1) * 2
        if not 0 <= start < end <= len(encoded):
            return None
        # Full-model test cases use inclusive UTF-16 indices. JSONL consumers
        # expect half-open Unicode scalar offsets on the original source string.
        lo = len(encoded[:start].decode("utf-16-le"))
        hi = len(encoded[:end].decode("utf-16-le"))
        if text[lo:hi].casefold() != item["Text"].casefold():
            return None
        gold.append({"text": text[lo:hi], "start_char": lo, "end_char": hi,
                     "type": item["TypeName"].split(".")[-1]})
    return gold


def contract_panel(path, license_path):
    cases = json.loads(checked(path.read_bytes(), MS_SPEC).decode("utf-8-sig"))
    license_text = checked(license_path.read_bytes(), MS_LICENSE)
    counts = Counter(case["Input"].casefold().strip() for case in cases)
    groups = defaultdict(dict)
    for i, case in enumerate(cases):
        text = case["Input"]
        if counts[text.casefold().strip()] != 1:
            continue
        gold = case_gold(text, case["Results"])
        if gold is None or (not gold and i not in EMPTY_CASES):
            continue
        group = gold[0]["type"] if gold else "negative"
        if gold and (group not in MS_TYPES or any(item["type"] != group for item in gold)):
            continue
        key = f"msrt-model-{i:04}"
        groups[group][key] = {
            "id": key, "source": "microsoft-recognizers-text-spec", "upstream_index": i,
            "stratum": group, "text": text, "time_expressions": gold,
        }
    rows = []
    for group, cases in sorted(groups.items()):
        keys = sample(cases, "anno-independent-contract-v1", 16 if group == "negative" else 5)
        rows.extend(cases[k] for k in keys)
    return rows, {"source_indices": [r["upstream_index"] for r in rows]}, license_text


def jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="source", required=True)
    for name in PANEL_SHA256:
        command = sub.add_parser(name)
        command.add_argument("file", type=Path)
        if name == "msrt":
            command.add_argument("license", type=Path)
        command.add_argument("output", type=Path, help="a new directory for this panel")
    args = parser.parse_args()
    if args.source == "meantime":
        rows, provenance, license_text = news_panel(args.file)
    else:
        rows, provenance, license_text = contract_panel(args.file, args.license)

    panel = checked(jsonl(rows), PANEL_SHA256[args.source])
    inputs = jsonl({"id": r["id"], "text": r["text"]} for r in rows)
    receipt = {
        "source": args.source, "source_sha256": MEANTIME if args.source == "meantime" else MS_SPEC,
        "license_sha256": sha256(license_text).hexdigest(), "provenance": provenance,
        "rows": len(rows), "empty_rows": sum(not r["time_expressions"] for r in rows),
        "spans": sum(len(r["time_expressions"]) for r in rows),
        "panel_sha256": PANEL_SHA256[args.source], "inputs_sha256": sha256(inputs).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as stage:
        out = Path(stage) / "panel"
        out.mkdir()
        (out / "panel.jsonl").write_bytes(panel)
        (out / "inputs.jsonl").write_bytes(inputs)
        (out / "LICENSE.txt").write_bytes(license_text)
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        # An existing nonempty directory keeps its complete, previous run.
        out.rename(args.output)
    print(f"{args.source}: {receipt['rows']} rows; SHA-256 {receipt['panel_sha256']}")


if __name__ == "__main__":
    main()
