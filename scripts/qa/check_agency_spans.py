"""Compare complete agency names in Federal Register abstracts with Anno's offline NER."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


def anchors(text, agencies):
    spans = set()
    for agency in agencies:
        for key in ("name", "raw_name"):
            name = agency.get(key)
            if name and len(name.split()) >= 2:
                pattern = r"(?<!\w)" + re.escape(name.strip()) + r"(?!\w)"
                spans.update(m.span() for m in re.finditer(pattern, text, re.IGNORECASE))
    # Nested issuing agencies can match overlapping parts of the same name.
    kept = []
    for span in sorted(spans, key=lambda s: (s[0], -s[1])):
        if not kept or span[0] >= kept[-1][1]:
            kept.append(span)
    return kept


def benchmark(binary, data):
    payload = json.loads(data)
    docs = payload["results"]
    if len({d["document_number"] for d in docs}) != len(docs) or len(docs) != payload["count"]:
        raise ValueError("expected a complete issue with unique document numbers")
    abstracts = {d["document_number"]: d["abstract"] for d in docs if d["abstract"]}
    gold = {d["document_number"]: anchors(d["abstract"], d["agencies"])
            for d in docs if d["abstract"]}
    inputs = "".join(json.dumps(dict(id=i, text=t)) + "\n" for i, t in abstracts.items())
    summary = {"input_sha256": hashlib.sha256(data).hexdigest(), "documents": len(docs),
               "with_abstract": len(abstracts), "with_anchors": sum(bool(v) for v in gold.values()),
               "anchor_count": sum(map(len, gold.values())), "models": {}}
    cases = []
    for model in ("pattern", "heuristic", "stacked"):
        run = subprocess.run([binary, "batch", "--stdin", "--model", model, "--format", "jsonl"],
                             input=inputs, text=True, capture_output=True, check=True)
        results = [json.loads(line) for line in run.stdout.splitlines()]
        if len(results) != len(abstracts) or {r["id"] for r in results} != set(abstracts):
            raise ValueError(f"{model} returned a different document panel")
        counts = Counter()
        for row in results:
            key = row["id"]
            for e in row["entities"]:
                start, end = e["start"], e["end"]
                if not (0 <= start < end <= len(abstracts[key]) and abstracts[key][start:end] == e["text"]):
                    raise ValueError(f"{model} returned an invalid source span for {key}")
            for start, end in gold[key]:
                same = [e["type"] for e in row["entities"] if (e["start"], e["end"]) == (start, end)]
                overlap = any(e["start"] < end and e["end"] > start for e in row["entities"])
                outcome = ("exact_org" if "ORG" in same else "exact_other_type" if same
                           else "overlap_only" if overlap else "miss")
                counts[outcome] += 1
                cases.append(dict(document=key, model=model, start=start, end=end,
                                  text=abstracts[key][start:end], outcome=outcome))
        summary["models"][model] = dict(counts)
    return summary, cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    parser.add_argument("json_file", type=Path)
    parser.add_argument("--cases", type=Path)
    args = parser.parse_args()
    summary, cases = benchmark(args.binary, args.json_file.read_bytes())
    if args.cases:
        args.cases.write_text("".join(json.dumps(row) + "\n" for row in cases))
    print(json.dumps(summary, indent=2))
