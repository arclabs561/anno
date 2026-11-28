"""Replay the 50-document REDACT comparison through the CONTACT-only privacy CLI."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

DATA_REVISION = "252232ac29669beeb99cb6474eeb2736761be508"
DATA_SHA256 = "754f562ce9c3c5b7b663f1ba6127c38a31d93f5631d38364341cb727936234c6"
PANEL_SHA256 = "7e7236d30e42fdaadf6e4adaab9f70055535999c93c8145bba70ed4fc9e5ae87"
MODEL_REVISION = "e594898629d452e8311796f5f329c7edbeda907c"
MANIFEST_SHA256 = "457d8e3e252cf09f1f2c6a926a696c8c2094afd00a8deaa8e0d32367bd6fdca6"
CONTACT_LABELS = {
    "Work_Email_Address", "Personal_Email_Address",
    "Telephone_Numbers_Personal", "Telephone_Numbers_Work",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def select_panel(data):
    if sha256(data) != DATA_SHA256:
        raise ValueError("unexpected REDACT file digest")
    rows = json.loads(data)
    locales = defaultdict(list)
    for i, row in enumerate(rows):
        if len(row["text"]) <= 900:
            locales[row["axes"]["language"]].append(i)

    # The earlier 75-document panel took three per locale by hashed record ID.
    # Source order broke ties among duplicate IDs. Exclude every duplicate text.
    seen = set()
    for indices in locales.values():
        earlier = sorted(indices, key=lambda i: sha256(str(rows[i]["record_id"]).encode()))[:3]
        seen.update(rows[i]["text"] for i in earlier)
    selected = []
    for locale in sorted(locales):
        ranked = sorted(locales[locale], key=lambda i: sha256(f"anno-contact-default-v1/{i}".encode()))
        count = 0
        for i in ranked:
            text = rows[i]["text"]
            if text not in seen:
                selected.append(i)
                seen.add(text)
                count += 1
                if count == 2:
                    break
    if sha256(json.dumps(selected).encode()) != PANEL_SHA256:
        raise ValueError("ordered panel fingerprint differs from the frozen cohort")
    return [rows[i] for i in selected]


def verify_checkpoint(root):
    # Hash the nine inference assets from the published fp32_v2 export.
    files = list((root / "fp32_v2").glob("*_fp32.onnx")) + [root / "fp32_v2/tokenizer.json"]
    hashes = {}
    for path in files:
        with path.open("rb") as stream:
            hashes[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha256(json.dumps(hashes, sort_keys=True).encode()) != MANIFEST_SHA256:
        raise ValueError("expected the nine files from the pinned FP32 export")


def report_spans(report, size):
    spans = {(int(a), int(b)) for a, b in re.findall(
        r'" @(\d+):(\d+) \[(?:HIGH|MEDIUM|LOW|CRITICAL)\]$', report, re.MULTILINE
    )}
    count = re.search(r"^  CONTACT:\s+(\d+) emails/phones$", report, re.MULTILINE)
    if not count or len(spans) != int(count[1]) or any(not 0 <= a < b <= size for a, b in spans):
        raise ValueError("could not verify CONTACT spans against the CLI report")
    return spans


def mask(spans):
    return {pos for a, b in spans for pos in range(a, b)}


def redacted(text, spans):
    merged = []
    for a, b in sorted(spans):
        if merged and a < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    for number, (a, b) in enumerate(reversed(merged), start=1):
        text = text[:a] + f"[CONTACT_{number}]" + text[b:]
    return text


def score(gold, found):
    expected, actual = mask(gold), mask(found)
    lengths = [sum(i in actual for i in range(a, b)) for a, b in sorted(gold)]
    full = sum(n == b - a for (a, b), n in zip(sorted(gold), lengths))
    missed = lengths.count(0)
    return Counter(
        tp=len(gold & found), fp=len(found - gold), fn=len(gold - found),
        fully_masked=full, partially_masked=len(gold) - full - missed,
        entirely_missed=missed, masked_contact_characters=len(expected & actual),
        unmasked_contact_characters=len(expected - actual), masked_noncontact_characters=len(actual - expected),
    )


def replay(index, row, routes):
    text = row["text"]
    gold = set()
    for entity in row["entities"]:
        if entity["disclosed"] and entity["entity_type"] in CONTACT_LABELS:
            start, end = entity["start"], entity["end"]
            if text[start:end] != entity["entity_string"]:
                raise ValueError(f"invalid gold offsets at cohort position {index}")
            gold.add((start, end))

    def run(route, action):
        cli, env = routes[route]
        proc = subprocess.run(
            [cli, "--color", "never", "privacy", "--types", "CONTACT", "--action", action, "--text=" + text],
            env=env, capture_output=True, text=True, encoding="utf-8", timeout=180,
        )
        if proc.returncode or "prepared GLiNER2-PII failed" in proc.stderr:
            raise RuntimeError(f"{route} {action} failed at cohort position {index}")
        return proc.stdout

    spans = {name: report_spans(run(name, "report"), len(text)) for name in routes}
    scores = {name: score(gold, found) for name, found in spans.items()}
    for name, found in spans.items():
        if index % 10 == 0 or (name != "fallback" and scores["prepared"]["partially_masked"]):
            if run(name, "redact") != redacted(text, found) + "\n":
                raise ValueError(f"{name} redaction differs from its reported spans at position {index}")
            scores[name]["redact_output_checks"] += 1
    paired = Counter(
        fallback_matches_previous=spans["fallback"] == spans["previous"],
        prepared_preserves_previous_mask=mask(spans["previous"]) <= mask(spans["prepared"]),
    )
    return scores, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--before-cli", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    panel = select_panel(args.data.read_bytes())
    verify_checkpoint(args.model_dir)
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, ANNO_NO_DOWNLOADS="1", HF_HOME=tmp, GLINER2_PII_MODEL_DIR=str(Path(tmp) / "absent"))
        routes = {
            "previous": (args.before_cli.resolve(), env),
            "fallback": (args.cli.resolve(), env),
            "prepared": (args.cli.resolve(), dict(env, GLINER2_PII_MODEL_DIR=str(args.model_dir.resolve()))),
        }
        totals = {name: Counter() for name in routes}
        comparisons = Counter()
        with ThreadPoolExecutor(args.jobs) as pool:
            for scores, paired in pool.map(lambda item: replay(*item, routes), enumerate(panel)):
                for name, counts in scores.items():
                    totals[name].update(counts)
                comparisons.update(paired)
    print(json.dumps({
        "dataset_revision": DATA_REVISION, "dataset_sha256": DATA_SHA256,
        "panel_sha256": PANEL_SHA256, "model_revision": MODEL_REVISION, "fp32_manifest_sha256": MANIFEST_SHA256,
        "documents": len(panel), "locales": len({row["axes"]["language"] for row in panel}),
        "routes": totals, "paired_documents": comparisons,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
