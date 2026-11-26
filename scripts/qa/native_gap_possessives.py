"""Compare two frozen native GAP trace test executables on unseen validation article URLs."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import gap_receipts
import subprocess
import tempfile

from coref_gap import DATA_SHA, REVISION, check_file, measure

VALIDATION_SHA = "2d784f66b390404f554704b9aef6dcde8845e79dda9886b8391cf7e9a24fdb98"


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def score(executable, docs, temp):
    raw = temp / "trace.jsonl"
    raw.unlink(missing_ok=True)
    subprocess.run(
        [executable.resolve(), "gap_trace::trace_native_gap", "--ignored"],
        check=True,
        env={
            **os.environ,
            "ANNO_GAP_TRACE_INPUT": str(temp / "input.jsonl"),
            "ANNO_GAP_TRACE_OUTPUT": str(raw),
        },
    )
    traces = [json.loads(line) for line in raw.read_text().splitlines()]
    pred = [
        dict(id=t["id"], producer="mention-ranking", clusters=t["clusters"])
        for t in traces
    ]
    measured = measure(docs, pred, "mention-ranking")
    proposals = []
    for (_, _, targets, labels), t in zip(docs, traces, strict=True):
        offered = set(map(tuple, t["proposed"]))
        proposals.append("".join("1" if s in offered else "0" for s in targets))
    measured["proposed_both_gold"] = sum(
        gold and mask[0] == mask[i + 1] == "1"
        for doc, mask in zip(docs, proposals, strict=True)
        for i, gold in enumerate(doc[3])
    )
    return measured, proposals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "test_tsv",
        "validation_tsv",
        "baseline_test",
        "candidate_test",
        "output",
    ]:
        p.add_argument(name, type=Path)
    p.add_argument("--local-details", type=Path, help="optional row-level replay file outside the repository")
    a = p.parse_args()
    check_file(a.test_tsv, DATA_SHA)
    check_file(a.validation_tsv, VALIDATION_SHA)
    test_articles = {r["URL"] for r in read_rows(a.test_tsv)}
    rows = read_rows(a.validation_tsv)
    docs = []
    urls = set()
    for r in rows:
        if r["URL"] in test_articles:
            continue
        urls.add(r["URL"])
        text = r["Text"]
        spans = []
        for key in ["Pronoun", "A", "B"]:
            start = int(r[key + "-offset"])
            end = start + len(r[key])
            if text[start:end] != r[key]:
                raise ValueError(f"wrong source boundary in {r['ID']}")
            spans.append((start, end))
        docs.append(
            (r["ID"], text, spans, [r["A-coref"] == "TRUE", r["B-coref"] == "TRUE"])
        )
    with tempfile.TemporaryDirectory() as d:
        temp = Path(d)
        (temp / "input.jsonl").write_text(
            "".join(json.dumps(dict(id=i, text=t)) + "\n" for i, t, *_ in docs)
        )
        before, bp = score(a.baseline_test, docs, temp)
        after, ap = score(a.candidate_test, docs, temp)
    results = []
    for (i, text, _, _), x, y, s, t in zip(
        docs, bp, ap, before.pop("per_id"), after.pop("per_id"), strict=True
    ):
        if x != y or s != t:
            results.append(
                dict(
                    id=i,
                    proposed_before=x,
                    proposed_after=y,
                    output_before=s,
                    output_after=t,
                )
            )
    out = dict(
        revision=REVISION,
        validation_sha256=VALIDATION_SHA,
        test_sha256=DATA_SHA,
        removed_shared_test_url_rows=len(rows) - len(docs),
        distinct_urls=len(urls),
        ids_sha256=hashlib.sha256(
            "".join(d[0] + "\n" for d in docs).encode()
        ).hexdigest(),
        baseline=before,
        candidate=after,
        changed_target_rows=results,
    )
    gap_receipts.write(a.output, gap_receipts.possessives(out), out, a.local_details)


if __name__ == "__main__":
    main()
