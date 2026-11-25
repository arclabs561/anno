# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate and lock blind bibliography predictions without a reference key."""
import argparse
from datetime import date
import json
from pathlib import Path
import unicodedata

from ..reader import read_one, read_bulk
from .common import ARMS, CUTS, doi, file_hash, file_map, fingerprint, load_rows, packed, require


def frame_rows(frame):
    seen = set()
    result = []
    for case in load_rows(frame):
        notice = doi(case["notice"])
        require(notice not in seen, f"duplicate notice {notice}")
        seen.add(notice)
        day = date.fromisoformat(case["day"]).isoformat()
        result.append({"notice": notice, "target": doi(case["target"]), "day": day})
    return result


def summarize(response):
    status = response["status"]
    if status == "unavailable":
        require(bool(response.get("reason")), "unavailable without reason")
        return {"status": status, "reason": response["reason"], "edition": None,
                "byline": None, "printed_byline": None, "evidence": None}
    require(status == "available", "unknown reader response")
    names = response["byline"]
    require(isinstance(names, list) and names, "empty byline")
    printed = [" ".join(unicodedata.normalize("NFC", " ".join(v for _, v in name)).split())
               for name in names]
    require(all(printed), "empty printed author")
    evidence = {k: response[k] for k in ("published_on", "source_url", "source_sha256", "license")}
    return {"status": status, "edition": response["edition"], "byline": fingerprint(names),
            "printed_byline": fingerprint(printed), "evidence": evidence}


def query(case, cut, arm, work):
    target, day = case["target"], case["day"]
    if arm == "mutable":
        # For an absent work the reader returns unavailable for the otherwise valid query.
        return {"doi": target, "cut": "as_of", "date": (work or {}).get("through") or day}
    if cut == "first":
        return {"doi": target, "cut": "first"}
    return {"doi": target, "cut": "before" if cut == "pre" else "as_of", "date": day}


def generate(frame, works):
    cases = frame_rows(frame)
    by_doi = {doi(w["doi"]): w for w in works}
    require(len(by_doi) == len(works), "duplicate work DOI")
    rows = []
    for arm in ARMS:
        ids = [(case, cut) for case in cases for cut in CUTS]
        asks = [query(case, cut, arm, by_doi.get(case["target"])) for case, cut in ids]
        batches = {"online": [read_one(by_doi.get(q["doi"]), q) for q in asks],
                   "bulk": read_bulk(works, asks)}
        for delivery, answers in batches.items():
            require(len(answers) == len(asks), "reader changed query cardinality")
            for (case, cut), ask, answer in zip(ids, asks, answers, strict=True):
                require(doi(answer["doi"]) == ask["doi"] and answer["cut"] == ask["cut"]
                        and answer.get("date") == ask.get("date"), "reader changed query order or identity")
                rows.append({"notice": case["notice"], "target": case["target"], "cut": cut,
                             "arm": arm, "delivery": delivery, **summarize(answer)})
    return {"frame": cases, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--works", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="new directory for predictions and lock")
    args = parser.parse_args()
    frame = json.loads(args.frame.read_bytes())
    works = json.loads(args.works.read_bytes())["works"]
    predictions = generate(frame, works)
    require(len(predictions["frame"]) == 17, "expected 17 selected notices")
    args.out.mkdir(parents=True, exist_ok=False)
    output = args.out / "predictions.json"
    output.write_bytes(packed(predictions) + b"\n")
    lock = {"predictions_sha256": file_hash(output), "frame_sha256": file_hash(args.frame),
            "works_sha256": file_hash(args.works), "code": file_map()}
    (args.out / "lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print("Predictions SHA-256", lock["predictions_sha256"])
    print("Lock SHA-256", file_hash(args.out / "lock.json"))


if __name__ == "__main__":
    main()
