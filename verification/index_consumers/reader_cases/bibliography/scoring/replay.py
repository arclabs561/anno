# SPDX-License-Identifier: MIT OR Apache-2.0
"""Re-score the public receipt and verify that decompression recovers the locked predictions."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from .common import ARMS, CUTS, DELIVERIES, ROOT, file_hash, packed, require
from .score import score


RECEIPTS = ROOT / "receipts"
ORIGINAL_KEY = "a85997443eca7a35c7ca3c56440aade47894d822b6ab66dddc4aa180edf022d5"
PUBLIC_KEY = "69a94864a70641168be407b1e032761081df19daa39c6d3521af25d50b9a633d"
ORIGINAL_LOCK = "6463b95febce6b63a3804d1c7b3d4642fa1751d5a2948cf0ab42887735ef1603"
KEY_FIELDS = ("notice", "work", "cut", "notice_day", "edition", "version_date", "edition_status",
              "byline", "printed_byline", "status", "locator", "pdf_url")
SOURCE_FIELDS = ("role", "url", "sha256", "location", "locator", "page", "layout_lines", "license",
                 "commit", "commit_time", "byline_equal", "xml_equal")


def public_key(original):
    rows = []
    for r in original["rows"]:
        row = {f: r[f] for f in KEY_FIELDS}
        if r["status"] != "verified":
            row["reason"] = r["reason"]
        row["sources"] = [{f: s[f] for f in SOURCE_FIELDS if f in s} for s in r["sources"]]
        rows.append(row)
    return {"original_reference_sha256": ORIGINAL_KEY, "pdf_transcription_sha256": original["pdf_transcription_sha256"],
            "rows": rows}


def compact(original):
    dimension = {"notice", "target", "cut", "arm", "delivery"}
    values = []
    ids = {}
    assignments = {}
    for row in original["rows"]:
        result = {k: v for k, v in row.items() if k not in dimension}
        signature = packed(result)
        if signature not in ids:
            ids[signature] = len(values)
            values.append(result)
        key = (row["notice"], row["cut"], row["arm"], row["delivery"])
        require(key not in assignments, "duplicate original prediction")
        assignments[key] = ids[signature]
    groups = {}
    for arm in ARMS:
        groups[arm] = {
            d: [assignments[r["notice"], cut, arm, d] for r in original["frame"] for cut in CUTS]
            for d in DELIVERIES}
    return {"frame": original["frame"], "answers": values, "assignment": groups}


def expand(receipt):
    rows = []
    frame = receipt["frame"]
    cases = [(row, cut) for row in frame for cut in CUTS]
    for arm in ARMS:
        for d in DELIVERIES:
            for (row, cut), index in zip(cases, receipt["assignment"][arm][d], strict=True):
                require(type(index) is int and 0 <= index < len(receipt["answers"]), "invalid answer index")
                rows.append({"notice": row["notice"], "target": row["target"], "cut": cut, "arm": arm,
                             "delivery": d, **receipt["answers"][index]})
    return {"frame": frame, "rows": rows}


def replay(original=None):
    lock_path = RECEIPTS / "lock.json"
    require(file_hash(lock_path) == ORIGINAL_LOCK, "wrong published prediction lock")
    lock = json.loads(lock_path.read_bytes())
    # The original study used the archived reader. New prediction runs hash the
    # current reader, whose native projection now identifies the XML source.
    for name, expected in lock["code"].items():
        source = RECEIPTS / "reader.py.txt" if name == "../reader.py" else ROOT / name
        require(file_hash(source) == expected, f"frozen study source changed: {name}")
    predictions = expand(json.loads((RECEIPTS / "predictions.json").read_bytes()))
    digest = sha256(packed(predictions) + b"\n").hexdigest()
    require(digest == lock["predictions_sha256"], "reconstructed predictions differ from original")
    public_path = RECEIPTS / "reference.json"
    require(file_hash(public_path) == PUBLIC_KEY, "public reference reduction changed")
    reference = json.loads(public_path.read_bytes())
    require(reference["original_reference_sha256"] == ORIGINAL_KEY, "wrong original reference identifier")
    if original:
        require(file_hash(original) == ORIGINAL_KEY, "original reference differs from its pre-unblinding digest")
        require(public_key(json.loads(original.read_bytes())) == reference, "public reduction differs")
    report = score(predictions, reference)
    report.update(reference_sha256=ORIGINAL_KEY, prediction_lock_sha256=ORIGINAL_LOCK)
    require(report == json.loads((RECEIPTS / "official.json").read_bytes()), "offline and official scores differ")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-key", type=Path, help="also verify the public reduction against the sealed bytes")
    args = parser.parse_args()
    report = replay(args.original_key)
    print(f"{report['queries']} queries, {report['delivery_pairs']} delivery pairs, "
          f"{len(report['delivery_mismatches'])} delivery mismatches. Offline score equals official score.")


if __name__ == "__main__":
    main()
