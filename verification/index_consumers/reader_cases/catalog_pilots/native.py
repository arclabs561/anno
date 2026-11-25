"""Export the scoped MusicBrainz observations to anno's existing two-reader suite."""

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from reader import SAMPLE


def export(panel):
    cases = []
    for key, record in panel["musicbrainz"].items():
        source = {"id": key, "edition": "api-observation-2026-11-25"}
        claims = []
        questions = []

        def ask(label, kind, **fields):
            questions.append({"id": label, "kind": kind, "at": 1, "policy": 1, "audience": "public", **fields})

        billing = "".join(c["credited"] + c["join"] for c in record.get("billing", []))
        lines = [billing] if billing else []
        if billing:
            claims.append({"id": "billing", "kind": "printed", "target": key, "start": 0, "end": len(billing)})
        ask("billing", "printed")
        for i, row in enumerate(record["roles"]):
            lines.append(row["role"] + ": " + row["catalog_name"])
            claims.append({"id": f"relationship-{i}", "kind": "credit", "target": row["artist"], "role": row["role"]})
        for role in ("producer", "writer", "lyricist"):
            ask(role, "credit", role=role)
        for claim in claims:
            claim.update(source=key, issuer="musicbrainz")
        source["text"] = "\n".join(lines)
        cases.append({
            "id": key, "work": key, "sources": [source], "steps": [{"revision": 1, "put": claims}],
            "policies": [{"revision": 1, "hidden": []}], "queries": questions,
        })
    return {"version": 1, "cases": cases}


def verify(suite, anno, rust_reader):
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    sys.path.insert(0, str(here.parents[1] / "congress"))
    from check import build_rust_reader, call_json, compare, read_suite
    from check_catalog_history import build_cli

    key = json.loads((here / "native_key.json").read_bytes())
    compare("MusicBrainz SQLite + anno CLI", read_suite(suite, anno or build_cli()), key)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "cases.json"
        path.write_text(json.dumps(suite, ensure_ascii=False), encoding="utf-8")
        compare("MusicBrainz Rust", call_json([rust_reader or build_rust_reader(), str(path)]), key)
    total = sum(len(rows) for rows in key["answers"].values())
    print(f"{total} MusicBrainz questions matched SQLite + anno CLI and Rust")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="score both anno readers against the fixed key")
    parser.add_argument("--anno", type=Path, help="existing anno CLI")
    parser.add_argument("--rust-reader", type=Path, help="existing reader_answer example")
    args = parser.parse_args()
    suite = export(json.loads(SAMPLE.read_bytes()))
    if args.verify:
        verify(suite, args.anno, args.rust_reader)
    else:
        print(json.dumps(suite, ensure_ascii=False))
