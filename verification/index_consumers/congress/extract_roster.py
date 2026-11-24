# SPDX-License-Identifier: MIT OR Apache-2.0
"""Project pinned CC0 current-roster editions to original, compact JSONL publications."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

BASE = "adcd211cad7ab2e0fdc8bcb6380a9b02c221dafc"
LAST = "82b1eaccd2cad9e4172059507f5129fc513962af"
FILE = "legislators-current.yaml"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def projected_rows(data):
    rows = {}
    missing_ids = 0
    for row in data:
        record_id = row["id"].get("bioguide")
        if not record_id:
            missing_ids += 1
            continue
        if record_id in rows:
            raise ValueError(f"duplicate Bioguide ID {record_id}")
        term = row["terms"][-1]
        name = row["name"]
        rows[record_id] = {
            "id": record_id,
            "official_full": name.get("official_full"),
            "first": name["first"],
            "last": name["last"],
            "state": term["state"],
            "chamber": term["type"],
        }
    return rows, missing_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="local congress-legislators Git repository")
    parser.add_argument("output", type=Path, help="destination JSONL file")
    args = parser.parse_args()
    git(args.repo, "merge-base", "--is-ancestor", BASE, LAST)
    patches = git(args.repo, "log", "--first-parent", "--reverse", "--format=%H",
                  f"{BASE}..{LAST}", "--", FILE).decode().splitlines()
    old = {}
    total_rows = 0
    missing_ids = 0
    publications = []
    previous_time = ""
    for revision in [BASE, *patches]:
        epoch = int(git(args.repo, "show", "-s", "--format=%ct", revision))
        timestamp = dt.datetime.fromtimestamp(epoch, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if timestamp < previous_time:
            raise ValueError(f"non-monotone first-parent commit time at {revision}")
        previous_time = timestamp
        # CSafeLoader and SafeLoader decode the same YAML subset. The C loader is faster.
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        data = yaml.load(git(args.repo, "show", f"{revision}:{FILE}"), Loader=loader)
        new, skipped = projected_rows(data)
        total_rows += len(data)
        missing_ids += skipped
        upsert = [row for key, row in sorted(new.items()) if row != old.get(key)]
        remove = sorted(old.keys() - new.keys())
        if upsert or remove:
            publications.append({"revision": revision, "published_at": timestamp,
                                 "upsert": upsert, "remove": remove})
        old = new
    output = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                      for row in publications)
    args.output.write_bytes(output)
    print(json.dumps({"editions_examined": 1 + len(patches), "publications": len(publications),
                      "projected_rows_seen": total_rows - missing_ids,
                      "rows_skipped_without_bioguide": missing_ids,
                      "sha256": hashlib.sha256(output).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
