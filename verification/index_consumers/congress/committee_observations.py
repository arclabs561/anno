# SPDX-License-Identifier: MIT OR Apache-2.0
"""Project first-parent committee rosters into timestamped changes to printed assertions.

Input: a local git clone of unitedstates/congress-legislators (CC0).
Python 3 with PyYAML is needed only for this extraction step.
"""

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

FIRST = "adcd211cad7ab2e0fdc8bcb6380a9b02c221dafc"
LAST = "82b1eaccd2cad9e4172059507f5129fc513962af"
COMMITTEES = "committee-membership-current.yaml"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def rows(repo, revision):
    raw = yaml.load(
        git(repo, "show", f"{revision}:{COMMITTEES}"),
        Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader),
    )
    printed = defaultdict(list)
    for committee, members in raw.items():
        for member in members or ():
            printed[committee, member["name"]].append(member.get("bioguide"))
    # Preserve source order if a committee prints two identical names in one release.
    return {
        (committee, name, ordinal): target
        for (committee, name), targets in printed.items()
        for ordinal, target in enumerate(targets)
    }


def extract(repo, output):
    revisions = (
        git(
            repo,
            "rev-list",
            "--reverse",
            "--first-parent",
            f"{FIRST}..{LAST}",
            "--",
            COMMITTEES,
        )
        .decode()
        .splitlines()
    )
    previous = {}
    last_time = None
    with output.open("w", encoding="utf-8") as stream:
        for revision in [FIRST, *revisions]:
            published = datetime.fromisoformat(
                git(repo, "show", "-s", "--format=%cI", revision).decode().strip()
            ).astimezone(timezone.utc)
            if last_time is not None and published < last_time:
                raise ValueError(
                    "Commit timestamps are nonmonotone on the selected path"
                )
            last_time = published
            current = rows(repo, revision)
            upserts = []
            for key in sorted(current):
                if key not in previous or previous[key] != current[key]:
                    committee, name, ordinal = key
                    upserts.append(
                        [committee, name, ordinal, current[key], key in previous]
                    )
            removed = sorted(previous.keys() - current.keys())
            if revision == FIRST or upserts or removed:
                message = {
                    "revision": revision,
                    "published_at": published.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "upsert": upserts,
                    "remove": removed,
                }
                stream.write(
                    json.dumps(message, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
            previous = current


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.repo, args.output)
