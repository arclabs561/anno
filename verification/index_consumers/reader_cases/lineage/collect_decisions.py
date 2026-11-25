# SPDX-License-Identifier: MIT OR Apache-2.0
"""Refresh the small factual decision manifest from public ROR curation issues."""

import argparse
import csv
import io
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

ROOT = "https://api.github.com"
REPO = "ror-community/ror-updates"
ROR = r"https://ror\.org/(0[a-z0-9]{8})\b"


def get(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "anno-ror-reader-study", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=60) as reply:
        return json.load(reply)


def extract(issue):
    if issue.get("state_reason") != "completed":
        return None
    body = issue["body"] or ""
    old = re.search(r"^ROR ID:\s*" + ROR, body, re.I | re.M)
    description = re.search(r"^Description of change:\s*([^\n]+)", body, re.I | re.M)
    if not old or not description or "status.replace==withdrawn" not in body.lower():
        return None
    target = re.search(r"\bduplicate of\b[^.\n]{0,100}?" + ROR, description[1], re.I)
    if not target or not re.search(
        r"https://ror\.org/" + target[1] + r"\s*\(successor(?:-np)?\)", body, re.I
    ):
        return None
    return issue["milestone"]["title"], old[1], target[1], issue["number"]


def collect():
    # These two published examples were identified before choosing the later cut.
    for number in [29292, 36233]:
        row = extract(get(f"{ROOT}/repos/{REPO}/issues/{number}"))
        if row is None:
            raise ValueError(f"could not verify development issue {number}")
        yield row
    for minor in range(11, 17):
        query = f'repo:{REPO} is:issue is:closed milestone:v2.{minor} "duplicate of" in:body'
        issues = []
        for page in range(1, 11):
            params = urllib.parse.urlencode({"q": query, "per_page": 100, "page": page})
            batch = get(f"{ROOT}/search/issues?{params}")
            if batch["incomplete_results"] or batch["total_count"] > 1000:
                raise ValueError(f"incomplete issue search for v2.{minor}")
            issues.extend(batch["items"])
            if len(issues) >= batch["total_count"]:
                break
        for issue in sorted(issues, key=lambda item: item["number"]):
            if row := extract(issue):
                yield row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["version", "source", "target", "issue"])
    writer.writerows(collect())
    if args.out:
        args.out.write_text(buffer.getvalue())
    else:
        print(buffer.getvalue(), end="")
