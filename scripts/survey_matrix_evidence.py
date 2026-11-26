"""Extract observed prechecks from archived main-branch CI logs using authenticated gh.

Example: python scripts/survey_matrix_evidence.py --limit 100 --output evidence.csv
Use --run multiple times to pin runs. --cache-dir retains downloaded log archives
locally for replay. The CSV contains observations only. Unavailable/expired logs
and runs without structured prechecks are reported separately on stderr.
"""

import argparse
import csv
import io
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "arclabs561/anno"
FIELDS = [
    "run_id",
    "head_sha",
    "observed_at",
    "dataset",
    "source",
    "dataset_sha256",
    "s3_fallback",
    "run_url",
]


def gh(*args):
    # Errors after GitHub redirects may include signed URLs. Keep those local.
    result = subprocess.run(["gh", *args], capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError("GitHub request failed")
    return result.stdout


def observations(run, cache_dir):
    run_id = run["databaseId"]
    archive = cache_dir / f"{run_id}.zip" if cache_dir else None
    try:
        if archive and archive.is_file():
            data = archive.read_bytes()
        else:
            data = gh("api", f"repos/{REPO}/actions/runs/{run_id}/logs")
            if archive:
                archive.write_bytes(data)
        rows = []
        with zipfile.ZipFile(io.BytesIO(data)) as logs:
            # The top-level job log contains steps; per-step files may duplicate them.
            matches = [
                name
                for name in logs.namelist()
                if re.fullmatch(
                    r"\d+_Randomized Matrix Test \(trusted branch\)\.txt", name
                )
            ]
            if not matches:
                return [], "no trusted matrix job log"
            for line in logs.read(matches[0]).decode("utf-8-sig").splitlines():
                match = re.fullmatch(
                    r"(\S+) matrix-muxer: dataset precheck (\{.*\})", line
                )
                if match:
                    entry = json.loads(match[2])
                    rows.append(
                        [
                            run_id,
                            run["headSha"],
                            match[1],
                            entry["dataset"],
                            entry["dataset_source"],
                            entry.get("dataset_sha256"),
                            entry.get("s3_fallback"),
                            run["url"],
                        ]
                    )
        return rows, "observed" if rows else "no structured prechecks"
    except (RuntimeError, subprocess.TimeoutExpired, zipfile.BadZipFile):
        return [], "logs unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, action="append")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    fields = "databaseId,headSha,headBranch,url,workflowName,conclusion"
    if args.run:
        runs = [
            json.loads(gh("run", "view", str(run), "-R", REPO, "--json", fields))
            for run in args.run
        ]
    else:
        runs = json.loads(
            gh(
                "run",
                "list",
                "-R",
                REPO,
                "--workflow",
                "ci.yml",
                "--branch",
                "main",
                "--status",
                "success",
                "--limit",
                str(args.limit),
                "--json",
                fields,
            )
        )
    if any(
        run["headBranch"] != "main"
        or run["workflowName"] != "CI"
        or run["conclusion"] != "success"
        for run in runs
    ):
        parser.error("Expected successful main-branch CI runs")
    if args.cache_dir:
        args.cache_dir.mkdir(exist_ok=True, parents=True)
    with ThreadPoolExecutor(5) as pool:
        results = list(pool.map(lambda run: observations(run, args.cache_dir), runs))
    print(dict(Counter(status for _, status in results)), file=sys.stderr)
    for run, (_, status) in zip(runs, results):
        if status != "observed":
            print(run["databaseId"], status, file=sys.stderr)
    rows = sorted(row for result, _ in results for row in result)
    if not rows:
        raise SystemExit("No structured prechecks found")
    with args.output.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
