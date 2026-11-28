"""Normalize the pinned CI matrix and sanity artifacts into dataset-level evidence.

Pass a manifest, a directory containing ARTIFACT_ID.zip files, and an output CSV.
The manifest records default-branch push runs, including runs with no artifacts.
This script reads named outcome records. A precheck or pool selection is never a
completed score. Only explicitly named pass/rejection records can be attributed
to a particular ID. The sampler reports a capped set of rejection examples.
"""

import argparse
import csv
import hashlib
from datetime import date
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
from zipfile import ZipFile

REPO = "arclabs561/anno"
WORKFLOW = "ci.yml"
ARTIFACTS = ("eval-sanity-report", "matrix-muxer-decisions")

FIELDS = (
    "id",
    "task",
    "backend",
    "reported_backend",
    "status",
    "examples",
    "unit",
    "scope",
    "seed",
    "detail",
    "metric",
    "metric_value",
    "read_source",
    "source_digest",
    "source",
    "member",
    "record",
)


def matrix_record(row, member):
    if row.get("record_type") != "outcome":
        return None
    failed, used = row["hard_junk"], row["cost_units"]
    if type(failed) is not bool or type(used) is not int or used < 0:
        raise ValueError("Invalid hard-failure flag or evaluation cost")
    if failed:
        status, n = "Failed evaluation", ""
    elif used == 0:
        status, n = "Empty evaluation", 0
    elif row.get("primary_f1") is None:
        raise ValueError("A completed matrix cell needs a recorded score")
    else:
        status, n = "Scored", used
    return dict(
        id=row["dataset"],
        task=row["task"],
        backend=row["backend"],
        reported_backend=row.get("backend_display", ""),
        status=status,
        examples=n,
        unit="examples",
        seed=row["seed"],
        detail=row.get("fail_kind", ""),
        scope=(
            "checkout fixture"
            if "smoke/" in member or "timex-control" in member
            else "sample of loaded input"
        ),
        metric="primary_f1" if status == "Scored" else "",
        metric_value=row["primary_f1"] if status == "Scored" else "",
        read_source=row.get("dataset_source", ""),
        source_digest=row.get("dataset_sha256", ""),
    )


def matrix_rows(archive):
    # Bootstrap history belongs to earlier runs. Each current outcome is read
    # once from the same named lanes consumed by the workflow.
    members = (
        "_temp/muxer-decisions.jsonl",
        "_temp/matrix-smoke/outcomes.jsonl",
        "_temp/matrix-smoke/temporal-outcomes.jsonl",
        "_temp/matrix-timex-control.jsonl",
    )
    if "_temp/muxer-decisions.jsonl" not in archive.namelist():
        raise ValueError("Missing current-run decision journal")
    for member in members:
        if member not in archive.namelist():
            continue
        for number, line in enumerate(archive.read(member).splitlines(), 1):
            record = matrix_record(json.loads(line), member)
            if record:
                yield dict(record, member=member, record=number)
    member = "_temp/matrix-dataset-pools.jsonl"
    if member in archive.namelist():
        for number, line in enumerate(archive.read(member).splitlines(), 1):
            row = json.loads(line)
            pool = row["pool"]
            for ident in pool["selected"]:
                yield dict(
                    id=ident,
                    task=row["slice"],
                    status="Passed load check",
                    seed=row["seed"],
                    detail=pool["selected_sources"].get(ident, ""),
                    scope=row["pass"],
                    member=member,
                    record=number,
                )
            for reject in pool["rejection_examples"]:
                yield dict(
                    id=reject["dataset"],
                    task=row["slice"],
                    status="Load rejected",
                    seed=row["seed"],
                    detail=reject["reason"],
                    scope=row["pass"],
                    member=member,
                    record=number,
                )


def sanity_rows(archive):
    member = "eval-sanity-report.json"
    for i, result in enumerate(json.loads(archive.read(member))["results"]):
        provenance = result["provenance"]
        dataset, backend = provenance["dataset"], provenance["backend"]
        if type(result["success"]) is not bool:
            raise ValueError("Missing sanity-run outcome")
        n = dataset.get("sentence_count")
        status = "Scored" if result["success"] else "Failed evaluation"
        if result["success"] and (
            not result.get("metrics") or type(n) is not int or n <= 0
        ):
            raise ValueError(
                "Scored sanity run needs nonempty input and recorded metrics"
            )
        yield dict(
            id=result["dataset"],
            task=result["task"].lower(),
            backend=result["backend"],
            reported_backend=backend.get("effective", ""),
            status=status,
            examples=n if result["success"] else "",
            unit="sentences",
            scope="sanity sample",
            seed=result["seed"],
            metric="f1" if result["success"] and "f1" in result["metrics"] else "",
            metric_value=result["metrics"].get("f1", "") if result["success"] else "",
            read_source=dataset.get("data_source", ""),
            source_digest=dataset.get("content_sha256", ""),
            member=member,
            record=i,
        )


def github_page_items(endpoint, key):
    """Read every API page, including a short or empty final page."""
    items = []
    separator = "&" if "?" in endpoint else "?"
    page = 1
    while True:
        url = f"{endpoint}{separator}per_page=100&page={page}"
        response = json.loads(subprocess.check_output(["gh", "api", url]))
        batch = response[key]
        items.extend(batch)
        if len(items) == response["total_count"]:
            return items
        if len(batch) < 100 or len(items) > response["total_count"]:
            raise ValueError(f"Incomplete GitHub pagination in {key}")
        page += 1


def discover(day):
    """Record every main-branch push of ci.yml on one UTC date, including cancellations."""
    day = date.fromisoformat(day).isoformat()
    endpoint = f"repos/{REPO}/actions/workflows/{WORKFLOW}/runs?branch=main&event=push&created={day}"
    found = github_page_items(endpoint, "workflow_runs")
    runs = []
    for run in sorted(found, key=lambda r: r["id"]):
        if run["status"] != "completed" or run["created_at"][:10] != day:
            raise ValueError(
                "The requested day contains an unfinished or mismatched CI run"
            )
        published = github_page_items(
            f"repos/{REPO}/actions/runs/{run['id']}/artifacts", "artifacts"
        )
        artifacts = []
        for item in sorted(published, key=lambda a: (a["name"], a["id"])):
            if item["name"] in ARTIFACTS:
                digest = (item.get("digest") or "").removeprefix("sha256:")
                if not re.fullmatch("[0-9a-f]{64}", digest):
                    raise ValueError(
                        f"GitHub did not supply a SHA-256 for {item['id']}"
                    )
                artifacts.append(dict(id=item["id"], name=item["name"], sha256=digest))
        runs.append(
            dict(
                run=run["id"],
                commit=run["head_sha"],
                conclusion=run["conclusion"],
                artifacts=artifacts,
                created_utc=run["created_at"],
            )
        )
    return dict(
        repository=REPO,
        workflow=WORKFLOW,
        event="push",
        branch="main",
        day_utc=day,
        artifact_names=list(ARTIFACTS),
        runs=runs,
    )


def collect(manifest, directory, *, download=False):
    seen_runs, seen_artifacts = set(), set()
    for run in sorted(manifest["runs"], key=lambda r: r["run"]):
        if run["run"] in seen_runs:
            raise ValueError("Duplicate CI run")
        seen_runs.add(run["run"])
        for artifact in sorted(run["artifacts"], key=lambda a: a["id"]):
            ident = artifact["id"]
            if ident in seen_artifacts:
                raise ValueError("Duplicate CI artifact")
            seen_artifacts.add(ident)
            path = directory / f"{ident}.zip"
            if download and not path.exists():
                request = subprocess.run(
                    ["gh", "api", f"repos/{REPO}/actions/artifacts/{ident}/zip"],
                    capture_output=True,
                )
                if request.returncode:
                    # gh may put a temporary signed archive URL in its stderr.
                    raise RuntimeError(
                        f"Could not download Actions artifact {ident} with gh. Supply the original {ident}.zip in {directory} to verify it offline."
                    )
                data = request.stdout
            else:
                data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
                raise ValueError(
                    f"CI artifact {ident} does not match the recorded digest"
                )
            if download and not path.exists():
                path.write_bytes(data)
            readers = {
                "matrix-muxer-decisions": matrix_rows,
                "eval-sanity-report": sanity_rows,
            }
            with ZipFile(BytesIO(data)) as archive:
                for row in readers[artifact["name"]](archive):
                    url = f"https://github.com/arclabs561/anno/actions/runs/{run['run']}/artifacts/{ident}"
                    yield dict(row, source=url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("archives", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--discover",
        metavar="YYYY-MM-DD",
        help="Find every CI push on a UTC day and save a new manifest after validating its archives",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Use the authenticated gh CLI to fetch missing archives",
    )
    args = parser.parse_args()
    manifest = (
        discover(args.discover)
        if args.discover
        else json.loads(args.manifest.read_text())
    )
    if args.download:
        args.archives.mkdir(parents=True, exist_ok=True)
    normalized = list(collect(manifest, args.archives, download=args.download))
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(normalized)
    if args.discover:
        args.manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
