"""Export a versioned catalog of dataset evidence and native charts for offline queries."""

import json
from collections import defaultdict

from figure_coverage_report import rows as csv_rows

NAME = "figure-catalog.json"
SCHEMA = "anno.figure-catalog/v1"


def studies(evidence, native_inputs):
    descriptors = csv_rows(evidence / "figure-studies.csv")
    guides = {row["study"]: row for row in descriptors}
    if len(guides) != len(descriptors):
        raise ValueError("Duplicate study ID in figure-studies.csv")
    for path in sorted(native_inputs):
        descriptor = guides.get(path.stem, {})
        yield {
            "id": path.stem,
            "datasets": [value for value in descriptor.get("datasets", "").split("|") if value],
            "guide": descriptor.get("guide", ""),
            "report": json.loads(path.read_text(encoding="utf-8")),
        }


def datasets(rendered):
    coverage = csv_rows(rendered / "dataset-coverage.csv")
    by_id = defaultdict(list)
    for row in csv_rows(rendered / "dataset-evaluation-coverage.csv"):
        by_id[row["id"]].append(row)
    if set(by_id) != {r["id"] for r in coverage} or len(coverage) != len(by_id):
        raise ValueError("Dataset IDs in the registry coverage and evaluation index disagree")
    metadata = {"id", "domain", "measurement", "declared_tasks", "declared_splits"}
    for source in sorted(coverage, key=lambda r: r["id"]):
        rows = by_id[source["id"]]
        first = rows[0]
        records = []
        for row in rows:
            if any(row[key] != first[key] for key in metadata):
                raise ValueError(f"Inconsistent catalog metadata for {source['id']}")
            if row["status"] == "No indexed evaluation":
                if len(rows) != 1:
                    raise ValueError(f"Missing and observed outcomes coexist for {source['id']}")
                continue
            record = {k: v for k, v in row.items() if k not in metadata and v}
            if "examples" in record:
                record["examples"] = int(record["examples"])
            if "metric_value" in record:
                record["metric_value"] = float(record["metric_value"])
            records.append(record)
        yield {
            "id": source["id"],
            "name": source["name"],
            "domain": source["domain"],
            "measurement": first["measurement"],
            "coverage_action": source["coverage_action"],
            "declared_tasks": first["declared_tasks"].split("|") if first["declared_tasks"] else [],
            "declared_splits": first["declared_splits"].split("|") if first["declared_splits"] else [],
            "records": records,
        }


def write_catalog(evidence, rendered, native_inputs):
    survey = json.loads((evidence / "dataset-ci-survey.json").read_text(encoding="utf-8"))
    catalog = {
        "schema": SCHEMA,
        "ci_survey_day": survey["day_utc"],
        "datasets": list(datasets(rendered)),
        "studies": list(studies(evidence, native_inputs)),
    }
    (rendered / NAME).write_text(
        json.dumps(catalog, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
