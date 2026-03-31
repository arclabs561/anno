#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Audit dataset registry URLs against S3 cache and live HTTP status.

Usage:
    uv run scripts/audit_dataset_health.py [--fix-urls] [--sync-s3]

Reads: scripts/registry_enrichment.json (health results from prior run)
Writes: scripts/dataset_audit_report.json

Does NOT modify the Rust registry -- that requires manual edits.
"""
import json
import sys
from pathlib import Path

ENRICHMENT = Path("scripts/registry_enrichment.json")
S3_INVENTORY = Path("/tmp/s3-inventory.txt")
REPORT_OUT = Path("scripts/dataset_audit_report.json")


def normalize(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "").replace(" ", "")


def load_s3_datasets() -> set[str]:
    """Parse S3 inventory file into a set of normalized dataset names."""
    if not S3_INVENTORY.exists():
        print("Warning: /tmp/s3-inventory.txt not found. Run: aws s3 ls s3://arc-anno-data/ --recursive > /tmp/s3-inventory.txt")
        return set()
    names = set()
    for line in S3_INVENTORY.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        path = parts[-1]
        if path.startswith("datasets/"):
            name = path.removeprefix("datasets/")
            # Strip extensions
            for ext in (".cache", ".json", ".txt", ".conll", ".bio", ".tsv", ".csv",
                        ".xml", ".tar.gz", ".rar", ".zip", ".parquet", ".iob2", ".ann"):
                name = name.removesuffix(ext)
            # Strip manifest/latest suffixes
            for suffix in (".latest", ".manifest"):
                name = name.removesuffix(suffix)
            if name and name != ".gitkeep" and not name.startswith("by-sha256"):
                names.add(normalize(name))
                names.add(name)  # Keep original casing too
    return names


def main():
    if not ENRICHMENT.exists():
        print(f"Error: {ENRICHMENT} not found")
        sys.exit(1)

    data = json.load(ENRICHMENT.open())
    health = data.get("health_results", {})
    s3_datasets = load_s3_datasets()

    report = {
        "summary": {},
        "healthy_cached": [],
        "healthy_not_cached": [],
        "broken_cached": [],
        "broken_not_cached": [],
    }

    for name, checks in health.items():
        primary = checks.get("primary", {})
        status = primary.get("status", "no_check")
        url = primary.get("url", "")
        http_code = primary.get("http_status", None)

        norm = normalize(name)
        in_s3 = norm in s3_datasets or name in s3_datasets

        entry = {"name": name, "url": url, "http_status": http_code, "in_s3": in_s3}

        if status == "ok":
            if in_s3:
                report["healthy_cached"].append(entry)
            else:
                report["healthy_not_cached"].append(entry)
        else:
            if in_s3:
                report["broken_cached"].append(entry)
            else:
                report["broken_not_cached"].append(entry)

    report["summary"] = {
        "total": len(health),
        "healthy_cached": len(report["healthy_cached"]),
        "healthy_not_cached": len(report["healthy_not_cached"]),
        "broken_cached": len(report["broken_cached"]),
        "broken_not_cached": len(report["broken_not_cached"]),
    }

    REPORT_OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"\nFull report: {REPORT_OUT}")


if __name__ == "__main__":
    main()
