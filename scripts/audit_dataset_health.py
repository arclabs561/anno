#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Compare previously checked registry URLs with an optional local S3 inventory.

Usage:
    python3 scripts/audit_dataset_health.py [--s3-inventory PATH]

Reads: scripts/registry_enrichment.json (health results from prior run)
Writes: scripts/dataset_audit_report.json

Does NOT modify the Rust registry -- that requires manual edits.
"""
import argparse
import json
import sys
from pathlib import Path

ENRICHMENT = Path("scripts/registry_enrichment.json")
REPORT_OUT = Path("scripts/dataset_audit_report.json")


def normalize(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "").replace(" ", "")


def load_s3_datasets(inventory: Path | None = None) -> set[str] | None:
    """Parse S3 inventory file into a set of normalized dataset names."""
    if inventory is None:
        return None
    names = set()
    for line in inventory.read_text().splitlines():
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-inventory", type=Path, help="Local output of aws s3 ls --recursive for your bucket")
    parser.add_argument("--enrichment", type=Path, default=ENRICHMENT)
    parser.add_argument("--output", type=Path, default=REPORT_OUT)
    args = parser.parse_args()
    if args.s3_inventory is not None and not args.s3_inventory.is_file():
        parser.error(f"S3 inventory not found: {args.s3_inventory}")
    if not args.enrichment.exists():
        print(f"Error: {args.enrichment} not found")
        sys.exit(1)

    data = json.loads(args.enrichment.read_text())
    health = data.get("health_results", {})
    s3_datasets = load_s3_datasets(args.s3_inventory)

    report = {
        "summary": {},
        "healthy_cached": [],
        "healthy_not_cached": [],
        "broken_cached": [],
        "broken_not_cached": [],
        "cache_unknown": [],
    }

    for name, checks in health.items():
        primary = checks.get("primary", {})
        status = primary.get("status", "no_check")
        url = primary.get("url", "")
        http_code = primary.get("http_status", None)

        norm = normalize(name)
        in_s3 = None if s3_datasets is None else (norm in s3_datasets or name in s3_datasets)

        entry = {"name": name, "url": url, "http_status": http_code, "in_s3": in_s3}

        if in_s3 is None:
            report["cache_unknown"].append({**entry, "status": status})
        elif status == "ok":
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
        "cache_unknown": len(report["cache_unknown"]),
        "healthy_cached": len(report["healthy_cached"]),
        "healthy_not_cached": len(report["healthy_not_cached"]),
        "broken_cached": len(report["broken_cached"]),
        "broken_not_cached": len(report["broken_not_cached"]),
    }

    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
