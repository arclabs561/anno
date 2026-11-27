"""Refresh the numerical sections of the dataset guide from the same ledgers as its plots."""

import csv
import re
from collections import Counter


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def coverage_text(evidence, rendered, report):
    coverage = rows(rendered / "dataset-coverage.csv")
    density = rows(rendered / "dataset-density-ranges.csv")
    domains = {
        r["domain"]: r for r in rows(rendered / "dataset-coverage-by-domain.csv")
    }
    measured = sum(int(r["measured"]) for r in coverage)
    tagged = sum(int(r["tagged_tokens_measured"]) for r in coverage)
    missing = len(coverage) - measured
    direct = [
        r
        for r in rows(evidence / "dataset-direct-files.csv")
        if r["status"] in {"records measured", "tagged tokens measured"}
    ]
    repo = [r for r in rows(evidence / "dataset-repo-files.csv") if r["status"] == "ok"]
    hub = [
        r
        for r in rows(evidence / "dataset-hf-splits.csv")
        if r["status"] in {"scanned", "scanned_unlabelled", "scanned_records"}
    ]
    hub_ids = {r["id"] for r in hub}
    repo_ids = {r["id"] for r in repo}
    direct_ids = {r["id"] for r in direct}
    git_ids = [r for r in coverage if int(r["github_measured_files"])]
    direct_files = len({(r["source"], r["sha256"]) for r in direct})
    actions = Counter(r["coverage_action"] for r in coverage if r["measured"] == "0")
    day = max(
        r["checked_utc"] for r in rows(evidence / "dataset-registry-pointers.csv")
    )
    clinical = domains["clinical"]

    def domain_count(label):
        row = domains[label]
        return f"{int(row['other_measured']) + int(row['tagged_tokens_measured'])} of {row['registered']}"

    sections = {}
    sections["overview"] = (
        f"The survey accounts for all **{len(coverage)} registry IDs**. It measured corpus content for **{measured}**, "
        f"including **{tagged}** with a tagged-token numerator and denominator. The other **{measured-tagged}** have counts "
        f"such as documents, words, questions or annotation offsets. **{missing}** have no independent content measurement. "
        f"The registry URL check is dated {day}. A source row identifies the edition, split, annotation layer or sample actually read. "
        "[All IDs, source scopes and exclusions](dataset-coverage.csv) · [Registry definitions](../../crates/anno-eval/src/eval/dataset_registry.rs)"
    )
    sections["domains"] = (
        f"The readers measured {domain_count('news')} news entries, {domain_count('biomedical')} biomedical entries and "
        f"{domain_count('wikipedia')} Wikipedia entries. Clinical coverage is **{domain_count('clinical')}**, including "
        f"{clinical['tagged_tokens_measured']} with a verified token-label denominator. The chart shows the eight largest "
        f"domains. The domain table includes all {len(domains)} registry labels. [Counts by domain](dataset-coverage-by-domain.csv) · "
        "[Source terms and measured layers for every clinical entry](dataset-clinical-access.csv)"
    )
    sections["routes"] = "\n".join(
        [
            "| Source | Observed scope | Reproduction evidence |",
            "|---|---|---|",
            f"| Registry files, release archives and Anno fixtures | {len(direct):,} partition or annotation-layer observations from {direct_files:,} distinct file references for {len(direct_ids)} IDs. | [Files, scope, counts and SHA-256](dataset-direct-files.csv) |",
            f"| Pinned publisher repositories | {len(repo):,} measured file rows for {len(repo_ids)} IDs. Their metrics include NER, entity links, syntax and part-of-speech tags. | [Repository, commit, file, metric and hash](dataset-repo-files.csv) |",
            f"| Hugging Face and publisher-linked releases | {len(hub):,} measured split or file observations for {len(hub_ids)} IDs. {sum(r['status']=='scanned' for r in hub):,} observations have tagged-token counts, {sum(r['status']=='scanned_unlabelled' for r in hub):,} have word counts without scored labels and {sum(r['status']=='scanned_records' for r in hub):,} have record counts. | [Revisions, scope, provider counts and measured columns](dataset-hf-splits.csv) |",
            f"| GitHub release survey | {len(git_ids)} IDs with reviewed corpus counts. A single ZIP can contain several measured partitions. | [Publisher paths, measured units, byte hashes and exclusions](dataset-github-sources.csv) |",
        ]
    )
    units = [int(r["n_observation_units"]) for r in density]
    sections["density"] = (
        f"The chart covers **{len(density)} registry IDs and {sum(units):,} observed files or partitions** after collapsing known "
        f"duplicates within each ID. Each partition contributes one percentage. The dot gives its ID’s median, the thin line "
        f"gives the observed range and the thick line gives the middle half for IDs with at least four observations. "
        f"For **{units.count(1)} IDs**, a single observation supplies the entire estimate. "
        "[Quartiles, endpoints, denominators and sources](dataset-density-ranges.csv)"
    )
    sections["exclusions"] = "\n".join(
        [
            f"The following actions partition all **{missing}** IDs without an independent content measurement. Each ID receives one primary action. "
            "Its row retains the status and reason from every route, including a separately reviewed publisher URL where available. "
            "[Filter by `coverage_action` and `source_disposition`](dataset-coverage.csv) · [Dated HTTP checks](dataset-registry-pointers.csv) · [Publisher access reviews](dataset-source-review.csv)",
            "",
            "| Next action | Registry IDs |",
            "|---|---:|",
            *(
                f"| {label} | {count} |"
                for label, count in sorted(actions.items(), key=lambda x: (-x[1], x[0]))
            ),
        ]
    )
    for key, replacement in sections.items():
        start = f"<!-- coverage-{key} -->"
        end = f"<!-- /coverage-{key} -->"
        pattern = re.escape(start) + ".*?" + re.escape(end)
        report, count = re.subn(
            pattern,
            lambda _: f"{start}\n\n{replacement}\n\n{end}",
            report,
            flags=re.DOTALL,
        )
        if count != 1:
            raise ValueError(f"Expected one {key} section in the coverage guide")
    return report
