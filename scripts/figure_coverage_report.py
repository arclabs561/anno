"""Refresh the numerical sections of the dataset guide from the same ledgers as its plots."""

import csv
import json
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
    direct_tagged = sum("direct" in r["tagged_routes"].split(" | ") for r in coverage)
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
        f"domains. The domain table includes all {len(domains)} registry labels. Open the explorer, choose a domain and select a scored outcome to see the backend and split. "
        "[Dataset explorer](dataset-explorer.html) · [Counts by domain](dataset-coverage-by-domain.csv) · "
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
    sections["routes"] += (
        f"\n\nThe direct reader measured files for **{len(direct_ids)} of {len(coverage)} registry IDs**. "
        f"It obtained tagged-token counts for **{direct_tagged} IDs**. Combining all four routes gives **{tagged} IDs** "
        "with a tagged-token denominator. The format and language panels describe selection by the direct reader."
    )
    evaluation = rows(rendered / "dataset-evaluation-coverage.csv")
    survey = json.loads(
        (evidence / "dataset-ci-survey.json").read_text(encoding="utf-8")
    )
    ci = rows(evidence / "dataset-ci-outcomes.csv")
    scored = [r for r in evaluation if r["status"] == "Scored"]
    scored_ids = {r["id"] for r in scored}
    states = Counter(r["status"] for r in ci)
    read_sources = Counter(
        r["read_source"] or "unreported source" for r in ci if r["status"] == "Scored"
    )
    reads = ", ".join(f"{n} from {name}" for name, n in sorted(read_sources.items()))
    split_sources = Counter(
        r["split_evidence"]
        for r in scored
        if r["source"].startswith("https://github.com/arclabs561/anno/actions/")
    )
    runs = survey["runs"]
    unknown = sum(r["status"] == "No indexed evaluation" for r in evaluation)
    sections["evaluations"] = (
        f"**{len(scored_ids)} of {len(coverage)} registry IDs have a completed score in these sources**. "
        f"{len({r['id'] for r in scored if r['measurement'] == 'Tagged-token counts'})} of those IDs also have tagged-token counts. "
        f"{unknown} IDs have no outcome in the indexed sources. The remaining IDs have a loader rejection, backend failure or exporter replay only. "
        "[Search by domain, backend and split](dataset-explorer.html) · [All outcomes with source receipts](dataset-evaluation-coverage.csv)\n\n"
        f"On **{survey['day_utc']} UTC**, `{survey['workflow']}` recorded {len(runs)} push runs on `{survey['branch']}`. "
        f"{sum(r['conclusion'] == 'success' for r in runs)} succeeded and {sum(r['conclusion'] == 'cancelled' for r in runs)} were cancelled. "
        f"The {sum(len(r['artifacts']) for r in runs)} sanity and matrix archives contain {states['Scored']} scored attempts, "
        f"{states['Failed evaluation']} failed evaluations, {states['Passed load check']} named loader selections and {states['Load rejected']} named pre-scoring load rejections. "
        f"The scored attempts reported {reads}. A local read can follow an S3 staging step. "
        f"A same-ID publisher-file digest identifies the input split or release for {split_sources['same-ID publisher file SHA-256']} scored attempts. "
        f"{split_sources['explicit checkout fixture']} used named checkout fixtures and {split_sources['unreported'] + split_sources['ambiguous same-ID file scopes']} had no established split. "
        "The same examples can recur across pushes. "
        "[Run and artifact digests](dataset-ci-survey.json) · [Normalized CI records](dataset-ci-outcomes.csv)"
    )
    units = [int(r["n_observation_units"]) for r in density]
    sections["density"] = (
        "The guide compares four biomedical term-annotation corpora. Each dot represents one source file or partition, "
        "with the number of labeled token positions divided by all token positions in that unit. The line spans the observed files. "
        "The releases label different term types and use their own tokenization.\n\n"
        f"The full reference covers **{len(density)} registry IDs and {sum(units):,} observed files or partitions** after collapsing known "
        f"duplicates within each ID. Each unit contributes one percentage. For **{units.count(1)} IDs**, a single unit supplies the entire estimate. "
        "Filter by annotation unit before comparing entity tags, slot tags, silver labels, entity links or subword PII labels. "
        "[Search partitions and denominators](dataset-explorer.html) · [Full distribution chart](../figures/dataset-density.png) · "
        "[Quartiles and endpoints](dataset-density-ranges.csv) · [Every numerator, denominator and source](dataset-density-units.csv)"
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
    cached = json.loads(
        (evidence / "s3-source-coverage.json").read_text(encoding="utf-8")
    )
    matches = rows(evidence / "s3-publisher-file-matches.csv")
    own_ids = {r["cache_id"] for r in matches if r["cache_id"] == r["publisher_id"]}
    if len(own_ids) != cached["registry_ids_with_own_publisher_match"] or (
        len({r["sha256"] for r in matches})
        != cached["distinct_matched_publisher_digests"]
    ):
        raise ValueError("The S3 file matches must agree with the aggregate receipt")
    kinds = cached["objects_by_kind"]
    from plot_s3_source_evidence import summary_counts

    totals, _ = summary_counts(cached)
    matched_elsewhere = cached["payload_objects_by_outcome"].get(
        "matches_other_registry_file_only", 0
    )
    unmatched = totals["read"] - totals["own"] - matched_elsewhere
    sections["s3"] = (
        f"On **{cached['listing_started_utc'][:10]} UTC**, trusted CI listed **{cached['objects_listed']} objects** "
        f"under `datasets/`. Its **{totals['payload']} recognized payload objects** yielded **{totals['read']} successful reads**. "
        f"Of those, {totals['own']} matched an independently measured publisher file for the same ID, {matched_elsewhere} matched only another ID "
        f"and {unmatched} lacked a verified match. The comparator skipped {totals['skipped']} objects and recorded {totals['failed']} read failures. "
        f"The {kinds.get('unrecognized', 0)} unrecognized keys contribute only to the inventory.\n\n"
        f"Of **{cached['registry_ids_with_payload']} distinct registry IDs** with a payload, **{len(own_ids)}** had an own-publisher match. "
        f"The comparison used {cached['publisher_digests_compared']:,} independently measured publisher digests. "
        "A match identifies the file or split whose bytes agree. The evaluation index records which inputs each backend scored. "
        f"[Trusted run](https://github.com/arclabs561/anno/actions/runs/{cached['workflow_run_id']}) · "
        "[Counts and exclusions](s3-source-coverage.json) · [File matches and measured units](s3-publisher-file-matches.csv) · "
        "[Backend and split outcomes](dataset-evaluation-coverage.csv)"
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
