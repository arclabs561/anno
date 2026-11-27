"""Compile unique-ID measurement coverage. The inventory alone reproduces the
historical direct-reader selection figure. Add source ledgers for a combined
coverage CSV and figure. Routes, configurations and splits can overlap.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def selection_plot(rows, output):
    # Each entry contributes once, regardless of corpus size.
    read = [
        row
        for row in rows
        if row["result"] in ("File read", "BIO token counts", "Tagged token counts")
    ]
    measured = [
        row
        for row in read
        if row["result"] in {"BIO token counts", "Tagged token counts"}
    ]
    stages = [rows, read, measured]
    stage_names = ["Registered", "File read", "Token counts"]
    colors = ["#36536a", "#009987", "#d57725", "#826a9e", "#c2c9cf"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")

    top_formats = [
        label for label, _ in Counter(row["format"] for row in rows).most_common(4)
    ]
    for ax, field, labels, title in [
        (axes[0], "format", top_formats, "Declared file format"),
        (axes[1], "language", ["en", "mul"], "Registry language code"),
    ]:
        start = [0.0] * len(stages)
        for label, color in zip(labels + ["Other"], colors):
            counts = [
                sum(
                    (
                        (r[field] not in labels)
                        if label == "Other"
                        else (r[field] == label)
                    )
                    for r in stage
                )
                for stage in stages
            ]
            widths = [
                100 * n / len(stage) if stage else 0 for n, stage in zip(counts, stages)
            ]
            bars = ax.barh(stage_names, widths, left=start, color=color, label=label)
            for bar, count, width in zip(bars, counts, widths):
                if width >= 8:
                    ax.text(
                        bar.get_x() + width / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(count),
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="black" if color == "#c2c9cf" else "white",
                    )
            start = [x + w for x, w in zip(start, widths)]
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(title, loc="left")
        ax.set_yticks(
            range(3),
            [f"{name} ({len(stage)})" for name, stage in zip(stage_names, stages)],
        )
        ax.set_xlabel("Share of entries at each stage (%)")
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=3, frameon=False
        )
    fig.suptitle("Which datasets reach token-level analysis?", fontsize=17)
    for extension in ("png", "svg"):
        fig.savefig(output / f"anno-selection.{extension}", dpi=160)
    plt.close(fig)


def load_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def group_by_id(path, ids, key="id"):
    groups = defaultdict(list)
    if path:
        for row in load_csv(path):
            ident = row[key]
            if ident not in ids:
                raise ValueError(f"{path}: unknown registry ID {ident!r}")
            groups[ident].append(row)
    return groups


def joined(values):
    return " | ".join(sorted(set(x for x in values if x)))


def counts(values):
    return " | ".join(f"{k} ({n})" for k, n in sorted(Counter(values).items()))


def has_count(row, key):
    return str(row.get(key, "")).isdigit()


def has_tagged(row):
    return (
        has_count(row, "tokens")
        and has_count(row, "tagged")
        and int(row["tokens"]) > 0
        and int(row["tagged"]) <= int(row["tokens"])
    )


def inventory_reason(row):
    if row.get("reason"):
        return row["reason"]
    if row["result"] not in {
        "Registry metadata only",
        "File read",
        "BIO token counts",
        "Tagged token counts",
    }:
        return row["result"]
    if row["access"] not in {"Public", "HuggingFace"}:
        return f"Registry access label: {row['access']}"
    if not row["url"]:
        return "Registry has no source URL"
    site = urlsplit(row["url"]).hostname
    if site == "github.com":
        return "Repository pointer; direct reader has no verified file mapping"
    if site in {"huggingface.co", "datasets-server.huggingface.co"}:
        return "Hub pointer; direct reader has no verified file mapping"
    if site == "universaldependencies.org":
        return "Treebank landing page; direct reader has no verified file mapping"
    if site in {"arxiv.org", "aclanthology.org", "doi.org", "www.sciencedirect.com"}:
        return "Publication pointer; direct reader has no verified corpus file"
    if row["result"] == "File read":
        return "Source bytes read; inventory supplies no validated record count"
    return f"No validated file measurement for registry URL on {site or 'unspecified host'}"


def inventory_observation(row):
    """Retain the older inventory semantics when no per-file ledger is supplied."""
    if row["result"] in {"BIO token counts", "Tagged token counts"} and has_tagged(row):
        return [
            dict(
                row,
                status="tagged tokens measured",
                records="",
                unit="tokens",
                reason="",
            )
        ]
    if row["result"] == "File read":
        return [
            dict(
                row,
                status="file read only",
                records="",
                unit="",
                reason=inventory_reason(row),
            )
        ]
    return []


def source_disposition(row, review):
    """Give one actionable outcome per ID. HTTP responses describe only the registry URL."""
    if row["measured"]:
        return "Corpus content measured"
    if row["same_release_as"]:
        return "Measured under another registry ID"
    if review:
        return review["disposition"]
    pointer = row["pointer_status"]
    if pointer == "local-source policy":
        return "Local source required"
    if row["access"] not in {"Public", "HuggingFace"}:
        return f"Registry access: {row['access']}"
    if "collection-level link" in row["github_reasons"]:
        return "General collection, matching dataset unresolved"
    if (
        "unparsed response" in row["direct_statuses"]
        or "file read only" in row["direct_statuses"]
    ):
        return "Source returned, contents unverified"
    if row["hf_provider_counts_only"]:
        return "Only provider-reported counts"
    if not row["url"]:
        return "Registry supplies no URL"
    if pointer in {
        "missing at this URL",
        "request denied",
        "sign-in page",
        "connection failed",
        "other HTTP response",
    }:
        return f"Registry pointer: {pointer}"
    if row["github_statuses"]:
        return "Repository or owner, matching files unresolved"
    if "Publication" in row["direct_reasons"]:
        return "Publication without verified corpus file"
    return "Linked source without verified corpus file"


def coverage_action(disposition):
    if disposition == "Corpus content measured":
        return "Inspect measured scope"
    if disposition == "Measured under another registry ID":
        return "Use the measured alias"
    if disposition in {
        "Registry access: Deprecated",
        "Registry access: NotYetReleased",
    }:
        return "Check release or replacement"
    if disposition.startswith("Registry pointer:"):
        return "Resolve the registry URL"
    if disposition in {
        "Only provider-reported counts",
        "Source returned, contents unverified",
    }:
        return "Validate corpus records"
    if disposition.startswith("Registry access:") or disposition in {
        "Local source required",
        "licensed distribution",
        "author request required",
        "credentials required",
        "credentials and source mapping",
        "individual data use agreement",
        "data use agreement and source mapping",
    }:
        return "Arrange source access"
    return "Identify the matching release"


def compile_entry(entry, sources, direct_supplied):
    ident = entry["id"]
    direct = (
        sources["direct"][ident] if direct_supplied else inventory_observation(entry)
    )
    repo = sources["repo"][ident]
    hub = sources["hf"][ident]
    github = sources["github"][ident]
    ci = sources["ci"][ident]
    statuses = {
        "direct": direct,
        "repo": repo,
        "hf": hub,
        "github": github,
    }

    direct_record = [
        r
        for r in direct
        if r["status"] in {"records measured", "tagged tokens measured"}
        and (has_count(r, "records") or has_count(r, "tokens"))
    ]
    repo_record = [
        r
        for r in repo
        if r["status"] == "ok" and (has_count(r, "rows") or has_count(r, "tokens"))
    ]
    hf_record = [
        r
        for r in hub
        if r["status"] in {"scanned", "scanned_unlabelled", "scanned_records"}
        and (has_count(r, "rows") or has_count(r, "tokens"))
    ]
    # GitHub rows can share a file across several metrics. Only observed record
    # or annotation counts belong here. Git trees and archive indices do not.
    github_record = [
        r
        for r in github
        if r["status"] == "ok"
        and r["metric"] == "count"
        and r["sha256"]
        and r["unit"] in RECORD_UNITS
        and has_count(r, "value")
    ]
    github_files = defaultdict(dict)
    for r in github_record:
        github_files[(r["revision"], r["path"], r["split"])][r["unit"]] = int(
            r["value"]
        )
    github_tagged = {
        key
        for key, units in github_files.items()
        if "tagged tokens" in units
        and units.get("tokens", 0) > 0
        and units["tagged tokens"] <= units["tokens"]
    }

    record_routes = {
        "direct": direct_record,
        "repo": repo_record,
        "hf": hf_record,
        "github": github_record,
    }
    tag_routes = {
        "direct": [
            r
            for r in direct_record
            if r["status"] == "tagged tokens measured" and has_tagged(r)
        ],
        "repo": [r for r in repo_record if has_tagged(r)],
        "hf": [r for r in hf_record if r["status"] == "scanned" and has_tagged(r)],
        "github": github_tagged,
    }
    row = {
        k: entry[k]
        for k in ("id", "name", "format", "language", "domain", "access", "url")
    }
    row.update(
        measured=int(any(record_routes.values())),
        tagged_tokens_measured=int(any(tag_routes.values())),
        measured_routes=joined(
            route for route, observed in record_routes.items() if observed
        ),
        tagged_routes=joined(
            route for route, observed in tag_routes.items() if observed
        ),
    )
    for route, observations in statuses.items():
        row[route + "_statuses"] = counts(r["status"] for r in observations)
        row[route + "_reasons"] = joined(r.get("reason", "") for r in observations)
    if not direct:
        row["direct_reasons"] = inventory_reason(entry)
    row.update(
        direct_measured_files=len(
            {(r.get("source"), r.get("sha256")) for r in direct_record}
        ),
        direct_tagged_files=len(
            {(r.get("source"), r.get("sha256")) for r in tag_routes["direct"]}
        ),
        direct_measured_partitions=len(direct_record),
        direct_units=joined(r["unit"] for r in direct_record),
        repo_measured_files=len(repo_record),
        repo_tagged_files=len(tag_routes["repo"]),
        repo_metrics=joined(r["metric"] for r in repo_record),
        hf_measured_splits=len(hf_record),
        hf_tagged_splits=len(tag_routes["hf"]),
        hf_provider_counts_only=sum(
            r["status"] == "provider_only" and has_count(r, "provider_rows")
            for r in hub
        ),
        github_measured_files=len(
            {(r["source_url"], r["sha256"]) for r in github_record}
        ),
        github_measured_partitions=len(github_files),
        github_tagged_files=len(
            {
                (r["source_url"], r["sha256"])
                for r in github_record
                if (r["revision"], r["path"], r["split"]) in github_tagged
            }
        ),
        github_tagged_partitions=len(github_tagged),
        github_units=joined(r["unit"] for r in github_record),
        ci_precheck_runs=len({r["run_id"] for r in ci}),
        ci_precheck_sources=counts(r["source"] for r in ci),
        ci_s3_precheck_runs=len({r["run_id"] for r in ci if r["source"] == "S3 cache"}),
    )
    if row["tagged_tokens_measured"]:
        row["outstanding_reason"] = ""
    elif row["measured"]:
        units = joined(
            [
                row["direct_units"],
                row["repo_metrics"],
                row["github_units"],
                "HF input rows/tokens without scored labels" if hf_record else "",
            ]
        )
        row["outstanding_reason"] = (
            f"Record counts available ({units}); no verified tagged-token numerator and denominator"
        )
    else:
        reasons = [
            f"{k}: {row[k + '_reasons']}" for k in statuses if row[k + "_reasons"]
        ]
        if ci:
            reasons.append(
                "CI precheck reports source availability; no corpus measurement in that observation"
            )
        row["outstanding_reason"] = "; ".join(reasons)
    pointers = sources.get("pointers", {}).get(ident, [])
    if pointers and (len(pointers) != 1 or pointers[0]["url"] != row["url"]):
        raise ValueError(f"Registry pointer ledger does not match {ident}")
    row["pointer_status"] = pointers[0]["pointer_status"] if pointers else ""
    row["same_release_as"] = {
        "FINERFood": "FINER",
        "LatinITTB": "LatinUD",
        "CSN": "CodeSearchNet",
        "CRAFTCorpusCoref": "CRAFTCoref",
    }.get(ident, "")
    review = sources.get("reviews", {}).get(ident, {})
    row["source_review_url"] = review.get("source", "")
    row["source_review_scope"] = review.get("scope_and_access", "")
    row["source_disposition"] = source_disposition(row, review)
    row["coverage_action"] = coverage_action(row["source_disposition"])
    return row


# Accepted count units produced by survey_github_sources.metrics and its
# reviewed release readers. Keep archive members and collection metadata out.
RECORD_UNITS = {
    "annotated argument pairs",
    "annotated segments",
    "classification examples",
    "concept mentions",
    "contexts",
    "dialogues",
    "documents",
    "event-pair labels",
    "image comparison records",
    "ingredient spans",
    "mention annotations",
    "omniscient separate mentions",
    "prompts",
    "pronoun decisions",
    "questions",
    "rebuttal sentences",
    "recipe records",
    "relation examples",
    "relations",
    "review sentences",
    "review-rebuttal pairs",
    "sentence pairs",
    "tagged tokens",
    "tokens",
    "topics",
    "turns",
    "utterances",
}


COVERAGE_CATEGORIES = ("tagged_tokens_measured", "other_measured", "unmeasured")


def coverage_by_domain(rows):
    by_domain = defaultdict(Counter)
    for row in rows:
        counts_for_domain = by_domain[row["domain"]]
        counts_for_domain["registered"] += 1
        if int(row["tagged_tokens_measured"]):
            counts_for_domain["tagged_tokens_measured"] += 1
        elif int(row["measured"]):
            counts_for_domain["other_measured"] += 1
        else:
            counts_for_domain["unmeasured"] += 1
    summary = [
        {
            "domain": domain,
            **{k: group[k] for k in ("registered", *COVERAGE_CATEGORIES)},
        }
        for domain, group in by_domain.items()
    ]
    return sorted(summary, key=lambda group: (-group["registered"], group["domain"]))


def coverage_plot(domains, output):
    def totals(groups):
        return {
            k: sum(r[k] for r in groups) for k in ("registered", *COVERAGE_CATEGORIES)
        }

    # Rank by registry size rather than measurement success. The final row
    # includes every domain beyond the eight shown by name.
    overall = totals(domains)
    labels = ["All domains"]
    groups = [overall]
    for group in domains[:8]:
        labels.append(group["domain"].replace("_", " ").capitalize() or "Unspecified")
        groups.append(group)
    if len(domains) > 8:
        labels.append(f"Other {len(domains) - 8} domains")
        groups.append(totals(domains[8:]))
    positions = [0, *(i + 1.35 for i in range(len(groups) - 1))]
    ink, muted = "#1e3440", "#4b616a"
    colors = ("#176d6b", "#d99142", "#dce3e6")

    fig, (bars, values) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(8.8, 2.55 + 0.30 * len(groups)),
        gridspec_kw={"width_ratios": (2.2, 1.2)},
    )
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.14, top=0.76, wspace=0.06)
    fig.text(
        0.025, 0.95, "Dataset coverage by domain", size=18, weight="bold", color=ink
    )
    fig.text(
        0.025,
        0.895,
        f"{overall['registered']:,} registry entries  ·  "
        f"{overall['tagged_tokens_measured']:,} with tagged-token counts  ·  "
        f"{overall['other_measured']:,} with other counts only",
        size=10,
        color=muted,
    )

    left = [0.0] * len(groups)
    for category, label, color in zip(
        COVERAGE_CATEGORIES,
        ("Tagged tokens", "Other counts only", "No corpus measurement"),
        colors,
    ):
        shares = [100 * row[category] / row["registered"] for row in groups]
        bars.barh(
            positions,
            shares,
            left=left,
            label=label,
            color=color,
            height=0.56,
            edgecolor="white",
            linewidth=0.6,
        )
        left = [a + b for a, b in zip(left, shares)]
    bars.set(xlim=(0, 100), ylim=(positions[-1] + 0.60, -0.80))
    bars.set_yticks(positions, labels, fontsize=10, color=ink)
    bars.get_yticklabels()[0].set_fontweight("bold")
    bars.set_xticks(
        [0, 25, 50, 75, 100],
        ["0%", "25%", "50%", "75%", "100%"],
        fontsize=9,
        color=muted,
    )
    bars.set_xlabel("Share of registry entries in the domain", fontsize=9, color=muted)
    bars.spines[:].set_visible(False)
    bars.tick_params(length=0)
    bars.set_axisbelow(True)
    bars.grid(axis="x", color="#edf0f1", linewidth=0.7)
    fig.legend(
        *bars.get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.019, 0.855),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=1.15,
        columnspacing=1.7,
    )

    values.set_xlim(0, 4)
    values.set_axis_off()
    for i, (key, label) in enumerate(
        zip((*COVERAGE_CATEGORIES, "registered"), ("Tagged", "Other", "None", "Total"))
    ):
        values.text(
            i + 0.5,
            -0.76,
            label,
            ha="center",
            va="center",
            size=9,
            weight="bold",
            color=muted,
        )
        for y, group in zip(positions, groups):
            values.text(
                i + 0.5,
                y,
                f"{group[key]:,}",
                ha="center",
                va="center",
                size=10,
                color=ink,
                weight="bold" if y == 0 else "normal",
            )
    for axis in (bars, values):
        axis.axhline(0.71, color="#b0bfc3", linewidth=0.7)
    for ext in ("png", "svg"):
        fig.savefig(output / f"dataset-coverage.{ext}", dpi=180)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    for option, description in [
        (
            "direct-files",
            "Direct-file measurement ledger. Supersedes inventory-level file status.",
        ),
        ("repo-files", "Pinned publisher-file manifest and observed counts."),
        ("hf-splits", "Hugging Face provider and measured split counts."),
        ("github-sources", "GitHub release survey with record-unit counts."),
        (
            "registry-pointers",
            "Header-only registry URL audit. Adds source dispositions, never content counts.",
        ),
        (
            "ci-prechecks",
            "Named, observed CI prechecks; recorded separately from measurement.",
        ),
    ]:
        parser.add_argument("--" + option, type=Path, help=description)
    parser.add_argument(
        "--source-review",
        type=Path,
        action="append",
        default=[],
        help="Publisher access or scope findings for specific IDs. Repeat for additional reviews.",
    )
    args = parser.parse_args()
    inventory = load_csv(args.inventory)
    ids = {r["id"] for r in inventory}
    if "" in ids or not ids or len(ids) != len(inventory):
        parser.error("Inventory requires a unique, nonempty ID for every entry")
    sources = {
        "direct": group_by_id(args.direct_files, ids),
        "repo": group_by_id(args.repo_files, ids),
        "hf": group_by_id(args.hf_splits, ids),
        "github": group_by_id(args.github_sources, ids),
        "ci": group_by_id(args.ci_prechecks, ids, key="dataset"),
        "pointers": group_by_id(args.registry_pointers, ids),
    }
    if args.registry_pointers and set(sources["pointers"]) != ids:
        parser.error(
            "Registry pointer audit requires exactly one outcome for each inventory ID"
        )
    sources["reviews"] = {}
    for path in args.source_review:
        for ident, reviews in group_by_id(path, ids).items():
            if len(reviews) != 1 or ident in sources["reviews"]:
                parser.error(f"Duplicate source review for {ident}")
            sources["reviews"][ident] = reviews[0]
    rows = [
        compile_entry(entry, sources, bool(args.direct_files)) for entry in inventory
    ]
    supplied = ["direct"]
    for route, path in [
        ("repo", args.repo_files),
        ("hf", args.hf_splits),
        ("github", args.github_sources),
        ("ci", args.ci_prechecks),
    ]:
        if path:
            supplied.append(route)
    for row in rows:
        row["survey_routes_supplied"] = joined(supplied)
    args.output.mkdir(parents=True, exist_ok=True)
    domains = coverage_by_domain(rows)
    write_csv(args.output / "dataset-coverage.csv", rows)
    write_csv(args.output / "dataset-coverage-by-domain.csv", domains)
    selection_plot(inventory, args.output)
    coverage_plot(domains, args.output)


if __name__ == "__main__":
    main()
