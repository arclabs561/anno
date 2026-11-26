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
        github_files[(r["revision"], r["path"])][r["unit"]] = int(r["value"])
    github_tagged = [
        units
        for units in github_files.values()
        if "tagged tokens" in units
        and "tokens" in units
        and units["tagged tokens"] <= units["tokens"]
    ]

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
        direct_measured_files=len(direct_record),
        direct_tagged_files=len(tag_routes["direct"]),
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
        github_measured_files=len(github_files),
        github_tagged_files=len(github_tagged),
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


def coverage_plot(rows, output, supplied):
    routes = {
        "direct": "Registry-linked files",
        "repo": "Publisher file manifest",
        "hf": "Hugging Face splits",
        "github": "GitHub release survey",
    }
    routes = {key: label for key, label in routes.items() if key in supplied}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    all_counts = [
        len(rows),
        sum(r["measured"] for r in rows),
        sum(r["tagged_tokens_measured"] for r in rows),
    ]
    bars = axes[0].barh(
        ["Registered", "Records or tokens counted", "Tagged tokens counted"],
        all_counts,
        color=["#b5c6c8", "#39737d", "#ae6328"],
        height=0.6,
    )
    axes[0].bar_label(bars, padding=4)
    axes[0].set_title("Unique registry entries", loc="left")
    for index, (route, label) in enumerate(routes.items()):
        record = sum(route in r["measured_routes"].split(" | ") for r in rows)
        tagged = sum(route in r["tagged_routes"].split(" | ") for r in rows)
        axes[1].barh(
            index,
            record,
            height=0.54,
            color="#39737d",
            label="Records or tokens" if index == 0 else None,
        )
        axes[1].barh(
            index,
            tagged,
            height=0.54,
            color="#ae6328",
            label="Of which tagged" if index == 0 else None,
        )
        axes[1].annotate(
            f"{tagged} of {record}",
            (record, index),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
        )
    axes[1].set_yticks(range(len(routes)), routes.values())
    axes[1].set_title("Coverage by source route", loc="left")
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False
    )
    for ax in axes:
        ax.set_xlabel("Registry entries")
        ax.invert_yaxis()
        ax.spines[["left", "top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.margins(x=0.24)
    fig.suptitle("How many Anno datasets have a verified measurement?", fontsize=16)
    fig.supxlabel(
        "Source routes overlap. Each entry counts once in the left panel.", fontsize=10
    )
    for ext in ("png", "svg"):
        fig.savefig(output / f"dataset-coverage.{ext}", dpi=160)
    plt.close(fig)


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
            "ci-prechecks",
            "Named, observed CI prechecks; recorded separately from measurement.",
        ),
    ]:
        parser.add_argument("--" + option, type=Path, help=description)
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
    }
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
    with (args.output / "dataset-coverage.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    selection_plot(inventory, args.output)
    coverage_plot(rows, args.output, supplied)


if __name__ == "__main__":
    main()
