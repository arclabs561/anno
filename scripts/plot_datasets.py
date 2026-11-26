"""Render Anno dataset figures (requires Matplotlib >= 3.6 and upstream HTTP access).

Usage: python scripts/plot_datasets.py OUTPUT_DIRECTORY
The CSV records per-file read failures. Remote URLs may change between runs.
"""

import argparse
import collections
import concurrent.futures
import csv
import json
import pathlib
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_file_survey import (
    FILE_FIELDS,
    ROOT,
    local_source_ids,
    registry_rows,
    survey_entry,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=pathlib.Path)
parser.add_argument(
    "--source-cache",
    type=pathlib.Path,
    help="Reuse downloaded bytes between runs; source hashes are recorded in the direct-file ledger",
)
args = parser.parse_args()
root = args.output
root.mkdir(exist_ok=True, parents=True)
repo = ROOT
registry, task_counts = registry_rows()
local_only = local_source_ids()
rows, direct_files, comparison, query_densities = [], [], {}, {}
with tempfile.TemporaryDirectory() as scratch:
    cache = args.source_cache or pathlib.Path(scratch)
    cache.mkdir(exist_ok=True, parents=True)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for row, files, forms, densities in pool.map(
            lambda entry: survey_entry(entry, cache, local_only), registry
        ):
            rows.append(row)
            direct_files.extend(files)
            if forms:
                comparison[row["name"]] = (
                    100 * row["tagged"] / row["tokens"],
                    100 * sum(n == 1 for n in forms.values()) / len(forms),
                )
            if densities:
                query_densities[row["name"]] = densities

for filename, fields, data in [
    ("anno-all.csv", list(rows[0]), rows),
    ("dataset-direct-files.csv", FILE_FIELDS, direct_files),
]:
    with (root / filename).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
measured_rows = sorted(
    (row for row in rows if row["tokens"]),
    key=lambda row: row["tagged"] / row["tokens"],
)
print(
    collections.Counter(row["result"] for row in rows),
    [
        (row["name"], round(100 * row["tagged"] / row["tokens"], 1))
        for row in measured_rows
    ],
)
fig, (access_axis, density_axis) = plt.subplots(
    1, 2, figsize=(13, 6), layout="constrained"
)
groups = collections.Counter(row["access"] for row in rows).most_common()[::-1]
for axis, labels, values, title, color in [
    (
        access_axis,
        [x for x, n in groups],
        [n for x, n in groups],
        f"Registry access labels · {len(rows)} entries",
        "#315b75",
    ),
    (
        density_axis,
        [row["name"] for row in measured_rows],
        [100 * row["tagged"] / row["tokens"] for row in measured_rows],
        "Entity-tagged tokens · % of file",
        "#008675",
    ),
]:
    bars = axis.barh(labels, values, color=color, height=0.63)
    axis.bar_label(bars, fmt="%.1f" if axis == density_axis else "%d", padding=4)
    axis.margins(x=0.15)
    axis.set_title(title, loc="left")
    axis.spines[["top", "right", "left"]].set_visible(False)
fig.suptitle("Anno dataset coverage and annotation density", fontsize=18)
for ext in ("png", "svg"):
    fig.savefig(root / ("anno-all." + ext), dpi=150)
plt.close(fig)


def bar_chart(name, title, counts):
    fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    labels, values = zip(*counts)
    bars = axis.barh(labels, values, color="#187673")
    axis.invert_yaxis()
    axis.bar_label(bars, padding=4)
    axis.margins(x=0.13)
    axis.set_title(title, loc="left")
    for ext in ("png", "svg"):
        fig.savefig(root / f"{name}.{ext}", dpi=150)
    plt.close(fig)


tags = ["ner", "coref", "el", "re", "temporal"]
bar_chart(
    "anno-tasks",
    f"Selected explicit task tags across {len(rows)} registry entries",
    [(tag, task_counts[tag]) for tag in tags],
)
fixture = repo / "crates/anno-eval/fixtures/pii_masking_fixture.jsonl"
counts = collections.Counter(
    entity["type"]
    for line in fixture.read_text().splitlines()
    for entity in json.loads(line)["entities"]
)
bar_chart(
    "anno-pii", "Annotated spans in the PII masking fixture", counts.most_common()
)
names = ["WikiGold", "WNUT-16"]
fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
for i, title in enumerate(
    ["Tokens inside entity mentions", "Distinct mention strings seen once"]
):
    for x, (name, color) in enumerate(zip(names, ["#176e70", "#d58642"])):
        if name in comparison:
            bars = axes[i].bar(x, comparison[name][i], color=color)
            axes[i].bar_label(bars, fmt="%.1f%%")
        else:
            axes[i].text(x, 4, "Unavailable", ha="center")
    axes[i].set_xticks(range(len(names)), names)
    axes[i].set_xlim(-0.6, len(names) - 0.4)
    axes[i].set_ylim(0, 100)
    axes[i].set_title(title)
for ext in ("png", "svg"):
    fig.savefig(root / f"anno-corpora.{ext}", dpi=150)
plt.close(fig)

fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
for name, densities in sorted(query_densities.items()):
    ordered = sorted(densities)
    cumulative = [100 * rank / len(ordered) for rank in range(1, len(ordered) + 1)]
    axis.step(
        [0, *ordered, 100],
        [0, *cumulative, 100],
        where="post",
        label=f"{name} ({len(ordered):,} queries)",
    )
axis.set(
    xlim=(0, 100),
    ylim=(0, 101),
    xlabel="Tokens tagged in a query (%)",
    ylabel="Queries at or below threshold (%)",
    title="How concentrated are slot annotations within individual queries?",
)
if query_densities:
    axis.legend()
else:
    axis.text(50, 50, "No measured query files in this survey", ha="center")
for ext in ("png", "svg"):
    fig.savefig(root / f"anno-query-density.{ext}", dpi=150)
plt.close(fig)
