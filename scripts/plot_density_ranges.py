"""Plot verified tagged-token ranges across measured files and provider partitions.

Pass INVENTORY OUTPUT_DIR and the four evidence CSVs. No dataset text or
network access is required. Percentages use the tokenization of each release.
Observation counts describe files/partitions, not independent examples.
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {
    "Entity tags": "#256a81",
    "Silver entity tags": "#669246",
    "Slot tags": "#ac5919",
    "Entity links": "#645aa5",
    "PII subword tags": "#a54768",
}
SPECIAL = {
    "WikiANN": "Silver entity tags",
    "WikiNeural": "Silver entity tags",
    "PolyglotNER": "Silver entity tags",
    "ATISFlightBooking": "Slot tags",
    "MitMovie": "Slot tags",
    "MitRestaurant": "Slot tags",
    "ZELDA": "Entity links",
    "PIIMasking200k": "PII subword tags",
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


@dataclass(frozen=True)
class Observation:
    dataset: str
    route: str
    source: str
    partition: str
    identity: tuple
    tagged: int
    tokens: int
    sha256: str

    @property
    def density(self):
        return Fraction(self.tagged, self.tokens)


def collect(direct, repository, hub, github):
    observations = []
    # Published NER partitions and their HF conversions can have identical counts.
    # Collapse only a known matching split AND equal numerator/denominator.
    physical_to_split = {}

    def add(dataset, route, source, partition, identity, tagged, tokens, digest=""):
        if tagged == "" or tokens == "":
            return
        tagged, tokens = int(tagged), int(tokens)
        if tagged < 0 or tokens < tagged:
            raise ValueError(f"Invalid tagged-token counts for {dataset}, {source}")
        if tokens:
            observations.append(
                Observation(
                    dataset, route, source, partition, identity, tagged, tokens, digest
                )
            )

    for row in repository:
        if row["status"] != "ok" or not row["tagged"]:
            continue
        dataset = row["id"]
        partition = "/".join(filter(None, [row["subset"], row["split"]]))
        identity = ("sha256", row["sha256"])
        if dataset in {"CrossNER", "MasakhaNER2"}:
            split = "validation" if row["split"] == "dev" else row["split"]
            identity = ("partition", row["subset"], split)
            physical_to_split[dataset, row["sha256"]] = identity
        add(
            dataset,
            "repository file",
            row["url"],
            partition,
            identity,
            row["tagged"],
            row["tokens"],
            row["sha256"],
        )

    for row in direct:
        if row["status"] == "tagged tokens measured":
            identity = physical_to_split.get(
                (row["id"], row["sha256"]), ("sha256", row["sha256"])
            )
            partition = row.get("split_or_subset") or row["scope"]
            add(
                row["id"],
                "direct file",
                row["source"],
                partition,
                identity,
                row["tagged"],
                row["tokens"],
                row["sha256"],
            )

    for row in hub:
        if row["status"] != "scanned" or (
            row["id"] == "TweetNER7" and row["split"].startswith("extra_")
        ):
            continue
        dataset, config, split = row["id"], row["config"], row["split"]
        revision = row["parquet_revision"] or row["source_revision"]
        source = f"https://huggingface.co/datasets/{row['repository']}/tree/{revision}"
        route = "Hub partition"
        partition = f"{config}/{split}"
        digest = row.get("payload_sha256", "")
        if row.get("payload_url"):
            source = row["payload_url"]
            route = "Hub-linked publisher file"
        elif dataset == "MultiNERD" and not row["parquet_revision"]:
            source = f"https://huggingface.co/datasets/{row['repository']}/resolve/{revision}/{config}"
            route, partition = "Hub file", split
        else:
            source += f"/{partition}"
        identity = ("sha256", digest) if digest else ("hub", config, split)
        if dataset in {"CrossNER", "MasakhaNER2"}:
            identity = ("partition", config, split)
        elif dataset == "FabNER" and config in {
            "fabner",
            "fabner_bio",
            "fabner_simple",
        }:
            identity = ("FabNER shared sentences", split)
        add(
            dataset,
            route,
            source,
            partition,
            identity,
            row["tagged"],
            row["tokens"],
            digest,
        )

    # The GitHub survey stores one metric per row. Join the numerator to its
    # denominator only for the very same source path, split and digest.
    counts = defaultdict(dict)
    for row in github:
        if row["status"] == "ok" and row["unit"] in {"tokens", "tagged tokens"}:
            key = (row["id"], row["source_url"], row["split"], row["sha256"])
            previous = counts[key].setdefault(row["unit"], row["value"])
            if previous != row["value"]:
                raise ValueError(f"Conflicting GitHub counts for {key}")
    for (dataset, source, partition, digest), values in counts.items():
        if {"tokens", "tagged tokens"} <= values.keys():
            add(
                dataset,
                "GitHub file",
                source,
                partition,
                ("sha256", digest),
                values["tagged tokens"],
                values["tokens"],
                digest,
            )
    return observations


def summarize(observations, names):
    groups = defaultdict(list)
    for item in observations:
        groups[item.dataset].append(item)
    rows = []
    for dataset, group in groups.items():
        units = {}
        for item in group:
            # Different files with coincidentally equal counts remain separate.
            units.setdefault((item.identity, item.tagged, item.tokens), item)
        row = {
            "id": dataset,
            "name": names[dataset],
            "annotation": SPECIAL.get(dataset, "Entity tags"),
            "n_observation_units": len(units),
            "n_route_observations": len(group),
        }
        for prefix, extreme in (("min", min), ("max", max)):
            endpoint = extreme(units.values(), key=lambda item: item.density)
            row.update(
                {
                    f"{prefix}_tagged": endpoint.tagged,
                    f"{prefix}_tokens": endpoint.tokens,
                    f"{prefix}_percent": float(100 * endpoint.density),
                    f"{prefix}_route": endpoint.route,
                    f"{prefix}_sha256": endpoint.sha256,
                    f"{prefix}_source": endpoint.source,
                    f"{prefix}_partition": endpoint.partition,
                }
            )
        rows.append(row)
    return sorted(
        rows, key=lambda row: (-row["max_percent"], -row["min_percent"], row["id"])
    )


def plot(rows, output):
    if not rows:
        raise ValueError("No verified positive token denominators")
    midpoint = (len(rows) + 1) // 2
    parts = [rows[:midpoint], rows[midpoint:]] if len(rows) > 1 else [rows]
    fig, axes = plt.subplots(
        1,
        len(parts),
        figsize=(8 * len(parts), 3.4 + 0.36 * midpoint),
        squeeze=False,
        layout="constrained",
    )
    for axis, part in zip(axes[0], parts):
        for index, row in enumerate(part):
            low, high = row["min_percent"], row["max_percent"]
            color = COLORS[row["annotation"]]
            axis.plot([low, high], [index, index], color=color, linewidth=2.2, zorder=2)
            positions = [low] if low == high else [low, high]
            axis.scatter(
                positions, [index] * len(positions), s=29, color=color, zorder=3
            )
        labels = [
            r["name"]
            + (" (fixture)" if r["id"] == "AnnoNerSmoke" else "")
            + f"   n={r['n_observation_units']}"
            for r in part
        ]
        axis.set(
            yticks=range(len(part)),
            yticklabels=labels,
            ylim=(midpoint - 0.48, -0.52),
            xlim=(-1.5, 101.5),
            xticks=range(0, 101, 20),
        )
        axis.set_xlabel("Tagged tokens (% of file or split)")
        axis.grid(axis="x", color="#dbe2e5", linewidth=0.8)
        axis.spines[["top", "left", "right"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    legend = [
        Line2D([], [], marker="o", color=color, lw=1.8, label=label)
        for label, color in COLORS.items()
        if any(r["annotation"] == label for r in rows)
    ]
    fig.suptitle(
        f"How much text is annotated in {len(rows)} Anno datasets?\n"
        "Minimum and maximum tagged-token coverage across n measured files or partitions",
        fontsize=17,
    )
    # Attach legend to an axis so constrained layout reserves its space.
    axes[0, 0].legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0, -0.075),
        frameon=False,
        ncol=2,
        fontsize=9,
    )
    for ext in ("png", "svg"):
        fig.savefig(output / f"dataset-density.{ext}", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    for option in ("direct-files", "repo-files", "hf-splits", "github-sources"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    names = {row["id"]: row["name"] for row in read_csv(args.inventory)}
    observations = collect(
        *(
            read_csv(path)
            for path in (
                args.direct_files,
                args.repo_files,
                args.hf_splits,
                args.github_sources,
            )
        )
    )
    rows = summarize(observations, names)
    args.output.mkdir(parents=True, exist_ok=True)
    plot(rows, args.output)
    with (args.output / "dataset-density-ranges.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
