"""Plot the distribution of verified tag densities across measured files and splits.

Pass INVENTORY OUTPUT_DIR and the four evidence CSVs. No dataset text or
network access is required. Percentages use the tokenization of each release.
Each deduplicated unit has equal weight. Related splits can share examples.
Quartiles use linear interpolation, and are reported for at least four units.
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import median, quantiles

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        rates = [100 * item.density for item in units.values()]
        quartiles = (
            quantiles(rates, n=4, method="inclusive") if len(rates) >= 4 else None
        )
        row = {
            "id": dataset,
            "name": names[dataset],
            "annotation": SPECIAL.get(dataset, "Entity tags"),
            "n_observation_units": len(units),
            "n_route_observations": len(group),
            "median_percent": float(median(rates)),
            "p25_percent": float(quartiles[0]) if quartiles else "",
            "p75_percent": float(quartiles[2]) if quartiles else "",
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
        rows, key=lambda row: (-row["median_percent"], row["name"], row["id"])
    )


def plot(rows, output):
    if not rows:
        raise ValueError("No verified positive token denominators")

    cohorts = [
        ("4+ files or partitions", [r for r in rows if r["n_observation_units"] >= 4]),
        (
            "2–3 files or partitions",
            [r for r in rows if 2 <= r["n_observation_units"] <= 3],
        ),
        ("One file or partition", [r for r in rows if r["n_observation_units"] == 1]),
    ]
    cohorts = [(label, part) for label, part in cohorts if part]
    plot_rows = len(rows) + 2.3 * len(cohorts)
    height = 1.30 + 0.201 * plot_rows
    fig, ax = plt.subplots(figsize=(8.6, height))
    fig.subplots_adjust(
        left=0.025, right=0.985, top=1 - 1.0 / height, bottom=0.18 / height
    )
    # The graph uses percentage points. Text columns sit in the margins of the
    # same coordinate system to align each denominator and median with its row.
    ax.set(xlim=(-112, 122), ylim=(plot_rows - 0.6, -0.6))
    ax.axis("off")
    ink = "#182e3a"
    muted = "#4d626c"
    teal = "#176e70"
    header_color = "#304c5b"
    title_x = 0.025
    fig.text(
        title_x,
        1 - 0.12 / height,
        "What share of tokens is tagged?",
        fontsize=17,
        weight="bold",
        color=ink,
        va="top",
    )
    units = sum(row["n_observation_units"] for row in rows)
    fig.text(
        title_x,
        1 - 0.47 / height,
        f"{len(rows)} registry entries · {units:,} measured files or partitions · equal weight per measurement",
        fontsize=9.2,
        color=muted,
        va="top",
    )
    # Draw a single, directly labelled key above the table.
    legend_y = 1 - 0.785 / height
    for x1, x2, label, color, width in [
        (0.027, 0.059, "Full observed range", "#80939b", 1.15),
        (0.280, 0.312, "Middle 50% (four or more units)", teal, 4.6),
    ]:
        line = plt.Line2D(
            [x1, x2],
            [legend_y, legend_y],
            color=color,
            linewidth=width,
            solid_capstyle="butt",
            transform=fig.transFigure,
        )
        fig.add_artist(line)
        fig.text(x2 + 0.007, legend_y, label, va="center", fontsize=8.2, color=ink)
    fig.add_artist(
        plt.Line2D(
            [0.640],
            [legend_y],
            marker="o",
            markersize=4.5,
            color=ink,
            transform=fig.transFigure,
        )
    )
    fig.text(
        0.651, legend_y, "Median across units", va="center", fontsize=8.2, color=ink
    )

    labels = {
        "Entity tags": "Entity",
        "Silver entity tags": "Silver entity",
        "Entity links": "Entity links",
        "Slot tags": "Slots",
        "PII subword tags": "PII subwords",
    }
    y = 0
    for heading, part in cohorts:
        ax.text(
            -111,
            y,
            heading,
            weight="bold",
            fontsize=8.7,
            va="center",
            color=header_color,
        )
        ax.text(-55, y, "Labels", fontsize=8, va="center", color=muted)
        ax.text(-12, y, "Units", fontsize=8, ha="right", va="center", color=muted)
        ax.text(121, y, "Median", fontsize=8, ha="right", va="center", color=muted)
        for percent in (0, 25, 50, 75, 100):
            ax.text(
                percent,
                y,
                f"{percent}%",
                ha="center",
                va="center",
                fontsize=7.6,
                color=muted,
            )
            ax.vlines(
                percent,
                y + 0.51,
                y + len(part) + 0.5,
                color="#e3e9e9",
                lw=0.65,
                zorder=0,
            )
        ax.hlines(y + 0.50, -111, 121, color="#b5c5c7", lw=0.8)
        for row in part:
            y += 1
            name = row["name"] + (" (fixture)" if row["id"] == "AnnoNerSmoke" else "")
            ax.text(-111, y, name, fontsize=8.2, va="center", color=ink)
            ax.text(
                -55,
                y,
                labels[row["annotation"]],
                fontsize=7.7,
                va="center",
                color=muted,
            )
            ax.text(
                -12,
                y,
                str(row["n_observation_units"]),
                ha="right",
                va="center",
                fontsize=8.3,
                color=ink,
            )
            ax.hlines(
                y, row["min_percent"], row["max_percent"], lw=1.05, color="#718991"
            )
            if row["n_observation_units"] >= 4:
                ax.hlines(y, row["p25_percent"], row["p75_percent"], lw=4.9, color=teal)
            ax.plot(
                row["median_percent"], y, "o", ms=4.5, mfc=ink, mec="white", mew=0.5
            )
            ax.text(
                121,
                y,
                f"{row['median_percent']:.1f}%",
                ha="right",
                va="center",
                fontsize=8.3,
                color=ink,
            )
        y += 2.3
    for ext in ("png", "svg"):
        fig.savefig(output / f"dataset-density.{ext}", dpi=160)
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
