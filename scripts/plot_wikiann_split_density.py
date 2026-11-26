"""Compare train-based estimates of tag density for each WikiANN held-out split.

Run python3 scripts/plot_wikiann_split_density.py HF_SPLIT_LEDGER OUTPUT_DIR.
Read complete, matched train/validation/test measurements from the same release.
Each language contributes one error per held-out split. The second estimate uses
the median training rate across the other measured languages.
"""

import argparse
import csv
from collections import defaultdict
from fractions import Fraction
from math import ceil
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HELD_OUT = {"validation": "Development", "test": "Test"}
SPLITS = ("train", *HELD_OUT)
INK = "#182e3a"
MUTED = "#4d626c"
SAME = "#176e70"
OTHER = "#aa571c"


def comparison_rows(ledger):
    languages = defaultdict(dict)
    releases = set()
    for row in ledger:
        if row["id"] != "WikiANN" or row["split"] not in SPLITS:
            continue
        lang, split = row["config"], row["split"]
        if not lang or row["status"] != "scanned":
            raise ValueError(f"Missing measured WikiANN partition {lang}/{split}")
        if split in languages[lang]:
            raise ValueError(f"Duplicate WikiANN partition {lang}/{split}")
        tagged, tokens = int(row["tagged"]), int(row["tokens"])
        if tokens <= 0 or not 0 <= tagged <= tokens:
            raise ValueError(f"Invalid tagged-token counts for {lang}/{split}")
        languages[lang][split] = (tagged, tokens)
        releases.add(
            (row["repository"], row["source_revision"], row["parquet_revision"])
        )
    if len(releases) != 1 or not all(next(iter(releases))):
        raise ValueError("The WikiANN measurements must identify one pinned release")
    if len(languages) < 2:
        raise ValueError("The comparison requires at least two languages")
    for lang, splits in languages.items():
        if set(splits) != set(SPLITS):
            raise ValueError(
                f"Incomplete WikiANN language {lang}, measured splits {sorted(splits)}"
            )

    repository, source_revision, parquet_revision = releases.pop()
    train = {lang: 100 * Fraction(*parts["train"]) for lang, parts in languages.items()}
    rows = []
    for lang, parts in sorted(languages.items()):
        other = median([rate for peer, rate in train.items() if peer != lang])
        row = {
            "config": lang,
            "repository": repository,
            "source_revision": source_revision,
            "parquet_revision": parquet_revision,
        }
        for split in SPLITS:
            tagged, tokens = parts[split]
            row[f"{split}_tagged"] = tagged
            row[f"{split}_tokens"] = tokens
            row[f"{split}_percent"] = float(100 * Fraction(tagged, tokens))
        row["other_languages_train_median_percent"] = float(other)
        for split in HELD_OUT:
            observed = 100 * Fraction(*parts[split])
            row[f"{split}_same_language_error_pp"] = float(abs(observed - train[lang]))
            row[f"{split}_other_languages_error_pp"] = float(abs(observed - other))
        rows.append(row)
    return rows


def plot(rows, output):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8.6, 4.1))
    fig.subplots_adjust(left=0.095, right=0.974, top=0.615, bottom=0.226, wspace=0.14)
    total = len(rows)
    fig.text(
        0.035,
        0.96,
        "How well does training density predict a held-out split?",
        fontsize=16,
        weight="bold",
        va="top",
        color=INK,
    )
    fig.text(
        0.035,
        0.855,
        f"WikiANN · {total} languages · silver-tagged tokens as a share of all tokens",
        fontsize=9.4,
        va="top",
        color=MUTED,
    )
    fig.legend(
        handles=[
            Line2D(
                [], [], color=SAME, lw=2.4, label="Training split in the same language"
            ),
            Line2D(
                [],
                [],
                color=OTHER,
                lw=2.1,
                ls="--",
                label=f"Median training rate in the other {total - 1} languages",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.027, 0.802),
        ncol=2,
        columnspacing=2,
        handlelength=2.7,
        frameon=False,
        fontsize=8.9,
    )
    largest = max(
        row[f"{split}_{reference}_error_pp"]
        for row in rows
        for split in HELD_OUT
        for reference in ("same_language", "other_languages")
    )
    limit = max(10, 10 * ceil(largest / 10))
    for ax, (split, label) in zip(axes, HELD_OUT.items()):
        medians = []
        within_five = []
        for reference, color, style in (
            ("same_language", SAME, "-"),
            ("other_languages", OTHER, "--"),
        ):
            errors = sorted(row[f"{split}_{reference}_error_pp"] for row in rows)
            ranks = [100 * i / total for i in range(1, total + 1)]
            ax.step(
                [0, *errors, limit],
                [0, *ranks, 100],
                where="post",
                color=color,
                ls=style,
                lw=2.0,
                zorder=3,
            )
            medians.append(median(errors))
            within_five.append(sum(error <= 5 for error in errors))
        ax.set_title(label, loc="left", fontsize=11, weight="bold", color=INK, pad=8)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, 103)
        ax.set_xticks(range(0, limit + 1, 10))
        ax.set_yticks((0, 25, 50, 75, 100))
        ax.grid(axis="y", color="#d6dfe0", lw=0.65)
        ax.axvline(5, linestyle=":", color="#72838a", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#98a9ac")
        ax.tick_params(length=2.5, color="#98a9ac", labelsize=8.4)
        ax.text(
            0.0,
            -0.37,
            f"Median error   {medians[0]:.1f} vs {medians[1]:.1f} points",
            transform=ax.transAxes,
            fontsize=9.0,
            color=INK,
            weight="bold",
        )
        ax.text(
            0.0,
            -0.50,
            f"Within 5 points   {within_five[0]} vs {within_five[1]} of {total} languages",
            transform=ax.transAxes,
            fontsize=8.0,
            color=MUTED,
        )
    axes[0].set_ylabel("Languages within error threshold (%)", fontsize=8.6, color=INK)
    for ax in axes:
        ax.set_xlabel("Absolute error (percentage points)", fontsize=8.5, color=INK)
    for ext in ("png", "svg"):
        fig.savefig(output / f"wikiann-split-density.{ext}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_splits", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.hf_splits.open(newline="", encoding="utf-8") as source:
        rows = comparison_rows(csv.DictReader(source))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "dataset-wikiann-split-density.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output)


if __name__ == "__main__":
    main()
