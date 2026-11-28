"""Compare train-based estimates of tag density for each WikiANN held-out split.

Run python3 scripts/plot_wikiann_split_density.py HF_SPLIT_LEDGER OUTPUT_DIR.
Read complete, matched train/validation/test measurements from the same release.
Each language contributes one error per held-out split. The second estimate uses
the median training rate across the other measured languages. Add --multiconer-v2
to write the same comparison for the twelve single-language MultiCoNER v2 tracks.
"""

import argparse
import csv
import re
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


def comparison_rows(ledger, dataset="WikiANN"):
    languages = defaultdict(dict)
    releases = set()
    raw_hashes = {}
    for row in ledger:
        if row["id"] != dataset:
            continue
        lang, split = row["config"], row["split"]
        if dataset == "MultiCoNERv2":
            if lang == "MULTI-Multilingual":
                continue
            split = "validation" if split == "dev" else split
        if split not in SPLITS:
            continue
        if not lang or row["status"] != "scanned":
            raise ValueError(f"Missing measured {dataset} partition {lang}/{split}")
        if split in languages[lang]:
            raise ValueError(f"Duplicate {dataset} partition {lang}/{split}")
        tagged, tokens = int(row["tagged"]), int(row["tokens"])
        if tokens <= 0 or not 0 <= tagged <= tokens:
            raise ValueError(f"Invalid tagged-token counts for {lang}/{split}")
        languages[lang][split] = (tagged, tokens)
        releases.add(
            (row["repository"], row["source_revision"], row["parquet_revision"])
        )
        if dataset == "MultiCoNERv2":
            sha = row["payload_sha256"]
            if (
                not re.fullmatch(r"[0-9a-f]{64}", sha)
                or not re.fullmatch(r"[0-9a-f]{40}", row["source_revision"])
                or f"/resolve/{row['source_revision']}/" not in row["payload_url"]
            ):
                raise ValueError(
                    f"Missing pinned original CoNLL file for {lang}/{split}"
                )
            raw_hashes[lang, split] = sha
    if len(releases) != 1:
        raise ValueError(f"The {dataset} measurements must identify one pinned release")
    repository, source_revision, parquet_revision = releases.pop()
    if (
        not repository
        or not source_revision
        or (dataset == "WikiANN" and not parquet_revision)
    ):
        raise ValueError(f"Incomplete source revisions for {dataset}")
    if len(languages) < 2:
        raise ValueError("The comparison requires at least two languages")
    for lang, splits in languages.items():
        if set(splits) != set(SPLITS):
            raise ValueError(
                f"Incomplete {dataset} language {lang}, measured splits {sorted(splits)}"
            )

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
            if raw_hashes:
                row[f"{split}_payload_sha256"] = raw_hashes[lang, split]
        row["other_languages_train_median_percent"] = float(other)
        for split in HELD_OUT:
            observed = 100 * Fraction(*parts[split])
            row[f"{split}_same_language_error_pp"] = float(abs(observed - train[lang]))
            row[f"{split}_other_languages_error_pp"] = float(abs(observed - other))
        rows.append(row)
    return rows


def plot(rows, output):
    total = len(rows)
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(4.9, 7.25))
    fig.subplots_adjust(left=0.13, right=0.965, top=0.748, bottom=0.157, hspace=0.97)
    fig.text(
        0.04,
        0.956,
        "How close is training density?",
        fontsize=16,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.04,
        0.912,
        f"WikiANN · {total} matched languages · silver-tagged tokens",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.04,
        0.878,
        "Curves count languages within each absolute error.",
        fontsize=11,
        color=MUTED,
    )
    fig.legend(
        handles=[
            Line2D([], [], color=SAME, lw=2.4, label="Same language’s training rate"),
            Line2D(
                [],
                [],
                color=OTHER,
                lw=2.2,
                ls="--",
                label=f"Median training rate in the other {total - 1} languages",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.018, 0.868),
        handlelength=2.3,
        frameon=False,
        fontsize=10.6,
        labelspacing=0.4,
    )
    largest = max(
        row[f"{split}_{reference}_error_pp"]
        for row in rows
        for split in HELD_OUT
        for reference in ("same_language", "other_languages")
    )
    limit = max(10, 10 * ceil(largest / 10))
    for ax, (split, label) in zip(axes, HELD_OUT.items()):
        medians, within_five = [], []
        ax.axvline(5, linestyle=":", color="#65767e", lw=1.2)
        for reference, color, style, marker in (
            ("same_language", SAME, "-", "o"),
            ("other_languages", OTHER, "--", "s"),
        ):
            errors = sorted(row[f"{split}_{reference}_error_pp"] for row in rows)
            ranks = [100 * i / total for i in range(1, total + 1)]
            ax.step(
                [0, *errors, limit],
                [0, *ranks, 100],
                where="post",
                color=color,
                ls=style,
                lw=2,
            )
            within = sum(error <= 5 for error in errors)
            ax.plot(5, 100 * within / total, marker=marker, ms=6, color=color, zorder=4)
            medians.append(median(errors))
            within_five.append(within)
        ax.set_title(
            f"{label}  ·  5-point tolerance",
            loc="left",
            fontsize=12.3,
            weight="bold",
            color=INK,
            pad=6,
        )
        ax.set(
            xlim=(0, limit),
            ylim=(0, 103),
            xticks=[0, 5, *range(20, limit + 1, 20)],
            yticks=(0, 50, 100),
            yticklabels=("0%", "50%", "100%"),
        )
        ax.grid(axis="y", color="#d6dfe0", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#98a9ac")
        ax.tick_params(length=2.5, color="#98a9ac", labelsize=10.6, labelbottom=True)
        # Position the labels in rank order at the observed tolerance.
        entries = sorted(
            zip(within_five, medians, ("Same", "Other"), (SAME, OTHER)), reverse=True
        )
        for i, (count, mid, name, color) in enumerate(entries):
            ax.text(
                0,
                -0.31 - i * 0.155,
                f"{name}  {count}/{total} within tolerance  ·  median error {mid:.1f} pp",
                transform=ax.transAxes,
                fontsize=10.2,
                color=color,
            )
    fig.text(
        0.04,
        0.016,
        "Horizontal scale is absolute error in percentage points.",
        fontsize=10.4,
        color=MUTED,
    )
    for ext in ("png", "svg"):
        fig.savefig(output / f"wikiann-split-density.{ext}", dpi=160)
    plt.close(fig)


def write_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_splits", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--multiconer-v2",
        action="store_true",
        help="Also write held-out estimates for MultiCoNER v2 single-language tracks",
    )
    args = parser.parse_args()
    with args.hf_splits.open(newline="", encoding="utf-8") as source:
        ledger = list(csv.DictReader(source))
    rows = comparison_rows(ledger)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output / "dataset-wikiann-split-density.csv")
    if args.multiconer_v2:
        transfer = comparison_rows(ledger, "MultiCoNERv2")
        write_csv(transfer, args.output / "dataset-multiconer-v2-split-density.csv")
    plot(rows, args.output)


if __name__ == "__main__":
    main()
