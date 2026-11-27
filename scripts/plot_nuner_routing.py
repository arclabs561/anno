"""Render the paired accuracy and review cost of running NuNER on skipped sentences."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

INK = "#213945"
MUTED = "#4a626d"
RULE = "#dce4e5"


def draw(receipt):
    rows = []
    for group, y in (("without_bert", 4.0), ("with_bert", 1.4)):
        for offset, split in enumerate(("Development", "Test")):
            arm = receipt["splits"][split][group]
            n = arm["stock_skipped"]
            counts = arm["skipped_subset"]
            exact = counts["exact_gained"] - counts["exact_lost"]
            unmatched = (
                counts["in_schema_unmatched_added"]
                - counts["in_schema_unmatched_removed"]
            )
            if n <= 0:
                raise ValueError(f"No skipped sentences in {split}, {group}")
            if exact != counts["candidate_exact"] - counts[
                "stock_exact"
            ] or unmatched != (
                counts["candidate_in_schema_unmatched"]
                - counts["stock_in_schema_unmatched"]
            ):
                raise ValueError(
                    f"Paired changes disagree with totals in {split}, {group}"
                )
            rows.append(
                (y - offset * 0.83, split, n, 100 * exact / n, 100 * unmatched / n)
            )

    fig, (labels, exact_ax, unmatched_ax) = plt.subplots(
        1, 3, figsize=(8.8, 4.85), sharey=True, width_ratios=(0.84, 1, 1)
    )
    fig.subplots_adjust(left=0.036, right=0.975, bottom=0.24, top=0.736, wspace=0.13)
    fig.text(
        0.036,
        0.93,
        "What does an extra NuNER call recover?",
        size=18,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.036,
        0.869,
        "Paired WNUT-17 predictions when Anno runs NuNER on sentences where the stock stack skipped that layer.",
        size=10.6,
        color=MUTED,
    )

    labels.set_ylim(0.05, 5.2)
    labels.axis("off")
    for y, title in ((4.64, "BERT unavailable"), (2.04, "BERT loaded")):
        labels.text(0, y, title, weight="bold", color=INK, size=11, va="center")
    for y, split, n, _, _ in rows:
        labels.text(
            0.04, y, f"{split} · {n:,} skipped", va="center", size=9.5, color=MUTED
        )

    all_rates = [0, *(rate for row in rows for rate in row[3:])]
    low, high = min(all_rates), max(all_rates)
    # Leave room for the signed value at either end, including a zero result.
    pad = max(high - low, 10) * 0.23
    for ax, title, column, color in (
        (exact_ax, "Exact matches", 3, "#146e70"),
        (unmatched_ax, "Unmatched predictions", 4, "#9b5e21"),
    ):
        ax.set_title(title, loc="left", size=11.3, color=INK, weight="bold", pad=5)
        ax.set_xlim(low - pad, high + pad)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.grid(axis="x", color=RULE, linewidth=0.7)
        ax.axvline(0, color=MUTED, linewidth=1.1)
        ax.spines[:].set_visible(False)
        ax.tick_params(axis="both", length=0, labelcolor=MUTED, labelsize=8.7)
        ax.set_yticks([])
        for row in rows:
            y, value = row[0], row[column]
            ax.hlines(y, 0, value, color=color, linewidth=2.3)
            ax.plot(value, y, "o", color=color, markersize=6)
            ax.annotate(
                f"{value:+.1f}" if value else "0",
                (value, y),
                xytext=(6 if value >= 0 else -6, 0),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="center",
                color=INK,
                size=9.2,
            )
    fig.text(
        0.036,
        0.122,
        "Change per 100 skipped sentences. Positive values mean more predictions of that kind.",
        weight="bold",
        color=INK,
        size=9.5,
    )
    fig.text(
        0.036,
        0.062,
        "Negative values mean fewer. Both columns use mapped WNUT labels. Exact matches require the reference span and type.",
        color=MUTED,
        size=8.8,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fig = draw(json.loads(args.receipt.read_text(encoding="utf-8")))
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "nuner-routing-tradeoff.png", dpi=165)
    with plt.rc_context({"svg.hashsalt": "anno-nuner-routing-tradeoff"}):
        fig.savefig(args.output / "nuner-routing-tradeoff.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
