"""Render the paired accuracy and review cost of running NuNER on skipped sentences."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#213945"
MUTED = "#4a626d"
RULE = "#dce4e5"


def draw(receipt):
    groups = {}
    for split in ("Test", "Development"):
        groups[split] = {}
        for name, group in (
            ("BERT absent", "without_bert"),
            ("BERT loaded", "with_bert"),
        ):
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
            if (
                exact != counts["candidate_exact"] - counts["stock_exact"]
                or unmatched
                != counts["candidate_in_schema_unmatched"]
                - counts["stock_in_schema_unmatched"]
            ):
                raise ValueError(
                    f"Paired changes disagree with totals in {split}, {group}"
                )
            timing = arm["warm_skipped_subset_latency"]
            if any(timing[name]["calls"] != n for name in ("stock", "candidate")):
                raise ValueError(
                    "Timed stack calls must cover the same skipped sentences"
                )
            groups[split][name] = dict(
                n=n,
                rates=(100 * exact / n, 100 * unmatched / n),
                suggestions=exact + unmatched,
                before=timing["stock"]["median_ms"],
                after=timing["candidate"]["median_ms"],
            )

    width, height = 374, 674
    fig, ax = plt.subplots(figsize=(width / 72, height / 72))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")

    def label(x, y, value, size=14.0, bold=False, color=INK, align="left"):
        ax.text(
            x,
            y,
            value,
            fontsize=size,
            color=color,
            ha=align,
            va="baseline",
            fontweight="bold" if bold else "normal",
        )

    label(13, 26, "What does a skipped NuNER", 19.5, True)
    label(13, 50, "call buy on WNUT-17?", 19.5, True)
    label(13, 78, "Net change per 100 skipped sentences.")
    label(13, 98, "Both outcomes share the same scale.")
    rates = [v for g in groups.values() for arm in g.values() for v in arm["rates"]]
    low, high = min(0, *rates), max(0, *rates)
    span = max(high - low, 1)

    def x_rate(value):
        return 117 + (value - low) * 163 / span

    y = 134
    for split, arms in groups.items():
        samples = {arm["n"] for arm in arms.values()}
        if len(samples) != 1:
            raise ValueError(
                f"Compared BERT conditions used different populations on {split}"
            )
        n = samples.pop()
        ax.hlines(y - 17, 13, width - 13, color="#70868c", lw=1)
        label(13, y + 3, f"{split} · {n:,} skipped sentences", 15, True)
        y += 32
        for name, arm in arms.items():
            label(13, y, name, 14, True)
            y += 23
            for (outcome, color), rate in zip(
                (("Exact", "#14736c"), ("Unmatched", "#9b5e21")),
                arm["rates"],
                strict=True,
            ):
                label(13, y, outcome)
                ax.hlines(y - 5, x_rate(low), x_rate(low + span), color=RULE, lw=2)
                ax.hlines(y - 5, x_rate(0), x_rate(rate), color=color, lw=4)
                ax.plot(x_rate(rate), y - 5, "o", color=color, ms=6)
                label(width - 13, y, f"{rate:+.1f}", 14, True, align="right")
                y += 20
            label(
                13,
                y + 2,
                f"{arm['suggestions']:+,} suggestions · {arm['before']:,.1f} → {arm['after']:,.1f} ms",
                color=MUTED,
            )
            y += 39
        y += 2
    label(13, y + 1, "Suggestions count net outputs in WNUT’s five")
    label(13, y + 21, "mapped types. Times are warm median stack")
    label(13, y + 41, "latency, stock → one extra call per sentence.")
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
