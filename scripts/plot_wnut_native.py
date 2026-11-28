"""Render the paired native WNUT-17 replay from its aggregate JSON."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INK = "#213945"
MUTED = "#4a626d"
OUTCOMES = {
    "exact": ("Exact span and type", "#146e70"),
    "wrong_type": ("Exact span, wrong type", "#c48b44"),
    "partial_overlap": ("Boundary overlap", "#ad9cbe"),
    "no_overlap": ("No overlap", "#dbe4e5"),
}


def draw(report):
    before, after = (report["arms"][name] for name in ("baseline", "candidate"))
    counts = after["counts"]
    test = report["source"]["test_sentences"]
    previous, current = (arm["counts"]["unseen"] for arm in (before, after))
    change = report["paired"]["unseen"]
    if (
        previous["exact"] != change["exact_retained"] + change["exact_lost"]
        or current["exact"] != change["exact_retained"] + change["exact_gained"]
    ):
        raise ValueError("Paired changes disagree with the native outcome counts")
    types = sorted(
        report["protocol"]["gold_projection"],
        key=lambda t: -counts[f"unseen/{t}"]["gold"],
    )
    names = [
        (kind.replace("-", " ").capitalize(), counts[f"unseen/{kind}"])
        for kind in types
    ]
    variants = {
        "upper": "Uppercase",
        "lower": "Lowercase",
        "uncased": "No cased letter",
    }
    casing = [
        (label, counts[f"unseen/case-{key}"])
        for key, label in variants.items()
        if f"unseen/case-{key}" in counts
    ]
    for rows in (names, casing):
        if sum(row["gold"] for _, row in rows) != current["gold"]:
            raise ValueError(
                "Unfamiliar-mention groups do not partition the test mentions"
            )

    changed = report["paired"]["all"]["changed_sentences"]
    if not 0 <= changed <= test:
        raise ValueError("Changed sentence count exceeds the scored test population")
    if changed == 0 and (
        before["counts"] != after["counts"]
        or any(
            report["paired"]["all"][key]
            for key in (
                "exact_gained",
                "exact_lost",
                "false_positives_added",
                "false_positives_removed",
            )
        )
    ):
        raise ValueError("Identical typed spans cannot have different scored outcomes")
    versions = [arm["cli_version"].removeprefix("anno ") for arm in (before, after)]
    fp = [
        100 * arm["counts"]["all"]["false_positives"] / test for arm in (before, after)
    ]
    title = (
        f"Identical typed spans on all\n{test:,} WNUT-17 test sentences"
        if changed == 0
        else f"Typed spans changed on {changed:,} of\n{test:,} WNUT-17 test sentences"
    )
    fig = plt.figure(figsize=(4.85, 6.45))
    fig.text(
        0.045,
        0.97,
        title,
        va="top",
        fontsize=15.4,
        linespacing=1.15,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.045,
        0.852,
        f"{report['protocol']['backend'].capitalize()} {versions[0]} → {versions[1]}"
        f"  ·  threshold {report['protocol']['threshold']}",
        fontsize=11.5,
        color=MUTED,
    )
    fig.text(
        0.045,
        0.794,
        f"Unfamiliar matches  ·  {change['exact_gained']:,} gained, {change['exact_lost']:,} lost",
        fontsize=12.4,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.045,
        0.755,
        f"Unmatched predictions / 100 sentences  ·  {fp[0]:.1f} → {fp[1]:.1f}",
        fontsize=10.8,
        color=INK,
    )
    fig.legend(
        handles=[
            Patch(facecolor=color, label=label) for label, color in OUTCOMES.values()
        ],
        loc="center left",
        bbox_to_anchor=(0.025, 0.678),
        ncol=2,
        handlelength=1.0,
        columnspacing=0.9,
        handletextpad=0.4,
        fontsize=10.4,
        frameon=False,
        labelcolor=INK,
    )
    fig.text(
        0.045,
        0.602,
        f"{versions[1]} outcomes by unfamiliar gold type",
        size=12,
        color=INK,
        weight="bold",
    )
    fig.text(0.965, 0.555, "Gold", size=10.4, color=MUTED, weight="bold", ha="right")

    def row_panel(rows, bottom, height, show_ticks=False):
        ax = fig.add_axes([0.25, bottom, 0.598, height])
        for position, (label, values) in enumerate(rows):
            start = 0
            for field, (_, color) in OUTCOMES.items():
                n = values[field]
                width = 100 * n / values["gold"]
                ax.barh(
                    position, width, left=start, height=0.66, color=color, linewidth=0
                )
                if width >= 9:
                    ax.text(
                        start + width / 2,
                        position,
                        f"{n:,}",
                        va="center",
                        ha="center",
                        fontsize=10.3,
                        color="white" if field == "exact" else INK,
                    )
                start += width
            if sum(values[key] for key in OUTCOMES) != values["gold"]:
                raise ValueError(
                    f"Outcome counts do not sum to {label}'s gold denominator"
                )
            ax.text(
                1.19,
                position,
                f"{values['gold']:,}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                size=11,
                color=INK,
            )
        ax.set(xlim=(0, 100), ylim=(len(rows) - 0.5, -0.5))
        ax.set_yticks(
            range(len(rows)), [label for label, _ in rows], color=INK, fontsize=10.7
        )
        if show_ticks:
            ax.set_xticks(
                [0, 25, 50, 75, 100],
                ["0%", "25%", "50%", "75%", "100%"],
                color=MUTED,
                fontsize=10.3,
            )
        else:
            ax.set_xticks([])
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)

    row_panel(names, 0.297, 0.255)
    fig.text(
        0.045,
        0.234,
        "First cased letter of the unfamiliar mention",
        size=12,
        color=INK,
        weight="bold",
    )
    row_panel(casing, 0.123, 0.073, show_ticks=True)
    fig.text(
        0.045,
        0.031,
        "Each bar uses its own gold count.\nGroup and corporation both score as ORG.",
        fontsize=10.3,
        color=MUTED,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.receipt.read_text(encoding="utf-8"))
    fig = draw(report)
    name = (
        "wnut-stacked-comparison"
        if report["protocol"]["backend"] == "stacked"
        else "wnut-native-comparison"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / f"{name}.png", dpi=165)
    with plt.rc_context({"svg.hashsalt": "anno-wnut-native-comparison"}):
        fig.savefig(args.output / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
