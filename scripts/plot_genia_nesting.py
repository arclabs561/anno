"""Plot GENIA test annotation overlap. Run: python scripts/plot_genia_nesting.py OUTPUT_DIR"""

import argparse
import hashlib
import urllib.request
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOURCE = (
    "https://raw.githubusercontent.com/thecharm/boundary-aware-nested-ner/"
    "7e87e1cb282a4441f1136c8eac5de5aedad60138/"
    "Our_boundary-aware_model/data/genia/genia.test.iob2"
)
DIGEST = "607cfab2744aff2d8af5c5613320c8138605aca64a987c8ab2d98e5dfe067691"


def extract_spans(tags):
    """Decode one IOB2 column into half-open intervals."""
    spans = []
    for index, tag in enumerate(tags):
        if tag == "O":
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B":
            spans.append((index, index + 1, label))
        elif prefix == "I" and spans and spans[-1][1:] == (index, label):
            spans[-1] = (spans[-1][0], index + 1, label)
        else:
            raise ValueError(f"Invalid continuation at token {index}: {tag}")
    return spans


def plot(data, output):
    depths = Counter()
    pairs = Counter()
    nested_mentions = total_mentions = 0
    for sentence in data.decode().strip().split("\n\n"):
        rows = [line.split()[1:] for line in sentence.splitlines() if line.strip()]
        if not all(len(row) == 4 for row in rows):
            raise ValueError("Expected four IOB2 annotation layers")
        depths.update(sum(tag != "O" for tag in row) for row in rows)
        spans = [span for column in zip(*rows) for span in extract_spans(column)]
        total_mentions += len(spans)
        contained = set()
        for first, second in combinations(enumerate(spans), 2):
            # Sort the longer span first. Equal intervals are not strict containment.
            (_, outer), (inner_id, inner) = sorted(
                (first, second), key=lambda item: item[1][0] - item[1][1]
            )
            if outer[:2] != inner[:2] and outer[0] <= inner[0] and inner[1] <= outer[1]:
                pairs[(outer[2], inner[2])] += 1
                contained.add(inner_id)
        nested_mentions += len(contained)

    fig, (depth_ax, pair_ax) = plt.subplots(
        1, 2, figsize=(11, 5), layout="constrained", gridspec_kw={"width_ratios": [1, 1.35]}
    )
    for ax in (depth_ax, pair_ax):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    active = sum(count for depth, count in depths.items() if depth)
    levels = sorted(depth for depth in depths if depth)
    shares = [100 * depths[level] / active for level in levels]
    bars = depth_ax.barh([str(n) for n in levels], shares, color="#187673")
    depth_ax.bar_label(bars, labels=[f"{p:.1f}%" for p in shares], padding=4)
    depth_ax.invert_yaxis()
    depth_ax.set(xlim=(0, 101), xlabel=f"Share of {active:,} annotated tokens", ylabel="Active layers")
    depth_ax.set_title("How often do layers overlap?", loc="left")
    common = pairs.most_common(7)
    bars = pair_ax.barh([f"{a} → {b}" for (a, b), _ in common], [n for _, n in common], color="#476585")
    pair_ax.bar_label(bars, padding=4)
    pair_ax.invert_yaxis()
    pair_ax.margins(x=.15)
    pair_ax.set_xlabel("Strict containment pairs (outer → inner)")
    pair_ax.set_title("Most frequent label combinations", loc="left")
    fig.suptitle("Nested entity annotations in the GENIA test file", fontsize=16)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(output / f"anno-genia-nesting.{ext}", dpi=160)
    plt.close(fig)
    print(f"{total_mentions=} {nested_mentions=} {depths=} {pairs=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input", type=Path, help="Local copy of the pinned test file")
    args = parser.parse_args()
    if args.input:
        contents = args.input.read_bytes()
    else:
        with urllib.request.urlopen(SOURCE, timeout=30) as response:
            contents = response.read()
    if hashlib.sha256(contents).hexdigest() != DIGEST:
        raise ValueError("GENIA test file differs from the pinned snapshot")
    plot(contents, args.output)
