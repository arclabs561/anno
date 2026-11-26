"""Plot WNUT-17 test mention novelty relative to training. Requires Matplotlib.

Uses the Anno registry's emerging.test.annotated file. Compare token-joined
mention surfaces across labels. Counts weight repeated test mentions by frequency.
Data: Derczynski, Nichols, van Erp & Limsopatham (2017), CC BY 4.0.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REVISION = "e52a2d2a71ac1a051ca2d532eef28653c5c02603"
SOURCE = f"https://raw.githubusercontent.com/leondz/emerging_entities_17/{REVISION}"


def mentions(filename, data_dir):
    if data_dir:
        text = (data_dir / filename).read_text(encoding="utf-8")
    else:
        with urlopen(f"{SOURCE}/{filename}", timeout=60) as response:
            text = response.read().decode("utf-8")
    words = []
    entity_type = None
    for line in [*text.splitlines(), ""]:
        token, tag = line.rsplit(None, 1) if line.strip() else ("", "O")
        if tag == "O" or tag.startswith("B-"):
            if words:
                yield entity_type, " ".join(words)
            words = []
            entity_type = None
        if tag != "O":
            prefix, label = tag.split("-", 1)
            if prefix == "I" and label != entity_type:
                raise ValueError(f"Inconsistent BIO tag {tag!r} in {filename}")
            entity_type = label
            words.append(token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-dir", type=Path, help="Use a local upstream checkout")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = {name for _, name in mentions("wnut17train.conll", args.data_dir)}
    folded = {name.casefold() for name in train}
    counts = {}
    for label, name in mentions("emerging.test.annotated", args.data_dir):
        status = (
            "exact"
            if name in train
            else "case_only" if name.casefold() in folded else "unseen"
        )
        counts.setdefault(label, Counter())[status] += 1
    counts["All types"] = sum(counts.values(), Counter())
    labels = sorted(counts, key=lambda k: counts[k]["unseen"] / sum(counts[k].values()))
    labels.remove("All types")
    labels.append("All types")
    with (args.output / "wnut-novelty.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["type", "test_mentions", "exact", "case_only", "unseen"])
        for label in labels:
            row = counts[label]
            writer.writerow(
                [
                    label,
                    sum(row.values()),
                    row["exact"],
                    row["case_only"],
                    row["unseen"],
                ]
            )

    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    left = [0.0] * len(labels)
    for key, title, color in [
        ("unseen", "Unseen in training", "#176b72"),
        ("case_only", "Case-only match", "#e0b260"),
        ("exact", "Exact match", "#bfcbd1"),
    ]:
        widths = [100 * counts[k][key] / sum(counts[k].values()) for k in labels]
        ax.barh(labels, widths, left=left, label=title, color=color)
        if key == "unseen":
            for i, width in enumerate(widths):
                ax.text(
                    width - 1,
                    i,
                    f"{width:.1f}%",
                    ha="right",
                    va="center",
                    color="white",
                )
        left = [x + y for x, y in zip(left, widths)]
    ax.set_yticks(
        range(len(labels)),
        [f"{k.replace('_', ' ')}  (n={sum(counts[k].values()):,})" for k in labels],
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of annotated test mentions (%)")
    ax.set_title(
        "Which WNUT-17 test names appeared in training?", loc="left", weight="bold"
    )
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    for ext in ("png", "svg"):
        fig.savefig(args.output / f"wnut-novelty.{ext}", dpi=160)


if __name__ == "__main__":
    main()
