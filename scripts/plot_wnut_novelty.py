"""Plot WNUT-17 development and test mention novelty relative to training. Requires Matplotlib.

Counts every annotated mention against all token-joined names in training,
regardless of label. The registry designates emerging.test.annotated; the other
test representations are excluded. wnut-novelty.csv retains the test-only schema.
Data: Derczynski, Nichols, van Erp & Limsopatham (2017), CC BY 4.0.
"""

import argparse
import csv
import hashlib
from collections import Counter
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REVISION = "e52a2d2a71ac1a051ca2d532eef28653c5c02603"
SOURCE = f"https://raw.githubusercontent.com/leondz/emerging_entities_17/{REVISION}"


@lru_cache(maxsize=1)
def split_hashes():
    manifest = Path(__file__).resolve().parents[1] / "docs/eval/dataset-repo-files.csv"
    with manifest.open(newline="", encoding="utf-8") as source:
        return {
            row["path"]: row["sha256"]
            for row in csv.DictReader(source)
            if row["id"] == "Wnut17"
            and row["revision"] == REVISION
            and row["status"] == "ok"
        }


def mentions(filename, data_dir):
    if data_dir:
        raw = (data_dir / filename).read_bytes()
    else:
        with urlopen(f"{SOURCE}/{filename}", timeout=60) as response:
            raw = response.read()
    if hashlib.sha256(raw).hexdigest() != split_hashes().get(filename):
        raise ValueError(
            f"WNUT-17 split differs from the pinned file manifest: {filename}"
        )
    text = raw.decode("utf-8")
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


def novelty_counts(filename, data_dir, train, folded):
    counts = {}
    for label, name in mentions(filename, data_dir):
        if name in train:
            status = "exact"
        elif name.casefold() in folded:
            status = "case_only"
        else:
            status = "unseen"
        counts.setdefault(label, Counter())[status] += 1
    counts["All types"] = sum(counts.values(), Counter())
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-dir", type=Path, help="Use a local upstream checkout")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = {name for _, name in mentions("wnut17train.conll", args.data_dir)}
    folded = {name.casefold() for name in train}
    files = {"Development": "emerging.dev.conll", "Test": "emerging.test.annotated"}
    splits = {
        split: novelty_counts(filename, args.data_dir, train, folded)
        for split, filename in files.items()
    }
    counts = splits["Test"]
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

    with (args.output / "wnut-novelty-splits.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "split",
                "filename",
                "type",
                "mentions",
                "exact",
                "case_only",
                "unseen",
                "unseen_percent",
            ]
        )
        for split, split_counts in splits.items():
            for label in labels:
                row = split_counts[label]
                total = sum(row.values())
                writer.writerow(
                    [
                        split,
                        files[split],
                        label,
                        total,
                        row["exact"],
                        row["case_only"],
                        row["unseen"],
                        100 * row["unseen"] / total,
                    ]
                )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained")
    for ax, (split, split_counts) in zip(axes, splits.items()):
        left = [0.0] * len(labels)
        for key, title, color in [
            ("unseen", "Unseen in training", "#176b72"),
            ("case_only", "Case-only match", "#e0b260"),
            ("exact", "Exact match", "#bfcbd1"),
        ]:
            widths = [
                100 * split_counts[k][key] / sum(split_counts[k].values())
                for k in labels
            ]
            ax.barh(range(len(labels)), widths, left=left, label=title, color=color)
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
            [
                f"{k.replace('_', ' ')}  (n={sum(split_counts[k].values()):,})"
                for k in labels
            ],
        )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of annotated mentions (%)")
        ax.set_title(f"{split} · {files[split]}", loc="left", fontsize=11)
        ax.spines[["left", "right", "top"]].set_visible(False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.17), ncol=3, frameon=False)
    fig.suptitle(
        "Which WNUT-17 names appeared in training?", fontsize=16, weight="bold"
    )
    for ext in ("png", "svg"):
        fig.savefig(args.output / f"wnut-novelty.{ext}", dpi=160)


if __name__ == "__main__":
    main()
