"""Plot overlaps in the three GENIA IOB2 release splits listed in the file ledger.

Run: python scripts/plot_genia_nesting.py OUTPUT_DIR [--cache-dir DOWNLOAD_CACHE]
For the original test-only analysis use --input PATH_TO_GENIA_TEST_IOB2.
For offline rendering use --from-counts docs/eval/genia-nesting-counts.json.
"""

import argparse
import csv
import hashlib
import json
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from survey_repo_files import file_contents

LEDGER = Path(__file__).resolve().parents[1] / "docs/eval/dataset-repo-files.csv"
SPLITS = ("train", "dev", "test")
LABELS = {"DNA", "RNA", "cell_line", "cell_type", "protein"}


def extract_spans(tags):
    """Decode one IOB2 column into half-open token intervals."""
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


@dataclass
class Counts:
    mentions: int = 0
    highlights: int = 0
    contained: int = 0
    depths: Counter = field(default_factory=Counter)
    pairs: Counter = field(default_factory=Counter)

    @property
    def hidden(self):
        return self.mentions - self.highlights


def measure(data):
    counts = Counts()
    for sentence in re.split(r"\n\s*\n", data.decode().strip()):
        rows = [line.split()[1:] for line in sentence.splitlines() if line.strip()]
        if not rows or not all(len(row) == 4 for row in rows):
            raise ValueError("Expected a token and four IOB2 annotation columns")
        counts.depths.update(sum(tag != "O" for tag in row) for row in rows)
        spans = [span for column in zip(*rows) for span in extract_spans(column)]
        counts.mentions += len(spans)
        # Token interval ordering matches character interval ordering in
        # anno::export::to_html. Every span comes from this sentence.
        highlight_end = 0
        for start, end, _ in sorted(spans, key=lambda span: (span[0], -span[1])):
            if start >= highlight_end:
                counts.highlights += 1
                highlight_end = end
        contained = set()
        for first, second in combinations(enumerate(spans), 2):
            (_, outer), (inner_id, inner) = sorted(
                (first, second), key=lambda item: item[1][0] - item[1][1]
            )
            if outer[:2] != inner[:2] and outer[0] <= inner[0] and inner[1] <= outer[1]:
                counts.pairs[(outer[2], inner[2])] += 1
                contained.add(inner_id)
        counts.contained += len(contained)
    return counts


def source_identity(row):
    return {
        "url": row["url"],
        "sha256": row["sha256"],
        "token_records": int(row["rows"]),
        "tagged_tokens": int(row["tagged"]),
    }


def check_counts(counts, row):
    tokens = sum(counts.depths.values())
    tagged = tokens - counts.depths[0]
    if (
        row["status"] != "ok"
        or tokens != int(row["tokens"])
        or tokens != int(row["rows"])
        or tagged != int(row["tagged"])
    ):
        raise ValueError(
            f"GENIA {row['split']} token denominators disagree with the ledger"
        )
    if not (
        0 < counts.highlights <= min(tagged, counts.mentions)
        and counts.contained <= min(counts.mentions, sum(counts.pairs.values()))
        and counts.mentions <= sum(depth * n for depth, n in counts.depths.items())
    ):
        raise ValueError(f"GENIA {row['split']} mention counts are inconsistent")


def write_counts(path, by_split, selected):
    splits = {}
    for split in SPLITS:
        c = by_split[split]
        pairs = {}
        for (outer, inner), n in sorted(c.pairs.items()):
            pairs.setdefault(outer, {})[inner] = n
        splits[split] = {
            "source": source_identity(selected[split]),
            "mentions": c.mentions,
            "highlights": c.highlights,
            "hidden_mentions": c.hidden,
            "strictly_contained_mentions": c.contained,
            "tokens_by_active_layer_count": [c.depths[level] for level in range(5)],
            "strict_pairs_by_outer_inner_label": pairs,
        }
    path.write_text(
        json.dumps({"dataset_id": "GENIANested", "splits": splits}, indent=2) + "\n",
        encoding="utf-8",
    )


def read_counts(path, selected):
    document = json.loads(path.read_text(encoding="utf-8"))
    splits = document["splits"]
    if document["dataset_id"] != "GENIANested" or set(splits) != set(SPLITS):
        raise ValueError(
            "Expected counts for the GENIANested train, dev and test splits"
        )
    by_split = {}
    for split in SPLITS:
        entry = splits[split]
        row = selected[split]
        if entry["source"] != source_identity(row):
            raise ValueError(
                f"GENIA {split} counts identify a different source from the ledger"
            )
        depths = entry["tokens_by_active_layer_count"]
        if not isinstance(depths, list) or len(depths) != 5:
            raise ValueError(
                f"GENIA {split} requires token counts at depths zero through four"
            )
        pairs = Counter()
        for outer, inner_counts in entry["strict_pairs_by_outer_inner_label"].items():
            if (
                outer not in LABELS
                or not isinstance(inner_counts, dict)
                or not set(inner_counts) <= LABELS
            ):
                raise ValueError(f"GENIA {split} contains an unknown nested label pair")
            for inner, n in inner_counts.items():
                pairs[(outer, inner)] = n
        mentions, visible, hidden, contained = (
            entry[key]
            for key in (
                "mentions",
                "highlights",
                "hidden_mentions",
                "strictly_contained_mentions",
            )
        )
        if not all(
            type(n) is int and n >= 0
            for n in (mentions, visible, hidden, contained, *depths, *pairs.values())
        ):
            raise ValueError(f"GENIA {split} requires nonnegative integer counts")
        if visible + hidden != mentions:
            raise ValueError(
                f"GENIA {split} hidden and highlighted mentions must sum to the total"
            )
        counts = Counts(
            mentions=mentions,
            highlights=visible,
            contained=contained,
            depths=Counter(dict(enumerate(depths))),
            pairs=pairs,
        )
        check_counts(counts, row)
        by_split[split] = counts
    return by_split


def plot(by_split, output):
    depths = sum((counts.depths for counts in by_split.values()), Counter())
    pairs = sum((counts.pairs for counts in by_split.values()), Counter())
    mentions = sum(c.mentions for c in by_split.values())
    hidden = sum(c.hidden for c in by_split.values())
    scope = "GENIA test split" if len(by_split) == 1 else "GENIA train, dev and test"
    fig = plt.figure(figsize=(12, 6.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0])
    hidden_ax = fig.add_subplot(grid[0, 0])
    depth_ax = fig.add_subplot(grid[1, 0])
    pair_ax = fig.add_subplot(grid[:, 1])
    for ax in (hidden_ax, depth_ax, pair_ax):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    shares = [100 * c.hidden / c.mentions for c in by_split.values()]
    bars = hidden_ax.barh(list(by_split), shares, color="#b86536", height=0.57)
    hidden_ax.bar_label(
        bars,
        padding=4,
        labels=[
            f"{c.hidden:,} of {c.mentions:,} ({rate:.1f}%)"
            for c, rate in zip(by_split.values(), shares)
        ],
    )
    hidden_ax.set_xlim(0, max(shares) * 1.95)
    hidden_ax.invert_yaxis()
    hidden_ax.set_xlabel("Share of reference mentions (%)")
    hidden_ax.set_title("Mentions the HTML preview would hide", loc="left")

    active = sum(count for depth, count in depths.items() if depth)
    levels = sorted(depth for depth in depths if depth)
    depth_shares = [100 * depths[level] / active for level in levels]
    bars = depth_ax.barh(
        [f"{n} layer" if n == 1 else f"{n} layers" for n in levels],
        depth_shares,
        color="#187673",
        height=0.57,
    )
    depth_ax.bar_label(bars, labels=[f"{p:.2f}%" for p in depth_shares], padding=4)
    depth_ax.set_xlim(0, 108)
    depth_ax.invert_yaxis()
    depth_ax.set_xlabel(f"Share of {active:,} annotated tokens")
    depth_ax.set_title("How many annotation layers share a token?", loc="left")

    common = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))[:7]
    bars = pair_ax.barh(
        [f"{outer} → {inner}" for (outer, inner), _ in common],
        [count for _, count in common],
        color="#476585",
        height=0.6,
    )
    pair_ax.bar_label(bars, labels=[f"{n:,}" for _, n in common], padding=4)
    pair_ax.invert_yaxis()
    pair_ax.margins(x=0.18)
    pair_ax.set_xlabel("Strict containment pairs (outer → inner)")
    pair_ax.set_title("Which mentions sit inside others?", loc="left")
    fig.suptitle(
        f"{scope} · {hidden:,} of {mentions:,} mentions hidden by HTML highlighting",
        fontsize=15,
    )
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "anno-genia-nesting.png", dpi=160)
    with matplotlib.rc_context({"svg.hashsalt": "anno-genia-nesting"}):
        fig.savefig(output / "anno-genia-nesting.svg", dpi=160, metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input",
        type=Path,
        help="Local copy of the pinned test file for test-only replay",
    )
    input_group.add_argument(
        "--from-counts",
        type=Path,
        help="Render the full release figure from verified split aggregates",
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Reuse files cached by survey_repo_files.py"
    )
    args = parser.parse_args()
    with LEDGER.open(newline="") as file:
        release = [row for row in csv.DictReader(file) if row["id"] == "GENIANested"]
    selected = {row["split"]: row for row in release}
    if set(selected) != set(SPLITS) or len(release) != len(SPLITS):
        raise ValueError(
            "Expected exactly one GENIANested ledger file per train/dev/test split"
        )
    if args.from_counts:
        by_split = read_counts(args.from_counts, selected)
    else:
        by_split = {}
        with tempfile.TemporaryDirectory() as tmp:
            cache = args.cache_dir or Path(tmp)
            cache.mkdir(parents=True, exist_ok=True)
            for split in (("test",) if args.input else SPLITS):
                row = selected[split]
                data = (
                    args.input.read_bytes() if args.input else file_contents(row, cache)
                )
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError(
                        f"GENIA {split} file differs from the verified ledger"
                    )
                counts = measure(data)
                check_counts(counts, row)
                by_split[split] = counts
    print(
        "split\ttokens\ttagged_tokens\tmentions\thighlights\thidden\tstrictly_contained"
    )
    for split, c in by_split.items():
        tokens = sum(c.depths.values())
        print(
            split,
            tokens,
            tokens - c.depths[0],
            c.mentions,
            c.highlights,
            c.hidden,
            c.contained,
            sep="\t",
        )
    plot(by_split, args.output)
    if not args.input and not args.from_counts:
        write_counts(args.output / "genia-nesting-counts.json", by_split, selected)


if __name__ == "__main__":
    main()
