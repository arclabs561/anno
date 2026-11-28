"""Plot overlaps in the three GENIA IOB2 release splits listed in the file ledger.

Run: python scripts/plot_genia_nesting.py OUTPUT_DIR [--cache-dir DOWNLOAD_CACHE]
For the original test-only analysis use --input PATH_TO_GENIA_TEST_IOB2.
For offline rendering use --from-counts docs/eval/genia-nesting-counts.json.
For a native replay, build `cargo build --locked -p anno-eval --example genia_export`
and pass --native-export target/debug/examples/genia_export.
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations, zip_longest
from pathlib import Path

from survey_repo_files import file_contents

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/eval/dataset-repo-files.csv"
SPLITS = ("train", "dev", "test")
LABELS = {"DNA", "RNA", "cell_line", "cell_type", "protein"}
# This sentence contains two gold mentions starting at the same character.
EXAMPLE_SENTENCE = 600


def release_files():
    with LEDGER.open(newline="") as file:
        release = [row for row in csv.DictReader(file) if row["id"] == "GENIANested"]
    selected = {row["split"]: row for row in release}
    if set(selected) != set(SPLITS) or len(release) != len(SPLITS):
        raise ValueError(
            "Expected exactly one GENIANested ledger file per train/dev/test split"
        )
    return selected


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
    sentences: int = 0
    sentences_with_hidden: int = 0
    depths: Counter = field(default_factory=Counter)
    pairs: Counter = field(default_factory=Counter)

    @property
    def hidden(self):
        return self.mentions - self.highlights


def sentences(data):
    for sentence in re.split(r"\n\s*\n", data.decode().strip()):
        records = [line.split() for line in sentence.splitlines() if line.strip()]
        if not records or not all(len(row) == 5 for row in records):
            raise ValueError("Expected a token and four IOB2 annotation columns")
        tokens = [record[0] for record in records]
        rows = [record[1:] for record in records]
        spans = [span for column in zip(*rows) for span in extract_spans(column)]
        yield tokens, rows, spans


def visible_spans(spans):
    """Expected earliest nonoverlapping intervals, longest first at a shared start."""
    visible = []
    last_end = 0
    for span in sorted(spans, key=lambda span: (span[0], -span[1])):
        if span[0] >= last_end:
            visible.append(span)
            last_end = span[1]
    return visible


def measure(data):
    counts = Counts()
    for _, rows, spans in sentences(data):
        counts.depths.update(sum(tag != "O" for tag in row) for row in rows)
        visible = len(visible_spans(spans))
        counts.sentences += 1
        counts.sentences_with_hidden += visible < len(spans)
        counts.mentions += len(spans)
        counts.highlights += visible
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


def read_html(fragment):
    """Read native <mark> offsets from the decoded text, excluding the <sup> labels."""
    root = ET.fromstring(f"<fragment>{fragment}</fragment>")
    parts = [root.text or ""]
    end = len(parts[0])
    marks = []
    for mark in root:
        if mark.tag != "mark" or len(mark) != 1 or mark[0].tag != "sup" or len(mark[0]):
            raise ValueError("Unexpected native HTML markup")
        label = mark[0].text
        classes = mark.get("class", "").split()
        if (
            not label
            or "entity" not in classes
            or f"entity-{label}" not in classes
            or mark[0].tail
        ):
            raise ValueError("Native HTML mark and superscript disagree")
        surface = mark.text or ""
        tail = mark.tail or ""
        marks.append((end, end + len(surface), label))
        parts.extend((surface, tail))
        end += len(surface) + len(tail)
    return "".join(parts), marks


def character_spans(tokens, token_spans):
    text = " ".join(tokens)
    # The IOB2 files carry tokens. Offsets refer to the one-space sentence,
    # with one index per Unicode scalar, including punctuation and whitespace.
    starts = []
    cursor = 0
    for token in tokens:
        starts.append(cursor)
        cursor += len(token) + 1
    gold = [
        (starts[a], starts[b - 1] + len(tokens[b - 1]), tag)
        for a, b, tag in token_spans
    ]
    return text, gold


def verify_export(number, tokens, token_spans, record):
    text, gold = character_spans(tokens, token_spans)
    if record["sentence"] != number:
        raise ValueError(f"Native output is out of order at sentence {number}")
    rendered_text, marks = read_html(record["html"])
    if rendered_text != text:
        raise ValueError(f"Native HTML changed the source text in sentence {number}")
    if marks != visible_spans(gold):
        raise ValueError(
            f"Native HTML highlighted different typed spans in sentence {number}"
        )
    rows = [json.loads(line) for line in record["jsonl"].splitlines()]
    observed = Counter(
        (r["start"], r["end"], r["type"], r["text"], r["source"]) for r in rows
    )
    expected = Counter((a, b, tag, text[a:b], str(number)) for a, b, tag in gold)
    if observed != expected:
        raise ValueError(
            f"Native JSONL entity list differs from gold in sentence {number}"
        )
    return len(gold), len(rows), len(marks)


def export_example(record):
    """Read only the two nested surfaces used in the release figure."""
    text, marks = read_html(record["html"])
    entities = [json.loads(line) for line in record["jsonl"].splitlines()]
    (outer,) = [
        r for r in entities if r["text"] == "cyclophilin gene" and r["type"] == "DNA"
    ]
    (inner,) = [
        r
        for r in entities
        if r["start"] == outer["start"]
        and r["end"] < outer["end"]
        and r["type"] == "protein"
    ]
    return {
        "sentence": record["sentence"],
        "jsonl": [
            {k: row[k] for k in ("text", "type", "start", "end")}
            for row in (outer, inner)
        ],
        "html": [
            {"text": text[a:b], "type": label, "start": a, "end": b}
            for a, b, label in marks
            if outer["start"] <= a < b <= outer["end"]
        ],
    }


def replay_native(binary, data, *, include_example=False):
    # One output record per sentence, including empty entity lists. Keep the
    # raw exports in a temporary file so large splits need no buffering in RAM.
    with tempfile.TemporaryFile() as output:
        subprocess.run([binary.resolve()], input=data, stdout=output, check=True)
        output.seek(0)
        build = json.loads(next(output))
        for field, path in {
            "example_rs_sha256": ROOT / "crates/anno-eval/examples/genia_export.rs",
            "loader_rs_sha256": ROOT / "crates/anno-eval/src/eval/loader.rs",
            "export_rs_sha256": ROOT / "crates/anno/src/export.rs",
        }.items():
            if build[field] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(
                    f"Native replay binary was built from a different {path.name}"
                )
        counts = Counter()
        example = {}
        for number, (source, native) in enumerate(
            zip_longest(sentences(data), output), 1
        ):
            if source is None or native is None:
                raise ValueError(
                    f"Native and gold sentence counts disagree at {number}"
                )
            tokens, _, spans = source
            record = json.loads(native)
            gold, rows, marks = verify_export(number, tokens, spans, record)
            if include_example and number == EXAMPLE_SENTENCE:
                example["export_example"] = export_example(record)
            counts["sentences_verified"] += 1
            counts["gold_mentions"] += gold
            counts["jsonl_rows"] += rows
            counts["html_marks"] += marks
            counts["hidden_mentions"] += gold - marks
            counts["sentences_with_hidden_mentions"] += gold > marks
            counts["sentences_without_gold_mentions"] += gold == 0
    if include_example and not example:
        raise ValueError("Missing GENIA test sentence for the export example")
    return build, {**counts, **example}


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
        if c.sentences:
            splits[split]["sentences"] = c.sentences
            splits[split]["sentences_with_hidden_mentions"] = c.sentences_with_hidden
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
        if "sentences" in entry or "sentences_with_hidden_mentions" in entry:
            total = entry.get("sentences")
            affected = entry.get("sentences_with_hidden_mentions")
            if (
                type(total) is not int
                or type(affected) is not int
                or not 0 < total <= sum(depths)
                or not 0 <= affected <= min(total, hidden)
                or bool(affected) != bool(hidden)
            ):
                raise ValueError(f"GENIA {split} has inconsistent sentence counts")
            counts.sentences = total
            counts.sentences_with_hidden = affected
        check_counts(counts, row)
        by_split[split] = counts
    return by_split


def plot(by_split, output, example):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The checked native records show how two gold mentions at one start reach each exporter.
    records = [
        (name, row)
        for name, key in (("HTML", "html"), ("JSONL", "jsonl"))
        for row in example[key]
    ]
    mentions = sum(c.mentions for c in by_split.values())
    visible = sum(c.highlights for c in by_split.values())
    scope = "GENIA test" if len(by_split) == 1 else "GENIA train, dev and test"
    ink, muted, teal, orange = "#183541", "#456371", "#126f70", "#a94e26"
    fig, ax = plt.subplots(figsize=(6.0, 5.35 + 0.26 * (len(records) + len(by_split))))
    fig.subplots_adjust(left=0.045, right=0.965, top=0.985, bottom=0.015)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def label(x, y, value, size=12.25, *, bold=False, ha="left", color=ink):
        ax.text(
            x,
            y,
            value,
            fontsize=size,
            color=color,
            weight="bold" if bold else "normal",
            ha=ha,
            va="center",
        )

    label(0, 0.955, "Which gold mentions survive export?", size=19, bold=True)
    label(
        0,
        0.9,
        f"GENIA test sentence {example['sentence']} · two nested gold mentions",
        color=muted,
    )
    columns = (0, 0.18, 0.56, 1)
    for x, title, align in zip(
        columns,
        ("Output", "Surface", "Type", "Characters"),
        ("left", "left", "left", "right"),
    ):
        label(x, 0.825, title, bold=True, ha=align)
    ax.axhline(0.795, color="#ccd9dc", lw=1)
    y = 0.755
    last = None
    for name, row in records:
        if name != last:
            label(0, y, name, bold=True, color=teal if name == "JSONL" else orange)
        label(columns[1], y, row["text"])
        label(columns[2], y, row["type"])
        label(columns[3], y, f"[{row['start']}, {row['end']})", ha="right")
        y -= 0.048
        last = name

    y -= 0.045
    label(0, y, f"Visible mentions in {scope}", size=15, bold=True)
    y -= 0.078
    for name, total, color in (("JSONL", mentions, teal), ("HTML", visible, orange)):
        label(0, y, name, bold=True)
        ax.barh(y, 0.30, left=0.19, height=0.023, color="#e4ecee")
        ax.barh(y, 0.30 * total / mentions, left=0.19, height=0.023, color=color)
        label(
            1,
            y,
            f"{total:,} / {mentions:,} ({100 * total / mentions:.1f}%)",
            ha="right",
        )
        y -= 0.05
    y -= 0.045
    label(0, y, "Where HTML hides a gold mention", size=15, bold=True)
    y -= 0.074
    for x, title, align in (
        (0, "Split", "left"),
        (0.53, "Hidden mentions", "right"),
        (1, "Affected sentences", "right"),
    ):
        label(x, y, title, bold=True, ha=align)
    y -= 0.034
    label(0.53, y, "of supplied mentions", ha="right", color=muted)
    label(1, y, "of all sentences", ha="right", color=muted)
    ax.axhline(y - 0.024, color="#ccd9dc", lw=1)
    for split, c in by_split.items():
        y -= 0.055
        label(0, y, split.capitalize())
        label(0.53, y, f"{c.hidden:,} ({100 * c.hidden / c.mentions:.1f}%)", ha="right")
        affected = (
            f"{c.sentences_with_hidden:,} ({100 * c.sentences_with_hidden / c.sentences:.1f}%)"
            if c.sentences
            else "—"
        )
        label(1, y, affected, ha="right")
    label(
        0,
        y - 0.045,
        "Unicode character intervals use an exclusive end.",
        color=muted,
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
    parser.add_argument(
        "--native-export",
        type=Path,
        help="Verify every gold sentence with the built genia_export Rust example",
    )
    args = parser.parse_args()
    if args.native_export and (args.input or args.from_counts):
        parser.error("--native-export requires all three original publisher splits")
    selected = release_files()
    native_splits = {}
    if args.from_counts:
        by_split = read_counts(args.from_counts, selected)
    else:
        by_split = {}
        native_build = None
        with tempfile.TemporaryDirectory() as tmp:
            cache = args.cache_dir or Path(tmp)
            cache.mkdir(parents=True, exist_ok=True)
            for split in ("test",) if args.input else SPLITS:
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
                if args.native_export:
                    build, verified = replay_native(
                        args.native_export, data, include_example=split == "test"
                    )
                    if native_build is not None and build != native_build:
                        raise ValueError("Native exporter build changed between splits")
                    native_build = build
                    if (
                        verified["jsonl_rows"] != counts.mentions
                        or verified["html_marks"] != counts.highlights
                        or verified["sentences_verified"] != counts.sentences
                        or verified["sentences_with_hidden_mentions"]
                        != counts.sentences_with_hidden
                    ):
                        raise ValueError(
                            f"GENIA {split} exports disagree with the figure counts"
                        )
                    native_splits[split] = {"source": source_identity(row), **verified}
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
    if native_splits:
        test_receipt = native_splits["test"]
    else:
        native = json.loads(
            LEDGER.with_name("genia-native-exports.json").read_text(encoding="utf-8")
        )
        test_receipt = native["splits"]["test"]
    if test_receipt["source"] != source_identity(selected["test"]):
        raise ValueError(
            "GENIA worked example came from a different publisher test split"
        )
    plot(by_split, args.output, test_receipt["export_example"])
    if not args.input and not args.from_counts:
        write_counts(args.output / "genia-nesting-counts.json", by_split, selected)
    if args.native_export:
        receipt = {
            "dataset_id": "GENIANested",
            "input": "All four gold IOB2 layers. Tokens within each sentence are joined by one ASCII space.",
            "native_build": native_build,
            "checks_per_sentence": [
                "Decoded anno::export::to_html markup recovers the entire input text.",
                "Every mark has the expected Unicode character interval and original label, in order.",
                "Every anno::export::to_jsonl row matches gold offsets, label, surface and sentence ID, including multiplicity.",
            ],
            "splits": native_splits,
        }
        (args.output / "genia-native-exports.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
