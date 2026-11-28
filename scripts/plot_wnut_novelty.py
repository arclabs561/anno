"""Replay WNUT-17 name exposure and published test predictions. Requires Matplotlib.

    python3 scripts/plot_wnut_novelty.py OUTPUT [--data-dir PINNED_WNUT_CHECKOUT]

The publisher distributes the inputs under CC BY 4.0. See Derczynski et al.,
Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition.
Only aggregate counts and figures are written.
"""

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

REVISION = "e52a2d2a71ac1a051ca2d532eef28653c5c02603"
SOURCE = f"https://raw.githubusercontent.com/leondz/emerging_entities_17/{REVISION}"
MANIFEST = Path(__file__).resolve().parents[1] / "docs/eval/dataset-repo-files.csv"
GOLD = {
    "Train": "wnut17train.conll",
    "Development": "emerging.dev.conll",
    "Test": "emerging.test.annotated",
}
SUBMISSIONS = {
    "Arcada": (
        "arcada",
        "d9289938419aa472c35ebb531ccced91c9031b6f01ff712bc0329531df8b0b33",
    ),
    "Drexel-CCI": (
        "drexel_cci",
        "8aa39196d3dd3ba907243d4084ee52406fac195f292999579ac25faedf5394cc",
    ),
    "FLYTXT": (
        "flytxt",
        "97ccbcaa389f71eee65861c20bb4f125ef488693fabec495b6982701c2a08978",
    ),
    "MIC-CIS": (
        "mic-cis.txt",
        "fe368ea408f3227298805a8e1be85fb94f9dd1279cf81554b564d64b064e249e",
    ),
    "SJTU-Adapt": (
        "sjtu_adapt.txt",
        "6b7fb7d91e80a181ec9f7af2913a1ea458a695e31e2fd09dbfeb2cc76688e104",
    ),
    "SpinningBytes": (
        "spinningbytes.txt",
        "d369a10530893ee5ca6681793e7797100121876dbd9e65c65cad3b4710b102d0",
    ),
    "UH-RiTUAL": (
        "uh_ritual",
        "733d6fa0fd7b3929854100042e0338accceb35254677d27e1b1b17dd80b9893d",
    ),
}
TYPES = {"corporation", "creative-work", "group", "location", "person", "product"}
OUTCOMES = ("correct", "wrong_type", "partial_overlap", "no_overlap")
RESULT_FIELDS = [
    "submission",
    "scope",
    "exposure",
    "entity_type",
    "sentences",
    "tokens",
    "changed_token_forms",
    "i_tags_opening_new_spans",
    "gold",
    "predicted",
    *OUTCOMES,
    "predictions_in_gold_empty_sentences",
]
INK = "#1f3841"
TEAL = "#116b70"
SEEN = "#774c89"
RULE = "#d8e0e0"
OUTCOME_COLORS = (TEAL, "#c28d4f", "#a89abf", "#dce4e4")


@dataclass
class TaggedText:
    tokens: list
    spans: set  # (sentence, start token, exclusive end token, entity type)
    repairs: Counter  # invalid predicted I tags treated as a new chunk, by sentence


def parse_bio(text, allow_repairs=False):
    sentences, spans, repairs = [], set(), Counter()
    tokens = []
    active = None
    start = 0
    for line in [*text.splitlines(), ""]:
        token, tag = line.rsplit(None, 1) if line.strip() else ("", "O")
        prefix, _, kind = tag.partition("-")
        if tag != "O" and (prefix not in {"I", "B"} or kind not in TYPES):
            raise ValueError(f"Unknown WNUT entity tag {tag!r}")
        if prefix == "I" and kind != active:
            if not allow_repairs:
                raise ValueError(f"Inconsistent reference tag {tag!r}")
            repairs[len(sentences)] += 1
            prefix = "B"
        if prefix != "I":
            if active is not None:
                spans.add((len(sentences), start, len(tokens), active))
            active = kind if prefix == "B" else None
            start = len(tokens)
        if token:
            tokens.append(token)
        elif tokens:
            sentences.append(tuple(tokens))
            tokens = []
    return TaggedText(sentences, spans, repairs)


def load_file(filename, digest, data_dir, prediction=False):
    if data_dir:
        raw = (data_dir / filename).read_bytes()
    else:
        with urlopen(f"{SOURCE}/{filename}", timeout=60) as response:
            raw = response.read()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"WNUT-17 input differs from the pinned release: {filename}")
    return parse_bio(raw.decode("utf-8"), allow_repairs=prediction)


def surface(span, tokens):
    sentence, start, end, _ = span
    return " ".join(tokens[sentence][start:end])


def novelty(reference, exact, folded):
    counts = defaultdict(Counter)
    for span in sorted(reference.spans):
        name = surface(span, reference.tokens)
        match = (
            "exact"
            if name in exact
            else "case_only" if name.casefold() in folded else "unseen"
        )
        counts[span[-1]][match] += 1
    counts["All types"] = sum(counts.values(), Counter())
    return counts


def write_novelty(output, splits):
    """Retain both original novelty CSVs and their test-label ordering."""
    test = splits["Test"]
    labels = sorted(
        TYPES, key=lambda label: test[label]["unseen"] / sum(test[label].values())
    ) + ["All types"]
    with (output / "wnut-novelty.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["type", "test_mentions", "exact", "case_only", "unseen"])
        for label in labels:
            r = test[label]
            writer.writerow(
                [label, sum(r.values()), r["exact"], r["case_only"], r["unseen"]]
            )
    with (output / "wnut-novelty-splits.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
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
        for split, counts in splits.items():
            for label in labels:
                row = counts[label]
                n = sum(row.values())
                writer.writerow(
                    [
                        split,
                        GOLD[split],
                        label,
                        n,
                        row["exact"],
                        row["case_only"],
                        row["unseen"],
                        100 * row["unseen"] / n,
                    ]
                )


def compare_tokens(reference, submitted):
    if list(map(len, reference)) != list(map(len, submitted)):
        raise ValueError(
            "Submission and reference have different sentence or token counts"
        )
    return [
        sum(a != b for a, b in zip(gold, pred))
        for gold, pred in zip(reference, submitted)
    ]


def submission_rows(name, scope, ids, mismatches, test, prediction, folded):
    gold = {s for s in test.spans if s[0] in ids}
    predicted = {s for s in prediction.spans if s[0] in ids}
    hit = gold & predicted
    gold_sentences = {s[0] for s in gold}

    def group(spans):
        return {s for s in spans if surface(s, test.tokens).casefold() in folded}

    seen_gold, seen_pred, seen_hit = map(group, (gold, predicted, hit))
    rows = []
    for exposure, gc, pc, hc in [
        ("all", gold, predicted, hit),
        ("seen", seen_gold, seen_pred, seen_hit),
        ("unseen", gold - seen_gold, predicted - seen_pred, hit - seen_hit),
    ]:
        rows.append(
            dict(
                submission=name,
                scope=scope,
                exposure=exposure,
                entity_type="all",
                sentences=len(ids),
                tokens=sum(len(test.tokens[i]) for i in ids),
                changed_token_forms=sum(mismatches[i] for i in ids),
                i_tags_opening_new_spans=sum(prediction.repairs[i] for i in ids),
                gold=len(gc),
                predicted=len(pc),
                correct=len(hc),
                predictions_in_gold_empty_sentences=sum(
                    s[0] not in gold_sentences for s in pc
                ),
            )
        )
    return rows


def shared_outcomes(reference, predictions, folded):
    gold = {
        s
        for s in reference.spans
        if surface(s, reference.tokens).casefold() not in folded
    }
    boundaries = {s[:3] for s in predictions}
    intervals = defaultdict(set)
    for sentence, start, end in boundaries:
        intervals[sentence].add((start, end))
    counts = defaultdict(Counter)
    for span in gold:
        sentence, start, end, kind = span
        if span in predictions:
            outcome = "correct"
        elif span[:3] in boundaries:
            outcome = "wrong_type"
        elif any(
            start < other_end and other_start < end
            for other_start, other_end in intervals[sentence]
        ):
            outcome = "partial_overlap"
        else:
            outcome = "no_overlap"
        counts[kind][outcome] += 1
        counts["all"][outcome] += 1
    return counts


def replay(test, folded, data_dir):
    rows, full = [], {}
    all_sentences = set(range(len(test.tokens)))
    for name, (filename, digest) in SUBMISSIONS.items():
        prediction = load_file(
            f"submissions/{filename}", digest, data_dir, prediction=True
        )
        changes = compare_tokens(test.tokens, prediction.tokens)
        exact_sentences = {i for i, change in enumerate(changes) if change == 0}
        if exact_sentences == all_sentences:
            full[name] = prediction
            rows.extend(
                submission_rows(
                    name,
                    "verbatim full test",
                    all_sentences,
                    changes,
                    test,
                    prediction,
                    folded,
                )
            )
        else:
            rows.extend(
                submission_rows(
                    name,
                    "verbatim sentences only",
                    exact_sentences,
                    changes,
                    test,
                    prediction,
                    folded,
                )
            )
            rows.extend(
                submission_rows(
                    name,
                    "full test by token position",
                    all_sentences,
                    changes,
                    test,
                    prediction,
                    folded,
                )
            )
    if not full:
        raise ValueError("No complete submission has the registered test tokens")
    best = shared_outcomes(test, set().union(*(p.spans for p in full.values())), folded)
    for kind in ["all", *sorted(TYPES)]:
        group = best[kind]
        rows.append(
            dict(
                submission="any verbatim full-test submission",
                scope="verbatim full test",
                exposure="unseen",
                entity_type=kind,
                sentences=len(test.tokens),
                tokens=sum(map(len, test.tokens)),
                changed_token_forms=0,
                gold=sum(group.values()),
                **{outcome: group[outcome] for outcome in OUTCOMES},
            )
        )
    return rows


def plot(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    individuals = defaultdict(dict)
    groups = {}
    for r in rows:
        if r["scope"] != "verbatim full test":
            continue
        if r["submission"] == "any verbatim full-test submission":
            groups[r["entity_type"]] = r
        else:
            individuals[r["submission"]][r["exposure"]] = r
    order = sorted(
        individuals, key=lambda name: -individuals[name]["unseen"]["correct"]
    )
    fig = plt.figure(figsize=(4.85, 6.9), facecolor="white")
    total = individuals[order[0]]
    unseen_gold, seen_gold = (total[key]["gold"] for key in ("unseen", "seen"))
    fig.text(
        0.045,
        0.951,
        "Unfamiliar names in WNUT-17",
        fontsize=16,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.045,
        0.912,
        f"{len(order)} systems on identical test tokens · exact span and type",
        fontsize=10.9,
        color=INK,
    )
    fig.text(
        0.045,
        0.853,
        f"Unseen in training  ·  {unseen_gold:,} gold mentions",
        color=TEAL,
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.045,
        0.816,
        f"Seen in training  ·  one of {seen_gold} hits shifts recall by {100 / seen_gold:.1f} points.",
        fontsize=10.8,
        color=INK,
    )

    table = fig.add_axes((0.045, 0.525, 0.915, 0.269))
    table.set(xlim=(0, 1), ylim=(len(order) - 0.3, -1.55))
    table.axis("off")
    for col, label in (
        (0, "Submission"),
        (0.48, "Unseen\nrecall"),
        (0.71, "Unseen\nprecision"),
        (1.0, "Seen\nhits"),
    ):
        table.text(
            col,
            -1.18,
            label,
            ha="left" if col == 0 else "right",
            va="center",
            fontsize=10.2,
            weight="bold",
            color=INK,
        )
    table.axhline(-0.51, color=RULE, lw=0.8)
    for row, name in enumerate(order):
        unseen, seen = (individuals[name][key] for key in ("unseen", "seen"))
        table.text(0, row, name, va="center", fontsize=10.8, color=INK)
        for col, value, color in (
            (0.48, f"{100 * unseen['correct'] / unseen['gold']:.1f}%", TEAL),
            (0.71, f"{100 * unseen['correct'] / unseen['predicted']:.1f}%", TEAL),
            (1.0, f"{seen['correct']} / {seen['gold']}", SEEN),
        ):
            table.text(
                col, row, value, ha="right", va="center", fontsize=11.6, color=color
            )

    combined = groups["all"]
    fig.text(
        0.045,
        0.476,
        f"Best match from any of {len(order)} systems",
        fontsize=14,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.045,
        0.439,
        f"At least one system found {combined['correct']:,} / {combined['gold']:,} unfamiliar mentions.",
        fontsize=10.8,
        color=INK,
    )
    bx = fig.add_axes([0.253, 0.123, 0.605, 0.238])
    kinds = sorted(TYPES, key=lambda k: -groups[k]["correct"] / groups[k]["gold"])
    for row, kind in enumerate(kinds):
        group = groups[kind]
        start = 0
        for outcome, color in zip(OUTCOMES, OUTCOME_COLORS):
            n = group[outcome]
            width = 100 * n / group["gold"]
            bx.barh(
                row,
                width,
                left=start,
                height=0.68,
                color=color,
                edgecolor="white",
                lw=0.6,
            )
            if width >= 10:
                bx.text(
                    start + width / 2,
                    row,
                    str(n),
                    va="center",
                    ha="center",
                    fontsize=10,
                    color="white" if outcome == "correct" else INK,
                )
            start += width
        bx.text(
            1.15,
            row,
            str(group["gold"]),
            transform=bx.get_yaxis_transform(),
            va="center",
            ha="right",
            fontsize=11,
            color=INK,
        )
    bx.text(
        1.15,
        1.015,
        "Gold",
        transform=bx.transAxes,
        ha="right",
        fontsize=10.4,
        weight="bold",
        color=INK,
    )
    bx.set(
        ylim=(len(kinds) - 0.45, -0.55),
        yticks=range(len(kinds)),
        yticklabels=[k.replace("-", " ").capitalize() for k in kinds],
        xlim=(0, 100),
        xticks=(0, 50, 100),
        xticklabels=("0%", "50%", "100%"),
    )
    bx.spines[:].set_visible(False)
    bx.tick_params(length=0, pad=3, colors=INK, labelsize=10.4)
    fig.legend(
        handles=[
            Patch(facecolor=c, label=l)
            for c, l in zip(
                OUTCOME_COLORS,
                [
                    "Exact span and type",
                    "Same span, wrong type",
                    "Boundary overlap",
                    "No overlap",
                ],
            )
        ],
        loc="center left",
        bbox_to_anchor=(0.027, 0.052),
        frameon=False,
        ncol=2,
        handlelength=1.0,
        columnspacing=0.9,
        fontsize=10.3,
    )
    for ext in ("png", "svg"):
        fig.savefig(output / f"wnut-novelty.{ext}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--data-dir", type=Path, help="Pinned upstream checkout including submissions/"
    )
    sources.add_argument(
        "--from-results",
        type=Path,
        help="Redraw only the figure from an aggregate submission CSV",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.from_results:
        with args.from_results.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            for field in RESULT_FIELDS[4:]:
                row[field] = int(row[field]) if row[field] else None
        plot(rows, args.output)
        return
    with MANIFEST.open(newline="", encoding="utf-8") as file:
        digests = {
            r["path"]: r["sha256"]
            for r in csv.DictReader(file)
            if r["id"] == "Wnut17" and r["revision"] == REVISION and r["status"] == "ok"
        }
    splits = {
        split: load_file(filename, digests[filename], args.data_dir)
        for split, filename in GOLD.items()
    }
    exact = {surface(s, splits["Train"].tokens) for s in splits["Train"].spans}
    folded = {name.casefold() for name in exact}
    write_novelty(
        args.output,
        {key: novelty(splits[key], exact, folded) for key in ("Development", "Test")},
    )
    rows = replay(splits["Test"], folded, args.data_dir)
    with (args.output / "wnut-submission-results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output)


if __name__ == "__main__":
    main()
