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
    from matplotlib.lines import Line2D
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
    fig = plt.figure(figsize=(8.7, 7.25), facecolor="white")
    fig.text(
        0.025,
        0.967,
        "Which unfamiliar names do WNUT-17 systems recover?",
        fontsize=17,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.025,
        0.926,
        f"{len(order)} published submissions with identical test tokens · Exact span and entity type",
        fontsize=10.5,
        color=INK,
    )

    ax = fig.add_axes([0.203, 0.546, 0.605, 0.255])
    fig.text(
        0.025,
        0.865,
        "Recall by exposure to training names",
        fontsize=13,
        weight="bold",
        color=INK,
    )
    totals = individuals[order[0]]
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=TEAL,
            label=f"Unseen ({totals['unseen']['gold']:,} gold mentions)",
        ),
        Line2D(
            [],
            [],
            marker="D",
            ls="",
            markerfacecolor="white",
            color=SEEN,
            label=f"Seen ({totals['seen']['gold']} gold mentions)",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(-0.016, 1.015),
        ncol=2,
        frameon=False,
        handletextpad=0.2,
        columnspacing=1.5,
        fontsize=9,
    )
    ax.text(
        1.20,
        1.085,
        "Unseen\nprecision",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        color=INK,
    )
    for i, name in enumerate(order):
        unseen, seen = individuals[name]["unseen"], individuals[name]["seen"]
        u, s = (100 * r["correct"] / r["gold"] for r in (unseen, seen))
        ax.plot([u, s], [i, i], color=RULE, lw=2.2, zorder=1)
        ax.scatter([u], [i], s=45, color=TEAL, zorder=3)
        ax.scatter(
            [s],
            [i],
            s=38,
            marker="D",
            facecolor="white",
            edgecolor=SEEN,
            linewidth=1.6,
            zorder=3,
        )
        ax.annotate(
            f"{u:.1f}%",
            (u, i),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            color=TEAL,
            fontsize=9,
        )
        ax.annotate(
            f"{seen['correct']}/{seen['gold']}",
            (s, i),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=SEEN,
            fontsize=9,
        )
        precision = 100 * unseen["correct"] / unseen["predicted"]
        ax.text(
            1.20,
            i,
            f"{precision:.1f}%",
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="center",
            color=INK,
            fontsize=10,
        )
    ax.set(
        ylim=(len(order) - 0.45, -0.55),
        yticks=range(len(order)),
        yticklabels=order,
        xlim=(0, 100),
        xticks=[0, 25, 50, 75, 100],
        xticklabels=["0%", "25%", "50%", "75%", "100%"],
    )
    ax.grid(axis="x", color="#ebeeee", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=5, labelcolor=INK)
    ax.spines[:].set_visible(False)

    fig.text(
        0.025,
        0.441,
        "What did the submissions find on each unseen mention?",
        fontsize=13,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.025,
        0.406,
        "Each gold mention gets the closest match made by any of them.",
        fontsize=10,
        color=INK,
    )
    bx = fig.add_axes([0.203, 0.135, 0.724, 0.24])
    kinds = sorted(TYPES, key=lambda k: -groups[k]["correct"] / groups[k]["gold"])
    for i, kind in enumerate(kinds):
        left = 0
        group = groups[kind]
        for outcome, color in zip(OUTCOMES, OUTCOME_COLORS):
            count = group[outcome]
            width = 100 * count / group["gold"]
            bx.barh(
                i,
                width,
                left=left,
                height=0.65,
                color=color,
                edgecolor="white",
                linewidth=0.7,
            )
            if width >= 4:
                bx.text(
                    left + width / 2,
                    i,
                    str(count),
                    va="center",
                    ha="center",
                    fontsize=8.5,
                    color="white" if outcome == "correct" else "#17262c",
                )
            left += width
        bx.text(
            101.5, i, str(group["gold"]), va="center", ha="left", fontsize=9, color=INK
        )
    bx.text(
        101.5,
        -0.70,
        "Gold",
        ha="left",
        va="bottom",
        fontsize=8.5,
        weight="bold",
        color=INK,
    )
    bx.set(
        ylim=(len(kinds) - 0.4, -0.55),
        yticks=range(len(kinds)),
        yticklabels=[k.title().replace("-", " ") for k in kinds],
        xlim=(0, 104),
        xticks=[0, 25, 50, 75, 100],
        xticklabels=["0%", "25%", "50%", "75%", "100%"],
    )
    bx.spines[:].set_visible(False)
    bx.tick_params(axis="both", length=0, pad=5, labelcolor=INK)
    fig.legend(
        handles=[
            Patch(facecolor=c, label=l)
            for c, l in zip(
                OUTCOME_COLORS,
                [
                    "Exact span + type",
                    "Same span, wrong type",
                    "Different overlapping span",
                    "No overlap",
                ],
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.50, 0.030),
        frameon=False,
        ncol=2,
        fontsize=9.5,
        columnspacing=1.7,
        handlelength=1.1,
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
