"""Render the benchmark chooser and CrossRE comparison from checked-in examples.

Run from a checkout with Matplotlib installed:
    python3 scripts/plot_benchmark_examples.py OUTPUT_DIRECTORY
"""

import argparse
import csv
import json
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CROSSRE_RECEIPT = ROOT / "docs/eval/crossre-full-prompt-comparison.json"
INK = "#243748"
TEAL = "#176d6c"
RULE = "#cbd7da"


def benchmark_chooser():
    fixture = ROOT / "scripts/qa/joint_baseline"
    (source,) = json.loads((fixture / "jerex_input.json").read_text())
    (prediction,) = json.loads((fixture / "jerex_prediction.json").read_text())
    tokens = [token for sentence in source["sents"] for token in sentence]
    if source["title"] != prediction["title"] or tokens != prediction["tokens"]:
        raise ValueError("The prediction must refer to the example document")

    mentions = [" ".join(tokens[m["start"] : m["end"]]) for m in prediction["mentions"]]
    clusters = [
        [mentions[index] for index in entity["mentions"]]
        for entity in prediction["entities"]
    ]
    (relation,) = prediction["relations"]
    head, tail = clusters[relation["head"]], clusters[relation["tail"]]
    if relation["type"] != "P19":
        raise ValueError("The worked example expects the place-of-birth predicate")

    fig, ax = plt.subplots(figsize=(4.8, 5.9))
    fig.subplots_adjust(left=0.045, right=0.96, bottom=0.01, top=0.985)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(
        0, 0.95, "Which output are you scoring?", fontsize=16, weight="bold", color=INK
    )
    ax.text(
        0,
        0.895,
        "Authored Anno passage · saved JEREX prediction",
        fontsize=11,
        color=TEAL,
    )
    ax.text(
        0, 0.86, fill(source["source_text"], width=42), va="top", fontsize=12, color=INK
    )

    def section(y, title, benchmark):
        ax.axhline(y + 0.05, color=RULE, linewidth=0.9)
        ax.text(0, y, title, fontsize=13, weight="bold", color=INK)
        ax.text(1, y, benchmark, ha="right", fontsize=11, color=TEAL)

    section(0.67, "Exact span and type", "WikiGold / WNUT-17")
    by_type = {}
    for entity in prediction["entities"]:
        by_type.setdefault(entity["type"], []).extend(entity["mentions"])
    for i, (kind, indices) in enumerate(by_type.items()):
        ordered = sorted(
            indices, key=lambda index: prediction["mentions"][index]["start"]
        )
        y = 0.61 - i * 0.05
        ax.text(0, y, kind, fontsize=11, color=TEAL, weight="bold")
        ax.text(
            0.13,
            y,
            " · ".join(mentions[index] for index in ordered),
            fontsize=12,
            color=INK,
        )

    section(0.44, "Mention cluster", "LitBank")
    ax.text(0, 0.375, "One person", fontsize=11, color=TEAL)
    ax.text(0.30, 0.375, " · ".join(head), fontsize=12, color=INK)

    section(0.264, "Directed predicate and endpoints", "DocRED")
    ax.text(0, 0.198, "Head", fontsize=11, color=TEAL)
    ax.text(0.18, 0.198, " / ".join(head), fontsize=12, color=INK)
    ax.text(0.05, 0.132, "↓", fontsize=17, color=TEAL, ha="center")
    ax.text(0.18, 0.135, "place of birth (P19)", fontsize=12, color=INK)
    ax.text(0, 0.072, "Tail", fontsize=11, color=TEAL)
    ax.text(0.18, 0.072, " / ".join(tail), fontsize=12, color=INK)
    return fig


def paired_prompt_counts(domain, run, splits):
    """Exact matches, predictions and references for the three plotted conditions."""
    corpus = run["corpus"]
    pair = [a for a in run["arms"] if a["backend"] == "gliner_relex"]
    if sorted(a["prompt"] for a in pair) != ["fixed", "oracle"]:
        raise ValueError(f"Expected two paired GLiNER-Relex arms in {domain}")
    fixed, gold = sorted(pair, key=lambda arm: arm["prompt"])
    if fixed["sample"]["sample_sha256"] != gold["sample"]["sample_sha256"]:
        raise ValueError(f"Prompt arms used different examples in {domain}")
    for split in ("train", "test"):
        if splits[domain, split]["sha256"] != corpus[f"{split}_sha256"]:
            raise ValueError(
                f"Prompt survey and model trial use different {domain} {split} files"
            )
    test_row = splits[domain, "test"]
    counts = []
    for arm in (fixed, gold):
        metrics = arm["metrics"]
        for field, key in (
            ("sentences", "num_docs"),
            ("gold_relations", "num_gold_relations"),
            ("no_predicate_sentences", "gold_empty_docs"),
        ):
            if int(test_row[field]) != metrics[key]:
                raise ValueError(f"Prompt study disagrees on {domain} {key}")
        matched, predicted, reference = (
            int(metrics[key])
            for key in (
                "strict_relation_matches",
                "num_predicted_relations",
                "num_gold_relations",
            )
        )
        if (
            not 0 <= matched <= min(predicted, reference)
            or abs(2 * matched / (predicted + reference) - metrics["strict_f1"]) > 1e-7
        ):
            raise ValueError(
                f"Strict relation F1 disagrees with the scored counts in {domain}"
            )
        counts.append((matched, predicted, reference))
    matched, predicted, reference = counts[0]
    suppressed = int(fixed["metrics"]["predictions_on_gold_empty_docs"])
    if not 0 <= suppressed <= predicted - matched:
        raise ValueError(
            f"Invalid count of predictions on relation-free sentences in {domain}"
        )
    if gold["metrics"]["predictions_on_gold_empty_docs"]:
        raise ValueError(f"The empty gold prompt produced relations in {domain}")
    return counts[0], (matched, predicted - suppressed, reference), counts[1]


def crossre_prompt_scores(receipt=CROSSRE_RECEIPT):
    study = ROOT / "docs/eval"
    report = json.loads(receipt.read_text(encoding="utf-8"))
    with (study / "crossre-prompt-scope.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        splits = {(r["domain"], r["split"]): r for r in csv.DictReader(file)}
    test = [row for (_, split), row in splits.items() if split == "test"]
    none = "no_predicate_sentences"
    single = "one_predicate_type_sentences"
    many = "multiple_predicate_type_sentences"
    test.sort(
        key=lambda r: (int(r[none]) + int(r[single])) / int(r["sentences"]),
        reverse=True,
    )

    names = {"ai": "AI"}
    test_domains = [split["domain"] for split in test]
    if set(test_domains) != report["domains"].keys():
        raise ValueError("Every surveyed test domain needs its paired model result")
    rows = [
        (
            names.get(domain, domain.title()),
            paired_prompt_counts(domain, report["domains"][domain], splits),
        )
        for domain in test_domains
    ]
    pooled = [
        tuple(sum(row[1][condition][key] for row in rows) for key in range(3))
        for condition in range(3)
    ]
    rows.insert(0, ("All domains", pooled))
    fig = plt.figure(figsize=(4.85, 7.3))
    fig.text(
        0.045,
        0.954,
        "CrossRE with and without test labels",
        color=INK,
        fontsize=15.3,
        weight="bold",
    )
    fig.text(
        0.045,
        0.916,
        f"GLiNER-Relex · {sum(int(r['sentences']) for r in test):,} test sentences",
        fontsize=11.5,
        color=INK,
    )
    fig.text(
        0.045,
        0.885,
        f"Strict directed-relation F1 (%) · {pooled[0][2]:,} gold relations",
        fontsize=11.2,
        color=INK,
    )

    # The first setting can be used on unseen text. The other two require answers.
    table = fig.add_axes((0.045, 0.532, 0.91, 0.284))
    table.set(xlim=(0, 1), ylim=(len(rows) - 0.3, -1.25))
    table.axis("off")
    fixed_x, filter_x, oracle_x = 0.43, 0.675, 0.92
    fig.text(
        0.425,
        0.84,
        "At inference",
        ha="center",
        color=TEAL,
        fontsize=11.8,
        weight="bold",
    )
    fig.text(
        0.77,
        0.84,
        "Uses test labels",
        ha="center",
        color="#74446b",
        fontsize=11.8,
        weight="bold",
    )
    table.axvline(0.543, color=RULE, lw=1.1)
    for x, name in (
        (0, "Domain"),
        (fixed_x, "Fixed"),
        (filter_x, "Filtered"),
        (oracle_x, "Gold"),
    ):
        table.text(
            x,
            -0.92,
            name,
            va="center",
            ha="left" if x == 0 else "center",
            color=INK,
            fontsize=10.8,
            weight="bold",
        )
    table.axhline(-0.43, color=RULE, lw=0.85)
    for n, (label, counts) in enumerate(rows):
        weight = "bold" if n == 0 else "normal"
        table.text(0, n, label, fontsize=11.4, weight=weight, color=INK, va="center")
        for x, (matched, predicted, reference) in zip(
            (fixed_x, filter_x, oracle_x), counts
        ):
            f1 = 200 * matched / (predicted + reference)
            table.text(
                x,
                n,
                f"{f1:.1f}",
                ha="center",
                va="center",
                fontsize=12,
                color=TEAL if x == fixed_x else INK,
                weight=weight,
            )
    table.axhline(0.5, color=RULE, lw=0.85)
    fig.text(
        0.045,
        0.495,
        "Fixed uses the domain’s training vocabulary.",
        fontsize=11,
        color=TEAL,
    )
    fig.text(
        0.045,
        0.455,
        "Filtered suppresses predictions on gold-empty sentences.\n"
        "Gold supplies each sentence’s reference labels.",
        va="center",
        fontsize=10.3,
        color=INK,
    )

    fig.text(
        0.045,
        0.39,
        "How much do test labels reveal?",
        fontsize=14,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.045,
        0.355,
        "Sentences by number of distinct gold predicates",
        fontsize=11,
        color=INK,
    )
    ax = fig.add_axes((0.225, 0.075, 0.725, 0.227))
    starts = [0.0] * len(test)
    for key, label, color, ink in (
        (none, "None", "#75456f", "white"),
        (single, "One", "#daa454", INK),
        (many, "Two or more", "#dae4e6", INK),
    ):
        counts = [int(row[key]) for row in test]
        widths = [
            100 * count / int(row["sentences"]) for count, row in zip(counts, test)
        ]
        bars = ax.barh(
            range(len(test)),
            widths,
            left=starts,
            height=0.72,
            color=color,
            edgecolor="white",
            lw=0.7,
            label=label,
        )
        ax.bar_label(bars, labels=counts, label_type="center", fontsize=10.5, color=ink)
        starts = [start + width for start, width in zip(starts, widths)]
    ax.set(
        yticks=range(len(test)),
        yticklabels=[names.get(r["domain"], r["domain"].title()) for r in test],
        xlim=(0, 100),
        ylim=(len(test) - 0.5, -0.5),
        xticks=(0, 50, 100),
        xticklabels=("0%", "50%", "100%"),
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.48, 1.01),
        ncol=3,
        frameon=False,
        handlelength=1.05,
        columnspacing=0.95,
        fontsize=10.5,
    )
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, colors=INK, labelsize=10.9)
    ax.set_xlabel("Share of each domain’s test sentences", color=INK, fontsize=11)

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--crossre-receipt",
        type=Path,
        default=CROSSRE_RECEIPT,
        help="complete six-domain CrossRE replay summary",
    )
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for name, draw in [
        ("benchmark-chooser", benchmark_chooser),
        ("crossre-prompts", lambda: crossre_prompt_scores(args.crossre_receipt)),
    ]:
        figure = draw()
        for extension in ("png", "svg"):
            figure.savefig(
                output / f"{name}.{extension}",
                dpi=160,
                bbox_inches="tight",
                pad_inches=0.12,
            )
        plt.close(figure)
