"""Render the benchmark chooser and CrossRE comparison from checked-in examples.

Run from a checkout with Matplotlib installed:
    python3 scripts/plot_benchmark_examples.py OUTPUT_DIRECTORY
"""

import argparse
import csv
import json
from pathlib import Path

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

    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.025, top=0.98)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(
        0,
        0.95,
        "Which benchmark scores the output you need?",
        fontsize=18,
        weight="bold",
        color=INK,
    )
    ax.text(0, 0.84, source["source_text"], fontsize=12, color=INK)
    for x, label in [
        (0, "Unit to score"),
        (0.24, "Saved JEREX prediction"),
        (0.73, "Example benchmark"),
    ]:
        ax.text(x, 0.71, label, fontsize=10, weight="bold", color=TEAL)
    for y in [0.67, 0.49, 0.29, 0.09]:
        ax.axhline(y, color=RULE, linewidth=0.8)

    # Preserve token order when displaying the mentions of each type.
    typed_mentions = {}
    for entity in prediction["entities"]:
        typed_mentions.setdefault(entity["type"], []).extend(entity["mentions"])
    for i, (kind, indices) in enumerate(typed_mentions.items()):
        ordered = sorted(
            indices, key=lambda index: prediction["mentions"][index]["start"]
        )
        y = 0.60 - i * 0.065
        ax.text(0.24, y, kind, fontsize=9, color=TEAL, weight="bold", va="center")
        ax.text(
            0.30,
            y,
            " · ".join(mentions[index] for index in ordered),
            fontsize=11,
            color=INK,
            va="center",
        )

    for y, unit, dataset, detail in [
        (0.575, "Typed mentions", "WikiGold or WNUT-17", "Exact span and entity type"),
        (0.39, "Mention cluster", "LitBank", "Mentions of one referent"),
        (0.19, "Directed relation", "DocRED", "Predicate and endpoint clusters"),
    ]:
        ax.text(0, y, unit, fontsize=12, color=INK, weight="bold", va="center")
        ax.text(0.73, y + 0.015, dataset, fontsize=12, color=INK, weight="bold")
        ax.text(0.73, y - 0.05, detail, fontsize=10, color=INK)

    def node(x, y, label):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color=INK,
            bbox={"boxstyle": "round,pad=.35", "fc": "#e6f2ef", "ec": TEAL, "lw": 0.8},
        )

    first_name, second_name = head
    ax.plot([0.32, 0.61], [0.39, 0.39], color=TEAL, linewidth=1.3)
    node(0.32, 0.39, first_name)
    node(0.61, 0.39, second_name)

    ax.annotate(
        "",
        xy=(0.58, 0.17),
        xytext=(0.39, 0.17),
        arrowprops={"arrowstyle": "-|>", "color": TEAL, "lw": 1.4},
    )
    ax.text(0.49, 0.22, "place of birth (P19)", fontsize=9, ha="center", color=INK)
    node(0.32, 0.17, " / ".join(head))
    node(0.62, 0.17, " / ".join(tail))
    ax.text(
        0,
        0.012,
        "The passage is an Anno fixture. The benchmark names identify other sources for scoring each kind of output.",
        fontsize=9,
        color=INK,
    )
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

    fig = plt.figure(figsize=(8.8, 8.05))
    fig.text(
        0.025,
        0.96,
        "What does the gold prompt reveal?",
        color=INK,
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.025,
        0.918,
        f"Distinct reference predicates in {sum(int(r['sentences']) for r in test):,} test sentences. Segments show sentence counts.",
        color=INK,
        fontsize=10.5,
    )
    ax = fig.add_axes((0.20, 0.571, 0.775, 0.250))
    groups = [
        (none, "No predicate", "#75456f", "white"),
        (single, "One predicate type", "#dca75a", INK),
        (many, "Two or more types", "#dae4e6", INK),
    ]
    starts = [0.0] * len(test)
    for key, title, color, ink in groups:
        counts = [int(row[key]) for row in test]
        widths = [100 * n / int(r["sentences"]) for n, r in zip(counts, test)]
        bars = ax.barh(
            range(len(test)),
            widths,
            left=starts,
            height=0.69,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=title,
        )
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,}",
                ha="center",
                va="center",
                color=ink,
                fontsize=10.5,
            )
        starts = [start + width for start, width in zip(starts, widths)]
    names = {"ai": "AI"}
    ax.set_yticks(
        range(len(test)),
        [
            f"{names.get(r['domain'], r['domain'].title())}  ({int(r['sentences']):,})"
            for r in test
        ],
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(len(test) - 0.55, -0.55)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.03),
        borderaxespad=0,
        frameon=False,
        ncol=3,
        handlelength=1.1,
        columnspacing=1.6,
        fontsize=10.5,
    )
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="both", length=0, labelcolor=INK, labelsize=10.5)

    fig.text(
        0.025,
        0.463,
        "How much of the F1 gain comes from relation-free sentences?",
        color=INK,
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.025,
        0.423,
        f"GLiNER-Relex on {sum(int(r['gold_relations']) for r in test):,} held-out relations. The first row pools counts across domains.",
        color=INK,
        fontsize=10.4,
    )
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
    ax = fig.add_axes((0.196, 0.135, 0.421, 0.209))
    table = fig.add_axes((0.647, 0.135, 0.337, 0.209), sharey=ax)
    columns = [
        ("Fixed", "o", TEAL),
        ("Filtered", "s", "#b16c14"),
        ("Gold", "D", "#75456f"),
    ]
    positions = [0] + [i + 0.35 for i in range(1, len(rows))]
    for (label, counts), y in zip(rows, positions):
        rates = [
            200 * matched / (predicted + reference)
            for matched, predicted, reference in counts
        ]
        fixed, _, gold = rates
        ax.plot([fixed, gold], [y, y], color="#c1cacc", lw=2.2, zorder=1)
        for i, (rate, (_, marker, color)) in enumerate(zip(rates, columns)):
            ax.plot(
                rate,
                y,
                marker=marker,
                markersize=6.8,
                linestyle="none",
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=2 + i,
            )
            weight = "bold" if y == 0 else "normal"
            table.text(
                i + 0.5,
                y,
                f"{rate:.1f}",
                ha="center",
                va="center",
                color=INK,
                weight=weight,
                fontsize=10.5,
            )
    ax.set_yticks(positions, [label for label, _ in rows])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set(
        ylim=(positions[-1] + 0.55, -0.63), xlim=(0, 100), xticks=[0, 25, 50, 75, 100]
    )
    ax.set_xlabel("Strict F1 (%)", fontsize=10, color=INK)
    ax.grid(axis="x", color="#e4ebeb", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, colors=INK, labelsize=10.2)
    table.set_xlim(0, 3)
    table.set_axis_off()
    for i, (column, marker, color) in enumerate(columns):
        table.plot(
            (i + 0.20) / 3,
            1.068,
            marker=marker,
            color=color,
            markersize=6,
            transform=table.transAxes,
            clip_on=False,
        )
        table.text(
            (i + 0.34) / 3,
            1.068,
            column,
            transform=table.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            weight="bold",
            color=INK,
        )
    for panel in (ax, table):
        panel.axhline(0.65, color=RULE, lw=0.8)
    fig.text(
        0.025,
        0.046,
        "Fixed uses the training vocabulary. Filtered removes its predictions on sentences with no gold relation.",
        fontsize=9.5,
        color=INK,
    )
    fig.text(
        0.025,
        0.019,
        "Gold supplies the entity and relation labels from each sentence’s reference relations.",
        fontsize=9.5,
        color=INK,
    )
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
