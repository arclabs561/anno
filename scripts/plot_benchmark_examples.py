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


def paired_prompt_f1(domain, run, splits):
    corpus = run["corpus"]
    arms = {a["prompt"]: a for a in run["arms"] if a["backend"] == "gliner_relex"}
    fixed, gold = arms["fixed"], arms["oracle"]
    if fixed["sample"]["sample_sha256"] != gold["sample"]["sample_sha256"]:
        raise ValueError(f"Prompt arms used different examples in {domain}")
    for split in ("train", "test"):
        if splits[domain, split]["sha256"] != corpus[f"{split}_sha256"]:
            raise ValueError(
                f"Prompt survey and model trial use different {domain} {split} files"
            )
    test_row = splits[domain, "test"]
    metrics = fixed["metrics"]
    for field, key in (
        ("sentences", "num_docs"),
        ("gold_relations", "num_gold_relations"),
        ("no_predicate_sentences", "gold_empty_docs"),
    ):
        if int(test_row[field]) != metrics[key] or metrics[key] != gold["metrics"][key]:
            raise ValueError(f"Prompt study disagrees on {domain} {key}")
    filtered = (
        metrics["num_predicted_relations"] - metrics["predictions_on_gold_empty_docs"]
    )
    rescored = (
        2
        * metrics["strict_relation_matches"]
        / (filtered + metrics["num_gold_relations"])
    )
    return [
        100 * metrics["strict_f1"],
        100 * rescored,
        100 * gold["metrics"]["strict_f1"],
    ]


def crossre_prompt_scores():
    study = ROOT / "docs/eval"
    report = json.loads((study / "crossre-full-prompt-comparison.json").read_text())
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

    fig = plt.figure(figsize=(8.8, 7.2))
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
    ax = fig.add_axes((0.20, 0.535, 0.775, 0.286))
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
        0.413,
        "How much of the F1 gain comes from relation-free sentences?",
        color=INK,
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.025,
        0.373,
        "Strict GLiNER-Relex relation F1 (%) on the domains with paired predictions.",
        color=INK,
        fontsize=10.5,
    )
    panels = fig.add_gridspec(
        1,
        len(report["domains"]),
        left=0.31,
        right=0.946,
        top=0.29,
        bottom=0.105,
        wspace=0.17,
    )
    row_labels = [
        "Training vocabulary",
        "Fixed predictions on\nrelation-bearing sentences",
        "Per-sentence gold vocabulary",
    ]
    for panel, (domain, run) in zip(panels, report["domains"].items()):
        values = paired_prompt_f1(domain, run, splits)
        ax = fig.add_subplot(panel)
        bars = ax.barh(
            range(3),
            values,
            height=0.47,
            color=[TEAL, "#93bab6", "#75456f"],
            edgecolor="none",
        )
        ax.bar_label(
            bars,
            labels=[f"{v:.1f}" for v in values],
            padding=4,
            weight="bold",
            color=INK,
            fontsize=10.5,
        )
        ax.set_title(
            names.get(domain, domain.title()),
            loc="left",
            color=INK,
            weight="bold",
            fontsize=12,
            pad=4,
        )
        ax.set(xlim=(0, 70), ylim=(2.5, -0.5), xticks=[0, 20, 40, 60])
        ax.set_yticks(range(3), row_labels if panel.is_first_col() else [])
        ax.tick_params(axis="both", length=0, labelcolor=INK, labelsize=10)
        ax.spines[:].set_visible(False)
        ax.grid(axis="x", color="#e4ebeb", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.text(
        0.025,
        0.028,
        "The middle row removes only fixed-run predictions on test sentences with no reference relation.",
        color=INK,
        fontsize=9.6,
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    for name, draw in [
        ("benchmark-chooser", benchmark_chooser),
        ("crossre-prompts", crossre_prompt_scores),
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
