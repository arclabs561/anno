"""Render the benchmark chooser and CrossRE comparison from checked-in examples.

Run from a checkout with Matplotlib installed:
    python3 scripts/plot_benchmark_examples.py OUTPUT_DIRECTORY
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
INK = "#243748"
TEAL = "#176d6c"
OCHRE = "#a65b18"
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


def crossre_prompt_scores():
    report = json.loads(
        (ROOT / "docs/eval/crossre-full-prompt-comparison.json").read_text()
    )
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    fig.subplots_adjust(left=0.18, right=0.965, top=0.68, bottom=0.22)
    fig.text(
        0.04,
        0.90,
        "Which labels did the relation extractor get to see?",
        fontsize=16,
        weight="bold",
        color=INK,
    )
    domains = report["domains"]
    tick_labels = []
    for y, (domain, run) in enumerate(domains.items()):
        arms = {a["prompt"]: a for a in run["arms"] if a["backend"] == "gliner_relex"}
        fixed, gold = arms["fixed"], arms["oracle"]
        if fixed["sample"]["sample_sha256"] != gold["sample"]["sample_sha256"]:
            raise ValueError(f"Prompt arms used different examples in {domain}")
        fixed_f1, gold_f1 = [100 * arm["metrics"]["strict_f1"] for arm in (fixed, gold)]
        ax.hlines(y, fixed_f1, gold_f1, color=RULE, linewidth=3)
        ax.plot(fixed_f1, y, "o", color=TEAL, markersize=9)
        ax.plot(gold_f1, y, "D", color=OCHRE, markersize=8)
        for value, color in [(fixed_f1, TEAL), (gold_f1, OCHRE)]:
            ax.annotate(
                f"{value:.1f}%",
                (value, y),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                color=color,
                weight="bold",
            )
        ax.annotate(
            f"{gold_f1 - fixed_f1:+.1f} points",
            ((fixed_f1 + gold_f1) / 2, y),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK,
        )
        name = "AI" if domain == "ai" else domain.title()
        tick_labels.append(f"{name}\n{run['corpus']['test_docs']:,} sentences")
    ax.set(
        xlim=(0, 100),
        ylim=(len(domains) - 0.5, -0.5),
        xlabel="Strict relation F1 (%)",
        yticks=range(len(domains)),
        yticklabels=tick_labels,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e6ecee", linewidth=0.8)
    fig.legend(
        handles=[
            Line2D(
                [], [], marker="o", lw=0, color=TEAL, label="Vocabulary from training"
            ),
            Line2D(
                [],
                [],
                marker="D",
                lw=0,
                color=OCHRE,
                label="Gold labels from each test sentence",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.025, 0.85),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.04,
        0.025,
        "GLiNER-Relex on the full CrossRE AI and literature test splits. Both runs predict their own endpoints.",
        fontsize=9,
        color=INK,
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
