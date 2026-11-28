"""Render HistNERo's held-out PERSON error and recall counts from the aggregate receipt."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NAME = "histnero-person-errors"
INK = "#213945"
MUTED = "#425d68"
TEAL = "#196f71"
TRACK = "#e5ecec"
OUTCOMES = {
    "no_overlap": ("No gold span overlaps", "#a3592e"),
    "same_boundary_other_type": ("Exact span, other type", "#867246"),
    "overlap_only": ("Overlaps, different span", "#716289"),
    "duplicate_exact_person": ("Repeated exact person", "#55727e"),
    "exact_person": ("Exact person", TEAL),
}


def held_out_splits(report):
    if (
        report["schema"] != "anno.histnero-ner-scoring/v1"
        or report["dataset"] != "HistNERo"
        or report["label_alignment"].get("PERS") != "PER"
    ):
        raise ValueError("Expected the HistNERo scoring receipt")
    splits = [report["splits"][split] for split in ("dev", "test")]
    for s in splits:
        outcomes, per = s["person_suggestions"], s["per_type"]["dataset"]["PER"]
        if (
            set(outcomes) != set(OUTCOMES)
            or any(type(n) is not int or n < 0 for n in outcomes.values())
            or sum(outcomes.values()) != per["predicted"]
            or outcomes["exact_person"] != per["correct"]
            or not 0 <= per["correct"] <= per["gold"]
            or min(per["predicted"], per["gold"], s["cases"]) <= 0
            or not 0 <= s["documents_also_in_training"] <= s["source_documents"]
        ):
            raise ValueError(
                "PERSON outcomes, gold counts or document overlap do not reconcile"
            )
    return splits


def draw(report):
    splits = held_out_splits(report)
    outcome_keys = [
        key
        for key in OUTCOMES
        if key != "duplicate_exact_person"
        or any(s["person_suggestions"][key] for s in splits)
    ]
    width = 326
    footer_y = 380 + 31 * (len(outcome_keys) - 4)
    height = footer_y + 51
    fig = plt.figure(figsize=(width / 72, height / 72), facecolor="white")
    cols = (161, 243)

    def text(x, y, content, *, size=10.8, color=INK, bold=False):
        fig.text(
            x / width,
            1 - y / height,
            content,
            ha="left",
            va="top",
            fontsize=size,
            color=color,
            weight="bold" if bold else "normal",
            linespacing=1.35,
        )

    def rectangle(x, y, w, h, color):
        fig.add_artist(
            Rectangle(
                (x / width, 1 - (y + h) / height),
                w / width,
                h / height,
                transform=fig.transFigure,
                linewidth=0,
                facecolor=color,
            )
        )

    def bar(x, y, n, total, color):
        rectangle(x, y, 69, 3.6, TRACK)
        rectangle(x, y, 69 * n / total, 3.6, color)

    backend = {"heuristic": "HeuristicNER"}.get(report["backend"], report["backend"])
    text(
        12,
        10,
        f"HistNERo  ·  {backend}  ·  PERS → PER",
        color=MUTED,
        bold=True,
    )
    majority = all(
        s["person_suggestions"]["no_overlap"]
        > s["per_type"]["dataset"]["PER"]["predicted"] / 2
        for s in splits
    )
    title = (
        "Most PERSON suggestions\nmiss every annotated span"
        if majority
        else "What do PERSON suggestions\nmatch on HistNERo?"
    )
    text(12, 29, title, size=17.4, bold=True)
    val, test = splits
    scope = (
        f"Publisher validation and test · {val['cases']:,} sentences each"
        if val["cases"] == test["cases"]
        else f"Publisher splits · {val['cases']:,} validation and {test['cases']:,} test sentences"
    )
    text(12, 79, scope, size=10.7, color=MUTED)

    text(12, 106, "What each PERSON suggestion matched", size=12.9, bold=True)
    for x, label in zip(cols, ("Validation", "Test")):
        text(x, 129, label, color=MUTED, bold=True)
    text(12, 148, "Total suggestions", bold=True)
    for x, s in zip(cols, splits):
        text(x, 148, f"{s['per_type']['dataset']['PER']['predicted']:,}", bold=True)
    rectangle(12, 165, width - 24, 0.8, "#bccacc")
    for i, key in enumerate(outcome_keys):
        label, color = OUTCOMES[key]
        y = 176 + 31 * i
        text(12, y, label)
        for x, s in zip(cols, splits):
            n = s["person_suggestions"][key]
            total = s["per_type"]["dataset"]["PER"]["predicted"]
            text(x, y, f"{n:,} ({n / total:.1%})")
            bar(x, y + 16, n, total, color)

    y = 176 + 31 * len(outcome_keys)
    rectangle(12, y - 3, width - 24, 0.8, "#bccacc")
    text(12, y + 8, "How many annotated people were found?", size=12.9, bold=True)
    text(12, y + 32, "Exact / gold people")
    text(12, y + 51, "Recall", bold=True)
    for x, s in zip(cols, splits):
        per = s["per_type"]["dataset"]["PER"]
        n, total = per["correct"], per["gold"]
        text(x, y + 32, f"{n:,} / {total:,}")
        text(x, y + 51, f"{n / total:.1%}", color=TEAL, bold=True)
        bar(x, y + 67, n, total, TEAL)

    text(
        12,
        footer_y + 3,
        "Source documents also present in the training split",
        color=MUTED,
    )
    text(
        12,
        footer_y + 18,
        f"{val['documents_also_in_training']:,}/{val['source_documents']:,} validation · "
        f"{test['documents_also_in_training']:,}/{test['source_documents']:,} test",
        color=MUTED,
    )
    return fig, title.replace("\n", " ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.receipt.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {"svg.hashsalt": NAME, "svg.fonttype": "none", "font.family": "DejaVu Sans"}
    ):
        fig, title = draw(report)
        fig.savefig(args.output / f"{NAME}.png", dpi=170)
        fig.savefig(
            args.output / f"{NAME}.svg", metadata={"Title": title, "Date": None}
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
