"""Plot same-input GLiNEREL/heuristic test comparisons from the saved NER receipt."""

import argparse
import json
import math
import re
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RECEIPT = Path(__file__).resolve().parents[1] / "docs/eval/ner-language-length.json"
LANGUAGES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ru": "Russian",
}
PANELS = {
    "Babelscape/REDFM": ("RED-FM", "paragraphs", "15–35", "70–130"),
    "Babelscape/multinerd": ("MultiNERD", "sentences", "8–16", "35–65"),
}
PER_CELL = 12
INK = "#203744"
TEAL = "#086a70"
MUTED = "#485d65"
RULE = "#d6dfe1"


class Cell(NamedTuple):
    language: str
    length: str
    delta: float
    lower: float
    upper: float
    gold: int
    gained: int
    lost: int


def f1_points(counts):
    matched, extra, missed = counts
    return 200 * matched / (2 * matched + extra + missed)


def read_panels(path):
    corpora = json.loads(path.read_text(encoding="utf-8"))["corpora"]
    panels = {}
    for revision, corpus in corpora.items():
        source = revision.split("@")[0]
        if source not in PANELS:
            continue
        if source in panels:
            raise ValueError(f"Ambiguous source revision for {source}")
        # All comparisons must come from one sampling budget and panel digest.
        panel = corpus["panels"][str(PER_CELL)]
        if not re.fullmatch(r"[0-9a-f]{64}", panel.get("panel_sha256", "")):
            raise ValueError(f"Missing or invalid panel digest for {source}")
        baseline = panel["reports"]["heuristic"]
        candidate = panel["reports"]["sauerkraut/fp32"]
        if (
            baseline["model"]["backend"] != "heuristic"
            or candidate["reference_model"] != baseline["model"]
        ):
            raise ValueError(f"Heuristic identities differ in {source}")
        if (candidate["model"]["backend"], candidate["model"]["precision"]) != (
            "sauerkraut",
            "fp32",
        ):
            raise ValueError(f"Expected the FP32 GLiNEREL candidate in {source}")
        for report, expected in ((baseline, "heuristic"), (candidate, "glinerel-fast")):
            observed = report["observed_runtimes_by_call"]
            if set(observed) != {expected} or observed[expected] <= 0:
                raise ValueError(f"Expected scored calls to {expected} in {source}")
        prior = baseline["test_correct_unmatched_missed"]
        current = candidate["test_correct_unmatched_missed"]
        paired = candidate["paired_vs_heuristic"]
        if prior.keys() != current.keys() or prior.keys() != paired.keys():
            raise ValueError(f"Paired observations have different cells in {source}")
        languages = sorted({cell.split("/")[0] for cell in paired})
        if set(paired) != {
            f"{lang}/{band}" for lang in languages for band in ("short", "long")
        }:
            raise ValueError(
                f"Expected one short and one long cell per language in {source}"
            )
        rows = []
        for lang in languages:
            for length in ("short", "long"):
                cell = f"{lang}/{length}"
                before, after, pair = prior[cell], current[cell], paired[cell]
                gold = before[0] + before[2]
                gained, lost = pair["correct_gained"], pair["correct_lost"]
                if gold != after[0] + after[2] or after[0] - before[0] != gained - lost:
                    raise ValueError(
                        f"Inconsistent gold or paired match counts for {source} {cell}"
                    )
                if not (
                    0 <= gained <= min(after[0], before[2])
                    and 0 <= lost <= min(before[0], after[2])
                ):
                    raise ValueError(
                        f"Paired gains or losses exceed available mentions for {source} {cell}"
                    )
                delta = f1_points(after) - f1_points(before)
                if not math.isclose(delta, pair["delta_f1_points"], abs_tol=0.00051):
                    raise ValueError(
                        f"Paired F1 disagrees with counts for {source} {cell}"
                    )
                low, high = pair["percentile_95"]
                if not -100 <= low <= high <= 100 or pair["defined_resamples"] != 2000:
                    raise ValueError(f"Invalid resampled interval for {source} {cell}")
                rows.append(
                    Cell(LANGUAGES[lang], length, delta, low, high, gold, gained, lost)
                )
        panels[source] = rows
    return [(*PANELS[source], panels[source]) for source in PANELS]


def plot(panels):
    fig, axes = plt.subplots(
        len(panels),
        3,
        figsize=(4.9, 7.9),
        gridspec_kw={
            "width_ratios": [1.62, 2.43, 2.58],
            "height_ratios": [len(rows) for *_, rows in panels],
        },
    )
    fig.subplots_adjust(
        left=0.042, right=0.975, bottom=0.158, top=0.80, hspace=0.65, wspace=0.09
    )
    fig.text(
        0.04, 0.964, "Change in exact-span F1", fontsize=16, weight="bold", color=INK
    )
    fig.text(
        0.04,
        0.924,
        "GLiNEREL-72m (FP32) versus Anno heuristic",
        fontsize=11.6,
        color=MUTED,
    )
    fig.text(
        0.04,
        0.895,
        f"The same {PER_CELL} test inputs per row go to both models.",
        fontsize=11,
        color=MUTED,
    )

    number_cols = [0.23, 0.52, 0.76, 1]
    for i, (name, unit, short, long, rows) in enumerate(panels):
        labels, effect, numbers = axes[i]
        for ax in axes[i]:
            ax.set_ylim(len(rows) - 0.48, -0.52)
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
        for ax in (labels, numbers):
            ax.set_xlim(0, 1)
            ax.set_xticks([])
        effect.set_xlim(-100, 100)
        effect.set_xticks([-100, 0, 100])
        effect.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{x:+.0f}" if x else "0")
        )
        effect.tick_params(axis="x", length=0, colors=MUTED, labelsize=10.3, pad=5)
        effect.set_axisbelow(True)
        effect.grid(axis="x", color=RULE, lw=0.7)
        effect.axvline(0, color="#798e94", lw=1.15, zorder=1)
        top = labels.get_position().y1
        fig.text(
            0.04,
            top + 0.058,
            f"{name}  ·  {PER_CELL * len(rows)} {unit}, {sum(row.gold for row in rows)} gold",
            fontsize=12.7,
            color=INK,
            weight="bold",
        )
        fig.text(
            0.04,
            top + 0.032,
            f"Short {short} words  ·  long {long} words",
            fontsize=10.8,
            color=MUTED,
        )
        effect.set_title("F1 change (points)", fontsize=10.3, color=MUTED, pad=2)
        for x, heading in zip(number_cols, ("Δ F1", "Gold", "Gain", "Loss")):
            numbers.text(
                x,
                1.005,
                heading,
                transform=numbers.transAxes,
                fontsize=10,
                ha="right",
                va="bottom",
                color=MUTED,
            )
        for y, (language, length, delta, low, high, gold, gained, lost) in enumerate(
            rows
        ):
            if length == "short":
                labels.text(0, y, language, fontsize=10.5, color=INK, va="center")
                if y:
                    for ax in axes[i]:
                        ax.axhline(y - 0.5, color=RULE, lw=0.6, zorder=0)
            labels.text(
                1,
                y,
                length.title(),
                fontsize=10.3,
                color=MUTED,
                ha="right",
                va="center",
            )
            effect.hlines(y, low, high, color=TEAL, lw=1.8, zorder=2)
            effect.plot(delta, y, "o", ms=5.3, color=TEAL, zorder=3)
            for x, value in zip(number_cols, (f"{delta:+.1f}", gold, gained, lost)):
                numbers.text(
                    x, y, str(value), fontsize=10.4, color=INK, ha="right", va="center"
                )
    fig.text(
        0.04,
        0.087,
        "Intervals resample whole inputs within each row\n"
        "(95% paired bootstrap, 2,000 draws).",
        fontsize=10.4,
        color=MUTED,
    )
    fig.text(
        0.04,
        0.025,
        "Gain and loss count gold mentions recovered by only one\n"
        "backend. Short and long rows use different texts.",
        fontsize=10.4,
        color=MUTED,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output directory for PNG and SVG.")
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()
    fig = plot(read_panels(args.receipt))
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"ner-language-length.{extension}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
