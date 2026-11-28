"""Measure two Norwegian review cues on every native PER suggestion in four NorNE splits."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile

from norne_source_replay import (
    MODEL,
    ROOT,
    SOURCE_REVISION,
    canonical,
    evaluate,
    sentence,
    sha,
)

CUES = {"i", "fra"}
SPLITS = ("nob-dev", "nno-dev", "nob-test", "nno-test")
OUTCOMES = ("LOC", "GPE_LOC", "PER", "other_exact", "partial_overlap", "no_overlap")


def previous_tokens(case):
    """Keep the previous publisher token at each word start, including punctuation."""
    preceding = {}
    pos, previous = 0, None
    for form, _ in case.tokens:
        if previous is not None and case.source[pos : pos + 1] == " ":
            pos += 1
        if not case.source.startswith(form, pos):
            raise ValueError("Token start disagrees with verified source")
        preceding[pos] = previous
        previous = form.casefold()
        pos += len(form)
    if pos != len(case.source):
        raise ValueError("Token boundaries leave unused source characters")
    return preceding


def measure(cases, observations):
    counts = {arm: Counter(dict.fromkeys(OUTCOMES, 0)) for arm in ("all", "flagged")}
    within_token = 0
    predictions = hashlib.sha256()
    for case, obs in zip(cases, observations, strict=True):
        predictions.update(canonical([case.identity, obs.typed]) + b"\n")
        prior = previous_tokens(case)
        exact = {(a, b): label for a, b, label in case.source_gold}
        for a, b, kind in obs.typed:
            if kind != "PER":
                continue
            flagged = prior.get(a) in CUES
            within_token += a not in prior
            label = exact.get((a, b))
            if label is not None:
                outcome = label if label in ("LOC", "GPE_LOC", "PER") else "other_exact"
            elif any(i < b and a < j for i, j in exact):
                outcome = "partial_overlap"
            else:
                outcome = "no_overlap"
            counts["all"][outcome] += 1
            if flagged:
                counts["flagged"][outcome] += 1
    queues = {}
    for arm, groups in counts.items():
        total = groups.total()
        places = groups["LOC"] + groups["GPE_LOC"]
        queues[arm] = {
            "suggestions": total,
            "exact_places": places,
            "outcomes": dict(groups),
        }
    return {
        "sentences": len(cases),
        "sentences_without_gold": sum(not case.source_gold for case in cases),
        "per_start_inside_token": within_token,
        "ordered_predictions_sha256": predictions.hexdigest(),
        **queues,
    }


def tradeoff(receipt):
    """Treat each place mistyped as PER as a review target at either observed budget."""
    if (
        receipt["schema"] != "anno.norne-place-review/v1"
        or set(receipt["cue_tokens"]) != CUES
    ):
        raise ValueError("Expected the NorNE i/fra place-review receipt")
    series = []
    for key, label in (("nob-test", "Bokmål test"), ("nno-test", "Nynorsk test")):
        panel = receipt["splits"][key]
        points = []
        for arm, policy in (("flagged", "After i or fra"), ("all", "All PER suggestions")):
            queue = panel[arm]
            outcomes = queue["outcomes"]
            correct = queue["exact_places"]
            if (
                any(type(n) is not int or n < 0 for n in outcomes.values())
                or correct != outcomes["LOC"] + outcomes["GPE_LOC"]
                or queue["suggestions"] != sum(outcomes.values())
                or any(n > panel["all"]["outcomes"][k] for k, n in outcomes.items())
            ):
                raise ValueError(f"Review counts disagree for {key} {arm}")
            points.append({
                "label": policy,
                "correct": correct,
                "unmatched": queue["suggestions"] - correct,
            })
        series.append({
            "label": label,
            "observations": panel["all"]["suggestions"],
            "gold": panel["all"]["exact_places"],
            "points": points,
        })
    return {
        "schema": "anno.tradeoff/v1",
        "title": "How many mislabeled places does a NorNE review find?",
        "unit": "PER suggestions",
        "x_axis": "suggestions",
        "series": series,
    }


def study(archive_path, binary, replay):
    if replay["source_revision"] != SOURCE_REVISION or replay["backend"] != MODEL:
        raise ValueError("Expected the pinned NorNE heuristic source replay")
    splits = {}
    development_text = set()
    with ZipFile(archive_path) as archive, tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "datasets").mkdir()
        originals = {
            row["variant"]: row["original_path"]
            for row in json.loads(archive.read("sources.json"))["inputs"]
            if row["id"] == "NorNE"
        }
        for key in SPLITS:
            ref = replay["splits"][key]
            raw = archive.read(originals[key])
            if sha(raw) != ref["source_sha256"]:
                raise ValueError(f"Publisher digest differs for {key}")
            cases = [
                sentence(block) for block in raw.decode().rstrip("\n").split("\n\n")
            ]
            observations = evaluate(binary, cases, raw, True, work)
            measured = measure(cases, observations)
            if (
                measured["sentences"] != ref["cases"]
                or measured["all"]["suggestions"]
                != ref["after_predictions_by_type"]["PER"]
                or measured["ordered_predictions_sha256"]
                != ref["after"]["ordered_predictions_sha256"]
            ):
                raise ValueError(
                    f"Observed predictions differ from the native source replay for {key}"
                )
            if key.endswith("-dev"):
                development_text.update(case.source for case in cases)
            else:
                repeats = [
                    (case, obs)
                    for case, obs in zip(cases, observations, strict=True)
                    if case.source in development_text
                ]
                repeated = measure([c for c, _ in repeats], [obs for _, obs in repeats])
                measured["exact_text_seen_on_either_development_split"] = {
                    "sentences": repeated["sentences"],
                    "per_suggestions": repeated["all"]["suggestions"],
                    "flagged_suggestions": repeated["flagged"]["suggestions"],
                    "flagged_places": repeated["flagged"]["exact_places"],
                }
            splits[key] = {"source_sha256": ref["source_sha256"], **measured}
    return {
        "schema": "anno.norne-place-review/v1",
        "dataset": "NorNE",
        "backend": MODEL,
        "source_revision": SOURCE_REVISION,
        "cue_tokens": sorted(CUES),
        "scope": (
            "Complete development and test files for both written standards. "
            "Count every PER output with multiplicity."
        ),
        "rule": (
            "Flag a PER suggestion starting at a publisher token boundary immediately "
            "after a case-folded cue. Include punctuation as a token. Judge a flagged "
            "or unflagged suggestion as an exact place only at the full boundary of "
            "a gold LOC or GPE_LOC."
        ),
        "splits": splits,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--archive", type=Path, help="Measure a NorNE ZIP with original CoNLL-U"
    )
    inputs.add_argument(
        "--tradeoff-from", type=Path, help="Build the portable chart from an aggregate review receipt"
    )
    parser.add_argument(
        "--anno", type=Path, help="Eval-enabled Anno with NorNE support for --archive"
    )
    parser.add_argument(
        "--replay", type=Path, default=ROOT / "docs/eval/norne-source-replay.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tradeoff_from:
        result = tradeoff(json.loads(args.tradeoff_from.read_bytes()))
    else:
        if args.anno is None:
            parser.error("--archive requires --anno")
        result = study(args.archive, args.anno.resolve(), json.loads(args.replay.read_bytes()))
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
