# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay the existing competing-name selector on aligned literary reflexives."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import audit_gap_decisions as gap
from coref_gap import FC_FILES, check_file
from litbank_pronoun_transfer import prepare


def summarize(rows, answers):
    outcomes = Counter(correct=0, wrong_distractor=0, wrong_neither=0, both=0, no_named_link=0)
    for row, answer in zip(rows, answers, strict=True):
        a, b = gap.gold(row)
        if answer in ("A", "B"):
            correct = a if answer == "A" else b
            outcome = "correct" if correct else "wrong_distractor" if a or b else "wrong_neither"
        else:
            outcome = answer
        outcomes[outcome] += 1
    return outcomes


def score(rows, outputs, backend, mode):
    observations = []
    enclosed = Counter()
    for row, out in zip(rows, outputs, strict=True):
        bits, status = gap.from_clusters(row, out, backend, mode)
        if mode == "supplied" and bits is not None:
            returned = {tuple(s) for group in out["clusters"] for s in group}
            supplied = set(gap.spans(row))
            if returned - supplied or (backend == "mention-ranking" and returned != supplied):
                raise ValueError("unexpected supplied mention output")
        if bits and bits[0] == "1" and bits[3:] == "00":
            # Examine only the cluster of the exact query. A longer mention may
            # cover a candidate name without referring to the named person.
            query, *names = gap.spans(row)
            group = next(group for group in out["clusters"] if list(query) in group)
            covered = [any(s <= a and b <= e and (s, e) != (a, b) for s, e in group) for a, b in names]
            if all(covered):
                enclosed["both_names_enclosed"] += 1
            elif any(covered):
                targets = gap.gold(row)
                matched = targets[covered.index(True)]
                label = "correct" if matched else "wrong_distractor" if any(targets) else "wrong_neither"
                enclosed["one_name_enclosed_" + label] += 1
            else:
                enclosed["neither_name_enclosed"] += 1
        observations.append((row["ID"], bits or status))
    choices = {"00": "no_named_link", "10": "A", "01": "B", "11": "both"}
    answers = []
    coverage = Counter()
    for _, bits in observations:
        if bits.startswith("refused:"):
            answers.append("refused")
        else:
            answers.append(choices[bits[3:]])
            coverage[bits[:3]] += 1
    by_gold = defaultdict(Counter)
    for row, answer in zip(rows, answers, strict=True):
        a, b = gap.gold(row)
        by_gold["A" if a else "B" if b else "neither"][answer] += 1
    return {"outcomes": summarize(rows, answers), "answers_by_gold": by_gold,
            "exact_query_A_B_returned": coverage, "enclosing_names_without_exact_link": enclosed,
            "id_and_decision_sha256": gap.sha(json.dumps(observations, separators=(",", ":")).encode())}


def measure(source, gap_dir, worker, with_fcoref):
    rows, selection = prepare(source, gap_dir, pronouns={"himself", "herself"})
    selection["ordered_rows_sha256"] = gap.sha(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())
    receipt = {"selection": selection, "forms": Counter(row["Pronoun"].casefold() for row in rows),
               "nearest": summarize(rows, ["B"] * len(rows))}
    arms = [("native_supplied", "mention-ranking", "supplied"), ("native_raw", "mention-ranking", "predicted")]
    if with_fcoref:
        for name, digest in FC_FILES.items():
            check_file(Path(os.environ["FCOREF_MODEL_PATH"]) / name, digest)
        arms.append(("fcoref_raw", "fcoref", "predicted"))
    for arm, backend, mode in arms:
        receipt[arm] = score(rows, gap.run(rows, backend, mode, worker), backend, mode)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("litbank", type=Path)
    parser.add_argument("gap", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--fcoref", action="store_true", help="also replay the pinned FP32 checkpoint")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = measure(args.litbank, args.gap, args.binary, args.fcoref)
    if args.check:
        expected = json.loads((gap.ROOT / "docs/eval/litbank-reflexives.json").read_text())
        if not args.fcoref:
            del expected["fcoref_raw"]
        if expected != result:
            raise SystemExit("Literary reflexive replay differs from the recorded result")
    print(json.dumps(result, indent=2, sort_keys=True))
