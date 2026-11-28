# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay the existing competing-name selector on aligned literary reflexives."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import audit_gap_decisions as gap
from coref_gap import model_identity
from coref_corpora import litbank
from litbank_pronoun_transfer import SOURCE_SHA256, WORKS, prepare


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


def chain_index(source):
    works = litbank.load_works(source, WORKS, SOURCE_SHA256)
    result = {}
    for work, (text, coref, entities, _) in works.items():
        _, members = litbank._mentions(text, coref, entities)
        result[work] = {span: chain for chain, spans in members.items() for span in spans}
    return result


def score(rows, outputs, backend, mode, chains=None):
    observations = []
    enclosed = Counter()
    enclosing_chains = Counter()
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
                if chains is not None:
                    _, work, start, _ = row["ID"].split("-")
                    offset = int(start) - query[0]
                    a, b = names[covered.index(True)]
                    candidate = chains[work][(a + offset, b + offset)]
                    for s, e in group:
                        if s <= a and b <= e:
                            outer = chains[work].get((s + offset, e + offset))
                            relation = "unannotated" if outer is None else "same" if outer == candidate else "different"
                            enclosing_chains[relation] += 1
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
    receipt = {"outcomes": summarize(rows, answers), "answers_by_gold": by_gold,
            "exact_query_A_B_returned": coverage, "enclosing_names_without_exact_link": enclosed,
            "id_and_decision_sha256": gap.sha(json.dumps(observations, separators=(",", ":")).encode())}
    if chains is not None:
        receipt["enclosing_span_gold_chain_vs_only_covered_name"] = enclosing_chains
    return receipt


def measure(source, gap_dir, worker, with_fcoref, with_maverick):
    rows, selection = prepare(source, gap_dir, pronouns={"himself", "herself"})
    selection["ordered_rows_sha256"] = gap.sha(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())
    receipt = {"selection": selection, "forms": Counter(row["Pronoun"].casefold() for row in rows),
               "nearest": summarize(rows, ["B"] * len(rows))}
    arms = [("native_supplied", "mention-ranking", "supplied"), ("native_raw", "mention-ranking", "predicted")]
    if with_fcoref:
        model_identity("fcoref", "predicted")
        arms.append(("fcoref_raw", "fcoref", "predicted"))
    if with_maverick:
        model_identity("maverick", "predicted")
        arms.extend([("maverick_raw", "maverick", "predicted"), ("maverick_supplied", "maverick", "supplied")])
    chains = chain_index(source) if with_fcoref or with_maverick else None
    for arm, backend, mode in arms:
        receipt[arm] = score(rows, gap.run(rows, backend, mode, worker), backend, mode,
                             chains if backend != "mention-ranking" and mode == "predicted" else None)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("litbank", type=Path)
    parser.add_argument("gap", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--fcoref", action="store_true", help="also replay the pinned FP32 checkpoint")
    parser.add_argument("--maverick", action="store_true", help="also replay the pinned Maverick checkpoint")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = measure(args.litbank, args.gap, args.binary, args.fcoref, args.maverick)
    if args.check:
        expected = json.loads((gap.ROOT / "docs/eval/litbank-reflexives.json").read_text())
        if not args.fcoref:
            del expected["fcoref_raw"]
        if not args.maverick:
            expected.pop("maverick_supplied", None)
            expected.pop("maverick_raw", None)
        if expected != result:
            raise SystemExit("Literary reflexive replay differs from the recorded result")
    print(json.dumps(result, indent=2, sort_keys=True))
