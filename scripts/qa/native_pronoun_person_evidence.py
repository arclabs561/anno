"""Replay frozen exact PERSON evidence through the native supplied-mention ranker."""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import audit_gap_decisions as gap
from native_gap_margin import paired, read_panel, summaries
from native_gap_neutral_type import ROOT, file_hash, read_sources


@dataclass(frozen=True)
class Answer:
    gold: int | None
    stock: int | None
    typed: int | None
    supported: frozenset[int]


ARMS = ("stock", "typed", "unique_support")


def answer(row, baseline, candidate):
    pronoun, *names = gap.spans(row)

    def issued(output):
        bits, _ = gap.from_clusters(row, output, "mention-ranking", "supplied")
        found = {tuple(span) for group in output["clusters"] for span in group}
        if found != {pronoun, *names}:
            raise ValueError("supplied mentions changed")
        return {"10": 0, "01": 1, "11": None, "00": None}[bits[-2:]]

    evidence = {tuple(span) for span in candidate["person_spans"]}
    if not evidence <= {pronoun, *names}:
        raise ValueError("unexpected PERSON evidence outside the supplied spans")
    targets = gap.gold(row)
    if all(targets):
        raise ValueError("expected at most one antecedent")
    return Answer(targets.index(True) if any(targets) else None, issued(baseline), issued(candidate),
                  frozenset(i for i, name in enumerate(names) if name in evidence))


def predictions(cases, arm):
    if arm == "stock":
        return [c.stock for c in cases]
    if arm == "typed":
        return [c.typed for c in cases]
    if arm == "unique_support":
        return [c.typed if c.supported == {c.typed} else None for c in cases]
    raise ValueError(f"unknown arm {arm}")


def select(results):
    # Fixed order breaks exact ties in favor of stock. An increase in wrong
    # issued answers on development disqualifies an arm.
    eligible = [a for a in ARMS if results[a]["wrong"] <= results["stock"]["wrong"]]
    return max(eligible, key=lambda a: (results[a]["correct"], -results[a]["wrong"]))


def measure(rows, binary):
    before = gap.run(rows, "mention-ranking", "supplied", binary)
    after = gap.run(rows, "mention-ranking-person-evidence", "supplied", binary)
    cases = [answer(*args) for args in zip(rows, before, after, strict=True)]
    answers = {arm: predictions(cases, arm) for arm in ARMS}
    works = [r["URL"] for r in rows]
    return {
        "rows": len(rows),
        "sources": len(set(works)),
        "supported_names": dict(Counter(str(len(c.supported)) for c in cases)),
        "arms": {arm: summaries(cases, choices) for arm, choices in answers.items()},
        "paired_with_stock": {arm: paired(cases, answers["stock"], answers[arm], works) for arm in ARMS[1:]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gap", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--litbank-panel", type=Path, help="private frozen literary panel")
    parser.add_argument("--check", action="store_true", help="compare all requested panels against stored evidence")
    args = parser.parse_args()
    rows, hashes = read_sources(args.gap)
    development = measure(rows.pop("development"), args.binary)
    selected = select(development["arms"])
    panels = {"development": development}
    for split, panel in rows.items():
        panels[split] = measure(panel, args.binary)
    receipt = {"gap_sha256": hashes, "selected_on_development": selected, "panels": panels}
    if args.litbank_panel is not None:
        digest = file_hash(args.litbank_panel)
        pin = json.loads((ROOT / "docs/eval/litbank-pronoun-transfer.json").read_text())
        if digest != pin["panel_tsv_sha256"]:
            raise ValueError("literary panel differs from the frozen selection")
        panels["litbank"] = measure(read_panel(args.litbank_panel), args.binary)
        receipt["litbank_sha256"] = digest
    if args.check:
        expected = json.loads((ROOT / "docs/eval/native-pronoun-person-evidence.json").read_text())
        if args.litbank_panel is None:
            del expected["panels"]["litbank"], expected["litbank_sha256"]
        if expected != receipt:
            raise ValueError("native pronoun replay differs from the stored evidence")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
