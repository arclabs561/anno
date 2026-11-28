"""Select a native supplied-pronoun abstention rule on development data, then apply it unchanged."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from dataclasses import dataclass
import json
from math import isfinite
import os
from pathlib import Path
import subprocess
import tempfile

from audit_gap_decisions import gold, payload, spans
from coref_gap import check_file
from native_gap_neutral_type import file_hash

DEV_SHA = "b9a01434fcf58d8c2f9bc762480c27e58ce466cf1ffe8b09cfecbc7a20d2d634"
MARGINS = (0, .025, .05, .075, .1, .125, .15, .175, .2, .225, .25, .3, .4)
FORMAT = "anno.qa.supplied-pronoun-margin.v1"
COMPARISON = Path(__file__).resolve().parents[2] / "docs/eval/native-gap-margin-comparison.json"
TSV_FIELDS = {"ID", "Text", "URL", "A", "A-coref", "A-offset", "B", "B-coref", "B-offset",
              "Pronoun", "Pronoun-offset"}


@dataclass(frozen=True)
class Lookup:
    gold: int | None
    issued: int | None
    direct: int | None
    margin: float | None
    above_threshold: float | None
    eligible: int
    gold_stage: str
    group_arity: int

    def gated(self, cutoff):
        if self.direct == self.issued and self.margin is not None and self.margin >= cutoff:
            return self.issued
        return None


def lookup(row, trace):
    if trace["id"] != row["ID"]:
        raise ValueError("trace row ID differs")
    targets = spans(row)
    ordered = sorted(targets)
    if len(set(ordered)) != 3 or [m[:2] for m in trace["members"]] != [list(s) for s in ordered]:
        raise ValueError("trace does not return the three unique supplied spans")
    groups = [m[2] for m in trace["members"]]
    pr, a, b = [ordered.index(target) for target in targets]
    names = (a, b)
    supported = [j for j in names if groups[j] == groups[pr]]
    issued = names.index(supported[0]) if len(supported) == 1 else None
    labels = gold(row)
    target = labels.index(True) if any(labels) else None
    decision = trace["decisions"][pr]
    scored = decision["candidates"]
    # The native scorer visits newer antecedents first and requires a strict threshold pass.
    visits = [j for j in range(pr - 1, -1, -1)
              if ordered[pr][0] - ordered[j][1] <= trace["max_distance"]]
    if [j for j, _, _ in scored] != visits or not all(
        isfinite(s) and isfinite(t) for _, s, t in scored
    ) or len({t for _, _, t in scored}) > 1:
        raise ValueError("eligibility, threshold or visit order disagrees with native trace")
    ranked = sorted(scored, key=lambda candidate: -candidate[1])
    winner = None
    margin = None
    above_threshold = None
    if ranked:
        best, score, threshold = ranked[0]
        if score > threshold:
            winner = best
            above_threshold = score - threshold
            rival = max(threshold, ranked[1][1]) if len(ranked) > 1 else threshold
            margin = score - rival
    if decision["selected"] != winner:
        raise ValueError("reconstructed winner disagrees with native direct decision")
    if winner is not None and groups[winner] != groups[pr]:
        raise ValueError("direct edge is missing from the final partition")
    if target is None:
        stage = "neither_gold"
    elif names[target] > pr:
        stage = "gold_follows_pronoun"
    elif names[target] not in visits:
        stage = "gold_outside_distance"
    elif ranked[0][0] != names[target]:
        scores = {j: score for j, score, _ in scored}
        stage = "gold_lost_tie" if scores[names[target]] == ranked[0][1] else "gold_outscored"
    elif winner is None:
        stage = "gold_first_below_or_at_threshold"
    else:
        stage = "gold_selected_directly"
    return Lookup(target, issued, names.index(winner) if winner is not None else None,
                  margin, above_threshold, len(visits), stage, len(supported))


def run(rows, worker, *, book_distance=False):
    with tempfile.TemporaryDirectory() as directory:
        source, output = (Path(directory) / name for name in ("input.jsonl", "trace.jsonl"))
        with source.open("w", encoding="utf-8") as f:
            for row in rows:
                print(json.dumps(payload(row, "supplied")), file=f)
        env = dict(os.environ, ANNO_GAP_TRACE_INPUT=str(source), ANNO_GAP_TRACE_OUTPUT=str(output))
        test = "trace_stock_book_distance" if book_distance else "trace_supplied_ranking"
        subprocess.run([str(worker.resolve()), f"gap_supplied_trace::{test}", "--ignored"],
                       env=env, check=True, stdout=subprocess.DEVNULL)
        with output.open(encoding="utf-8") as f:
            return [lookup(row, json.loads(trace)) for row, trace in zip(rows, f, strict=True)]


def summaries(rows, predictions):
    result = Counter(correct=0, wrong=0, abstained=0, wrong_when_neither=0, wrong_distractor=0)
    for row, decision in zip(rows, predictions, strict=True):
        if decision is None:
            result["abstained"] += 1
        elif row.gold == decision:
            result["correct"] += 1
        else:
            result["wrong"] += 1
            result["wrong_when_neither" if row.gold is None else "wrong_distractor"] += 1
    return dict(result)


def diagnostics(rows):
    return {"gold_stage": dict(Counter(row.gold_stage for row in rows)),
            "eligible_names": dict(Counter(str(row.eligible) for row in rows)),
            "group_links": dict(Counter(str(row.group_arity) for row in rows)),
            "unique_link_without_direct_edge": sum(row.issued is not None and row.direct is None for row in rows)}


def best_feasible(baseline, grid):
    if [r["margin"] for r in grid] != list(MARGINS):
        raise ValueError("development margin grid differs")
    best = max((r for r in grid if r["wrong"] <= baseline["wrong"]),
               key=lambda r: (r["correct"], -r["wrong"], r["margin"]), default=None)
    return best


def choose(stock, neutral):
    baseline = summaries(stock, [r.issued for r in stock])
    grid = [{**summaries(neutral, [r.gated(m) for r in neutral]), "margin": m} for m in MARGINS]
    best = best_feasible(baseline, grid)
    rule = {"arm": "stock"}
    if best and (best["correct"], -best["wrong"]) > (baseline["correct"], -baseline["wrong"]):
        rule = {"arm": "neutral_direct_unique", "margin": best["margin"]}
    return rule, grid


def paired(stock, before, after, works):
    counts = Counter()
    by_work = defaultdict(Counter)
    for row, prev, curr, work in zip(stock, before, after, works, strict=True):
        def verdict(answer):
            return "abstain" if answer is None else ("correct" if answer == row.gold else "wrong")
        old, new = verdict(prev), verdict(curr)
        counts[f"{old}_to_{new}"] += 1
        by_work[work][old] -= 1
        by_work[work][new] += 1
    work_changes = {}
    for label in ("correct", "wrong"):
        directions = Counter("increased" if change[label] > 0 else "decreased" if change[label] < 0 else "equal"
                             for change in by_work.values())
        work_changes[label] = dict(directions)
    return {"queries": dict(counts), "changed_issued_target": sum(a != b for a, b in zip(before, after, strict=True)),
            "works": work_changes}


def measure(stock, neutral, rule, works, comparator):
    if rule == {"arm": "stock"}:
        predictions = [r.issued for r in stock]
    elif (set(rule) == {"arm", "margin"} and rule["arm"] == "neutral_direct_unique"
          and type(rule["margin"]) in (int, float) and rule["margin"] in MARGINS):
        predictions = [r.gated(rule["margin"]) for r in neutral]
    else:
        raise ValueError("unsupported frozen rule")
    stock_answers = [r.issued for r in stock]
    compared = [r.gated(comparator) for r in neutral]
    # Is a rival ever decisive after the same lead over abstaining has already been met?
    rival_vetoes = sum(r.issued is not None and r.direct == r.issued and r.above_threshold is not None
                       and r.above_threshold >= comparator
                       and answer is None for r, answer in zip(neutral, compared, strict=True))
    return {"rows": len(stock), "gold_links": sum(r.gold is not None for r in stock),
            "stock": summaries(stock, stock_answers),
            "neutral": summaries(neutral, [r.issued for r in neutral]),
            "frozen": summaries(stock, predictions), "stock_to_frozen": paired(stock, stock_answers, predictions, works),
            "comparison_margin": comparator, "comparison_vetoes_due_to_rival": rival_vetoes,
            "neutral_at_comparison_margin": summaries(neutral, compared),
            "stock_to_comparison_margin": paired(stock, stock_answers, compared, works),
            "stock_diagnostics": diagnostics(stock), "neutral_diagnostics": diagnostics(neutral)}


def read_panel(path):
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not TSV_FIELDS.issubset(columns):
            raise ValueError("missing or duplicate GAP column")
        rows = list(reader)
    ids = set()
    for row in rows:
        if (None in row or any(row[k] in (None, "") for k in TSV_FIELDS)
                or row["ID"] in ids or not row["ID"].strip() or not row["URL"].strip()):
            raise ValueError("missing field, duplicate ID or malformed TSV row")
        ids.add(row["ID"])
        coords = spans(row)
        if len(set(coords)) != 3 or any(start < 0 or start >= end for start, end in coords):
            raise ValueError("expected three distinct valid character ranges")
        if all(gold(row)):
            raise ValueError("both candidates labeled correct")
    if not rows:
        raise ValueError("empty panel")
    return rows


def comparison_margin(receipt, policy_hash, commitment):
    evidence = {"stock": receipt["stock"], "grid": receipt["development_grid"]}
    digest = hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (commitment["policy_sha256"] != policy_hash or commitment["development_evidence_sha256"] != digest
            or commitment["development_sha256"] != receipt["source_sha256"]):
        raise ValueError("development comparison differs from its frozen commitment")
    best = best_feasible(evidence["stock"], evidence["grid"])
    if best is None or best["margin"] != commitment["margin"]:
        raise ValueError("inconsistent development comparator")
    return best["margin"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("freeze", "score"):
        command = actions.add_parser(action)
        for field in ("tsv", "stock", "neutral", "policy", "result"):
            command.add_argument(field, type=Path)
        if action == "score":
            command.add_argument("--development-receipt", type=Path, required=True,
                                 help="local freeze result pinning both executable digests and the comparator")
    args = parser.parse_args()
    hashes = {"stock_binary_sha256": file_hash(args.stock), "neutral_binary_sha256": file_hash(args.neutral)}
    if args.action == "freeze":
        check_file(args.tsv, DEV_SHA)
    else:
        policy = json.loads(args.policy.read_bytes())
        if policy.get("format") != FORMAT or policy.get("development_sha256") != DEV_SHA:
            raise ValueError("unsupported policy or development set")
        dev = json.loads(args.development_receipt.read_bytes())
        if (dev["source_sha256"] != DEV_SHA or dev["policy_sha256"] != file_hash(args.policy)
                or any(dev[key] != value for key, value in hashes.items())):
            raise ValueError("development data, binaries or frozen policy differs from freeze receipt")
        comparator = comparison_margin(dev, file_hash(args.policy), json.loads(COMPARISON.read_bytes()))
    inputs = read_panel(args.tsv)
    witness = dict(ID="original-type-contrast", Text="Paz says he saw Zed.", Pronoun="he", A="Paz", B="Zed",
                   **{"Pronoun-offset": "9", "A-offset": "0", "B-offset": "16", "A-coref": "TRUE", "B-coref": "FALSE"})
    rivals = dict(ID="original-rival-veto", Text="Hera told Zoe her news.", Pronoun="her", A="Hera", B="Zoe",
                  **{"Pronoun-offset": "14", "A-offset": "0", "B-offset": "10", "A-coref": "FALSE", "B-coref": "TRUE"})
    stock_witness, stock_rivals, *stock = run([witness, rivals, *inputs], args.stock)
    neutral_witness, neutral_rivals, *neutral = run([witness, rivals, *inputs], args.neutral)
    if stock_witness.issued is not None or neutral_witness.issued != 0:
        raise ValueError("workers failed the stock/neutral contrast")
    if (stock_rivals.issued != 1 or neutral_rivals.direct != 1 or neutral_rivals.issued != 1
            or neutral_rivals.above_threshold is None or neutral_rivals.above_threshold < .15
            or neutral_rivals.gated(.15) is not None):
        raise ValueError("workers failed the scored rival-veto control")
    if args.action == "freeze":
        rule, grid = choose(stock, neutral)
        policy = {"format": FORMAT, "development_sha256": DEV_SHA, "rule": rule}
        args.policy.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
        best = best_feasible(summaries(stock, [r.issued for r in stock]), grid)
        if best is None:
            raise ValueError("no feasible development comparator")
        comparator = best["margin"]
    result = measure(stock, neutral, policy["rule"], [r["URL"] for r in inputs], comparator)
    if args.action == "freeze":
        result["development_grid"] = grid
    result.update(source_sha256=file_hash(args.tsv), policy_sha256=file_hash(args.policy),
                  **hashes, articles=len({r["URL"] for r in inputs}))
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
