"""Replay native supplied-mention decisions with competing-name and threshold margins."""

import argparse
import json
import math
import os
import random
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import audit_gap_decisions as gap
from coref_corpora import winogender
from coref_gap import REVISION as GAP_REVISION
from gap_receipts import digest
from native_gap_neutral_type import (
    PATCH as NEUTRAL_PATCH,
)
from native_gap_neutral_type import (
    PRODUCER,
    ROOT,
    file_hash,
    paired,
    read_sources,
)

MARGINS = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
RULES = ("rival", "threshold")
RETENTION = 0.80
BASE = "01370e52ba97d5a094c3a3a0e9d3b038ef7063f9"


@dataclass
class Decision:
    span: tuple
    kind: str
    threshold: float
    candidates: list
    selected: int | None

    def parent(self, rule, margin):
        if self.selected is None or self.kind != "pronominal" or margin == 0:
            return self.selected
        scores = dict(self.candidates)
        if rule == "rival":
            others = [
                score for index, score in self.candidates if index != self.selected
            ]
            if not others:
                return self.selected
            comparator = max(others)
        elif rule == "threshold":
            comparator = self.threshold
        else:
            raise ValueError(f"Unknown abstention rule {rule}")
        return self.selected if scores[self.selected] - comparator >= margin else None


@dataclass
class Observation:
    decisions: list
    roles: tuple

    def groups(self, rule="rival", margin=0.0):
        roots = []
        for i, decision in enumerate(self.decisions):
            parent = decision.parent(rule, margin)
            roots.append(i if parent is None else roots[parent])
        return roots

    def mask(self, rule="rival", margin=0.0):
        roots = self.groups(rule, margin)
        p, a, b = self.roles
        return "111" + str(int(roots[p] == roots[a])) + str(int(roots[p] == roots[b]))

    def partition(self, rule="rival", margin=0.0):
        by_root = {}
        for decision, root in zip(
            self.decisions, self.groups(rule, margin), strict=True
        ):
            by_root.setdefault(root, []).append(decision.span)
        return sorted(by_root.values())


def validate_trace(row, records, output):
    spans = sorted(gap.spans(row))
    if len(set(spans)) != 3 or len(records) != 3:
        raise ValueError(
            "Expected three distinct supplied mentions and one decision per mention"
        )
    decisions = []
    for i, (record, span) in enumerate(zip(records, spans, strict=True)):
        if record["index"] != i or tuple(record["span"]) != span:
            raise ValueError("Native trace changed source order or supplied boundaries")
        candidates = [(int(j), float(s)) for j, s in record["candidates"]]
        gate = record["threshold"]
        if not math.isfinite(gate) or not all(math.isfinite(s) for _, s in candidates):
            raise ValueError("Nonfinite rank score")
        indices = [j for j, _ in candidates]
        if indices != sorted(set(indices), reverse=True) or not all(
            0 <= j < i for j in indices
        ):
            raise ValueError("Invalid native antecedent order")
        # The fixed shipped configuration examines all three marked mentions within the window.
        eligible = [
            j
            for j in reversed(range(i))
            if max(0, span[0] - spans[j][1]) <= record["max_distance"]
        ]
        if indices != eligible:
            raise ValueError(
                "Trace candidates differ from the native three-mention search window"
            )
        best = max(candidates, key=lambda candidate: candidate[1], default=None)
        winner = best[0] if best and best[1] > gate else None
        if winner != record["selected"]:
            raise ValueError(
                "Native winner differs from the strict threshold and nearest-first tie order"
            )
        decisions.append(
            Decision(span, record["mention_type"], gate, candidates, winner)
        )
    observation = Observation(decisions, tuple(spans.index(s) for s in gap.spans(row)))
    bits, status = gap.from_clusters(row, output, PRODUCER, "supplied")
    actual = sorted(sorted(tuple(s) for s in group) for group in output["clusters"])
    if (
        observation.partition() != actual
        or status != "ok"
        or bits != observation.mask()
    ):
        raise ValueError(
            "Replayed parent edges do not reproduce the public native partitions"
        )
    return observation, actual


def native(rows, binary, classify=False):
    command = [str(binary.resolve()), PRODUCER, "--trace-ranking"]
    if classify:
        command.append("--classify-mentions")
    payload = "".join(json.dumps(gap.payload(r, "supplied")) + "\n" for r in rows)
    proc = subprocess.run(
        command,
        input=payload,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "ANNO_NO_DOWNLOADS": "1", "HF_HUB_OFFLINE": "1"},
    )
    traced = {}
    records = None
    for line in proc.stderr.splitlines():
        rec = json.loads(line)
        if set(rec) == {"id"}:
            if rec["id"] in traced:
                raise ValueError("Repeated native trace document")
            records = traced[rec["id"]] = []
        elif records is None:
            raise ValueError("Rank trace has no document")
        else:
            records.append(rec)
    if list(traced) != [r["ID"] for r in rows]:
        raise ValueError("Native trace IDs do not match input rows")
    observed, partitions = [], []
    for row, line in zip(rows, proc.stdout.splitlines(), strict=True):
        obs, groups = validate_trace(row, traced[row["ID"]], json.loads(line))
        observed.append(obs)
        partitions.append(groups)
    return observed, digest(partitions)


def verify_margin_worker(rows, observations, binary, rule):
    """Check every connected component against an independent native intervention."""
    payload = "".join(json.dumps(gap.payload(row, "supplied")) + "\n" for row in rows)
    fingerprints = []
    for margin in MARGINS:
        run = subprocess.run(
            [str(binary.resolve()), PRODUCER],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
            env={
                **os.environ,
                "ANNO_NO_DOWNLOADS": "1",
                "HF_HUB_OFFLINE": "1",
                "ANNO_QA_MARGIN_RULE": rule,
                "ANNO_QA_MIN_MARGIN": str(margin),
            },
        )
        partitions = []
        for row, obs, line in zip(
            rows, observations, run.stdout.splitlines(), strict=True
        ):
            native_result = json.loads(line)
            _, status = gap.from_clusters(row, native_result, PRODUCER, "supplied")
            actual = sorted(
                sorted(tuple(span) for span in group)
                for group in native_result["clusters"]
            )
            if status != "ok" or actual != obs.partition(rule, margin):
                raise ValueError(
                    f"{row['ID']}: native clusters differ at {rule} margin {margin}"
                )
            partitions.append(actual)
        fingerprints.append(digest(partitions))
    return fingerprints


def winogender_rows(directory):
    rows = {"Winogender phrases": [], "Winogender heads": []}
    groups = set()
    for case in winogender.cases(directory):
        groups.add(case["group"])
        for key, values in rows.items():
            spans = case["spans"].copy()
            if key == "Winogender heads":
                spans.update(case["boundaries"]["head"])
            row = {
                "ID": case["id"],
                "Text": case["text"],
                "template": case["group"],
                "A-coref": str(case["gold"] == "A").upper(),
                "B-coref": str(case["gold"] == "B").upper(),
            }
            for source, label in (("target", "Pronoun"), ("A", "A"), ("B", "B")):
                start, end = spans[source]
                row[label], row[label + "-offset"] = case["text"][start:end], start
            values.append(row)
    if len(groups) != 120:
        raise ValueError("Winogender template groups changed")
    return rows


def count(rows, masks):
    counts = gap.decision_counts(rows, masks)
    return {k: counts[k] for k in ("tp", "fp", "fn", "tn")}


def diagnose(rows, observations):
    eligibility, missed, wrong, kind = Counter(), Counter(), Counter(), Counter()
    positive = 0
    for row, obs in zip(rows, observations, strict=True):
        pi, *names = obs.roles
        decision = obs.decisions[pi]
        scores = dict(decision.candidates)
        targets = gap.gold(row)
        eligibility[str(len(scores))] += 1
        kind[decision.kind] += 1
        mask = obs.mask()[3:]
        for name, gold, link in zip(names, targets, mask, strict=True):
            positive += gold
            if gold and link == "0":
                if name > pi:
                    reason = "name_after_target"
                elif name not in scores:
                    reason = "prior_name_outside_window"
                elif scores[name] <= decision.threshold:
                    reason = "prior_gold_at_or_below_gate"
                else:
                    if decision.selected == name:
                        raise ValueError("A direct native gold edge disappeared")
                    reason = "prior_gold_above_gate_other_won"
                missed[reason] += 1
            if not gold and link == "1":
                if decision.selected != name:
                    reason = "incoming_or_transitive"
                elif decision.kind != "pronominal":
                    reason = "target_inferred_nonpronominal"
                elif len(scores) == 1:
                    reason = "direct_no_scored_rival"
                else:
                    reason = "direct_scored_rival"
                wrong[reason] += 1
    totals = count(rows, [x.mask() for x in observations])
    if sum(missed.values()) != totals["fn"] or sum(wrong.values()) != totals["fp"]:
        raise ValueError("Diagnostic counts failed to reconcile")
    return {
        "target_eligible_name_counts": eligibility,
        "target_inferred_kind": kind,
        "gold_links": positive,
        "missed_gold_by_target_search": missed,
        "wrong_links_by_target_search": wrong,
    }


def score_rule(rows, observations, rule, margin):
    return [x.mask(rule, margin) for x in observations]


def sweep(rows, observations):
    return {
        rule: [
            {"margin": m, **count(rows, score_rule(rows, observations, rule, m))}
            for m in MARGINS
        ]
        for rule in RULES
    }


def select(dev_curve):
    floor = math.ceil(RETENTION * dev_curve[0]["tp"])
    return min(
        (p for p in dev_curve if p["tp"] >= floor),
        key=lambda p: (p["fp"], -p["tp"], p["margin"]),
    )["margin"]


def paired_interval(rows, before, after, group_field, draws=5000):
    """Resample paired clusters of the same size, holding the excerpt count fixed."""
    by_group = {}
    for row, old, new in zip(rows, before, after, strict=True):
        # Two candidate-name links share a target. Keep both with the source article or template.
        group = by_group.setdefault(row[group_field], [0, 0, 0, 0])
        group[0] += 1
        for gold, prev, curr in zip(gap.gold(row), old[3:], new[3:], strict=True):
            if gold:
                group[1] += prev == "1"
                group[2] += curr == "1"
                group[3] += prev == curr == "1"
    strata = {}
    for key in sorted(by_group):
        size, baseline, candidate, kept = by_group[key]
        strata.setdefault(size, []).append((baseline, candidate - baseline, kept))
    rng = random.Random(428)
    net, retention = [], []
    for _ in range(draws):
        baseline = delta = retained = 0
        for values in strata.values():
            for original, change, kept in rng.choices(values, k=len(values)):
                baseline += original
                delta += change
                retained += kept
        net.append(delta)
        if baseline:
            retention.append(100 * retained / baseline)

    def bounds(values):
        values = sorted(values)
        return [
            round(values[math.ceil(q * len(values)) - 1], 2) for q in (0.025, 0.975)
        ]

    return {
        "cluster": "article URL" if group_field == "URL" else "Winogender template",
        "clusters": len(by_group),
        "draws": draws,
        "seed": 428,
        "net_correct_95": bounds(net),
        "retention_defined_draws": len(retention),
        "retention_percent_95": bounds(retention) if retention else None,
    }


def chart(panels):
    panel = panels["GAP test"]
    points = []

    def add(label, counts):
        points.append(
            {"label": label, "correct": counts["tp"], "unmatched": counts["fp"]}
        )

    for name in ("Stock", "Neutral type", "Classified mentions"):
        add(name, panel["operating_points"][name])
    for rule, label, name in (
        ("rival", "Rival", "Rival margin"),
        ("threshold", "Threshold", "Threshold margin"),
    ):
        chosen = select(panels["GAP development"]["sweeps"][rule])
        add(f"{label} {chosen:g} (dev choice)", panel["operating_points"][name])
        for point in panel["sweeps"][rule]:
            if point["margin"] in (0.1, 0.3) and point["margin"] != chosen:
                add(f"{label} {point['margin']:g}", point)
    return {
        "schema": "anno.tradeoff/v1",
        "title": "Choosing when to link a pronoun on GAP",
        "unit": "excerpts",
        "series": [
            {
                "label": "GAP test, supplied mentions",
                "observations": panel["rows"],
                "gold": panel["diagnostic"]["gold_links"],
                "points": points,
            }
        ],
    }


def paired_chart(receipt):
    cutoff = receipt["selected_development_margins"]["threshold"]
    target = receipt["development_true_link_retention"]
    labels = (
        ("GAP development", "GAP development", "Cutoff selection"),
        ("GAP validation", "GAP validation", "Fixed cutoff"),
        ("GAP test", "GAP test", "Fixed cutoff"),
        ("Winogender phrases", "Full role phrases", "Winogender · same sentences"),
        ("Winogender heads", "Bare role heads", "Winogender · same sentences"),
    )
    rows = []
    for key, label, role in labels:
        group = receipt["panels"][key]
        arms = group["operating_points"]
        baseline, candidate = arms["Neutral type"], arms["Threshold margin"]
        pair = group["paired_vs_neutral"]["Threshold margin"]
        row = {
            "label": label,
            "role": role,
            "observations": group["rows"],
            "baseline": {"correct": baseline["tp"], "unmatched": baseline["fp"]},
            "candidate": {"correct": candidate["tp"], "unmatched": candidate["fp"]},
            "gained": pair["correct_gained"],
            "lost": pair["correct_lost"],
        }
        if ci := group.get("threshold_pair_resampling"):
            row["interval"] = {
                "lower": ci["net_correct_95"][0],
                "upper": ci["net_correct_95"][1],
                "method": f"95% paired bootstrap by {ci['cluster']}, within cluster size",
            }
        rows.append(row)
    validation = next(row for row in rows if row["label"] == "GAP validation")
    correct = validation["baseline"]["correct"]
    retained = correct - validation["lost"]
    title = (
        f"GAP validation retained {retained / correct:.1%} of correct links (target {target:.0%})"
        if correct
        else "GAP validation had no baseline correct links to retain"
    )
    return {
        "schema": "anno.paired/v1",
        "title": title,
        "unit": "supplied-span examples",
        "baseline": "Neutral mention type",
        "candidate": f"Threshold margin {cutoff:g}",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in (
        "gap",
        "winogender",
        "stock",
        "neutral",
        "margin-worker",
        "output",
        "chart",
    ):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument(
        "--paired-chart",
        type=Path,
        help="Write development, validation, test and both Winogender boundary comparisons",
    )
    args = parser.parse_args()
    gap_splits, pins = read_sources(args.gap)
    panels = {
        f"GAP {name}": gap_splits[name]
        for name in ("development", "validation", "test")
    }
    panels.update(winogender_rows(args.winogender))
    previous = json.loads(
        (ROOT / "docs/eval/native-gap-neutral-mention.json").read_text()
    )
    results, cutoffs = {}, {}
    for label, rows in panels.items():
        stock, stock_digest = native(rows, args.stock)
        neutral, neutral_digest = native(rows, args.neutral)
        classified, classified_digest = native(rows, args.neutral, classify=True)
        if label.startswith("GAP "):
            frozen = previous["panels"][label.split()[-1]]["supplied"]
            if (stock_digest, neutral_digest) != (
                frozen["stock_partitions_sha256"],
                frozen["candidate_partitions_sha256"],
            ):
                raise ValueError(
                    f"{label}: earlier full-split native partitions did not reproduce"
                )
        stock_masks, neutral_masks = (
            [x.mask() for x in stock],
            [x.mask() for x in neutral],
        )
        curves = sweep(rows, neutral)
        if label == "GAP development":
            cutoffs = {rule: select(curves[rule]) for rule in RULES}
        outputs = {"Stock": stock_masks, "Neutral type": neutral_masks}
        for rule, title in (
            ("rival", "Rival margin"),
            ("threshold", "Threshold margin"),
        ):
            outputs[title] = score_rule(rows, neutral, rule, cutoffs[rule])
        outputs["Classified mentions"] = [x.mask() for x in classified]
        result = {
            "rows": len(rows),
            "diagnostic": diagnose(rows, neutral),
            "operating_points": {k: count(rows, v) for k, v in outputs.items()},
            "paired_vs_neutral": {
                k: paired(rows, neutral_masks, v)
                for k, v in outputs.items()
                if k != "Neutral type"
            },
            "threshold_pair_resampling": paired_interval(
                rows,
                neutral_masks,
                outputs["Threshold margin"],
                "URL" if label.startswith("GAP ") else "template",
            ),
            "sweeps": curves,
            "native_partitions_sha256": {
                "stock": stock_digest,
                "neutral": neutral_digest,
                "classified": classified_digest,
            },
            "checked_native_margin_partitions_sha256": {
                rule: verify_margin_worker(rows, neutral, args.margin_worker, rule)
                for rule in RULES
            },
        }
        results[label] = result
    receipt = {
        "anno_revision": BASE,
        "gap_revision": GAP_REVISION,
        "gap_sha256": pins,
        "winogender_publisher": winogender.PUBLISHER,
        "winogender_sha256": winogender.PINS,
        "winogender_templates": 120,
        "development_true_link_retention": RETENTION,
        "selected_development_margins": cutoffs,
        "checked_native_margin_grid": MARGINS,
        "panels": results,
        "patches_sha256": {
            path.name: file_hash(path)
            for path in (
                NEUTRAL_PATCH,
                NEUTRAL_PATCH.with_name("native-gap-score-margin.patch"),
            )
        },
        "workers_sha256": {
            "stock": file_hash(args.stock),
            "neutral": file_hash(args.neutral),
            "margin": file_hash(args.margin_worker),
        },
    }
    files = [(args.output, receipt), (args.chart, chart(results))]
    if args.paired_chart:
        files.append((args.paired_chart, paired_chart(receipt)))
    for target, value in files:
        target.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
