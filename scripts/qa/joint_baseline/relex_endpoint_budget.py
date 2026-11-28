# SPDX-License-Identifier: MIT OR Apache-2.0
"""Rank a fixed CrossRE review queue using the weaker predicted endpoint's confidence."""

import argparse
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
import json
from math import ceil
from pathlib import Path
from random import Random

import relex_full_test as full
import relex_predicate_ranking as study
from jointie_coverage import gold_graph
from relex_roles import gold


WEIGHTS = (0.5, 1, 2)
BUDGETS = (25, 50, 100)
PRIMARY_BUDGET = 50
CACHE_SCHEMA = "crossre-endpoint-scores/v1"


def inputs(directory):
    data, schemas, eligible, pins, _ = full.membership(directory)
    _, _, balanced_ids, _, _ = study.cohorts(directory)
    _, _, all_ids, _, _ = study.cohorts(directory, all_eligible=True)
    _, outside = full.outside_holdout(data, eligible)
    calibration = [(domain, "dev", index) for domain in study.DOMAINS
                   for index in balanced_ids["dev", domain]]
    used = {study.text_id(data[d, s][i]) for d, s, i in calibration}
    used.update(study.text_id(doc) for d in study.DOMAINS for doc in data[d, "test"])
    confirmation = [(domain, "dev", index) for domain in study.DOMAINS
                    for index in all_ids["dev", domain]
                    if study.text_id(data[domain, "dev"][index]) not in used]
    cohorts = {
        "calibration": calibration,
        "confirmation": confirmation,
        "additional_test": [(d, "test", i) for d, i, orig in eligible if not orig],
        "balanced_test": [(d, "test", i) for d, i, orig in eligible if orig],
        "outside_content_holdout": [(d, "test", i) for d, i, _ in outside],
    }
    return data, schemas, pins, cohorts


def condensed(entities, relations):
    """Keep only model-output coordinates and scores needed to replay queue choices."""
    by_span = {(e["start"], e["end"]): float(e["score"]) for e in entities}
    candidates, stock = [], []
    seen = set()
    for rel in relations:
        head = (rel["head"]["start"], rel["head"]["end"])
        tail = (rel["tail"]["start"], rel["tail"]["end"])
        key = (*head, rel["relation"], *tail)
        confidence = float(rel["score"])
        endpoint = min(by_span[head], by_span[tail])
        if key in seen or not (study.FLOOR < confidence <= 1 and study.STOCK_GATE < endpoint <= 1):
            raise ValueError("Duplicate relation or invalid public model score")
        seen.add(key)
        if confidence > study.STOCK_GATE:
            stock.append(key)
        else:
            candidates.append((*key, confidence, endpoint))
    return {"stock": sorted(stock), "candidates": sorted(candidates)}


@dataclass
class Observation:
    domain: str
    split: str
    index: int
    text_sha256: str
    candidates: list
    reference: Counter
    reference_mentions: set
    stock: Counter

    @property
    def location(self):
        return self.domain, self.split, self.index


def observations(data, cells, cache):
    result = []
    for domain, split, index in cells:
        doc = data[domain, split][index]
        text_hash = study.text_id(doc)
        path = cache / f"{domain}-{split}-{index}.json"
        raw = json.loads(path.read_text())
        source = raw["source"]
        if (source["schema"], source["model_revision"], source["text_sha256"]) != (
            CACHE_SCHEMA, study.MODEL_REVISION, text_hash
        ):
            raise ValueError(f"Model or source mismatch in {path}")
        ref = Counter({(*h, label, *t): n for (h, label, t), n in gold(doc).items()})
        _, nodes, _ = gold_graph(doc)
        raw_stock = Counter(map(tuple, raw["stock"]))
        pool = [(tuple(c[:5]), c[5], c[6]) for c in raw["candidates"]]
        pool_keys = set()
        for edge, prob, endpoint in pool:
            if (edge in pool_keys or edge in raw_stock or not study.FLOOR < prob <= study.STOCK_GATE
                    or not study.STOCK_GATE < endpoint <= 1):
                raise ValueError(f"Invalid cached review pool in {path}")
            pool_keys.add(edge)
        result.append(Observation(domain, split, index, text_hash, pool, ref,
                                  {(start, end) for _, start, end in nodes}, raw_stock))
    return result


def nominees(cases, weight):
    """Use only model scores and input identity, before accessing any evaluation labels."""
    ordered = []
    for case in cases:
        if not case.candidates:
            continue
        value, confidence, edge = min(
            (-score * endpoint ** weight, -score, key)
            for key, score, endpoint in case.candidates
        )
        ordered.append((value, confidence, case.text_sha256, case.location, edge))
    return [(location, edge) for _, _, _, location, edge in sorted(ordered)]


def compare(cases, baseline, candidate, percent):
    if len(baseline) != len(candidate) or {loc for loc, _ in baseline} != {loc for loc, _ in candidate}:
        raise ValueError("Review arms must start from the same eligible texts")
    budget = len(baseline) * percent // 100
    left, right = dict(baseline[:budget]), dict(candidate[:budget])
    by_domain = {d: Counter() for d in study.DOMAINS}
    states = Counter()
    for case in cases:
        old, new = left.get(case.location), right.get(case.location)
        changed = old != new
        old_hit, new_hit = old in case.reference, new in case.reference
        info = by_domain[case.domain]
        info["sentences"] += 1
        for arm, edge, hit in (("confidence", old, old_hit), ("endpoint", new, new_hit)):
            if edge is None:
                continue
            info[f"{arm}_suggestions"] += 1
            info[f"{arm}_correct"] += int(hit)
            info[f"{arm}_unmatched"] += int(not hit)
            unknown_endpoint = edge[:2] not in case.reference_mentions or edge[3:] not in case.reference_mentions
            if hit and unknown_endpoint:
                raise ValueError("Correct reference relation has an unannotated endpoint")
            info[f"{arm}_unmatched_with_inexact_endpoint"] += int(unknown_endpoint)
        info["gained_correct_relations"] += int(changed and new_hit)
        info["lost_correct_relations"] += int(changed and old_hit)
        info["both_correct_changed_relation"] += int(changed and old_hit and new_hit)
        if old is None and new is None:
            continue
        if old is None or new is None:
            states["candidate_only_sentence" if old is None else "baseline_only_sentence"] += 1
        elif changed:
            states["both_reviewed_with_different_relation"] += 1
        else:
            states["both_reviewed_with_same_relation"] += 1
    total = sum(by_domain.values(), Counter())
    for arm in ("confidence", "endpoint"):
        if total[f"{arm}_correct"] + total[f"{arm}_unmatched"] != budget:
            raise ValueError("Fixed review budget did not reconcile")
    if total["gained_correct_relations"] - total["lost_correct_relations"] != (
        total["endpoint_correct"] - total["confidence_correct"]
    ):
        raise ValueError("Paired gained and lost reference identities did not reconcile")
    return {"review_slots": budget, "counts": dict(total), "sentence_allocation": dict(states),
            "domains": {d: dict(c) for d, c in by_domain.items()}}


def resampled_hits(ranking, copies, slots):
    """Walk the sorted nominees, filling a fresh global queue with sampled copies."""
    hits = 0
    for index, correct in ranking:
        taken = min(copies[index], slots)
        if correct:
            hits += taken
        slots -= taken
        if slots == 0:
            break
    if slots:
        raise ValueError("The resampled nominees could not fill the review budget")
    return hits


def queue_interval(cases, baseline, candidate, slots, *, replicates=10_000, seed=20261128):
    """Paired percentile interval conditional on each domain's eligible sentence count."""
    eligible = [c for c in cases if c.candidates]
    lookup = {c.location: i for i, c in enumerate(eligible)}
    strata = {d: [] for d in study.DOMAINS}
    for i, case in enumerate(eligible):
        strata[case.domain].append(i)
    if (not 0 <= slots <= len(eligible) or replicates < 1 or len(lookup) != len(eligible)
            or any(len(ranking) != len(eligible) or {loc for loc, _ in ranking} != lookup.keys()
                   for ranking in (baseline, candidate))):
        raise ValueError("Paired resampling requires the same eligible sentences and a feasible budget")

    def indexed(ranking):
        return [(lookup[loc], edge in eligible[lookup[loc]].reference) for loc, edge in ranking]

    left, right = indexed(baseline), indexed(candidate)
    rng = Random(seed)
    differences = []
    for _ in range(replicates):
        copies = [0] * len(eligible)
        for stratum in strata.values():
            for index in rng.choices(stratum, k=len(stratum)):
                copies[index] += 1
        # A multiplicity is shared by the two arms. Each fills all slots in its own
        # ranking order, so the sentence at the budget boundary can change every draw.
        differences.append(resampled_hits(right, copies, slots) - resampled_hits(left, copies, slots))
    differences.sort()
    return {
        "lower": differences[ceil(.025 * replicates) - 1],
        "upper": differences[ceil(.975 * replicates) - 1],
        "method": "95% paired bootstrap with both global review queues reselected",
        "replicates": replicates, "seed": seed, "fixed_review_slots": slots,
        "eligible_sentences_by_domain": {d: len(s) for d, s in strata.items()},
        "sampling": "Eligible sentences sampled with replacement within each domain at its observed count; both arms share each draw; empirical inverse-CDF percentiles of candidate minus baseline exact matches.",
    }


def corpus_scope(cases):
    gold_count = sum(c.reference.total() for c in cases)
    matched = sum((c.stock & c.reference).total() for c in cases)
    return {
        "sentences": len(cases), "distinct_texts": len({c.text_sha256 for c in cases}),
        "eligible_for_one_suggestion": sum(bool(c.candidates) for c in cases),
        "gold_relations": gold_count, "stock_correct": matched,
        "stock_predictions": sum(c.stock.total() for c in cases), "gold_missed_by_stock": gold_count - matched,
        "input_sha256": study.digest("".join(
            f"{c.domain}\t{c.split}\t{c.index}\t{c.text_sha256}\n" for c in cases
        ).encode()),
    }


def select(cases, pins):
    baseline = nominees(cases, 0)
    grid = []
    for weight in (0, *WEIGHTS):
        ranked = nominees(cases, weight)
        result = compare(cases, baseline, ranked, PRIMARY_BUDGET)
        grid.append({"weight": weight, "review_slots": result["review_slots"],
                     "correct": result["counts"].get("endpoint_correct", 0)})
    winner = max(grid[1:], key=lambda r: (r["correct"], -r["weight"]))
    return {
        "model_revision": study.MODEL_REVISION, "model_files_sha256": study.MODEL_FILES,
        "crossre_revision": study.REVISION, "source_sha256": pins,
        "packages": json.loads(full.BASELINE.read_text())["package_versions"],
        "scope": corpus_scope(cases), "primary_budget_percent_of_eligible": PRIMARY_BUDGET,
        "weights_checked": grid, "selected_endpoint_weight": winner["weight"],
        "selection": "Among positive weights, most exact development additions at the primary budget; tie to smaller weight. Zero is the independent confidence baseline.",
    }


def measure(cases, weight, *, primary_interval=False):
    baseline = nominees(cases, 0)
    candidate = nominees(cases, weight)
    budgets = {str(p): compare(cases, baseline, candidate, p) for p in BUDGETS}
    if primary_interval:
        primary = budgets[str(PRIMARY_BUDGET)]
        primary["net_match_interval"] = queue_interval(cases, baseline, candidate, primary["review_slots"])
    return {"scope": corpus_scope(cases), "budgets": budgets}


def check_prior(measured, prior):
    scope, baseline = measured["scope"], measured["budgets"]["100"]["counts"]
    expected = prior["stock"]
    if (scope["sentences"], scope["gold_relations"], scope["stock_correct"], scope["stock_predictions"],
        baseline.get("confidence_correct", 0), baseline.get("confidence_suggestions", 0)) != (
        expected["sentences"], expected["gold"], expected["matched"], expected["predicted"],
        prior["additional_edges"]["score"]["correct_added"], prior["additional_edges"]["score"]["added"]
    ):
        raise ValueError("Public decoded stock or review pool differs from the prior scored-tensor replay")


def measure_part(data, groups, cache, selected, part):
    """Score complete cohorts locally so each review queue spans all six domains."""
    weight = selected["selected_endpoint_weight"]
    panels = {"calibration": measure(observations(data, groups["calibration"], cache), weight)}
    check_prior(panels["calibration"], json.loads(full.BASELINE.read_text())["panels"]["dev"])
    if part == "confirmation":
        panels[part] = measure(observations(data, groups[part], cache), weight, primary_interval=True)
    else:
        official = []
        prior = json.loads((full.BASELINE.parent / "crossre-full-test.json").read_text())["panels"]
        for name in ("additional_test", "balanced_test", "outside_content_holdout"):
            cases = observations(data, groups[name], cache)
            panels[name] = measure(cases, weight)
            check_prior(panels[name], prior[name])
            official.extend(cases)
        panels["full_official_test"] = measure(official, weight)
        check_prior(panels["full_official_test"], prior["full_official_test"])
    return {"part": part, "selection": selected, "panels": panels}


def combine(confirmation, test):
    if (
        confirmation["part"] != "confirmation" or test["part"] != "test"
        or confirmation["selection"] != test["selection"]
        or confirmation["panels"]["calibration"] != test["panels"]["calibration"]
        or set(confirmation["panels"]) != {"calibration", "confirmation"}
        or set(test["panels"]) != {"calibration", "additional_test", "balanced_test",
                                    "outside_content_holdout", "full_official_test"}
    ):
        raise ValueError("Incomplete study or different development evidence across workers")
    return {
        "selection": confirmation["selection"], "entity_gate": study.STOCK_GATE,
        "relation_confidence_lower_exclusive": study.FLOOR, "relation_confidence_upper_inclusive": study.STOCK_GATE,
        "ranking": "relation probability multiplied by the lesser predicted endpoint probability raised to the selected weight; rank one per sentence, then rank nominees across the cohort",
        "budgets_percent_of_eligible": list(BUDGETS), "panels": {**test["panels"], **confirmation["panels"]},
    }


def endpoint_chart(receipt):
    panels = receipt["panels"]
    rows = []
    for name, label, role in (
        ("confirmation", "Unused development texts", "Independent confirmation"),
        ("additional_test", "Additional official test texts", "Previously examined for other rules"),
        ("calibration", "Original development panel", "Selection"),
        ("full_official_test", "Complete official test release", "Includes the official test subsets"),
    ):
        panel = panels[name]
        budget = panel["budgets"][str(PRIMARY_BUDGET)]
        counts = budget["counts"]
        scope = panel["scope"]
        row = {
            "label": label, "role": f"{role} · {scope['eligible_for_one_suggestion']:,} eligible",
            "observations": scope["sentences"],
            "baseline": {"correct": counts.get("confidence_correct", 0), "unmatched": counts.get("confidence_unmatched", 0)},
            "candidate": {"correct": counts.get("endpoint_correct", 0), "unmatched": counts.get("endpoint_unmatched", 0)},
            "gained": counts.get("gained_correct_relations", 0), "lost": counts.get("lost_correct_relations", 0),
        }
        if interval := budget.get("net_match_interval"):
            row["interval"] = {key: interval[key] for key in ("lower", "upper", "method")}
        rows.append(row)
    return {"schema": "anno.paired/v1", "title": "CrossRE reviews at 50% of eligible sentences", "unit": "sentences",
            "baseline": "Predicate confidence", "candidate": "Endpoint-weighted confidence", "rows": rows}


def collect(args):
    import torch

    data, schemas, _, groups = inputs(args.data)
    packages = json.loads(full.BASELINE.read_text())["package_versions"]
    if any(version(p) != v for p, v in packages.items()):
        raise ValueError("Use the packages pinned in the predicate ranking study")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model, _ = study.load_model(args.checkpoint)
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.scope == "all":
        requested = list(groups)
    elif args.scope == "test":
        requested = ["additional_test", "balanced_test", "outside_content_holdout"]
    else:
        requested = [args.scope]
    processed = 0
    for group in requested:
        cells = groups[group]
        for domain in study.DOMAINS:
            needed = []
            for d, split, index in cells:
                if d != domain:
                    continue
                doc = data[d, split][index]
                path = args.cache / f"{d}-{split}-{index}.json"
                meta = {"schema": CACHE_SCHEMA, "model_revision": study.MODEL_REVISION,
                        "text_sha256": study.text_id(doc), "batch_size": args.batch_size}
                if path.exists():
                    old = json.loads(path.read_text())
                    if old["source"] != meta:
                        raise ValueError(f"Different inference configuration or source at {path}")
                else:
                    needed.append((doc, path, meta))
            schema = schemas[domain]
            for start in range(0, len(needed), args.batch_size):
                batch = needed[start:start + args.batch_size]
                texts = [" ".join(doc["sentence"]) for doc, _, _ in batch]
                entities, rels = model.inference(
                    texts, schema["entity_types"], relations=schema["relation_types"],
                    threshold=study.STOCK_GATE, adjacency_threshold=study.STOCK_GATE,
                    relation_threshold=study.FLOOR, batch_size=args.batch_size,
                )
                for (_, path, meta), mentions, edges in zip(batch, entities, rels, strict=True):
                    full.write_json(path, {"source": meta, **condensed(mentions, edges)})
                processed += len(batch)
                if processed == len(batch) or processed // 64 > (processed - len(batch)) // 64:
                    print(f"Collected {processed} predictions, current cohort {group}, domain {domain}", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="command", required=True)
    collect_cli = actions.add_parser("collect")
    collect_cli.add_argument("data", type=Path)
    collect_cli.add_argument("checkpoint", type=Path)
    collect_cli.add_argument("cache", type=Path)
    collect_cli.add_argument("--scope", choices=("calibration", "confirmation", "test", "balanced_test",
                                               "additional_test", "outside_content_holdout", "all"), default="all")
    collect_cli.add_argument("--threads", type=int, default=2)
    collect_cli.add_argument("--batch-size", type=int, default=1)
    selection = actions.add_parser("select", help="Freeze a positive weight on development before evaluation")
    selection.add_argument("data", type=Path)
    selection.add_argument("cache", type=Path)
    selection.add_argument("selection", type=Path)
    evaluate = actions.add_parser("measure", help="Apply the frozen choice to a complete six-domain cohort")
    evaluate.add_argument("data", type=Path)
    evaluate.add_argument("cache", type=Path)
    evaluate.add_argument("selection", type=Path)
    evaluate.add_argument("--part", choices=("confirmation", "test"), required=True)
    evaluate.add_argument("--output", required=True, type=Path)
    assemble = actions.add_parser("assemble", help="Reconcile the two calibrations and combine aggregate counts")
    assemble.add_argument("confirmation", type=Path)
    assemble.add_argument("test", type=Path)
    assemble.add_argument("--output", required=True, type=Path)
    assemble.add_argument("--paired-output", required=True, type=Path)
    args = cli.parse_args()
    if args.command == "collect":
        if min(args.threads, args.batch_size) < 1:
            cli.error("threads and batch size must be positive")
        collect(args)
    elif args.command == "assemble":
        receipt = combine(json.loads(args.confirmation.read_text()), json.loads(args.test.read_text()))
        full.write_json(args.output, receipt)
        full.write_json(args.paired_output, endpoint_chart(receipt))
    else:
        data, _, pins, groups = inputs(args.data)
        selected = select(observations(data, groups["calibration"], args.cache), pins)
        if args.command == "select":
            if args.selection.exists() and json.loads(args.selection.read_text()) != selected:
                cli.error("The existing frozen development selection differs")
            full.write_json(args.selection, selected)
        else:
            if json.loads(args.selection.read_text()) != selected:
                cli.error("Frozen selection or calibration inputs differ")
            full.write_json(args.output, measure_part(data, groups, args.cache, selected, args.part))


if __name__ == "__main__":
    main()
