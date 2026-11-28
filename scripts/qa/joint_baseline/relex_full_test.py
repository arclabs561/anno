# SPDX-License-Identifier: MIT OR Apache-2.0
"""Check the frozen CrossRE suggestion rules on every eligible official test text."""

import argparse
from collections import Counter, defaultdict
import gzip
import inspect
from importlib.metadata import version
from itertools import zip_longest
import json
from pathlib import Path
import tempfile
import time

import relex_predicate_ranking as study
from relex_roles import classify, decode, gold

BASELINE = study.ROOT / "docs/eval/crossre-predicate-ranking.json"


def single_forward(model, doc, schema):
    """Read the predicate heads used by the public GLiNER decoder in the same call."""
    from gliner.calibration import calibrated_sigmoid

    text = " ".join(doc["sentence"])
    captured = []

    def capture(_module, _args, kwargs, output):
        captured.append((kwargs, output))

    handle = model.model.register_forward_hook(capture, with_kwargs=True)
    try:
        entities, relations = model.predict_relations(
            text, schema["entity_types"], schema["relation_types"], threshold=study.STOCK_GATE
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Expected a single batch-one public model forward")
    inputs, output = captured.pop()
    words = list(model.data_processor.words_splitter(text))

    def chars(bounds):
        first, last = bounds
        return words[first][1], words[last][2]

    slots = inputs["span_idx"][0].tolist()
    valid = inputs["span_mask"].reshape(-1).bool()
    grid = {chars(slots[i]) for i in valid.nonzero().flatten().tolist()}
    entity_probs = calibrated_sigmoid(output.logits[0], model.config, kind="entity")
    proposed_mask = (entity_probs.flatten(0, 1).amax(-1) > study.STOCK_GATE) & valid
    proposed = {chars(slots[i]) for i in proposed_mask.nonzero().flatten().tolist()}
    scores = {}
    if output.rel_logits is not None:
        relation_probs = calibrated_sigmoid(output.rel_logits[0], model.config, kind="relation")
        candidates = output.entity_spans[0].tolist()
        indices = output.rel_idx[0]
        if output.rel_mask is not None:
            mask = output.rel_mask[0].bool()
            indices, relation_probs = indices[mask], relation_probs[mask]
        for (head, tail), row in zip(indices.tolist(), relation_probs.tolist(), strict=True):
            key = (chars(candidates[head]), chars(candidates[tail]))
            if key in scores:
                raise ValueError("Duplicate directed pair from the public predicate head")
            scores[key] = row
    kept = {(e["start"], e["end"]) for e in entities}
    graph = decode(scores, kept, schema["relation_types"], study.STOCK_GATE)
    expected = Counter(
        ((r["head"]["start"], r["head"]["end"]), r["relation"], (r["tail"]["start"], r["tail"]["end"]))
        for r in relations
    )
    if expected != graph:
        raise ValueError("Recorded scores did not reproduce the public directed stock edges")
    losses = classify(gold(doc), graph, schema["relation_types"], grid, proposed, scores, kept)
    return kept, scores, graph, losses


def cached_reference(row):
    kept = {tuple(span) for span in row["kept"]}
    scores = {(tuple(h), tuple(t)): values for h, t, values in row["scores"]}
    graph = Counter((tuple(h), label, tuple(t)) for h, label, t in row["stock"])
    return kept, scores, graph, Counter(row["losses"])


def check_reference(doc, schema, observed, reference):
    kept, scores, graph, losses = observed
    other_kept, other_scores, other_graph, other_losses = reference
    if (
        kept != other_kept
        or graph != other_graph
        or losses != other_losses
        or scores.keys() != other_scores.keys()
    ):
        raise ValueError(
            "Single-forward and trained export candidates, stock edges or losses differ"
        )
    max_error = max(
        (
            abs(a - b)
            for edge, row in scores.items()
            for a, b in zip(row, other_scores[edge], strict=True)
        ),
        default=0.0,
    )
    if max_error > 0.0002:
        raise ValueError(
            f"Single-forward predicate scores disagree with trained export by {max_error}"
        )
    labels = schema["relation_types"]
    if study.suggestions(scores, kept, labels) != study.suggestions(
        other_scores, other_kept, labels
    ):
        raise ValueError("A single-forward suggestion differs from the trained export")
    if study.document_counts(doc, schema, observed) != study.document_counts(
        doc, schema, reference
    ):
        raise ValueError(
            "Single-forward and trained export suggestion or ranking statistics differ"
        )
    return max_error


def write_json(path, data):
    staged = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    tmp = Path(staged.name)
    try:
        with staged:
            json.dump(data, staged, sort_keys=True, ensure_ascii=False, indent=2)
            staged.write("\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def membership(directory):
    data, schemas, all_ids, pins, scope = study.cohorts(directory, all_eligible=True)
    _, _, balanced_ids, _, balanced_scope = study.cohorts(directory)
    receipt = json.loads(BASELINE.read_text())
    if (
        receipt["source_sha256"] != pins
        or receipt["selection"] != balanced_scope
        or receipt["prompts_from_training"] != schemas
    ):
        raise ValueError("Balanced input identities or training-only prompts changed")
    if (
        receipt["selected_on_development"] != "score"
        or receipt["candidate_weights_label_direction"]
        != {k: list(v) for k, v in study.RULES.items()}
        or receipt["stock_gate"] != study.STOCK_GATE
        or receipt["suggestion_floor"] != study.FLOOR
    ):
        raise ValueError("Expanded replay must retain the balanced study's frozen rules")
    by_domain = {}
    for domain in study.DOMAINS:
        indices = all_ids["test", domain]
        original = set(balanced_ids["test", domain])
        if not original.issubset(indices):
            raise ValueError("An original test case fell outside the expanded cohort")
        by_domain[domain] = [(domain, i, i in original) for i in indices]
    # Interleave domains so a resumable job makes progress on each corpus.
    rows = [r for group in zip_longest(*by_domain.values()) for r in group if r is not None]
    return data, schemas, rows, pins, scope["test"]


def cache_identity(pins, baseline_sha):
    return study.digest(
        json.dumps(
            {
                "sources": pins,
                "model": study.MODEL_FILES,
                "baseline": baseline_sha,
                "score": inspect.getsource(single_forward),
                "aggregate": inspect.getsource(study.document_counts),
                "rank": inspect.getsource(study.compare_scores),
                "suggest": inspect.getsource(study.suggestions),
            },
            sort_keys=True,
        ).encode()
    )


def outside_holdout(data, eligible):
    """Explain each official row outside the distinct-text test cohort."""
    seen = set()
    training = {study.text_id(doc) for d in study.DOMAINS for doc in data[d, "train"]}
    historical = {study.text_id(doc) for doc in data["ai", "test"][:128]}
    development = {study.text_id(doc) for d in study.DOMAINS for doc in data[d, "dev"]}
    stages = (
        ("training_text", training),
        ("historical_ai_text", historical),
        ("development_text", development),
        ("repeated_test_text", seen),
    )
    selected = {(d, i) for d, i, _ in eligible}
    by_domain, extras = {}, []
    for domain in study.DOMAINS:
        omitted = {name: [] for name, _ in stages}
        for index, doc in enumerate(data[domain, "test"]):
            key = study.text_id(doc)
            reason = next((name for name, values in stages if key in values), None)
            if (reason is None) != ((domain, index) in selected):
                raise ValueError("Holdout exclusions disagree with the balanced-study selector")
            if reason is not None:
                omitted[reason].append(index)
                extras.append((domain, index, False))
            seen.add(key)
        by_domain[domain] = omitted
    return by_domain, extras


def repeated_annotations(data, counts):
    """Check whether repeated test text carries the same directed relation reference."""
    groups = defaultdict(list)
    for domain in study.DOMAINS:
        for i, doc in enumerate(data[domain, "test"]):
            reference = frozenset(gold(doc).items())
            stock = counts[domain, i]["stock"]
            groups[study.text_id(doc)].append((domain, reference, stock))
    repeated = [rows for rows in groups.values() if len(rows) > 1]
    changed = [rows for rows in repeated if len({ref for _, ref, _ in rows}) > 1]
    return {
        "distinct_test_texts": len(groups),
        "repeated_text_groups": len(repeated),
        "test_occurrences_in_repeated_groups": sum(map(len, repeated)),
        "different_directed_references_groups": len(changed),
        "occurrences_with_different_directed_references": sum(map(len, changed)),
        "single_domain_groups_with_different_directed_references": sum(
            len({d for d, _, _ in rows}) == 1 for rows in changed
        ),
        "different_references_and_different_stock_match_counts_groups": sum(
            len({stock["matched"] for _, _, stock in rows}) > 1 for rows in changed
        ),
    }


def collect(args):
    import torch

    data, schemas, eligible, pins, _ = membership(args.data)
    _, extras = outside_holdout(data, eligible)
    rows = {"official": eligible + extras, "eligible": eligible, "supplement": extras}[args.scope]
    frozen = json.loads(BASELINE.read_text())
    packages = frozen["package_versions"]
    if {k: version(k) for k in packages} != packages:
        raise ValueError(
            "The full replay must use the same model package versions as the frozen panel"
        )
    export_source = study.ROOT / "scripts/export_gliner_relex_joint.py"
    if study.digest(export_source.read_bytes()) != frozen["export_source_sha256"]:
        raise ValueError("The trained export source differs from the balanced study")
    args.cache.mkdir(parents=True, exist_ok=True)
    identity = cache_identity(pins, study.digest(BASELINE.read_bytes()))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model, wrapper = study.load_model(args.checkpoint)
    references = {}
    if args.reference_cache is not None:
        for domain in study.DOMAINS:
            with gzip.open(args.reference_cache / f"test-{domain}.jsonl.gz", "rt") as stream:
                records = list(map(json.loads, stream))
            indices = {i for d, i, original in eligible if d == domain and original}
            if {r["index"] for r in records} != indices or len(records) != len(indices):
                raise ValueError("Cached reference membership changed")
            for rec in records:
                references[domain, rec["index"]] = rec
    start = time.monotonic()
    processed = 0
    with torch.inference_mode():
        for serial, (domain, i, original) in enumerate(rows):
            if serial % args.parts != args.part:
                continue
            doc = data[domain, "test"][i]
            text_hash = study.text_id(doc)
            path = args.cache / f"{domain}-{i}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["identity"] != identity or saved["text_sha256"] != text_hash:
                    raise ValueError("Stale incremental row in cache")
                continue
            observed = single_forward(model, doc, schemas[domain])
            max_error = None
            if original:
                if references:
                    saved = references[domain, i]
                    if saved["text_sha256"] != text_hash:
                        raise ValueError("Reference cache text differs from the pinned source")
                    reference = cached_reference(saved)
                else:
                    reference = study.infer(model, wrapper, doc, schemas[domain])
                max_error = check_reference(doc, schemas[domain], observed, reference)
            counts = study.document_counts(doc, schemas[domain], observed)
            write_json(
                path,
                {
                    "identity": identity,
                    "text_sha256": text_hash,
                    "export_reference_max_abs_error": max_error,
                    "counts": counts,
                },
            )
            processed += 1
            if processed == 1 or processed % 25 == 0:
                print(
                    f"Processed {processed} test sentences in {time.monotonic()-start:.1f}s",
                    flush=True,
                )


def panel_report(rows):
    counts = study.aggregate(rows)
    return {
        "stock": counts["stock"],
        "misses_by_first_loss": counts["first_loss"],
        "suggestion_opportunity": counts["suggestion_opportunity"],
        "additional_edges": {name: counts[name] for name in (*study.RULES, "gate_0.40")},
        "retained_low_scored_keys": counts["retained"],
        "suppressed_low_scored_keys": counts["suppressed"],
        "reference_structure": counts["reference_structure"],
        "stronger_label_confusions": counts["stronger_label_confusions"],
    }


def paired_review_slots(rows):
    """Count sentence-level outcomes separately from changes in correct relation identity."""
    counts = Counter(dict.fromkeys((
        "no_suggestion", "neither_correct", "confidence_only_correct",
        "both_margins_only_correct", "both_correct_same_relation", "both_correct_different_relation",
    ), 0))
    for row in rows:
        baseline, candidate = row["score"], row["both_margins"]
        budget = baseline["added"]
        if budget not in (0, 1) or candidate["added"] != budget:
            raise ValueError("The paired sentence comparison requires one shared review slot")
        correct = baseline["correct_added"], candidate["correct_added"]
        changed = candidate["different_from_score_only"]
        gained = candidate["correct_gained_vs_score_only"]
        lost = candidate["correct_lost_vs_score_only"]
        if (any(value not in (0, 1) or value > budget for value in (*correct, changed))
                or (not changed and correct[0] != correct[1])
                or gained != changed * correct[1] or lost != changed * correct[0]):
            raise ValueError("Paired review counts disagree with the change in relation identity")
        if budget == 0:
            key = "no_suggestion"
        elif correct == (1, 1):
            key = "both_correct_different_relation" if changed else "both_correct_same_relation"
        else:
            key = {(0, 0): "neither_correct", (1, 0): "confidence_only_correct",
                   (0, 1): "both_margins_only_correct"}[correct]
        counts[key] += 1
    return dict(counts)


def paired_interval(strata, *, alpha=0.05):
    """Paired percentile interval for net matches, resampling review slots within each domain.

    The contribution of a paired sentence is -1, 0 or +1. Convolving these empirical
    distributions gives the ordinary stratified bootstrap without a random seed.
    """
    if not 0 < alpha < 1:
        raise ValueError("Bootstrap alpha must be between zero and one")
    probabilities = [1.0]
    observations = 0
    for counts in strata:
        slots = sum(counts.values()) - counts["no_suggestion"]
        if not slots:
            continue
        plus = counts["both_margins_only_correct"] / slots
        minus = counts["confidence_only_correct"] / slots
        zero = 1 - plus - minus
        for _ in range(slots):
            updated = [0.0] * (len(probabilities) + 2)
            for i, probability in enumerate(probabilities):
                updated[i] += minus * probability
                updated[i + 1] += zero * probability
                updated[i + 2] += plus * probability
            probabilities = updated
        observations += slots
    if not observations:
        return None
    total = sum(probabilities)
    quantiles = [alpha / 2 * total, (1 - alpha / 2) * total]
    endpoints = []
    cumulative = 0.0
    for i, probability in enumerate(probabilities):
        cumulative += probability
        while len(endpoints) < 2 and cumulative >= quantiles[len(endpoints)]:
            endpoints.append(i - observations)
    lower, upper = endpoints
    return {"lower": lower, "upper": upper,
            "method": f"{100 * (1 - alpha):.5g}% paired bootstrap within domains at fixed review slots"}


def confidence_pair_support(rows):
    """Can another stock predicate on the same directed span pair identify better suggestions?"""
    strata = {key: {"suggestions": 0, "correct": 0} for key in ("stock_pair_present", "stock_pair_absent")}
    for row in rows:
        arm = row["score"]
        if arm["added"]:
            key = "stock_pair_present" if arm["on_already_emitted_pair"] else "stock_pair_absent"
            strata[key]["suggestions"] += arm["added"]
            strata[key]["correct"] += arm["correct_added"]
    return strata


def summarize(directory, cache):
    baseline = json.loads(BASELINE.read_text())
    baseline_sha = study.digest(BASELINE.read_bytes())
    data, schemas, eligible, pins, scope = membership(directory)
    excluded, extras = outside_holdout(data, eligible)
    identity = cache_identity(pins, baseline_sha)
    rows, parity_errors = {}, []
    for domain, index, original in eligible + extras:
        path = cache / f"{domain}-{index}.json"
        if not path.is_file():
            raise ValueError(f"Missing {domain} test row {index}. Resume collect with the official scope.")
        record = json.loads(path.read_text())
        doc = data[domain, "test"][index]
        text_hash = study.text_id(doc)
        if record["text_sha256"] != text_hash or record["identity"] != identity:
            raise ValueError(
                f"Cached model or input identity differs for {domain} test row {index}"
            )
        reference = gold(doc)
        stock = record["counts"]["stock"]
        if (
            stock["sentences"] != 1
            or stock["gold"] != reference.total()
            or stock["unique_gold"] != len(reference)
            or stock["empty_gold"] != int(not reference)
            or stock["matched"] + stock["unmatched"] != stock["predicted"]
            or stock["matched"] + sum(record["counts"]["first_loss"].values()) != stock["gold"]
        ):
            raise ValueError(
                f"Cached counts disagree with the source or native loss accounting for {domain} row {index}"
            )
        error = record["export_reference_max_abs_error"]
        if (error is not None) != original or (original and not 0 <= error <= 0.0002):
            raise ValueError("Expected predicate-tensor comparisons on every original panel row")
        if original:
            parity_errors.append(error)
        rows[domain, index] = record["counts"]
    if len(rows) != sum(len(data[d, "test"]) for d in study.DOMAINS):
        raise ValueError("The release-wide replay must contain every official test occurrence")
    groups = {
        "balanced_test": [r for r in eligible if r[2]],
        "additional_test": [r for r in eligible if not r[2]],
        "all_eligible_test": eligible,
        "outside_content_holdout": extras,
        "full_official_test": eligible + extras,
    }
    controls = study.aggregate(rows[d, i] for d, i, _ in groups["balanced_test"])
    if study.public_counts(controls) != baseline["panels"]["test"]:
        raise ValueError("Original-panel predictions differ from the frozen study")
    panels = {}
    for name, cells in groups.items():
        group = panel_report(rows[d, i] for d, i, _ in cells)
        domains = {}
        for domain in study.DOMAINS:
            indices = [i for d, i, _ in cells if d == domain]
            sub = panel_report(rows[domain, i] for i in indices)
            domains[domain] = {
                "paired_review_slots": paired_review_slots(rows[domain, i] for i in indices),
                "index_text_sha256": study.digest(
                    "".join(
                        f"{i}\t{study.text_id(data[domain, 'test'][i])}\n" for i in indices
                    ).encode()
                ),
                "stock": sub["stock"],
                "one_per_sentence_upper_bound": sub["suggestion_opportunity"][
                    "one_per_sentence_upper_bound"
                ],
                "suggestions": {
                    arm: {
                        key: counts[key]
                        for key in (
                            "added",
                            "correct_added",
                            "unmatched_added",
                            "correct_gained_vs_score_only",
                            "correct_lost_vs_score_only",
                            "different_from_score_only",
                        )
                        if key in counts
                    }
                    for arm, counts in sub["additional_edges"].items()
                },
            }
        panels[name] = {
            **group, "domains": domains,
            "paired_review_slots": paired_review_slots(rows[d, i] for d, i, _ in cells),
            "confidence_by_stock_pair": confidence_pair_support(rows[d, i] for d, i, _ in cells),
        }
    # Each domain was inspected. Record both pointwise intervals and the wider
    # percentile intervals at alpha / six for familywise comparison.
    additional = panels["additional_test"]
    additional["domain_net_match_intervals"] = {
        domain: {"pointwise_95": paired_interval([entry["paired_review_slots"]]),
                 "bonferroni_family_95": paired_interval([entry["paired_review_slots"]], alpha=0.05 / len(study.DOMAINS))}
        for domain, entry in additional["domains"].items()
    }
    prior_native = study.ROOT / "docs/eval/crossre-full-prompt-comparison.json"
    prior = json.loads(prior_native.read_text())
    if (
        prior["crossre_revision"] != study.REVISION
        or prior["relation_threshold"] != study.STOCK_GATE
    ):
        raise ValueError("Independent native stock comparison uses a different input or threshold")
    comparison = {}
    for domain in study.DOMAINS:
        source = prior["domains"][domain]["corpus"]
        if (
            source["schema"] != schemas[domain]
            or source["train_sha256"] != pins[f"{domain}-train.json"]
            or source["test_sha256"] != pins[f"{domain}-test.json"]
        ):
            raise ValueError(
                f"Independent native input or training-only prompt differs for {domain}"
            )
        (arm,) = [
            item
            for item in prior["domains"][domain]["arms"]
            if item["backend"] == "gliner_relex" and item["prompt"] == "fixed"
        ]
        counts = panels["full_official_test"]["domains"][domain]["stock"]
        metrics = arm["metrics"]
        target = {
            name: int(metrics[field])
            for name, field in (
                ("gold", "num_gold_relations"),
                ("matched", "strict_relation_matches"),
                ("predicted", "num_predicted_relations"),
                ("sentences", "num_docs"),
            )
        }
        comparison[domain] = {
            "native_onnx": target,
            "public_pytorch_minus_native_onnx": {n: counts[n] - t for n, t in target.items()},
        }
    checked_probabilities = sum(
        panels["balanced_test"]["domains"][domain]["stock"]["scored_pairs"]
        * len(schemas[domain]["relation_types"])
        for domain in study.DOMAINS
    )
    return {
        "crossre_revision": study.REVISION,
        "source_sha256": pins,
        "content_holdout": scope,
        "excluded_from_content_holdout_indices": excluded,
        "exclusion_precedence": [
            "training_text",
            "historical_ai_text",
            "development_text",
            "repeated_test_text",
        ],
        "test_text_repetition": repeated_annotations(data, rows),
        "checkpoint": baseline["checkpoint"],
        "checkpoint_revision": study.MODEL_REVISION,
        "checkpoint_files_sha256": study.MODEL_FILES,
        "package_versions": baseline["package_versions"],
        "export_source_sha256": baseline["export_source_sha256"],
        "balanced_study_sha256": baseline_sha,
        "selected_on_original_development": baseline["selected_on_development"],
        "stock_gate": study.STOCK_GATE,
        "suggestion_floor": study.FLOOR,
        "candidate_weights_label_direction": study.RULES,
        "checks": {
            "source_occurrences_with_directed_stock_graph_parity": len(rows),
            "balanced_test_occurrences_compared_to_trained_export": len(parity_errors),
            "maximum_absolute_predicate_probability_difference": max(parity_errors),
            "balanced_test_directed_score_rows_checked": controls["stock"]["scored_pairs"],
            "balanced_test_predicate_probabilities_checked": checked_probabilities,
            "native_full_release_receipt_sha256": study.digest(prior_native.read_bytes()),
            "native_full_release_comparison": comparison,
        },
        "panels": panels,
    }


def tradeoff(receipt):
    """Draw the incremental review workload at measured policies after stock extraction."""
    series = []
    for key, label in (
        ("balanced_test", "Original test panel"),
        ("additional_test", "Additional content-disjoint test texts"),
        ("full_official_test", "All official test occurrences"),
    ):
        panel = receipt["panels"][key]
        stock = panel["stock"]
        points = [{"label": "Stock only", "correct": 0, "unmatched": 0}]
        for arm, name in (
            ("score", "Confidence"),
            ("label_margin", "Label margin"),
            ("direction_margin", "Direction margin"),
            ("both_margins", "Both margins"),
            ("gate_0.40", "All above 0.40"),
        ):
            data = panel["additional_edges"][arm]
            points.append(
                {
                    "label": name,
                    "correct": data["correct_added"],
                    "unmatched": data["unmatched_added"],
                }
            )
        series.append(
            {
                "label": label + " · gold missed by stock",
                "observations": stock["sentences"],
                "gold": stock["gold"] - stock["matched"],
                "points": points,
            }
        )
    return {
        "schema": "anno.tradeoff/v1",
        "title": "Which extra CrossRE relations are worth reviewing?",
        "unit": "sentences",
        "series": series,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="command", required=True)
    run = actions.add_parser(
        "collect", help="Run or resume the pinned model on test source occurrences"
    )
    run.add_argument("data", type=Path)
    run.add_argument("checkpoint", type=Path)
    run.add_argument(
        "cache", type=Path, help="Directory for resumable per-occurrence aggregate counts"
    )
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--parts", type=int, default=1)
    run.add_argument("--part", type=int, default=0)
    run.add_argument("--scope", choices=("official", "eligible", "supplement"), default="official")
    run.add_argument(
        "--reference-cache",
        type=Path,
        help="Optional local scores from the balanced trained export",
    )
    report = actions.add_parser(
        "summarize", help="Summarize all completed rows, without loading the model"
    )
    report.add_argument("data", type=Path)
    report.add_argument("cache", type=Path)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--tradeoff-output", type=Path, required=True)
    args = cli.parse_args()
    if args.command == "summarize":
        if args.output.resolve() == args.tradeoff_output.resolve():
            cli.error("Use distinct paths for the measurement and the chart input")
        receipt = summarize(args.data, args.cache)
        write_json(args.output, receipt)
        write_json(args.tradeoff_output, tradeoff(receipt))
    else:
        if args.threads < 1 or not 0 <= args.part < args.parts:
            cli.error("Set positive threads and a selected shard index within the part count")
        collect(args)
