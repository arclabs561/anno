# SPDX-License-Identifier: MIT OR Apache-2.0
"""Verify external joint-extraction diagnostics against a public aggregate and digest."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


CROSSRE_COUNTS = """scored_windows gold_mentions scored_mentions gold_mentions_scored
decoded_mentions gold_mentions_decoded gold_directed_pairs gold_directed_pairs_co_window
gold_edges gold_edges_co_window gold_edges_both_mentions_scored actual_scored_edges
gold_edges_scored decoded_edges gold_edges_decoded sparse_directed_predicate_span_pairs
feasible_documents""".split()
CROSSRE_LOSSES = """predicate_outside_schema endpoint_types_outside_schema outside_shared_window
missing_typed_mention_in_shared_window not_in_sparse_edge_output below_edge_candidate_threshold
edge_cap optimizer_endpoint_pruning optimizer_did_not_select result_builder_omitted matched""".split()
JEREX_STAGES = """mention_spans within_cluster_mention_pairs exact_clusters exact_typed_clusters
directed_cluster_pairs labeled_relations labeled_relations_typed_endpoints""".split()
JEREX_LOSSES = """ambiguous_cluster_alignment endpoint_mention_not_enumerated
endpoint_mention_below_threshold cluster_membership_mismatch directed_pair_not_scored
predicate_outside_checkpoint below_relation_threshold matched""".split()
PREDICTION_OUTCOMES = """matched endpoint_has_no_exact_cluster ambiguous_exact_membership
different_predicate_on_annotated_pair same_predicate_only_in_reverse directed_pair_unannotated""".split()


def counts(values, permitted):
    if not values.keys() <= set(permitted) or any(type(n) is not int or n < 0 for n in values.values()):
        raise ValueError("unexpected aggregate category or non-count value")
    return dict(sorted(values.items()))


def equal(actual, expected, description):
    if actual != expected:
        raise ValueError(description)


def jerex_stages(rows):
    summary = {}
    for stage in JEREX_STAGES:
        metrics = Counter()
        for row in rows:
            metrics.update(counts(row["stages"][stage], ("gold", "matched", "predicted", "matched_upper_bound")))
        summary[stage] = dict(metrics)
    return summary


def external_output(path):
    if path.resolve().is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("write full annotations and predictions outside the Anno checkout")


def project(report, kind):
    from jerex_stages import CHECKPOINT, SPLIT
    from jointie_coverage import TEST_SHA, WEIGHTS_SHA

    rows = report["documents"]
    if not rows:
        raise ValueError("expected external per-document diagnostic rows")
    serialized = json.dumps(report, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    summary = {"study": kind, "documents": len(rows),
               "full_receipt_sha256": hashlib.sha256(serialized.encode()).hexdigest()}
    if kind == "jerex-predictions":
        equal(report["source_sha256"], SPLIT, "wrong JEREX split")
        outcomes = Counter()
        for row in rows:
            observed = Counter(item["outcome"] for item in row["predictions"])
            equal(observed, row["outcomes"], "prediction outcomes differ from individual edges")
            outcomes.update(observed)
        equal(dict(outcomes), report["outcomes"], "prediction outcome totals differ")
        summary.update(source_sha256=SPLIT, outcomes=counts(outcomes, PREDICTION_OUTCOMES))
    elif kind == "jerex-stages":
        equal(report["split_sha256"], SPLIT, "wrong JEREX split")
        equal(report["checkpoint_files"], CHECKPOINT, "wrong JEREX checkpoint")
        stages = jerex_stages(rows)
        equal(stages, report["stages"], "JEREX stage totals differ from individual documents")
        losses = Counter()
        for row in rows:
            trace = row["edge_trace"]
            edges = {(e["head"], e["predicate"], e["tail"]) for e in trace}
            labeled = row["stages"]["labeled_relations"]
            equal(len(trace), len(edges), "duplicate reference edges")
            equal(len(trace), labeled["gold"], "reference inventory differs")
            observed = Counter(e["first_loss"] for e in trace)
            equal(dict(observed), row["first_loss"], "reference-edge outcomes differ")
            equal(observed["matched"], labeled["matched"], "reference matches differ")
            for edge in trace:
                if "predicate_probability" in edge:
                    equal(edge["predicate_probability"] >= row["relation_threshold"],
                          edge["first_loss"] == "matched", "relation threshold disagrees with match")
            losses.update(observed)
        equal(dict(losses), report["first_loss"], "reference-edge loss totals differ")
        baseline = json.loads((Path(__file__).parent / "receipts/jerex.json").read_bytes())
        prefix = rows[:len(baseline["titles"])]
        equal([row["document"] for row in prefix], baseline["titles"], "JEREX historical prefix differs")
        equal(jerex_stages(prefix), baseline["stages"], "JEREX historical stage counts differ")
        summary.update(source_sha256=SPLIT, model_sha256=CHECKPOINT["model.ckpt"],
                       stages=stages, first_loss=counts(losses, JEREX_LOSSES))
    elif kind == "jointie":
        equal(report["dataset_sha256"], TEST_SHA, "wrong CrossRE split")
        equal(report["checkpoint_files"]["model.safetensors"], WEIGHTS_SHA, "wrong JointIE checkpoint")
        totals = Counter()
        permitted = CROSSRE_COUNTS + ["first_loss_" + name for name in CROSSRE_LOSSES]
        for row in rows:
            totals.update(counts(row["counts"], permitted))
        equal(dict(totals), report["totals"], "CrossRE totals differ from individual documents")
        for setting in ("window_words", "overlap_words"):
            value = report[setting]
            if type(value) is not int or value < 0:
                raise ValueError("invalid window configuration")
            summary[setting] = value
        summary.update(source_sha256=TEST_SHA, model_sha256=WEIGHTS_SHA,
                       totals=dict(sorted(totals.items())))
        if "training_sha256" in report:
            training = "44d54f2669c2a41f613aac70095aed8a3b8f2e6befde454abb99bf97c77385d3"
            equal(report["training_sha256"], training, "wrong training route source")
            summary["training_sha256"] = training
    else:
        raise ValueError("unknown study")
    if kind != "jerex-predictions":
        equal(len(rows), report["enrolled_documents"], "enrollment differs from source rows")
        equal(len(rows), report["scored_documents"], "score count differs from source rows")
    return summary


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("study", choices=("jointie", "jerex-stages", "jerex-predictions"))
    cli.add_argument("full", type=Path, help="local, authorized full diagnostic")
    cli.add_argument("published", type=Path, help="checked-in aggregate receipt")
    args = cli.parse_args()
    summary = project(json.loads(args.full.read_bytes()), args.study)
    equal(summary, json.loads(args.published.read_bytes()), "public counts or full receipt digest differ")
    print(f"Verified {summary['documents']} documents against {args.published.name}.")


if __name__ == "__main__":
    main()
