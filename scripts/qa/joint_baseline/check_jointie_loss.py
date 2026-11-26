"""Audit the expanded CrossRE receipts against source annotations and frozen eight-row receipts."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from jointie_coverage import SCHEMA, gold_graph


def check_stage(item, selected, routes):
    head, predicate, tail = tuple(item["head"]), item["predicate"], tuple(item["tail"])
    matched = (head, predicate, tail) in selected
    loss = item["first_loss"]
    if (loss == "matched") != matched:
        raise ValueError("matched attribution disagrees with the named decoded edge")

    co_windows, local = item["co_windows"], item["joint_mention_windows"]
    if local != sorted(set(local)) or not set(local) <= set(co_windows):
        raise ValueError("local mention windows disagree with shared windows")
    route = routes.get(predicate)
    allowed = route is not None and head[0] in route[0] and tail[0] in route[1]
    if "schema_allows_gold_types" in item and item["schema_allows_gold_types"] is not allowed:
        raise ValueError("schema flag disagrees with declared endpoint types")

    if route is None:
        expected = "predicate_outside_schema"
    elif not allowed:
        expected = "endpoint_types_outside_schema"
    elif not co_windows:
        expected = "outside_shared_window"
    elif not local:
        expected = "missing_typed_mention_in_shared_window"
    elif "probability" not in item:
        expected = "not_in_sparse_edge_output"
    else:
        probability, gate = item["probability"], item["candidate_threshold"]
        rank, cap = item["rank_after_gate"], item["per_predicate_cap"]
        endpoints = item["optimizer_endpoint_count"]
        if not (0 <= probability <= 1 and 0 <= gate <= 1 and endpoints in (0, 1, 2)
                and type(cap) is int and cap >= 0):
            raise ValueError("invalid candidate evidence")
        if probability < gate:
            if rank is not None:
                raise ValueError("sub-threshold candidate has a rank")
            expected = "below_edge_candidate_threshold"
        elif type(rank) is not int or rank < 1:
            raise ValueError("eligible candidate lacks a positive rank")
        elif rank > cap:
            expected = "edge_cap"
        elif endpoints < 2:
            expected = "optimizer_endpoint_pruning"
        else:
            if not math.isfinite(item.get("optimizer_edge_utility", math.nan)):
                raise ValueError("optimizer attribution lacks solver input")
            # Old receipts did not persist selected solver edges separately.
            if not matched and loss in ("optimizer_did_not_select", "result_builder_omitted"):
                return
            expected = "matched"
    if loss != expected:
        raise ValueError(f"edge stage disagrees with evidence: {loss}, expected {expected}")


def check(source, expanded, previous=None):
    rows = expanded["documents"]
    if len(rows) != len(source) or not rows:
        raise ValueError("source and receipt counts differ")
    if expanded["enrolled_documents"] != len(rows) or expanded[
        "scored_documents"
    ] != len(rows):
        raise ValueError("enrollment or scored count differs from the document rows")
    vocab = json.loads(SCHEMA.read_bytes())
    routes = expanded.get("allowed_endpoint_types", {
        p: (vocab["entity_types"], vocab["entity_types"]) for p in vocab["relation_types"]
    })
    totals = Counter()
    for document, row in zip(source, rows, strict=True):
        text, gold_mentions, gold = gold_graph(document)
        if hashlib.sha256(text.encode()).hexdigest() != row["source_sha256"]:
            raise ValueError("source text digest differs")
        if row["document"] != document["doc_key"]:
            raise ValueError("document order changed")
        selected = {(tuple(h), p, tuple(t)) for h, p, t in row["decoded_edge_keys"]}
        predicted_mentions = set(map(tuple, row["decoded_mention_keys"]))
        if len(selected) != len(row["decoded_edge_keys"]):
            raise ValueError("duplicate decoded edge in written graph")
        if any(
            h not in predicted_mentions or t not in predicted_mentions
            for h, _, t in selected
        ):
            raise ValueError("decoded edge references an absent mention")
        counts = row["counts"]
        if (
            len(gold),
            len(selected),
            len(gold & selected),
            len(gold_mentions & predicted_mentions),
            len(gold_mentions),
            len(predicted_mentions),
        ) != (
            counts["gold_edges"],
            counts["decoded_edges"],
            counts["gold_edges_decoded"],
            counts["gold_mentions_decoded"],
            counts["gold_mentions"],
            counts["decoded_mentions"],
        ):
            raise ValueError(
                f"{row['document']}: decoded edge or mention counts differ"
            )
        keys = set()
        losses = Counter()
        for item in row["edge_trace"]:
            h, t = tuple(item["head"]), tuple(item["tail"])
            key = (h, item["predicate"], t)
            if key in keys:
                raise ValueError("duplicate annotated edge in trace")
            keys.add(key)
            if item["co_windows"] != [
                i
                for i, w in enumerate(row["windows"])
                if w["start"] <= min(h[1], t[1]) and max(h[2], t[2]) <= w["end"]
            ]:
                raise ValueError("co-window coverage disagrees with source spans")
            loss = item["first_loss"]
            losses[loss] += 1
            check_stage(item, selected, routes)
        if keys != gold or losses != {
            k.removeprefix("first_loss_"): n
            for k, n in counts.items()
            if k.startswith("first_loss_")
        }:
            raise ValueError(
                f"{row['document']}: gold coverage or loss categories differ"
            )
        totals.update(counts)
    if dict(totals) != expanded["totals"]:
        raise ValueError("aggregate totals disagree with per-document counts")
    if previous:
        for old, current in zip(
            previous["documents"], rows[: len(previous["documents"])], strict=True
        ):
            if (old["document"], old["windows"], old["source_sha256"]) != (
                current["document"],
                current["windows"],
                current["source_sha256"],
            ) or any(n != current["counts"][k] for k, n in old["counts"].items()):
                raise ValueError(f"historical trace differs on {old['document']}")
            for field in ("decoded_mention_keys", "decoded_edge_keys", "edge_trace"):
                if field in old and current[field] != old[field]:
                    raise ValueError(f"historical {field} differs on {old['document']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crossre", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    expanded = json.loads(args.receipt.read_bytes())
    raw_source = args.crossre.read_bytes()
    if hashlib.sha256(raw_source).hexdigest() != expanded["dataset_sha256"]:
        raise ValueError("source dataset digest differs")
    source = [
        json.loads(s) for s in raw_source.splitlines()[: len(expanded["documents"])]
    ]
    check(
        source,
        expanded,
        json.loads(args.previous.read_bytes()) if args.previous else None,
    )
    print(
        f"Verified {len(source)} source documents and {expanded['totals']['gold_edges']} labeled edges."
    )


if __name__ == "__main__":
    main()
