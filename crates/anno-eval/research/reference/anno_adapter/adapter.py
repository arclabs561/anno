"""Replay Anno's actual predicted mentions with document-local memory policies."""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[4] / "scripts/qa"))

from entity_memory_replay import Memory, link_confidence, prefix_vector
from score_person_mentions import annotation_maps, pair_counts, selection

WORD = re.compile(r"\w+|[^\w\s]")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def predict(binary, documents):
    """Run the named offline backend and bind each result to its source ID."""
    with TemporaryDirectory(prefix="anno-memory-") as temporary:
        root = Path(temporary)
        names = {}
        for i, doc in enumerate(documents):
            name = f"doc-{i:04}"
            names[name] = doc["id"]
            (root / f"{name}.txt").write_bytes(doc["text"].encode("utf-8"))
        env = {**os.environ, "ANNO_NO_DOWNLOADS": "1", "HF_HUB_OFFLINE": "1",
               "FCOREF_MODEL_PATH": str(root / "no-model")}
        result = subprocess.run(
            [str(binary), "pipeline", "--dir", str(root), "--model", "heuristic",
             "--coref", "--format", "grounded", "--quiet"],
            cwd=root, env=env, capture_output=True, check=True,
        )
    predictions = {}
    for row in json.loads(result.stdout):
        key = names[row["id"]]
        if key in predictions:
            raise ValueError(f"duplicate CLI document {key}")
        predictions[key] = row
    if predictions.keys() != set(names.values()):
        raise ValueError("CLI predictions do not match submitted document IDs")
    return predictions


def read_grounded(doc, row, backend):
    """Validate exact source coordinates and preserve original signal and track IDs."""
    text = doc["text"]
    if row["text"] != text:
        raise ValueError(f"{doc['id']}: CLI text differs from annotation source")
    run = row.get("coref_run") or {}
    if run.get("status") != "completed" or run.get("backend") != backend:
        raise ValueError(f"{doc['id']}: expected completed {backend} run, got {run}")
    ids = set()
    locations = {}
    unsupported = []
    for signal in row["signals"]:
        signal_id = signal["id"]
        if type(signal_id) is not int or signal_id < 0 or signal_id in ids:
            raise ValueError(f"{doc['id']}: invalid or duplicate signal ID {signal_id}")
        ids.add(signal_id)
        location = signal["location"]
        if set(location) != {"Text"}:
            unsupported.append(signal_id)
            continue
        start, end = location["Text"]["start"], location["Text"]["end"]
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(text)
                or text[start:end] != signal["surface"]):
            raise ValueError(f"{doc['id']}: signal {signal_id} is not source-aligned")
        locations[signal_id] = (start, end)
    assigned = set()
    tracks = {}
    for key, track in row["tracks"].items():
        track_id = track["id"]
        if type(track_id) is not int or str(track_id) != key or track_id < 0:
            raise ValueError(f"{doc['id']}: inconsistent track ID {key}")
        members = []
        for reference in track["signals"]:
            if set(reference) != {"signal_id", "position"}:
                raise ValueError(f"{doc['id']}: expected a document-local signal reference")
            signal_id = reference["signal_id"]
            if type(signal_id) is not int or signal_id not in ids or signal_id in assigned:
                raise ValueError(f"{doc['id']}: missing or duplicate track member {signal_id}")
            assigned.add(signal_id)
            members.append(signal_id)
        tracks[track_id] = members
    return locations, tracks, sorted(unsupported)


def partitions(groups, locations, mentions, lookup, negatives):
    """Separate document-level mistakes from missing and unjudged signal spans."""
    spans = set(locations.values())
    cluster_spans = [
        {locations[signal] for signal in signals if signal in locations}
        for signals in groups.values()
    ]
    counts = pair_counts(mentions, cluster_spans, lookup, spans)
    counts["false_split_pairs_covered"] = counts["covered_positive"] - counts["joined_positive"]
    counts["false_merge_pairs"] = counts["joined_negative"]
    counts["missed_pairs_detection"] = counts["positive_pairs"] - counts["covered_positive"]

    gold_groups = defaultdict(set)
    for mention_id, mention in mentions.items():
        if (person := mention["person_id"]) is not None:
            gold_groups[person].add(mention_id)
    owners = defaultdict(set)
    track_rows = []
    for key, group in groups.items():
        excluded = [q for q in group if q not in locations]
        spans_in_group = [locations[q] for q in group if q in locations]
        matched = [lookup[bounds] for bounds in spans_in_group if bounds in lookup]
        mapped = set(matched)
        duplicate_occurrences = len(matched) - len(mapped)
        people = {mentions[q]["person_id"] for q in mapped} - {None}
        unjudged = sum(p not in lookup and p not in negatives for p in spans_in_group)
        distractors = sum(p in negatives for p in spans_in_group)
        unresolved = sum(mentions[q]["person_id"] is None for q in mapped)
        for q in mapped:
            owners[q].add(key)
        track_rows.append({
            "cluster_id": key,
            "signal_ids": group,
            "gold_mention_ids": sorted(mapped),
            "known_identities": sorted(people),
            "false_merge": len(people) > 1,
            "exact_gold_cluster": (
                len(people) == 1
                and not (excluded or unjudged or distractors or unresolved or duplicate_occurrences)
                and mapped == gold_groups[next(iter(people))]
            ),
            "excluded_signal_ids": excluded,
            "duplicate_gold_occurrences": duplicate_occurrences,
            "unjudged_signals": unjudged,
            "explicit_negative_signals": distractors,
            "unresolved_mentions": unresolved,
        })
    duplicate_memberships = sum(len(groups_at_span) > 1 for groups_at_span in owners.values())
    return {
        "pairs": counts,
        "clusters": {
            "gold_entities": len(gold_groups),
            "predicted_clusters": len(groups),
            "mixed_known_identity_clusters": sum(t["false_merge"] for t in track_rows),
            "exact_gold_clusters": sum(t["exact_gold_cluster"] for t in track_rows),
            "clusters_with_excluded_signals": sum(bool(t["excluded_signal_ids"]) for t in track_rows),
            "gold_entities_split_across_clusters": sum(
                len(set().union(*(owners[q] for q in gold))) > 1
                for gold in gold_groups.values()
            ),
            "gold_mentions_in_multiple_predicted_clusters": duplicate_memberships,
        },
        "cluster_details": track_rows,
    }


def feature(text, start, end, dimension):
    context = WORD.findall(text[:start])[-12:]
    tokens = context + WORD.findall(text[start:end])
    return prefix_vector(tokens, len(context), dimension)


def run_memory(text, locations, policy, dimension, budget, k, threshold):
    """Close the loop using only Anno's signal positions and the lexical encoder."""
    memory = Memory(policy, dimension, budget)
    order = sorted(locations, key=lambda q: (locations[q][1], locations[q][0], q))
    state = defaultdict(list)
    rows = []
    for step, signal_id in enumerate(order):
        query = feature(text, *locations[signal_id], dimension)
        ranked = memory.rank(query, k)
        confidence = link_confidence(ranked) if ranked else None
        target = ranked[0][0] if ranked else None
        attached = target is not None and confidence >= threshold
        cluster_id = target if attached else signal_id
        rows.append({
            "signal_id": signal_id,
            "resident_cluster_ids": [int(q) for q in memory.ids if q >= 0],
            "candidate_cluster_ids": [q for q, _ in ranked],
            "target_cluster_id": target,
            "target_prior_signal_ids": list(state[target]) if target is not None else [],
            "assigned_cluster_id": cluster_id,
            "accepted": attached,
            "ranking_support": confidence,
        })
        state[cluster_id].append(signal_id)
        memory.update(cluster_id, query, step)
    return dict(state), rows, {"bytes": memory.bytes, "capacity": memory.capacity}


def audit_replay(rows, locations, mentions, lookup):
    """Annotate a finished causal trace. Gold labels cannot affect its decisions."""
    known = defaultdict(set)
    counts = Counter()
    decisions = []
    for row in rows:
        gold = lookup.get(locations[row["signal_id"]])
        person = mentions[gold]["person_id"] if gold is not None else None
        correct_clusters = {
            owner for previous, owner in known[person] if previous != gold
        } if person is not None else set()
        target = row["target_cluster_id"]
        if gold is None:
            outcome = "unmatched_source_span"
        elif person is None:
            outcome = "unresolved_gold_mention"
        elif not correct_clusters:
            outcome = "no_observed_distinct_antecedent"
        elif correct_clusters.isdisjoint(row["resident_cluster_ids"]):
            outcome = "retention_miss"
        elif correct_clusters.isdisjoint(row["candidate_cluster_ids"]):
            outcome = "candidate_miss"
        elif target not in correct_clusters:
            outcome = "ranking_miss"
        elif not row["accepted"]:
            outcome = "threshold_abstention"
        else:
            outcome = "accepted_gold_supported"
        counts[outcome] += 1

        target_owners = {
            mentions[member]["person_id"]
            for signal in row["target_prior_signal_ids"]
            if (member := lookup.get(locations[signal])) is not None
        } - {None}
        wrong = row["accepted"] and person is not None and bool(target_owners - {person})
        counts["accepted_links"] += row["accepted"]
        counts["known_wrong_or_impure_accepted_targets"] += wrong
        decisions.append({
            **row,
            "gold_mention_id": gold,
            "outcome": outcome,
            "known_wrong_or_impure_target": wrong,
        })
        if person is not None:
            known[person].add((gold, row["assigned_cluster_id"]))
    return {"counts": dict(counts), "decisions": decisions}


def audit(doc, prediction, args):
    text = doc["text"]
    locations, tracks, unsupported = read_grounded(doc, prediction, args.backend)
    mentions, contracts, negatives = annotation_maps(doc)
    runs = {
        policy: run_memory(
            text, locations, policy, args.dimension, args.budget, args.k, args.threshold,
        )
        for policy in args.policy
    }
    native = {}
    replay = {}
    for contract, lookup in contracts.items():
        native[contract] = partitions(tracks, locations, mentions, lookup, negatives)
        replay[contract] = {
            policy: {
                **memory,
                **audit_replay(rows, locations, mentions, lookup),
                **partitions(groups, locations, mentions, lookup, negatives),
            }
            for policy, (groups, rows, memory) in runs.items()
        }
    emitted = set(locations.values())
    return {
        "id": doc["id"],
        "anno_document_id": prediction["id"],
        "source_sha256": digest(text.encode("utf-8")),
        "coref_run": prediction["coref_run"],
        "signals": [{"id": q, "start": p[0], "end": p[1]} for q, p in sorted(locations.items())],
        "duplicate_contiguous_spans": len(locations) - len(emitted),
        "excluded_noncontiguous_signal_ids": unsupported,
        "native_wire_track_confidence": {
            str(track["id"]): track.get("cluster_confidence") for track in prediction["tracks"].values()
        },
        "alignment": {
            contract: selection(emitted, lookup, negatives, len(mentions))
            for contract, lookup in contracts.items()
        },
        "native": native,
        "replay": replay,
    }


def summarize(documents, fields):
    total = Counter()
    for doc in documents:
        value = doc
        for field in fields:
            value = value[field]
        total.update(value)
    return dict(total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--backend", default="mention-ranking")
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=-0.5)
    parser.add_argument("--policy", action="append", choices=("exemplars", "box", "midpoint"))
    parser.add_argument("--raw-output", type=Path, help="Optional local file with full CLI outputs")
    parser.add_argument("--summary-only", action="store_true", help="Omit per-document traces")
    args = parser.parse_args()
    args.policy = list(dict.fromkeys(args.policy or ("exemplars", "midpoint", "box")))
    if min(args.budget, args.dimension, args.k) < 1 or not math.isfinite(args.threshold):
        parser.error("budget, dimension and k must be positive, and threshold must be finite")
    binary = args.anno.resolve(strict=True)
    panel_bytes = args.gold.read_bytes()
    panel = json.loads(panel_bytes)
    if panel.get("offset_unit") != "unicode_scalar" or panel.get("schema_version") != 1:
        parser.error("expected version 1 person panel with Unicode scalar offsets")
    documents = panel["documents"]
    ids = [doc["id"] for doc in documents]
    if not ids or len(set(ids)) != len(ids):
        parser.error("expected unique document IDs in a nonempty panel")
    predictions = predict(binary, documents)
    if args.raw_output is not None:
        args.raw_output.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    rows = [audit(doc, predictions[doc["id"]], args) for doc in documents]
    totals = {}
    for contract in rows[0]["alignment"]:
        totals[contract] = {
            "alignment": summarize(rows, ("alignment", contract)),
            "native_pairs": summarize(rows, ("native", contract, "pairs")),
            "native_clusters": summarize(rows, ("native", contract, "clusters")),
            "replay": {
                policy: {
                    key: summarize(rows, ("replay", contract, policy, key))
                    for key in ("counts", "pairs", "clusters")
                }
                for policy in args.policy
            },
        }
    print(json.dumps({
        "schema_version": 1,
        "offset_unit": "unicode_scalar",
        "corpus_sha256": digest(panel_bytes),
        "anno_binary_sha256": digest(binary.read_bytes()),
        "anno_version": subprocess.check_output([str(binary), "--version"], text=True).strip(),
        "ner_backend": "heuristic",
        "expected_coref_backend": args.backend,
        "replay_config": {
            "encoder": "lexical_hash_visible_prefix",
            "dimension": args.dimension,
            "budget": args.budget,
            "k": args.k,
            "threshold": args.threshold,
            "policies": args.policy,
        },
        "totals": totals,
        "document_sources": [
            {key: row[key] for key in (
                "id", "anno_document_id", "source_sha256", "coref_run",
                "native_wire_track_confidence",
            )}
            for row in rows
        ],
        **({} if args.summary_only else {"documents": rows}),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
