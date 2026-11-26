"""Decompose gold DocRED relation errors using the pinned JEREX forward pass."""

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from aggregate_receipt import external_output

from jerex_stages import CHECKPOINT, DOCRED, REVISION, SPLIT, digest


def gold_clusters(doc):
    offsets = []
    start = 0
    for sentence in doc["sents"]:
        offsets.append(start)
        start += len(sentence)
    return [
        frozenset(tuple(i + offsets[m["sent_id"]] for i in m["pos"]) for m in cluster)
        for cluster in doc["vertexSet"]
    ]


class Observer:
    def __init__(self):
        self.documents = []

    def attach(self, model):
        apply = model._apply_thresholds

        def observe(
            mention,
            coref,
            entity,
            relation,
            mention_mask,
            coref_mask,
            entity_mask,
            pair_mask,
        ):
            self.mention_logits = mention[0].detach()
            self.mention_mask = mention_mask[0].bool()
            self.logits = relation[0].detach()
            self.valid_pairs = pair_mask[0].bool()
            return apply(
                mention,
                coref,
                entity,
                relation,
                mention_mask,
                coref_mask,
                entity_mask,
                pair_mask,
            )

        model._apply_thresholds = observe

    def record(self, doc, batch, result, hp):
        expected = gold_clusters(doc)
        span = batch["mention_orig_spans"][0]
        retained = {tuple(v) for v in span[result["mention_clf"][0] > 0].tolist()}
        proposed = {
            tuple(v): i for i, v in enumerate(span.tolist()) if self.mention_mask[i]
        }
        missing = {}
        for member in set().union(*expected) - retained:
            idx = proposed.get(member)
            probability = (
                None if idx is None else self.mention_logits[idx].sigmoid().item()
            )
            if probability is not None and probability >= hp["mention_threshold"]:
                raise ValueError("missing gold mention has an accepted detector score")
            missing[member] = {"span": member, "probability": probability}
        clusters = defaultdict(list)
        for index, (indices, keep) in enumerate(
            zip(result["clusters"][0], result["clusters_sample_masks"][0], strict=True)
        ):
            if keep.any():
                members = frozenset(
                    tuple(v) for v in span[indices[keep.bool()]].tolist()
                )
                clusters[members].append(index)
        duplicated = {v for v, n in Counter(expected).items() if n > 1}
        duplicated.update(v for v, indices in clusters.items() if len(indices) > 1)
        pairs = {
            tuple(p): i
            for i, p in enumerate(result["rel_entity_pairs"][0].tolist())
            if self.valid_pairs[i]
        }
        relations = hp["relation_types"]
        entity_types = {v.index: name for name, v in hp["entity_types"].items()}
        top_type = result["entity_clf"][0].argmax(-1).tolist()
        above = result["rel_clf"][0]
        outcomes = []
        for head, label, tail in sorted(
            {(e["h"], e["r"], e["t"]) for e in doc["labels"]}
        ):
            h, t = expected[head], expected[tail]
            item = {
                "head": head,
                "predicate": label,
                "tail": tail,
                "head_missing_mentions": len(h - retained),
                "tail_missing_mentions": len(t - retained),
                "head_exact_clusters": len(clusters.get(h, [])),
                "tail_exact_clusters": len(clusters.get(t, [])),
            }
            if h in duplicated or t in duplicated:
                reason = "ambiguous_cluster_alignment"
            elif not (h | t) <= retained:
                misses = [missing[m] for m in sorted((h | t) - retained)]
                item["missing_mentions"] = misses
                reason = (
                    "endpoint_mention_not_enumerated"
                    if any(m["probability"] is None for m in misses)
                    else "endpoint_mention_below_threshold"
                )
            elif h not in clusters or t not in clusters:
                for role, members in [("head", h), ("tail", t)]:
                    touching = [p for p in clusters if members.intersection(p)]
                    item[role + "_partition"] = {
                        "clusters_overlapping": len(touching),
                        "extraneous_mentions": len(set().union(*touching) - members),
                    }
                reason = "cluster_membership_mismatch"
            else:
                a, b = clusters[h][0], clusters[t][0]
                item["endpoint_types_match"] = (
                    entity_types[top_type[a]] == doc["vertexSet"][head][0]["type"]
                    and entity_types[top_type[b]] == doc["vertexSet"][tail][0]["type"]
                )
                pair = pairs.get((a, b))
                if pair is None:
                    reason = "directed_pair_not_scored"
                elif label not in relations:
                    reason = "predicate_outside_checkpoint"
                else:
                    column = relations[label].index
                    probability = self.logits[pair, column].sigmoid().item()
                    item["predicate_probability"] = probability
                    selected = bool(above[pair, column].item())
                    if selected != (probability >= hp["rel_threshold"]):
                        raise ValueError(
                            "observed predicate decision disagrees with stage logits"
                        )
                    reason = "matched" if selected else "below_relation_threshold"
            item["first_loss"] = reason
            outcomes.append(item)
        self.documents.append(
            {
                "document": doc["title"],
                "status": "scored",
                "encoded_wordpieces": batch["encodings"].shape[-1],
                "distinct_directed_pairs_scored": len(pairs),
                "relation_threshold": hp["rel_threshold"],
                "mention_threshold": hp["mention_threshold"],
                "edge_trace": outcomes,
                "first_loss": dict(Counter(o["first_loss"] for o in outcomes)),
            }
        )
        print(
            f"Traced {len(self.documents)}: {doc['title']} ({len(outcomes)} gold edges)",
            file=sys.stderr,
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("test_joint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--documents", type=int, default=32)
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional raw predicted spans, entities, and labels for external scoring",
    )
    args = parser.parse_args()
    external_output(args.output)
    if args.predictions:
        external_output(args.predictions)
    if digest(args.test_joint) != SPLIT or any(
        digest(args.model_dir / f) != h for f, h in CHECKPOINT.items()
    ):
        raise ValueError("source or checkpoint differs from the pinned release")
    docs = json.loads(args.test_joint.read_bytes())
    if not 1 <= args.documents <= len(docs):
        parser.error("document count must be within the held-out split")
    docs = docs[: args.documents]
    sys.path.insert(0, str(DOCRED))
    import subprocess

    import jerex
    from predict_jerex import predict
    from score_jerex import convert, stage_counts

    checkout = Path(jerex.__file__).resolve().parent.parent
    if (
        subprocess.check_output(
            ["git", "-C", checkout, "rev-parse", "HEAD"], text=True
        ).strip()
        != REVISION
    ):
        raise ValueError("unexpected upstream JEREX checkout")
    subprocess.run(
        ["git", "-C", checkout, "diff", "--quiet", "HEAD", "--", "jerex"], check=True
    )
    observer = Observer()
    with tempfile.TemporaryDirectory() as tmp:
        output = args.predictions or Path(tmp) / "predictions.json"
        predict(args.model_dir, docs, output, observer=observer)
        raw = json.loads(output.read_bytes())
    gold, pred = convert(docs, raw)
    totals = Counter()
    for d, g, p, r in zip(observer.documents, gold, pred, raw, strict=True):
        stages = stage_counts([g], [p], [r])
        d["stages"] = stages
        d["source_sha256"] = hashlib.sha256(
            json.dumps(g["sents"], ensure_ascii=False).encode()
        ).hexdigest()
        if (
            len(d["edge_trace"]) != stages["labeled_relations"]["gold"]
            or d["first_loss"].get("matched", 0)
            != stages["labeled_relations"]["matched"]
        ):
            raise ValueError(
                f"{d['document']}: edge trace differs from independent writer score"
            )
        totals.update(d["first_loss"])
    report = {
        "split_sha256": SPLIT,
        "checkpoint_files": CHECKPOINT,
        "upstream_revision": REVISION,
        "selection": "first N in published order",
        "enrolled_documents": len(docs),
        "scored_documents": len(raw),
        "stages": stage_counts(gold, pred, raw),
        "first_loss": dict(totals),
        "documents": observer.documents,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
