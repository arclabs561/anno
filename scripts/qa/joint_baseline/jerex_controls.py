# SPDX-License-Identifier: MIT OR Apache-2.0
"""Measure JEREX clustering and decision-threshold effects on the same detected mentions."""
import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jerex_loss import gold_clusters
from jerex_revised_reference import REVISION as REVISED_REVISION, load_revised
from jerex_stages import CHECKPOINT, DOCRED, REVISION, SPLIT, digest

THRESHOLDS = (0.25, 0.5, 0.75)
ARMS = ("predicted_clusters", "reference_partition_of_detected_mentions")


def partition_detected(spans, reference):
    """Every detected span appears once, including unannotated singleton detections."""
    if len(spans) != len(set(spans.values())):
        raise ValueError("duplicate detected span")
    assigned = set()
    groups = []
    for cluster in reference:
        members = [i for i, span in spans.items() if span in cluster]
        if assigned.intersection(members):
            raise ValueError("detected span assigned to multiple reference entities")
        if members:
            groups.append(members)
            assigned.update(members)
    groups.extend([i] for i in spans if i not in assigned)
    return groups


def reference_edges(document):
    clusters = gold_clusters(document)
    if len(clusters) != len(set(clusters)) or any(not group for group in clusters):
        raise ValueError("exact cluster membership is empty or ambiguous")
    return {(clusters[e["h"]], e["r"], clusters[e["t"]]) for e in document["labels"]}


class ControlledReplay:
    def __init__(self, partitions, progress=None, revised=None, start=0):
        self.progress = progress
        self.revised = revised
        self.start = start
        self.partitions = partitions
        self.runs = []

    def attach(self, model):
        import torch
        from jerex.models import misc

        forward = model._forward_inference

        def controlled_forward(**kwargs):
            index, arm = divmod(len(self.runs), len(ARMS))
            if arm == 0:
                return forward(**kwargs)
            spans = kwargs["mention_orig_spans"][0].tolist()

            def make_partition(_coref, _pairs, _sample_mask, valid, _threshold):
                if valid.shape[0] != 1:
                    raise ValueError("controlled replay expects one document per batch")
                selected = {i: tuple(spans[i]) for i in valid[0].nonzero().flatten().tolist()}
                groups = partition_detected(selected, self.partitions[index])
                width = max(map(len, groups), default=1)
                ids = torch.zeros((1, max(1, len(groups)), width), dtype=torch.long, device=valid.device)
                mask = torch.zeros_like(ids, dtype=torch.bool)
                for i, group in enumerate(groups):
                    ids[0, i, :len(group)] = torch.tensor(group, device=ids.device)
                    mask[0, i, :len(group)] = True
                return ids, mask

            # The pinned upstream joint module calls this function between detection and relation scoring.
            with patch.object(misc, "create_clusters", make_partition):
                return forward(**kwargs)

        model._forward_inference = controlled_forward
        apply = model._apply_thresholds

        def observe(mention, coref, entity, relation, mention_mask, coref_mask, entity_mask, pair_mask):
            self.scores = relation[0].sigmoid().detach()
            self.valid = pair_mask[0].bool()
            return apply(mention, coref, entity, relation, mention_mask, coref_mask, entity_mask, pair_mask)

        model._apply_thresholds = observe

    def record(self, document, batch, result, hp):
        gold = reference_edges(document)
        spans = batch["mention_orig_spans"][0]
        detected = {tuple(s) for s in spans[result["mention_clf"][0] > 0].tolist()}
        clusters = [
            frozenset(map(tuple, spans[indices[mask.bool()]].tolist()))
            for indices, mask in zip(result["clusters"][0], result["clusters_sample_masks"][0], strict=True)
        ]
        pairs = result["rel_entity_pairs"][0].tolist()
        vocabulary = {v.index: k for k, v in hp["relation_types"].items()}
        default = float(hp["rel_threshold"])
        by_threshold = {}
        for threshold in sorted(set((*THRESHOLDS, default))):
            chosen = (self.scores >= threshold) & self.valid.unsqueeze(-1)
            if threshold == default and not chosen.equal(result["rel_clf"][0] > 0):
                raise ValueError("observed scores disagree with upstream decoder")
            selected = set()
            for row, col in chosen.nonzero().tolist():
                h, t = pairs[row]
                selected.add((clusters[h], vocabulary[col], clusters[t]))
            by_threshold[threshold] = (selected & gold, selected - gold)
        self.runs.append((detected, gold, by_threshold, default))
        if len(self.runs) % len(ARMS) == 0:
            pairs = len(self.runs) // len(ARMS)
            if self.progress is not None:
                final_pair = self.runs[-len(ARMS):]
                row = {"document_index": self.start + pairs - 1, "arms": summarize(final_pair)}
                if self.revised is not None:
                    row["revised_arms"] = summarize(rescore(final_pair, [document], self.revised))
                self.progress.write(json.dumps(row) + "\n")
                self.progress.flush()
            print(f"Completed {pairs} paired documents", file=sys.stderr, flush=True)


def summarize(runs):
    totals = {arm: {} for arm in ARMS}
    for base, intervention in zip(runs[::2], runs[1::2], strict=True):
        detected, gold, original, default = base
        if (detected, gold, default) != (intervention[0], intervention[1], intervention[3]):
            raise ValueError("mention detections, source gold or threshold differ between paired arms")
        baseline_tp = original[default][0]
        for arm, run in zip(ARMS, (base, intervention), strict=True):
            for threshold, (tp, fp) in run[2].items():
                row = totals[arm].setdefault(str(threshold), Counter())
                row.update(gold=len(gold), matched=len(tp), missed=len(gold - tp), extras=len(fp),
                           gained=len(tp - baseline_tp), lost=len(baseline_tp - tp))
    return totals


def rescore(runs, documents, revised):
    rescored = []
    for i, (detected, _gold, thresholds, default) in enumerate(runs):
        doc = documents[i // len(ARMS)]
        alternate = revised[doc["title"]][1]
        if alternate["sents"] != doc["sents"]:
            raise ValueError("revised edition changed source tokens")
        gold = reference_edges(alternate)
        by_threshold = {}
        for threshold, (tp, fp) in thresholds.items():
            predictions = tp | fp
            by_threshold[threshold] = (predictions & gold, predictions - gold)
        rescored.append((detected, gold, by_threshold, default))
    return rescored


def paired_intervals(runs):
    rows = [summarize([base, intervention]) for base, intervention in zip(runs[::2], runs[1::2], strict=True)]
    return intervals_from_counts(rows, str(runs[0][3]))


def intervals_from_counts(rows, default):
    """Document bootstrap of micro-F1 differences in percentage points, with pairing intact."""
    thresholds = sorted(rows[0][ARMS[0]])

    def counts(row):
        return row["matched"], row["gold"] + row["matched"] + row["extras"]

    def micro_f1(sample, values):
        matched = sum(values[i][0] for i in sample)
        denominator = sum(values[i][1] for i in sample)
        return 200 * matched / denominator if denominator else None

    baseline = [counts(row[ARMS[0]][default]) for row in rows]
    rng = Random(53)
    draws = [rng.choices(range(len(rows)), k=len(rows)) for _ in range(2000)]
    references = [micro_f1(draw, baseline) for draw in draws]
    intervals = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for threshold in thresholds:
            candidate = [counts(row[arm][threshold]) for row in rows]
            changes = []
            for indices, ref in zip(draws, references, strict=True):
                value = micro_f1(indices, candidate)
                if ref is not None and value is not None:
                    changes.append(value - ref)
            changes.sort()
            if len(changes) != len(draws):
                raise ValueError("undefined F1 in a paired bootstrap replicate")
            intervals[arm][str(threshold)] = [
                round(changes[int(q * (len(changes) - 1))], 3) for q in (0.025, 0.975)
            ]
    return intervals


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model", type=Path)
    cli.add_argument("test_joint", type=Path)
    cli.add_argument("--start", type=int, default=0)
    cli.add_argument("--documents", type=int, default=32)
    cli.add_argument("--revised-directory", type=Path, help="also rescore the identical predictions against Re-DocRED")
    cli.add_argument("--progress", type=Path, help="new external JSONL file for per-document aggregate counts")
    args = cli.parse_args()
    if digest(args.test_joint) != SPLIT or any(digest(args.model / f) != h for f, h in CHECKPOINT.items()):
        raise ValueError("unexpected dataset or weights")
    docs = json.loads(args.test_joint.read_bytes())
    if args.start < 0 or args.documents < 1 or args.start + args.documents > len(docs):
        cli.error("document slice outside the published test split")
    docs = docs[args.start:args.start + args.documents]
    import jerex
    root = Path(jerex.__file__).resolve().parents[1]
    if subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip() != REVISION:
        raise ValueError("unexpected upstream JEREX code revision")
    subprocess.run(["git", "-C", root, "diff", "--quiet", "HEAD", "--", "jerex"], check=True)
    sys.path.insert(0, str(DOCRED))
    from predict_jerex import predict
    from score_jerex import convert, stage_counts

    revised = load_revised(args.revised_directory, docs) if args.revised_directory else None
    from contextlib import nullcontext
    if args.progress:
        if any((parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
               for parent in args.progress.resolve().parents):
            raise ValueError("per-document progress must be written outside Git")
        progress_file = args.progress.open("x")
    else:
        progress_file = nullcontext()
    with TemporaryDirectory() as tmp, progress_file as progress:
        observer = ControlledReplay([gold_clusters(doc) for doc in docs], progress, revised, args.start)
        writer = Path(tmp) / "writer.json"
        predict(args.model, [doc for doc in docs for _ in ARMS], writer, observer=observer)
        written = json.loads(writer.read_bytes())
    totals = summarize(observer.runs)
    for i, arm in enumerate(ARMS):
        raw = written[i::len(ARMS)]
        g, p = convert(docs, raw)
        stages = stage_counts(g, p, raw)["labeled_relations"]
        result = totals[arm][str(observer.runs[0][3])]
        if (stages["gold"], stages["predicted"], stages["matched"]) != (
            result["gold"], result["matched"] + result["extras"], result["matched"]
        ):
            raise ValueError("independent writer score differs from observed default decisions")
    report = dict(source_sha256=SPLIT, model_sha256=CHECKPOINT["model.ckpt"], revision=REVISION,
                  document_start=args.start, documents=len(docs), reference_threshold=observer.runs[0][3],
                  arms=totals, paired_f1_difference_95pct_points=paired_intervals(observer.runs))
    if revised is not None:
        rerated = rescore(observer.runs, docs, revised)
        report["revised_reference"] = dict(revision=REVISED_REVISION, arms=summarize(rerated),
                                           paired_f1_difference_95pct_points=paired_intervals(rerated))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
