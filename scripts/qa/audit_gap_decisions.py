"""Independent GAP decision audit over saved observations and replayed cluster partitions."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIGEST = "1c35e36d5b14f6313ec3f6cd67b275de282595dd59e59390e00cfff9897a6819"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_rows(tsv, published):
    if sha(tsv.read_bytes()) != INPUT_DIGEST:
        raise ValueError("GAP test file digest mismatch")
    with tsv.open(encoding="utf-8", newline="") as f:
        rows = sorted(csv.DictReader(f, delimiter="\t"),
                      key=lambda row: sha(("anno-gap-v1/" + row["ID"]).encode()))
    old = rows[:128]
    if [r["ID"] for r in old] != published["ids"] or len({r["ID"] for r in rows}) != len(rows):
        raise ValueError("GAP IDs disagree with the saved panel or are duplicated")
    seen = {r["URL"] for r in old}
    holdout = []
    for row in rows[128:]:
        if row["URL"] not in seen:
            holdout.append(row)
            seen.add(row["URL"])
        if len(holdout) == 128:
            break
    if len(holdout) != 128:
        raise ValueError("insufficient held-out articles")
    return {"published": old, "article_holdout": holdout}


def spans(row):
    result = []
    for key in ["Pronoun", "A", "B"]:
        start = int(row[key + "-offset"])
        span = (start, start + len(row[key]))
        if row["Text"][slice(*span)] != row[key]:
            raise ValueError(f"{row['ID']}: {key} offset or surface mismatch")
        result.append(span)
    return result


def gold(row):
    labels = [row["A-coref"], row["B-coref"]]
    if any(v not in {"TRUE", "FALSE"} for v in labels):
        raise ValueError(f"{row['ID']}: invalid gold decision")
    return tuple(v == "TRUE" for v in labels)


def decision_counts(rows, observations):
    """Each labeled target pair is one decision. A refused row has no decisions."""
    counts = dict.fromkeys(("attempted", "scored", "refused", "gold_links", "returned_gold_pairs",
                            "all_gold_links", "tp", "fp", "fn", "tn", "both_decisions_correct"), 0)
    for row, obs in zip(rows, observations, strict=True):
        targets = gold(row)
        counts["attempted"] += 1
        counts["all_gold_links"] += sum(targets)
        if obs is None:
            counts["refused"] += 1
            continue
        if len(obs) != 5 or set(obs) - {"0", "1"}:
            raise ValueError(f"{row['ID']}: invalid per-ID observation")
        p, a, b, pa, pb = (v == "1" for v in obs)
        if (pa and not (p and a)) or (pb and not (p and b)):
            raise ValueError(f"{row['ID']}: positive link with an absent endpoint")
        counts["scored"] += 1
        counts["gold_links"] += sum(targets)
        counts["both_decisions_correct"] += (pa, pb) == targets
        for y, endpoint, link in zip(targets, (a, b), (pa, pb)):
            counts["returned_gold_pairs"] += y and p and endpoint
            counts[{(True, True): "tp", (False, True): "fp",
                    (True, False): "fn", (False, False): "tn"}[y, link]] += 1
    if sum(counts[k] for k in ("tp", "fp", "fn", "tn")) != 2 * counts["scored"]:
        raise ValueError("GAP denominator disagreement")
    return counts


def from_clusters(row, result, producer):
    if result["id"] != row["ID"] or result["producer"] != producer:
        raise ValueError("unexpected source document or producer")
    if "unsupported" in result:
        if "clusters" in result or result["unsupported"] != "supplied_span_splits_token":
            raise ValueError("unrecognized refusal or ambiguous refusal with clusters")
        return None, "refused:" + result["unsupported"]
    groups = result["clusters"]
    if not isinstance(groups, list) or any(not g for g in groups):
        raise ValueError("empty cluster or invalid partition")
    flattened = [tuple(span) for group in groups for span in group]
    if len(set(flattened)) != len(flattened) or any(
        not (0 <= start < end <= len(row["Text"])) for start, end in flattened
    ):
        raise ValueError("duplicate or invalid output mention")
    p, a, b = spans(row)
    owners = {tuple(span): j for j, group in enumerate(groups) for span in group}
    flags = (p in owners, a in owners, b in owners,
             p in owners and a in owners and owners[p] == owners[a],
             p in owners and b in owners and owners[p] == owners[b])
    return "".join("1" if flag else "0" for flag in flags), "ok" if groups else "empty_success"


def payload(row, policy):
    targets = spans(row)
    return {"id": row["ID"], "text": row["Text"],
            "mentions": sorted(targets) if policy == "supplied" else None}


def run(rows, backend, policy, binary):
    env = dict(os.environ, ANNO_NO_DOWNLOADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false")
    data = "".join(json.dumps(payload(row, policy)) + "\n" for row in rows)
    child = subprocess.run([str(binary.resolve()), backend], input=data, text=True,
                           env=env, stdout=subprocess.PIPE, check=True)
    return [json.loads(line) for line in child.stdout.splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", type=Path)
    parser.add_argument("binary", type=Path, help="built coref_gap_panel example")
    parser.add_argument("output", type=Path)
    parser.add_argument("--extra-holdout", action="append",
                        choices=["fcoref/predicted", "maverick/predicted", "maverick/supplied"],
                        default=[], help="also check a loaded checkpoint on the new articles")
    parser.add_argument("--candidate", type=Path, help="frozen native candidate for article-disjoint comparison")
    args = parser.parse_args()
    stored = json.loads((ROOT / "docs/eval/coref-panels.json").read_text())["gap"]
    panels = load_rows(args.tsv, stored)
    # Share artifact validation only. The decision and coverage scorer above is independent.
    from coref_gap import model_identity
    models = {arm: model_identity(*arm.split("/")) for arm in sorted(set(args.extra_holdout))}
    with args.binary.open("rb") as executable:
        binary_sha256 = hashlib.file_digest(executable, "sha256").hexdigest()
    report = {"input_sha256": INPUT_DIGEST, "binary_sha256": binary_sha256,
              "holdout_checkpoints": models, "panels": {}}
    for name, rows in panels.items():
        observations = {}
        ids = [r["ID"] for r in rows]
        labels = ["".join("1" if v else "0" for v in gold(r)) for r in rows]
        panel = {"ids": ids, "article_count": len({r["URL"] for r in rows}), "gold_pairs": labels,
                 "id_list_sha256": sha(("".join(i + "\n" for i in ids)).encode()),
                 "arms": observations}
        report["panels"][name] = panel
        if name == "published":
            if labels != stored["gold_pairs"]:
                raise ValueError("reference labels differ from original TSV")
            for arm, record in stored["arms"].items():
                counts = decision_counts(rows, record["per_id"])
                for k in ("tp", "fp", "fn", "tn", "gold_links"):
                    if counts[k] != record["counts"][k]:
                        raise ValueError(f"{name} {arm}: saved {k} differs")
                if counts["returned_gold_pairs"] != record["counts"]["gold_both_covered"]:
                    raise ValueError(f"{arm}: saved gold-pair coverage differs")
                observations[arm] = {"evidence": "saved_target_decisions", "counts": counts}
        arms = ["mention-ranking/predicted", "mention-ranking/supplied"]
        if name == "article_holdout":
            arms += args.extra_holdout
        for arm in arms:
            backend, policy = arm.split("/")
            outputs = run(rows, backend, policy, args.binary)
            decoded = [from_clusters(row, out, backend) for row, out in zip(rows, outputs, strict=True)]
            bits = [v for v, _ in decoded]
            if name == "published" and bits != stored["arms"][arm]["per_id"]:
                raise ValueError(f"{arm}: native cluster replay does not reproduce saved per-ID decisions")
            outcomes = [s for _, s in decoded]
            observations[arm] = {"evidence": "replayed_clusters", "counts": decision_counts(rows, bits),
                                 "empty_successes": outcomes.count("empty_success"),
                                 "per_id": bits, "refusals": {i: s for i, s in zip(ids, outcomes)
                                                              if s.startswith("refused:")}}
    if args.candidate:
        with args.candidate.open("rb") as f:
            candidate_sha = hashlib.file_digest(f, "sha256").hexdigest()
        comparison = {"binary_sha256": candidate_sha, "arms": {}}
        rows = panels["article_holdout"]
        for mode in ("predicted", "supplied"):
            before = report["panels"]["article_holdout"]["arms"]["mention-ranking/" + mode]["per_id"]
            raw = run(rows, "mention-ranking", mode, args.candidate)
            after = [from_clusters(r, o, "mention-ranking")[0] for r, o in zip(rows, raw, strict=True)]
            changes = []
            transitions = {"both_correct": 0, "won": 0, "lost": 0, "both_wrong": 0}
            for r, old, new in zip(rows, before, after, strict=True):
                labels = "".join("1" if v else "0" for v in gold(r))
                flags = old[3:] == labels, new[3:] == labels
                transitions[{(True, True): "both_correct", (False, True): "won",
                             (True, False): "lost", (False, False): "both_wrong"}[flags]] += 1
                if old != new:
                    changes.append({"id": r["ID"], "before": old, "after": new})
            comparison["arms"][mode] = {
                "counts": decision_counts(rows, after), "changed_observations": changes,
                "paired_exact_decisions": transitions,
                "coverage_pronoun_A_B": [sum(bits[i] == "1" for bits in after) for i in range(3)],
            }
        report["article_holdout_candidate"] = comparison
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
