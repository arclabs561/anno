"""Replay the native ranker's unknown-type score on the three official GAP files."""

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import audit_gap_decisions as gap
import gap_receipts
from coref_gap import REVISION

ROOT = Path(__file__).resolve().parents[2]
PATCH = Path(__file__).parent / "fixtures" / "native-gap-neutral-mention.patch"
BASE = "75f961e54631774b4e74547a383eaf8fc79a772d"
PRODUCER = "mention-ranking"


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(rows, policy, binary):
    predictions = gap.run(rows, PRODUCER, policy, binary)
    bits, partitions = [], []
    for row, prediction in zip(rows, predictions, strict=True):
        observation, status = gap.from_clusters(row, prediction, PRODUCER, policy)
        if status not in {"ok", "empty_success"}:
            raise ValueError("A native run did not return a valid partition")
        groups = sorted(
            sorted(tuple(span) for span in group) for group in prediction["clusters"]
        )
        if policy == "supplied":
            returned = {span for group in groups for span in group}
            if returned != set(gap.spans(row)):
                raise ValueError("The supplied spans changed")
        bits.append(observation)
        partitions.append(groups)
    return bits, partitions


def paired(rows, before, after):
    counts = {
        "correct_gained": 0,
        "correct_lost": 0,
        "incorrect_added": 0,
        "incorrect_removed": 0,
        "incorrect_added_to_distractor": 0,
        "incorrect_added_when_neither_gold": 0,
        "incorrect_removed_from_distractor": 0,
        "incorrect_removed_when_neither_gold": 0,
        "changed_documents": 0,
    }
    for row, old, new in zip(rows, before, after, strict=True):
        if old[:3] != new[:3]:
            raise ValueError("Returned endpoint coverage changed")
        gold = gap.gold(row)
        counts["changed_documents"] += old != new
        for positive, prev, curr in zip(gold, old[3:], new[3:], strict=True):
            if prev == curr:
                continue
            if positive:
                counts["correct_gained" if curr == "1" else "correct_lost"] += 1
            elif curr == "0":
                counts["incorrect_removed"] += 1
                subtype = (
                    "incorrect_removed_from_distractor"
                    if any(gold)
                    else "incorrect_removed_when_neither_gold"
                )
                counts[subtype] += 1
            else:
                counts["incorrect_added"] += 1
                subtype = (
                    "incorrect_added_to_distractor"
                    if any(gold)
                    else "incorrect_added_when_neither_gold"
                )
                counts[subtype] += 1
    return counts


def read_sources(directory):
    with (ROOT / "docs/eval/dataset-github-sources.csv").open(
        newline="", encoding="utf-8"
    ) as manifest:
        pins = {
            r["split"]: r
            for r in csv.DictReader(manifest)
            if r["id"] == "GAP" and r["revision"] == REVISION and r["status"] == "ok"
        }
    if set(pins) != {"test", "development", "validation"}:
        raise ValueError("Incomplete GAP source manifest")
    rows = {}
    for split, pin in pins.items():
        path = directory / pin["path"]
        if file_hash(path) != pin["sha256"]:
            raise ValueError(f"{split} source digest differs from the registry receipt")
        with path.open(encoding="utf-8", newline="") as f:
            rows[split] = list(csv.DictReader(f, delimiter="\t"))
        if len(rows[split]) != int(pin["value"]) or len(
            {r["ID"] for r in rows[split]}
        ) != len(rows[split]):
            raise ValueError(f"{split} row count or ID uniqueness differs")
    return rows, {split: pin["sha256"] for split, pin in pins.items()}


def record_panel(rows, arms, index, frozen=None):
    positions = [index[r["ID"]] for r in rows]
    id_hash = gap.sha("".join(r["ID"] + "\n" for r in rows).encode())
    result = {
        "documents": len(rows),
        "distinct_articles": len({r["URL"] for r in rows}),
        "id_list_sha256": id_hash,
        "gold_patterns": dict.fromkeys(("A_only", "B_only", "both", "neither"), 0),
    }
    pattern_name = {
        (True, False): "A_only",
        (False, True): "B_only",
        (True, True): "both",
        (False, False): "neither",
    }
    for row in rows:
        result["gold_patterns"][pattern_name[gap.gold(row)]] += 1
    if frozen is not None and id_hash != frozen["id_list_sha256"]:
        raise ValueError("Frozen panel membership or order changed")
    for policy, (stock, candidate) in arms.items():
        before, original = [[arr[i] for i in positions] for arr in stock]
        after, changed = [[arr[i] for i in positions] for arr in candidate]
        counts = gap.decision_counts(rows, before)
        if frozen is not None:
            previous = frozen["arms"][PRODUCER + "/" + policy]
            if (
                counts != previous["counts"]
                or gap_receipts.digest([before, {}]) != previous["observations_sha256"]
            ):
                raise ValueError(
                    "The control failed to reproduce frozen native decisions"
                )
        result[policy] = {
            "stock": counts,
            "candidate": gap.decision_counts(rows, after),
            "paired": paired(rows, before, after),
            "stock_partitions_sha256": gap_receipts.digest(original),
            "candidate_partitions_sha256": gap_receipts.digest(changed),
            "stock_linked_both": sum(obs[3:] == "11" for obs in before),
            "candidate_linked_both": sum(obs[3:] == "11" for obs in after),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "stock", "candidate", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    stock_hash = file_hash(args.stock)
    candidate_hash = file_hash(args.candidate)
    if stock_hash == candidate_hash:
        parser.error("The two workers are the same build")
    # At this distance a generic antecedent's type term changes the winning score.
    witness = {
        "ID": "authored",
        "Text": "Paz says he saw Zed.",
        "Pronoun": "he",
        "Pronoun-offset": "9",
        "A": "Paz",
        "A-offset": "0",
        "B": "Zed",
        "B-offset": "16",
    }
    for worker, expected in ((args.stock, "11100"), (args.candidate, "11110")):
        flags, _ = evaluate([witness], "supplied", worker)
        if flags != [expected]:
            raise ValueError("A worker failed the authored type-score contrast")
    historic = json.loads((ROOT / "docs/eval/gap-decision-audit.json").read_text())[
        "panels"
    ]
    published = json.loads((ROOT / "docs/eval/coref-panels.json").read_text())["gap"]
    inputs, hashes = read_sources(args.data)
    panels = gap.load_rows(args.data / "gap-test.tsv", published)
    inspected = [r for group in panels.values() for r in group]
    inspected_ids = {r["ID"] for r in inspected}
    inspected_articles = {r["URL"] for r in inspected}
    outside = [r for r in inputs["test"] if r["ID"] not in inspected_ids]
    panels["outside_panels"] = outside
    panels["outside_panels_new_articles"] = [
        r for r in outside if r["URL"] not in inspected_articles
    ]
    articles = {split: {r["URL"] for r in rows} for split, rows in inputs.items()}
    results = {}
    for split, rows in inputs.items():
        arms = {}
        for policy in ("predicted", "supplied"):
            before = evaluate(rows, policy, args.stock)
            after = evaluate(rows, policy, args.candidate)
            if policy == "predicted" and before[1] != after[1]:
                raise ValueError("Raw-text control changed a cluster")
            arms[policy] = before, after
        index = {r["ID"]: i for i, r in enumerate(rows)}
        results[split] = record_panel(rows, arms, index)
        if split == "test":
            results["test_subsets"] = {
                name: record_panel(group, arms, index, historic.get(name))
                for name, group in panels.items()
            }
    receipt = {
        "anno_revision": BASE,
        "source_revision": REVISION,
        "source_sha256": hashes,
        "patch_sha256": file_hash(PATCH),
        "stock_binary_sha256": stock_hash,
        "candidate_binary_sha256": candidate_hash,
        "shared_article_urls_between_splits": {
            f"{a}_{b}": len(articles[a] & articles[b])
            for a, b in combinations(sorted(articles), 2)
        },
        "test_rows_outside_panels_sharing_a_panel_article": sum(
            r["URL"] in inspected_articles for r in outside
        ),
        "panels": results,
    }
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
