# SPDX-License-Identifier: MIT OR Apache-2.0
"""Score frozen commitment rules on verified, absent and authored affiliation references."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from commitments.check import check as check_fixture, policies
from ngram_retrieval import Query, compiled_index, lookup
from segments import replicate as study


def source_negatives(benchmark, reference):
    with reference.open(newline="", encoding="utf8") as stream:
        prior = {study.rerank.key(row["original_affiliation"]) for row in csv.DictReader(stream)}
    groups = defaultdict(list)
    with benchmark.open(newline="", encoding="utf8") as stream:
        for number, row in enumerate(csv.DictReader(stream), 1):
            groups[study.rerank.key(row["aff_str"])].append((number, row))
    cases = []
    for name, entries in groups.items():
        if name and name not in prior and {row["match_status"] for _, row in entries} == {"TN"}:
            number, row = entries[0]
            cases.append((number, row["aff_str"], frozenset(), len(entries)))
    return cases


def run(args):
    lib = study.rerank
    benchmark = lib.get(args.cache, "zbmath-g2.csv", study.G2_URL, study.G2_HASH, args.fetch)
    reference = lib.get(args.cache, "s2aff-gold.csv", lib.SOURCE, lib.LABEL_HASH, args.fetch)
    archive = lib.get(args.cache, lib.ARCHIVE, lib.ROR_URL, lib.ROR_HASH, args.fetch)
    catalog, _ = lib.read_catalog(archive)
    cases, _ = study.select_cases(benchmark, reference, catalog)
    positives = [(c.row, c.affiliation, c.available, c.occurrences) for c in cases]
    panels = (
        ("verified_available", positives, False),
        ("same_rows_all_labels_withheld", positives, True),
        ("source_TN", source_negatives(benchmark, reference), False),
    )
    counts, negative_links = {}, []
    args.out.mkdir(parents=True, exist_ok=True)
    fields = ("panel", "row", "source_row_occurrences", "policy", "outcome", "reason", "suggestions")
    with TemporaryDirectory() as tmp, (args.out / "rows.csv").open("w", newline="", encoding="utf8") as stream:
        binary = compiled_index(tmp)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, rows, withheld in panels:
            requests = [Query(label, str(number), name, None) for number, name, _, _ in rows]
            masks = {str(number): gold for number, _, gold, _ in rows} if withheld else None
            results = lookup(binary, catalog, requests, catalog_masks=masks)
            totals = defaultdict(Counter)
            for (number, _, gold, repeats), output in zip(rows, results, strict=True):
                q, _, exact, _, ranked, block = output
                assert int(q.source) == number
                if withheld:
                    assert gold.isdisjoint(block)
                for arm, (owner, reason) in policies(ranked).items():
                    if owner is None:
                        outcome = "abstained"
                    elif owner in gold:
                        outcome = "correct"
                    else:
                        outcome = "wrong"
                    totals[arm][outcome] += 1
                    writer.writerow(dict(panel=label, row=number, source_row_occurrences=repeats,
                                         policy=arm, outcome=outcome, reason=reason, suggestions=len(ranked)))
                    if label == "source_TN" and arm == "gate" and owner is not None:
                        negative_links.append({
                            "source_row": number, "chosen": owner, "exact_alias": owner in exact,
                            "winner": ranked[0], "runner_up": ranked[1] if len(ranked) > 1 else None,
                            "candidate_owners": len(block),
                        })
            counts[label] = {
                "n": len(rows), "source_rows": sum(repeats for _, _, _, repeats in rows),
                "policies": dict(totals),
            }
        authored, receipts = check_fixture(binary)
    result = {"public": counts, "authored": authored, "source_TN_gate_commits": negative_links}
    for name, values in (("results", result), ("fixture-receipts", receipts)):
        (args.out / f"{name}.json").write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true", help="download missing hash-verified inputs")
    run(parser.parse_args())
