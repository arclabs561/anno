# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay the frozen search methods on independently labeled zbMATH affiliations."""
import argparse
from contextlib import contextmanager
import ast
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from hashlib import sha256
from itertools import groupby, islice
import json
from math import ceil
from pathlib import Path
import re
import platform
import shutil
import subprocess
from time import perf_counter_ns
from statistics import median
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "context"))
import rerank

G2_URL = "https://zenodo.org/api/records/18403221/files/g2.csv/content"
G2_HASH = "cedc2611cdde06998a75f8edbbad271f71236fff520c9db66355e6d266cfe329"


@dataclass(frozen=True)
class Case:
    row: int
    affiliation: str
    gold: frozenset
    available: frozenset
    occurrences: int
    disjoint: bool


def terms(text):
    whole = rerank.key(text)
    # Preserve first occurrence order; repeated or empty parts spend no lookup.
    parts = dict.fromkeys(rerank.key(part) for part in re.split("[,;]", text))
    return [whole, *islice((s for s in parts if s and s != whole), 3)]


def merge(options):
    chosen = []
    for depth in range(5):
        for ranked in options:
            if depth < len(ranked) and ranked[depth] not in chosen:
                chosen.append(ranked[depth])
                if len(chosen) == 5:
                    return chosen
    return chosen


def select_cases(benchmark, reference, catalog):
    prior_strings, prior_ids = set(), set()
    with reference.open(encoding="utf8", newline="") as source:
        for row in csv.DictReader(source):
            prior_strings.add(rerank.key(row["original_affiliation"]))
            prior_ids.update(s.removeprefix("https://ror.org/") for s in ast.literal_eval(row["labels"])
                             if s.startswith("https://ror.org/"))
    groups, statuses, papers = defaultdict(list), Counter(), set()
    with benchmark.open(encoding="utf8", newline="") as source:
        for number, row in enumerate(csv.DictReader(source), 1):
            groups[rerank.key(row["aff_str"])].append((number, row))
            statuses[row["match_status"]] += 1
            papers.add(row["an"])

    cases, exclusions = [], Counter()
    for name, records in groups.items():
        marks = {r["match_status"] for _, r in records}
        labels = {frozenset(v.strip() for v in r["ror_ids"].split(";") if v.strip()) for _, r in records}
        if not name:
            exclusions["empty_string"] += 1
        elif marks != {"TP"}:
            exclusions["non_TP_or_mixed_status"] += 1
        elif len(labels) != 1:
            exclusions["inconsistent_TP_ID_sets"] += 1
        else:
            gold, = labels
            available = gold.intersection(catalog)
            if not gold or not all(re.fullmatch(r"0[0-9a-z]{6}[0-9]{2}", v) for v in gold):
                exclusions["missing_or_malformed_TP_ID"] += 1
            elif not available:
                exclusions["no_target_in_ROR_v1_1"] += 1
            elif name in prior_strings:
                exclusions["normalized_query_overlap_with_S2AFF"] += 1
            else:
                n, first = records[0]
                cases.append(Case(n, first["aff_str"], gold, available, len(records), gold.isdisjoint(prior_ids)))
    return cases, {
        "rows": sum(statuses.values()), "unique_publication_IDs": len(papers),
        "row_status_counts": dict(statuses), "unique_normalized_inputs": len(groups),
        "repeated_row_occurrences": sum(statuses.values()) - len(groups),
        "normalized_inputs_overlapping_S2AFF_all_statuses": len(set(groups) & prior_strings),
        "groups_by_first_exclusion": {**exclusions, "primary": len(cases)},
        "S2AFF_unique_strings_all_splits": len(prior_strings), "S2AFF_distinct_ROR_IDs_all_splits": len(prior_ids),
    }


def first_rank(ranked, gold):
    return next((i for i, accession in enumerate(ranked, 1) if accession in gold), None)


def row_result(case, lists, blocks):
    whole = lists[0]
    segmented = merge(lists)
    gold = case.available
    i, j = first_rank(whole, gold), first_rank(segmented, gold)
    union = set().union(*blocks)
    if j:
        stage = "shown"
    elif any(first_rank(ranked, gold) for ranked in lists):
        stage = "displaced_at_fusion"
    elif union.intersection(gold):
        stage = "below_all_component_fives"
    else:
        stage = "absent"
    return {
        "row": case.row, "occurrences": case.occurrences, "gold_count": len(case.gold),
        "active_gold_count": len(gold), "unseen_target_in_S2AFF": case.disjoint, "lookups": len(lists),
        "whole_rank": i, "segment_rank": j, "whole_block": len(blocks[0]), "union_block": len(union),
        "whole_cards": len(whole), "segment_cards": len(segmented),
        "whole_stage": "shown" if i else "below_five" if gold.intersection(blocks[0]) else "absent",
        "segment_stage": stage,
    }


def panels(rows):
    return {"primary": rows,
            "S2AFF_disjoint_target": [r for r in rows if r["unseen_target_in_S2AFF"]],
            "has_nontrivial_segment": [r for r in rows if r["lookups"] > 1],
            "whole_only": [r for r in rows if r["lookups"] == 1]}


def summary(rows):
    result = {"n": len(rows), "represented_source_rows": sum(r["occurrences"] for r in rows)}
    for k in (1, 3, 5):
        old = {r["row"] for r in rows if (r["whole_rank"] or 6) <= k}
        new = {r["row"] for r in rows if (r["segment_rank"] or 6) <= k}
        result[str(k)] = {"whole": len(old), "segment": len(new), "gain": len(new - old), "loss": len(old - new),
                          "whole_shown": sum(min(k, r["whole_cards"]) for r in rows),
                          "segment_shown": sum(min(k, r["segment_cards"]) for r in rows)}
    for arm in ("whole", "segment"):
        stages = Counter(r[f"{arm}_stage"] for r in rows if r[f"{arm}_stage"] != "shown")
        result[f"{arm}_misses"] = dict(stages)
    for field in ("whole_block", "union_block", "lookups"):
        values = sorted(r[field] for r in rows)
        result[field] = {"median": median(values), "p95": values[ceil(.95 * len(values)) - 1], "sum": sum(values)} if values else {}
    return result


@contextmanager
def serving(catalog):
    # The published batch driver buffers stdout. Flush each response to time a complete lookup.
    source_path = HERE.parent / "ngrams.rs"
    source = source_path.read_text()
    call = '                index.query(fields.next().expect("query"), excluded, None, &mut out)?;'
    if source.count(call) != 1:
        raise ValueError("Unexpected query driver entry point")
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "ngrams.rs").write_text(source.replace(call, call + "\n                out.flush()?;"))
        (directory / "context").mkdir()
        shutil.copyfile(HERE.parent / "context/masked.rs", directory / "context/masked.rs")
        executable = directory / "index"
        subprocess.run(["rustc", "--edition", "2021", "-O", str(directory / "ngrams.rs"), "-o", str(executable)], check=True)
        with subprocess.Popen([executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True, encoding="utf8") as child:
            try:
                for owner, record in catalog.items():
                    fields = ["I", owner, rerank.key(record.display), *sorted({rerank.key(n) for n in record.names})]
                    child.stdin.write("\t".join(fields) + "\n")
                child.stdin.flush()

                def lookup(term):
                    child.stdin.write("Q\t\t" + term + "\n")
                    child.stdin.flush()
                    _, _, count, ranks, owners = child.stdout.readline().rstrip("\n").split("\t")
                    ranked = [part.split(":")[0] for part in ranks.split(",") if part]
                    # Consume the diagnostic block payload on every request in both arms.
                    block = set(owners.split(",")) if owners else set()
                    assert len(block) == int(count)
                    return ranked

                yield lookup
            finally:
                child.stdin.close()
                if child.wait():
                    raise RuntimeError("Native retrieval failed")


def benchmark_queries(catalog, cases, rows, expected):
    times = defaultdict(list)
    with serving(catalog) as lookup:
        for repeat in range(6):
            for case in cases:
                order = (0, 1) if (case.row + repeat) % 2 else (1, 0)
                for arm in order:
                    start = perf_counter_ns()
                    if arm:
                        result = merge([lookup(s) for s in terms(case.affiliation)])
                    else:
                        result = lookup(rerank.key(case.affiliation))
                    elapsed = perf_counter_ns() - start
                    assert result == expected[case.row][arm]
                    if repeat:  # First pass warms both arms and verifies candidate parity.
                        times[case.row, arm].append(elapsed)

    stats = {}
    for group, members in panels(rows).items():
        if not members:
            continue
        stats[group] = {"queries": len(members)}
        for arm, label in enumerate(("whole", "segment")):
            values = sorted(median(times[r["row"], arm]) / 1e6 for r in members)
            stats[group][label] = {"median_ms": round(median(values), 3),
                                   "p95_ms": round(values[ceil(.95 * len(values)) - 1], 3),
                                   "mean_ms": round(sum(values) / len(values), 3)}
    return {"rounds_after_warmup": 5, "machine": platform.machine(), "python": platform.python_version(),
            "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(), "panels": stats}


def run(args):
    benchmark = rerank.get(args.cache, "zbmath-g2.csv", G2_URL, G2_HASH, args.fetch)
    archive = rerank.get(args.cache, rerank.ARCHIVE, rerank.ROR_URL, rerank.ROR_HASH, args.fetch)
    reference = rerank.get(args.cache, "s2aff-gold.csv", rerank.SOURCE, rerank.LABEL_HASH, args.fetch)
    catalog, _ = rerank.read_catalog(archive)
    cases, counts = select_cases(benchmark, reference, catalog)
    requests = [rerank.Query("", str(case.row), term, None) for case in cases for term in terms(case.affiliation)]
    rows, expected_options = [], {}
    with TemporaryDirectory() as tmp:
        native = rerank.lookup(rerank.compiled_index(tmp), catalog, requests)
        groups = groupby(native, key=lambda v: v[0].source)
        for case, (source, group) in zip(cases, groups, strict=True):
            assert source == str(case.row)
            lists, blocks = [], []
            for _, _, _, count, ranked, block in group:
                assert len(block) == count and len(ranked) <= 5
                lists.append([owner for owner, _, _ in ranked])
                blocks.append(set(block))
            assert len(lists) == len(terms(case.affiliation))
            rows.append(row_result(case, lists, blocks))
            expected_options[case.row] = (lists[0], merge(lists))
    result = {"G2_SHA256": G2_HASH, "ROR_SHA256": rerank.ROR_HASH, "S2AFF_SHA256": rerank.LABEL_HASH,
              "native_source_sha256": sha256((HERE.parent / "ngrams.rs").read_bytes()).hexdigest(),
              "catalog_active_owners": len(catalog), "selection": counts,
              "partially_available_primary": sum(r["gold_count"] != r["active_gold_count"] for r in rows),
              "panels": {name: summary(group) for name, group in panels(rows).items()}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (args.out / "rows.csv").open("w", encoding="utf8", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.timing:
        timing = benchmark_queries(catalog, cases, rows, expected_options)
        (args.out / "timing.json").write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true", help="download missing, hash-pinned inputs")
    parser.add_argument("--timing", action="store_true", help="add a warm-process timing run")
    run(parser.parse_args())
