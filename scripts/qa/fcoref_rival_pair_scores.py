# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare native exact-name, enclosing-mention and null antecedent scores."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
from struct import pack, unpack
import subprocess

from backend_residuals.private_ledger import external_path
from coref_gap import FC_REV, decode_clusters, index_partition
from fcoref_boundary_ranks import WHITE_SPACE
from fcoref_emitted_boundaries import MODEL_FILES, check_file, panel
from fcoref_endpoint_stages import marked
from gap_receipts import digest
from run_fcoref_endpoint_stages import observations


def f32(value):
    return unpack("f", pack("f", value))[0]


def project(text, offsets, i, j):
    if not 0 <= i <= j < len(offsets):
        raise ValueError("Invalid token pair")
    a, b = offsets[i][0], offsets[j][1]
    raw = text[a:b]
    left = raw.lstrip(WHITE_SPACE)
    value = left.rstrip(WHITE_SPACE)
    if value:
        a += len(raw) - len(left)
        return a, a + len(value)
    return None


def retained_indices(table, text, offsets):
    """Verify native f32 sums and tie order; keep the best token realization per source span."""
    indices = {}
    previous = None
    for i, (start, end, mention, chosen, pairs) in enumerate(table):
        if previous is not None and (start, end) <= previous:
            raise ValueError("Retained token pairs are not strictly source ordered")
        previous = (start, end)
        if len(pairs) != i or not math.isfinite(mention):
            raise ValueError("Incomplete native score row")
        target = i
        best = -math.inf
        for j, (pair, total) in enumerate(pairs):
            if not math.isfinite(pair) or not math.isfinite(total):
                raise ValueError("Non-finite native pair score")
            expected = f32(f32(f32(mention) + f32(table[j][2])) + f32(pair))
            if expected != f32(total):
                raise ValueError("Pair total disagrees with native component sum")
            if total > best:
                target, best = j, total
        if chosen != (target if best >= 0 else i):
            raise ValueError("Returned antecedent differs from native argmax")
        span = project(text, offsets, start, end)
        if span is not None and (span not in indices or mention > table[indices[span]][2]):
            indices[span] = i
    return indices


def edge(table, indices, pronoun, name):
    p, n = indices[pronoun], indices[name]
    query, antecedent = max(p, n), min(p, n)
    pair, total = table[query][4][antecedent]
    winner = table[query][3]
    best_real = max(score for _, score in table[query][4])
    return dict(pronoun_mention=table[p][2], name_mention=table[n][2], pair=pair, total=total,
                query="pronoun" if query == p else "name", best_real=best_real,
                winning_score=max(0., best_real),
                winner="direct" if winner == antecedent else "null" if winner == query else "other")


def run_arm(cases, binary, model, reserve):
    inputs = []
    for row, _, _ in cases:
        text = row["Text"]
        marks = [tuple(len(text[:i].encode()) for i in marked(row, field))
                 for field in ("Pronoun", "A", "B")] if reserve else []
        inputs.append(json.dumps(dict(id=row["ID"], text=text, marked_bytes=marks)))
    env = dict(os.environ, FCOREF_MODEL_PATH=str(model.resolve()), ANNO_NO_DOWNLOADS="1",
               HF_HUB_OFFLINE="1", ANNO_FCOREF_PAIR_SCORES="1")
    proc = subprocess.run([binary.resolve(), "fcoref-marked-study"],
                          input="\n".join(inputs) + "\n", text=True, capture_output=True,
                          env=env, check=True)
    return parse_arm(proc.stdout.splitlines(), proc.stderr.splitlines())


def parse_arm(outputs, diagnostics):
    tables = (json.loads(line.removeprefix("FCOREF_PAIR_SCORES ")) for line in diagnostics
              if line.startswith("FCOREF_PAIR_SCORES "))
    return list(zip(map(json.loads, outputs), tables, strict=True))


def per_case(cases, stock_ranks, tokenizer, stock_arm, reserved_arm):
    results = []
    for (row, field, rival), stock, reserve in zip(cases, stock_arm, reserved_arm, strict=True):
        doc_id, text = row["ID"], row["Text"]
        offsets = tokenizer.encode(text).offsets
        name, pronoun = marked(row, field), marked(row, "Pronoun")
        frozen = stock_ranks[doc_id]
        outputs = []
        for pos, (prediction, table) in enumerate((stock, reserve)):
            groups = decode_clusters(prediction, doc_id, text, "fcoref")
            groups_by_span = index_partition(groups)
            ranks = frozen["trace"][1]
            k = frozen["trace"][0]
            if len(table) != k:
                raise ValueError("Retained slot count changed")
            # Verify that trace and stdout refer to the same document and arm.
            # The reserved arm promotes only the best realization of the three marks.
            forced = set()
            if pos == 1:
                marks = {marked(row, f) for f in ("Pronoun", "A", "B")}
                for a, b, _ in ranks:
                    span = project(text, offsets, a, b)
                    if span in marks:
                        marks.remove(span)
                        forced.add((a, b))
                        if not marks:
                            break
                if marks:
                    raise ValueError("Expected three legal marked proposals")
            expected = sorted(ranks, key=lambda candidate: (candidate[0], candidate[1]) not in forced)[:k]
            if [(a, b, f32(s)) for a, b, s in sorted(expected)] != [(a, b, f32(s)) for a, b, s, _, _ in table]:
                raise ValueError("Retained native candidates differ from independent rank trace")
            indices = retained_indices(table, text, offsets)
            if pos == 0:
                prior = decode_clusters(frozen["stock"], doc_id, text, "fcoref")
                if sorted(map(sorted, groups)) != sorted(map(sorted, prior)) or name in indices:
                    raise ValueError("Stock run did not reproduce frozen case")
            if groups_by_span.get(pronoun) is None or groups_by_span.get(rival) != groups_by_span[pronoun]:
                raise ValueError("Enclosing rival no longer shares the pronoun's cluster")
            item = {"rival": edge(table, indices, pronoun, rival)}
            if pos == 1:
                item["exact"] = edge(table, indices, pronoun, name)
                if groups_by_span.get(name) == groups_by_span[pronoun]:
                    raise ValueError("Marked admission changed this gold link")
            outputs.append(item)
        results.append(dict(id=doc_id, field=field, stock_rival=outputs[0]["rival"], **outputs[1]))
    return results


def describe(values):
    return dict(min=min(values), median=median(values), max=max(values))


def aggregate(results):
    summary = dict(cases=len(results))
    for arm in ("stock_rival", "rival", "exact"):
        rows = [item[arm] for item in results]
        summary[arm] = dict(
            **{key: describe([row[key] for row in rows])
               for key in ("pronoun_mention", "name_mention", "pair", "total", "winning_score")},
            gap_to_winner=describe([row["winning_score"] - row["total"] for row in rows]),
            below_null=sum(row["total"] < 0 for row in rows),
            query_counts=dict(Counter(row["query"] for row in rows)),
            winner_counts=dict(Counter(row["winner"] for row in rows)))
    summary["rival_minus_exact"] = {
        key: describe([r["rival"][key] - r["exact"][key] for r in results])
        for key in ("name_mention", "pair", "total")}
    summary["rival_higher"] = {
        key: sum(r["rival"][key] > r["exact"][key] for r in results)
        for key in ("name_mention", "pair", "total")}
    summary["rival_total_changed_from_stock"] = sum(
        f32(r["stock_rival"]["total"]) != f32(r["rival"]["total"]) for r in results)
    return summary


def main():
    from tokenizers import Tokenizer
    from fcoref_identity_review import select_cases

    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("development", "test", "validation", "model", "binary", "rank_binary"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--private-rows", type=Path, help="New file outside Git for blinded-label joins")
    args = parser.parse_args()
    for file, expected in MODEL_FILES.items():
        check_file(args.model / file, expected)
    rows = panel(args.development, args.test, args.validation)
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    ranks = list(observations(rows, args.rank_binary, args.model))
    cases = select_cases(rows, ranks, tokenizer)
    results = per_case(cases, {r["id"]: r for r in ranks}, tokenizer,
                       run_arm(cases, args.binary, args.model, False),
                       run_arm(cases, args.binary, args.model, True))
    if args.private_rows:
        path = external_path(args.private_rows)
        with os.fdopen(os.open(path, os.O_EXCL | os.O_CREAT | os.O_WRONLY, 0o600), "w") as stream:
            print(json.dumps(results, allow_nan=False), file=stream)
    summary = aggregate(results)
    with args.binary.open("rb") as stream:
        summary.update(binary_sha256=hashlib.file_digest(stream, "sha256").hexdigest(),
                       model_revision=FC_REV, model_files=MODEL_FILES, private_rows_digest=digest(results))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
