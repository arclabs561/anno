# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay original paired signer questions against the public coreference example."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

import audit_gap_decisions as gap

ROOT = Path(__file__).resolve().parents[2]
ARMS = {"native-entities": ("mention-ranking", "supplied"),
        "native-raw": ("mention-ranking", "predicted"),
        "native-copula-raw": ("mention-ranking-copula", "predicted"),
        "fcoref-raw": ("fcoref", "predicted"),
        "maverick-raw": ("maverick", "predicted"),
        "maverick-supplied": ("maverick", "supplied")}


def panel():
    rows = []

    def add(family, pair, text, query, expected):
        a, b = names
        row = {"ID": f"{pair}-{expected or 'unresolved'}", "pair": pair, "family": family,
               "Text": text, "Pronoun": query, "Pronoun-offset": text.rindex(query),
               "A": a, "A-offset": text.index(a), "B": b, "B-offset": text.index(b),
               "gold": expected}
        if len(set(gap.spans(row))) != 3:
            raise ValueError("the three query endpoints must be distinct occurrences")
        rows.append(row)

    namesets = [("Ada", "Béa"), ("Mia", "Zoe")]
    for group, base in enumerate(namesets):
        for direction, names in enumerate((base, base[::-1])):
            a, b = names
            layout = f"{group}-{direction}"
            for order, roles in enumerate((("pilot", "medic"), ("medic", "pilot"))):
                context = f"{a} is the {roles[0]}. {b} is the {roles[1]}. "
                for expected, cue in zip(("A", "B"), roles, strict=True):
                    add("appositive", f"appositive-{layout}-{order}",
                        context + f"Today she, the {cue}, signed.", "she", expected)
                    add("exclusive", f"exclusive-{layout}-{order}",
                        context + f"Only the {cue} signed the register. The log confirms that she signed it.",
                        "she", expected)
            for expected, cue in (("A", "former"), ("B", "latter")):
                add("order_word", f"order-word-{layout}", f"{a} and {b} arrived. Only the {cue} signed "
                    "the register. The log confirms that she signed it.", "she", expected)
            for expected, name in zip(("A", "B"), names, strict=True):
                add("repeated_name", f"repeated-name-{layout}",
                    f"{a} met {b}. In the evening, {name} signed.", name, expected)
            add("ambiguous", f"ambiguous-{layout}",
                f"{a} and {b} both signed the register. Later she went home.", "she", None)
    return rows


def observe(row, result, producer, mode):
    bits, status = gap.from_clusters(row, result, producer, mode)
    if bits is None:
        return {"status": status, "choice": "unsupported"}
    choice = {"10": "A", "01": "B", "11": "both", "00": "none"}[bits[3:]]
    query, *names = gap.spans(row)
    # A longer output may enclose the query. Keep this diagnostic separate from exact evidence.
    enclosing = [group for group in result["clusters"]
                 if any(tuple(span) != query and span[0] <= query[0] and query[1] <= span[1]
                        for span in group)]
    enclosing_names = [k for k, span in zip("AB", names, strict=True)
                       if any(list(span) in group for group in enclosing)]
    if choice in ("A", "B"):
        outcome = "unresolved_gold_commitment" if row["gold"] is None else (
            "correct" if choice == row["gold"] else "wrong")
    elif choice == "both":
        outcome = "both_linked"
    elif bits[0] == "0":
        outcome = "query_missing"
    elif bits[1:3] != "11":
        outcome = "candidate_missing"
    else:
        outcome = "abstained"
    return {"status": outcome, "choice": choice, "returned_query_A_B": bits[:3],
            "names_linked_to_strictly_enclosing_query": "".join(enclosing_names)}


def summarize(rows, observations):
    outcomes = defaultdict(Counter)
    paired = defaultdict(list)
    for row, result in zip(rows, observations, strict=True):
        outcomes[row["family"]][result["status"]] += 1
        if row["gold"] is not None:
            paired[(row["family"], row["pair"])].append((row["gold"], result["choice"]))
    pairs = defaultdict(Counter)
    for (family, _), values in paired.items():
        by_gold = dict(values)
        if len(values) != 2 or set(by_gold) != {"A", "B"}:
            raise ValueError("expected opposite gold answers in each pair")
        a, b = by_gold["A"], by_gold["B"]
        if (a, b) == ("A", "B"):
            decision = "both_correct"
        elif (a, b) == ("B", "A"):
            decision = "both_wrong"
        elif a == b and a in ("A", "B"):
            decision = "same_unique_choice"
        else:
            decision = "incomplete_unique_choices"
        pairs[family][decision] += 1
    return {"by_family": dict(outcomes), "paired_unique_decisions": dict(pairs)}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def replay(rows, binary, arm):
    backend, mode = ARMS[arm]
    if backend == "fcoref":
        expected = json.loads((ROOT / "docs/eval/fcoref-emitted-boundaries.json").read_text())
        for name, sha in expected["model_files"].items():
            with (Path(os.environ["FCOREF_MODEL_PATH"]) / name).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != sha:
                    raise ValueError(f"FCoref export differs for {name}")
    calls = "".join(json.dumps(gap.payload(r, mode)) + "\n" for r in rows)
    proc = subprocess.run([str(binary.resolve()), backend], input=calls, text=True, capture_output=True,
                          env={**os.environ, "ANNO_NO_DOWNLOADS": "1", "HF_HUB_OFFLINE": "1",
                               "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    if proc.returncode:
        raise ValueError(f"{backend} worker failed: {proc.stderr}")
    records = [observe(r, json.loads(line), backend, mode)
               for r, line in zip(rows, proc.stdout.splitlines(), strict=True)]
    details = [{**{"id": row["ID"]}, **observation} for row, observation in zip(rows, records, strict=True)]
    return {"backend": backend, "mode": mode, **summarize(rows, records),
            "observations_sha256": digest(details)}, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path, help="aggregate replay receipt")
    parser.add_argument("--arm", choices=ARMS, required=True, action="append")
    parser.add_argument("--details", type=Path, help="optional per-question evidence")
    args = parser.parse_args()
    if len(set(args.arm)) != len(args.arm):
        parser.error("each arm can be requested once")
    rows = panel()
    result, details = {}, {}
    for arm in args.arm:
        result[arm], details[arm] = replay(rows, args.binary, arm)
    receipt = {"panel_sha256": digest(rows), "documents": len(rows), "arms": result}
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if args.details:
        args.details.write_text(json.dumps(details, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
