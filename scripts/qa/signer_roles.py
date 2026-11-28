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
from coref_gap import check_file, FC_FILES

ARMS = {"native-entities": ("mention-ranking", "supplied"),
        "native-raw": ("mention-ranking", "predicted"),
        "native-copula-raw": ("mention-ranking-copula", "predicted"),
        "fcoref-raw": ("fcoref", "predicted"),
        "fcoref-supplied": ("fcoref", "supplied"),
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


def reflexive_panel(infinitive=False):
    """Contrast the subject of a matrix clause with the subject of a reported clause."""
    rows = []
    layouts = [(("Iris", "Léna"), "herself"), (("Omar", "Evan"), "himself")] if infinitive else [
        (("Ada", "Béa"), "herself"), (("Leo", "Max"), "himself")]
    actions = ("washed", "dressed", "defended") if infinitive else ("praised", "blamed", "introduced")
    family = "reflexive_infinitive" if infinitive else "reflexive_clause"
    for names, pronoun in layouts:
        for a, b in (names, names[::-1]):
            for action in actions:
                pair = f"clause-{a}-{b}-{action}"
                passages = (f"Yesterday {a}, who saw {b}, had {action} {pronoun}.",
                            f"Yesterday {a} said how {b} once {action} {pronoun}.")
                if infinitive:
                    passages = (f"Today {a}, beside {b}, {action} {pronoun}.",
                                f"Today {a} advised {b} to {action.removesuffix('ed')} {pronoun}.")
                for gold_name, text in zip("AB", passages, strict=True):
                    rows.append({"ID": f"{pair}-{gold_name}", "pair": pair, "family": family,
                                 "Text": text, "Pronoun": pronoun, "Pronoun-offset": text.index(pronoun),
                                 "A": a, "A-offset": text.index(a), "B": b, "B-offset": text.index(b),
                                 "gold": gold_name})
                if gap.spans(rows[-2]) != gap.spans(rows[-1]):
                    raise ValueError(f"marked endpoints moved in {pair}")
    return rows


def observe(row, result, producer, mode):
    bits, status = gap.from_clusters(row, result, producer, mode)
    if bits is None:
        return {"status": status, "choice": "unsupported"}
    choice = {"10": "A", "01": "B", "11": "both", "00": "none"}[bits[3:]]
    query, *names = gap.spans(row)
    if mode == "supplied":
        supplied = {query, *names}
        if any(tuple(span) not in supplied for group in result["clusters"] for span in group):
            raise ValueError("supplied-mention worker returned an unrequested span")
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
    exact_group = next((group for group in result["clusters"] if list(query) in group), [])
    diagnostic = None
    if outcome == "candidate_missing" and row["family"] in ("exclusive", "order_word"):
        start = row["Text"].index("Only the ")
        end = row["Text"].index(" signed the register", start)
        if [start, end] in exact_group:
            diagnostic = "exact_query_linked_to_cue_phrase_without_named_answer"
    elif outcome == "candidate_missing" and row["family"] == "repeated_name":
        start, end = names["AB".index(row["gold"])]
        if row["Text"][end:end + 1] == "." and [start, end + 1] in exact_group:
            diagnostic = "exact_query_linked_to_expected_name_with_period"
    elif outcome == "query_missing" and len(enclosing_names) == 1 and row["gold"]:
        diagnostic = ("enclosing_query_names_correct_candidate" if enclosing_names[0] == row["gold"]
                      else "enclosing_query_names_wrong_candidate")
    if row["family"] in ("reflexive_clause", "reflexive_infinitive"):
        a, b = names
        # The two authored parentheticals modify A. This exact source phrase
        # contains both marked names, so containment alone cannot identify its head.
        if (row["Text"][a[1]:b[0]] in (", who saw ", ", beside ")
                and [a[0], b[1]] in exact_group):
            diagnostic = "exact_query_linked_to_outer_phrase_headed_by_A"
    return {"status": outcome, "choice": choice, "returned_query_A_B": bits[:3],
            "names_linked_to_strictly_enclosing_query": "".join(enclosing_names),
            **({"diagnostic": diagnostic} if diagnostic else {})}


def summarize(rows, observations):
    outcomes = defaultdict(Counter)
    coverage = defaultdict(Counter)
    diagnostics = defaultdict(Counter)
    paired = defaultdict(list)
    for row, result in zip(rows, observations, strict=True):
        outcomes[row["family"]][result["status"]] += 1
        if "diagnostic" in result:
            diagnostics[row["family"]][result["diagnostic"]] += 1
        if "returned_query_A_B" in result:
            coverage[row["family"]][result["returned_query_A_B"]] += 1
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
    return {"by_family": dict(outcomes), "source_span_diagnostics": dict(diagnostics),
            "returned_query_A_B_by_family": dict(coverage),
            "paired_unique_decisions": dict(pairs)}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def replay(rows, binary, arm):
    backend, mode = ARMS[arm]
    if backend == "fcoref":
        for name, sha in FC_FILES.items():
            check_file(Path(os.environ["FCOREF_MODEL_PATH"]) / name, sha)
    if backend == "maverick":
        manifest_path = gap.ROOT / "crates/anno/src/backends/coref/maverick_models.json"
        checkpoint = json.loads(manifest_path.read_text())["MODELS"]["ontonotes"]
        check_file(Path(os.environ["MAVERICK_MODEL_PATH"]), checkpoint["prepared_sha256"])
    calls = "".join(json.dumps(gap.payload(r, mode)) + "\n" for r in rows)
    proc = subprocess.run([str(binary.resolve()), backend], input=calls, text=True, capture_output=True,
                          env={**os.environ, "ANNO_NO_DOWNLOADS": "1", "HF_HUB_OFFLINE": "1",
                               "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    unsupported = "Feature not available: fcoref cannot cluster supplied mentions"
    if arm == "fcoref-supplied" and proc.returncode and not proc.stdout and (
            f"fcoref on {rows[0]['ID']}: {unsupported}" in proc.stderr):
        return {"backend": backend, "mode": mode, "status": "unsupported_operation",
                "rejected_document": rows[0]["ID"], "reason": unsupported}, []
    if proc.returncode:
        raise ValueError(f"{backend} worker failed: {proc.stderr}")
    outputs = [json.loads(line) for line in proc.stdout.splitlines()]
    records = [observe(row, output, backend, mode)
               for row, output in zip(rows, outputs, strict=True)]
    details = [{"id": row["ID"], **observation, **(
        {"clusters": sorted(sorted(group) for group in output["clusters"])} if "clusters" in output else {})}
        for row, observation, output in zip(rows, records, outputs, strict=True)]
    return {"backend": backend, "mode": mode, **summarize(rows, records),
            "observations_sha256": digest(details)}, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path, help="aggregate replay receipt")
    parser.add_argument("--arm", choices=ARMS, required=True, action="append")
    parser.add_argument("--details", type=Path, help="optional per-question evidence")
    parser.add_argument("--panel", choices=("signer", "reflexive", "reflexive-infinitive"), default="signer")
    parser.add_argument("--check", type=Path, help="compare the selected arms with a saved receipt")
    args = parser.parse_args()
    if args.check and args.check.resolve() in {args.output.resolve(), args.details.resolve() if args.details else None}:
        parser.error("use a separate output path for the checked receipt")
    if len(set(args.arm)) != len(args.arm):
        parser.error("each arm can be requested once")
    rows = panel() if args.panel == "signer" else reflexive_panel(args.panel == "reflexive-infinitive")
    result, details = {}, {}
    for arm in args.arm:
        result[arm], details[arm] = replay(rows, args.binary, arm)
    receipt = {"panel_sha256": digest(rows), "documents": len(rows), "arms": result}
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if args.check:
        expected = json.loads(args.check.read_bytes())
        if (expected["panel_sha256"] != receipt["panel_sha256"] or expected["documents"] != len(rows)
                or any(result[arm] != expected["arms"].get(arm) for arm in args.arm)):
            raise ValueError("panel or backend decisions differ from the stored receipt")
    if args.details:
        args.details.write_text(json.dumps(details, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
