"""Replay native mention-ranking stages on the frozen GAP panel without supplying gold to the model."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import tempfile

from coref_gap import measure, panel


def endpoint(span, proposed, retained, seen, decisions):
    if span not in proposed:
        overlaps = [(a, b) for a, b in proposed if a < span[1] and span[0] < b]
        if not overlaps:
            return "not_proposed_no_overlap"
        if all(span[0] <= a and b <= span[1] for a, b in overlaps):
            return "not_proposed_only_subspans"
        return "not_proposed_other_overlap"
    if span not in retained:
        return "overlap_dedup"
    if span in seen:
        return "returned"
    i = retained.index(span)
    d = decisions[i]
    if d["selected"] is not None or any(x["selected"] == i for x in decisions):
        raise ValueError("selected endpoint missing from clusters")
    if not d["candidates"]:
        return "unlinked_no_earlier_candidate"
    if all(score <= threshold for j, score, threshold in d["candidates"]):
        return "unlinked_no_earlier_candidate_above_threshold"
    raise ValueError("eligible antecedent not selected by the default ranker")


def audit(docs, traces, published):
    predictions = [
        dict(id=r["id"], producer="mention-ranking", clusters=r["clusters"])
        for r in traces
    ]
    verified = measure(docs, predictions, "mention-ranking")
    if [d[0] for d in docs] != published["ids"] or verified != published["arms"][
        "mention-ranking/predicted"
    ]:
        raise ValueError("native outputs differ from published decisions or counts")
    counts, combinations, relations = Counter(), Counter(), Counter()
    rows = []
    names = Counter()
    for (doc_id, text, targets, labels), trace in zip(docs, traces, strict=True):
        proposed = set(map(tuple, trace["proposed"]))
        retained = list(map(tuple, trace["retained"]))
        groups = [set(map(tuple, c)) for c in trace["clusters"]]
        seen = set().union(*groups)
        decisions = trace["decisions"]
        for side, g in enumerate(labels, 1):
            if not g:
                continue
            p, n = targets[0], targets[side]
            statuses = [
                endpoint(s, proposed, retained, seen, decisions) for s in (p, n)
            ]
            if any(s not in proposed for s in (p, n)):
                stage = "proposal"
            elif any(s not in retained for s in (p, n)):
                stage = "overlap_dedup"
            elif any(s not in seen for s in (p, n)):
                stage = "cluster_emission"
            elif not any(p in c and n in c for c in groups):
                stage = "returned_separately"
            else:
                stage = "linked"
            row = {
                "id": doc_id,
                "candidate": "AB"[side - 1],
                "first_loss_or_link": stage,
                "pronoun": statuses[0],
                "name": statuses[1],
            }
            if n not in proposed:
                a, b = n
                pieces = {s for s in proposed if s[0] < b and a < s[1]}
                if (
                    len(text[a:b].split()) > 1
                    and statuses[1] == "not_proposed_only_subspans"
                ):
                    shape = "multiword_only_subspans"
                elif (a, b + 2) in pieces and text[b : b + 2] in ("'s", "’s"):
                    shape = "including_possessive_clitic"
                elif not pieces and (
                    not text[:a].strip() or text[:a].rstrip()[-1] in ".!?"
                ):
                    shape = "unproposed_sentence_initial"
                elif not pieces and text[max(0, a - 1) : a] == "(":
                    shape = "unproposed_after_open_paren"
                else:
                    shape = "other_unresolved"
                row["name_boundary_observation"] = shape
                names[shape] += 1
            if p in retained and n in retained:
                earlier, later = sorted((retained.index(p), retained.index(n)))
                decision = decisions[later]
                pair = next(
                    (c for c in decision["candidates"] if c[0] == earlier), None
                )
                if pair is None:
                    beyond_count = later - earlier > decision["max_antecedents"]
                    beyond_distance = (
                        retained[later][0] - retained[earlier][1]
                        > trace["max_distance"]
                    )
                    rank = (
                        "beyond_distance_and_count"
                        if beyond_distance and beyond_count
                        else "beyond_distance"
                        if beyond_distance
                        else "beyond_count"
                        if beyond_count
                        else "outside_eligible_unknown"
                    )

                elif decision["selected"] == earlier:
                    rank = "selected"
                elif pair[1] <= pair[2]:
                    rank = "at_or_below_threshold"
                else:
                    rank = "another_antecedent_selected"
                row["later_endpoint"] = "pronoun" if retained[later] == p else "name"
                row["direct_pair"] = rank
                relations[rank] += 1
            counts[stage] += 1
            combinations[" / ".join(statuses)] += 1
            rows.append(row)
    return dict(
        data_sha256=published["sha256"],
        revision=published["revision"],
        split=published["split"],
        documents=len(docs),
        gold_pairs=len(rows),
        name_boundary_observations=dict(sorted(names.items())),
        stages=dict(sorted(counts.items())),
        endpoint_status_pairs=dict(sorted(combinations.items())),
        direct_pair_when_retained=dict(sorted(relations.items())),
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    docs = panel(args.tsv)
    with tempfile.TemporaryDirectory() as tmp:
        source, trace = Path(tmp) / "input.jsonl", Path(tmp) / "trace.jsonl"
        source.write_text(
            "".join(json.dumps(dict(id=i, text=t)) + "\n" for i, t, *_ in docs)
        )
        subprocess.run(
            [
                "cargo",
                "test",
                "--locked",
                "-p",
                "anno",
                "--no-default-features",
                "--lib",
                "gap_trace::trace_native_gap",
                "--",
                "--ignored",
            ],
            cwd=root,
            env={
                **os.environ,
                "ANNO_GAP_TRACE_INPUT": str(source),
                "ANNO_GAP_TRACE_OUTPUT": str(trace),
            },
            check=True,
        )
        traces = [json.loads(line) for line in trace.read_text().splitlines()]
    published = json.loads((root / "docs/eval/coref-panels.json").read_text())["gap"]
    args.output.write_text(json.dumps(audit(docs, traces, published), indent=2) + "\n")


if __name__ == "__main__":
    main()
