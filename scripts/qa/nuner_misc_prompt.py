"""Replay one frozen fourth NuNER prompt on the full WikiGold release and WNUT test.

Build the two native workers at the same revision, applying only the example
patch to the candidate. The fourth prompt is mapped to WikiGold MISC. WNUT
has no such gold class, so its MISC outputs receive no correctness judgment.
"""

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import wikigold_nuner_routing as wiki
import wnut_native_replay as wnut
from wikigold_entity_errors import STOCK_SPANS_SHA256

PATCH = Path(__file__).resolve().parent / "fixtures/nuner-wikigold-misc-prompt.patch"
PROMPT = "miscellaneous"


def row_digest(rows):
    # Same ordered representation as the independently evaluated WikiGold stock run.
    h = hashlib.sha256()
    for row in rows:
        pair = [row["input_sha256"], row["spans"]]
        h.update(
            (
                json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode()
        )
    return h.hexdigest()


def predictions(row, text):
    if row["input_sha256"] != wiki.routing.text_sha(text):
        raise ValueError("Worker and reference inputs differ in text or order")
    aligned = set()
    for start, end, raw_label in row["spans"]:
        if not 0 <= start < end <= len(text):
            raise ValueError("Native span is outside the input")
        label = {"misc": "MISC", PROMPT: "MISC"}.get(raw_label, raw_label)
        aligned.add((start, end, label))
    if len(aligned) != len(row["spans"]):
        raise ValueError("Projected native predictions collide")
    return aligned


def sensitivity(groups, unit, include_misc):
    """Paired percentile bootstrap of a per-sentence difference at the original volume."""
    # Order the empirical distribution canonically so the saved histogram is
    # sufficient for an exact offline replay, including groups with zero change.
    if not groups:
        raise ValueError("A paired prompt study requires sentence occurrences")
    histogram = Counter(tuple(row) for row in groups.values())
    clusters = sorted(histogram.elements())
    n = sum(row[0] for row in clusters)
    samples = {
        key: [] for key in (("all", "misc", "other") if include_misc else ("all",))
    }
    replicates, seed = 4000, 430
    rng = random.Random(seed)
    for _ in range(replicates):
        picked = rng.choices(clusters, k=len(clusters))
        size, delta, misc = (sum(column) for column in zip(*picked, strict=True))
        factor = n / size
        samples["all"].append(factor * delta)
        if include_misc:
            samples["misc"].append(factor * misc)
            samples["other"].append(factor * (delta - misc))
    intervals = {}
    for name, values in samples.items():
        quantiles = statistics.quantiles(values, n=40, method="inclusive")
        intervals[name] = {
            "lower": round(quantiles[0], 2),
            "upper": round(quantiles[-1], 2),
            "method": f"95% paired bootstrap by {unit}, scaled to {n:,} sentences",
        }
    distribution = []
    for (size, delta, misc), count in sorted(histogram.items()):
        row = {"sentences": size, "net_exact": delta, "groups": count}
        if include_misc:
            row["net_misc_exact"] = misc
        distribution.append(row)
    return {
        "unit": unit,
        "groups": len(clusters),
        "sentences": n,
        "replicates": replicates,
        "seed": seed,
        "histogram": distribution,
        "intervals": intervals,
    }


def score(
    cases, stock, candidate, projection, *, group_ids=None, unit="sentence occurrence"
):
    inventory = set(projection.values())
    if group_ids is None:
        group_ids = range(1, len(cases) + 1)
    groups = defaultdict(lambda: [0, 0, 0])
    membership = hashlib.sha256()
    by_type = {label: Counter() for label in sorted(inventory)}
    totals, target = Counter(), Counter()
    prior_misc = dict.fromkeys(
        ("same_span_other_type", "overlap_only", "no_overlap"), 0
    )
    candidate_misc = dict.fromkeys(("same_span_misc", *prior_misc), 0)

    def overlap(span, alternatives):
        a, b, _ = span
        if any((a, b) == other[:2] for other in alternatives):
            return "same_span_other_type"
        if any(a < end and start < b for start, end, _ in alternatives):
            return "overlap_only"
        return "no_overlap"

    for (text, annotations), before, after, group in zip(
        cases, stock, candidate, group_ids, strict=True
    ):
        gold = {(a, b, projection[t]) for a, b, t in annotations}
        if len(gold) != len(annotations):
            raise ValueError("Projected gold spans collide")
        old, new = (predictions(row, text) for row in (before, after))
        if before["ran"] != after["ran"] or (not before["ran"] and old != new):
            raise ValueError("The arms must share routing and skipped-sentence outputs")
        gains, losses = (new - old) & gold, (old - new) & gold
        row = groups[group]
        row[0] += 1
        row[1] += len(gains) - len(losses)
        row[2] += sum(x[2] == "MISC" for x in gains) - sum(
            x[2] == "MISC" for x in losses
        )
        membership.update(f"{group} {before['input_sha256']}\n".encode("ascii"))
        totals.update(
            sentences=1,
            gold_empty_sentences=int(not gold),
            nuner_calls=int(before["ran"]),
            stock_skipped=int(not before["ran"]),
            sentences_changed=int(old != new),
            gold=len(gold),
            stock_exact=len(old & gold),
            candidate_exact=len(new & gold),
            exact_gained=len(gains),
            exact_lost=len(losses),
            stock_suggestions=len(old),
            candidate_suggestions=len(new),
            stock_mapped_suggestions=sum(s[2] in inventory for s in old),
            candidate_mapped_suggestions=sum(s[2] in inventory for s in new),
            stock_mapped_unmatched=sum(p[2] in inventory for p in old - gold),
            candidate_mapped_unmatched=sum(p[2] in inventory for p in new - gold),
            stock_misc_predictions=sum(p[2] == "MISC" for p in old),
            candidate_misc_predictions=sum(p[2] == "MISC" for p in new),
        )
        for label, c in by_type.items():
            for name, spans in (
                ("gold", gold),
                ("stock_exact", old & gold),
                ("candidate_exact", new & gold),
                ("gained", gains),
                ("lost", losses),
            ):
                c[name] += sum(s[2] == label for s in spans)
        for span in new:
            if span[2] == "MISC":
                match = "same_span_misc" if span in gold else overlap(span, gold)
                candidate_misc[match] += 1
        if "MISC" in inventory:
            for span in gains:
                if span[2] == "MISC":
                    prior_misc[overlap(span, old)] += 1
            old_wrong = {(a, b) for a, b, t in old if t != "MISC"}
            new_bounds = {(a, b) for a, b, _ in new}
            for g in gold - old:
                if g[2] == "MISC" and g[:2] in old_wrong:
                    target["stock_same_bounds_wrong_type"] += 1
                    target["corrected"] += int(g in new)
                    target["same_bounds_wrong_type_after"] += int(
                        g not in new and g[:2] in new_bounds
                    )
                    target["boundaries_missing_after"] += int(g[:2] not in new_bounds)
    result = {
        **totals,
        "by_gold_label": by_type,
        "candidate_misc_reference_overlap": candidate_misc,
        "resampling": sensitivity(groups, unit, "MISC" in inventory),
        "ordered_group_and_input_sha256": membership.hexdigest(),
        "stock_ordered_output_sha256": row_digest(stock),
        "candidate_ordered_output_sha256": row_digest(candidate),
    }
    if "MISC" in inventory:
        result["stock_mislabeled_misc"] = target
        result["new_exact_misc_by_stock_overlap"] = prior_misc
    return result


def receipt(
    wiki_cases,
    wiki_source,
    wiki_documents,
    wnut_cases,
    wnut_source,
    wiki_rows,
    wnut_rows,
    workers,
):
    for rows in (*wiki_rows, *wnut_rows):
        for row in rows:
            expected_layers = [
                ["nuner", "succeeded" if row["ran"] else "skipped"],
                ["regex", "succeeded"],
                ["heuristic", "succeeded"],
            ]
            if row.get("layers") != expected_layers:
                raise ValueError(
                    "Prompt comparison requires the recorded BERT-absent native layer outcomes"
                )
    if len(set(wiki_documents)) != wiki_source["documents_with_sentences"]:
        raise ValueError(
            "WikiGold resampling groups differ from the pinned document boundaries"
        )
    wikigold = score(
        wiki_cases,
        *wiki_rows,
        wiki.PROJECTION,
        group_ids=wiki_documents,
        unit="WikiGold document",
    )
    if wikigold["stock_ordered_output_sha256"] != STOCK_SPANS_SHA256:
        raise ValueError("WikiGold stock disagrees with the original error study")
    if wikigold["stock_mislabeled_misc"]["stock_same_bounds_wrong_type"] != 263:
        raise ValueError(
            "WikiGold target population differs from the original error study"
        )
    test = score(wnut_cases, *wnut_rows, wnut.PROJECTION)
    recorded = json.loads(wiki.STUDY.read_text())["splits"]["Test"]["without_bert"]
    expected = recorded["full_split"]
    if (
        test["stock_skipped"],
        sum(c["stock_exact"] for c in test["by_gold_label"].values()),
        test["stock_mapped_unmatched"],
    ) != (
        recorded["stock_skipped"],
        expected["stock_exact"],
        expected["stock_in_schema_unmatched"],
    ):
        raise ValueError("WNUT stock disagrees with the original routing study")
    return {
        "model": wiki.routing.CHECKPOINT,
        "checkpoint_sha256": wiki.routing.FILES,
        "stock_prompts": ["person", "organization", "location"],
        "candidate_prompts": ["person", "organization", "location", PROMPT],
        "nuner_threshold": 0.9,
        "final_threshold": 0.5,
        "bert": "absent",
        "routing": "stock, no fallback opt-in",
        "worker_sha256": {
            role: wiki.routing.file_sha(path) for role, path in workers.items()
        },
        "scoring": "Exact typed Unicode-character spans on identical ordered sentences. Suggestions include every final native output. Mapped suggestions include only the corpus scoring inventory. The fourth prompt maps to MISC. WNUT has no gold MISC class and excludes candidate MISC outputs from mapped suggestions.",
        "candidate_patch_sha256": wiki.routing.file_sha(PATCH),
        "wikigold": {"source": wiki_source, **wikigold},
        "wnut_test": {"source": wnut_source, **test},
    }


def paired(report):
    rows = []
    for corpus, group, role, labels in [
        (
            "wikigold",
            "WikiGold · all four mapped types",
            "Complete English release",
            None,
        ),
        (
            "wikigold",
            "WikiGold · MISC only",
            "Type subset of the same release",
            {"MISC"},
        ),
        (
            "wikigold",
            "WikiGold · PER, ORG and LOC",
            "Type subset of the same release",
            {"PER", "ORG", "LOC"},
        ),
        (
            "wnut_test",
            "WNUT-17 · five mapped types",
            "Official test · MISC has no gold class",
            None,
        ),
    ]:
        arm = report[corpus]
        counts = arm["by_gold_label"]
        selected = set(counts) if labels is None else labels
        if not selected <= counts.keys():
            raise ValueError("Missing a scored gold label in the prompt comparison")
        row = {
            "label": group,
            "role": role,
            "observations": arm["sentences"],
            "gained": sum(counts[t]["gained"] for t in selected),
            "lost": sum(counts[t]["lost"] for t in selected),
        }
        for field, key in (("baseline", "stock"), ("candidate", "candidate")):
            correct = sum(counts[t][f"{key}_exact"] for t in selected)
            suggestions = arm[f"{key}_mapped_suggestions"]
            if labels is not None:
                misc = arm[f"{key}_misc_predictions"]
                suggestions = misc if "MISC" in selected else suggestions - misc
            unmatched = suggestions - correct
            if unmatched < 0 or (
                labels is None
                and (correct, unmatched)
                != (arm[f"{key}_exact"], arm[f"{key}_mapped_unmatched"])
            ):
                raise ValueError(
                    "Prompt comparison suggestions disagree with exact and unmatched totals"
                )
            row[field] = {
                "correct": correct,
                "unmatched": unmatched,
                "calls": arm["nuner_calls"],
            }
        if (
            row["candidate"]["correct"] - row["baseline"]["correct"]
            != row["gained"] - row["lost"]
        ):
            raise ValueError(
                "Prompt comparison counts and paired gain/loss identities differ"
            )
        key = "all" if labels is None else ("misc" if "MISC" in labels else "other")
        row["interval"] = arm["resampling"]["intervals"][key]
        rows.append(row)
    return {
        "schema": "anno.paired/v1",
        "title": "What did a fourth NuNER prompt change?",
        "baseline": "Three prompts",
        "candidate": "Add miscellaneous",
        "unit": "sentences",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    replay_parser = commands.add_parser(
        "replay", help="Run both native workers on each complete source"
    )
    for arg in ("wikigold", "wnut", "stock", "candidate", "checkpoint", "output"):
        replay_parser.add_argument("--" + arg, type=Path, required=True)
    replay_parser.add_argument(
        "--paired", type=Path, help="Also write the native paired chart input"
    )
    chart = commands.add_parser(
        "chart", help="Rebuild the native chart input from a saved receipt"
    )
    chart.add_argument("receipt", type=Path)
    chart.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "chart":
        wiki.write_json(
            args.output, paired(json.loads(args.receipt.read_text(encoding="utf-8")))
        )
        return
    for member, digest in wiki.routing.FILES.items():
        if wiki.routing.file_sha(args.checkpoint / member) != digest:
            raise ValueError(f"Unexpected NuNER checkpoint member {member}")
    wc, ws, wiki_documents = wiki.load_release_with_documents(args.wikigold)
    ns, nc, _ = wnut.load_cases(args.wnut)

    def replay(cases):
        return [
            wiki.routing.run_worker(binary, args.checkpoint, cases)
            for binary in (args.stock, args.candidate)
        ]

    results = receipt(
        wc,
        ws,
        wiki_documents,
        nc,
        ns,
        replay(wc),
        replay(nc),
        {"stock": args.stock, "candidate": args.candidate},
    )
    wiki.write_json(args.output, results)
    if args.paired:
        wiki.write_json(args.paired, paired(results))


if __name__ == "__main__":
    main()
