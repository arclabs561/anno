# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay predicate competitors and fixed-budget suggestions on pinned CrossRE splits."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

from jointie_coverage import gold_graph
from relex_roles import SOURCE_SHA, classify, decode, gold, score_document

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/qa"))
from crossre_prompt_replay import DOMAINS, MANIFEST, REVISION, digest, read_split

MODEL_REVISION = "4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290"
MODEL_FILES = {
    "model.safetensors": SOURCE_SHA,
    "gliner_config.json": "009031313c27dbdc255d4e9d85bdfb89f17259ccf4c82e8982ac763c249c1cdb",
    "tokenizer.json": "d9f64f3126faa24e3ef6bdc47e7120eef528a49cd8b447bdd6865bcf2043cd65",
    "tokenizer_config.json": "ee20eaa4db6f56045339841383f83e98cb79f1672a93bb76a9735243ca65d0fa",
    "special_tokens_map.json": "b2f1b2f15f29a6b6d9d6ea4eca1675d2c231a71477f151d48f79cc83a625ba21",
    "spm.model": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
    "added_tokens.json": "14fedd7e9f345ea45fd9b74245468121d5ccac83e523cad1680529d00bb0e8e8",
}
STOCK_GATE = 0.5
CLI_PREDICATE_GATE = 0.55
FLOOR = 0.30
RULES = {
    "score": (0.0, 0.0),
    "label_margin": (0.5, 0.0),
    "direction_margin": (0.0, 0.5),
    "both_margins": (0.5, 0.5),
}


def text_id(doc):
    return digest(" ".join(doc["sentence"]).encode())


def cohorts(directory, all_eligible=False):
    """Freeze input files, training prompts and disjoint text-hash panels before scoring."""
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        ledger = {}
        for row in csv.DictReader(f):
            if (
                row["id"] != "CrossRE"
                or row["repository"] != "mainlp/CrossRE"
                or row["revision"] != REVISION
            ):
                continue
            key = (row["subset"], row["split"])
            if key in ledger or row["status"] != "ok":
                raise ValueError(f"Missing or duplicate CrossRE ledger entry for {key}")
            ledger[key] = row
    expected = {(d, s) for d in DOMAINS for s in ("train", "dev", "test")}
    if ledger.keys() != expected:
        raise ValueError("Expected all 18 pinned CrossRE split files")
    data, schemas, pins = {}, {}, {}
    for domain in DOMAINS:
        for split in ("train", "dev", "test"):
            r = ledger[domain, split]
            name = f"{domain}-{split}.json"
            docs = read_split(directory / name, r["sha256"])
            if len(docs) != int(r["rows"]):
                raise ValueError(f"Sentence count mismatch for {name}")
            pins[name] = r["sha256"]
            data[domain, split] = docs
        train = data[domain, "train"]
        schemas[domain] = {
            "entity_types": sorted({e[2] for doc in train for e in doc["ner"]}),
            "relation_types": sorted({r[4] for doc in train for r in doc["relations"]}),
        }
    training = {text_id(d) for domain in DOMAINS for d in data[domain, "train"]}
    historical = {text_id(d) for d in data["ai", "test"][:128]}
    blocked = training | historical
    panels = {("historical", "ai"): list(range(64, 96))}
    counts = {}
    for split, size in (("dev", 16), ("test", 32)):
        for domain in DOMAINS:
            docs = data[domain, split]
            earliest = {}
            for i, doc in enumerate(docs):
                key = text_id(doc)
                if key not in blocked:
                    earliest.setdefault(key, i)
            keys = sorted(earliest)
            if not all_eligible:
                keys = keys[:size]
                if len(keys) != size:
                    raise ValueError(f"Too few distinct {domain} {split} sentences")
            panels[split, domain] = [earliest[key] for key in keys]
            counts[domain, split] = len(earliest)
            blocked.update(keys)
        if split == "dev":
            blocked.update(text_id(d) for domain in DOMAINS for d in data[domain, split])
    scope = {}
    for (split, domain), rows in panels.items():
        dataset = data[domain, "test" if split == "historical" else split]
        scope.setdefault(split, {})[domain] = {
            "source_sentences": len(dataset),
            "selected": len(rows),
            "index_text_sha256": digest(
                "".join(f"{i}\t{text_id(dataset[i])}\n" for i in rows).encode()
            ),
        }
        if split != "historical":
            scope[split][domain]["eligible_unique_texts"] = counts[domain, split]
    return data, schemas, panels, pins, scope


def load_model(checkpoint):
    for name, expected in MODEL_FILES.items():
        with (checkpoint / name).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError(f"Unexpected Relex file {name}")
    if version("gliner") != "0.2.30":
        raise ValueError("Use the pinned upstream gliner==0.2.30")
    from gliner import GLiNER

    sys.path.insert(0, str(ROOT / "scripts"))
    from export_gliner_relex_joint import JointRelex

    model = GLiNER.from_pretrained(
        str(checkpoint), local_files_only=True, low_cpu_mem_usage=True
    ).eval()
    return model, JointRelex(model.model).eval()


def infer(model, wrapper, doc, schema):
    grid, proposed, kept, scores, stock = score_document(model, wrapper, doc, schema, STOCK_GATE)
    reference = gold(doc)
    losses = classify(reference, stock, schema["relation_types"], grid, proposed, scores, kept)
    if (stock & reference).total() + losses.total() != reference.total():
        raise ValueError("Exact matches and first losses do not reconcile")
    return kept, scores, stock, losses


def compare_scores(reference, kept, scores, predicates):
    """Compare each low, scored, directed reference key with valid competitors once."""
    gold_labels = defaultdict(set)
    for head, label, tail in reference:
        gold_labels[head, tail].add(label)
    columns = {label: i for i, label in enumerate(predicates)}
    counts = {
        stage: Counter(
            unique_keys=0,
            reference_occurrences=0,
            literal_first=0,
            literal_tied_first=0,
            unannotated_label_higher=0,
            unannotated_reverse_higher=0,
            reverse_unavailable=0,
            reverse_also_correct=0,
            both_rivals_higher=0,
        )
        for stage in ("retained", "suppressed")
    }
    wrong_labels = Counter()
    for (head, label, tail), copies in reference.items():
        row = scores.get((head, tail))
        if row is None or label not in columns:
            continue
        col = columns[label]
        score = row[col]
        if score > STOCK_GATE:
            continue
        where = "retained" if head in kept and tail in kept else "suppressed"
        c = counts[where]
        c["unique_keys"] += 1
        c["reference_occurrences"] += copies
        c["in_suggestion_band_keys"] += score > FLOOR
        c["in_suggestion_band_occurrences"] += copies if score > FLOOR else 0
        better = sum(other > score for other in row)
        tied = sum(other == score for other in row)
        rank = "below_first" if better else "tied_first" if tied > 1 else "first"
        c["literal_" + rank] += 1
        labels = gold_labels[head, tail]
        c["on_multilabel_directed_pair"] += len(labels) > 1
        unexpected = [
            (other, name) for name, other in zip(predicates, row, strict=True) if name not in labels
        ]
        stronger_label = False
        if unexpected:
            best = max(other for other, _ in unexpected)
            stronger_label = best > score
            if stronger_label:
                c["unannotated_label_higher"] += 1
                # Credit one confusion per reference key, retaining an explicit category for ties.
                names = [name for other, name in unexpected if other == best]
                rival = names[0] if len(names) == 1 else "[tied labels]"
                wrong_labels[label + " → " + rival] += 1
            elif best == score:
                c["unannotated_label_tied"] += 1
            else:
                c["above_all_unannotated_labels"] += 1
        else:
            c["all_prompted_labels_annotated"] += 1
        if better and not stronger_label:
            c["only_annotated_labels_higher"] += 1
        reverse = scores.get((tail, head))
        reverse_gold = gold_labels.get((tail, head), set())
        c["same_predicate_annotated_in_reverse"] += label in reverse_gold
        c["different_predicate_annotated_in_reverse"] += bool(reverse_gold - {label})
        stronger_reverse = False
        if reverse is None:
            c["reverse_unavailable"] += 1
        elif label in reverse_gold:
            c["reverse_also_correct"] += 1
        elif reverse[col] > score:
            stronger_reverse = True
            c["unannotated_reverse_higher"] += 1
        elif reverse[col] == score:
            c["unannotated_reverse_tied"] += 1
        else:
            c["above_unannotated_reverse"] += 1
        if stronger_label and stronger_reverse:
            c["both_rivals_higher"] += 1
        if stronger_label or stronger_reverse:
            c["at_least_one_unannotated_rival_higher"] += 1
    return counts, wrong_labels


def suggestions(scores, kept, predicates):
    """Rank at most one additional edge per sentence. Every arm shares the pool and budget."""
    selections = {name: None for name in RULES}
    best = {}
    low_gate = set()
    for (head, tail), row in scores.items():
        if head not in kept or tail not in kept:
            continue
        reverse = scores.get((tail, head))
        for col, (label, score) in enumerate(zip(predicates, row, strict=True)):
            if not FLOOR < score <= STOCK_GATE:
                continue
            edge = (head, label, tail)
            if score > 0.4:
                low_gate.add(edge)
            other = max((s for i, s in enumerate(row) if i != col), default=score)
            label_margin = score - other
            direction_margin = 0.0 if reverse is None else score - reverse[col]
            for name, (label_weight, direction_weight) in RULES.items():
                value = score + label_weight * label_margin + direction_weight * direction_margin
                # A fixed raw-score and lexical tie break makes all arms order-independent.
                key = (-value, -score, edge)
                if name not in best or key < best[name]:
                    best[name] = key
                    selections[name] = edge
    additions = {name: {edge} if edge is not None else set() for name, edge in selections.items()}
    additions["gate_0.40"] = low_gate
    return additions


def document_counts(doc, schema, observation):
    kept, scores, stock, losses = observation
    predicates = schema["relation_types"]
    for row in scores.values():
        if len(row) != len(predicates) or not all(math.isfinite(s) and 0 <= s <= 1 for s in row):
            raise ValueError("Invalid predicate score row")
    reference = gold(doc)
    ranks, confusions = compare_scores(reference, kept, scores, predicates)
    if ranks["retained"]["reference_occurrences"] != losses["predicate_below_threshold"]:
        raise ValueError("Retained low scores failed to reproduce the first-loss audit")
    if ranks["suppressed"]["reference_occurrences"] > losses["flat_overlap_suppression"]:
        raise ValueError("More suppressed low scores than first losses")
    hits = (reference & stock).total()
    stock_counts = Counter(
        sentences=1,
        empty_gold=int(not reference),
        gold=reference.total(),
        unique_gold=len(reference),
        scored_pairs=len(scores),
        matched=hits,
        predicted=stock.total(),
        unmatched=stock.total() - hits,
    )
    added = suggestions(scores, kept, predicates)
    eligible_gold = ranks["retained"]["in_suggestion_band_keys"]
    opportunity = Counter(
        eligible_gold_keys=eligible_gold,
        eligible_gold_occurrences=ranks["retained"]["in_suggestion_band_occurrences"],
        sentences_with_an_eligible_edge=int(bool(added["score"])),
        one_per_sentence_upper_bound=int(eligible_gold > 0),
    )
    _, gold_nodes, _ = gold_graph(doc)
    entity_spans = {(start, end) for _, start, end in gold_nodes}
    reference_pairs = Counter((head, tail) for head, _, tail in reference)
    reference_structure = Counter(
        directed_pairs=len(reference_pairs),
        multilabel_directed_pairs=sum(n > 1 for n in reference_pairs.values()),
        directed_pairs_with_annotated_reverse=sum(
            h != t and (t, h) in reference_pairs for h, t in reference_pairs
        ),
        keys_also_annotated_in_reverse=sum(
            h != t and (t, r, h) in reference for h, r, t in reference
        ),
        self_relation_keys=sum(h == t for h, _, t in reference),
    )
    stock_pairs = {(head, tail) for head, _, tail in stock}
    checked = {}
    for name, edges in added.items():
        if any(edge in stock for edge in edges):
            raise ValueError("An added suggestion already appeared in stock")
        correct = sum(edge in reference for edge in edges)
        checked[name] = Counter(
            added=len(edges),
            correct_added=correct,
            unmatched_added=len(edges) - correct,
            gold_empty_unmatched=len(edges) if not reference else 0,
            on_already_emitted_pair=sum((head, tail) in stock_pairs for head, _, tail in edges),
        )
        for head, label, tail in edges:
            if (head, label, tail) in reference:
                continue
            if (tail, label, head) in reference:
                reason = "unmatched_reverses_exact_reference"
            elif (head, tail) in reference_pairs:
                reason = "unmatched_other_label_on_reference_pair"
            elif (tail, head) in reference_pairs:
                reason = "unmatched_other_label_on_reversed_pair"
            elif head not in entity_spans or tail not in entity_spans:
                reason = "unmatched_with_inexact_endpoint"
            else:
                reason = "unmatched_unrelated_annotated_entities"
            checked[name][reason] += 1
        if name in RULES:
            if len(edges) != len(added["score"]):
                raise ValueError("Equal-budget arms returned different numbers of suggestions")
            checked[name].update(
                different_from_score_only=int(edges != added["score"]),
                correct_gained_vs_score_only=sum(e in reference for e in edges - added["score"]),
                correct_lost_vs_score_only=sum(e in reference for e in added["score"] - edges),
            )
    return {
        "stock": stock_counts,
        "first_loss": losses,
        "retained": ranks["retained"],
        "suppressed": ranks["suppressed"],
        "stronger_label_confusions": confusions,
        "suggestion_opportunity": opportunity,
        "reference_structure": reference_structure,
        **checked,
    }


def aggregate(rows):
    result = defaultdict(Counter)
    for doc in rows:
        for name, counts in doc.items():
            result[name].update(counts)
    return dict(result)


def public_counts(counts):
    return {
        "stock": counts["stock"],
        "misses_by_first_loss": counts["first_loss"],
        "low_predicate_keys": {k: counts[k] for k in ("retained", "suppressed")},
        "stronger_label_confusions": counts["stronger_label_confusions"],
        "suggestion_opportunity": counts["suggestion_opportunity"],
        "reference_structure": counts["reference_structure"],
        "additional_edges": {k: counts[k] for k in (*RULES, "gate_0.40")},
    }


def run_study(directory, observation, all_eligible=False, *, cli_gate_counts=None):
    data, schemas, panels, pins, scope = cohorts(directory, all_eligible)

    def evaluate(panel):
        by_domain = {}
        for (split, domain), indices in panels.items():
            if split != panel:
                continue
            source = data[domain, "test" if split == "historical" else split]
            rows = []
            for index in indices:
                doc = source[index]
                scored = observation(split, domain, index, doc, schemas[domain])
                if cli_gate_counts is not None and split in cli_gate_counts:
                    kept, scores, _, _ = scored
                    filtered = decode(scores, kept, schemas[domain]["relation_types"], CLI_PREDICATE_GATE)
                    reference = gold(doc)
                    cli_gate_counts[split].update(
                        sentences=1, gold=reference.total(),
                        predicted=filtered.total(), matched=(filtered & reference).total(),
                    )
                rows.append(document_counts(doc, schemas[domain], scored))
            by_domain[domain] = aggregate(rows)
        return aggregate(by_domain.values()), by_domain

    hist, _ = evaluate("historical")
    previous = json.loads(
        (Path(__file__).parent / "receipts/relex_ai32_suppressed.json").read_text()
    )
    stock = hist["stock"]
    observed = {k: stock[k] for k in ("gold", "matched", "predicted")}
    observed["directed_pairs"] = stock["scored_pairs"]
    if observed != previous["totals"] or hist["first_loss"] != previous["missed"]:
        raise ValueError("The historical stock replay differs from the recorded diagnostic")
    dev, _ = evaluate("dev")
    # A tie goes to the simpler, earlier predeclared rule. Freeze it before test inference.
    winner = max(RULES, key=lambda name: dev[name]["correct_added"])
    test, by_domain = evaluate("test")
    pooled = {"historical": hist, "dev": dev, "test": test}
    held_out = {}
    for domain, c in by_domain.items():
        held_out[domain] = {
            "stock": c["stock"],
            "suggestion_opportunity": c["suggestion_opportunity"],
            "additional_edges": {
                name: c[name]
                for name in dict.fromkeys(("score", winner, "both_margins", "gate_0.40"))
            },
            "low_retained": {
                k: c["retained"][k]
                for k in (
                    "unique_keys",
                    "literal_first",
                    "unannotated_label_higher",
                    "unannotated_reverse_higher",
                    "at_least_one_unannotated_rival_higher",
                )
            },
        }
    return {
        "crossre_revision": REVISION,
        "source_sha256": pins,
        "checkpoint": "knowledgator/gliner-relex-large-v1.0",
        "checkpoint_revision": MODEL_REVISION,
        "checkpoint_files_sha256": MODEL_FILES,
        "upstream_package": "gliner==0.2.30",
        "stock_parity": {
            "checked_sentences": sum(c["stock"]["sentences"] for c in pooled.values()),
            "comparison": "upstream predict_relations vs. trained Anno export, exact directed edge multiset",
            "runtime": "CPU PyTorch",
        },
        "export_source_sha256": digest(
            (ROOT / "scripts/export_gliner_relex_joint.py").read_bytes()
        ),
        "package_versions": {
            name: version(name) for name in ("gliner", "torch", "tokenizers", "transformers")
        },
        "prompts_from_training": schemas,
        "selection": scope,
        "fresh_panel_policy": {
            "size": "all eligible distinct texts" if all_eligible else "16 dev and 32 test per domain",
            "order": "SHA-256 of the UTF-8 space-joined sentence, ascending",
            "duplicate_policy": "first source occurrence per distinct eligible text",
            "excluded": "all training texts, first 128 AI test texts, previously selected texts",
            "test_also_excludes": "all six development files' texts",
        },
        "stock_gate": STOCK_GATE,
        "suggestion_floor": FLOOR,
        "candidate_weights_label_direction": RULES,
        "selected_on_development": winner,
        "panels": {split: public_counts(c) for split, c in pooled.items()},
        "test_by_domain": held_out,
    }


def tradeoff(report):
    names = {
        "score": "Confidence",
        "label_margin": "Label margin",
        "direction_margin": "Direction margin",
        "both_margins": "Both margins",
        "gate_0.40": "All above 0.40",
    }
    selected = report["selected_on_development"]
    # Show the fixed both-margin comparison even when development selected confidence alone.
    arms = list(dict.fromkeys(("score", selected, "both_margins", "gate_0.40")))
    series = []
    for name, label in (
        ("dev", "Development · six domains"),
        ("test", "Held-out test · six domains"),
    ):
        panel = report["panels"][name]
        base = panel["stock"]
        points = [
            {
                "label": "Stock",
                "correct": base["matched"],
                "unmatched": base["unmatched"],
            }
        ]
        for arm in arms:
            add = panel["additional_edges"][arm]
            points.append(
                {
                    "label": names[arm],
                    "correct": base["matched"] + add["correct_added"],
                    "unmatched": base["unmatched"] + add["unmatched_added"],
                }
            )
        series.append(
            {
                "label": label,
                "observations": base["sentences"],
                "gold": base["gold"],
                "points": points,
            }
        )
    return {
        "schema": "anno.tradeoff/v1",
        "title": "How much do extra CrossRE relation suggestions recover?",
        "unit": "sentences",
        "series": series,
    }


def cli_gate_report(report, counted):
    """Retrospective gate comparison using the original paired sentences and trained scores."""
    panels = {}
    for split in ("dev", "test"):
        panel = report["panels"][split]
        stock, additions = panel["stock"], panel["additional_edges"]["gate_0.40"]
        cli = counted[split]
        if cli["sentences"] != stock["sentences"] or cli["gold"] != stock["gold"]:
            raise ValueError("CLI gate and original study must cover the same reference occurrences")
        panels[split] = {
            "sentences": stock["sentences"], "gold": stock["gold"],
            "index_text_sha256_by_domain": {d: s["index_text_sha256"] for d, s in report["selection"][split].items()},
            "predicate_gates": {
                "0.40": {"matched": stock["matched"] + additions["correct_added"],
                         "predicted": stock["predicted"] + additions["added"]},
                "0.50": {k: stock[k] for k in ("matched", "predicted")},
                f"{CLI_PREDICATE_GATE:.2f}": {k: cli[k] for k in ("matched", "predicted")},
            },
        }
    return {"study": "Retrospective predicate thresholds with fixed retained endpoints and training prompts",
            "checkpoint_revision": report["checkpoint_revision"], "crossre_revision": report["crossre_revision"],
            "source_sha256": report["source_sha256"], "entity_and_adjacency_gate": STOCK_GATE, "panels": panels}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "data", type=Path, help="Directory containing all 18 pinned CrossRE JSONL files"
    )
    cli.add_argument("checkpoint", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--tradeoff-output", type=Path, required=True)
    cli.add_argument("--cli-gate-output", type=Path, help="Also compare the 0.55 CLI predicate gate on dev and test")
    cli.add_argument("--threads", type=int, default=2, help="PyTorch CPU inference threads")
    cli.add_argument(
        "--all-eligible", action="store_true", help="Replay every eligible distinct dev and test text"
    )
    args = cli.parse_args()
    if args.threads < 1:
        cli.error("--threads must be positive")
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model, wrapper = load_model(args.checkpoint)
    cli_counts = {s: Counter() for s in ("dev", "test")} if args.cli_gate_output else None
    with torch.inference_mode():
        result = run_study(
            args.data,
            lambda split, domain, index, doc, schema: infer(model, wrapper, doc, schema),
            all_eligible=args.all_eligible,
            cli_gate_counts=cli_counts,
        )
    outputs = [(args.output, result), (args.tradeoff_output, tradeoff(result))]
    if cli_counts is not None:
        outputs.append((args.cli_gate_output, cli_gate_report(result, cli_counts)))
    for path, data in outputs:
        path.write_text(
            json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
