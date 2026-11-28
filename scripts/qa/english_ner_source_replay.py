"""Score the CoNLL04 entity layer and SEC FIN3 on their complete pinned test files.

The reference reader and the native dataset reports must agree on every input
and gold occurrence. Both trained and heuristic predictions come from Anno.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from nuner_fallback_routing import BERT_CHECKPOINT, BERT_FILES, file_sha

EVAL = Path(__file__).resolve().parents[2] / "docs/eval"
SOURCES = {
    "CoNLL04RE": (
        "data/CoNLL04/test.txt",
        "288-case joint extraction test · entity layer",
        "7f40a0f7c1892821591358518c542139e88f693e",
        "d172819deff771c059c9c6688d67c3ff2bdaa25cf71ec56b1f7e8c6705f96700",
    ),
    "SECFilingsNER": (
        "data/SEC-filings/CONLL-format/data/test/FIN3.txt",
        "SEC FIN3 · complete mirrored test file",
        "8522f4b39cf9ed323b77f2d8a5005628f8861908",
        "fc71db3cb718bf7400af08c9080d435ec2a002ddd9070c8bcbb4e8e3396835bd",
    ),
}
SHARED = {"PER", "ORG", "LOC"}
CONLL04 = {"Peop": "PER", "Org": "ORG", "Loc": "LOC", "Other": "Other"}
PREDICTION_ALIASES = {"misc": "MISC"}
BACKENDS = ("heuristic", "bert-onnx")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()


def source_for(dataset):
    path, _, revision, digest = SOURCES[dataset]
    with (EVAL / "dataset-repo-files.csv").open(encoding="utf-8", newline="") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["id"] == dataset
            and (r["path"], r["revision"], r["sha256"]) == (path, revision, digest)
            and r["split"] == "test"
            and r["status"] == "ok"
        ]
    if len(rows) != 1:
        raise ValueError(f"Expected one verified test file for {dataset}")
    return rows[0]


def blocks(dataset, lines):
    current, identity = [], None
    for line in lines:
        if dataset == "CoNLL04RE":
            if line.startswith("#doc "):
                if current:
                    yield identity, current
                identity = line.removeprefix("#doc ").strip()
                if not identity.isdecimal():
                    raise ValueError("Invalid CoNLL04 document identifier")
                current = []
            elif line.strip():
                fields = line.split("\t")
                if (
                    len(fields) != 5
                    or identity is None
                    or int(fields[0]) != len(current)
                ):
                    raise ValueError("Invalid indexed CoNLL04 release row")
                current.append((fields[1], fields[2]))
        elif line.startswith("-DOCSTART-") or not line.strip():
            if current:
                yield None, current
                current = []
        else:
            word, _, _, tag = line.split()
            current.append((word, tag))
    if current:
        yield identity, current


def cases_for(dataset, raw):
    cases, labels, words, tagged = [], Counter(), 0, 0
    for index, (identity, tokens) in enumerate(
        blocks(dataset, raw.decode("utf-8").splitlines()), 1
    ):
        text = " ".join(word for word, _ in tokens)
        gold, previous, pos = [], None, 0
        for word, tag in tokens:
            end = pos + len(word)
            if tag == "O":
                previous = None
            else:
                prefix, label = tag.split("-", 1)
                if prefix not in ("I", "B"):
                    raise ValueError("Expected an IOB entity tag")
                mapped = CONLL04[label] if dataset == "CoNLL04RE" else label
                if prefix == "I" and previous == label:
                    start = gold[-1][0]
                    gold[-1] = (start, end, mapped)
                else:
                    gold.append((pos, end, mapped))
                    labels[label] += 1
                previous = label
                tagged += 1
            pos = end + 1
        words += len(tokens)
        cases.append((identity or f"FIN3-{index}", text, gold))
    if len(cases) != len({case[0] for case in cases}):
        raise ValueError("Repeated record identity")
    return cases, labels, words, tagged


def native_reports(anno, dataset, backend, env, path):
    subprocess.run(
        [
            anno,
            "dataset",
            "eval",
            "--dataset",
            dataset,
            "--task",
            "ner",
            "--model",
            backend,
            "--reports",
            path,
        ],
        stdout=subprocess.DEVNULL,
        check=True,
        env=env,
    )
    return [json.loads(row) for row in path.read_bytes().splitlines()]


def predict(anno, backend, cases, env):
    inputs = b"".join(
        canonical({"id": identity, "text": text}) + b"\n" for identity, text, _ in cases
    )
    output = subprocess.run(
        [
            anno,
            "batch",
            "--stdin",
            "--format",
            "jsonl",
            "--batch-size",
            "1",
            "--parallel",
            "1",
            "--model",
            backend,
            "-q",
        ],
        input=inputs,
        stdout=subprocess.PIPE,
        check=True,
        env=env,
    ).stdout
    return [json.loads(row) for row in output.splitlines()]


def metric(gold, predicted, correct):
    def rate(n, d):
        return round(n / d, 8) if d else None

    return {
        "gold": gold,
        "predicted": predicted,
        "correct": correct,
        "precision": rate(correct, predicted),
        "recall": rate(correct, gold),
        "f1": rate(2 * correct, predicted + gold),
    }


def native_gold_key(label, start, end):
    return sha(canonical([label, [{"start": start, "end": end}]]))


def score(dataset, backend, cases, native, predictions):
    counts, correct_ids, all_outputs = Counter(), [], []
    by_type = {}
    for index, ((identity, text, gold), report, pred) in enumerate(
        zip(cases, native, predictions, strict=True), 1
    ):
        digest = sha(text.encode())
        if (
            pred["id"],
            pred["model"],
            pred["runtime_model"],
            pred["source_sha256"],
            pred["text_length"],
        ) != (identity, backend, backend, digest, len(text)):
            raise ValueError(f"{identity}: native prediction source or model differs")
        pair = report["pairing"]
        keys = {
            str(i): native_gold_key(label, a, b) for i, (a, b, label) in enumerate(gold)
        }
        if (
            pair["document_id"],
            pair["source_sha256"],
            pair["gold_keys"],
            report["model"],
            report["gold_count"],
        ) != (f"{dataset}:{index}", digest, keys, backend, len(gold)):
            raise ValueError(
                f"{identity}: native reader disagrees with independently recovered source text, gold spans or model"
            )
        raw = []
        for p in pred["entities"]:
            a, b, label = p["start"], p["end"], p["type"]
            if not 0 <= a < b <= len(text) or text[a:b] != p["text"]:
                raise ValueError(f"{identity}: invalid native Unicode span")
            raw.append((a, b, label))
        if len(raw) != pred["entity_count"] or report["predicted_count"] != len(raw):
            raise ValueError(
                f"{identity}: native report and prediction stream have different output counts"
            )
        gold_set = set(gold)
        if len(gold) != len(gold_set):
            raise ValueError("Duplicate typed source mention")
        native_exact = {
            gold[e["gold_id"]] for e in report["matches"] if e["type"] == "correct"
        }
        if (
            native_exact != set(raw) & gold_set
            or len(native_exact) != report["correct"]
        ):
            raise ValueError(f"{identity}: independent native exact matches disagree")
        projected = [
            (a, b, PREDICTION_ALIASES.get(label, label)) for a, b, label in raw
        ]
        distinct = set(projected)
        if len(distinct) != len(set(raw)):
            raise ValueError(
                "The MISC alias collapsed two otherwise distinct native predictions"
            )
        hits = distinct & gold_set
        types_per_boundary = Counter((a, b) for a, b, _ in distinct)
        correct_ids.append(hits)
        all_outputs.append([identity, sorted(raw)])
        counts.update(
            gold_empty_predictions=0 if gold else len(raw),
            gold_empty_cases_with_prediction=int(bool(raw) and not gold),
            literal_native_matches=len(native_exact),
            repeated_typed_predictions=len(projected) - len(distinct),
            cases_with_repeated_typed_prediction=int(len(projected) != len(distinct)),
            distinct_spans_with_competing_types=sum(
                n > 1 for n in types_per_boundary.values()
            ),
        )
        for field, spans in (
            ("gold", gold_set),
            ("predicted", projected),
            ("correct", hits),
        ):
            for _, _, label in spans:
                by_type.setdefault(label, Counter())[field] += 1

    def totals(labels):
        vals = [
            sum(by_type[t][f] for t in labels if t in by_type)
            for f in ("gold", "predicted", "correct")
        ]
        return metric(*vals)

    all_labels = totals(by_type)
    return {
        "all_labels": all_labels,
        "deduplicated_typed_predictions_counterfactual": metric(
            all_labels["gold"],
            all_labels["predicted"] - counts["repeated_typed_predictions"],
            all_labels["correct"],
        ),
        "shared_labels": totals(SHARED),
        "by_label": {t: totals([t]) for t in sorted(by_type)},
        **counts,
        "ordered_predictions_sha256": sha(canonical(all_outputs)),
    }, correct_ids


def sec_contract_roles(
    cases, baseline, candidate, baseline_predictions, candidate_predictions
):
    """Describe FIN3's PER annotations by the two most frequent gold surfaces."""
    role_counts = Counter(
        text[a:b].casefold()
        for _, text, gold in cases
        for a, b, label in gold
        if label == "PER"
    )
    top = {surface for surface, _ in role_counts.most_common(2)}
    strata = {}
    for (_, text, gold), before, after, base_pred, cand_pred in zip(
        cases,
        baseline,
        candidate,
        baseline_predictions,
        candidate_predictions,
        strict=True,
    ):
        entities = cand_pred["entities"]
        base_bounds = {(p["start"], p["end"]) for p in base_pred["entities"]}
        cand_bounds = {(p["start"], p["end"]) for p in entities}
        for mention in gold:
            a, b, label = mention
            if label != "PER":
                continue
            surface = text[a:b].casefold()
            name = surface if surface in top else "other_PER"
            row = strata.setdefault(
                name,
                {
                    "exact": Counter(),
                    "exact_boundary_any_type": Counter(),
                    "candidate_on_lost_baseline_mentions": Counter(),
                },
            )
            lost = mention in before and mention not in after
            row["exact"].update(
                gold=1,
                baseline=int(mention in before),
                candidate=int(mention in after),
                gained=int(mention in after and mention not in before),
                lost=int(lost),
            )
            earlier = (a, b) in base_bounds
            later = (a, b) in cand_bounds
            row["exact_boundary_any_type"].update(
                baseline=int(earlier),
                candidate=int(later),
                gained=int(later and not earlier),
                lost=int(earlier and not later),
            )
            if lost:
                same = sorted(
                    {
                        PREDICTION_ALIASES.get(p["type"], p["type"])
                        for p in entities
                        if (p["start"], p["end"]) == (a, b)
                    }
                )
                if same:
                    outcome = "same_span_" + "_and_".join(same)
                elif any(p["start"] < b and a < p["end"] for p in entities):
                    outcome = "partial_overlap"
                else:
                    outcome = "no_overlap"
                row["candidate_on_lost_baseline_mentions"][outcome] += 1
    return strata


def measure(args):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        cache = folder / "datasets"
        cache.mkdir()
        checkpoint = folder / "bert-fp32"
        checkpoint.mkdir()
        for member, digest in BERT_FILES.items():
            model_file = (args.bert / member).resolve()
            if file_sha(model_file) != digest:
                raise ValueError(f"Unexpected model file {member}")
            (checkpoint / member).symlink_to(model_file)
        env = {
            **os.environ,
            "ANNO_CACHE_DIR": str(folder),
            "ANNO_NO_DOWNLOADS": "1",
            "ANNO_S3_CACHE": "0",
            "HF_HUB_OFFLINE": "1",
            "BERT_ONNX_MODEL_PATH": str(checkpoint),
        }
        datasets = {}
        for name, source_file in (
            ("CoNLL04RE", args.conll04),
            ("SECFilingsNER", args.sec),
        ):
            if source_file is None:
                continue
            provenance = source_for(name)
            raw = source_file.read_bytes()
            if sha(raw) != provenance["sha256"]:
                raise ValueError(f"Unexpected pinned {name} file")
            (cache / (name + ".cache")).write_bytes(raw)
            cases, labels, words, tagged = cases_for(name, raw)
            if (words, tagged) != (
                int(provenance["tokens"]),
                int(provenance["tagged"]),
            ):
                raise ValueError(
                    f"{name}: source counts differ from published byte audit"
                )
            result = {
                "split": "test",
                "scope": SOURCES[name][1],
                "source": {
                    k: provenance[k]
                    for k in ("repository", "url", "revision", "path", "sha256")
                },
                "cases": len(cases),
                "tokens": words,
                "tagged_tokens": tagged,
                "gold": sum(labels.values()),
                "gold_empty_cases": sum(not g for _, _, g in cases),
                "distinct_texts": len({t for _, t, _ in cases}),
                "observed_gold_labels": dict(sorted(labels.items())),
                "source_label_mapping": (
                    CONLL04 if name == "CoNLL04RE" else {t: t for t in sorted(labels)}
                ),
                "ordered_cases_sha256": sha(
                    canonical(
                        [
                            [identity, sha(text.encode()), gold]
                            for identity, text, gold in cases
                        ]
                    )
                ),
                "backends": {},
            }
            result["source"]["bytes"] = len(raw)
            if name == "CoNLL04RE" and args.prior_anno:
                old = native_reports(
                    args.prior_anno, name, "heuristic", env, folder / "old.jsonl"
                )
                if (
                    len(old) != 1
                    or "['N']" not in old[0]["label_outcomes"]["by_gold_label"]
                ):
                    raise ValueError(
                        "--prior-anno does not reproduce the pre-fix CoNLL04 reader"
                    )
                result["prior_reader"] = {
                    "revision": "fe6a4c1",
                    "cases": len(old),
                    "gold": sum(r["gold_count"] for r in old),
                    "distinct_gold_labels": len(
                        {t for r in old for t in r["label_outcomes"]["by_gold_label"]}
                    ),
                }
            hits, predictions = [], []
            for backend in BACKENDS:
                reports = native_reports(
                    args.anno, name, backend, env, folder / (backend + ".jsonl")
                )
                output = predict(args.anno, backend, cases, env)
                metrics, exact = score(name, backend, cases, reports, output)
                result["backends"][backend] = metrics
                hits.append(exact)
                predictions.append(output)
            if name == "SECFilingsNER":
                result["person_role_audit"] = sec_contract_roles(
                    cases, *hits, *predictions
                )
            gains, losses = Counter(), Counter()
            for before, after in zip(*hits, strict=True):
                gains.update(label for _, _, label in after - before)
                losses.update(label for _, _, label in before - after)
            result["paired"] = {
                "gained": sum(gains.values()),
                "lost": sum(losses.values()),
                "by_label": {
                    t: {"gained": gains[t], "lost": losses[t]}
                    for t in sorted(gains.keys() | losses.keys())
                },
            }
            datasets[name] = result
    return {
        "schema": "anno.english-ner-source/v1",
        "checkpoint": BERT_CHECKPOINT,
        "checkpoint_sha256": BERT_FILES,
        "prediction_label_aliases": PREDICTION_ALIASES,
        "shared_types": sorted(SHARED),
        "datasets": datasets,
    }


def paired(study):
    rows = []
    for name, result in study["datasets"].items():
        p = result["paired"]
        row = {
            "label": name,
            "role": result["scope"],
            "observations": result["cases"],
            "gained": p["gained"],
            "lost": p["lost"],
        }
        for field, backend in zip(("baseline", "candidate"), BACKENDS, strict=True):
            score = result["backends"][backend]["all_labels"]
            row[field] = {
                "correct": score["correct"],
                "unmatched": score["predicted"] - score["correct"],
            }
        if (
            row["candidate"]["correct"] - row["baseline"]["correct"]
            != row["gained"] - row["lost"]
        ):
            raise ValueError("Paired identities disagree with the aggregate")
        rows.append(row)
    title = "What does BERT recover on news and filing text?"
    if len(rows) == 1:
        title = f"What does BERT recover on {rows[0]['label']}?"
    return {
        "schema": "anno.paired/v1",
        "title": title,
        "unit": "source records",
        "baseline": "HeuristicNER",
        "candidate": "FP32 BERT NER",
        "rows": rows,
    }


def write(path, obj):
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="command", required=True)
    replay = cmds.add_parser("replay")
    for field in ("anno", "bert", "output", "paired"):
        replay.add_argument("--" + field, required=True, type=Path)
    replay.add_argument("--conll04", type=Path, help="Pinned adapted CoNLL04 test file")
    replay.add_argument("--sec", type=Path, help="Pinned FIN3 test file")
    replay.add_argument(
        "--prior-anno",
        type=Path,
        help="Previously built fe6a4c1 CLI for the failing reader audit",
    )
    chart = cmds.add_parser("chart")
    chart.add_argument("source", type=Path)
    chart.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "chart":
        write(args.output, paired(json.loads(args.source.read_text(encoding="utf-8"))))
    else:
        if not (args.conll04 or args.sec):
            parser.error("replay requires --conll04, --sec, or both")
        result = measure(args)
        write(args.output, result)
        write(args.paired, paired(result))


if __name__ == "__main__":
    main()
