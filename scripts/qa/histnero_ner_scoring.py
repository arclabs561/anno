"""Replay the three complete HistNERo source files with literal and dataset scoring."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from english_ner_source_replay import canonical, metric, predict, sha

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from build_offline_native_cache import native_bytes

DATASET = "HistNERo"
MODEL = "heuristic"
LABELS = [
    "O",
    "B-PERS",
    "I-PERS",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-PROD",
    "I-PROD",
    "B-DATE",
    "I-DATE",
]
ALIGNED = {"PERS": "PER", "PROD": "PRODUCT"}
MODES = ("source", "dataset")


def source_case(row):
    tokens, ids = row["tokens"], row["ner_tags"]
    if len(tokens) != len(ids) or any(
        not isinstance(t, str) or t != t.strip(" ") for t in tokens
    ):
        raise ValueError("Unexpected source token or edge space in a nonempty token")
    source, aligned, starts = {}, {}, []
    pos = 0
    for t in tokens:
        starts.append(pos)
        pos += len(t) + 1
    text = " ".join(tokens)
    for i, k in enumerate(ids):
        if type(k) is not int or k < 0 or k >= len(LABELS):
            raise ValueError("Unrecognized publisher class index")
        if k == 0:
            continue
        if k % 2 == 0:
            if i == 0 or ids[i - 1] not in (k - 1, k):
                raise ValueError("Orphan inside tag in pinned source")
            continue
        j = i + 1
        while j < len(ids) and ids[j] == k + 1:
            j += 1
        label = LABELS[k][2:]
        key = len(source)
        source[key] = (starts[i], starts[j - 1] + len(tokens[j - 1]), label)
        observed = [p for p in range(i, j) if tokens[p]]
        if observed:
            first, last = observed[0], observed[-1]
            aligned[key] = (
                starts[first],
                starts[last] + len(tokens[last]),
                ALIGNED.get(label, label),
            )
    return text, source, aligned


def run_reports(anno, mode, env, work):
    path = work / "counts.jsonl"
    args = [
        anno,
        "dataset",
        "eval",
        "--cached-only",
        "--dataset",
        DATASET,
        "--model",
        MODEL,
        "--reports",
        path,
    ]
    if mode:
        args.extend(["--ner-scoring", mode])
    result = subprocess.run(args, check=True, env=env, capture_output=True, text=True)
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    total = re.search(r"Gold: (\d+)\s+Predicted: (\d+)\s+Correct: (\d+)", result.stdout)
    if not total or tuple(map(int, total.groups())) != tuple(
        sum(r[k] for r in rows) for k in ("gold_count", "predicted_count", "correct")
    ):
        raise ValueError("Native text summary and JSON reports disagree")
    return rows


def fingerprints(gold):
    return {
        str(i): sha(canonical([label, [{"start": a, "end": b}] if a < b else []]))
        for i, (a, b, label) in gold.items()
    }


def visible_result(gold, predicted, report, row_id, text, model):
    distinct = set(predicted)
    exact = {i for i, span in gold.items() if span in distinct}
    if (
        report["pairing"]["document_id"],
        report["pairing"]["source_sha256"],
        report["pairing"]["gold_keys"],
        report["model"],
    ) != (row_id, sha(text.encode()), fingerprints(gold), model):
        raise ValueError(
            f"{row_id}: independently reconstructed source/gold differ from the native report"
        )
    if exact != {m["gold_id"] for m in report["matches"] if m["type"] == "correct"} or (
        len(exact),
        len(gold),
        len(predicted),
    ) != (report["correct"], report["gold_count"], report["predicted_count"]):
        raise ValueError(f"{row_id}: exact gold identities or suggestion count differ")
    return exact


def person_workload(gold, predicted):
    counts = Counter(
        {
            k: 0
            for k in (
                "exact_person",
                "duplicate_exact_person",
                "same_boundary_other_type",
                "overlap_only",
                "no_overlap",
            )
        }
    )
    seen = set()
    wrong_types = Counter()
    for a, b, t in predicted:
        if t != "PER":
            continue
        if (a, b, "PER") in gold:
            category = "duplicate_exact_person" if (a, b) in seen else "exact_person"
        elif same := {label for c, d, label in gold if (a, b) == (c, d)}:
            category = "same_boundary_other_type"
            if len(same) != 1:
                raise ValueError(
                    "More than one gold type shares a boundary in flat BIO"
                )
            wrong_types.update(same)
        elif any(a < d and c < b for c, d, _ in gold):
            category = "overlap_only"
        else:
            category = "no_overlap"
        seen.add((a, b))
        counts[category] += 1
    return counts, wrong_types


def protocol_totals(gold, predictions, exact):
    by_label = {
        k: metric(gold[k], predictions[k], exact[k])
        for k in sorted(gold.keys() | predictions.keys())
    }
    return (
        metric(sum(gold.values()), sum(predictions.values()), sum(exact.values())),
        by_label,
    )


def analyze(anno, before, env, work, raw, info):
    rows = json.loads(raw)
    cache = work / "datasets"
    cache.mkdir(exist_ok=True)
    (cache / "HistNERo.cache").write_bytes(native_bytes("histnero-rows", raw))
    cases = [source_case(row) for row in rows]
    inputs = [
        (str(row["id"]), text, [])
        for row, (text, _, _) in zip(rows, cases, strict=True)
    ]
    if len({i for i, _, _ in inputs}) != len(inputs):
        raise ValueError("Duplicate source ID within split")
    reports = {mode: run_reports(anno, mode, env, work) for mode in MODES}
    predictions = predict(anno, MODEL, inputs, env)
    if before:
        previous = run_reports(before, None, env, work)
        for a, b in zip(previous, reports["source"], strict=True):
            if any(
                a[k] != b[k]
                for k in (
                    "pairing",
                    "matches",
                    "label_outcomes",
                    "gold_count",
                    "predicted_count",
                    "correct",
                )
            ):
                raise ValueError(
                    "Earlier native result differs from the source protocol"
                )
        prior_predictions = predict(before, MODEL, inputs, env)
        if any(
            a["entities"] != b["entities"]
            for a, b in zip(prior_predictions, predictions, strict=True)
        ):
            raise ValueError("Earlier native model emitted different suggestions")
    totals = Counter()
    golds = {mode: Counter() for mode in MODES}
    hits = {mode: Counter() for mode in MODES}
    preds = Counter()
    work_items = Counter()
    same_boundary_types = Counter()
    outputs = []
    effects = Counter()
    trimmed_types = Counter()
    empty_types = Counter()
    for index, (row, (text, source, aligned), p, srep, arep) in enumerate(
        zip(
            rows, cases, predictions, reports["source"], reports["dataset"], strict=True
        ),
        1,
    ):
        row_id = f"HistNERo:{index}"
        digest = sha(text.encode())
        if (p["id"], p["runtime_model"], p["source_sha256"], p["text_length"]) != (
            str(row["id"]),
            MODEL,
            digest,
            len(text),
        ):
            raise ValueError("Native prediction source or backend disagrees")
        typed = []
        for v in p["entities"]:
            a, b, t = v["start"], v["end"], v["type"]
            if not 0 <= a < b <= len(text) or text[a:b] != v["text"]:
                raise ValueError("Invalid Unicode prediction extent")
            typed.append((a, b, t))
        outputs.append([p["id"], digest, typed])
        predset = set(typed)
        preds.update(t for _, _, t in typed)
        exact_source = visible_result(source, typed, srep, row_id, text, MODEL)
        exact_aligned = visible_result(aligned, typed, arep, row_id, text, MODEL)
        label_only = {
            i for i, (a, b, t) in source.items() if (a, b, ALIGNED.get(t, t)) in predset
        }
        for label, first, last in [
            ("labels", exact_source, label_only),
            ("extents", label_only, exact_aligned),
        ]:
            effects[label + "_gained"] += len(last - first)
            effects[label + "_lost"] += len(first - last)
        for mode, g, matched in [
            ("source", source, exact_source),
            ("dataset", aligned, exact_aligned),
        ]:
            golds[mode].update(t for _, _, t in g.values())
            hits[mode].update(g[i][2] for i in matched)
        excluded = source.keys() - aligned.keys()
        trimmed = {i for i in aligned if source[i][:2] != aligned[i][:2]}
        empty_types.update(source[i][2] for i in excluded)
        trimmed_types.update(source[i][2] for i in trimmed)
        if arep["ner_scoring"] != {
            "mode": "dataset",
            "reference": {
                "source_gold": len(source),
                "scored_gold": len(aligned),
                "excluded_empty_gold": len(excluded),
                "trimmed_gold": len(trimmed),
            },
        }:
            raise ValueError(
                "Native denominator accounting disagrees with original tokens"
            )
        workload, wrong_types = person_workload(set(aligned.values()), typed)
        work_items.update(workload)
        same_boundary_types.update(wrong_types)
        totals.update(
            tokens=len(row["tokens"]),
            empty_tokens=row["tokens"].count(""),
            tagged_empty_tokens=sum(
                not t and bool(k)
                for t, k in zip(row["tokens"], row["ner_tags"], strict=True)
            ),
            tagged_tokens=sum(bool(k) for k in row["ner_tags"]),
            empty_gold_cases=int(not source),
        )
    recorded = {}
    by_type = {}
    for mode in MODES:
        recorded[mode], by_type[mode] = protocol_totals(golds[mode], preds, hits[mode])
    if sum(work_items.values()) != preds["PER"]:
        raise ValueError("Person workload does not reconcile")
    return {
        "cases": len(rows),
        "source": {"url": info["url"], "sha256": sha(raw), "bytes": len(raw)},
        "source_gold": sum(golds["source"].values()),
        **totals,
        "zero_length_gold_spans": sum(empty_types.values()),
        "trimmed_gold": sum(trimmed_types.values()),
        "excluded_source_labels": dict(empty_types),
        "trimmed_source_labels": dict(trimmed_types),
        "source_labels": dict(golds["source"]),
        "dataset_labels": dict(golds["dataset"]),
        "scoring": recorded,
        "per_type": by_type,
        "effects": dict(effects),
        "person_suggestions": dict(work_items),
        "person_same_boundary_gold_types": dict(same_boundary_types),
        "ordered_predictions_sha256": sha(canonical(outputs)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno", required=True, type=Path)
    parser.add_argument(
        "--prior-anno",
        type=Path,
        help="Optional earlier CLI with dataset eval --reports to verify the original baseline",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sources",
        type=Path,
        help="Directory with train.json, valid.json, test.json, README.md and publisher-class-labels.md",
    )
    group.add_argument(
        "--archive",
        type=Path,
        help="Native general cache ZIP containing the original publisher files",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    catalog = json.loads((SCRIPTS / "data/offline-native-sources.json").read_text())
    spec = next(x for x in catalog["datasets"] if x["id"] == DATASET)
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        source = folder / "sources" if args.archive else args.sources
        if args.archive:
            source.mkdir()
            with zipfile.ZipFile(args.archive) as z:
                for item in spec["inputs"]:
                    fname = item["url"].rsplit("/", 1)[1]
                    (source / fname).write_bytes(
                        z.read(f"originals/HistNERo/{item['variant']}/{fname}")
                    )
                for notice in spec["notices"]:
                    (source / notice["name"]).write_bytes(
                        z.read("notices/HistNERo/" + notice["name"])
                    )
        notices = {
            n["name"]: n
            for n in spec["notices"]
            if n["name"] in ("README.md", "publisher-class-labels.md")
        }
        for name, n in notices.items():
            if sha((source / name).read_bytes()) != n["sha256"]:
                raise ValueError("Unexpected source schema or readme")
        author = (source / "publisher-class-labels.md").read_text()
        found = [s for _, s in re.findall(r"^\s+'(\d+)': (\S+)$", author, re.MULTILINE)]
        if found != LABELS:
            raise ValueError("Publisher class order does not match")
        study = {
            "schema": "anno.histnero-ner-scoring/v1",
            "dataset": DATASET,
            "backend": MODEL,
            "anno_version": subprocess.check_output(
                [args.anno, "--version"], text=True
            ).strip(),
            "source_protocol_reference_revision": "1b6fcc97df321e1614bdc94f00c9b18ab36bb49a",
            "source_semantics": notices,
            "class_order": LABELS,
            "label_alignment": ALIGNED,
            "splits": {},
        }
        docs = {}
        texts = {}
        for item in spec["inputs"]:
            key = item["variant"]
            raw = (source / item["url"].rsplit("/", 1)[1]).read_bytes()
            if sha(raw) != item["sha256"]:
                raise ValueError(f"Unexpected {key} source bytes")
            work = folder / key
            work.mkdir()
            env = {
                **os.environ,
                "ANNO_CACHE_DIR": str(work),
                "ANNO_S3_CACHE": "0",
                "ANNO_NO_DOWNLOADS": "1",
                "HF_HUB_OFFLINE": "1",
                "NO_COLOR": "1",
            }
            study["splits"][key] = analyze(
                args.anno.resolve(),
                args.prior_anno.resolve() if args.prior_anno else None,
                env,
                work,
                raw,
                item,
            )
            input_rows = json.loads(raw)
            docs[key] = [r["doc_id"] for r in input_rows]
            texts[key] = [sha(" ".join(r["tokens"]).encode()) for r in input_rows]
        for key, rec in study["splits"].items():
            rec["source_documents"] = len(set(docs[key]))
            rec["distinct_texts"] = len(set(texts[key]))
            if key != "train":
                training_docs = set(docs["train"])
                training_texts = set(texts["train"])
                rec["documents_also_in_training"] = len(set(docs[key]) & training_docs)
                rec["records_from_training_documents"] = sum(
                    d in training_docs for d in docs[key]
                )
                rec["records_with_identical_training_text"] = sum(
                    t in training_texts for t in texts[key]
                )
        args.output.write_text(
            json.dumps(study, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )


if __name__ == "__main__":
    main()
