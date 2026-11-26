#!/usr/bin/env python3
"""Compare exact NER spans by observed language and length on RED-FM paragraphs or MultiNERD sentences."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import tempfile
import time
from urllib.request import urlopen

from multinerd_spans import text_and_gold

REVISION = "eae75dc9bde14d1172351423bcbb82c5b9a94f66"
HASHES = {
    "dev.de": "0d87affeb21a22e65cd5881a668cbbaa48e2e90d4a20b19f076e25f9d8c0d79e",
    "dev.en": "05f8f0125f42d21e6eed886ae673d473b90271e795013e856eb823247f56cc2f",
    "dev.es": "2935599781a53894f57ab0634700b36b93eb30b660e4730a92f40b57dd8d7c5f",
    "dev.fr": "8fbbd1bd04beb54e09587e835a0ee70805077d4ecf376dd8d085e689a0ac6943",
    "dev.it": "1e25587703327ef21f21461a1e9d0c797762386d2f020c19c6bb985bc2c38e9d",
    "test.de": "a588b3614f752909584fe4a912a7f93f82d2fceb4791aec18b73e0ab7cd78c31",
    "test.en": "ec139112f999d4ff551a8323590d6cb2da5dea45e1535e1239793452e940292f",
    "test.es": "e798b831c493462c36286b2c118bdafab42a533094b122708610959d87c9d1e9",
    "test.fr": "e3232a8074bf58406e0ad4964fec8b1f0d03175ce526d9e275402f060ee93bd5",
    "test.it": "4c38c9c34ae69077708ba042cbab2939f33ca7c5d5352040c04fbf43402b3c18",
}
MULTINERD_REVISION = "2814b78e7af4b5a1f1886fe7ad49632de4d9dd25"
MULTINERD_HASHES = {
    "val_en": "f4fd6ad17cfe5bb2b11e9f2a0d425074c67d975c930d0158945556e7ea5f6bfd",
    "val_ru": "1ffb6f7352cd03c746bd9b562038b5c9e4ed9790a64067f673c9a36779a0bd56",
    "test_en": "f89c5544643ec7223ca80c02ac4b50bdc3dab980f2e50d77ca8cae93d86e3428",
    "test_ru": "5beb14bfb0b53ef50a644e9879001ecf936a07b54309920eefefaaadbc715257",
}
MULTINERD_LENGTHS = {"short": (8, 16), "long": (35, 65)}
LANGUAGES = ("de", "en", "es", "fr", "it")
LENGTHS = {"short": (15, 35), "long": (70, 130)}
LABELS = {"person": "PER", "organization": "ORG", "location": "LOC"}
SAUERKRAUT = {
    "config.json": "c54d5f4fa936588ebc03bce9bd9e8dc92b12fd1f182f8c4cc099c58023b2aa21",
    "encoder_config/config.json": "5071796838d57b3cef6af7e9533a135db4483e8a4e7a69479a27f2c89a734986",
    "tokenizer.json": "01f6231290ac28d521e36842a9fbd5e436bf5ad1d361de52f8e545bcab7bd1d9",
    "model.safetensors": "2386dcb222870ea0da86653ebcbfbe85457d5f8d7465b3759299050c409180c2",
}
MODEL_ENV = {"nuner": "NUNER_MODEL_PATH", "bert-onnx": "BERT_ONNX_MODEL_PATH"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_rows(directory, key, fetch, *, hashes=HASHES, prefix=None):
    path = directory / f"{key}.jsonl"
    if fetch:
        resolved = directory.resolve()
        markers = (d / ".git" for d in (resolved, *resolved.parents))
        if any(marker.is_file() or (marker / "HEAD").is_file() for marker in markers):
            raise ValueError("Use an external data cache outside every Git worktree")
    if fetch and not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        if prefix is None:
            prefix = f"https://huggingface.co/datasets/Babelscape/REDFM/resolve/{REVISION}/data"
        url = f"{prefix}/{path.name}"
        with urlopen(url, timeout=90) as source, tempfile.NamedTemporaryFile(dir=directory) as tmp:
            while chunk := source.read(1 << 20):
                tmp.write(chunk)
            tmp.flush()
            if digest(Path(tmp.name)) != hashes[key]:
                raise ValueError(f"Downloaded source differs for {key}")
            # A hard link publishes complete, verified bytes without replacing a concurrent writer.
            try:
                os.link(tmp.name, path)
            except FileExistsError:
                pass
    if digest(path) != hashes[key]:
        raise ValueError(f"Source fingerprint differs for {key}")
    with path.open() as stream:
        return [json.loads(line) for line in stream]


@dataclass
class Case:
    split: str
    cell: str
    text: str
    gold: set
    topic: str
    fingerprint: str
    empty_references: int


def select(rows, split, language, used_topics, per_cell, *, corpus="redfm", lengths=LENGTHS):
    """Sample by identity, without seeing predictions or filtering for entity presence."""
    if any(row["lan"] != language for row in rows):
        raise ValueError(f"Document language differs from {language}")
    cases = []
    for length, (lo, hi) in lengths.items():
        candidates = [r for r in rows if lo <= len(r["text"].split()) <= hi]
        def rank(r):
            key = f"anno-{corpus}-ner-v1/{split}/{language}/{r['docid']}"
            return hashlib.sha256(key.encode()).hexdigest()
        n = 0
        for row in sorted(candidates, key=rank):
            topic = row["uri"]
            if not topic:
                raise ValueError("Missing topic identity")
            if topic in used_topics:
                continue
            if corpus == "multinerd":
                text, labels = text_and_gold([row["original"]])
                if text != row["text"]:
                    raise ValueError("Source reconstruction differs between selection and evaluation")
                gold = {(a, b, ("PER", "ORG", "LOC")[i]) for a, b, i in labels}
                empty = 0
            else:
                gold = spans(row["text"], row["entities"], "boundaries")
                empty = sum(e["type"] in LABELS.values() and not e["surfaceform"] for e in row["entities"])
            cases.append(Case(split, f"{language}/{length}", row["text"], gold, topic, rank(row), empty))
            used_topics.add(topic)
            n += 1
            if n == per_cell:
                break
        if n != per_cell:
            raise ValueError(f"{split}/{language}/{length}: only {n} unique topics for {per_cell} cases")
    return cases


def spans(text, entities, source):
    result = set()
    for entity in entities:
        if source == "boundaries":
            label = entity["type"]
            if label not in LABELS.values():
                continue
            start, end = entity["boundaries"]
            surface = entity["surfaceform"]
            if 0 <= start == end <= len(text) and surface == "":
                continue
        else:
            label = entity[source]
            start, end, surface = entity["start"], entity["end"], entity["text"]
        if not (0 <= start < end <= len(text)) or text[start:end] != surface:
            raise ValueError("Returned span does not slice back to source text")
        label = LABELS.get(label.lower(), label.upper())
        if label in LABELS.values():
            result.add((start, end, label))
    return result


def counts(gold, pred):
    return len(gold & pred), len(pred - gold), len(gold - pred)


def f1(totals):
    tp, fp, fn = totals
    den = 2 * tp + fp + fn
    return 2 * tp / den if den else None


def summary(outcomes):
    tp, fp, fn = (sum(col) for col in zip(*outcomes, strict=True))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None, "exact_f1": f1((tp, fp, fn))}


def paired_delta(pairs):
    def difference(sample):
        a, b = zip(*sample, strict=True)
        x, y = summary(a)["exact_f1"], summary(b)["exact_f1"]
        return None if x is None or y is None else 100 * (y - x)
    rng = random.Random(73)
    values = [difference(rng.choices(pairs, k=len(pairs))) for _ in range(2000)]
    defined = sorted(n for n in values if n is not None)
    return {"points": difference(pairs), "defined_resamples": len(defined),
            "percentile_95": [defined[math.floor(0.025 * (len(defined)-1))],
                              defined[math.ceil(0.975 * (len(defined)-1))]] if defined else None}


def predictor(args):
    if args.backend == "sauerkraut":
        from glinerel_fast import GLiNERel
        import torch
        if not args.checkpoint or any(digest(args.checkpoint / name) != sha for name, sha in SAUERKRAUT.items()):
            raise ValueError("Use the verified SauerkrautLM-GLiNEREL-72m checkpoint")
        if importlib.metadata.version("glinerel-fast") != "0.1.1":
            raise ValueError("Replay requires glinerel-fast==0.1.1")
        model = GLiNERel.from_pretrained(args.checkpoint, device="cpu", precision=args.precision, threads=1)
        def predict(text, threshold):
            output = model.extract(text, entities=list(LABELS), entity_threshold=threshold)
            return spans(text, output["entities"], "label"), "glinerel-fast"
        return predict, {"backend": args.backend, "precision": args.precision, "torch": torch.__version__,
                         "runtime": "glinerel-fast==0.1.1", "checkpoint_sha256": SAUERKRAUT}

    if getattr(args, "precision", "fp32") != "fp32":
        raise ValueError("For the native CLI, precision is determined by the ONNX graph")
    if args.anno is None:
        raise ValueError("Provide --anno for an Anno backend")
    cmd = [str(args.anno.resolve()), "extract", "--model", args.backend, "--format", "json"]
    env = dict(os.environ, ANNO_NO_DOWNLOADS="1")
    identity = {"backend": args.backend, "cli_sha256": digest(args.anno)}
    if args.backend in MODEL_ENV:
        if not args.checkpoint or not args.checkpoint.is_dir():
            raise ValueError("Use an explicit, local checkpoint directory")
        artifacts = {p.name: digest(p) for p in sorted(args.checkpoint.iterdir())
                     if p.is_file() and p.suffix in {".onnx", ".data", ".json"}}
        if not any(k.endswith(".onnx") for k in artifacts):
            raise ValueError("No ONNX artifact in checkpoint directory")
        env[MODEL_ENV[args.backend]] = str(args.checkpoint.resolve())
        identity["checkpoint_sha256"] = artifacts
        if args.backend == "nuner":
            cmd.extend(["--extract-types", ",".join(LABELS)])
    elif args.checkpoint:
        raise ValueError("This backend does not accept a local checkpoint")
    def predict(text, threshold):
        output = subprocess.run([*cmd, "--threshold", str(threshold), "--text", text],
                                capture_output=True, env=env)
        if output.returncode:
            raise RuntimeError(f"{args.backend} exited {output.returncode}; panel incomplete")
        record = json.loads(output.stdout)
        prov = record["provenance"]
        if prov["model"] != args.backend or prov["runtime_model"] != args.backend:
            raise ValueError(f"Expected {args.backend}, received {prov.get('runtime_model')}")
        if args.checkpoint and prov.get("checkpoint") != str(args.checkpoint.resolve()):
            raise ValueError("Model provenance differs from the evaluated checkpoint")
        return spans(text, record["entities"], "type"), prov["runtime_model"]
    return predict, identity


def run(args):
    if args.per_cell < 1:
        raise ValueError("per-cell budget must be positive")
    cases, used = [], set()
    if args.corpus == "multinerd":
        corpus = f"Babelscape/multinerd@{MULTINERD_REVISION}"
        hashes, lengths = MULTINERD_HASHES, MULTINERD_LENGTHS
    else:
        corpus = f"Babelscape/REDFM@{REVISION}"
        hashes, lengths = HASHES, LENGTHS
    for split in ("dev", "test"):
        for lang in (("en", "ru") if args.corpus == "multinerd" else LANGUAGES):
            if args.corpus == "multinerd":
                partition = "val" if split == "dev" else "test"
                prefix = f"https://huggingface.co/datasets/Babelscape/multinerd/resolve/{MULTINERD_REVISION}/{partition}"
                source = source_rows(args.data_dir, f"{partition}_{lang}", args.fetch, hashes=hashes, prefix=prefix)
                rows = []
                for i, raw in enumerate(source):
                    text = " ".join(raw["tokens"])
                    rows.append({"uri": hashlib.sha256(text.encode()).hexdigest(), "docid": i,
                                 "text": text, "lan": raw["lang"], "original": raw})
            else:
                rows = source_rows(args.data_dir, f"{split}.{lang}", args.fetch)
            cases.extend(select(rows, split, lang, used, args.per_cell, corpus=args.corpus, lengths=lengths))
    predict, identity = predictor(args)
    reference_predict = reference_identity = None
    reference_name = "heuristic"
    if args.compare_heuristic:
        baseline = argparse.Namespace(backend="heuristic", checkpoint=None, anno=args.anno)
        reference_predict, reference_identity = predictor(baseline)
    elif args.compare_fp32:
        if args.backend != "sauerkraut" or args.precision == "fp32" or args.calibrate:
            raise ValueError("--compare-fp32 requires the external BF16 arm and the fixed threshold 0.5")
        reference_name = "fp32"
        reference_predict, reference_identity = predictor(argparse.Namespace(**{**vars(args), "precision": "fp32"}))
    changed_documents = Counter()
    reference_pairs, recovered = defaultdict(list), defaultdict(lambda: [0, 0])
    thresholds = [0.5, 0.3, 0.7] if args.calibrate else [0.5]
    reports, runtime, timings = {}, Counter(), defaultdict(list)
    selected = 0.5
    for split in ("dev", "test"):
        score = defaultdict(lambda: defaultdict(list))
        gains = defaultdict(lambda: [0, 0])
        split_thresholds = thresholds if split == "dev" else list(dict.fromkeys([0.5, selected]))
        for case in (case for case in cases if case.split == split):
            reference = None
            if split == "test" and reference_predict:
                baseline_pred, _ = reference_predict(case.text, 0.5)
            for t in split_thresholds:
                before = time.perf_counter()
                pred, observed = predict(case.text, t)
                elapsed = time.perf_counter() - before
                runtime[observed] += 1
                candidate_counts = counts(case.gold, pred)
                score[t][case.cell].append(candidate_counts)
                if split == "test" and t == selected and reference_predict:
                    reference_pairs[case.cell].append((counts(case.gold, baseline_pred), candidate_counts))
                    changed_documents[case.cell] += pred != baseline_pred
                    recovered[case.cell][0] += len((pred - baseline_pred) & case.gold)
                    recovered[case.cell][1] += len((baseline_pred - pred) & case.gold)
                if split == "test" and t == 0.5:
                    timings[case.cell].append(elapsed)
                if t == 0.5:
                    reference = pred
                elif split == "test":
                    gains[case.cell][0] += len((pred - reference) & case.gold)
                    gains[case.cell][1] += len((reference - pred) & case.gold)
        reports[split] = {t: {cell: summary(samples) for cell, samples in groups.items()}
                          for t, groups in score.items()}
        if split == "dev":
            def quality(t):
                value = summary([x for samples in score[t].values() for x in samples])["exact_f1"]
                return -1 if value is None else value
            selected = max(thresholds, key=quality)
        elif selected != 0.5:
            reports["test_selected_minus_fixed"] = {
                cell: {**paired_delta(list(zip(score[0.5][cell], score[selected][cell], strict=True))),
                       "correct_gained": gains[cell][0], "correct_lost": gains[cell][1]}
                for cell in score[0.5]}
    if reference_predict:
        reports[f"test_selected_minus_{reference_name}"] = {
            cell: {**paired_delta(pairs), "changed_documents": changed_documents[cell], "correct_gained": recovered[cell][0],
                   "correct_lost": recovered[cell][1]}
            for cell, pairs in reference_pairs.items()}
    empty_references = Counter()
    for case in cases:
        if case.empty_references:
            empty_references[f"{case.split}/{case.cell}"] += case.empty_references
    return {"empty_reference_annotations_omitted": dict(empty_references),
            "reference_model": reference_identity, "dataset": corpus,
            "source_sha256": hashes, "per_cell": args.per_cell,
            "panel_sha256": hashlib.sha256("\n".join(c.fingerprint for c in cases).encode()).hexdigest(),
            "labels": list(LABELS.values()), "length_words": lengths, "model": identity,
            "observed_runtimes_by_call": dict(runtime), "dev_selected_threshold": selected,
            "resource_scope": "wall time per test call at 0.5; peak RSS for process or maximum child, Linux KiB",
            "test_p95_seconds": {k: sorted(v)[math.ceil(0.95 * len(v))-1] for k, v in timings.items()},
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF if args.backend == "sauerkraut"
                                               else resource.RUSAGE_CHILDREN).ru_maxrss,
            "scores": reports}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", choices=("redfm", "multinerd"), default="redfm",
                   help="RED-FM paragraphs (CC BY-SA 4.0) or English/Russian MultiNERD sentences (CC BY-NC-SA 4.0)")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--fetch", action="store_true", help="Fetch missing pinned data into an external cache")
    p.add_argument("--anno", type=Path)
    p.add_argument("--backend", choices=("heuristic", "crf", "hmm", "nuner", "bert-onnx", "sauerkraut"), required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32",
                   help="CPU arithmetic in the optional GLiNEREL runtime")
    p.add_argument("--per-cell", type=int, default=12)
    compare = p.add_mutually_exclusive_group()
    compare.add_argument("--compare-fp32", action="store_true", help="Pair the external BF16 arm against FP32 at 0.5")
    compare.add_argument("--compare-heuristic", action="store_true",
                         help="Score the same test examples with Anno heuristic at 0.5")
    p.add_argument("--calibrate", action="store_true", help="Select a threshold on development paragraphs")
    print(json.dumps(run(p.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
