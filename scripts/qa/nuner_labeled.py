# /// script
# requires-python = ">=3.11"
# dependencies = ["gliner==0.2.30", "torch==2.15.0", "transformers==5.22.2", "numpy", "onnxruntime==1.32.0", "sentencepiece"]
# ///
"""Prepare paired MultiNERD inputs, run one graph per process, and score both predictions."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np

REV = "2814b78e7af4b5a1f1886fe7ad49632de4d9dd25"
FILES = {
    "en": "f89c5544643ec7223ca80c02ac4b50bdc3dab980f2e50d77ca8cae93d86e3428",
    "ru": "5beb14bfb0b53ef50a644e9879001ecf936a07b54309920eefefaaadbc715257",
}
LABELS = ["person", "organization", "location"]
THRESHOLDS = [0.3, 0.5, 0.7]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def dataset(directory, lang):
    filename = f"test_{lang}.jsonl"
    path = directory / filename
    if not path.exists():
        url = f"https://huggingface.co/datasets/Babelscape/multinerd/resolve/{REV}/test/{filename}"
        with urlopen(url, timeout=90) as f:
            path.write_bytes(f.read())
    if digest(path) != FILES[lang]:
        raise ValueError(f"Wrong test file digest: {filename}")
    return [json.loads(line) for line in path.read_text().splitlines()]


def text_and_gold(rows):
    offset = 0
    sentences, gold = [], []
    for row in rows:
        sentences.append(" ".join(row["tokens"]))
        prev = 0
        for token, tag in zip(row["tokens"], row["ner_tags"], strict=True):
            end = offset + len(token)
            if 1 <= tag <= 6:
                label = (tag - 1) // 2
                if tag % 2:
                    gold.append([offset, end, label])
                elif prev in (tag - 1, tag):
                    gold[-1][1] = end
                else:
                    raise ValueError("Orphan I tag in source data")
            prev = tag
            offset = end + 1
    return "\n".join(sentences), gold


def prepare(directory, export):
    from gliner.config import UniEncoderSpanConfig
    from gliner.data_processing.collator import UniEncoderSpanDataCollator
    from gliner.data_processing.processor import UniEncoderSpanProcessor
    from gliner.data_processing.tokenizer import WordsSplitter
    from transformers import AutoTokenizer

    cfg = UniEncoderSpanConfig.from_json_file(export / "gliner_config.json")
    tokenizer = AutoTokenizer.from_pretrained(export, local_files_only=True)
    split = WordsSplitter(cfg.words_splitter_type)
    processor = UniEncoderSpanProcessor(cfg, tokenizer, split)
    collate = UniEncoderSpanDataCollator(
        cfg, data_processor=processor, prepare_labels=False
    )
    data = {lang: dataset(directory, lang) for lang in FILES}
    order = sorted(
        range(len(data["en"])),
        key=lambda i: hashlib.sha256(f"nuner-multinerd-v1/en/{i}".encode()).digest(),
    )
    ordinary = order[:24]
    unicode_only = [
        i for i in order[24:] if not " ".join(data["en"][i]["tokens"]).isascii()
    ][:12]
    selections = [("en-random", "en", [i]) for i in ordinary]
    selections += [("en-unicode", "en", [i]) for i in unicode_only]
    # Pack complete, contiguous test sentences. The source does not expose document IDs.
    for lang, target, start in [("en", 1800, 2000), ("ru", 4096, 1500)]:
        ids, n_tokens, n_words = [], 0, 0
        for i in range(start, len(data[lang])):
            text, _ = text_and_gold([data[lang][i]])
            words = [w for w, _, _ in split(text)]
            piece_count = len(
                tokenizer(words, is_split_into_words=True, add_special_tokens=False)[
                    "input_ids"
                ]
            )
            # Three single-token labels, three ENT tokens, SEP, BOS and EOS.
            if (
                n_tokens + piece_count + 9 > target
                or n_words + len(words) > cfg.max_len
            ):
                break
            ids.append(i)
            n_tokens += piece_count
            n_words += len(words)
        if lang == "en" and set(ids).intersection(ordinary + unicode_only):
            raise ValueError("English stress and sentence panels must be disjoint")
        selections.append((f"{lang}-packed", lang, ids))

    panel = []
    for index, (group, lang, ids) in enumerate(selections):
        text, gold = text_and_gold([data[lang][i] for i in ids])
        words = list(split(text))
        batch = collate(
            [{"tokenized_text": [w for w, _, _ in words], "ner": None}],
            entity_types=LABELS,
        )
        names = [
            "input_ids",
            "attention_mask",
            "words_mask",
            "text_lengths",
            "span_idx",
            "span_mask",
        ]
        feed = {key: batch[key].numpy() for key in names}
        if feed["input_ids"].shape[1] > 4096 or feed["text_lengths"].item() != len(
            words
        ):
            raise ValueError(f"Truncated or oversized input at case {index}")
        np.savez(directory / f"input-{index}.npz", **feed)
        panel.append(
            {
                "group": group,
                "language": lang,
                "test_rows_zero_based": ids,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "chars": len(text),
                "unicode": not text.isascii(),
                "tokens": int(feed["input_ids"].shape[1]),
                "words": [[a, b] for _, a, b in words],
                "gold": gold,
            }
        )
    write(directory / "panel.json", panel)
    print([(p["group"], p["tokens"], len(p["gold"])) for p in panel], flush=True)


def run(directory, graph, key):
    import onnxruntime as ort

    panel = json.loads((directory / "panel.json").read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    graph_sha = digest(graph)
    start = time.perf_counter()
    runtime = ort.InferenceSession(
        str(graph), options, providers=["CPUExecutionProvider"]
    )
    load_s = time.perf_counter() - start
    inputs = [dict(np.load(directory / f"input-{i}.npz")) for i in range(len(panel))]
    runtime.run(None, inputs[0])  # same warm-up, excluded from measured calls
    timings = []
    cpu_start = time.process_time()
    for index, (doc, feed) in enumerate(zip(panel, inputs, strict=True)):
        start = time.perf_counter()
        logits = runtime.run(None, feed)[0]
        timings.append(time.perf_counter() - start)
        if (
            logits.shape != (1, len(doc["words"]), 1, len(LABELS))
            or not np.isfinite(logits).all()
        ):
            raise ValueError(f"Invalid output at case {index}: {logits.shape}")
        np.save(directory / f"{key}-{index}.npy", logits[0, :, 0, :])
        print(
            f"{key} case {index + 1}/{len(panel)}, {doc['tokens']} tokens: {timings[-1]:.2f}s",
            flush=True,
        )
    write(
        directory / f"{key}-run.json",
        {
            "sha256": graph_sha,
            "bytes": graph.stat().st_size,
            "ort": ort.__version__,
            "provider": runtime.get_providers(),
            "machine": platform.machine(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "intra_threads": 2,
            "inter_threads": 1,
            "session_load_seconds": load_s,
            "timings_seconds": timings,
            "inference_loop_cpu_seconds": time.process_time() - cpu_start,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )


def decode(logits, words, text, threshold):
    # Match Anno's NuNER width-one decoder: best label per word, then merge adjacent same-label words.
    result, previous = [], None
    for row, (start, end) in zip(logits, words, strict=True):
        label = int(row.argmax())
        if 1 / (1 + np.exp(-float(row[label]))) < threshold:
            previous = None
            continue
        if previous == label:
            result[-1][1] = end
        else:
            result.append([start, end, label])
        previous = label
    cleaned = []
    for start, end, label in result:
        # textprep::spans::clean_span_tail, called by NuNER::create_entity
        surface = text[start:end].rstrip(".,;:!?)]")
        for suffix in ("'s", "’s"):
            while surface.endswith(suffix):
                surface = surface[:-2]
        if surface:
            cleaned.append([start, start + len(surface), label])
    return cleaned


def metrics(tp, pred, gold):
    return {
        "tp": tp,
        "fp": pred - tp,
        "fn": gold - tp,
        "f1": 2 * tp / (pred + gold) if pred + gold else None,
        "precision": tp / pred if pred else None,
        "recall": tp / gold if gold else None,
    }


def paired_interval(cases):
    """Resample sentences jointly across graphs; retain empty-gold and empty-prediction rows."""
    counts = []
    for case in cases:
        gold = set(map(tuple, case["gold"]))
        row = []
        for key in ("fp32", "u8"):
            pred = set(map(tuple, case[key]))
            row.append([2 * len(gold & pred), len(gold) + len(pred)])
        counts.append(row)
    indices = np.random.default_rng(20261125).integers(
        len(cases), size=(10000, len(cases))
    )
    totals = np.asarray(counts)[indices].sum(axis=1)
    valid = (totals[:, :, 1] > 0).all(axis=1)
    scores = totals[valid, :, 0] / totals[valid, :, 1]
    delta = scores[:, 1] - scores[:, 0]
    return {
        "resamples": len(indices),
        "defined": len(delta),
        "seed": 20261125,
        "percentile_95": np.quantile(delta, [0.025, 0.975]).tolist(),
    }


def report(directory, output):
    panel = json.loads((directory / "panel.json").read_text())
    data = {lang: dataset(directory, lang) for lang in FILES}
    texts = []
    for case in panel:
        rows = [data[case["language"]][i] for i in case["test_rows_zero_based"]]
        text, gold = text_and_gold(rows)
        if (
            hashlib.sha256(text.encode()).hexdigest() != case["text_sha256"]
            or gold != case["gold"]
        ):
            raise ValueError(
                "Test input or annotations changed between preparation and scoring"
            )
        texts.append(text)
    receipt = {
        "checkpoint": "numind/NuNER_Zero-4k",
        "checkpoint_revision": "7a7cd8d65af2572c297054dbaf8f25c0d46da55d",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("gliner", "transformers", "torch", "numpy", "onnxruntime")
        },
        "dataset": "Babelscape/multinerd",
        "dataset_revision": REV,
        "test_file_sha256": FILES,
        "labels_by_index": LABELS,
        "models": {},
        "cases": [],
    }
    for graph in ("fp32", "u8"):
        stats = json.loads((directory / f"{graph}-run.json").read_text())
        groups = {}
        for group in dict.fromkeys(p["group"] for p in panel):
            indices = [i for i, p in enumerate(panel) if p["group"] == group]
            scores = {}
            for t in THRESHOLDS:
                tp = pred = gold = 0
                for i in indices:
                    case = panel[i]
                    labels = set(map(tuple, case["gold"]))
                    estimates = set(
                        map(
                            tuple,
                            decode(
                                np.load(directory / f"{graph}-{i}.npy"),
                                case["words"],
                                texts[i],
                                t,
                            ),
                        )
                    )
                    tp += len(estimates & labels)
                    pred += len(estimates)
                    gold += len(labels)
                scores[str(t)] = metrics(tp, pred, gold)
            times = [stats["timings_seconds"][i] for i in indices]
            groups[group] = {
                "n": len(indices),
                "thresholds": scores,
                "elapsed_seconds": sum(times),
                "median_seconds": statistics.median(times),
            }
        receipt["models"][graph] = {**stats, "groups": groups}
    for i, p in enumerate(panel):
        case = {k: v for k, v in p.items() if k != "words"}
        case["feed_sha256"] = digest(directory / f"input-{i}.npz")
        probabilities = {}
        for graph in ("fp32", "u8"):
            logits = np.load(directory / f"{graph}-{i}.npy")
            case[graph] = decode(logits, p["words"], texts[i], 0.5)
            probabilities[graph] = 1 / (1 + np.exp(-logits.astype(np.float64)))
        a, b = probabilities["fp32"], probabilities["u8"]
        case["compared_word_labels"] = int(a.size)
        case["max_absolute_probability_change"] = float(np.max(np.abs(a - b)))
        case["threshold_crossings"] = {
            str(t): int(((a >= t) != (b >= t)).sum()) for t in THRESHOLDS
        }
        receipt["cases"].append(case)
    receipt["paired_f1_delta_u8_minus_fp32_at_0.5"] = {
        group: paired_interval(
            [case for case in receipt["cases"] if case["group"] == group]
        )
        for group in ("en-random", "en-unicode")
    }
    write(output, receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare").add_argument("export_dir", type=Path)
    runner = sub.add_parser("run")
    runner.add_argument("graph", type=Path)
    runner.add_argument("key", choices=["fp32", "u8"])
    sub.add_parser("report").add_argument("output", type=Path)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        prepare(args.work, args.export_dir)
    elif args.action == "run":
        run(args.work, args.graph, args.key)
    else:
        report(args.work, args.output)
