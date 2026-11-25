# /// script
# requires-python = ">=3.11"
# dependencies = ["gliner==0.2.30", "torch==2.15.0", "transformers==5.22.2", "sentencepiece", "onnx==1.23.2", "onnxruntime==1.32.0"]
# ///
"""Compare a verified NuNER Zero-4k FP32 export and a dynamically quantized candidate.

uv run scripts/qa/check_nuner_4k_quantization.py /models/nuner-4k/model.onnx /tmp/zero-4k-u8.onnx --quantize
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nuner_4k_import import CHECKPOINT, REVISION
from nuner_longformer_onnx import parity_cases


def checkpoint_feeds(export_dir):
    from gliner.config import UniEncoderSpanConfig
    from gliner.data_processing.collator import UniEncoderSpanDataCollator
    from gliner.data_processing.processor import UniEncoderSpanProcessor
    from gliner.data_processing.tokenizer import WordsSplitter
    from transformers import AutoTokenizer

    config = UniEncoderSpanConfig.from_json_file(export_dir / "gliner_config.json")
    tokenizer = AutoTokenizer.from_pretrained(export_dir, local_files_only=True)
    splitter = WordsSplitter(config.words_splitter_type)
    processor = UniEncoderSpanProcessor(config, tokenizer, splitter)
    collator = UniEncoderSpanDataCollator(config, data_processor=processor, prepare_labels=False)
    names = ("input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask")
    feeds = []
    for text, labels in parity_cases():
        words = [word for word, _, _ in splitter(text)]
        batch = collator([{"tokenized_text": words, "ner": None}], entity_types=labels)
        feeds.append({name: batch[name].numpy() for name in names})
    return feeds


def infer(path, feeds):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    names = [item.name for item in session.get_inputs()]
    logits = []
    for feed in feeds:
        logits.append(session.run(None, {name: feed[name] for name in names})[0])
        print(f"{path.name} evaluated {feed['input_ids'].shape[1]} tokens", file=sys.stderr, flush=True)
    return logits


def artifact(path):
    with path.open("rb") as f:
        return {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(f, "sha256").hexdigest()}


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -80, 80)))


def compare(feeds, reference, candidate):
    rows = []
    for ((_, labels), feed, expected, actual) in zip(parity_cases(), feeds, reference, candidate, strict=True):
        if expected.shape != actual.shape or not (np.isfinite(expected).all() and np.isfinite(actual).all()):
            raise ValueError("Graph output shape or nonfinite values prevent score comparison")
        before, after = expected > 0, actual > 0
        changes = []
        for index in np.argwhere(before != after)[:5]:
            at = tuple(index)
            changes.append({
                "word_index": int(index[1]),
                "label": labels[index[-1]],
                "fp32_probability": float(sigmoid(expected[at])),
                "candidate_probability": float(sigmoid(actual[at])),
            })
        rows.append({
            "input_tokens": int(feed["input_ids"].shape[-1]),
            "words": int(expected.shape[1]),
            "labels": int(expected.shape[-1]),
            "retained_positive": int((after & before).sum()),
            "lost_positive": int((~after & before).sum()),
            "new_positive": int((after & ~before).sum()),
            "first_changed_decisions": changes,
            "max_abs_logit_difference": float(np.abs(actual - expected).max()),
            "max_abs_probability_difference": float(np.abs(sigmoid(actual) - sigmoid(expected)).max()),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fp32", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--quantize", action="store_true", help="Create a new dynamic QUInt8 candidate")
    args = parser.parse_args()
    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        if args.candidate.exists():
            parser.error("--quantize requires a new candidate path")
        quantize_dynamic(str(args.fp32), str(args.candidate), weight_type=QuantType.QUInt8)
    feeds = checkpoint_feeds(args.fp32.parent)
    results = compare(feeds, infer(args.fp32, feeds), infer(args.candidate, feeds))
    print(json.dumps({
        "fixture_checkpoint": f"{CHECKPOINT}@{REVISION}",
        "onnxruntime": ort.__version__,
        "provider": "CPUExecutionProvider",
        "threshold": 0.5,
        "fp32": artifact(args.fp32),
        "candidate": artifact(args.candidate),
        "cases": results,
    }, indent=2))


if __name__ == "__main__":
    main()
