#!/usr/bin/env python3
"""Emit a cache-only ONNX Runtime reference trace for BERT token classification.

This is a parity aid for ``BertNEROnnx::debug_trace``. It deliberately runs the
same local graph and tokenizer as anno, then records the stages which can
otherwise be conflated: tokenizer inputs, byte/character offset conventions,
full logits, argmax labels, strict-BIO entities, and anno's decoded entities.

It never calls a model hub. Install its dependencies in an isolated environment
when needed, for example:

  uv run --with onnxruntime --with tokenizers --with numpy \
    scripts/bert_onnx_reference_trace.py --model-dir PATH --text 'Alice met Bob.'

The comparison tolerance is specified before any run: abs <= 1e-5 OR relative
<= 1e-5 for each f32 logit. Exact equality is required for IDs, masks, tokens,
and the reference argmax labels. The strict BIO decoder is intentionally kept
separate from anno's decoder, which has documented first-subword and
name-completion policies.

Do not substitute ``transformers.AutoTokenizer`` for the raw tokenizer here.
On the pinned protectai snapshot, that loader applied hub-facing defaults and
produced different IDs/subwords for ``Paris London`` than the selected
``tokenizer.json``. That is a valid *different reference configuration*, not
evidence of a graph or anno decoding defect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ABS_TOLERANCE = 1e-5
REL_TOLERANCE = 1e-5
MAX_TRACE_TOKENS = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_bio_entities(
    text: str, offsets: list[tuple[int, int]], labels: list[str]
) -> list[dict[str, Any]]:
    """Decode regular BIO tags using Python character offsets.

    This describes the conventional reference policy only. It does not try to
    reproduce anno's intentionally different postprocessing policy.
    """

    entities: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def finish() -> None:
        nonlocal current
        if current is not None:
            start, end = current["start"], current["end"]
            current["text"] = text[start:end]
            entities.append(current)
            current = None

    for (start, end), label in zip(offsets, labels):
        if start == end or label == "O":
            finish()
            continue
        if label.startswith("B-"):
            prefix, entity_type = "B", label[2:]
        elif label.startswith("I-"):
            prefix, entity_type = "I", label[2:]
        else:
            prefix, entity_type = "B", label

        if prefix == "I" and current is not None and current["entity_type"] == entity_type:
            current["end"] = end
        else:
            finish()
            current = {"start": start, "end": end, "entity_type": entity_type}
    finish()
    return entities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, help="write JSON instead of stdout")
    args = parser.parse_args()

    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as error:
        raise SystemExit(
            "install isolated dependencies: onnxruntime, tokenizers, numpy "
            f"({error})"
        ) from error

    graph = args.model_dir / "model.onnx"
    tokenizer_path = args.model_dir / "tokenizer.json"
    config_path = args.model_dir / "config.json"
    missing = [str(path) for path in (graph, tokenizer_path, config_path) if not path.is_file()]
    if missing:
        raise SystemExit(f"model directory is missing required files: {', '.join(missing)}")

    # Keep the reference host deterministic and comparable to anno's CPU
    # default. Do not permit provider fallback to hide a host difference.
    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    # anno loads exactly this raw tokenizer JSON through the Rust tokenizers
    # library. Avoid AutoTokenizer here: it can apply model-hub configuration
    # defaults that are not encoded in this selected artifact.
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # The serialized artifact can impose the model's 512-token truncation.
    # anno disables that rule before its own chunking, and debug_trace rejects
    # multi-window input. Do the same here so a trace can never silently
    # compare only an input prefix.
    tokenizer.no_truncation()
    encoded = tokenizer.encode(args.text, add_special_tokens=True)
    if len(encoded.ids) > MAX_TRACE_TOKENS:
        raise SystemExit(
            "BERT parity trace accepts one window "
            f"(at most {MAX_TRACE_TOKENS} tokens including special tokens), "
            f"got {len(encoded.ids)}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_names = {item.name for item in session.get_inputs()}
    feed: dict[str, Any] = {
        "input_ids": np.asarray([encoded.ids], dtype=np.int64),
        "attention_mask": np.asarray([encoded.attention_mask], dtype=np.int64),
    }
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.asarray(
            [[0 for _ in encoded.ids]], dtype=np.int64
        )
    logits = session.run(["logits"], feed)[0]
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise SystemExit(f"unexpected logits shape: {logits.shape!r}")

    label_ids = logits[0].argmax(axis=-1).tolist()
    id2label = config.get("id2label", {})
    labels = [str(id2label.get(str(label_id), f"LABEL_{label_id}")) for label_id in label_ids]
    offsets = [tuple(map(int, pair)) for pair in encoded.offsets]
    input_ids = [int(value) for value in encoded.ids]
    attention_mask = [int(value) for value in encoded.attention_mask]
    token_type_ids = (
        [0 for _ in input_ids] if "token_type_ids" in input_names else None
    )

    trace = {
        "reference": {
            "host": "onnxruntime-python",
            "providers": session.get_providers(),
            "decoder": "strict_bio",
            "logit_tolerance": {"absolute": ABS_TOLERANCE, "relative": REL_TOLERANCE},
            "offset_unit": "unicode_scalar_index",
        },
        "artifacts": {
            "graph": {"path": str(graph), "sha256": sha256(graph)},
            "tokenizer": {"path": str(tokenizer_path), "sha256": sha256(tokenizer_path)},
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
        },
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "tokens": encoded.tokens,
        "offsets": offsets,
        "logits": logits[0].astype(np.float32).tolist(),
        "labels": labels,
        "strict_bio_entities": strict_bio_entities(args.text, offsets, labels),
    }
    rendered = json.dumps(trace, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
