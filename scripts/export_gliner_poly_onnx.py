#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gliner>=0.2.16",
#     "torch>=2.0.0",
#     "onnx>=1.14.0",
#     "onnxruntime>=1.16.0",
#     "transformers>=4.30.0",
#     "numpy>=1.21.0",
#     "onnxscript>=0.1.0",
# ]
# ///
"""
Export GLiNER poly-encoder model to ONNX format for use in the anno Rust crate.

GLiNER poly-encoder models use a bi-encoder architecture with post-fusion:
  - Text encoder: DeBERTa-v3 (processes the input text)
  - Label encoder: BGE-small-en (encodes entity type labels)
  - Post-fusion: cross-attention between text and label representations

This creates two ONNX files:
  1. model.onnx         -- main span model (text encoder + fusion + span scorer)
  2. label_encoder.onnx  -- label encoder (sentence-transformer for entity labels)

The split allows the Rust consumer to pre-compute label embeddings once and reuse
them across many texts (the main inference advantage of the poly-encoder design).

Usage:
    uv run scripts/export_gliner_poly_onnx.py
    uv run scripts/export_gliner_poly_onnx.py --model knowledgator/gliner-bi-large-v1.0
    uv run scripts/export_gliner_poly_onnx.py --output ~/.cache/anno/models/gliner-poly/
    uv run scripts/export_gliner_poly_onnx.py --quantize

Compatible models (bi-encoder / poly-encoder with label encoder):
    knowledgator/gliner-bi-large-v2.0       (v2, 61.5% CrossNER zero-shot SOTA)
    knowledgator/gliner-bi-base-v2.0        (v2, smaller/faster variant)
    knowledgator/gliner-bi-large-v1.0       (DeBERTa-v3-large + BGE-small-en)
    knowledgator/gliner-bi-small-v1.0       (DeBERTa-v3-small + BGE-small-en)
    knowledgator/modern-gliner-bi-large-v1.0 (ModernBERT-large + BGE-base-en)
    knowledgator/modern-gliner-bi-base-v1.0  (ModernBERT-base + BGE-base-en)

Note: knowledgator/gliner-poly-{base,small}-v1.0 are model cards only (no weights).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Heavy imports (torch, onnx, numpy, gliner) are deferred to function bodies
# so that `python script.py --help` works without them installed.
# When run via `uv run`, the PEP 723 dependencies are installed first.

DEFAULT_MODEL = "knowledgator/gliner-bi-large-v2.0"

DEFAULT_ENTITY_TYPES = [
    "person",
    "organization",
    "location",
    "date",
    "time",
    "money",
    "percent",
    "product",
    "event",
    "facility",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """Print a timestamped log message to stderr."""
    import datetime

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def check_onnx_model(path: str | Path) -> None:
    """Load and verify an ONNX model, printing input/output specs."""
    import onnx

    path = Path(path)
    size_mb = path.stat().st_size / (1024 * 1024)

    # Large models (>2GB) can't be serialized for full check due to protobuf limits.
    if size_mb > 1900:
        log(f"  Model is {size_mb:.0f} MB (>2GB protobuf limit), using path-based check.")
        onnx.checker.check_model(str(path))
    else:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)

    # Load with external data support for input/output inspection.
    model = onnx.load(str(path), load_external_data=False)
    print(f"  ONNX check passed: {path}")
    print("  Inputs:")
    for inp in model.graph.input:
        shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        dtype = inp.type.tensor_type.elem_type
        print(f"    {inp.name}: shape={shape} dtype={dtype}")
    print("  Outputs:")
    for out in model.graph.output:
        shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        dtype = out.type.tensor_type.elem_type
        print(f"    {out.name}: shape={shape} dtype={dtype}")


def verify_with_onnxruntime(onnx_path: str | Path, input_dict: dict) -> list:
    """Run a quick inference with ONNX Runtime to verify the export works."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    input_names = [inp.name for inp in sess.get_inputs()]

    # Filter to only the inputs the model expects
    feed = {k: v for k, v in input_dict.items() if k in input_names}

    missing = set(input_names) - set(feed.keys())
    if missing:
        raise RuntimeError(
            f"ONNX model expects inputs {missing} not provided. "
            f"Available: {list(feed.keys())}"
        )

    return sess.run(None, feed)


def quantize_model(onnx_path: Path, output_path: Path) -> None:
    """Apply INT8 dynamic quantization to an ONNX model."""
    log(f"Quantizing {onnx_path.name}...")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            str(onnx_path),
            str(output_path),
            weight_type=QuantType.QUInt8,
        )
        log(f"Quantized model saved to {output_path}")

        orig_size = onnx_path.stat().st_size / (1024 * 1024)
        quant_size = output_path.stat().st_size / (1024 * 1024)
        log(
            f"  Original: {orig_size:.1f} MB, Quantized: {quant_size:.1f} MB "
            f"({quant_size / orig_size * 100:.0f}%)"
        )
    except Exception as e:
        log(f"Quantization failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Strategy 1: use GLiNER library's built-in export
# ---------------------------------------------------------------------------

def try_library_export(model_id: str, output_dir: Path, quantize: bool) -> bool:
    """
    Attempt export using GLiNER's built-in export_to_onnx method.

    Returns True on success, False on failure (caller should fall back).
    """
    log(f"Strategy 1: trying GLiNER library export_to_onnx for '{model_id}'...")
    try:
        from gliner import GLiNER

        model = GLiNER.from_pretrained(model_id, load_tokenizer=True)

        result = model.export_to_onnx(
            save_dir=str(output_dir),
            onnx_filename="model.onnx",
            quantized_filename="model_quantized.onnx",
            quantize=quantize,
            opset=17,
        )

        onnx_path = output_dir / "model.onnx"
        if onnx_path.exists():
            log("Strategy 1 succeeded.")
            check_onnx_model(onnx_path)
            return True
        else:
            log(f"Strategy 1: export_to_onnx returned {result} but model.onnx not found.")
            return False

    except Exception as e:
        log(f"Strategy 1 failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Strategy 2: manual torch.onnx.export with correct bi-encoder inputs
# ---------------------------------------------------------------------------

def _build_wrapper_class():
    """
    Build the GLiNERPolyONNXWrapper class at call time (requires torch).

    Returned as a class object so callers can instantiate it.
    """
    import torch
    import torch.nn as nn

    class GLiNERPolyONNXWrapper(nn.Module):
        """
        Wrapper that adapts a GLiNER poly-encoder model for torch.onnx.export.

        Two modes:
          "main"          -- full span model with pre-computed label embeddings
          "label_encoder" -- label encoder only (sentence-transformer)
        """

        def __init__(self, gliner_model, mode: str = "main"):
            super().__init__()
            self.gliner_model = gliner_model
            self.model = gliner_model.model
            self.mode = mode

        def forward(self, *args):
            if self.mode == "main":
                return self._forward_main(*args)
            elif self.mode == "label_encoder":
                return self._forward_label_encoder(*args)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        def _forward_main(
            self,
            input_ids,
            attention_mask,
            words_mask,
            text_lengths,
            span_idx,
            span_mask,
            labels_embeddings,
        ):
            """Forward pass with pre-computed label embeddings."""
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                words_mask=words_mask,
                text_lengths=text_lengths,
                span_idx=span_idx,
                span_mask=span_mask,
                labels_embeddings=labels_embeddings,
            )
            if hasattr(output, "logits"):
                return output.logits
            elif isinstance(output, dict):
                return output.get(
                    "logits",
                    output.get("output", next(iter(output.values()))),
                )
            elif isinstance(output, (tuple, list)):
                return output[0]
            return output

        def _forward_label_encoder(self, labels_input_ids, labels_attention_mask):
            """Forward pass for the label encoder only."""
            # Find the label encoder sub-module
            if hasattr(self.model, "label_encoder"):
                label_encoder = self.model.label_encoder
            elif hasattr(self.model, "entity_encoder"):
                label_encoder = self.model.entity_encoder
            else:
                raise AttributeError(
                    "Could not find label/entity encoder on the model. "
                    f"Available attrs: {[a for a in dir(self.model) if not a.startswith('_')]}"
                )

            outputs = label_encoder(
                input_ids=labels_input_ids,
                attention_mask=labels_attention_mask,
            )

            # Extract embeddings (mean-pool if we get hidden states)
            if hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state
                mask = labels_attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                return pooled
            elif isinstance(outputs, torch.Tensor):
                return outputs
            elif isinstance(outputs, dict):
                return outputs.get(
                    "sentence_embedding",
                    outputs.get("pooler_output", next(iter(outputs.values()))),
                )
            return outputs[0]

    return GLiNERPolyONNXWrapper


def make_dummy_inputs(gliner_model, entity_types: list[str], max_width: int = 12) -> dict:
    """
    Create dummy inputs by running the model's own tokenization pipeline.

    Falls back to manual construction if the data_processor API is unavailable.
    """
    import torch

    text = "John Smith works at Google in New York City since January 2020."
    text_words = text.split()
    num_words = len(text_words)

    # Try the model's data processor first
    try:
        raw_batch = [{"tokenized_text": text_words, "ner": []}]

        if hasattr(gliner_model, "data_processor"):
            processor = gliner_model.data_processor
            batch = processor.collate_fn(raw_batch, entity_types)
        elif hasattr(gliner_model, "prepare_model_inputs"):
            batch = gliner_model.prepare_model_inputs(raw_batch, entity_types)
        else:
            raise AttributeError("No data_processor or prepare_model_inputs")

        inputs = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v
            elif isinstance(v, list):
                inputs[k] = torch.tensor(v)
        return inputs

    except Exception as e:
        log(f"Data processor failed ({e}), falling back to manual tokenization.")
        return _make_dummy_inputs_manual(gliner_model, text_words, entity_types, max_width)


def _make_dummy_inputs_manual(
    gliner_model,
    text_words: list[str],
    entity_types: list[str],
    max_width: int = 12,
) -> dict:
    """
    Manually construct dummy inputs following the GLiNER prompt format.

    Mirrors the tokenization logic in anno's gliner_onnx/inference.rs.
    """
    import torch

    # Find the tokenizer
    hf_tokenizer = None
    if hasattr(gliner_model, "data_processor"):
        dp = gliner_model.data_processor
        hf_tokenizer = getattr(dp, "transformer_tokenizer", None) or getattr(dp, "tokenizer", None)
    if hf_tokenizer is None:
        hf_tokenizer = getattr(gliner_model, "tokenizer", None)
    if hf_tokenizer is None:
        raise RuntimeError("Cannot access the model's tokenizer for manual input construction.")

    num_words = len(text_words)

    # Build prompt: [CLS] <<ENT>> type1 <<ENT>> type2 ... <<SEP>> word1 word2 ... [SEP]
    entity_prompt = " ".join(f"<<ENT>> {et}" for et in entity_types)
    full_text = " ".join(text_words)
    prompt = f"{entity_prompt} <<SEP>> {full_text}"

    encoding = hf_tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = encoding["input_ids"]  # [1, seq_len]
    attention_mask = encoding["attention_mask"]  # [1, seq_len]
    seq_len = input_ids.shape[1]

    # Approximate words_mask (zeros -- the model infers word boundaries internally)
    words_mask = torch.zeros(1, seq_len, dtype=torch.long)
    text_lengths = torch.tensor([[num_words]], dtype=torch.long)

    # Span tensors
    num_spans = num_words * max_width
    span_idx = torch.zeros(1, num_spans, 2, dtype=torch.long)
    span_mask = torch.zeros(1, num_spans, dtype=torch.bool)

    for start in range(num_words):
        for width in range(min(max_width, num_words - start)):
            dim = start * max_width + width
            if dim < num_spans:
                span_idx[0, dim, 0] = start
                span_idx[0, dim, 1] = start + width
                span_mask[0, dim] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "words_mask": words_mask,
        "text_lengths": text_lengths,
        "span_idx": span_idx,
        "span_mask": span_mask,
    }


def compute_label_embeddings(gliner_model, entity_types: list[str]):
    """Compute label embeddings via the model's encode_labels method."""
    import numpy as np
    import torch

    log(f"Computing label embeddings for {len(entity_types)} entity types...")

    with torch.no_grad():
        if hasattr(gliner_model, "encode_labels"):
            embeddings = gliner_model.encode_labels(entity_types, batch_size=len(entity_types))
            if isinstance(embeddings, torch.Tensor):
                return embeddings
            elif isinstance(embeddings, np.ndarray):
                return torch.from_numpy(embeddings)
            return torch.tensor(embeddings)
        raise AttributeError(
            "Model lacks encode_labels -- it may not be a bi/poly-encoder variant."
        )


def export_main_model(gliner_model, output_dir: Path, dummy_inputs: dict, labels_embeddings, opset: int = 17) -> Path:
    """Export the main span model (text encoder + fusion + scorer) to ONNX."""
    import torch

    log("Exporting main model to ONNX...")

    WrapperCls = _build_wrapper_class()
    wrapper = WrapperCls(gliner_model, mode="main")
    wrapper.eval()

    onnx_path = output_dir / "model.onnx"

    all_inputs = (
        dummy_inputs["input_ids"],
        dummy_inputs["attention_mask"],
        dummy_inputs["words_mask"],
        dummy_inputs["text_lengths"],
        dummy_inputs["span_idx"],
        dummy_inputs["span_mask"],
        labels_embeddings,
    )

    input_names = [
        "input_ids",
        "attention_mask",
        "words_mask",
        "text_lengths",
        "span_idx",
        "span_mask",
        "labels_embeddings",
    ]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "words_mask": {0: "batch_size", 1: "sequence"},
        "text_lengths": {0: "batch_size"},
        "span_idx": {0: "batch_size", 1: "num_spans"},
        "span_mask": {0: "batch_size", 1: "num_spans"},
        "labels_embeddings": {0: "num_labels"},
        "logits": {0: "batch_size"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            all_inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    log(f"Main model exported to {onnx_path}")
    check_onnx_model(onnx_path)
    return onnx_path


def export_label_encoder(gliner_model, output_dir: Path, opset: int = 17) -> Path:
    """Export the label encoder (sentence-transformer) to a separate ONNX file."""
    import torch

    log("Exporting label encoder to ONNX...")

    WrapperCls = _build_wrapper_class()
    wrapper = WrapperCls(gliner_model, mode="label_encoder")
    wrapper.eval()

    onnx_path = output_dir / "label_encoder.onnx"

    dummy_labels = ["person", "organization", "location"]

    # Find the label tokenizer
    label_tokenizer = None
    if hasattr(gliner_model, "data_processor"):
        label_tokenizer = getattr(gliner_model.data_processor, "label_tokenizer", None)
    if label_tokenizer is None:
        label_tokenizer = getattr(gliner_model, "label_tokenizer", None)
    if label_tokenizer is None and hasattr(gliner_model, "config"):
        from transformers import AutoTokenizer

        cfg = gliner_model.config
        label_model_name = getattr(cfg, "labels_encoder", None) or getattr(cfg, "entity_encoder", None)
        if label_model_name:
            label_tokenizer = AutoTokenizer.from_pretrained(label_model_name)
    if label_tokenizer is None:
        raise RuntimeError("Cannot find label tokenizer on the model.")

    encoding = label_tokenizer(
        dummy_labels,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )

    labels_input_ids = encoding["input_ids"]
    labels_attention_mask = encoding["attention_mask"]

    input_names = ["labels_input_ids", "labels_attention_mask"]
    dynamic_axes = {
        "labels_input_ids": {0: "num_labels", 1: "label_sequence"},
        "labels_attention_mask": {0: "num_labels", 1: "label_sequence"},
        "labels_embeddings": {0: "num_labels"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (labels_input_ids, labels_attention_mask),
            str(onnx_path),
            input_names=input_names,
            output_names=["labels_embeddings"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    log(f"Label encoder exported to {onnx_path}")
    check_onnx_model(onnx_path)
    return onnx_path


def save_tokenizer(gliner_model, output_dir: Path) -> None:
    """Save the tokenizer to output_dir/tokenizer.json."""
    log("Saving tokenizer...")

    saved = False

    if hasattr(gliner_model, "data_processor"):
        dp = gliner_model.data_processor
        tok = getattr(dp, "transformer_tokenizer", None) or getattr(dp, "tokenizer", None)
        if tok is not None:
            if hasattr(tok, "save_pretrained"):
                tok.save_pretrained(str(output_dir))
                saved = True
            elif hasattr(tok, "save"):
                tok.save(str(output_dir / "tokenizer.json"))
                saved = True

    if not saved and hasattr(gliner_model, "tokenizer"):
        tok = gliner_model.tokenizer
        if hasattr(tok, "save_pretrained"):
            tok.save_pretrained(str(output_dir))
            saved = True
        elif hasattr(tok, "save"):
            tok.save(str(output_dir / "tokenizer.json"))
            saved = True

    tj = output_dir / "tokenizer.json"
    if saved and tj.exists():
        log(f"Tokenizer saved to {tj}")
    elif saved:
        log(f"Warning: tokenizer saved but tokenizer.json not found in {output_dir}")
        for f in sorted(output_dir.iterdir()):
            log(f"  {f.name}")
    else:
        log("Warning: could not save tokenizer. Download it manually from HuggingFace.")


def save_metadata(
    gliner_model,
    model_id: str,
    output_dir: Path,
    entity_types: list[str],
    label_embed_shape: tuple,
) -> None:
    """Save model metadata to gliner_config.json."""
    log("Saving model metadata...")

    metadata: dict = {
        "model_id": model_id,
        "architecture": "poly-encoder",
        "max_width": 12,
        "label_embed_dim": int(label_embed_shape[-1]) if len(label_embed_shape) > 1 else None,
        "default_entity_types": entity_types,
        "onnx_files": {
            "main_model": "model.onnx",
            "label_encoder": "label_encoder.onnx",
        },
        "inputs": {
            "main_model": [
                "input_ids",
                "attention_mask",
                "words_mask",
                "text_lengths",
                "span_idx",
                "span_mask",
                "labels_embeddings",
            ],
            "label_encoder": [
                "labels_input_ids",
                "labels_attention_mask",
            ],
        },
        "outputs": {
            "main_model": ["logits"],
            "label_encoder": ["labels_embeddings"],
        },
    }

    if hasattr(gliner_model, "config"):
        cfg = gliner_model.config
        for attr in [
            "max_width",
            "max_length",
            "labels_encoder",
            "class_token_index",
            "vocab_size",
            "span_mode",
            "model_type",
        ]:
            val = getattr(cfg, attr, None)
            if val is not None:
                metadata[attr] = val

    config_path = output_dir / "gliner_config.json"
    with open(config_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"Config saved to {config_path}")


def try_manual_export(
    model_id: str,
    output_dir: Path,
    quantize: bool,
    entity_types: list[str],
    opset: int = 17,
) -> bool:
    """
    Strategy 2: manual torch.onnx.export with correct bi-encoder inputs.

    Exports two ONNX files:
      - model.onnx: main span model (accepts pre-computed label embeddings)
      - label_encoder.onnx: label encoder (sentence-transformer)

    Returns True on success.
    """
    import numpy as np
    import torch

    log(f"Strategy 2: manual export for '{model_id}'...")

    from gliner import GLiNER

    log("Loading model...")
    model = GLiNER.from_pretrained(model_id, load_tokenizer=True)
    model.eval()

    # 1. Compute label embeddings (needed as dummy input for main model export)
    labels_embeddings = compute_label_embeddings(model, entity_types)
    log(f"Label embeddings shape: {labels_embeddings.shape}")

    # 2. Prepare dummy inputs for the main model
    dummy_inputs = make_dummy_inputs(model, entity_types)
    log(f"Dummy inputs prepared: {list(dummy_inputs.keys())}")
    for k, v in dummy_inputs.items():
        if isinstance(v, torch.Tensor):
            log(f"  {k}: shape={v.shape} dtype={v.dtype}")

    # 3. Export the main model
    main_onnx = export_main_model(model, output_dir, dummy_inputs, labels_embeddings, opset=opset)

    # 4. Export the label encoder
    label_onnx = export_label_encoder(model, output_dir, opset=opset)

    # 5. Save tokenizer.json
    save_tokenizer(model, output_dir)

    # 6. Save model metadata
    save_metadata(model, model_id, output_dir, entity_types, labels_embeddings.shape)

    # 7. Verify with ONNX Runtime
    log("Verifying exports with ONNX Runtime...")
    try:
        # Verify label encoder
        label_input_np = {
            "labels_input_ids": np.array([[1, 2, 3, 0]], dtype=np.int64),
            "labels_attention_mask": np.array([[1, 1, 1, 0]], dtype=np.int64),
        }
        label_outputs = verify_with_onnxruntime(label_onnx, label_input_np)
        log(f"Label encoder output shape: {label_outputs[0].shape}")

        # Verify main model
        main_input_np = {}
        for k, v in dummy_inputs.items():
            if isinstance(v, torch.Tensor):
                main_input_np[k] = v.numpy()
        main_input_np["labels_embeddings"] = labels_embeddings.numpy()
        main_outputs = verify_with_onnxruntime(main_onnx, main_input_np)
        log(f"Main model output shape: {main_outputs[0].shape}")
    except Exception as e:
        log(f"ONNX Runtime verification failed (non-fatal): {e}")
        traceback.print_exc(file=sys.stderr)

    # 8. Optional quantization
    if quantize:
        quantize_model(main_onnx, output_dir / "model_quantized.onnx")
        quantize_model(label_onnx, output_dir / "label_encoder_quantized.onnx")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export GLiNER poly-encoder model to ONNX for the anno Rust crate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        default=os.path.expanduser("~/.cache/anno/models/gliner-poly"),
        help="Output directory (default: ~/.cache/anno/models/gliner-poly)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 dynamic quantization",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--entity-types",
        nargs="+",
        default=DEFAULT_ENTITY_TYPES,
        help="Entity types to use as dummy labels for export",
    )
    parser.add_argument(
        "--strategy",
        choices=["auto", "library", "manual"],
        default="auto",
        help=(
            "Export strategy: 'library' uses GLiNER's built-in export, "
            "'manual' uses torch.onnx.export with correct bi-encoder inputs, "
            "'auto' tries library first then falls back to manual (default: auto)"
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Model:  {args.model}")
    log(f"Output: {output_dir}")
    log(f"Opset:  {args.opset}")
    log(f"Labels: {args.entity_types}")

    success = False

    if args.strategy in ("auto", "library"):
        success = try_library_export(args.model, output_dir, args.quantize)

    if not success and args.strategy in ("auto", "manual"):
        try:
            success = try_manual_export(
                args.model,
                output_dir,
                args.quantize,
                args.entity_types,
                opset=args.opset,
            )
        except Exception as e:
            log(f"Strategy 2 (manual) failed: {e}")
            traceback.print_exc(file=sys.stderr)
            success = False

    if success:
        print("\n" + "=" * 60)
        print("Export complete.")
        print(f"Output directory: {output_dir}")
        print()
        print("Files:")
        for f in sorted(output_dir.iterdir()):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name:40s} {size_mb:8.1f} MB")
            else:
                print(f"  {f.name}/")
        print()
        print("Usage with anno (once GLiNERPoly backend is implemented):")
        print(f"  GLINER_POLY_MODEL_PATH={output_dir} anno extract --model gliner-poly 'Your text'")
        print("=" * 60)
    else:
        print("\nExport failed. See errors above.", file=sys.stderr)
        print("\nKnown issues:", file=sys.stderr)
        print(
            "  - GLiNER library's export_to_onnx has bugs with bi/poly-encoder models",
            file=sys.stderr,
        )
        print(
            "    (see https://github.com/urchade/GLiNER/issues/225)",
            file=sys.stderr,
        )
        print(
            "  - The manual export requires correct model introspection which may",
            file=sys.stderr,
        )
        print("    fail if the GLiNER library API has changed.", file=sys.stderr)
        print(
            f"\nTry: uv run {__file__} --strategy manual --model {args.model}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
