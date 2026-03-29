#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "gliner>=0.2.16",
#     "torch>=2.0.0",
#     "onnx>=1.14.0",
#     "onnxruntime>=1.16.0",
#     "transformers>=4.30.0",
# ]
# ///
"""
Export GLiNER-RelEx (joint NER + relation extraction) to ONNX.

GLiNER-RelEx uses scatter-based subword pooling that requires real
tokenized inputs for JIT tracing (random dummy inputs cause index
out-of-bounds). This script:

1. Loads the model and runs a real inference to capture valid inputs
2. Uses the captured inputs for torch.onnx.export with JIT tracing
3. Saves tokenizer and config alongside the ONNX model

Usage:
    uv run scripts/export_gliner_relex_onnx.py
    uv run scripts/export_gliner_relex_onnx.py --model knowledgator/gliner-relex-large-v1.0
    uv run scripts/export_gliner_relex_onnx.py --output /path/to/relex-onnx/
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export GLiNER-RelEx to ONNX")
    parser.add_argument(
        "--model",
        default="knowledgator/gliner-relex-large-v1.0",
        help="HuggingFace model ID",
    )
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from gliner import GLiNER

    model_id = args.model
    if args.output:
        out = Path(args.output)
    else:
        safe_name = model_id.replace("/", "--")
        cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        out = cache_dir / "hub" / f"models--{safe_name}" / "onnx"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_id}...")
    m = GLiNER.from_pretrained(model_id, load_tokenizer=True)
    m.eval()

    # Save tokenizer
    if hasattr(m, "data_processor") and hasattr(m.data_processor, "transformer_tokenizer"):
        m.data_processor.transformer_tokenizer.save_pretrained(str(out))
        print(f"Saved tokenizer to {out}")

    # Save config
    if hasattr(m, "config"):
        config_dict = vars(m.config) if not hasattr(m.config, "to_dict") else m.config.to_dict()
        with open(out / "gliner_config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        print(f"Saved config")

    # Step 1: Capture real model inputs by hooking into forward
    print("\nCapturing model inputs from a real inference...")
    captured = {}
    original_forward = m.model.forward

    def capturing_forward(**kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                captured[k] = v.clone()
        return original_forward(**kwargs)

    m.model.forward = lambda **kw: capturing_forward(**kw)
    text = "Marie Curie worked at the University of Paris and discovered radium."
    labels = ["person", "organization", "location"]
    entities = m.predict_entities(text, labels)
    m.model.forward = original_forward  # restore

    print(f"  Inference found {len(entities[0]) if entities else 0} entities")
    for k, v in captured.items():
        print(f"  {k}: {v.shape}")

    # Step 2: ONNX export with JIT tracing using real inputs
    class RelExWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, words_mask, text_lengths, span_idx, span_mask):
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                words_mask=words_mask,
                text_lengths=text_lengths,
                span_idx=span_idx,
                span_mask=span_mask,
            )
            if isinstance(out, dict):
                return out.get("logits", next(iter(out.values())))
            return out

    wrapper = RelExWrapper(m.model)
    wrapper.eval()

    input_keys = ["input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask"]
    args_tuple = tuple(captured[k] for k in input_keys)

    print(f"\nExporting to ONNX (JIT trace, opset 14)...")
    onnx_path = out / "model.onnx"
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                args_tuple,
                str(onnx_path),
                input_names=input_keys,
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "words_mask": {0: "batch", 1: "seq"},
                    "text_lengths": {0: "batch"},
                    "span_idx": {0: "batch", 1: "num_spans"},
                    "span_mask": {0: "batch", 1: "num_spans"},
                    "logits": {0: "batch"},
                },
                opset_version=14,
                dynamo=False,
            )
        if onnx_path.exists():
            print(f"ONNX model: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
        else:
            print("ERROR: no output file produced")
            return
    except Exception as e:
        print(f"ONNX export failed: {e}")
        # Fallback: save PyTorch weights
        pt_path = out / "model.pt"
        torch.save(m.model.state_dict(), str(pt_path))
        print(f"Saved PyTorch state_dict to {pt_path}")
        return

    # Step 3: Verify
    print("\nVerifying ONNX inference...")
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(str(onnx_path))
    feed = {k: captured[k].numpy() for k in input_keys}
    results = session.run(None, feed)
    print(f"  Output shape: {results[0].shape}")
    print("Export complete.")


if __name__ == "__main__":
    main()
