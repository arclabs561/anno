#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "optimum[onnxruntime]>=1.16.0",
#     "transformers==4.47.0",
#     "sentencepiece>=0.1.99",
#     "protobuf>=3.20.0",
#     "onnxruntime>=1.16.0",
# ]
# ///
"""
Export DeBERTa-v3 NER model to ONNX format.

Uses optimum for the ONNX export (handles DeBERTa's disentangled
attention correctly). Pins transformers to 4.47.0 to avoid the
tekken.json tokenizer bug in newer versions.

The tokenizer is copied from HF cache (spm.model + tokenizer_config.json)
because DeBERTa-v2's fast tokenizer conversion is broken in recent
transformers.

Usage:
    uv run scripts/export_deberta_ner_to_onnx.py
    uv run scripts/export_deberta_ner_to_onnx.py --model microsoft/deberta-v3-large
    uv run scripts/export_deberta_ner_to_onnx.py --output /path/to/deberta-ner/

Then:
    DEBERTA_MODEL_PATH=/path/to/deberta-ner anno extract --model deberta-v3 'Your text'
"""

import argparse
import os
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export DeBERTa-v3 NER to ONNX")
    parser.add_argument(
        "--model",
        default="microsoft/deberta-v3-base",
        help="HuggingFace model ID (default: microsoft/deberta-v3-base)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: /tmp/deberta-ner-onnx)",
    )
    args = parser.parse_args()

    from optimum.onnxruntime import ORTModelForTokenClassification
    from huggingface_hub import hf_hub_download

    model_id = args.model
    out = Path(args.output) if args.output else Path("/tmp/deberta-ner-onnx")
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Export ONNX model via optimum
    print(f"Exporting {model_id} to ONNX...")
    model = ORTModelForTokenClassification.from_pretrained(model_id, export=True)
    model.save_pretrained(str(out))

    onnx_path = out / "model.onnx"
    if onnx_path.exists():
        print(f"ONNX model: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Step 2: Copy tokenizer files from HF cache
    # DeBERTa-v2's fast tokenizer conversion is broken in transformers.
    # We copy the raw SentencePiece model and config directly.
    print("Fetching tokenizer files from HF cache...")
    for fname in ["spm.model", "tokenizer_config.json", "special_tokens_map.json",
                   "tokenizer.json", "added_tokens.json"]:
        try:
            src = hf_hub_download(model_id, fname)
            shutil.copy(src, out / fname)
            print(f"  {fname}")
        except Exception:
            pass  # Not all files exist for all models

    # Step 3: Verify ONNX inference
    print("\nVerifying ONNX inference...")
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(str(onnx_path))
    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    print(f"  Inputs: {input_names}")
    print(f"  Outputs: {output_names}")

    dummy = {k: np.ones((1, 16), dtype=np.int64) for k in input_names}
    results = session.run(None, dummy)
    print(f"  Output shape: {results[0].shape}")

    print(f"\nExport complete. Output: {out}")
    print(f"\nTo use with anno:")
    print(f"  DEBERTA_MODEL_PATH={out} anno extract --model deberta-v3 'Your text'")


if __name__ == "__main__":
    main()
