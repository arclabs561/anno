#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "torch>=2.0.0",
#     "transformers>=4.30.0",
#     "onnx>=1.14.0",
#     "optimum[onnxruntime]>=1.16.0",
# ]
# ///
"""
Export NuNER Zero models to ONNX format.

Supports both the standard NuNER Zero and the 4k-context variant.

Usage:
    uv run scripts/export_nuner_to_onnx.py
    uv run scripts/export_nuner_to_onnx.py --model numind/NuNER_Zero-4k --output ~/.cache/huggingface/hub/models--numind--NuNER_Zero-4k/
    uv run scripts/export_nuner_to_onnx.py --model numind/NuNER_Zero-span

The default exports numind/NuNER_Zero-4k to a directory that hf-hub will find.
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export NuNER Zero model to ONNX")
    parser.add_argument(
        "--model",
        default="numind/NuNER_Zero-4k",
        help="HuggingFace model ID (default: numind/NuNER_Zero-4k)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: HF cache location for the model)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Also export quantized (INT8) version",
    )
    args = parser.parse_args()

    from optimum.onnxruntime import ORTModelForTokenClassification
    from transformers import AutoTokenizer

    model_id = args.model

    if args.output:
        output_dir = Path(args.output)
    else:
        # Place in HF cache so anno's hf_loader finds it
        safe_name = model_id.replace("/", "--")
        cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        output_dir = cache_dir / "hub" / f"models--{safe_name}" / "onnx"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {model_id} to ONNX...")
    print(f"Output: {output_dir}")

    # Export using optimum
    model = ORTModelForTokenClassification.from_pretrained(
        model_id, export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    onnx_path = output_dir / "model.onnx"
    print(f"ONNX model saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    if args.quantize:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        print("Quantizing to INT8...")
        quantizer = ORTQuantizer.from_pretrained(output_dir)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
        quantizer.quantize(save_dir=output_dir / "quantized", quantization_config=qconfig)
        q_path = output_dir / "quantized" / "model_quantized.onnx"
        if q_path.exists():
            print(f"Quantized model: {q_path} ({q_path.stat().st_size / 1e6:.1f} MB)")

    # Verify the export
    print("\nVerifying export...")
    verify_model = ORTModelForTokenClassification.from_pretrained(output_dir)
    inputs = tokenizer("Marie Curie worked at the Sorbonne.", return_tensors="pt")
    outputs = verify_model(**inputs)
    print(f"Verification passed: output shape {outputs.logits.shape}")
    print("Export complete.")


if __name__ == "__main__":
    main()
