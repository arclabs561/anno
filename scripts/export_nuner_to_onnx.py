#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "torch>=2.0.0",
#     "transformers>=4.30.0",
#     "onnx>=1.14.0",
#     "optimum[onnxruntime]>=1.16.0",
#     "huggingface-hub>=0.20.0",
# ]
# ///
"""
Export NuNER Zero models to ONNX format.

NuNER models come in two architectures:
  - Token classifier (e.g., deepanwa/NuNerZero_onnx): standard BIO tagging
  - GLiNER-based (e.g., numind/NuNER_Zero-4k): span classification

This script auto-detects the architecture. For GLiNER-based models, it
delegates to export_gliner_poly_onnx.py.

Usage:
    uv run scripts/export_nuner_to_onnx.py --model deepanwa/NuNerZero_onnx
    uv run scripts/export_nuner_to_onnx.py --model numind/NuNER_Zero-4k

For the 4k variant (GLiNER-based), prefer using export_gliner_poly_onnx.py directly:
    uv run scripts/export_gliner_poly_onnx.py --model numind/NuNER_Zero-4k
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def is_gliner_model(model_id: str) -> bool:
    """Check if a HuggingFace model uses GLiNER architecture."""
    from huggingface_hub import list_repo_files

    try:
        files = list_repo_files(model_id)
        return "gliner_config.json" in files
    except Exception:
        return False


def export_token_classifier(model_id: str, output_dir: Path, quantize: bool):
    """Export a standard token classifier to ONNX."""
    from optimum.onnxruntime import ORTModelForTokenClassification
    from transformers import AutoTokenizer

    print(f"Detected: token classifier architecture")
    print(f"Exporting {model_id} to ONNX...")
    print(f"Output: {output_dir}")

    model = ORTModelForTokenClassification.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    onnx_path = output_dir / "model.onnx"
    print(f"ONNX model saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    if quantize:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        print("Quantizing to INT8...")
        quantizer = ORTQuantizer.from_pretrained(output_dir)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
        quantizer.quantize(
            save_dir=output_dir / "quantized", quantization_config=qconfig
        )

    # Verify
    print("\nVerifying export...")
    verify_model = ORTModelForTokenClassification.from_pretrained(output_dir)
    inputs = tokenizer("Marie Curie worked at the Sorbonne.", return_tensors="pt")
    outputs = verify_model(**inputs)
    print(f"Verification passed: output shape {outputs.logits.shape}")


def export_gliner_model(model_id: str, output_dir: Path, quantize: bool):
    """Delegate GLiNER-based models to export_gliner_poly_onnx.py."""
    print(f"Detected: GLiNER architecture (has gliner_config.json)")
    print(f"Delegating to export_gliner_poly_onnx.py...")

    script_dir = Path(__file__).parent
    gliner_script = script_dir / "export_gliner_poly_onnx.py"

    if not gliner_script.exists():
        print(f"ERROR: {gliner_script} not found", file=sys.stderr)
        print(f"For GLiNER-based NuNER models, use export_gliner_poly_onnx.py directly.")
        sys.exit(1)

    cmd = ["uv", "run", str(gliner_script), "--model", model_id, "--output", str(output_dir)]
    if quantize:
        cmd.append("--quantize")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


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

    model_id = args.model

    if args.output:
        output_dir = Path(args.output)
    else:
        safe_name = model_id.replace("/", "--")
        cache_dir = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        )
        output_dir = cache_dir / "hub" / f"models--{safe_name}" / "onnx"

    output_dir.mkdir(parents=True, exist_ok=True)

    if is_gliner_model(model_id):
        export_gliner_model(model_id, output_dir, args.quantize)
    else:
        export_token_classifier(model_id, output_dir, args.quantize)

    print("Export complete.")


if __name__ == "__main__":
    main()
