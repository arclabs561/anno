#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "gliner==0.2.30",
#     "sentencepiece",
#     "transformers>=4.30.0",
#     "onnx>=1.14.0",
#     "optimum[onnxruntime]>=1.16.0",
#     "huggingface-hub>=0.20.0",
# ]
# ///
"""
Export NuNER Zero models to ONNX format.

NuNER models come in two architectures:
  - Fixed-label token classifier: standard BIO tagging
  - NuNER Zero family: label-conditioned word or span scoring

GLiNER-based NuNER checkpoints use an inline label prompt and one encoder.
The GLiNER library exports the matching token or span graph and its tokenizer.

Usage:
    uv run scripts/export_nuner_to_onnx.py --model numind/NuNER_Zero --output /tmp/nuner
    NUNER_MODEL_PATH=/tmp/nuner anno extract --model nuner 'Ada lives in London.'
"""

import argparse
import os
from pathlib import Path


def is_gliner_model(model_id: str) -> bool:
    """Check if a HuggingFace model uses GLiNER architecture."""
    from huggingface_hub import list_repo_files

    return "gliner_config.json" in list_repo_files(model_id)


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
    """Export the trained shared encoder, prompt tokenizer and width metadata."""
    import json
    from gliner import GLiNER
    from huggingface_hub import hf_hub_download

    config_file = hf_hub_download(model_id, filename="gliner_config.json")
    config = json.loads(Path(config_file).read_text())
    if config.get("model_name") == "allenai/longformer-large-4096":
        raise ValueError(
            "NuNER Zero-4k requires a parity-tested Longformer export. "
            "GLiNER 0.2.30 exports a runnable graph with different predictions "
            "from PyTorch on both short and long inputs. Use the upstream PyTorch "
            "model for Zero-4k or onnx-community/NuNER_Zero-span for Anno."
        )
    model = GLiNER.from_pretrained(model_id)
    model.eval()
    model.export_to_onnx(save_dir=output_dir, quantize=quantize)


def main():
    parser = argparse.ArgumentParser(description="Export NuNER Zero model to ONNX")
    parser.add_argument(
        "--model",
        default="numind/NuNER_Zero",
        help="HuggingFace model ID (default: numind/NuNER_Zero)",
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
