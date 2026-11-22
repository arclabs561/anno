# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0",
#     "transformers>=4.30",
#     "onnx>=1.14",
#     "onnxruntime>=1.15",
#     "onnxscript>=0.1",
#     "safetensors>=0.4",
# ]
# ///
"""Export biu-nlp/f-coref to ONNX encoder + safetensors scorer weights.

Loads the f-coref model directly via transformers (bypassing fastcoref's
spacy-dependent high-level API) and exports:

  <output_dir>/
    encoder.onnx          -- DistilRoBERTa encoder
    scorer_weights.safetensors -- Mention/antecedent scorer MLP weights
    tokenizer.json        -- HuggingFace tokenizer
    config.json           -- Model config with coref head params

Usage:
    uv run scripts/export_fcoref.py [--output-dir fcoref_onnx] [--quantize]
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Reproduce the f-coref scorer head architecture (from fastcoref source)
# so we can load the pretrained weights without importing fastcoref.
# ---------------------------------------------------------------------------

class FullyConnectedLayer(nn.Module):
    """Linear + activation + LayerNorm + Dropout.

    Uses named attributes (dense, layer_norm) matching f-coref's checkpoint format.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout_prob: float):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        return self.dropout(self.layer_norm(self.activation(self.dense(x))))


def build_scorer_heads(config) -> dict[str, nn.Module]:
    """Build scorer head modules matching f-coref's architecture."""
    coref_head = config.coref_head if hasattr(config, "coref_head") else {}
    if isinstance(coref_head, dict):
        ffnn_size = coref_head.get("ffnn_size", 1024)
        dropout_prob = coref_head.get("dropout_prob", 0.3)
    else:
        ffnn_size = getattr(coref_head, "ffnn_size", 1024)
        dropout_prob = getattr(coref_head, "dropout_prob", 0.3)

    hidden_size = config.hidden_size  # 768 for DistilRoBERTa

    heads = {
        "start_mention_mlp": FullyConnectedLayer(hidden_size, ffnn_size, dropout_prob),
        "end_mention_mlp": FullyConnectedLayer(hidden_size, ffnn_size, dropout_prob),
        "mention_start_classifier": nn.Linear(ffnn_size, 1),
        "mention_end_classifier": nn.Linear(ffnn_size, 1),
        "mention_s2e_classifier": nn.Linear(ffnn_size, ffnn_size),
        "start_coref_mlp": FullyConnectedLayer(hidden_size, ffnn_size, dropout_prob),
        "end_coref_mlp": FullyConnectedLayer(hidden_size, ffnn_size, dropout_prob),
        "antecedent_s2s_classifier": nn.Linear(ffnn_size, ffnn_size),
        "antecedent_e2e_classifier": nn.Linear(ffnn_size, ffnn_size),
        "antecedent_s2e_classifier": nn.Linear(ffnn_size, ffnn_size),
        "antecedent_e2s_classifier": nn.Linear(ffnn_size, ffnn_size),
    }
    return heads


def load_fcoref_model(model_id: str = "biu-nlp/f-coref", revision: str | None = None):
    """Load f-coref model weights directly via transformers."""
    from huggingface_hub import hf_hub_download

    config = AutoConfig.from_pretrained(model_id, revision=revision)

    # Load the full state dict from the HF checkpoint
    weights_path = hf_hub_download(model_id, "pytorch_model.bin", revision=revision)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Separate encoder vs scorer head weights
    encoder_state = {}
    scorer_state = {}

    # The f-coref model prefixes encoder weights with "roberta." or "base_model."
    scorer_prefixes = (
        "start_mention_mlp", "end_mention_mlp",
        "mention_start_classifier", "mention_end_classifier", "mention_s2e_classifier",
        "start_coref_mlp", "end_coref_mlp",
        "antecedent_s2s_classifier", "antecedent_e2e_classifier",
        "antecedent_s2e_classifier", "antecedent_e2s_classifier",
    )

    for key, value in state_dict.items():
        is_scorer = False
        for prefix in scorer_prefixes:
            if key.startswith(prefix):
                scorer_state[key] = value
                is_scorer = True
                break
        if not is_scorer:
            encoder_state[key] = value

    # Load encoder
    # Try loading the base DistilRoBERTa model
    encoder = AutoModel.from_pretrained(model_id, config=config, revision=revision)

    # Build and load scorer heads
    heads = build_scorer_heads(config)
    for head_name, module in heads.items():
        # Collect this head's weights from scorer_state
        head_state = {}
        for key, value in scorer_state.items():
            if key.startswith(head_name + "."):
                # Strip the head_name prefix
                sub_key = key[len(head_name) + 1:]
                # Map FullyConnectedLayer's Sequential indices to our structure
                # f-coref uses: fc.0.weight (Linear), fc.2.weight (LayerNorm)
                # Our structure: fc.0 (Linear), fc.1 (GELU), fc.2 (LayerNorm), fc.3 (Dropout)
                head_state[sub_key] = value
            elif key == head_name + ".weight":
                head_state["weight"] = value
            elif key == head_name + ".bias":
                head_state["bias"] = value
        if head_state:
            module.load_state_dict(head_state, strict=False)

    return encoder, heads, config


def export_encoder(encoder, output_dir: Path, opset: int = 14):
    """Export the DistilRoBERTa encoder to ONNX."""
    encoder.eval()

    dummy_input_ids = torch.ones(1, 128, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, 128, dtype=torch.long)

    encoder_path = output_dir / "encoder.onnx"

    # Wrap to only return last_hidden_state
    class EncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state

    wrapper = EncoderWrapper(encoder)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        str(encoder_path),
        opset_version=opset,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "last_hidden_state": {0: "batch", 1: "seq_len"},
        },
    )

    # Newer torch may export with external data files (.data); reload all tensors
    # into memory and re-save as a single self-contained protobuf.
    onnx_model = onnx.load(str(encoder_path), load_external_data=True)
    # Clear external_data references so onnx.save writes everything inline
    for init in onnx_model.graph.initializer:
        del init.external_data[:]
        init.data_location = 0  # DEFAULT = inline
    onnx.save(onnx_model, str(encoder_path))
    # Clean up any leftover .data files from the initial export
    for f in encoder_path.parent.glob("*.data"):
        f.unlink()

    onnx.checker.check_model(onnx_model)
    print(f"  encoder.onnx: {encoder_path.stat().st_size / 1024 / 1024:.1f} MB")
    return encoder_path


def export_scorer_weights(heads: dict, output_dir: Path):
    """Export all scorer head weights to safetensors."""
    tensors = OrderedDict()

    for head_name, module in heads.items():
        for name, param in module.named_parameters():
            tensors[f"{head_name}.{name}"] = param.data.clone()

    weights_path = output_dir / "scorer_weights.safetensors"
    save_file(tensors, str(weights_path))
    print(f"  scorer_weights.safetensors: {weights_path.stat().st_size / 1024:.1f} KB")
    print(f"  tensors ({len(tensors)}): {list(tensors.keys())}")
    return weights_path


def export_tokenizer(output_dir: Path, model_id: str = "biu-nlp/f-coref", revision: str | None = None):
    """Save the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.save_pretrained(str(output_dir))
    # Keep only essential files
    keep = {"tokenizer.json", "encoder.onnx", "scorer_weights.safetensors",
            "config.json", "encoder_quantized.onnx"}
    for f in output_dir.iterdir():
        if f.name not in keep:
            f.unlink()
    print("  tokenizer.json saved")


def export_config(config, output_dir: Path):
    """Save model config with coref head params."""
    config_dict = config.to_dict()
    coref_head = config_dict.get("coref_head", {})
    export_cfg = {
        "model_type": "fcoref",
        "encoder": "distilroberta-base",
        "hidden_size": config_dict.get("hidden_size", 768),
        "coref_head": {
            "ffnn_size": coref_head.get("ffnn_size", 1024),
            "max_span_length": coref_head.get("max_span_length", 30),
            "top_lambda": coref_head.get("top_lambda", 0.25),
            "max_segment_len": coref_head.get("max_segment_len", 512),
        },
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(export_cfg, f, indent=2)
    print("  config.json saved")
    return config_path


def quantize_onnx(encoder_path: Path):
    """Optionally quantize the encoder to INT8."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized_path = encoder_path.parent / "encoder_quantized.onnx"
        quantize_dynamic(
            str(encoder_path),
            str(quantized_path),
            weight_type=QuantType.QInt8,
        )
        print(f"  encoder_quantized.onnx: {quantized_path.stat().st_size / 1024 / 1024:.1f} MB")
        return quantized_path
    except ImportError:
        print("  quantization skipped (onnxruntime.quantization not available)")
        return None


def verify_export(output_dir: Path):
    """Verify the exported model produces valid output."""
    import numpy as np
    import onnxruntime as ort
    from safetensors import safe_open

    session = ort.InferenceSession(str(output_dir / "encoder.onnx"))
    dummy_ids = np.ones((1, 10), dtype=np.int64)
    dummy_mask = np.ones((1, 10), dtype=np.int64)
    outputs = session.run(None, {"input_ids": dummy_ids, "attention_mask": dummy_mask})
    hidden = outputs[0]
    assert hidden.shape == (1, 10, 768), f"unexpected shape: {hidden.shape}"
    print(f"  encoder verification: OK (output shape {hidden.shape})")

    with safe_open(str(output_dir / "scorer_weights.safetensors"), framework="pt") as f:
        keys = list(f.keys())
        assert len(keys) > 0, "no tensors in scorer_weights"
        print(f"  scorer weights verification: OK ({len(keys)} tensors)")

    with open(output_dir / "config.json") as f:
        cfg = json.load(f)
        assert cfg["coref_head"]["ffnn_size"] == 1024
        print("  config verification: OK")


def main():
    parser = argparse.ArgumentParser(description="Export f-coref to ONNX + safetensors")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("fcoref_onnx"),
        help="Output directory (default: fcoref_onnx)",
    )
    parser.add_argument("--revision", help="Hugging Face revision for the model, config, and tokenizer")
    parser.add_argument("--quantize", action="store_true", help="Also produce INT8 quantized encoder")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version (default: 14)")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading f-coref model (via transformers, no spacy)...")
    encoder, heads, config = load_fcoref_model(revision=args.revision)

    print("Exporting encoder to ONNX...")
    encoder_path = export_encoder(encoder, output_dir, opset=args.opset)

    print("Exporting scorer weights to safetensors...")
    export_scorer_weights(heads, output_dir)

    print("Saving tokenizer...")
    export_tokenizer(output_dir, revision=args.revision)

    print("Saving config...")
    export_config(config, output_dir)

    if args.quantize:
        print("Quantizing encoder...")
        quantize_onnx(encoder_path)

    print("Verifying export...")
    verify_export(output_dir)

    print(f"\nExport complete: {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
