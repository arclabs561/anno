#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
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
Export GLiNER-RelEx (joint NER + relation extraction) to ONNX.

GLiNER-RelEx extends GLiNER with relation scoring between entity span
pairs. It produces two outputs:
  1. Entity span scores (same as standard GLiNER)
  2. Relation scores between span pairs

Usage:
    uv run scripts/export_gliner_relex_onnx.py
    uv run scripts/export_gliner_relex_onnx.py --model knowledgator/gliner-relex-large-v1.0
    uv run scripts/export_gliner_relex_onnx.py --quantize

The model requires the `gliner` library for loading and the export
produces model.onnx + tokenizer files in the HuggingFace cache.
"""

import argparse
import os
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class GLiNERRelExWrapper(nn.Module):
    """Wraps a GLiNER-RelEx model for ONNX export with dual outputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids,
        attention_mask,
        words_mask,
        text_lengths,
        span_idx,
        span_mask,
    ):
        """Forward pass producing both entity and relation scores.

        Returns:
            entity_scores: [batch, num_spans, num_entity_types]
            relation_scores: [batch, num_spans, num_spans, num_relation_types]
        """
        # Get token representations from the encoder
        token_type_ids = torch.zeros_like(input_ids)
        outputs = self.model.token_rep_layer(
            input_ids, attention_mask, token_type_ids
        )

        if isinstance(outputs, dict):
            token_reps = outputs.get("last_hidden_state", outputs.get("token_embeddings"))
        elif isinstance(outputs, (tuple, list)):
            token_reps = outputs[0]
        else:
            token_reps = outputs

        # Word-level pooling (subword -> word)
        B, _, H = token_reps.shape
        words_mask_float = words_mask.unsqueeze(-1).float()
        word_reps = token_reps * words_mask_float

        # Get span representations
        span_reps = self.model.rnn(word_reps, span_idx)
        if hasattr(span_reps, 'last_hidden_state'):
            span_reps = span_reps.last_hidden_state

        # Entity scoring
        entity_scores = torch.zeros(B, span_idx.shape[1], 1)
        if hasattr(self.model, 'span_rep_layer'):
            entity_scores = self.model.span_rep_layer(span_reps, span_mask)

        # Relation scoring between span pairs
        num_spans = span_idx.shape[1]
        relation_scores = torch.zeros(B, num_spans, num_spans, 1)
        if hasattr(self.model, 'rel_rep_layer'):
            # Compute pairwise span representations for relations
            head_reps = span_reps.unsqueeze(2).expand(-1, -1, num_spans, -1)
            tail_reps = span_reps.unsqueeze(1).expand(-1, num_spans, -1, -1)
            pair_reps = torch.cat([head_reps, tail_reps], dim=-1)
            relation_scores = self.model.rel_rep_layer(pair_reps)

        return entity_scores, relation_scores


def main():
    parser = argparse.ArgumentParser(
        description="Export GLiNER-RelEx to ONNX"
    )
    parser.add_argument(
        "--model",
        default="knowledgator/gliner-relex-large-v1.0",
        help="HuggingFace model ID",
    )
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument(
        "--quantize", action="store_true", help="Also export INT8 quantized"
    )
    parser.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version"
    )
    args = parser.parse_args()

    model_id = args.model
    if args.output:
        out = Path(args.output)
    else:
        safe_name = model_id.replace("/", "--")
        cache_dir = Path(
            os.environ.get(
                "HF_HOME", Path.home() / ".cache" / "huggingface"
            )
        )
        out = cache_dir / "hub" / f"models--{safe_name}" / "onnx"

    out.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {model_id} to {out}")

    # Load the GLiNER model
    from gliner import GLiNER

    model = GLiNER.from_pretrained(model_id, load_tokenizer=True)
    model.eval()

    # Save tokenizer
    if hasattr(model, "data_processor") and hasattr(
        model.data_processor, "transformer_tokenizer"
    ):
        model.data_processor.transformer_tokenizer.save_pretrained(str(out))
        print(f"Saved tokenizer to {out}")

    # Save config
    import json

    config_path = out / "gliner_config.json"
    if hasattr(model, "config"):
        config_dict = (
            model.config.to_dict()
            if hasattr(model.config, "to_dict")
            else vars(model.config)
        )
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        print(f"Saved config to {config_path}")

    # Try direct ONNX export first
    print("Attempting ONNX export...")
    onnx_path = out / "model.onnx"

    try:
        # Use the GLiNER library's export if available
        if hasattr(model, "to_onnx"):
            model.to_onnx(str(out), quantize=args.quantize)
            print(f"Exported via GLiNER library to {out}")
        else:
            # Manual export
            wrapper = GLiNERRelExWrapper(model.model)
            wrapper.eval()

            # Create dummy inputs
            batch_size, seq_len, num_words, num_spans = 1, 128, 20, 50
            dummy_inputs = {
                "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
                "words_mask": torch.ones(
                    batch_size, seq_len, dtype=torch.long
                ),
                "text_lengths": torch.tensor([num_words]),
                "span_idx": torch.randint(
                    0, num_words, (batch_size, num_spans, 2)
                ),
                "span_mask": torch.ones(
                    batch_size, num_spans, dtype=torch.bool
                ),
            }

            torch.onnx.export(
                wrapper,
                (
                    dummy_inputs["input_ids"],
                    dummy_inputs["attention_mask"],
                    dummy_inputs["words_mask"],
                    dummy_inputs["text_lengths"],
                    dummy_inputs["span_idx"],
                    dummy_inputs["span_mask"],
                ),
                str(onnx_path),
                input_names=[
                    "input_ids",
                    "attention_mask",
                    "words_mask",
                    "text_lengths",
                    "span_idx",
                    "span_mask",
                ],
                output_names=["entity_scores", "relation_scores"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq_len"},
                    "attention_mask": {0: "batch", 1: "seq_len"},
                    "words_mask": {0: "batch", 1: "seq_len"},
                    "text_lengths": {0: "batch"},
                    "span_idx": {0: "batch", 1: "num_spans"},
                    "span_mask": {0: "batch", 1: "num_spans"},
                    "entity_scores": {0: "batch", 1: "num_spans"},
                    "relation_scores": {
                        0: "batch",
                        1: "num_spans",
                        2: "num_spans",
                    },
                },
                opset_version=args.opset,
            )
            print(f"Exported model.onnx to {onnx_path}")

            if onnx_path.exists():
                size_mb = onnx_path.stat().st_size / 1e6
                print(f"Model size: {size_mb:.1f} MB")

    except Exception as e:
        print(f"ONNX export failed: {e}")
        traceback.print_exc()
        # Fallback: save PyTorch weights
        pt_path = out / "model.pt"
        torch.save(model.model.state_dict(), str(pt_path))
        print(f"Saved PyTorch state_dict to {pt_path}")
        print(
            "Note: Rust GLiNER-RelEx backend will need candle feature for PyTorch loading."
        )
        return

    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q_path = out / "model_quantized.onnx"
        quantize_dynamic(
            str(onnx_path), str(q_path), weight_type=QuantType.QInt8
        )
        if q_path.exists():
            print(
                f"Quantized: {q_path} ({q_path.stat().st_size / 1e6:.1f} MB)"
            )

    print("Export complete.")


if __name__ == "__main__":
    main()
