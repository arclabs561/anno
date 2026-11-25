#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "torch>=2.6.0",
#     "transformers>=4.30.0",
#     "safetensors>=0.4.0",
#     "numpy>=1.21.0",
#     "onnx>=1.14.0",
#     "onnxscript>=0.1.0",
# ]
# ///
"""
Export W2NER model to ONNX format.

W2NER (Word-to-Word NER) uses a grid-based approach for nested/discontinuous NER.
This script exports a simplified version suitable for inference.

Usage:
    # Clone W2NER repo and train a model first, or download pretrained weights
    uv run scripts/export_w2ner_to_onnx.py --model-dir ./w2ner-conll2003 --output model.onnx
    
    # Or with HuggingFace model (if you have access)
    uv run scripts/export_w2ner_to_onnx.py --hf-model ljynlp/w2ner-bert-base --output model.onnx

Note: The original W2NER model uses dynamic operations (pack_padded_sequence) that
require careful handling for ONNX export. This script creates a simplified export
suitable for inference with fixed-length inputs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class W2NERSimplified(nn.Module):
    """
    Simplified W2NER for ONNX export.
    
    Key simplifications:
    - Fixed sequence length (no dynamic packing)
    - Removed LSTM in favor of attention pooling
    - Simplified grid generation
    """
    
    def __init__(self, bert_name: str, num_labels: int = 7, hidden_size: int = 768):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        
        # Biaffine scoring layers
        self.head_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.tail_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Biaffine weight: [num_labels, head_dim+1, tail_dim+1]
        biaffine_dim = hidden_size // 2
        self.biaffine_weight = nn.Parameter(
            torch.zeros(num_labels, biaffine_dim + 1, biaffine_dim + 1)
        )
        nn.init.xavier_normal_(self.biaffine_weight)
        
        # Distance embeddings for position information
        self.dist_emb = nn.Embedding(20, 32)
        self.dist_proj = nn.Linear(32, num_labels)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for ONNX export.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            
        Returns:
            grid_logits: [batch, seq_len, seq_len, num_labels]
        """
        batch_size, seq_len = input_ids.shape
        
        # Get BERT embeddings
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Project to biaffine space
        head_repr = self.head_mlp(hidden)  # [batch, seq_len, dim]
        tail_repr = self.tail_mlp(hidden)  # [batch, seq_len, dim]
        
        # Add bias dimension
        head_repr = torch.cat([
            head_repr,
            torch.ones(batch_size, seq_len, 1, device=head_repr.device)
        ], dim=-1)
        tail_repr = torch.cat([
            tail_repr,
            torch.ones(batch_size, seq_len, 1, device=tail_repr.device)
        ], dim=-1)
        
        # Biaffine scoring: [batch, seq_len, seq_len, num_labels]
        # einsum: batch head_pos head_dim, labels head_dim tail_dim, batch tail_pos tail_dim
        #      -> batch labels head_pos tail_pos
        grid_logits = torch.einsum(
            'bxi,oij,byj->bxyo',
            head_repr,
            self.biaffine_weight,
            tail_repr
        )
        
        # Add distance bias
        positions = torch.arange(seq_len, device=input_ids.device)
        dist_matrix = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        dist_matrix = dist_matrix.clamp(0, 19)  # Cap at max distance
        dist_embs = self.dist_emb(dist_matrix)  # [seq_len, seq_len, 32]
        dist_bias = self.dist_proj(dist_embs)  # [seq_len, seq_len, num_labels]
        
        grid_logits = grid_logits + dist_bias.unsqueeze(0)
        
        return grid_logits


def export_to_onnx(
    model: nn.Module,
    tokenizer,
    output_path: str,
    max_length: int = 128,
):
    """Export model to ONNX format."""
    model.eval()
    
    # Create dummy inputs
    dummy_text = "This is a sample sentence for export."
    encoding = tokenizer(
        dummy_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    # Export
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["grid_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "grid_logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    
    print(f"✓ Exported to {output_path}")
    
    # Verify
    import onnx
    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)
    print("✓ ONNX model verified")


def main():
    parser = argparse.ArgumentParser(description="Export W2NER to ONNX")
    parser.add_argument(
        "--bert-model",
        default="bert-base-cased",
        help="Base BERT model to use (HuggingFace model id)",
    )
    # Back-compat: earlier docs used `--hf-model`.
    parser.add_argument(
        "--hf-model",
        dest="bert_model",
        help="Alias for --bert-model (deprecated)",
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=7,
        help="Number of relation labels (default: 7 for NNW/THW * entity_types + None)",
    )
    parser.add_argument(
        "--output",
        default="w2ner.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--weights",
        help="Path to pretrained weights (optional)",
    )
    args = parser.parse_args()
    
    print(f"Creating W2NER model with {args.bert_model}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    
    # Create model
    model = W2NERSimplified(
        bert_name=args.bert_model,
        num_labels=args.num_labels,
    )
    
    # Load weights if provided
    if args.weights and os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}...")
        state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    
    # Export
    export_to_onnx(model, tokenizer, args.output, args.max_length)
    
    # Save tokenizer alongside
    tokenizer_dir = Path(args.output).parent
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"✓ Tokenizer saved to {tokenizer_dir}")
    
    print("\nTo use with anno:")
    print(f"  anno extract --model w2ner --model-path {tokenizer_dir} 'Your text here'")


if __name__ == "__main__":
    main()

