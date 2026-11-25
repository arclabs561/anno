#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "onnx>=1.14.0",
#     "onnxruntime>=1.16.0",
#     "transformers>=4.30.0",
#     "numpy>=1.21.0",
# ]
# ///
"""
Export TPLinker (Wang et al., COLING 2020) to ONNX for use in anno.

TPLinker uses a handshaking tagging scheme for joint entity-relation extraction.
The model produces three output heads over a handshaking sequence of length
L*(L+1)/2, where L is the token sequence length:

  - ent_logits:      [batch, hs_len, num_entity_tags]   -- entity boundary tags
  - head_rel_logits: [batch, hs_len, num_relation_tags]  -- head-to-tail relations
  - tail_rel_logits: [batch, hs_len, num_relation_tags]  -- tail-to-head relations

The handshaking maps position pairs (i, j) where i <= j to a flat index:
  idx = i * L - i * (i - 1) / 2 + (j - i)

Entity tags: {NONE, SH2OH, OH2ST, ST2OT, OT2ST}
  - SH2OH: Subject Head to Object Head
  - OH2ST: Object Head to Subject Tail
  - ST2OT: Subject Tail to Object Tail
  - OT2ST: Object Tail to Subject Tail (reverse direction)

Relation tags per relation type: {NONE, SH2OH, OH2ST}
  - Applied per (head_entity, tail_entity) pair

Usage:
    uv run scripts/export_tplinker_onnx.py
    uv run scripts/export_tplinker_onnx.py --checkpoint path/to/tplinker_nyt.pt
    uv run scripts/export_tplinker_onnx.py --output ~/.cache/anno/models/tplinker/
    uv run scripts/export_tplinker_onnx.py --dataset nyt --quantize

Checkpoints:
    Supply weights matching the three heads and relation order used below (NYT or WebNLG).
    The authors' TPLinkerPlus checkpoints use a different tag inventory and output head.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast

DEFAULT_OUTPUT = os.path.expanduser("~/.cache/anno/models/tplinker")


# ============================================================================
# TPLinker model definition (matches the original paper's architecture)
# ============================================================================


class HandshakingKernel(nn.Module):
    """Handshaking kernel that maps token pairs (i, j) to representations."""

    def __init__(self, hidden_size: int, shaking_type: str = "cat"):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == "cat":
            self.combine = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine = nn.Linear(hidden_size * 3, hidden_size)

    def forward(self, seq_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_hidden: [batch, seq_len, hidden]
        Returns:
            shaking_hiddens: [batch, hs_len, hidden]
            where hs_len = seq_len * (seq_len + 1) / 2
        """
        seq_len = seq_hidden.size(1)
        # Build upper-triangular index pairs
        guide = torch.triu(torch.ones(seq_len, seq_len, device=seq_hidden.device))
        row_idx, col_idx = torch.where(guide > 0)

        head_hidden = seq_hidden[:, row_idx, :]  # [batch, hs_len, hidden]
        tail_hidden = seq_hidden[:, col_idx, :]  # [batch, hs_len, hidden]

        if self.shaking_type == "cat":
            shaking = torch.cat([head_hidden, tail_hidden], dim=-1)
            shaking = self.combine(shaking)
        elif self.shaking_type == "cat_plus":
            diff = head_hidden - tail_hidden
            shaking = torch.cat([head_hidden, tail_hidden, diff], dim=-1)
            shaking = self.combine(shaking)
        else:
            shaking = head_hidden * tail_hidden

        return torch.tanh(shaking)


class TPLinkerModel(nn.Module):
    """TPLinker: Single-stage Joint Extraction of Entities and Relations."""

    def __init__(
        self,
        encoder_name: str = "bert-base-cased",
        num_entity_tags: int = 5,
        num_relation_types: int = 24,
        hidden_size: int = 768,
        shaking_type: str = "cat",
    ):
        super().__init__()
        self.encoder = BertModel.from_pretrained(encoder_name)
        self.hidden_size = hidden_size

        self.handshaking_kernel = HandshakingKernel(hidden_size, shaking_type)

        # Entity head: predicts entity boundary tags
        self.ent_fc = nn.Linear(hidden_size, num_entity_tags)

        # Relation heads: one per relation type (head-to-tail and tail-to-head)
        # Each outputs 3 tags: {NONE, SH2OH, OH2ST}
        self.head_rel_fc = nn.Linear(hidden_size, num_relation_types * 3)
        self.tail_rel_fc = nn.Linear(hidden_size, num_relation_types * 3)

        self.num_entity_tags = num_entity_tags
        self.num_relation_types = num_relation_types

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            ent_logits: [batch, hs_len, num_entity_tags]
            head_rel_logits: [batch, hs_len, num_relation_types * 3]
            tail_rel_logits: [batch, hs_len, num_relation_types * 3]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        seq_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Handshaking: map all (i, j) pairs to representations
        shaking_hidden = self.handshaking_kernel(seq_hidden)

        # Predict tags
        ent_logits = self.ent_fc(shaking_hidden)
        head_rel_logits = self.head_rel_fc(shaking_hidden)
        tail_rel_logits = self.tail_rel_fc(shaking_hidden)

        return ent_logits, head_rel_logits, tail_rel_logits


# ============================================================================
# Dataset configs (relation type schemas)
# ============================================================================

NYT_RELATIONS = [
    "/business/company/advisors",
    "/business/company/founders",
    "/business/company/industry",
    "/business/company/major_shareholders",
    "/business/company/place_founded",
    "/business/company_shareholder/major_shareholder_of",
    "/business/person/company",
    "/location/administrative_division/country",
    "/location/country/administrative_divisions",
    "/location/country/capital",
    "/location/location/contains",
    "/location/neighborhood/neighborhood_of",
    "/people/deceased_person/place_of_death",
    "/people/ethnicity/geographic_distribution",
    "/people/ethnicity/people",
    "/people/person/ethnicity",
    "/people/person/nationality",
    "/people/person/place_lived",
    "/people/person/place_of_birth",
    "/people/person/profession",
    "/people/person/religion",
    "/sports/sports_team/location",
    "/sports/sports_team_location/teams",
    "/time/event/locations",
]

WEBNLG_RELATIONS = [
    "abbreviation",
    "academicDiscipline",
    "activeYearsEndYear",
    "activeYearsStartYear",
    "affiliation",
    "almaMater",
    "anthem",
    "architect",
    "areaCode",
    "areaOfLand",
    "areaTotal",
    "assembly",
    "associatedBand/associatedMusicalArtist",
    "award",
    "battle",
    "bedpiecessSacristy",
    "birthDate",
    "birthName",
    "birthPlace",
    "birthYear",
    "broadcastArea",
    "campus",
    "capital",
    "chairman",
    "child",
    "city",
    "cityServed",
    "class",
    "club",
    "coachedBy",
    "commander",
    "completionDate",
    "comlestersystem",
    "country",
    "county",
    "course",
    "creator",
    "currency",
    "dean",
]

DATASET_CONFIGS = {
    "nyt": {"relations": NYT_RELATIONS, "encoder": "bert-base-cased"},
    "webnlg": {"relations": WEBNLG_RELATIONS, "encoder": "bert-base-cased"},
}


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    """Allow an omitted pretrained encoder, but require every learned task parameter."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for wrapper in ("model_state_dict", "state_dict"):
            if wrapper in state:
                state = state[wrapper]
                break
    if not isinstance(state, dict):
        raise ValueError("TPLinker checkpoint must contain a tensor state dictionary")
    if state and all(isinstance(key, str) and key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    expected = model.state_dict()
    task_keys = [key for key in expected if not key.startswith("encoder.")]
    missing = [key for key in task_keys if key not in state]
    if missing:
        raise ValueError(
            "Incompatible TPLinker checkpoint: missing task tensors "
            + ", ".join(missing)
            + ". Anno requires its handshaking kernel and three trained heads. "
            "TPLinkerPlus uses a single classifier with a different tag inventory."
        )
    for key, tensor in expected.items():
        if key in state and (not isinstance(state[key], torch.Tensor)
                             or state[key].shape != tensor.shape):
            actual = getattr(state[key], "shape", type(state[key]).__name__)
            raise ValueError(f"Incompatible TPLinker tensor {key}: expected {tensor.shape}, got {actual}")
    model.load_state_dict(state, strict=False)


def export_tplinker(
    checkpoint_path: str | None,
    output_dir: str,
    dataset: str = "nyt",
    quantize: bool = False,
    opset: int = 14,
) -> None:
    """Export TPLinker model to ONNX."""
    if checkpoint_path is not None and not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"TPLinker checkpoint does not exist: {checkpoint_path}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = DATASET_CONFIGS.get(dataset, DATASET_CONFIGS["nyt"])
    relations = config["relations"]
    encoder_name = config["encoder"]
    num_relations = len(relations)

    print(f"[tplinker-export] Dataset: {dataset} ({num_relations} relation types)")
    print(f"[tplinker-export] Encoder: {encoder_name}")

    # Build model
    model = TPLinkerModel(
        encoder_name=encoder_name,
        num_entity_tags=5,
        num_relation_types=num_relations,
        hidden_size=768,
        shaking_type="cat",
    )

    # Load checkpoint if provided
    if checkpoint_path is not None:
        print(f"[tplinker-export] Loading checkpoint: {checkpoint_path}")
        load_checkpoint(model, checkpoint_path)
        print("[tplinker-export] Checkpoint loaded.")
    else:
        print(
            "[tplinker-export] No checkpoint provided -- exporting with random weights."
        )
        print("[tplinker-export] Download pretrained weights from:")
        print(
            "[tplinker-export]   https://github.com/131250208/TPLinker-joint-extraction"
        )

    model.eval()

    # Save tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(encoder_name)
    tokenizer.save_pretrained(str(out))

    # Save config
    tplinker_config = {
        "dataset": dataset,
        "encoder": encoder_name,
        "num_entity_tags": 5,
        "num_relation_types": num_relations,
        "hidden_size": 768,
        "shaking_type": "cat",
        "entity_tags": ["NONE", "SH2OH", "OH2ST", "ST2OT", "OT2ST"],
        "relation_tags": ["NONE", "SH2OH", "OH2ST"],
        "relations": relations,
    }
    with open(out / "tplinker_config.json", "w") as f:
        json.dump(tplinker_config, f, indent=2)
    print(f"[tplinker-export] Config saved.")

    # Export to ONNX
    max_seq_len = 100
    dummy_input_ids = torch.randint(0, 30000, (1, max_seq_len))
    dummy_attention_mask = torch.ones(1, max_seq_len, dtype=torch.long)

    print("[tplinker-export] Exporting to ONNX...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(out / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["ent_logits", "head_rel_logits", "tail_rel_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "ent_logits": {0: "batch", 1: "hs_len"},
            "head_rel_logits": {0: "batch", 1: "hs_len"},
            "tail_rel_logits": {0: "batch", 1: "hs_len"},
        },
        opset_version=opset,
    )
    print(f"[tplinker-export] Exported model.onnx")

    if quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(
                str(out / "model.onnx"),
                str(out / "model_quantized.onnx"),
                weight_type=QuantType.QInt8,
            )
            print("[tplinker-export] Quantized model saved.")
        except Exception as e:
            print(f"[tplinker-export] Quantization failed: {e}")

    # Verify
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out / "model.onnx"))
        print("[tplinker-export] ONNX verification passed.")
        for inp in sess.get_inputs():
            print(f"  Input:  {inp.name}: {inp.shape} ({inp.type})")
        for out_node in sess.get_outputs():
            print(f"  Output: {out_node.name}: {out_node.shape} ({out_node.type})")

        # Test inference
        test_ids = np.ones((1, 20), dtype=np.int64)
        test_mask = np.ones((1, 20), dtype=np.int64)
        results = sess.run(None, {"input_ids": test_ids, "attention_mask": test_mask})
        hs_len = 20 * 21 // 2  # L*(L+1)/2
        print(f"  ent_logits shape: {results[0].shape} (expected: [1, {hs_len}, 5])")
        print(
            f"  head_rel_logits shape: {results[1].shape} (expected: [1, {hs_len}, {num_relations * 3}])"
        )
        print(
            f"  tail_rel_logits shape: {results[2].shape} (expected: [1, {hs_len}, {num_relations * 3}])"
        )
    except Exception as e:
        print(f"[tplinker-export] Verification failed: {e}")

    print(f"[tplinker-export] Done. Output: {out}")


def main():
    parser = argparse.ArgumentParser(description="Export TPLinker to ONNX")
    parser.add_argument(
        "--checkpoint", default=None, help="Path to TPLinker .pt checkpoint"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output dir (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dataset", default="nyt", choices=["nyt", "webnlg"], help="Dataset schema"
    )
    parser.add_argument("--quantize", action="store_true", help="Quantize to INT8")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()
    export_tplinker(
        args.checkpoint, args.output, args.dataset, args.quantize, args.opset
    )


if __name__ == "__main__":
    main()
