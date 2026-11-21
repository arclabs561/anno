"""Export the pinned FP32 checkpoint for the word-budget regression."""

import argparse
import hashlib
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, RobertaModel

CHECKPOINT = "30ea6efe2bce49a857fe8aba41dcfe28b4147f77e3dec9f14100950bf4de1aa5"


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask).last_hidden_state


def export(source, output):
    weights = source / "pytorch_model.bin"
    with weights.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINT:
            raise ValueError("Unexpected checkpoint SHA-256")

    config = AutoConfig.from_pretrained(source, local_files_only=True)
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    # The checkpoint includes the position buffer. Transformers 4.51 rebuilds it.
    position_ids = state.pop("roberta.embeddings.position_ids")
    expected = torch.arange(config.max_position_embeddings).unsqueeze(0)
    torch.testing.assert_close(position_ids, expected)
    with torch.device("meta"):
        model = RobertaModel(config)
    prefix = "roberta."
    encoder = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    model.load_state_dict(encoder, strict=True, assign=True)
    model.embeddings.position_ids = position_ids
    model.embeddings.token_type_ids = torch.zeros_like(position_ids)
    wrapper = Encoder(model).eval()

    output.mkdir(exist_ok=True, parents=True)
    ids = torch.tensor([[0, 123, 5, 2]])
    with torch.inference_mode():
        torch.onnx.export(
            wrapper, (ids, torch.ones_like(ids)), output / "encoder.onnx",
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={k: {0: "batch", 1: "seq_len"}
                          for k in ["input_ids", "attention_mask", "last_hidden_state"]},
            opset_version=17, dynamo=False, do_constant_folding=False,
        )
    heads = {k: v for k, v in state.items() if not k.startswith(prefix)}
    save_file(heads, str(output / "scorer_weights.safetensors"))
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.backend_tokenizer.save(str(output / "tokenizer.json"))
    shutil.copyfile(source / "config.json", output / "config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.source, args.output)
