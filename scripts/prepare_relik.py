#!/usr/bin/env python3
"""Prepare the AIDA ReLiK reader for offline inference from its released weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

READER = "sapienzanlp/relik-reader-deberta-v3-base-aida"
REVISION = "3bed88483cabb40188d7a3f8e79ec895c45b3a28"
BASE = "microsoft/deberta-v3-base"
BASE_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
WEIGHTS_SHA = "60ae577b1b5cb7bfff9776ccdbea4075e699976ecc9eb1e97afbf5a4d0933f1a"
TOKENIZER = ("added_tokens.json", "special_tokens_map.json", "spm.model", "tokenizer.json", "tokenizer_config.json")

SOURCE_FILES = {
    "added_tokens.json": "2b1f18d6bd57a905d4b9279a142b5a19d99b0354da66d3d2978483b9ee01f096",
    "config.json": "6f151acbc62ed12ea40c2f865c436605d9ac55bf3adfcfa42f6c0a95e81d506b",
    "special_tokens_map.json": "06fd042591f252d92006a0031f441623869384234d3d637e2b896f9cad933f03",
    "spm.model": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
    "tokenizer.json": "c94431ae3eb6e9cf3955117d600aad23156adafcedd2a8fa508bf50670a2f2d8",
    "tokenizer_config.json": "94962ac55de7927fac634e9ac1f4b1e54667d933ea603c0fa9ab74118e5ab4fa",
}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare(source, base_config, stage):
    import torch
    from safetensors.torch import save_file
    weights = source / "pytorch_model.bin"
    if digest(weights) != WEIGHTS_SHA:
        raise ValueError("ReLiK checkpoint differs from the pinned release")
    for name, expected in SOURCE_FILES.items():
        if digest(source / name) != expected:
            raise ValueError(f"unexpected ReLiK revision for {name}")
    config = json.loads((source / "config.json").read_bytes())
    if config["architectures"] != ["RelikReaderSpanModel"] or config["transformer_model"] != BASE:
        raise ValueError("expected the AIDA span reader")
    base = stage / "encoder"
    base.mkdir()
    if digest(base_config) != "649f6a1ec33c6bdd9a6486d5c66019d461139e54957073eafe9bbc2d34c75b0b":
        raise ValueError("unexpected DeBERTa base config revision")
    shutil.copyfile(base_config, base / "config.json")
    size = json.loads(base_config.read_bytes())["vocab_size"]
    state = torch.load(weights, mmap=True, map_location="cpu", weights_only=True)
    encoder = {key.removeprefix("transformer_model."): tensor for key, tensor in state.items()
               if key.startswith("transformer_model.")}
    head = {key: tensor for key, tensor in state.items() if not key.startswith("transformer_model.")}
    # The released reader constructor loads an encoder and then appends 101 vocabulary rows.
    # Store the original trained rows on either side of that boundary, without duplicating the encoder.
    word_key = "embeddings.word_embeddings.weight"
    original = encoder[word_key]
    if original.shape[0] != size + config["additional_special_symbols"]:
        raise ValueError("unexpected ReLiK embedding table size")
    encoder[word_key] = original[:size]
    head["additional_word_embeddings"] = original[size:]
    save_file(encoder, str(base / "model.safetensors"), metadata={"format": "pt"})
    save_file(head, str(stage / "reader.safetensors"), metadata={"format": "pt"})
    (stage / "tokenizer").mkdir()
    for name in TOKENIZER:
        shutil.copyfile(source / name, stage / "tokenizer" / name)
    (stage / "reader.json").write_text(json.dumps(config))
    (stage / "receipt.json").write_text(json.dumps({"reader": READER, "revision": REVISION,
        "source_weights_sha256": WEIGHTS_SHA, "encoder_revision": BASE_REVISION,
        "files": {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob("*")) if p.is_file()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, help="existing pinned upstream reader snapshot")
    parser.add_argument("--base-config", type=Path, help="existing pinned DeBERTa-v3-base config.json")
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new directory")
    from huggingface_hub import hf_hub_download, snapshot_download
    source = args.source or Path(snapshot_download(READER, revision=REVISION,
        allow_patterns=["pytorch_model.bin", "config.json", *TOKENIZER]))
    base = args.base_config or Path(hf_hub_download(BASE, "config.json", revision=BASE_REVISION))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as scratch:
        stage = Path(scratch, "model")
        stage.mkdir()
        prepare(source, base, stage)
        os.rename(stage, dest)
    print(dest)


if __name__ == "__main__":
    main()
