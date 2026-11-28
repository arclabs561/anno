"""Line-oriented CPU inference using the original Otter prediction and decoding code."""
import hashlib
import json
import os
from pathlib import Path
import sys
import shutil
import tempfile

# Configure the libraries before import. Loading a local checkpoint must not download
# tokenizers, model code, or fallback weights.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def verified_snapshot(root, target, files):
    """Expose only the pinned files, including the Python code AutoModel imports."""
    for name, expected in files.items():
        source = root / name
        size = source.stat().st_size
        if size != expected["size"]:
            raise ValueError(f"unexpected file size for {name}")
        with source.open("rb") as stream:
            if "sha256" in expected:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                wanted = expected["sha256"]
            else:
                blob = hashlib.sha1(f"blob {size}\0".encode())
                blob.update(stream.read())
                digest, wanted = blob.hexdigest(), expected["git_blob"]
        if digest != wanted:
            raise ValueError(f"checkpoint fingerprint differs for {name}")
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.symlink_to(source)
        except OSError:
            # Windows can require a privilege for symbolic links.
            try:
                path.hardlink_to(source.resolve())
            except OSError:
                shutil.copyfile(source, path)


def run():
    path = Path(sys.argv[1]).resolve(strict=True)
    variant = sys.argv[2]
    with tempfile.TemporaryDirectory(prefix="anno-otter-") as tmp:
        staging = Path(tmp)
        snapshot = staging / "otter"
        verified_snapshot(path, snapshot, json.loads(sys.argv[3])[variant]["files"])
        os.environ["HF_MODULES_CACHE"] = str(staging / "modules")
        os.environ.setdefault("HF_HOME", str(staging / "cache"))
        import torch
        from transformers import AutoModel, PreTrainedTokenizerFast

        torch.set_num_threads(1)
        model = AutoModel.from_pretrained(
            snapshot, trust_remote_code=True, local_files_only=True, use_safetensors=True
        ).eval()
        if variant == "bi":
            # The published mmBERT tokenizer declares the Transformers 5 generic
            # TokenizersBackend class. Open its serialized fast tokenizer directly
            # to support the 4.x runtime without resolving the remote base model.
            folder = snapshot / "token_tokenizer"
            config = json.loads((folder / "tokenizer_config.json").read_text())
            fields = ("bos_token", "cls_token", "eos_token", "mask_token", "pad_token",
                      "sep_token", "unk_token", "padding_side", "model_input_names",
                      "model_max_length", "clean_up_tokenization_spaces")
            model._token_tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(folder / "tokenizer.json"),
                **{field: config[field] for field in fields},
            )
        token = model.tokenizer if variant == "cross" else model.token_tokenizer
        if variant == "bi":
            type_token = model.type_tokenizer
        limit = model.config.max_seq_length
    print('"ready"', flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            text, labels, threshold = request["text"], request["labels"], request["threshold"]
            prompt = model.build_prompt(labels) if variant == "cross" else ""
            length = len(token(prompt + text, truncation=False)["input_ids"])
            if length > limit:
                raise ValueError(f"text and prompt need {length} tokens, maximum is {limit}")
            if variant == "bi":
                label_limit = min(type_token.model_max_length, model.config.type_encoder_config.max_position_embeddings)
                if any(len(ids) > label_limit for ids in type_token(labels, truncation=False)["input_ids"]):
                    raise ValueError(f"label exceeds type encoder limit of {label_limit} tokens")
            with torch.inference_mode():
                spans = model.predict(text, labels=labels, threshold=threshold)
            print(json.dumps({"ok": spans}, allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps({"error": str(error)}), flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        print(json.dumps({"error": str(error)}), flush=True)
        sys.exit(1)
