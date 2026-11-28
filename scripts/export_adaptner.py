# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.8.0", "transformers==4.55.4", "onnx==1.19.0", "onnxruntime==1.32.0"]
# ///
"""Export the published Arabic AdaptNER development checkpoint as local FP32 ONNX."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForTokenClassification, AutoTokenizer

REPO = "NAMAA-Space/KnowledgeGraphEval-2026-AdaptNER-AraBERTv02-Nested"
REVISION = "c3096335b1e8d00f51451e267b8034a0cc59876f"
FILES = {
    "config.json": "be08ce6e6298c1be622c7346d06d1528b2367bc9ac3b16baaa2bf00e302ac83b",
    "tokenizer.json": "b822d30d37de1d87400e891701268c0f8acf979c5ecc0eb424e831407cbe3498",
    "tokenizer_config.json": "72e0dad2b800a9b128eab62bb51c0fd129b06530f2f430a67f31eb0a98a68088",
    "model.safetensors": "fa6c90741b73d86c4cf49ec9f467eca1b19b99b1f38dd1fccd71c6daf031d5d4",
}


class TokenClassifier(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Use a previously downloaded pinned snapshot",
    )
    args = parser.parse_args()
    source = args.checkpoint_dir or Path(
        snapshot_download(REPO, revision=REVISION, allow_patterns=list(FILES))
    )
    for filename, digest in FILES.items():
        with (source / filename).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != digest:
                raise ValueError(f"Unexpected checksum for {filename}")
    cfg = json.loads((source / "config.json").read_text())
    labels = [cfg["id2label"][str(i)] for i in range(42)]
    model = TokenClassifier(
        AutoModelForTokenClassification.from_pretrained(
            source, local_files_only=True, attn_implementation="eager"
        )
    ).eval()
    tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
    torch.set_num_threads(2)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as scratch:
        dest = Path(scratch) / "export"
        dest.mkdir()
        words = "حرب 1948 ( النكبة )".split()
        enc = tok(words, is_split_into_words=True, return_tensors="pt")
        names = ["input_ids", "attention_mask"]
        sample = tuple(enc[k] for k in names)
        with torch.inference_mode():
            torch.onnx.export(
                model,
                sample,
                str(dest / "model.onnx"),
                dynamo=False,
                opset_version=17,
                input_names=names,
                output_names=["logits"],
                dynamic_axes={k: {1: "sequence"} for k in [*names, "logits"]},
            )
            session = ort.SessionOptions()
            session.intra_op_num_threads = 2
            ort_model = ort.InferenceSession(
                str(dest / "model.onnx"),
                sess_options=session,
                providers=["CPUExecutionProvider"],
            )
            for text in [words, ["فلسطين"], "صورة رئيس مجلس محلي بيرزيت".split()]:
                enc = tok(text, is_split_into_words=True, return_tensors="pt")
                pt_logits = model(*(enc[k] for k in names)).numpy()
                (ort_logits,) = ort_model.run(
                    ["logits"], {k: enc[k].numpy() for k in names}
                )
                np.testing.assert_allclose(ort_logits, pt_logits, atol=2e-4, rtol=2e-4)
        shutil.copyfile(source / "tokenizer.json", dest / "tokenizer.json")
        (dest / "adaptner.json").write_text(
            json.dumps({"revision": REVISION, "labels": labels}, indent=2) + "\n"
        )
        dest.rename(args.output)


if __name__ == "__main__":
    main()
