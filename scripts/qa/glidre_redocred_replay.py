# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay the six-document Re-DocRED panel on a prepared local GLiDRE checkpoint."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "docs/eval/glidre-redocred-panel.json"
SCORER = ROOT / "crates/anno-eval/examples/docred/score.py"


def read_source(path, expected_digest):
    data = path.read_bytes()
    if data.startswith(b"\x1f\x8b"):
        data = gzip.decompress(data)
    if hashlib.sha256(data).hexdigest() != expected_digest:
        raise ValueError(f"Unexpected source SHA-256 for {path}")
    return json.loads(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--anno", default="anno")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    receipt = json.loads(RECEIPT.read_bytes())
    sources = receipt["sources"]
    docs = read_source(args.split, sources["split"]["sha256"])
    names = read_source(args.schema, sources["schema"]["decompressed_sha256"])
    prompts = {key: name.upper().replace(" ", "_") for key, name in names.items()}
    settings = receipt["settings"]
    if len(set(prompts.values())) != settings["predicates"] or len(prompts) != settings["predicates"]:
        raise ValueError("Predicate inventory differs from the recorded study")

    enrollment = receipt["enrollment"]
    if len({doc["title"] for doc in docs}) != enrollment["split_documents"]:
        raise ValueError("Unexpected development split")

    def rank(index):
        return hashlib.sha256((enrollment["rank_prefix"] + docs[index]["title"]).encode()).digest()

    indices = sorted(range(len(docs)), key=rank)[:enrollment["size"]]
    if indices != enrollment["indices"]:
        raise ValueError("Ranked document indices differ from enrollment")
    gold = [docs[i] for i in indices]
    blinded = [{k: doc[k] for k in ("title", "sents", "vertexSet")} for doc in gold]

    weights = args.model_dir / "pytorch_model.bin"
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != receipt["checkpoint"]["pytorch_model.bin_sha256"]:
        raise ValueError(f"Unexpected checkpoint SHA-256 for {weights}")

    with tempfile.TemporaryDirectory(prefix="anno-glidre-redocred-") as tmp:
        scratch = Path(tmp)
        for name, value in (("gold", gold), ("documents", blinded), ("prompts", prompts)):
            (scratch / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
        predictions = scratch / "predictions.json"
        with predictions.open("wb") as output:
            subprocess.run([
                args.anno, "glidre", "--upstream", args.upstream, "--model-dir", args.model_dir,
                "--python", args.python, "--input", scratch / "documents.json",
                "--schema", scratch / "prompts.json", "--threshold", str(settings["threshold"]),
                "--device", settings["device"],
            ], stdout=output, check=True)
        result = json.loads(subprocess.check_output([
            sys.executable, SCORER, scratch / "gold.json", predictions,
            "--endpoints", settings["endpoints"],
        ]))
    measured = {key: result[key] for key in receipt["result"]}
    print(json.dumps(measured, indent=2))
    if measured != receipt["result"]:
        raise SystemExit("Measured results differ from the recorded receipt")


if __name__ == "__main__":
    main()
