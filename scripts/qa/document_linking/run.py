"""Replay the authored document workflow with local checkpoints; write per-stage outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from prepare_linking_document import prepare, read


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args):
    work = args.output
    work.mkdir(parents=True, exist_ok=True)
    sources = read(HERE / "documents.json")
    # Withhold reference identities and relations from every inference process.
    refs = {r["id"]: r["entities"] for r in read(HERE / "reviewed.json")}
    bank = read(HERE / "candidates.json")
    predictions, reviewed = [], []

    def call(command, filename, *, source=None, env=None):
        result = subprocess.run([args.anno, *command], input=source, encoding="utf-8",
                                stdout=subprocess.PIPE, env=env, check=True)
        (work / filename).write_text(result.stdout, encoding="utf-8")

    for doc in sources:
        stem = doc["id"]
        call(["extract", "--model", "bert-onnx", "--format", "json", "--text", doc["text"]],
             f"{stem}-ner.json")
        env = os.environ.copy()
        if args.corpipe_cache:
            env["HF_HUB_CACHE"] = args.corpipe_cache
        call(["corpipe", "--predictor", args.corpipe_predictor, "--checkpoint", args.corpipe_model,
              "--tokenizer-model", args.udpipe, "--python", args.corpipe_python],
             f"{stem}-coref.json", source=doc["text"], env=env)
        predictions.append(prepare(doc, read(work / f"{stem}-ner.json"),
                                   read(work / f"{stem}-coref.json")))
        reviewed.append({**doc, "entities": refs[stem]})

    for arm, docs in [("predicted", predictions), ("reviewed", reviewed)]:
        inputs, candidates = work / f"{arm}-documents.json", work / f"{arm}-candidates.json"
        save(inputs, docs)
        save(candidates, [{"doc_id": doc["id"], "entity_id": e["id"], "start": m["start"],
                           "end": m["end"], "candidates": bank}
                          for doc in docs for e in doc["entities"] for m in e["mentions"]])
        call(["glidre", "--upstream", args.glidre_upstream, "--model-dir", args.glidre_model,
              "--python", args.glidre_python, "--input", inputs, "--schema", HERE / "schema.json",
              "--threshold", "0.3"], f"{arm}-relations.json")
        call(["glidre", "--upstream", args.glidre_upstream, "--model-dir", args.glidre_model,
              "--python", args.glidre_python, "--input", inputs, "--schema", HERE / "schema.json",
              "--threshold", "0"], f"{arm}-scores.json")
        call(["mhel", "rerank", "--upstream", args.mhel_upstream, "--model-dir", args.mhel_model,
              "--python", args.mhel_python, "--documents", inputs, "--candidates", candidates,
              "--threshold", "0"], f"{arm}-links.jsonl")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--anno", default="anno")
    for arg in ("corpipe-predictor", "corpipe-model", "udpipe", "corpipe-python",
                "glidre-upstream", "glidre-model", "glidre-python", "mhel-upstream",
                "mhel-model", "mhel-python"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--corpipe-cache", help="HF_HUB_CACHE override for the CorPipe process")
    run(p.parse_args())
