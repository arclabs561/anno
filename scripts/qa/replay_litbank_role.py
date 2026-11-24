"""Replay four selected LitBank identity judgments with Anno's fcoref_word_spans example."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from check_span_assertions import indexed, source_sha256
from score_person_mentions import model_spans, span


FIXTURE = Path(__file__).parent / "fixtures/litbank-role-link.json"


def prepare(fixture, checkout):
    # Read bytes to preserve the pinned line endings and document-relative scalar offsets.
    source = (checkout / fixture["source_path"]).read_bytes().decode("utf-8")
    if source_sha256(source) != fixture["source_sha256"]:
        raise ValueError("LitBank source differs from the pinned text")
    start, end = span(fixture["excerpt"], source)
    text = source[start:end]
    if source_sha256(text) != fixture["excerpt_sha256"]:
        raise ValueError("LitBank excerpt differs from the pinned context")
    named = {}
    for key, bounds in fixture["spans"].items():
        lo, hi = span(bounds, source)
        named[key] = span([lo - start, hi - start], text)
    return {"id": fixture["id"], "text": text}, named


def score(doc, named, pairs, predictions):
    rows = indexed(predictions["results"], "prediction documents")
    if rows.keys() != {doc["id"]}:
        raise ValueError("expected exactly the pinned LitBank document")
    proposed, clusters = model_spans(doc, rows[doc["id"]])
    endpoints = {}
    for key, bounds in named.items():
        groups = [index for index, group in enumerate(clusters) if bounds in group]
        if len(groups) > 1:
            raise ValueError(f"{key} appears in multiple emitted clusters")
        endpoints[key] = {"proposed": bounds in proposed, "cluster": groups[0] if groups else None}

    judgments = []
    for left, right, expected in pairs:
        a, b = endpoints[left]["cluster"], endpoints[right]["cluster"]
        if a is None or b is None:
            observed = "unresolved"
        else:
            observed = "same" if a == b else "different"
        judgments.append({"left": left, "right": right, "expected": expected, "observed": observed})
    matches = all(pair["expected"] == pair["observed"] for pair in judgments)
    return {"id": doc["id"], "source_sha256": source_sha256(doc["text"]),
            "status": "pass" if matches else "mismatch", "endpoints": endpoints, "pairs": judgments}


def unavailable(doc, reason):
    print(json.dumps({"id": doc["id"], "status": "not_run", "reason": reason}))
    return 77


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("litbank", type=Path, help="root of the pinned LitBank checkout")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--predictions", type=Path, help="rescore an existing word-aligned panel receipt")
    choice.add_argument("--runner", type=Path, help="fcoref_word_spans executable")
    parser.add_argument("--model-dir", type=Path, help="local FP32 FCoref export for the runner")
    args = parser.parse_args()
    if bool(args.runner) != bool(args.model_dir):
        parser.error("supply --runner and --model-dir together")
    fixture = json.loads(FIXTURE.read_text())
    doc, named = prepare(fixture, args.litbank)
    if args.predictions:
        predictions = json.loads(args.predictions.read_bytes())
    elif args.runner:
        if not args.runner.is_file() or not os.access(args.runner, os.X_OK):
            return unavailable(doc, "configured fcoref_word_spans executable is unavailable")
        required = ("config.json", "tokenizer.json", "scorer_weights.safetensors", "encoder.onnx")
        missing = [name for name in required if not (args.model_dir / name).is_file()]
        if missing:
            return unavailable(doc, "missing FCoref model files: " + ", ".join(missing))
        with tempfile.TemporaryDirectory() as temp:
            panel = Path(temp) / "panel.json"
            panel.write_text(json.dumps([doc]), encoding="utf-8")
            output = subprocess.check_output([args.runner.resolve(), args.model_dir, panel])
            predictions = json.loads(output)
    else:
        return unavailable(doc, "no model or saved predictions supplied")
    result = score(doc, named, fixture["pairs"], predictions)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, ValueError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"LitBank replay error: {exc}", file=sys.stderr)
        sys.exit(2)
