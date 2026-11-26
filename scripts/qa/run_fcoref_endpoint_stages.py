# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay the frozen GAP panel with the native FCoref boundary-rank readout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from coref_gap import FC_REV, check_file
from fcoref_emitted_boundaries import MODEL_FILES, panel


def observations(rows, binary, model):
    env = dict(os.environ, FCOREF_MODEL_PATH=str(model.resolve()),
               ANNO_NO_DOWNLOADS="1", HF_HUB_OFFLINE="1", ANNO_FCOREF_BOUNDARY_RANKS="1")
    env.pop("ANNO_FCOREF_DOUBLE_K", None)
    # Only one document is in flight. Reading its stdout response also establishes
    # that its rank table has already been written to stderr, even for equal-length inputs.
    with tempfile.TemporaryFile("w+b") as diagnostic:
        with subprocess.Popen([binary.resolve(), "fcoref"], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=diagnostic, text=True,
                              encoding="utf-8", env=env) as proc:
            try:
                for row in rows:
                    proc.stdin.write(json.dumps({"id": row["ID"], "text": row["Text"]}) + "\n")
                    proc.stdin.flush()
                    output = proc.stdout.readline()
                    if not output:
                        raise ValueError("Native resolver exited without returning its document")
                    stock = json.loads(output)
                    if stock.get("id") != row["ID"] or stock.get("producer") != "fcoref":
                        raise ValueError("Native resolver returned an unexpected document or producer")
                    diagnostic.seek(0)
                    tables = [json.loads(line[len(b"FCOREF_RANKS "):]) for line in diagnostic
                              if line.startswith(b"FCOREF_RANKS ")]
                    if len(tables) != 1:
                        raise ValueError("Expected one rank table per document; build with the study patch")
                    diagnostic.seek(0)
                    diagnostic.truncate()
                    yield {"id": row["ID"], "stock": stock, "trace": tables[0]}
                proc.stdin.close()
                if proc.stdout.read() or proc.wait():
                    raise ValueError("Native resolver did not finish cleanly")
            finally:
                if proc.poll() is None:
                    proc.kill()


def main():
    from tokenizers import Tokenizer
    from fcoref_endpoint_stages import score

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "test", "validation", "model", "binary"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    rows = panel(args.development, args.test, args.validation)
    for name, expected in MODEL_FILES.items():
        check_file(args.model / name, expected)
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    result = score(rows, observations(rows, args.binary, args.model), tokenizer)
    result["model_revision"] = FC_REV
    result["model_files"] = MODEL_FILES
    with args.binary.open("rb") as stream:
        result["diagnostic_binary_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
