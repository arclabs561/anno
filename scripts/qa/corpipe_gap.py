"""Score the released CorPipe one-stage model on Anno's existing English GAP panel."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from ufal.udpipe import Model, Pipeline, ProcessingError

import coref_gap

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "crates/anno-cli/src/cli/commands")
)
from corpipe import ground


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["gap", "predictor", "checkpoint", "udpipe", "output"]:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    docs = coref_gap.panel(args.gap)
    coref_gap.check_file(
        args.checkpoint / "model.pt",
        "023759452da9ffe53c3de06bf517027422d481973388949a2efed937c9638bf5",
    )
    coref_gap.check_file(
        args.predictor,
        "0dc86353a1c6a7246a296587a0c06577f42bcbecd66fb563e7d2827b8e9a0365",
    )
    coref_gap.check_file(
        args.udpipe, "784bd0fa85e3d831fd02a55290d0acfd05c953159dc38cc33d52e1b28add9957"
    )
    tokenizer = Model.load(str(args.udpipe))
    pipeline = Pipeline(
        tokenizer, "tokenizer=ranges", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu"
    )
    start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        files = []
        for i, (_, text, _, _) in enumerate(docs):
            source = root / f"source-{i}.conllu"
            error = ProcessingError()
            source.write_text(pipeline.process(text, error), encoding="utf-8")
            if error.occurred():
                raise ValueError(error.message)
            files.append(str(source))
        completed = subprocess.run(
            [
                sys.executable,
                str(args.predictor.resolve()),
                "--load",
                str(args.checkpoint.resolve()),
                "--epoch",
                "0",
                "--exp",
                tmp,
                "--test",
                *files,
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        if "Truncating a long sentence" in completed.stdout:
            raise ValueError("A test sentence exceeded the model window")
        records = []
        for i, (docid, text, _, _) in enumerate(docs):
            output = (root / f"source-{i}.00.conllu").read_text()
            clusters = [
                [(m["start"], m["end"]) for m in c["mentions"] if "start" in m]
                for c in ground(text, output)
            ]
            records.append(dict(id=docid, producer="corpipe26-base", clusters=clusters))
        result = coref_gap.measure(docs, records, "corpipe26-base")
        result.update(
            checkpoint_revision="cb62351dd97b58c95bb7e12258a669e471cb577d",
            checkpoint="ufal/corpipe26-onestage-corefud1.4-base-260702",
            predictor_revision="ec853c3b2d97890f4ef04114b10af6a1ac01d78f",
            udpipe="english-ewt-ud-2.5-191206",
            elapsed_seconds=time.monotonic() - start,
            versions={
                n: version(n)
                for n in ["torch", "transformers", "minnt", "ufal.udpipe", "udapi"]
            },
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
