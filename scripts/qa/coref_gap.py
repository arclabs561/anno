"""Run one matched coreference comparison on the frozen GAP test panel (stdlib only)."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

REVISION = "83135f26025cdd705c04f77e940d0c9201116b80"
DATA_SHA = "1c35e36d5b14f6313ec3f6cd67b275de282595dd59e59390e00cfff9897a6819"
PANEL = 128
FC_REV = "e91dfff12879495d882ba9460d0c5d5dd44ade59"
FC_FILES = {
    "encoder.onnx": "74b98b0f03b27154065d0918a566ed5968d14665e601ecaea6905014bfe6f065",
    "scorer_weights.safetensors": "54c3e13017825750b725dbb682db861c3b7933db6ec1306ff5f51d46c533d0e7",
    "tokenizer.json": "5ec67f22fb423648c32c22819353703e5d3dddd3a553b11989f37702fb910aa1",
    "config.json": "12361347b196a78de4fad99045cd127c19baa55590fc7885f84a89a7a785bfbf",
}


def check_file(path, expected):
    with Path(path).open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"{path} expected {expected}, received {digest}")


def panel(path):
    check_file(path, DATA_SHA)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    rows.sort(
        key=lambda r: hashlib.sha256(("anno-gap-v1/" + r["ID"]).encode()).digest()
    )
    docs = []
    for row in rows[:PANEL]:
        text = row["Text"]
        spans = []
        for column in ["Pronoun", "A", "B"]:
            start = int(row[column + "-offset"])
            end = start + len(row[column])
            if text[start:end] != row[column]:
                raise ValueError(f"invalid {column} span on {row['ID']}")
            spans.append((start, end))
        labels = [row[c + "-coref"] for c in ["A", "B"]]
        if any(label not in {"TRUE", "FALSE"} for label in labels):
            raise ValueError(f"invalid gold labels on {row['ID']}")
        docs.append((row["ID"], text, spans, [label == "TRUE" for label in labels]))
    if len(docs) != PANEL or len({d[0] for d in docs}) != PANEL:
        raise ValueError("panel IDs are missing or duplicated")
    return docs


def measure(docs, records, backend):
    counts = dict.fromkeys(
        [
            "documents",
            "gold_links",
            "predicted_links",
            "tp",
            "fp",
            "fn",
            "tn",
            "exact_decisions",
            "gold_both_covered",
            "eligible_pairs",
            "all_output_spans",
        ],
        0,
    )
    coverage = [0, 0, 0]
    observations = []
    for (doc_id, text, targets, gold), record in zip(docs, records, strict=True):
        if record["id"] != doc_id or record["producer"] != backend:
            raise ValueError("resolver or document mismatch")
        groups = [set(map(tuple, c)) for c in record["clusters"]]
        seen = set().union(*groups)
        if any(not 0 <= start < end <= len(text) for start, end in seen):
            raise ValueError(f"invalid returned character offsets on {doc_id}")
        covered = [span in seen for span in targets]
        pred = [
            any(targets[0] in c and target in c for c in groups)
            for target in targets[1:]
        ]
        counts["documents"] += 1
        counts["exact_decisions"] += pred == gold
        counts["all_output_spans"] += len(seen)
        for i, hit in enumerate(covered):
            coverage[i] += hit
        for i, (g, p) in enumerate(zip(gold, pred)):
            counts["gold_links"] += g
            counts["predicted_links"] += p
            counts["tp" if g and p else "fp" if p else "fn" if g else "tn"] += 1
            eligible = covered[0] and covered[i + 1]
            counts["eligible_pairs"] += eligible
            counts["gold_both_covered"] += g and eligible
        observations.append("".join("1" if flag else "0" for flag in covered + pred))
    counts["linked_output_span_coverage_pronoun_A_B"] = coverage
    tp, fp, fn = (counts[k] for k in ["tp", "fp", "fn"])
    counts["precision"] = round(tp / (tp + fp), 6) if tp + fp else None
    counts["recall"] = round(tp / (tp + fn), 6) if tp + fn else None
    counts["f1"] = round(2 * tp / (2 * tp + fp + fn), 6)
    return {"counts": counts, "per_id": observations}


def model_identity(backend, mode):
    model = {"backend": backend}
    if backend == "fcoref":
        if mode == "supplied":
            raise ValueError("the FCoref backend only accepts raw text")
        for name, digest in FC_FILES.items():
            check_file(Path(os.environ["FCOREF_MODEL_PATH"]) / name, digest)
        model.update(repo="biu-nlp/f-coref", revision=FC_REV, files=FC_FILES)
    if backend == "maverick":
        pins = json.loads(
            Path(__file__).with_name("maverick").joinpath("models.json").read_text()
        )
        reqs = Path(__file__).with_name("maverick") / "requirements-cpu-py312.txt"
        names = {"maverick-coref", "spacy", "tokenizers", "torch", "transformers"}
        required = dict(
            line.split("==")
            for line in reqs.read_text().splitlines()
            if line.partition("==")[0] in names
        )
        check = subprocess.run(
            [
                os.environ["MAVERICK_PYTHON"],
                "-c",
                "from importlib.metadata import version; import json,sys; print(json.dumps({n:version(n) for n in sys.argv[1:]}))",
                *sorted(names),
            ],
            text=True,
            check=True,
            stdout=subprocess.PIPE,
        )
        if json.loads(check.stdout) != required:
            raise ValueError(f"Maverick runtime differs from {reqs}")
        if (
            os.environ.get("TRANSFORMERS_CACHE", os.environ["HF_HUB_CACHE"])
            != os.environ["HF_HUB_CACHE"]
        ):
            raise ValueError("Maverick must use the verified HF_HUB_CACHE")
        check_file(os.environ["MAVERICK_MODEL_PATH"], pins["MODEL_SHA256"])
        cached = Path(os.environ["HF_HUB_CACHE"]) / (
            "models--" + pins["ENCODER"].replace("/", "--")
        )
        if (cached / "refs/main").read_text().strip() != pins["ENCODER_REVISION"]:
            raise ValueError("offline encoder revision mismatch")
        for name, digest in pins["ENCODER_FILES"].items():
            check_file(cached / "snapshots" / pins["ENCODER_REVISION"] / name, digest)
        model.update(
            repo="sapienzanlp/maverick-mes-ontonotes",
            revision=pins["MODEL_REVISION"],
            sha256=pins["MODEL_SHA256"],
            encoder=pins["ENCODER"],
            encoder_revision=pins["ENCODER_REVISION"],
            singletons=False,
        )
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("backend", choices=["mention-ranking", "fcoref", "maverick"])
    parser.add_argument("mode", choices=["predicted", "supplied"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    docs = panel(args.tsv)
    model = model_identity(args.backend, args.mode)
    payloads = [
        {
            "id": d,
            "text": text,
            "mentions": sorted(spans) if args.mode == "supplied" else None,
        }
        for d, text, spans, gold in docs
    ]
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "ANNO_NO_DOWNLOADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    run = subprocess.run(
        [args.binary.resolve(), args.backend],
        text=True,
        check=True,
        env=env,
        input="".join(json.dumps(r) + "\n" for r in payloads),
        stdout=subprocess.PIPE,
    )
    result = measure(
        docs, [json.loads(x) for x in io.StringIO(run.stdout)], args.backend
    )
    result.update(
        model=model,
        mode=args.mode,
        ids=[d[0] for d in docs],
        gold_pairs=["".join("1" if x else "0" for x in d[3]) for d in docs],
        split="gap-test.tsv",
        revision=REVISION,
        sha256=DATA_SHA,
    )
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
