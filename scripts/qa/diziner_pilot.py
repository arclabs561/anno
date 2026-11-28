# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.3.2", "pandas==2.3.1", "scikit-learn==1.7.1"]
# ///
"""Prepare a local CoNLL schema-review packet using DiZiNER's disagreement stage."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

UPSTREAM = "2577f8ce7f06f0554e88b1931f9e0c9626c24f17"
DATA_SHA = "c8869cec4f419e7b635843aae285990581bf4685b7193c81696d87e1939f0c63"
DIGESTS = {
    "disagreement_analysis.py": "bef24acfffd82843cc83db45df7a36116489887ba3bf0f7738be2a561917aa0e",
    "utils_disagreement.py": "ce960aaf403c79270ab6a51dc81aa142d065f320025fe300f0c9cdfea6a9bdce",
    "experiments_settings/crossner_conll2003_default_config.json": "d3ef3df44bca715425eed2ecd8833918541b4fd4c415e684962c4a3d0ff7e3b5",
}


def verify(path, sha):
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError(f"Unexpected contents of {path}")


def tags_for_entities(tokens, entities, scheme):
    starts, ends = {}, {}
    pos = 0
    labels = ["O"] * len(tokens)
    text = " ".join(tokens)
    for i, word in enumerate(tokens):
        starts[pos] = i
        pos += len(word)
        ends[pos] = i
        pos += 1
    for ent in entities:
        label, start, end = ent["type"], ent["start"], ent["end"]
        if label not in scheme:
            continue
        if start not in starts or end not in ends or text[start:end] != ent["text"]:
            raise ValueError("entity cannot be aligned to source token boundaries")
        a, b = starts[start], ends[end] + 1
        if any(t != "O" for t in labels[a:b]):
            raise ValueError(
                "overlapping entities cannot be projected onto one BIO stream"
            )
        labels[a:b] = ["B-" + label] + ["I-" + label] * (b - a - 1)
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--anno", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    verify(args.dev, DATA_SHA)
    for name, sha in DIGESTS.items():
        verify(args.upstream / name, sha)
    sys.path.insert(0, str(args.upstream.resolve()))
    from disagreement_analysis import analyze_ner_disagreement_dataset

    config = json.loads(
        (
            args.upstream
            / "experiments_settings/crossner_conll2003_default_config.json"
        ).read_text()
    )
    scheme = config["ner_scheme"]
    dataset = [
        [line.split("\t")[0] for line in block.splitlines()]
        for block in args.dev.read_text().strip().split("\n\n")
    ]
    ids = sorted(
        range(len(dataset)),
        key=lambda i: hashlib.sha256(f"diziner-anno-pilot/{i}".encode()).digest(),
    )[:20]
    inputs = "".join(
        json.dumps({"id": str(i), "text": " ".join(dataset[i])}) + "\n" for i in ids
    )
    pilot = [{"tokens": dataset[i], "labels_by_models": {}} for i in ids]
    excluded = {}
    for model in ("bert-onnx", "heuristic"):
        result = subprocess.run(
            [
                str(args.anno.resolve()),
                "batch",
                "--stdin",
                "--model",
                model,
                "--format",
                "jsonl",
                "--quiet",
            ],
            input=inputs,
            text=True,
            capture_output=True,
            check=True,
            env={**os.environ, "ANNO_NO_DOWNLOADS": "1"},
        )
        for i, row, line in zip(ids, pilot, result.stdout.splitlines(), strict=True):
            prediction = json.loads(line)
            if str(i) != prediction.get("id") or prediction.get("model") != model:
                raise ValueError(f"Unexpected {model} batch output at {i}")
            try:
                row["labels_by_models"][model] = tags_for_entities(
                    row["tokens"], prediction["entities"], scheme
                )
            except ValueError as exc:
                excluded.setdefault(str(i), []).append(f"{model}: {exc}")
    usable = [
        (i, row) for i, row in zip(ids, pilot, strict=True) if str(i) not in excluded
    ]
    result = analyze_ner_disagreement_dataset(
        [row for _, row in usable], save_results=False, use_gold_standard=False
    )
    sections = []
    for block in result["hotspots"]:
        i, row = usable[block["sentence_id"]]
        a, b = block["start"], block["end"] + 1
        annotations = {k: v[a:b] for k, v in row["labels_by_models"].items()}
        if len(set(map(tuple, annotations.values()))) == 1:
            continue
        sections.append(
            {
                "source_row": i,
                "tokens": row["tokens"],
                "focus": [a, b],
                "annotations": annotations,
                "u_star_mean": block["U_star_mean"],
            }
        )
    packet = {
        "upstream_revision": UPSTREAM,
        "dev_sha256": DATA_SHA,
        "source_rows": ids,
        "excluded": excluded,
        "schema": scheme,
        "task_goal": config["final_task_goal"],
        "weights": result["weights_used"],
        "global_thresholds": result["global_thresholds"],
        "summary": result["dataset_summary"],
        "review_blocks": sections,
    }
    args.output.write_text(
        json.dumps(packet, indent=2, ensure_ascii=False, default=lambda x: x.item())
        + "\n"
    )
    print(
        f"Prepared {len(sections)} disagreement blocks from {len(usable)} pilot rows."
    )


if __name__ == "__main__":
    main()
