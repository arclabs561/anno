#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Build and replay the pinned IBM ViRA dialog act benchmark."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

SPLITS = {"train": 571, "validation": 71, "test": 72}
DATASET = "ViraDialogActsLive"
TASK = "speech-act-classification"
BACKEND = "gliner2_fastino"
MODEL = "jugaadsrl/gliner2-multi-v1-onnx"
MODEL_REV = "4241d7c66b648e618c89c150bf4cf418d2f83159"
SEED = 45929075593
STEM = "vira-dialog-acts-offline"
RECIPE = Path(__file__).parent / "fixtures/vira-dialog-act-sources.json"
if not RECIPE.is_file():
    RECIPE = Path(__file__).parent / "source-manifest.json"
ACTS = "ibm-classifier/dialog_act_dataset/dialog_acts.csv"
README = """# IBM ViRA Dialog Acts (Live)

Each of the 714 English user turns has one of seven dialog acts. The official splits contain 571 training, 71 validation and 72 held-out test turns. In `ibm-classifier/dialog_act_dataset/dialog_acts.csv` row order defines the label indices. The split CSVs also spell out each label. During packing all 714 ordered (text, label ID) pairs are compared to the pinned Hugging Face Parquet partitions. `native-inputs/` holds complete JSON inputs accepted by Anno's ViRA loader.

`source-manifest.json` gives immutable URLs and SHA-256 digests for both IBM releases. `ibm-classifier/LICENSE` contains the source repository's Apache-2.0 notice. The Hugging Face dataset card leaves its license field unspecified. The pinned ONNX conversion likewise has no license field on its card. Model weights are a separate prerequisite for inference.

Verify every file with Python 3, without a network connection or additional packages.

```sh
python3 vira_speech_act.py verify .
```

To reproduce the evaluation offline, build Anno with the `eval` and `gliner2-fastino` features. Set `HF_HOME` to a cache containing `jugaadsrl/gliner2-multi-v1-onnx@4241d7c66b648e618c89c150bf4cf418d2f83159` with its `fp32_v2` tokenizer and graphs. The command stages a fresh dataset cache and tells both Anno and Hugging Face to stay offline.

```sh
python3 vira_speech_act.py replay . --split test --anno /path/to/anno --output test.json
```

Repeat with `--split validation` and `--split train` to write `validation.json` and `train.json`. The script can then reconstruct the published evidence.

```sh
python3 vira_speech_act.py summarize . --reports . --output vira-speech-act.json
```
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def record(path, data):
    return {"path": path, "bytes": len(data), "sha256": digest(data)}


def check(data, expected):
    require(
        digest(data) == expected["sha256"] and len(data) == expected["bytes"],
        f"Bytes differ for {expected['path']}",
    )


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))


def native_input(rows):
    # Reproduce the viewer response for a single page. Training spans multiple pages.
    features = [
        {"feature_idx": i, "name": key, "type": {"dtype": typ, "_type": "Value"}}
        for i, (key, typ) in enumerate([("text", "string"), ("label", "int64")])
    ]
    doc = {
        "features": features,
        "rows": [
            {"row_idx": i, "row": row, "truncated_cells": []}
            for i, row in enumerate(rows)
        ],
    }
    if len(rows) <= 100:
        doc.update(num_rows_total=len(rows), num_rows_per_page=100, partial=False)
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode()


def packed_files(recipe_path, sources):
    # PyArrow is only required to audit the two publishers' source formats when packing.
    import pyarrow.parquet as pq

    recipe = json.loads(recipe_path.read_bytes())
    files = {
        "source-manifest.json": recipe_path.read_bytes(),
        "README.md": README.encode(),
        "vira_speech_act.py": Path(__file__).read_bytes(),
    }
    for item in recipe["files"]:
        if sources is None:
            with urlopen(item["url"], timeout=60) as response:
                data = response.read()
        else:
            data = (sources / item["path"]).read_bytes()
        check(data, item)
        files[item["path"]] = data
    labels = [row["dialog_act"] for row in csv_rows(files[ACTS])]
    require(
        len(labels) == len(set(labels)) == 7, "Expected seven distinct IBM dialog acts"
    )
    split_paths = {}
    for split, total in SPLITS.items():
        source = f"ibm-classifier/dialog_act_dataset/{split}.csv"
        (parquet,) = [p for p in files if p.startswith(f"publisher/data/{split}-")]
        source_rows = csv_rows(files[source])
        rows = [
            {"text": row["sentence"], "label": int(row["label_idx"])}
            for row in source_rows
        ]
        require(
            len(rows) == total
            and all(
                0 <= row["label"] < len(labels) and labels[row["label"]] == raw["label"]
                for row, raw in zip(rows, source_rows)
            ),
            f"IBM label dictionary or row count differs on {split}",
        )
        require(
            rows == pq.read_table(io.BytesIO(files[parquet])).to_pylist(),
            f"Ordered text or label differs between publishers on {split}",
        )
        native = f"native-inputs/{split}.json"
        files[native] = native_input(rows)
        split_paths.update(dict.fromkeys([source, parquet, native], split))
    records = []
    for path, data in sorted(files.items()):
        item = record(path, data)
        if path in split_paths:
            item.update(split=split_paths[path], examples=SPLITS[split_paths[path]])
        records.append(item)
    manifest = {
        "schema": "anno.vira-offline/v1",
        "dataset": DATASET,
        "splits": SPLITS,
        "labels": labels,
        "files": records,
    }
    files["manifest.json"] = encoded(manifest)
    return files, manifest


def pack(output, sources, recipe):
    files, manifest = packed_files(recipe, sources)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{STEM}.zip"
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        target = Path(tmp) / archive.name
        with ZipFile(target, "w", compression=ZIP_DEFLATED) as z:
            for path, data in sorted(files.items()):
                info = ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o100644 << 16
                info.compress_type = ZIP_DEFLATED
                z.writestr(info, data)
        with ZipFile(target) as z:
            require(
                z.testzip() is None and set(z.namelist()) == set(files),
                "Corrupt or incomplete data archive",
            )
        manifest["archive"] = record(archive.name, target.read_bytes())
        manifest["files"].append(record("manifest.json", files["manifest.json"]))
        sidecar = output / f"{STEM}-manifest.json"
        (Path(tmp) / sidecar.name).write_bytes(encoded(manifest))
        target.replace(archive)
        (Path(tmp) / sidecar.name).replace(sidecar)
    print(f"{manifest['archive']['sha256']}  {archive}")


def verify(root):
    manifest = json.loads((root / "manifest.json").read_bytes())
    require(
        manifest["dataset"] == DATASET and manifest["splits"] == SPLITS,
        "Unexpected dataset or split sizes",
    )
    for item in manifest["files"]:
        check((root / item["path"]).read_bytes(), item)
    for split, n in SPLITS.items():
        rows = json.loads((root / f"native-inputs/{split}.json").read_bytes())["rows"]
        require(
            len(rows) == n
            and all(
                r["row_idx"] == i
                and r["row"]["text"]
                and 0 <= r["row"]["label"] < len(manifest["labels"])
                for i, r in enumerate(rows)
            ),
            f"Invalid native input on {split}",
        )
    return manifest


def replay(root, split, anno, output):
    manifest = verify(root)
    data = (root / f"native-inputs/{split}.json").read_bytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="anno-vira-") as tmp:
        cache = Path(tmp) / "datasets"
        cache.mkdir()
        (cache / f"{DATASET}.cache").write_bytes(data)
        env = dict(
            os.environ,
            ANNO_CACHE_DIR=tmp,
            ANNO_EVAL_HISTORY=f"{tmp}/history.jsonl",
            ANNO_NO_DOWNLOADS="1",
            HF_HUB_OFFLINE="1",
        )
        raw = Path(tmp) / "result.json"
        subprocess.run(
            [
                anno,
                "benchmark",
                "--tasks",
                TASK,
                "--datasets",
                DATASET,
                "--backends",
                BACKEND,
                "--max-examples",
                str(SPLITS[split]),
                "--seed",
                str(SEED),
                "--cached-only",
                "--output-json",
                str(raw),
            ],
            env=env,
            check=True,
        )
        result = json.loads(raw.read_bytes())
        split_score(result, split, data, manifest["labels"])
        output.write_bytes(raw.read_bytes())


def split_score(raw, split, data, labels):
    (result,) = raw["results"]
    p = result["provenance"]
    require(
        result["success"]
        and result["backend"] == BACKEND
        and result["dataset"] == DATASET
        and result["task"] == "SpeechActClassification"
        and result["seed"] == SEED,
        f"Unexpected or unsuccessful benchmark on {split}",
    )
    n = SPLITS[split]
    require(
        p["backend"]["configured_labels"] == labels
        and result["num_examples"] == n
        and p["dataset"]["content_sha256"] == digest(data)
        and p["evaluated_sample"]["unit_count"] == n
        and p["evaluated_sample"]["gold_item_count"] == n,
        f"Wrong input or coverage on {split}",
    )
    m = result["metrics"]
    support = Counter(labels[r["row"]["label"]] for r in json.loads(data)["rows"])
    counts = {}
    for label in labels:
        gold, predicted, correct = (
            m[f"class/{label}/{key}"] for key in ("gold", "predicted", "correct")
        )
        require(
            all(v >= 0 and int(v) == v for v in (gold, predicted, correct))
            and gold == support[label]
            and 0 < gold
            and correct <= min(gold, predicted),
            f"Invalid native counts for {split}/{label}",
        )
        counts[label] = {
            "gold": int(gold),
            "predicted": int(predicted),
            "correct": int(correct),
            "f1": 2 * correct / (gold + predicted),
        }
    correct = sum(r["correct"] for r in counts.values())
    abstained = n - sum(r["predicted"] for r in counts.values())
    macro = sum(r["f1"] for r in counts.values()) / len(labels)
    require(
        abstained >= 0
        and correct == m["correct"]
        and abstained == m["abstained"]
        and math.isclose(correct / n, m["accuracy"])
        and math.isclose(macro, m["macro_f1"])
        and math.isclose(macro, m["f1"]),
        f"Native and recomputed metrics differ on {split}",
    )
    artifacts = p["backend"]["model_artifacts"]
    require(
        artifacts["model_id"] == MODEL
        and all(
            MODEL_REV in Path(f["path"]).parts and f["sha256"]["status"] == "known"
            for f in artifacts["files"]
        ),
        "Unpinned model",
    )
    model_files = {
        Path(f["path"]).name: f["sha256"]["value"] for f in artifacts["files"]
    }
    require(len(model_files) == 9, "Expected tokenizer and eight graphs")
    return {
        "examples": n,
        "correct": correct,
        "abstained": abstained,
        "accuracy": correct / n,
        "macro_f1": macro,
        "input_sha256": digest(data),
        "by_label": counts,
    }, model_files


def summarize(root, reports, output):
    bundle = verify(root)
    recipe = json.loads((root / "source-manifest.json").read_bytes())
    labels = bundle["labels"]
    splits = {}
    model_files = None
    for split in SPLITS:
        data = (root / f"native-inputs/{split}.json").read_bytes()
        score, files = split_score(
            json.loads((reports / f"{split}.json").read_bytes()), split, data, labels
        )
        require(
            model_files is None or files == model_files,
            "Model files changed between splits",
        )
        model_files = files
        splits[split] = score
    majority = max(labels, key=lambda l: splits["train"]["by_label"][l]["gold"])
    for score in splits.values():
        n, correct = score["examples"], score["by_label"][majority]["gold"]
        score["training_majority"] = {
            "label": majority,
            "correct": correct,
            "accuracy": correct / n,
            "macro_f1": (2 * correct / (n + correct)) / len(labels),
        }
    evidence = {
        "schema": "anno.vira-speech-act/v1",
        "dataset": DATASET,
        "task": TASK,
        "backend": BACKEND,
        "seed": SEED,
        "unit": "user turns",
        "labels": labels,
        "input_sources": {
            "hub": recipe["repository"],
            "revision": recipe["revision"],
            "ibm": recipe["verified_against"]["repo"],
            "ibm_revision": recipe["verified_against"]["revision"],
        },
        "model": {"id": MODEL, "revision": MODEL_REV, "sha256_by_file": model_files},
        "method": "Fastino's single-label prediction uses the highest of the seven IBM dialog act scores. An all-zero result abstains. Each abstention counts as an incorrect turn and a false negative for its gold class.",
        "macro_f1": "Mean of seven per-class values 2 × correct / (gold + predicted), including classes with zero recall. Every label has gold support in each full split.",
        "split_roles": {
            "test": "official held-out test",
            "validation": "development",
            "train": "training-set diagnostic and source of the majority baseline",
        },
        "splits": splits,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded(evidence))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    build = subs.add_parser(
        "pack",
        help="Verify IBM CSV against Hub Parquet and build the archive (requires PyArrow)",
    )
    build.add_argument(
        "--sources",
        type=Path,
        help="Read the pinned source tree locally instead of downloading",
    )
    build.add_argument("--recipe", type=Path, default=RECIPE)
    build.add_argument("--output", type=Path, required=True)
    for command in ("verify", "replay", "summarize"):
        sub = subs.add_parser(command)
        sub.add_argument("root", type=Path, help="Extracted data archive")
        if command in ("summarize", "replay"):
            sub.add_argument("--output", type=Path, required=True)
        if command == "replay":
            sub.add_argument("--anno", default="anno")
            sub.add_argument("--split", choices=SPLITS, default="test")
        if command == "summarize":
            sub.add_argument(
                "--reports",
                type=Path,
                required=True,
                help="Directory containing train.json, validation.json and test.json from anno benchmark",
            )
    args = p.parse_args()
    if args.command == "pack":
        pack(args.output, args.sources, args.recipe)
    elif args.command == "verify":
        bundle = verify(args.root)
        for split in SPLITS:
            rows = json.loads((args.root / f"native-inputs/{split}.json").read_bytes())[
                "rows"
            ]
            support = Counter(bundle["labels"][r["row"]["label"]] for r in rows)
            print(f"{split} ({len(rows)}): {dict(sorted(support.items()))}")
        print(f"Verified {len(bundle['files'])} files and {sum(SPLITS.values())} turns")
    elif args.command == "replay":
        replay(args.root, args.split, args.anno, args.output)
    else:
        summarize(args.root, args.reports, args.output)


if __name__ == "__main__":
    main()
