#!/usr/bin/env python3
"""
Download HuggingFace datasets and convert to local cache format.

Usage:
    uv run scripts/download_hf_datasets.py [--all] [--dataset DATASET] [--list]
"""

import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen

MAX_ARTIFACT_BYTES = 32 * 1024 * 1024


def get_cache_dir() -> Path:
    """Get the anno cache directory, matching Rust's env.rs logic.

    Priority:
    1. ANNO_CACHE_DIR environment variable
    2. Platform-specific default:
       - macOS: ~/Library/Caches/anno
       - Linux/other: $XDG_CACHE_HOME/anno or ~/.cache/anno
    """
    if custom := os.environ.get("ANNO_CACHE_DIR"):
        return Path(custom)

    if platform.system() == "Darwin":  # macOS
        return Path.home() / "Library/Caches/anno"
    else:  # Linux, Windows, etc.
        xdg_cache = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        return Path(xdg_cache) / "anno"


CACHE_DIR = get_cache_dir() / "datasets"

# These fields mirror `DatasetId::cache_filename()` and `DatasetId::download_url()`
# in crates/anno-eval/src/eval/dataset_registry.rs. The loader uses the filename as
# both the on-disk cache key and the manifest entry key.
DATASETS = {
    "conll2003_sample": {
        "identity_verified": True,
        "hf_id": "tner/conll2003",
        "config": "conll2003",
        "split": "test",
        "revision": "b18612dee0007b1f7129731dbf2f5f2ed4039ad3",
        "cache_filename": "CoNLL2003Sample.cache",
        "format": "conll",
        "tag_col": "tags",
        # Inverse of tner/conll2003's pinned dataset/label.json mapping.
        "label_names": [
            "O",
            "B-ORG",
            "B-MISC",
            "B-PER",
            "I-PER",
            "B-LOC",
            "I-ORG",
            "I-MISC",
            "I-LOC",
        ],
        "registry_source_url": "https://huggingface.co/datasets/tner/conll2003",
        "resolved_url": "https://huggingface.co/datasets/tner/conll2003/resolve/b18612dee0007b1f7129731dbf2f5f2ed4039ad3/dataset/test.json",
        "note": "Verified CoNLL-2003 test split from TNER",
    },
    "jnlpba": {
        "identity_verified": True,
        "hf_id": "tner/bionlp2004",
        "config": "bionlp2004",
        "split": "test",
        "revision": "8d75081cb3dae70b3f59db7e8d851dbc42f9275d",
        "cache_filename": "JNLPBA.cache",
        "format": "conll",
        "tag_col": "tags",
        "label_names": [
            "O",
            "B-DNA",
            "I-DNA",
            "B-protein",
            "I-protein",
            "B-cell_type",
            "I-cell_type",
            "B-cell_line",
            "I-cell_line",
            "B-RNA",
            "I-RNA",
        ],
        "registry_source_url": "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/JNLPBA/test.tsv",
        "resolved_url": "https://huggingface.co/datasets/tner/bionlp2004/resolve/8d75081cb3dae70b3f59db7e8d851dbc42f9275d/dataset/test.json",
        "note": "Verified BioNLP2004/JNLPBA test split from TNER",
    },
    "bc2gm_full": {
        "identity_verified": False,
        "hf_id": "disi-unibo-nlp/bc2gm",
        "config": None,
        "split": "test",
        "cache_filename": "BC2GMFull.cache",
        "format": "conll",
        "registry_source_url": "https://biocreative.bioinformatics.udel.edu/resources/biocreative-ii-corpus/",
        "note": "The HuggingFace BC2GM source is not verified as the registry's BC2GM Full artifact; use the registry corpus source",
    },
    "uner": {
        "identity_verified": False,
        "hf_id": "unimelb-nlp/wikiann",
        "config": "en",
        "split": "test",
        "cache_filename": "UNER.cache",
        "format": "wikiann_json",
        "registry_source_url": "https://github.com/UniversalNER/UNER",
        "note": "WikiANN is a separate corpus; use the UNER project's released test data",
    },
    "biomner": {
        "identity_verified": False,
        "hf_id": "tner/bionlp2004",
        "config": "bionlp2004",
        "split": "test",
        "cache_filename": "BioMNER.cache",
        "format": "conll",
        "registry_source_url": "https://huggingface.co/datasets/tner/bionlp2004",
        "note": "The source identifies itself as BioNLP2004/JNLPBA, not the registry's BioMNER method corpus; reconcile the registry before caching",
    },
    "craft": {
        "identity_verified": False,
        "hf_id": "bigbio/anat_em",
        "config": "anat_em_bigbio_kb",
        "split": "test",
        "cache_filename": "CRAFT.cache",
        "format": "conll",
        "registry_source_url": "https://github.com/UCDenver-ccp/CRAFT/archive/refs/heads/master.zip",
        "note": "AnatEM is a separate corpus; use the upstream CRAFT archive and its conversion path",
    },
    "msner": {
        "identity_verified": False,
        "cache_filename": "MSNER.cache",
        "format": "wikiann_json",
        "registry_source_url": "https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/ZTVMIX",
        "note": "No verified downloadable artifact; use the registry source rather than a synthetic placeholder",
    },
}


@dataclass(frozen=True)
class ConversionStats:
    sentence_count: int
    entity_count: int


def normalise_tags(tags, label_names):
    """Convert numeric dataset labels to the loader's string BIO labels."""
    if not tags:
        return tags

    numeric = [isinstance(tag, int) and not isinstance(tag, bool) for tag in tags]
    if any(numeric):
        if not all(numeric):
            raise ValueError("NER tags mix numeric and string labels")
        if not label_names:
            raise ValueError(
                "numeric NER tags require an explicit verified label mapping"
            )
        invalid = [tag for tag in tags if tag < 0 or tag >= len(label_names)]
        if invalid:
            raise ValueError(f"NER tag IDs outside verified label mapping: {invalid}")
        return [label_names[tag] for tag in tags]

    if not all(isinstance(tag, str) for tag in tags):
        raise ValueError(
            "NER tags must be strings or integer IDs with a verified mapping"
        )
    return tags


def entity_count(tags) -> int:
    """Count BIO/BILOU entity starts, matching the loader's sentence semantics."""
    count = 0
    index = 0
    while index < len(tags):
        tag = tags[index]
        if not isinstance(tag, str):
            index += 1
        elif tag.startswith("B-"):
            count += 1
            entity_type = tag.removeprefix("B-")
            index += 1
            while (
                index < len(tags)
                and isinstance(tags[index], str)
                and tags[index].startswith("I-")
                and tags[index].removeprefix("I-") == entity_type
            ):
                index += 1
        elif tag != "O" and not tag.startswith(("I-", "TAG_")):
            count += 1
            index += 1
            while index < len(tags) and tags[index] == tag:
                index += 1
        else:
            index += 1
    return count


def to_conll(
    examples,
    output_path: Path,
    token_col="tokens",
    tag_col="ner_tags",
    label_names=None,
) -> ConversionStats:
    """Convert to CoNLL format."""
    # Get label names if available
    if (
        label_names is None
        and hasattr(examples, "features")
        and tag_col in examples.features
    ):
        feat = examples.features[tag_col]
        if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
            label_names = feat.feature.names

    converted = []
    for ex in examples:
        tokens = ex.get(token_col, [])
        tags = normalise_tags(ex.get(tag_col, []), label_names)
        if len(tokens) != len(tags):
            raise ValueError(
                f"token/tag length mismatch: {len(tokens)} tokens, {len(tags)} tags"
            )
        converted.append((tokens, tags))

    sentence_count = 0
    entities = 0
    with output_path.open("w", encoding="utf-8") as f:
        for tokens, tags in converted:
            written = len(tokens)
            for tok, tag in zip(tokens, tags):
                f.write(f"{tok}\t{tag}\n")
            if written:
                f.write("\n")
                sentence_count += 1
                entities += entity_count(tags[:written])
    return ConversionStats(sentence_count, entities)


def to_json(examples, output_path: Path, label_names=None) -> ConversionStats:
    """Convert to JSON format."""
    if (
        label_names is None
        and hasattr(examples, "features")
        and "ner_tags" in examples.features
    ):
        feat = examples.features["ner_tags"]
        if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
            label_names = feat.feature.names

    data = []
    for ex in examples:
        tokens = ex.get("tokens", [])
        tags = normalise_tags(ex.get("ner_tags", []), label_names)
        if len(tokens) != len(tags):
            raise ValueError(
                f"token/tag length mismatch: {len(tokens)} tokens, {len(tags)} tags"
            )
        data.append(
            {
                "text": ex.get("text", " ".join(tokens)),
                "tokens": tokens,
                "ner_tags": tags,
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    valid = [
        item
        for item in data
        if item["tokens"] and len(item["tokens"]) == len(item["ner_tags"])
    ]
    return ConversionStats(
        len(valid), sum(entity_count(item["ner_tags"]) for item in valid)
    )


def write_manifest_entry(
    output_path: Path, config: dict, stats: ConversionStats
) -> None:
    """Record the cache artifact in the Rust loader's manifest schema."""
    manifest_path = output_path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"version": 1, "entries": {}}

    content = output_path.read_bytes()
    manifest.setdefault("entries", {})[config["cache_filename"]] = {
        "dataset_id": config["cache_filename"],
        # The loader compares this logical registry URL before accepting the cache.
        "source_url": config["registry_source_url"],
        # Preserve the exact pinned artifact whose bytes were converted.
        "resolved_url": config["resolved_url"],
        "sha256": sha256(content).hexdigest(),
        "file_size": len(content),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "sentence_count": stats.sentence_count,
        "entity_count": stats.entity_count,
        "anno_version": "unknown",
    }
    temp_path = manifest_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temp_path.replace(manifest_path)


def load_pinned_json_artifact(url: str) -> list[dict]:
    """Fetch a bounded, pinned JSON array or JSONL artifact without executing a dataset script."""
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()
        if status != 200:
            raise ValueError(f"pinned artifact returned HTTP {status}")

        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError as error:
                raise ValueError(
                    "pinned artifact has an invalid Content-Length"
                ) from error
            if declared_size < 0 or declared_size > MAX_ARTIFACT_BYTES:
                raise ValueError(
                    f"pinned artifact exceeds {MAX_ARTIFACT_BYTES} byte limit"
                )

        content = response.read(MAX_ARTIFACT_BYTES + 1)
    if len(content) > MAX_ARTIFACT_BYTES:
        raise ValueError(f"pinned artifact exceeds {MAX_ARTIFACT_BYTES} byte limit")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in content.splitlines() if line.strip()]

    if not isinstance(parsed, list) or not all(isinstance(row, dict) for row in parsed):
        raise ValueError("pinned artifact must be a JSON array or JSONL objects")
    return parsed


def download_dataset(name: str, config: dict) -> bool:
    """Download single dataset."""
    if not config.get("identity_verified"):
        print(f"  [REFUSE] {name}: {config['note']}")
        return False

    output_path = CACHE_DIR / config["cache_filename"]

    if output_path.exists():
        print(f"  [SKIP] {name}: exists at {output_path}")
        return True

    hf_id = config["hf_id"]
    cfg = config.get("config")
    split = config.get("split", "test")

    print(f"  [DOWNLOAD] {name}: {hf_id} ({cfg or 'default'}, {split})")
    if "note" in config:
        print(f"    Note: {config['note']}")

    try:
        ds = load_pinned_json_artifact(config["resolved_url"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if config["format"] == "wikiann_json":
            stats = to_json(ds, output_path, label_names=config.get("label_names"))
        else:
            stats = to_conll(
                ds,
                output_path,
                tag_col=config.get("tag_col", "ner_tags"),
                label_names=config.get("label_names"),
            )
        write_manifest_entry(output_path, config, stats)

        print(f"  [OK] {output_path} ({len(ds)} examples)")
        return True
    except Exception as e:  # noqa: BLE001 - CLI boundary reports dataset library errors.
        print(f"  [ERROR] {name}: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name, cfg in DATASETS.items():
            hf = cfg.get("hf_id", "no verified HuggingFace source")
            note = cfg.get("note", "")
            print(f"  {name}: {hf}" + (f" ({note})" if note else ""))
        return

    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"Unknown: {args.dataset}. Use --list to see available.")
            return
        items = [(args.dataset, DATASETS[args.dataset])]
    elif args.all:
        items = list(DATASETS.items())
    else:
        print("Use --all or --dataset NAME")
        return

    ok, fail = 0, 0
    for name, cfg in items:
        if download_dataset(name, cfg):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} ok, {fail} failed")


if __name__ == "__main__":
    main()
