"""Build validated, notice-bearing local Anno caches from the pinned native source index."""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from itertools import chain
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "scripts/data/offline-native-sources.json"
STAMP = (2026, 11, 28, 0, 0, 0)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def archived_originals(directory):
    sources = {}
    for artifact in sorted(directory.glob("anno-native-cache-*.zip")):
        with zipfile.ZipFile(artifact) as archive:
            for item in json.loads(archive.read("sources.json"))["inputs"]:
                sources[item["sha256"]] = artifact, item["original_path"]
                for notice in item["notices"]:
                    sources[notice["sha256"]] = artifact, notice["path"]
    return sources


def source_bytes(item, caches, online, archived=None):
    key, url = item["sha256"], item["url"]
    for cache in caches:
        for name in (key, digest(url.encode())):
            file = cache / name
            if file.is_file():
                raw = file.read_bytes()
                if digest(raw) != key:
                    raise ValueError(f"Cached source digest failed for {url}: {file}")
                return raw
    if archived and key in archived:
        artifact, member_name = archived[key]
        with zipfile.ZipFile(artifact) as archive:
            raw = archive.read(member_name)
        if digest(raw) != key:
            raise ValueError(
                f"Archived source digest failed for {artifact}/{member_name}"
            )
        return raw
    if not online:
        raise FileNotFoundError(
            f"Missing {key} for {url}. Supply its verified bytes via --cache or use --fetch."
        )
    request = urllib.request.Request(
        url, headers={"User-Agent": "Anno offline dataset source builder"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
    if digest(raw) != key:
        raise ValueError(f"Publisher digest changed for {url}")
    if caches:
        caches[0].mkdir(exist_ok=True, parents=True)
        with tempfile.TemporaryDirectory(dir=caches[0]) as tmp:
            verified = Path(tmp) / key
            verified.write_bytes(raw)
            os.replace(verified, caches[0] / key)
    return raw


def json_bytes(record):
    return (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode()


def counted(count, noun):
    return f"{count:,} {noun}{'' if count == 1 else 's'}"


def stats(anno, directory, dataset_id):
    env = {
        **os.environ,
        "ANNO_CACHE_DIR": str(directory),
        "ANNO_S3_CACHE": "0",
        "ANNO_NO_DOWNLOADS": "1",
        "NO_COLOR": "1",
    }
    # dataset info uses DatasetLoader::load, which has no publisher/S3 fallback.
    proc = subprocess.run(
        [anno, "dataset", "info", "--dataset", dataset_id],
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
    match = re.search(
        r"Loaded Statistics:\s+Sentences:\s*(\d+)\s+Tokens:\s*(\d+)\s+Entities:\s*(\d+)",
        proc.stdout,
    )
    if not match or int(match[1]) == 0:
        raise ValueError(
            f"{dataset_id} did not load: {proc.stdout[-500:]} {proc.stderr[-250:]}"
        )
    return dict(
        zip(
            ("sentences", "tokens", "native_labels"),
            map(int, match.groups()),
            strict=True,
        )
    )


def attribution_index(raw, field):
    rows = list(csv.DictReader(io.StringIO(raw.decode()), delimiter="\t"))
    if not rows or any(
        not r[field].startswith(
            ("https://en.wikipedia.org/wiki/", "http://en.wikipedia.org/wiki/")
        )
        for r in rows
    ):
        raise ValueError("Missing article attribution in a GAP row")
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(("case_id", "wikipedia_article_url"))
    writer.writerows((r["ID"], r[field]) for r in rows)
    return out.getvalue().encode()


def native_bytes(kind, original):
    """Build the adapter's parser input. The archive always retains the exact original separately."""
    if not kind:
        return original
    text = original.decode("utf-8")
    lines = []
    if kind == "agcner-jsonl":
        out = io.StringIO()
        tokens, tags = [], []
        for line in chain(io.StringIO(text), ("",)):
            line = line.removesuffix("\n").removesuffix("\r")
            if line.strip():
                match = re.fullmatch(r"(.)[ \t](O|[BI]-[^\s]+)", line, re.DOTALL)
                if not match:
                    raise ValueError("Invalid AgCNER publisher character/BIO pair")
                token, tag = match.groups()
                tokens.append(token)
                tags.append(tag)
            elif tokens:
                out.write(
                    json.dumps(
                        {"tokens": tokens, "ner_tags": tags},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                tokens, tags = [], []
        return out.getvalue().encode()
    if kind in {"histnero-rows", "masakhapos-rows"}:
        if kind == "histnero-rows":
            # Published class order, also retained verbatim in publisher-class-labels.md.
            labels = [
                "O",
                "B-PERS",
                "I-PERS",
                "B-ORG",
                "I-ORG",
                "B-LOC",
                "I-LOC",
                "B-PROD",
                "I-PROD",
                "B-DATE",
                "I-DATE",
            ]
            rows = json.loads(text)
            for row in rows:
                for token, label in zip(row["tokens"], row["ner_tags"], strict=True):
                    if (
                        not isinstance(token, str)
                        or type(label) is not int
                        or not 0 <= label < len(labels)
                    ):
                        raise ValueError("Invalid HistNERo token or class ID")
            features = [{"name": "ner_tags", "type": {"feature": {"names": labels}}}]
        else:
            rows, tokens, tags = [], [], []
            for line in [*text.splitlines(), ""]:
                if line.strip():
                    token, tag = line.split()
                    tokens.append(token)
                    tags.append("B-" + tag)
                elif tokens:
                    rows.append({"tokens": tokens, "tags": tags})
                    tokens, tags = [], []
            features = [{"name": "tags", "type": {"feature": {"dtype": "string"}}}]
        # The native reader validates coverage from the ordered indices and advertised total.
        cache = {
            "features": features,
            "num_rows_total": len(rows),
            "rows": [{"row_idx": i, "row": row} for i, row in enumerate(rows)],
        }
        return (
            json.dumps(cache, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode()
    if kind == "norne-bio":
        for line in text.splitlines():
            if not line.strip():
                lines.append("")
            elif not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) != 10:
                    raise ValueError("Unexpected NorNE CoNLL-U row")
                if not parts[0].isdigit():
                    continue
                labels = [
                    v.removeprefix("name=")
                    for v in parts[9].split("|")
                    if v.startswith("name=")
                ]
                if (
                    len(labels) != 1
                    or not re.fullmatch(r"O|[BI]-[^\s]+", labels[0])
                    or parts[1].split() != [parts[1]]
                ):
                    raise ValueError(
                        "NorNE token or name attribute cannot be projected faithfully"
                    )
                lines.append(f"{parts[1]}\t{labels[0]}")
    else:
        raise ValueError(f"Unknown native projection {kind}")
    return ("\n".join(lines) + "\n").encode()


def validate_index(index):
    seen = set()
    for data in index["datasets"]:
        key = data["id"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", key) or key in seen:
            raise ValueError(f"Invalid or duplicate dataset ID: {key}")
        seen.add(key)
        inputs = data["inputs"]
        variants = [item["variant"] for item in inputs]
        if sum(item["default"] for item in inputs) != 1 or len(set(variants)) != len(
            inputs
        ):
            raise ValueError(f"{key} needs unique variants and exactly one default")
        if not all(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", v) for v in variants):
            raise ValueError(f"{key} has an unsafe cache profile name")
        for item in inputs + data["notices"]:
            if not re.fullmatch("[a-f0-9]{64}", item["sha256"]) or not item[
                "url"
            ].startswith("https://"):
                raise ValueError(f"Invalid source URL or digest in {key}")


def select_index(index, dataset_id, variant):
    if not dataset_id:
        if variant:
            raise ValueError("--variant requires --dataset")
        return index
    datasets = {data["id"]: data for data in index["datasets"]}
    if dataset_id not in datasets:
        raise ValueError(
            f"Unknown --dataset {dataset_id}. Available: {', '.join(datasets)}"
        )
    data = datasets[dataset_id]
    inputs = data["inputs"]
    selection = {"dataset": dataset_id, "indexed_variants": len(inputs)}
    if variant:
        variants = {item["variant"]: item for item in inputs}
        if variant not in variants:
            raise ValueError(
                f"Unknown --variant {variant} for {dataset_id}. Available: {', '.join(variants)}"
            )
        source = variants[variant]
        # A single selected split must open via the archive's ordinary cache root.
        inputs = [{**source, "default": True, "index_default": source["default"]}]
        selection["variant"] = variant
    return {**index, "datasets": [{**data, "inputs": inputs}], "selection": selection}


def archive_stem(index, group):
    selection = index.get("selection")
    if not selection:
        return f"anno-native-cache-{group}"
    stem = f"anno-native-cache-dataset-{selection['dataset']}"
    return f"{stem}-{selection['variant']}" if "variant" in selection else stem


def member(archive, name, raw):
    if any(part in {"", ".", ".."} for part in name.split("/")) or "\\" in name:
        raise ValueError(f"Unsafe archive path {name}")
    if name in archive.NameToInfo:
        raise ValueError(f"Duplicate archive path {name}")
    info = zipfile.ZipInfo(name, STAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, raw, compresslevel=6)


def build(index, output, anno, caches, online, group, archived=None):
    datasets = [d for d in index["datasets"] if d["usage"] == group]
    version = (
        subprocess.check_output([anno, "--version"], text=True).strip().split()[-1]
    )
    output.mkdir(parents=True, exist_ok=True)
    stem = archive_stem(index, group)
    records = []
    manifests = {}
    with tempfile.TemporaryDirectory(dir=output) as work:
        scratch = Path(work)
        check = scratch / "check"
        files = check / "datasets"
        files.mkdir(parents=True)
        staged = scratch / "pack.zip"
        with zipfile.ZipFile(staged, "w", allowZip64=True) as archive:
            for data in datasets:
                dataset_id = data["id"]
                for notice in data["notices"]:
                    member(
                        archive,
                        f"notices/{dataset_id}/{notice['name']}",
                        source_bytes(notice, caches, online, archived),
                    )
                for item in data["inputs"]:
                    original = source_bytes(item, caches, online, archived)
                    raw = native_bytes(item.get("projection"), original)
                    if digest(raw) != item.get("cache_sha256", item["sha256"]):
                        raise ValueError(
                            f"Cache projection digest changed for {dataset_id}/{item['variant']}"
                        )
                    filename = dataset_id + ".cache"
                    local = files / filename
                    local.write_bytes(raw)
                    actual = stats(anno, check, dataset_id)
                    if item["counts"] != actual:
                        raise ValueError(
                            f"{dataset_id}/{item['variant']} native counts changed: {actual}"
                        )
                    root = (
                        "cache"
                        if item["default"]
                        else f"profiles/{dataset_id}/{item['variant']}"
                    )
                    name = f"{root}/datasets/{filename}"
                    member(archive, name, raw)
                    original_path = name
                    if item.get("projection"):
                        basename = (
                            item.get("original_filename") or item["url"].split("/")[-1]
                        )
                        original_path = (
                            f"originals/{dataset_id}/{item['variant']}/{basename}"
                        )
                        member(archive, original_path, original)
                    if item.get("article_url_column"):
                        member(
                            archive,
                            f"notices/{dataset_id}/articles-{item['variant']}.csv",
                            attribution_index(raw, item["article_url_column"]),
                        )
                    entry = {
                        "dataset_id": filename,
                        "source_url": data["registry_url"],
                        "resolved_url": item["url"],
                        "sha256": digest(raw),
                        "file_size": len(raw),
                        "downloaded_at": index["snapshot_date"],
                        "sentence_count": actual["sentences"],
                        "entity_count": actual["native_labels"],
                        "anno_version": version,
                    }
                    manifests.setdefault(root, {})[filename] = entry
                    records.append(
                        {
                            **{
                                k: data[k]
                                for k in (
                                    "id",
                                    "usage",
                                    "attribution",
                                    "license",
                                    "interpretation",
                                )
                            },
                            "notices": [
                                {
                                    "path": f"notices/{dataset_id}/{n['name']}",
                                    "sha256": n["sha256"],
                                    "url": n["url"],
                                }
                                for n in data["notices"]
                            ],
                            **item,
                            "path": name,
                            "original_path": original_path,
                            "source_bytes": len(original),
                            "cache_sha256": digest(raw),
                            "bytes": len(raw),
                        }
                    )
            for root, entries in sorted(manifests.items()):
                member(
                    archive,
                    f"{root}/datasets/manifest.json",
                    json_bytes({"version": 1, "entries": entries}),
                )
            catalog = {
                "schema": "anno.offline-native-cache/v1",
                "verification": "Original publisher digests and native cache digests are verified separately. DatasetLoader::load validates the counts for every cache input.",
                "version": version,
                "package": group,
                "pack_date": index["snapshot_date"],
                "dataset_ids": len(datasets),
                "file_count": len(records),
                "distinct_payloads": len({r["sha256"] for r in records}),
                "inputs": records,
            }
            if "selection" in index:
                catalog["selection"] = index["selection"]
            member(archive, "sources.json", json_bytes(catalog))
            member(archive, "README.md", readme(catalog).encode())
        artifact = output / (stem + ".zip")
        os.replace(staged, artifact)
        (output / (stem + ".sources.json")).write_bytes(json_bytes(catalog))
        (output / (stem + ".md")).write_text(readme(catalog))
        print(
            f"{artifact.name}: {counted(len(datasets), 'registry ID')}, {counted(len(records), 'verified input')}, {artifact.stat().st_size:,} compressed bytes"
        )
        if "selection" in catalog:
            for item in records:
                counts = item["counts"]
                print(
                    f"  {item['id']}/{item['variant']}: {counts['sentences']:,} sentences, {counts['tokens']:,} tokens, {counts['native_labels']:,} native labels"
                )
    return artifact


def verify(artifact, anno):
    """Restore one native profile at a time, including the exact manifest from the ZIP."""
    with zipfile.ZipFile(artifact) as archive, tempfile.TemporaryDirectory() as work:
        if len(archive.namelist()) != len(archive.NameToInfo):
            raise ValueError("Duplicate members in native ZIP")
        catalog = json.loads(archive.read("sources.json"))
        if catalog.get("schema") != "anno.offline-native-cache/v1":
            raise ValueError("Unsupported native cache archive schema")
        inputs = catalog["inputs"]
        if (
            not inputs
            or len({(item["id"], item["variant"]) for item in inputs}) != len(inputs)
            or len(inputs) != catalog["file_count"]
            or len({item["id"] for item in inputs}) != catalog["dataset_ids"]
            or len({item["sha256"] for item in inputs}) != catalog["distinct_payloads"]
        ):
            raise ValueError("Native archive counts disagree with its inputs")
        cache = Path(work)
        files = cache / "datasets"
        files.mkdir()
        checked = set()
        for item in catalog["inputs"]:
            if not re.fullmatch(
                r"[A-Za-z][A-Za-z0-9]*", item["id"]
            ) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", item["variant"]):
                raise ValueError("Invalid dataset ID or variant in native ZIP")
            expected_root = (
                "cache"
                if item["default"]
                else f"profiles/{item['id']}/{item['variant']}"
            )
            if item["path"] != f"{expected_root}/datasets/{item['id']}.cache":
                raise ValueError("Native cache path disagrees with its profile")
            manifest = archive.read(f"{expected_root}/datasets/manifest.json")
            filename = f"{item['id']}.cache"
            entry = json.loads(manifest)["entries"][filename]
            raw = archive.read(item["path"])
            if (
                digest(raw) != item["cache_sha256"]
                or entry["sha256"] != item["cache_sha256"]
                or len(raw) != entry["file_size"]
            ):
                raise ValueError(f"Corrupt native cache member {item['path']}")
            if digest(archive.read(item["original_path"])) != item["sha256"]:
                raise ValueError(f"Corrupt publisher copy {item['original_path']}")
            for notice in item["notices"]:
                if notice["path"] not in checked:
                    if digest(archive.read(notice["path"])) != notice["sha256"]:
                        raise ValueError(f"Corrupt rights notice {notice['path']}")
                    checked.add(notice["path"])
            (files / "manifest.json").write_bytes(manifest)
            (files / filename).write_bytes(raw)
            if stats(anno, cache, item["id"]) != item["counts"]:
                raise ValueError(f"Restored native counts differ for {item['path']}")
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP integrity failed for {bad}")
    print(
        f"Restored and verified {counted(catalog['file_count'], 'native input')} from {artifact.name}"
    )


def readme(catalog):
    restrictions = {
        "agcner": "The individual AgCNER split releases carry CC0 1.0. The three Figshare version-specific release notices and CC0 legal text accompany the originals.",
        "general": "Sources have individual attribution and share-alike terms. Keep their notices with copies.",
        "noncommercial": "Use these separately licensed data for noncommercial research. Each source retains its own attribution and share-alike conditions.",
        "annotated-collection": "NorNE retains the Norwegian Dependency Treebank terms. Keep the annotated collection, original works' source lists and license together. Third-party works may not be republished as stand-alone texts.",
    }[catalog["package"]]
    ids = ", ".join(sorted({r["id"] for r in catalog["inputs"]}))
    examples = {
        "agcner": "anno dataset eval --dataset AgCNER --task ner --model heuristic --cached-only --reports chinese-agricultural-ner.jsonl",
        "general": "anno dataset eval --dataset SECFilingsNER --task ner --model heuristic --cached-only --reports sec.jsonl",
        "noncommercial": "anno dataset eval --dataset MasakhaNER --task ner --model heuristic --cached-only --reports yoruba.jsonl",
        "annotated-collection": "anno dataset eval --dataset NorNE --task ner --model heuristic --cached-only --reports norne.jsonl",
    }
    example = "\n" + examples[catalog["package"]]
    scope = ids
    usage = 'Each ID has one selected release file in `cache/datasets`. Use `ANNO_CACHE_DIR="$PWD/profiles/ID/VARIANT"` for another split or language listed in `sources.json`.'
    interpretation = "`native_labels` is the entity count reported by the loader. A dependency treebank normally has one part-of-speech label per word. HIPE exposes coarse literal labels and its other annotation columns remain in the original TSV. GAP's token view returns whitespace-tokenized text with zero NER labels. Use the coreference view for its supplied pronoun and candidates. Consult each source's interpretation before comparing models."
    description = f"{catalog['dataset_ids']} registry IDs and {catalog['file_count']} release files"
    if "selection" in catalog:
        selection = catalog["selection"]
        dataset = selection["dataset"]
        variants = ", ".join(item["variant"] for item in catalog["inputs"])
        scope = f"{dataset}. Included {catalog['file_count']} of {selection['indexed_variants']} indexed variants ({variants})."
        restrictions = catalog["inputs"][0]["license"]
        description = (
            f"one registry ID and {counted(catalog['file_count'], 'release file')}"
        )
        lines = [
            "| Variant | Sentences | Tokens | Native labels |",
            "|---|---:|---:|---:|",
        ]
        for item in catalog["inputs"]:
            counts = item["counts"]
            lines.append(
                f"| {item['variant']} | {counts['sentences']:,} | {counts['tokens']:,} | {counts['native_labels']:,} |"
            )
        interpretation = (
            "\n".join(lines)
            + "\n\n`native_labels` is the entity count reported by the dataset loader."
        )
        if catalog["file_count"] == 1:
            usage = f"The {variants} variant is installed as the default at `{catalog['inputs'][0]['path']}`."
        if dataset == "GAP":
            example = "\nanno dataset eval --dataset GAP --task coref --model heuristic --cached-only"
        elif dataset in {
            "AgCNER",
            "HIPE2022",
            "HistNERo",
            "MasakhaNER",
            "MasakhaNER2",
            "NorNE",
            "SECFilingsNER",
        }:
            example = f"\nanno dataset eval --dataset {dataset} --task ner --model heuristic --cached-only --reports ner.jsonl"
        else:
            example = ""
    return f"""# Anno native offline cache

{restrictions}

The archive contains {description}. Each original is present in full. `sources.json` lists the source URL, split or language, publisher digest, native cache digest, rights notices, interpretation, and counts for every file. Where a format adapter is required, `original_path` points to the unchanged publisher file and `path` points to the native projection.

## Use the cache

Extract this ZIP into a dedicated directory. From that directory run an eval-enabled Anno build with the `--cached-only` dataset option.

```sh
export ANNO_CACHE_DIR="$PWD/cache" ANNO_S3_CACHE=0 ANNO_NO_DOWNLOADS=1
anno dataset info --dataset {catalog["inputs"][0]["id"]}{example}
```

{usage} Keep each archive in its own directory. Loading a new release in the same directory can replace a cache file.

## Scope

{scope}

{interpretation}
"""


def write_inventory(index, registry_file, output):
    """Distinguish packaged profiles, individually reviewed omissions and untested registry IDs."""
    packaged = {}
    for artifact in sorted(output.glob("anno-native-cache-*.zip")):
        with zipfile.ZipFile(artifact) as archive:
            for item in json.loads(archive.read("sources.json"))["inputs"]:
                packaged.setdefault(item["id"], []).append(
                    (artifact.name, item["variant"])
                )
    reviewed = index.get("other_reviewed_candidates", {})
    with (
        registry_file.open() as registry,
        (output / "native-scope.csv").open("w", newline="") as handle,
    ):
        fields = [
            "registry_id",
            "access",
            "native_status",
            "included_profiles",
            "archive",
            "reason",
            "registry_url",
        ]
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        for row in csv.DictReader(registry):
            key = row["id"]
            profiles = packaged.get(key, [])
            if profiles:
                status, reason = (
                    "native_cache",
                    "Profile variants and original-file digests are in the archive catalog.",
                )
            elif key in reviewed:
                status, reason = reviewed[key]["status"], reviewed[key]["reason"]
            elif row["access"] == "Local":
                status, reason = (
                    "local_only",
                    "Local fixtures and user-supplied corpora are outside these publisher caches.",
                )
            elif row["access"] in {
                "Registration",
                "ContactAuthors",
                "DependsOnOther",
                "NotYetReleased",
                "Deprecated",
            }:
                status, reason = (
                    "registry_access",
                    f"Registry access is {row['access']}. No original was included in these native caches.",
                )
            else:
                status, reason = (
                    "outside_native_selection",
                    "This registry ID was outside the native input selection. See the combined offline archive inventory for publisher-format availability.",
                )
            writer.writerow(
                dict(
                    zip(
                        fields,
                        [
                            key,
                            row["access"],
                            status,
                            len({variant for _, variant in profiles}),
                            " | ".join(sorted({a for a, _ in profiles})),
                            reason,
                            row["url"],
                        ],
                        strict=True,
                    )
                )
            )
    (output / "native-omissions.json").write_bytes(
        json_bytes({"reviewed": reviewed, "files": index.get("omitted_inputs", [])})
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "output", type=Path, help="Output directory, or a ZIP file with --verify-only"
    )
    ap.add_argument(
        "--anno",
        default="anno",
        help="Eval-enabled Anno executable, by default anno on PATH",
    )
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument(
        "--cache",
        type=Path,
        action="append",
        default=[],
        help="Reusable publisher byte cache (default: OUTPUT/source-cache). Fetches write to the first cache.",
    )
    ap.add_argument(
        "--dataset",
        help="Build only this registry ID and its associated rights notices from the pinned source index",
    )
    ap.add_argument(
        "--variant",
        help="Build one named split or language profile of --dataset, e.g. test for SECFilingsNER",
    )
    ap.add_argument(
        "--from-archives",
        type=Path,
        help="Read original publisher files and notices from a directory containing existing native ZIPs, for a network-free rebuild",
    )
    ap.add_argument(
        "--fetch",
        action="store_true",
        help="Permit digest-checked retrieval of missing pinned data and notices",
    )
    ap.add_argument(
        "--usage",
        choices=("general", "noncommercial", "annotated-collection", "agcner"),
        action="append",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="Reopen existing ZIPs, verify manifest and source digests, and load every restored input without network access",
    )
    ap.add_argument(
        "--inventory",
        type=Path,
        help="Also map every row of the registry coverage CSV to the native ZIPs present in the output directory",
    )
    a = ap.parse_args()
    anno = shutil.which(a.anno) or shutil.which(str(Path(a.anno).resolve()))
    if not anno:
        ap.error(
            f"Cannot find executable {a.anno!r}. Build anno-cli with --features eval and pass its path with --anno."
        )
    anno = Path(anno).resolve()
    if a.verify_only and (a.output.is_file() or a.output.suffix.lower() == ".zip"):
        if not a.output.is_file():
            ap.error(f"Cannot find native ZIP {a.output}")
        if a.usage or a.dataset or a.variant or a.inventory:
            ap.error(
                "A ZIP file specifies the verification scope. Use the output directory for selection flags or --inventory."
            )
        verify(a.output, anno)
        return
    index = json.loads(a.index.read_text())
    validate_index(index)
    try:
        index = select_index(index, a.dataset, a.variant)
    except ValueError as err:
        ap.error(str(err))
    groups = {data["usage"] for data in index["datasets"]}
    if a.usage and not set(a.usage) <= groups:
        ap.error(f"This source selection belongs to: {', '.join(sorted(groups))}")
    archived = (
        archived_originals(a.from_archives)
        if a.from_archives and not a.verify_only
        else None
    )
    caches = a.cache or [a.output / "source-cache"]
    for group in dict.fromkeys(a.usage or sorted(groups)):
        if a.verify_only:
            verify(a.output / (archive_stem(index, group) + ".zip"), anno)
        else:
            build(index, a.output, anno, caches, a.fetch, group, archived)
    if a.inventory:
        write_inventory(index, a.inventory, a.output)


if __name__ == "__main__":
    main()
