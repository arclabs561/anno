#!/usr/bin/env python3
"""Pack reviewed, anonymously distributed HF releases from Anno's measured split ledger.

    uv run --with pyarrow==26.0.1 scripts/pack_hub_datasets.py OUT --only WIESP2022NER

Set ANNO_HF_SURVEY_CACHE for resumable downloads of content and pinned Hub manifests.
Existing ZIPs can be validated offline using their bundled verify.py and Python alone.
"""

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from xml.etree import ElementTree
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, unquote, urlsplit

import pyarrow.compute as pc
import pyarrow.parquet as pq

import survey_hf_releases as hub

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/eval/dataset-hf-splits.csv"


@dataclass(frozen=True)
class Release:
    ids: tuple[str, ...]
    terms: str
    scope: str
    splits: tuple[str, ...] = ()
    original: str = ""
    extras: tuple[str, ...] = ()
    omit: tuple[str, ...] = ()


# Source-specific scope and terms are reviewed independently of the file readers.
SOURCE_RULES = json.loads((ROOT / "scripts/qa/hub_pack_sources.json").read_text(encoding="utf-8"))
RELEASES = {name: Release(**spec) for name, spec in SOURCE_RULES["releases"].items()}
ORIGINAL_NOTICES = SOURCE_RULES["original_notices"]

VERIFIER = '''"""Verify all extracted payloads and source documents using Python's standard library."""
import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
for row in manifest['files']:
    path = root / row['path']
    with path.open('rb') as stream:
        observed = hashlib.file_digest(stream, 'sha256').hexdigest()
    if path.stat().st_size != row['bytes'] or observed != row['sha256']:
        raise SystemExit(f"File differs: {row['path']}")
print(f"Verified {len(manifest['files'])} files in {manifest['release']}.")
'''


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def source_path(url, repo, revision):
    parsed = urlsplit(url)
    if parsed.hostname != "huggingface.co":
        raise ValueError(f"Unexpected release host: {url}")
    prefix = f"/datasets/{repo}/resolve/{revision}/"
    if not parsed.path.startswith(prefix):
        raise ValueError(f"Unexpected release revision: {url}")
    return unquote(parsed.path[len(prefix) :])


def safe_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe source path: {name}")
    return name


def partition_files(row, spec, manifest):
    config, split = row["config"], row["split"]
    if spec.original == "wiesp":
        name = {"train": "TRAINING", "validation": "VALIDATION", "test": "TESTING"}[split]
        return [f"WIESP2022-NER-{name}.jsonl"]
    if spec.original == "mewslix":
        return [f"wikinews_mentions/{config}/{split}.jsonl"]
    if spec.original == "multinerd":
        return [config]
    if row["payload_url"]:
        revision = row["parquet_revision"] or row["source_revision"]
        return [source_path(row["payload_url"], row["repository"], revision)]
    prefix = f"{config}/{split}/"
    return sorted(
        e["rfilename"]
        for e in manifest["siblings"]
        if re.fullmatch(re.escape(prefix) + r"[^/]+\.parquet", e["rfilename"])
    )


def measure(path, kind):
    if kind == ".gz":
        count = 0
        fields = set()
        with tarfile.open(path, mode="r|gz") as archive:
            for item in archive:
                if item.isfile() and item.name.endswith(".json"):
                    with archive.extractfile(item) as source:
                        patent = json.load(source)
                    if not patent["application_number"] or not isinstance(
                        patent["abstract"], str
                    ):
                        raise ValueError("Invalid patent application in the January 2016 archive")
                    fields.update(patent)
                    count += 1
        return count, {
            "format": "gzip tar containing JSON patent records",
            "columns": sorted(fields),
        }
    if kind == ".tsv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            fields = reader.fieldnames
            count = 0
            for record in reader:
                if (
                    not record["query-id"]
                    or not record["corpus-id"]
                    or not 0 <= float(record["score"]) <= 5
                ):
                    raise ValueError("Invalid ACORD relevance judgment")
                count += 1
        return count, {"format": "TSV", "columns": fields}
    if kind == ".parquet":
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        metadata = schema.metadata or {}
        features = json.loads(metadata.get(b"huggingface", b"{}")).get("info", {}).get("features")
        columns = {field.name: str(field.type) for field in schema}
        pairs = next(
            (
                (word, tag)
                for word, tag in [
                    ("tokens", "ner_tags"),
                    ("tokens", "tags"),
                    ("mbert_text_tokens", "mbert_bio_labels"),
                    ("words", "ner"),
                ]
                if word in columns and tag in columns
            ),
            None,
        )
        total = 0
        for batch in parquet.iter_batches(batch_size=8192):
            total += batch.num_rows
            if pairs:
                words, tags = (batch.column(batch.schema.get_field_index(col)) for col in pairs)
                if pc.any(
                    pc.not_equal(pc.list_value_length(words), pc.list_value_length(tags))
                ).as_py():
                    raise ValueError(f"Unequal token/tag lengths in {path.name}")
        if total != parquet.metadata.num_rows:
            raise ValueError(f"Incomplete Parquet partition: {path}")
        return total, {"columns": columns, "features": features}
    if kind == ".conll":
        with path.open("rb") as stream:
            count, tokens, tagged = hub.conll_counts(stream)
        return count, {
            "format": "CoNLL sentence blocks",
            "tokens": tokens,
            "tagged_tokens": tagged,
        }
    if kind == ".jsonl":
        total, columns = 0, set()
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or not record:
                    raise ValueError(f"Expected a JSON object in {path}")
                columns.update(record)
                if (
                    "tokens" in record
                    and "ner_tags" in record
                    and len(record["tokens"]) != len(record["ner_tags"])
                ):
                    raise ValueError(f"Unequal token/tag lengths in {path}")
                total += 1
        return total, {"columns": sorted(columns), "format": "JSON Lines"}
    raise ValueError(f"Unexpected data format {kind}")


def member(out, name, content):
    info = zipfile.ZipInfo(safe_path(name))
    info.compress_type = (
        zipfile.ZIP_STORED
        if name.endswith((".gz", ".zip", ".docx", ".xlsx"))
        else zipfile.ZIP_DEFLATED
    )
    info.external_attr = 0o100644 << 16
    info._compresslevel = 4
    with out.open(info, "w", force_zip64=True) as dest:
        if isinstance(content, bytes):
            dest.write(content)
        else:
            shutil.copyfileobj(content, dest, 1 << 20)


def supplemental_schema(path, name):
    if name.endswith(".tsv"):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            table = csv.DictReader(stream, delimiter="\t")
            return {
                "format": "TSV",
                "columns": table.fieldnames,
                "data_rows": sum(1 for _ in table),
            }
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as workbook:
        sheets = ElementTree.fromstring(workbook.read("xl/workbook.xml")).iter(ns + "sheet")
        layout = {sheet.attrib["name"]: sheet.attrib["sheetId"] for sheet in sheets}
        counts = {}
        for source in workbook.namelist():
            if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", source):
                rows = ElementTree.fromstring(workbook.read(source)).iter(ns + "row")
                counts[source] = sum(
                    any(
                        cell.find(ns + tag) is not None
                        for cell in row
                        for tag in ("v", "is", "f")
                    )
                    for row in rows
                )
    return {"format": "XLSX", "worksheets": layout, "nonempty_rows_including_headings": counts}


def write_turns(archive, scratch, row, source, records, schemas):
    """Download small batches directly into the output ZIP without a second corpus copy."""
    repo, revision = row["repository"], row["source_revision"]
    entries = sorted(
        (
            entry
            for entry in source["siblings"]
            if re.fullmatch(r"filtered/[0-9a-f]+\.jsonl", entry["rfilename"])
        ),
        key=lambda entry: entry["rfilename"],
    )

    def fetch_one(entry):
        path = Path(scratch) / PurePosixPath(entry["rfilename"]).name
        url = hub.file_url(repo, revision, entry["rfilename"])
        with path.open("w+b") as data:
            hub.download_file(url, data)
            checksum = hub.validate_file(data, entry)
        count = 0
        fields = set()
        with path.open("rb") as data:
            for line in data:
                turn = json.loads(line)
                if not all(
                    isinstance(turn[key], list)
                    for key in ("commands_norm", "combat_state_before", "combat_state_after")
                ):
                    raise ValueError(f'Invalid turn in {entry["rfilename"]}')
                fields.update(turn)
                count += 1
        return path, url, checksum, count, sorted(fields)

    names, total = [], 0
    receipt = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=12) as workers:
        for start in range(0, len(entries), 12):
            group = entries[start : start + 12]
            for entry, result in zip(group, workers.map(fetch_one, group), strict=True):
                path, url, checksum, count, fields = result
                dest = "data/filtered/turns/" + path.name
                with path.open("rb") as data:
                    member(archive, dest, data)
                path.unlink()
                records.append(
                    {
                        "path": dest,
                        "bytes": entry["size"],
                        "url": url,
                        "sha256": checksum,
                        "hub_blob_id": entry["blobId"],
                        "role": "data",
                        "rows": count,
                    }
                )
                schemas[dest] = {"format": "JSON Lines", "columns": fields}
                names.append(dest)
                total += count
                receipt.update(f"{checksum}  {entry['rfilename']}\n".encode())
            if start and start % 240 == 0:
                print(
                    "Verified filtered FIREBALL files",
                    start + len(group),
                    "of",
                    len(entries),
                    flush=True,
                )
    if receipt.hexdigest() != row["payload_sha256"]:
        raise ValueError("Filtered session files differ from the combined survey receipt")
    return total, names


def pack(key, spec, all_rows, out):
    ids = spec.ids if key == "BookCoref" else spec.ids[:1]
    rows = [
        r
        for r in all_rows
        if r["id"] in ids
        and r["status"].startswith("scanned")
        and (not spec.splits or any(re.fullmatch(s, r["split"]) for s in spec.splits))
        and r["config"] not in spec.omit
    ]
    if not rows:
        raise ValueError(f"{key}: no complete surveyed splits")
    repos = {(r["repository"], r["source_revision"]) for r in rows}
    if len(repos) != 1:
        raise ValueError(f"{key}: expected one pinned source release, got {repos}")
    repo, revision = repos.pop()
    source = hub.manifest(repo, revision)
    source_files = {e["rfilename"]: e for e in source["siblings"]}
    manifest = {
        "schema": "anno.offline-hub/v1",
        "release": key,
        "registry_ids": list(spec.ids),
        "repository": repo,
        "source_revision": revision,
        "terms": spec.terms,
        "scope": spec.scope,
        "source_url": f"https://huggingface.co/datasets/{repo}/tree/{revision}",
        "splits": [],
        "files": [],
    }
    schemas = {}
    with tempfile.TemporaryDirectory(dir=out, prefix="." + key + "-") as prefix:
        temporary = Path(prefix) / "release.zip"
        with zipfile.ZipFile(temporary, "w") as archive:

            def original(hub_revision, entry, destination, data=False, inner=None):
                url = hub.file_url(repo, hub_revision, entry["rfilename"])
                with hub.verified_file(url, entry) as (local, digest):
                    result = {
                        "path": destination,
                        "bytes": entry["size"],
                        "url": url,
                        "sha256": digest,
                        "hub_blob_id": entry["blobId"],
                        "role": "data" if data else "publisher documentation",
                    }
                    path = Path(local.name)
                    if inner:
                        path = Path(prefix) / "archive-member"
                        with zipfile.ZipFile(local) as contents, contents.open(
                            safe_path(inner)
                        ) as src, path.open("wb") as output:
                            shutil.copyfileobj(src, output)
                        with path.open("rb") as src:
                            result.update(
                                archive_member=inner,
                                archive_sha256=digest,
                                sha256=hashlib.file_digest(src, "sha256").hexdigest(),
                                bytes=path.stat().st_size,
                            )
                    with path.open("rb") as src:
                        member(archive, destination, src)
                    if data:
                        n, layout = measure(path, PurePosixPath(destination).suffix)
                        result["rows"] = n
                        schemas[destination] = layout
                    manifest["files"].append(result)
                return result

            extra = sorted(
                {
                    n
                    for n in source_files
                    if re.fullmatch(
                        r"(readme\.md|licen[sc]e(?:\.\w+)?|notice(?:\.\w+)?|copying(?:\.\w+)?)",
                        n,
                        re.I,
                    )
                }
                | set(spec.extras)
            )
            if "README.md" not in extra:
                raise ValueError(f"{repo}: missing pinned card")
            for name in extra:
                original(revision, source_files[name], "source/" + name)

            def outside(name, url, digest, content):
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError(f"Upstream source notice changed: {url}")
                destination = "source/" + name
                member(archive, destination, content)
                manifest["files"].append(
                    {
                        "path": destination,
                        "bytes": len(content),
                        "url": url,
                        "sha256": digest,
                        "role": "publisher documentation",
                    }
                )

            if key in ORIGINAL_NOTICES:
                repository, commit, name, digest = ORIGINAL_NOTICES[key]
                url = f"https://raw.githubusercontent.com/{repository}/{commit}/{name}"
                outside(
                    "upstream-" + PurePosixPath(name).name,
                    url,
                    digest,
                    hub.fetch(url, lambda response: response.read()),
                )
            if key == "MSNER":
                url = "https://api.datacite.org/dois/10.48804/ZTVMIX"
                record = hub.fetch(url, json.load)["data"]["attributes"]
                fields = (
                    "doi",
                    "version",
                    "titles",
                    "creators",
                    "rightsList",
                    "publisher",
                    "url",
                    "descriptions",
                )
                outside(
                    "author-dataset-deposit.json",
                    url,
                    "86c779fb8a01f1eb198141f7c9fb16b5f5438a41e4fce2d2c9d5e3a2f1416dfa",
                    json_bytes({k: record[k] for k in fields}),
                )
            if key == "KORE50":
                url = "https://mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads"
                page = hub.fetch(url, lambda response: response.read().decode())
                visible = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", page)).split()).replace(
                    " .", "."
                )
                phrases = [
                    "KORE50.tar.gz",
                    "50 hand-crafted, difficult sentences containing a large number of very ambiguous mentions.",
                    "All datasets are licensed under a Creative Commons Attribution-ShareAlike 3.0 Unported License.",
                ]
                if not all(text in visible for text in phrases):
                    raise ValueError("Original KORE50 source terms changed")
                content = ("\n".join(phrases) + "\n" + url + "\n").encode()
                outside(
                    "original-data-terms.txt",
                    url,
                    "bb7c4e97134932d7526274d0f293343a4a0476b20efd0695c1d09f6a8a9b2d65",
                    content,
                )
            if key == "ACORD":
                entry = source_files["ACORD Dataset & ReadMe.zip"]
                for supplemental in (
                    "ACORD readme.docx",
                    "supplemental materials/acord query (short_medium_long).tsv",
                    "supplemental materials/ACORD 2-5 Star Clause Pairs.xlsx",
                ):
                    file = original(
                        revision,
                        entry,
                        "source/" + supplemental,
                        inner="ACORD Dataset & ReadMe (external)/" + supplemental,
                    )
                    if supplemental.startswith("supplemental materials/"):
                        file["role"] = "supplemental data"
                        schemas[file["path"]] = supplemental_schema(
                            Path(prefix) / "archive-member", supplemental
                        )
            for row in sorted(rows, key=lambda r: (r["config"], r["split"])):
                data_revision = (
                    revision if spec.original else (row["parquet_revision"] or revision)
                )
                pin = hub.manifest(repo, data_revision)
                entries = {e["rfilename"]: e for e in pin["siblings"]}
                actual = 0
                names = []
                if spec.original == "fireball":
                    actual, names = write_turns(
                        archive, prefix, row, pin, manifest["files"], schemas
                    )
                    paths = []
                else:
                    paths = partition_files(row, spec, pin)
                    if not paths:
                        raise ValueError(f"No complete shards for {row['config']}/{row['split']}")
                for path in paths:
                    inner = parse_qs(urlsplit(row["payload_url"]).fragment).get("member", [None])[
                        0
                    ]
                    dest = (
                        f"data/{row['config']}/{row['split']}/{PurePosixPath(inner or path).name}"
                    )
                    f = original(data_revision, entries[path], dest, data=True, inner=inner)
                    if (
                        len(paths) == 1
                        and row["payload_sha256"]
                        and not spec.original
                        and row["payload_sha256"] != f["sha256"]
                    ):
                        raise ValueError(
                            f"{key}: bytes differ from published split receipt at {path}"
                        )
                    actual += f["rows"]
                    names.append(dest)
                if actual != int(row["rows"]):
                    raise ValueError(
                        f"{key}: {row['config']}/{row['split']} contains {actual}, expected {row['rows']}"
                    )
                manifest["splits"].append(
                    {
                        "config": row["config"],
                        "split": row["split"],
                        "rows": actual,
                        "paths": names,
                        "observation_column": row["observation_column"],
                        "survey_status": row["status"],
                        "survey_note": row["scope_note"],
                        "surveyed_rows": int(row["rows"]),
                        "pinned_data_revision": data_revision,
                    }
                )
            member(archive, "SCHEMAS.json", json_bytes(schemas))
            member(archive, "MANIFEST.json", json_bytes(manifest))
            member(archive, "verify.py", VERIFIER.encode())
            first = manifest["splits"][0]["paths"][0]
            suffix = PurePosixPath(first).suffix
            sample = (
                f"import pyarrow.parquet as pq\nf = pq.ParquetFile({first!r})\nprint(f.schema_arrow)\nprint(next(f.iter_batches(batch_size=1)).to_pylist()[0].keys())"
                if suffix == ".parquet"
                else f'from pathlib import Path\nwith Path({first!r}).open(encoding="utf-8") as f:\n    print(next(f).strip()[:200])'
            )
            if suffix == ".gz":
                sample = f'import json, tarfile\nwith tarfile.open({first!r}, "r|gz") as archive:\n    first = next(m for m in archive if m.isfile() and m.name.endswith(".json"))\n    print(json.load(archive.extractfile(first)).keys())'
            readme = f'# {key}\n\n{spec.scope}\n\n{spec.terms}. See the original dataset card and any supplied notices in source/.\n\nRegistry IDs: {", ".join(spec.ids)}. Source: {manifest["source_url"]}\n\nExtract the ZIP, then run `python verify.py` from the extracted directory. MANIFEST.json gives every split, its row unit, original URL, byte count and SHA-256; SCHEMAS.json records the actual columns and any label mapping in the Parquet metadata. Configurations can reuse observations.\n\nRead one record locally:\n\n```python\n{sample}\n```\n'
            member(archive, "README.md", readme.encode())
        with zipfile.ZipFile(temporary) as check:
            if check.testzip():
                raise ValueError(f"{key}: ZIP member failed CRC check")
        destination = out / f"{key}.zip"
        os.replace(temporary, destination)
    return (
        key,
        len(manifest["splits"]),
        sum(x["rows"] for x in manifest["splits"]),
        destination.stat().st_size,
    )


def index(out, ledger_rows):
    packs, splits, files, covered = [], [], [], set()
    measured_ids = defaultdict(set)
    for row in ledger_rows:
        if row["status"].startswith("scanned"):
            measured_ids[
                (row["repository"], row["source_revision"], row["config"], row["split"])
            ].add(row["id"])
    for path in sorted(out.glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            m = json.loads(archive.read("MANIFEST.json"))
            if m.get("schema") != "anno.offline-hub/v1":
                raise ValueError(f"Unexpected archive {path.name}")
            for f in m["files"]:
                with archive.open(f["path"]) as reader:
                    if (
                        archive.getinfo(f["path"]).file_size != f["bytes"]
                        or hashlib.file_digest(reader, "sha256").hexdigest() != f["sha256"]
                    ):
                        raise ValueError(f"{path.name}: corrupt {f['path']}")
            lookup = {f["path"]: f for f in m["files"]}
            member_ids = defaultdict(set)
            for group in m["splits"]:
                key = (m["repository"], m["source_revision"], group["config"], group["split"])
                split_ids = [ident for ident in m["registry_ids"] if ident in measured_ids[key]]
                if not split_ids:
                    raise ValueError(f"{path.name}: missing registry split identity for {key}")
                for name in group["paths"]:
                    member_ids[name].update(split_ids)
                for ident in split_ids:
                    covered.add((ident, m["repository"], group["config"], group["split"]))
                payloads = [lookup[q] for q in group["paths"]]
                splits.append(
                    {
                        "archive": path.name,
                        "registry_ids": "|".join(split_ids),
                        "repository": m["repository"],
                        "source_revision": m["source_revision"],
                        "data_revision": group["pinned_data_revision"],
                        "config": group["config"],
                        "split": group["split"],
                        "rows": group["rows"],
                        "files": len(payloads),
                        "uncompressed_payload_bytes": sum(r["bytes"] for r in payloads),
                        "status": group["survey_status"],
                        "observation_column": group["observation_column"],
                        "survey_note": group["survey_note"],
                    }
                )
            for file in m["files"]:
                record = {"archive": path.name, **file}
                if file["role"] == "data":
                    record["registry_ids"] = sorted(member_ids[file["path"]])
                    if not record["registry_ids"]:
                        raise ValueError(f"{path.name}: source data file missing its split")
                elif file["role"] == "supplemental data":
                    record["registry_ids"] = m["registry_ids"]
                files.append(record)
            data = [q for q in m["files"] if q["role"] in {"data", "supplemental data"}]
            supplemental = [q for q in data if q["role"] == "supplemental data"]
            documents = [q for q in m["files"] if q["role"] == "publisher documentation"]
            with path.open("rb") as source:
                checksum = hashlib.file_digest(source, "sha256").hexdigest()
            packs.append(
                {
                    "archive": path.name,
                    "registry_ids": m["registry_ids"],
                    "repository": m["repository"],
                    "source_revision": m["source_revision"],
                    "sha256": checksum,
                    "archive_bytes": path.stat().st_size,
                    "data_files": len(data),
                    "data_bytes": sum(q["bytes"] for q in data),
                    "partition_data_files": len(data) - len(supplemental),
                    "supplemental_data_files": len(supplemental),
                    "supplemental_data_bytes": sum(q["bytes"] for q in supplemental),
                    "source_document_files": len(documents),
                    "source_document_bytes": sum(q["bytes"] for q in documents),
                    "receipt_partitions": len(m["splits"]),
                    "record_instances": sum(q["rows"] for q in m["splits"]),
                    "terms": m["terms"],
                    "scope": m["scope"],
                }
            )
    if not packs:
        raise ValueError("No verified Hub pack in output directory")
    reasons = {
        ident: reason
        for group, reason in SOURCE_RULES["exclusions"].items()
        for ident in group.split()
    }
    held = {}
    scope_by_id = {ident: p["scope"] for p in packs for ident in p["registry_ids"]}
    for row in ledger_rows:
        key = (row["id"], row["repository"], row["config"], row["split"])
        if key in covered:
            continue
        reason = (
            ("Outside the packaged edition. " + scope_by_id[row["id"]])
            if row["id"] in scope_by_id
            else reasons.get(
                row["id"], row["reason"] or "Outside the reviewed releases in this Hub pack."
            )
        )
        held.setdefault((row["id"], row["repository"], reason), []).append(row)
    excluded = [
        {
            "id": ident,
            "repository": repo,
            "ledger_partitions": len(items),
            "configs": "|".join(sorted({r["config"] for r in items})),
            "splits": "|".join(sorted({r["split"] for r in items})),
            "reason": reason,
        }
        for (ident, repo, reason), items in sorted(held.items())
    ]

    def save_csv(name, records):
        with (out / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    save_csv("splits.csv", splits)
    if excluded:
        save_csv("exclusions.csv", excluded)
    (out / "inventory.json").write_bytes(
        json_bytes(
            {
                "schema": "anno.offline-hub-inventory/v1",
                "registry_ids": sorted(scope_by_id),
                "registry_ids_in_hf_ledger": len({r["id"] for r in ledger_rows}),
                "included_ledger_entries": len(covered),
                "packs": packs,
            }
        )
    )
    (out / "files.json").write_bytes(json_bytes(files))
    (out / "SHA256SUMS").write_text(
        "".join(p["sha256"] + "  " + p["archive"] + "\n" for p in packs), encoding="utf-8"
    )
    inventory = [
        f"# Pinned dataset releases\n",
        f'{len(packs)} self-contained ZIPs cover {len(scope_by_id)} of the {len({r["id"] for r in ledger_rows})} registry IDs in the Hugging Face split ledger. Each ZIP holds original data files, the source card, an inventory, schema information and an offline verifier.\n',
        "Read inventory.json for per-archive bytes, hashes, sources, conditions and scope. Read splits.csv for each exact split, configuration, row count and data bytes. files.json inventories individual source URLs. exclusions.csv explains unbundled registry entries and split choices.\n",
        "Counts are partition records. Some editions repeat the same text, as documented in the source manifests. Candidate tables, clauses, judgments and patent applications also have distinct row units.\n",
        "Check a downloaded ZIP against SHA256SUMS, extract it, and run `python verify.py` inside the extracted directory. Each archive’s README has an offline reading example. JSONL, TSV and CoNLL can be read with Python alone. Parquet examples use PyArrow.\n",
        "| Archive | Source data files | Partitions | Registry IDs |\n| --- | ---: | ---: | --- |",
    ]
    inventory.extend(
        f"| {p['archive']} | {p['data_files']:,} | {p['receipt_partitions']:,} | {', '.join(p['registry_ids'])} |"
        for p in packs
    )
    (out / "README.md").write_text("\n".join(inventory) + "\n", encoding="utf-8")
    print(
        f"Indexed {len(packs)} archives, {len(covered)} ledger entries, {len(scope_by_id)} registry IDs.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--only", nargs="+", choices=sorted(RELEASES))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Rehash existing ZIPs and write consolidated file, split, coverage and exclusion indexes.",
    )
    args = parser.parse_args()
    args.directory.mkdir(exist_ok=True, parents=True)
    with args.ledger.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if args.index_only:
        index(args.directory, rows)
        return
    with tempfile.TemporaryDirectory(prefix="anno-hub-content-") as scratch:
        os.environ.setdefault("ANNO_HF_SURVEY_CACHE", scratch)
        failed = False
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {
                pool.submit(pack, k, RELEASES[k], rows, args.directory): k
                for k in (args.only or RELEASES)
            }
            for future in as_completed(jobs):
                try:
                    print("PACK", *future.result(), flush=True)
                except Exception as exc:
                    print(
                        f"FAILED {jobs[future]}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    failed = True
        if failed:
            raise SystemExit(1)
    if not args.only:
        index(args.directory, rows)


if __name__ == "__main__":
    main()
