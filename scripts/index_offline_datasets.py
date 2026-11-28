"""Index a collection of Anno offline archives against the complete registry snapshot.

Run after building the publisher, hub, direct or native packs. Only archives present
locally count as included. Repeated IDs, files and split variants stay traceable.
"""

import argparse
from collections import defaultdict
import csv
import json
import tempfile
from io import TextIOWrapper
from zipfile import ZipFile
from pathlib import Path

from offline_dataset_archives import check, file_hash, local_path

EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "eval"


def rows(path, delimiter=","):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter=delimiter))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, fields, data):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def build(root, evidence, split_receipts, s3_evidence=None):
    coverage = {r["id"]: r for r in rows(evidence / "dataset-coverage.csv")}
    sources, notes, omitted = (defaultdict(set) for _ in range(3))
    hashes = defaultdict(set)
    expected = {}
    native_ids = set()

    def add(ident, archive, scope, sha=None):
        if ident not in coverage:
            raise ValueError(f"Unknown registry ID {ident} in {archive}")
        sources[ident].add(archive)
        if scope:
            notes[ident].add(scope)
        if sha:
            hashes[ident, sha].add(archive)

    hub = root / "hub"
    if (hub / "inventory.json").is_file():
        packs = load(hub / "inventory.json")["packs"]
        for pack in packs:
            archive = "hub/" + pack["archive"]
            expected[archive] = {
                "bytes": pack["archive_bytes"],
                "sha256": pack["sha256"],
            }
            for ident in pack["registry_ids"]:
                add(ident, archive, pack["scope"])
        for member in load(hub / "files.json"):
            for ident in member.get("registry_ids", []):
                archive = "hub/" + member["archive"]
                if archive not in sources[ident]:
                    raise ValueError(f"Unlisted release for {ident}: {archive}")
                if member["role"] != "publisher documentation":
                    hashes[ident, member["sha256"]].add(archive)
        for row in rows(hub / "exclusions.csv"):
            omitted[row["id"]].add(row["reason"])

    publisher = root / "publisher"
    if (publisher / "publisher-packages.json").is_file():
        for name, record in load(publisher / "publisher-packages.json")[
            "packages"
        ].items():
            expected["publisher/" + name] = record
        for row in rows(publisher / "publisher-coverage.csv"):
            if row["member"]:
                name = f"publisher/publisher-{row['bundle']}.zip"
                scope = "Publisher Git files: " + ", ".join(
                    x for x in (row["subset"], row["split"]) if x
                )
                add(row["id"], name, scope.rstrip(": "), row["sha256"])
            else:
                omitted[row["id"]].add(row["terms"] or row["reason"])

    for manifest in sorted((root / "direct").glob("*.manifest.json")):
        pack = load(manifest)
        archive = "direct/" + pack["archive"]
        expected[archive] = {
            "bytes": pack["archive_bytes"],
            "sha256": pack["archive_sha256"],
        }
        for ident, detail in pack["datasets"].items():
            add(ident, archive, detail["included"])
            if detail["excluded"]:
                omitted[ident].add(detail["excluded"])
        for row in rows(
            manifest.with_name(manifest.name.replace(".manifest.json", ".files.tsv")),
            "\t",
        ):
            if row["role"] == "data":
                hashes[row["registry_id"], row["sha256"]].add(archive)

    for manifest in sorted((root / "native").glob("*.sources.json")):
        archive = "native/" + manifest.name.replace(".sources.json", ".zip")
        with ZipFile(root / archive) as pack:
            if pack.read("sources.json") != manifest.read_bytes():
                raise ValueError(f"Native source index differs from {archive}")
        for item in load(manifest)["inputs"]:
            ident = item["id"]
            native_ids.add(ident)
            add(ident, archive, "Native cache: " + item["variant"], item["sha256"])

    omissions = root / "native" / "native-omissions.json"
    if omissions.is_file():
        for ident, item in load(omissions)["reviewed"].items():
            omitted[ident].add("Native cache: " + item["reason"])

    speech = root / "speech" / "vira-dialog-acts-offline.zip"
    if speech.is_file():
        speech_pack = load(speech.with_name("vira-dialog-acts-offline-manifest.json"))
        expected["speech/" + speech.name] = speech_pack["archive"]
        for member in speech_pack["files"]:
            if member["path"].startswith(
                ("publisher/data/", "ibm-classifier/dialog_act_dataset/")
            ):
                hashes["ViraDialogActsLive", member["sha256"]].add(
                    "speech/" + speech.name
                )
        add(
            "ViraDialogActsLive",
            "speech/" + speech.name,
            "IBM train, validation and test utterances and ordered label dictionary",
        )

    if s3_evidence is None:
        snapshots = sorted((root / "evidence").glob("cache-source-evidence-*.zip"))
        if len(snapshots) > 1:
            raise ValueError(
                "Multiple S3 snapshots in this collection. Select one with --s3-evidence."
            )
        s3_evidence = next(iter(snapshots), None)
    info = None
    matched = []
    if s3_evidence is not None:
        with ZipFile(s3_evidence) as snapshot:
            info = json.loads(snapshot.read("cache-source-coverage.json"))
            with snapshot.open("publisher-file-matches.csv") as stream:
                matched = list(csv.DictReader(TextIOWrapper(stream, encoding="utf-8")))
    matches = {}
    for row in matched:
        if (
            row["cache_id"] == row["publisher_id"]
            and row["match"] == "same registry ID"
        ):
            # Several publisher file ledgers can refer to the same object.
            key = (row["cache_id"], row["object_kind"], row["sha256"])
            matches[key] = {
                "id": key[0],
                "object_kind": key[1],
                "sha256": key[2],
                "archives": " | ".join(sorted(hashes[key[0], key[2]])),
            }
    if (
        info is not None
        and len(matches)
        != info["payload_objects_by_outcome"]["matches_own_publisher_file"]
    ):
        raise ValueError("Distinct matched objects disagree with the S3 evidence")
    matched_ids = {k[0] for k, r in matches.items() if r["archives"]}
    fields = [
        "id",
        "name",
        "domain",
        "language",
        "status",
        "archives",
        "native_cache",
        "scope",
        "exclusions_or_limits",
        "same_release_as",
        "available_via",
        "matched_dated_s3_object",
    ]
    inventory = []
    for ident, row in sorted(coverage.items()):
        available = bool(sources.get(ident))
        reason = set(filter(None, omitted.get(ident, ())))
        if not available and not reason:
            reason.add(
                row["outstanding_reason"]
                or "No archive in this collection. See the survey for independently measured files."
            )
        inventory.append(
            dict(
                id=ident,
                name=row["name"],
                domain=row["domain"],
                language=row["language"],
                status="included" if available else "absent",
                archives=" | ".join(sorted(sources.get(ident, ()))),
                native_cache="yes" if ident in native_ids else "",
                same_release_as=row["same_release_as"],
                available_via=" | ".join(
                    sorted(sources.get(row["same_release_as"], ()))
                ),
                scope=" | ".join(sorted(notes.get(ident, ()))),
                exclusions_or_limits=" | ".join(sorted(reason)),
                matched_dated_s3_object="yes" if ident in matched_ids else "",
            )
        )
    covered = {r["id"] for r in inventory if r["status"] == "included"}
    corpus_archives = {a for v in sources.values() for a in v}
    known = corpus_archives | {
        p.relative_to(root).as_posix() for p in (root / "evidence").glob("*.zip")
    }
    tools = root / "native" / "native-build-tools.zip"
    if tools.is_file():
        known.add(tools.relative_to(root).as_posix())
    receipts = {r["path"]: r for r in (load(p) for p in split_receipts)}
    if set(receipts) - known:
        raise ValueError(
            f"Part manifest references unknown archives: {sorted(set(receipts) - known)}"
        )
    archives = []
    for name in sorted(known):
        path = local_path(root, name)
        record = {
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
            "kind": (
                "corpus"
                if name in corpus_archives
                else ("tools" if name.startswith("native/") else "analysis")
            ),
        }
        for original in (expected.get(name), receipts.get(name)):
            if original and (record["bytes"], record["sha256"]) != (
                original["bytes"],
                original["sha256"],
            ):
                raise ValueError(f"Archive differs from its source manifest: {name}")
        if name in receipts:
            record["parts"] = receipts[name]["parts"]
            for part in record["parts"]:
                check(local_path(root, part["path"]), part)
        archives.append(record)
    result = {
        "schema": "anno.offline-archives/v1",
        "registry_ids": len(coverage),
        "included_ids": len(covered),
        "native_ids": len(native_ids),
        "corpus_archives": len(corpus_archives),
        "corpus_archive_bytes": sum(
            r["bytes"] for r in archives if r["kind"] == "corpus"
        ),
        "matched_s3_objects": (
            sum(bool(r["archives"]) for r in matches.values())
            if info is not None
            else None
        ),
        "publisher_matched_s3_objects_in_snapshot": (
            len(matches) if info is not None else None
        ),
        "matched_s3_ids": len(matched_ids) if info is not None else None,
        "s3_snapshot": (
            {
                key: info[key]
                for key in (
                    "listing_started_utc",
                    "workflow_run_id",
                    "repository_commit",
                )
            }
            if info is not None
            else None
        ),
        "archives": archives,
    }
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        staging = Path(tmp)
        write_csv(staging / "datasets.csv", fields, inventory)
        write_csv(
            staging / "s3-file-matches.csv",
            ["id", "object_kind", "sha256", "archives"],
            (v for _, v in sorted(matches.items())),
        )
        (staging / "archives.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        for name in ("datasets.csv", "s3-file-matches.csv", "archives.json"):
            (staging / name).replace(root / name)
    print(json.dumps({k: v for k, v in result.items() if k != "archives"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument(
        "--s3-evidence",
        type=Path,
        help="Compare local releases with this dated cache-source-evidence workflow ZIP. "
        "By default only a snapshot bundled in the collection's evidence directory is used.",
    )
    parser.add_argument("--part-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    build(args.root, args.evidence, args.part_manifest, args.s3_evidence)


if __name__ == "__main__":
    main()
