"""Verify offline dataset archives, or restore an archive transported in smaller pieces."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected a collection-relative path: {name}")
    return root.joinpath(*path.parts)


def check(path, record):
    if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
        raise ValueError(f"Size or SHA-256 differs for {path}")


def split_archive(root, name, maximum):
    source = local_path(root, name)
    if maximum <= 0 or source.stat().st_size <= maximum:
        raise ValueError("Choose a positive part size smaller than the archive")
    parts = []
    archive_hash = hashlib.sha256()
    with source.open("rb") as src:
        while count := src.read(maximum):
            path = source.with_name(f"{source.name}.part{len(parts) + 1:03}")
            expected = {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(count),
                "sha256": hashlib.sha256(count).hexdigest(),
            }
            if path.exists():
                check(path, expected)
            else:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
                    temp = Path(out.name)
                    try:
                        out.write(count)
                        out.flush()
                        check(temp, expected)
                        try:
                            os.link(temp, path)
                        except FileExistsError:
                            check(path, expected)
                    finally:
                        temp.unlink(missing_ok=True)
            parts.append(expected)
            archive_hash.update(count)
    return {
        "path": name,
        "bytes": sum(p["bytes"] for p in parts),
        "sha256": archive_hash.hexdigest(),
        "parts": parts,
    }


def restore(root, record):
    dest = local_path(root, record["path"])
    if dest.is_file():
        check(dest, record)
        return
    if not record.get("parts"):
        raise FileNotFoundError(f"Download {record['path']} from the collection")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as out:
        temp = Path(out.name)
        try:
            for part in record["parts"]:
                path = local_path(root, part["path"])
                size, digest = 0, hashlib.sha256()
                with path.open("rb") as src:
                    while chunk := src.read(1024 * 1024):
                        out.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if size != part["bytes"] or digest.hexdigest() != part["sha256"]:
                    raise ValueError(f"Size or SHA-256 differs for {path}")
            out.flush()
            check(temp, record)
            os.link(temp, dest)
        finally:
            temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--archive",
        action="append",
        help="Check just this collection-relative archive (repeatable)",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Rejoin missing ZIPs from their parts and verify them",
    )
    parser.add_argument(
        "--split",
        metavar="ARCHIVE",
        help="Prepare an existing ZIP for size-limited file transfer",
    )
    parser.add_argument("--part-bytes", type=int, default=90_000_000)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.split:
        print(json.dumps(split_archive(root, args.split, args.part_bytes), indent=2))
        return
    records = json.loads((root / "archives.json").read_text())["archives"]
    selected = set(args.archive or [r["path"] for r in records])
    unknown = selected.difference(r["path"] for r in records)
    if unknown:
        parser.error(f"Unknown archive(s): {', '.join(sorted(unknown))}")
    for record in records:
        if record["path"] in selected:
            if args.restore:
                restore(root, record)
            else:
                check(local_path(root, record["path"]), record)
            print(f"{record['sha256']}  {record['path']}")


if __name__ == "__main__":
    main()
