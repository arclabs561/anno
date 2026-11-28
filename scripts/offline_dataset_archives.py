"""Verify offline dataset archives, or restore an archive transported in smaller pieces."""

import argparse
from contextlib import contextmanager
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
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or any(c in name for c in "\\:")
    ):
        raise ValueError(f"Expected a collection-relative path: {name}")
    return root.joinpath(*path.parts)


def check(path, record):
    if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
        raise ValueError(f"Size or SHA-256 differs for {path}")


def publish(staged, dest, record):
    """Install a closed, verified file. Concurrent packers must agree on its bytes."""
    if os.name == "nt":
        # Windows rename already fails if another writer installed the destination.
        try:
            staged.rename(dest)
            return
        except FileExistsError:
            pass
    else:
        import fcntl

        # POSIX rename replaces an existing file. Cooperating writers lock a
        # regular file opened for writing, as required by NFS implementations.
        # Keep its inode stable. The OS releases the lock when a process exits.
        with (dest.parent / ".anno-archive-publish.lock").open("ab") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not os.path.lexists(dest):
                staged.rename(dest)
                return
    check(dest, record)


@contextmanager
def staged_output(dest, record):
    # A sibling directory keeps rename on the destination filesystem. All file
    # handles are closed before hashing, moving, or cleaning up the staged file.
    with tempfile.TemporaryDirectory(prefix=".anno-archive-", dir=dest.parent) as temp:
        staged = Path(temp) / "payload"
        with staged.open("wb") as out:
            yield out
        check(staged, record)
        publish(staged, dest, record)


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
                with staged_output(path, expected) as out:
                    out.write(count)
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
    with staged_output(dest, record) as out:
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
