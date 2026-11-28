"""Install pinned model files after checking all downloads against the release manifest."""

import hashlib
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory


def matches(path, spec):
    """Check a Hub LFS SHA-256 or a Git blob ID and its declared byte length."""
    if not path.is_file():
        return False
    size = path.stat().st_size
    if "size" in spec and spec["size"] != size:
        return False
    with path.open("rb") as stream:
        if "sha256" in spec:
            return hashlib.file_digest(stream, "sha256").hexdigest() == spec["sha256"]
        blob = hashlib.sha1(f"blob {size}\0".encode())
        for block in iter(lambda: stream.read(1 << 20), b""):
            blob.update(block)
        return blob.hexdigest() == spec["git_blob"]


def pinned_inputs(source, stage, hashes):
    """Expose exactly the pinned files to libraries that discover optional sidecars."""
    stage.mkdir()
    for name, digest in hashes.items():
        origin = (source / name).resolve(strict=True)
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(origin, target)
        except OSError:
            shutil.copyfile(origin, target)
        if not matches(target, {"sha256": digest}):
            raise ValueError(f"Unexpected SHA-256 for {name}")
    return stage


def prepare(directory, repo, revision, files, *, other_configs=()):
    directory = Path(directory)
    if directory.is_symlink() and not directory.exists():
        raise ValueError(f"Model destination is a dangling symlink: {directory}")
    directory = directory.resolve()
    if directory.is_dir() and all(
        matches(directory / name, spec) for name, spec in files.items()
    ):
        return

    from filelock import FileLock

    directory.parent.mkdir(parents=True, exist_ok=True)
    # Serialize competing repairs and recheck the destination after taking the lock.
    with FileLock(str(directory.parent / f".{directory.name}.prepare.lock")):
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"Model destination is not a directory: {directory}")
        if any(matches(directory / "config.json", config) for config in other_configs):
            raise ValueError(
                f"{directory} contains another model variant; choose its own directory"
            )
        missing = [
            name for name, spec in files.items() if not matches(directory / name, spec)
        ]
        if not missing:
            return

        from huggingface_hub import snapshot_download, try_to_load_from_cache

        with TemporaryDirectory(dir=directory.parent) as scratch:
            stage = Path(scratch) / "model"
            pending = []
            for name in missing:
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                # snapshot_download(local_dir=...) does not consult cached snapshots
                # offline. Reuse individual verified files from the shared Hub cache.
                cached = try_to_load_from_cache(repo, name, revision=revision)
                if isinstance(cached, str) and matches(Path(cached), files[name]):
                    shutil.copyfile(cached, target)
                else:
                    pending.append(name)
            if pending:
                download = Path(scratch) / "download"
                received = Path(
                    snapshot_download(
                        repo_id=repo,
                        revision=revision,
                        local_dir=download,
                        allow_patterns=pending,
                    )
                )
                for name in pending:
                    source = received / name
                    if not source.is_file():
                        raise ValueError(
                            f"Incomplete snapshot for {repo}@{revision}: {name}"
                        )
                    target = stage / name
                    if received == download and not source.is_symlink():
                        source.rename(target)
                    else:
                        shutil.copyfile(source, target)
            for name in missing:
                if not matches(stage / name, files[name]):
                    raise ValueError(
                        f"Prepared file does not match {repo}@{revision}: {name}"
                    )
                target = directory / name
                if target.is_dir() or any(
                    part.is_symlink() or (part.exists() and not part.is_dir())
                    for part in target.parents
                    if part.is_relative_to(directory) and part != directory
                ):
                    raise ValueError(f"Cannot install file at {target}")
            if not directory.exists():
                stage.rename(directory)
                return
            for name in missing:
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / name, target)
