"""Compare the bundled SQLite releases in Anno #157 on a local Windows volume.

Run from a full Anno checkout with Python 3.11+ and Cargo:
    python scripts/review-sqlite-wal/run.py
"""

import json
import mmap
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REVISIONS = {
    "3.53.2": "9756b458810759e6c5cd45eef78d81a9c83037a6",
    "3.54.1": "3d0101b38b5b8fc27c7dee2248adbb659eb8d963",
}


def at_revision(revision, name):
    return subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)


def build(work, version):
    revision = REVISIONS[version]
    original_lock = at_revision(revision, "Cargo.lock")
    original = tomllib.loads(original_lock.decode())["package"]
    manifest = tomllib.loads(at_revision(revision, "Cargo.toml").decode())
    dependency = manifest["workspace"]["dependencies"]["rusqlite"]
    features = json.dumps(dependency["features"])
    defaults = str(dependency.get("default-features", True)).lower()
    crate = work / version
    crate.mkdir()
    (crate / "Cargo.toml").write_text(
        '[package]\nname="wal-probe"\nversion="0.0.0"\nedition="2021"\n'
        '[[bin]]\nname="wal-probe"\npath="probe.rs"\n[dependencies]\n'
        f'rusqlite={{version="={dependency["version"]}", default-features={defaults}, features={features}}}\n',
        encoding="utf-8",
    )
    shutil.copyfile(HERE / "probe.rs", crate / "probe.rs")
    (crate / "Cargo.lock").write_bytes(original_lock)
    subprocess.run(["cargo", "build", "--quiet", "--target-dir", str(crate / "target")], cwd=crate, check=True)
    compiled = tomllib.loads((crate / "Cargo.lock").read_text(encoding="utf-8"))["package"]
    pins = {(p["name"], p["version"], p.get("checksum")) for p in original}
    assert all((p["name"], p["version"], p["checksum"]) in pins for p in compiled if "source" in p)
    return crate / "target/debug" / ("wal-probe.exe" if os.name == "nt" else "wal-probe")


def exercise(case, writer, reader, binaries, mapped):
    case.mkdir()
    db = case / "history.db"
    subprocess.run([binaries[writer], "seed", db, writer], check=True)
    shm = case / "history.db-shm"
    assert shm.stat().st_size > 0 and (case / "history.db-wal").stat().st_size > 0
    reopen = [binaries[reader], "reopen", db, reader]
    outcome = "reopened"
    if mapped:
        with shm.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as view:
            assert any(view[:8]), "expected the writer's initialized WAL-index header"
            result = subprocess.run(reopen, capture_output=True, text=True)
            if os.name == "nt" and reader == "3.53.2":
                assert result.returncode == 42, (result.stdout, result.stderr)
                assert result.stdout.strip() == "SQLITE_IOERR_TRUNCATE"
                outcome = "SQLITE_IOERR_TRUNCATE, then reopened after unmap"
            elif result.returncode:
                raise RuntimeError(f"{result.returncode}: {result.stdout} {result.stderr}")
        if result.returncode == 42:
            subprocess.run(reopen, check=True)
    else:
        subprocess.run(reopen, check=True)
    for version, binary in binaries.items():
        subprocess.run([binary, "verify", db, version], check=True)
    print(f"writer={writer} reader={reader} mapping={mapped}: {outcome}, integrity ok", flush=True)


def main():
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        binaries = {version: build(work, version) for version in REVISIONS}
        for writer in binaries:
            for reader in binaries:
                for mapped in [False, True]:
                    exercise(work / f"{writer}_{reader}_{mapped}", writer, reader, binaries, mapped)


if __name__ == "__main__":
    main()
