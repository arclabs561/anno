"""Adapter for the tracked catalog operation corpus and its Rust assignment oracle."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

SCOPE = "synthetic_contract"
ORACLE_FILES = ("crates/anno/tests/reviewed_kb_seeds.rs",
                "crates/anno/tests/support/reviewed_kb_fuzz.rs")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def seed_path(root, case):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}", case):
        raise ValueError("invalid reviewed KB seed name")
    path = root / "fuzz/seeds/reviewed_kb" / case
    if path.is_symlink() or not path.is_file():
        raise ValueError("expected a reviewed seed file")
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", str(path.relative_to(root))],
                   cwd=root, check=True, stdout=subprocess.DEVNULL)
    return path


def describe(root, case):
    parts = [digest(root / file) for file in ORACLE_FILES]
    parts.append(digest(Path(__file__)))
    return {"source_sha256": digest(seed_path(root, case)),
            "oracle_sha256": hashlib.sha256("\n".join(parts).encode()).hexdigest(),
            # Related minimized witnesses can use '<family>--<variant>'.
            "family": case.partition("--")[0]}


def replay(root, case):
    seed = seed_path(root, case)
    cmd = ["cargo", "test", "--locked", "--offline", "-p", "anno", "--no-default-features",
           "--test", "reviewed_kb_seeds", "--no-run", "--message-format=json"]
    messages = subprocess.check_output(cmd, cwd=root, text=True).splitlines()
    artifacts = [json.loads(line) for line in messages]
    binary, = [Path(a["executable"]) for a in artifacts
               if a.get("executable") and a["target"]["name"] == "reviewed_kb_seeds"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "receipt.json"
        env = dict(os.environ, ANNO_REVIEWED_SEED=str(seed), ANNO_REVIEWED_RECEIPT=str(path))
        subprocess.run([binary, "replay_one_reviewed_seed", "--exact", "--ignored"], cwd=root,
                       env=env, check=True, stdout=subprocess.DEVNULL)
        result = json.loads(path.read_bytes())
    if (result["source_sha256"] != digest(seed)
            or result["oracle_files"] != [digest(root / f) for f in ORACLE_FILES]):
        raise ValueError("compiled oracle or executed seed differs from the pinned source")
    return {"outcome": result["outcome"], "signature": result["signature"],
            "producer_sha256": digest(binary)}
