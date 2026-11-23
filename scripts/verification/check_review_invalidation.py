#!/usr/bin/env python3
"""Check that removing review invalidation makes the public Rust regression fail."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
GATE = Path("crates/anno/src/core/grounded/corpus_state.rs")
TEST = "handing_out_a_mutable_document_retires_its_outstanding_preview"
MUTATION_SITE = "        self.review_revision.take();\n"


def snapshot(destination):
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    for encoded in paths.split(b"\0"):
        if not encoded:
            continue
        path = Path(os.fsdecode(encoded))
        source = ROOT / path
        if source.is_file() or source.is_symlink():
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target, follow_symlinks=False)


def run_test(workspace, target):
    env = {
        **os.environ,
        "CARGO_TARGET_DIR": str(target),
        "CARGO_BUILD_BUILD_DIR": str(target),
        "CARGO_TERM_COLOR": "never",
        "RUST_BACKTRACE": "0",
    }
    env.setdefault("CARGO_PROFILE_TEST_DEBUG", "0")
    return subprocess.run(
        ["cargo", "test", "--locked", "-p", "anno", "--no-default-features",
         "--test", "reviewed_kb_assignment", "--", "--exact", TEST],
        cwd=workspace, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def main():
    with tempfile.TemporaryDirectory(prefix="anno-invalidation-") as tmp:
        workspace = Path(tmp) / "workspace"
        target = Path(tmp) / "target"
        snapshot(workspace)
        gate = workspace / GATE
        if not gate.resolve().is_relative_to(workspace):
            raise SystemExit(f"Mutation site points outside the working copy: {gate}")
        source = gate.read_text()
        if source.count(MUTATION_SITE) != 1:
            raise SystemExit(f"Expected one revision invalidation in {GATE}; review the mutation site.")

        baseline = run_test(workspace, target)
        if (baseline.returncode or f"test {TEST} ... ok" not in baseline.stdout
                or "test result: ok. 1 passed; 0 failed;" not in baseline.stdout):
            raise SystemExit(f"Positive control failed:\n{baseline.stdout}")

        gate.write_text(source.replace(MUTATION_SITE, ""))
        mutant = run_test(workspace, target)
        if not (mutant.returncode == 101 and f"test {TEST} ... FAILED" in mutant.stdout
                and "mutable access must retire the preview: KbAssignmentReceipt" in mutant.stdout
                and "test result: FAILED. 0 passed; 1 failed;" in mutant.stdout):
            raise SystemExit(f"Expected a stale preview to be applied by the mutant:\n{mutant.stdout}")
    print(f"Passed positive control. Removing revision invalidation made {TEST} return a stale receipt.")


if __name__ == "__main__":
    main()
