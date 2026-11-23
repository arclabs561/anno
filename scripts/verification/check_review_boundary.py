#!/usr/bin/env python3
"""Compile valid and invalid Rust callers of anno's reviewed catalog assignment API."""

import json
import os
from pathlib import Path
import subprocess
import tempfile

FIXTURE = Path(__file__).resolve().with_name("review_boundary.rs")
ROOT = FIXTURE.parents[2]
CASES = {
    "permitted": None,
    "direct_data": "E0616",
    "mutable_read": "E0596",
    "mutable_deref": "E0277",
    "restore_revision": "E0616",
    "edited_copy": "E0308",
    "change_target": "E0594",
    "direct_proposal": "E0616",
    "read_token": "E0616",
    "double_apply": "E0382",
}


def build_library():
    build = subprocess.run(
        ["cargo", "build", "--locked", "-p", "anno", "--lib", "--no-default-features",
         "--message-format=json-render-diagnostics"],
        cwd=ROOT, stdout=subprocess.PIPE, text=True, check=True,
    )
    for line in build.stdout.splitlines():
        event = json.loads(line)
        if event["reason"] == "compiler-artifact" and event["target"]["name"] == "anno":
            return next(Path(f) for f in event["filenames"] if f.endswith(".rlib"))
    raise SystemExit("Cargo built no anno library")


def check(library, output_dir, case, expected):
    invocation = subprocess.run(
        [os.environ.get("RUSTC", "rustc"), "--edition=2021", "--crate-type=lib",
         "--emit=metadata", "--error-format=json", "--out-dir", output_dir,
         "--extern", f"anno={library}", "-L", f"dependency={library.parent / 'deps'}",
         "--cfg", f'review_case="{case}"', str(FIXTURE)],
        cwd=ROOT, capture_output=True, text=True,
    )
    errors = [msg for line in invocation.stderr.splitlines()
              if (msg := json.loads(line)).get("level") == "error"]
    codes = [msg["code"]["code"] for msg in errors if msg.get("code")]
    if expected is None:
        matched = invocation.returncode == 0
    else:
        matched = invocation.returncode != 0 and codes == [expected]
    if not matched:
        details = "".join(msg.get("rendered", "") for msg in errors)
        raise SystemExit(f"{case}: expected {expected or 'success'}, got {codes}\n{details}")
    print(f"{case}: {expected or 'compiled'}")


def main():
    library = build_library()
    with tempfile.TemporaryDirectory() as output_dir:
        for case, expected in CASES.items():
            check(library, output_dir, case, expected)


if __name__ == "__main__":
    main()
