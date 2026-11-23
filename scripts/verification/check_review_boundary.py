#!/usr/bin/env python3
"""Compile valid and invalid Rust callers of anno's reviewed catalog assignment API."""

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
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


def check(case, expected):
    # Cargo supplies the target toolchain, dependencies and build directories.
    # Arguments following -- affect this fixture alone.
    result = subprocess.run(
        ["cargo", "rustc", "--locked", "-p", "anno", "--no-default-features",
         "--test", "reviewed_kb_compile", "--message-format=json", "--",
         "--cfg", f'review_case="{case}"'],
        cwd=ROOT, capture_output=True, text=True,
    )
    errors = []
    for line in result.stdout.splitlines():
        event = json.loads(line)
        if event["reason"] == "compiler-message" and event["message"]["level"] == "error":
            errors.append(event["message"])
    codes = [error["code"]["code"] for error in errors if error.get("code")]
    if expected is None:
        matched = result.returncode == 0
    else:
        matched = result.returncode != 0 and codes == [expected]
    if not matched:
        diagnostics = "".join(error.get("rendered", "") for error in errors)
        raise SystemExit(f"{case}: expected {expected or 'success'}, got {codes}\n"
                         f"{diagnostics}{result.stderr}")
    print(f"{case}: {expected or 'compiled'}")


if __name__ == "__main__":
    for name, diagnostic in CASES.items():
        check(name, diagnostic)
