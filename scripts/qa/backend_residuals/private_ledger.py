# SPDX-License-Identifier: MIT OR Apache-2.0
"""Keep source-derived diagnostic rows in the caller's external working directory."""

import json
from pathlib import Path


def external_path(path):
    path = Path(path).resolve()
    if any((folder / ".git").is_file() or (folder / ".git/HEAD").is_file()
           for folder in (path, *path.parents)):
        raise ValueError("use a path outside Git checkouts for per-document diagnostics")
    return path


def save_new(path, value):
    data = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with external_path(path).open("xb") as output:
        output.write(data)
