"""Verify and prepare the pinned Otter, GLiNEREL and Kniv release snapshots for CI."""

import argparse
import subprocess
import sys
from pathlib import Path

from ner_language_length import SAUERKRAUT

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from _prepare_snapshot import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    for variant in ("cross", "bi"):
        subprocess.run(
            [
                sys.executable,
                SCRIPTS / "prepare_otter.py",
                variant,
                root / f"otter-{variant}",
            ],
            check=True,
        )
    subprocess.run(
        [sys.executable, SCRIPTS / "download_kniv.py", "--ner-only", root / "kniv"],
        check=True,
    )
    prepare(
        root / "glinerel-72m",
        "VAGOsolutions/SauerkrautLM-GLiNEREL-72m",
        "34c024bb2ea85326526f6c5d150b9afe8bbe4808",
        {name: {"sha256": expected} for name, expected in SAUERKRAUT.items()},
    )
    print(f"Verified the GLiNEREL checkpoint in {root / 'glinerel-72m'}")


if __name__ == "__main__":
    main()
