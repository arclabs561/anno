# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.8", "numpy==2.3.5", "pillow==13.0.0"]
# ///
"""Build the dataset figures embedded in Anno's documentation from checked-in evidence."""

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EVAL = ROOT / "docs" / "eval"
FIGURES = ROOT / "docs" / "figures"
SUMMARIES = (
    "dataset-coverage.csv",
    "dataset-coverage-by-domain.csv",
    "dataset-density-ranges.csv",
    "dataset-wikiann-split-density.csv",
)


def render(directory, config):
    inventory = EVAL / "dataset-figure-inventory.csv"
    hub = EVAL / "dataset-hf-splits.csv"
    sources = [
        "--direct-files",
        EVAL / "dataset-direct-files.csv",
        "--repo-files",
        EVAL / "dataset-repo-files.csv",
        "--hf-splits",
        hub,
        "--github-sources",
        EVAL / "dataset-github-sources.csv",
    ]
    jobs = [
        (
            "plot_measurement_coverage.py",
            inventory,
            directory,
            *sources,
            "--ci-prechecks",
            EVAL / "dataset-matrix-evidence.csv",
        ),
        ("plot_density_ranges.py", inventory, directory, *sources),
        ("plot_wikiann_split_density.py", hub, directory),
        (
            "plot_wnut_novelty.py",
            directory,
            "--from-results",
            EVAL / "wnut-submission-results.csv",
        ),
        (
            "plot_genia_nesting.py",
            directory,
            "--from-counts",
            EVAL / "genia-nesting-counts.json",
        ),
        ("plot_ner_language_length.py", directory),
        ("plot_benchmark_examples.py", directory),
    ]
    config.mkdir()
    (config / "matplotlibrc").write_text(
        "svg.hashsalt: anno-dataset-figures\n", encoding="utf-8"
    )
    env = {
        **os.environ,
        "MPLCONFIGDIR": str(config),
        "SOURCE_DATE_EPOCH": "0",
        "XDG_CACHE_HOME": str(config.parent / "cache"),
    }
    for script, *args in jobs:
        subprocess.run(
            [sys.executable, SCRIPTS / script, *args], check=True, env=env, cwd=ROOT
        )


def same_result(built, published):
    if not published.is_file():
        return False
    if filecmp.cmp(built, published, shallow=False):
        return True
    if built.suffix == ".csv":
        return built.read_text(encoding="utf-8") == published.read_text(
            encoding="utf-8"
        )
    if built.suffix == ".png":
        # Encoder metadata and compression can differ without a visible change.
        from PIL import Image

        with Image.open(built) as rendered:
            try:
                with Image.open(published) as existing:
                    return (
                        rendered.size == existing.size
                        and rendered.convert("RGBA").tobytes()
                        == existing.convert("RGBA").tobytes()
                    )
            except OSError:
                return False
    return False


def publish(built, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination so replacement stays atomic across filesystems.
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as f:
        staging = Path(f.name)
    try:
        shutil.copy(built, staging)
        staging.replace(destination)
    finally:
        staging.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Export PNGs, SVGs and derived CSVs to this directory.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--update",
        action="store_true",
        help="Refresh changed figures and derived CSVs in the documentation.",
    )
    modes.add_argument(
        "--check",
        action="store_true",
        help="Report stale published results and exit nonzero for differences.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as work:
        directory = Path(work) / "figures"
        render(directory, Path(work) / "matplotlib")
        pngs = sorted(directory.glob("*.png"))
        if not pngs or {p.stem for p in pngs} != {
            p.stem for p in directory.glob("*.svg")
        }:
            parser.exit(1, "Every figure must render both PNG and SVG.\n")
        results = {p: FIGURES / p.name for p in pngs}
        results.update({directory / name: EVAL / name for name in SUMMARIES})
        if any(not built.is_file() for built in results):
            parser.exit(1, "The generators did not produce every derived CSV.\n")
        if args.output or not (args.update or args.check):
            output = args.output or ROOT / ".generated" / "dataset-figures"
            for path in sorted(directory.iterdir()):
                publish(path, output / path.name)
            print(
                f"Wrote {len(pngs)} PNG/SVG pairs and {len(SUMMARIES)} CSVs to {output}"
            )
            if not (args.update or args.check):
                return

        missing = sorted(set(FIGURES.glob("*.png")) - set(results.values()))
        if missing:
            names = ", ".join(p.name for p in missing)
            parser.exit(1, f"Published PNGs without a rendered counterpart: {names}\n")
        changed = [(a, b) for a, b in results.items() if not same_result(a, b)]
        for built, destination in changed:
            if args.update:
                publish(built, destination)
            print(
                f"{'Updated' if args.update else 'Stale'} {destination.relative_to(ROOT)}"
            )
        if changed and args.check:
            parser.exit(
                1, "Run the figure generator with --update to refresh the docs.\n"
            )
        print(f"Verified {len(pngs)} published PNGs and {len(SUMMARIES)} derived CSVs.")


if __name__ == "__main__":
    main()
