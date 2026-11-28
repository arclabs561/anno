# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.8", "numpy==2.3.5", "pillow==13.0.0", "resvg-py==0.5.0"]
# ///
"""Build the dataset figures embedded in Anno's documentation from checked-in evidence."""

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import distribution
from pathlib import Path
from figure_catalog import NAME as CATALOG_NAME, write_catalog
from figure_coverage_report import coverage_text

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EVAL = ROOT / "docs" / "eval"
FIGURES = ROOT / "docs" / "figures"
NATIVE_VIEWS = ("tradeoff", "overlap", "errors", "paired")
SUMMARIES = (
    "dataset-coverage.csv",
    "dataset-coverage-by-domain.csv",
    "dataset-density-ranges.csv",
    "dataset-density-units.csv",
    "dataset-evaluation-coverage.csv",
    "dataset-wikiann-split-density.csv",
    "dataset-multiconer-v2-split-density.csv",
)
DATASET_REFERENCE = "dataset-explorer.html"


def build_anno():
    command = [
        "cargo",
        "build",
        "--locked",
        "-p",
        "anno-cli",
        "--bin",
        "anno",
        "--no-default-features",
        "--features",
        "extractor-html2text",
        "--message-format=json-render-diagnostics",
    ]
    messages = subprocess.check_output(command, cwd=ROOT, text=True)
    for line in messages.splitlines():
        record = json.loads(line)
        if (
            record.get("reason") == "compiler-artifact"
            and record["target"]["name"] == "anno"
            and record["target"]["kind"] == ["bin"]
            and record.get("executable")
        ):
            return Path(record["executable"])
    raise RuntimeError("Cargo did not report the anno executable.")


def native_paths(directory, extension):
    return sorted(
        path
        for view in NATIVE_VIEWS
        for path in directory.glob(f"*-{view}.{extension}")
    )


def render_native(inputs, directory, anno):
    """Render each portable input through the same schema reader as a user's report."""
    from resvg_py import svg_to_bytes

    executable = anno.resolve() if anno else build_anno()
    font_dir = distribution("matplotlib").locate_file("matplotlib/mpl-data/fonts/ttf")
    fonts = [
        str(font_dir / f"DejaVuSans{style}.ttf")
        for style in ("", "-Bold", "-Oblique", "-BoldOblique")
    ]
    for source in inputs:
        svg = directory / source.with_suffix(".svg").name
        png = svg.with_suffix(".png")
        html = svg.with_suffix(".html")
        if svg.exists() or png.exists() or html.exists():
            raise ValueError(f"More than one figure has the name {source.stem!r}.")
        for target in (svg, html):
            subprocess.run([executable, "viz", "render", source, "--output", target], check=True)
        png.write_bytes(
            svg_to_bytes(
                svg_path=str(svg),
                skip_system_fonts=True,
                font_files=fonts,
                font_family="DejaVu Sans",
                sans_serif_family="DejaVu Sans",
            )
        )
        shutil.copyfile(source, directory / source.name)


def render(directory, config, anno):
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
            "--registry-pointers",
            EVAL / "dataset-registry-pointers.csv",
            "--source-review",
            EVAL / "dataset-clinical-access.csv",
            "--source-review",
            EVAL / "dataset-source-review.csv",
        ),
        ("plot_density_ranges.py", inventory, directory, *sources),
        ("plot_wikiann_split_density.py", hub, directory, "--multiconer-v2"),
        (
            "plot_wnut_novelty.py",
            directory,
            "--from-results",
            EVAL / "wnut-submission-results.csv",
        ),
        ("plot_wnut_native.py", EVAL / "wnut-native-comparison.json", directory),
        ("plot_nuner_routing.py", EVAL / "nuner-fallback-routing.json", directory),
        ("plot_s3_source_evidence.py", EVAL / "s3-source-coverage.json", directory),
        (
            "plot_genia_nesting.py",
            directory,
            "--from-counts",
            EVAL / "genia-nesting-counts.json",
        ),
        ("plot_ner_language_length.py", directory),
        ("plot_benchmark_examples.py", directory),
        ("dataset_evidence_index.py", EVAL, directory),
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
    native = native_paths(EVAL, "json")
    write_catalog(EVAL, directory, native)
    if native:
        render_native(native, directory, anno)
    return [directory / source.with_suffix(".svg").name for source in native]


def same_result(built, published):
    if not published.is_file():
        return False
    if filecmp.cmp(built, published, shallow=False):
        return True
    if built.suffix in {".csv", ".md", ".svg", ".html"}:
        try:
            return built.read_text(encoding="utf-8") == published.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            return False
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
        help="Export figures, offline HTML, native inputs and derived CSVs to this directory.",
    )
    parser.add_argument(
        "--anno",
        type=Path,
        help="Use a prebuilt anno executable for native figures. By default, build this checkout.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--update",
        action="store_true",
        help="Refresh published figures, offline reports and derived CSVs.",
    )
    modes.add_argument(
        "--check",
        action="store_true",
        help="Report stale published results and exit nonzero for differences.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as work:
        directory = Path(work) / "figures"
        native_svgs = render(directory, Path(work) / "matplotlib", args.anno)
        pngs = sorted(directory.glob("*.png"))
        if not pngs or {p.stem for p in pngs} != {
            p.stem for p in directory.glob("*.svg")
        }:
            parser.exit(1, "Every figure must render both PNG and SVG.\n")
        results = {p: FIGURES / p.name for p in pngs}
        results.update({p: FIGURES / p.name for p in native_svgs})
        html_reports = [svg.with_suffix(".html") for svg in native_svgs]
        results.update({p: FIGURES / p.name for p in html_reports})
        results.update({directory / name: EVAL / name for name in SUMMARIES})
        results[directory / DATASET_REFERENCE] = EVAL / DATASET_REFERENCE
        results[directory / CATALOG_NAME] = EVAL / CATALOG_NAME
        if any(not built.is_file() for built in results):
            parser.exit(1, "A figure, offline report or derived CSV is missing.\n")
        report = EVAL / "dataset-coverage.md"
        built_report = directory.parent / report.name
        built_report.write_text(
            coverage_text(EVAL, directory, report.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        results[built_report] = report
        if args.output or not (args.update or args.check):
            output = args.output or ROOT / ".generated" / "dataset-figures"
            for path in sorted(directory.iterdir()):
                publish(path, output / path.name)
            print(
                f"Wrote {len(pngs)} PNG/SVG pairs, {len(native_svgs)} native inputs and HTML reports, the dataset explorer, CLI catalog and {len(SUMMARIES)} CSVs to {output}"
            )
            if not (args.update or args.check):
                return

        published_figures = (
            set(FIGURES.glob("*.png"))
            | set(native_paths(FIGURES, "svg"))
            | set(native_paths(FIGURES, "html"))
        )
        missing = sorted(published_figures - set(results.values()))
        if missing:
            names = ", ".join(p.name for p in missing)
            parser.exit(
                1, f"Published figures without a rendered counterpart: {names}\n"
            )
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
        print(
            f"Verified {len(pngs)} published PNGs, {len(native_svgs)} native SVG/HTML reports, the dataset explorer, CLI catalog, {len(SUMMARIES)} derived CSVs and the coverage guide's counts."
        )


if __name__ == "__main__":
    main()
