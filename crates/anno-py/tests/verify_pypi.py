# SPDX-License-Identifier: MIT OR Apache-2.0
"""Verify the documented PyPI wheel, or all files from a completed release job."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen


BASELINE = "0.1.0"
# cp39-abi3 manylinux2014_x86_64, the published wheel used by ubuntu-latest.
BASELINE_WHEEL_SHA256 = "8d2549c3dcbbb3339e96d0aa2382711e10efe30b0f2e519ded4bc31e3318d901"
INDEX = "https://pypi.org/simple"
# The public ABI and platforms promised by python-wheels.yml.
PLATFORMS = {
    "Linux x86_64": r"manylinux(?:[0-9]+|_[0-9]+_[0-9]+)_x86_64",
    "Linux aarch64": r"manylinux(?:[0-9]+|_[0-9]+_[0-9]+)_aarch64",
    "macOS arm64": r"macosx_[0-9]+_[0-9]+_arm64",
    "Windows x64": r"win_amd64",
}


def release_inventory(directory, ref=""):
    # Keep artifact subdirectories separate so equal filenames cannot overwrite one another.
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    wheels = [p for p in files if p.suffix == ".whl"]
    sdists = [p for p in files if p.name.endswith(".tar.gz")]
    if not wheels or len(sdists) != 1 or len(files) != len(wheels) + 1:
        raise ValueError("Expected wheels and one source distribution in release artifacts")
    sdist = re.fullmatch(r"anno[-_]py-([a-zA-Z0-9.!+_]+)\.tar\.gz", sdists[0].name)
    if not sdist:
        raise ValueError(f"Unexpected source distribution name: {sdists[0].name}")
    version = sdist.group(1)
    prefix = "refs/tags/anno-py-v"
    if ref.startswith(prefix) and ref != prefix + version:
        raise ValueError(f"Tag {ref} does not match distribution version {version}")
    seen = set()
    for wheel in wheels:
        prefix = f"anno_py-{version}-cp39-abi3-"
        if not wheel.stem.startswith(prefix):
            raise ValueError(f"Unexpected project, version, Python tag or ABI in {wheel.name}")
        tags = wheel.stem[len(prefix):].split(".")
        platform_names = [name for name, pattern in PLATFORMS.items()
                          if all(re.fullmatch(pattern, tag) for tag in tags)]
        if len(platform_names) != 1:
            raise ValueError(f"Unexpected platform tags in {wheel.name}")
        name = platform_names[0]
        if name in seen:
            raise ValueError(f"Duplicate {name} wheel: {wheel.name}")
        seen.add(name)
    missing = sorted(PLATFORMS.keys() - seen)
    if missing:
        raise ValueError(f"Missing release wheels: {', '.join(missing)}")
    return version, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def missing_files(expected, registry_rows):
    registry = {row["filename"]: row for row in registry_rows}
    for filename in expected.keys() & registry.keys():
        row = registry[filename]
        if row["yanked"]:
            raise ValueError(f"{filename} is yanked on PyPI")
        actual = row["digests"]["sha256"]
        if actual != expected[filename]:
            raise ValueError(
                f"{filename} has SHA-256 {actual} on PyPI but this build has {expected[filename]}"
            )
    return sorted(expected.keys() - registry.keys())


def verify_release(version, expected):
    url = f"https://pypi.org/pypi/anno-py/{version}/json"
    for attempt in range(6):
        try:
            with urlopen(url, timeout=15) as response:
                missing = missing_files(expected, json.load(response)["urls"])
            if not missing:
                return
            detail = f"missing files: {missing}"
        except HTTPError as error:
            if error.code not in {404, 429, 500, 502, 503, 504}:
                raise
            detail = str(error)
        except (URLError, TimeoutError) as error:
            detail = str(error)
        if attempt < 5:
            time.sleep(10)
    raise RuntimeError(f"PyPI did not expose release {version}: {detail}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, help="Local files submitted to PyPI")
    parser.add_argument("--inventory-only", action="store_true",
                        help="Check built release files before uploading them")
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.inventory_only and not args.release_dir:
        parser.error("--inventory-only requires --release-dir")
    version, expected = BASELINE, {}
    if args.release_dir:
        version, expected = release_inventory(args.release_dir, os.environ.get("GITHUB_REF", ""))
        if args.inventory_only:
            inventory = {"version": version, "built_release_files": expected}
            args.receipt.write_text(json.dumps(inventory, indent=2) + "\n")
            print(json.dumps(inventory, indent=2))
            return
        verify_release(version, expected)

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        subprocess.run([sys.executable, "-m", "venv", directory / "venv"], check=True)
        python = directory / "venv" / "bin" / "python"
        report = directory / "install.json"
        hashes = [digest for name, digest in expected.items() if name.endswith(".whl")]
        if not expected:
            hashes = [BASELINE_WHEEL_SHA256]
        requirements = directory / "requirements.txt"
        pins = " ".join(f"--hash=sha256:{digest}" for digest in hashes)
        requirements.write_text(f"anno-py=={version} {pins}\n")
        subprocess.run([
            python, "-m", "pip", "--isolated", "install", "--index-url", INDEX,
            "--no-cache-dir", "--no-deps", "--only-binary=:all:", "--report", report,
            "--require-hashes", "-r", requirements,
        ], cwd=directory, check=True)
        package, = json.loads(report.read_text())["install"]
        info = package["download_info"]
        url = urlsplit(info["url"])
        filename = Path(unquote(url.path)).name
        digest = info["archive_info"]["hashes"]["sha256"]
        if (url.scheme != "https" or url.hostname != "files.pythonhosted.org"
                or not filename.endswith(".whl") or package["metadata"]["name"] != "anno-py"
                or package["metadata"]["version"] != version):
            raise ValueError("pip did not install the requested public PyPI wheel")
        if expected and expected.get(filename) != digest:
            raise ValueError(f"Installed {filename} does not match this workflow's build")
        surface = "checkout" if expected else f"pypi-{BASELINE}"
        subprocess.run(
            [python, "-I", Path(__file__).with_name("test_bulletin.py").resolve(), "-v"],
            cwd=directory, env={**os.environ, "ANNO_BULLETIN_SURFACE": surface}, check=True,
        )
    receipt = {
        "checkout": os.environ.get("GITHUB_SHA"), "version": version,
        "python": platform.python_version(), "platform": platform.platform(),
        "installed_wheel": filename, "url": info["url"], "sha256": digest,
        "verified_release_files": expected, "bulletin_surface": surface,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
