"""Verify the documented PyPI wheel, or all files from a completed release job."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
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


def release_inventory(directory):
    files = sorted(directory.iterdir())
    wheels = [p for p in files if p.suffix == ".whl"]
    sdists = [p for p in files if p.name.endswith(".tar.gz")]
    if not wheels or len(sdists) != 1 or len(files) != len(wheels) + 1:
        raise ValueError("Expected wheels and one source distribution in release artifacts")
    versions = {wheel.name.split("-")[1] for wheel in wheels}
    if len(versions) != 1:
        raise ValueError("Wheel versions disagree")
    version = versions.pop()
    if not all(p.name.startswith(f"anno_py-{version}-") for p in wheels):
        raise ValueError("Unexpected wheel project or version")
    if sdists[0].name not in {f"anno_py-{version}.tar.gz", f"anno-py-{version}.tar.gz"}:
        raise ValueError("Source distribution project or version differs from wheels")
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
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    version, expected = BASELINE, {}
    if args.release_dir:
        version, expected = release_inventory(args.release_dir)
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
