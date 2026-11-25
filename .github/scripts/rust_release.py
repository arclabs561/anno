"""Publish the Anno crates in dependency order and verify their immutable registry checksums."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

CRATES = ("anno", "anno-eval", "anno-cli")


def release_version(metadata, ref):
    versions = {
        item["name"]: item["version"]
        for item in metadata["packages"]
        if item["id"] in metadata["workspace_members"] and item["name"] in CRATES
    }
    if set(versions) != set(CRATES) or len(set(versions.values())) != 1:
        raise ValueError(f"Expected all three workspace crates at one version, found {versions}")
    version = versions["anno"]
    if ref not in {"refs/heads/main", f"refs/tags/v{version}"}:
        raise ValueError(f"Ref {ref} does not identify the workspace release v{version}")
    return version


def indexed(crate, version, expected_digest):
    # The three supported names all have the same crates.io sparse-index prefix.
    request = Request(f"https://index.crates.io/an/no/{crate}",
                      headers={"User-Agent": "anno-release (github.com/arclabs561/anno)"})
    try:
        with urlopen(request, timeout=20) as response:
            for line in response:
                row = json.loads(line)
                if row["vers"] == version:
                    if row["name"] != crate or row["yanked"] or row["cksum"] != expected_digest:
                        raise ValueError(f"{crate} {version} on crates.io is yanked or differs from this build")
                    return True
    except HTTPError as error:
        if error.code != 404:
            raise
    return False


def publish(crate, version, target):
    package = ["-p", crate, "--locked", "--registry", "crates-io", "--target-dir", str(target)]
    subprocess.run(["cargo", "package", "--no-verify", *package], check=True)
    archive = target / "package" / f"{crate}-{version}.crate"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if indexed(crate, version, digest):
        print(f"{crate} {version} already matches crates.io SHA-256 {digest}", flush=True)
        return

    # A registry may accept an upload even if the client loses the response.
    # In both cases the sparse index must confirm the bytes before proceeding.
    result = subprocess.run(["cargo", "publish", *package])
    for attempt in range(6):
        if indexed(crate, version, digest):
            print(f"Verified {crate} {version} on crates.io at SHA-256 {digest}", flush=True)
            return
        if attempt < 5:
            time.sleep(10)
    raise RuntimeError(f"{crate} {version} absent from crates.io after cargo publish exited {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "publish"])
    args = parser.parse_args()
    metadata = json.loads(subprocess.check_output([
        "cargo", "metadata", "--locked", "--format-version", "1", "--no-deps",
    ]))
    version = release_version(metadata, os.environ["GITHUB_REF"])
    print(f"Workspace release v{version} ({', '.join(CRATES)})", flush=True)
    if args.command == "publish":
        with tempfile.TemporaryDirectory(prefix="anno-publish-") as temporary:
            for crate in CRATES:
                publish(crate, version, Path(temporary))


if __name__ == "__main__":
    main()
