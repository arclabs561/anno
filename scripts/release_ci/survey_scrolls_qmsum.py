"""Measure the public SCROLLS QMSum transformation at a verified Hub revision."""

import hashlib
import json
import re
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

HUB = "https://huggingface.co"
REPOSITORY = "tau/scrolls"
# The SCROLLS publisher README links REPOSITORY's qmsum.zip.
PAPER_REVISION = "bfc0da0747976418cd0c4b8837db023ea567ba84"


def fetch_json(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "anno-public-corpus-survey"}
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def partition_counts(file):
    totals = Counter()
    for line in file:
        if not line.strip():
            continue
        item = json.loads(line)
        if (
            not isinstance(item["input"], str)
            or not item["input"].strip()
            or not item["id"]
        ):
            raise ValueError("SCROLLS record lacks an input or ID")
        output = item.get("output")
        if isinstance(output, str):
            count = int(bool(output.strip()))
        elif output is None:
            count = 0
        elif isinstance(output, list) and all(isinstance(ref, str) for ref in output):
            count = sum(bool(ref.strip()) for ref in output)
        else:
            raise ValueError("unknown SCROLLS reference-summary representation")
        totals["questions"] += 1
        totals["questions with released reference summaries"] += bool(count)
        totals["released reference summaries"] += count
    if not totals["questions"]:
        raise ValueError("empty SCROLLS QMSum partition")
    return totals


def survey(registry_url, cache, revision=None):
    source = f"{HUB}/datasets/{REPOSITORY}/resolve/{revision or 'main'}/qmsum.zip"
    base = dict(
        id="ScrollsQMSum",
        registry_url=registry_url,
        repository="tau-nlp/scrolls",
        revision=PAPER_REVISION,
    )
    try:
        metadata = fetch_json(
            f"{HUB}/api/datasets/{REPOSITORY}"
            + (f"/revision/{revision}" if revision else "")
        )
        if metadata.get("private") or metadata.get("gated") not in (
            None,
            False,
            "false",
        ):
            raise ValueError(
                "publisher marks the transformed SCROLLS release private or gated"
            )
        resolved = metadata["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", resolved) or (
            revision and resolved != revision
        ):
            raise ValueError("could not resolve the requested SCROLLS Hub revision")
        source = f"{HUB}/datasets/{REPOSITORY}/resolve/{resolved}/qmsum.zip"
        tree = fetch_json(f"{HUB}/api/datasets/{REPOSITORY}/tree/{resolved}")
        entry = next((item for item in tree if item["path"] == "qmsum.zip"), None)
        if not entry:
            raise ValueError("qmsum.zip is absent from the SCROLLS publisher tree")
        expected = entry["lfs"]["oid"]
        expected_size = int(entry["size"])
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("invalid SCROLLS archive digest")
        dest = cache / expected
        if not dest.exists():
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as tmp:
                temp_path = Path(tmp.name)
                try:
                    with urllib.request.urlopen(
                        source + "?download=true", timeout=120
                    ) as response:
                        while chunk := response.read(1024 * 1024):
                            tmp.write(chunk)
                    tmp.close()
                    if temp_path.stat().st_size != expected_size:
                        raise ValueError("incomplete SCROLLS ZIP")
                    temp_path.replace(dest)
                finally:
                    temp_path.unlink(missing_ok=True)
        with dest.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(
                "SCROLLS archive differs from the publisher's LFS checksum"
            )
        rows = []
        with zipfile.ZipFile(dest) as z:
            for split in ("train", "validation", "test"):
                paths = [
                    n
                    for n in z.namelist()
                    if "/__MACOSX/" not in "/" + n and Path(n).name == split + ".jsonl"
                ]
                if len(paths) != 1:
                    raise ValueError("ambiguous or absent SCROLLS split " + split)
                path = paths[0]
                with z.open(path) as f:
                    totals = partition_counts(f)
                for unit, value in totals.items():
                    rows.append(
                        dict(
                            base,
                            path=path,
                            split=split,
                            status="ok",
                            reason="SCROLLS-transformed QMSum release; bytes and SHA-256 describe the publisher ZIP.",
                            source_url=source,
                            sha256=digest,
                            bytes=expected_size,
                            metric="count",
                            unit=unit,
                            value=value,
                        )
                    )
        return rows
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
        return [
            dict(
                base,
                source_url=source,
                status="unmeasured",
                reason=f"Public SCROLLS QMSum ZIP: {type(exc).__name__}: {exc}",
            )
        ]
