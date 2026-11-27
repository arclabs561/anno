"""Read only VoxPopuli transcript TSVs from the author's SLUE v0.2 blind ZIP."""

import ast
import csv
import hashlib
import io
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from http_range_file import RemoteFile

# The pinned asappresearch/slue-toolkit scripts/download_datasets.sh publishes
# this blind release. data/README.md defines the normalized character offsets.
SOURCE = "https://public-dataset-model-store.awsdev.asapp.com/users/sshon/public/slue/slue-voxpopuli_v0.2_blind.zip"
SPLITS = ("fine-tune", "dev", "test_blind")


def counts(raw, split):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter="\t")
    required = {"id", "normalized_text"}
    if split != "test_blind":
        required.add("normalized_ner")
    if not required <= set(reader.fieldnames or []):
        raise ValueError("unexpected SLUE transcript fields")
    result = Counter()
    for row in reader:
        transcript = row["normalized_text"]
        if not row["id"] or not transcript:
            raise ValueError("missing SLUE identifier or normalized transcript")
        result["utterances"] += 1
        result["tokens"] += len(transcript.split())
        if split == "test_blind":
            continue
        annotations = ast.literal_eval(row["normalized_ner"])
        if not isinstance(annotations, list):
            raise TypeError("expected normalized NER annotation list")
        spans = []
        for kind, start, length in annotations:
            if not kind or start < 0 or length <= 0 or start + length > len(transcript):
                raise ValueError("invalid SLUE normalized entity span")
            spans.append((start, start + length))
        result["mention annotations"] += len(spans)
        result["tagged tokens"] += sum(
            any(word.start() < end and start < word.end() for start, end in spans)
            for word in re.finditer(r"\S+", transcript)
        )
    if not result["utterances"]:
        raise ValueError("empty SLUE transcript file")
    return result


def survey(base, fetch=urllib.request.urlopen):
    scope = "VoxPopuli v0.2 blind release; bytes and SHA-256 cover the uncompressed normalized-transcript TSV. Audio and later test labels excluded."
    rows = []
    try:
        request = urllib.request.Request(
            SOURCE, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"}
        )
        with fetch(request, timeout=30) as response:
            match = re.fullmatch(
                r"bytes 0-0/(\d+)", response.headers.get("Content-Range", "")
            )
            if response.status != 206 or not match or len(response.read(2)) != 1:
                raise OSError("publisher endpoint did not return a one-byte range")
            size = int(match[1])
            etag = response.headers.get("ETag")

        def read_range(request, check):
            if etag:
                request.add_header("If-Match", etag)
            with fetch(request, timeout=60) as response:
                return check(response)

        with (
            RemoteFile(SOURCE, size, read_range) as stream,
            zipfile.ZipFile(stream) as archive,
        ):
            for split in SPLITS:
                name = f"slue-voxpopuli_{split}.tsv"
                matches = [
                    member for member in archive.namelist() if Path(member).name == name
                ]
                if len(matches) != 1:
                    raise ValueError("missing or ambiguous SLUE manifest " + name)
                path = matches[0]
                raw = archive.read(path)
                digest = hashlib.sha256(raw).hexdigest()
                for unit, value in counts(raw, split).items():
                    rows.append(
                        dict(
                            base,
                            path=path,
                            split=split,
                            status="ok",
                            reason=scope,
                            source_url=SOURCE,
                            sha256=digest,
                            bytes=len(raw),
                            metric="count",
                            unit=unit,
                            value=value,
                        )
                    )
    except (
        OSError,
        ValueError,
        SyntaxError,
        KeyError,
        TypeError,
        zipfile.BadZipFile,
    ) as exc:
        rows.append(
            dict(
                base,
                source_url=SOURCE,
                status="unmeasured",
                reason=f"Official blind-release transcript ZIP: {type(exc).__name__}: {exc}",
            )
        )
    return rows
