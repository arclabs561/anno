"""Count score annotations in the Distant Listening Corpus Zenodo release.

The combined TSV and the per-piece project tree are parallel encodings.
Only metadata-listed scores and their canonical harmonies contribute to the
record count. The project tree also contains older and alternative analyses.
"""

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath

METADATA = "distant_listening_corpus.metadata.tsv"
EXPANDED = "distant_listening_corpus.expanded.tsv"
POSITION_AND_LABEL = ("mc", "mn", "mc_onset", "staff", "voice", "label")


def table(archive, member, columns):
    with archive.open(member) as file:
        reader = csv.DictReader(
            io.TextIOWrapper(file, encoding="utf-8-sig", newline=""), delimiter="\t"
        )
        if not set(columns) <= set(reader.fieldnames or []):
            raise ValueError(f"Missing expected score columns in {member}")
        for row in reader:
            if None in row or any(not row.get(key) for key in columns):
                raise ValueError(f"Incomplete or misaligned row in {member}")
            yield row


def annotate_digest(digest, fields):
    # Keep field boundaries and row order reproducible across the two schemas.
    digest.update(
        json.dumps(fields, separators=(",", ":"), ensure_ascii=False).encode()
    )
    digest.update(b"\n")


def score_annotations(archive, member):
    digest = hashlib.sha256()
    count = 0
    for row in table(archive, member, POSITION_AND_LABEL):
        annotate_digest(digest, [row[key] for key in POSITION_AND_LABEL])
        count += 1
    return count, digest.hexdigest()


def describe_distant_listening(path):
    """Return direct-survey fields for one ZIP, including the comparable scope."""
    with zipfile.ZipFile(path) as archive:
        expected = {}
        for row in table(archive, METADATA, ("corpus", "piece", "label_count")):
            key = row["corpus"], row["piece"]
            if key in expected or not row["label_count"].isdigit():
                raise ValueError(
                    "Duplicate score identifier or invalid published label count"
                )
            expected[key] = int(row["label_count"])
        if not expected:
            raise ValueError("Empty Distant Listening score catalogue")

        counts = Counter()
        signatures = {}
        supplements = ""
        if EXPANDED in archive.NameToInfo:
            layout = "Combined expanded annotation table"
            digests = defaultdict(hashlib.sha256)
            for row in table(
                archive, EXPANDED, ("corpus", "piece", *POSITION_AND_LABEL)
            ):
                key = row["corpus"], row["piece"]
                if key not in expected:
                    raise ValueError("Expanded annotation refers to an unknown score")
                annotate_digest(
                    digests[key], [row[field] for field in POSITION_AND_LABEL]
                )
                counts[key] += 1
            signatures = {key: digest.hexdigest() for key, digest in digests.items()}
        else:
            layout = "Canonical per-piece harmonies in the project archive"
            canonical = set()
            for key in expected:
                corpus, piece = key
                member = f"{corpus}/harmonies/{piece}.harmonies.tsv"
                if member in archive.NameToInfo:
                    counts[key], signatures[key] = score_annotations(archive, member)
                    canonical.add(member)
            # Some corpora also publish an older table under PIECE.tsv, or an
            # alternative analysis that is absent from the main score catalogue.
            copies = alternatives = 0
            for member in archive.namelist():
                parts = PurePosixPath(member).parts
                if (
                    len(parts) != 3
                    or parts[1] != "harmonies"
                    or not member.endswith(".tsv")
                    or member in canonical
                ):
                    continue
                piece = parts[2].removesuffix(".tsv").removesuffix(".harmonies")
                count, signature = score_annotations(archive, member)
                if signatures.get((parts[0], piece)) == signature:
                    copies += count
                else:
                    alternatives += count
            supplements = (
                f" Additional harmonies tables contain {copies} duplicate row occurrences"
                f" and {alternatives} alternate-analysis rows outside the catalogue."
            )

    if any(counts[key] != published for key, published in expected.items()):
        raise ValueError(
            "Parsed annotations disagree with a score's published label count"
        )
    # Hash only the logical score ID, count, and ordered positions/labels. Metadata,
    # column ordering, container timestamps and encoding details differ by release.
    catalogue = hashlib.sha256()
    for key in sorted(expected):
        annotate_digest(catalogue, [*key, counts[key], signatures.get(key, "")])
    annotated = sum(bool(n) for n in counts.values())
    return {
        "unit": "catalogued harmonic annotation rows",
        "records": sum(counts.values()),
        "split_or_subset": layout,
        "reason": (
            f"{annotated} annotated scores among {len(expected)} catalogue entries. "
            f"Parallel representations share ordered score/position/label fingerprint "
            f"{catalogue.hexdigest()}.{supplements}"
        ),
    }
