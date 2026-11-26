"""Read the four annotated text members of Zenodo release 6463699.

The other 15 GB of this ZIP contains embeddings, model weights and logs.
Pinned ranges are half-open offsets from its ZIP64 directory, including each
member's local header and data descriptor. A sparse temp file lets zipfile
validate headers and CRCs without materializing intervening model bytes.
"""

from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile
from zlib import error as DeflateError

ARCHIVE_BYTES = 15_034_863_872
DIRECTORY = (
    15_034_850_638,
    ARCHIVE_BYTES,
    "25a3601a6797fac1a2d3e79a21537d9af0e2de086f16cb3e74b10cba03002489",
)
PARTS = {
    "datasets/train_set_all_17_03_2022.txt": (
        "train",
        1335416,
        7089043,
        "d937d13e7406a336a613e751859e948a9b3b3a526dc6132d3d967fb0c7b33aff",
    ),
    "datasets/val_set_all_17_03_2022.txt": (
        "validation",
        7089043,
        7516144,
        "b0c9e4eab0474746efa0f012add217b98bf7a695049d90223b0b71c91d46012e",
    ),
    "datasets/test_set_all_17_03_2022.txt": (
        "test",
        300370,
        1335416,
        "4f8bc49565f0279efa8f58b894f3c767d44418f3d54a5da5d16835d110a72682",
    ),
    "datasets/Nesle_Eslonza.txt": (
        "Nesle_Eslonza",
        71,
        300370,
        "3b991d06a0df721a29decd2f6d2e569263acf86135e78c63ae6e1c3801371939",
    ),
}
LANGUAGES = {"FR", "lat", "es", "latin", "spanish"}
PERSON = {"O", "B-PERS", "I-PERS"}
PLACE = {"O", "B-LOC", "I-LOC"}


def range_block(source, cache, start, stop, expected_hash):
    destination = cache / expected_hash
    length = stop - start
    if destination.is_file() and destination.stat().st_size == length:
        cached = destination.read_bytes()
        if sha256(cached).hexdigest() == expected_hash:
            return cached
    request = Request(
        source,
        headers={
            "Range": f"bytes={start}-{stop - 1}",
            "Accept-Encoding": "identity",
            "User-Agent": "Anno dataset survey",
        },
    )
    for attempt in range(2):
        try:
            with urlopen(request, timeout=45) as result:
                expected = f"bytes {start}-{stop - 1}/{ARCHIVE_BYTES}"
                if (
                    result.status != 206
                    or result.headers.get("Content-Range") != expected
                ):
                    raise OSError(
                        "Zenodo did not return exactly the requested ZIP range"
                    )
                data = result.read(length + 1)
                if len(data) != length or sha256(data).hexdigest() != expected_hash:
                    raise ValueError(
                        "Medieval charter ZIP range differs from the pinned release"
                    )
            break
        except HTTPError as exc:
            if attempt or exc.code not in (429, 500, 502, 503, 504):
                raise
    with NamedTemporaryFile(dir=cache, delete=False) as temp:
        partial = Path(temp.name)
        try:
            temp.write(data)
            temp.flush()
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
    return data


@contextmanager
def charter_zip(source, cache):
    complete = cache / sha256(source.encode()).hexdigest()
    if complete.is_file() and complete.stat().st_size == ARCHIVE_BYTES:
        with complete.open("rb") as file:
            file.seek(DIRECTORY[0])
            if sha256(file.read()).hexdigest() != DIRECTORY[2]:
                raise ValueError("Cached ZIP directory differs from the pinned release")
            with ZipFile(file) as archive:
                yield archive
        return

    range_cache = cache / "medieval-charter-6463699"
    range_cache.mkdir(exist_ok=True, parents=True)
    with TemporaryFile(dir=range_cache) as sparse:
        for start, stop, digest in [DIRECTORY, *(part[1:] for part in PARTS.values())]:
            sparse.seek(start)
            sparse.write(range_block(source, range_cache, start, stop, digest))
        with ZipFile(sparse) as archive:
            yield archive


def survey_charter_files(source_url, cache_dir):
    """Return per-member direct-file ledger rows, with hashes of uncompressed data."""
    previous_ids = set()
    rows = []
    try:
        with charter_zip(source_url, Path(cache_dir)) as archive:
            members = Counter(
                n
                for n in archive.namelist()
                if n.startswith("datasets/") and not n.endswith("/")
            )
            if members != Counter(PARTS.keys()):
                raise ValueError(
                    "Expected the four published medieval charter text members"
                )
            for name, (split, start, _, _) in PARTS.items():
                member = archive.getinfo(name)
                if member.header_offset != start:
                    raise ValueError(f"Unexpected ZIP position for {name}")
                documents = set()
                tokens = tagged = empty_tokens = 0
                digest = sha256()
                with archive.open(member) as stream:
                    for number, raw in enumerate(stream, 1):
                        digest.update(raw)
                        line = raw.decode("utf-8")
                        if not line.strip():
                            continue
                        # One train document adds an empty trailing TSV column.
                        fields = line.rstrip("\r\n\t").split("\t")
                        if len(fields) != 5:
                            raise ValueError(
                                f"{name}:{number}: expected five TSV fields"
                            )
                        token, person, place, language, charter = fields
                        if (
                            person not in PERSON
                            or place not in PLACE
                            or language not in LANGUAGES
                            or not charter.strip()
                        ):
                            raise ValueError(
                                f"{name}:{number}: unrecognized charter annotation"
                            )
                        if not token.strip():
                            if person != "O" or place != "O":
                                raise ValueError(
                                    f"{name}:{number}: annotated empty token"
                                )
                            empty_tokens += 1
                            continue
                        documents.add(charter.strip())
                        tokens += 1
                        tagged += person != "O" or place != "O"
                if not tokens or previous_ids & documents:
                    raise ValueError(
                        f"{name}: empty corpus or charter ID reused across release parts"
                    )
                previous_ids.update(documents)
                rows.append(
                    dict(
                        id="MedievalCharterNER",
                        source=f"{source_url}#{name}",
                        scope="publisher archive member",
                        split_or_subset=split,
                        status="tagged tokens measured",
                        sha256=digest.hexdigest(),
                        bytes=member.file_size,
                        unit="charter IDs",
                        records=len(documents),
                        tokens=tokens,
                        tagged=tagged,
                        reason=f"PERS/LOC layer union; excluded {empty_tokens} empty-token rows labelled O/O",
                    )
                )
    except (BadZipFile, DeflateError) as exc:
        raise ValueError("Medieval charter ZIP failed member integrity checks") from exc
    return rows
