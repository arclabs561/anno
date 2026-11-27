"""Count the publisher's ApNER split files and 2016 ChemDataExtractor journal gold."""

import hashlib
import json
from pathlib import PurePosixPath
import zipfile

from publisher_release_survey import bio

# Cranfield data deposit https://doi.org/10.57996/cran.ceres-2614
CRANFIELD = "https://dspace.lib.cranfield.ac.uk/server/api/core/bitstreams"
APNER = {
    f"{CRANFIELD}/fbf82d16-509b-4f65-9bfa-1fa4333f1990/content": (
        "train",
        "d09c547350e4ca2d2e64eb143699a11448bd15f7236eb2c928e2625134ea2583",
    ),
    f"{CRANFIELD}/f3bc9b83-9661-410e-9351-4068456e63ed/content": (
        "dev",
        "99e81c22a7861df0d5e9e7f7f2d1027be9f94792570523517b5e1ce9e010847b",
    ),
    f"{CRANFIELD}/ff1778f0-a768-4942-b8bf-30770cdf7e93/content": (
        "test",
        "1908cc8c0cbce97fe2bd90539268804e5840d4b0032e3cc6a899c8b7bddc1b18",
    ),
}
# The publisher links this exact HTTP archive at http://chemdataextractor.org/download.
CHEM_GOLD = "http://data.chemdataextractor.org/evaluation/chemdataextractor-journal-articles-gold.zip"
CHEM_SHA256 = "78ec15b10c47f821e682debba529232fe8a25b95b02a61f0d71d5c7f450211a7"
SOURCES = {"AviationProductsNER": list(APNER), "ChemDataExtractor": [CHEM_GOLD]}
APNER_SCOPE = "Publisher-segmented token positions from Cranfield's ApNER gold splits. The three PDF renditions are excluded."
CHEM_SCOPE = "Manual gold JSON from all 50 journal evaluation articles. Counts structured reference records. Separate model outputs, article source texts and patent predictions are excluded."


def journal_gold(path):
    articles = records = 0
    with zipfile.ZipFile(path) as archive:
        members = [
            m
            for m in archive.infolist()
            if m.filename.lower().endswith(".json")
            and not any(
                part.startswith(".") or part == "__MACOSX"
                for part in PurePosixPath(m.filename).parts
            )
        ]
        if len({m.filename for m in members}) != len(members):
            raise ValueError("Duplicate JSON member in the journal gold archive")
        for member in members:
            with archive.open(member) as source:
                article = json.load(source)
            if (
                not isinstance(article, list)
                or not article
                or any(
                    not isinstance(record, dict)
                    or not record
                    or any(not isinstance(values, list) for values in record.values())
                    for record in article
                )
            ):
                raise ValueError(
                    "Expected a nonempty list of structured manual records per article"
                )
            articles += 1
            records += len(article)
    if articles != 50:
        raise ValueError("Expected all 50 journal gold files")
    return dict(
        unit="manual structured reference records",
        records=records,
        split_or_subset="2016 gold journal evaluation, 50 articles",
        reason=CHEM_SCOPE,
    )


def measure(dataset, source, path):
    if source not in SOURCES[dataset]:
        raise ValueError("Unrecognized annotation deposit source")
    expected = APNER[source][1] if dataset == "AviationProductsNER" else CHEM_SHA256
    with path.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != expected:
            raise ValueError(
                "Publisher annotation file differs from the pinned release"
            )
    if dataset == "AviationProductsNER":
        with path.open(encoding="utf-8-sig") as file:
            return [
                dict(bio(file, 2), split_or_subset=APNER[source][0], reason=APNER_SCOPE)
            ]
    return [journal_gold(path)]
