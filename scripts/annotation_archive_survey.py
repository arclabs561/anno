"""Read original Elgold, GermEval, SemEval ABSA and Saraga corpus annotations."""

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import PurePosixPath
import re
from urllib.parse import quote
from uuid import UUID
from xml.etree import ElementTree as ET
import zipfile

from download_elgold import CATEGORY_NAMES, ELGOLD_ENTITY_CLASSES, parse_elgold_text

# Elgold concept DOI 10.34808/f3q9-9k64 resolves to release DOI 10.34808/8zbz-1e66.
ELGOLD = "https://mostwiedzy.pl/en/open-research-data/elgold-gold-standard-multi-genre-dataset-for-named-entity-recognition-and-linking,202503241553453028951-0/download"
# The GermEval organizers list their three gold files at https://sites.google.com/site/germeval2014ner/data.
RELEASES = {
    "ELGold": {
        ELGOLD: ("v1.1", "4c2afa2eb6622307da4bf279aa9d0420fc75abec13f2dc143d49a9ec8e307aec"),
    },
    "GermEval2014": {
        "https://drive.usercontent.google.com/download?id=1Jjhbal535VVz2ap4v4r_rN1UEHTdLK5P&export=download": (
            "train",
            "98682c21700e5bd7959ac361150679929e4af5d91e4dd7c379cc89133a810994",
        ),
        "https://drive.usercontent.google.com/download?id=1ZfRcQThdtAR5PPRjIDtrVP7BtXSCUBbm&export=download": (
            "dev",
            "2ffe8405acc6bb6111b62a88623aa1e1f2e6db16b77fc005c54163541807aa05",
        ),
        "https://drive.usercontent.google.com/download?id=1u9mb7kNJHWQCWyweMDRMuTFoOHOfeBTH&export=download": (
            "test",
            "ec6e6adce5ed71cee119ae758e569ea4be54c6be5319b3b49dfcdcb05e51301b",
        ),
    },
    "ProductReviewNER": {
        "https://alt.qcri.org/semeval2014/task4/data/uploads/laptops-trial.xml": (
            "laptops",
            "0971a30e6add454d344ac011385fb8407db4193db048c8cd8e7a82908ac5172d",
        ),
        "https://alt.qcri.org/semeval2014/task4/data/uploads/restaurants-trial.xml": (
            "restaurants",
            "48d758803cd5183e9b8ee8a04520889a4b7ff371a2ff66e399ba193620e5bc41",
        ),
    },
}
SOURCES = {ident: list(files) for ident, files in RELEASES.items()}
SARAGA_COMMIT = "bd103295689ad11a5ed19038b7469e91b5c3c124"
SARAGA_ZIP = f"https://codeload.github.com/MTG/saraga/zip/{SARAGA_COMMIT}"
SARAGA_RAW = f"https://raw.githubusercontent.com/MTG/saraga/{SARAGA_COMMIT}/"
SOURCES["Saraga"] = [SARAGA_ZIP]
BIO = re.compile(r"[BI]-(?:PER|LOC|ORG|OTH)(?:part|deriv)?\Z")


def elgold(path):
    genres = defaultdict(Counter)
    with zipfile.ZipFile(path) as archive:
        files = [m.filename for m in archive.infolist() if not m.is_dir()]
        if not files or len(set(files)) != len(files):
            raise ValueError("Missing or duplicate Elgold document")
        for name in files:
            member = re.fullmatch(r"data/(\d[a-z]?)_\d+\.txt", name)
            if not member or member[1] not in CATEGORY_NAMES:
                raise ValueError("Unexpected Elgold archive member or category")
            raw = archive.read(name).decode("utf-8-sig")
            doc = parse_elgold_text(PurePosixPath(name).stem, raw)
            annotations = doc.annotations
            if not (raw.count("{{") == raw.count("}}") == len(annotations)):
                raise ValueError("Unparsed Elgold mention markup")
            for ann in annotations:
                if (
                    ann.entity_class not in ELGOLD_ENTITY_CLASSES
                    or doc.text[ann.start_char : ann.end_char] != ann.mention
                ):
                    raise ValueError("Invalid Elgold mention offsets or class")
            genres[doc.category].update(
                documents=1, mentions=len(annotations), linked=sum(bool(a.target) for a in annotations)
            )
    return [
        dict(
            split_or_subset=genre,
            unit="annotated documents",
            records=c["documents"],
            reason=f"{c['mentions']} mentions, {c['linked']} Wikipedia-linked, {c['mentions'] - c['linked']} NIL. Elgold v1.1 groups documents by genre without train/test splits.",
        )
        for genre, c in sorted(genres.items())
    ]


def germeval(path, split):
    counts = Counter()
    last_token = 0
    previous = ("O", "O")
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip() or line.startswith("#"):
                last_token = 0
                previous = ("O", "O")
                continue
            number, token, outer, inner = line.rstrip("\r\n\t").split("\t")
            if not token or int(number) != last_token + 1:
                raise ValueError("Invalid GermEval token sequence")
            counts["sentences"] += last_token == 0
            last_token += 1
            for layer, tag, prior in zip(("outer", "inner"), (outer, inner), previous):
                if tag != "O" and (
                    not BIO.fullmatch(tag) or tag.startswith("I-") and tag[2:] != prior[2:]
                ):
                    raise ValueError("Invalid GermEval BIO sequence")
                counts[layer + "_spans"] += tag.startswith("B-")
                counts[layer + "_tokens"] += tag != "O"
            counts["tokens"] += 1
            counts["tagged"] += outer != "O" or inner != "O"
            previous = (outer, inner)
    if not counts["sentences"] or not counts["inner_spans"]:
        raise ValueError("Expected both GermEval annotation layers")
    return [
        dict(
            split_or_subset=split,
            unit="sentences with two BIO layers",
            records=counts["sentences"],
            tokens=counts["tokens"],
            tagged=counts["tagged"],
            reason=f"Outer layer: {counts['outer_spans']} spans on {counts['outer_tokens']} tokens. Embedded layer: {counts['inner_spans']} spans on {counts['inner_tokens']} tokens. Tagged totals count each token once across layers.",
        )
    ]


def semeval(path, domain):
    root = ET.parse(path).getroot()
    ids = set()
    mentions = categories = 0
    for sentence in root:
        ident = sentence.get("id")
        text = sentence.findtext("text")
        if (
            root.tag != "sentences"
            or sentence.tag != "sentence"
            or not ident
            or ident in ids
            or not text
        ):
            raise ValueError("Invalid SemEval sentence")
        ids.add(ident)
        for term in sentence.findall("aspectTerms/aspectTerm"):
            start, end = int(term.attrib["from"]), int(term.attrib["to"])
            if (
                not 0 <= start < end <= len(text)
                or text[start:end] != term.attrib["term"]
                or not term.get("polarity")
            ):
                raise ValueError("Invalid SemEval aspect-term offsets or polarity")
            mentions += 1
        for category in sentence.findall("aspectCategories/aspectCategory"):
            if not category.get("category") or not category.get("polarity"):
                raise ValueError("Invalid SemEval aspect category")
            categories += 1
    if not ids or not mentions:
        raise ValueError("Missing SemEval aspect annotations")
    return [
        dict(
            split_or_subset=f"{domain}/trial",
            unit="review sentences",
            records=len(ids),
            reason=f"{mentions} offset-verified aspect terms and {categories} category judgments. Public organizer trial; training and final gold releases have separate access terms.",
        )
    ]


def saraga(path):
    """Identify each recording by MusicBrainz UUID and each measured file by its own SHA."""
    rows = []
    prefix = f"saraga-{SARAGA_COMMIT}/"

    def emit(member, data, partition, unit, count, note):
        rows.append(
            dict(
                source=SARAGA_RAW + quote(member.removeprefix(prefix)),
                sha256=hashlib.sha256(data).hexdigest(),
                bytes=len(data),
                scope="publisher metadata and annotation repository",
                split_or_subset=partition,
                unit=unit,
                records=count,
                reason=note,
            )
        )

    with zipfile.ZipFile(path) as archive:
        names = sorted(archive.namelist())
        if len(names) != len(set(names)):
            raise ValueError("Ambiguous paths in Saraga release")
        for tradition, raag, taal in (("carnatic", "raaga", "taala"), ("hindustani", "raags", "taals")):
            base = f"{prefix}dataset/{tradition}/"
            index_file = base + "file_paths.csv"
            raw_index = archive.read(index_file)
            index = {}
            for entry in csv.DictReader(io.StringIO(raw_index.decode("utf-8"))):
                mbid = str(UUID(entry["mbid"]))
                if mbid in index or not entry["filepath"]:
                    raise ValueError("Invalid Saraga recording index")
                index[mbid] = entry["filepath"]
            records = {}
            by_member = {}
            totals = Counter()
            for member in names:
                file = PurePosixPath(member)
                if not member.startswith(base) or file.name != file.parent.name + ".json":
                    continue
                data = archive.read(member)
                recording = json.loads(data)
                mbid = str(UUID(recording["mbid"]))
                duration = recording["length"]
                if (
                    mbid in records
                    or not recording["title"]
                    or type(duration) is not int
                    or duration <= 0
                    or not all(isinstance(recording[k], list) for k in ("artists", raag, taal))
                ):
                    raise ValueError("Invalid or duplicate Saraga editorial record")
                stem = member.removeprefix(base).removesuffix(".json")
                records[mbid] = stem
                by_member[member.removesuffix(".json")] = duration / 1000
                selection = "indexed" if mbid in index else "outside publisher index"
                emit(
                    member,
                    data,
                    f"{tradition}/{selection}",
                    "editorial recording records",
                    1,
                    "One recording UUID with a title, duration and artist, raag and taal fields. Audio and pitch files are outside this survey.",
                )
            if not index or not index.keys() <= records.keys():
                raise ValueError("A Saraga index record lacks corresponding editorial metadata")
            for member in names:
                if not member.startswith(base) or not member.endswith(".sama-manual.txt"):
                    continue
                duration = by_member[member.removesuffix(".sama-manual.txt")]
                data = archive.read(member)
                times = [float(v) for v in data.decode("utf-8-sig").splitlines() if v.strip()]
                if not times:
                    totals["empty"] += 1
                    continue
                if any(not math.isfinite(t) for t in times) or any(
                    a > b for a, b in zip(times, times[1:])
                ):
                    raise ValueError("Invalid Saraga manual sama timestamp sequence")
                earlier = sum(t < 0 for t in times)
                later = sum(t > duration for t in times)
                totals.update(files=1, timestamps=len(times), early=earlier, late=later)
                emit(
                    member,
                    data,
                    tradition,
                    "manual sama timestamps within recording duration",
                    len(times) - earlier - later,
                    f"{earlier} negative and {later} after-duration rows excluded. Times are seconds and reported durations are milliseconds.",
                )
            nonindexed = len(records.keys() - index.keys())
            path_diffs = sum(stem != records[uid] for uid, stem in index.items())
            emit(
                index_file,
                raw_index,
                tradition,
                "distinct indexed recording UUIDs",
                len(index),
                f"Related editorial files contain {nonindexed} additional UUIDs and {path_diffs} index path differences. {totals['files']} nonempty manual sama files contain {totals['timestamps']} timestamp rows, including {totals['early']} negative and {totals['late']} after-duration values. {totals['empty']} empty files are excluded. The publisher assigns CC BY-NC 4.0 to audio and manual/automatic annotations and AGPLv3 to code. Editorial metadata terms are unspecified in LICENSE.md.",
            )
    return rows


def measure(ident, url, path):
    if ident == "Saraga" and url == SARAGA_ZIP:
        return saraga(path)
    scope, digest = RELEASES[ident][url]
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != digest:
            raise ValueError("Publisher corpus bytes differ from the pinned release")
    if ident == "ELGold":
        return elgold(path)
    if ident == "GermEval2014":
        return germeval(path, scope)
    return semeval(path, scope)
