"""Read registry-linked dataset files and describe exactly which units were counted."""

import csv
import hashlib
import io
import json
import re
import shutil
import tarfile
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from medieval_charter_survey import survey_charter_files
from survey_repo_files import crossre_counts, measure
from tweetnerd_survey import describe as describe_tweetnerd
from zenodo_file_survey import describe_distant_listening

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "crates/anno-eval/src/eval/dataset_registry.rs"
SOURCE_RULES = REGISTRY.with_name("shared_cache_exclusions.tsv")
LOCAL_FILES = {
    "AnnoNerSmoke": ["crates/anno-eval/fixtures/matrix-smoke.conll"],
    "AnnoTimexSmoke": ["scripts/qa/fixtures/matrix-smoke-timex.jsonl"],
    "AnnoTimexContrast": ["scripts/qa/fixtures/timex-contrast/rows.jsonl"],
    "HumanVoiceAgentInteraction": [
        "testdata/human_voice_agent/transcripts.jsonl",
        "testdata/human_voice_agent/response_tokens.jsonl",
        "testdata/human_voice_agent/discourse_deixis.jsonl",
    ],
}
PUBLISHER_FILES = {
    "FINER": ["https://ndownloader.figshare.com/files/36144501"],
    "WinogradSchemaChallengeWSC": [
        "https://cs.nyu.edu/~davise/papers/WinogradSchemas/WSCollection.xml?download=1"
    ],
    "BioRED": ["https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip"],
    "MACCROBAT": [
        "https://ndownloader.figshare.com/files/17493650",
        "https://ndownloader.figshare.com/files/21489405",
    ],
    "TweetNERD": [
        f"https://zenodo.org/records/6617192/files/{part}.public.tsv"
        for part in ["Academic", "OOD", *(f"part_{n}" for n in range(12))]
    ],
    "DistantListeningCorpus": [
        "https://zenodo.org/records/15150283/files/distant_listening_corpus.zip",
        "https://zenodo.org/records/15150283/files/distant_listening_corpus-v3.1.zip",
    ],
    "MedievalCharterNER": [
        "https://zenodo.org/api/records/6463699/files/Multilingual_NER_medieval_charters.zip/content"
    ],
}
TAG_FIRST = {"MitMovie", "MitRestaurant"}
BIO_FILES = TAG_FIRST | {
    "WikiGold",
    "Wnut17",
    "WNUT16",
    "GENIANested",
    "ZELDA",
    "NCBIDisease",
    "MasakhaNER",
    "NoiseBench",
    "CrossWeigh",
    "JNLPBA",
    "SECFilingsNER",
    "AnnoNerSmoke",
}
FILE_FIELDS = [
    "id",
    "source",
    "scope",
    "split_or_subset",
    "status",
    "sha256",
    "bytes",
    "unit",
    "records",
    "tokens",
    "tagged",
    "reason",
]
TAG = re.compile(r"O|[BI]-.+")
DIRECT = re.compile(r"\.(?:bio|conllu?|tsv|csv|jsonl?|txt|zip|gz|xml|ann|bz2)$", re.I)
PAPERS = {
    "aclanthology.org",
    "arxiv.org",
    "direct.mit.edu",
    "www.sciencedirect.com",
    "pmc.ncbi.nlm.nih.gov",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "link.springer.com",
}
CONTROLLED = {
    "ContactAuthors": "Registry requires contacting the authors",
    "Registration": "Registry requires registration",
    "Local": "Local dataset has no provisioned file in this checkout",
    "DependsOnOther": "Registry requires an independently provisioned source corpus",
    "Deprecated": "Registry marks this source deprecated",
    "NotYetReleased": "Registry marks the dataset unreleased",
}


def registry_rows():
    body = REGISTRY.read_text().split("\ndefine_datasets! {", 1)[1]
    rows = []
    task_counts = Counter()
    for key, entry in re.findall(r"^    (\w+) \{\n(.*?)^    \},?", body, re.M | re.S):
        fields = dict(
            re.findall(
                r'^        (name|url|format|domain|language): "([^"\n]*)"', entry, re.M
            )
        )
        if "name" not in fields:
            continue
        match = re.search(r"^        access_status: (\w+)", entry, re.M)
        fields.update(id=key, access=match[1] if match else "Public")
        for name in ("tasks", "splits"):
            values = re.search(r"^        " + name + r": \[(.*?)\]", entry, re.M | re.S)
            values = re.findall('"([^"]+)"', values[1]) if values else []
            if name == "tasks":
                task_counts.update(set(values))
            else:
                fields["declared_splits"] = "|".join(values)
        rows.append(fields)
    if len({row["id"] for row in rows}) != len(
        re.findall(r'^        name: "', body, re.M)
    ):
        raise ValueError(
            "The inventory must account for every named registry entry exactly once"
        )
    return rows, task_counts


def direct_sources(row, local_only):
    ident, url = row["id"], row["url"]
    if ident in LOCAL_FILES:
        return LOCAL_FILES[ident], ""
    if ident in local_only:
        return [], "Raw-source rules require a provisioned local copy"
    if row["access"] in CONTROLLED:
        return [], CONTROLLED[row["access"]]
    if ident == "FINERFood":
        return (
            [],
            "Same Figshare release as FINER; see FINER to avoid counting the same file twice",
        )
    if ident == "TempEval3":
        return [], "Release bundles TimeBank source text; no provisioned local archive"
    if ident in PUBLISHER_FILES:
        return PUBLISHER_FILES[ident], ""
    if not url:
        return [], "Registry supplies no URL"
    point = urlsplit(url)
    if point.scheme != "https":
        return [], "Registry URL is outside the HTTPS file reader"
    if (
        point.hostname == "raw.githubusercontent.com"
        or DIRECT.search(point.path)
        or "/resolve/" in point.path
        or "/ndownloader/files/" in point.path
        or ("/bitstreams/" in point.path and point.path.endswith("/content"))
    ):
        return [url], ""
    if point.hostname == "github.com":
        return (
            [],
            "GitHub project or collection pointer; see the repository file surveys",
        )
    if point.hostname in ("huggingface.co", "datasets-server.huggingface.co"):
        return [], "Hub reference; see the configuration and split survey"
    if point.hostname == "universaldependencies.org":
        return [], "Treebank reference; see the repository file survey"
    if point.hostname in PAPERS or point.path.lower().endswith(".pdf"):
        return [], "Publication reference without a verified corpus file"
    if point.hostname in (
        "catalog.ldc.upenn.edu",
        "physionet.org",
        "www.i2b2.org",
        "docs.google.com",
    ):
        return [], "Access or catalog page without a provisioned corpus file"
    return [], "Project or publisher page without a resolved direct file in this reader"


def cached_file(source, cache):
    if not source.startswith("https://"):
        return ROOT / source
    path = cache / hashlib.sha256(source.encode()).hexdigest()
    if path.is_file():
        return path
    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as tmp:
        partial = Path(tmp.name)
        try:
            for attempt in range(2):
                try:
                    with urlopen(
                        Request(source, headers={"User-Agent": "Anno dataset survey"}),
                        timeout=45,
                    ) as response:
                        if response.status != 200 or response.headers.get(
                            "Content-Range"
                        ):
                            raise OSError("Expected the complete response body")
                        shutil.copyfileobj(response, tmp)
                        expected = response.headers.get("Content-Length")
                        if expected is not None and tmp.tell() != int(expected):
                            raise OSError("Incomplete download")
                    break
                except HTTPError as exc:
                    if attempt or exc.code not in (429, 500, 502, 503, 504):
                        raise
                    tmp.seek(0)
                    tmp.truncate()
                    time.sleep(1)
            tmp.flush()
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    return path


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bio_counts(lines, ident):
    """Count covered tokens once across layers. Keep sequence boundaries for diagnostics."""
    tokens = tagged = blocks = block_tokens = block_tagged = 0
    forms = Counter()
    densities = []
    words = []
    previous = ""
    keep_forms = ident in {"WikiGold", "WNUT16"}
    for line in lines:
        parts = line.split()
        if parts and ident == "ZELDA" and parts[0] == "#":
            continue
        if not parts or parts[0] == "-DOCSTART-":
            if words:
                forms[" ".join(words).casefold()] += 1
                words = []
            previous = ""
            if block_tokens:
                blocks += 1
                if ident in TAG_FIRST:
                    densities.append(100 * block_tagged / block_tokens)
                block_tokens = block_tagged = 0
            continue
        if len(parts) < 2 or (ident == "GENIANested" and len(parts) != 5):
            raise ValueError("A token is missing its annotation column")
        token, tag = (
            (parts[-1], parts[0]) if ident in TAG_FIRST else (parts[0], parts[-1])
        )
        tags = parts[1:] if ident == "GENIANested" else [tag]
        if not all(TAG.fullmatch(value) for value in tags):
            raise ValueError("Unrecognized BIO label")
        covered = any(value != "O" for value in tags)
        tokens += 1
        tagged += covered
        block_tokens += 1
        block_tagged += covered
        if keep_forms:
            kind = tag[2:]
            if words and (kind != previous or tag.startswith("B-")):
                forms[" ".join(words).casefold()] += 1
                words = []
            if covered:
                words.append(token)
            previous = kind
    if not tokens:
        raise ValueError("Empty token file")
    if block_tokens:
        blocks += 1
        if ident in TAG_FIRST:
            densities.append(100 * block_tagged / block_tokens)
    if words:
        forms[" ".join(words).casefold()] += 1
    return (
        dict(
            unit="nonempty token blocks", records=blocks, tokens=tokens, tagged=tagged
        ),
        forms,
        densities,
    )


def annotated_json(lines, ident):
    if ident in {"CrossREAI", "CrossRELiterature"}:
        records, tokens, tagged = crossre_counts(lines)
        return dict(unit="sentences", records=records, tokens=tokens, tagged=tagged)
    records = 0
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Expected one JSON object per line")
        records += 1
    if not records:
        raise ValueError("Empty JSONL file")
    return dict(unit="JSONL records", records=records)


def describe_zip(path, ident):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if ident == "MACCROBAT":
            spans = 0
            for name in names:
                if name.endswith(".ann") and "__MACOSX/" not in name:
                    with archive.open(name) as ann:
                        spans += sum(bool(re.match(rb"^T\d+\t", line)) for line in ann)
            if not spans:
                raise ValueError("No BRAT annotations in MACCROBAT release")
            return dict(
                unit="BRAT text-bound annotations in .ann members",
                records=spans,
            )
        if ident == "BioRED":
            docs = 0
            for split in ("Train", "Dev", "Test"):
                with archive.open(f"BioRED/{split}.BioC.JSON") as part:
                    docs += len(json.load(part)["documents"])
            return dict(unit="BioC documents in Train/Dev/Test JSON", records=docs)
        if ident == "CRAFT":
            files = [
                n
                for n in names
                if "/structural-annotation/dependency/conllu/" in n
                and n.endswith(".conllu")
            ]
            if not files:
                raise ValueError("No CRAFT dependency annotation files")
            words = 0
            for name in files:
                with archive.open(name) as part:
                    words += sum(line.split(b"\t", 1)[0].isdigit() for line in part)
            return dict(
                unit="dependency-annotated articles",
                records=len(files),
                tokens=words,
            )
        if ident == "ECBPlus":
            files = [n for n in names if re.fullmatch(r"ECB\+/\d+/[^/]+\.xml", n)]
            words = 0
            for name in files:
                with archive.open(name) as part:
                    doc = ElementTree.parse(part).getroot()
                    if doc.tag != "Document":
                        raise ValueError("Unexpected ECB+ document")
                    words += len(doc.findall("token"))
            if not words:
                raise ValueError("No ECB+ document tokens")
            return dict(unit="ECB+ XML documents", records=len(files), tokens=words)
        if ident == "AnnoCTR":
            tokens = tagged = 0
            for split in ("train", "dev", "test"):
                (name,) = [
                    n for n in names if n.endswith(f"/AnnoCTR/ner_bio/{split}.bio")
                ]
                with archive.open(name) as part:
                    for line in io.TextIOWrapper(part, encoding="utf-8"):
                        if line.strip():
                            fields = line.rstrip("\r\n").split("\t")
                            if len(fields) != 7 or not all(
                                TAG.fullmatch(t) for t in fields[1:]
                            ):
                                raise ValueError("Unexpected AnnoCTR NER annotation")
                            tokens += 1
                            tagged += any(t != "O" for t in fields[1:])
            return dict(
                unit="tokens across train/dev/test; six tag layers combined, train_ext excluded",
                records=tokens,
                tokens=tokens,
                tagged=tagged,
            )
        members = sum(not item.is_dir() for item in archive.infolist())
    return dict(unit="archive file entries", records=members)


def describe(path, source, row):
    ident = row["id"]
    with path.open("rb") as f:
        header = f.read(200).lstrip().lower()
    if header.startswith((b"<!doctype html", b"<html", b"<head")):
        raise ValueError("URL returned a web page instead of the linked data file")
    if header.startswith(b"%pdf-"):
        return dict(
            reason="Linked bitstream contains a PDF; no validated annotation file"
        )
    if ident == "DistantListeningCorpus":
        return describe_distant_listening(path)
    if ident == "TweetNERD":
        return describe_tweetnerd(path)
    if ident == "FINER" and header.startswith(b"rar!"):
        try:
            import libarchive
        except ImportError:
            return dict(
                reason="The FINER RAR release needs libarchive-c and libarchive"
            )
        with libarchive.file_reader(str(path)) as archive:
            for member in archive:
                if member.pathname == "finer.conll":
                    with tempfile.TemporaryFile() as plain:
                        for block in member.get_blocks():
                            plain.write(block)
                        plain.seek(0)
                        return bio_counts(
                            io.TextIOWrapper(plain, encoding="utf-8-sig"), ident
                        )[0]
        raise ValueError("FINER release lacks finer.conll")
    if zipfile.is_zipfile(path):
        return describe_zip(path, ident)
    name = unquote(urlsplit(source).path).lower()
    if ident == "WinoBias":
        with path.open(encoding="utf-8-sig") as f:
            records = 0
            for line in f:
                if line.strip():
                    if not re.match(r"^\d+\s", line):
                        raise ValueError("Missing WinoBias example number")
                    records += 1
        return dict(unit="numbered WinoBias examples", records=records)
    if ident in {"GICoref", "TwiConv", "QxoRef"}:
        with path.open(encoding="utf-8-sig") as f:
            if ident == "QxoRef":
                records = sum(
                    row["index"].isdigit() for row in csv.DictReader(f, delimiter="\t")
                )
            else:
                records = sum(
                    bool(line.strip()) and not line.startswith(("#begin", "#end"))
                    for line in f
                )
        return dict(
            unit="released coreference token rows", records=records, tokens=records
        )
    if ident == "HIPE2022":
        records, tokens, tagged = measure(
            path.read_bytes(), {"metric": "HIPE_coarse_literal"}
        )
        return dict(
            unit="coarse literal NER token rows",
            records=records,
            tokens=tokens,
            tagged=tagged,
        )
    if ident == "ENer":
        with path.open(encoding="utf-8-sig", newline="") as f:
            tokens = invalid = tagged = 0
            for record in csv.reader(f):
                if not record or record[0] in ("", "-DOCSTART-"):
                    continue
                if len(record) != 2:
                    raise ValueError("Expected a token and tag")
                tokens += 1
                invalid += not bool(TAG.fullmatch(record[1]))
                tagged += record[1] != "O"
        result = dict(unit="nonempty token rows", records=tokens, tokens=tokens)
        if invalid:
            result["reason"] = (
                f"{invalid} rows use tags outside the BIO schema; tagged total omitted"
            )
        else:
            result["tagged"] = tagged
        return result
    if ident == "MovieCoref" and tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            records = 0
            for member in archive:
                if member.isfile() and member.name.endswith(".coref.mapped.csv"):
                    with archive.extractfile(member) as part:
                        records += sum(
                            1
                            for _ in csv.DictReader(
                                io.TextIOWrapper(part, encoding="utf-8-sig")
                            )
                        )
        return dict(unit="mapped coreference table rows", records=records)
    if name.endswith(".conllu"):
        tokens = 0
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) != 10:
                        raise ValueError("Unexpected CoNLL-U columns")
                    tokens += fields[0].isdigit()
        return dict(unit="CoNLL-U syntactic words", records=tokens, tokens=tokens)
    if name.endswith((".tsv", ".csv")) and ident != "HIPE2022":
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter="\t" if name.endswith(".tsv") else ",")
            header = next(reader)
            if len(header) < 2:
                raise ValueError("Expected a delimited table header")
            records = sum(bool(record) for record in reader)
        return dict(unit="table rows after header", records=records)
    if name.endswith(".ann"):
        with path.open(encoding="utf-8-sig") as f:
            spans = sum(bool(re.match(r"^T\d+\t", line)) for line in f)
        return dict(unit="BRAT text-bound annotations", records=spans)
    if name.endswith(".jsonl") or ident in {
        "GoogleRE",
        "CrossREAI",
        "CrossRELiterature",
    }:
        with path.open(encoding="utf-8-sig") as f:
            return annotated_json(f, ident)
    if name.endswith(".json"):
        with path.open(encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, list):
            return dict(unit="JSON array records", records=len(data))
        if ident == "FewRel" and all(isinstance(v, list) for v in data.values()):
            return dict(unit="relation examples", records=sum(map(len, data.values())))
        if ident == "MuDoCo" and isinstance(data.get("dialogs"), (list, dict)):
            return dict(unit="dialogs", records=len(data["dialogs"]))
    if ident == "WinogradSchemaChallengeWSC":
        records = sum(tag.tag == "schema" for _, tag in ElementTree.iterparse(path))
        if records:
            return dict(unit="Winograd schemas", records=records)
    return {}


# Names as released, including archives that bundle parallel representations.
DIRECT_SUBSETS = {
    "AnnoCTR": "train/dev/test ner_bio; extended training and linking views excluded",
    "BioRED": "Train/Dev/Test BioC JSON; alternate XML and PubTator encodings excluded",
    "CRAFT": "structural-annotation/dependency/conllu only",
    "ECBPlus": "ECB+/topic/document XML; metadata and duplicate macOS files excluded",
    "FINER": "finer.conll; combined file without split boundaries",
    "HIPE2022": "German HIPE-2020 test, v2.1",
    "MasakhaNER": "Yoruba test only",
    "MitMovie": "English test",
    "MitRestaurant": "English test",
    "MultiNERD": "English test file only",
    "WikiGold": "complete linked release",
    "Wnut17": "registered annotated test file",
    "WNUT16": "test",
    "MovieCoref": "mapped coreference CSVs; parallel exports excluded",
    "SECFilingsNER": "FIN3 only within the test directory",
    "GENIANested": "test, all four layers",
}


def survey_entry(row, cache, local_only):
    sources, reason = direct_sources(row, local_only)
    evidence, forms, densities = [], Counter(), []
    for source in sources:
        obs = dict.fromkeys(FILE_FIELDS, "")
        scope = (
            "publisher release file"
            if row["id"] in PUBLISHER_FILES
            else (
                "registry-linked file"
                if source.startswith("https://")
                else "checked-in fixture"
            )
        )
        member = unquote(Path(urlsplit(source).path).name)
        subset = DIRECT_SUBSETS.get(
            row["id"], member or "linked file; split unspecified"
        )
        if row["id"] == "MACCROBAT":
            subset = (
                "2018 BRAT release" if member == "17493650" else "2020 BRAT release"
            )
        obs.update(id=row["id"], source=source, scope=scope, split_or_subset=subset)
        if row["id"] == "MedievalCharterNER":
            try:
                evidence.extend(survey_charter_files(source, cache))
            except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
                evidence.append(dict(obs, status="read failed", reason=str(exc)[:160]))
            continue
        try:
            path = cached_file(source, cache)
            obs.update(bytes=path.stat().st_size, sha256=file_digest(path))
        except (URLError, OSError, ValueError) as exc:
            obs.update(status="read failed", reason=str(exc)[:160])
        else:
            try:
                if row["id"] in BIO_FILES:
                    with path.open(encoding="utf-8-sig") as file:
                        counts, names, samples = bio_counts(file, row["id"])
                    forms.update(names)
                    densities.extend(samples)
                else:
                    counts = describe(path, source, row)
                obs.update(counts)
                if counts.get("tagged") is not None:
                    obs.update(status="tagged tokens measured")
                elif counts.get("unit") == "archive file entries":
                    obs.update(
                        status="archive index only",
                        reason="Archive members counted; corpus records require a format-specific reader",
                    )
                elif "records" in counts:
                    obs.update(
                        status="records measured",
                        reason=counts.get(
                            "reason",
                            "No comparable entity-token density measured for this representation",
                        ),
                    )
                else:
                    obs.update(
                        status="file read only",
                        reason=counts.get(
                            "reason", "No validated record reader for these bytes"
                        ),
                    )
            except (
                UnicodeError,
                ValueError,
                KeyError,
                TypeError,
                IndexError,
                csv.Error,
                zipfile.BadZipFile,
                ElementTree.ParseError,
                StopIteration,
            ) as exc:
                obs.update(
                    status="unparsed response",
                    reason=f"{type(exc).__name__}: could not validate file against its reader",
                )
        evidence.append(obs)
    successful = [
        o for o in evidence if o["status"] not in ("read failed", "unparsed response")
    ]
    measured = [o for o in successful if o["tagged"] != ""]
    updates = dict(
        result="Registry metadata only", bytes="", tokens="", tagged="", reason=reason
    )
    if evidence:
        updates["result"] = (
            "Tagged token counts"
            if measured
            else ("File read" if successful else evidence[0]["status"])
        )
        updates["reason"] = " | ".join(
            dict.fromkeys(o["reason"] for o in evidence if o["reason"])
        )
        sizes = [o["bytes"] for o in evidence if o["bytes"] != ""]
        updates["bytes"] = sum(sizes) if sizes else ""
    if measured:
        updates.update(
            tokens=sum(o["tokens"] for o in measured),
            tagged=sum(o["tagged"] for o in measured),
        )
    return row | updates, evidence, forms, densities


def local_source_ids():
    return {
        fields[0].removesuffix(".cache")
        for line in SOURCE_RULES.read_text().splitlines()
        if (fields := line.split("\t"))[1] == "local"
    }
