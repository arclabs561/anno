"""Read scoped public corpus releases selected by source identity.

Called by survey_hf_splits.py. Every file is checked against the pinned Hub Git
or LFS manifest before counting. ANNO_HF_SURVEY_CACHE optionally retains the
downloaded public files outside the repository for repeated local measurements.
"""

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
import zipfile
from collections import defaultdict
from concurrent.futures import as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from hf_parquet_projection import text_partition


@dataclass(frozen=True)
class Release:
    ids: str
    repo: str
    column: str
    scope: str
    configs: tuple = ()


# Resolve the revisions once per survey. The CSV stores both exact commits for replay.
RELEASES = (
    Release(
        "DocRED",
        "thunlp/docred",
        "sents",
        "Publisher documents. Human and distant training annotations are separate. Test relations are held out.",
    ),
    Release(
        "MELO",
        "Avature/MELO-Benchmark",
        "text",
        "Queries and candidate occupation records by retrieval configuration. Configurations reuse candidates and queries.",
    ),
    Release(
        "MathDial",
        "eth-nlped/mathdial",
        "conversation",
        "Publisher tutoring conversations. One row is one dialogue.",
    ),
    Release(
        "WIESPAstro",
        "adsabs/WIESP2022-NER",
        "BIO",
        "WIESP 2022 astrophysics NER release. Also registered as WIESP2022NER.",
    ),
    Release(
        "HAREM",
        "Linguateca/harem",
        "BIO",
        "Full HAREM label inventory. The selective encoding repeats the documents.",
        ("default",),
    ),
    Release(
        "GermEval2014",
        "GermanT5/germeval2014",
        "BIO",
        "GermanT5 conversion of GermEval 2014. The supplied BIO column encodes one entity layer.",
    ),
    Release(
        "BC2GMFull",
        "spyysalo/bc2gm_corpus",
        "BIO",
        "BioCreative II gene mention corpus conversion. BC2GM is a separate registry ID for this underlying corpus.",
    ),
    Release(
        "S800", "spyysalo/species_800", "BIO", "Public BIO conversion of the Species-800 corpus."
    ),
    Release(
        "BC4CHEMD CHEMDNER",
        "disi-unibo-nlp/bc4chemd",
        "BIO",
        "Chemical-only BC4CHEMD conversion. Literal tags are validated against the chemical label inventory. BC4CHEMD and CHEMDNER name the same BioCreative IV corpus.",
    ),
    Release(
        "MSNER",
        "qmeeus/MSNER-nlp",
        "tokens",
        "Four MSNER transcript languages from the paper. The separate English comparison configuration and audio are outside this measurement.",
        ("de", "es", "fr", "nl"),
    ),
    # LinCE §3.3 identifies the source corpus of each NER configuration.
    # The 2018 CodaLab organizers link LinCE as their improved release.
    Release(
        "CALCS",
        "lince-benchmark/lince",
        "words",
        "2018 CALCS NER competition inputs in the organizers' LinCE release. English-Spanish and Egyptian-MSA only. Other workshop years and tasks are outside this scope. https://aclanthology.org/W18-3219/",
        ("ner_spaeng", "ner_msaea"),
    ),
    Release(
        "CALCS2018",
        "lince-benchmark/lince",
        "words",
        "The registry's English-Spanish CALCS 2018 track in the improved LinCE release. LinCE partitioning. https://competitions.codalab.org/competitions/18725",
        ("ner_spaeng",),
    ),
    Release(
        "HindiEnglishSocialMediaNER",
        "lince-benchmark/lince",
        "words",
        "The registry-linked corpus catalog cites Singh et al. (2018). LinCE republishes those 2,079 Hindi-English tweets in its own partitions. https://aclanthology.org/W18-2405/",
        ("ner_hineng",),
    ),
    Release(
        "GLUECoS",
        "lince-benchmark/lince",
        "words",
        "Two of the eleven GLUECoS input corpora. The authors identify Singh's Hindi-English and CALCS Spanish-English NER corpora. Counts follow LinCE's republished partitions. Original GLUECoS partition assignments and the other tasks were outside this measurement. https://aclanthology.org/2020.acl-main.329/",
        ("ner_hineng", "ner_spaeng"),
    ),
    Release(
        "LinCE",
        "lince-benchmark/lince",
        "words",
        "All released language identification, NER, POS and sentiment configurations. Tasks can share inputs.",
    ),
    Release(
        "KORE50", "naist-nlp/kore50", "text", "NAIST conversion of KORE50 entity-linking inputs."
    ),
    Release(
        "WNEDWiki",
        "naist-nlp/wned-wiki",
        "text",
        "NAIST conversion of the WNED-WIKI entity-linking documents.",
    ),
    Release(
        "MHERCL",
        "n28div/MHERCL-EN",
        "sentence",
        "English MHERCL mention-level records. Multiple records can refer to one sentence.",
    ),
    Release(
        "MHERCL",
        "n28div/MHERCL-ITA",
        "sentence",
        "Italian MHERCL mention-level records. Multiple records can refer to one sentence.",
    ),
    Release(
        "WEBNLG",
        "webnlg-challenge/web_nlg",
        "original_triple_sets.otriple_set",
        "Original WebNLG release_v1 inputs. Later revisions and constrained encodings are separate releases.",
        ("release_v1",),
    ),
    Release(
        "CodeSearchNet CSN",
        "code-search-net/code_search_net",
        "func_code_string",
        "All six language partitions of CodeSearchNet. The pooled all configuration repeats these functions. CSN and CodeSearchNet refer to this same snapshot.",
        ("go", "java", "javascript", "php", "python", "ruby"),
    ),
    Release(
        "DROC",
        "lattice-nlp/DROC",
        "text",
        "Public conversion of DROC literary excerpts, cited and linked to the original dataset by its release card.",
    ),
    Release(
        "NovelCR",
        "shuaiwa16/novelCR",
        "src",
        "Authors’ Chinese NovelCR release. English books are outside this release. The Hub publishes a single partition.",
    ),
    Release(
        "Zcoref",
        "psuwannapich/Z-coref-dataset",
        "text",
        "Author release of Thai Z-coref. The registry still describes the dataset as unreleased.",
    ),
    Release(
        "MultiCoNER",
        "samanjoy2/multiconer_v1",
        "conll",
        "Original v1 CoNLL sentence blocks. The multilingual configuration repeats language-specific examples. Test inputs have withheld labels.",
    ),
    Release(
        "MultiCoNERv2",
        "MultiCoNER/multiconer_v2",
        "conll",
        "Original v2 CoNLL sentence blocks. The multilingual configuration repeats language-specific examples.",
    ),
    Release(
        "BookCoref",
        "sapienzanlp/bookcoref",
        "book-full",
        "Publisher full-book coreference annotation records. Character clusters refer to separately sourced book text.",
    ),
    Release(
        "BookCorefSplit",
        "sapienzanlp/bookcoref",
        "book-split",
        "Publisher segmented-book coreference annotation records. The alternate splitted directory repeats the split directory.",
    ),
    Release(
        "ACORD",
        "theatticusproject/acord",
        "acord",
        "Publisher legal retrieval release. Queries, candidate clauses and graded judgments are different units.",
    ),
    Release(
        "HUPD",
        "HUPD/hupd",
        "patents",
        "Publisher sample-jan-2016 archive only. Annual archives and the cumulative full collection extend beyond this measurement.",
    ),
    Release(
        "BEANSZero",
        "EarthSpeciesProject/BEANS-Zero",
        "audio-test",
        "All publisher test shards. Read the text instruction for every audio example using Parquet column ranges. Waveforms were outside the projection.",
    ),
    Release(
        "NatureLMAudio",
        "EarthSpeciesProject/NatureLM-audio-training",
        "audio-sample",
        "Systematic training sample across all four volume parts of the 17.6 TB release. Read the first, quarter, middle, three-quarter and final shard of each part. Each observation is an audio example’s text instruction. The remaining shards and waveform columns were outside the projection.",
    ),
    Release(
        "FictionNER750M",
        "SaladTechnologies/fiction-ner-750m",
        "fiction",
        "All published data/batch_*.parquet files in data.zip. Rows are tokenized chunks. Counts use raw words and Qwen-generated silver BIO tags, excluding subword labels.",
    ),
    Release(
        "B2NERD",
        "Umean/B2NERD",
        "b2nerd",
        "Refined English release. Selected public underlying corpora only. All other components, B2NERD_raw and B2NERD_all are omitted. One row is one annotated sentence.",
    ),
    Release(
        "MentionResolutionLLM",
        "KawshikManikantan/IdentifyMe",
        "identifyme",
        "Publisher IdentifyMe archive. Each released dev and test row is a multiple-choice mention question. Only aggregate counts are published.",
    ),
    Release(
        "InterlingueWikipedia",
        "wikimedia/wikipedia",
        "text",
        "All articles in the November 2023 Interlingue Wikipedia dump conversion.",
        ("20231101.ie",),
    ),
    Release(
        "FIREBALL",
        "lara-martin/FIREBALL",
        "fireball",
        "All publisher filtered gameplay-turn JSONL files. One row is one filtered turn. The card also describes a larger raw-session collection.",
    ),
)
# These name matches have positive evidence for a specific exclusion. Each
# revision identifies the reviewed candidate and can be inspected independently.
REVIEWED_EXCLUSIONS = (
    (
        "CharacterCodex",
        "NousResearch/CharacterCodex",
        "0620a3c13d05627dc4a7bd6648e6f98464f6e321",
        "The card describes personas for character role-play. The registry describes a literary character recognition corpus.",
    ),
    (
        "TRIDIS",
        "magistermilitum/Tridis",
        "98cb8e28456644eb162d70ad993a42490db264c4",
        "The published Tridis files are document images for manuscript transcription. The registry describes discourse relations.",
    ),
    (
        "HOMOMEX",
        "jhovany/Homomex2024",
        "ed41880fc2be34eab8326d652c291891735e9811",
        "The HOMO-MEX 2024 shared task evaluates LGBT-phobia detection. The registry describes Spanish homonym disambiguation.",
    ),
    (
        "ReasoningNER",
        "HuiHuang/NER-CoT",
        "f90cbd5bb4944dd642a876b69a59bd7cebf3eadc",
        "Authors identify NER-CoT as reasoning-augmented PILE-NER training examples. The registry describes a 20-dataset zero-shot evaluation suite.",
    ),
    (
        "SLUE",
        "asapp/slue",
        "67f7da031721a14cc391c7fa7c8d96411282d8a3",
        "The asapp/slue Hub distribution requires acceptance of its access terms.",
    ),
    (
        "SPoRC",
        "blitt/SPoRC",
        "d37755f5046d8fd46853e121982931b4db30388d",
        "The blitt/SPoRC Hub distribution requires acceptance of its access terms.",
    ),
    (
        "InterslavicCorpus",
        "interslavic-oss/isv-corpus",
        "8a49bd93ef46e629b6f18e3d8c447eb7bba757ae",
        "The Interslavic Linguistic Forum publishes a later MT collection under its own data license. The registry links a language homepage and lists a different license. Corpus identity is unresolved.",
    ),
    (
        "CorefUD",
        "FrancophonIA/CorefUD_1.2",
        "73ede7049b26872f4e1aa5cf43b45d229057887e",
        "The mirror identifies CorefUD 1.2 and a different publisher DOI from the linked registry release.",
    ),
    (
        "CorefUD",
        "lattice-nlp/corefud-1-4",
        "3ded5a1638e43d89b81ee24bf5b98f66943e1c60",
        "The conversion identifies CorefUD 1.4. The registry links another publisher release.",
    ),
)


def excluded_rows(ids):
    return [
        dict(
            id=i,
            repository=repo,
            source_revision=sha,
            status="excluded",
            reason=why,
            scope_note="Publisher metadata review only.",
            payload_url=f"https://huggingface.co/datasets/{repo}/tree/{sha}",
        )
        for i, repo, sha, why in REVIEWED_EXCLUSIONS
        if i in ids
    ]


BY_ID_REPO = {(i, r.repo): r for r in RELEASES for i in r.ids.split()}
RAW = {
    "conll",
    "book-full",
    "book-split",
    "acord",
    "patents",
    "fireball",
    "fiction",
    "b2nerd",
    "identifyme",
    "audio-test",
    "audio-sample",
}
B2_COMPONENTS = (
    "CrossNER_AI",
    "CrossNER_literature",
    "CrossNER_music",
    "CrossNER_politics",
    "CrossNER_science",
    "AnatEM",
    "FabNER",
    "MultiNERD",
    "TweetNER7",
    "WikiANN en",
    "WikiNeural",
    "bc5cdr",
    "ncbi",
)


def fetch(url, read):
    for attempt in range(4):
        try:
            request = (
                url
                if isinstance(url, Request)
                else Request(url, headers={"Accept-Encoding": "identity"})
            )
            with urlopen(request, timeout=150) as response:
                return read(response)
        except (HTTPError, URLError, TimeoutError) as error:
            if (
                isinstance(error, HTTPError) and error.code not in {429, 500, 502, 503, 504}
            ) or attempt == 3:
                raise
            time.sleep(2**attempt)


@lru_cache(maxsize=None)
def manifest(repo, rev):
    pinned = re.fullmatch(r"[0-9a-f]{40}", rev)
    cache = os.environ.get("ANNO_HF_SURVEY_CACHE")
    directory = Path(cache) / "manifests" / repo if cache else None
    path = directory / f"{rev}.json" if directory and pinned else None
    if path and path.is_file():
        result = json.loads(path.read_text())
    else:
        revision = "" if rev == "main" else f"/revision/{quote(rev, safe='')}"
        url = f"https://huggingface.co/api/datasets/{repo}{revision}?blobs=true"
        result = fetch(url, json.load)
    if result["id"].lower() != repo.lower() or result.get("gated") or result.get("private"):
        raise ValueError(f"Expected an anonymously distributed corpus at {repo}")
    if not re.fullmatch(r"[0-9a-f]{40}", result["sha"]) or (pinned and result["sha"] != rev):
        raise ValueError(f"Source revision differs for {repo}")
    if directory and (not path or not path.is_file()):
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as tmp:
            try:
                json.dump(result, tmp)
                tmp.close()
                os.replace(tmp.name, directory / f"{result['sha']}.json")
            finally:
                Path(tmp.name).unlink(missing_ok=True)
    return result


def file_url(repo, revision, path):
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(path)}"


@lru_cache(maxsize=1)
def shared_archives():
    return tempfile.TemporaryDirectory(prefix="anno-hf-archives-")


@lru_cache(maxsize=None)
def file_lock(url):
    return Lock()


def validate_file(handle, entry):
    digest = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {entry['size']}\0".encode())
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
        if not entry.get("lfs"):
            git_blob.update(block)
    digest_hex = digest.hexdigest()
    matches = (
        digest_hex == entry["lfs"]["sha256"]
        if entry.get("lfs")
        else git_blob.hexdigest() == entry["blobId"]
    )
    if handle.tell() != entry["size"] or not matches:
        raise ValueError(f"File differs from pinned Hub manifest: {entry['rfilename']}")
    handle.seek(0)
    return digest_hex


def download_file(url, handle):
    def copy(source):
        handle.seek(0)
        handle.truncate()
        shutil.copyfileobj(source, handle, 1024 * 1024)

    fetch(url + "?download=true", copy)
    handle.seek(0)


@contextmanager
def verified_file(url, entry, reused=False):
    cache = os.environ.get("ANNO_HF_SURVEY_CACHE")
    if reused and not cache:
        scratch_dir = shared_archives()
        cache = scratch_dir.name
    if not cache:
        with tempfile.TemporaryFile() as local:
            download_file(url, local)
            yield local, validate_file(local, entry)
        return
    dest = Path(cache) / hashlib.sha256(url.encode()).hexdigest()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(url):
        if not dest.exists():
            with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                try:
                    download_file(url, tmp)
                    validate_file(tmp, entry)
                    tmp.close()
                    os.replace(tmp.name, dest)
                finally:
                    Path(tmp.name).unlink(missing_ok=True)
    with dest.open("rb") as local:
        yield local, validate_file(local, entry)


def partitions(release, source, conversion):
    kind = release.column
    entries = defaultdict(list)
    for f in (conversion or source)["siblings"]:
        path = f["rfilename"]
        parts = path.split("/")
        if kind not in RAW:
            if (
                len(parts) != 3
                or not re.fullmatch(r"\d+\.parquet", parts[2])
                or parts[1].startswith("partial-")
            ):
                continue
            cfg, split = parts[:2]
            if release.configs and cfg not in release.configs:
                continue
        elif kind == "conll" and path.endswith(".conll"):
            cfg, split = parts[0], Path(path).stem.rsplit("_", 1)[1]
        elif (
            kind.startswith("book-")
            and path.startswith(f"bookcoref_annotations/{kind[5:]}/")
            and path.endswith(".jsonl")
        ):
            cfg, split = kind[5:], Path(path).stem.removesuffix("_split")
        elif kind == "audio-test" and re.fullmatch(r"shard_\d+\.parquet", path):
            cfg, split = "published", "test"
        elif kind == "audio-sample" and re.fullmatch(r"train/part\d+/shard_\d+\.parquet", path):
            cfg, split = parts[1], "train-sampled"
        elif kind == "fiction" and path == "data.zip":
            cfg, split = "published", "all"
        elif kind == "b2nerd" and path == "B2NERD_data.zip":
            for component in B2_COMPONENTS:
                for split in ("dev", "test", "train"):
                    entries[component, split].append(f)
            continue
        elif kind == "identifyme" and path == "IdentifyMe.zip":
            for split in ("dev", "test"):
                entries["published", split].append(f)
            continue
        elif kind == "patents" and path == "data/sample-jan-2016.tar.gz":
            cfg, split = "sample", "2016-01"
        elif kind == "acord" and path == "ACORD Dataset & ReadMe.zip":
            for split in ("queries", "corpus", "qrels/train", "qrels/valid", "qrels/test"):
                entries["published", split].append(f)
            continue
        elif kind == "fireball" and re.fullmatch(r"filtered/[0-9a-f]+\.jsonl", path):
            cfg, split = "filtered", "turns"
        else:
            continue
        if release.ids == "MHERCL":
            cfg = "en" if release.repo.endswith("-EN") else "it"
        entries[cfg, split].append(f)
    if kind == "audio-sample":
        for key, shards in entries.items():
            ordered = sorted(shards, key=lambda f: f["rfilename"])
            indices = sorted({(len(ordered) - 1) * quarter // 4 for quarter in range(5)})
            entries[key] = [dict(ordered[i], part_shards=len(ordered)) for i in indices]
    if not entries:
        raise ValueError(f"No complete {kind} partitions found in {release.repo}")
    return entries


def plan(release, source_rev="main", parquet_rev="refs/convert/parquet"):
    source = manifest(release.repo, source_rev)
    conversion = None if release.column in RAW else manifest(release.repo, parquet_rev)
    return [
        (
            release,
            source["sha"],
            conversion["sha"] if conversion else "",
            cfg,
            split,
            sorted(files, key=lambda f: f["rfilename"]),
        )
        for (cfg, split), files in sorted(partitions(release, source, conversion).items())
    ]


def text_rows(parquet, column):
    root, *fields = column.split(".")
    if root not in parquet.schema_arrow.names:
        raise ValueError(f"Missing observation column {column}")
    rows = 0
    for batch in parquet.iter_batches(columns=[column], batch_size=2048):
        values = batch.column(batch.schema.get_field_index(root))
        for name in fields:
            values = pc.struct_field(values, name)
        while pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
            if values.null_count:
                raise ValueError(f"Null observations in {column}")
            values = pc.list_flatten(values)
        if values.null_count or not (
            pa.types.is_string(values.type) or pa.types.is_large_string(values.type)
        ):
            raise ValueError(f"Invalid observation field {column}")
        rows += batch.num_rows
    return rows


def bio_counts(parquet, release, word_name="tokens", tag_name="ner_tags", blind=False):
    features = (
        json.loads((parquet.schema_arrow.metadata or {}).get(b"huggingface", b"{}"))
        .get("info", {})
        .get("features", {})
    )
    feature = features.get(tag_name, {})
    label_names = (feature[0] if isinstance(feature, list) else feature.get("feature", {})).get(
        "names"
    )
    rows = tokens = tagged = 0
    for batch in parquet.iter_batches(columns=[word_name, tag_name], batch_size=4096):
        words, tags = batch.column(word_name), batch.column(tag_name)
        if (
            words.null_count
            or tags.null_count
            or not pc.list_value_length(words).equals(pc.list_value_length(tags))
        ):
            raise ValueError("Token and tag vectors differ")
        values = pc.list_flatten(tags)
        if values.null_count or pc.list_flatten(words).null_count:
            raise ValueError("Null token or tag")
        if blind:
            if len(values) and not pc.all(pc.equal(values, "")).as_py():
                raise ValueError("Blind split contains a nonempty tag")
            rows += batch.num_rows
            tokens += len(values)
            continue
        if pa.types.is_integer(values.type):
            if (
                not label_names
                or label_names.count("O") != 1
                or (
                    len(values)
                    and (pc.min(values).as_py() < 0 or pc.max(values).as_py() >= len(label_names))
                )
            ):
                raise ValueError("Invalid embedded ClassLabel mapping")
            outside = label_names.index("O")
        else:
            pattern = r"^(O|[BIESUL]-.+)$"
            if release.column == "fiction":
                pattern = r"^(O|[BI]-(CHA|LOC|FAC|OBJ|EVT|ORG|MISC))$"
            if "BC4CHEMD" in release.ids.split():
                pattern = r"^(O|[BI]-(CHEM|Chemical|CHEMICAL))$"
            if (
                len(values)
                and not pc.all(pc.match_substring_regex(pc.unique(values), pattern)).as_py()
            ):
                raise ValueError(
                    f"Unexpected tags in {release.repo}: {pc.unique(values).to_pylist()[:40]}"
                )
            outside = "O"
        rows += batch.num_rows
        tokens += len(values)
        tagged += pc.sum(pc.not_equal(values, outside)).as_py() or 0
    return rows, tokens, None if blind else tagged


def conll_counts(file):
    rows = tokens = tagged = 0
    labelled = None
    inside = False
    for line in file:
        line = line.rstrip(b"\r\n")
        if line.startswith(b"# id "):
            inside = False
            continue
        if not line.strip():
            inside = False
            continue
        if line.endswith(b" _ _"):
            has_label = False
        else:
            word, sep, tag = line.rpartition(b" _ _ ")
            if not sep or not (tag == b"O" or re.fullmatch(rb"[BI]-.+", tag)):
                raise ValueError("Unrecognized token row or entity tag")
            has_label = True
            tagged += tag != b"O"
        if labelled is not None and labelled != has_label:
            raise ValueError("Partially annotated CoNLL partition")
        labelled = has_label
        rows += not inside
        inside = True
        tokens += 1
    return rows, tokens, tagged if labelled else None


def fiction_overlap(file):
    # Ordered checks use only word equality and document IDs. A natural repeated
    # token also counts, so the diagnostic cannot establish unique corpus size.
    overlapping = pairs = 0
    with zipfile.ZipFile(file) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith(".parquet"))
        selected = sorted({names[0], names[len(names) // 2], names[-1]})
        for name in selected:
            parquet = pq.ParquetFile(io.BytesIO(archive.read(name)))
            rows = next(
                parquet.iter_batches(batch_size=512, columns=["doc_id", "raw_tokens"])
            ).to_pylist()
            for before, after in zip(rows, rows[1:]):
                if before["doc_id"] != after["doc_id"]:
                    continue
                pairs += 1
                left, right = before["raw_tokens"], after["raw_tokens"]
                overlapping += any(
                    left[-n:] == right[:n] for n in range(1, min(len(left), len(right)) + 1)
                )
    file.seek(0)
    return (
        f" Read {len(names)} Parquet members. In the first 512 rows of the first, middle and final members, "
        f"{overlapping:,} of {pairs:,} adjacent same-document pairs share a nonempty suffix and prefix. "
        "The word-position total includes repeated context."
    )


def content_counts(file, release, cfg, split):
    kind = release.column
    if kind == "conll":
        return conll_counts(file)
    if kind in {"book-full", "book-split", "fireball"}:
        rows = 0
        for line in file:
            record = json.loads(line)
            if kind == "fireball":
                valid = all(
                    isinstance(record[key], list)
                    for key in ("combat_state_before", "combat_state_after", "commands_norm")
                )
            else:
                valid = isinstance(record["doc_key"], str) and isinstance(record["clusters"], list)
            if not valid:
                raise ValueError(f"Unrecognized {kind} record")
            rows += 1
        return rows, None, None
    if kind == "fiction":
        counts = [0, 0, 0]
        with zipfile.ZipFile(file) as archive:
            names = sorted(n for n in archive.namelist() if n.endswith(".parquet"))
            if not names or any(
                not re.fullmatch(r"data/batch_(?:\d+|final)\.parquet", n) for n in names
            ):
                raise ValueError("Unrecognized fiction archive layout")
            for member in names:
                parquet = pq.ParquetFile(io.BytesIO(archive.read(member)))
                for pos, value in enumerate(bio_counts(parquet, release, "raw_tokens", "raw_tags")):
                    counts[pos] += value
        return tuple(counts)
    if kind == "b2nerd":
        records = json.load(file)
        for example in records:
            if not isinstance(example["sentence"], str) or not isinstance(
                example["entities"], list
            ):
                raise ValueError("Unrecognized B2NERD annotated example")
        return len(records), None, None
    if kind == "identifyme":
        rows = 0
        for record in map(json.loads, file):
            if (
                not isinstance(record["text"], str)
                or not isinstance(record["options"], list)
                or "answer" not in record
            ):
                raise ValueError("Unrecognized IdentifyMe question")
            rows += 1
        return rows, None, None
    if kind == "acord":
        judgments = split.startswith("qrels/")
        rows = 0
        for record in (
            csv.DictReader(io.TextIOWrapper(file, encoding="utf-8"), delimiter="\t")
            if judgments
            else map(json.loads, file)
        ):
            if judgments:
                valid = (
                    record["query-id"] and record["corpus-id"] and 0 <= float(record["score"]) <= 5
                )
            else:
                valid = record["_id"] and isinstance(record["text"], str)
            if not valid:
                raise ValueError("Invalid ACORD record")
            rows += 1
        return rows, None, None
    if kind == "patents":
        rows = 0
        with tarfile.open(fileobj=file, mode="r|gz") as archive:
            for member in archive:
                if member.isfile() and member.name.endswith(".json"):
                    with archive.extractfile(member) as entry:
                        patent = json.load(entry)
                    if not patent["application_number"] or not isinstance(patent["abstract"], str):
                        raise ValueError("Unrecognized patent record")
                    rows += 1
        return rows, None, None
    parquet = pq.ParquetFile(file)
    if release.ids == "MSNER":
        return bio_counts(parquet, release, "tokens", "tags")
    if release.repo == "lince-benchmark/lince" and cfg.startswith("ner_"):
        return bio_counts(parquet, release, "words", "ner", blind=split == "test")
    if kind == "BIO":
        return bio_counts(parquet, release)
    return text_rows(parquet, kind), None, None


OBSERVATION = {
    "BIO": "tokens,ner_tags",
    "fiction": "raw_tokens,raw_tags",
    "b2nerd": "sentence,entities",
    "identifyme": "text,options,answer",
    "conll": "CoNLL token and optional BIO tag",
    "book-full": "doc_key,clusters",
    "book-split": "doc_key,clusters",
    "patents": "application_number,abstract",
    "fireball": "commands_norm,combat_state_before,combat_state_after",
}


def archive_member(release, cfg, split):
    if release.column == "acord":
        return (
            "ACORD Dataset & ReadMe (external)/"
            + split
            + (".tsv" if split.startswith("qrels/") else ".jsonl")
        )
    if release.column == "identifyme":
        return f"qas_{split}.jsonl"
    if release.column == "b2nerd":
        return f"B2NERD/NER_en/{cfg}/{split}.json"
    return None


def measure(job):
    release, source, conversion, cfg, split, files = job
    if release.column in {"audio-test", "audio-sample"}:
        return text_partition(job, file_url, fetch)
    extra_scope = ""
    hashes = []
    rows = tokens = tagged = 0
    has_tokens = has_tags = None
    revision = conversion or source
    for entry in files:
        path = entry["rfilename"]
        url = file_url(release.repo, revision, path)
        member = archive_member(release, cfg, split)
        with verified_file(url, entry, reused=bool(member)) as (file, digest):
            if release.column == "fiction":
                extra_scope = fiction_overlap(file)
            if member:
                with zipfile.ZipFile(file) as archive:
                    payload = archive.read(member)
                digest = hashlib.sha256(payload).hexdigest()
                url += "#member=" + quote(member)
                count, words, labels = content_counts(io.BytesIO(payload), release, cfg, split)
            else:
                count, words, labels = content_counts(file, release, cfg, split)
        if has_tokens is not None and (
            has_tokens != (words is not None) or has_tags != (labels is not None)
        ):
            raise ValueError(f"Inconsistent annotation presence in {release.repo}/{cfg}/{split}")
        has_tokens, has_tags = words is not None, labels is not None
        rows += count
        tokens += words or 0
        tagged += labels or 0
        hashes.append((path, digest))
    if not rows:
        raise ValueError(f"Empty corpus partition at {release.repo}/{cfg}/{split}")
    scope = release.scope + extra_scope
    if len(hashes) > 1:
        digest = hashlib.sha256(
            "".join(f"{digest}  {path}\n" for path, digest in hashes).encode()
        ).hexdigest()
        url = f"https://huggingface.co/datasets/{release.repo}/tree/{revision}/{quote(os.path.commonpath([p for p, _ in hashes]))}"
        scope += f" Read all {len(hashes)} files. SHA-256 covers sorted SHA256, two spaces, path and LF lines."
    column = OBSERVATION.get(release.column, release.column)
    if release.ids == "MSNER":
        column = "tokens,tags"
        scope += " Manual evaluation labels." if split == "test" else " Silver annotation labels."
    if release.repo == "lince-benchmark/lince" and cfg.startswith("ner_"):
        column = "words,ner"
        if split == "test":
            scope += " Gold test labels are withheld."
    if release.column == "acord":
        column = "query-id,corpus-id,score" if split.startswith("qrels/") else "_id,text"
    return [
        dict(
            id=i,
            repository=release.repo,
            source_revision=source,
            parquet_revision=conversion,
            config=cfg,
            split=split,
            status=(
                "scanned" if has_tags else "scanned_unlabelled" if has_tokens else "scanned_records"
            ),
            rows=rows,
            tokens=tokens if has_tokens else "",
            tagged=tagged if has_tags else "",
            scope_note=scope,
            observation_column=column,
            payload_url=url,
            payload_sha256=digest,
            reason=(
                "Publisher file contains token inputs with withheld labels"
                if has_tokens and not has_tags
                else ""
            ),
        )
        for i in release.ids.split()
    ]


def survey(releases, pool):
    rows = []
    jobs = []
    for release, pending in [(r, pool.submit(plan, r)) for r in releases]:
        try:
            jobs.extend(pending.result())
        except (OSError, ValueError, KeyError) as error:
            rows.extend(
                dict(
                    id=i,
                    repository=release.repo,
                    status="unavailable",
                    reason=str(error),
                    scope_note=release.scope,
                )
                for i in release.ids.split()
            )
    futures = {pool.submit(measure, job): job for job in jobs}
    for future in as_completed(futures):
        release, source, conversion, cfg, split, files = futures[future]
        try:
            rows.extend(future.result())
        except (OSError, ValueError, KeyError, pa.ArrowException) as error:
            print(f"{release.repo}/{cfg}/{split}: {error}", flush=True)
            rows.extend(
                dict(
                    id=i,
                    repository=release.repo,
                    source_revision=source,
                    parquet_revision=conversion,
                    config=cfg,
                    split=split,
                    status="scan_failed",
                    scope_note=release.scope,
                    reason=str(error),
                )
                for i in release.ids.split()
            )
    return rows


def replay(row):
    release = BY_ID_REPO[row["id"], row["repository"]]
    jobs = plan(release, row["source_revision"], row["parquet_revision"])
    job = next(j for j in jobs if j[3:5] == (row["config"], row["split"]))
    current = next(r for r in measure(job) if r["id"] == row["id"])
    if row.get("provider_rows") and current.get("provider_rows") is not None:
        if str(current["provider_rows"]) != row["provider_rows"]:
            raise ValueError("Pinned provider row count differs")
    for field in (
        "status",
        "rows",
        "tokens",
        "tagged",
        "payload_url",
        "payload_sha256",
        "observation_column",
        "scope_note",
    ):
        if str(current[field]) != row[field]:
            raise ValueError(
                f"{row['repository']}/{row['config']}/{row['split']} differs on {field}"
            )
    return current
