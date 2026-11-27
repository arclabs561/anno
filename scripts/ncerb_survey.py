"""Reproduce the public NCER benchmarking recipe and count its retained documents and spans.

The four component names come from NCER.tasks and the span filters from prepare_dataset
at github.com/WadoodAbdul/clinical_ner_benchmark, revision identified below.
"""

import io
import math
import re
import zipfile
from collections import Counter
from xml.etree import ElementTree as ET

REVISION = "de0480188cbea9266611a33738ffbbeb3608c15a"
CHIA = "https://huggingface.co/datasets/bigbio/chia/resolve/b3296f43da51c5d703afce13dd435096af89f4cc/chia_source/train/0000.parquet"
BIORED = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip"
BC5CDR = "https://raw.githubusercontent.com/JHnlp/BioCreative-V-CDR-Corpus/dd0b3165a94338874979a2a8aaedf74b67466b9c/CDR_Data.zip"
NCBI_SPLITS = {"train": "train", "validation": "develop", "test": "test"}
NCBI = {
    f"https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBI{name}set_corpus.zip": split
    for split, name in NCBI_SPLITS.items()
}
SOURCES = [CHIA, BIORED, BC5CDR, *NCBI]
LABELS = {
    "CHIA": {
        "Condition": "condition",
        "Drug": "drug",
        "Procedure": "procedure",
        "Measurement": "measurement",
    },
    "BioRED": {
        "GeneOrGeneProduct": "gene",
        "DiseaseOrPhenotypicFeature": "condition",
        "ChemicalEntity": "drug",
        "SequenceVariant": "gene variant",
    },
    "BC5CDR": {"Chemical": "drug", "Disease": "condition"},
    "NCBI": dict.fromkeys(
        ("CompositeMention", "DiseaseClass", "Modifier", "SpecificDisease"), "condition"
    ),
}


def retained(component, text, mentions):
    """Apply the benchmark's source-specific offset policy, label mapping and empty-row filter."""
    if component == "CHIA":
        text = re.sub(r"(?<=[a-zA-Z,.])\n(?=[a-zA-Z,.])", "\n\n", text)
    if not text.strip().lower().replace("na", ""):
        return Counter()
    labels = Counter()
    for label, surface, start, end in mentions:
        mapped = LABELS[component].get(label)
        if mapped is None:
            continue
        if component in {"CHIA", "NCBI"} and surface != text[start:end]:
            if (
                component == "NCBI"
                or sum(1 for _ in re.finditer(re.escape(surface), text)) != 1
            ):
                continue
        # BC5CDR repairs mismatches with a string search and always keeps the mention.
        labels[mapped] += 1
    return labels


def pubtator(stream):
    document, annotations = [], []
    for raw in stream:
        line = raw.rstrip("\r\n")
        if not line:
            if document:
                if len(document) != 2:
                    raise ValueError("Missing PubTator title or abstract")
                yield " ".join(document), annotations
            document, annotations = [], []
        elif re.match(r"^\d+\|[ta]\|", line):
            document.append(line.split("|", 2)[2])
        else:
            fields = line.split("\t")
            if fields[1].isdigit():
                docid, start, end, surface, label, *_ = fields
                annotations.append((label, surface, int(start), int(end)))
    if document:
        if len(document) != 2:
            raise ValueError("Missing PubTator title or abstract")
        yield " ".join(document), annotations


def bioc(stream):
    for doc in ET.parse(stream).iter("document"):
        passages = doc.findall("passage")
        if len(passages) != 2 or passages[0].findtext("infon[@key='type']") != "title":
            raise ValueError("Unexpected BC5CDR passages")
        text = " ".join(p.findtext("text") for p in passages)
        mentions = []
        for passage in passages:
            for entry in passage.findall("annotation"):
                start, length = (
                    int(entry.find("location").attrib[k]) for k in ("offset", "length")
                )
                mentions.append(
                    (
                        entry.findtext("infon[@key='type']"),
                        entry.findtext("text"),
                        start,
                        start + length,
                    )
                )
        yield text, mentions


def chia_splits(path):
    import numpy as np
    import pyarrow.parquet as pq

    rows = pq.read_table(path, columns=["text", "entities"]).to_pylist()
    # datasets.Dataset.train_test_split uses the first ceil(n * test_size) permutation
    # positions for the test partition. The benchmark repeats it on the holdout.
    order = np.random.default_rng(42).permutation(len(rows))
    cut = math.ceil(len(rows) * 0.2)
    holdout = order[:cut][np.random.default_rng(42).permutation(cut)]
    second = math.ceil(cut * 0.5)
    for split, indices in (
        ("train", order[cut:]),
        ("validation", holdout[second:]),
        ("test", holdout[:second]),
    ):
        docs = []
        for index in indices:
            row = rows[index]
            mentions = []
            for entity in row["entities"]:
                (surface,) = entity["text"]
                start, end = entity["offsets"][0]
                mentions.append((entity["type"], surface, start, end))
            docs.append((row["text"], mentions))
        yield split, docs


def split_result(component, split, documents):
    seen = kept = 0
    labels = Counter()
    for text, mentions in documents:
        seen += 1
        mapped = retained(component, text, mentions)
        if mapped:
            kept += 1
            labels.update(mapped)
    if not kept:
        raise ValueError("No eligible benchmark documents")
    distribution = ", ".join(f"{name} {n}" for name, n in sorted(labels.items()))
    return dict(
        split_or_subset=f"{component} {split}",
        unit="documents with a benchmark-mapped mention",
        records=kept,
        reason=f"{sum(labels.values())} retained mentions ({distribution}) from {seen} source documents. "
        f"NCER recipe {REVISION[:12]}.",
    )


def measure(source, path):
    if source == CHIA:
        return [split_result("CHIA", split, docs) for split, docs in chia_splits(path)]
    with zipfile.ZipFile(path) as archive:
        if source in NCBI:
            split = NCBI[source]
            member = f"NCBI{NCBI_SPLITS[split]}set_corpus.txt"
            with archive.open(member) as f:
                return [
                    split_result(
                        "NCBI",
                        split,
                        pubtator(io.TextIOWrapper(f, encoding="utf-8-sig")),
                    )
                ]
        output = []
        labels = (
            ("Train", "Dev", "Test")
            if source == BIORED
            else ("TrainingSet", "DevelopmentSet", "TestSet")
        )
        for split, label in zip(("train", "validation", "test"), labels):
            if source == BIORED:
                with archive.open(f"BioRED/{label}.PubTator") as f:
                    result = split_result(
                        "BioRED",
                        split,
                        pubtator(io.TextIOWrapper(f, encoding="utf-8-sig")),
                    )
            elif source == BC5CDR:
                with archive.open(
                    f"CDR_Data/CDR.Corpus.v010516/CDR_{label}.BioC.xml"
                ) as f:
                    result = split_result("BC5CDR", split, bioc(f))
            else:
                raise ValueError("Unknown NCER benchmark source")
            output.append(result)
        return output
