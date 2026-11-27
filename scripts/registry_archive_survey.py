"""Counts from versioned publisher archives behind registry landing pages."""

import bz2
import json
import re
import zipfile
from collections import Counter
from itertools import zip_longest
from xml.etree import ElementTree

RELEASES = {
    "CorefUD": (
        "https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/abf56419-4563-4ea7-a51f-f6ffca302ba7/content",
        "9f4e610a1d7fea2e89717b53e4e61567141d852f9740dcb48ae64ac01358c497",
        "CRAC 2025, unconstrained track, gold train/minidev and blind minitest",
    ),
    "InterlingueWikipedia": (
        "https://dumps.wikimedia.org/iewiki/20261101/iewiki-20261101-pages-articles.xml.bz2",
        "0c992b61ee2ab1cef8f992694502e4da7ececb6c8f5c14d6c11e8c2f4fa29898",
        "1 November 2026, current revision, main-namespace nonredirect pages",
    ),
    "LojbanTatoeba": (
        "https://object.pouta.csc.fi/OPUS-Tatoeba/v2026-07-08/moses/en-jbo.txt.zip",
        "70fffec6ba1d7152d7627aa9999b58882ca6298ba1de3a875955f532616e6b2a",
        "8 July 2026 OPUS Tatoeba English–Lojban parallel release",
    ),
    "DutchArchaeology": (
        "https://zenodo.org/api/records/3544544/files/alexbrandsen/dutch-archaeo-NER-dataset-v1.0.zip/content",
        "7e8385771bac8f526226d8a26ce976f2a0f873dc32127bf63d4570ca054c8869",
        "Brandsen's v1.0 complete-bio-gs.bio, all six archaeological categories",
    ),
    "SanskritNERBhagavadGita": (
        "https://www.kaggle.com/api/v1/datasets/download/akashsuklabaidya/ner-dataset-fyp-25?datasetVersionNumber=1",
        "7727a766d86c2ea386080c5467acc539a150fea01070b6a548ca0db4b5b89a66",
        "publisher version 1, Bhagavad_Gita_NER.json",
    ),
    "Quenya": (
        "https://github.com/pfstrack/eldamo/releases/download/v0.8.14/eldamo-0.8.14.zip",
        "e4324c1c89bb10e47206a6b4938eeda88ad8406bd9f33d94f59f3cc6731e9ca4",
        "Eldamo 0.8.14 top-level lexical entries explicitly labelled Quenya (q)",
    ),
}


def corefud(path):
    directories = {"train": "gold", "minidev": "gold", "minitest": "input_blind"}
    total = Counter()
    summary = []
    with zipfile.ZipFile(path) as archive:
        for split, view in directories.items():
            prefix = f"CRAC25-Shared-Task-Data/for_shared_task/unconstrained_track/{view}/{split}/"
            members = [
                n
                for n in archive.namelist()
                if n.startswith(prefix) and n.endswith(".conllu")
            ]
            if len(members) != 22:
                raise ValueError(f"Expected all 22 {split} treebanks")
            counts = Counter()
            for member in members:
                sentences = tokens = 0
                with archive.open(member) as data:
                    for line in data:
                        if line.startswith(b"# sent_id = "):
                            sentences += 1
                        elif line.strip() and not line.startswith(b"#"):
                            fields = line.rstrip(b"\r\n").split(b"\t")
                            if len(fields) != 10:
                                raise ValueError("Expected ten CoNLL-U columns")
                            if fields[0].isdigit():
                                tokens += 1
                            elif not re.fullmatch(rb"\d+[-.]\d+", fields[0]):
                                raise ValueError("Invalid CoNLL-U token ID")
                if not (tokens >= sentences > 0):
                    raise ValueError("Empty or malformed treebank")
                counts.update(sentences=sentences, tokens=tokens)
            total.update(counts)
            summary.append(
                f"{split} {counts['sentences']:,} sentences/{counts['tokens']:,} words"
            )
    return dict(
        unit="CoNLL-U sentences",
        records=total["sentences"],
        tokens=total["tokens"],
        reason=", ".join(summary)
        + ". Integer word IDs only. Blind test contains no reference links. Alternate inputs, LLM views and submissions excluded.",
    )


def wikipedia(path):
    pages = Counter()
    with bz2.open(path) as data:
        for _, page in ElementTree.iterparse(data):
            if page.tag.rsplit("}", 1)[-1] != "page":
                continue
            namespace = page.findtext("{*}ns")
            if namespace is None or page.find("{*}revision") is None:
                raise ValueError("Missing page namespace or revision")
            if namespace != "0":
                pages["other namespace"] += 1
            elif page.find("{*}redirect") is not None:
                pages["redirect"] += 1
            else:
                pages["article"] += 1
            page.clear()
    if not pages["article"]:
        raise ValueError("No main-namespace articles")
    return dict(
        unit="main-namespace nonredirect articles",
        records=pages["article"],
        reason=f"Parsed all {pages.total():,} pages. Excluded {pages['redirect']:,} main-namespace redirects and {pages['other namespace']:,} pages in other namespaces. No NER labels.",
    )


def tatoeba(path):
    with zipfile.ZipFile(path) as archive:
        with archive.open("Tatoeba.en-jbo.en") as english, archive.open(
            "Tatoeba.en-jbo.jbo"
        ) as lojban:
            pairs = 0
            for left, right in zip_longest(english, lojban):
                if (
                    not left
                    or not right
                    or not left.decode().strip()
                    or not right.decode().strip()
                ):
                    raise ValueError("Empty or unpaired sentence")
                pairs += 1
    if not pairs:
        raise ValueError("No aligned sentences")
    return dict(
        unit="aligned sentence pairs",
        records=pairs,
        reason="Both files read in order. Each published alignment counts once, including repeated sentences. No entity labels.",
    )


def archaeology(path):
    records = tokens = tagged = 0
    in_sentence = False
    with zipfile.ZipFile(path) as archive:
        (member,) = [
            n for n in archive.namelist() if n.endswith("/complete-bio-gs.bio")
        ]
        with archive.open(member) as data:
            for line in data:
                fields = line.decode("utf-8").split()
                if not fields:
                    in_sentence = False
                    continue
                if len(fields) != 4 or not re.fullmatch(
                    r"O|[BI]-(ART|CON|LOC|MAT|PER|SPE)", fields[-1]
                ):
                    raise ValueError("Invalid Dutch archaeology BIO row")
                records += not in_sentence
                in_sentence = True
                tokens += 1
                tagged += fields[-1] != "O"
    return dict(
        unit="annotated sentence blocks",
        records=records,
        tokens=tokens,
        tagged=tagged,
        reason="Single combined gold file. PER denotes time periods. Alternate encodings, per-user views and the agreement sample excluded.",
    )


def sanskrit(path):
    with zipfile.ZipFile(path) as archive:
        with archive.open("Bhagavad_Gita_NER.json") as data:
            rows = json.load(data)
    tokens = tagged = 0
    for row in rows:
        words, labels = row["tokens"], row["ner_tags"]
        if (
            not words
            or not all(isinstance(word, str) and word for word in words)
            or len(words) != len(labels)
        ):
            raise ValueError("Unaligned Sanskrit token labels")
        if not all(re.fullmatch(r"O|[BI]-\w+", tag) for tag in labels):
            raise ValueError("Invalid Sanskrit BIO label")
        tokens += len(words)
        tagged += sum(tag != "O" for tag in labels)
    return dict(
        unit="tokenized records",
        records=len(rows),
        tokens=tokens,
        tagged=tagged,
        reason="All records in the single published JSON file. The release identifies neither a separate Yoga Sutras file nor train/test splits.",
    )


def quenya(path):
    with zipfile.ZipFile(path) as archive:
        with archive.open("eldamo-0.8.14/content/data-model/eldamo-data.xml") as data:
            root = ElementTree.parse(data).getroot()
    if root.tag != "word-data" or root.get("version") != "0.8.14":
        raise ValueError("Unexpected Eldamo release")
    entries = root.findall("word[@l='q']")
    if not entries or any(not entry.get("v") for entry in entries):
        raise ValueError("Missing lexical headword")
    return dict(
        unit="Quenya lexical entries",
        records=len(entries),
        reason="Includes words, phrases and names. Nested variant elements, early/middle/neo-Quenya and other languages excluded. This source is a dictionary.",
    )


def describe(ident, path, digest):
    if digest != RELEASES[ident][1]:
        raise ValueError(f"Publisher archive digest changed for {ident}")
    readers = {
        "CorefUD": corefud,
        "InterlingueWikipedia": wikipedia,
        "LojbanTatoeba": tatoeba,
        "DutchArchaeology": archaeology,
        "SanskritNERBhagavadGita": sanskrit,
        "Quenya": quenya,
    }
    return readers[ident](path)
