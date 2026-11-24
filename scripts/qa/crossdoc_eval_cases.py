"""Replay tiny authored cross-document cases against the Rust cdcr_eval executable."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import re
import subprocess
import tempfile
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZipInfo

SEEDS = {"Singletons": "singletons", "Gold within document": "gold_local"}
METRICS = {
    "gold_clusters": "Gold clusters",
    "predicted_clusters": "Pred clusters",
    "correct_links": "Correct links",
    "gold_links": "Gold links",
    "predicted_links": "Pred links",
}
THRESHOLDS = (0.3, 0.5, 0.7)
KINDS = {
    "person": "HUMAN_PART_PER",
    "organization": "HUMAN_PART_ORG",
    "participant": "NON_HUMAN_PART",
    "event": "ACTION_OCCURRENCE",
    "location": "LOC_GEO",
    "time": "TIME_DATE",
}


def _items(value, label, minimum, maximum):
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{label} requires {minimum}–{maximum} entries")
    return value


def _number(value, label, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _string(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 96:
        raise ValueError(f"{label} must contain 1–96 characters")
    if not all(c.isprintable() for c in value):
        raise ValueError(f"{label} must contain printable text")
    return value


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() <= set(required) | set(optional):
        raise ValueError(f"expected fields {sorted(required)}, with optional {sorted(optional)}")


def _doc_key(doc):
    topic = _number(doc["topic"], "topic", 1, 999999)
    name = doc["id"]
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9]{1,32}", name):
        raise ValueError("document id must contain 1–32 ASCII letters or digits")
    return topic, name


def _document_xml(doc):
    root = ET.Element("Document")
    token_count = 0
    for sentence_index, sentence in enumerate(_items(doc["sentences"], "sentences", 1, 8)):
        for position, word in enumerate(_items(sentence, "sentence tokens", 1, 32)):
            token_count += 1
            token = ET.SubElement(root, "token", t_id=str(token_count),
                                  sentence=str(sentence_index), number=str(position))
            token.text = _string(word, "token")

    markables = ET.SubElement(root, "Markables")
    groups = {}
    for mark_id, mention in enumerate(_items(doc["mentions"], "mentions", 0, 24), 1):
        _fields(mention, ["tokens"], ["kind", "local", "cross"])
        kind = mention.get("kind", "person")
        if not isinstance(kind, str) or kind not in KINDS:
            raise ValueError(f"kind must be one of {sorted(KINDS)}")
        mark = ET.SubElement(markables, KINDS[kind], m_id=str(mark_id))
        for token_id in _items(mention["tokens"], "mention token anchors", 1, 32):
            _number(token_id, "anchor", 1, token_count)
            ET.SubElement(mark, "token_anchor", t_id=str(token_id))
        if "cross" in mention and "local" in mention:
            raise ValueError("a mention may belong to one local or cross-document group")
        for scope in ("cross", "local"):
            if scope in mention:
                label = _string(mention[scope], f"{scope} annotation label")
                groups.setdefault((scope, label), []).append(mark_id)

    relations = ET.SubElement(root, "Relations")
    for relation_id, ((scope, label), members) in enumerate(groups.items(), 1):
        tag = "CROSS_DOC_COREF" if scope == "cross" else "INTRA_DOC_COREF"
        relation = ET.SubElement(relations, tag, r_id=str(relation_id))
        if scope == "cross":
            relation.set("note", label)
        for mark_id in members:
            ET.SubElement(relation, "source", m_id=str(mark_id))
    return ET.tostring(root, encoding="utf-8")


def _write_panel(case, directory):
    known = {}
    cache = directory / "datasets"
    cache.mkdir()
    with ZipFile(cache / "ECBPlus.cache", "w") as archive:
        for doc in _items(case.get("documents"), "documents", 2, 8):
            _fields(doc, ["topic", "id", "sentences", "mentions"])
            topic, name = _doc_key(doc)
            if (topic, name) in known:
                raise ValueError("each (topic, document id) must be unique")
            entry = ZipInfo(f"ECB+/{topic}/{topic}_{name}.xml")  # Fixed timestamp for reproducible bytes.
            archive.writestr(entry, _document_xml(doc))
            known[topic, name] = len(doc["sentences"])
    if "selected_sentences" not in case:
        return []

    index_path = directory / "sentences.csv"
    selected = set()
    with index_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["Topic", "File", "Sentence Number"])
        for panel in _items(case["selected_sentences"], "selected_sentences", 1, 8):
            _fields(panel, ["topic", "id", "sentences"])
            key = _doc_key(panel)
            if key not in known or key in selected:
                raise ValueError("each selected document must be unique and present in the case")
            selected.add(key)
            used_indices = set()
            for index in _items(panel["sentences"], "selected sentence indices", 1, known[key]):
                _number(index, "selected sentence", 0, known[key] - 1)
                if index in used_indices:
                    raise ValueError("each selected sentence must appear once")
                used_indices.add(index)
                writer.writerow([*key, index])
    return ["--sentence-index", str(index_path)]


def _expectations(case):
    rows = {}
    for assertion in _items(case.get("assertions"), "assertions", 1, 6):
        _fields(assertion, {"seeds", "threshold"}, METRICS)
        if not METRICS.keys() & assertion.keys():
            raise ValueError("each assertion needs at least one expected counter")
        seeds, threshold = assertion["seeds"], assertion["threshold"]
        if seeds not in SEEDS.values() or threshold not in THRESHOLDS:
            raise ValueError("select singletons or gold_local and threshold 0.3, 0.5, or 0.7")
        key = seeds, threshold
        if key in rows:
            raise ValueError("each seed/threshold assertion must be unique")
        rows[key] = {name: _number(assertion[name], name, 0, 20000)
                     for name in METRICS if name in assertion}
    return rows


def _observations(output):
    rows = {}
    columns = None
    for line in output.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] == "Local seeds":
            columns = cells
        elif cells[0] in SEEDS:
            if columns is None or len(cells) != len(columns):
                raise RuntimeError("cdcr_eval emitted a malformed results table")
            values = dict(zip(columns, cells))
            try:
                key = SEEDS[cells[0]], float(values["Threshold"])
                counts = {name: int(values[column]) for name, column in METRICS.items()}
            except (KeyError, ValueError) as error:
                raise RuntimeError("cdcr_eval emitted missing or invalid result fields") from error
            if key in rows:
                raise RuntimeError("cdcr_eval emitted duplicate seed/threshold results")
            rows[key] = counts
    required_rows = {(seeds, threshold) for seeds in SEEDS.values() for threshold in THRESHOLDS}
    if rows.keys() != required_rows:
        raise RuntimeError("cdcr_eval did not emit its full seed/threshold results table")
    return rows


def run_case(case: dict, binary: Path) -> dict:
    """Run one fixed executable once. XML, selection and expectations are data supplied by the case."""
    if not isinstance(case, dict) or case.get("kind") != "cross_doc_eval":
        raise ValueError("expected a cross_doc_eval case")
    expected = _expectations(case)
    executable = Path(binary).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="anno-qa-cdcr-") as tmp:
        directory = Path(tmp)
        args = _write_panel(case, directory)
        env = os.environ.copy()
        env.update(ANNO_CACHE_DIR=tmp, ANNO_S3_CACHE="0", ANNO_CDCR_DATASETS="ECBPlus")
        env.pop("ANNO_CDCR_MAX_DOCS", None)
        try:
            result = subprocess.run([str(executable), *args], cwd=directory, env=env,
                                    capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"could not execute cdcr_eval: {error}") from error
        if result.returncode:
            raise RuntimeError(f"cdcr_eval exited {result.returncode}: {result.stderr[-1000:]}")
        observed = _observations(result.stdout)
    checks = []
    for key, counts in expected.items():
        for metric, wanted in counts.items():
            actual = observed[key][metric]
            checks.append({"name": f"{key[0]}@{key[1]} {metric}", "expected": wanted,
                           "actual": actual, "passed": actual == wanted})
    return {"checks": checks}
