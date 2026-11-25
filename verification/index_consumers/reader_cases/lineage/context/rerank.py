# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare two fixed orders of the same native affiliation shortlist."""

import argparse
import ast
from collections import Counter
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from evaluate import Record, key
from ngram_retrieval import Query, compiled_index, decision, lookup
from s2aff_names import ARCHIVE, ROR_URL, SOURCE, get

ROR_HASH = "d71fd8d20d753fed8f7b5e08aa2215ad4cda2b2378c7f2a9f977ad17605504a6"
LABEL_HASH = "2385f28c6c1275f0114cf15c33d164672af7a2a6a88d0c03178e04236c3de8c3"
PROTOCOL_HASH = "a6e0b906d2761db9fcc568312b4e3b8cdf2fc5d591be9f0d99a8e3989bd03eec"
CHOICES_HASH = "30fb3686a45bc076cebda41161bbbed5568a03592edd1bdbd5f7f06c115c60e1"
ARMS = ("native", "location")
ACTIONS = ("correct", "wrong", "deferred", "unjudged_link")
BUDGETS = (1, 3, 5)


def tokens(value):
    return tuple(re.findall(r"[^\W_]+", key(value)))


def place(value):
    parts = tokens(value or "")
    return parts if sum(c.isalpha() for part in parts for c in part) >= 3 else ()


def matches(parts, phrase):
    return bool(phrase) and any(
        parts[i : i + len(phrase)] == phrase for i in range(len(parts) - len(phrase) + 1)
    )


@dataclass(frozen=True)
class Location:
    cities: tuple
    country: tuple


def rerank(name, ranked, locations):
    parts = tokens(name)
    scored = []
    for accession, n, d in ranked:
        loc = locations[accession]
        city = any(matches(parts, phrase) for phrase in loc.cities)
        country = matches(parts, loc.country)
        adjusted = min(Fraction(1), Fraction(n, d) + Fraction(2 * city + country, 25))
        scored.append((accession, adjusted.numerator, adjusted.denominator))
    # Python's stable sort preserves native position on exact adjusted-score ties.
    return sorted(scored, key=lambda item: Fraction(item[1], item[2]), reverse=True)


def read_catalog(path):
    with ZipFile(path) as archive, archive.open(ARCHIVE.replace(".zip", ".json")) as f:
        records = json.load(f)
    catalog = {}
    locations = {}
    for row in records:
        if row["status"] != "active":
            continue
        accession = row["id"].removeprefix("https://ror.org/")
        names = {
            row["name"], *row["aliases"], *row["acronyms"], *(v["label"] for v in row["labels"])
        }
        catalog[accession] = Record(row["name"], tuple(sorted(names)), "active", ())
        cities = {place(a["city"]) for a in row["addresses"]}
        locations[accession] = Location(
            tuple(sorted(cities - {()})), place(row["country"]["country_name"])
        )
    return catalog, locations


@dataclass(frozen=True)
class Label:
    split: str
    ids: frozenset


def read_queries(path):
    queries, labels = [], {}
    with path.open(encoding="utf-8", newline="") as f:
        for number, row in enumerate(csv.DictReader(f), 1):
            if row["split"] not in ("val", "test"):
                continue
            accession = str(number)
            ids = frozenset(
                s.removeprefix("https://ror.org/") for s in ast.literal_eval(row["labels"])
                if s.startswith("https://ror.org/")
            )
            queries.append(Query("", accession, row["original_affiliation"], None))
            labels[accession] = Label(row["split"], ids)
    return queries, labels


def panel(available_target=True):
    return {
        "queries": 0,
        "order_changed": 0,
        "shortlist_sizes": Counter(),
        "budget": {
            str(k): {
                "shown_each": 0,
                "gains": 0 if available_target else None,
                "losses": 0 if available_target else None,
                "hits": dict.fromkeys(ARMS, 0) if available_target else None,
                "misses": dict.fromkeys(ARMS, 0) if available_target else None,
            }
            for k in BUDGETS
        },
        "gate": {arm: dict.fromkeys(ACTIONS, 0) for arm in ARMS},
        "gate_pairs": Counter(),
    }


def record(panel, rankings, ids):
    lists = [[a for a, _, _ in rank] for rank in rankings]
    native_ids, location_ids = lists
    if (len(native_ids) > 5 or len(native_ids) != len(set(native_ids))
            or sorted(native_ids) != sorted(location_ids)):
        raise ValueError("reranking changed the native options or duplicated an accession")
    panel["order_changed"] += lists[0] != lists[1]
    panel["queries"] += 1
    panel["shortlist_sizes"][str(len(lists[0]))] += 1
    for k in BUDGETS:
        b = panel["budget"][str(k)]
        b["shown_each"] += min(len(lists[0]), k)
        if b["hits"] is not None:  # This panel has an available annotated target.
            native, location = [not ids.isdisjoint(options[:k]) for options in lists]
            b["gains"] += location and not native
            b["losses"] += native and not location
            for arm, hit in zip(ARMS, (native, location)):
                b["hits"][arm] += hit
                b["misses"][arm] += not hit
    states = []
    for arm, rank in zip(ARMS, rankings):
        prediction, _ = decision(rank)
        if prediction is None:
            state = "deferred"
        elif not ids:
            state = "unjudged_link"
        else:
            state = "correct" if prediction in ids else "wrong"
        panel["gate"][arm][state] += 1
        states.append(state)
    panel["gate_pairs"]["/".join(states)] += 1


def run(cache, fetch, binary, output):
    freezes = (("protocol.txt", PROTOCOL_HASH), ("implementation.txt", CHOICES_HASH))
    for filename, expected in freezes:
        if hashlib.sha256((HERE / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{filename} differs from its frozen hash")
    archive = get(cache, ARCHIVE, ROR_URL, ROR_HASH, fetch)
    label_path = get(cache, "s2aff-gold.csv", SOURCE, LABEL_HASH, fetch)
    catalog, locations = read_catalog(archive)
    queries, labels = read_queries(label_path)
    test_ids = {i for label in labels.values() if label.split == "test" for i in label.ids}
    available = {n: label.ids.intersection(catalog) for n, label in labels.items()}
    masks = {n: label.ids for n, label in labels.items() if available[n]}
    withheld = [q for q in queries if available[q.source]]
    panels = {
        "natural": {
            s: {c: panel(c == "available") for c in ("available", "unavailable", "no_ror_label")}
            for s in ("val", "test")
        },
        "withheld": {s: panel(available_target=False) for s in ("val", "test")},
        "val_entity_disjoint": panel(),
    }
    metadata = {s: Counter() for s in ("val", "test")}
    digest = hashlib.sha256()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "context-ledger.jsonl").open("wb") as ledger:
        conditions = (("natural", queries, None), ("withheld", withheld, masks))
        for condition, requests, excluded in conditions:
            retrieved = lookup(binary, catalog, requests, catalog_masks=excluded)
            for q, _, _, nblocked, native, block in retrieved:
                label = labels[q.source]
                ids = label.ids
                known = bool(available[q.source]) and condition == "natural"
                if excluded is not None and not ids.isdisjoint(block):
                    raise ValueError("a withheld ID survived the native candidate block")
                geo = rerank(q.name, native, locations)
                ranks = (native, geo)
                if condition == "natural":
                    cohort = "available" if known else ("unavailable" if ids else "no_ror_label")
                    record(panels["natural"][label.split][cohort], ranks, ids)
                    if label.split == "val" and known and ids.isdisjoint(test_ids):
                        record(panels["val_entity_disjoint"], ranks, ids)
                    m = metadata[label.split]
                    m["annotated_id_occurrences"] += len(ids)
                    m["unavailable_id_occurrences"] += len(ids - available[q.source])
                    m["multi_id_rows"] += len(ids) > 1
                    m["partial_availability_rows"] += bool(available[q.source]) and bool(
                        ids - available[q.source]
                    )
                    m["known_target_in_block"] += not ids.isdisjoint(block)
                else:
                    record(panels["withheld"][label.split], ranks, ids)
                row = dict(
                    row=int(q.source), split=label.split, condition=condition, gold=sorted(ids),
                    available_before_mask=sorted(available[q.source]), block_size=nblocked,
                    target_in_block=not ids.isdisjoint(block), native=native, location=geo,
                )
                line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                entry = line.encode("ascii")
                ledger.write(entry)
                digest.update(entry)
    for split in metadata:
        split_ids = {i for label in labels.values() if label.split == split for i in label.ids}
        metadata[split]["distinct_annotated_ids"] = len(split_ids)
        metadata[split]["distinct_unavailable_ids"] = len(split_ids.difference(catalog))
    summary = dict(
        panels=panels, metadata=metadata, active_catalog=len(catalog),
        ledger_sha256=digest.hexdigest(), ror_sha256=ROR_HASH, labels_sha256=LABEL_HASH,
        protocol_sha256=PROTOCOL_HASH, implementation_sha256=CHOICES_HASH,
    )
    (output / "context-results.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("output", type=Path, help="directory for the summary and text-free ledger")
    p.add_argument("--fetch", action="store_true")
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        run(args.cache, args.fetch, compiled_index(tmp), args.output)
