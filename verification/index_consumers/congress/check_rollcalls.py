# SPDX-License-Identifier: MIT OR Apache-2.0
"""Reproduce two Clerk publication comparisons. Requires Git, PyYAML and the roster clone.

python3 check_rollcalls.py PATH_TO_CONGRESS_LEGISLATORS [PATH_TO_XML_CACHE]
The optional cache contains roll001-2023.xml and roll400-2023.xml.
Without it, fetch the original files from the Clerk.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import argparse
import json
import re
import subprocess
import unicodedata
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import yaml


OPENING = "238b522772f264f3eb59b1ee83f9308f96248a03"
UPDATE = "c2bd05283e3dbf18dde3134df1710421c959a890"
AUTUMN = "98e8dceb1a11f11efec77b616e6bc344d38e2233"
ROLLS = (
    (1, "2023-01-03", "d5600044c18f39e9d263b336b8bc88210773dac127b5529903041dfd8191b812",
     [("as-of", OPENING), ("first-new-roster", UPDATE)]),
    (400, "2023-09-20", "7c4bf38a7cd038b3aeeed3164f4d3205e09d452740bf8e6594358fcd53fb0408",
     [("as-of", AUTUMN)]),
)


@dataclass(frozen=True)
class Query:
    printed: str
    state: str


def normal(value):
    folded = unicodedata.normalize("NFKD", value.casefold())
    return "".join(c for c in folded if c.isalnum())


def build(roster, date, active, state):
    candidates = defaultdict(set)
    for person in roster:
        terms = person["terms"]
        if active:
            term = next((t for t in reversed(terms) if t["start"] <= date < t["end"]), None)
        else:
            term = terms[-1]
        if term and term["type"] == "rep":
            name = person["name"]
            last = normal(name["last"])
            key = last, term["state"] if state else ""
            candidates[(*key, "")].add(person["id"]["bioguide"])
            # Preserve forms such as "C. Scott" that the source printed explicitly.
            full = normal(name.get("official_full", ""))
            given_names = {normal(name.get(field, "")) for field in ("first", "nickname")}
            if full.endswith(last):
                given_names.add(full[:-len(last)])
            for given in given_names - {""}:
                candidates[(*key, given)].add(person["id"]["bioguide"])
    return candidates


def match(query, candidates, state, preserve_given):
    literal = re.sub(r"\s+\([A-Z]{2}\)$", "", query.printed)
    last, _, given = literal.partition(",")
    key = normal(last), query.state if state else "", normal(given) if preserve_given else ""
    found = candidates.get(key, set())
    if len(found) == 1:
        return next(iter(found)), "unique"
    return None, "ambiguous" if found else "missing"


def load_queries(number, digest, cache):
    roll = f"roll{number:03d}"
    if cache:
        raw = (cache / f"{roll}-2023.xml").read_bytes()
    else:
        with urlopen(f"https://clerk.house.gov/evs/2023/{roll}.xml", timeout=30) as stream:
            raw = stream.read()
    if sha256(raw).hexdigest() != digest:
        raise ValueError(f"{roll} XML differs from measured input")
    root = ET.fromstring(raw)
    rows = root.findall("vote-data/recorded-vote/legislator")
    queries = [Query(r.text, r.attrib["state"]) for r in rows]
    gold = [r.attrib["name-id"] for r in rows]
    return queries, gold


def pilot(repo, cache, output_dir):
    for number, date, digest, editions in ROLLS:
        queries, gold = load_queries(number, digest, cache)
        previous = {}
        for label, rev in editions:
            raw = subprocess.check_output(["git", "-C", repo, "show", rev + ":legislators-current.yaml"])
            roster = yaml.load(raw, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
            for active, state, given in ((False, True, False), (False, True, True),
                                         (True, False, False), (True, True, False), (True, True, True)):
                index = build(roster, date, active, state)
                guesses = [match(q, index, state, given) for q in queries]
                audit = Counter("right" if g == truth else "wrong" if g is not None else why
                                for (g, why), truth in zip(guesses, gold))
                key = (active, state, given)
                print(number, label, rev[:8], "active" if active else "latest-any-term",
                      "surname+state" if state else "surname", "use-given" if given else "surname-only", dict(sorted(audit.items())))
                if key in previous:
                    flips = Counter(("right" if g1 == target else "missing" if g1 is None else "wrong",
                                     "right" if g2 == target else "missing" if g2 is None else "wrong")
                                    for ((g1, _), (g2, _), target) in zip(previous[key], guesses, gold))
                    print("  prior->now:", sorted(flips.items()))
                previous[key] = guesses
                if active and state and given and output_dir:
                    from publication_input import grounded_names

                    number_text = f"{number:03d}"
                    doc_id = f"house-clerk/2023/{number_text}"
                    occurrences = [(q.printed, guess[0]) for q, guess in zip(queries, guesses)]
                    envelope = {
                        "stream": doc_id + "/bioguide",
                        "sequence": 2 if label == "first-new-roster" else 1,
                        "source_revision": "clerk-xml-sha256:" + digest,
                        "document": grounded_names(doc_id, occurrences, catalog="bioguide"),
                    }
                    suffix = "asof" if number == 400 else "before" if label == "as-of" else "after"
                    path = output_dir / f"house-2023-{number_text}-{suffix}.json"
                    path.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
        print("  occurrences", len(queries), "unique IDs", len(set(gold)), "state-qualified text",
              sum(bool(re.search(r"\([A-Z]{2}\)$", q.printed)) for q in queries),
              "non-ASCII", sum(not q.printed.isascii() for q in queries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roster", type=Path)
    parser.add_argument("cache", type=Path, nargs="?")
    parser.add_argument("--output-dir", type=Path, help="emit three anno query publication inputs")
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(exist_ok=True, parents=True)
    pilot(args.roster, args.cache, args.output_dir)
