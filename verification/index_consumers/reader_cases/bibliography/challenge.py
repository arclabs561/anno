"""Replay original dated-bibliography examples through the shared bulk reader.

Run `python3 challenge.py --seed 41 --rounds 32` for deterministic local exploration.
Use `--fault --save NEW_DIR` to export a minimized off-by-one control and its
separate expected answers. Run `--replay DIR` to score private or saved cases.
Seed 41 is a public development control. Promote a new seed only after independent
review of the expectation and its reuse rights. Original synthetic inputs and code:
MIT OR Apache-2.0. No source is fetched.
"""
import argparse
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import random
import sys
from xml.etree import ElementTree as ET

from reader import read_bulk


def digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def source_xml(byline):
    article = ET.Element("article")
    meta = ET.SubElement(ET.SubElement(article, "front"), "article-meta")
    group = ET.SubElement(meta, "contrib-group")
    for parts in byline:
        entry = ET.SubElement(group, "contrib", {"contrib-type": "author"})
        name = entry if parts[0][0] == "collab" else ET.SubElement(entry, "name")
        for field, value in parts:
            ET.SubElement(name, field).text = value
    return ET.tostring(article, encoding="utf-8", xml_declaration=True)


def example(seed):
    """Author the key at each release, before shuffling the reader's edition events."""
    rng = random.Random(seed)
    doi = f"10.0000/shore-almanac-{seed}"
    day = date(2020, 1, 1) + timedelta(days=rng.randrange(365))
    works = [{"doi": doi, "editions": []}]
    editions = works[0]["editions"]
    queries, key = [], []

    def ask(cut, boundary, chosen):
        q = {"doi": doi, "cut": cut}
        if boundary is not None:
            q["date"] = boundary.isoformat()
        queries.append(q)
        head = {**q, "date": q.get("date")}
        if chosen is None:
            key.append({**head, "status": "unavailable", "reason": "no_edition_at_cut"})
        else:
            key.append({**head, "status": "available", **deepcopy(chosen)})

    # The first revision day is a single release. A later day always contains
    # a record and a same-date correction ordered by publisher edition.
    group_sizes = [1, 1, 2] + [rng.choice([1, 2]) for _ in range(rng.randint(1, 4))]
    for size in group_sizes:
        day += timedelta(days=rng.randint(3, 12))
        prior = editions[-1] if editions else None
        for _ in range(size):
            n = len(editions) + 1
            byline = [[["given-names", "Leïla"], ["surname", "Cove"]],
                      [["collab", f"Survey group {n}"]]]
            if n % 2 == 0:
                byline.reverse()
            edition = {"edition": n, "published_on": day.isoformat(),
                       "source_url": f"https://shore.example/{seed}/v{n}.xml",
                       "source_sha256": hashlib.sha256(source_xml(byline)).hexdigest(),
                       "license": "https://spdx.org/licenses/MIT.html",
                       "byline": byline}
            editions.append(edition)
        if prior is None:
            ask("first", None, editions[0])
        ask("before", day, prior)
        ask("as_of", day, editions[-1])
        ask("as_of", day + timedelta(days=1), editions[-1])
    works[0]["through"] = (day + timedelta(days=2)).isoformat()
    rng.shuffle(editions)
    return works, queries, key


def observations(works, queries, fault=False):
    # Model a caller that accidentally uses yesterday's inclusive cut for an
    # as-of question. Preserve the user's question in the returned projection.
    effective = []
    for q in queries:
        if fault and q["cut"] == "as_of":
            yesterday = date.fromisoformat(q["date"]) - timedelta(days=1)
            effective.append({**q, "date": yesterday.isoformat()})
        else:
            effective.append(q)
    rows = read_bulk(works, effective)
    if not isinstance(rows, list) or len(rows) != len(queries):
        raise ValueError("reader returned an incomplete response")
    result = []
    for q, row in zip(queries, rows):
        if not isinstance(row, dict) or row.get("status") not in {"available", "unavailable"}:
            raise ValueError("reader returned an invalid answer envelope")
        result.append({**row, "date": q["date"]} if fault and q["cut"] == "as_of" else row)
    return result


def failures(works, queries, key, fault):
    if not queries or len(key) != len(queries):
        raise ValueError("expected one independent answer per question")
    for q, expected in zip(queries, key):
        if (expected["doi"], expected["cut"], expected["date"]) != (q["doi"].lower(), q["cut"], q.get("date")):
            raise ValueError("key belongs to another question")
        fields = {"doi", "cut", "date", "status"}
        if expected["status"] == "available":
            fields.update(("edition", "published_on", "source_url", "source_sha256", "license", "byline"))
        elif expected["status"] == "unavailable":
            fields.add("reason")
        else:
            raise ValueError("unsupported status in answer key")
        if set(expected) != fields:
            raise ValueError("unexpected or missing fields in answer key")
    rows = observations(works, queries, fault)
    return [(i, row) for i, row in enumerate(rows) if digest(row) != digest(key[i])]


def shrink(generated, mismatches, fault):
    """Keep one original question and the shortest complete prefix reproducing its wrong answer."""
    works, queries, key = generated
    # Prefer an actually substituted credit to a blank result: these have
    # different consequences for somebody publishing a bibliography.
    index, observed = next(((i, row) for i, row in mismatches
                            if key[i]["status"] == row["status"] == "available"), mismatches[0])
    question, expected = queries[index], key[index]
    work, = works
    for length in range(max(1, expected.get("edition", 1)), len(work["editions"]) + 1):
        prefix = [e for e in work["editions"] if e["edition"] <= length]
        candidate = ([{**work, "editions": prefix}], [question], [expected])
        if failures(*candidate, fault) == [(0, observed)]:
            return candidate, observed
    raise ValueError("could not reproduce the measured failure")


FILES = ("works.json", "queries.json", "expected.json")


def save(directory, case):
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in zip(FILES, case):
        (directory / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--rounds", type=int, default=1, help="number of consecutive seeds")
    parser.add_argument("--replay", type=Path, help="local directory with works, queries and expected JSON")
    parser.add_argument("--fault", action="store_true", help="simulate an exclusive as-of boundary")
    parser.add_argument("--save", type=Path, help="write the selected case and separate key to a new directory")
    args = parser.parse_args()
    if args.rounds < 1 or ((args.replay or args.save) and args.rounds != 1):
        parser.error("local replay and export require a single case")
    checked = 0
    for seed in range(args.seed, args.seed + args.rounds):
        if args.replay:
            case = tuple(json.loads((args.replay / name).read_bytes()) for name in FILES)
        else:
            case = example(seed)
        mismatches = failures(*case, args.fault)
        checked += len(case[1])
        if mismatches:
            original = {"editions": sum(len(w["editions"]) for w in case[0]), "queries": len(case[1])}
            if args.replay:
                i, observed = mismatches[0]
                case = case[0], [case[1][i]], [case[2][i]]
            else:
                case, observed = shrink(case, mismatches, args.fault)
            if args.save:
                save(args.save, case)
            print(json.dumps({"outcome": "fail", "fault_control": args.fault,
                              "seed": None if args.replay else seed, "original": original,
                              "retained_editions": sum(len(w["editions"]) for w in case[0]),
                              "input_sha256": digest(case[:2]), "key_sha256": digest(case[2]),
                              "expected_edition": case[2][0].get("edition"),
                              "observed_edition": observed.get("edition")}, sort_keys=True))
            return 1
        if args.save:
            save(args.save, case)
    print(json.dumps({"outcome": "pass", "cases": args.rounds, "questions": checked}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({"outcome": "error", "detail": str(exc)}), file=sys.stderr)
        sys.exit(2)
