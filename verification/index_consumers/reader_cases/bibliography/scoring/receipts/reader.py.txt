#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Read publisher-printed credits from a trusted, rights-reviewed bibliography manifest.

Supply the entire publisher release index through the certified UTC date `through`.
Each edition records its publisher version, day and (when available) an ordered JATS
byline, raw publisher XML digest, source URL and license. A citation to the raw XML
and anno's digest of the separately projected byline text identify different bytes.

    python reader.py works.json queries.json
    python reader.py works.json --ask < query.json
Add --anno PATH to round-trip the available rows through anno query publication.
The application authenticates and authorizes the viewer before serving the results.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from copy import deepcopy
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit


PERSON_FIELDS = ("prefix", "given-names", "surname", "suffix")
EVIDENCE = ("source_url", "source_sha256", "license", "byline")


def doi_key(value):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("expected a nonempty DOI without surrounding whitespace")
    return value.lower()


def utc_day(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError("expected an ISO UTC date (YYYY-MM-DD)")
    return date.fromisoformat(value)


def request(query):
    if not isinstance(query, dict):
        raise ValueError("expected a query object")
    cut, day = query.get("cut"), query.get("date")
    if cut == "first":
        if day is not None:
            raise ValueError("first does not take a date")
    elif cut in ("before", "as_of"):
        utc_day(day)
    else:
        raise ValueError("cut must be first, before or as_of")
    return {"doi": doi_key(query.get("doi")), "cut": cut, "date": day}


def ordered_editions(work):
    # Independent deliveries of the exact same release agree; conflicting bodies
    # or backwards publisher dates mean this index cannot order the releases.
    editions = {}
    if not isinstance(work.get("editions"), list):
        raise ValueError("expected publisher editions")
    for edition in work["editions"]:
        if not isinstance(edition, dict):
            raise ValueError("expected edition metadata")
        version = edition.get("edition")
        if type(version) is not int or not 0 < version <= (1 << 64) - 1:
            raise ValueError("expected a positive publisher version")
        utc_day(edition.get("published_on"))
        if version in editions and editions[version] != edition:
            raise ValueError("conflicting publisher edition")
        editions[version] = edition
    ordered = [editions[n] for n in sorted(editions)]
    if any(a["published_on"] > b["published_on"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("publisher dates contradict version order")
    return ordered


def select(work, query):
    """Return the provable release at the cut, or a reason to abstain."""
    try:
        editions = ordered_editions(work)
        through = work.get("through")
        if through is not None:
            utc_day(through)
            if editions and editions[-1]["published_on"] > through:
                raise ValueError("edition beyond the certified coverage date")
    except ValueError:
        return None, "invalid_chronology"

    if query["cut"] == "first":
        if editions and editions[0]["edition"] == 1:
            return editions[0], None
        return None, "incomplete_history"

    day = query["date"]
    if through is not None and day > through:
        # 'before' on the day immediately following coverage only reads covered days.
        bound = utc_day(day).toordinal() - int(query["cut"] == "before")
        if bound > utc_day(through).toordinal():
            return None, "outside_coverage"

    candidate = None
    for edition in editions:
        released = edition["published_on"]
        if released > day or (released == day and query["cut"] == "before"):
            # A missing successor might fall on either side of the cut. Earlier
            # gaps no longer matter when a later eligible edition was retained.
            previous = candidate["edition"] if candidate else 0
            if edition["edition"] != previous + 1:
                return None, "incomplete_history"
            break
        candidate = edition
    else:
        if through is None:
            return None, "incomplete_history"
    if candidate is None:
        return None, "no_edition_at_cut"
    return candidate, None


def source_url(value):
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except ValueError:
        return False


def valid_byline(authors):
    if not isinstance(authors, list) or not authors:
        return False
    for author in authors:
        if not isinstance(author, list) or not author:
            return False
        fields = []
        for pair in author:
            if (not isinstance(pair, list) or len(pair) != 2
                    or not all(isinstance(v, str) and v.strip() for v in pair)):
                return False
            fields.append(pair[0])
        if fields in (["collab"], ["string-name"]):
            continue
        if fields != [field for field in PERSON_FIELDS if field in fields]:
            return False
    return True


def read_one(work, query):
    """Read one queried work at a publisher-date cut. None denotes an unknown work."""
    result = request(query)
    if work is None or doi_key(work.get("doi")) != result["doi"]:
        return {**result, "status": "unavailable", "reason": "unknown_work"}
    edition, reason = select(work, result)
    if reason:
        return {**result, "status": "unavailable", "reason": reason}
    digest = edition.get("source_sha256")
    if not (source_url(edition.get("source_url")) and source_url(edition.get("license"))
            and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
            and valid_byline(edition.get("byline"))):
        return {**result, "status": "unavailable", "reason": "selected_evidence_unavailable"}
    return {**result, "status": "available", "edition": edition["edition"],
            "published_on": edition["published_on"],
            **{key: deepcopy(edition[key]) for key in EVIDENCE}}


def index_works(works):
    indexed = {}
    for work in works:
        key = doi_key(work.get("doi"))
        if key in indexed:
            raise ValueError(f"duplicate work manifest for {key}")
        indexed[key] = work
    return indexed


def read_bulk(works, queries):
    """Answer in request order, including repeats, under the same frozen input."""
    indexed = index_works(works)
    return [read_one(indexed.get(doi_key(q.get("doi"))), q) for q in queries]


def json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def verify_source(row, retrieved_url, source_bytes):
    """Independently check a retrieved raw publisher artifact against its scoped receipt."""
    if row["status"] != "available" or retrieved_url != row["source_url"]:
        raise ValueError("source locator differs")
    if hashlib.sha256(source_bytes).hexdigest() != row["source_sha256"]:
        raise ValueError("source bytes differ")


def publish_byline(row, anno):
    """Check the available row's projected, edition-pinned author occurrences via anno."""
    if row["status"] != "available":
        raise ValueError("only an available byline can be published")
    text = json_text(row["byline"])
    signals = []
    start = 1  # Opening '[' of the canonical author list.
    for position, author in enumerate(row["byline"]):
        surface = json_text(author)
        end = start + len(surface)
        signals.append({"id": position, "label": "PRINTED_AUTHOR", "confidence": 1.0,
                        "modality": "Symbolic", "negated": False, "surface": surface,
                        "location": {"Text": {"start": start, "end": end}}})
        start = end + 1
    revision = f"v{row['edition']}"
    payload = {"stream": "bibliography.ordered-byline.v1", "sequence": row["edition"],
               "source_revision": revision, "document": {"id": row["doi"], "text": text,
                   "signals": signals, "tracks": {}, "identities": {}}}
    run = subprocess.run([anno, "query", "publication", "-"], input=json_text(payload),
                         text=True, capture_output=True, timeout=90)
    if run.returncode:
        raise ValueError(f"anno query publication failed: {run.stderr.strip()}")
    native = json.loads(run.stdout)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if (native["source_sha256"], native["source_revision"]) != (digest, revision):
        raise ValueError("anno returned a different byline edition")
    postings = sorted(native["rows"], key=lambda posting: posting["signal_id"])
    if len(postings) != len(signals):
        raise ValueError("anno returned a different author count")
    for signal, posting in zip(signals, postings):
        expected = (row["doi"], revision, digest, signal["id"], signal["location"], signal["surface"])
        actual = tuple(posting[key] for key in
                       ("document", "source_revision", "source_sha256", "signal_id", "location", "surface"))
        if actual != expected:
            raise ValueError("anno returned a different printed author occurrence")
    return native


def array(path, key):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get(key)
    if not isinstance(data, list):
        raise ValueError(f"expected an array of {key}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("works", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("queries", nargs="?", type=Path)
    group.add_argument("--ask", action="store_true", help="read one online query from standard input")
    parser.add_argument("--anno", help="validate each available edition with this anno executable")
    args = parser.parse_args()
    works = array(args.works, "works")
    if args.ask:
        query = json.load(sys.stdin)
        key = request(query)["doi"]
        rows = [read_one(index_works(works).get(key), query)]
    else:
        rows = read_bulk(works, array(args.queries, "queries"))
    if args.anno:
        validated = set()
        for row in rows:
            key = (row["doi"], row.get("edition"))
            if row["status"] == "available" and key not in validated:
                publish_byline(row, args.anno)
                validated.add(key)
    print(json_text(rows[0] if args.ask else rows))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        sys.exit(str(error))
