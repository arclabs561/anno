#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay an authored bibliography correction against native anno publications."""
import argparse
from copy import deepcopy
from datetime import date, timedelta
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from attribution_history import DatedEvidence, History, View
from check_cases import build_cli, publish

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reader_cases" / "bibliography"))
import reader as bibliography
from sources.capture import names


FIXTURE = Path(__file__).with_name("bibliography")
STREAM = "fixture-bibliography"
TIDE, SALT = "10.0000/anno-tide", "10.0000/anno-salt"
A, B, COAUTHOR = ("university", "17"), ("archive", "17"), ("archive", "26")


def check_answer(actual, expected, label):
    if actual != expected:
        print(f"{label}: expected {expected!r}, got {actual!r}", file=sys.stderr)
        print(f"attribution-history mismatch: {label}")
        raise AssertionError(f"attribution-history mismatch: {label}")


def articles(cli):
    papers = {}
    for filename in ("tide.xml", "salt.xml"):
        data = (FIXTURE / filename).read_bytes()
        meta = ET.fromstring(data).find("front/article-meta")
        doi = meta.findtext("article-id[@pub-id-type='doi']")
        url = f"https://publisher.example/{filename}"
        release = {"edition": 1, "published_on": meta.find("pub-date").attrib["iso-8601-date"],
                   "byline": names(meta), "source_url": url, "source_sha256": hashlib.sha256(data).hexdigest(),
                   "license": "https://spdx.org/licenses/MIT.html"}
        row, = bibliography.read_bulk([{"doi": doi, "editions": [release]}], [{"doi": doi, "cut": "first"}])
        bibliography.verify_source(row, url, data)
        native = bibliography.publish_byline(row, cli)
        byline = {}
        affiliations = {a.attrib["id"]: a.text for a in meta.findall("aff")}
        authors = meta.findall("contrib-group/contrib[@contrib-type='author']")
        for position, posting in enumerate(sorted(native["rows"], key=lambda r: r["signal_id"])):
            author = authors[position]
            byline_field, = json.loads(posting["surface"])
            if byline_field[0] != "string-name":
                raise ValueError("expected a literal JATS name")
            slot = author.attrib["id"]
            byline[slot] = {"surface": byline_field[1], "source": f"{url}#{slot}",
                            "listed": tuple(affiliations[ref] for x in author.findall("xref[@ref-type='aff']")
                                            for ref in x.attrib["rid"].split())}
        papers[doi] = {"byline": byline, "day": row["published_on"]}
    return papers


def deliveries(cli, papers, catalog):
    states, manifests = [], []
    for state in catalog["editions"]:
        credits, sources = [], {}
        for key in state["decisions"]:
            decision = catalog["decisions"][key]
            work, occurrence = decision["work"], decision["occurrence"]
            credit = papers[work]["byline"][occurrence]
            namespace, accession = decision["person"]
            doc = f"{work}#{occurrence}"
            credits.append({"document": doc, "revision": "v1", "work": work, "catalog": namespace,
                            "accession": accession, "surface": credit["surface"],
                            "predicate": "attributed_to", "audience": "public"})
            sources[doc] = {"occurrence": occurrence, "person": decision["person"], "source": f"catalog.json#/decisions/{key}",
                            "printed_source": credit["source"]}
        claims = [dict(catalog["claims"][key], source=f"catalog.json#/claims/{key}") for key in state["claims"]]
        states.append({"credits": credits})
        manifests.append({"credits": sources, "claims": claims})
    pubs = publish(cli, {"sources": {STREAM: states}})
    return [(pubs[STREAM, seq], dict(manifest, checkpoint=pubs[STREAM, seq].checkpoint))
            for seq, manifest in enumerate(manifests, 1)]


def supported(value, record):
    return value, f"catalog.json#/claims/{record}"


HARBOR = supported("Harbor Institute", "harbor-term")
QUAY = supported("Quay Archive", "quay-term")
REED = supported("Reed Observatory", "reed-term")
MARSH = supported("Marsh Library", "undated")


def check_view(view, cut, papers):
    expected_author = B if cut == 1 else A
    expected_decision = "tide-import" if cut == 1 else "tide-correction"
    check_answer(view.authors(TIDE), (("by1", expected_author, f"catalog.json#/decisions/{expected_decision}"),
                                    ("by2", COAUTHOR, "catalog.json#/decisions/tide-coauthor")),
                 "corrected paper attribution")
    check_answer(view.authors(SALT), (("by1", B, "catalog.json#/decisions/salt-import"),), "other namesake paper")
    check_answer(view.printed(TIDE), (("by1", "Rowan Mire", "https://publisher.example/tide.xml#by1"),
                                    ("by2", "Iren Sen", "https://publisher.example/tide.xml#by2")),
                 "unchanged printed tide credit")
    check_answer(view.printed(SALT), (("by1", "Rowan Mire", "https://publisher.example/salt.xml#by1"),),
                 "unchanged printed salt credit")
    check_answer(view.works(COAUTHOR), (TIDE,), "coauthor bibliography")
    check_answer(view.works(A), () if cut == 1 else (TIDE,), "corrected person's bibliography")
    check_answer(view.works(B), (SALT, TIDE) if cut == 1 else (SALT,), "namesake's bibliography")

    # Use the author returned by the query and the date read from the original article.
    _, attributed, _ = view.authors(TIDE)[0]
    expected_affiliations = {1: (QUAY,), 2: (HARBOR,), 3: (HARBOR, REED)}
    at_publication = view.affiliations(attributed, papers[TIDE]["day"])
    check_answer(at_publication.undetermined, (MARSH,) if cut == 3 else (), "undated affiliation evidence")
    check_answer(at_publication.confirmed, expected_affiliations[cut],
                 "affiliations on publication day at reader edition")
    check_answer(view.affiliations(B, papers[SALT]["day"]).confirmed, (QUAY,), "unaffected namesake affiliation")
    check_answer(view.affiliations(A, "2018-05-31").confirmed, (HARBOR,), "before concurrent term")
    check_answer(view.affiliations(A, "2018-09-01").confirmed, (HARBOR,), "end of concurrent term")
    check_answer(view.names(A, papers[TIDE]["day"]).confirmed, (supported("Rowan Mire", "early-name"),), "name at publication")
    later = (supported("Rowan Vale", "later-name"),) if cut == 3 else ()
    check_answer(view.names(A, "2021-01-01").confirmed, later, "later name at reader edition")
    check_answer(view.names(B, "2021-01-01").confirmed, (supported("Rowan Mire", "archive-name"),), "namesake did not rename")


def deliver(index, pub, evidence):
    # After bootstrap the online reader receives only the native delta.
    payload = {"rows": pub.rows} if pub.checkpoint["sequence"] == 1 else {"delta": pub.delta}
    return index.receive(pub.checkpoint, pub.manifest, evidence, **payload)


def reject_delivery(action, label):
    try:
        action()
    except ValueError:
        return
    check_answer("accepted", "rejected", label)


def dated_evidence_answers(pub, evidence, reader):
    # A missing boundary can lie beyond any query. Reserve a completion day on either side.
    days = [(date(2020, 2, 26) + timedelta(days=i)).isoformat() for i in range(8)]
    bounds = [None, *days[2:-2]]
    completions = list(combinations(days, 2))
    queries = days[1:-1]
    claims = []
    expected = {}
    for kind, value in (("name", "Rowan Mire"), ("affiliation", "Reed Observatory")):
        for when in queries:
            expected[kind, when] = {"confirmed": [], "undetermined": []}
        for start, end in product(bounds, repeat=2):
            compatible = [(a, b) for a, b in completions
                          if (start is None or a == start) and (end is None or b == end)]
            if not compatible:  # No nonempty term has these bounds.
                continue
            # Equal display values from different sources remain independently citable.
            source = f"fixture:{kind}/{start or '?'}/{end or '?'}"
            claims.append(dict(person=list(A), kind=kind, value=value, source=source, start=start, end=end))
            for when in queries:
                occupied = [a <= when < b for a, b in compatible]
                if any(occupied):
                    status = "confirmed" if all(occupied) else "undetermined"
                    expected[kind, when][status].append((value, source))

    view = reader(pub, dict(evidence, claims=claims))
    for kind, lookup in (("name", view.names), ("affiliation", view.affiliations)):
        for when in queries:
            result = lookup(A, when)
            for status in ("confirmed", "undetermined"):
                yield (kind, status, when), getattr(result, status), tuple(sorted(expected[kind, when][status]))
            namesake = lookup(B, when)
            yield (kind, "other person", when), (namesake.confirmed, namesake.undetermined), ((), ())


def check_dated_evidence(pub, evidence):
    for (kind, status, when), actual, expected in dated_evidence_answers(pub, evidence, View):
        check_answer(actual, expected, f"{kind} {status} on {when}")

    class ConfirmedNamesOnly(View):
        def names(self, person, when):
            return DatedEvidence(super().names(person, when).confirmed, ())

    failures = {(kind, status) for (kind, status, _), actual, expected
                in dated_evidence_answers(pub, evidence, ConfirmedNamesOnly) if actual != expected}
    check_answer(failures, {("name", "undetermined")}, "confirmed-only name reader detected")


def rejected_corrections(editions):
    # All invalid deliveries use the same valid predecessor. A failure must be retryable.
    index = History(STREAM)
    deliver(index, *editions[0])
    pub, evidence = editions[1]
    reject_delivery(lambda: index.receive(pub.checkpoint, pub.manifest, evidence, rows=pub.rows),
                    "full replacement after bootstrap")
    empty_patch = dict(pub.delta, removed=[], upserted=[])
    reject_delivery(lambda: index.receive(pub.checkpoint, pub.manifest, evidence, delta=empty_patch),
                    "truncated correction")
    bad_credit = deepcopy(evidence)
    bad_credit["credits"][f"{TIDE}#by1"]["person"] = list(B)
    reject_delivery(lambda: deliver(index, pub, bad_credit), "correction evidence contradicts native binding")
    foreign = deepcopy(pub.checkpoint)
    scope = json.loads(foreign["stream"])
    scope[2] = 0.75  # same application source but a different resolver threshold
    foreign["stream"] = json.dumps(scope, separators=(",", ":"))
    reject_delivery(lambda: index.receive(foreign, dict(pub.manifest, checkpoint=foreign),
                                          dict(evidence, checkpoint=foreign), delta=dict(pub.delta, target=foreign)),
                    "cross-policy delta")
    check_answer(index.writes, 1, "rejected correction has no writes")
    check_answer(deliver(index, pub, evidence), "accepted", "retry after rejected correction")
    reject_delivery(lambda: deliver(index, pub, bad_credit), "conflicting redelivery")
    check_answer(index.writes, 2, "conflicting redelivery has no writes")


def replay(cli):
    catalog = json.loads((FIXTURE / "catalog.json").read_text())
    papers = articles(cli)
    check_answer(papers[TIDE]["byline"]["by1"]["listed"], ("Harbor Institute",), "publisher affiliation on tide")
    check_answer(papers[SALT]["byline"]["by1"]["listed"], ("Quay Archive",), "publisher affiliation on salt")
    editions = deliveries(cli, papers, catalog)
    check_dated_evidence(*editions[0])
    rejected_corrections(editions)
    correction = editions[1][0].delta
    check_answer([key[0] for key, _ in correction["removed"]], [f"{TIDE}#by1"], "retracted occurrence")
    check_answer([key[0] for key, _ in correction["upserted"]], [f"{TIDE}#by1"], "reassigned occurrence")
    update = editions[2][0].delta
    check_answer((update["removed"], update["upserted"]), ([], []), "authority-only edition has no native row changes")
    index = History(STREAM)
    for sequence, (pub, evidence) in enumerate(editions, 1):
        check_answer(deliver(index, pub, evidence), "accepted", "accept new edition")
        retained = index.at(sequence)
        check_answer(deliver(index, pub, deepcopy(evidence)), "duplicate", "duplicate delivery")
        check_answer(index.writes, sequence, "zero additional history writes on redelivery")
        check_answer(index.at(sequence) is retained, True, "same retained view on redelivery")
        for cut in range(1, sequence + 1):
            old_pub, old_evidence = editions[cut - 1]
            check_view(View(old_pub, old_evidence), cut, papers)  # independent complete native snapshot
            check_view(index.at(cut), cut, papers)              # receiver's retained delta-applied state
    check_answer(deliver(index, *editions[0]), "stale", "ignore superseded delivery")
    check_answer(index.writes, len(editions), "superseded delivery has no writes")
    check_view(index.at(3), 3, papers)
    print("attribution-history ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno", type=Path, help="use an existing anno executable")
    arguments = parser.parse_args()
    replay((arguments.anno or build_cli()).resolve())
