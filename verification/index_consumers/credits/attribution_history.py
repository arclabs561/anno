# SPDX-License-Identifier: MIT OR Apache-2.0
"""Read an authored public bibliography at a retained membership edition."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import date
import json
from types import MappingProxyType

from check_cases import Publication, apply_delta


def text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected a nonempty string")
    return value


def person_id(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("expected a catalog and accession")
    return text(value[0]), text(value[1])


def day(value):
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("expected a full ISO calendar day")
    return parsed


def membership_source(checkpoint):
    scope = json.loads(checkpoint["stream"])
    if (not isinstance(scope, list) or len(scope) != 4 or scope[0] != "anno.query.memberships.v1"
            or type(scope[2]) not in (int, float) or not 0 <= scope[2] <= 1 or type(scope[3]) is not bool):
        raise ValueError("expected a native membership projection and resolver policy")
    return text(scope[1])


@dataclass(frozen=True)
class Credit:
    work: str
    occurrence: str
    person: tuple
    literal: str
    printed_source: str
    binding_source: str


@dataclass(frozen=True)
class Claim:
    person: tuple
    kind: str
    value: str
    start: date | None
    end: date | None
    source: str

    def at(self, when):
        if (self.start is not None and when < self.start) or (self.end is not None and self.end <= when):
            return "outside"
        if self.start is not None and self.end is not None:
            return "confirmed"
        # Every assertion in this fixture describes a nonempty interval of whole days.
        if when == self.start or (self.end is not None and (self.end - when).days == 1):
            return "confirmed"
        return "undetermined"


@dataclass(frozen=True)
class DatedEvidence:
    confirmed: tuple
    undetermined: tuple


@dataclass(frozen=True, init=False)
class View:
    """One edition's native credits and the dated evidence available to its reader."""

    _credits: tuple
    _claims: tuple

    def __init__(self, pub, evidence):
        membership_source(pub.checkpoint)
        if pub.checkpoint != evidence["checkpoint"] or pub.checkpoint != pub.manifest["checkpoint"]:
            raise ValueError("credit evidence belongs to another native edition")
        if pub.rows.keys() != pub.manifest["credits"].keys():
            raise ValueError("native postings and credit manifest disagree")
        documents = {key[0] for key in pub.rows}
        if documents != evidence["credits"].keys() or len(documents) != len(pub.rows):
            raise ValueError("expected one evidenced track per credit document")

        credits = []
        for key, row in pub.rows.items():
            work, predicate, audience = pub.manifest["credits"][key]
            if (predicate, audience) != ("attributed_to", "public"):
                raise ValueError("this fixture reads public author attributions")
            signal, = row["signals"]
            record = row["catalog_record"]
            citation = evidence["credits"][key[0]]
            if not isinstance(record, dict):
                raise ValueError("expected a qualified native author binding")
            person = person_id((record.get("catalog"), record.get("accession")))
            if person != person_id(citation.get("person")):
                raise ValueError("native catalog binding differs from its cited attribution")
            credits.append(Credit(text(work), text(citation["occurrence"]), person,
                                  text(signal["surface"]), text(citation["printed_source"]),
                                  text(citation["source"])))
        if len({(c.work, c.occurrence) for c in credits}) != len(credits):
            raise ValueError("duplicate occurrence on a work")

        claims = []
        for item in evidence["claims"]:
            if item["kind"] not in ("affiliation", "name"):
                raise ValueError("unknown dated assertion kind")
            start = day(item["start"]) if item["start"] is not None else None
            end = day(item["end"]) if item["end"] is not None else None
            if start is not None and end is not None and end <= start:
                raise ValueError("empty or reversed assertion period")
            claims.append(Claim(person_id(item["person"]), item["kind"], text(item["value"]),
                                start, end, text(item["source"])))
        object.__setattr__(self, "_credits", tuple(credits))
        object.__setattr__(self, "_claims", tuple(claims))

    def printed(self, work):
        return tuple(sorted((c.occurrence, c.literal, c.printed_source)
                            for c in self._credits if c.work == work))

    def authors(self, work):
        return tuple(sorted((c.occurrence, c.person, c.binding_source)
                            for c in self._credits if c.work == work))

    def works(self, person):
        person = person_id(person)
        return tuple(sorted({c.work for c in self._credits if c.person == person}))

    def _dated(self, person, when, kind):
        person, when = person_id(person), day(when)
        results = {"confirmed": [], "undetermined": []}
        for claim in self._claims:
            if claim.kind == kind and claim.person == person:
                status = claim.at(when)
                if status in results:
                    results[status].append((claim.value, claim.source))
        return DatedEvidence(tuple(sorted(results["confirmed"])), tuple(sorted(results["undetermined"])))

    def affiliations(self, person, when):
        return self._dated(person, when, "affiliation")

    def names(self, person, when):
        return self._dated(person, when, "name")


class History:
    """Apply one stream's native deltas and retain answers at each accepted sequence."""

    def __init__(self, source):
        self.source = source
        self._head = None
        self._views = {}
        self._latest_evidence = None
        self.writes = 0

    @property
    def revisions(self):
        return MappingProxyType(self._views)

    def at(self, sequence):
        return self._views[sequence]

    def receive(self, checkpoint, manifest, evidence, *, rows=None, delta=None):
        head = self._head
        if membership_source(checkpoint) != self.source:
            raise ValueError("unexpected source stream")
        if head is not None and checkpoint["stream"] != head.checkpoint["stream"]:
            raise ValueError("membership stream changed")
        if evidence["checkpoint"] != checkpoint or manifest["checkpoint"] != checkpoint:
            raise ValueError("evidence checkpoint differs from the delivery")
        sequence = checkpoint["sequence"]
        if type(sequence) is not int or sequence < 1:
            raise ValueError("expected a positive edition sequence")
        if sequence == 1:
            if rows is None or delta is not None:
                raise ValueError("the initial edition requires only a full snapshot")
        elif rows is not None or delta is None:
            raise ValueError("later editions require only a native delta")

        previous = head.checkpoint["sequence"] if head else 0
        if sequence < previous:
            return "stale"
        if sequence == previous:
            if (checkpoint != head.checkpoint or manifest != head.manifest or delta != head.delta
                    or (rows is not None and rows != head.rows) or evidence != self._latest_evidence):
                raise ValueError("conflicting evidence at the same edition")
            return "duplicate"
        if sequence != previous + 1:
            return "gap"

        # A subscriber receives a complete native posting set only at bootstrap.
        # Build and validate the new view before advancing either retained head.
        delta = deepcopy(delta)
        staged_rows = deepcopy(rows) if head is None else apply_delta(head, checkpoint, delta)
        if head is not None:
            has_edits = bool(delta["removed"] or delta["upserted"])
            if has_edits != (checkpoint["fingerprint"] != head.checkpoint["fingerprint"]):
                raise ValueError("native changes disagree with the target row fingerprint")
        staged_head = Publication(deepcopy(checkpoint), staged_rows, deepcopy(manifest), delta)
        view = View(staged_head, evidence)
        self._latest_evidence = deepcopy(evidence)
        self._views[sequence] = view
        self._head = staged_head
        self.writes += 1
        return "accepted"
