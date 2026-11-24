#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Synthetic current-view index study. Run with Python 3.10+ and no dependencies.

The public edition prints an ambiguous byline. Only a staff register identifies
its writer. A link is a supplied assertion about an exact source occurrence,
with dependencies on particular document revisions. Heads advance monotonically
per document. Redelivery can change arrival order, and deletions are tombstones.

Queries return literal byline hits, identity-filtered source citations, identity
facet labels/counts, and the total attributed occurrences. There is one candidate
identity, one quote per edition, and two fixed audiences. This oracle exercises
projection semantics; it supplies no authentication, storage engine or live ACL.
"""

from collections import Counter
from dataclasses import dataclass


Ref = tuple[str, int]
QUOTE = "M. Vale"
IDENTITY = "catalog/Mira-Vale"


@dataclass(frozen=True)
class Document:
    ref: Ref
    audience: str
    text: str | None


@dataclass(frozen=True, order=True)
class Citation:
    ref: Ref
    start: int
    end: int


@dataclass(frozen=True)
class Attribution:
    citation: Citation
    identity: str
    evidence: tuple[Ref, ...]


@dataclass(frozen=True)
class View:
    literal_hits: tuple[Citation, ...]
    identity_hits: tuple[Citation, ...]
    facets: tuple[tuple[str, int], ...]
    attributed_count: int


class Index:
    def __init__(self) -> None:
        self.heads: dict[str, Document] = {}
        self.claims: set[Attribution] = set()

    def receive(self, event: Document | Attribution) -> None:
        if isinstance(event, Attribution):
            self.claims.add(event)
            return
        doc_id, revision = event.ref
        head = self.heads.get(doc_id)
        if head is None or revision > head.ref[1]:
            self.heads[doc_id] = event
        elif head.ref == event.ref and head != event:
            raise ValueError("conflicting contents for one source revision")

    def query(self, audience: str, evidence_acl: bool = True, pin_revisions: bool = True) -> View:
        def accessible(doc: Document) -> bool:
            return doc.audience == "public" or audience == "staff"

        def live(ref: Ref) -> Document | None:
            doc = self.heads.get(ref[0])
            if doc is None or doc.text is None or (pin_revisions and doc.ref != ref):
                return None
            return doc

        # The requested full-text collection contains published editions.
        hits = []
        for doc in self.heads.values():
            if doc.ref[0] == "edition" and doc.text and accessible(doc) and QUOTE in doc.text:
                start = doc.text.index(QUOTE)
                hits.append(Citation(doc.ref, start, start + len(QUOTE)))

        authorized = set()
        for claim in self.claims:
            anchor = claim.citation
            source = live(anchor.ref)
            if source is None or not accessible(source):
                continue
            candidate = Citation(source.ref, anchor.start, anchor.end)
            if candidate not in hits:
                continue
            supports = [live(ref) for ref in claim.evidence]
            if all(doc is not None and (not evidence_acl or accessible(doc)) for doc in supports):
                authorized.add((candidate, claim.identity))

        facets = Counter(identity for _, identity in authorized)
        selected = sorted(citation for citation, identity in authorized if identity == IDENTITY)
        return View(tuple(sorted(hits)), tuple(selected), tuple(sorted(facets.items())), len(authorized))


PAGE1 = Document(("edition", 1), "public", "By M. Vale. First printing.")
PAGE2 = Document(("edition", 2), "public", "By M. Vale. Revised printing.")
DELETE_PAGE = Document(("edition", 3), "public", None)
NOTE1 = Document(("register", 1), "staff", "Mira Vale supplied the byline.")
DELETE_NOTE = Document(("register", 2), "staff", None)
NOTE3 = Document(("register", 3), "staff", "The renewed register names Mira Vale.")
GUIDE = Document(("guide", 1), "public", "Index of published editions.")
A1, A2 = Citation(PAGE1.ref, 3, 10), Citation(PAGE2.ref, 3, 10)
LINK1 = Attribution(A1, IDENTITY, (GUIDE.ref, NOTE1.ref))
LINK2 = Attribution(A2, IDENTITY, (GUIDE.ref, NOTE1.ref))
LINK3 = Attribution(A2, IDENTITY, (GUIDE.ref, NOTE3.ref))

# Explicit independent answers. A raw result has no identity facet or count.
EMPTY = View((), (), (), 0)
RAW1, RAW2 = View((A1,), (), (), 0), View((A2,), (), (), 0)
RESOLVED1 = View((A1,), (A1,), ((IDENTITY, 1),), 1)
RESOLVED2 = View((A2,), (A2,), ((IDENTITY, 1),), 1)
CASES = (
    ("public source alone", [PAGE1, GUIDE], RAW1, RAW1),
    ("claim before evidence", [LINK1], RAW1, RAW1),
    ("staff evidence arrives", [NOTE1], RAW1, RESOLVED1),
    ("duplicate deliveries", [PAGE1, NOTE1, LINK1], RAW1, RESOLVED1),
    ("replacement with identical quote and offsets", [PAGE2], RAW2, RAW2),
    ("old page and citation redelivered", [PAGE1, LINK1], RAW2, RAW2),
    ("new citation on the revised page", [LINK2], RAW2, RESOLVED2),
    ("support deleted", [DELETE_NOTE], RAW2, RAW2),
    ("deleted support redelivered", [NOTE1, LINK2], RAW2, RAW2),
    ("support republished under new revision", [NOTE3], RAW2, RAW2),
    ("new evidence explicitly linked", [LINK3], RAW2, RESOLVED2),
    ("source itself deleted", [DELETE_PAGE], EMPTY, EMPTY),
    ("old source and claims redelivered", [PAGE2, LINK1, LINK2, LINK3], EMPTY, EMPTY),
)


def main() -> None:
    index = Index()
    access_misses, revision_misses = [], []
    for name, events, public, staff in CASES:
        for event in events:
            index.receive(event)
        for audience, expected in (("public", public), ("staff", staff)):
            actual = index.query(audience)
            if actual != expected:
                raise AssertionError((name, audience, expected, actual))
            source_only = index.query(audience, evidence_acl=False)
            if source_only != expected:
                if audience != "public" or source_only != staff:
                    raise AssertionError(("source-only leak", name, source_only))
                access_misses.append((name, audience))
            unpinned = index.query(audience, pin_revisions=False)
            if unpinned != expected:
                if audience != "staff" or unpinned != RESOLVED2:
                    raise AssertionError(("wrong revision", name, unpinned))
                revision_misses.append((name, audience))
    expected_leaks = [(CASES[i][0], "public") for i in (2, 3, 6, 10)]
    expected_rebinds = [(CASES[i][0], "staff") for i in (4, 5, 9)]
    if access_misses != expected_leaks or revision_misses != expected_rebinds:
        raise AssertionError(("unexpected baseline behavior", access_misses, revision_misses))
    print(f"Dependency visibility and revision pins: {2 * len(CASES)} of {2 * len(CASES)} views match")
    print(f"Source-only access exposes the identity citation, facet and count in {len(access_misses)} public views")
    print(f"Document-ID-only matching reattaches stale evidence or citations in {len(revision_misses)} staff views")
    print(f"All checks use the same two audiences and hand-labeled results over {len(CASES)} ingestion stages.")


if __name__ == "__main__":
    main()
