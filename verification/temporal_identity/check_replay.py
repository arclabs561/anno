#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay a synthetic one-subject alias catalog at two independent date cutoffs.

Run with Python 3.10+ and no dependencies. Dates are integer years, and valid
intervals include the start and exclude the end. Arrival is the catalog's
locally recorded year, fixed on each immutable receipt before storage reorders
it. Revision numbers are source-assigned within one assertion lineage.

Each new revision wholly replaces the earlier claim, including on historical
dates. A withdrawal retracts that source's assertion. A different real-world
period uses a separate claim. Conflicting bodies under a single revision ID
produce an integrity error for this subject's snapshot. The original mention
is read by source location independently of catalog assertions.
"""

from dataclasses import dataclass
from itertools import permutations


@dataclass(frozen=True)
class Claim:
    lineage: str
    revision: int
    alias: str | None
    start: int | None = None
    end: int | None = None

    def applies(self, valid_year: int) -> bool:
        return (
            self.alias is not None
            and self.start is not None
            and self.start <= valid_year
            and (self.end is None or valid_year < self.end)
        )


@dataclass(frozen=True)
class Receipt:
    arrival: int
    claim: Claim


class RevisionConflict(ValueError):
    """The same source revision identifier arrived with two different bodies."""


def catalog_at(receipts: tuple[Receipt, ...], recorded_through: int) -> tuple[Claim, ...]:
    by_revision: dict[tuple[str, int], Claim] = {}
    for receipt in receipts:
        if receipt.arrival > recorded_through:
            continue
        claim = receipt.claim
        key = (claim.lineage, claim.revision)
        if by_revision.setdefault(key, claim) != claim:
            raise RevisionConflict(key)
    current: dict[str, Claim] = {}
    for (lineage, revision), claim in by_revision.items():
        if lineage not in current or revision > current[lineage].revision:
            current[lineage] = claim
    return tuple(current.values())


def latest_only_at(receipts: tuple[Receipt, ...], recorded_through: int) -> tuple[Claim, ...]:
    """One most recently received row for the subject, with its valid interval."""
    eligible = [receipt for receipt in receipts if receipt.arrival <= recorded_through]
    return (max(eligible, key=lambda receipt: receipt.arrival).claim,) if eligible else ()


def aliases(claims: tuple[Claim, ...], valid_year: int) -> frozenset[str]:
    return frozenset(c.alias for c in claims if c.applies(valid_year) and c.alias is not None)


TEXT = "By M. Lea."
MENTION = ("edition-1", 3, 9)


def read_mention(mention: tuple[str, int, int]) -> str:
    edition, start, end = mention
    return {"edition-1": TEXT}[edition][start:end]


OLD = "Mira Lea"
CORRECTED = "Mira Léa"
NEW = "Mira Vale"
PEN = "North Quay"
PEN_CLAIM = Claim("pen", 1, PEN, 1995, 2005)
RECEIPTS = (
    Receipt(2010, Claim("old", 1, OLD, 1980, 2000)),
    Receipt(2011, Claim("new", 1, NEW, 2000)),
    Receipt(2014, PEN_CLAIM),
    Receipt(2018, Claim("old", 2, CORRECTED, 1980, 2000)),
    Receipt(2020, Claim("pen", 2, None)),
    Receipt(2021, PEN_CLAIM),  # stale import of the identical source revision
)
CONFLICT = Receipt(2022, Claim("old", 2, "Mira Lía", 1980, 2000))

# Independent hand-labeled query answers. Columns bracket interval boundaries
# and overlap; rows bracket the arrival of each claim or correction.
VALID_YEARS = (1979, 1980, 1997, 2000, 2005)
GOLD = {
    2009: [(), (), (), (), ()],
    2010: [(), (OLD,), (OLD,), (), ()],
    2011: [(), (OLD,), (OLD,), (NEW,), (NEW,)],
    2014: [(), (OLD,), (OLD, PEN), (NEW, PEN), (NEW,)],
    2018: [(), (CORRECTED,), (CORRECTED, PEN), (NEW, PEN), (NEW,)],
    2020: [(), (CORRECTED,), (CORRECTED,), (NEW,), (NEW,)],
    2021: [(), (CORRECTED,), (CORRECTED,), (NEW,), (NEW,)],
}


def main() -> None:
    replay_hits = baseline_hits = orderings = withdrawals = 0
    baseline_misses: set[tuple[int, int]] = set()
    for storage_order in permutations(RECEIPTS):
        orderings += 1
        for as_of, expected_row in GOLD.items():
            state = catalog_at(storage_order, as_of)
            single = latest_only_at(storage_order, as_of)
            # An empty list of current aliases cannot by itself distinguish
            # "this source never made a claim" from "the source withdrew it".
            pen_records = [c for c in state if c.lineage == "pen"]
            if as_of < 2014 and pen_records:
                raise AssertionError("a pen-name claim appears before its arrival")
            if as_of >= 2020:
                if pen_records != [Claim("pen", 2, None)]:
                    raise AssertionError("withdrawal erased or superseded by replay")
                withdrawals += 1
            for valid_year, expected in zip(VALID_YEARS, expected_row, strict=True):
                actual = aliases(state, valid_year)
                if actual != frozenset(expected):
                    raise AssertionError((storage_order, as_of, valid_year, actual, expected))
                replay_hits += 1
                if aliases(single, valid_year) == frozenset(expected):
                    baseline_hits += 1
                else:
                    baseline_misses.add((as_of, valid_year))
                if read_mention(MENTION) != "M. Lea":
                    raise AssertionError("source mention changed during catalog replay")

    rejected = unaffected = 0
    for storage_order in permutations((*RECEIPTS, CONFLICT)):
        # A conflicting body ingested in 2022 cannot change a snapshot recorded in 2021.
        if aliases(catalog_at(storage_order, 2021), 1997) != {CORRECTED}:
            raise AssertionError("future conflicting data leaked into old snapshot")
        unaffected += 1
        try:
            catalog_at(storage_order, 2022)
        except RevisionConflict as error:
            if error.args != (("old", 2),):
                raise
            rejected += 1
        else:
            raise AssertionError("accepted conflicting source revision")

    date_pairs = len(GOLD) * len(VALID_YEARS)
    comparisons = date_pairs * orderings
    print(f"{date_pairs} fixed date pairs × {orderings} storage orders")
    print(f"Two-time replay {replay_hits}/{comparisons}; latest-only {baseline_hits}/{comparisons}")
    print(f"Latest-only missed {len(baseline_misses)} distinct date pairs")
    print(f"Original grounded byline preserved in {replay_hits} query checks")
    print(f"Withdrawal remains inspectable in {withdrawals} catalog snapshots")
    print(f"Conflicting revision rejected in all {rejected} seven-receipt orders")
    print(f"Earlier snapshot unaffected in all {unaffected} seven-receipt orders")


if __name__ == "__main__":
    main()
