# SPDX-License-Identifier: MIT OR Apache-2.0
"""Return native and proposed hits for one authorized occurrence gallery and query on stdin."""

import argparse
import json
import sys

from .policy import CandidateIndex


LOCATOR = ("source", "revision", "occurrence")


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def fields(obj, required, optional=()):
    if not isinstance(obj, dict) or not set(required) <= obj.keys() <= set(
        required
    ) | set(optional):
        raise ValueError("missing or unexpected fields")


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def read(payload):
    fields(payload, ("variant", "query", "gallery"))
    variant, query, gallery = payload["variant"], payload["query"], payload["gallery"]
    if variant not in ("native", "fold", "expanded") or not isinstance(gallery, list):
        raise ValueError("expected a supported variant and a gallery array")
    fields(query, ("surface", "group"))
    if not nonempty(query["surface"]) or (
        query["group"] is not None and not nonempty(query["group"])
    ):
        raise ValueError("expected a query surface and a string or null group")

    records = {}
    for row in gallery:
        fields(row, (*LOCATOR, "surface", "group"), ("snippet",))
        if not all(nonempty(row[k]) for k in (*LOCATOR, "surface", "group")):
            raise ValueError(
                "gallery identifiers, surface and group require nonempty strings"
            )
        if "snippet" in row and not isinstance(row["snippet"], str):
            raise ValueError("snippet must be text")
        locator = tuple(row[k] for k in LOCATOR)
        if locator in records:
            raise ValueError("duplicate source/revision/occurrence locator")
        records[locator] = row["surface"], row["group"]

    def evidence(locators):
        return [
            {**dict(zip(LOCATOR, key)), "surface": records[key][0]}
            for key in sorted(locators)
        ]

    native = [key for key, (_, group) in records.items() if group == query["group"]]
    decision = CandidateIndex(records).suggest(
        query["surface"], query["group"], variant
    )
    return {
        "native_hits": evidence(native),
        "proposal": {
            "decision": decision.decision,
            "tier": decision.tier,
            "hits": evidence(decision.hits),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        result = read(json.load(sys.stdin, object_pairs_hook=unique_keys))
    except ValueError as exc:
        parser.exit(2, f"invalid request: {exc}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
