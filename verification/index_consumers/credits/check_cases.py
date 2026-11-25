#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay synthetic credit cases against anno. Pass --anno to use an existing executable."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "memberships"))
from replay import capture, document, edition

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "congress"))
from check_catalog_history import build_cli


class ViewMismatch(AssertionError):
    pass


@dataclass
class Publication:
    checkpoint: dict
    rows: dict
    manifest: dict
    delta: dict | None


def native_rows(items):
    rows = {tuple(key): value for key, value in items}
    if len(rows) != len(items):
        raise ValueError("ambiguous native row key")
    return rows


def publish(cli, scenario):
    publications = {}
    for source, states in scenario["sources"].items():
        previous = None
        for sequence, state in enumerate(states, 1):
            inputs, credits = [], {}
            for c in state["credits"]:
                doc_id = c["document"]
                if doc_id in credits or not c["surface"].strip():
                    raise ValueError("each fixture row needs a unique document and a literal surface")
                if (c["predicate"] not in (None, "printed_credit", "attributed_to")
                        or c["audience"] not in ("public", "restricted")):
                    raise ValueError("invalid credit predicate or audience")
                item = document(doc_id, c["surface"], c["revision"], (c["catalog"], c["accession"]))
                doc = item["document"]
                label = c.get("label", "MENTION")
                doc["signals"][0]["label"] = label
                doc["tracks"]["0"]["entity_type"] = label
                doc["identities"]["0"]["entity_type"] = label
                doc["tracks"]["0"]["canonical_surface"] = c.get("canonical_surface", c["surface"])
                inputs.append(item)
                credits[doc_id] = (c["work"], c["predicate"], c["audience"])
            current = edition(sequence, inputs)
            current["stream"] = source
            snapshot = capture(cli, current)
            checkpoint = snapshot["checkpoint"]
            scope = ["anno.query.memberships.v1", source, current["threshold"], current["require_type_match"]]
            if checkpoint["sequence"] != sequence or json.loads(checkpoint["stream"]) != scope:
                raise ValueError("unexpected native stream or sequence")
            rows = native_rows(snapshot["rows"])
            if {k[0] for k in rows} != set(credits) or len(rows) != len(credits):
                raise ValueError("native rows disagree with submitted source documents")
            # Predicate, work and audience are edition-specific facts supplied by this fixture.
            manifest = {"checkpoint": checkpoint, "credits": {key: credits[key[0]] for key in rows}}
            delta = capture(cli, current, previous) if previous is not None else None
            if state.get("unchanged_rows") and (delta is None or delta["removed"] or delta["upserted"]):
                raise AssertionError(f"{source} edition {sequence}: policy changed native membership rows")
            publications[source, sequence] = Publication(checkpoint, rows, manifest, delta)
            previous = current
    return publications


class Reader:
    """A single-process fixture reader. Only complete validated source transitions are installed."""
    def __init__(self, *, stale_cache=False):
        self.heads = {}
        self.cache = {}
        self.stale_cache = stale_cache

    def receive(self, source, pub):
        if pub.manifest["checkpoint"] != pub.checkpoint:
            raise ValueError("credit manifest belongs to another native edition")
        old = self.heads.get(source)
        sequence = pub.checkpoint["sequence"]
        head = old.checkpoint["sequence"] if old else 0
        if sequence < head:
            return "stale"
        if sequence == head:
            if pub != old:
                raise ValueError("conflicting edition")
            return "duplicate"
        if sequence != head + 1:
            return "gap"
        if old is None:
            self.heads[source] = pub
        else:
            delta = pub.delta
            if delta is None or delta["base"] != old.checkpoint or delta["target"] != pub.checkpoint:
                raise ValueError("mismatched native delta endpoints")
            rows = old.rows.copy()
            for key, old_value in native_rows(delta["removed"]).items():
                if rows.pop(key) != old_value:
                    raise ValueError("invalid native removal")
            for key, new_value in native_rows(delta["upserted"]).items():
                if key in rows:
                    raise ValueError("upsert without retraction")
                rows[key] = new_value
            if rows != pub.rows:
                raise ValueError("applied native delta differs from full snapshot")
            self.heads[source] = Publication(pub.checkpoint, rows, pub.manifest, delta)
        if old is None or not self.stale_cache:
            self.cache.clear()
        return "accepted"

    def query(self, work, subject, audience):
        cache_key = work, subject, audience
        if cache_key not in self.cache:
            surfaces, supports = [], set()
            for source, pub in self.heads.items():
                for key, (target, predicate, access) in pub.manifest["credits"].items():
                    if target != work or (access == "restricted" and audience == "public"):
                        continue
                    row = pub.rows[key]
                    if predicate == "printed_credit" and subject is None:
                        # This fixture gives each credit one unambiguous printed signal.
                        occurrence, = row["signals"]
                        surfaces.append(occurrence["surface"])
                    elif predicate == "attributed_to" and subject is not None:
                        record = row["catalog_record"]
                        if record and (record["catalog"], record["accession"]) == subject:
                            supports.add(source)
            self.cache[cache_key] = tuple(sorted(surfaces if subject is None else supports))
        return self.cache[cache_key]


def replay(scenario, publications, *, stale_cache=False):
    reader = Reader(stale_cache=stale_cache)
    live = {}
    for i, step in enumerate(scenario["steps"], 1):
        source, seq = step["source"], step["edition"]
        status = reader.receive(source, publications[source, seq])
        expected_status = step.get("reject", "duplicate" if live.get(source) == seq else "accepted")
        if status != expected_status:
            raise AssertionError(f"{scenario['name']} step {i}: {status}, expected {expected_status}")
        if status == "accepted":
            live[source] = seq
        rebuilt = Reader()
        rebuilt.heads = {s: publications[s, n] for s, n in live.items()}
        if {s: p.checkpoint for s, p in reader.heads.items()} != {s: p.checkpoint for s, p in rebuilt.heads.items()}:
            raise AssertionError(f"{scenario['name']} step {i}: wrong acknowledged checkpoints")
        work, expected = scenario["work"], step["expect"]
        for audience in ("restricted", "public"):
            if expected[audience].keys() - scenario["queries"].keys():
                raise ValueError(f"unknown expected query alias at step {i}")
            printed = expected["printed"]
            wanted = [(None, printed[audience] if isinstance(printed, dict) else printed)]
            for alias, qualified in scenario["queries"].items():
                wanted.append((tuple(qualified), expected[audience].get(alias, [])))
            for subject, answer in wanted:
                clean = rebuilt.query(work, subject, audience)
                actual = reader.query(work, subject, audience)
                if sorted(answer) != list(clean):
                    raise AssertionError(f"{scenario['name']} step {i}: fixture/rebuild differ for {audience} {subject}")
                if actual != clean:
                    raise ViewMismatch(f"{scenario['name']} step {i}: stale {audience} {subject} answer")
    return len(scenario["steps"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="?", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--anno", type=Path, help="use an existing anno executable")
    parser.add_argument("--sources", type=Path, default=Path(__file__).with_name("sources.json"))
    args = parser.parse_args()
    binary = (args.anno or build_cli()).resolve()
    sources = json.loads(args.sources.read_text(encoding="utf-8"))["sources"]
    source_ids = {s["id"] for s in sources}
    if len(source_ids) != len(sources):
        raise ValueError("duplicate source register ID")
    report = []
    for scenario in json.loads(args.cases.read_text(encoding="utf-8"))["scenarios"]:
        refs = scenario["references"]
        if not refs or set(refs) - source_ids or len(set(refs)) != len(refs):
            raise ValueError(f"missing, repeated or unknown source reference in {scenario['name']}")
        publications = publish(binary, scenario)
        steps = replay(scenario, publications)
        stale = None
        try:
            replay(scenario, publications, stale_cache=True)
        except ViewMismatch as fault:
            stale = str(fault)
        row_empty = sum(pub.delta is not None and not pub.delta["removed"] and not pub.delta["upserted"]
                        for pub in publications.values())
        report.append({"case": scenario["name"], "steps": steps, "editions": len(publications),
                       "deltas_with_unchanged_rows": row_empty, "stale_cache_control": stale})
    if not any(case["stale_cache_control"] for case in report):
        raise AssertionError("cases failed to distinguish a reader retaining stale cached responses")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
