# SPDX-License-Identifier: MIT OR Apache-2.0
"""Check native predecessor shortlists against the complete previous published roster."""

import argparse
import hashlib
import json
import random
import struct
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from committee_pilot import load, similarity
from publication_input import grounded_names

DOC_ID = "published-roster"
STREAM = "legislators-name-projection"


@dataclass
class Edition:
    path: Path
    sequence: int
    revision: str
    text_hash: str
    keys: list
    names: list

    def evidence(self, track_id):
        return {"doc_id": DOC_ID, "track_id": track_id, "source_revision": self.revision,
                "text_sha256": self.text_hash, "surface": self.names[track_id]}


def project(records, sequence, revision, path):
    # Catalog IDs stay in the evaluator. Reassign track handles at each cut.
    keys = list(records)
    random.Random(sequence).shuffle(keys)
    named = [key for key in keys if records[key]]
    keys = named + [key for key in keys if not records[key]]
    names = [records[key] for key in keys]
    doc = grounded_names(DOC_ID, [(records[key], None) for key in named])
    for i in range(len(named), len(keys)):
        doc["tracks"][str(i)] = {"id": i, "signals": [], "canonical_surface": "",
                                 "cluster_confidence": 1.0}
    edition = {"stream": STREAM, "sequence": sequence, "threshold": 0.7, "require_type_match": True,
               "documents": [{"source_revision": revision, "document": doc}]}
    path.write_text(json.dumps(edition, separators=(",", ":")), encoding="utf-8")
    return Edition(path, sequence, revision, hashlib.sha256(doc["text"].encode()).hexdigest(), keys, names)


def f32(value):
    return struct.unpack("!f", struct.pack("!f", value))[0]


def expected(name, prior, limit, minimum):
    groups = defaultdict(list)
    for i, earlier in enumerate(prior.names):
        score = f32(similarity(name, earlier))
        if score > 0 and score >= minimum:
            groups[score].append(i)
    cards = []
    for score, ids in sorted(groups.items(), reverse=True):
        remaining = limit - len(cards)
        if not remaining:
            return cards, None, True
        if len(ids) > remaining:
            return cards, {"score": score, "count": len(ids)}, True
        cards.extend((i, score) for i in ids)
    return cards, None, False


def verify(binary, current, prior, i, limit, minimum, flags):
    command = [binary, "query", "predecessors", current.path, "--prior", prior.path,
               "--doc-id", DOC_ID, "--track-id", str(i), *flags]
    result = json.loads(subprocess.run(command, stdout=subprocess.PIPE, check=True).stdout)
    cards, tie, truncated = expected(current.names[i], prior, limit, minimum)
    assert result["scorer"] == "word_jaccard_v1" and result["stream"] == STREAM
    assert (result["current_sequence"], result["prior_sequence"]) == (current.sequence, prior.sequence)
    assert result["query"] == current.evidence(i)
    assert (result["limit"], result["minimum_score"]) == (limit, minimum)
    assert result["truncated"] == truncated
    withheld = result["withheld_tie"]
    if withheld is not None:
        withheld = {"score": f32(withheld["score"]), "count": withheld["count"]}
    assert withheld == tie
    actual = [(c["track"], f32(c["score"])) for c in result["candidates"]]
    assert actual == [(prior.evidence(j), score) for j, score in cards], (current.sequence, i)
    return [prior.keys[j] for j, _ in cards]


def replay(binary, roster):
    records, counts = {}, defaultdict(lambda: defaultdict(Counter))
    policies = [("default", 1, 0.5, []), ("wide", 5, 0.0, ["--limit", "5", "--minimum-score", "0"])]
    queries = pairs = 0
    editions = load(roster)
    if len(editions) < 2:
        raise ValueError("at least two roster editions are required")
    with TemporaryDirectory() as directory:
        prior = None
        for n, edition in enumerate(editions, start=1):
            old = records.copy()
            for key in edition["remove"]:
                del records[key]
            records.update((row["id"], row["official_full"] or "") for row in edition["upsert"])
            current = project(records, n, edition["revision"], Path(directory) / f"{n % 2}.json")
            if prior:
                current_ids = {key: i for i, key in enumerate(current.keys)}
                for row in edition["upsert"]:
                    key, name = row["id"], row["official_full"]
                    if not name:
                        continue
                    queries += 1
                    pairs += len(prior.keys)
                    group = "other"
                    if key not in old:
                        group = "absent"
                    elif old[key] and old[key] != name:
                        group = "renamed"
                    for policy, limit, minimum, flags in policies:
                        chosen = verify(binary, current, prior, current_ids[key], limit, minimum, flags)
                        out = counts[policy][group]
                        out["queries"] += 1
                        out["with_suggestions"] += bool(chosen)
                        out["candidate_cards"] += len(chosen)
                        out["correct_predecessor_shown"] += key in chosen
            prior = current
    return {"editions": len(editions), "named_queries": queries, "eligible_pairs_per_policy": pairs,
            "policies": counts, "roster_sha256": hashlib.sha256(roster.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roster", type=Path)
    parser.add_argument("--anno", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.anno.resolve(), args.roster), indent=2, sort_keys=True))
