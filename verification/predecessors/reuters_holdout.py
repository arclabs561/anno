# SPDX-License-Identifier: MIT OR Apache-2.0
"""Evaluate read-only predecessor suggestions on independently linked Reuters passages.

Usage: python reuters_holdout.py Reuters-128.nt target/debug/anno
Requires rdflib (tested with 7.6.0). Download Reuters-128.nt from
https://github.com/dice-group/n3-collection/tree/c136acb69b9b43555ec1b863c9a2956904581fcc.
The AKSW collection describes the dataset as CC BY-NC-SA 4.0.
Only aggregate counts are written. Gold URI labels stay in the evaluator.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from rdflib import Graph, Namespace

NIF = Namespace("http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#")
ITS = Namespace("http://www.w3.org/2005/11/its/rdf#")
RSS_DIGEST = "4ee0d831cb7fdcc7ca4263aea0444984cb210294dd9afad825caa5a66590f3cc"
DIGEST = "7b0b9a158a77edab8219999fd88b38993df748410aed9ad063f58fe8c15c0f9a"


def split(source):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    rss = digest == RSS_DIGEST
    if not rss and digest != DIGEST:
        raise ValueError("expected the pinned Reuters-128 or RSS-500 input")
    graph = Graph().parse(source, format="nt")
    catalog = defaultdict(Counter)
    queries = []
    documents = defaultdict(set)
    seen = set()
    for mention, target in sorted(graph.subject_objects(ITS.taIdentRef)):
        context = graph.value(mention, NIF.referenceContext)
        text = str(graph.value(context, NIF.isString))
        start = int(graph.value(mention, NIF.beginIndex))
        end = int(graph.value(mention, NIF.endIndex))
        surface = str(graph.value(mention, NIF.anchorOf))
        if text[start:end] != surface:
            raise ValueError(f"source span mismatch at {mention}")
        target = str(target)
        # RSS-500 explicitly labels mentions with no Wikipedia identity.
        if rss:
            if target.startswith("http://aksw.org/notInWiki/"):
                target = None
            else:
                target = target.replace("http://dbpedia.org/page/", "http://dbpedia.org/resource/")
        key = (text, start, end, target)
        if key in seen:
            continue
        seen.add(key)
        digest = hashlib.sha256(text.encode()).hexdigest()
        holdout = int(digest, 16) % 3 == 0
        documents[holdout].add(digest)
        if holdout:
            queries.append((surface, target))
        elif target is not None:
            catalog[target][surface] += 1
    return catalog, queries, len(documents[False]), len(documents[True])


def write_edition(path, sequence, names):
    # The lookup reads canonical track surfaces. Source text is a name projection.
    tracks = {str(i): {"id": i, "canonical_surface": name, "signals": [],
                       "cluster_confidence": 1.0} for i, name in enumerate(names)}
    document = {"id": "names", "text": "\n".join(names), "tracks": tracks, "signals": [], "identities": {}}
    edition = {"stream": "reuters-name-projection", "sequence": sequence,
               "threshold": 0.7, "require_type_match": True,
               "documents": [{"document": document}]}
    path.write_text(json.dumps(edition), encoding="utf-8")


def lookup(binary, current, prior, track_id, flags):
    raw = subprocess.check_output([binary, "query", "predecessors", current,
                                   "--prior", prior, "--doc-id", "names",
                                   "--track-id", str(track_id), *flags])
    return json.loads(raw)


def run(source, binary, directory):
    catalog, queries, training_docs, heldout_docs = split(source)
    surfaces = sorted({name for name, _ in queries})
    current = directory / "current.json"
    prior = directory / "prior.json"
    write_edition(current, 2, surfaces)
    result = {"prior_documents": training_docs, "held_out_documents": heldout_docs,
              "prior_identities": len(catalog), "test_mentions": len(queries), "policies": {}}
    baseline_hits = set()
    for policy in ("default", "wide", "observed_aliases"):
        entries = []
        for target, counts in sorted(catalog.items()):
            ordered = sorted(counts, key=lambda s: (-counts[s], s))
            for name in ordered if policy == "observed_aliases" else ordered[:1]:
                entries.append((name, target))
        write_edition(prior, 1, [name for name, _ in entries])
        flags = ["--limit", "5", "--minimum-score", "0"] if policy == "wide" else []
        choices = {}
        qualifying = {}
        for i, name in enumerate(surfaces):
            result_page = lookup(binary, current, prior, i, flags)
            choices[name] = [entries[c["track"]["track_id"]][1] for c in result_page["candidates"]]
            if policy == "default":
                full = lookup(binary, current, prior, i, ["--limit", "128", "--minimum-score", "0"])
                if full["truncated"]:
                    raise ValueError("diagnostic lookup truncated; cannot classify the miss")
                qualifying[name] = {entries[c["track"]["track_id"]][1]: c["score"]
                                    for c in full["candidates"]}
        stats = Counter()
        hits = set()
        for occurrence, (name, target) in enumerate(queries):
            ids = choices[name]
            present = target in catalog
            if target is None:
                stats["explicit_nil"] += 1
                stats["shown_on_explicit_nil"] += bool(ids)
            stats["in_catalog" if present else "absent_from_catalog"] += 1
            if policy == "default" and present and target not in ids:
                score = qualifying[name].get(target, 0)
                if score == 0:
                    reason = "no_positive_overlap"
                elif score < 0.5:
                    reason = "below_threshold"
                else:
                    reason = "rank_or_tie_limit"
                stats[reason] += 1
            if target in ids:
                hits.add(occurrence)
            stats["gold_shown"] += target in ids
            stats["nonempty"] += bool(ids)
            stats["shown_on_absent"] += bool(ids) and not present
            stats["abstentions"] += not ids
            stats["cards"] += len(ids)
            stats["nongold_cards"] += sum(t != target for t in ids)
            stats["first_correct"] += bool(ids) and ids[0] == target
            stats["first_wrong"] += bool(ids) and ids[0] != target
        stats["eligible_track_comparisons_per_mention_summed"] = len(entries) * len(queries)
        if policy == "default":
            baseline_hits = hits
        stats["gold_gained_vs_default"] = len(hits - baseline_hits)
        stats["gold_lost_vs_default"] = len(baseline_hits - hits)
        result["policies"][policy] = dict(stats)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("anno", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory() as temp:
        print(json.dumps(run(args.source, args.anno.resolve(), Path(temp)), sort_keys=True, indent=2))
