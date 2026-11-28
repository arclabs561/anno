#!/usr/bin/env python3
"""Join anno extract and anno corpipe output for anno glidre and anno mhel."""
import argparse
import hashlib
import json
from pathlib import Path


def span(text, item):
    start, end = item["start"], item["end"]
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
            or item["text"] != text[start:end]):
        raise ValueError(f"source mismatch at {start}:{end}")
    return start, end


def prepare(source, ner, coref):
    text = source["text"]
    encoded = text.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if ner["source_map"]["source_sha256"] != digest:
        raise ValueError("NER source differs from the input document")
    if (coref["source_sha256"] != digest
            or coref["offset_unit"] != "unicode_scalar" or coref["source_length"] != len(text)):
        raise ValueError("CorPipe source differs from the input document")
    detected = {}
    for entity in ner["entities"]:
        detected.setdefault(span(text, entity), []).append(entity)

    entities, used, cluster_ids = [], set(), set()
    for cluster in coref["clusters"]:
        cid = cluster["id"]
        if cid in cluster_ids:
            raise ValueError(f"duplicate CorPipe cluster ID {cid!r}")
        cluster_ids.add(cid)
        spans = {}
        for mention in cluster["mentions"]:
            if "start" in mention or "end" in mention:
                start, end = span(text, mention)
                spans[start, end] = {"start": start, "end": end, "text": text[start:end]}
        anchors = [hit for key in spans for hit in detected.get(key, [])]
        if not anchors:
            continue
        for key, mention in spans.items():
            if key in detected:
                mention["ner_ids"] = [hit["id"] for hit in detected[key]]
                used.add(key)
        # CorPipe clusters have no NER type. Use the highest-confidence exact NER
        # anchor; stable input order breaks ties and no name-string matching is used.
        best = max(anchors, key=lambda e: e["confidence"])
        entities.append({"id": f"coref:{cid}", "corpipe_id": cid, "type": best["type"],
                         "mentions": sorted(spans.values(), key=lambda m: (m["start"], m["end"]))})
    for key, hits in detected.items():
        if key in used:
            continue
        best = max(hits, key=lambda e: e["confidence"])
        start, end = key
        entities.append({"id": f"ner:{best['id']}", "type": best["type"], "mentions": [
            {"start": start, "end": end, "text": text[start:end], "ner_ids": [h["id"] for h in hits]}]})
    if len({e["id"] for e in entities}) != len(entities):
        raise ValueError("source provider assigned the same ID to distinct entities")
    return {**source, "entities": entities}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="one source JSON object with id, language and exact text")
    parser.add_argument("ner", help="anno extract --format json output")
    parser.add_argument("coref", help="anno corpipe --format json output")
    args = parser.parse_args()
    print(json.dumps([prepare(read(args.source), read(args.ner), read(args.coref))], ensure_ascii=False))


if __name__ == "__main__":
    main()
