#!/usr/bin/env python3
"""Regenerate the sparse-tag decoder fixture with the pinned upstream executable tagger.

Pass tplinker_plus/tplinker_plus.py from the revision below. Requires CPU PyTorch.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import torch

REVISION = "c9305b14aaa528aa0ecc69236e1fafac3e36f95f"
DIGEST = "dd4037c483c733a3a9161d690d77ebbb96b1b84efeeca334c2da66a194f8fbdd"
OUT = Path(__file__).resolve().parents[2] / "crates/anno/tests/fixtures/tplinker_plus_tags.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream_file", type=Path)
    source = parser.parse_args().upstream_file.read_bytes()
    if hashlib.sha256(source).hexdigest() != DIGEST:
        raise ValueError("source digest differs from pinned upstream")
    # The tagger is standalone. Extract its class verbatim to avoid loading
    # the training framework and unrelated debugger and encoder dependencies.
    node, = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
             and n.name == "HandshakingTaggingScheme"]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "tplinker_plus.py", "exec"), scope)
    text = "Zoë at Élan Labs met Zoë"
    offsets = [[0, 3], [4, 6], [7, 11], [12, 16], [17, 20], [21, 24]]
    rels = {"works at": 4, "knows": 0, "absent": 3}
    types = {"org": 2, "person": 0, "team": 1}
    tagger = scope[node.name](rels, len(offsets), types)
    common = {"text": text, "tok2char_span": offsets, "rel2id": rels, "entity_type2id": types}
    entities = [("person", 0, 1), ("org", 2, 4), ("team", 2, 3), ("person", 5, 6)]
    edges = [("knows", 0, 0), ("works at", 0, 1), ("knows", 3, 0), ("works at", 1, 2)]
    def make(name, mentions, pairs, extra):
        sample = {"entity_list": [{"type": t, "tok_span": [s, e]} for t, s, e in mentions],
                  "relation_list": [{"predicate": p, "subj_tok_span": mentions[s][1:],
                                     "obj_tok_span": mentions[o][1:]} for p, s, o in pairs]}
        spots = tagger.get_spots(sample) + extra
        shaking = tagger.spots2shaking_tag(spots)
        relations, found = tagger.decode_rel(text, shaking, offsets)
        return {"name": name, "input": {**common, "spots": spots},
                "output": {"entity_list": found, "relation_list": relations}}
    cases = [
        make("reverse_self_nested_and_repeated", entities, edges,
             [(0, 2, tagger.tag2id["absent⺀SH2OH"])]),
        make("same_span_two_types", [("org", 2, 4), ("team", 2, 4)], [("knows", 0, 1)], []),
        make("empty", [], [], []),
    ]
    OUT.write_text(json.dumps({"source_revision": REVISION, "source_file_sha256": DIGEST,
                    "tags": tagger.tags, "cases": cases}, ensure_ascii=False, indent=2) + "\n")

if __name__ == "__main__":
    main()
