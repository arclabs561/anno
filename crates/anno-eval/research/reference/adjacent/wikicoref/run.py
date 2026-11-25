"""Replay the frozen different-text queue on WikiCoref's OntoNotes projection."""
import argparse
import json
import re
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.stats import rankdata
from tokenizers import Tokenizer

from encode_entity_memory import encode_document, file_digest, verify_prefix_invariance
from entity_memory_replay import Document, Mention
from predicted_review_queue import evaluate, trace

ARCHIVE_SHA = "1204c04ada223d0880e6b08ba2964d7139f1615a580f4aff85de01e3b2ee2892"
MODEL_SHA = "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21"
TOKENIZER_SHA = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
THRESHOLD = -0.15900000000000025  # PreCo calibration, frozen before transfer.
BUDGET = 74304


def documents(archive):
    suffix = "_coref_level_OntoNotesScheme.xml"
    with tarfile.open(archive) as source:
        paths = sorted(p for p in source.getnames() if "/Markables/" in p and p.endswith(suffix))
        if len(paths) != 30:
            raise ValueError("expected one OntoNotes projection for every article")
        for path in paths:
            root, filename = path.split("/Markables/")
            title = filename.removesuffix(suffix)
            words = ET.fromstring(source.extractfile(f"{root}/Basedata/{title}_words.xml").read())
            if any(w.attrib["id"] != f"word_{i}" for i, w in enumerate(words, start=1)):
                raise ValueError(f"{title}: nonsequential word IDs")
            tokens = [w.text for w in words]
            marks = ET.fromstring(source.extractfile(path).read())
            annotated = []
            for node in marks:
                m = node.attrib
                span = re.fullmatch(r"word_(\d+)(?:\.\.word_(\d+))?", m["span"])
                if not span:
                    raise ValueError(f"{title}: unsupported span {m['span']}")
                start = int(span[1]) - 1
                end = int(span[2] or span[1])
                if not 0 <= start < end <= len(tokens):
                    raise ValueError(f"{title}: invalid source coordinates")
                annotated.append((Mention(start, end, m["coref_class"]), m["mentiontype"]))
            annotated.sort(key=lambda x: (x[0].end, x[0].start, x[0].entity))
            mentions, types = zip(*annotated)
            if len({(m.start, m.end) for m in mentions}) != len(mentions):
                raise ValueError(f"{title}: duplicate mention coordinates")
            yield Document(title, tokens, list(mentions)), np.asarray(types)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "model", "tokenizer"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    for path, digest in ((args.archive, ARCHIVE_SHA), (args.model, MODEL_SHA),
                         (args.tokenizer, TOKENIZER_SHA)):
        if file_digest(path) != digest:
            raise ValueError(f"digest mismatch for {path}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    verify_prefix_invariance(tokenizer, session, 48)
    results, mention_types = [], Counter()
    for doc, types in documents(args.archive):
        features = encode_document(doc, tokenizer, session, 48)
        rows = trace(doc, features, BUDGET, THRESHOLD)
        if len(rows) != len(doc.mentions) - 1:
            raise ValueError("expected one proposal for every noninitial mention")
        unmatched = rows[rows[:, 5] == 0]
        stats = evaluate(unmatched, len(unmatched), .20)[:3]
        different_types = types[1:][rows[:, 5] == 0]
        ranks = [rankdata(unmatched[:, i]) for i in range(3)]
        scores = [ranks[0], ranks[0] + ranks[1], ranks[0] + ranks[2]]
        slices = []
        for score, (keep, _, _) in zip(scores, stats, strict=True):
            selected = np.argsort(-score, kind="stable")[:keep]
            slices.append({t: [int((different_types[selected] == t).sum()),
                               int(unmatched[selected[different_types[selected] == t], 3].sum())]
                           for t in ("ne", "np", "pro")})
        mention_types.update(types)
        results.append(dict(document=doc.name, tokens=len(doc.tokens), mentions=len(types),
                            different_text=len(unmatched), counts=stats, type_slices=slices))
    counts = np.array([row["counts"] for row in results])
    samples = np.random.default_rng(7).integers(len(results), size=(2000, len(results)))
    pooled = counts[samples].sum(axis=1)
    gain = (pooled[:, 0, 1:] - pooled[:, 2, 1:]) / pooled[:, 0, :1]
    totals = {key: sum(r[key] for r in results) for key in ("tokens", "mentions", "different_text")}
    out = dict(archive_sha256=ARCHIVE_SHA, model_sha256=MODEL_SHA, tokenizer_sha256=TOKENIZER_SHA,
               threshold=THRESHOLD, budget=BUDGET, onnxruntime=ort.__version__,
               arms=["exemplar", "exemplar+midpoint", "exemplar+box"],
               outcomes=["reviews", "wrong_visible_pair", "wrong_or_impure_cluster"],
               totals=totals, mention_types=dict(mention_types),
               counts=counts.sum(axis=0).tolist(),
               exemplar_minus_box_ci95=np.quantile(gain,[.025,.975],axis=0).tolist(),
               documents=results)
    print(json.dumps(out, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
