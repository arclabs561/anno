"""Adapter for the MHEL-LLaMo authors' retriever, prompt chain and prediction files."""
import contextlib
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REVISION = "b262dfb77285ea00993af8d453387fd7bf81228b"


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def documents(dataset):
    docs = {}
    for row in rows(Path(dataset) / "paragraphs_test.csv"):
        if row["doc_id"] in docs:
            raise ValueError("duplicate source document")
        docs[row["doc_id"]] = row
    return docs


def mention(row, docs):
    doc = docs[row["doc_id"]]
    start, end = int(row["start_pos"]), int(row["end_pos"])
    if not 0 <= start < end <= len(doc["text"]) or doc["text"][start:end] != row["surface"]:
        raise ValueError(f"source mismatch in {row['doc_id']} at {start}:{end}")
    return doc, start, end


def import_results(predictions, docs):
    result = []
    for row in rows(predictions):
        _, start, end = mention(row, docs)
        qid = row["identifier"]
        if qid != "NIL" and not re.fullmatch(r"Q[1-9][0-9]*", qid):
            raise ValueError(f"invalid predicted Wikidata identifier {qid!r}")
        result.append(dict(doc_id=row["doc_id"], start=start, end=end, text=row["surface"],
                           type=row["type"], wikidata_id=None if qid == "NIL" else qid))
    for entry in result:
        print(json.dumps(entry, ensure_ascii=False))


def retrieval_context(transform, doc, start, end):
    text = doc["text"]
    left = list(re.finditer(r"\S+", text[:start]))[-100:]
    right = list(re.finditer(r"\S+", text[end:]))[:100]
    hi = end + right[-1].end() if right else end
    while True:
        lo = left[0].start() if left else start
        chunk = text[lo:hi]
        # The pinned BELA transform truncates at 254 SentencePiece tokens.
        # OCR and long words can put the mention outside its 100-word window.
        bounds = transform.transform({"texts": [chunk]})["sp_tokens_boundaries"][0]
        if bounds and end - lo <= bounds[-1][1]:
            return chunk, start - lo
        if not left:
            raise ValueError(f"mention in {doc['doc_id']} at {start}:{end} exceeds the BELA tokenizer window")
        left = left[(len(left) + 1) // 2:]


def retrieve(upstream, model_dir, dataset, top_k, device):
    docs = documents(dataset)
    mentions = []
    missing = {}
    for row in rows(Path(dataset) / "annotations_test.csv"):
        doc_id = row["doc_id"]
        if doc_id not in docs:
            missing[doc_id] = missing.get(doc_id, 0) + 1
            continue
        doc, start, end = mention(row, docs)
        mentions.append((row, doc, start, end))
    if missing:
        # The authors' test sets include annotations outside the supplied paragraphs.
        print(f"Skipped {sum(missing.values())} annotation rows with missing source "
              f"documents ({len(missing)} document IDs)", file=sys.stderr)
    if not mentions:
        print("[]")
        return

    for asset in ("model_wiki.ckpt", "faiss.index", "index.txt", "knowledge_base.sqlite"):
        if not (model_dir / asset).is_file():
            raise FileNotFoundError(f"missing WikiBELA asset: {model_dir / asset}")
    sys.path.insert(0, str(upstream))
    with contextlib.redirect_stdout(sys.stderr):
        from src.retriever import load_disambiguator
        model = load_disambiguator(str(model_dir), device=device)
    result = []
    for row, doc, start, end in mentions:
        with contextlib.redirect_stdout(sys.stderr):
            chunk, offset = retrieval_context(model.transform, doc, start, end)
            candidates, = model.get_candidates_batch(
                [chunk], [[offset]], [[end - start]], k=top_k, lang=doc["lang"])
        if not candidates:
            raise ValueError(f"no candidates for {row['doc_id']} at {start}:{end}")
        result.append({**row, "identifier": row.get("identifier", ""), "candidates": candidates})
    json.dump(result, sys.stdout, ensure_ascii=False)
    print()


def rerank(upstream, model_dir, dataset, candidate_path, top_k, threshold):
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be nonnegative and finite")
    docs = documents(dataset)
    candidates = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    if not candidates:
        return
    for row in candidates:
        mention(row, docs)
        if not row["candidates"]:
            raise ValueError("reranker requires candidates for every supplied mention")
        row["identifier"] = ""  # upstream CSV schema includes a reference column
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidates.json"
        path.write_text(json.dumps(candidates), encoding="utf-8")
        subprocess.run([sys.executable, str(upstream/"filter_and_prompt_chain.py"),
                        "--json_f", str(path), "--dataset_path", str(dataset),
                        "--output_dir", tmp, "--model_id", str(model_dir),
                        "--threshold", str(threshold), "--n_candidates", str(top_k)],
                       check=True, stdout=sys.stderr)
        import_results(Path(tmp)/"output.csv", docs)


def main(args):
    stage, *values = args
    if stage == "import":
        dataset, predictions = values
        import_results(predictions, documents(dataset))
        return
    upstream, model_dir, dataset = (Path(x).resolve(strict=True) for x in values[:3])
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"expected MHEL-LLaMo revision {REVISION}, got {revision}")
    top_k = int(values[3] if stage == "retrieve" else values[4])
    if top_k < 1:
        raise ValueError("top-k must be positive")
    if stage == "retrieve":
        retrieve(upstream, model_dir, dataset, top_k, values[4])
    else:
        rerank(upstream, model_dir, dataset, values[3], top_k, float(values[5]))


if __name__ == "__main__":
    main(sys.argv[1:])
