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


def valid_id(value):
    return type(value) is int or (isinstance(value, str) and bool(value))


class SourceDocuments:
    """Keep caller IDs and original offsets at the upstream string CSV boundary."""

    def __init__(self, source):
        if not isinstance(source, list):
            raise ValueError("expected a JSON document array")
        self.docs = {}
        self.ids = {}
        self.occurrences = {}
        seen = set()
        for index, doc in enumerate(source):
            doc_id, text, lang = doc["id"], doc["text"], doc["language"]
            if not valid_id(doc_id) or doc_id in seen:
                raise ValueError(f"invalid or duplicate document ID {doc_id!r}")
            if not isinstance(text, str) or lang not in ("de", "en", "fi", "fr", "it", "nl", "sv"):
                raise ValueError(f"{doc_id}: expected text and a supported language")
            seen.add(doc_id)
            # One string key per document keeps integer and string IDs distinct in CSV.
            key = str(index)
            paragraph = dict(doc_id=key, text=text, lang=lang)
            for field in ("publication_date", "genre"):
                value = doc.get(field, "")
                if not isinstance(value, str):
                    raise ValueError(f"{doc_id}: expected a string for {field}")
                paragraph[field] = value
            self.docs[key] = paragraph
            self.ids[key] = doc_id
            if not isinstance(doc["entities"], list):
                raise ValueError(f"{doc_id}: expected an entities array")
            entity_ids = set()
            for entity in doc["entities"]:
                eid, typ, spans = entity["id"], entity["type"], entity["mentions"]
                if not valid_id(eid) or eid in entity_ids:
                    raise ValueError(f"{doc_id}: invalid or duplicate entity ID {eid!r}")
                if not isinstance(typ, str) or not typ or not isinstance(spans, list) or not spans:
                    raise ValueError(f"{doc_id}: entity {eid!r} requires a type and mention spans")
                entity_ids.add(eid)
                for span in spans:
                    start, end = span["start"], span["end"]
                    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                        raise ValueError(f"{doc_id}: invalid character range for entity {eid!r}")
                    surface = text[start:end]
                    occurrence = (doc_id, eid, start, end)
                    if occurrence in self.occurrences or span.get("text", surface) != surface:
                        raise ValueError(f"{doc_id}: duplicate or mismatched mention {eid!r} at {start}:{end}")
                    self.occurrences[occurrence] = dict(doc_id=key, entity_id=eid, surface=surface,
                                                        start_pos=start, end_pos=end, type=typ)

    def candidate_rows(self, entries, threshold):
        if not isinstance(entries, list):
            raise ValueError("expected a candidate array")
        remaining = self.occurrences.copy()
        result = []
        for item in entries:
            key = tuple(item[field] for field in ("doc_id", "entity_id", "start", "end"))
            if not all(valid_id(i) for i in key[:2]) or not all(type(i) is int for i in key[2:]):
                raise ValueError(f"invalid candidate occurrence {key!r}")
            row = remaining.pop(key, None)
            if row is None:
                raise ValueError(f"unknown or duplicate candidate occurrence {key!r}")
            if item.get("text", row["surface"]) != row["surface"] or item.get("type", row["type"]) != row["type"]:
                raise ValueError(f"candidate text or type differs from source at {key!r}")
            choices = item["candidates"]
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"expected a nonempty candidate list at {key!r}")
            normalized = []
            for choice in choices:
                qid, label = choice["wb_id"], choice["label"]
                if not isinstance(qid, str) or not re.fullmatch(r"Q[1-9][0-9]*", qid) or not isinstance(label, str):
                    raise ValueError(f"invalid candidate Wikidata ID or label at {key!r}")
                score = choice.get("score")
                if score is not None and (type(score) not in (int, float) or not math.isfinite(score)):
                    raise ValueError(f"invalid candidate score at {key!r}")
                if threshold > 0 and score is None:
                    raise ValueError(f"a positive threshold requires scored candidates at {key!r}")
                normalized.append(dict(wb_id=qid, label=label, type=choice.get("type", ""),
                                       descr=choice.get("descr", ""), min_date=choice.get("min_date", ""),
                                       score=score))
            result.append({**row, "candidates": normalized})
        if remaining:
            raise ValueError(f"missing candidates for {len(remaining)} supplied occurrences")
        return result

    def candidate_output(self, row, candidates):
        return dict(doc_id=self.ids[row["doc_id"]], entity_id=row["entity_id"],
                    start=row["start_pos"], end=row["end_pos"], text=row["surface"],
                    type=row["type"], candidates=candidates)


def mention(row, docs):
    doc = docs[row["doc_id"]]
    start, end = int(row["start_pos"]), int(row["end_pos"])
    if not 0 <= start < end <= len(doc["text"]) or doc["text"][start:end] != row["surface"]:
        raise ValueError(f"source mismatch in {row['doc_id']} at {start}:{end}")
    return doc, start, end


def import_results(predictions, docs, native=None, expected=()):
    result = []
    predictions = rows(predictions)
    if native is not None and len(predictions) != len(expected):
        raise ValueError("reranker returned an unexpected number of occurrences")
    for index, row in enumerate(predictions):
        _, start, end = mention(row, docs)
        qid = row["identifier"]
        if qid != "NIL" and not re.fullmatch(r"Q[1-9][0-9]*", qid):
            raise ValueError(f"invalid predicted Wikidata identifier {qid!r}")
        result.append(dict(doc_id=row["doc_id"], start=start, end=end, text=row["surface"],
                           type=row["type"], wikidata_id=None if qid == "NIL" else qid))
        if native is not None:
            source = expected[index]
            # The pinned upstream script emits one row per candidate entry in input order.
            if (source["doc_id"], source["start_pos"], source["end_pos"], source["type"]) != (
                    row["doc_id"], start, end, row["type"]):
                raise ValueError("reranker returned a different source occurrence")
            result[-1].update(doc_id=native.ids[row["doc_id"]], entity_id=source["entity_id"])
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


def retrieve(upstream, model_dir, dataset, top_k, device, native=None):
    docs = native.docs if native is not None else documents(dataset)
    annotations = native.occurrences.values() if native is not None else rows(Path(dataset) / "annotations_test.csv")
    mentions = []
    missing = {}
    for row in annotations:
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
        if native is not None:
            result.append(native.candidate_output(row, candidates))
        else:
            result.append({**row, "identifier": row.get("identifier", ""), "candidates": candidates})
    json.dump(result, sys.stdout, ensure_ascii=False)
    print()


def rerank(upstream, model_dir, dataset, candidate_path, top_k, threshold, native=None):
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be nonnegative and finite")
    docs = native.docs if native is not None else documents(dataset)
    candidates = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    if native is not None:
        candidates = native.candidate_rows(candidates, threshold)
    if not candidates:
        return
    for row in candidates:
        mention(row, docs)
        if not row["candidates"]:
            raise ValueError("reranker requires candidates for every supplied mention")
        row["identifier"] = ""  # upstream CSV schema includes a reference column
    with tempfile.TemporaryDirectory() as tmp:
        if native is not None:
            dataset = Path(tmp)
            with (dataset / "paragraphs_test.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("doc_id", "text", "lang", "publication_date", "genre"))
                writer.writeheader()
                for doc in docs.values():
                    # The upstream CSV reader uses universal newlines. Keep one code
                    # point per source character when the prompt contains CRLF.
                    writer.writerow({**doc, "text": doc["text"].replace("\r\n", " \n").replace("\r", "\n")})
        path = Path(tmp) / "candidates.json"
        path.write_text(json.dumps(candidates), encoding="utf-8")
        subprocess.run([sys.executable, str(upstream/"filter_and_prompt_chain.py"),
                        "--json_f", str(path), "--dataset_path", str(dataset),
                        "--output_dir", tmp, "--model_id", str(model_dir),
                        "--threshold", str(threshold), "--n_candidates", str(top_k)],
                       check=True, stdout=sys.stderr)
        import_results(Path(tmp)/"output.csv", docs, native, candidates)


def main(args):
    stage, *values = args
    if stage == "import":
        dataset, predictions = values
        import_results(predictions, documents(dataset))
        return
    upstream, model_dir, dataset = (Path(x).resolve(strict=True) for x in values[:3])
    native = None
    if values[3] == "--documents":
        native = SourceDocuments(json.loads(dataset.read_text(encoding="utf-8")))
        del values[3]
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"expected MHEL-LLaMo revision {REVISION}, got {revision}")
    top_k = int(values[3] if stage == "retrieve" else values[4])
    if top_k < 1:
        raise ValueError("top-k must be positive")
    if stage == "retrieve":
        retrieve(upstream, model_dir, dataset, top_k, values[4], native)
    else:
        rerank(upstream, model_dir, dataset, values[3], top_k, float(values[5]), native)


if __name__ == "__main__":
    main(sys.argv[1:])
