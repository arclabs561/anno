"""Run the pinned GLiDRE tokenized-document model with supplied mention clusters."""
import contextlib
import json
import math
from pathlib import Path
import subprocess
import sys

UPSTREAM_REVISION = "1a830bb2154fc503c968081e4d583d320fa5c309"


def prepare(doc):
    title = doc["title"]
    sents = doc["sents"]
    if not isinstance(title, str) or not title:
        raise ValueError("expected a nonempty document title")
    if not isinstance(sents, list) or any(
        not isinstance(sent, list) or any(not isinstance(w, str) or not w for w in sent)
        for sent in sents
    ):
        raise ValueError(f"{title}: expected sentences of nonempty tokens")
    tokens, starts = [], []
    for sent in sents:
        starts.append(len(tokens))
        tokens.extend(sent)
    if not isinstance(doc["vertexSet"], list):
        raise ValueError(f"{title}: expected a vertexSet array")
    vertices = []
    for vertex_id, cluster in enumerate(doc["vertexSet"]):
        if not isinstance(cluster, list) or not cluster:
            raise ValueError(f"{title}: empty or invalid vertex {vertex_id}")
        mentions = []
        for mention in cluster:
            sent, (start, end) = mention["sent_id"], mention["pos"]
            if (any(type(i) is not int for i in (sent, start, end)) or
                    not 0 <= sent < len(sents) or not 0 <= start < end <= len(sents[sent])):
                raise ValueError(f"{title}: invalid mention span in vertex {vertex_id}")
            mentions.append({"start": starts[sent] + start, "end": starts[sent] + end - 1,
                             "value": " ".join(sents[sent][start:end])})
        vertices.append({"id": vertex_id, "mentions": mentions})
    return tokens, vertices


def valid_id(value):
    return type(value) is int or (isinstance(value, str) and bool(value))


def prepare_text(doc, splitter):
    doc_id, text, entities = doc["id"], doc["text"], doc["entities"]
    if not valid_id(doc_id) or not isinstance(text, str) or not isinstance(entities, list):
        raise ValueError("expected a document ID, text string and entities array")
    tokens, starts, ends = [], {}, {}
    for token, start, end in splitter(text):
        if start >= end or text[start:end] != token:
            raise ValueError(f"{doc_id}: upstream tokenizer lost its source offsets")
        starts[start] = ends[end] = len(tokens)
        tokens.append(token)

    vertices, grounded, seen = [], [], set()
    for entity in entities:
        eid, mentions = entity["id"], entity["mentions"]
        if not valid_id(eid) or eid in seen:
            raise ValueError(f"{doc_id}: invalid or duplicate entity ID {eid!r}")
        seen.add(eid)
        if not isinstance(mentions, list) or not mentions:
            raise ValueError(f"{doc_id}: entity {eid!r} requires at least one mention")
        indexed, occurrences = [], []
        for mention in mentions:
            start, end = mention["start"], mention["end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                raise ValueError(f"{doc_id}: invalid character span on entity {eid!r}")
            surface = text[start:end]
            if "text" in mention and mention["text"] != surface:
                raise ValueError(f"{doc_id}: mention text mismatch at {start}:{end}")
            if start not in starts or end not in ends:
                raise ValueError(f"{doc_id}: mention {start}:{end} must align with GLiDRE token boundaries")
            indexed.append({"start": starts[start], "end": ends[end], "value": surface})
            occurrences.append({**mention, "text": surface})
        # The pinned pair classifier consumes mention boundaries and evaluates
        # every ordered cluster pair; entity type is optional metadata.
        vertices.append({"id": eid, "mentions": indexed})
        grounded.append({**entity, "mentions": occurrences})
    return tokens, vertices, grounded


def schema_labels(schema):
    if not isinstance(schema, dict) or not schema or any(
        not isinstance(value, str) or not value.strip()
        for pair in schema.items() for value in pair
    ):
        raise ValueError("schema must map nonempty predicate IDs to nonempty text prompts")
    ids = {label: predicate for predicate, label in schema.items()}
    if len(ids) != len(schema):
        raise ValueError("each predicate needs a distinct text prompt")
    return ids


def project(decoded, vertices, ids):
    """Recover original vertex IDs from their ordered slots in the pair classifier."""
    edges = []
    n = len(vertices)
    if len(decoded) != n * n:
        raise ValueError("GLiDRE returned an unexpected number of entity pairs")
    for pair_id, predictions in enumerate(decoded):
        head, tail = divmod(pair_id, n)
        for prediction in predictions:
            for vertex, key in ((head, "entity_1"), (tail, "entity_2")):
                spans = [(m["start"], m["end"]) for m in vertices[vertex]["mentions"]]
                if prediction[key] != spans:
                    raise ValueError("GLiDRE decoded an endpoint inconsistent with its source vertex")
            score = float(prediction["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid GLiDRE prediction score")
            edges.append({"h": head, "t": tail, "r": ids[prediction["relation_type"]],
                          "score": score})
    return edges


def classify(model, doc_id, tokens, vertices, ids, threshold):
    import torch

    if not vertices:
        return []

    processor = model.data_processor
    # Upstream clips words first, then its bi-encoder tokenizer clips subwords.
    if len(tokens) > model.config.max_len:
        raise ValueError(f"{doc_id}: {len(tokens)} words exceed the GLiDRE limit of {model.config.max_len}")
    texts = processor.prepare_texts([tokens]) if processor.preprocess_text else [tokens]
    tokenizer = processor.transformer_tokenizer
    subwords = tokenizer(texts, is_split_into_words=True, truncation=False)["input_ids"][0]
    labels = list(ids)
    raw = processor.collate_raw_batch([{"tokenized_text": tokens, "mentions": vertices}], labels)
    inputs = processor.collate_fn(raw)
    if inputs["input_ids"].shape[1] != len(subwords):
        raise ValueError(f"{doc_id}: GLiDRE tokenizer truncated the document")
    inputs.update(text_lengths=raw["seq_length"], rel_idx=raw["rel_idx"])
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model.model(**inputs).logits
        # The authors' display wrapper reconstructs an ID from the first shared
        # mention. Decode one ordered pair per batch slot to retain vertex identity.
        pairs = len(vertices) ** 2
        decoded = model.decoder.decode(
            [tokens] * pairs, raw["id_to_classes"], raw["rel_idx"][0].unsqueeze(1),
            logits[0].unsqueeze(1), threshold=threshold, multi_label=True)
    return project(decoded, vertices, ids)


def predict(model, doc, ids, threshold):
    if "text" in doc:
        tokens, vertices, entities = prepare_text(doc, model.data_processor.words_splitter)
        edges = classify(model, doc["id"], tokens, vertices, ids, threshold)
        relations = [{"head": vertices[e["h"]]["id"], "tail": vertices[e["t"]]["id"],
                      "predicate": e["r"], "score": e["score"]} for e in edges]
        return {"id": doc["id"], "text": doc["text"], "entities": entities, "relations": relations}
    tokens, vertices = prepare(doc)
    edges = classify(model, doc["title"], tokens, vertices, ids, threshold)
    return {**{k: doc[k] for k in ("title", "sents", "vertexSet")}, "labels": edges}


def main(upstream, model_dir, input_path, schema_path, threshold, device):
    upstream = Path(upstream).resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"expected GLiDRE revision {UPSTREAM_REVISION}, got {revision}")
    threshold = float(threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    ids = schema_labels(json.loads(Path(schema_path).read_text(encoding="utf-8")))
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    docs = [payload] if isinstance(payload, dict) else payload
    if not isinstance(docs, list) or any(not isinstance(d, dict) for d in docs):
        raise ValueError("expected a JSON document or an array of documents")
    keys = [("text", d["id"]) if "text" in d else ("docred", d["title"]) for d in docs]
    if any(not valid_id(key) for _, key in keys) or len(set(keys)) != len(keys):
        raise ValueError("invalid or duplicate document ID")
    model_dir = Path(model_dir).resolve(strict=True)
    sys.path.insert(0, str(upstream))
    with contextlib.redirect_stdout(sys.stderr):
        from glidre import GLiDRE
        model = GLiDRE.from_pretrained(str(model_dir), strict=True).to(device).eval()
        result = [predict(model, doc, ids, threshold) for doc in docs]
    json.dump(result[0] if isinstance(payload, dict) else result, sys.stdout,
              ensure_ascii=False, allow_nan=False)
    print()


if __name__ == "__main__":
    main(*sys.argv[1:])
