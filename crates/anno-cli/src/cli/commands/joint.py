"""JSONL interface to the optional upstream GLiNER2.5 JointIE engine."""

from contextlib import redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
import sys


def checkpoint_receipt(directory):
    # A local directory name can survive an in-place checkpoint replacement.
    files = sorted(p for p in directory.rglob("*") if p.suffix in {".json", ".safetensors", ".model"})
    receipt = {}
    for path in files:
        if ".cache" in path.relative_to(directory).parts:
            continue
        digest = sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        receipt[str(path.relative_to(directory))] = digest.hexdigest()
    return receipt


def model_windows(model, schema, text, words, window_words, overlap_words, token_limit):
    """Fit the schema and source subwords together at each advancing word offset."""
    start = 0
    while True:
        char_start = words[start][1] if start else 0
        end = min(start + window_words, len(words))

        def fit(stop):
            char_end = words[stop - 1][2] if stop < len(words) else len(text)
            fragment = text[char_start:char_end]
            encoded = model.scorer.processor.transform_record(fragment, schema)
            return len(encoded.input_ids) <= token_limit, fragment, char_end

        fits, fragment, char_end = fit(end)
        if not fits:
            # The schema uses encoder positions too. Find the longest prefix
            # that fits before applying the requested overlap.
            lo, hi = start + 1, end - 1
            best = None
            while lo <= hi:
                mid = (lo + hi) // 2
                fits, fragment, char_end = fit(mid)
                if fits:
                    best = mid, fragment, char_end
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best is None:
                raise ValueError(f"schema and source word at {char_start} exceed the encoder budget")
            end, fragment, char_end = best
        scores = model.score(fragment, schema)
        yield start, end, char_start, char_end, scores
        if end == len(words):
            return
        start = end - min(overlap_words, end - start - 1)


def merge_window_candidates(text, words, fragments):
    """Lift only locally scored candidates to document word coordinates."""
    from dataclasses import replace
    from gliner2.joint_ie.candidate_scores import CandidateScoreSet

    mentions = {}
    edges = {}
    windows = []
    covered = 0
    for start, end, char_start, char_end, scores in fragments:
        if start > covered or not 0 <= start <= end <= len(words):
            raise ValueError("invalid candidate window coverage")
        covered = max(covered, end)
        expected = [(left - char_start, right - char_start) for _, left, right in words[start:end]]
        if (text[char_start:char_end] != scores.text
                or list(zip(scores.start_mappings, scores.end_mappings, strict=True)) != expected):
            raise ValueError("candidate offsets disagree with the source tokenization")
        windows.append({"start": char_start, "end": char_end})
        local_keys = set()
        for candidate in scores.mentions:
            if not 0 <= candidate.start < candidate.end <= end - start:
                raise ValueError("mention candidate lies outside its scoring window")
            local_keys.add(candidate.key)
            lifted = replace(candidate, start=start + candidate.start, end=start + candidate.end)
            previous = mentions.get(lifted.key)
            if previous is None or lifted.logit > previous.logit:
                mentions[lifted.key] = lifted
        for edge in scores.edges:
            if edge.head not in local_keys or edge.tail not in local_keys:
                raise ValueError("relation candidate refers to a missing window endpoint")
            ht, hs, he = edge.head
            tt, ts, te = edge.tail
            lifted = replace(edge, head=(ht, start + hs, start + he),
                             tail=(tt, start + ts, start + te))
            key = lifted.relation_type, lifted.head, lifted.tail
            previous = edges.get(key)
            if previous is None or lifted.logit > previous.logit:
                edges[key] = lifted
    if not windows or covered != len(words):
        raise ValueError("scoring windows did not cover the complete document")
    return CandidateScoreSet(text=text, mentions=tuple(mentions.values()), edges=tuple(edges.values()),
                             text_tokens=tuple(word for word, _, _ in words),
                             start_mappings=tuple(start for _, start, _ in words),
                             end_mappings=tuple(end for _, _, end in words)), windows


def extract_document(model, schema, text, window_words, overlap_words, token_limit):
    from dataclasses import replace
    from gliner2.inference.chunking import iter_word_offsets
    from gliner2.processing.word_splitter import word_splitter_from
    from gliner2.joint_ie import JointIEConfig

    words = list(iter_word_offsets(text, word_splitter=word_splitter_from(model)))
    fragments = model_windows(model, schema, text, words, window_words, overlap_words, token_limit)
    candidates, windows = merge_window_candidates(text, words, fragments)
    # This is the same decoder used by JointIE.extract in gliner2 2.2.0.
    # Give the document one default candidate budget per scored window.
    # Explicit per-type schema limits and hard constraints still apply globally.
    defaults = JointIEConfig()
    config = replace(defaults, top_k_entities=defaults.top_k_entities * len(windows),
                     relation_pair_cap=defaults.relation_pair_cap * len(windows),
                     max_edges_per_type=defaults.max_edges_per_type * len(windows))
    result = model._decode(candidates, schema, config)
    return result, windows


def run(checkpoint, schema_path, window_words, overlap_words):
    window_words = int(window_words)
    overlap_words = int(overlap_words)
    if window_words < 1 or not 0 <= overlap_words < window_words:
        raise ValueError("expected a positive window and overlap in [0, window size)")

    # Some ML libraries print progress to stdout; stdout is the JSONL stream.
    with redirect_stdout(sys.stderr):
        from gliner2 import __version__
        from gliner2.joint_ie import JointIE, JointSchema

        checkpoint = Path(checkpoint)
        config = json.loads((checkpoint / "config.json").read_bytes())
        if config.get("architecture") != "boundary":
            raise ValueError("JointIE expects a local boundary checkpoint")
        token_limit = json.loads((checkpoint / "encoder_config/config.json").read_bytes())["max_position_embeddings"]
        raw_schema = Path(schema_path).read_bytes()
        schema = JointSchema.from_dict(json.loads(raw_schema))
        if schema.relation_specs and not config.get("boundary_head", {}).get("enable_relations"):
            raise ValueError("the selected checkpoint has no trained relation head")
        model = JointIE.from_pretrained(str(checkpoint), device="cpu", local_files_only=True)
        schema = model.compile_schema(schema)
        artifacts = checkpoint_receipt(checkpoint)

    for row, line in enumerate(sys.stdin, start=1):
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"document {row}: invalid JSON: {exc}") from exc
        if not isinstance(doc, dict) or not isinstance(doc.get("text"), str):
            raise ValueError(f"document {row}: expected an object containing string 'text'")
        text = doc["text"]
        with redirect_stdout(sys.stderr):
            result, windows = extract_document(model, schema, text, window_words, overlap_words, token_limit)
        graph = result.to_dict(include_confidence=True, include_spans=True)
        for entity in result.entities:
            if not (0 <= entity.start < entity.end <= len(text)) or text[entity.start:entity.end] != entity.text:
                raise ValueError(f"document {row}: entity {entity.id} has invalid source offsets")
        graph.update(
            feasible=result.feasible,
            scored_windows=windows,
            backend="gliner25_joint",
            gliner2_version=__version__,
            checkpoint=str(checkpoint),
            checkpoint_files=artifacts,
            source_sha256=sha256(text.encode()).hexdigest(),
            schema_sha256=sha256(raw_schema).hexdigest(),
        )
        if "doc_id" in doc:
            graph["doc_id"] = doc["doc_id"]
        print(json.dumps(graph, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        run(*sys.argv[1:])
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
        sys.exit(f"GLiNER2.5 JointIE: {exc}")
