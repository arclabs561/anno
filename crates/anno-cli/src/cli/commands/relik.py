"""Read one unmodified document and link mentions to caller-supplied catalog IDs."""
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys

REVISION = "3bed88483cabb40188d7a3f8e79ec895c45b3a28"


def load_reader(directory):
    import torch
    from relik.reader.pytorch_modules.hf.configuration_relik import RelikReaderConfig
    from relik.reader.pytorch_modules.hf.modeling_relik import RelikReaderSpanModel
    from relik.reader.pytorch_modules.span import RelikReaderForSpanExtraction
    from safetensors.torch import load_file
    from transformers import DebertaV2TokenizerFast

    receipt = json.loads((directory / "receipt.json").read_bytes())
    if receipt["revision"] != REVISION:
        raise ValueError("expected the prepared AIDA base reader")
    expected = {"encoder/config.json", "encoder/model.safetensors", "reader.json", "reader.safetensors",
                *("tokenizer/" + n for n in ("added_tokens.json", "special_tokens_map.json", "spm.model",
                                             "tokenizer.json", "tokenizer_config.json"))}
    if expected != set(receipt["files"]):
        raise ValueError("incomplete ReLiK preparation receipt")
    for name in sorted(expected):
        with (directory / name).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != receipt["files"][name]:
                raise ValueError(f"ReLiK prepared artifact mismatch: {name}")
    config = RelikReaderConfig.from_json_file(str(directory / "reader.json"))
    config.transformer_model = str(directory / "encoder")
    # The published encoder includes the trained reader vocabulary. Restore its
    # extra embeddings after the official reader constructor resizes the base.
    core = RelikReaderSpanModel(config)
    head = load_file(str(directory / "reader.safetensors"))
    extra = head.pop("additional_word_embeddings")
    targets = {k: v for k, v in core.state_dict().items() if not k.startswith("transformer_model.")}
    if targets.keys() != head.keys() or extra.shape[0] != config.additional_special_symbols:
        raise ValueError("incomplete ReLiK reader head")
    with torch.no_grad():
        core.transformer_model.get_input_embeddings().weight[-len(extra):].copy_(extra)
        for name, tensor in targets.items():
            tensor.copy_(head[name])
    tok = DebertaV2TokenizerFast.from_pretrained(str(directory / "tokenizer"), local_files_only=True)
    return RelikReaderForSpanExtraction(core, tokenizer=tok, use_nme=True, device="cpu", precision=32).eval()


def catalog(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise ValueError("supply a JSON array of 1 to 100 candidates")
    titles, ids = {}, set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each candidate needs an id and title")
        title, key = row.get("title"), row.get("id")
        if not isinstance(title, str) or not title.strip() or not isinstance(key, str) or not key.strip():
            raise ValueError("candidate id and title must be nonempty strings")
        if key in ids or title in titles or title == "--NME--":
            raise ValueError("candidate IDs and titles must be unique; --NME-- is reserved")
        ids.add(key)
        titles[title] = key
    return titles


def link(reader, text, titles):
    import spacy
    from relik.reader.data.relik_reader_sample import RelikReaderSample
    words = [w for w in spacy.blank("en").make_doc(text) if not w.is_space]
    if not words:
        return []
    tokenizer = reader.tokenizer
    schema = ["--NME--", *(f"[E-{i}] {title}" for i, title in enumerate(titles))]
    pieces = tokenizer(schema, add_special_tokens=False).input_ids
    # ReLiK independently truncates each title to 22 pieces. Require every
    # candidate to fit in every scored window, instead of silently dropping titles.
    room = 512 - 20 - sum(min(22, len(row)) for row in pieces)
    if any(sum(token_id in tokenizer.additional_special_tokens_ids for token_id in row) != 1 for row in pieces):
        raise ValueError("candidate titles must not contain ReLiK reserved markers")
    if room < 1:
        raise ValueError("candidate titles exceed the reader window; supply fewer candidates")
    lengths = [len(p) for p in tokenizer([w.text for w in words], add_special_tokens=False).input_ids]
    output = {}
    start = 0
    while start < len(words):
        end, used = start, 0
        while end < min(start + 32, len(words)) and used + lengths[end] <= room:
            used += lengths[end]
            end += 1
        if end == start:
            raise ValueError(f"word at character {words[start].idx} cannot fit beside the candidate list")
        chunk = words[start:end]
        tokens = [w.text for w in chunk]
        sample = RelikReaderSample(tokens=tokens, words=tokens, span_candidates=list(titles),
            token2char_start={str(i): w.idx for i,w in enumerate(chunk)},
            token2char_end={str(i): w.idx+len(w) for i,w in enumerate(chunk)},
            char2token_start={str(w.idx): i for i,w in enumerate(chunk)},
            char2token_end={str(w.idx+len(w)): i for i,w in enumerate(chunk)})
        (prediction,) = reader.read(samples=[sample], annotation_type="char", max_length=512,
                                    max_batch_size=1, top_k=-1)
        for left, right, title in sorted(prediction.predicted_spans):
            if not chunk[0].idx <= left < right <= chunk[-1].idx + len(chunk[-1]) or title not in titles:
                raise ValueError("reader returned an ungrounded mention or unknown title")
            probability = max(p for t,p in prediction.probs_window_labels_chars[(left,right)] if t == title)
            item = dict(start=left, end=right, text=text[left:right], id=titles[title], title=title,
                        candidate_probability=probability)
            key = (left, right)
            if key not in output or probability > output[key]["candidate_probability"]:
                output[key] = item
        if end == len(words):
            break
        start = min(start + 16, max(start + 1, end - 1))
    return [output[k] for k in sorted(output)]


def main():
    raw = sys.stdin.buffer.read()
    text = raw.decode("utf-8")
    candidates = Path(sys.argv[2]).read_bytes()
    titles = catalog(json.loads(candidates))
    with redirect_stdout(sys.stderr):
        reader = load_reader(Path(sys.argv[1]))
        mentions = link(reader, text, titles)
    print(json.dumps({"reader": "sapienzanlp/relik-reader-deberta-v3-base-aida", "revision": REVISION,
                      "source_sha256": hashlib.sha256(raw).hexdigest(),
                      "candidates_sha256": hashlib.sha256(candidates).hexdigest(),
                      "candidate_count": len(titles), "mentions": mentions}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
        sys.exit(f"ReLiK: {exc}")
