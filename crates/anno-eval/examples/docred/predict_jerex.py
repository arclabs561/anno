"""Run the upstream JEREX joint model with a local checkpoint on CPU.

Give Python access to the lavis-nlp/jerex checkout (e.g. via PYTHONPATH).
Use --input-format text for one UTF-8 document or pass tokenized JSON by default.
The published checkpoint supplies the entity types, predicates and thresholds.
The output uses JEREX joint-mode predictions.json with document-wide token spans.
Text input also records exact Unicode character offsets into the original text.
"""

import argparse
import hashlib
import json
import re
import sys
from itertools import accumulate
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import is_zipfile


def text_document(text, title):
    """Create upstream sentence tokens directly from unchanged Unicode source text."""
    sentences = []
    spans = []
    terminator = False
    previous_end = 0
    for match in re.finditer(r"\w+|[^\w\s]", text):
        token = match.group()
        gap = text[previous_end:match.start()]
        if not sentences or re.search(r"[\r\n\u0085\u2028\u2029]", gap) or (gap and terminator):
            sentences.append([])
            terminator = False
        sentences[-1].append(token)
        spans.append(match.span())
        terminator = token in ".?!。！？" or (terminator and token in "\"'”’»)]}")
        previous_end = match.end()
    if not spans:
        raise ValueError("Text input must contain at least one non-whitespace token")
    return {"title": title, "sents": sentences, "source_text": text, "token_char_spans": spans}


def attach_text_offsets(row, document):
    """Keep upstream token indices and add verified ranges into raw input text."""
    tokens = [token for sentence in document["sents"] for token in sentence]
    if row["tokens"] != tokens:
        raise ValueError("JEREX output tokens differ from the raw input tokens")
    spans = document["token_char_spans"]
    row["source_text"] = document["source_text"]
    row["token_char_spans"] = spans
    row["sentence_token_ends"] = list(accumulate(map(len, document["sents"])))
    for mention in row["mentions"]:
        start, end = mention["start"], mention["end"]
        if not 0 <= start < end <= len(spans):
            raise ValueError(f"Invalid JEREX mention token span [{start}, {end})")
        mention["char_start"] = spans[start][0]
        mention["char_end"] = spans[end - 1][1]


class CheckpointMetadata:
    """Passive replacement for a Lightning callback type stored as a dict key."""


def load_inference_state(model, state):
    """Keep the module's parameter and buffer representations on CPU."""
    import torch

    expected = model.state_dict()
    # Assignment can reuse mmap-backed tensors. A regular copy casts and
    # transfers into the model and preserves non-contiguous or tied buffers.
    can_assign = expected.keys() == state.keys()
    destinations, sources = set(), set()
    if can_assign:
        for name, target in expected.items():
            source = state[name]
            if not (isinstance(target, torch.Tensor) and isinstance(source, torch.Tensor)
                    and source.dtype == target.dtype and source.device == target.device
                    and source.shape == target.shape and source.layout == target.layout == torch.strided
                    and not source.is_quantized and source.stride() == target.stride()):
                can_assign = False
                break
            if target.numel():
                dest_storage = target.untyped_storage().data_ptr()
                source_storage = source.untyped_storage().data_ptr()
                if dest_storage in destinations or source_storage in sources:
                    can_assign = False
                    break
                destinations.add(dest_storage)
                sources.add(source_storage)
    model.load_state_dict(state, strict=True, assign=can_assign)


def predict(model_dir, documents, output, include_mention_scores=False, observer=None, relation_threshold=None,
            include_text_offsets=False):
    import torch
    from jerex import models
    from jerex.datasets import DocREDDataset
    from jerex.entities import EntityType, RelationType
    from jerex.sampling.sampling_common import collate_fn_padding
    from transformers import BertConfig, BertTokenizer

    if relation_threshold is not None and not 0 <= relation_threshold <= 1:
        raise ValueError("relation threshold must be a probability in [0, 1]")
    metadata = (CheckpointMetadata, "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint")
    with torch.serialization.safe_globals([metadata, EntityType, RelationType]):
        path = model_dir / "model.ckpt"
        checkpoint = torch.load(path, weights_only=True, map_location="cpu", mmap=is_zipfile(path))
    hp = checkpoint["hyper_parameters"]
    if relation_threshold is not None:
        hp = dict(hp, rel_threshold=relation_threshold)
    if hp["model_type"] != "joint_multi_instance":
        raise ValueError("Expected the joint_multi_instance checkpoint")
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=hp["lowercase"], local_files_only=True)
    config = BertConfig.from_pretrained(model_dir, local_files_only=True)
    parameters = {key: hp[key] for key in (
        "entity_types", "relation_types", "prop_drop", "size_embeddings_count", "meta_embedding_size",
        "ed_embeddings_count", "token_dist_embeddings_count", "sentence_dist_embeddings_count",
        "mention_threshold", "coref_threshold", "rel_threshold", "position_embeddings_count",
    )}
    cls = models.get_model(hp["model_type"])
    model = models.create_model(cls, config, tokenizer, **parameters)
    state = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items()}
    # Transformers 4.48 computes position IDs from a nonpersistent buffer.
    position_ids = state.pop("bert.embeddings.position_ids")
    if not torch.equal(position_ids, model.bert.embeddings.position_ids):
        raise ValueError("Checkpoint and encoder position indices differ")
    if "bert.embeddings.position_ids" in model.state_dict():
        state["bert.embeddings.position_ids"] = position_ids
    load_inference_state(model, state)
    del state, checkpoint
    # Keep the default token-type buffer aligned with JEREX's extended positions.
    model.bert.embeddings.register_buffer("token_type_ids", torch.zeros_like(position_ids), persistent=False)
    model.eval()
    if observer is not None:
        observer.attach(model)
    evaluator = cls.EVALUATOR(hp["entity_types"], hp["relation_types"], tokenizer)
    with TemporaryDirectory() as temporary:
        inputs = Path(temporary) / "input.json"
        inputs.write_text(json.dumps([
            {"title": doc["title"], "sents": doc["sents"], "vertexSet": []} for doc in documents
        ]), encoding="utf-8")
        dataset = DocREDDataset(str(inputs), hp["entity_types"], hp["relation_types"], tokenizer,
                                max_span_size=hp["max_span_size"])
    dataset.switch_mode(dataset.INFERENCE_MODE)
    dataset.switch_task(cls.TASK_TYPE)
    predictions = []
    mention_scores = []
    with torch.inference_mode():
        for index, parsed in enumerate(dataset.documents):
            # The upstream sampler allocates a wordpiece mask for each candidate span.
            encoded_length = len(parsed.encodings)
            if encoded_length > position_ids.shape[-1]:
                raise ValueError(f"{documents[index]['title']}: {encoded_length} wordpieces exceed "
                                 f"checkpoint capacity {position_ids.shape[-1]}")
            batch = collate_fn_padding([dataset[index]])
            result = model(**batch, inference=True, max_spans=32, max_coref_pairs=128, max_rel_pairs=64)
            if observer is not None:
                # Diagnostics see annotations only after the unmodified forward pass.
                observer.record(documents[index], batch, result, hp)
            predictions.extend(evaluator.convert_batch(**result, batch=batch))
            if include_mention_scores:
                # The joint model returns masked sigmoid probabilities at this stage.
                scores = result["mention_clf"][0]
                keep = scores > 0
                spans = batch["mention_orig_spans"][0][keep].tolist()
                mention_scores.append(dict(zip(map(tuple, spans), scores[keep].tolist(), strict=True)))
    with TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "predictions.json"
        evaluator.store_predictions(predictions, dataset.documents, str(staged))
        rows = json.loads(staged.read_text(encoding="utf-8"))
        for row, document in zip(rows, documents, strict=True):
            row["title"] = document["title"]
            if include_text_offsets:
                attach_text_offsets(row, document)
            if relation_threshold is not None:
                row["relation_threshold"] = relation_threshold
        if include_mention_scores:
            with (model_dir / "model.ckpt").open("rb") as stream:
                checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
            for row, scores in zip(rows, mention_scores, strict=True):
                row["checkpoint_sha256"] = checkpoint_sha
                for mention in row["mentions"]:
                    mention["confidence"] = scores.pop((mention["start"], mention["end"]))
                if scores:
                    raise ValueError("Upstream writer omitted a retained mention")
        staged.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        staged.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("documents", type=Path, help="Tokenized JSON array or one UTF-8 text file (- for text on stdin)")
    parser.add_argument("--input-format", choices=("tokenized", "text"), default="tokenized",
                        help="Default: JSON array of {title, sents}. Use text for one ordinary document.")
    parser.add_argument("--title", help="Title for text input (defaults to the filename, or stdin)")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--mention-scores", action="store_true", help="Include the trained mention detector probabilities")
    parser.add_argument("--relation-threshold", type=float, help="explicit sigmoid cutoff, defaults to the checkpoint setting")
    args = parser.parse_args()
    raw = args.input_format == "text"
    if args.title is not None and not raw:
        parser.error("--title is only used with --input-format text")
    if raw:
        stdin = args.documents == Path("-")
        content = sys.stdin.buffer.read() if stdin else args.documents.read_bytes()
        try:
            title = args.title if args.title is not None else ("stdin" if stdin else args.documents.name)
            documents = [text_document(content.decode("utf-8"), title)]
        except ValueError as error:
            parser.error(str(error))
    else:
        documents = json.loads(args.documents.read_text(encoding="utf-8"))
    predict(args.model_dir, documents, args.predictions, args.mention_scores,
            relation_threshold=args.relation_threshold, include_text_offsets=raw)


if __name__ == "__main__":
    main()
