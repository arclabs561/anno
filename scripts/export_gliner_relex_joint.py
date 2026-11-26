"""Batch-one GLiNER-Relex export using the trained span and directed pair heads.

The original Python pair builder converts candidate counts to Python integers.
Build the same ordered, off-diagonal pairs with tensor operations so the trace
also works for documents with different numbers of mentions (including zero).
"""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn


class JointRelex(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
        cfg = core.config
        if not (
            hasattr(core, "pair_rep_layer")
            and cfg.relations_layer == "none"
            and cfg.embed_ent_token
            and cfg.embed_rel_token
            and cfg.subtoken_pooling == "first"
            and cfg.words_splitter_type == "whitespace"
            and cfg.max_width == 12
        ):
            raise ValueError(
                "Unsupported Relex architecture or tokenizer configuration"
            )
        self.entity_temperature = getattr(cfg, "logit_temperature", 1.0) or 1.0
        self.entity_bias = getattr(cfg, "logit_bias", 0.0) or 0.0
        self.relation_temperature = getattr(cfg, "rel_logit_temperature", 1.0) or 1.0
        self.relation_bias = getattr(cfg, "rel_logit_bias", 0.0) or 0.0

    def forward(
        self,
        input_ids,
        word_positions,
        entity_positions,
        relation_positions,
        span_idx,
        span_mask,
        threshold,
    ):
        core = self.core
        states = core.token_rep_layer(input_ids, torch.ones_like(input_ids))[0]
        words = states.index_select(0, word_positions).unsqueeze(0)
        if hasattr(core, "rnn"):
            # No batch padding: running the underlying LSTM yields precisely the
            # same states as pack_padded_sequence on this one full-length item.
            words, _ = core.rnn.lstm(words)
        spans = core.span_rep_layer(words, span_idx)
        prompts = core.prompt_rep_layer(states.index_select(0, entity_positions))
        logits = (spans[0] @ prompts.T) / self.entity_temperature + self.entity_bias
        selected = torch.where(
            (logits.amax(-1).reshape(-1).sigmoid() > threshold) & span_mask
        )[0]
        row, col = torch.where(selected[:, None] != selected[None, :])
        # A sentinel gives the projection a nonempty batch for zero actual edges.
        sentinel = span_idx.new_zeros(1)
        head = torch.cat((selected[row], sentinel))
        tail = torch.cat((selected[col], sentinel))
        flat = spans.flatten(0, 2)
        pair = core.pair_rep_layer(torch.cat((flat[head], flat[tail]), dim=-1))
        rel = states.index_select(0, relation_positions)
        scores = (
            (pair @ rel.T) / self.relation_temperature + self.relation_bias
        ).sigmoid()
        return logits.unsqueeze(0), torch.stack((head, tail), dim=-1)[:-1], scores[:-1]


INPUTS = [
    "input_ids",
    "word_positions",
    "entity_positions",
    "relation_positions",
    "span_idx",
    "span_mask",
    "threshold",
]


def prepare(model, text, entities, relations, threshold):
    """Use the upstream processor for a real example and retain reference predictions."""
    captured = {}

    def capture(_module, _args, kwargs):
        captured.update(kwargs)

    handle = model.model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        prediction = model.predict_relations(
            text, entities, relations, threshold=threshold
        )
    finally:
        handle.remove()
    ids = captured["input_ids"]
    return (
        ids,
        torch.where(captured["words_mask"][0] > 0)[0],
        torch.where(ids[0] == model.config.class_token_index)[0],
        torch.where(ids[0] == model.config.rel_token_index)[0],
        captured["span_idx"] * captured["span_mask"].unsqueeze(-1),
        captured["span_mask"].reshape(-1).bool(),
        torch.tensor(threshold, dtype=torch.float32),
    ), prediction


REFERENCE_FILE = Path(__file__).resolve().parents[1] / "testdata/grounded/gliner_relex_reference.json"


def source_identity(model_id, revision):
    """Describe the weights and library versions used by the reference producer."""
    from huggingface_hub import try_to_load_from_cache

    root = Path(model_id)
    if not root.is_dir():
        cached = try_to_load_from_cache(model_id, "model.safetensors", revision=revision)
        if not isinstance(cached, str):
            raise ValueError(f"Cannot identify source checkpoint for {model_id} at {revision}")
        root = Path(cached).parent
        revision = root.name
    checkpoint = root / "model.safetensors"
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return {
        "checkpoint": str(model_id), "revision": revision,
        "checkpoint_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("torch", "gliner", "transformers", "onnx", "onnxruntime")},
    }


def export_joint(model, output: Path):
    model.eval()
    wrapper = JointRelex(model.model).eval()
    fixture = json.loads(REFERENCE_FILE.read_text())
    cases = [(c["text"], c["entity_labels"], c["relation_labels"], c["threshold"])
             for c in fixture["cases"]]
    with torch.inference_mode():
        prepared = [prepare(model, *case) for case in cases]
        sample = prepared[0][0]
        graph = output / "model.onnx"
        torch.onnx.export(
            wrapper,
            sample,
            str(graph),
            input_names=INPUTS,
            output_names=["logits", "pair_slots", "relation_scores"],
            dynamic_axes={
                "input_ids": {1: "sequence"},
                "word_positions": {0: "words"},
                "entity_positions": {0: "entity_types"},
                "relation_positions": {0: "relations"},
                "span_idx": {1: "slots"},
                "span_mask": {0: "slots"},
                "logits": {1: "words", 3: "entity_types"},
                "pair_slots": {0: "pairs"},
                "relation_scores": {0: "pairs", 1: "relations"},
            },
            opset_version=17,
            dynamo=False,
        )
        references = [
            reference_output(model, wrapper, case, sample)
            for case, sample in zip(cases, prepared)
        ]
    return references


def reference_output(model, wrapper, case, prepared):
    inputs, prediction = prepared
    feed = dict(zip(INPUTS, (v.numpy() for v in inputs)))
    expected = [v.numpy() for v in wrapper(*inputs)]
    words = list(model.data_processor.words_splitter(case[0]))
    return case, feed, expected, words, prediction


def verify_joint(graph, references):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(graph), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    saw_edge = False
    for case, feed, expected, words, (entities, relations) in references:
        logits, pairs, scores = session.run(None, feed)
        for actual, reference in zip((logits, pairs, scores), expected):
            np.testing.assert_allclose(actual, reference, atol=2e-4, rtol=1e-4)
        # Check the actual trained relation output against Python by source span.
        text, entity_labels, relation_labels, threshold = case
        slots = feed["span_idx"][0]

        def char_span(slot):
            begin, end = slots[slot]
            return words[begin][1], words[end][2]

        # The Python decoder sorts all positive (span, label) candidates by
        # score and greedily suppresses overlapping spans for flat NER.
        probs = 1 / (1 + np.exp(-np.clip(logits[0], -100, 100).astype(np.float64)))
        candidates = np.argwhere(probs > threshold)
        occupied = set()
        decoded = {}
        for start, width, cls in sorted(candidates, key=lambda c: -probs[tuple(c)]):
            indices = range(start, start + width + 1)
            if start + width >= len(words) or occupied.intersection(indices):
                continue
            occupied.update(indices)
            span = (words[start][1], words[start + width][2])
            decoded[span] = (entity_labels[cls], probs[start, width, cls])
        reference_entities = {
            (e["start"], e["end"]): (e["label"], e["score"]) for e in entities
        }
        if {s: v[0] for s, v in decoded.items()} != {
            s: v[0] for s, v in reference_entities.items()
        }:
            raise ValueError(f"Entity mismatch: {decoded} vs {reference_entities}")
        for span, (_, score) in decoded.items():
            np.testing.assert_allclose(
                score, reference_entities[span][1], atol=2e-4
            )
        kept = decoded.keys()
        actual_edges = {
            (char_span(h), label, char_span(t)): float(score)
            for (h, t), row in zip(pairs, scores)
            for label, score in zip(relation_labels, row)
            if score > threshold and char_span(h) in kept and char_span(t) in kept
        }
        expected_edges = {
            (
                (r["head"]["start"], r["head"]["end"]),
                r["relation"],
                (r["tail"]["start"], r["tail"]["end"]),
            ): r["score"]
            for r in relations
        }
        if actual_edges.keys() != expected_edges.keys():
            raise ValueError(
                f"Directed edge mismatch: {actual_edges} vs {expected_edges}"
            )
        for key in actual_edges:
            np.testing.assert_allclose(
                actual_edges[key], expected_edges[key], atol=2e-4
            )
        saw_edge |= bool(expected_edges)
    if not saw_edge:
        raise ValueError(
            "Reference checkpoint produced no positive edge on export validation cases"
        )
