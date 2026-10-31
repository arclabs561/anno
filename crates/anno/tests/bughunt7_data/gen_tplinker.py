"""Generate tiny TPLinker ONNX fixtures for tests/bughunt7.rs.

Run (from this directory's parent or anywhere):
    uv run --no-project --python 3.12 --with onnx==1.23.2 --with tokenizers==0.23.3 \
        --with numpy python -I gen_tplinker.py <out_dir>

Each fixture directory holds model.onnx, tokenizer.json and tplinker_config.json in
the layout `TPLinkerOnnx::from_local` reads:
  inputs  input_ids [1, L] int64, attention_mask [1, L] int64
  outputs ent_logits [1, hs, 5], head_rel_logits [1, hs, R*3], tail_rel_logits [1, hs, R*3]
with hs = L*(L+1)/2 and flat index idx(i, j) = i*L - i*(i-1)/2 + (j-i) for i <= j.
The model ignores its inputs and returns constant logits that encode one handcrafted
analysis of one fixed sentence (the test passes exactly that sentence).
Logit convention: NONE = 0, a set tag = 10 (softmax prob ~= 1).
"""

import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

ENT_TAGS = ["NONE", "SH2OH", "OH2ST", "ST2OT", "OT2ST"]
REL_TAGS = ["NONE", "SH2OH", "OH2ST"]
ON = 10.0


def hs_index(i, j, L):
    assert i <= j
    return i * L - i * (i - 1) // 2 + (j - i)


def wordpiece_tokenizer(words):
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for w in words:
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tok.decoder = decoders.WordPiece()
    return tok


def char_tokenizer(chars):
    """One token per character, no special tokens: token index == char index."""
    vocab = {"[UNK]": 0}
    for c in chars:
        vocab.setdefault(c, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    return tok


def constant_model(ent, head, tail):
    nodes = []
    outs = []
    for name, arr in (("ent_logits", ent), ("head_rel_logits", head), ("tail_rel_logits", tail)):
        t = numpy_helper.from_array(arr.astype(np.float32), name=name + "_value")
        nodes.append(helper.make_node("Constant", [], [name], value=t))
        outs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(arr.shape)))
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "L"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, "L"]),
    ]
    graph = helper.make_graph(nodes, "tplinker_const", inputs, outs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def write_fixture(out_dir, name, tok, text, relations, ent_tags, head_tags, tail_tags):
    """ent_tags: {(i, j): tag}; head_tags/tail_tags: {(rel_idx, i, j): tag} over token positions."""
    enc = tok.encode(text)
    L = len(enc.ids)
    hs = L * (L + 1) // 2
    R = len(relations)
    ent = np.zeros((1, hs, 5), np.float32)
    head = np.zeros((1, hs, R * 3), np.float32)
    tail = np.zeros((1, hs, R * 3), np.float32)
    for (i, j), tag in ent_tags.items():
        ent[0, hs_index(i, j, L), ENT_TAGS.index(tag)] = ON
    for (r, i, j), tag in head_tags.items():
        head[0, hs_index(i, j, L), r * 3 + REL_TAGS.index(tag)] = ON
    for (r, i, j), tag in tail_tags.items():
        tail[0, hs_index(i, j, L), r * 3 + REL_TAGS.index(tag)] = ON
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    onnx.save(constant_model(ent, head, tail), os.path.join(d, "model.onnx"))
    tok.save(os.path.join(d, "tokenizer.json"))
    cfg = {
        "dataset": "bughunt7",
        "encoder": "none",
        "num_entity_tags": 5,
        "num_relation_types": R,
        "hidden_size": 4,
        "shaking_type": "cat",
        "entity_tags": ENT_TAGS,
        "relation_tags": REL_TAGS,
        "relations": relations,
    }
    with open(os.path.join(d, "tplinker_config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(d, "fixture.json"), "w") as f:
        json.dump(
            {
                "text": text,
                "tokens": enc.tokens,
                "offsets": enc.offsets,
                "ent_tags": {f"{i},{j}": t for (i, j), t in ent_tags.items()},
                "head_tags": {f"{r},{i},{j}": t for (r, i, j), t in head_tags.items()},
                "tail_tags": {f"{r},{i},{j}": t for (r, i, j), t in tail_tags.items()},
            },
            f,
            indent=1,
        )
    print(name, L, enc.tokens)


def main(out_dir):
    wp = wordpiece_tokenizer(["alice", "met", "bob", "a", "b", "in", "paris"])

    # (a1) token positions != char offsets. [CLS] alice met bob [SEP]: alice = tok 1 (chars 0..5),
    # bob = tok 3 (chars 10..13). Relation "met" at token pair (1, 3), subject alice -> object bob.
    write_fixture(
        out_dir, "tplinker_tokpos", wp, "alice met bob", ["met"],
        ent_tags={(1, 1): "SH2OH", (3, 3): "SH2OH"},
        head_tags={(0, 1, 3): "SH2OH"},
        tail_tags={(0, 1, 3): "SH2OH"},
    )

    # (a2) "a b": [CLS] a b [SEP]; a = tok 1 (char 0), b = tok 2 (char 2). The only relation logit
    # is at token pair (0, 2) = ([CLS], b), which is meaningless; the true pair (1, 2) is NONE.
    write_fixture(
        out_dir, "tplinker_cls_cell", wp, "a b", ["rel"],
        ent_tags={(1, 1): "SH2OH", (2, 2): "SH2OH"},
        head_tags={(0, 0, 2): "SH2OH"},
        tail_tags={(0, 0, 2): "SH2OH"},
    )



if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
