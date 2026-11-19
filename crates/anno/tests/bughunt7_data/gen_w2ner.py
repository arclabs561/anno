"""Generate tiny W2NER ONNX fixtures for tests/bughunt7.rs.

    uv run --no-project --python 3.12 --with onnx==1.23.2 --with tokenizers==0.23.3 \
        --with numpy python -I gen_w2ner.py <bughunt7_data dir>

Layout follows scripts/export_w2ner_to_onnx.py: inputs input_ids/attention_mask [1, T]
int64, one output grid_logits [1, T, T, num_labels] of raw logits over tokenizer
positions. Labels here are the 3 anno assumes (0 None, 1 NNW, 2 THW); the export
script's default is 7. THW is stored at (row = tail, col = head), NNW at (row = i, col = i+1).
"""

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

ON, OFF = 10.0, -10.0


def model(grid):
    t = numpy_helper.from_array(grid.astype(np.float32), name="v")
    node = helper.make_node("Constant", [], ["grid_logits"], value=t)
    ins = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "T"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, "T"]),
    ]
    out = helper.make_tensor_value_info("grid_logits", TensorProto.FLOAT, list(grid.shape))
    m = helper.make_model(helper.make_graph([node], "w2ner", ins, [out]),
                          opset_imports=[helper.make_opsetid("", 14)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def grid_none(T):
    g = np.full((1, T, T, 3), OFF, np.float32)
    g[..., 0] = ON
    return g


def setcell(g, i, j, rel, val=ON):
    g[0, i, j, :] = OFF
    g[0, i, j, 0] = -val if rel != 0 else val
    g[0, i, j, rel] = val


def save(root, name, tok, grid, text):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    onnx.save(model(grid), os.path.join(d, "model.onnx"))
    tok.save(os.path.join(d, "tokenizer.json"))
    print(name, tok.encode(text).tokens, grid.shape)


def main(root):
    text = "John Smith works"
    # (1) BERT-style tokenizer: [CLS] john smith works [SEP]; token t = word t-1.
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "john": 4, "smith": 5, "works": 6}
    wp = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    wp.normalizer = normalizers.BertNormalizer(lowercase=True)
    wp.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    wp.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    g = grid_none(5)
    setcell(g, 1, 2, 1)  # NNW john -> smith
    setcell(g, 2, 1, 2)  # THW tail smith, head john
    save(root, "w2ner_tokgrid", wp, g, text)

    # (2) word-level tokenizer, no special tokens: token index == word index.
    wl = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "John": 1, "Smith": 2, "works": 3},
                                    unk_token="[UNK]"))
    wl.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    g = grid_none(3)
    setcell(g, 0, 1, 1)  # NNW John -> Smith
    setcell(g, 1, 0, 2)  # THW tail Smith, head John  (p ~ 1)
    # (2, 2): argmax is None (logit 10) but the THW logit is 2.0 (softmax p ~ 3e-4).
    g[0, 2, 2, :] = [ON, OFF, 2.0]
    save(root, "w2ner_wordgrid", wl, g, text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
