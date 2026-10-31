"""Generate tiny BERT-token-classification and GLiNER ONNX fixtures for tests/bughunt7.rs.

    uv run --no-project --python 3.12 --with onnx==1.23.2 --with tokenizers==0.23.3 \
        --with numpy python -I gen_ner.py <bughunt7_data dir>

Every model ignores its inputs and returns constant logits for one fixed sentence.
GLiNER fixtures are written as a Hugging Face cache (hf_home/hub/models--bh7--*/...)
because GLiNEROnnx only loads through hf-hub; the tests set HF_HOME to hf_home and
ANNO_NO_DOWNLOADS=1, so nothing is fetched.
"""

import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

ON = 10.0
OFF = -10.0
CONLL = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def const_model(inputs, arr, out_name="logits"):
    t = numpy_helper.from_array(arr.astype(np.float32), name="v")
    node = helper.make_node("Constant", [], [out_name], value=t)
    out = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, list(arr.shape))
    ins = [helper.make_tensor_value_info(n, ty, shp) for n, ty, shp in inputs]
    g = helper.make_graph([node], "const", ins, [out])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 14)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def bert_tokenizer(words):
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for w in words:
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=False, handle_chinese_chars=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return tok


def bert_fixture(root, name, words, text, tags):
    """tags: list of labels, one per token of the [CLS] ... [SEP] encoding."""
    tok = bert_tokenizer(words)
    enc = tok.encode(text)
    assert len(tags) == len(enc.ids), (enc.tokens, tags)
    L = len(enc.ids)
    logits = np.full((1, L, len(CONLL)), OFF, np.float32)
    for i, t in enumerate(tags):
        logits[0, i, CONLL.index(t)] = ON
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    inputs = [
        ("input_ids", TensorProto.INT64, [1, "L"]),
        ("attention_mask", TensorProto.INT64, [1, "L"]),
    ]
    onnx.save(const_model(inputs, logits), os.path.join(d, "model.onnx"))
    tok.save(os.path.join(d, "tokenizer.json"))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"id2label": {str(i): l for i, l in enumerate(CONLL)}}, f)
    print(name, enc.tokens, tags)


def gliner_tokenizer(words):
    vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "<<ENT>>": 4, "<<SEP>>": 5}
    for w in words:
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tok


def gliner_fixture(root, repo, words, logits, span_inputs, class_token_index=4):
    snap = os.path.join(root, "hf_home", "hub", "models--bh7--" + repo)
    d = os.path.join(snap, "snapshots", "fixture")
    os.makedirs(os.path.join(d, "onnx"), exist_ok=True)
    os.makedirs(os.path.join(snap, "refs"), exist_ok=True)
    with open(os.path.join(snap, "refs", "main"), "w") as f:
        f.write("fixture")
    inputs = [
        ("input_ids", TensorProto.INT64, [1, "S"]),
        ("attention_mask", TensorProto.INT64, [1, "S"]),
        ("words_mask", TensorProto.INT64, [1, "S"]),
        ("text_lengths", TensorProto.INT64, [1, 1]),
    ]
    if span_inputs:
        inputs += [
            ("span_idx", TensorProto.INT64, [1, "N", 2]),
            ("span_mask", TensorProto.BOOL, [1, "N"]),
        ]
    onnx.save(const_model(inputs, logits), os.path.join(d, "onnx", "model.onnx"))
    gliner_tokenizer(words).save(os.path.join(d, "tokenizer.json"))
    with open(os.path.join(d, "gliner_config.json"), "w") as f:
        json.dump({"class_token_index": class_token_index, "max_width": 12}, f)
    print(repo, logits.shape)


def main(root):
    # BERT: Chinese text, BertNormalizer splits every CJK ideograph into its own word.
    # "我在北京工作。" -> [CLS] 我 在 北 京 工 作 。 [SEP]; 北京 = B-LOC I-LOC.
    bert_fixture(
        root, "bert_cjk", list("我在北京工作。"), "我在北京工作。",
        ["O", "O", "O", "B-LOC", "I-LOC", "O", "O", "O", "O"],
    )
    words = ["Call", "John", "Smith", "now", "person"]
    # GLiNER span model: [1, num_words=4, max_width=12, num_classes=1]; "John Smith" =
    # start word 1, width index 1; a weaker nested "John" (start 1, width 0).
    span = np.full((1, 4, 12, 1), OFF, np.float32)
    span[0, 1, 1, 0] = 3.0
    span[0, 1, 0, 0] = 1.0
    gliner_fixture(root, "gliner-span", words, span, span_inputs=True)

    # GLiNER token-level model (span_mode "token_level", e.g. gliner-pii-edge): output
    # [1, num_words, num_classes, 3] with the last axis = (start, end, inside), as in
    # gliner.modeling.scorers.Scorer and gliner.decoding.decoder.TokenDecoder.
    tok = np.full((1, 4, 1, 3), OFF, np.float32)
    tok[0, 1, 0, 0] = ON  # start at "John"
    tok[0, 2, 0, 1] = ON  # end at "Smith"
    tok[0, 1, 0, 2] = ON  # inside
    tok[0, 2, 0, 2] = ON  # inside
    gliner_fixture(root, "gliner-token", words, tok, span_inputs=False)



if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
