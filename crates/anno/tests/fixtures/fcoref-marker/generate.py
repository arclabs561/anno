"""Generate the four-token model used by the segment-boundary regression."""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors.numpy import save_file
from tokenizers import Tokenizer, models, pre_tokenizers, processors

out = Path(__file__).resolve().parent
vocab = {"x": 0, "<s>": 1, "</s>": 2, "Zoë": 3}
tok = Tokenizer(models.WordLevel(vocab, unk_token="x"))
tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
tok.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
)
tok.save(str(out / "tokenizer.json"))

# Only Zoë activates the first hidden dimension.
embedding = np.array([[0, 1], [0, 1], [0, 1], [1, 0]], dtype=np.float32)
inputs = [helper.make_tensor_value_info(name, TensorProto.INT64, [1, "tokens"])
          for name in ("input_ids", "attention_mask")]
output = helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, [1, "tokens", 2])
gather = helper.make_node("Gather", ["embedding", "input_ids"], ["last_hidden_state"])
graph = helper.make_graph([gather], "marker", inputs, [output],
                          [numpy_helper.from_array(embedding, "embedding")])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
onnx.checker.check_model(model)
onnx.save(model, out / "encoder.onnx")

# Mention scores alone link the three marked positions. Pair weights are zero.
weights = {}
for layer in ("start_mention_mlp", "end_mention_mlp", "start_coref_mlp", "end_coref_mlp"):
    weights[layer + ".dense.weight"] = np.eye(2, dtype=np.float32)
    weights[layer + ".dense.bias"] = np.zeros(2, dtype=np.float32)
    weights[layer + ".layer_norm.weight"] = np.ones(2, dtype=np.float32)
    weights[layer + ".layer_norm.bias"] = np.zeros(2, dtype=np.float32)
for end in ("start", "end"):
    weights[f"mention_{end}_classifier.weight"] = np.array([[1, 0]], dtype=np.float32)
for layer in ("mention_s2e", "antecedent_s2s", "antecedent_e2e", "antecedent_s2e", "antecedent_e2s"):
    weights[layer + "_classifier.weight"] = np.zeros((2, 2), dtype=np.float32)
save_file(weights, out / "scorer_weights.safetensors")
