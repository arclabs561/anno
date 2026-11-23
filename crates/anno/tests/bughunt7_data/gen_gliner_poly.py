"""Generate small split GLiNER graphs to test tokenizer, label and span transport."""

from pathlib import Path

import onnx
from onnx import TensorProto as T, helper as h

DEST = Path(__file__).parent / "gliner_poly_split"
DEST.mkdir(exist_ok=True)


def save(name, inputs, output, nodes, constants):
    graph = h.make_graph(nodes, name, inputs, [output], constants)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, DEST / name)


info = h.make_tensor_value_info
tensor = h.make_tensor
node = h.make_node

# Encode each label as its largest unpadded vocabulary ID.
save("label_encoder.onnx", [
    info("labels_input_ids", T.INT64, ["labels", "seq"]),
    info("labels_attention_mask", T.INT64, ["labels", "seq"]),
], info("labels_embeddings", T.FLOAT, ["labels", 1]), [
    node("Mul", ["labels_input_ids", "labels_attention_mask"], ["tokens"]),
    node("ReduceMax", ["tokens"], ["label_ids"], axes=[1], keepdims=1),
    node("Cast", ["label_ids"], ["labels_embeddings"], to=T.FLOAT),
], [])

# A pruned text graph scores the second and third words together if the
# corresponding label embedding exceeds six. Other spans have negative logits.
save("model.onnx", [info("labels_embeddings", T.FLOAT, ["labels", 1])],
     info("logits", T.FLOAT, [1, 3, 2, "labels"]), [
         node("Sub", ["labels_embeddings", "threshold"], ["centered"]),
         node("Transpose", ["centered"], ["per_label"], perm=[1, 0]),
         node("Where", ["span", "per_label", "low"], ["logits"]),
     ], [tensor("span", T.BOOL, [1, 3, 2, 1], [0, 0, 0, 1, 0, 0]),
         tensor("threshold", T.FLOAT, [1], [6.0]), tensor("low", T.FLOAT, [1], [-12.0])])
