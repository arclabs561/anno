"""Run the upstream JEREX joint model with a local checkpoint on CPU.

Give Python access to the lavis-nlp/jerex checkout (e.g. via PYTHONPATH).
Input is a DocRED-format array. Only titles and sentence tokens enter inference.
The published DocRED joint checkpoint supplies the task schema and thresholds.
Output uses its joint-mode predictions.json schema, accepted by score_jerex.py.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from jerex import models
from jerex.datasets import DocREDDataset
from jerex.entities import EntityType, RelationType
from jerex.sampling.sampling_common import collate_fn_padding
from transformers import BertConfig, BertTokenizer


class CheckpointMetadata:
    """Passive replacement for a Lightning callback type stored as a dict key."""


def predict(model_dir, documents, output):
    metadata = (CheckpointMetadata, "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint")
    with torch.serialization.safe_globals([metadata, EntityType, RelationType]):
        checkpoint = torch.load(model_dir / "model.ckpt", weights_only=True, map_location="cpu")
    hp = checkpoint["hyper_parameters"]
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
    model.load_state_dict(state, strict=True)
    # Keep the default token-type buffer aligned with JEREX's extended positions.
    model.bert.embeddings.register_buffer("token_type_ids", torch.zeros_like(position_ids), persistent=False)
    model.eval()
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
    with torch.inference_mode():
        for index in range(len(dataset)):
            batch = collate_fn_padding([dataset[index]])
            encoded_length = batch["encodings"].shape[-1]
            if encoded_length > position_ids.shape[-1]:
                raise ValueError(f"{documents[index]['title']}: {encoded_length} wordpieces exceed "
                                 f"checkpoint capacity {position_ids.shape[-1]}")
            result = model(**batch, inference=True, max_spans=32, max_coref_pairs=128, max_rel_pairs=64)
            predictions.extend(evaluator.convert_batch(**result, batch=batch))
    with TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "predictions.json"
        evaluator.store_predictions(predictions, dataset.documents, str(staged))
        rows = json.loads(staged.read_text(encoding="utf-8"))
        for row, document in zip(rows, documents, strict=True):
            row["title"] = document["title"]
        staged.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        staged.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("documents", type=Path)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    predict(args.model_dir, json.loads(args.documents.read_text(encoding="utf-8")), args.predictions)


if __name__ == "__main__":
    main()
