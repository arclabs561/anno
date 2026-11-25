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


def run(checkpoint, schema_path):
    # Some ML libraries print progress to stdout; stdout is the JSONL stream.
    with redirect_stdout(sys.stderr):
        from gliner2 import __version__
        from gliner2.joint_ie import JointIE, JointSchema

        checkpoint = Path(checkpoint)
        config = json.loads((checkpoint / "config.json").read_bytes())
        if config.get("architecture") != "boundary":
            raise ValueError("JointIE expects a local boundary checkpoint")
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
            # A union of independently optimized windows can violate graph-wide
            # cardinality constraints. Only publish a graph covering the source.
            result, lattice = model.extract(text, schema, return_lattice=True)
        if text[max(lattice.end_mappings, default=0):].strip():
            raise ValueError(f"document {row}: checkpoint did not cover the full text")
        graph = result.to_dict(include_confidence=True, include_spans=True)
        for entity in result.entities:
            if not (0 <= entity.start < entity.end <= len(text)) or text[entity.start:entity.end] != entity.text:
                raise ValueError(f"document {row}: entity {entity.id} has invalid source offsets")
        graph.update(
            feasible=result.feasible,
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
