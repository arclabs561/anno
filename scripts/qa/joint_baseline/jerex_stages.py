"""Verify source assets and run the frozen stage decomposition on the authors' test split."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
DOCRED = HERE.parents[2] / "crates/anno-eval/examples/docred"
SPLIT = "f7872af697e8a887efcf2f3381bb815e791e9488186dd17167894808c89385ed"
CHECKPOINT = {
    "model.ckpt": "55bf677bcdfa66973a4e22859051327dd63bfd7a249e5b3e34cd08b95681d4c2",
    "config.json": "b576f8e6dfed53d8bba577c1b6a7d405c02ac847ca4ec2820b8da533bc51286c",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer_config.json": "0f6d13e6f4da6f9e24f22ada6bc3be571123d858d7c0c05a8a7cd55a9c23c2e8",
    "vocab.txt": "eeaa9875b23b04b4c54ef759d03db9d1ba1554838f8fb26c5d96fa551df93d02",
}
REVISION = "85851dfcbb485d9bfa2bf662b8073e826c760d7f"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def canonical(row):
    spans = [(m["start"], m["end"]) for m in row["mentions"]]
    if len(set(spans)) != len(spans):
        raise ValueError("duplicate JEREX mention span")
    clusters = [(v["type"], tuple(sorted(spans[i] for i in v["mentions"]))) for v in row["entities"]]
    edges = sorted((clusters[e["head"]], e["type"], clusters[e["tail"]]) for e in row["relations"])
    return sorted(clusters), edges, dict(zip(spans, (m["confidence"] for m in row["mentions"]), strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("test_joint", type=Path)
    args = parser.parse_args()
    for filename, expected in CHECKPOINT.items():
        if digest(args.model_dir / filename) != expected:
            raise ValueError(f"unexpected checkpoint file {filename}")
    if digest(args.test_joint) != SPLIT:
        raise ValueError("unexpected DocRED joint test edition")
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", upstream, "diff", "--quiet", "HEAD", "--", "jerex"], check=True)
    if revision != REVISION:
        raise ValueError("unexpected upstream JEREX checkout")
    sample = json.loads(args.test_joint.read_bytes())[:3]
    authored, = json.loads((HERE / "jerex_input.json").read_bytes())
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(upstream), os.environ.get("PYTHONPATH", "")]),
               HF_HUB_OFFLINE="1")
    sys.path.insert(0, str(DOCRED))
    from score_jerex import convert, stage_counts

    with tempfile.TemporaryDirectory() as tmp:
        input_path, prediction_path = (Path(tmp) / name for name in ("input.json", "predictions.json"))
        input_path.write_text(json.dumps([*sample, authored]))
        subprocess.run([sys.executable, DOCRED / "predict_jerex.py", args.model_dir,
                        input_path, prediction_path, "--mention-scores"], check=True, env=env)
        predictions = json.loads(prediction_path.read_bytes())
    expected, actual = convert(sample, predictions[:-1])
    stages = stage_counts(expected, actual, predictions[:-1])
    reference = json.loads((HERE / "receipts/jerex.json").read_bytes())
    if stages != reference["stages"] or [d["title"] for d in sample] != reference["titles"]:
        raise ValueError(f"JEREX stage counts changed: {stages}")
    saved, = json.loads((HERE / "jerex_prediction.json").read_bytes())
    current = predictions[-1]
    if (current["tokens"], current["checkpoint_sha256"]) != (saved["tokens"], saved["checkpoint_sha256"]):
        raise ValueError("frozen projection source tokens or checkpoint changed")
    c, e, p = canonical(current)
    ec, ee, ep = canonical(saved)
    if (c, e, p.keys()) != (ec, ee, ep.keys()) or any(abs(p[s] - ep[s]) > 2e-5 for s in p):
        raise ValueError("frozen projection fixture differs from trained inference")
    print(json.dumps(stages, indent=2))


if __name__ == "__main__":
    main()
