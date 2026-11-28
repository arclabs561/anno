"""Line-oriented glinerel-fast worker, embedded in the Rust library."""
import json
import os
from pathlib import Path
import sys


def load(checkpoint):
    if os.getenv("ANNO_NO_DOWNLOADS", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
        os.environ["HF_HUB_OFFLINE"] = "1"

    from glinerel_fast import GLiNERel

    if checkpoint:
        path = Path(checkpoint).expanduser()
        if not path.is_dir():
            raise ValueError(f"ANNO_GLINEREL_MODEL_DIR does not exist: {path}")
        return GLiNERel(path, device="cpu", precision="fp32", threads=1)

    return GLiNERel.from_pretrained(
        "VAGOsolutions/SauerkrautLM-GLiNEREL-72m",
        revision="34c024bb2ea85326526f6c5d150b9afe8bbe4808",
        device="cpu", precision="fp32", threads=1,
    )


def main():
    try:
        model = load(sys.argv[1])
    except Exception as error:
        print(json.dumps({"error": str(error)}), flush=True)
        return 1
    print(json.dumps("ready"), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            result = model.extract(
                request["text"],
                entities=request["entities"],
                relations=request["relations"],
                entity_threshold=request["threshold"],
                relation_threshold=request["threshold"],
            )
            reply = {"ok": result}
        except Exception as error:
            reply = {"error": str(error)}
        print(json.dumps(reply, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
