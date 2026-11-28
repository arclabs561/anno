"""Prepare and validate the pinned CPU WikiBELA lookup on a Linux runner."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

RELEASE = "sntcristian/WikiBELA"
REV = "82b71b08c596b821e4a44c29a23d1c5fecd8165c"
FILES = {
    "faiss.index": (19765027245, "17e71167c992ec4ecc6e7747b6d78a3efa3d32d2a89856b2fc0b465cbc3e4575"),
    "index.txt": (158216234, "65403dd2c236629cba026156acc8fb31c1907525824d512a90679d16ea1fe124"),
    "knowledge_base.sqlite": (3159400448, "af09701a1f4c7b78c92a87b2c55747cf4c097dbde4bbd15f351a2a6a8f043068"),
    "model_wiki.ckpt": (2243434498, "b13ec79092a6a60db30e3e366aa1e0dc55ecc85a02ab4427ef4d163847a7dd26"),
}
XLM_FILES = {
    "config.json": (616, "8e5fb14e1352fd8fc678a7b293b63cfb5cf091f6"),
    "model.safetensors": (2244817354, "2dfa19f172412917cab174da04b46e2134811b723666965fd0aabd97caa6e23b"),
    "sentencepiece.bpe.model": (5069051, "db9af13bf09fd3028ca32be90d3fb66d5e470399"),
    "tokenizer.json": (9096718, "463f3414782c1c9405828c9b31bfa36dda1f45c5"),
    "tokenizer_config.json": (25, "34ddbd64a4cd3f2d9d8a9120d3662d0bf91baead"),
}
XLM_REV = "c23d21b0620b635a76227c604d44e43a9f0ee389"
GIB = 1024 ** 3


def budget(work):
    # The pinned retriever calls faiss.read_index. The complete index remains
    # allocated alongside the query encoder and Python objects during lookup.
    info = {line.split(":")[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines() if line.endswith("kB")}
    ram, swap = info["MemAvailable"], info["SwapFree"]
    cg = Path("/sys/fs/cgroup")
    for label, limit, current in [("ram", "memory.max", "memory.current"),
                                  ("swap", "memory.swap.max", "memory.swap.current")]:
        cf = cg / limit
        if cf.is_file() and cf.read_text().strip() != "max":
            cap = int(cf.read_text()) - int((cg / current).read_text())
            if label == "ram":
                # File cache is reclaimable. MemAvailable already accounts for it.
                cached = int(re.search(r"^inactive_file (\d+)$", (cg / "memory.stat").read_text(), re.M)[1])
                ram = min(ram, max(0, cap + cached))
            else:
                swap = min(swap, max(0, cap))
    missing_swap = math.ceil(max(0, 28 * GIB - ram - swap) / GIB)
    disk = shutil.disk_usage(work).free
    required = sum(size for size, _ in [*FILES.values(), *XLM_FILES.values()]) + 2 * GIB
    print(f"Available RAM {ram:,} bytes, swap {swap:,} bytes, disk {disk:,} bytes. "
          f"Need {required:,} bytes free for models and working space plus {missing_swap} GiB new swap.",
          file=sys.stderr)
    if disk < required + missing_swap * GIB:
        raise SystemExit("insufficient disk for this full release and its RAM shortfall")
    return missing_swap


def download(root, repo, revision, files):
    root.mkdir(exist_ok=True)
    for name, (size, expected) in files.items():
        path = root / name
        if not path.exists():
            partial = root / (name + ".partial")
            subprocess.run(["curl", "-fLsS", "--retry", "4", "-C", "-", "-o", str(partial),
                            f"https://huggingface.co/{repo}/resolve/{revision}/{name}"], check=True)
            check_file(partial, size, expected)
            partial.rename(path)
        else:
            check_file(path, size, expected)


def check_file(path, size, expected):
    if path.stat().st_size != size:
        raise ValueError(f"wrong size for {path}")
    # HF publishes SHA-256 for LFS and the Git blob ID for the small XLM assets.
    hasher = hashlib.sha256() if len(expected) == 64 else hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 << 20), b""):
            hasher.update(chunk)
    if hasher.hexdigest() != expected:
        raise ValueError(f"checksum mismatch for {path}")


def run_lookup(work, anno):
    model = work / "wikibela"
    download(model, RELEASE, REV, FILES)
    download(work / "xlm-roberta-large", "FacebookAI/xlm-roberta-large", XLM_REV, XLM_FILES)
    with (model / "faiss.index").open("rb") as file:
        header = file.read(45)
    if struct.unpack_from("<4sIQ", header) != (b"IxFI", 300, 16470856):
        raise ValueError("unexpected released FAISS index header")
    source = {"id": 14, "language": "en", "text": "📰 In Paris, the French government met.",
              "entities": [{"id": "city", "type": "LOC", "mentions": [{"start": 5, "end": 10}]}]}
    sourcefile = work / "source.json"
    sourcefile.write_text(json.dumps([source]))
    command = [anno, "mhel", "retrieve", "--python", sys.executable, "--upstream", work / "mhel",
               "--model-dir", model, "--documents", sourcefile, "--device", "cpu", "--top-k", "5"]
    # The pinned upstream encoder asks for xlm-roberta-large by relative name.
    output = subprocess.check_output(command, cwd=work, text=True)
    row, = json.loads(output)
    if (row["doc_id"], row["entity_id"], row["text"], row["start"], row["end"]) != (14, "city", "Paris", 5, 10):
        raise ValueError("retriever lost source identity or character coordinates")
    candidates = row["candidates"]
    if len(candidates) != 5 or any(not re.fullmatch(r"Q[1-9][0-9]*", c["wb_id"])
                                   or not math.isfinite(c["score"]) for c in candidates):
        raise ValueError("unexpected candidate count, identifier or score")
    report = {"wiki_revision": REV, "xlm_revision": XLM_REV, "index_vectors": 16470856,
              "verified_asset_bytes": sum(n for n, _ in FILES.values()), "result": row}
    (work / "receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=["budget", "lookup"])
    p.add_argument("work", type=Path)
    p.add_argument("--anno", default="anno")
    args = p.parse_args()
    if args.operation == "budget":
        print(budget(args.work))
    else:
        run_lookup(args.work.resolve(), args.anno)
