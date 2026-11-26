"""Replay pinned native extractors on Linux. Each CLI invocation handles one document."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import tempfile
import time
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
ORT_SHA256 = "7d5242d7bfbb8b0a2c6ae4d4883adac34c883129c8091f74f3e1bc9f740b1d62"
SCORE_TOLERANCE = 0.001


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_assets(directory, spec, fetch):
    directory.mkdir(parents=True, exist_ok=True)
    for filename, item in spec["files"].items():
        target = directory / filename
        if target.exists() and digest(target) == item["sha256"]:
            continue
        if not fetch:
            raise ValueError(f"Missing or incorrect pinned artifact: {target}")
        url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{item['remote']}"
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as download:
            temporary = Path(download.name)
        try:
            with temporary.open("wb") as download, urlopen(url, timeout=180) as source:
                shutil.copyfileobj(source, download)
            if digest(temporary) != item["sha256"]:
                raise ValueError(f"Artifact hash mismatch for {url}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def check_case(binary, directory, spec, case):
    text = case["text"]
    args = [binary, "extract", "--quiet", "--format", "json", "--text", text,
            "--model", spec["backend"]]
    if case["labels"]:
        args += ["--extract-types", ",".join(case["labels"])]
    if "threshold" in case:
        args += ["--threshold", str(case["threshold"])]
    env = {**os.environ, spec["env"]: str(directory), "ANNO_NO_DOWNLOADS": "1",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    start = time.monotonic()
    result = json.loads(subprocess.check_output(args, env=env, timeout=180))
    elapsed = time.monotonic() - start
    producer = result["provenance"]
    if producer["model"].replace("_", "-") != spec["backend"] or producer["checkpoint"] != str(directory):
        raise ValueError(f"Unexpected producer for {spec['backend']}: {producer}")
    if producer["runtime_model"] != spec["runtime_model"]:
        raise ValueError(f"Wrong runtime for {spec['backend']}: {producer['runtime_model']}")
    found = []
    occurrence_ids = set()
    for ent in result["entities"]:
        a, b = ent["start"], ent["end"]
        if not (0 <= a < b <= len(text)) or text[a:b] != ent["text"]:
            raise ValueError(f"Invalid source span: {ent}")
        byte_range = {"start": len(text[:a].encode()), "end": len(text[:b].encode())}
        if ent["analyzed_byte_ranges"] != [byte_range]:
            raise ValueError(f"Wrong analyzed bytes for source span [{a}, {b})")
        if ent["id"] in occurrence_ids:
            raise ValueError(f"Duplicate occurrence ID at [{a}, {b}): {ent['id']}")
        occurrence_ids.add(ent["id"])
        found.append([a, b, ent["type"], ent["confidence"]])
    found.sort()
    expected = sorted(case["entities"])
    if [r[:3] for r in found] != [r[:3] for r in expected]:
        raise ValueError(f"{spec['backend']} on {text!r}: expected {expected}, observed {found}")
    for actual, ref in zip(found, expected):
        if len(ref) == 4 and not math.isclose(actual[3], ref[3], abs_tol=SCORE_TOLERANCE, rel_tol=0):
            raise ValueError(f"{spec['backend']} score outside tolerance: {actual}, reference {ref}")
    return {"source_sha256": hashlib.sha256(text.encode()).hexdigest(), "entities": found,
            "backend": producer["model"], "runtime_model": producer["runtime_model"],
            "elapsed_seconds_including_load": elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True, help="Model cache directory")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true", help="Download missing or incorrect pinned artifacts")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    linked = subprocess.check_output(["ldd", binary], text=True)
    ort = next((Path(row.split()[2]) for row in linked.splitlines()
                if row.strip().startswith("libonnxruntime.so") and "=>" in row), None)
    if not ort or not ort.is_file() or digest(ort) != ORT_SHA256:
        raise SystemExit("Expected the Linux x86_64 ONNX Runtime 1.24.2 library from the pinned Python wheel")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    manifest = HERE / "models.json"
    report = {"binary_sha256": digest(binary), "manifest_sha256": digest(manifest),
              "checkout_revision": subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True).strip(),
              "onnxruntime_version": "1.24.2", "onnxruntime_sha256": ORT_SHA256,
              "python_version": platform.python_version(), "platform": platform.platform(),
              "score_absolute_tolerance": SCORE_TOLERANCE, "cpus": sorted(os.sched_getaffinity(0)),
              "models": {}, "complete": False}
    try:
        for name, spec in json.loads(manifest.read_text()).items():
            directory = (args.models / name).absolute()
            record = {"repo": spec["repo"], "revision": spec["revision"], "cases": []}
            report["models"][name] = record
            verify_assets(directory, spec, args.fetch)
            record["verified_sha256"] = {k: v["sha256"] for k, v in spec["files"].items()}
            # Some loaders select a precision variant by filename. Prevent other
            # files in a populated user cache from influencing the replay.
            with tempfile.TemporaryDirectory(prefix="anno-verified-") as model_dir:
                verified = Path(model_dir)
                for filename in spec["files"]:
                    (verified / filename).symlink_to(directory / filename)
                for case in spec["cases"]:
                    record["cases"].append(check_case(binary, verified, spec, case))
        report["complete"] = True
    finally:
        report["maximum_child_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        args.receipt.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
