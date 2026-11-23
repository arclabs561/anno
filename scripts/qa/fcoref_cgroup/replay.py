"""Run pinned word-budget fixtures in isolated Linux Docker containers."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import tempfile

MODEL_SHA = {
    "config.json": "12361347b196a78de4fad99045cd127c19baa55590fc7885f84a89a7a785bfbf",
    "encoder.onnx": "74b98b0f03b27154065d0918a566ed5968d14665e601ecaea6905014bfe6f065",
    "scorer_weights.safetensors": "54c3e13017825750b725dbb682db861c3b7933db6ec1306ff5f51d46c533d0e7",
    "tokenizer.json": "5ec67f22fb423648c32c22819353703e5d3dddd3a553b11989f37702fb910aa1",
}
CASES = [(511, 521), (512, 521), (513, 521), (520, 521),
         (521, 521), (522, 521), (521, 520)]


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


def require_controls(info):
    required = ["CpuCfsQuota", "CPUSet", "MemoryLimit", "SwapLimit"]
    if str(info.get("CgroupVersion")) != "2" or not all(info.get(k) is True for k in required):
        raise RuntimeError("Docker must support cgroup v2 CPU quota, cpuset, memory and swap limits")


def run_case(args, image, threads, tokens, budget):
    mounts = {args.binary: "/measure", args.model: "/weights",
              Path(__file__).with_name("guard.sh"): "/guard.sh", args.ort: "/ort.so"}
    options = []
    for src, dst in mounts.items():
        options += ["--mount", f"type=bind,source={src.resolve()},target={dst},readonly"]
    for key, value in {"EXPECTED_CPU": args.cpu, "ORT_DYLIB_PATH": "/ort.so",
                       "ANNO_NO_DOWNLOADS": 1, "MALLOC_ARENA_MAX": 2, "RAYON_NUM_THREADS": 1,
                       "OMP_NUM_THREADS": 1, "OPENBLAS_NUM_THREADS": 1,
                       "TOKENIZERS_PARALLELISM": "false"}.items():
        options += ["--env", f"{key}={value}"]
    container = docker("create", "--pull=never", "--network=none", "--read-only",
                       "--cap-drop=ALL", "--security-opt=no-new-privileges", "--cgroupns=private",
                       "--cpus=1", f"--cpuset-cpus={args.cpu}", "--memory=2g", "--memory-swap=2g",
                       "--user", f"{os.getuid()}:{os.getgid()}", *options, "--entrypoint", "/bin/sh",
                       image, "/guard.sh", str(threads), str(tokens), str(budget))
    row = {"threads": threads, "tokens": tokens, "budget": budget, "timed_out": False}
    try:
        started = time.monotonic()
        docker("start", container)
        try:
            subprocess.run(["docker", "wait", container], stdout=subprocess.DEVNULL,
                           check=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            row["timed_out"] = True
            docker("kill", container)
        row["wall_seconds"] = time.monotonic() - started
        row["state"] = json.loads(docker("inspect", "--format", "{{json .State}}", container))
        logs = subprocess.run(["docker", "logs", container], capture_output=True, text=True, check=True)
        row["stderr"] = logs.stderr
        if row["state"]["ExitCode"] == 0 and not row["timed_out"]:
            row["measurement"] = json.loads(logs.stdout)
        else:
            row["stdout"] = logs.stdout
        return row
    finally:
        docker("rm", "--force", container)


def verify(row, cpu):
    if row["timed_out"] or row["state"]["OOMKilled"] or row["state"]["ExitCode"] != 0:
        raise RuntimeError("container did not complete successfully")
    m = row["measurement"]
    cg = m["cgroup"]
    quota, period = cg["cpu.max"].split()
    if not (int(quota) == int(period) > 0 and cg["memory.max"].strip() == "2147483648"
            and cg["memory.swap.max"].strip() == "0"
            and cg["cpuset.cpus.effective"].strip() == str(cpu)
            and set(m["thread_affinities"]) == {str(cpu)}):
        raise RuntimeError("unexpected resource controls reported by the measured process")
    if (m["encoded_tokens"], m["budget"], m["threads_configured"]) != (
            row["tokens"], row["budget"], row["threads"]):
        raise RuntimeError("benchmark output belongs to a different case")
    events = dict(line.split() for line in cg["memory.events"].splitlines())
    if int(events["oom_kill"]) or int(events["oom"]):
        raise RuntimeError("kernel recorded a cgroup OOM")
    if len(m["warm_ms"]) != 5:
        raise RuntimeError("expected five warm calls")
    durations = [row["wall_seconds"], m["load_ms"], m["first_call_ms"], *m["warm_ms"]]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in durations):
        raise RuntimeError("expected finite positive timings")

    tokens, budget = row["tokens"], row["budget"]
    suffix = "Alice reviewed the proposal. She approved it. Bob objected. She overruled him."
    text = "x " * (tokens - 19) + suffix
    if m["text"] != text:
        raise RuntimeError("unexpected fixture text for this token count")
    predictions = m["predictions"]
    if tokens > budget:
        expected = f"Invalid input: coreference document has {tokens} tokens, exceeding max_doc_tokens {budget}"
        if predictions != {"error": expected}:
            raise RuntimeError("expected the document-budget rejection for this case")
        return

    if "error" in predictions or not predictions.get("mentions"):
        raise RuntimeError("expected successful predictions for this budget")
    mentions = predictions["mentions"]
    for start, end, surface, score in mentions:
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or text[start:end] != surface or type(score) not in (int, float) or not math.isfinite(score)):
            raise RuntimeError("unexpected mention span or score")

    # Only the repeated ASCII prefix changes; the discourse and its three
    # expected source-grounded clusters are the same at each accepted length.
    shift = 2 * (tokens - 521)
    canonical_clusters = [[(1004,1009),(1033,1036),(1064,1067)],
                          [(1019,1031),(1046,1048)], [(1050,1053),(1078,1081)]]
    expected = [[[start + shift, end + shift] for start, end in cluster]
                for cluster in canonical_clusters]
    mention_spans = {(m[0], m[1]) for m in mentions}
    if predictions["clusters"] != expected or any(
            tuple(span) not in mention_spans for cluster in expected for span in cluster):
        raise RuntimeError("unexpected reference clusters")
    if tokens == 521:
        reference_path = Path(__file__).resolve().parents[3] / "crates/anno/tests/fixtures/fcoref_word_budget.json"
        reference = json.loads(reference_path.read_text())
        if len(mentions) != len(reference):
            raise RuntimeError("unexpected reference mention count")
        for (start, end, _, logit), (gold_start, gold_end, gold_logit) in zip(mentions, reference):
            if (start, end) != (gold_start, gold_end) or not abs(logit - gold_logit) < 0.01:
                raise RuntimeError("unexpected reference mention or score")


def checkpoint(path, receipt, *, create=False):
    # Publish a complete same-filesystem snapshot. A hard link reserves the
    # initial name exclusively without exposing an empty or partial file.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        temporary = Path(out.name)
        try:
            json.dump(receipt, out, indent=2, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
            if create:
                os.link(temporary, path)
            else:
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, help="local image with matching Linux C runtime and /bin/sh")
    p.add_argument("--binary", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--ort", required=True, type=Path, help="CPU libonnxruntime.so.1.24.4")
    p.add_argument("--cpu", required=True, type=int)
    p.add_argument("--repeats", type=int, choices=range(1, 11), default=3)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--output", required=True, type=Path, help="new JSON receipt path")
    args = p.parse_args()
    if args.cpu < 0 or args.timeout < 1:
        p.error("CPU must be nonnegative and timeout must be positive")
    receipt = {"status": "incomplete", "runs": []}
    checkpoint(args.output, receipt, create=True)
    try:
        info = json.loads(docker("info", "--format", "{{json .}}"))
        receipt["daemon"] = {k: info.get(k) for k in
                             ("CgroupVersion", "CpuCfsQuota", "CPUSet", "MemoryLimit", "SwapLimit", "Warnings")}
        require_controls(info)
        image = docker("image", "inspect", "--format", "{{.Id}}", args.image)
        hashes = {name: sha(args.model / name) for name in MODEL_SHA}
        if hashes != MODEL_SHA or (args.model / "encoder_quantized.onnx").exists():
            raise RuntimeError("expected the pinned FP32 export")
        ort_hash = sha(args.ort)
        if ort_hash != "2d518e11c7767f9fdc4413bdbeb45acf30db504ff76a82fc19d9238a3f9f6c24":
            raise RuntimeError("expected the pinned Linux x86_64 ONNX Runtime CPU library 1.24.4")
        receipt.update(image=image, model_sha256=hashes, ort_sha256=ort_hash,
                       binary_sha256=sha(args.binary), cpu=args.cpu, repeats=args.repeats)
        receipt["status"] = "in_progress"
        for rep in range(args.repeats):
            for n, b in (CASES if rep % 2 == 0 else reversed(CASES)):
                for t in ((1, 2) if (rep+n) % 2 == 0 else (2, 1)):
                    receipt["active_case"] = {"repeat": rep + 1, "threads": t, "tokens": n, "budget": b}
                    checkpoint(args.output, receipt)
                    row = run_case(args, image, t, n, b)
                    row["repeat"] = rep + 1
                    try:
                        if (row["threads"], row["tokens"], row["budget"]) != (t, n, b):
                            raise RuntimeError("container row belongs to a different scheduled case")
                        verify(row, args.cpu)
                        # Ancillary Docker/model diagnostics must also be valid receipt JSON.
                        json.dumps(row, allow_nan=False)
                    except Exception:
                        # Retain the container failure details even when its
                        # measurement is malformed and cannot be serialized.
                        receipt["failed_run"] = {k: row[k] for k in ("state", "stderr", "stdout", "timed_out") if k in row}
                        raise
                    receipt["runs"].append(row)
                    del receipt["active_case"]
                    checkpoint(args.output, receipt)
        receipt["status"] = "completed"
    except Exception as exc:
        receipt.update(status="failed", error=str(exc))
        checkpoint(args.output, receipt)
        raise
    checkpoint(args.output, receipt)


if __name__ == "__main__":
    main()
