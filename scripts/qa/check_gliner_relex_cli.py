#!/usr/bin/env python3
"""Check a pinned Relex export via anno extract and the grounded reader.

Refresh reference predictions with the exporter Python environment:
    python scripts/qa/check_gliner_relex_cli.py --refresh-reference
Verify the native CLI using only the standard library:
    python scripts/qa/check_gliner_relex_cli.py --anno target/debug/anno --receipt result.json
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

FIXTURE = Path(__file__).resolve().parents[2] / "testdata/grounded/gliner_relex_reference.json"
LABELS = {"person": "PER", "organization": "ORG", "company": "ORG", "product": "PRODUCT",
          "misc": "MISC", "location": "LOC"}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def refresh(spec):
    from gliner import GLiNER
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from export_gliner_relex_joint import source_identity

    model = GLiNER.from_pretrained(spec["checkpoint"], revision=spec["revision"]).eval()
    producer = source_identity(spec["checkpoint"], spec["revision"])
    if producer["checkpoint_sha256"] != spec["checkpoint_sha256"]:
        raise ValueError("Unexpected source checkpoint digest")
    spec["reference_runtime"] = {"python": producer["python"], "packages": producer["packages"]}
    for case in spec["cases"]:
        entities, relations = model.predict_relations(case["text"], case["entity_labels"],
            case["relation_labels"], threshold=case["threshold"])
        case.update(reference_predictions(entities, relations))
    FIXTURE.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")


def reference_predictions(entities, relations):
    """Translate the pinned prompt labels to the CLI's public entity type strings."""
    return {
        "entities": [{"start": e["start"], "end": e["end"], "text": e["text"],
                      "type": LABELS.get(e["label"], e["label"]), "confidence": e["score"]}
                     for e in entities],
        "relations": [{"head": [r["head"]["start"], r["head"]["end"]],
                       "tail": [r["tail"]["start"], r["tail"]["end"]],
                       "type": r["relation"], "confidence": r["score"]} for r in relations],
    }


def stock_decision_digest(cases):
    """Bind model decisions to their document, ignoring within-document order and score noise."""
    decisions = []
    for case in cases:
        mentions = sorted((e["start"], e["end"], e["type"]) for e in case["entities"])
        edges = sorted((e["head"], e["type"], e["tail"]) for e in case["relations"])
        decisions.append([case["name"], mentions, edges])
    return hashlib.sha256(json.dumps(decisions, separators=(",", ":")).encode()).hexdigest()


def check_producer(producer, generation):
    if (producer["source"], producer["method"], producer["model_version"]) != (
            "knowledgator/gliner-relex-large-v1.0", "Neural", "sha256:" + generation):
        raise ValueError("Incorrect relation producer identity")


def native_runtime(binary, explicit):
    # In load-dynamic builds, ort reads ORT_DYLIB_PATH. Linked Linux builds
    # resolve their dependency with the same environment used for the replay.
    path = None
    if sys.platform.startswith("linux"):
        linked = subprocess.check_output(["ldd", str(binary)], text=True)
        match = re.search(r"libonnxruntime\.so[^ ]* => (\S+)", linked)
        if match:
            path = Path(match[1]).resolve(strict=True)
    if path is None:
        selected = os.environ.get("ORT_DYLIB_PATH") or explicit
        if not selected:
            raise ValueError("Cannot resolve native ORT; supply --ort-library")
        path = Path(selected).resolve(strict=True)
        os.environ["ORT_DYLIB_PATH"] = str(path)
    if explicit and explicit.resolve() != path:
        raise ValueError("Selected ORT library differs from the requested runtime")
    # OrtApiBase is stable across API versions.
    class ApiBase(ctypes.Structure):
        _fields_ = [("get_api", ctypes.c_void_p), ("version", ctypes.CFUNCTYPE(ctypes.c_char_p))]
    lib = ctypes.CDLL(str(path))
    lib.OrtGetApiBase.restype = ctypes.POINTER(ApiBase)
    return {"version": lib.OrtGetApiBase().contents.version().decode(), "sha256": digest(path)}


def score_map(rows, key):
    scores = {}
    for row in rows:
        item = key(row)
        if item in scores:
            raise ValueError("Duplicate prediction identity")
        scores[item] = row["confidence"]
    return scores


def parity(observed, expected):
    if observed.keys() != expected.keys():
        raise ValueError(f"Prediction identities differ: {len(expected.keys() - observed.keys())} missing, {len(observed.keys() - expected.keys())} extra")
    errors = [abs(observed[k] - expected[k]) for k in expected]
    if any(not math.isfinite(e) or e > 2e-4 for e in errors):
        raise ValueError(f"Prediction score drift across {len(errors)} aligned predictions")
    return max(errors, default=0.0)


def check_case(binary, case, generation, export):
    command = [str(binary), "extract", "--model", "gliner-relex", "--extract-relations",
               "--text", case["text"], "--extract-types", ",".join(case["entity_labels"]),
               "--relation-types", ",".join(case["relation_labels"]), "--relation-threshold",
               str(case["threshold"]), "--format", "json", "--export", str(export), "--quiet"]
    process = subprocess.run(command, capture_output=True)
    if process.returncode:
        raise ValueError(f"Native extraction exited {process.returncode}")
    output = json.loads(process.stdout)
    provenance = output["provenance"]
    if (provenance["model"], provenance["runtime_model"]) != ("gliner-relex", "GLiNER-ONNX"):
        raise ValueError("Unexpected Relex runtime")
    check_producer(provenance["relation_producer"], generation)
    for edge in output["relations"]:
        check_producer(edge["provenance"], generation)
    def mention(e):
        return (e["start"], e["end"], e["type"], e["text"])
    def flat_edge(e):
        return ((e["head"]["start"], e["head"]["end"]), e["type"], (e["tail"]["start"], e["tail"]["end"]))
    def ref_edge(e):
        return (tuple(e["head"]), e["type"], tuple(e["tail"]))
    entity_delta = parity(score_map(output["entities"], mention), score_map(case["entities"], mention))
    target = score_map(case["relations"], ref_edge)
    edge_delta = parity(score_map(output["relations"], flat_edge), target)
    saved = json.loads(export.read_text())
    reloaded = subprocess.run([str(binary), "enhance", str(export), "--quiet", "--format", "grounded"], capture_output=True)
    if reloaded.returncode:
        raise ValueError(f"Native grounded reload exited {reloaded.returncode}")
    loaded = json.loads(reloaded.stdout)
    if saved != loaded:
        raise ValueError(f'{case["name"]}: grounded document changed on reload')
    signals = {s["id"]: s for s in loaded["signals"]}
    evidence = loaded.get("relation_evidence", {})
    scored = {}
    if evidence and evidence["text_sha256"] != hashlib.sha256(case["text"].encode()).hexdigest():
        raise ValueError("Relation evidence is bound to a different source edition")
    for item in evidence.get("mentions", []):
        def span(role):
            signal = signals[item[f"{role}_signal_id"]]
            loc = signal["location"]["Text"]
            actual = item["relation"][role]
            if (actual["text"], actual["start"], actual["end"]) != (signal["surface"], loc["start"], loc["end"]):
                raise ValueError(f"Grounded {role} does not match its signal")
            return loc["start"], loc["end"]
        key = (span("head"), item["relation"]["relation_type"], span("tail"))
        if key in scored:
            raise ValueError("Duplicate grounded relation identity")
        scored[key] = item["relation"]["confidence"]
        check_producer(item["provenance"], generation)
    parity(scored, target)
    return {"name": case["name"], "mentions": len(output["entities"]), "directed_edges": len(target),
            "max_mention_score_difference": entity_delta, "max_edge_score_difference": edge_delta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--crossre-cases", type=Path, help="Additional upstream cases from relex_roles.py")
    parser.add_argument("--ort-library", type=Path)
    parser.add_argument("--refresh-reference", action="store_true")
    args = parser.parse_args()
    spec = json.loads(FIXTURE.read_text())
    if args.refresh_reference:
        refresh(spec)
        return
    binary = (args.anno or Path(shutil.which("anno") or "anno")).resolve(strict=True)
    os.environ["ANNO_NO_DOWNLOADS"] = "1"
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    install = hf / "hub" / ("models--" + spec["checkpoint"].replace("/", "--")) / "onnx/joint"
    generation = (install / "current").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", generation):
        raise ValueError("Invalid joint export generation")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from export_gliner_relex_onnx import artifact_digest

    if artifact_digest(install / generation) != generation:
        raise ValueError("Installed graph, tokenizer or metadata has changed")
    producer = json.loads((install / generation / "joint-relex-v1").read_text())
    for field in ("checkpoint", "revision", "checkpoint_sha256"):
        if producer[field] != spec[field]:
            raise ValueError(f"Export {field} differs from the pinned reference")
    receipt = {"source": producer, "generation": generation, "binary_sha256": digest(binary),
               "native_ort": native_runtime(binary, args.ort_library), "reference_sha256": digest(FIXTURE)}
    cases = spec["cases"]
    if args.crossre_cases:
        cases = cases + json.loads(args.crossre_cases.read_bytes())
        receipt["crossre_cases_sha256"] = digest(args.crossre_cases)
    with tempfile.TemporaryDirectory() as directory:
        receipt["cases"] = []
        for index, case in enumerate(cases):
            try:
                result = check_case(binary, case, generation, Path(directory) / f"{index}.json")
            except subprocess.CalledProcessError as error:
                raise ValueError(f'{case["name"]}: native command exited {error.returncode}') from None
            except KeyError:
                raise ValueError(f'{case["name"]}: required response field missing') from None
            except ValueError as error:
                raise ValueError(f'{case["name"]}: {error}') from None
            receipt["cases"].append(result)
            print(f'{result["name"]}: {result["mentions"]} mentions, {result["directed_edges"]} directed edges',
                  file=sys.stderr, flush=True)
    report = json.dumps(receipt, indent=2) + "\n"
    if args.receipt:
        args.receipt.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
