"""Check, replay, or score source-bound coreference QA cases using local publisher files."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import BadZipFile

from check_span_assertions import indexed, source_sha256
from coref_corpora import corefud, gap, gum, litbank, phrase_detectives, winogender
from coref_gap import index_partition
from score_person_mentions import model_spans, span

READERS = {
    "dutch-openboek": corefud.dutch,
    "french-litbankfr": corefud.french,
    "gap": gap.cases,
    "gum": gum.cases,
    "litbank": litbank.cases,
    "phrase-detectives": phrase_detectives.cases,
    "winogender": winogender.cases,
}
REGISTRY = Path(__file__).with_name("coref_corpora") / "sources.json"


class CorpusError(ValueError):
    """A diagnostic suitable for shared logs, independent of corpus contents."""


def check_case(case):
    text, ranges, kind, gold = (case[k] for k in ("text", "spans", "kind", "gold"))
    if not isinstance(text, str) or not text or not isinstance(case["group"], str) or not case["group"]:
        raise ValueError("missing case text or source group")
    marks = {k: span(v, text) for k, v in ranges.items()}
    if kind == "nested":
        if marks.keys() != {"inner", "outer"} or gold not in {"same", "different"}:
            raise ValueError("invalid nested pair")
        a, b = marks["inner"]
        c, d = marks["outer"]
        if not (c <= a < b <= d) or (a, b) == (c, d):
            raise ValueError("expected strict containment")
    elif kind == "choice":
        if "target" not in marks or not marks.keys() <= {"target", "A", "B"}:
            raise ValueError("invalid choice spans")
        if gold not in {"A", "B", "both", "neither", "unscored"}:
            raise ValueError("invalid choice gold")
        required = {"A", "B"} if gold == "both" else {gold} & {"A", "B"}
        if not required <= marks.keys() or len(set(marks.values())) != len(marks):
            raise ValueError("duplicate endpoints or unavailable gold choice")
    else:
        raise ValueError("unknown case kind")
    for alias in case.get("aliases", []):
        span(alias, text)
    for name, overrides in case.get("boundaries", {}).items():
        if kind != "choice" or name == "default" or not overrides or not overrides.keys() <= marks.keys():
            raise ValueError("invalid alternative scoring boundary")
        for r in overrides.values():
            span(r, text)
    tags = case.get("tags", [])
    if not isinstance(tags, list) or any(not isinstance(t, str) or not t for t in tags):
        raise ValueError("invalid case tags")


def select_cases(rows, *, limit=None, seed="anno-coref-v1", case_id=None, tag=None):
    index = indexed(list(rows), "corpus cases")
    for case in index.values():
        check_case(case)
    chosen = [case for case in index.values() if (case_id is None or case["id"] == case_id)
              and (tag is None or tag in case.get("tags", []))]
    if not chosen:
        raise CorpusError("no cases matched the requested corpus and selection")
    chosen.sort(key=lambda r: hashlib.sha256((seed + "/" + r["id"]).encode()).digest())
    return chosen[:limit]


def score_case(case, row):
    proposed, clusters = model_spans(case, row)
    groups = index_partition(clusters)
    reports = {}
    for policy, overrides in {"default": {}, **case.get("boundaries", {})}.items():
        marks = {k: tuple(v) for k, v in (case["spans"] | overrides).items()}
        proposed_roles = sorted(k for k, r in marks.items() if r in proposed)
        published_roles = sorted(k for k, r in marks.items() if r in groups)
        result = {"proposed": proposed_roles, "published": published_roles}
        if case["kind"] == "nested":
            if len(published_roles) < 2:
                decision = "unresolved"
            else:
                decision = "same" if groups[marks["inner"]] == groups[marks["outer"]] else "different"
        else:
            target = groups.get(marks["target"])
            linked = [k for k in ("A", "B") if k in marks and target is not None
                      and groups.get(marks[k]) == target]
            decision = "both" if len(linked) == 2 else next(iter(linked), "neither")
            result["target_linked_anywhere"] = target is not None
            if "aliases" in case:
                alternatives = {tuple(r) for r in case["aliases"]} - set(marks.values())
                result["alternate_gold_in_window"] = len(alternatives)
                result["linked_alternate_gold"] = None
                if alternatives:
                    result["linked_alternate_gold"] = target is not None and any(
                        groups.get(r) == target for r in alternatives)
        result["observed"] = decision
        result["matches_marked_gold"] = None if case["gold"] == "unscored" else decision == case["gold"]
        reports[policy] = result
    return {"id": case["id"], "kind": case["kind"], "gold": case["gold"],
            "tags": case.get("tags", []), "source_sha256": row["source_sha256"], "boundaries": reports}


def score(cases, predictions):
    try:
        rows = indexed(predictions["results"], "predictions")
        if rows.keys() != {c["id"] for c in cases}:
            raise CorpusError("prediction IDs differ from the selected cases")
        return [score_case(c, rows[c["id"]]) for c in cases]
    except CorpusError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        # General-purpose span validators may quote offending strings in their diagnostics.
        raise CorpusError("invalid prediction receipt; check source hashes, spans and cluster membership") from exc


def inputs(cases):
    return [{"id": c["id"], "text": c["text"]} for c in cases]


def export_inputs(output, cases, source, name, registry):
    # Keep source notices and answer-bearing publisher locators out of the model file.
    lineage = {"schema_version": 1, "corpus": name, "input_file": output.name,
               "publisher_input": str(source.resolve()), "source_record": registry,
               "cases": [{"id": c["id"], "group": c["group"], "source_sha256": source_sha256(c["text"]),
                          "provenance": c["provenance"]} for c in cases]}
    sidecar = output.with_name(output.name + ".provenance.json")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".anno-inputs-") as temp:
        data, metadata = Path(temp) / "data", Path(temp) / "metadata"
        for path, value in ((data, inputs(cases)), (metadata, lineage)):
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            path.chmod(0o600)
        # Link complete files without replacing an existing input or lineage record.
        os.link(metadata, sidecar)
        try:
            os.link(data, output)
        except OSError:
            sidecar.unlink()
            raise


def path_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def producer_identity(runner, model_dir):
    encoder = "encoder_quantized.onnx" if (model_dir / "encoder_quantized.onnx").exists() else "encoder.onnx"
    files = [model_dir / name for name in (encoder, "tokenizer.json", "scorer_weights.safetensors")]
    if (model_dir / "config.json").exists():
        files.append(model_dir / "config.json")
    # External tensors are separate from the ONNX graph in some exports.
    files.extend(p for p in model_dir.rglob("*") if p.is_file() and p.suffix in {".data", ".onnx_data"})
    return {"runner_sha256": path_sha(runner),
            "model_files_sha256": {str(p.relative_to(model_dir)): path_sha(p) for p in sorted(files)}}


def run_native(cases, runner, model_dir, batch_size):
    env = os.environ | {"ANNO_NO_DOWNLOADS": "1", "HF_HUB_OFFLINE": "1"}
    results = []
    with tempfile.TemporaryDirectory(prefix="anno-coref-") as tmp:
        batch_file = Path(tmp) / "inputs.json"
        for offset in range(0, len(cases), batch_size):
            batch = cases[offset:offset + batch_size]
            batch_file.write_text(json.dumps(inputs(batch), ensure_ascii=False), encoding="utf-8")
            proc = subprocess.run([runner.resolve(), model_dir.resolve(), batch_file],
                                  env=env, capture_output=True, check=False)
            if proc.returncode:
                raise CorpusError(f"local producer exited {proc.returncode} at batch {offset // batch_size + 1}")
            results.extend(score(batch, json.loads(proc.stdout)))
    return results


def positive(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return n


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show source locations, transformations and reuse terms")
    for cmd in ("check", "inputs", "run", "score", "score-document"):
        op = sub.add_parser(cmd)
        choices = sorted(READERS) + (["all"] if cmd == "check" else [])
        if cmd == "score-document":
            choices = ["dutch-openboek", "french-litbankfr"]
        op.add_argument("corpus", choices=choices)
        source = op.add_mutually_exclusive_group(required=True)
        source.add_argument("--source", type=Path, help="local publisher file or directory for one corpus")
        source.add_argument("--sources-file", type=Path, help="local JSON mapping corpus names to paths")
        op.add_argument("--limit", type=positive, help="maximum cases per corpus after filtering")
        op.add_argument("--seed", default="anno-coref-v1")
        op.add_argument("--case", help="replay one opaque case ID")
        op.add_argument("--tag", help="select a case tag")
        if cmd == "inputs":
            op.add_argument("--output", type=Path, required=True, help="new private local text-bearing file")
        if cmd == "run":
            op.add_argument("--runner", type=Path, required=True, help="local fcoref_word_spans-compatible executable")
            op.add_argument("--model-dir", type=Path, required=True)
            op.add_argument("--batch-size", type=positive, default=64)
        if cmd == "score-document":
            op.add_argument("--predictions", type=Path, required=True, help="full aligned CorefUD prediction .conllu")
            op.add_argument("--include-cases", action="store_true", help="include per-case outcomes by opaque ID")
        if cmd == "score":
            op.add_argument("--predictions", type=Path, required=True, help="existing source-bound prediction JSON")
    return p


def source_path(args, name):
    if args.source is not None:
        if args.corpus == "all":
            raise CorpusError("check all requires --sources-file")
        return args.source
    paths = json.loads(args.sources_file.read_bytes())
    if not isinstance(paths, dict) or not isinstance(paths.get(name), str) or not paths[name].strip():
        raise CorpusError(f"missing local source path for {name}")
    return args.sources_file.parent / paths[name]


def main():
    args = parser().parse_args()
    registry = json.loads(REGISTRY.read_bytes())
    if registry.keys() != READERS.keys():
        raise ValueError("source registry and adapters differ")
    if args.command == "list":
        return registry
    out = {}
    for name in sorted(READERS) if args.corpus == "all" else [args.corpus]:
        try:
            source = source_path(args, name)
            cases = select_cases(READERS[name](source), limit=args.limit, seed=args.seed,
                                 case_id=args.case, tag=args.tag)
        except CorpusError:
            raise
        except (ValueError, TypeError, OSError, KeyError, IndexError, BadZipFile) as exc:
            raise CorpusError(f"{name}: source validation failed; check the local path, input format and pinned release") from exc
        report = {"corpus": name, "source_revision": registry[name]["revision"], "cases": len(cases),
                  "groups": len({c["group"] for c in cases}), "kinds": Counter(c["kind"] for c in cases),
                  "tags": Counter(t for c in cases for t in c.get("tags", [])),
                  "selection_sha256": source_sha256(json.dumps([c["id"] for c in cases])),
                  "seed": args.seed, "limit": args.limit, "tag": args.tag, "case": args.case}
        if args.command == "inputs":
            export_inputs(args.output, cases, source, name, registry[name])
        elif args.command == "run":
            if not args.model_dir.is_dir():
                raise ValueError("model directory is unavailable")
            report.update(producer_identity(args.runner, args.model_dir))
            report["records"] = run_native(cases, args.runner, args.model_dir, args.batch_size)
        elif args.command == "score-document":
            from coref_corpora.system_output import score_document_predictions
            try:
                report["document_scores"] = score_document_predictions(
                    cases, source, name, args.predictions, include_cases=args.include_cases)
            except (ValueError, KeyError, OSError, BadZipFile) as exc:
                raise CorpusError(f"{name}: cannot score document predictions; check the component, "
                                  "sentence and token alignment, and supported Entity layout") from exc
        elif args.command == "score":
            data = args.predictions.read_bytes()
            report["predictions_sha256"] = hashlib.sha256(data).hexdigest()
            report["records"] = score(cases, json.loads(data))
        out[name] = report
    return out


if __name__ == "__main__":
    try:
        print(json.dumps(main(), indent=2, ensure_ascii=False))
    except CorpusError as exc:
        sys.exit(f"coreference QA: {exc}")
    except FileExistsError:
        sys.exit("coreference QA: output already exists; choose a new private output file")
    except (OSError, ValueError, KeyError, TypeError, BadZipFile):
        sys.exit("coreference QA: invalid or unavailable local input, model, producer or output")
