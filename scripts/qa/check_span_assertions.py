"""Check typed assertions against anno batch JSONL using half-open Unicode code-point ranges."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

DEFAULT_FIXTURE = Path(__file__).with_name("fixtures") / "typed-span-assertions.json"
# Built-in labels from EntityType::as_label used for assertion type sets.
TYPES = set("PER ORG LOC DATE TIME MONEY PERCENT QUANTITY CARDINAL ORDINAL EMAIL URL PHONE".split())


def objects(items, description):
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError(f"expected an array of objects for {description}")
    return items


def indexed(rows, description):
    result = {}
    for row in objects(rows, description):
        key = row.get("id")
        if not isinstance(key, str) or not key.strip() or key in result:
            raise ValueError(f"{description}: missing or duplicate id {key!r}")
        result[key] = row
    return result


def check_range(item, text, key):
    start, end = item.get("start"), item.get("end")
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
            or item.get("text") != text[start:end]):
        raise ValueError(f"{key}: range [{start}, {end}) does not select {item.get('text')!r}")


def valid_label(label):
    return isinstance(label, str) and bool(label) and label == label.strip() and label.isprintable()


def load_cases(rows, custom_types=()):
    if any(not valid_label(label) for label in custom_types):
        raise ValueError("custom assertion labels must be nonempty printable names without outer spaces")
    vocabulary = TYPES.union(custom_types)
    cases = indexed(rows, "fixture")
    if not cases:
        raise ValueError("empty fixture")
    for key, case in cases.items():
        text = case.get("text")
        if not isinstance(text, str) or not text or not objects(case.get("assertions"), key):
            raise ValueError(f"{key}: missing source text or assertions")
        for rule in case["assertions"]:
            check_range(rule, text, key)
            if rule.get("kind") not in ("exact_span", "no_overlap"):
                raise ValueError(f"{key}: unsupported assertion kind")
            labels = rule.get("types")
            if (not isinstance(labels, list) or not labels
                    or any(not isinstance(label, str) or label not in vocabulary for label in labels)
                    or len(set(labels)) != len(labels)):
                raise ValueError(f"{key}: expected distinct declared entity labels, got {labels!r}")
    return cases


def score(cases, predictions):
    outputs = indexed(predictions, "predictions")
    if outputs.keys() != cases.keys():
        raise ValueError(f"prediction IDs differ from fixture: {sorted(outputs.keys() ^ cases.keys())}")
    findings = []
    for key, case in cases.items():
        entities = objects(outputs[key].get("entities"), key)
        for entity in entities:
            check_range(entity, case["text"], key)
            label = entity.get("type")
            if not valid_label(label):
                raise ValueError(f"{key}: malformed prediction type {label!r}")
        for rule in case["assertions"]:
            overlaps = [
                {field: entity[field] for field in ("start", "end", "text", "type")}
                for entity in entities
                if entity["start"] < rule["end"] and rule["start"] < entity["end"]
            ]
            typed = [e for e in overlaps if e["type"] in rule["types"]]
            if rule["kind"] == "exact_span":
                passed = any((e["start"], e["end"]) == (rule["start"], rule["end"]) for e in typed)
            else:
                passed = not typed
            findings.append({"id": key, **rule, "passed": passed, "overlaps": overlaps})
    return {"passed": sum(row["passed"] for row in findings), "total": len(findings),
            "findings": findings}


def compare_predictions(cases, baseline_jsonl, candidate):
    # Both reports iterate the same validated fixture, regardless of JSONL row order.
    baseline = score(cases, [json.loads(line) for line in baseline_jsonl.splitlines()])
    gained = lost = 0
    for old, new in zip(baseline["findings"], candidate["findings"], strict=True):
        new["baseline_passed"] = old["passed"]
        if old["passed"] != new["passed"]:
            new["baseline_overlaps"] = old["overlaps"]
            if new["passed"]:
                gained += 1
            else:
                lost += 1
    return {"baseline_passed": baseline["passed"], "gained": gained, "lost": lost}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--binary", default="anno", help="anno CLI executable (default: anno)")
    source.add_argument("--predictions", type=Path, help="saved anno batch JSONL")
    parser.add_argument("--model", choices=("pattern", "heuristic", "stacked"), default="heuristic")
    parser.add_argument("--allow-type", action="append", default=[], metavar="LABEL",
                        help="declare a custom, case-sensitive assertion label (repeatable)")
    parser.add_argument("--baseline-predictions", type=Path,
                        help="compare against saved batch JSONL and fail on any lost assertion")
    args = parser.parse_args()
    try:
        cases = load_cases(json.loads(args.fixture.read_text(encoding="utf-8")), args.allow_type)
        if args.predictions:
            output = args.predictions.read_text(encoding="utf-8")
        else:
            inputs = "".join(json.dumps({"id": key, "text": case["text"]}) + "\n"
                             for key, case in cases.items())
            cmd = [args.binary, "batch", "--stdin", "--format", "jsonl", "--model", args.model]
            output = subprocess.run(cmd, input=inputs, encoding="utf-8", stdout=subprocess.PIPE,
                                    check=True, env={**os.environ, "ANNO_NO_DOWNLOADS": "1"}).stdout
        report = score(cases, [json.loads(line) for line in output.splitlines()])
        if args.baseline_predictions:
            report["comparison"] = compare_predictions(
                cases, args.baseline_predictions.read_text(encoding="utf-8"), report)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"invalid span QA run: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.baseline_predictions:
        return int(report["comparison"]["lost"] > 0)
    return int(report["passed"] != report["total"])


if __name__ == "__main__":
    sys.exit(main())
