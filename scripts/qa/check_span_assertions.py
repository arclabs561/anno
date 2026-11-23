"""Check typed assertions against anno batch JSONL using half-open Unicode code-point ranges."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from domain_coverage import valid_domain_name

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
        if "domain" in case and not valid_domain_name(case["domain"]):
            raise ValueError(f"{key}: expected a nonempty printable domain name")
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


def jsonl_rows(text):
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def source_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_batch(cases, binary, model):
    inputs = "".join(json.dumps({"id": key, "text": case["text"]}) + "\n"
                     for key, case in cases.items())
    cmd = [binary, "batch", "--stdin", "--format", "jsonl", "--model", model]
    output = subprocess.run(cmd, input=inputs, encoding="utf-8", stdout=subprocess.PIPE,
                            check=True, env={**os.environ, "ANNO_NO_DOWNLOADS": "1"}).stdout
    rows = jsonl_rows(output)
    # Older binaries omit this field. Bind their live output to the text we sent.
    for key, row in indexed(rows, "batch output").items():
        if key in cases and "source_sha256" not in row:
            row["source_sha256"] = source_sha256(cases[key]["text"])
    return rows


def score(cases, predictions):
    outputs = indexed(predictions, "predictions")
    if outputs.keys() != cases.keys():
        raise ValueError(f"prediction IDs differ from fixture: {sorted(outputs.keys() ^ cases.keys())}")
    findings = []
    predicted = unassessed = 0
    for key, case in cases.items():
        row = outputs[key]
        if "source_sha256" not in row:
            raise ValueError(f"{key}: saved prediction lacks source_sha256; "
                             "rerun the original binary with --save-predictions")
        digest = row["source_sha256"]
        if not isinstance(digest, str) or digest.lower() != source_sha256(case["text"]):
            raise ValueError(f"{key}: prediction source_sha256 differs from fixture text")
        entities = objects(row.get("entities"), key)
        for entity in entities:
            check_range(entity, case["text"], key)
            label = entity.get("type")
            if not valid_label(label):
                raise ValueError(f"{key}: malformed prediction type {label!r}")
        assessed = set()
        for rule in case["assertions"]:
            overlapping_indices = [i for i, e in enumerate(entities)
                                   if e["start"] < rule["end"] and rule["start"] < e["end"]]
            overlaps = [
                {field: entities[i][field] for field in ("start", "end", "text", "type")}
                for i in overlapping_indices
            ]
            matches = [i for i in overlapping_indices if entities[i]["type"] in rule["types"]]
            if rule["kind"] == "exact_span":
                matches = [i for i in matches if (entities[i]["start"], entities[i]["end"])
                           == (rule["start"], rule["end"])]
                passed = bool(matches)
            else:
                passed = not matches
            assessed.update(matches)
            findings.append({**rule, "id": key, "passed": passed, "overlaps": overlaps})
        predicted += len(entities)
        unassessed += len(entities) - len(assessed)
    return {"passed": sum(row["passed"] for row in findings), "total": len(findings),
            "prediction_coverage": {"scope": "supplied_assertions", "total": predicted,
                                    "unassessed": unassessed},
            "findings": findings}


def compare_predictions(cases, baseline_jsonl, candidate):
    # Both reports iterate the same validated fixture, regardless of JSONL row order.
    baseline = score(cases, jsonl_rows(baseline_jsonl))
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


def summarize_domains(cases, findings):
    """Keep positive and protected-range denominators visible for each labeled domain."""
    if not any("domain" in case for case in cases.values()):
        return []
    groups = {}
    for case in cases.values():
        domain = case.get("domain")
        if domain not in groups:
            groups[domain] = {
                "domain": domain, "documents": 0,
                "exact_span": {"passed": 0, "total": 0},
                "no_overlap": {"passed": 0, "total": 0},
            }
        groups[domain]["documents"] += 1
    for finding in findings:
        group = groups[cases[finding["id"]].get("domain")]
        counts = group[finding["kind"]]
        counts["total"] += 1
        counts["passed"] += finding["passed"]
        if "baseline_passed" in finding:
            changes = group.setdefault("comparison", {"baseline_passed": 0, "gained": 0, "lost": 0})
            changes["baseline_passed"] += finding["baseline_passed"]
            changes["gained"] += finding["passed"] and not finding["baseline_passed"]
            changes["lost"] += finding["baseline_passed"] and not finding["passed"]
    return list(groups.values())


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
    parser.add_argument("--save-predictions", type=Path, metavar="FILE",
                        help="capture a live batch run in a new source-bound JSONL file")
    args = parser.parse_args()
    if args.save_predictions and args.predictions:
        parser.error("--save-predictions requires a live run; use --binary")
    try:
        fixture = args.fixture.read_bytes()
        cases = load_cases(json.loads(fixture.decode("utf-8")), args.allow_type)
        if args.predictions:
            predictions = jsonl_rows(args.predictions.read_text(encoding="utf-8"))
        else:
            predictions = run_batch(cases, args.binary, args.model)
        report = score(cases, predictions)
        if args.baseline_predictions:
            report["comparison"] = compare_predictions(
                cases, args.baseline_predictions.read_text(encoding="utf-8"), report)
        domains = summarize_domains(cases, report["findings"])
        if domains:
            report.update(kind="anno.qa.domain-spans", schema_version=1,
                          fixture_sha256=hashlib.sha256(fixture).hexdigest(), domains=domains)
        if args.save_predictions:
            with args.save_predictions.open("x", encoding="utf-8") as capture:
                for row in predictions:
                    capture.write(json.dumps(row) + "\n")
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"invalid span QA run: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.baseline_predictions:
        return int(report["comparison"]["lost"] > 0)
    return int(report["passed"] != report["total"])


if __name__ == "__main__":
    sys.exit(main())
