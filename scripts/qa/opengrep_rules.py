"""Run the repository's OpenGrep rules with their production targets or fixtures."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


# Artifact name, rule pack, production targets, annotated fixture.
CORE_AND_EVAL = ("crates/anno", "crates/anno-eval")
PACKS = (
    ("validated", "rust-validated", ("crates",), "eval/graph.rs"),
    ("security", "rust-security", CORE_AND_EVAL, "backends/eval/library_security.rs"),
    ("nlp", "rust-nlp-ml-patterns", CORE_AND_EVAL, "nlp/backends/gliner_nlp_onnx_cases.rs"),
    ("eval", "rust-evaluation-framework", ("crates/anno-eval", "crates/anno/src/metrics/coref_metrics.rs"),
     "evaluation/eval/evaluation_framework_cases.rs"),
    ("anno", "rust-anno-specific", ("crates/anno",), "anno_specific/entity.rs"),
    ("error", "rust-error-handling", CORE_AND_EVAL, "backends/eval/library_error_handling.rs"),
    ("memory", "rust-memory-patterns", CORE_AND_EVAL, "eval/backends/reference_memory.rs"),
)


def scan_counts(report):
    result = json.loads(report.read_text())
    if result["errors"] != []:
        raise ValueError(f"{report} contains scanner errors: {result['errors']}")
    scanned, findings = result["paths"]["scanned"], result["results"]
    if not isinstance(scanned, list) or not scanned or not isinstance(findings, list):
        raise ValueError(f"{report} is missing scanned paths or results")
    return len(scanned), len(findings)


def check_target_coverage(log, scanned):
    # The pinned CLI reports the number of configured and applicable rules.
    # JSON paths are pack-wide, so they cannot detect one rule with no targets.
    configured = re.search(r"with (\d+) Code rules?:\s*$", log, re.MULTILINE)
    executed = re.search(r"^Ran (\d+) rules? on (\d+) files?:", log, re.MULTILINE)
    if not configured or not executed:
        raise ValueError("OpenGrep did not report its production rule coverage")
    total = int(configured[1])
    active, paths = map(int, executed.groups())
    if not total or active != total or paths != scanned:
        raise ValueError(f"OpenGrep applied {active}/{total} configured rules to {paths} files (JSON: {scanned})")
    return active


def scan(name, config, targets):
    report = Path(f"opengrep-{name}-results.json")
    report.unlink(missing_ok=True)
    command = ["opengrep", "scan", "-f", config, "--json", "--output", report, *targets]
    completed = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    print(completed.stderr, end="", file=sys.stderr)
    completed.check_returncode()
    scanned, findings = scan_counts(report)
    rules = check_target_coverage(completed.stderr, scanned)
    print(f"{name}: {rules} rules, {scanned} scanned, {findings} findings", flush=True)


def check_fixture_ids(config, declared, checks):
    missing = declared - checks.keys()
    unexpected = checks.keys() - declared
    if not declared or missing or unexpected:
        raise ValueError(f"{config}: missing fixture checks {sorted(missing)}, unexpected {sorted(unexpected)}")
    for rule_id, check in checks.items():
        if not check["passed"] or not any(match["expected_lines"] for match in check["matches"].values()):
            raise ValueError(f"{config}: {rule_id} has no passing positive case")


def test_fixture(engine, config, fixture):
    # Ask the pinned scanner to parse the configuration independently of target filtering.
    rules = subprocess.check_output([engine, "-prefilter_of_rules", config], text=True)
    declared = {rule["rule_id"] for rule in json.loads(rules)}

    command = ["opengrep", "test", "--json", "-f", config, Path(".opengrep/fixtures") / fixture]
    completed = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    if completed.returncode:
        print(completed.stdout, file=sys.stderr)
        completed.check_returncode()
    result = json.loads(completed.stdout)
    checks = result["results"][str(config)]["checks"]
    if result["config_with_errors"]:
        raise ValueError(f"{config} has scanner errors for {fixture}")
    # OpenGrep can exit successfully when a rule loses its whole fixture case.
    check_fixture_ids(config, declared, checks)
    print(f"{config.stem}: {len(checks)} fixture checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", action="store_true", help="run annotated positive and negative cases")
    args = parser.parse_args()
    engine = None
    if args.fixtures:
        engine = subprocess.check_output(["opengrep", "scan", "--dump-engine-path"], text=True).strip()
    failed = False
    for name, pack, targets, fixture in PACKS:
        config = Path(".opengrep/rules") / f"{pack}.yaml"
        try:
            if args.fixtures:
                test_fixture(engine, config, fixture)
            else:
                scan(name, config, targets)
        except (KeyError, OSError, TypeError, ValueError, subprocess.CalledProcessError) as error:
            print(f"{pack}: {error}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
