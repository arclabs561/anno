"""Run the repository's OpenGrep rules with their production targets or fixtures."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


# Artifact name, rule pack, production targets, annotated fixture.
CORE_AND_EVAL = ("crates/anno", "crates/anno-eval")
PACKS = (
    ("validated", "rust-validated", ("crates",), "eval/graph.rs"),
    ("security", "rust-security", CORE_AND_EVAL, "backends/eval/library_security.rs"),
    ("nlp", "rust-nlp-ml-patterns", CORE_AND_EVAL, "nlp/backends/gliner_nlp_onnx_cases.rs"),
    ("eval", "rust-evaluation-framework", ("crates/anno-eval",), "evaluation/eval/evaluation_framework_cases.rs"),
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


def scan(name, config, targets):
    report = Path(f"opengrep-{name}-results.json")
    report.unlink(missing_ok=True)
    subprocess.run(["opengrep", "scan", "-f", config, "--json", "--output", report, *targets], check=True)
    scanned, findings = scan_counts(report)
    print(f"{name}: {scanned} scanned, {findings} findings", flush=True)


def test_fixture(config, fixture):
    command = ["opengrep", "test", "--json", "-f", config, Path(".opengrep/fixtures") / fixture]
    completed = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    if completed.returncode:
        print(completed.stdout, file=sys.stderr)
        completed.check_returncode()
    result = json.loads(completed.stdout)
    checks = result["results"][str(config)]["checks"]
    # OpenGrep can exit successfully with no checks when fixture annotations drift.
    if result["config_with_errors"] or not checks:
        raise ValueError(f"{config} has no successful fixture run for {fixture}")
    print(f"{config.stem}: {len(checks)} fixture checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", action="store_true", help="run annotated positive and negative cases")
    args = parser.parse_args()
    failed = False
    for name, pack, targets, fixture in PACKS:
        config = Path(".opengrep/rules") / f"{pack}.yaml"
        try:
            if args.fixtures:
                test_fixture(config, fixture)
            else:
                scan(name, config, targets)
        except (KeyError, OSError, TypeError, ValueError, subprocess.CalledProcessError) as error:
            print(f"{pack}: {error}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
