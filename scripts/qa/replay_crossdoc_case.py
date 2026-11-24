"""Run an authored cross-document case locally or replay a registered synthetic witness."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import crossdoc_cases
import crossdoc_eval_cases

SCOPE = "crossdoc_authored_cases_v1"
PASS_SIGNATURE = ""
FAIL_SIGNATURE = r"[0-9a-f]{64}"
RUNNERS = {
    "cross_doc": ("cli", crossdoc_cases, "ANNO_BIN"),
    "cross_doc_eval": ("eval", crossdoc_eval_cases, "ANNO_CDCR_BIN"),
}
# Only these project-authored fixtures may become shared reviewed witnesses.
REVIEWED = {
    "catalog-revision": ("crossdoc", "cross_doc"),
    "topic-scoped-homonyms": ("crossdoc_eval", "cross_doc_eval"),
}


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_case(path):
    # Local authors may keep sensitive examples in private untracked directories.
    with path.open("rb") as source:
        raw = source.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("cross-document case exceeds 1 MiB")
    def unique_fields(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"duplicate case field: {key}")
            obj[key] = value
        return obj

    case = json.loads(raw, object_pairs_hook=unique_fields)
    if not isinstance(case, dict) or not isinstance(case.get("kind"), str) or case["kind"] not in RUNNERS:
        raise ValueError("expected kind cross_doc or cross_doc_eval")
    if not isinstance(case.get("assertions"), list) or not 1 <= len(case["assertions"]) <= 256:
        raise ValueError("expected 1–256 frozen assertions")
    return case


def fingerprints(case, root=None):
    _, runner, _ = RUNNERS[case["kind"]]
    scorer = []
    for module in (Path(__file__), Path(runner.__file__)):
        if root is not None and module.read_bytes() != (root / "scripts/qa" / module.name).read_bytes():
            raise ValueError("executed cross-document scorer differs from the selected checkout")
        scorer.append(file_digest(module))
    return {
        "source_sha256": digest({k: v for k, v in case.items() if k != "assertions"}),
        "oracle_sha256": digest([scorer, case["assertions"]]),
    }


def execute(case, binary):
    name, runner, _ = RUNNERS[case["kind"]]
    binary = binary.resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("selected producer must be an executable file")
    producer = file_digest(binary)
    pins = fingerprints(case)
    checks = runner.run_case(case, binary)["checks"]
    if (not isinstance(checks, list) or not 1 <= len(checks) <= 256
            or any(not isinstance(c, dict) or type(c.get("passed")) is not bool for c in checks)):
        raise ValueError("cross-document oracle did not return 1–256 measured assertions")
    if file_digest(binary) != producer or fingerprints(case) != pins:
        raise ValueError("producer or oracle changed during replay")
    failed = [i for i, check in enumerate(checks) if not check["passed"]]
    receipt = {
        "kind": "anno.qa.crossdoc-case.v1", "runner": name, **pins,
        "producer_sha256": producer, "outcome": "fail" if failed else "pass",
        "passed": len(checks) - len(failed), "failed": len(failed),
    }
    return receipt, failed, checks


def family_for_case(root, case):
    if case not in REVIEWED:
        raise ValueError("unknown reviewed cross-document case")
    return case


def reviewed_case(root, case):
    family_for_case(root, case)
    directory, kind = REVIEWED[case]
    path = root / "scripts/qa/fixtures" / directory / (case + ".json")
    if path.is_symlink():
        raise ValueError("expected a regular reviewed fixture")
    data = load_case(path)
    if data["kind"] != kind:
        raise ValueError("reviewed case changed runner")
    return data


def describe(root, case):
    return {"family": family_for_case(root, case), **fingerprints(reviewed_case(root, case), root)}


def replay(root, case):
    data = reviewed_case(root, case)
    _, _, env_var = RUNNERS[data["kind"]]
    if not os.environ.get(env_var):
        raise ValueError(f"set {env_var} to the built executable for this cross-document case")
    receipt, failed, _ = execute(data, Path(os.environ[env_var]))
    return {
        "outcome": receipt["outcome"], "signature": digest(failed) if failed else PASS_SIGNATURE,
        "producer_sha256": receipt["producer_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("binary", type=Path, help="anno CLI or cdcr_eval executable for this case")
    parser.add_argument("--details", action="store_true", help="write per-check details to local stderr")
    args = parser.parse_args()
    try:
        receipt, failed, checks = execute(load_case(args.case), args.binary)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"Cannot execute cross-document case: {error}\n")
    print(json.dumps(receipt, sort_keys=True))
    if args.details:
        print(json.dumps(checks, ensure_ascii=False), file=sys.stderr)
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
