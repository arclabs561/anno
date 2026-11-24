# SPDX-License-Identifier: MIT OR Apache-2.0
"""Check the named admission properties and positive/mutant witnesses with Z3."""
from itertools import product
from pathlib import Path
import sys
import subprocess

MODEL = Path(__file__).with_name("admission.smt2")
EXPECTED = {
    "local-file-permission-cannot-grant-origin-or-shared-access": "unsat",
    "publisher-permission-does-not-grant-shared-raw-publication": "unsat",
    "replay-cannot-relicense-external-data-or-promote-a-holdout": "unsat",
    "judge-only-or-uncredited-seeds-cannot-enter-shared-state": "unsat",
    "authorized-local-source-has-a-witness": "sat",
    "publisher-origin-fetch-has-a-witness": "sat",
    "reviewed-independent-synthetic-replay-has-a-witness": "sat",
    "trusting-replay-alone-allows-a-restricted-source": "sat",
}


def main():
    output = subprocess.run(["z3", "-smt2", str(MODEL)], check=True, text=True,
                            capture_output=True, timeout=30)
    expected = [item for pair in EXPECTED.items() for item in pair]
    if output.stderr or output.stdout.splitlines() != expected:
        raise SystemExit(f"Unexpected source-rights solver output\n{output.stdout}{output.stderr}")
    sys.path.insert(0, str(MODEL.parents[2] / "scripts/qa/reviewed_span_seeds"))
    from replay import reviewed_source

    # Exhaustively compare the actual registry admission predicate to the SMT
    # specification. Use a single solver invocation for the finite input space.
    origins = {"LocalOnly": "licensed-local", "Publisher": "publisher", "ReviewedSynthetic": "project-authored-synthetic"}
    queries = []
    for (source, origin), registered, license_id, held in product(
        origins.items(), (False, True), (None, "unknown", "MIT OR Apache-2.0"), (False, True)
    ):
        entry = {"scope": "holdout" if held else "exposed-regression", "origin": origin,
                 "license": license_id}
        actual = reviewed_source(entry if registered else {})
        args = [source, str(registered).lower(), str(license_id == "MIT OR Apache-2.0").lower(),
                str(held).lower()]
        queries.extend(["(push)", f"(assert (distinct (can-register {' '.join(args)}) {str(actual).lower()}))",
                        "(check-sat)", "(pop)"])
    result = subprocess.run(["z3", "-in"], input=MODEL.read_text() + "\n" + "\n".join(queries),
                            check=True, text=True, capture_output=True, timeout=30)
    cases = len(queries) // 4
    if result.stderr or result.stdout.splitlines() != expected + ["unsat"] * cases:
        raise SystemExit("The reviewed seed source gate disagrees with its formal model")
    print(f"Verified {len(EXPECTED)} named queries and {cases} model/implementation comparisons.")


if __name__ == "__main__":
    main()
