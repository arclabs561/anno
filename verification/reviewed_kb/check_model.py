"""Check every named query in the reviewed catalog assignment model using Z3."""

from pathlib import Path
import subprocess
import sys


QUERIES = (
    ("stale-rejection-preserves-world", "unsat"),
    ("fresh-acceptance-preserves-frame", "unsat"),
    ("fresh-change-with-unaffected-peer", "sat"),
    ("changed-and-restored-binding-rejected", "sat"),
    ("value-comparison-misses-intervening-write", "sat"),
    ("revision-comparison-misses-foreign-owner", "sat"),
    ("overbroad-apply-changes-unreported-peer", "sat"),
)


def main() -> int:
    model = Path(__file__).with_name("revision_guard.smt2")
    try:
        solver = subprocess.run(
            ["z3", "-smt2", str(model)], capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        sys.exit("Install the Z3 command-line tool and make z3 available on PATH.")
    except subprocess.TimeoutExpired:
        sys.exit("Timed out checking the reviewed assignment model.")

    expected = [line for pair in QUERIES for line in pair]
    if solver.returncode or solver.stderr or solver.stdout.splitlines() != expected:
        print(f"Z3 exit code {solver.returncode}", file=sys.stderr)
        print("Expected named results:", QUERIES, file=sys.stderr)
        print(solver.stdout, end="", file=sys.stderr)
        print(solver.stderr, end="", file=sys.stderr)
        return 1

    for name, result in QUERIES:
        print(f"{name}: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
