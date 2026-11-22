# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.5", "scipy==1.18.1", "udapi==0.5.2"]
# ///
"""Replay one paired CorefUD dataset with the official CRAC scorer."""
import argparse
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

SCORER_COMMIT = "4fd7b0e0c661aeeff88bc60c19ef507b84d1b590"
MODES = {
    "primary": ("head", False),
    "partial": ("partial", False),
    "exact": ("exact", False),
    "singletons": ("head", True),
}


def score(root, key, response, mode, zero_match="dependent", capture=False):
    match, keep_singletons = MODES[mode]
    flags = ["-a", match, "-z", zero_match]
    if keep_singletons:
        flags.append("--keep-singletons")
    if not capture:
        singletons = "included" if keep_singletons else "excluded"
        print(f"CorefUD {mode} mode on one dataset with {match} mention matching, "
              f"singletons {singletons}, {zero_match} zero matching.", flush=True)
    return subprocess.run(
        [sys.executable, str(root / "corefud-scorer.py"), str(key), str(response),
         *flags, "-m", "muc", "bcub", "ceafe", "zero"],
        check=True, text=True, stdout=subprocess.PIPE if capture else None,
    ).stdout


def verify(root):
    fixtures = root / "tests" / "tests-corefud"
    # Same mention heads with changed boundaries; then a missing entity; then
    # a relocated empty node with the same dependency attachment.
    cases = [
        ("head-match/TC-HMA", 2, "primary", "dependent", "100.00", None),
        ("head-match/TC-HMA", 2, "exact", "dependent", "52.78", None),
        ("head-match/TC-HMA", 2, "partial", "dependent", "100.00", None),
        ("zeros/TC-ZA", 2, "primary", "dependent", "48.48", None),
        ("zeros/TC-ZA", 2, "singletons", "dependent", "59.33", None),
        ("zeros/TC-ZF", 2, "primary", "dependent", "100.00", "100.00"),
        ("zeros/TC-ZF", 2, "primary", "linear", "72.22", "0.00"),
    ]
    for name, variant, mode, method, conll, zero in cases:
        key = fixtures / f"{name}.key"
        response = fixtures / f"{name}-{variant}.response"
        output = score(root, key, response, mode, method, capture=True)
        if f"CoNLL score: {conll}" not in output.splitlines():
            raise SystemExit(f"{name} {mode} {method}: expected CoNLL {conll}\n{output}")
        if zero is not None and f"zero\nRecall: {zero}  Precision: {zero}  F1: {zero}\n" not in output:
            raise SystemExit(f"{name} {method}: expected zero P/R/F1 {zero}\n{output}")
    print(f"{len(cases)} official fixture checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", type=Path, required=True, help="local ufal/corefud-scorer repository")
    parser.add_argument("--mode", choices=MODES, default="primary")
    parser.add_argument("--check", action="store_true", help="run the pinned upstream fixture controls")
    parser.add_argument("pair", type=Path, nargs="*", metavar="CONLLU")
    args = parser.parse_args()
    if len(args.pair) != (0 if args.check else 2):
        parser.error("supply KEY.conllu RESPONSE.conllu, or --check without input files")
    paths = ["corefud-scorer.py", "ua-scorer.py", "scorer"]
    if args.check:
        paths.append("tests/tests-corefud")
    # Read the pinned Git tree without touching the checkout or executing local edits.
    archive = subprocess.check_output(["git", "-C", str(args.scorer), "archive",
                                       SCORER_COMMIT, "--", *paths])
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        with tarfile.open(fileobj=io.BytesIO(archive)) as files:
            files.extractall(root, filter="data")
        print(f"ufal/corefud-scorer {SCORER_COMMIT}", flush=True)
        if args.check:
            verify(root)
        else:
            score(root, *args.pair, args.mode)


if __name__ == "__main__":
    main()
