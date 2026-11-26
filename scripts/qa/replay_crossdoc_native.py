"""Execute one repository-authored cross-document contract with the offline Rust harness."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import NamedTuple


SCOPE = "crossdoc_native_contract_v1"
PASS_SIGNATURE = ""
FAIL_SIGNATURE = r"assertion:[0-9a-f]{64}"


class Contract(NamedTuple):
    package: str
    target: str
    test: str

    @property
    def source(self):
        return Path("crates") / self.package / "tests" / (self.target + ".rs")


CASES = {
    "unlinked-revision-history": Contract(
        "anno", "index_reconciliation", "unlinked_revision_histories_match_fresh_members_and_identity_metadata"),
    "guarded-catalog-checkpoints": Contract(
        "anno", "index_reconciliation", "catalog_decisions_preserve_the_members_reviewed_at_each_checkpoint"),
    "raw-blocking-boundary": Contract(
        "anno-cli", "crossdoc_candidates", "raw_candidate_policy_controls_pell_pair_across_size_boundary"),
    "raw-pair-explanation": Contract(
        "anno-cli", "crossdoc_candidates", "explain_one_pair_reports_actual_candidate_and_threshold_decisions"),
    "grounded-pair-explanation": Contract(
        "anno-cli", "membership_queries", "explain_submitted_pair_from_cli"),
    "grounded-candidate-policy": Contract(
        "anno-cli", "crossdoc_candidates", "imported_tracks_report_their_own_policy_and_reject_raw_options"),
    "grounded-replacement-order": Contract(
        "anno-cli", "crossdoc_report", "mixed_imports_apply_same_document_replacements_in_argument_order"),
    "query-catalog-publication": Contract(
        "anno-cli", "crossdoc_report", "query_reads_the_produced_report_and_matches_the_exact_catalog_record"),
}


def family_for_case(root, case):
    if case not in CASES:
        raise ValueError("unknown native cross-document contract")
    return case


def digest(path):
    with path.open("rb") as reader:
        return hashlib.file_digest(reader, "sha256").hexdigest()


def describe(root, case):
    family = family_for_case(root, case)
    contract = CASES[case]
    source = digest(root / contract.source)
    # These integration test files contain the synthetic inputs and their expectations.
    fields = [SCOPE, contract, source, digest(Path(__file__))]
    oracle = hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()
    return {"family": family, "source_sha256": source, "oracle_sha256": oracle}


def _artifacts(root, contract, env):
    command = ["cargo", "test", "--locked", "--offline", "-p", contract.package, "--no-default-features"]
    if contract.package == "anno-cli":
        command.extend(["--features", "eval extractor-readability"])
    command.extend(["--test", contract.target, "--no-run", "--message-format=json-render-diagnostics"])
    messages = subprocess.check_output(command, cwd=root, env=env, text=True)
    binaries = {}
    for line in messages.splitlines():
        record = json.loads(line)
        if record.get("reason") == "compiler-artifact" and record.get("executable"):
            key = (tuple(record["target"]["kind"]), Path(record["target"]["src_path"]), record["profile"]["test"])
            if key in binaries:
                raise ValueError("ambiguous Cargo executable")
            binaries[key] = Path(record["executable"])
    try:
        harness = binaries[(("test",), root / contract.source, True)]
        # The CLI contracts spawn env!("CARGO_BIN_EXE_anno"). Cargo reports that
        # same ordinary bin alongside the test harness in this invocation.
        if contract.package == "anno-cli":
            producer = binaries[(("bin",), root / "crates/anno-cli/src/main.rs", False)]
        else:
            producer = harness
    except KeyError as exc:
        raise ValueError("Cargo did not report the required cross-document executable") from exc
    return harness, producer


def _verdict(contract, result):
    lines = result.stdout.splitlines()
    name = contract.test
    if result.returncode == 0:
        status, run, passed, failed = "ok", "ok", 1, 0
    elif result.returncode == 101:
        status, run, passed, failed = "FAILED", "FAILED", 0, 1
    else:
        raise ValueError("test harness did not finish normally")
    summary = rf"test result: {status}\. {passed} passed; {failed} failed; 0 ignored; 0 measured; \d+ filtered out; finished in .+"
    if (lines.count("running 1 test") != 1 or lines.count(f"test {name} ... {run}") != 1
            or sum(re.fullmatch(summary, line) is not None for line in lines) != 1):
        raise ValueError("the named contract did not produce exactly one completed test")
    if passed:
        return "pass", PASS_SIGNATURE

    # A build error, abort, missing binary, or harness error is not an observed
    # contract violation. Restrict failed receipts to a panic in the pinned oracle.
    output = result.stdout.replace("\\", "/")
    source = re.escape(contract.source.as_posix())
    site = re.search(rf"(?m)^thread '{re.escape(name)}'(?: \(\d+\))? panicked at (?:[^\n]*?/)?{source}:(\d+):(\d+):\n([^\n]+)", output)
    if not site:
        raise ValueError("failure did not identify a panic in the named contract")
    reason = site.group(3)
    if ("Os {" in reason or "error while loading shared libraries" in reason
            or "No space left on device" in reason):
        raise ValueError("test could not access a required local resource")
    # Assertion locations identify failing checks without exporting CLI diagnostics
    # or unstable temp paths and allocated IDs from assertion values.
    signature = hashlib.sha256(f"{name}:{site.group(1)}:{site.group(2)}".encode()).hexdigest()
    return "fail", "assertion:" + signature


def replay(root, case):
    root = Path(root).resolve()
    before = describe(root, case)
    contract = CASES[case]
    env = dict(os.environ, ANNO_NO_DOWNLOADS="1", ANNO_S3_CACHE="0", HF_HUB_OFFLINE="1",
               CARGO_NET_OFFLINE="true", RUST_BACKTRACE="0")
    harness, producer = _artifacts(root, contract, env)
    pinned = {path: digest(path) for path in {harness, producer}}
    result = subprocess.run([harness, "--exact", contract.test, "--format", "pretty", "--color", "never"],
                            cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    outcome, signature = _verdict(contract, result)
    if describe(root, case) != before or any(digest(path) != hashed for path, hashed in pinned.items()):
        raise ValueError("source, harness, or executable changed during native replay")
    return {"outcome": outcome, "signature": signature, "producer_sha256": pinned[producer]}
