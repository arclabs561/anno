"""Run one reviewed document-delivery trace through the native fuzz oracle."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import replay_crossdoc_native as native

SCOPE = "document_delivery_synthetic_contract_v1"
PASS_SIGNATURE = native.PASS_SIGNATURE
FAIL_SIGNATURE = native.FAIL_SIGNATURE
CASES = frozenset({
    "historical-receipt", "issuer-token-collision", "late-restored-review",
    "reviewed-peers-and-forks", "reviewed-source-edit",
})
CONTRACT = native.Contract("anno", "document_delivery_seeds", "replay_one_delivery_seed")
SUPPORT = native.Contract("anno", "support/document_delivery", CONTRACT.test)


def family_for_case(root, case):
    if case not in CASES:
        raise ValueError("unknown reviewed delivery seed")
    return case


def pins(root, case):
    family_for_case(root, case)
    seed = Path("fuzz/seeds/document_delivery") / case
    if (root / seed).is_symlink() or not subprocess.check_output(
            ["git", "ls-files", "-z", "--", seed], cwd=root):
        raise ValueError("expected a tracked delivery seed")
    return [native.digest(root / path) for path in (seed, CONTRACT.source, SUPPORT.source)]


def describe(root, case):
    source, *oracle = pins(root, case)
    for module in (Path(__file__), Path(native.__file__)):
        checked_in = root / "scripts/qa" / module.name
        if module.read_bytes() != checked_in.read_bytes():
            raise ValueError("delivery replay code differs from its checkout")
        oracle.append(native.digest(module))
    return {"family": case, "source_sha256": source,
            "oracle_sha256": hashlib.sha256(json.dumps(oracle).encode()).hexdigest()}


def build_env(root, case):
    # An absolute target/build directory can otherwise reuse another worktree's
    # newer producer even when Cargo reports this worktree's test target.
    env = dict(os.environ, ANNO_DELIVERY_CASE=case, RUST_BACKTRACE="0")
    local = root / "target"
    target = env.get("CARGO_TARGET_DIR", local)
    for key in ("CARGO_TARGET_DIR", "CARGO_BUILD_BUILD_DIR"):
        configured = (root / env.get(key, target)).resolve()
        if configured.is_relative_to(root):
            env[key] = str(configured)
        else:
            checkout = hashlib.sha256(os.fsencode(root)).hexdigest()
            env[key] = str(configured / "anno-delivery" / checkout)
    return env


def replay(root, case):
    source_pins = pins(root, case)
    checkout = hashlib.sha256(os.fsencode(root / "crates/anno")).hexdigest()
    marker = "delivery-oracle " + " ".join([*source_pins, checkout])
    env = build_env(root, case)
    binary, _ = native._artifacts(root, CONTRACT, env)
    digest = native.digest(binary)
    result = subprocess.run(
        [binary, "--exact", CONTRACT.test, "--ignored", "--show-output", "--format", "pretty", "--color", "never"],
        cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
    )
    if (result.stdout.splitlines().count(marker) != 1
            or native.digest(binary) != digest):
        raise ValueError("executed delivery seed, compiled oracle or executable differs from its pin")
    site = SUPPORT if f"{SUPPORT.source}:" in result.stdout else CONTRACT
    outcome, signature = native._verdict(site, result)
    return {"outcome": outcome, "signature": signature, "producer_sha256": digest}
