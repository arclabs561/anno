"""Choose one journey per policy before measuring the same small, frozen CLI panel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time

import replay_crossdoc_case as catalog
import replay_crossdoc_native as native

ROOT = Path(__file__).resolve().parents[2]
NATIVE = {
    "span-roundtrip": native.Contract("anno-cli", "brat_dataset_roundtrip",
        "repeated_segmented_mentions_keep_their_ranges_in_each_revision"),
    "stale-revision": native.Contract("anno-cli", "corpus_read_many",
        "pinning_a_review_rejects_an_advanced_head_even_when_the_reviewed_documents_match"),
}
JOURNEYS = ("span-roundtrip", "catalog-update", "stale-revision")
FIXTURE = Path("scripts/qa/fixtures/crossdoc/catalog-revision.json")
QUERIES = FIXTURE.with_suffix(".queries.json")
POLICIES = ("balanced", "random", "muxer")
WINDOW = 16
MAX_BYTES = 8192
OFFLINE = {"ANNO_NO_DOWNLOADS": "1", "ANNO_S3_CACHE": "0", "HF_HUB_OFFLINE": "1",
           "CARGO_NET_OFFLINE": "true", "RUST_BACKTRACE": "0"}


SELECTOR_SOURCE = Path("crates/anno-eval/examples/journey_pick.rs")


def cohort_inputs():
    files = [FIXTURE, QUERIES, Path("Cargo.lock"), Path("Cargo.toml"),
             Path("crates/anno-eval/Cargo.toml"), SELECTOR_SOURCE, *(c.source for c in NATIVE.values())]
    for filename in ("journey_pilot.py", "crossdoc_cases.py",
                     "crossdoc_query_checks.py", "replay_crossdoc_case.py", "replay_crossdoc_native.py"):
        files.append(Path("scripts/qa") / filename)
    return files


def cohort():
    return hashlib.sha256(json.dumps([(str(p), native.digest(ROOT / p)) for p in cohort_inputs()]).encode()).hexdigest()


def checkout_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def checkout_provenance():
    # Capture the checkout before writing the plan, including untracked pilot sources.
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT)
    return {"revision": checkout_revision(), "checkout_dirty": bool(status)}


def built_selector(supplied):
    # Re-resolve the example from the checked-out policy on every fresh round.
    # Keeping this build separate leaves the CLI under test untouched.
    command = ["cargo", "build", "--offline", "--locked", "-p", "anno-eval", "--no-default-features",
               "--features", "eval", "--example", "journey_pick", "--message-format=json-render-diagnostics"]
    messages = subprocess.check_output(command, cwd=ROOT, text=True)
    candidates = []
    for line in messages.splitlines():
        record = json.loads(line)
        if record.get("reason") != "compiler-artifact" or not record.get("executable"):
            continue
        target = record["target"]
        if target["kind"] == ["example"] and Path(target["src_path"]) == ROOT / SELECTOR_SOURCE:
            candidates.append(Path(record["executable"]))
    if len(candidates) != 1:
        raise ValueError("Cargo did not report exactly one journey selector")
    selector, = candidates
    digest = native.digest(selector)
    if supplied is not None and native.digest(supplied) != digest:
        raise ValueError("supplied selector differs from this checkout's locked Cargo build")
    return selector, digest


def failure(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_state(path, version, *, canonical=False):
    if path.is_symlink():
        raise ValueError("expected a regular journey ledger")
    if not path.exists():
        return dict(schema=1, cohort=version, rounds=0, last_run=0, history=[])
    if not path.is_file():
        raise ValueError("expected a regular journey ledger")
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("oversized journey ledger")
    def unique(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs):
            raise ValueError("duplicate journey ledger keys")
        return obj
    state = json.loads(data, object_pairs_hook=unique)
    if (type(state) is not dict or set(state) != {"schema", "cohort", "rounds", "last_run", "history"}
            or type(state["schema"]) is not int or state["schema"] != 1 or state["cohort"] != version
            or type(state["rounds"]) is not int or type(state["last_run"]) is not int
            or not 1 <= state["rounds"] <= state["last_run"] < 2**63
            or type(state["history"]) is not list or len(state["history"]) != min(WINDOW, state["rounds"])):
        raise ValueError("invalid or incompatible journey ledger")
    for row in state["history"]:
        if type(row) is not dict or set(row) != set(POLICIES):
            raise ValueError("invalid journey policy history")
        for pick in row.values():
            if (type(pick) is not list or len(pick) != 2 or pick[0] not in JOURNEYS
                    or (pick[1] != "pass" and not failure(pick[1]))):
                raise ValueError("invalid selected journey observation")
    if canonical and data != (json.dumps(state, indent=2, sort_keys=True) + "\n").encode():
        raise ValueError("noncanonical trusted journey ledger")
    return state


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(stream.name, path)


def picks(state, seed, selector):
    rotation = state["rounds"] % len(JOURNEYS)
    order = JOURNEYS[rotation:] + JOURNEYS[:rotation]
    arms = []
    for name in order:
        values = [row["muxer"][1] for row in state["history"] if row["muxer"][0] == name]
        arms.append(dict(name=name, measured=len(values),
                         failures=sum(map(failure, values))))
    choice = subprocess.check_output([selector], input=json.dumps(arms), text=True, timeout=30).strip()
    if choice not in JOURNEYS:
        raise ValueError("unexpected Muxer choice")
    return {"balanced": order[0], "random": random.Random(seed).choice(JOURNEYS), "muxer": choice}


def measure_case(name, binary):
    if name == "catalog-update":
        proposal, proposal_hash = catalog.read_json(ROOT / FIXTURE)
        queries, query_hash = catalog.read_json(ROOT / QUERIES)
        receipt, rules = catalog.execute_with_queries(proposal, proposal_hash, binary, queries, query_hash)
        if receipt["outcome"] == "pass":
            return "pass"
        failed = [(judge, index) for judge, checks in rules.items()
                  for index, check in enumerate(checks) if not check["passed"]]
        if not failed:
            raise ValueError("missing catalog assertion failure")
        return hashlib.sha256(json.dumps(failed).encode()).hexdigest()
    contract = NATIVE[name]
    env = {**os.environ, **OFFLINE}
    harness, producer = native._artifacts(ROOT, contract, env)
    if native.digest(producer) != native.digest(binary):
        raise ValueError("integration harness and public CLI use different producers")
    run = subprocess.run([harness, "--exact", contract.test, "--format", "pretty", "--color", "never"],
                         cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, errors="replace", timeout=90)
    verdict, signature = native._verdict(contract, run)
    return "pass" if verdict == "pass" else signature.removeprefix("assertion:")


def confirmed(run):
    start = time.monotonic()
    attempts = 1
    try:
        observation = run()
        if failure(observation):
            attempts += 1
            if run() != observation:
                observation = "unavailable"
        elif observation != "pass":
            raise ValueError("invalid contract result")
    except (ValueError, OSError, subprocess.SubprocessError):
        observation = "unavailable"
    return {"observation": observation, "attempts": attempts,
            "elapsed_ms": round(1000 * (time.monotonic() - start))}


def pilot(args):
    version = cohort()
    state = read_state(args.state, version, canonical=args.prior_run is not None)
    if args.prior_run is not None and state["last_run"] != args.prior_run:
        raise ValueError("downloaded ledger does not match its verified producing run")
    args.output.mkdir(parents=True, exist_ok=False)
    if not 0 < args.run_id < 2**63 or not 0 <= args.seed < 2**64:
        raise ValueError("invalid run identity or seed")
    if state["last_run"] >= args.run_id:
        write_json(args.output / "report.json", {"status": "already-recorded-or-newer", "run": args.run_id})
        return 0
    selector, selector_hash = built_selector(args.selector)
    source = checkout_provenance()
    selected = picks(state, args.seed, selector)
    plan = dict(schema=1, cohort=version, seed=args.seed, run=args.run_id, **source, selector_sha256=selector_hash,
                previous_run=state["last_run"], previous_rounds=state["rounds"],
                selections=selected, budget_per_policy=1, census_size=len(JOURNEYS))
    write_json(args.output / "plan.json", plan)
    os.environ.update(OFFLINE)
    binary = args.anno.resolve()
    producer = native.digest(binary)
    census = {name: confirmed(lambda: measure_case(name, binary)) for name in JOURNEYS}
    if (cohort() != version or native.digest(binary) != producer
            or native.digest(selector) != selector_hash or checkout_revision() != source["revision"]):
        raise ValueError("source, oracle or producer changed during measurement")
    row = {name: result["observation"] for name, result in census.items()}
    results = {}
    for policy, name in selected.items():
        observed = row[name]
        is_hit = failure(observed)
        results[policy] = dict(case=name, outcome="confirmed-failure" if is_hit else observed,
                               unseen_in_window=is_hit and all(prior[policy] != [name, observed] for prior in state["history"]))
    measured = sum(v != "unavailable" for v in row.values())
    random_hit_rate = sum(map(failure, row.values())) / len(JOURNEYS) if measured == len(JOURNEYS) else None
    write_json(args.output / "report.json", {**plan, "producer_sha256": producer,
        "uniform_random_expected_hit_rate": random_hit_rate,
        "census": census, "measured": measured, "policy_results": results})
    if measured == len(JOURNEYS):
        selections = {policy: [name, row[name]] for policy, name in selected.items()}
        updated = {**state, "rounds": state["rounds"] + 1, "last_run": args.run_id,
                   "history": (state["history"] + [selections])[-WINDOW:]}
        write_json(args.state, updated)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as out:
                print("publish=true", file=out)
    return 0 if measured == len(JOURNEYS) else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("cohort")
    run = sub.add_parser("run")
    run.add_argument("--selector", type=Path, help="Verify this executable against the checkout's locked Cargo build")
    for arg in ("anno", "state", "output"):
        run.add_argument("--" + arg, type=Path, required=True)
    for arg in ("run-id", "seed"):
        run.add_argument("--" + arg, type=int, required=True)
    run.add_argument("--prior-run", type=int, help="verified artifact producer, or zero for a hosted cold start")
    args = parser.parse_args()
    if args.command == "cohort":
        print(cohort())
    else:
        sys.exit(pilot(args))
