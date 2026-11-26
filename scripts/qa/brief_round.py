"""Prepare a coordinator's challenge from an existing QA run."""
import argparse
import json
import math
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

from round_trace import initialize
from domain_coverage import render_coverage, render_domain_scores
from run_panel import render_markdown

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "ADVERSARIAL_QA.md"


def measured_counts(values, *keys):
    """Legacy or malformed observations cannot support a structural diagnosis."""
    if not isinstance(values, dict):
        return None
    result = [values.get(key) for key in keys]
    if not all(type(v) in (int, float) and 0 <= v <= 2**53 and int(v) == v for v in result):
        return None
    return tuple(map(int, result))


def registry_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.+-]{1,128}", value):
        raise ValueError("expected registry identifiers for the dataset and backend")
    return value


def outcome_evidence(path):
    lines = []
    selected = {}
    observed = set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            scope = tuple(row.get(k) for k in ("run_id", "strategy", "slice"))
            scoped = all(isinstance(value, str) and value for value in scope)
            if row.get("record_type") != "outcome":
                # Some selectors name only the first backend; these rows give a lower bound.
                if scoped and "record_type" not in row and row.get("chosen"):
                    if not isinstance(row["chosen"], str) or not isinstance(row.get("datasets"), list):
                        raise ValueError("invalid matrix selection")
                    backends = [registry_id(value) for value in row["chosen"].split(",")]
                    for value in row["datasets"]:
                        dataset = registry_id(value)
                        for backend in backends:
                            selected.setdefault((scope, dataset, backend), number)
                continue
            dataset, backend = registry_id(row["dataset"]), registry_id(row["backend"])
            if scoped:
                observed.add((scope, dataset, backend))
            score = row.get("primary_f1")
            if row.get("hard_junk") or row.get("cost_units") == 0 or score is None:
                lines.append(f"- Line {number}, {dataset} / {backend}: no usable score. Check execution first.")
                continue
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid primary F1")
            if type(row.get("ok")) is not bool:
                raise ValueError("invalid outcome status")
            lines.append(f"- Line {number}, {dataset} / {backend}: ok={row['ok']}, primary F1={score}")
            counts = row.get("structural_counts")
            coref = measured_counts(counts, "gold_coref_links", "reachable_gold_links",
                                    "matched_coref_links", "predicted_coref_links")
            if coref:
                gold, reachable, matched, predicted = coref
                if reachable > gold or matched > min(predicted, gold):
                    raise ValueError("inconsistent coreference link counts")
                lines.append(f"  Resolver inputs cover both ends of {reachable}/{gold} gold links. "
                             f"The resolver output contains {matched} correct and {predicted-matched} extra links.")
                if gold - reachable > reachable:
                    lines.append("  Freeze a mention-proposal comparison on the same documents and resolver. "
                                 "Check extra links on a second, authorized source and no-referent controls.")
                elif reachable > matched:
                    lines.append(f"  At least {reachable-matched} reachable links were missed. "
                                 "Hold mention proposals fixed and vary only the resolver.")
            elif (links := measured_counts(counts, "gold_coref_links", "matched_coref_links",
                                           "predicted_coref_links")) is not None:
                gold, matched, predicted = links
                if matched > min(predicted, gold):
                    raise ValueError("inconsistent coreference link counts")
                lines.append(f"  Output links: {matched} correct, {predicted-matched} extra, "
                             f"{gold-matched} missed. Resolver input coverage was unobserved.")
            temporal = measured_counts(counts, "unmatched_predictions_overlapping_missed_gold",
                                       "matched_spans", "predicted_spans", "gold_empty_sentences",
                                       "gold_empty_sentences_with_predictions")
            if temporal:
                near, matched, predicted, empty, alarms = temporal
                if matched > predicted or near > predicted - matched or alarms > empty:
                    raise ValueError("inconsistent temporal span counts")
                gold_count = measured_counts(counts, "gold_spans")
                if gold_count is not None:
                    gold, = gold_count
                    if matched > gold:
                        raise ValueError("inconsistent temporal span counts")
                    if (row.get("task") == "temporal" and backend == "pattern"
                            and gold > matched and predicted == matched and empty and not alarms):
                        lines.append(f"  Pattern missed {gold-matched}/{gold} gold spans. "
                                     "Probe weekday recall with an authored name, title and numeric-date control: "
                                     "just qa-spans target/debug/anno pattern "
                                     "scripts/qa/fixtures/weekday-span-assertions.json")
                if near:
                    lines.append(f"  {near}/{predicted-matched} unmatched predictions overlap missed gold. "
                                 "Compare native and reference decoding on identical frozen model outputs.")
                if empty:
                    lines.append(f"  {alarms}/{empty} gold-empty sentences triggered. Preserve that control.")
                else:
                    lines.append("  No gold-empty sentences were scored. Reserve a labeled negative panel.")
    for key, number in selected.items():
        if key not in observed:
            _, dataset, backend = key
            lines.append(f"- Line {number} selects {dataset} / {backend}. This file contains no outcome "
                         "with a matching run, strategy and slice. Execution and score are unknown.")
    if not lines:
        raise ValueError("matrix receipt contains no outcome or identifiable selection rows")
    return "\n".join(lines)


REPLAY_DIGESTS = {"source_sha256", "oracle_sha256", "producer_sha256"}


def check_replay_digests(receipt):
    if not all(isinstance(receipt[k], str) and re.fullmatch("[a-f0-9]{64}", receipt[k])
               for k in REPLAY_DIGESTS):
        raise ValueError("invalid source, oracle or producer digest")


def crossdoc_evidence(receipt):
    if set(receipt) != REPLAY_DIGESTS | {"kind", "runner", "passed", "failed", "outcome"}:
        raise ValueError("unexpected cross-document receipt fields")
    runner, passed, failed = receipt["runner"], receipt["passed"], receipt["failed"]
    if runner not in ("cli", "eval"):
        raise ValueError("invalid cross-document runner")
    check_replay_digests(receipt)
    if (type(passed) is not int or type(failed) is not int or min(passed, failed) < 0
            or not 1 <= passed + failed <= 256):
        raise ValueError("expected 1 to 256 evaluated assertions")
    expected_outcome = "fail" if failed else "pass"
    if receipt["outcome"] != expected_outcome:
        raise ValueError("cross-document outcome disagrees with assertion counts")
    return (f"Cross-document {runner} case outcome {expected_outcome}. "
            f"Evaluated {passed + failed} assertions ({passed} passed, {failed} failed).")


def query_case_evidence(receipt):
    if set(receipt) != {"kind", "authored", "consumer", "proposal_sha256", "outcome"}:
        raise ValueError("unexpected query case receipt fields")
    authored, reader = receipt["authored"], receipt["consumer"]
    if (not isinstance(authored, dict) or authored.get("kind") != "anno.qa.crossdoc-case.v1"
            or authored.get("runner") != "cli"):
        raise ValueError("expected an authored CLI case")
    summary = crossdoc_evidence(authored)
    if not isinstance(reader, dict) or set(reader) != {
        "rubric", "scorer_sha256", "holdout_sha256", "passed", "failed", "outcome"
    } or reader["rubric"] != "anno.qa.query-postings.v1":
        raise ValueError("unexpected reader rubric or fields")
    hashes = [receipt["proposal_sha256"], reader["scorer_sha256"], reader["holdout_sha256"]]
    if not all(isinstance(value, str) and re.fullmatch("[a-f0-9]{64}", value) for value in hashes):
        raise ValueError("invalid query case digest")
    passed, failed = reader["passed"], reader["failed"]
    if (type(passed) is not int or type(failed) is not int or min(passed, failed) < 0
            or not 1 <= passed + failed <= 32):
        raise ValueError("expected 1 to 32 evaluated reader queries")
    combined = "fail" if failed or authored["failed"] else "pass"
    if reader["outcome"] != ("fail" if failed else "pass") or receipt["outcome"] != combined:
        raise ValueError("query case outcomes disagree with the counts")
    return f"{summary} Evaluated {passed + failed} reader queries ({passed} passed, {failed} failed)."


def replay_evidence(receipt):
    from replay_feedback import ADAPTERS, validate_signature

    outcome = receipt["outcome"]
    if outcome in ("error", "unavailable"):
        if (set(receipt) != {"outcome", "detail"} or not isinstance(receipt["detail"], str)
                or not 1 <= len(receipt["detail"]) <= 256):
            raise ValueError("unexpected unmeasured replay receipt")
        return f"Reviewed replay reported {outcome}. Evaluated 0 cases."
    if set(receipt) != REPLAY_DIGESTS | {"corpus", "case", "signature", "outcome"}:
        raise ValueError("unexpected reviewed replay receipt fields")
    check_replay_digests(receipt)
    if (not isinstance(receipt["corpus"], str) or receipt["corpus"] not in ADAPTERS
            or not isinstance(receipt["case"], str) or not 1 <= len(receipt["case"]) <= 128):
        raise ValueError("invalid reviewed replay corpus or case")
    validate_signature(ADAPTERS[receipt["corpus"]], outcome, receipt["signature"])
    return f"Reviewed replay evaluated 1 case. Outcome {outcome}."


def benchmark_evidence(receipt):
    """Project recorded benchmark fields into a source-free reviewer handoff."""
    rows = receipt["results"]
    if not isinstance(rows, list):
        raise ValueError("expected benchmark result rows")
    lines = [f"Benchmark result rows: {len(rows)}."]
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("expected a benchmark result object")
        dataset, backend = row.get("dataset"), row.get("backend")
        if not all(isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_.+-]{1,128}", v)
                   for v in (dataset, backend)):
            raise ValueError("invalid benchmark dataset or backend identifier")
        success = row.get("success")
        if type(success) is not bool:
            raise ValueError("expected a benchmark success flag")
        provenance = row.get("provenance") or {}
        if not isinstance(provenance, dict):
            raise ValueError("invalid benchmark provenance")
        observed = provenance.get("dataset") or {}
        if not isinstance(observed, dict):
            raise ValueError("invalid benchmark dataset provenance")
        tier = observed.get("data_source")
        if tier not in ("S3 cache", "local cache", "original URL", "embedded",
                        "skipped (unavailable)", "unavailable before dataset load"):
            tier = "unrecorded"
        content_digest = observed.get("content_sha256")
        version = observed.get("version")
        digest = "unrecorded"
        if isinstance(content_digest, str) and re.fullmatch(r"[0-9a-f]{64}", content_digest):
            digest = "sha256:" + content_digest
        elif isinstance(version, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", version):
            digest = version
        lines.append(f"- {dataset} / {backend}: success={success}, source={tier}, digest={digest}")
    return "\n".join(lines)


def arm_counts(values, *keys):
    counts = measured_counts(values, *keys)
    if counts is None:
        raise ValueError(f"expected nonnegative integer counts for {', '.join(keys)}")
    return counts


def paired_arm_evidence(cohorts):
    """Compare final outputs within each reported cohort without inferring item-level pairing."""
    if not isinstance(cohorts, dict) or not cohorts:
        raise ValueError("expected paired arm cohorts")
    lines = ["Aggregate arm comparison. Verify common item IDs and cohort separation in the source ledgers."]
    for cohort, group in cohorts.items():
        registry_id(cohort)
        if not isinstance(group, dict) or "common" not in group or "baseline" not in group or len(group) < 3:
            raise ValueError("expected common counts, a baseline and at least one alternative per cohort")
        common = group["common"]
        docs, gold = arm_counts(common, "documents", "gold")
        if not docs and gold:
            raise ValueError("nonzero gold count on an empty cohort")
        sample_digests = set()
        for entry in group.values():
            if isinstance(entry, dict) and "sample_sha256" in entry:
                digest = entry["sample_sha256"]
                if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
                    raise ValueError("invalid cohort sample digest")
                sample_digests.add(digest)
        if len(sample_digests) > 1:
            raise ValueError("arms report different cohort sample digests")
        eligible = gold
        if "gold_scored" in common:
            eligible, = arm_counts(common, "gold_scored")
            if eligible > gold:
                raise ValueError("scored gold exceeds cohort gold")
        lines.append(f"\n{cohort} ({docs} documents, {gold} gold items)")
        for name in ["baseline", *(k for k in group if k not in ("common", "baseline"))]:
            registry_id(name)
            arm = group[name]
            tp, fp, fn = arm_counts(arm, "tp", "fp", "fn")
            if tp + fn != gold or tp > eligible or (not docs and tp + fp):
                raise ValueError("final counts disagree with cohort totals")
            for key, expected in (("documents", docs), ("gold", gold), ("predicted", tp + fp)):
                if key in arm and arm_counts(arm, key) != (expected,):
                    raise ValueError(f"inconsistent arm {key}")
            denom = 2 * tp + fp + fn
            f1 = f"{2 * tp / denom:.4f}" if denom else "undefined"
            line = f"- {name}: TP {tp}, FP {fp}, FN {fn}; {tp + fp} predicted, micro F1 {f1}."
            if name == "baseline":
                base = tp, fp, fn
            else:
                line += f" Change from baseline: TP {tp-base[0]:+d}, FP {fp-base[1]:+d}, FN {fn-base[2]:+d}."
            lines.append(line)
            if "gold_solver" in arm:
                reached, = arm_counts(arm, "gold_solver")
                if not tp <= reached <= eligible:
                    raise ValueError("solver gold inconsistent with scored gold or final matches")
                lines.append(f"  Gold items in solver input: {reached}/{gold} total.")
            if "typed_solver_edges" in arm:
                edges, = arm_counts(arm, "typed_solver_edges")
                lines.append(f"  Typed edges in solver input: {edges}.")
            churn = ("new_solver_gold", "lost_solver_gold", "retained_solver_gold")
            if any(k in arm for k in churn):
                gained, lost, kept = arm_counts(arm, *churn)
                prev, = arm_counts(group["baseline"], "gold_solver")
                curr, = arm_counts(arm, "gold_solver")
                if name == "baseline" or kept + lost != prev or kept + gained != curr or kept + lost + gained > eligible:
                    raise ValueError("inconsistent solver gold gains and losses")
                lines.append(f"  Solver input gained {gained}, lost {lost} and retained {kept} gold items.")
            if "decode_seconds" in arm:
                secs = arm["decode_seconds"]
                if type(secs) not in (int, float) or not 0 <= secs < math.inf:
                    raise ValueError("invalid decode time")
                lines.append(f"  Reported decode time: {secs:.3f} seconds.")
    return "\n".join(lines)


def evidence(directory):
    if directory.is_file():
        if directory.suffix == ".jsonl":
            return outcome_evidence(directory)
        receipt = json.loads(directory.read_text(encoding="utf-8"))
        if isinstance(receipt, dict):
            if "kind" not in receipt and "paired_arms" in receipt:
                return paired_arm_evidence(receipt["paired_arms"])
            if "kind" not in receipt and "results" in receipt and "summary" in receipt:
                return benchmark_evidence(receipt)
            if receipt.get("kind") == "anno.qa.crossdoc-query-case.v1":
                return query_case_evidence(receipt)
            if receipt.get("kind") == "anno.qa.crossdoc-case.v1":
                return crossdoc_evidence(receipt)
            if "kind" not in receipt and "outcome" in receipt:
                return replay_evidence(receipt)
            if receipt.get("kind") == "anno.qa.domain-spans":
                return render_domain_scores(receipt)
        return render_coverage(receipt)
    panel = directory / "panel-summary.json"
    schedule = directory / "schedule.json"
    if panel.exists() and schedule.exists():
        raise ValueError("give the directory of a single panel or matrix slot")
    if panel.is_file():
        return render_markdown(json.loads(panel.read_text())["results"]).rstrip()
    if not schedule.is_file():
        raise ValueError("expected panel-summary.json or schedule.json in the run directory, or a supported QA receipt file")
    plan = json.loads(schedule.read_text())
    lines = [f"Matrix slot {plan['slot']}, requested {plan['requested_strategy']}, seed {plan['seed']}."]
    result = directory / "result.json"
    if not result.is_file():
        return lines[0] + " The slot did not produce a validated result. Investigate execution first."
    run = json.loads(result.read_text())
    if plan["requested_strategy"] == "reviewed-replay":
        case = plan["selected_case"]
        if plan["plan_only"]:
            return lines[0] + " Selected one reviewed case, oracle replay pending."
        actual = json.loads((directory / "replay.json").read_text())
        from replay_feedback import key
        outcome_id = actual.pop("observation_id")
        if (key(actual) != outcome_id or outcome_id != run["outcome_observation_id"]
                or any(case[k] != actual[k] for k in
                       ("corpus", "case", "scope", "family", "source_sha256", "oracle_sha256"))):
            raise ValueError("saved oracle outcome does not match the selected witness")
        outcome = actual["outcome"]
        if (outcome not in ("pass", "fail") or type(actual["replays"]) is not int
                or actual["replays"] != (2 if outcome == "fail" else 1)):
            raise ValueError("inconsistent reviewed replay outcome or attempt count")
        attempts = "2 attempts" if outcome == "fail" else "1 attempt"
        return lines[0] + f" Evaluated 1 reviewed case in {attempts}. Oracle result: {outcome}."
    if not run["strategy_matches"]:
        return lines[0] + " The observed strategy differed. Exclude this slot from policy comparisons."
    if run["plan_only"]:
        return lines[0] + " Selection only. No backend was evaluated."
    return lines[0] + "\n" + outcome_evidence(directory / "decisions.jsonl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("claim")
    parser.add_argument("--trace", type=Path, help="create a local round trace at a new SQLite path")
    parser.add_argument("--compact", action="store_true", help="include the next review step and a guide reference")
    args = parser.parse_args()
    try:
        summary = evidence(args.receipts)
        guidance = (
            "Check the claim against the original receipt. Record one counterexample or the next "
            "replay that would distinguish the explanations. For independent labeling, share the "
            "input and contract before the producer output. See docs/ADVERSARIAL_QA.md in this checkout."
            if args.compact else GUIDE.read_text()
        )
        handoff = (f"# Coordinator handoff\n\nClaim to investigate: {args.claim}\n\n"
                   f"Starting evidence\n\n{summary}\n\n{guidance}\n")
        if args.trace:
            trace_id = initialize(args.trace, handoff)
            print(f"QA trace {trace_id} at {args.trace}", file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Cannot prepare QA round: {exc}\n")
    print(handoff, end="")


if __name__ == "__main__":
    main()
