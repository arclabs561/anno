"""Public GAP receipts contain aggregates and digests of the ordered local replay."""

import hashlib
import json
from pathlib import Path


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("ascii")).hexdigest()


def stages(details):
    result = {key: details[key] for key in (
        "data_sha256", "revision", "split", "documents", "gold_pairs",
        "name_boundary_observations", "stages", "endpoint_status_pairs",
        "direct_pair_when_retained", "output_errors",
    )}
    result["pair_rows_sha256"] = digest(details["rows"])
    result["false_link_paths_sha256"] = digest(details["false_links"])
    return result


def possessives(details):
    result = {key: details[key] for key in (
        "revision", "validation_sha256", "test_sha256", "removed_shared_test_url_rows",
        "distinct_urls", "ids_sha256",
    )}
    for name in ("baseline", "candidate"):
        run = details[name]
        result[name] = {"counts": run["counts"], "proposed_both_gold": run["proposed_both_gold"]}
    changed = details["changed_target_rows"]
    result.update(changed_target_count=len(changed), changed_target_rows_sha256=digest(changed))
    return result


def decisions(details):
    result = {key: details[key] for key in ("input_sha256", "binary_sha256", "holdout_checkpoints")}
    result["panels"] = {}
    for name, panel in details["panels"].items():
        arms = {}
        for label, arm in panel["arms"].items():
            summary = {"evidence": arm["evidence"], "counts": arm["counts"]}
            if "per_id" in arm:
                summary["empty_successes"] = arm["empty_successes"]
                summary["observations_sha256"] = digest([arm["per_id"], arm["refusals"]])
            arms[label] = summary
        result["panels"][name] = {
            "article_count": panel["article_count"], "id_list_sha256": panel["id_list_sha256"],
            "gold_pairs_sha256": digest(panel["gold_pairs"]), "arms": arms,
        }
    if "article_holdout_candidate" in details:
        candidate = details["article_holdout_candidate"]
        arms = {}
        for mode, arm in candidate["arms"].items():
            changes = arm["changed_observations"]
            arms[mode] = {
                "counts": arm["counts"], "coverage_pronoun_A_B": arm["coverage_pronoun_A_B"],
                "paired_exact_decisions": arm["paired_exact_decisions"],
                "changed_rows": len(changes), "changed_observations_sha256": digest(changes),
            }
        result["article_holdout_candidate"] = {"binary_sha256": candidate["binary_sha256"], "arms": arms}
    return result


def write(output, summary, details, local_path):
    if local_path is not None:
        root = Path(__file__).resolve().parents[2]
        if local_path.resolve().is_relative_to(root) or local_path.resolve() == output.resolve():
            raise ValueError("local details require a separate file outside the repository")
        local_path.write_text(json.dumps(details, indent=2) + "\n")
    output.write_text(json.dumps(summary, indent=2) + "\n")
