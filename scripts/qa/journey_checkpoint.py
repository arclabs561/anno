"""Find the latest successful default-branch producer of this exact journey cohort."""
import json
import os
import re
import subprocess

WORKFLOW = ".github/workflows/journey-sampling.yml"
MAX_ARCHIVE_BYTES = 16384


def select(artifacts, name, repo, repo_id, branch, workflow_id, lookup):
    candidates = sorted(artifacts, key=lambda a: (a["workflow_run"]["id"], a["id"]), reverse=True)
    for artifact in candidates:
        producer = artifact["workflow_run"]
        if (artifact["name"] != name or artifact["expired"] or producer["head_branch"] != branch
                or producer["repository_id"] != repo_id or producer["head_repository_id"] != repo_id):
            continue
        run_id = producer["id"]
        run = lookup(run_id, None)
        # A retry changes the run's current status. Check the actual attempt that
        # owned the archive, identified by the window in which GitHub created it.
        while artifact["created_at"] < run["run_started_at"] and run["run_attempt"] > 1:
            run = lookup(run_id, run["run_attempt"] - 1)
        if (run["id"] != run_id or run["workflow_id"] != workflow_id or run["path"] != WORKFLOW
                or run["event"] not in {"schedule", "workflow_dispatch"}
                or run["head_branch"] != branch or run["head_sha"] != producer["head_sha"]
                or run["head_repository"]["id"] != repo_id or run["repository"]["full_name"] != repo
                or run["status"] != "completed" or run["conclusion"] != "success"
                or not run["run_started_at"] <= artifact["created_at"] <= run["updated_at"]):
            continue
        if not 0 < artifact["size_in_bytes"] <= MAX_ARCHIVE_BYTES:
            raise ValueError("trusted journey archive exceeds its expected size")
        return run_id, artifact["id"]
    return None


def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    cohort = os.environ["JOURNEY_COHORT"]
    if not re.fullmatch("[a-f0-9]{64}", cohort):
        raise ValueError("invalid cohort")
    def api(endpoint, *options):
        return json.loads(subprocess.check_output(["gh", "api", f"repos/{repo}/actions/{endpoint}", *options]))
    workflow = api("workflows/journey-sampling.yml")
    if workflow["path"] != WORKFLOW:
        raise ValueError("unexpected workflow identity")
    name = f"journey-ledger-{cohort}"
    pages = api(f"artifacts?name={name}&per_page=100", "--paginate", "--slurp")
    result = select([a for page in pages for a in page["artifacts"]], name, repo,
                    int(os.environ["GITHUB_REPOSITORY_ID"]), os.environ["DEFAULT_BRANCH"], workflow["id"],
                    lambda run_id, attempt: api(f"runs/{run_id}" + (f"/attempts/{attempt}" if attempt else "")))
    run_id, artifact_id = result or (0, 0)
    with open(os.environ["GITHUB_OUTPUT"], "a") as out:
        print(f"prior_run={run_id}\nartifact_id={artifact_id}", file=out)
    print(f"Restoring reviewed journey state from run {run_id}." if run_id else
          "Cold start: no compatible successful default-branch journey ledger was available.")


if __name__ == "__main__":
    main()
