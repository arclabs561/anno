# Evaluation sampling

The muxer integration schedules a bounded evaluation slice. It does not choose
the backend used by `anno` at annotation time. Runtime backend selection remains
an application-level choice.

The sampler is useful for three separate jobs:

- a fixed-panel smoke run that proves selected dataset/backend cells still run;
- an adaptive triage run that spends a small budget on recently poor or failing
  cells after there is history to learn from.
- a coverage run that fills the least-observed parts of a fixed panel.

Start with the fixed panel. It makes missing model features, incompatible
datasets, and cold caches visible before interpreting adaptive choices.

## Reproducible smoke run

Create a fresh receipt directory and keep history, results, and decision logs
there. Use local cached datasets and retain their manifest checksums, the source
commit, and `Cargo.lock` with the run. The legacy `muxer_version` log field is
`unrecorded`; use the retained lockfile to identify the resolved dependency.

```bash
mkdir -p .generated/sampler
receipt_dir=$(mktemp -d "$PWD/.generated/sampler/smoke.XXXXXX")

ANNO_MATRIX_REQUIRE_CACHED=1 \
ANNO_MATRIX_TRY_DOWNLOAD_ON_EMPTY=0 \
ANNO_MATRIX_TASK=ner \
ANNO_SAMPLE_STRATEGY=random \
ANNO_CI_SEED=42 \
ANNO_MAX_EXAMPLES=5 \
ANNO_MUXER_FIXED_BACKEND=heuristic \
ANNO_MUXER_FIXED_DATASETS=WikiGold \
ANNO_HISTORY_FILE="$receipt_dir/history.json" \
ANNO_EVAL_HISTORY="$receipt_dir/eval-results.jsonl" \
ANNO_MUXER_DECISIONS_FILE="$receipt_dir/decisions.jsonl" \
cargo test --locked -p anno-eval --lib --features "eval discourse" \
  muxer_matrix::test_randomized_matrix_sample -- --exact --ignored --nocapture
```

The matrix test is intentionally ignored because it can be slow. `--ignored` is
therefore required. Add `ANNO_ML_IN_MATRIX=1` and the relevant feature (for
example `onnx`) only after the fixed panel is known to be available.
The harness records failures and can return early when nothing is runnable;
an exit code of zero alone is not acceptance. Check that outcome rows exist,
the expected cells ran, and recorded failures are understood.

CI first runs the sampler with the heuristic backend on one checked-in sentence
with known entity spans. The fixture is the dataset registry's WikiGold format
example. It lives in a separate temporary cache, with downloads disabled, and
must produce one completed result with F1 of 1. Its receipt is separate from
evaluation results on real corpora.

External sampling restores the newest compatible checkpoint produced by this
workflow on the default branch. The artifact contains muxer windows, evaluation
JSONL, and its SQLite index. Checkpoints expire after 60 days. A new cohort
starts when the history or provenance implementation changes. PRs can read
this state. Only default-branch runs with new external observations publish it.
An active default-branch run can finish its checkpoint before the next run
starts. A retry stays read-only if that run or a newer run already published
the same cohort. Each retry retains its own sampling receipt.

Both kinds of run measure five examples per dataset from a local cache. A
checksum-pinned public WikiGold corpus supplies a real input even after cache
eviction. A cold default-branch run records a heuristic baseline on that corpus.
The synthetic offline fixture uses separate storage. Worst-first runs before
the four randomized task controls (NER, relation extraction, intra-document
coreference, and temporal extraction). On the default branch, a named control
may fetch data when that task's local cache is empty. The job summary links
the restored run and reports the first worst-first selection, its observed
result, and completed or missing coverage by task.

To replay the cross-run feedback test locally, run
`python3 scripts/qa/replay_matrix_feedback.py`. Two isolated histories use the
same seed, text, backend, and budget. Swapping which dataset scored zero must
swap the next run's first cell after checkpoint restore. A successful retry
must clear the pending observation. This test enables low-score retries
explicitly. CI's default adaptive lane targets operational failures.

To require full external coverage on a provisioned runner, use
`python3 scripts/qa/check_matrix_outcomes.py muxer-decisions.jsonl`. It
requires a completed sample with examples on each named task and an attempted
worst-first outcome.

`ANNO_MUXER_PIN_BACKEND` filters the normal candidate set, so the selected
policy still runs. `ANNO_MUXER_FIXED_BACKEND` is for a direct backend smoke
test: it bypasses the policy and must not be used to evaluate it. Similarly,
`ANNO_MUXER_FIXED_DATASETS` fixes the dataset panel when comparing policies.

## Adaptive triage

Use the same cache, fixed datasets, candidate panel, example cap, and seed set
for a random control and a `worst-first` run. Isolate each policy's history and
logs. Compare failure discovery and distinct cell coverage at the same budget,
then inspect the JSONL with `muxer_audit` or `anno muxer decisions`.

`worst-first` prioritizes observed badness; compare its discovery rate with
the control rather than assuming an advantage. It does not maximize average F1. The default
`ml-only` path uses EXP3-IX. The optional MAB and LinUCB paths are experiments,
not a claim of general improvement.

Muxer 0.5.3 is enabled for evaluation with `serde`, `stochastic`, and
`contextual`. The sampler uses its bounded outcome history, candidate summaries,
and policy helpers. It does not need every muxer policy or feature: extra knobs
make a small matrix harder to interpret.

## Coverage

Use `--mode coverage` or `--strategy estimate` when the question is which
eligible fixed-panel cells have been run least often. The implementation adds
each backend's historical observation counts across the selected datasets and
chooses the lowest total. Counts are scoped to a recorded policy cohort: task,
dataset, example cap, cache policy, enabled evaluator features, and the
primary-score contract. Seed remains in each receipt but does not split coverage
cohorts, so repeated-seed panels can accumulate coverage. Legacy and nonmatching receipts do not count. Source
revision deliberately does not define a cohort, so a new revision can fill the
same panel rather than starting its coverage from zero.

This is a coverage heuristic, not an artifact-controlled quality comparison:
model-artifact and provider receipts are not available for every candidate at
selection time. It does not estimate uncertainty, variance, recency, or a
regression probability. Keep the dataset panel fixed and inspect the resulting
receipt before treating missing cells as the next work item. Coverage decisions
now record each candidate's count and deterministic tie-break selection; each
outcome carries the same opaque observation ID as its evaluation-history row.

## Interpreting receipts

A low F1 is an observed quality result, not automatically a code regression.
Dataset subset variation, model availability, and changed candidate panels can
also change it. Record the actual candidates and datasets in the decision log;
the sampler observes only selected cells, so absence from a log is not evidence
that a cell is healthy.

Drift and change monitors need enough independent-looking history to be useful.
Do not infer a distribution shift from a handful of selected runs, and do not
enable monitor thresholds merely because a policy exposes them.

Decision logs support diagnosis, not off-policy evaluation by themselves.
IPS-style comparisons require valid, positive logging propensities for every
selected action and a predeclared target policy. The current mixed deterministic
and adaptive sampler logs do not provide that common contract.

For the surrounding evaluation architecture, see [Architecture](ARCHITECTURE.md).
