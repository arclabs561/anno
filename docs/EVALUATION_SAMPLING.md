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

## Checked-in public evidence

The [compact sampler summary](eval/muxer-public-summary.json) records two offline
NER backends on pinned WikiGold. Each cell reports three completed seed trials,
the number of example evaluations and the unweighted mean of trial-level
strict span-and-type F1. Sentences can recur across trials. The evaluator Git
revision and the public corpus checksum identify the code and data used for the snapshot.
Trusted default-branch checkpoints supply operational feedback to the sampler.
The public summary is for inspecting changes across revisions.

To refresh, start from a clean checkout and run the public replay. The exporter
verifies the input checksum, evaluates each backend at seeds 41, 42 and 43
using 32 examples per trial, and joins completed outcomes to the evaluator
journal by observation ID. Repeated receipts count once per backend and seed.
Only the fixed public cell names and aggregate metrics enter the snapshot.

```bash
cache=$(mktemp -d)
ANNO_CACHE_DIR="$cache" bash scripts/qa/prepare_matrix_wikigold.sh
python3 scripts/qa/export_muxer_summary.py "$cache/datasets/WikiGold.cache" > "$cache/summary.json"
diff -u docs/eval/muxer-public-summary.json "$cache/summary.json"
```

After inspecting the revision and per-cell differences, copy the refreshed JSON
into `docs/eval/muxer-public-summary.json` to retain it in version control.

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

For a prospective comparison, record the adaptive and random first picks before
running every cell on a small fixed panel. Score both picks against that same
panel. Count newly observed backend failures separately from repeat dataset
and I/O failures. A run that measures only its chosen cell cannot tell whether
the other policy would have found a failure sooner.

Run `python3 scripts/qa/compare_matrix_picks.py --output-dir /tmp/anno-matched`
with an unused output path for an offline prospective control. It builds the
test binary from cached dependencies and runs both production selectors through
their normal first-pick path. Each reads an independent copy of the same
frozen history and neutral dataset cache. The experiment writes `picks.json`
before any held-out scoring. Separate exact-cell runs then write `panels.json`
and retain the observation IDs. Both mirrored histories use eight seeds by
default. Each panel has one NER task, one heuristic backend and two dataset
slots containing the same synthetic text. Only the gold annotations vary.
The persistent and switched cases show whether a policy rediscovers a known
quality failure or detects it after it moves. Neither creates a backend crash
or an I/O error. The experiment does not estimate discovery rates on live data.

In the seed-zero control, giving the WikiGold slot the earlier zero-F1 result
made both selectors choose WikiGold. Moving that result to the CoNLL2003Sample
slot switched the worst-first pick to CoNLL2003Sample. The random pick stayed
WikiGold. Across eight seeds and two mirrored histories, worst-first found the
zero-F1 slot on its first pick in 16/16 persistent cases and 0/16 switched
cases. Random found it in 8/16 cases under either condition. The checked-in
[paired control](../scripts/qa/compare_matrix_picks.py) reproduces these
synthetic decisions. Dataset names here designate fixture slots.

### Reserve part of a trial for coverage

For a bounded trial, `scripts/qa/run_matrix_slot.py` runs one existing selector
per explicit, one-based slot. In the example, slots 1, 3, and 4 use
`worst-first`, and slot 2 uses `estimate` with the same persistent cache.

```bash
for slot in 1 2 3 4; do
  ANNO_MATRIX_TASK=ner ANNO_MATRIX_REQUIRE_CACHED=1 ANNO_NO_DOWNLOADS=1 \
    python3 scripts/qa/run_matrix_slot.py --policy mixed --slot "$slot" \
      --period 4 --coverage-slot 2 --seed "$((100 + slot))" \
      --cache-dir .generated/mixed-cache --receipts ".generated/mixed/slot-$slot"
done
```

The period repeats if the trial continues. The reserved fraction counts
selection slots. Record observed cost and completed task/dataset/backend cells
when choices run different numbers of examples. Each fresh receipt directory
contains the requested schedule, native decision/outcome rows, and the strategy
observed in those rows. Use `--test-binary PATH` to reuse a compiled evaluator.
Use `--plan-only` with a copy of the starting state to inspect a choice before
revealing its outcome.

Run `--policy random` with the same slot numbers, seeds, and candidate panel
in a separate copy of the initial cache. Preserve both the journal and scoped
coverage state between slots within a policy. Comparing discovery at the same
budget measures triage. Estimating overall quality requires a separate random
sample or an independently scored complete panel.

## Coverage

Use `--mode coverage` or `--strategy estimate` when the question is which
eligible fixed-panel cells have the fewest recorded evaluation receipts. The
implementation sums each backend's receipts across the selected datasets and
chooses the lowest total. Skips, dataset preflight failures, and successful
unscored runs all count. To measure scored coverage, count successful receipts
with a finite primary F1. Counts are scoped to a recorded policy cohort: task,
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

Backend rolling windows exclude skips and failed dataset preflights. An executed
backend failure contributes a hard failure. A successful run without a finite
primary F1 records execution and cost with unknown quality. A measured zero
qualifies as low quality for a positive junk threshold.

The history index stores finite primary scores from successful evaluations,
including coreference `conll_f1` and relation `strict_f1`. For a previously
created SQLite index, `anno history --history-file PATH rebuild` recovers these
scores from readable evaluator metadata in the original JSONL. Old backend
windows retain their recorded outcomes until new attempts displace them.

In worst-first mode, comparable unresolved failures have priority. When none is
eligible, one quarter of stable seed buckets also consider cells with at least
two recovered episodes of the same kind in the retained window. An unscored
completion can close an execution failure. Only a scored measurement can close
a low-F1 episode. For recovered cells the draw weight is the configured
hard/soft weight times `episodes / (relevant attempts + 4)`.
The decision records the reason, number of recovered episodes and latest
supporting failure ID. Other buckets use the usual worst-first exploration.

Two consecutive attempts in a cell with the same explicit dataset or download
error share a retry budget with other repeated infrastructure failures. One
eighth of seed buckets give them normal priority. In the remaining buckets
the sampler prefers an alternative backend when one is eligible. A sole
eligible backend still runs. A changed error or recovery starts a new episode.
A repeated-infrastructure retry records both observation IDs and the failure kind.

## What the selector learns

A test cell identifies a task, dataset and backend under a comparable evaluation configuration. The selection policy uses observations from earlier attempts to allocate a bounded test budget. The library also supports multi-arm routing, contextual selection, change monitoring and triage. Anno uses particular pieces of [muxer 0.5.3](https://docs.rs/muxer/0.5.3/muxer/) for evaluation. Production model selection remains with the caller.

Anno wires EXP3-IX, an alternative MAB selector, and LinUCB into its ML-only sampling path. It uses bounded history windows, outcome and cost summaries, and optional drift monitors. The library also offers Thompson sampling, sticky routing, a stateful Router with snapshots, and a triage session. These have no active Anno selection path. Enabling more policies would divide an already small observation budget. The [locked muxer API](https://docs.rs/muxer/0.5.3/muxer/) and [Anno integration](../crates/anno-eval/src/matrix_muxer_ci.rs) show the distinction.

| Question | Appropriate comparison |
| --- | --- |
| Which checks find regressions sooner? | Compare first discovery and unique failures at equal runner-minute budgets. Record time spent fetching data separately from backend execution. Historical duration, time since the last execution, and previous outcomes are useful test-prioritization inputs. [Spieker et al.](https://dl.acm.org/doi/10.1145/3092703.3092709) |
| Which interactions have we missed? | Reserve coverage for compatible task, backend and data combinations, including formats and languages. A combinatorial design can cover selected interactions within a fixed budget. Its coverage guarantee applies only to the dimensions and interaction strength specified. [NIST SP 800-142](https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-142.pdf) |
| Would a different policy have done better? | Seal competing picks before running the same small held-out panel, or design a randomized logging experiment. Ordinary adaptive logs contain outcomes only for chosen cells. Unbiased replay needs a suitable randomized logging policy and assumptions about the observations. [Li et al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/published-3.pdf) |

Keep live-web sampling outcomes separate from labeled dataset F1. A broken feed is evidence about retrieval. A wrong labeled span is evidence about extraction or scoring. Track them separately so a publisher outage cannot teach the selector that a model regressed.

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

## Rescore saved entity predictions

Save one grounded document per source. Supply the same source text and gold annotations to compare entity matching and character masking without running the model again.

```sh
anno extract --model pattern --file report.txt --format grounded > report.predictions.json
anno eval --predictions report.predictions.json --file report.txt --gold-file report.gold.jsonl --json
anno eval --predictions report.predictions.json --file report.txt --gold-file report.gold.jsonl --metric char-masking --json
```

A gold JSONL record has an `entities` array with `text`, `label`, `start`, and `end`. Offsets are half-open Unicode character ranges. An optional `segments` array identifies the covered characters of a discontinuous mention. For example, this record marks A and C while leaving the intervening B unlabeled.

```json
{"text":"😀A B C","entities":[{"label":"PER","start":1,"end":6,"segments":[{"start":1,"end":2},{"start":5,"end":6}],"text":"A C"}]}
```

The ordered ranges must agree with the outer bounds and the original source. The surface can join the selected pieces with spaces or preserve the exact bounding substring from a legacy record. Empty BRAT boundary markers add no covered characters. Each scored mention must cover at least one character. A missing or null `segments` field identifies a contiguous span. Entity matching compares the covered positions and label. Boundary IoU, character masking, the HTML highlights and the detailed text error counts use the same positions.

Replay validates the stored signals and matches the entire source text, including whitespace. The JSON report retains the grounded document ID, the SHA-256 of the evaluated UTF-8 text, and the saved signal IDs in the matches. Model and inference time are null during replay. The grounded artifact contains source text and detected surfaces, so store it with the same access controls as the original.

Gold JSONL records may carry their source `text`, `source_sha256`, and a document ID (`id`, `doc_id`, or `document_id`). The evaluator checks any supplied text or hash against its input, requires one document ID throughout the file, and checks that ID against the saved predictions. To preserve a corpus document ID, use `anno batch --stdin --model pattern --format grounded` with an input record such as `{"id":"report-v1","text":"…"}`. Legacy entity-only gold records still validate each annotated surface. Identical text from distinct revisions can be distinguished only when the records carry distinct document IDs.
