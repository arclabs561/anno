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

In “Maya arrived Tuesday. She left Friday,” the `intra-coref` task scores
whether “Maya” and “She” belong to the same mention chain. A grounded document
represents that grouping as a `Track`. The `temporal` task scores the text
spans “Tuesday” and “Friday” as time expressions.

## Inspect dataset eligibility

Run `anno sampler eligibility --task ner --require-cached` before comparing policies on a local panel. Its JSON lists each registered dataset for that task, whether the standard loader sees a local cache hit, and the first policy exclusion. For a compact list of locally cached sources and why each will or will not enter selection:

```bash
anno sampler eligibility --task ner --require-cached |
  jq '.datasets[] | select(.locally_cached) | {dataset, eligible, exclusion}'
```

An automatic run applies the registry's source policy before checking the cache. For example, cached WNUT17 can report `non_automatable_source`. To include a known, verified local source in an offline comparison, scope `ANNO_MATRIX_INCLUDE_NON_AUTOMATABLE=1`, `ANNO_NO_DOWNLOADS=1` and `ANNO_S3_CACHE=0` to both the preflight and run commands, using an isolated dataset cache. Language/domain environment pins and fast-profile limits affect the reported pool. When S3 cache is enabled, `require_cached` permits a remote cache lookup during execution. The preflight itself neither downloads data nor parses corpus contents. Backend compatibility, fixed-dataset overrides, and successful loading are checked at run time.

Relation extraction matches each predicted pair to gold labels in the same document. Corpus precision, recall and F1 use the summed match and item counts. The `anno-relation-document-micro-v1` policy cohort distinguishes these measurements from older scores that pooled document-relative offsets. Re-score the same inputs to build a comparable baseline.

GoogleRE provides judged entity-ID pairs and evidence snippets. The standard loader reads its snippets; `anno sampler eligibility --task re` reports `missing_relation_spans` because the span scorer needs gold character offsets for both arguments.

Temporal evaluation scores distinct TIMEX start/end pairs on each original sentence. The
pinned TIMEX test split has 390 news sentences, 592 annotations and 591 distinct spans. One
annotation repeats an interval. Every test sentence contains a time expression, so this split cannot
measure false alarms on sentences without one. Nested and adjacent intervals remain separate. The pattern
baseline maps DATE and TIME spans to TIMEX. The GLiNER multi-task backend receives the label
`time expression`. Event, signal, normalized value and temporal ordering annotations are outside
this metric. The receipt records raw annotations, distinct gold spans, predictions, matches and
micro F1. The generic tokenized NER view has a different denominator.

Matrix rows report `unmatched_predictions_overlapping_missed_gold`. Each unmatched prediction counts once if it intersects any missed annotation in its sentence. Several predictions can overlap the same missed annotation. Use `gold_empty_sentences_with_predictions` and `gold_empty_sentences` to measure how often a sentence with no gold spans raises a false alarm.

CI scores the 34-document English contrast panel as `AnnoTimexContrast` and the two-sentence synthetic smoke control as `AnnoTimexSmoke`. Each has its own cache filename, pinned digest and sampling history. Stage the checked-in file before requesting that fixture. The native temporal reader rejects a replacement file with different bytes. [Fixture sources and local replay](../scripts/qa/fixtures/timex-contrast/README.md)

The public metadata for the historical 390-sentence split identifies TimeBank 1.2 as its source. Its pinned Hugging Face repository provides no separate license. The LDC catalog names the underlying news and annotation rights holders. Its non-member agreement limits use and redistribution. For independent reproduction, obtain the applicable source permissions first. Stage an authorized local file in a private cache with `ANNO_CACHE_DIR=/private/cache python3 scripts/qa/prepare_matrix_timex.py /path/to/test.jsonl`. The helper verifies the historical file digest offline. Keep original text and annotations out of shared caches and artifacts. [Pinned dataset card](https://huggingface.co/datasets/mdg-nlp/timex-recognition-sentence-original/blob/904e35a7d2d0ee97c5594735621f94b04f8b7cef/README.md) · [TimeBank 1.2](https://catalog.ldc.upenn.edu/LDC2006T08) · [LDC non-member agreement](https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf)

### Compare temporal decoders

The GLiNER multitask token-level ONNX graph scores start, end and inside independently.
The Rust decoder follows GLiNER v0.2.16's flat, single-label selection. An exact span needs both
ends above the threshold and every word inside at or above it. A strong start with a weak end
can disappear entirely. [Upstream decoder](https://github.com/urchade/GLiNER/blob/a865555af53920321c4593f7c091dfc9059e9028/gliner/decoding/decoder.py#L61-L94)

Two English panels show why exact boundaries and false alarms belong together. With the same
graph, labels and threshold, correcting the decoder raised exact matches from 23 to 26 of 53
Wikinews spans and from 14 to 18 of 40 Microsoft specification spans. The latter change gained
six spans and lost two compact slash dates. False alarms occurred in zero of 34 gold-empty news
sentences and three of 16 gold-empty specification cases after the correction. An additional
English phrase-expansion rule gave no incremental gain on either panel. Researchers with authorized access can replay the 390-sentence news split.
Both now-inspected independent panels are regression controls under their source terms. Freeze new source
documents before evaluating a subsequent candidate. [Prepare the independent panels](../scripts/qa/temporal_sources/README.md)

## Choose a mode

`anno sampler` (also `anno muxer`) exposes three presets. An explicit `--strategy` overrides the preset.

| CLI mode | Default strategy | What gets the next test |
| --- | --- | --- |
| `--mode triage` | `worst-first` | Cells with unresolved failures or comparable evidence of recurring problems. |
| `--mode measure` | `ml-only` | Eligible backends prioritized by the configured policy and available outcomes. Selection changes the sample, so use a fixed panel for a head-to-head quality comparison. |
| `--mode coverage` | `estimate` | Backends with fewer recorded observations across the eligible dataset panel. |

`--strategy random` supplies a seeded control. The historical strategy name `ml-only` selects a policy. Feature flags, task compatibility and `--include-ml` determine which backends enter the candidate set. Use `--mode coverage` when filling gaps. The legacy spelling `--mode estimate` selects measure, while `--strategy estimate` selects coverage. In muxer itself, the stateful `Router` can change from normal operation to triage when a drift alarm fires. Anno's CLI selects its preset explicitly. [CLI definitions](../crates/anno-cli/src/cli/commands/muxer.rs) · [Upstream RouterMode](https://docs.rs/muxer/0.5.3/muxer/enum.RouterMode.html)

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
with known entity spans. The registry's WikiGold format example is staged as
`AnnoNerSmoke.cache` in a separate temporary cache with downloads disabled.
The loader verifies the fixture digest, and CI requires one completed
`AnnoNerSmoke` result with F1 of 1. The five-example bootstrap uses the
pinned WikiGold release and records its own `WikiGold` result.

External sampling restores the newest compatible checkpoint produced by this
workflow on the default branch. The artifact carries typed muxer windows,
registered arm IDs, scores, observation IDs and a SQLite index rebuilt from
the exported journal. Full evaluator receipts and diagnostic messages stay in
the writer's cache. For repeated infrastructure failures, the exporter assigns
a group to each distinct diagnostic within the run. Confirmed groups survive
later exports. A new private diagnostic cannot confirm an older exported group,
so a cell with one failure per run can remain eligible for ordinary retries.
Checkpoints expire after 60 days. A new cohort starts when the history,
provenance or export schema changes. PRs can read this state. Only
default-branch runs with new external observations publish it.
An active default-branch run can finish its checkpoint before the next run
starts. A retry stays read-only if that run or a newer run already published
the same cohort. Each retry retains its own sampling receipt.

Both kinds of run measure five examples per selected dataset from job-local storage. A
checksum-pinned public WikiGold corpus supplies a real input even after cache
eviction. A cold default-branch run records a heuristic baseline on that corpus.
The synthetic offline checks use separate storage. A temporal check scores two
authored sentences with the pattern backend: one contains a date and one has no
TIMEX span. Event and signal labels do not count as temporal gold. Worst-first runs before
the four randomized task controls (NER, relation extraction, intra-document
coreference, and temporal extraction). On the default branch, a named control
may fetch data when that task's local cache is empty. The job summary links
the restored run and reports the first worst-first selection, its observed
result, completed or missing coverage by task, and which dataset/backend pairs
failed. The job log names the selected datasets and backends before evaluation
starts, so an interrupted run can still identify its intended inputs.

The Actions summary prints the evaluator checkout, features and a command for each recorded cell with replay coordinates. For example, replay the one-sentence checked-in NER control from the repository root:

```sh
python3 scripts/qa/replay_matrix_cell.py ner AnnoNerSmoke heuristic --seed 42 --max-examples 1
```

The helper stages checked-in controls in a fresh temporary cache. For an external cell, supply its authorized local dataset cache and any required model assets. The command fixes the measured task, backend, dataset, seed and example limit. A reported dataset digest checks the input version. For a WikiGold cache without a manifest, the loader recognizes the pinned public release by hashing the bytes it parsed. When CI has prepared that release, each scored WikiGold outcome must carry its digest. Without a digest or pinned model weights, a rerun can establish the cell’s current score but cannot establish identity with the original inputs.

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

After a slot, use its receipts to decide which behavior to challenge. For example:

```sh
just qa-round .generated/mixed/slot-2 "Does this backend preserve the source occurrence after a name repeats?"
```

The handoff labels plan-only selections, incomplete runs, and strategy mismatches. The coordinator can use those to investigate execution. Only native evaluator outcomes supply quality evidence. Keep agent hypotheses in the round report and let completed evaluations update muxer history.

## Coverage

Use `--mode coverage` or `--strategy estimate` when the question is which
eligible fixed-panel cells have the fewest completed evaluations. The
implementation counts runs that evaluated at least one example without an error,
sums each backend's counts across the selected datasets, and chooses the lowest
total. Tasks with their own metrics can contribute coverage without a primary
F1. The evaluation history also retains skipped and failed attempts for triage. To measure scored coverage, count successful receipts
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

### Compare detection against a known annotation fault

Use the [fault comparison](../scripts/qa/compare_sampler_annotation_faults.py) with an eval-enabled CLI, Python 3.11 or newer, and a directory holding the pinned `WikiGold.cache` and `Wnut17.cache` files. The script checks both source hashes, copies them into temporary caches, and changes labels only in those copies. The WikiGold source is pinned by [the cache preparation script](../scripts/qa/prepare_matrix_wikigold.sh); the WNUT-17 input is the [upstream annotated test file](https://github.com/leondz/emerging_entities_17/blob/master/emerging.test.annotated).

```bash
python3 scripts/qa/compare_sampler_annotation_faults.py \
  --anno "$PWD/target/debug/anno" --cache "$ANNO_CACHE_DIR/datasets" \
  --out .generated/annotation-fault-comparison
```

The experiment removes WikiGold location labels or WNUT-17 person labels in separate trials. Each of coverage, triage, and seeded random gets three 20-example selections from the same two eligible corpora and the same starting history. Eight schedules cross both planted faults and three pre-fault histories. The sampler sees only its normal outcomes. A separate oracle compares the selected evaluation's gold count with a pristine evaluation of the same dataset and seed. Published receipts include the per-slot count differences and unchanged-corpus checks. Choose a new output directory for each run.

In the initial 432 selections, coverage detected the planted loss within two slots in 39/48 trials, compared with 33/48 each for triage and random. With clean history from the corpus that later lost labels, coverage detected it by slot two in 8/16 trials versus 11/16 for each alternative. Coverage detected both kinds of faults by slot three in all 48 trials. Seven draws of the corrupted WikiGold corpus had *higher* sampled gold counts on one seed: NER stratifies by whether a sentence has any gold, so changing gold can change which sentences the seed selects. All 213 draws of the intact corpus matched their pristine controls. The endpoint measures detection of these planted data losses under this sampling policy. It does not estimate sensitivity to other defects.

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

Benchmark JSON records the observed generic loader tier in `provenance.dataset.data_source`, the input digest in `content_sha256`, and a reason in `s3_fallback` when the publisher supplied an input after an S3 miss or exclusion. The digest identifies the loader input before sampling. Document-oriented scorers may also use companion files. Matrix outcomes project these fields as `dataset_source`, `dataset_sha256`, and `s3_fallback`, alongside the observed split and evaluator revision when available. Leave absent historical fields unknown when comparing runs. The matrix logs a separate `dataset precheck` when it loads a generic input to check eligibility. That check can warm the local cache before scoring. Its record is available for the lifetime of the matrix job log.

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

Use `{"text":"quiet","entities":[]}` for a document reviewed and found to contain no entities. Every predicted entity in that document counts as a false positive. With `--metric char-masking`, every character covered by its predictions contributes to the false-positive rate over all source characters. A blank gold file has no labeled document and is rejected.

Replay validates the stored signals and matches the entire source text, including whitespace. The JSON report retains the grounded document ID, the SHA-256 of the evaluated UTF-8 text, and the saved signal IDs in the matches. Model and inference time are null during replay. The grounded artifact contains source text and detected surfaces, so store it with the same access controls as the original.

For saved predictions, the JSON measurement's `predictions.source_sha256` identifies the UTF-8 text scored. When the grounded document carries mapped HTML, `predictions.html_source.source_sha256` and `predictions.html_source.renderer` identify the original bytes and how they were rendered, as recorded during extraction. Keep the grounded document for its per-span fragment map. [Trace a signal to its original HTML](EXTRACTION_WORKFLOWS.md#trace-a-signal-to-its-original-html)

Gold JSONL records may carry their source `text`, `source_sha256`, and a document ID (`id`, `doc_id`, or `document_id`). The evaluator checks any supplied text or hash against its input, requires one document ID throughout the file, and checks that ID against the saved predictions. To preserve a corpus document ID, use `anno batch --stdin --model pattern --format grounded` with an input record such as `{"id":"report-v1","text":"…"}`. Legacy entity-only gold records still validate each annotated surface. To bind gold to one of several documents with identical rendered text, give each document a distinct ID and include that ID in the gold record.

## Interpret coreference dataset scores

`anno dataset eval --dataset gap --task coref` reports an unweighted mean of document CoNLL F1 scores. It compares exact character spans and keeps supplied singleton chains. A document with no gold or predicted mentions contributes zero CoNLL F1. A dataset with one perfectly resolved two-mention chain and one correctly empty document therefore scores 0.50.

The link counts show how many within-document mention pairs the resolver matched, added or missed across the dataset. In the example above the counts are one matched, zero extra and zero missed. Add `--coref-oracle-mentions` to supply gold mentions and measure clustering separately from mention detection. The resolver receives the spans in document order and any type attached to each mention. On corpora with many singletons, report correct links and total gold links alongside CoNLL F1. A system that places every mention in a separate chain can score well on B³ and CEAF-e while recovering zero links.

Matrix decision logs store the following counts in the `structural_counts` object on each scored outcome. Coreference rows record `matched_coref_links`, `predicted_coref_links` and `gold_coref_links`. For resolvers that take mention spans, divide `matched_input_mentions` by `input_mentions` for exact mention precision and by `gold_mentions` for recall. Divide `reachable_gold_links` by `gold_coref_links` to see how many gold links the resolver could form from its inputs. Counts sum distinct spans and pairs within each document. A zero denominator leaves the corresponding rate undefined. Raw-text FCoref exposes its final clusters only, so its rows omit input coverage.

The matrix sampler uses separate coverage histories for detected and supplied mentions. Earlier results that omitted the mention source keep their historical cohort. PreCo’s authors limit the compilation to academic study and research and identify third-party rights in its passages. For an authorized evaluation, put the original development JSONL at `$ANNO_CACHE_DIR/datasets/PreCo.cache` on your own machine. Keep passages and per-example outputs local. About two thirds of the passages come from RACE, whose authors specify noncommercial research use. The mirror labels its license unknown. [RACE release](https://www.cs.cmu.edu/~glai1/data/race/) · [Authors’ Copyright and Privacy Notice](https://preschool-lab.github.io/PreCo/) · [Pinned mirror card](https://huggingface.co/datasets/coref-data/preco_raw/blob/02c8ffedb9d0b49168809a572ff553c2f6f9f7ca/README.md)

To compare mention sources on all 500 development documents, run the same command twice with `ANNO_COREF_GOLD=0` and `ANNO_COREF_GOLD=1`.

```sh
ANNO_MATRIX_INCLUDE_NON_AUTOMATABLE=1 ANNO_NO_DOWNLOADS=1 ANNO_S3_CACHE=0 \
anno sampler --strategy random run --task intra-coref --fixed-datasets PreCo \
  --fixed-backend coref_resolver --require-cached --max-examples 500 --runs 1 --seed-base 42
```

Use separate decision files for the two runs. The sampler pools document-scoped chains for corpus CoNLL F1; the dataset CLI averages the individual document scores.

## Replay a reviewed corpus witness in a triage slot

For a single developer replay use `just qa-replay typed_spans currency-print-spacing`
or the fixed-cell command in the Actions matrix summary. The single-case command
works on the current working tree. Use the ledger below to persist a confirmed
outcome for future Muxer selections. [Local replay](QA.md#run-a-check)

Native matrix history records which dataset/backend/task cells ran. A catalog
operation trace instead checks a document-edit contract. Give it a separate
persistent ledger (Python 3.11 or newer) when reproducing a finding from fuzzing or adversarial QA.
A clean checkout and a tracked seed pin the case, source and oracle. The
`reviewed_kb` adapter executes the named Cargo regression directly. Build
errors, missing cases and inconsistent reruns cannot open a finding. A failure
must reproduce twice with the same diagnostic and producer hash.

```sh
python3 scripts/qa/replay_feedback.py --cache-dir "$cache" observe reviewed_kb changed-and-restored
python3 scripts/qa/replay_feedback.py --cache-dir "$cache" next
python3 scripts/qa/run_matrix_slot.py --cache-dir "$cache" --receipts "$slot_receipts" \
  --policy mixed --slot 1 --period 4 --coverage-slot 2 --seed 41
```

The mixed-slot coordinator consults an existing vetted ledger by default.
An absent ledger leaves native selection intact. Pass `--no-replay-history` to
reproduce a native-only comparison. Direct `anno sampler --strategy ...`
continues to select native evaluation cells. A passing reviewed seed yields no priority. An unresolved witness can occupy
the first eligible triage slot in each block. Each such slot runs one named
case, with a second replay only to confirm a failure. Random controls, the
reserved coverage slot and other triage slots run the native selection. The
oldest-served eligible corpus gets the next turn, then its oldest-served
family and case.
The catalog adapter admits the seven named, repository-authored regression seeds.
A new seed needs provenance review and admission before it can enter the ledger.
Rerunning the same evidence adds no observations. A clean replay closes that
witness alone. When a seed or oracle changes, re-observe it under the new
hashes. A changed producer can close a finding against unchanged inputs.

The `typed_spans` adapter runs one authored case from the typed assertion panel
through the offline stacked CLI (`ANNO_BIN` or `anno` on PATH). It records the
actual executable hash and the pinned span scorer. Its result is scoped to
the enumerated assertions for that case. Judge comments may nominate cases;
they cannot insert an observation without a successful oracle invocation.

For labeled data, `observe wikigold sentence-0` runs the stacked backend on
one zero-based sentence from the checksum-pinned WikiGold source. The oracle
compares exact PER, ORG, LOC and MISC spans. Both extraction adapters require
the native backend ID and source hash in the batch output. Install the pinned
`datasets/WikiGold.cache` under `--cache-dir`, or set `ANNO_CACHE_DIR` to a
separate verified corpus cache. Each sentence remains independently
addressable. Replays retain a domain-separated digest of a discrepancy. Source
text, gold spans and predicted spans stay in the local scorer. Its replay
result is a sentence-scoped contract assertion;
only the native evaluator contributes corpus-level model observations.

The slot receipt and adversarial round handoff identify the corpus, case,
scope and oracle. A plan-only call spends no turn or evidence. Contract
replays add no dataset quality score, backend failure or live URL observation.
Keep an intentionally faulty producer's ledger in its own experiment directory.

```sh
python3 scripts/qa/replay_feedback.py --cache-dir "$cache" snapshot "$trusted_backup"
python3 scripts/qa/replay_feedback.py --cache-dir "$fresh_cache" restore "$trusted_backup"
```

Snapshot and restore build a new database from each witness's latest validated
receipt, generation and rotation position. Earlier observations, auxiliary
tables and free-text fields stay in the local ledger. These transfers support
up to 512 witnesses and 2 KiB per receipt. Obsolete scopes, including the
old WikiGold span-list format, are omitted. Re-observe those cases with the
current adapter. When a validated typed-span fixture retires a case, selection
and transfer skip that witness so the remaining cases can still be replayed. Both commands require a new destination and a trusted input.
Store the replay snapshot in a trusted project archive. On a new machine, restore the existing native matrix
checkpoint into the fresh cache, then restore the separate replay snapshot
into that same directory using the command above. Native CI checkpoints keep
their existing file list, provenance checks and publication rules. They
continue to restore without a replay ledger. Do not import a replay snapshot
from an untrusted PR artifact.

For an English mention-proposal experiment, run the same sampler command with
`ANNO_COREF_GOLD=0 ANNO_COREF_ENGLISH_PRONOUNS=1` and a separate decision file.
This supplies the entity resolver with NER proposals plus exact source spans from
Anno's English pronoun detector. Existing NER types take precedence on matching
spans. Compare proposal coverage, correct and extra links, link precision and
corpus CoNLL F1 on the same documents. Repeated pronouns, quoted tokens and
complementizers such as “that” can introduce false links. Tests on
397 held-back PreCo documents raised simple-resolver link recall from 1.55% to
28.30% and precision from 47.17% to 55.02%. On 12 independently selected LitBank
excerpts, recall rose from 0.96% to 22.25% while precision fell from 69.13% to
45.83%. The extra false links exceeded the three-percentage-point precision
limit set before testing. [LitBank source](https://github.com/dbamman/litbank/tree/3e50db0ffc033d7ccbb94f4d88f6b99210328ed8/coref)
The sampler keeps a separate history cohort for this opt-in research mode.
Gold mentions and raw-text FCoref runs have their own mention sources and reject
the additional proposals.
