---
name: anno-qa
description: Run anno's code checks and bounded dataset/backend evaluations, using capability matching and muxer history to investigate failures and estimate quality. Use before merging changes or when auditing evaluation coverage.
---

# Anno QA

Run commands from the repository root. This is a tool-based procedure usable by
developers or coding agents; it does not require a particular harness.

## Procedure

1. Define the changed boundary and evaluation budget: tasks, datasets, backends,
   examples, seeds, and runtime limit. Read `justfile`, the relevant tests, and
   `docs/EVALUATION_SAMPLING.md`. Keep receipts in a fresh directory:

   ```bash
   REC="$PWD/.generated/anno-qa/run-001"
   mkdir -p "$REC"
   git rev-parse HEAD > "$REC/commit.txt"
   git diff > "$REC/working-tree.patch"
   cp Cargo.lock "$REC/Cargo.lock"
   ```

   Keep receipt paths absolute: Cargo tests run from their package directory.

2. Run `just check-cached`. Retain the command, exit status, and complete output.
   Use the machine's configured sccache wrapper; do not replace its remote-cache
   configuration. Add focused feature checks for the changed boundary; consult
   `just check-feature-matrix` and CI rather than using `--all-features` across
   incompatible hardware features.

3. Inventory the current catalog and eligible pairs before choosing a panel:

   ```bash
   cargo run -p anno-cli --features "eval onnx discourse" -- dataset list
   cargo run -p anno-cli --features "eval onnx discourse" -- models list
   ANNO_MATRIX_REQUIRE_CACHED=1 ANNO_ML_IN_MATRIX=1 \
   ANNO_MATRIX_COVERAGE_REPORT="$REC/coverage.json" \
   cargo test -p anno-eval --lib --features "eval onnx discourse" \
     muxer_matrix::test_matrix_coverage_report -- --nocapture
   ```

   Inspect the report and test count. Registered, declared loadable, compiled,
   compatible, cached, and actually executed are different states. Capability
   matching filters candidates; it does not prove runtime readiness or quality.
   Choose small compatible panels by task/domain instead of a blind Cartesian
   product. Model aliases are not independent architectures.

4. Establish fixed acceptance checks before adaptive exploration. Run
   `bash scripts/eval-sanity.sh` for the bounded WikiGold loader/backend smoke
   check (requires `jq`, permits capped downloads). This is separate from the
   cached-only policy comparison below. Its JSON gate requires both
   expected nonempty results. For affected tasks, run additional fixed panels
   with `anno benchmark --help`, retaining JSON results and source/model hashes.
   Keep datasets and seeds identical when comparing revisions.

5. Use muxer to allocate the remaining budget according to the question:

   | Question | Evidence to seek |
   |---|---|
   | Did a known behavior regress? | Fixed cells and a comparable baseline |
   | Where are failures or poor results concentrated? | Triage history, reproduced failures, low-quality slices |
   | Where is quality poorly known? | Under-observed eligible cells, repeated estimates across seeds |
   | Does adaptive selection help? | Random control at the same panel and budget |

   Read `anno muxer --help` and `anno muxer run --help` for available policies.
   `--mode coverage` / `--strategy estimate` prioritizes lower observation counts
   across the selected panel. It helps gather evidence for quality estimation;
   it does not calculate uncertainty or account for variance or stale results.
   `--pin-backend` filters candidates while retaining selection;
   `--fixed-backend` bypasses selection. For a bounded paired triage experiment,
   set `DATASETS` and `PANEL` to comma-separated IDs from the coverage report:

   ```bash
   for POLICY in random worst-first; do
     mkdir -p "$REC/$POLICY"
     ANNO_EVAL_HISTORY="$REC/$POLICY/eval-results.jsonl" \
     cargo run -p anno-cli --features "eval onnx discourse" -- muxer \
       --strategy "$POLICY" --include-ml \
       --history-file "$REC/$POLICY/history.json" run \
       --runs 3 --seed-base 4100 --task ner --max-examples 3 \
       --datasets-per-run 2 --backends-per-run 1 --per-dataset --require-cached \
       --fixed-datasets "$DATASETS" --pin-backend "$PANEL" \
       --decisions-file "$REC/$POLICY/decisions.jsonl" \
       --agg-out "$REC/$POLICY/aggregate.json"
   done
   ```

   Three runs check plumbing, not policy superiority. Preserve history within
   a comparable cohort; isolate histories between policies. Inspect actual
   outcomes with `anno muxer decisions`, `stats`, and `regress`. Repeat bounded
   panels for other supported tasks and domains as budget allows.

6. Report executed versus eligible cells, failures, skips, missing prerequisites,
   dataset/model provenance, seeds, and measured quality. Link receipts. Stop
   when required fixed checks pass and exploratory findings are reproduced,
   fixed, or assigned an explicit follow-up with an owner and reason.

## Constraints

- A successful process or an outcome row is not proof that every selected cell
  passed. Exploratory matrix tests can record failures without failing the test.
- Low F1 is not automatically a code bug. Separate hard errors, weak model fit,
  annotation incompatibility, scorer defects, and baseline regressions.
- Adaptive samples are selected observations, not unbiased population estimates.
  Do not claim policy uplift, calibrated uncertainty, or off-policy evaluation
  without an appropriate experiment and valid selection probabilities.
- Example counts are not monetary cost; retain elapsed time separately. Debug
  runs are not deployment throughput benchmarks.
- GPU compilation does not prove provider execution. Require actual placement
  and numerical evidence on the target hardware before claiming support.
- Coreference receipts must disclose adapter and scorer limitations; an
  approximate CEAF assignment does not produce official leaderboard scores.
