# Run an adversarial QA round for Anno

Use a round when a result could change what a maintainer ships, reverts, or measures next. Start in the developer’s session with the [named replay or local fixture](QA.md#run-a-check). Run the cheapest check that distinguishes the competing explanations. Ask an agent for a counterexample or an independent label when that step needs judgment. CI runs the executable checks and Muxer sampler. A brief is optional when work crosses sessions.

## Coordinator prompt

To challenge a corrected byline, attribution, catalog redirect, or visibility decision, propose a short event history and run the [credit reader replay](../verification/index_consumers/reader_cases/README.md#propose-and-judge-another-history). It checks the Rust and SQLite readers against a separate answer key and accepts local proposals without contacting a model provider.

> At the baseline revision, identify one user-visible claim and the decision that depends on it. Read the starting receipt. State two plausible explanations for the observation, the smallest test on which they predict different outcomes, and what result would change your mind.
>
> Choose the job for this round. **Verify** a recurring failure using triage history. **Explore** an under-observed part of a compatible panel using coverage. **Discriminate** between explanations using a paired experiment on the same inputs. These are assignments for the coordinator. The existing sampler modes select cells; the coordinator chooses the claim to test inside a cell.
>
> Give the scout the public API or CLI, a time or example budget, and one starting case. Search a pinned source, use corpus labels, or apply a seeded transformation. Look for a boundary between two correct behaviors. Give the oracle reviewer only the inputs, units and contract. Have them record expected output independently of the implementation. Ask the skeptic for a legitimate neighboring case a repair might break, and reserve a different document or transformation family before exposing any result to the repair agent.
>
> Bring the three agents together over one disagreement. Ask each for the replay that would settle it. If the gold is ambiguous, record the ambiguity. If a failure reproduces, give the repair agent the minimized case and visible controls. Have the reviewer run both revisions on identical inputs and then reveal the reserved cases. Retire any exposed holdout from future blind rounds.
>
> Return the revisions, receipt path, seed or source hash, command, expected and observed result, gains, losses, ties, denominator and runner cost. Classify the claim as confirmed and fixed, confirmed and open, disproved, or inconclusive. Group variants by root cause. Finish by identifying the next decision and the cheapest experiment that could change it.

## Challenge cross-document identity decisions

Choose the claim before choosing a binary. A production report groups supplied tracks and qualified catalog records. The ECB+ evaluator measures a different heuristic on annotated mentions in known topics. Each accepts a small, independently authored JSON case. The scout can propose a new source history in a private directory, and an independent oracle can freeze its assertions before either version of the producer runs.

| Claim | Input the agent can change | Replay |
|---|---|---|
| Two sources refer to the same catalog record, a correction supersedes a link, or a mention is unassigned | Grounded observations in submission order, Unicode character spans, local track memberships, namespaces/accessions, type settings or supplied embeddings. Assertions target the active edition after a named number of submissions. | [Catalog revision case](../scripts/qa/fixtures/crossdoc/catalog-revision.json) and [coverage control](../scripts/qa/fixtures/crossdoc/coverage-control.json) with the `anno` CLI. |
| A scorer confuses local and shared IDs, counts a same-document pair, or overmerges same-spelling names | Tiny synthetic documents grouped by topic, token anchors, local chains, cross-document labels, optional sentence selection, and expected counters for each seed condition and threshold. | [Topic-scoped homonyms](../scripts/qa/fixtures/crossdoc_eval/topic-scoped-homonyms.json) and [local seeds](../scripts/qa/fixtures/crossdoc_eval/local-seeds.json) with the `cdcr_eval` executable. |
| A clean rebuild and a series of source revisions disagree, blocking drops a candidate, or a corrected catalog posting fails to reach the reader | Select the compiled regression that exercises the full production entry point. Its assertions cover retry/withdrawal, representative metadata, input order, the 100/101-document threshold, or catalog posting changes as applicable. | `just qa-replay crossdoc_native unlinked-revision-history`. Other names include `guarded-catalog-checkpoints`, `raw-blocking-boundary`, `grounded-candidate-policy`, `grounded-replacement-order` and `query-catalog-publication`. |
| A withdrawn credit survives in public search or one publisher's correction erases another publisher's support | Give the scout fictional source editions with a work, qualified target, printed credit or attribution, audience, and delivery order. Have the reviewer write public and staff answers before running the reader. | Copy the [credit corpus](../verification/index_consumers/credits/cases.json) into the private round, add the proposed case, and run `python3 verification/index_consumers/credits/check_cases.py PRIVATE_CASES.json --anno PATH_TO_ANNO`. [Evidence and reuse terms](../verification/index_consumers/credits/README.md) |

Build the executables and copy a suitable fixture into the private round directory. These commands run one authored case and pass its aggregate result to the same coordinator used by other QA rounds.

```sh
cargo build --locked -p anno-cli --no-default-features --features "eval extractor-html2text"
cargo build --locked -p anno-eval --example cdcr_eval --features eval
round_dir=$(mktemp -d)
cp scripts/qa/fixtures/crossdoc/catalog-revision.json "$round_dir/case.json"
just qa-challenge "$round_dir/case.json" target/debug/anno > "$round_dir/result.json"
just qa-round "$round_dir/result.json" 'Did correcting the stage record change the harbor record?' "$round_dir/round.sqlite3"
```

Run the same frozen file with a separately built baseline executable to compare versions. Exit 0 means every assertion passed, 1 means at least one measured assertion failed, and 2 means the case could not be evaluated. Inspect a failed run locally with `python3 scripts/qa/replay_crossdoc_case.py CASE BINARY --details`. A completed receipt carries the binary hash, separate source and oracle hashes, and assertion counts. The oracle hash includes the scorer code. Keep the case and detailed output with the authorized reviewers. A named reviewed case can enter the existing replay ledger through its normal two-run failure confirmation. Set `ANNO_BIN` for CLI cases and `ANNO_CDCR_BIN` for evaluator cases when using `just qa-replay crossdoc_cases CASE`.

For a historical or model-quality comparison, record which assertions distinguish the revisions, which both pass, and which both fail. A name collision can be a real matching error even when ingestion and publication behave correctly. `separate` requires two reported memberships. Use `unresolved` for a supplied signal without a reported group, and `absent` when that source occurrence is missing. An identical numeric report ID across runs is insufficient evidence for a shared referent. Public, project-authored examples can be reviewed for promotion into the named corpus. Raw source text, externally licensed annotations and local witness directories stay out of shared reports unless their terms and the user permit sharing.

The coordinator uses the agent host selected for the round. The repository’s Codex example dispatches a subagent with the oracle prompt. The local replay commands call the selected binary and need no LLM service or API key.


### Check the published report with an independent reader

A case can group the right occurrences while a name search returns somebody else's sources. An independent reviewer can freeze a JSON query holdout for a grounded proposal. Give the reviewer the source documents, the reader contract and the source digest before exposing any producer output. The digest covers the ordered inputs, case ID and resolver options. It excludes the author's assertions, so a change to those assertions can be scored against the same reader expectations.

```sh
case=scripts/qa/fixtures/crossdoc/lamp-ledger.json
holdout=scripts/qa/fixtures/crossdoc/lamp-ledger.queries.json
python3 scripts/qa/replay_crossdoc_case.py "$case" --source-digest
python3 scripts/qa/replay_crossdoc_case.py "$case" target/debug/anno --query-holdout "$holdout" --details
```

Use `kind: anno.qa.query-postings.v1`, that source digest in `source_sha256`, and a `queries` array. Each query sets `after` to a document submission count, supplies `entity` (native name search), `catalog_record` (exact catalog and accession), or both, and lists the expected `postings`. Each posting is a document ID and a zero-based position in the active edition's signal array. An empty array asserts zero hits. The reader checks every returned posting, including multiplicity, in any output order. The two judges share the actual report bytes produced once at each checkpoint. The harness allows 32 queries across up to eight combined checkpoints.

Without `--details`, the receipt carries separate authored and consumer outcomes, the proposal and holdout file hashes, the source digest, producer hash, and both scorer commitments. `--details` writes expected and observed postings to local stderr. A missed expectation in either judge exits 1. Invalid inputs or an execution error exit 2. A hash identifies the frozen bytes. Use a separately recorded reviewer dispatch when the experiment requires evidence of authorship or independence.

For an in-session agent or deterministic local reply, use `qa_oracle.py accept TASK ANSWER`. A headless agent can use `qa_oracle.py codex TASK ANSWER`. A raw gateway can use `qa_oracle.py openrouter TASK ANSWER --model MODEL`. Supply the resulting proposal and the separate reviewer response as the two replay files. Compare each dispatch receipt's `answer_sha256` to the corresponding file hash. The transport commands and task isolation are described below.

The lamp ledger, names, accession assignments and query expectations are original synthetic project examples released under the repository's MIT OR Apache-2.0 license. They are exposed regression controls. The native tests deliberately exchange display names or catalog namespaces between two report rows and rerun the reader. Keep future experimental holdouts outside the proposal author's context. [Proposal](../scripts/qa/fixtures/crossdoc/lamp-ledger.json) · [Reader expectations](../scripts/qa/fixtures/crossdoc/lamp-ledger.queries.json) · [Mutation controls](../scripts/qa/test_crossdoc_query_holdout.py)

## Start from a rotating domain sample

The default URL sampler balances catalog topics and publishers. Its JSON receipt records which subjects were eligible, which sources were selected, and which actually yielded a comparable rendered input. A local run retains the detailed receipt for replay or an optional reviewer brief. The hosted job exports catalog coverage counts.

```sh
just qa-round qa/reports/publishers.json 'Which selected topic needs a new source or independent labels?'
```

Check for duplicate text and redirects before assigning a scout. Two publisher URLs returning the same notice provide one rendered input. A valid page with zero extracted entities can still contribute to retrieval coverage. For quality comparisons freeze the source revision, use independent positive and protected-range labels, and report gains and losses separately by domain. Keep deliberately challenging punctuation or alias cases separate from a sample intended to describe ordinary documents.

For a labeled round, pass the [domain span checker’s JSON report](DOMAIN_QA.md) to the same command. The handoff includes exact hits, protected-range passes and paired changes for each group. It omits the original mentions. A receipt with repeated subjects from one publisher can support a genre-specific result; evidence of transfer to other publishers requires separate source families.

## Assign a reviewer and account for the run

Start a routine round with one author and the executable replay. Bring in an independent labeler when the annotation convention is disputed or the change affects the scorer. Give fresh-history authors just the public contract and a common output budget in a historical trial. Keep the revision pair, patch and known regression with the executor until proposals are frozen. Score a historical rediscovery only when the proposed check separates the old and repaired behavior. An expectation that fails on both revisions may identify another defect.

For a benchmark handoff, pass the saved JSON to `python3 scripts/qa/brief_round.py REPORT.json "Which dataset source supplied this run?" --compact`. Each row names the backend, recorded source and available SHA-256. A missing digest stays unrecorded. The compact form also works for matrix outcomes and named replays. Omit `--compact` to include the full guide for a reviewer without a checkout.

### Choose how to ask the oracle

The oracle receives one UTF-8 task file containing the approved source, annotation contract, units and requested JSON shape. Keep the coordinator's handoff, previous answers and producer results elsewhere. All three routes take the same task. The resulting answer file can go directly to the checker for that task.

| Where the round runs | How to obtain a proposal | When to choose it |
|---|---|---|
| Inside Codex or another agent harness | `just qa-oracle prompt TASK` writes the exact task. Dispatch it to a fresh reviewer with no inherited conversation. Pipe its final reply to `just qa-oracle accept TASK ANSWER`. | The coordinator already owns dispatch, credentials and cancellation. Start a trace span at the actual dispatch, as in the example below. Import records the answer supplied by the caller. |
| Headless with a configured Codex CLI | `just qa-oracle codex TASK ANSWER --timeout 90` | A script needs a fresh Codex turn using the CLI's own login or API credentials. |
| Headless Codex with the OpenRouter key | `just qa-oracle codex TASK ANSWER --openrouter --model openai/gpt-4.1-mini` | The round requires the Codex runtime and the available credential belongs to OpenRouter. Codex uses its Responses endpoint with a temporary login directory. |
| Directly through OpenRouter | `just qa-oracle openrouter TASK ANSWER --model openai/gpt-4.1-mini --max-tokens 256` | A data-only labeling job needs a single model call with a token cap and provider-reported usage. Set `OPENROUTER_API_KEY_FILE` to the private file holding the key, or use `OPENROUTER_API_KEY`. The file setting takes precedence. |

The headless adapter runs in a temporary directory with read-only sandboxing and model tools disabled. On a new Codex CLI version, run `ANNO_TEST_CODEX_LOOPBACK=1 python3 -m unittest discover -s scripts/qa -p test_oracle_headless.py` to inspect the actual request against a local fixture. This has been exercised with Codex CLI 0.172.0 on Linux. For the default Codex provider, supply a writable CLI login directory or `CODEX_API_KEY`. [OpenRouter’s Codex integration](https://openrouter.ai/docs/cookbook/coding-agents/codex-cli)

Supply `--shape array` for a typed-span fixture. The default is an object, as used by the cross-document case formats. Include the required fields and their meaning in the task. OpenRouter uses JSON object mode for an object and requests text for an array. Every route accepts a complete JSON value, either plain or inside a single Markdown block labeled `json`. It rejects surrounding prose, the wrong root, duplicate fields and nonfinite numbers. The receipt hashes both the original reply and the JSON bytes written to the answer file. Before live replay, compare the returned IDs and source text with the inputs the coordinator supplied. A live span replay runs the binary on the text in the proposal. A saved-predictions replay also checks the source hash. The native checker validates offsets and scores the assertions. Keep a proposal local until its contents and source rights have been reviewed.

Each dispatch needs a new answer path. Success writes a mode-0600 answer and prints a JSON receipt with prompt and answer hashes, transport, and any reported model, token counts and cost. The direct routes take `--trace DB` for an initialized round and record the attempt, including an error or interrupt. Use `--retry-of SPAN` for an explicit retry. The runner issues one dispatch per invocation. With `accept`, the caller identifies which task produced the reply and owns the dispatch timing. CI exercises synthetic fixtures offline. The same commands work with `python3 scripts/qa/qa_oracle.py` in place of `just qa-oracle`.

For a concrete comparison, give two fresh reviewers a topic with two mentions of X in document M and one in document N. They must count two cross-document pairs. Withdraw N and neither pair remains. Freeze both answers before invoking the evaluator. This control detects a reviewer that counts the within-document pair even when it returns well-formed JSON. [Cross-document cases](#challenge-cross-document-identity-decisions) · [Codex non-interactive execution](https://developers.openai.com/codex/noninteractive/) · [OpenRouter response formats](https://openrouter.ai/docs/guides/features/structured-outputs) · [OpenRouter usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting)

### Trace one author and one replay

Run from the repository root with a completed [panel or muxer slot](QA.md#run-a-check). Set `PANEL` to its receipt directory and `ANNO_BIN` to the executable to test. This example asks an independent reviewer to label one public synthetic sentence, then checks the actual CLI against the reviewer's answer.

```sh
umask 077
round_dir=$(mktemp -d)
db="$round_dir/round.sqlite3"
trace() { python3 scripts/qa/round_trace.py "$db" "$@"; }
just qa-round "$PANEL" 'Does pattern extraction report the email in Unicode scalar offsets?' \
    "$db" > "$round_dir/handoff.md"
cat > "$round_dir/oracle-prompt.txt" <<'PROMPT'
Use only this text, including its leading emoji: 🦊 Email ada@example.org
Return a JSON array with one object. Use id "mail", the exact text above
(starting at the emoji), and an assertions array with this rule:
{"kind":"exact_span","text":"ada@example.org","types":["EMAIL"],"start":N,"end":M}
Determine N and M in half-open Unicode scalar offsets. Return plain JSON.
PROMPT
# Read handoff.md and prepare dispatch before starting the timer.
span=$(trace start --role oracle --tool collaboration.spawn_agent \
    --prompt-file "$round_dir/oracle-prompt.txt")
```

In a Codex session, call `collaboration.spawn_agent` with a fresh task name, `fork_turns="none"`, and `message` equal to the contents of `oracle-prompt.txt`. Give the oracle only that prompt. Wait for its final answer, then pipe the raw response into `just qa-oracle accept "$round_dir/oracle-prompt.txt" "$round_dir/oracle.json" --shape array`. Take the start timestamp immediately before dispatch and end the span when the answer arrives. For this single-case task, verify the returned source with `jq -e 'length == 1 and .[0].id == "mail" and .[0].text == "🦊 Email ada@example.org"' "$round_dir/oracle.json"`. Set `dispatch_status=ok` after saving and checking the answer, `dispatch_status=error` if dispatch fails, or `dispatch_status=interrupted` if cancelled. Run the following block in the same shell.

```sh
(
    set -eu
    status=$dispatch_status
    case "$status" in
        ok)
            trace end "$span" --status ok --result-file "$round_dir/oracle.json"
            rc=0
            trace run --role replay --tool anno --assertion-failure-code 1 --input-file "$round_dir/oracle.json" \
                --result-file "$round_dir/replay.json" -- \
                python3 scripts/qa/check_span_assertions.py --model pattern \
                --binary "$ANNO_BIN" --fixture "$round_dir/oracle.json" \
                > "$round_dir/replay.json" || rc=$?
            case "$rc" in
                0) ;;
                1) status=assertion_failed ;;
                *) status=error ;;
            esac
            ;;
        error|interrupted)
            trace end "$span" --status "$status"
            if [ "$status" = interrupted ]; then rc=130; else rc=1; fi
            ;;
        *) echo "Expected dispatch_status=ok, error or interrupted" >&2; exit 2 ;;
    esac
    trace end round --status "$status"
    trace report > "$round_dir/report.json"
    exit "$rc"
)
```

After writing the report, the subshell returns 0 for a pass, 1 for a failed assertion, or 2 for an invalid fixture. A failed or cancelled dispatch closes both spans and skips replay.

The fixture can be checked against the source by hand: the email occupies characters 8–23. A passing replay reports one exact span. If the response is malformed or the extractor disagrees, keep the original answer and the exit status. Record a retry as a separate call. In a larger round, use a start/end pair for each dispatch and pass `--retry-of` with the failed or interrupted span ID when retrying.

The coordinator measures elapsed time from dispatch to observed completion, including scheduling and waiting. Provider inference time, the resolved model, token counts and cost require separate evidence. Supply `--model`, `--input-tokens`, `--output-tokens` or `--cost-usd` on `end` only for values the host reports. The checker uses exit 1 for an unmet assertion and exit 2 for invalid input. The wrapper preserves the child's exit status. Only explicitly listed exit codes appear in `assertion_failed_spans`. Other unsuccessful commands appear in `failed_spans`. A missing result artifact appears in `recording_issues`; the span preserves the child's own outcome. Command duration uses a monotonic clock. Automatically dispatched oracle attempts also use a monotonic clock. Manually recorded agent spans and round durations use the coordinator's UTC start and finish times. The `agent_usage.attempts` count includes every recorded agent dispatch, including retries and failures before an agent returned. The historical `agents` field is an alias for `attempts`. The status counts partition the attempts into `completed` (ended with `ok`), `errors`, `interrupted`, `assertion_failed` and `open`. A failed dispatch followed by a successful retry produces two attempts and one completion. The host supplies no unique actor identifiers to this trace. Token and cost totals sum the observed values across attempts. Each total has a coverage count and is JSON `null` when no attempt reported that value.

The trace records file digests. Its top-level `revision` and `tracked_changes` describe the coordinator checkout when the round started. Record each tested binary’s SHA-256, build revision and features alongside its replay result to identify the baseline and candidate. Keep raw prompts, responses and source documents in the private round directory. Review any report before sharing, and use public or approved material in prompts sent to an external model. [Span assertion contracts](QA.md#check-what-a-negative-span-means)

## Challenge the judge itself

For a span assertion, start with a successful batch receipt from an authored fixture. Keep one copy as a control and duplicate a single entity row in a second copy, updating the reported entity count if present. Both copies still contain the expected text, offsets and type. A caller that counts occurrences can distinguish them. Use `expected_count: 1` in the exact-span rule when that distinction belongs to the fixture contract. An assertion that asks only whether the span is present should pass on both copies.

Have an independent reviewer freeze three possible outcomes for each case before running the judge: pass, assertion failure, or invalid input. Include a repeated surface at another offset, a legitimate nested span and a mismatched source digest. Compare the baseline and candidate on the same receipts. Report missed assertion failures, false alarms on valid controls and mishandled invalid inputs separately. A seeded duplicate can establish a scorer blind spot; estimating how often the extractor produces it requires unmodified outputs.

When comparing mutation policies, hold the oracle and mutation recipe fixed, include a clean producer, and charge the immediate confirmation replay to the same CPU budget as discovery. Count distinct confirmed causes at the cutoff and record the time spent on repeated confirmations. A known case can be valuable to verify a repair even when it finds no additional cause.

## From a panel to a decision

The fixed WikiGold panel can stay green even if organization boundaries fail around punctuation. Two explanations are that the backend mishandles aliases generally or that token spacing masks the bug. Keep the ordinary corpus as the control. Have the scout remove only the spaces immediately inside parentheses and have the reviewer remap the gold offsets to the transformed text. Run both versions on both renderings.

On 1,696 WikiGold sentences the ordinary rendering produced identical predictions. On the transformed rendering the candidate removed 17 incorrectly merged organizations, recovered 30 exact gold organizations and introduced three false positives. The next triage question is what distinguishes those three false positives from genuine aliases. A separate coverage slot can sample a new document family while the skeptic investigates them. Keep synthetic transformations identifiable in the receipt.

For a policy comparison, give the seeded random control the same candidate set, starting history and budget. For a historical blind replay, give the scout a pre-fix revision, withhold the patch and its tests, and count distinct rediscovered failures per runner-minute. Use a fresh case when the agent has already seen the public fix.

### Replay a price-span regression

A typography trial generated eight variants of each of 16 labeled documents across eight domains. With a budget of 16 probes, Muxer's domain selector and balanced round-robin each found a failure in 17 of the same 24 orderings. Across all 128 variants, the stacked extractor missed three prices with narrow no-break spacing and five dates with nonbreaking hyphens. Domain selection showed no discovery advantage on this panel. The orderings reused the same documents.

The reviewed `currency-print-spacing` case pairs one exact price with a bare currency sign followed by punctuation. The old CLI failed both assertions on two replays. After fixing price boundaries, the same case passed after restoring the checkpoint into an empty cache. To replay it against a local CLI, run:

```sh
ANNO_BIN=target/debug/anno python3 scripts/qa/replay_feedback.py \
  --cache-dir /tmp/anno-price-replay observe typed_spans currency-print-spacing
```

The separate date candidate recovered printed dates and also mislabeled date-shaped identifiers. Both outcomes are recorded in [the date issue](https://github.com/arclabs561/anno/issues/295).

## Start from a measured muxer outcome

Pass the sampler's decision JSONL directly to the coordinator. The same analysis runs on a completed matrix slot. A low score with few reachable gold links prompts a mention-proposal experiment. If most gold links are reachable, the brief quantifies a lower bound on missed reachable links and suggests holding proposals fixed. Temporal near misses prompt a reference-decoder comparison. A sample without gold-empty sentences calls for a separate negative control.

```sh
python3 scripts/qa/brief_round.py /path/to/decisions.jsonl \
  "Which stage accounts for the missed links?" --trace /tmp/anno-muxer-round.sqlite
```

The brief reads counts and registry identifiers. Earlier receipts that contain only the overall score still work. Scope a proposed replay to the source permissions, pin the corpus and evaluation version, and give the scout fresh examples that were outside the selection history. Confirm the verdict with an independent oracle and a reproducible native run before retaining a regression seed.

## Choose a challenge that fits the claim

| Target | Scout's challenge | Oracle and skeptic |
|---|---|---|
| Offsets and interchange | Prepend an emoji, repeat the same name twice, change LF to CRLF, or insert a gap in a mention. | Slice the original text in the format's specified units. A same-looking mention at a different occurrence must fail. |
| Organization boundaries | Sample filings, contracts or research affiliations containing parenthesized names. Vary surrounding punctuation. | Use annotated spans or independently supplied names. Preserve a legal name such as `Acme (UK) Ltd` and distinguish a separate alias. Score only anchors the source actually labels. |
| Coreference and scoring | Move the same predicted mentions from their gold document into a second document with identical local offsets. | Compare with the reference scorer. Moving the prediction should eliminate credit. Permuting document order should preserve the score. |
| Matrix coverage | Remove a cached dataset before one run, restore it for the next, then repeat with a measured zero F1. | Inspect actual evaluator outcomes. A preflight failure belongs in triage history. A successful nonempty run with zero F1 counts as an evaluation. |

## Route the result

- **Pull requests** run the deterministic regression and its near-neighbor offline. Property tests cover the surrounding invariant. A small mutation check can show whether the regression catches the old behavior.
- **Scheduled or opt-in research** runs rotating public sources, larger seeded campaigns and model comparisons. Save the resolved input, retrieval outcome and content hash before comparing backends. Review material from the web as data.
- **Muxer** can spend a triage slot on a previously broken cell and a coverage slot on a different compatible cell. Keep a fixed panel for comparisons and a seeded random control for policy experiments. Feed it measured execution outcomes. Keep retrieval failures and disputed annotations distinct from model errors.

Record the baseline and candidate separately even when their outputs match. A count such as “3 gained, 0 lost across 175 anchors in 92 contracts” lets the next reviewer audit a claim; whole-document precision requires labels for the other mentions too. [Fixtures and dataset commands](QA.md) · [Muxer modes](EVALUATION_SAMPLING.md)

The weekly and manual live job also scores up to eight discovery examples per intent from the publisher’s Snips weather and restaurant validation sets. The shared artifact contains the source pins, split sizes and aggregate scores. Run the same panel locally for per-example inspection. Authorized local panels can cover other corpora and languages. [Source terms and commands](DOMAIN_QA.md#run-the-hosted-panel-locally)

To give a minimized, reviewed counterexample a future triage slot, run its
registered corpus oracle and retain the resulting source and producer hashes.
An adjudicator's report can nominate a case. Reproducing the same oracle
failure twice opens the witness. The next eligible slot names and replays the
case, while passing controls and model scores keep their own denominators.
[Replay state and slot selection](EVALUATION_SAMPLING.md#replay-a-reviewed-corpus-witness-in-a-triage-slot)

For a completed synthetic organization-span round, the [reviewed seed replay](../scripts/qa/reviewed_span_seeds/README.md)
binds the independent oracle to native results and emits digest-keyed regression evidence for scheduling.
