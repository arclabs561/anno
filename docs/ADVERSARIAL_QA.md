# Run an adversarial QA round for Anno

Use a round when a result could change what a maintainer ships, reverts, or measures next. Start with the user's operation and the evidence already available. A green panel can motivate a challenge just as readily as a failed cell.

## Coordinator prompt

> At the baseline revision, identify one user-visible claim and the decision that depends on it. Read the starting receipt. State two plausible explanations for the observation, the smallest test on which they predict different outcomes, and what result would change your mind.
>
> Choose the job for this round. **Verify** a recurring failure using triage history. **Explore** an under-observed part of a compatible panel using coverage. **Discriminate** between explanations using a paired experiment on the same inputs. These are assignments for the coordinator. The existing sampler modes select cells; the coordinator chooses the claim to test inside a cell.
>
> Give the scout the public API or CLI, a time or example budget, and one starting case. Search a pinned source, use corpus labels, or apply a seeded transformation. Look for a boundary between two correct behaviors. Give the oracle reviewer only the inputs, units and contract. Have them record expected output independently of the implementation. Ask the skeptic for a legitimate neighboring case a repair might break, and reserve a different document or transformation family before exposing any result to the repair agent.
>
> Bring the three agents together over one disagreement. Ask each for the replay that would settle it. If the gold is ambiguous, record the ambiguity. If a failure reproduces, give the repair agent the minimized case and visible controls. Have the reviewer run both revisions on identical inputs and then reveal the reserved cases. Retire any exposed holdout from future blind rounds.
>
> Return the revisions, receipt path, seed or source hash, command, expected and observed result, gains, losses, ties, denominator and runner cost. Classify the claim as confirmed and fixed, confirmed and open, disproved, or inconclusive. Group variants by root cause. Finish by identifying the next decision and the cheapest experiment that could change it.

## Start from a rotating domain sample

The default URL sampler balances catalog topics and publishers. Its JSON receipt records which subjects were eligible, which sources were selected, and which actually yielded a comparable rendered input. Pass that file directly to the coordinator. The scheduled live QA job includes both the receipt and a handoff in its artifact.

```sh
just qa-round qa/reports/publishers.json 'Which selected topic needs a new source or independent labels?'
```

Check for duplicate text and redirects before assigning a scout. Two publisher URLs returning the same notice provide one rendered input. A valid page with zero extracted entities can still contribute to retrieval coverage. For quality comparisons freeze the source revision, use independent positive and protected-range labels, and report gains and losses separately by domain. Keep deliberately challenging punctuation or alias cases separate from a sample intended to describe ordinary documents.

For a labeled round, pass the [domain span checker’s JSON report](DOMAIN_QA.md) to the same command. The handoff includes exact hits, protected-range passes and paired changes for each group. It omits the original mentions. A receipt with repeated subjects from one publisher can support a genre-specific result; evidence of transfer to other publishers requires separate source families.

## Assign a reviewer and account for the run

Start a routine round with one author and the executable replay. Bring in an independent labeler when the annotation convention is disputed or the change affects the scorer. Give fresh-history authors just the public contract and a common output budget in a historical trial. Keep the revision pair, patch and known regression with the executor until proposals are frozen. Score a historical rediscovery only when the proposed check separates the old and repaired behavior. An expectation that fails on both revisions may identify another defect.

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

In a Codex session, call `collaboration.spawn_agent` with a fresh task name, `fork_turns="none"`, and `message` equal to the contents of `oracle-prompt.txt`. Give the oracle only that prompt. Wait for its final answer, then save the response body in `$round_dir/oracle.json`. Take the start timestamp immediately before dispatch and end the span when the answer arrives. Set `dispatch_status=ok` after saving the answer, `dispatch_status=error` if dispatch fails, or `dispatch_status=interrupted` if cancelled. Run the following block in the same shell.

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

The fixture can be checked against the source by hand: the email occupies characters 8–23. A passing replay reports one exact span. If the response is malformed or the extractor disagrees, keep the original answer and the exit status. Record a retry as a separate call. In a larger round, use the same start/end pair per agent so concurrent calls and the replay remain distinguishable.

The coordinator measures elapsed time from dispatch to observed completion, including scheduling and waiting. Provider inference time, the resolved model, token counts and cost require separate evidence. Supply `--model`, `--input-tokens`, `--output-tokens` or `--cost-usd` on `end` only for values the host reports. The checker uses exit 1 for an unmet assertion and exit 2 for invalid input. The wrapper preserves the child's exit status. Only explicitly listed exit codes appear in `assertion_failed_spans`. Other unsuccessful commands appear in `failed_spans`. A missing result artifact appears in `recording_issues`; the span preserves the child's own outcome. Command duration uses a monotonic clock. Agent and round durations use the coordinator's UTC start and finish times. The usage report sums only observed costs and token counts and lists how many agent spans contributed each value. The trace records file digests. Keep raw prompts, responses and source documents in the private round directory. Review any report before sharing, and use public or approved material in prompts sent to an external model. [Span assertion contracts](QA.md#check-what-a-negative-span-means)

## From a panel to a decision

The fixed WikiGold panel can stay green even if organization boundaries fail around punctuation. Two explanations are that the backend mishandles aliases generally or that token spacing masks the bug. Keep the ordinary corpus as the control. Have the scout remove only the spaces immediately inside parentheses and have the reviewer remap the gold offsets to the transformed text. Run both versions on both renderings.

On 1,696 WikiGold sentences the ordinary rendering produced identical predictions. On the transformed rendering the candidate removed 17 incorrectly merged organizations, recovered 30 exact gold organizations and introduced three false positives. The next triage question is what distinguishes those three false positives from genuine aliases. A separate coverage slot can sample a new document family while the skeptic investigates them. Keep synthetic transformations identifiable in the receipt.

For a policy comparison, give the seeded random control the same candidate set, starting history and budget. For a historical blind replay, give the scout a pre-fix revision, withhold the patch and its tests, and count distinct rediscovered failures per runner-minute. Use a fresh case when the agent has already seen the public fix.

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

The weekly and manual live QA job also scores up to eight discovery anchors per group on pinned English, Dutch, Spanish and Chinese datasets. The `live-document-qa` artifact includes the source pins and split counts in `domain-manifest.json`, the revision, backend and aggregate scores in `domain-discovery.json`, and the reviewer prompt in `labeled-round.md`. Source text, individual predictions and reserved examples stay on the ephemeral runner. Use the same fixture digest for comparisons over time. A missed anchor counts toward the reported error rate. Failure to download, verify, run or score the fixture fails the job.
