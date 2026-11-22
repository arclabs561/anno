# Run an adversarial QA round for Anno

A round answers one falsifiable question through a public Anno entry point. Use the following prompt with a coordinator that can delegate to agents. A single reviewer can run the same stages, but should record when the oracle and repair share an author.

## Coordinator prompt

> Find a small, reproducible case where Anno violates **[behavior]** for someone using **[CLI command or library API]**. Start at **[commit]**. Success means that independent evidence establishes the expected behavior, a replay distinguishes a real failure from a bad expectation, and a proposed repair survives a test its author did not see. A round that disproves a plausible bug is also a valid result.
>
> Start three independent investigations. Give the **scout** the task and a budget. Give the **oracle reviewer** the task and source material, withholding Anno's predictions and the proposed patch. Give the **skeptic** the claimed contract and ask for a counterexample to the expectation itself. Keep the repair agent out of the held-out material.
>
> The scout should try a checked-in fixture, an annotated dataset or an authoritative public source, and a seeded transformation of a real example. Prefer different document families when the claim spans domains. Search can find a source; use a pinned record, revision or content hash to replay it. With fuzzing, save the seed, minimize the first interesting failure, and explain which output should change and which should stay the same.
>
> Have the oracle reviewer write expected spans, labels, scores or error behavior before seeing the output. Name the evidence: a source slice and its units, an independent implementation, a corpus annotation, or a mathematical invariant. Record ambiguities. Ask the skeptic to try a legitimate near-neighbor that a tempting fix would break. Where practical, reserve a different source, document or transformation family. Commit to its cases or selection rule in a separate receipt before the repair agent begins.
>
> Convene a short review. Each agent gives its claim, evidence, strongest objection and a command that could settle it. Resolve disagreement by running that command or consulting the source. If the task has two defensible interpretations, record the disputed contract and stop that repair. Agent votes cannot supply missing ground truth.
>
> Give the repair agent the minimized reproducer, agreed contract and visible controls. Ask for the narrowest change through the normal code path. Have a different agent replay baseline and candidate on identical inputs, then open the reserved cases. Report gains, losses, ties, missing inputs and the denominator. If a held-out failure informs a second repair, reserve fresh material for the next evaluation.
>
> Return one result per finding: **confirmed and fixed**, **confirmed and open**, **disproved**, or **inconclusive**. Include revisions, seeds or source hashes, an executable replay, expected and observed output, and the most important limit. Promote the smallest redistributable regression to the repository. Keep raw pages, model outputs and large receipts in local storage or CI artifacts.

A starter budget for one claim is three discovery sources, 32 seeded transformations, one minimized fixture and one held-out family. Scale it to the cost of the backend. Commit the selection rule before running a comparison and include attempts that timed out or found nothing.

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
