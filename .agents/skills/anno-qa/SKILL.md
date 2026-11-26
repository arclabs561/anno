---
name: anno-qa
description: Replay a reported Anno regression, interpret a benchmark or matrix receipt, hand evidence to a reviewer, or run a bounded dataset and backend evaluation.
---

# Anno QA

Choose the entry point for the question. Run commands from the repository root.

| Task | First action |
|---|---|
| Reproduce a named case | Run its fixed replay with the binary from the checkout being tested. For example, use `ANNO_BIN=target/debug/anno just qa-replay typed_spans currency-print-spacing`. |
| Find how a CI benchmark loaded its data | Open the run’s `eval-sanity-report.json` or benchmark JSON and inspect each result’s `provenance.dataset.data_source`. A bucket count measures objects. A workflow setting shows S3 was enabled. The result records which source supplied the input. |
| Explain a selected, missing or zero-scoring matrix cell | Run the compact handoff below on the decisions JSONL. It cites the observed rows and flags named selections without a matching run, strategy, slice, dataset and backend. Some selectors log only part of the selection. For the full denominator, compare the same run’s `matrix-muxer:` dispatch lines. Replay a measured cell at its recorded revision and features. |
| Compare a backend change on discovery and confirmation data | Run `python3 scripts/qa/brief_round.py SUMMARY.json 'Did the final predictions improve?'` on the paired-arm summary. Verify item identities and split separation in the underlying ledgers. Report final TP, FP and FN for each cohort alongside stage gains, losses and cost. |
| Review a scorer change | Replay both versions on the same saved outputs, including a known error and a valid neighboring case. Record which past comparisons can be recovered from the receipts. See [scorer revisions](../../../docs/ADVERSARIAL_QA.md#keep-scores-interpretable-as-the-harness-changes). |
| Hand a claim to another reviewer | Run `python3 scripts/qa/brief_round.py RECEIPT 'claim to investigate' --compact`. Include the source run or artifact link and the next check that could change the conclusion. For an independent oracle, share the input and labeling contract before revealing the producer’s answer. |
| Compare policies, sample coverage or launch a multi-backend campaign | Follow the [campaign procedure](references/campaign.md) for budgets, eligibility, fixed acceptance checks and isolated histories. |

For a historical run, keep the observed source, content digest, split, evaluator revision, seed and feature set when present. Mark missing fields as unknown. The current cache checksum establishes today’s bytes. A historical digest identifies the bytes used then.

For cross-document identity, select the grounded track query, fixed-mention benchmark, or named lifecycle replay in the [adversarial QA guide](../../../docs/ADVERSARIAL_QA.md#challenge-cross-document-identity-decisions). For a new regression seed, check the [fixture and source rights guidance](../../../docs/QA.md#decide-what-to-keep) before sharing an excerpt.
