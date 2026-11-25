# Reader cases and answer key

Each case concerns a single work. A reader receives [cases.json](cases.json). The scorer receives the reader's output and [answer_key.json](answer_key.json). The key was written from the source observations and original case narratives before running either reader. The PLOS case checks two named contributor roles on two surfaces; it also retains one author's name as a literal excerpt. The remaining cases are original, fictional histories. These are specified regression outcomes, not a random sample of corrections or observations about a reader's speed.

## Questions and evidence

| Case | Question a reader should be able to answer |
|---|---|
| `plos-role-lists` | Which people does the selected PLOS surface list for data curation or visualization? Which published entries support the answer? |
| `shared-byline` | Which catalog records does a shared byline make discoverable? What additional evidence identifies the writer of this issue? |
| `undo-reference-rewrite` | After one reversible redirect ends, which card still independently cites its target? |
| `two-support-withdrawals` | Which issuer still supports translation after one withdraws its credit? What may a public reader see at the same content revision after permission changes? |
| `issuer-local-credit-citations` | Two issuers each signed entry 7 on the same page. Whose credit survives withdrawal, which words support it, and what may a public reader display now? |
| `conflated-at-ingest` | What can the reader recover when the exporting catalog used one label for a place and a collective and omitted individual assignments? |

The sibling [credit scenarios](../../credits/README.md) exercise the index subscriber, including gaps, replay, and public or restricted support. Here the reader must answer role-specific questions, cite a particular surface, and apply a separately selected viewer policy to an exact edition. Two pieces of evidence may come from the same publisher. Counting rows does not establish the number of independent informants. A missing credit means this view supplies no matching support. The key makes no determination about what an individual actually did.

## Wire format, version 1

- All identifiers are local to one case unless a source URL is explicitly cited here. `work` identifies its sole work. A source `id` identifies one immutable edition of its text. Source edition strings are opaque. Synthetic catalog targets use qualified identifiers such as `people:remy`.
- Read steps through the exact requested content revision `at`. Each step first withdraws listed supports, then inserts or replaces each `put` under its own `(issuer, id)` key. An omitted list is empty. These patches start at 1 and require each intervening revision. Complete fixture files can list steps out of order. A live subscriber needs missing predecessors before publishing a later patch. Content and policy revision numbers are JSON integers in the range 0 through 2⁶³ − 1.
- The publisher owns its local assertion ID. Two publishers can both issue entry `"7"` on one page. Withdrawing `{"issuer":"press","id":"7"}` leaves the guild's entry intact. A bare withdrawal string resolves against issuers seen through that patch, including its puts. Multiple owners make the withdrawal ambiguous and reject the patch. A selector with no matching assertion supplies no evidence.
- Each assertion has an ID, kind, issuer, source and target. A `printed` assertion also has a half-open `[start,end)` range counted in Unicode scalar values. A `credit` has an exact, case-sensitive role and may have an approved `excerpt` with `start` and `end` on that source edition. A replacement must supply its own excerpt approval. A `reference` is a published catalog reference. A `redirect` has a `from` target and a destination `target`.
- The independently selected policy revision lists complete `hidden` supports and optional `hidden_excerpts`. A hidden support disappears from every public answer. An excerpt-only hide still permits the role credit and its issuer to be returned. Qualified selectors can pre-register a restriction before an assertion exists. A legacy bare policy selector resolves against *all issuers in the full imported work history*, including later factual editions. A bare ID with multiple owners rejects the manifest, even for an early factual read. An unknown bare ID grants no evidence. Use explicit issuer and ID for durable policy rules, including pre-registered hides when another issuer may later reuse the number.
- A trusted application selects the current disclosure policy and authorized audience even when it reads older facts. The example accepts these fields from its trusted caller. `staff` permits proof locators for approved excerpts. Binding real credentials to that privilege belongs to the consuming service. The wire assumes visible work, role, target, issuer and local ID labels themselves may be disclosed.
- A query's optional `source` and `issuer` independently filter the assertions answering the question. `printed` returns exact source substrings. `credit` returns targets for a role. `evidence` returns visible supporting IDs for an exact role and target. It rejects ambiguous results if two matching issuers used the same local ID. Qualify the query by issuer or request `cited_credit` to retain both. `reference` follows at most one visible redirect, which can come from another source. The histories contain at most one active redirect per input and no chains. Credit refers to the target originally asserted.
- These four questions return sorted unique strings. `cited_credit` takes an exact role and target and returns records in issuer/local-ID order. Each record contains `work`, `issuer`, `id`, `role`, `target` and `excerpt`. The public excerpt is either `{"quote":"…"}` or `null`. Missing approval and withheld wording yield the same public value. Staff also receives a serializable `TextCitation` for an approved excerpt, with the source edition, whole-text SHA-256 and scalar range. No proof locator appears in the public response. The batch envelope is `{"version":1,"answers":{"case-id":{"query-id":[…]}}}`.

## Follow a published credit to its wording

Pass one JSON question on standard input to either reader. For example, from the repository root:

```sh
cases=verification/index_consumers/reader_cases/fixtures/cases.json
echo '{"id":"read","kind":"credit","role":"Data curation","at":2,"policy":1,"audience":"public"}' |
  cargo run --quiet --locked -p anno --no-default-features --example reader_answer -- "$cases" --ask plos-role-lists
```

For each returned target, send a second question with `kind:"cited_credit"` and that `target`. Timur Nuritidinow has one quoted attribution on the Authors page and one in the correction. Arne Mueller has a visible attribution but no approved excerpt on his own assertion. Sharing a source or appearing in someone else's approved quotation gives no additional excerpt approval. The same question works through `python3 verification/index_consumers/reader_cases/check.py --cases "$cases" --ask plos-role-lists`. The [client exercise](../citation_client.py) sends the runtime requests, then independently checks each staff receipt against the scoped source, exact UTF-8 digest and quoted scalar range. The full suite runs this exercise.

## Source attribution and reuse

The `plos-role-lists` case and its answer-key entry adapt two PLOS publications. Their publisher identifies the copyright holders as **© 2019 Keppler et al.** and **© 2025 Keppler et al.**, respectively. The adapted material remains under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

- Keppler et al., *Validity of accelerometry in step detection and gait speed measurement in orthogeriatric patients* (2019). We transcribed one name and selected the people listed under two roles on the article Authors page observed November 25, 2026. The source text in the fixture rearranges those entries, drops other roles and normalizes whitespace. [Article and license](https://doi.org/10.1371/journal.pone.0221732) · [Authors page](https://journals.plos.org/plosone/article/authors?id=10.1371/journal.pone.0221732)
- Keppler et al., *Correction: Validity of accelerometry in step detection and gait speed measurement in orthogeriatric patients* (November 24, 2025). The second source contains only the data curation and visualization lists, with presentation markup removed. The fixture and key convert printed names to local identifiers. [Correction and license](https://doi.org/10.1371/journal.pone.0337268)

Both surfaces were observed on November 25, 2026. The two fixture revisions represent ingesting those surfaces in succession. They provide no archived view of the article's 2019 contributor page. Source-specific questions remain answerable after both surfaces are ingested.

The five synthetic cases, their corresponding answers, and the original documentation use the repository's `MIT OR Apache-2.0` terms. Keep the PLOS attribution and license with copies of the adapted material. The synthetic examples use original wording and contain no external images or identifying private records.
