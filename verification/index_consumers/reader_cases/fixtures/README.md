# Reader cases and answer key

Each case concerns a single work. A reader receives [cases.json](cases.json). The scorer receives the reader's output and [answer_key.json](answer_key.json). The key was written from the source observations and original case narratives before running either reader. The PLOS case checks two named contributor roles on two surfaces; it also retains one author's name as a literal excerpt. The remaining cases are original, fictional histories. These are specified regression outcomes, not a random sample of corrections or observations about a reader's speed.

## Questions and evidence

| Case | Question a reader should be able to answer |
|---|---|
| `plos-role-lists` | Which people does the selected PLOS surface list for data curation or visualization? Which published entries support the answer? |
| `shared-byline` | Which catalog records does a shared byline make discoverable? What additional evidence identifies the writer of this issue? |
| `undo-reference-rewrite` | After one reversible redirect ends, which card still independently cites its target? |
| `two-support-withdrawals` | Which issuer still supports translation after one withdraws its credit? What may a public reader see at the same content revision after permission changes? |
| `conflated-at-ingest` | What can the reader recover when the exporting catalog used one label for a place and a collective and omitted individual assignments? |

The sibling [credit scenarios](../../credits/README.md) exercise the index subscriber, including gaps, replay, and public or restricted support. Here the reader must answer role-specific questions, cite a particular surface, and apply a separately selected viewer policy to an exact edition. Two pieces of evidence may come from the same publisher. Counting rows does not establish the number of independent informants. A missing credit means this view supplies no matching support. The key makes no determination about what an individual actually did.

## Wire format, version 1

- All identifiers are local to one case unless a source URL is explicitly cited here. `work` identifies its sole work. A source `id` identifies one immutable edition of its text. Source edition strings are opaque. Synthetic catalog targets use qualified identifiers such as `people:remy`.
- Read steps through the exact requested content revision `at`. Each step first withdraws the listed assertion IDs, then inserts or replaces the assertions in `put` by ID. An omitted list is empty. These patch histories start at 1 and require each intervening revision. Full fixture files may list steps out of order. A live subscriber that receives a later patch first needs the missing patches before it can publish that revision.
- Assertion IDs identify supports within the entire work history. An adapter importing two publishers' independent local assertion number 17 must qualify those IDs before writing the fixture. Reuse an ID when the issuer revises that same support. Record a separate ID for genuinely independent support so either can be withdrawn alone.
- Each assertion has an ID, kind, issuer, source and target. A `printed` assertion also has a half-open `[start,end)` range counted in Unicode scalar values. Read its exact substring. A `credit` assertion also has an exact, case-sensitive role. A `reference` is a published catalog reference. A `redirect` has a `from` target and a destination `target`.
- Use the complete policy identified by the query's `policy` revision. Remove its hidden assertion IDs for the `public` audience before any projection. The authorized `staff` audience sees active assertions. This fixture chooses the audience and policy for each question; a service must supply its own trusted viewer context.
- A query's optional `source` and `issuer` independently restrict the assertions answering the question. Two issuers can publish different attributions in one source edition, as in the joint meeting minutes. `printed` returns literal substrings. `credit` returns targets with the requested role. `evidence` returns IDs of visible credit assertions matching both role and target. `reference` returns reference targets after any visible one-hop redirect; the redirect may come from another source. The histories have at most one active redirect per input and no chains. Credit never traverses a redirect.
- Return sorted, unique strings per question. Output shape is `{"version":1,"answers":{"case-id":{"query-id":["value"]}}}`. Empty lists are valid. Evidence IDs let a caller locate the assertion and its issuer and source within the case.

## Source attribution and reuse

The `plos-role-lists` case and its answer-key entry adapt two PLOS publications. Their publisher identifies the copyright holders as **© 2019 Keppler et al.** and **© 2025 Keppler et al.**, respectively. The adapted material remains under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

- Keppler et al., *Validity of accelerometry in step detection and gait speed measurement in orthogeriatric patients* (2019). We transcribed one name and selected the people listed under two roles on the article Authors page observed November 25, 2026. The source text in the fixture rearranges those entries, drops other roles and normalizes whitespace. [Article and license](https://doi.org/10.1371/journal.pone.0221732) · [Authors page](https://journals.plos.org/plosone/article/authors?id=10.1371/journal.pone.0221732)
- Keppler et al., *Correction: Validity of accelerometry in step detection and gait speed measurement in orthogeriatric patients* (November 24, 2025). The second source contains only the data curation and visualization lists, with presentation markup removed. The fixture and key convert printed names to local identifiers. [Correction and license](https://doi.org/10.1371/journal.pone.0337268)

Both surfaces were observed on November 25, 2026. The two fixture revisions represent ingesting those surfaces in succession. They provide no archived view of the article's 2019 contributor page. Source-specific questions remain answerable after both surfaces are ingested.

The four synthetic cases, their corresponding answers, and the original documentation use the repository's `MIT OR Apache-2.0` terms. Keep the PLOS attribution and license with copies of the adapted material. The synthetic examples use original wording and contain no external images or identifying private records.
