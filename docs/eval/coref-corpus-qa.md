# Replay the coreference QA corpora

Use these local panels to investigate exact mention boundaries, competing antecedents, and pronouns with no visible antecedent. The seven adapters read original publisher artifacts. They reconstruct the cases and gold labels on demand. The same caller can check source pins, replay an individual case, or score predictions from another backend. The command is also available as `python3 scripts/qa/coref_corpus.py`. [Source revisions, attribution, licenses and storage](coref-corpus-sources.md)

| Corpus name | Frozen cases | Source unit and scoring contract |
|---|---:|---|
| `litbank` | 47 | 24 nested pairs and 23 pronoun choices across 21 held-out literary works. Pairs use the original entity chains. The choice scorer also checks other annotated mentions of the same person. |
| `gum` | 235 | 219 nested pairs with a proper-name-headed person inside, plus 16 pronoun choices. The containing phrase can refer to a different entity. |
| `phrase-detectives` | 173 | Held-out pronouns across 16 source groups. The expert export supplies semantic types. With the two offered spans, 15 are explicitly non-referring and 64 have an earlier referent elsewhere. Another 85 have a visible correct candidate, including seven with both choices correct. Seven without a recoverable prior mention and two discourse-new cases remain unscored. |
| `gap` | 88 | One case per article. The ordinary 64 yield 35 A, 20 B and nine neither. The separate 24-case challenge has two wrong options in every case. Compared with the initial three-panel reserve, this selection replaces four articles overlapping a fourth historical evaluation. |
| `winogender` | 720 | 120 templates with six variants each. Keep all variants together for development splits. Each native receipt scores the full role phrase and the bare role slot. |
| `dutch-openboek` | 1,396 | 795 nested pairs and 601 antecedent choices across two development works in Dutch. |
| `french-litbankfr` | 1,608 | 571 nested pairs and 1,037 antecedent choices across four development works in French. |

## Check and run

`list` prints the source registry without opening any data. Supply a local source only after checking its terms for your use. For one corpus use `--source PATH`. To check all seven, make a private JSON map from corpus name to a local path. Relative paths resolve against the map's directory. The two CorefUD entries can point to the same release ZIP. Winogender expects the directory containing both TSV files.

```sh
just qa-coref-corpora list
just qa-coref-corpora check all --sources-file /private/anno/sources.json
just qa-coref-corpora check gum --source /private/anno/en_gum-ud-test.conllu
cargo build --locked -p anno --example fcoref_word_spans --features onnx
just qa-coref-corpora run gum --sources-file /private/anno/sources.json \
    --runner target/debug/examples/fcoref_word_spans --model-dir /private/models/fcoref \
    --limit 4 --seed first-pass > /private/anno/gum-receipt.json
```

Use a model appropriate to the corpus language. The native runner reads a local FCoref export. `--limit` caps cases after filtering. Omit it to run the panel. `--tag` filters by a tag from `check`; `--case` selects an opaque ID from a scored receipt. Replay uses deterministic sampling and checks the text hash and Unicode scalar offsets for every returned document. Counts across panels use different sampling units and label contracts.

## Use another producer

The explicit `inputs` command writes a new mode-0600 local file containing only `id` and `text`. A separate owner-only `<output>.provenance.json` records the pinned release, terms, attribution, source file location and per-case publisher locators. A publisher locator can reveal the answer. Keep both files on storage authorized for the text, pass only the inputs file to the producer, and reuse the same case filters for `score`. The native path uses a temporary directory and discards the text-bearing prediction stream after scoring.

```sh
just qa-coref-corpora inputs french-litbankfr --sources-file /private/anno/sources.json \
    --limit 4 --output /private/anno/fr-inputs.json
just qa-coref-corpora score french-litbankfr --sources-file /private/anno/sources.json \
    --limit 4 --predictions /private/anno/fr-predictions.json
```

A producer returns an object with a `results` array. Each result includes the input ID, `source_sha256` of the exact UTF-8 text, `mentions` with Unicode scalar `start`, exclusive `end` and `surface`, and `clusters` of distinct proposed spans. Each published cluster has at least two members. Every selected case needs one result. The scorer rejects stale hashes, mismatched surfaces, duplicate assignments and missing cases.

For a nested pair, missing either endpoint produces `unresolved`, including when the two gold entities differ. For a choice, `neither` means the exact target joined neither offered span. Use `target_linked_anywhere`, optional `alternate_gold_in_window` and `linked_alternate_gold` to distinguish the observable outcomes. With no annotated alternative in the window the last field is null. A silent model can agree with a gold neither on its offered choices. That does not establish a semantic non-reference judgment. Tags distinguish explicit non-reference, omitted referents and unresolved gold.

## Promote a measured case to adversarial replay

Set `ANNO_COREF_SOURCES` to the local path map, `ANNO_COREF_RUNNER` to the native executable, and `ANNO_COREF_MODEL_DIR` to the local export. Copy the opaque case ID from a scored receipt. For a read-only replay use `just qa-replay coref_corpus SLUG/ID`. A nested pair with a missing endpoint produces an explicit failure. An unscored gold case is ineligible for feedback. To retain a reproducible failure, use the existing [reviewed corpus witness workflow](../EVALUATION_SAMPLING.md#replay-a-reviewed-corpus-witness-in-a-triage-slot). The replay ledger binds the source, oracle, producer and model fingerprints and retains a compact outcome without passages or source notices.
