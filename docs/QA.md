# Fixtures, datasets and QA

Choose the evidence for the question you want to answer. A source offset bug, a model comparison, and an unreadable web page each need a different test.

| Question | Input and check |
| --- | --- |
| Did this code change break a known behavior? | Run a small checked-in input through the public API or CLI. The public-source suite checks exact text, label and Unicode character range on excerpts from NASA, the Spanish official gazette, Japanese literature and IANA. Include a negative example such as an entity-shaped string inside a gap. [Fixtures and source terms](../crates/anno-cli/tests/fixtures/public_sources/README.md) |
| Does the scorer itself give the right answer? | Compare against an independent oracle. Examples include enumerating character masks, transforming Unicode or line endings while checking the original text, and replaying the official CorefUD scorer. Property tests exercise many inputs. The optional mutation job checks whether tests detect selected changes to span logic. [Reference scorer](../scripts/README.md) · [Mutation workflow](../.github/workflows/research.yml) |
| Did extraction quality change on labeled documents? | Run the same examples, labels, backend configuration and scorer for both versions. The fixed panels cover WikiGold, CoNLL2003Sample, WNUT17 and GAP. The eight-document PII fixture measures masking errors on its authored examples. Use a separate representative held-out split for a model decision. [Panel manifest](../scripts/qa/core-panel.json) · [PII fixture and scoring](eval/pii-char-masking.md) |
| Does extraction work on complete real documents? | Replay a pinned source with independent expected results. The NASA check derives 457 date labels from catalog link targets and checks their exact ranges on the rendered page. A score on those cells describes calendar dates in those two catalogs. [Full-page replay](../crates/anno-cli/tests/fixtures/public_sources/README.md) |
| What breaks on unfamiliar sites? | Sample live pages from different publishers, languages and formats. Record the resolved URL, content identity, backend and failure stage. A page with zero entities can be valid. Live results measure retrieval, decoding and output integrity unless someone has supplied labels. [Live URL harness](../scripts/qa_random_urls.py) |
| What should run in a limited CI budget? | Keep required regression checks deterministic. Fixed panels compare like with like. Muxer can give extra attempts to previously failing cells and direct coverage toward undersampled candidates. Track which cells actually ran, including missing data and infrastructure failures. [Sampling policies and controls](EVALUATION_SAMPLING.md) |

## Run a check

From the repository root, these checks use checked-in inputs. Cargo needs its dependencies installed or cached.

```sh
python3 -m unittest discover -s scripts/qa -p 'test_*.py'
cargo test --locked -p anno-cli --no-default-features --features eval,extractor-html2text --test public_sources
```

To run the fixed WikiGold smoke panel, build the CLI and prepare its cache once. The preparation script verifies the pinned corpus digest and downloads it if needed. The panel disables dataset and model downloads. It expects a measured result from each of two backends on up to 20 examples at seed 42.

```sh
cargo build --locked -p anno-cli --no-default-features --features eval,extractor-html2text
export ANNO_CACHE_DIR="$PWD/.generated/qa-cache"
bash scripts/qa/prepare_matrix_wikigold.sh
mkdir -p .generated/anno-qa
OUT=$(mktemp -d "$PWD/.generated/anno-qa/fixed-panel.XXXXXXXX")
just qa-panel target/debug/anno smoke "$OUT"
```

Open the printed output directory. Read `panel-summary.md` for per-cell outcomes and `smoke-seed42.json` for the underlying measurements. The panel checks coverage and finite scores. A change in F1 needs a paired comparison on the same inputs. The one-sentence WikiGold-format CI fixture, the labeled WikiGold corpus, and a random Wikipedia page are three distinct sources.

To investigate a result, pass the panel's printed output directory and the claim to the coordinator. This prints a local handoff with the observed cells, the independent review prompt, and a worked WikiGold example. Give the oracle reviewer only the source and contract so its expected answer remains independent.

```sh
just qa-round "$OUT" "Does removing spaces inside parentheses change organization boundaries?"
```

The same command accepts a matrix slot's receipt directory. To time agent calls and their replay, follow the [traced round](ADVERSARIAL_QA.md#trace-one-author-and-one-replay). Start directly with the [coordinator prompt](ADVERSARIAL_QA.md) for fuzzing or web discovery before a panel exists.

#For an equal-budget comparison on public, labeled text across ten domains, prepare [the domain fixtures](DOMAIN_QA.md).

## Check what a negative span means

Use `exact_span` when the source range must receive one of the listed types. Use `no_overlap` when the specified types must leave the entire range clear. Both use half-open Unicode scalar offsets. For example, an ORG spanning `Iona Tress (CPA` violates a protected range for `CPA` even if the extractor never emits the credential on its own. A prediction ending at the first character of the protected range may pass. Pair a negative range with a positive mention so an empty extractor cannot pass the whole case.

```sh
just qa-spans target/debug/anno stacked
python3 scripts/qa/check_span_assertions.py --fixture cases.json --predictions batch.jsonl
```

The [diagnostic fixture](../scripts/qa/fixtures/typed-span-assertions.json) includes credentials, a delivery channel and organization initials. Its failures expose current backend behavior. Exit 0 means all assertions passed, 1 means at least one assertion failed, and 2 means the input or run was invalid. For a custom assertion type, pass its exact spelling with `--allow-type LABEL`. Other custom predictions still appear in the overlap report. Saved batch JSONL must include a per-document `source_sha256` matching the exact UTF-8 fixture text, including line endings. The report lists overlapping entities so a reviewer can inspect both boundaries and labels.

To compare weekday handling, run `just qa-spans target/debug/anno pattern scripts/qa/fixtures/weekday-span-assertions.json`.
Each case protects a person or film title and checks a date in the same sentence. The two weekday
dates expose known misses in the pattern backend. The offline matrix smoke test also protects
“Wednesday Addams” in a sentence with no date.

For a change to a partially passing backend, add `--baseline-predictions old.jsonl` to the replay command. Both versions must match the same source revision. For an older CLI that omits the digest, capture a fresh baseline from the original binary with `python3 scripts/qa/check_span_assertions.py --fixture cases.json --binary /path/to/old/anno --model stacked --save-predictions old.jsonl`. Use a new output path. The checker saves valid predictions even when assertions fail. Older saved files without a source digest require a new run. In comparison mode exit 1 means a previously passing assertion was lost, even when the total improved. Unchanged failures remain visible in the report. The fixture includes the same string, `CPA`, as a credential and as the initials of Coastal Planning Authority. A blanket suppression can improve the credential count while losing the organization.

## What runs in GitHub Actions

The required Ubuntu CLI integration job runs against local fixtures, including public-source excerpts. The additional eval-feature job exercises the loopback HTTP cases. The Python QA job tests the harness itself with fixed inputs. A separate path-filtered benchmark fetches checksum-pinned WikiGold and records corpus scores. The full 457-date NASA replay is an explicit command. Fuzzing is available locally, and a maintainer can launch the bounded mutation job from a matching branch. [CI workflow](../.github/workflows/ci.yml) · [WikiGold benchmark](../.github/workflows/bench-extraction.yml) · [Fuzz target](../fuzz/README.md)

For live discovery, `just qa-random` balances catalog topics and publishers by default. For offline exact-span assertions, `just qa-domain-spans target/debug/anno` exercises a labeled panel. The live runner needs an eval-enabled CLI.

```sh
export ANNO=target/debug/anno
python3 scripts/qa_random_urls.py --count 10 --seed 42 --backends pattern --output qa/reports/field.md
just qa-round qa/reports/field.json 'Which selected topic still needs an independent label?'
python3 scripts/qa_random_urls.py --replay qa/reports/field.json --output qa/reports/revisit.md
```

The catalog assigns one topic to each publisher, including its feeds. Each pass samples distinct publishers. The next publisher is chosen from the least represented available topics. A topic describes a catalog entry. The Markdown report counts eligible, selected and verified entries alongside distinct final hosts and rendered text hashes. A source counts as verified when every requested backend reports the same final URL and text hash. Repeated content, failures and mismatches remain visible. These checks measure input coverage and output integrity. Accuracy requires independent labels.

The sibling JSON receipt records the original feed or page, selected article, final URL, requested backends and rendered text hashes. Replaying a receipt revisits its selected URLs and compares them to its original successful results. A changed final URL or rendered hash, a missing baseline or an incomplete current run yields a nonzero exit. The prior receipt stays available. Live RSS/Atom entries may change between seeded selections. `--no-rss` restricts sampling to standing pages. Use `--category all` for the wider multilingual discovery pool, or select a category such as `wikipedia`. A language filter alone searches the multilingual pool; pass `--cross-domain --lang ...` to limit the topical catalog. Contributor PRs run deterministic offline checks.

## Score person highlights and referring expressions

Use the local CPU FP32 export of `biu-nlp/f-coref` at revision `e91dfff12879495d882ba9460d0c5d5dd44ade59`. The model workflow pins the Python dependencies, verifies the original checkpoint, exports it and verifies all four converted files. Set `FCOREF_MODEL_DIR` to that export directory. The panel mode uses two inference threads, the example’s Unicode word boundaries and a 4,096-token document budget. [Export procedure and artifact hashes](../.github/workflows/fcoref-cli.yml)

```sh
panel=testdata/grounded/dual_span_people.json
mkdir -p target/qa-person
jq '[.documents[] | {id, text}]' "$panel" > target/qa-person/inputs.json
ANNO_NO_DOWNLOADS=1 cargo run --locked -p anno --no-default-features --features onnx \
  --example fcoref_word_spans -- "$FCOREF_MODEL_DIR" target/qa-person/inputs.json \
  > target/qa-person/predictions.json
python3 scripts/qa/score_person_mentions.py "$panel" target/qa-person/predictions.json
```

The example runs `FCoref::analyze_words` with caller-supplied word boundaries. The raw-text route used by `FCoref::resolve` tokenizes the original string and can produce different clusters, even with the same weights. Identify the inference route when reporting either result.

In version 1 gold, each mention supplies `highlight_surface` and one `referential_surfaces` entry per `referential_spans` entry. Each recorded surface must equal the corresponding source slice. Every accepted referring expression contains its highlight. Each negative span supplies its own `surface` and must be distinct and disjoint from the positive mentions.

Each row in the receipt's `results` array supplies the document `id`, `source_sha256` of the verbatim UTF-8 input, `mentions` with scalar `start`, `end` and `surface`, and `clusters` of scalar range pairs. Each cluster member must appear among the candidates. A coarse caller-supplied word may project mentions from two model clusters onto the same source span. The scorer counts that span once in selection and each pair once if its endpoints share a cluster. It preserves the original clusters without inferring links between their other members. Documents with empty model output still need a row.

The scorer evaluates the same output against each gold occurrence's compact highlight and accepted referring expressions. It reports all proposed spans and the subset used in clusters. For one-to-one person selection, use `covered_mentions / emitted` for precision and `covered_mentions / gold_mentions` for recall. Several accepted ranges can cover one occurrence. An exact hit on a marked distractor counts separately from other unmatched ranges. A general coreference model can legitimately propose non-person entities outside this panel's person task.

For each boundary contract, `joined_positive / positive_pairs` measures end-to-end same-person pair recall. `joined_negative / negative_pairs` measures the fraction of distinct-person pairs merged. `covered_positive` and `covered_negative` count pairs whose endpoints were both proposed. A missed endpoint stays in the end-to-end denominator. Report zero denominators as unavailable. Unresolved references are reported separately and have no pair truth. The human–distractor counts cover pairs of an annotated human occurrence and an exact marked distractor within the same document. They expose contaminated clusters even when no distinct people were merged. These are within-document measurements. The receipt carries no external catalog or linking decisions.

### Compare compact name displays

Use the pinned local FCoref export described above and the stacked NER backend. Both commands below consume only IDs and verbatim text. Set `FCOREF_MODEL_DIR` to the export directory and prefetch build dependencies for an offline replay.

```sh
panel=testdata/grounded/paired_display_people.json
out=target/qa-display
mkdir -p "$out"
jq '[.documents[] | {id, text}]' "$panel" > "$out/inputs.json"
jq -c '.[]' "$out/inputs.json" > "$out/inputs.jsonl"
ANNO_NO_DOWNLOADS=1 cargo run --offline --locked -p anno --no-default-features --features onnx \
  --example fcoref_word_spans -- "$FCOREF_MODEL_DIR" "$out/inputs.json" > "$out/coref.json"
ANNO_NO_DOWNLOADS=1 cargo run --offline --locked -p anno-cli --bin anno --no-default-features --features onnx,extractor-readability \
  -- batch --stdin --format jsonl --model stacked < "$out/inputs.jsonl" > "$out/ner.jsonl"
python3 scripts/qa/score_compact_display.py "$panel" "$out/coref.json" "$out/ner.jsonl" > "$out/scores.json"
jq '{totals, stronger_baseline, gate, passes}' "$out/scores.json"
```

The global comparison counts each distinct source range once per document and includes all eligible name occurrences. The full-phrase baseline displays the source ranges used in decoded clusters. Standalone NER displays every PER range, including predictions on documents with no decoded phrases. On each decoded phrase the candidate displays the sole distinct fully contained PER range and abstains with zero or multiple choices. Its displays therefore form a subset of standalone NER. Reading the global counts gives selection precision as `correct / emitted` and coverage as `correct / eligible`. Zero denominators indicate an unavailable rate.

For attribution, all three phrase views use the same source ranges and cluster memberships. Only an exact licensed referring expression supplies a gold occurrence owner. An explicitly protected expression requires no compact display. A nested name can be a valid standalone highlight and an incorrect assignment to its parent. A different occurrence counts as a known wrong person only when both local identities are resolved and differ. The contained-all diagnostic exposes ambiguous choices. Each document retains the paired variant ID, source occurrences, protected cases and decisions for inspection. No external identity, KB linking or new cluster inference is evaluated.

The gate requires more correct global highlights than both existing baselines at no more wrong highlights than the stronger baseline (most correct, then fewest wrong). Source memberships must be preserved and no new wrong protected assignment may be introduced. Because selection only removes standalone PER ranges, the strict global gain is impossible for this candidate. Phrase-level precision and abstention describe its behavior under a different, conditional unit.

On 24 November 2026, the pinned CPU model and stacked NER produced these results on the 24 paired documents. The fixture SHA-256 was `19db4b9d544ec861de0aeb81ea821487e6e47175e0a442de342ddfd6bfe46f33`. The FCoref revision was `e91dfff12879495d882ba9460d0c5d5dd44ade59`.

| Display rule | Correct highlights | Wrong highlights | Eligible occurrences |
|---|---:|---:|---:|
| Full referring phrase | 18 | 19 | 44 |
| Standalone stacked PER | 20 | 14 | 44 |
| Single contained PER | 8 | 5 | 44 |

The candidate dropped 12 correct and nine wrong standalone highlights. In `possessor_a`, it also selected the possessor’s name for a phrase referring to the assistant. The candidate failed both the global recovery gate and the protected-attribution gate.

## Find a dataset or add a case

Use an eval-enabled CLI to browse the [dataset registry](../crates/anno-eval/src/eval/dataset_registry.rs).

```sh
anno dataset list --task ner --loadable
anno dataset info --dataset WikiGold
```

Check the upstream source, license, available split, loader and cache manifest before quoting a result. A catalog entry describes a dataset. A loader describes a supported format. A result requires actual examples, an identified backend and completed scoring. Record the revision or digest, exact split or sample IDs, seed, labels, feature flags, scorer policy and number of measured examples. Keep calibration documents separate from the held-out comparison.

When a live run reveals a bug, preserve the original revision and rendering details, then reduce it to the smallest redistributable input that still fails. Put the expected behavior beside the test that consumes it. For a restricted source, use an independently authored reproducer or a permitted fetch recipe with a digest. Preserve the coordinate system of the original data, including whether source offsets address bytes, Unicode characters or token-reconstructed text. [Source and offset contract](CONTRACT.md)

### Agency names in government publications

Federal Register records provide an independent source of organization names.
The following replay compares each issuing agency's metadata against its own
document abstract. It scores exact, case-insensitive occurrences of multiword
names, counts overlapping aliases once, and checks that every predicted span
slices back to its reported text. The report gives the document and anchor
denominators. Other mentions in the abstract have no gold labels, so this panel
supports recall on those anchors only.

```sh
curl --fail --silent --show-error --get 'https://www.federalregister.gov/api/v1/documents.json' \
  --data-urlencode 'conditions[publication_date][is]=2024-06-03' \
  --data-urlencode 'per_page=1000' --data-urlencode 'order=document_number' -o /tmp/fr-june.json
python3 scripts/qa/check_agency_spans.py target/debug/anno /tmp/fr-june.json
```

The June 3, 2024 issue yielded 53 anchors in 47 abstracts. A separate September 3,
2024 issue yielded 50 in 44. Save each response and its reported SHA-256 for
repeat comparisons. Use `--cases PATH` to inspect individual spans and look
up the corresponding Federal Register document number.
