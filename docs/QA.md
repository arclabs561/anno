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

### Check what a negative span means

Use `exact_span` when the source range must receive one of the listed types. Use `no_overlap` when the specified types must leave the entire range clear. Both use half-open Unicode scalar offsets. For example, an ORG spanning `Iona Tress (CPA` violates a protected range for `CPA` even if the extractor never emits the credential on its own. A prediction ending at the first character of the protected range may pass. Pair a negative range with a positive mention so an empty extractor cannot pass the whole case.

```sh
just qa-spans target/debug/anno stacked
python3 scripts/qa/check_span_assertions.py --fixture cases.json --predictions batch.jsonl
```

The [diagnostic fixture](../scripts/qa/fixtures/typed-span-assertions.json) includes credentials, a delivery channel and organization initials. Its failures expose current backend behavior. Exit 0 means all assertions passed, 1 means at least one assertion failed, and 2 means the input or run was invalid. For a custom assertion type, pass its exact spelling with `--allow-type LABEL`. Other custom predictions still appear in the overlap report. Saved batch JSONL must include a per-document `source_sha256` matching the exact UTF-8 fixture text, including line endings. The report lists overlapping entities so a reviewer can inspect both boundaries and labels.

For a change to a partially passing backend, add `--baseline-predictions old.jsonl` to the replay command. Both versions must match the same source revision. For an older CLI that omits the digest, capture a fresh baseline from the original binary with `python3 scripts/qa/check_span_assertions.py --fixture cases.json --binary /path/to/old/anno --model stacked --save-predictions old.jsonl`. Use a new output path. The checker saves valid predictions even when assertions fail. Older saved files without a source digest require a new run. In comparison mode exit 1 means a previously passing assertion was lost, even when the total improved. Unchanged failures remain visible in the report. The fixture includes the same string, `CPA`, as a credential and as the initials of Coastal Planning Authority. A blanket suppression can improve the credential count while losing the organization.

## What runs in GitHub Actions

The required Ubuntu CLI integration job runs against local fixtures, including public-source excerpts. The additional eval-feature job exercises the loopback HTTP cases. The Python QA job tests the harness itself with fixed inputs. A separate path-filtered benchmark fetches checksum-pinned WikiGold and records corpus scores. The full 457-date NASA replay is an explicit command. Fuzzing is available locally, and a maintainer can launch the bounded mutation job from a matching branch. [CI workflow](../.github/workflows/ci.yml) · [WikiGold benchmark](../.github/workflows/bench-extraction.yml) · [Fuzz target](../fuzz/README.md)

For discovery across publishers, choose an eval-enabled CLI and a bounded number of pages:

```sh
ANNO=target/debug/anno python3 scripts/qa_random_urls.py --cross-domain --count 3 --seed 42 --backends pattern
```

The pool combines standing pages and same-publisher articles discovered from RSS or Atom feeds. The sampler visits distinct publishers within each pass and bounds feed reads. Use `--no-rss` for standing pages only. The seed reproduces the choice among unchanged candidates. Publishers can change a page, redirect a URL or stop responding. Compare backend outputs only when their rendered source hashes agree. Scheduled or manually dispatched live QA keeps the reports when some pages fail. The standard contributor path uses local inputs.

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
