# Fixtures, datasets and QA

Choose the evidence for the question you want to answer. A source offset bug, a model comparison, and an unreadable web page each need a different test.

| Question | Input and check |
| --- | --- |
| Did this code change break a known behavior? | Run a small checked-in input through the public API or CLI. The public-source suite checks exact text, label and Unicode character range on excerpts from NASA, the Spanish official gazette, Japanese literature and IANA. Include a negative example such as an entity-shaped string inside a gap. [Fixtures and source terms](../crates/anno-cli/tests/fixtures/public_sources/README.md) |
| Does the scorer itself give the right answer? | Compare against an independent oracle. Examples include enumerating character masks, transforming Unicode or line endings while checking the original text, and replaying the official CorefUD scorer. Property tests exercise many inputs. The optional mutation job checks whether tests detect selected changes to span logic. [Reference scorer](../scripts/README.md) · [Mutation workflow](../.github/workflows/research.yml) |
| Did a reference disappear at detection, retrieval, or track assignment? | Run the [reference-resolution checks](../crates/anno-eval/research/reference/README.md). Compare actual Anno spans with labeled occurrences, then use bounded replays to separate eviction, ranking, and cluster errors. |
| Did extraction quality change on labeled documents? | Run the same examples, labels, backend configuration and scorer for both versions. The fixed panels cover WikiGold, CoNLL2003Sample, WNUT17 and GAP. The eight-document PII fixture measures masking errors on its authored examples. Use a separate representative held-out split for a model decision. [Panel manifest](../scripts/qa/core-panel.json) · [PII fixture and scoring](eval/pii-char-masking.md) |
| Does extraction work on complete real documents? | Replay a pinned source with independent expected results. The NASA check derives 457 date labels from catalog link targets and checks their exact ranges on the rendered page. A score on those cells describes calendar dates in those two catalogs. [Full-page replay](../crates/anno-cli/tests/fixtures/public_sources/README.md) |
| What breaks on unfamiliar sites? | Sample live pages from different publishers, languages and formats. Record the resolved URL, content identity, backend and failure stage. A page with zero entities can be valid. Live results measure retrieval, decoding and output integrity unless someone has supplied labels. [Live URL harness](../scripts/qa_random_urls.py) |
| What should run in a limited CI budget? | Keep required regression checks deterministic. Fixed panels compare like with like. Muxer can give extra attempts to previously failing cells and direct coverage toward undersampled candidates. Track which cells actually ran, including missing data and infrastructure failures. [Sampling policies and controls](EVALUATION_SAMPLING.md) |

## Checks for silent data loss and false passes

| Contract | Regression that exercises it |
| --- | --- |
| Import and export preserve the dataset or report an error | Put a malformed BRAT mention after a valid one and prefill the destination. Check the exit status and the destination bytes. Round-trip a mention with a word in the gap through each advertised format and compare the exact segments. |
| Scores depend on the annotations | Permute predictions, reverse an undirected edge with differently typed endpoints, and include duplicate spans with different labels. Check one-to-one matches, both per-label denominators, and deterministic tied votes. |
| A model result still points into the source | Pad the same token sequence or repeat whitespace between words while holding the model outputs fixed. Check the source spans and surfaces at the public interface. |
| QA credits the implementation that ran | Give two requested backends the same reported runtime identity, replay with a different selector, or change the model inventory. Check the saved receipt and the downstream coverage decision. |
| A release or cache can be verified by another job | Compare the tag, manifest version and registry checksum. Exercise duplicate uploads with mismatched bytes. Check the workflow’s saved paths against its retention policy. |

A type can require a parsed span, a validated threshold or a resolved identity at an internal boundary. Comparing against the original input, running producer or registry still needs a behavioral check. Keep these small cases in the ordinary PR suites so they also run for contributors.

## Try a workflow as a new reader

Start outside the checkout with a published package or a named build. Give a reader
a question and the public docs. For the harbor bulletin, ask which deadline
changed, whether the drawing moved, and what evidence an editor can keep.
Let the reader find the command, then have them replace one notice with their
own text. A second process or person should be able to reopen the result from
the files they receive. [Python bulletin example](../crates/anno-py/README.md#compare-dates-in-two-bulletins)

Record the artifact version and digest, first command attempted, help the
reader needed, and what the recipient could verify. Measure installation and
model setup separately from time spent doing the task. Mark simulated readers
as agents. Keep private inputs and exact prompts local.

For a language-sensitive variation, try the same December 4 to December 7
amendment as `2026年12月4日` and `2026年12月7日`, or as `٤ ديسمبر ٢٠٢٦` and
`٧ ديسمبر ٢٠٢٦`. Check the original sentences even if the change list is empty.
Use a known contract failure to gate a patch once its expected behavior is
agreed. Record newly uncovered language gaps with the input, backend and scope
per the [seed retention guidance](#decide-what-to-keep).

## Run a check

From the repository root, these checks use checked-in inputs. Cargo needs its dependencies installed or cached.

```sh
python3 -m unittest discover -s scripts/qa -p 'test_*.py'
cargo test --locked -p anno-cli --no-default-features --features eval,extractor-html2text --test public_sources
```

To investigate one reviewed failure, run its named case on the binary you are editing. This works on a dirty checkout. The currency case uses a project-authored sentence and requires no dataset, model download, agent or Muxer history.

```sh
cargo build --locked -p anno-cli --no-default-features --features extractor-html2text
ANNO_BIN=target/debug/anno just qa-replay typed_spans currency-print-spacing
```

Exit 0 means the assertions passed, 1 means an assertion failed, 2 means the command could not evaluate the case, and 3 means a required corpus input is unavailable. For a local fixture still under development, run `just qa-spans target/debug/anno stacked /path/to/fixture.json`. That command prints detailed findings, which can include source text. Keep its output with the private inputs.

To run the fixed WikiGold smoke panel, build the CLI and prepare its cache once. The preparation script verifies the pinned corpus digest and downloads it if needed. The panel disables dataset and model downloads. It expects a measured result from each of two backends on up to 20 examples at seed 42.

The authors’ versioned Figshare release lists the manual annotations in `wikigold.conll.txt` under CC BY 4.0. Its file 9446377 is 318,530 bytes with MD5 `0c1a4f91b9ea70f84ed43f4ff62fdfd5`. The pinned Git mirror used here has those same bytes, verified locally, and the preparation script checks SHA-256 `c797a64d0cf73ed058363f77671486bcfd413e70fda1726ad4a6ba624455225f`. The underlying Wikipedia passages retain separate reuse terms, including attribution, applicable share-alike provisions and notices. Raw corpus files stay in the job-local dataset directory. [Author release](https://doi.org/10.6084/m9.figshare.5462500.v1) · [Annotation paper](https://aclanthology.org/W09-3302/) · [Wikimedia reuse terms, §7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content)

```sh
cargo build --locked -p anno-cli --no-default-features --features eval,extractor-html2text
export ANNO_CACHE_DIR="$PWD/.generated/qa-cache"
bash scripts/qa/prepare_matrix_wikigold.sh
mkdir -p .generated/anno-qa
OUT=$(mktemp -d "$PWD/.generated/anno-qa/fixed-panel.XXXXXXXX")
just qa-panel target/debug/anno smoke "$OUT"
```

Open the printed output directory. Read `panel-summary.md` for per-cell outcomes and `smoke-seed42.json` for the underlying measurements. The panel checks coverage and finite scores. A change in F1 needs a paired comparison on the same inputs. The one-sentence CI fixture has the dataset ID `AnnoNerSmoke`. A WikiGold score requires the pinned labeled corpus.

To challenge a result in your current session, give a reviewer the contract and an input cleared for that reviewer. Freeze the expected answer before showing the candidate output. Use `just qa-oracle --help` for harness, headless Codex and direct OpenRouter dispatch. When another person or session needs context, `just qa-round "$OUT" "Does punctuation change organization boundaries?"` renders a brief from the saved receipt. [Adversarial QA](ADVERSARIAL_QA.md) · [Domain fixtures and source terms](DOMAIN_QA.md)

## Score document-level relations

The [document-level relation scorer](../crates/anno-eval/examples/docred/score.py) reads two DocRED-format arrays with the same documents and sentence tokens. Each endpoint indexes a document-local vertex containing one or more mention spans. For raw-text extraction, pass model-produced vertices and relations with `--endpoints predicted`. For a classifier given the gold vertices, use `--endpoints supplied`. The scorer checks the complete supplied inventory. Both modes score directed edges and preserve the original document boundary.

```sh
python3 crates/anno-eval/examples/docred/score.py dev.json predictions.json --endpoints predicted
```

This diagnostic requires exact endpoint mention sets. Types are ignored by default. Pass `--require-entity-type` to also require each endpoint's entity type, using its first mention as in the upstream JEREX loader. Vertex order is immaterial for uniquely identifiable sets. Two distinct Re-DocRED vertices in “Cambodia–Vietnam Friendship Monument” share a mention set. Predicted mode reports the number of ambiguous edges, a lower bound using unambiguous matches, and an upper bound allowing each ambiguous edge its own matching endpoint assignment. Supplied mode preserves ambiguous vertices by their original IDs. Scoring the revised development gold against itself yielded 17,284 matched edges in 500 documents with supplied vertices and 17,266 confirmed plus 18 ambiguous with independently aligned vertices. [Re-DocRED development annotations](https://github.com/tonytan48/Re-DocRED/blob/main/data/dev_revised.json)

For DWIE, use one directory of gold documents and one of predictions, with exact article `content` in both. The adapter verifies each half-open character range against its mention text. Its ordinary scores cover relations between anchored concepts. It separately counts relations involving concepts with no mentions. With supplied concept IDs it also scores the full inventory. The DWIE authors’ official scorer offers additional entity-centric and graded metrics. [Corpus and scorer](https://github.com/klimzaporojets/DWIE)

```sh
python3 crates/anno-eval/examples/docred/score_dwie.py gold/ predicted/ --endpoints predicted
```

The native mention-span evaluator reports strict and boundary scores separately for documents with predicted endpoints and those scored by heuristics over gold endpoints. An endpoint group with zero gold and zero predicted relations retains its counts and omits F1. The legacy aggregate scores an empty reference and empty prediction as 1.0. The JSON metric `gold_label_prompt_docs` counts documents whose extractor received the reference entity and relation label vocabulary. Predicted spans in these documents are still conditioned on knowing which labels appear. Use `anno benchmark --tasks relation --relation-schema schema.json` to provide the same entity and relation labels in the same order for all documents. The file has two nonempty arrays, `entity_types` and `relation_types`. The JSON receipt records the requested vocabulary and counts documents by the prompt passed to the extractor and the producer that returned the scored predictions. Producers can interpret or filter labels differently. Every reference relation stays in the recall denominator, including those with labels absent from the supplied vocabulary. [Full CrossRE AI and literature comparison](eval/crossre-full-prompt-comparison.md)

### Run the JEREX joint checkpoint

The three-document JEREX replay counts mention detection, coreference, entity typing and directed relations separately. A companion JointIE replay counts gold pairs sharing a scoring window, gold labeled edges that received scores, and final decoded edges on eight CrossRE records. An authored case checks that regrouping a mention preserves a saved concept relation and changes its current track projection. [Joint extraction results and replay commands](eval/joint-stage-baseline.md)

The [CPU inference adapter](../crates/anno-eval/examples/docred/predict_jerex.py) runs the upstream joint model on a local DocRED array. It drops gold vertices and labels before inference. The [JEREX prediction reader](../crates/anno-eval/examples/docred/score_jerex.py) accepts the upstream joint writer's `predictions.json`. The upstream writer omits document IDs, so the reader requires the original document order, verifies every source token, and converts sentence-local gold offsets onto JEREX's document token axis. The CPU adapter adds a title to each prediction, which permits disambiguation when two documents have identical token sequences. Pass `--require-entity-type` to include the endpoint classifications in the relation score.

The verified replay used the [JEREX checkout](https://github.com/lavis-nlp/jerex/tree/85851dfcbb485d9bfa2bf662b8073e826c760d7f), the five files from the [authors’ joint checkpoint directory](https://lavis.cs.hs-rm.de/storage/jerex/public/models/docred_joint/joint_multi_instance/), and the [authors’ end-to-end test split](https://lavis.cs.hs-rm.de/storage/jerex/public/datasets/docred_joint/test_joint.json). The checkpoint SHA-256 is `55bf677bcdfa66973a4e22859051327dd63bfd7a249e5b3e34cd08b95681d4c2`. Use Python 3.12 with torch 2.8 CPU, transformers 4.48.3, scikit-learn 1.3.2, NumPy 1.26.4, hydra-core, tqdm and Jinja2. Set `PYTHONPATH` to the upstream checkout.

```sh
python3 crates/anno-eval/examples/docred/predict_jerex.py ./joint_multi_instance ./documents.json ./predictions.json
python3 crates/anno-eval/examples/docred/score_jerex.py ./documents.json ./predictions.json
```

On “Jeff Prosserman”, the first document of the authors’ 700-document split, this CPU replay recovered all 31 mention clusters, assigned 30 correct entity types, and matched seven of nine gold relations from 14 predictions. The upstream scorer also returned 60.87% micro-F1 on this document.

## Score a LitBank book or cohort

The matched backend replay runs native mention-ranking, FCoref and Maverick on prefixes of all ten LitBank fold-0 test works and on 128 GAP pronoun cases. It reports exact mention coverage, cluster quality and the effective producer for each mention-input mode. [Results and replay](eval/coref-panels.md)

The registry selector `LitBank` downloads one pinned *Bleak House* excerpt. Its score describes one book. The upstream collection has 100 excerpts. For a cohort, record the upstream revision and the selected filenames, then call `CorefLoader::load_litbank_from_path` once per `coref/brat/*_brat.ann`. Keep each same-stem `.txt` beside its annotations. The loader returns the exact source and document-local gold chains, including singletons. Report the number of books, mentions and gold links with the result. [LitBank corpus and license](https://github.com/dbamman/litbank/tree/3e50db0ffc033d7ccbb94f4d88f6b99210328ed8)

The pinned LitBank release grants CC BY 4.0, which permits commercial reuse of the licensors’ material with attribution, a license link and an indication of changes. The authors selected pre-1923 Project Gutenberg source texts under U.S. public-domain criteria. When distributing prose, check the particular edition and the countries where it will be distributed. A fresh registry download embeds the sample’s attribution in `LitBank.cache`. Carry the notice with shared copies, including HTML evaluation reports that reproduce the excerpt or gold annotations. The [standalone notice](../crates/anno-eval/src/eval/litbank_attribution.json) also applies to older copies if **both** the decoded source text and annotations match its SHA-256 values. Record any downstream changes. [Release terms](https://github.com/dbamman/litbank/blob/3e50db0ffc033d7ccbb94f4d88f6b99210328ed8/README.md) · [Source selection, §2](https://aclanthology.org/N19-1220/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

For an older local cache containing raw annotations in `LitBank.cache`, place the matching source beside it as `LitBank.txt`. An annotation-only cache returns an error naming the missing path. Reading a compatible local cache preserves its existing bytes. To obtain the attributed publisher pair separately, point `ANNO_CACHE_DIR` at an empty directory and run `anno dataset download --dataset LitBank`. Flat NER uses the leftmost outer mention when labels nest.

## Score a local TACRED or Re-TACRED split

Copy an authorized TACRED split JSON to `$ANNO_CACHE_DIR/datasets/TACRED.cache`. For Re-TACRED, apply the authors’ patch to a licensed TACRED split and place the resulting split at `$ANNO_CACHE_DIR/datasets/ReTACRED.cache`. The public patch contains ID-to-label mappings and requires the original sentences. Both inputs stay local. [TACRED distribution](https://catalog.ldc.upenn.edu/LDC2018T24) · [Re-TACRED patch and release notes](https://github.com/gstoica27/Re-TACRED)

Run `anno dataset eval --task relation --dataset retacred --model heuristic` (or use `--dataset tacred`). Each record contributes one document. Positive labels create directed subject-to-object relations. A `no_relation` record has empty gold and can still incur false positives. The scorer evaluates extracted argument spans and relation labels. Published TACRED classification benchmarks supply the argument pair to the model.

## Check what a negative span means

Use `exact_span` when the source range must receive one of the listed types. Use `no_overlap` when the specified types must leave the entire range clear. Both use half-open Unicode scalar offsets. For example, an ORG spanning `Iona Tress (CPA` violates a protected range for `CPA` even if the extractor never emits the credential on its own. A prediction ending at the first character of the protected range may pass. Pair a negative range with a positive mention so an empty extractor cannot pass the whole case.

For an occurrence that must appear exactly once, add `"expected_count": 1` to its `exact_span` rule. The judge counts individual predictions at that exact range across all allowed types and reports `observed_count`. This detects duplicate emissions even when both have a valid type. Zero forbids the listed types at that exact range and permits partial overlaps. Omitting the field keeps the original presence check.

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

The native checkpoint replay runs BERT-large, ModernBERT, NuNER Zero-span INT8, and GLiNER2.5 through `anno extract` using hashed artifacts. Seven CPU cases check source offsets and labels. Four also compare GLiNER2.5 scores with upstream probabilities. The receipt records the actual binary and runtime hashes. [Reproduction instructions](../scripts/qa/checkpoint_replay/README.md)

The required Ubuntu CLI integration job runs against local fixtures, including public-source excerpts. The additional eval-feature job exercises the loopback HTTP cases. The Python QA job tests the harness itself with fixed inputs. A separate path-filtered benchmark fetches checksum-pinned WikiGold and records corpus scores. The full 457-date NASA replay is an explicit command. Fuzzing is available locally, and a maintainer can launch the bounded mutation job from a matching branch. [CI workflow](../.github/workflows/ci.yml) · [WikiGold benchmark](../.github/workflows/bench-extraction.yml) · [Fuzz target](../fuzz/README.md)

The Python binding and Linux wheel jobs run the bulletin example from a
temporary working directory. The weekly installed-package job exercises the
pinned public PyPI wheel and records the filename and hash. The release
workflow also verifies the submitted distribution hashes after publication.
The real-weight FCoref job checks that a reader can reopen a caption at its
original cue time after a Unicode edit. [Python wheel build](../.github/workflows/python-wheels.yml) · [Public wheel check](../.github/workflows/python-package-check.yml) · [Caption replay](../crates/anno/examples/caption_links/README.md)

Maintainers can run the weekly checks immediately from the default branch.
The journey report records the policies’ selections and the independent
result for each of the three cases. [Policy comparison and local replay](EVALUATION_SAMPLING.md#compare-three-developer-journeys)

```sh
gh workflow run python-package-check.yml --ref main
gh workflow run journey-sampling.yml --ref main
```

For live discovery, `just qa-random` balances catalog topics and publishers by default. For offline exact-span assertions, `just qa-domain-spans target/debug/anno` exercises a labeled panel. The live runner needs an eval-enabled CLI.

```sh
export ANNO=target/debug/anno
python3 scripts/qa_random_urls.py --count 10 --seed 42 --backends pattern --output qa/reports/field.md
just qa-round qa/reports/field.json 'Which selected topic still needs an independent label?'
python3 scripts/qa_random_urls.py --replay qa/reports/field.json --output qa/reports/revisit.md
```

The cross-domain receipt records the requested backend and the runtime model named by the executing Anno instance. A matching URL and text hash establish input agreement. Backend coverage also requires a compatible runtime name. Legacy receipts support input comparisons and report their backend identity as unverified.

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

### Replay a quoted role and two named people

This fixture uses a 2,003-scalar excerpt from document 208 in [LitBank](https://github.com/dbamman/litbank/tree/3e50db0ffc033d7ccbb94f4d88f6b99210328ed8), by David Bamman, Olivia Lewke and Anya Mansoor (CC BY 4.0). The [fixture](../scripts/qa/fixtures/litbank-role-link.json) pins the verbatim source and excerpt hashes. All saved offsets are half-open Unicode scalar indices into the original document. The runner receives only the unmarked excerpt and its ID. Use the same local FP32 export as the word-aligned example above.

```sh
git clone https://github.com/dbamman/litbank.git target/litbank
git -C target/litbank checkout --detach 3e50db0ffc033d7ccbb94f4d88f6b99210328ed8
cargo build --locked -p anno --no-default-features --features onnx --example fcoref_word_spans
python3 scripts/qa/replay_litbank_role.py target/litbank \
  --runner target/debug/examples/fcoref_word_spans --model-dir "$FCOREF_MODEL_DIR"
```

The oracle expects the pronoun, role and correct name to share a cluster, and the other named person to appear in a different cluster. The scorer reports all four selected pairs, with an unresolved judgment when either exact span lacks an emitted cluster. Stock FCoref on 942eb6a6 proposed all four endpoints but grouped the pronoun and role with the other named person. This single case has no frequency estimate.

Live runs verify the four pinned FP32 files before and after inference and include their SHA-256 digests in `prediction_source`. A co-located quantized encoder or a mismatched digest exits 2. Use `--predictions FILE` to score saved predictions from the example. That mode records the SHA-256 of the supplied receipt and leaves the model identity unverified. A match exits 0 and a mismatch exits 1. An unavailable runner or model, or a source-only check with neither execution mode, reports `not_run` and exits 77. A bad source, malformed receipt, or failed invocation exits 2. Synthetic same-person, wrong-person and missing-endpoint controls run with `python3 -m unittest discover -s scripts/qa -p test_replay_litbank_role.py`.

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

### Decide what to keep

Promote a discovery when a small input and an independent expected result reproduce the behavior. For a repair, run both revisions and include a neighboring case the change could break. Preserve the offset units. Once a patch author or agent has seen a holdout, count it as exposed in future comparisons. [Source and offset contract](CONTRACT.md)

| What the reviewer has established | What belongs in Git | Where the input can run |
| --- | --- | --- |
| The fixture or generator is project-authored and cleared for the project license | The small input or generator and seed, expected result, and provenance alongside the owning test | Contributor checkouts and offline PR CI |
| A third-party release permits redistribution of the particular text, labels and embedded assets | Only the cleared material, with the revision, license, attribution, notices and changes its terms require | Each destination and purpose covered by the terms |
| Testing is permitted but sharing source bytes is restricted | A public reference or permitted acquisition recipe with a version, plus non-revealing assertions if shareable | An authorized local cache, or a hosted runner if specifically covered |
| Rights or provenance are unknown | An independently authored example after review, if it reproduces the same failure | Obtain permission before using the external material |

Keep generated candidates in a private working directory. `fuzz/corpus/` and `.generated/` are Git-ignored. Review logs, prompts, Actions artifacts and S3 as separate destinations. Predictions and minimized cases can retain original text. A digest identifies bytes and short or sensitive values can be guessable. For shared receipts, select approved fields before writing output, including on error paths. Keep source-bearing traces with their inputs.

The [fuzz corpus](../fuzz/README.md) keeps generated inputs separate from reviewed regression seeds. Muxer tracks which cases and dataset cells to try next. Its trusted checkpoint can preserve scheduling history between runs. Fixed cases remain replayable with an empty history. Count a confirmed cause once when evaluating discovery, even if several variants reproduce it. [Checkpoint and source policy](S3_CACHE_CI.md#what-ci-shares) · [Reviewed replay adapters](EVALUATION_SAMPLING.md#replay-a-reviewed-corpus-witness-in-a-triage-slot)

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

## Score a SciERC split

Put an authorized `processed_data/json/{train,dev,test}.json` split from the authors’ processed archive at `$ANNO_CACHE_DIR/datasets/SciERC.cache`. Run `anno dataset eval --task relation --dataset SciERC --model heuristic`. The scorer evaluates each sentence separately, retains sentences with no gold relations, and resolves nested arguments using the release’s document-relative token indices. Flat NER keeps the leftmost non-overlapping mentions, preferring the longest at a shared start. The separate `SciERCNER.cache` input accepts an already normalized sentence-level NER projection. [SciERC release and annotation guidelines](https://nlp.cs.washington.edu/sciIE/)

## Score cross-document linkage on ECB+

Run the cross-document example with the pinned ECB+ release in the dataset cache, or let the loader fetch it. Pass the published sentence-index CSV to restrict annotations to the recommended sentences. `ANNO_CDCR_MAX_DOCS=12` selects a sorted prefix for a bounded diagnostic. Omit the variable to score the full panel. CI runs independently authored fixtures without downloading the release.

```sh
ANNO_CDCR_MAX_DOCS=12 just cdcr-eval --sentence-index /path/to/ECBplus_coreference_sentences.csv
cargo test --locked -p anno-eval --features eval --example cdcr_eval
```

For the cited release, the archive SHA-256 is `fe6ac2d76d8f7015f773ccc969927d7f3c33b331ed643b71efe31f28373eca90` and the sentence-index SHA-256 is `afee49e0f5fe5084eec690e934aaade842d5eeff7a5704a36f59e7a49505f434`. The report identifies the sentence index used and the number of panel documents scored.

The example measures merging of supplied mention spans in known numeric topics. It compares two inputs to the same heuristic character-trigram encoder and cosine scorer. One begins with singletons. The other supplies true within-document chains. Transitive closure can connect two seeds from the same document through a third seed elsewhere. The example reports pooled clustering metrics alongside correct, reference and predicted cross-document mention-pair counts. Pair precision divides correct by predicted links. Pair recall divides correct by reference links. Neither depends on the large number of unrelated pairs.

Across the complete archive there are 32,297 supplied mentions in 982 documents and 216,535 reference cross-document pairs. The recommended sentence selection gives 15,122 mentions in 976 documents and 91,185 such pairs. Seven raw instance labels recur in different topics. Group by the supplied topic and instance label together. A relation confined to one document or an unlinked singleton cannot supply a cross-document gold link.

The numbers from this command depend on the selected documents, supplied mention boundaries, local seeds and gold topic grouping. They do not measure detection, infer unknown topics, identify people in a catalog, or score the native production track resolver. A deployment threshold needs validation on held-out topics and on the downstream cost of false merges and splits.

Cybulska and Vossen (2014) describe ECB+ and its annotation process. The annotation license is CC BY 3.0. Underlying news text has separate provenance. The source ZIP and sentence list are obtained from the publisher. [Paper](https://aclanthology.org/L14-1646/) · [Pinned release, sentence index and license](https://github.com/cltl/ecbPlus/tree/3aa071e12fd08fc0705b16bb1841e7a9c5c22420/ECB%2B_LREC2014)
