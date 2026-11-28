# Does the NuNER fallback transfer to encyclopedia text?

When BERT was unavailable, running NuNER on the 313 WikiGold sentences the default stack skipped raised exact entity matches from **231 to 435** and reduced unmatched predictions from **307 to 274**. With BERT loaded, the same extra calls gained four exact matches and added 21 net unmatched predictions. **Keep the length-and-casing gate when BERT succeeds.** Forcing NuNER raised the WikiGold warm median from 78 to 829 ms and lowered full-corpus F1. When BERT is absent, use the opt-in fallback for WikiGold-like encyclopedia extraction if recall justifies a 641 ms median on this host. A WNUT-like review queue must also accommodate more unmatched suggestions. [WikiGold counts and model identities](wikigold-nuner-routing.json) · [WNUT routing trial](nuner-fallback-routing.md)

Rust callers can select the measured fallback with `StackedNER::default().with_nuner_fallback()`. A separate native replay of the BERT-absent configuration on the same WikiGold release reproduced the exact gains and losses. [Caller behavior and independent replay](nuner-fallback-routing.md#does-the-fallback-help-on-encyclopedia-text)

On the **same 313 stock-skipped WikiGold sentences**, enabling NuNER added 313 model calls in either BERT condition. Suggestions count every prediction in the four mapped WikiGold classes, including correct matches. Warm latency measures a complete stack call on the study host. For an application that surfaces every native label, each arm emitted another 102 suggestions outside WikiGold’s inventory on these sentences. Total output rose from 640 to 811 without BERT and from 898 to 923 with BERT.

| BERT | Net exact | Net unmatched | Net suggestions to review | Warm median stack time, stock → extra |
|---|---:|---:|---:|---:|
| Absent | +204 | −33 | **+171** | 0.2 → 640.7 ms |
| Loaded | +4 | +21 | **+25** | 78.0 → 829.1 ms |

The paired view gives the measured calls, latency, exact gains and regressions for each WikiGold and WNUT condition. Each row uses the skipped sentences for that corpus and split. [Paired chart](../figures/wikigold-routing-paired.svg) · [Accessible tables](../figures/wikigold-routing-paired.html) · [Portable measurements](wikigold-routing-paired.json)

| Corpus and BERT state | Extra calls | Gold on these inputs | Exact gained / lost | Unmatched added / removed | F1 on full corpus or test split, stock → extra call |
|---|---:|---:|---:|---:|---:|
| WikiGold, absent | 313 | 731 | 220 / 16 | 115 / 148 | 58.1% → 62.5% |
| WikiGold, loaded | 313 | 731 | 4 / 0 | 23 / 2 | 73.7% → 73.5% |
| WNUT-17 test, absent | 295 | 264 | 82 / 2 | 186 / 46 | 35.2% → 38.0% |
| WNUT-17 test, loaded | 295 | 264 | 14 / 0 | 126 / 7 | 37.0% → 36.2% |

The F1 column uses all 1,696 WikiGold sentences or all 1,287 WNUT test sentences. An exact match requires both the Unicode character span and the mapped type. WikiGold scores PER, ORG, LOC and MISC. WNUT maps groups and corporations to ORG and includes products and creative works. Both retain all source gold types. The inference thresholds, FP32 checkpoints and stack order match between the studies. [Inputs to the native paired chart](wikigold-routing-paired.json)

For a WikiGold reviewer seeing every in-inventory prediction, the BERT-absent fallback produces **171 more suggestions** on the skipped inputs. Its 204 net new exact matches account for that increase even though the unmatched count falls by 33. Of the 115 newly unmatched outputs, 31 had the right span with a different type, 60 overlapped an annotated span with a different boundary, and 24 had no reference overlap. On WNUT, 132 of 186 newly unmatched outputs had no overlap. A useful next test for WikiGold is to rerank competing extents and types on those 91 overlapping predictions while counting the exact spans lost in the same sentences. [Paired additions and removals](wikigold-nuner-routing.json) · [WNUT error breakdown](nuner-fallback-routing.json)

## Which types benefit?

The BERT-absent fallback gained 82 exact PER, 76 LOC and 62 ORG mentions on WikiGold. It lost two PER, two LOC and twelve ORG. NuNER's three prompts offered no path to the 145 gold MISC mentions on those skipped inputs. All four exact gains with BERT present were ORG. [Counts by gold type](wikigold-nuner-routing.json) · [NuNER prompt and output mapping](../../crates/anno/src/backends/nuner/inference.rs)

![Across all 1,696 WikiGold sentences, the stock stack with BERT absent returned 2,034 exact matches. The entity error view distinguishes same-span type errors, partial matches, misses and extra predictions.](../figures/wikigold-entity-errors.svg)

The BERT-absent stock stack emitted zero MISC predictions across the **712 gold MISC mentions**. Anno's one-to-one entity matcher found **263 (36.9%)** at exactly the right boundaries with another label. It called 134 ORG, 90 PER and 39 LOC. Another 121 partially matched a predicted span, and 328 had no paired prediction. Those 263 mentions define the target for the fourth-prompt experiment below. [Native error counts and label pairs](wikigold-entity-errors.json) · [Error-view replay](../../scripts/qa/wikigold_entity_errors.py)

The error chart counts all 3,718 stock predictions, including 280 outside the routing trial's four-label inventory. Its one-to-one matcher distinguishes 536 same-span type errors, 347 partial overlaps and 801 unpaired extra predictions. **210 of the 347 partial pairs also disagreed on type**. The remaining 137 retained their gold label. The routing table counts each in-inventory prediction that lacks an exact span-and-type match as unmatched. [Complete error tables](../figures/wikigold-entity-errors.html) · [Native per-type outcomes and joint boundary and type errors](wikigold-entity-errors.json)

With BERT loaded, its adapter emits the checkpoint's MISC class as the custom Anno label `misc`. The WikiGold scorer maps this spelling to the publisher's `MISC`. Both policies emitted 857 such spans and matched 497 exactly over the full corpus. This mapping lets a caller distinguish support for the class in BERT from the missing prompt in NuNER. [BERT label mapping](../../crates/anno/src/backends/onnx.rs) · [Alias and exact-match counts](wikigold-nuner-routing.json)

## Does a fourth prompt recover MISC?

**Keep the three stock prompts.** Adding the literal label `miscellaneous` corrected **2 of the 263** WikiGold MISC mentions for which stock already found the exact boundaries with another type. The candidate still gave 247 the wrong type at those boundaries and no longer proposed the exact boundaries for 14. Across the whole release, the extra prompt identified 20 of 712 MISC mentions, including 18 with no overlap from any stock prediction. Its 34 MISC suggestions included 14 without an exact MISC match. [Paired outcomes](nuner-misc-prompt.json) · [Native comparison](../figures/nuner-misc-prompt-paired.svg)

| WikiGold gold type | Gold mentions | Stock exact | Four prompts exact | Newly exact | Previously exact lost |
|---|---:|---:|---:|---:|---:|
| MISC | 712 | 0 | 20 | 20 | 0 |
| ORG | 898 | 576 | 564 | 2 | 14 |
| LOC | 1,014 | 727 | 726 | 14 | 15 |
| PER | 934 | 731 | 730 | 1 | 2 |
| All four | 3,558 | 2,034 | 2,040 | 37 | 31 |

The net gain of six exact mentions hides 31 displaced correct predictions. Both arms returned 3,718 native suggestions, including 3,438 in the WikiGold inventory. Mapped unmatched fell from 1,404 to 1,398. A more targeted next experiment would keep a proposed span fixed and rerank its type against WikiGold’s MISC definition. Freeze that rule on separate development data, then count corrected and displaced spans on held-out documents. [Label confusions that motivated the experiment](wikigold-entity-errors.json)

The all-type gain of six is small relative to variation across WikiGold documents. The 95% paired bootstrap interval was **−13.2 to +26.0 exact mentions** at the original release volume. The MISC-only interval was +10.1 to +32.1, and the other three types together ranged from −30.4 to +2.2. The scorer read the pinned CoNLL bytes, started a new document at each standalone `-DOCSTART- O` boundary and assigned each subsequent sentence its document ordinal while parsing its gold spans. The 145 resulting nonempty groups include every occurrence, including 355 sentences with zero gold and all nine repeated texts beyond their first occurrence. Each of 4,000 paired draws sampled 145 entire document groups with replacement and scaled the net change to 1,696 sentences because document lengths differ. The receipt stores the group sizes, paired changes and ordered group-to-input hash. [Resampling and counts](nuner-misc-prompt.json)

On the full WNUT-17 test set, the same prompt changed mapped exact matches from **476 to 474** (five gained, seven lost, including five previously correct PER). The 95% paired sentence interval was −8 to +3. Mapped suggestions fell from 1,627 to 1,591, with unmatched dropping from 1,151 to 1,117. Counting all native labels, suggestions fell from 2,128 to 2,102. The candidate emitted ten MISC predictions outside WNUT’s reference inventory. One had the exact span of a different annotated type and nine had no reference overlap. These ten contribute to the all-output workload, and their MISC correctness is unmeasured on WNUT. [Full paired counts](nuner-misc-prompt.json) · [Accessible comparison](../figures/nuner-misc-prompt-paired.html)

Both arms used stock sentence routing, BERT absent, identical FP32 NuNER weights, a 0.90 NuNER threshold and a 0.50 final confidence filter. The candidate added one prompt after person, organization and location and mapped the resulting native label to MISC for scoring. Each arm ran NuNER on the same 1,383 WikiGold sentences and returned identical outputs on the 313 skipped ones. On WNUT the same arms made 992 NuNER calls each across all 1,287 official test sentences. Its resampling unit was a sentence occurrence, retaining all 598 with zero gold mentions and the seven occurrences beyond the 1,280 distinct texts. The WNUT file supplies sentence boundaries for this comparison. The WikiGold investigation was retrospective on an unsplit release. Both corpora tested one literal prompt. The concurrent Linux CPU workers used FP32 ONNX Runtime 1.24.2 in a Rust development build, so the prompt comparison reports calls as its inference-cost measure. [Pinned model, datasets, worker and output digests](nuner-misc-prompt.json) · [Prompt patch](../../scripts/qa/fixtures/nuner-wikigold-misc-prompt.patch)

## What does an extra call cost?

| BERT | Gained exact without new unmatched | Gained exact and added unmatched | Added unmatched without gain | Neither | Warm median, ms | Warm p95, ms |
|---|---:|---:|---:|---:|---:|---:|
| Absent | 115 | 33 | 57 | 108 | 0.2 → 640.7 | 1.1 → 1,088.6 |
| Loaded | 4 | 0 | 22 | 287 | 78.0 → 829.1 | 145.0 → 1,416.7 |

The first four columns count sentences. A gain means at least one newly exact mention. A sentence can also lose an existing exact match. Times show stock → extra NuNER inference over the same 313 previously skipped sentences, in release order, after warm-up. They measure the final stack call and exclude model loading, input parsing and scoring. The development-build workers used FP32 ONNX Runtime 1.24.2 on a shared Linux host with a four-CPU quota. The process that assembled the receipt used logical CPUs 0 and 1. The separate timing workers used CPUs 3 and 4. WNUT timings came from a different run. [Timing counts and distributions](wikigold-nuner-routing.json)

## What reached the evaluator

The complete, unsplit English WikiGold release contains 145 document groups, 1,696 sentence occurrences, 39,007 tokens and 3,558 gold mentions. The replay retained all 355 sentences with zero gold mentions and all nine repeated sentence occurrences (1,687 distinct strings). It excluded 145 standalone document delimiters. Offsets index Unicode characters after joining tokens with one space. An initial or type-changing I tag begins a mention. Consecutive I tags of the same type extend it. WikiGold supplies no training split, so this comparison pools familiar and unfamiliar names. [Pinned publisher file](https://github.com/juand-r/entity-recognition-datasets/blob/8522f4b39cf9ed323b77f2d8a5005628f8861908/data/wikigold/CONLL-format/data/wikigold.conll.txt) · [Span reader](../../scripts/qa/replay_wikigold.py)

The study used local publisher bytes. Their SHA-256 matches the WikiGold S3 payload. The scorer hashes each reconstructed sentence, surface, type and offset in native evaluator order. The result matches the Rust loader's authenticated cold-S3 receipt. CI independently reproduced the file and span hashes on November 27. [Matched S3 and publisher files](s3-publisher-file-matches.csv) · [Cold-cache verification](https://github.com/arclabs561/anno/actions/runs/46032959618/artifacts/15214793714) · [Native probe](../../crates/anno-eval/examples/wikigold_cache_probe.rs)

The stock stack processed all 1,696 inputs with BERT absent and again with BERT required to succeed. Its length and casing rule skipped NuNER on 313 and ran it on 1,383. Each candidate ran NuNER on all 313 skipped inputs. Eight controls per condition verified that the already-routed path reproduced the stock spans. Candidate totals use the measured stock outputs for the other 1,375 already-routed sentences. Sixteen additional controls checked that the guarded fallback preserves both stock routes when BERT succeeds. The timing runs reproduced the same routing decisions and ordered spans for all 313 inputs under every policy. [Native worker](../../crates/anno/examples/stacked_nuner_routing.rs) · [Scorer and recorded input checks](../../scripts/qa/wikigold_nuner_routing.py)

## Reproduce

Use the three native workers and model files from the WNUT trial. Set `trial`, `NUNER` and `BERT` as in its build instructions. The recorded executables match the WNUT workers byte for byte. Their backend baseline was `75f961e`. The replay verifies the source, checkpoint files and reconstructed gold spans. [Build the workers](nuner-fallback-routing.md#reproduce-the-comparison) · [Prepare the pinned WikiGold input](../../scripts/qa/prepare_matrix_wikigold.sh)

```sh
export ANNO_CACHE_DIR="${ANNO_CACHE_DIR:-$trial/cache}"
bash scripts/qa/prepare_matrix_wikigold.sh
python3 scripts/qa/wikigold_nuner_routing.py replay \
  --stock "$trial/stock" --candidate "$trial/guarded" --always "$trial/always" \
  --checkpoint "$NUNER" --bert "$BERT" --data "$ANNO_CACHE_DIR/datasets/WikiGold.cache" \
  --output "$trial/wikigold.json" --paired "$trial/chart.json"
```

The committed measurements reproduce the routing, error and fourth-prompt charts offline.

```sh
python3 scripts/qa/wikigold_nuner_routing.py chart \
  docs/eval/wikigold-nuner-routing.json "$trial/recorded-chart.json" --view paired
anno viz paired "$trial/recorded-chart.json" -o "$trial/recorded-chart.svg"
anno viz errors docs/eval/wikigold-entity-errors.json -o "$trial/errors.svg"
python3 scripts/qa/nuner_misc_prompt.py chart \
  docs/eval/nuner-misc-prompt.json "$trial/misc-chart.json"
anno viz paired "$trial/misc-chart.json" -o "$trial/misc.svg"
```

To recompute the error counts from the full stock output, use the same pinned files and the native evaluator.

```sh
python3 scripts/qa/wikigold_entity_errors.py --anno anno \
  --data "$ANNO_CACHE_DIR/datasets/WikiGold.cache" --stock-worker "$trial/stock" \
  --checkpoint "$NUNER" --output-dir "$trial/errors"
```

To rerun the fourth-prompt experiment, build two copies of the stock example at its recorded revision. Set `WNUT` to the directory with the pinned WNUT training and annotated test files and `NUNER` to the FP32 checkpoint directory. Training annotations are read to verify the source manifest. Inference uses only WNUT test and the full WikiGold release. [Prompt patch](../../scripts/qa/fixtures/nuner-wikigold-misc-prompt.patch) · [Scorer](../../scripts/qa/nuner_misc_prompt.py)

```sh
set -eu
patch_file="$PWD/scripts/qa/fixtures/nuner-wikigold-misc-prompt.patch"
git worktree add --detach "$trial/prompt-sources" 0874e06
(
  cd "$trial/prompt-sources"
  worker="${CARGO_TARGET_DIR:-target}/debug/examples/stacked_nuner_routing"
  cargo build --locked -p anno --no-default-features --features onnx --example stacked_nuner_routing
  cp "$worker" "$trial/stock-prompts"
  git apply "$patch_file"
  cargo build --locked -p anno --no-default-features --features onnx --example stacked_nuner_routing
  cp "$worker" "$trial/four-prompts"
  git apply -R "$patch_file"
)
git worktree remove "$trial/prompt-sources"
python3 scripts/qa/nuner_misc_prompt.py replay \
  --wikigold "$ANNO_CACHE_DIR/datasets/WikiGold.cache" --wnut "$WNUT" \
  --stock "$trial/stock-prompts" --candidate "$trial/four-prompts" --checkpoint "$NUNER" \
  --output "$trial/nuner-misc-prompt.json" --paired "$trial/nuner-misc-prompt-paired.json"
```
