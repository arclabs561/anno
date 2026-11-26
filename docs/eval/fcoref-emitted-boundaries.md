# Can a repeated name repair an emitted coreference boundary?

FCoref can link a pronoun to a span that contains the annotated name while missing the exact name boundary. We tested whether the cluster itself contains enough evidence to recover that boundary on another occurrence. The [earlier rank diagnostic](native-gap-loss.md#fcoref-scores-for-competing-mention-boundaries) motivated the question. It used different articles.

## Frozen rule and panel

The [replay](../../scripts/qa/fcoref_emitted_boundaries.py) first runs Anno's raw-text FCoref resolver. Within each returned cluster, it collects exact, case-sensitive surfaces consisting of at least two alphabetic words starting with uppercase letters. A longer mention is shortened only if exactly one proper substring matches one of those surfaces at word boundaries. The surface has to have been emitted elsewhere in that cluster. A candidate that collides with any assigned span or another proposed replacement is discarded. The replacement retains its original cluster and adds no aliases. The rule uses only source text and emitted spans.

We froze the rule, selection and paired scorer before scoring the panel. We used the original development TSV, ordered by the SHA-256 of the UTF-8 string `anno-fcoref-emitted-v1/` and the row ID concatenated. We selected 256 distinct URLs and texts, excluding the entire test and validation splits and the 128 development articles in the [previous budget trial](coref-panels.md). The replay checks all three input hashes and the uniqueness and separation of decoded, Unicode-normalized Wikipedia titles. Article separation applies to these Anno experiments. The model's training exposure is unknown.

The scorer checks both exact candidate–pronoun pairs per article and compares the same document before and after the boundary change. GAP labels only the offered pairs. For other output spans we can count replacements, while a judgment of their correctness would require additional annotation.

## Result and decision

The rule shortened 15 of 2,311 returned spans across 14 articles. The A-name coverage rose from 110 to 111. Exactly one more marked pair had both endpoints (228 to 229), and the number of gold pairs with both endpoints stayed at 162. The new name thus belonged to a negative pair whose pronoun was already present. The two remained in different clusters, as all 512 marked link decisions stayed the same. Both arms recovered 149 of 231 gold links, produced 12 false links and missed 82. Of those misses, **69 lacked at least one exact emitted endpoint and 13 had both endpoints in different clusters**. The replacement removed 15 spans and introduced 14 spans outside GAP’s three marked mentions per article. GAP cannot tell whether those 29 changes improved their boundaries. [Machine-readable aggregate](fcoref-emitted-boundaries.json)

Keep FCoref’s existing output policy. This candidate reused a single inference and did a text search over each cluster’s emitted names. In this English panel it paid that extra matching cost without recovering a gold link. The candidate requires a second, independently emitted matching surface. Other boundary mechanisms and languages need separate trials. The [follow-up stage trace](fcoref-endpoint-stages.md) found an exact legal proposal for both endpoints in all 69 cases and measured where each pair disappeared.

## Replay

Obtain the three GAP TSVs in external storage. The publisher repository carries Apache-2.0. The Wikipedia passages have their own attribution and reuse terms. The tracked receipt contains aggregates and SHA-256 commitments. The standard Rust runner (ONNX Runtime 1.24.4) and a separate diagnostic build in stock mode (1.24.2) agreed on all confusion counts, changes and replacement counts. We used the pinned FP32 encoder, scorer, tokenizer and configuration of the native CPU experiment. [GAP revision and credits](https://github.com/google-research-datasets/gap-coreference/tree/83135f26025cdd705c04f77e940d0c9201116b80) · [Rights and source notes](coref-corpus-sources.md) · [Model revision](https://huggingface.co/biu-nlp/f-coref/tree/e91dfff12879495d882ba9460d0c5d5dd44ade59)

```sh
cargo build --locked -p anno --features onnx --example coref_gap_panel
python3 scripts/qa/fcoref_emitted_boundaries.py \
  "$GAP/gap-development.tsv" "$GAP/gap-test.tsv" "$GAP/gap-validation.tsv" \
  "$FCOREF_MODEL_PATH" target/debug/examples/coref_gap_panel
python3 -m unittest discover -s scripts/qa -p 'test_fcoref_emitted_boundaries.py'
```
