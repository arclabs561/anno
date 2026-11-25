# Coreference on GAP and LitBank

Three resolvers see the same English passages in each comparison. The raw-text arms call Anno's `CorefBackend::resolve` and use only the clusters it returns. A model must initialize and identify itself as the requested producer. A failure cannot select a fallback. The supplied-mention arms receive the same exact character boundaries, without gold links. Maverick receives source text and the boundaries. Native mention-ranking's entity interface receives the surfaces in source order with a neutral entity type. That interface has no source-text parameter, so context-dependent features differ from its raw-text path. FCoref's public raw-text resolver detects its own mentions.

## GAP pronoun decisions

The published GAP test split contains Wikipedia excerpts annotated for two decisions per excerpt. We sorted IDs by the SHA-256 digest of `anno-gap-v1/` plus the ID and took the first 128. The panel contains 128 distinct passages from 127 article URLs. There are 64 A links, 53 B links and 11 cases in which neither name is the referent. The scorer checks every original character offset. It awards a positive decision when the exact pronoun and designated candidate occur in the same returned cluster. It ignores other links because GAP leaves them unlabeled. Output coverage counts target spans visible in any returned cluster, including singletons if the producer emits them. An absent span can reflect proposal, boundary or singleton policy. [GAP corpus and scoring](https://github.com/google-research-datasets/gap-coreference/tree/83135f26025cdd705c04f77e940d0c9201116b80)

| Resolver | Mentions | TP / FP / FN | Pair P / R / F1 | True pairs with both spans returned | Exact two decisions |
|---|---|---|---|---|---|
| Native mention-ranking | predicted | 6 / 4 / 111 | 0.600 / 0.051 / 0.094 | 8/117 | 16/128 |
| FCoref | predicted | 73 / 6 / 44 | 0.924 / 0.624 / 0.745 | 76/117 | 82/128 |
| Maverick | predicted | 76 / 1 / 41 | 0.987 / 0.650 / 0.784 | 77/117 | 87/128 |
| Native mention-ranking | supplied | 0 / 1 / 117 | 0.000 / 0.000 / 0.000 | 117/117 | 11/128 |
| Maverick | supplied | 113 / 13 / 4 | 0.897 / 0.966 / 0.930 | 113/117 | 114/128 |

On the same GAP excerpts, raw Maverick got both decisions right on 7 cases where FCoref missed at least one. FCoref got both right on 2 cases where Maverick missed. They both got the two decisions right on 80 cases. Coverage explains most of the missed links. Maverick linked 76 of the 77 true pairs whose exact endpoints appeared in its raw output. FCoref linked 73 of 76. Native raw mention-ranking returned both endpoints of only eight true pairs and linked six.

## LitBank cluster quality

The panel uses all ten works in the public fold-0 test partition. Each excerpt begins at the start of the annotated passage and ends at the last complete sentence within 180 source tokens. We chose that cap before model inference to keep the English documents inside the short-context comparison. Every token is matched against the original source in order. Only intervening whitespace may be skipped. The raw source prefix, including line breaks and punctuation, is the actual model input. No prose is redistributed in the receipt. [Pinned test split](https://huggingface.co/datasets/coref-data/litbank_indiscrim/tree/58b5fdd88d139b17ffd75365a5e1ec5b53b8dd43/split_0) · [Bamman, Lewke and Mansoor, LitBank annotations and CC BY 4.0 terms](https://github.com/dbamman/litbank/tree/3e50db0ffc033d7ccbb94f4d88f6b99210328ed8)

LitBank retains nested mentions and singletons for six ACE entity categories. The scorer retains annotated mentions in each excerpt, including mentions whose only coreferent is outside that prefix. Exact mention precision and recall compare all returned spans with all gold spans. A second measure counts spans in non-singleton clusters. MUC, B³ and entity-based CEAF use only clusters of at least two mentions on each side. Missing mentions split MUC chains. B³ sums squared intersection sizes over mentions. CEAF uses document-local maximum one-to-one assignment with the entity Dice similarity. The scorer sums numerators and denominators across documents, then averages the three F1 scores for CoNLL F1. It supplies every annotated span, including singletons, when running a supplied-mention arm. The metric numerators and denominators matched CoVal on 256 seeded partitions with independently omitted mentions and on the ten real native raw-text predictions. [Scoring implementation](../../scripts/qa/coref_litbank.py) · [CoVal reference](https://github.com/ns-moosavi/coval/blob/87071a6293dc2e786dcfe2ed78e9369c17e41b3b/coval/eval/evaluator.py)

| Resolver | Mentions | Works scored | Exact mention P / R | MUC F1 | B³ F1 | CEAFₑ F1 | CoNLL F1 |
|---|---|---:|---|---:|---:|---:|---:|
| Native mention-ranking | predicted | 10 | 0.602 / 0.348 | 0.376 | 0.278 | 0.291 | 0.315 |
| FCoref | predicted | 10 | 0.828 / 0.520 | 0.741 | 0.668 | 0.674 | 0.695 |
| Maverick | predicted | 10 | 0.801 / 0.593 | 0.777 | 0.704 | 0.681 | 0.721 |
| Native mention-ranking | supplied | 10 | 1.000 / 1.000 | 0.428 | 0.314 | 0.362 | 0.368 |
| Maverick | supplied | 9 | 1.000 / 0.970 | 0.800 | 0.749 | 0.791 | 0.780 |

The ten excerpts contain 1,566 source tokens, 66 sentences and 204 annotated mentions. Of these mentions, 145 belong to the 33 clusters with at least two members inside the excerpt. Leaving out one work at a time gives a raw Maverick-minus-FCoref CoNLL F1 difference from +0.012 to +0.051. This range measures sensitivity to one excerpt in this panel.

On *Dear Enemy*, three gold mentions split tokens under Maverick’s English tokenizer. The supplied-mode adapter rejected that document. On the same 9 accepted works, supplied native mention-ranking scored 0.397 and Maverick scored 0.780 CoNLL F1. With predicted mentions on those same works, native scored 0.320, FCoref 0.722 and Maverick 0.756. Native returns singleton IDs for every supplied mention. Maverick was configured to emit linked mentions only, which affects both its output mention recall and the interpretation of the supplied-mention comparison. These short excerpts leave long-distance and cross-document linking unmeasured.

## What a coreference result needs to preserve

On GAP, supplying the three exact spans raised Maverick's true links from 76 to 113 and its false links from one to 13. Native mention-ranking received all the same spans through its entity-only interface and linked none of the 117 true pairs. A replay therefore needs the source edition, whether the caller or resolver proposed the mentions, the actual resolver entry point, and the singleton policy. A returned cluster covers only the spans its producer emitted. A token-alignment refusal is a document-level outcome. Counting it as an empty successful prediction would change recall and conceal why a caller received no result.

## Replay

Use Python 3.12 with `pyarrow==20.0.0` and `scipy==1.16.3` to read the corpus and score it. Install the pinned Maverick Python environment and run its verification recipe. Set `MAVERICK_PYTHON` to that interpreter, `HF_HUB_CACHE` to its verified hub directory, and `MAVERICK_MODEL_PATH` to the verified MES OntoNotes checkpoint. Set `TRANSFORMERS_CACHE` to the same hub if the host normally overrides that cache. Export the pinned FCoref CPU FP32 model and set `FCOREF_MODEL_PATH` to its directory. Each scorer checks the weights before running a neural arm. The recorded run used ONNX Runtime 1.24.2, maverick-coref 1.0.7, PyTorch CPU 2.8.0 and spaCy 3.8.18. Each neural arm ran with single-CPU affinity. Other jobs ran on separate logical CPUs, so the receipt reports predictive quality and omits timing comparisons. [Maverick runtime and encoder verification](../../.github/workflows/maverick-coref.yml) · [FCoref artifact hashes and export](../../.github/workflows/fcoref-cli.yml)

```sh
mkdir -p /tmp/anno-coref-replay
out=/tmp/anno-coref-replay
curl -fLsS https://raw.githubusercontent.com/google-research-datasets/gap-coreference/83135f26025cdd705c04f77e940d0c9201116b80/gap-test.tsv -o "$out/gap-test.tsv"
curl -fLsS https://huggingface.co/datasets/coref-data/litbank_indiscrim/resolve/58b5fdd88d139b17ffd75365a5e1ec5b53b8dd43/split_0/test-00000-of-00001.parquet -o "$out/litbank-test.parquet"
cargo build --locked --release -p anno --no-default-features --features onnx --example coref_gap_panel
curl -fLsS https://raw.githubusercontent.com/ns-moosavi/coval/87071a6293dc2e786dcfe2ed78e9369c17e41b3b/coval/eval/evaluator.py -o "$out/coval.py"
COVAL_EVALUATOR="$out/coval.py" python -m unittest discover -s scripts/qa -p test_coref_panels.py
driver=target/release/examples/coref_gap_panel
for backend in mention-ranking fcoref maverick; do
    stem=$backend
    if [ "$backend" = mention-ranking ]; then stem=native; fi
    for mode in predicted supplied; do
        if [ "$backend/$mode" = fcoref/supplied ]; then continue; fi
        python scripts/qa/coref_gap.py "$out/gap-test.tsv" "$driver" "$backend" "$mode" "$out/$stem-$mode.json" || exit 1
        python scripts/qa/coref_litbank.py "$out/litbank-test.parquet" "$driver" "$backend" "$mode" "$out/litbank-$stem-$mode.json" || exit 1
    done
done
python scripts/qa/summarize_coref_panels.py "$out" "$out/coref-panels.json"
```

The receipt lists model identities, corpus digests, GAP IDs and decisions, and per-excerpt counts and source digests for LitBank. Each GAP observation contains five bits. The first three record output coverage of the pronoun, A and B. The last two are the predicted links to A and B. Gold pairs have two bits in A, B order. LitBank's three-component counts are exact matches, predictions and gold mentions. Its four-component counts are the precision numerator and denominator followed by the recall numerator and denominator. [Machine-readable receipt](coref-panels.json)
