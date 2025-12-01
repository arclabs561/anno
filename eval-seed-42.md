# Eval Report

Total: 208 | ✓: 31 | ⊘: 177 | ✗: 0 | Avg examples: 17 | Avg time: 656ms

## Results

**Note**: 177 combinations skipped (features not enabled or incompatible). Showing successful and failed results only.

### Abstract Anaphora Resolution

| Dataset | Backend | CoNLL | MUC | B³ | N | ms |
|---------|---------|-------|-----|----|---|----|
| GAP | coref_resolver | 35.0 | 0.3 | 75.2 | 20 | 69944 |
| PreCo | coref_resolver | 0.0 | 0.0 | 0.0 | 0 | 1 |
| LitBank | coref_resolver | 0.0 | 0.0 | 0.0 | 1 | 1 |

### Intra-document Coreference

| Dataset | Backend | CoNLL | MUC | B³ | N | ms |
|---------|---------|-------|-----|----|---|----|
| GAP | coref_resolver | 35.0 | 0.3 | 75.2 | 20 | 66455 |
| PreCo | coref_resolver | 0.0 | 0.0 | 0.0 | 0 | 0 |
| LitBank | coref_resolver | 0.0 | 0.0 | 0.0 | 1 | 0 |

### Named Entity Recognition

| Dataset | Backend | F1 | P | R | N | ms |
|---------|---------|----|----|----|---|----|
| WikiANN | stacked | 42.3 | 42.3 | 42.3 | 20 | 0 |
| MultiNERD | stacked | 38.1 | 37.5 | 38.7 | 20 | 1 |
| CoNLL2003Sample | stacked | 36.7 | 35.5 | 37.9 | 20 | 1 |
| WikiGold | stacked | 29.1 | 27.9 | 30.4 | 20 | 28 |
| OntoNotesSample | stacked | 26.0 | 25.6 | 26.3 | 20 | 1 |
| MultiCoNERv2 | stacked | 24.3 | 23.5 | 25.2 | 20 | 1 |
| PolyglotNER | stacked | 16.2 | 14.6 | 18.2 | 20 | 0 |
| TweetNER7 | stacked | 11.5 | 8.9 | 16.4 | 20 | 1 |
| BroadTwitterCorpus | stacked | 10.8 | 7.7 | 18.2 | 20 | 0 |
| Wnut17 | stacked | 9.0 | 5.0 | 42.9 | 20 | 2 |
| CrossNER | stacked | 7.4 | 8.0 | 6.9 | 20 | 1 |
| MultiCoNER | stacked | 5.7 | 7.4 | 4.6 | 20 | 0 |
| FewNERD | stacked | 4.2 | 5.9 | 3.2 | 20 | 1 |
| MitMovie | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| MitRestaurant | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| BC5CDR | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| NCBIDisease | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| GENIA | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| AnatEM | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC2GM | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC4CHEMD | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| FabNER | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| UniversalNERBench | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| WikiNeural | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| UniversalNER | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |

## Backend Summary

| Backend | ✓ | ⊘ | ✗ | Avg F1 |
|---------|---|---|---|--------|
| stacked | 25 | 0 | 0 | 10.4 |
| coref_resolver | 6 | 0 | 0 | 11.7 |
| gliner_onnx | 0 | 25 | 0 | 0.0 |
| gliner_candle | 0 | 25 | 0 | 0.0 |
| gliner2 | 0 | 27 | 0 | 0.0 |
| nuner | 0 | 25 | 0 | 0.0 |
| candle_ner | 0 | 25 | 0 | 0.0 |
| bert_onnx | 0 | 25 | 0 | 0.0 |
| w2ner | 0 | 25 | 0 | 0.0 |

