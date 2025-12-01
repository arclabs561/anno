# Eval Report

Total: 208 | ✓: 31 | ⊘: 177 | ✗: 0 | Avg examples: 17 | Avg time: 616ms

## Results

**Note**: 177 combinations skipped (features not enabled or incompatible). Showing successful and failed results only.

### Abstract Anaphora Resolution

| Dataset | Backend | CoNLL | MUC | B³ | N | ms |
|---------|---------|-------|-----|----|---|----|
| GAP | coref_resolver | 35.0 | 0.3 | 75.2 | 20 | 63877 |
| PreCo | coref_resolver | 0.0 | 0.0 | 0.0 | 0 | 0 |
| LitBank | coref_resolver | 0.0 | 0.0 | 0.0 | 1 | 0 |

### Intra-document Coreference

| Dataset | Backend | CoNLL | MUC | B³ | N | ms |
|---------|---------|-------|-----|----|---|----|
| GAP | coref_resolver | 35.0 | 0.3 | 75.2 | 20 | 64085 |
| PreCo | coref_resolver | 0.0 | 0.0 | 0.0 | 0 | 0 |
| LitBank | coref_resolver | 0.0 | 0.0 | 0.0 | 1 | 0 |

### Named Entity Recognition

| Dataset | Backend | F1 | P | R | N | ms |
|---------|---------|----|----|----|---|----|
| WikiGold | stacked | 37.3 | 31.1 | 46.3 | 20 | 23 |
| MultiNERD | stacked | 36.4 | 35.7 | 37.0 | 20 | 1 |
| MultiCoNERv2 | stacked | 28.2 | 26.9 | 29.5 | 20 | 5 |
| WikiANN | stacked | 27.3 | 25.0 | 30.0 | 20 | 1 |
| CoNLL2003Sample | stacked | 23.9 | 23.4 | 24.4 | 20 | 4 |
| OntoNotesSample | stacked | 20.4 | 21.7 | 19.2 | 20 | 1 |
| TweetNER7 | stacked | 15.7 | 13.8 | 18.1 | 20 | 4 |
| PolyglotNER | stacked | 13.8 | 12.9 | 14.8 | 20 | 3 |
| BroadTwitterCorpus | stacked | 12.8 | 8.3 | 27.3 | 20 | 2 |
| Wnut17 | stacked | 12.2 | 7.5 | 33.3 | 20 | 1 |
| MultiCoNER | stacked | 9.5 | 13.0 | 7.5 | 20 | 2 |
| FewNERD | stacked | 4.9 | 7.0 | 3.8 | 20 | 3 |
| CrossNER | stacked | 1.1 | 1.3 | 0.9 | 20 | 3 |
| MitMovie | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| MitRestaurant | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| BC5CDR | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| NCBIDisease | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| GENIA | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| AnatEM | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| BC2GM | stacked | 0.0 | 0.0 | 0.0 | 20 | 3 |
| BC4CHEMD | stacked | 0.0 | 0.0 | 0.0 | 20 | 2 |
| FabNER | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| UniversalNERBench | stacked | 0.0 | 0.0 | 0.0 | 20 | 2 |
| WikiNeural | stacked | 0.0 | 0.0 | 0.0 | 20 | 3 |
| UniversalNER | stacked | 0.0 | 0.0 | 0.0 | 20 | 2 |

## Backend Summary

| Backend | ✓ | ⊘ | ✗ | Avg F1 |
|---------|---|---|---|--------|
| stacked | 25 | 0 | 0 | 9.7 |
| coref_resolver | 6 | 0 | 0 | 11.7 |
| candle_ner | 0 | 25 | 0 | 0.0 |
| gliner2 | 0 | 27 | 0 | 0.0 |
| gliner_onnx | 0 | 25 | 0 | 0.0 |
| w2ner | 0 | 25 | 0 | 0.0 |
| bert_onnx | 0 | 25 | 0 | 0.0 |
| gliner_candle | 0 | 25 | 0 | 0.0 |
| nuner | 0 | 25 | 0 | 0.0 |

