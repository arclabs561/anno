# Eval Report

Total: 208 | ✓: 25 | ⊘: 177 | ✗: 6 | Avg examples: 20 | Avg time: 0ms

## Failures

| Task | Dataset | Backend | Error |
|------|---------|---------|-------|
| Intra-document Coreference | GAP | coref_resolver | Invalid input: Unknown backend: 'coref_resolver'. Available: pattern, heuristic, stacked |
| Intra-document Coreference | PreCo | coref_resolver | Invalid input: Unknown backend: 'coref_resolver'. Available: pattern, heuristic, stacked |
| Intra-document Coreference | LitBank | coref_resolver | Invalid input: Unknown backend: 'coref_resolver'. Available: pattern, heuristic, stacked |
| Abstract Anaphora Resolution | GAP | coref_resolver | Invalid input: Unknown backend: 'coref_resolver'. Available: pattern, heuristic, stacked |
| Abstract Anaphora Resolution | PreCo | coref_resolver | Invalid input: Unknown backend: 'coref_resolver'. Available: pattern, heuristic, stacked |
| Abstract Anaphora Resolution | LitBank | coref_resolver | Invalid input: Unknown backend: 'coref_resolver'. Available: pattern, heuristic, stacked |

## Error Patterns

- [6x] Invalid input: Unknown backend: 'coref_resolver'. ...

## Results

**Note**: 177 combinations skipped (features not enabled or incompatible). Showing successful and failed results only.

### Named Entity Recognition

| Dataset | Backend | F1 | P | R | N | ms |
|---------|---------|----|----|----|---|----|
| WikiANN | stacked | 59.2 | 52.1 | 68.5 | 20 | 1 |
| PolyglotNER | stacked | 41.4 | 35.3 | 50.0 | 20 | 0 |
| OntoNotesSample | stacked | 32.4 | 30.6 | 34.4 | 20 | 0 |
| Wnut17 | stacked | 31.6 | 20.7 | 66.7 | 20 | 1 |
| MultiNERD | stacked | 31.6 | 34.6 | 29.0 | 20 | 0 |
| WikiGold | stacked | 29.3 | 28.1 | 30.5 | 20 | 32 |
| CoNLL2003Sample | stacked | 25.0 | 24.3 | 25.7 | 20 | 1 |
| MultiCoNERv2 | stacked | 23.7 | 23.0 | 24.4 | 20 | 1 |
| BroadTwitterCorpus | stacked | 10.5 | 6.5 | 28.6 | 20 | 0 |
| TweetNER7 | stacked | 8.7 | 7.3 | 10.8 | 20 | 1 |
| MultiCoNER | stacked | 4.1 | 5.8 | 3.2 | 20 | 1 |
| FewNERD | stacked | 2.8 | 3.9 | 2.2 | 20 | 1 |
| CrossNER | stacked | 1.9 | 2.2 | 1.7 | 20 | 1 |
| MitMovie | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| MitRestaurant | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC5CDR | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| NCBIDisease | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| GENIA | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| AnatEM | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC2GM | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC4CHEMD | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| FabNER | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| UniversalNERBench | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| WikiNeural | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| UniversalNER | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |

### Intra-document Coreference

| Dataset | Backend | CoNLL | MUC | B³ | N | ms |
|---------|---------|-------|-----|----|---|----|
| GAP | coref_resolver | ✗ | unknown-backend | 0 |
| PreCo | coref_resolver | ✗ | unknown-backend | 0 |
| LitBank | coref_resolver | ✗ | unknown-backend | 0 |

### Abstract Anaphora Resolution

| Dataset | Backend | CoNLL | MUC | B³ | N | ms |
|---------|---------|-------|-----|----|---|----|
| GAP | coref_resolver | ✗ | unknown-backend | 0 |
| PreCo | coref_resolver | ✗ | unknown-backend | 0 |
| LitBank | coref_resolver | ✗ | unknown-backend | 0 |

## Backend Summary

| Backend | ✓ | ⊘ | ✗ | Avg F1 |
|---------|---|---|---|--------|
| stacked | 25 | 0 | 0 | 12.1 |
| nuner | 0 | 25 | 0 | 0.0 |
| gliner_candle | 0 | 25 | 0 | 0.0 |
| coref_resolver | 0 | 0 | 6 | 0.0 |
| bert_onnx | 0 | 25 | 0 | 0.0 |
| w2ner | 0 | 25 | 0 | 0.0 |
| candle_ner | 0 | 25 | 0 | 0.0 |
| gliner_onnx | 0 | 25 | 0 | 0.0 |
| gliner2 | 0 | 27 | 0 | 0.0 |

