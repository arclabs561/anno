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

### Abstract Anaphora Resolution

| Dataset | Backend | CoNLL | MUC | B³ | N | ms |
|---------|---------|-------|-----|----|---|----|
| GAP | coref_resolver | ✗ | unknown-backend | 0 |
| PreCo | coref_resolver | ✗ | unknown-backend | 0 |
| LitBank | coref_resolver | ✗ | unknown-backend | 0 |

### Named Entity Recognition

| Dataset | Backend | F1 | P | R | N | ms |
|---------|---------|----|----|----|---|----|
| WikiANN | stacked | 37.3 | 35.0 | 40.0 | 20 | 0 |
| WikiGold | stacked | 29.5 | 28.1 | 31.0 | 20 | 10 |
| MultiCoNERv2 | stacked | 28.0 | 25.9 | 30.4 | 20 | 1 |
| MultiNERD | stacked | 24.6 | 25.9 | 23.3 | 20 | 1 |
| CoNLL2003Sample | stacked | 22.5 | 23.1 | 22.0 | 20 | 0 |
| TweetNER7 | stacked | 15.0 | 12.6 | 18.6 | 20 | 1 |
| BroadTwitterCorpus | stacked | 10.3 | 6.5 | 25.0 | 20 | 0 |
| PolyglotNER | stacked | 9.8 | 9.1 | 10.7 | 20 | 0 |
| Wnut17 | stacked | 9.2 | 5.5 | 30.0 | 20 | 1 |
| OntoNotesSample | stacked | 7.7 | 8.0 | 7.4 | 20 | 0 |
| CrossNER | stacked | 6.9 | 7.9 | 6.1 | 20 | 1 |
| MultiCoNER | stacked | 3.4 | 4.8 | 2.7 | 20 | 0 |
| FewNERD | stacked | 2.1 | 2.7 | 1.7 | 20 | 1 |
| MitMovie | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| MitRestaurant | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC5CDR | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| NCBIDisease | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| GENIA | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| AnatEM | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC2GM | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC4CHEMD | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| FabNER | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| UniversalNERBench | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| WikiNeural | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| UniversalNER | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |

### Intra-document Coreference

| Dataset | Backend | CoNLL | MUC | B³ | N | ms |
|---------|---------|-------|-----|----|---|----|
| GAP | coref_resolver | ✗ | unknown-backend | 0 |
| PreCo | coref_resolver | ✗ | unknown-backend | 0 |
| LitBank | coref_resolver | ✗ | unknown-backend | 0 |

## Backend Summary

| Backend | ✓ | ⊘ | ✗ | Avg F1 |
|---------|---|---|---|--------|
| stacked | 25 | 0 | 0 | 8.3 |
| gliner2 | 0 | 27 | 0 | 0.0 |
| gliner_candle | 0 | 25 | 0 | 0.0 |
| gliner_onnx | 0 | 25 | 0 | 0.0 |
| bert_onnx | 0 | 25 | 0 | 0.0 |
| nuner | 0 | 25 | 0 | 0.0 |
| candle_ner | 0 | 25 | 0 | 0.0 |
| w2ner | 0 | 25 | 0 | 0.0 |
| coref_resolver | 0 | 0 | 6 | 0.0 |

