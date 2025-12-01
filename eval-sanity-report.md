# Eval Report

Total: 258 | ✓: 75 | ⊘: 177 | ✗: 6 | Avg examples: 20 | Avg time: 33ms

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

### Abstract Anaphora Resolution

| Dataset | Backend | CoNLL | MUC | B³ | N | ms |
|---------|---------|-------|-----|----|---|----|
| GAP | coref_resolver | ✗ | unknown-backend | 0 |
| PreCo | coref_resolver | ✗ | unknown-backend | 0 |
| LitBank | coref_resolver | ✗ | unknown-backend | 0 |

### Intra-document Coreference

| Dataset | Backend | CoNLL | MUC | B³ | N | ms |
|---------|---------|-------|-----|----|---|----|
| GAP | coref_resolver | ✗ | unknown-backend | 0 |
| PreCo | coref_resolver | ✗ | unknown-backend | 0 |
| LitBank | coref_resolver | ✗ | unknown-backend | 0 |

### Named Entity Recognition

| Dataset | Backend | F1 | P | R | N | ms |
|---------|---------|----|----|----|---|----|
| WikiANN | heuristic | 42.3 | 37.4 | 48.6 | 20 | 1 |
| WikiANN | stacked | 42.2 | 37.3 | 48.6 | 20 | 1 |
| MultiNERD | stacked | 39.5 | 39.8 | 39.2 | 20 | 1860 |
| MultiNERD | heuristic | 39.1 | 40.0 | 38.2 | 20 | 1655 |
| CoNLL2003Sample | heuristic | 36.3 | 36.5 | 36.2 | 20 | 1416 |
| CoNLL2003Sample | stacked | 35.9 | 34.7 | 37.3 | 20 | 1692 |
| OntoNotesSample | stacked | 34.6 | 33.6 | 35.7 | 20 | 106 |
| OntoNotesSample | heuristic | 34.2 | 34.7 | 33.6 | 20 | 74 |
| WikiGold | heuristic | 33.3 | 31.1 | 35.8 | 20 | 46 |
| WikiGold | stacked | 32.7 | 30.4 | 35.5 | 20 | 55 |
| MultiCoNERv2 | heuristic | 31.7 | 30.1 | 33.5 | 20 | 2 |
| MultiCoNERv2 | stacked | 31.6 | 29.9 | 33.5 | 20 | 4 |
| PolyglotNER | heuristic | 20.0 | 16.9 | 24.4 | 20 | 1 |
| PolyglotNER | stacked | 19.8 | 16.7 | 24.4 | 20 | 1 |
| Wnut17 | heuristic | 18.8 | 12.1 | 41.7 | 20 | 66 |
| BroadTwitterCorpus | heuristic | 18.7 | 12.9 | 34.2 | 20 | 7 |
| BroadTwitterCorpus | stacked | 16.3 | 10.7 | 34.2 | 20 | 13 |
| Wnut17 | stacked | 14.1 | 8.5 | 41.4 | 20 | 101 |
| TweetNER7 | heuristic | 13.9 | 11.7 | 17.3 | 20 | 19 |
| TweetNER7 | stacked | 12.5 | 10.4 | 15.7 | 20 | 34 |
| FewNERD | heuristic | 1.6 | 2.2 | 1.2 | 20 | 1 |
| FewNERD | stacked | 1.3 | 1.9 | 1.1 | 20 | 2 |
| MultiCoNER | heuristic | 1.0 | 1.5 | 0.8 | 20 | 1 |
| MultiCoNER | stacked | 1.0 | 1.5 | 0.8 | 20 | 2 |
| CrossNER | heuristic | 0.9 | 1.0 | 0.8 | 20 | 2 |
| CrossNER | stacked | 0.9 | 1.0 | 0.8 | 20 | 3 |
| MitRestaurant | pattern | 0.0 | 0.4 | 0.0 | 20 | 32 |
| MitRestaurant | stacked | 0.0 | 0.4 | 0.0 | 20 | 34 |
| WikiGold | pattern | 0.0 | 0.0 | 0.0 | 20 | 27 |
| Wnut17 | pattern | 0.0 | 0.0 | 0.0 | 20 | 42 |
| MitMovie | pattern | 0.0 | 0.0 | 0.0 | 20 | 41 |
| MitMovie | heuristic | 0.0 | 0.0 | 0.0 | 20 | 25 |
| MitMovie | stacked | 0.0 | 0.0 | 0.0 | 20 | 44 |
| MitRestaurant | heuristic | 0.0 | 0.0 | 0.0 | 20 | 19 |
| CoNLL2003Sample | pattern | 0.0 | 0.0 | 0.0 | 20 | 97 |
| OntoNotesSample | pattern | 0.0 | 0.0 | 0.0 | 20 | 18 |
| MultiNERD | pattern | 0.0 | 0.0 | 0.0 | 20 | 156 |
| BC5CDR | pattern | 0.0 | 0.0 | 0.0 | 20 | 30 |
| BC5CDR | heuristic | 0.0 | 0.0 | 0.0 | 20 | 43 |
| BC5CDR | stacked | 0.0 | 0.0 | 0.0 | 20 | 65 |
| NCBIDisease | pattern | 0.0 | 0.0 | 0.0 | 20 | 34 |
| NCBIDisease | heuristic | 0.0 | 0.0 | 0.0 | 20 | 138 |
| NCBIDisease | stacked | 0.0 | 0.0 | 0.0 | 20 | 170 |
| GENIA | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| GENIA | heuristic | 0.0 | 0.0 | 0.0 | 20 | 1 |
| GENIA | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| AnatEM | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| AnatEM | heuristic | 0.0 | 0.0 | 0.0 | 20 | 1 |
| AnatEM | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| BC2GM | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| BC2GM | heuristic | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC2GM | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| BC4CHEMD | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| BC4CHEMD | heuristic | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC4CHEMD | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| TweetNER7 | pattern | 0.0 | 0.0 | 0.0 | 20 | 17 |
| BroadTwitterCorpus | pattern | 0.0 | 0.0 | 0.0 | 20 | 7 |
| FabNER | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| FabNER | heuristic | 0.0 | 0.0 | 0.0 | 20 | 0 |
| FabNER | stacked | 0.0 | 0.0 | 0.0 | 20 | 2 |
| FewNERD | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| CrossNER | pattern | 0.0 | 0.0 | 0.0 | 20 | 2 |
| UniversalNERBench | pattern | 0.0 | 0.0 | 0.0 | 20 | 31 |
| UniversalNERBench | heuristic | 0.0 | 0.0 | 0.0 | 20 | 67 |
| UniversalNERBench | stacked | 0.0 | 0.0 | 0.0 | 20 | 73 |
| WikiANN | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| MultiCoNER | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| MultiCoNERv2 | pattern | 0.0 | 0.0 | 0.0 | 20 | 2 |
| WikiNeural | pattern | 0.0 | 0.0 | 0.0 | 20 | 2 |
| WikiNeural | heuristic | 0.0 | 0.0 | 0.0 | 20 | 2 |
| WikiNeural | stacked | 0.0 | 0.0 | 0.0 | 20 | 3 |
| PolyglotNER | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| UniversalNER | pattern | 0.0 | 0.0 | 0.0 | 20 | 2 |
| UniversalNER | heuristic | 0.0 | 0.0 | 0.0 | 20 | 2 |
| UniversalNER | stacked | 0.0 | 0.0 | 0.0 | 20 | 2 |
| WikiGold | bert_onnx | ⊘ | no-feature | 0 |
| WikiGold | candle_ner | ⊘ | no-feature | 0 |
| WikiGold | nuner | ⊘ | no-feature | 0 |
| WikiGold | gliner_onnx | ⊘ | no-feature | 0 |
| WikiGold | gliner_candle | ⊘ | no-feature | 0 |
| WikiGold | gliner2 | ⊘ | no-feature | 0 |
| WikiGold | w2ner | ⊘ | no-feature | 0 |
| Wnut17 | bert_onnx | ⊘ | no-feature | 0 |
| Wnut17 | candle_ner | ⊘ | no-feature | 0 |
| Wnut17 | nuner | ⊘ | no-feature | 0 |
| Wnut17 | gliner_onnx | ⊘ | no-feature | 0 |
| Wnut17 | gliner_candle | ⊘ | no-feature | 0 |
| Wnut17 | gliner2 | ⊘ | no-feature | 0 |
| Wnut17 | w2ner | ⊘ | no-feature | 0 |
| MitMovie | bert_onnx | ⊘ | no-feature | 0 |
| MitMovie | candle_ner | ⊘ | no-feature | 0 |
| MitMovie | nuner | ⊘ | no-feature | 0 |
| MitMovie | gliner_onnx | ⊘ | no-feature | 0 |
| MitMovie | gliner_candle | ⊘ | no-feature | 0 |
| MitMovie | gliner2 | ⊘ | no-feature | 0 |
| MitMovie | w2ner | ⊘ | no-feature | 0 |
| MitRestaurant | bert_onnx | ⊘ | no-feature | 0 |
| MitRestaurant | candle_ner | ⊘ | no-feature | 0 |
| MitRestaurant | nuner | ⊘ | no-feature | 0 |
| MitRestaurant | gliner_onnx | ⊘ | no-feature | 0 |
| MitRestaurant | gliner_candle | ⊘ | no-feature | 0 |
| MitRestaurant | gliner2 | ⊘ | no-feature | 0 |
| MitRestaurant | w2ner | ⊘ | no-feature | 0 |
| CoNLL2003Sample | bert_onnx | ⊘ | no-feature | 0 |
| CoNLL2003Sample | candle_ner | ⊘ | no-feature | 0 |
| CoNLL2003Sample | nuner | ⊘ | no-feature | 0 |
| CoNLL2003Sample | gliner_onnx | ⊘ | no-feature | 0 |
| CoNLL2003Sample | gliner_candle | ⊘ | no-feature | 0 |
| CoNLL2003Sample | gliner2 | ⊘ | no-feature | 0 |
| CoNLL2003Sample | w2ner | ⊘ | no-feature | 0 |
| OntoNotesSample | bert_onnx | ⊘ | no-feature | 0 |
| OntoNotesSample | candle_ner | ⊘ | no-feature | 0 |
| OntoNotesSample | nuner | ⊘ | no-feature | 0 |
| OntoNotesSample | gliner_onnx | ⊘ | no-feature | 0 |
| OntoNotesSample | gliner_candle | ⊘ | no-feature | 0 |
| OntoNotesSample | gliner2 | ⊘ | no-feature | 0 |
| OntoNotesSample | w2ner | ⊘ | no-feature | 0 |
| MultiNERD | bert_onnx | ⊘ | no-feature | 0 |
| MultiNERD | candle_ner | ⊘ | no-feature | 0 |
| MultiNERD | nuner | ⊘ | no-feature | 0 |
| MultiNERD | gliner_onnx | ⊘ | no-feature | 0 |
| MultiNERD | gliner_candle | ⊘ | no-feature | 0 |
| MultiNERD | gliner2 | ⊘ | no-feature | 0 |
| MultiNERD | w2ner | ⊘ | no-feature | 0 |
| BC5CDR | bert_onnx | ⊘ | no-feature | 0 |
| BC5CDR | candle_ner | ⊘ | no-feature | 0 |
| BC5CDR | nuner | ⊘ | no-feature | 0 |
| BC5CDR | gliner_onnx | ⊘ | no-feature | 0 |
| BC5CDR | gliner_candle | ⊘ | no-feature | 0 |
| BC5CDR | gliner2 | ⊘ | no-feature | 0 |
| BC5CDR | w2ner | ⊘ | no-feature | 0 |
| NCBIDisease | bert_onnx | ⊘ | no-feature | 0 |
| NCBIDisease | candle_ner | ⊘ | no-feature | 0 |
| NCBIDisease | nuner | ⊘ | no-feature | 0 |
| NCBIDisease | gliner_onnx | ⊘ | no-feature | 0 |
| NCBIDisease | gliner_candle | ⊘ | no-feature | 0 |
| NCBIDisease | gliner2 | ⊘ | no-feature | 0 |
| NCBIDisease | w2ner | ⊘ | no-feature | 0 |
| GENIA | bert_onnx | ⊘ | no-feature | 0 |
| GENIA | candle_ner | ⊘ | no-feature | 0 |
| GENIA | nuner | ⊘ | no-feature | 0 |
| GENIA | gliner_onnx | ⊘ | no-feature | 0 |
| GENIA | gliner_candle | ⊘ | no-feature | 0 |
| GENIA | gliner2 | ⊘ | no-feature | 0 |
| GENIA | w2ner | ⊘ | no-feature | 0 |
| AnatEM | bert_onnx | ⊘ | no-feature | 0 |
| AnatEM | candle_ner | ⊘ | no-feature | 0 |
| AnatEM | nuner | ⊘ | no-feature | 0 |
| AnatEM | gliner_onnx | ⊘ | no-feature | 0 |
| AnatEM | gliner_candle | ⊘ | no-feature | 0 |
| AnatEM | gliner2 | ⊘ | no-feature | 0 |
| AnatEM | w2ner | ⊘ | no-feature | 0 |
| BC2GM | bert_onnx | ⊘ | no-feature | 0 |
| BC2GM | candle_ner | ⊘ | no-feature | 0 |
| BC2GM | nuner | ⊘ | no-feature | 0 |
| BC2GM | gliner_onnx | ⊘ | no-feature | 0 |
| BC2GM | gliner_candle | ⊘ | no-feature | 0 |
| BC2GM | gliner2 | ⊘ | no-feature | 0 |
| BC2GM | w2ner | ⊘ | no-feature | 0 |
| BC4CHEMD | bert_onnx | ⊘ | no-feature | 0 |
| BC4CHEMD | candle_ner | ⊘ | no-feature | 0 |
| BC4CHEMD | nuner | ⊘ | no-feature | 0 |
| BC4CHEMD | gliner_onnx | ⊘ | no-feature | 0 |
| BC4CHEMD | gliner_candle | ⊘ | no-feature | 0 |
| BC4CHEMD | gliner2 | ⊘ | no-feature | 0 |
| BC4CHEMD | w2ner | ⊘ | no-feature | 0 |
| TweetNER7 | bert_onnx | ⊘ | no-feature | 0 |
| TweetNER7 | candle_ner | ⊘ | no-feature | 0 |
| TweetNER7 | nuner | ⊘ | no-feature | 0 |
| TweetNER7 | gliner_onnx | ⊘ | no-feature | 0 |
| TweetNER7 | gliner_candle | ⊘ | no-feature | 0 |
| TweetNER7 | gliner2 | ⊘ | no-feature | 0 |
| TweetNER7 | w2ner | ⊘ | no-feature | 0 |
| BroadTwitterCorpus | bert_onnx | ⊘ | no-feature | 0 |
| BroadTwitterCorpus | candle_ner | ⊘ | no-feature | 0 |
| BroadTwitterCorpus | nuner | ⊘ | no-feature | 0 |
| BroadTwitterCorpus | gliner_onnx | ⊘ | no-feature | 0 |
| BroadTwitterCorpus | gliner_candle | ⊘ | no-feature | 0 |
| BroadTwitterCorpus | gliner2 | ⊘ | no-feature | 0 |
| BroadTwitterCorpus | w2ner | ⊘ | no-feature | 0 |
| FabNER | bert_onnx | ⊘ | no-feature | 0 |
| FabNER | candle_ner | ⊘ | no-feature | 0 |
| FabNER | nuner | ⊘ | no-feature | 0 |
| FabNER | gliner_onnx | ⊘ | no-feature | 0 |
| FabNER | gliner_candle | ⊘ | no-feature | 0 |
| FabNER | gliner2 | ⊘ | no-feature | 0 |
| FabNER | w2ner | ⊘ | no-feature | 0 |
| FewNERD | bert_onnx | ⊘ | no-feature | 0 |
| FewNERD | candle_ner | ⊘ | no-feature | 0 |
| FewNERD | nuner | ⊘ | no-feature | 0 |
| FewNERD | gliner_onnx | ⊘ | no-feature | 0 |
| FewNERD | gliner_candle | ⊘ | no-feature | 0 |
| FewNERD | gliner2 | ⊘ | no-feature | 0 |
| FewNERD | w2ner | ⊘ | no-feature | 0 |
| CrossNER | bert_onnx | ⊘ | no-feature | 0 |
| CrossNER | candle_ner | ⊘ | no-feature | 0 |
| CrossNER | nuner | ⊘ | no-feature | 0 |
| CrossNER | gliner_onnx | ⊘ | no-feature | 0 |
| CrossNER | gliner_candle | ⊘ | no-feature | 0 |
| CrossNER | gliner2 | ⊘ | no-feature | 0 |
| CrossNER | w2ner | ⊘ | no-feature | 0 |
| UniversalNERBench | bert_onnx | ⊘ | no-feature | 0 |
| UniversalNERBench | candle_ner | ⊘ | no-feature | 0 |
| UniversalNERBench | nuner | ⊘ | no-feature | 0 |
| UniversalNERBench | gliner_onnx | ⊘ | no-feature | 0 |
| UniversalNERBench | gliner_candle | ⊘ | no-feature | 0 |
| UniversalNERBench | gliner2 | ⊘ | no-feature | 0 |
| UniversalNERBench | w2ner | ⊘ | no-feature | 0 |
| WikiANN | bert_onnx | ⊘ | no-feature | 0 |
| WikiANN | candle_ner | ⊘ | no-feature | 0 |
| WikiANN | nuner | ⊘ | no-feature | 0 |
| WikiANN | gliner_onnx | ⊘ | no-feature | 0 |
| WikiANN | gliner_candle | ⊘ | no-feature | 0 |
| WikiANN | gliner2 | ⊘ | no-feature | 0 |
| WikiANN | w2ner | ⊘ | no-feature | 0 |
| MultiCoNER | bert_onnx | ⊘ | no-feature | 0 |
| MultiCoNER | candle_ner | ⊘ | no-feature | 0 |
| MultiCoNER | nuner | ⊘ | no-feature | 0 |
| MultiCoNER | gliner_onnx | ⊘ | no-feature | 0 |
| MultiCoNER | gliner_candle | ⊘ | no-feature | 0 |
| MultiCoNER | gliner2 | ⊘ | no-feature | 0 |
| MultiCoNER | w2ner | ⊘ | no-feature | 0 |
| MultiCoNERv2 | bert_onnx | ⊘ | no-feature | 0 |
| MultiCoNERv2 | candle_ner | ⊘ | no-feature | 0 |
| MultiCoNERv2 | nuner | ⊘ | no-feature | 0 |
| MultiCoNERv2 | gliner_onnx | ⊘ | no-feature | 0 |
| MultiCoNERv2 | gliner_candle | ⊘ | no-feature | 0 |
| MultiCoNERv2 | gliner2 | ⊘ | no-feature | 0 |
| MultiCoNERv2 | w2ner | ⊘ | no-feature | 0 |
| WikiNeural | bert_onnx | ⊘ | no-feature | 0 |
| WikiNeural | candle_ner | ⊘ | no-feature | 0 |
| WikiNeural | nuner | ⊘ | no-feature | 0 |
| WikiNeural | gliner_onnx | ⊘ | no-feature | 0 |
| WikiNeural | gliner_candle | ⊘ | no-feature | 0 |
| WikiNeural | gliner2 | ⊘ | no-feature | 0 |
| WikiNeural | w2ner | ⊘ | no-feature | 0 |
| PolyglotNER | bert_onnx | ⊘ | no-feature | 0 |
| PolyglotNER | candle_ner | ⊘ | no-feature | 0 |
| PolyglotNER | nuner | ⊘ | no-feature | 0 |
| PolyglotNER | gliner_onnx | ⊘ | no-feature | 0 |
| PolyglotNER | gliner_candle | ⊘ | no-feature | 0 |
| PolyglotNER | gliner2 | ⊘ | no-feature | 0 |
| PolyglotNER | w2ner | ⊘ | no-feature | 0 |
| UniversalNER | bert_onnx | ⊘ | no-feature | 0 |
| UniversalNER | candle_ner | ⊘ | no-feature | 0 |
| UniversalNER | nuner | ⊘ | no-feature | 0 |
| UniversalNER | gliner_onnx | ⊘ | no-feature | 0 |
| UniversalNER | gliner_candle | ⊘ | no-feature | 0 |
| UniversalNER | gliner2 | ⊘ | no-feature | 0 |
| UniversalNER | w2ner | ⊘ | no-feature | 0 |

### Relation Extraction

| Dataset | Backend | Strict | Boundary | N | ms |
|---------|---------|--------|----------|---|----|
| DocRED | gliner2 | ⊘ | no-feature | 0 |
| ReTACRED | gliner2 | ⊘ | no-feature | 0 |

## Backend Summary

| Backend | ✓ | ⊘ | ✗ | Avg F1 |
|---------|---|---|---|--------|
| pattern | 25 | 0 | 0 | 0.0 |
| stacked | 25 | 0 | 0 | 11.3 |
| heuristic | 25 | 0 | 0 | 11.7 |
| gliner_candle | 0 | 25 | 0 | 0.0 |
| nuner | 0 | 25 | 0 | 0.0 |
| gliner2 | 0 | 27 | 0 | 0.0 |
| gliner_onnx | 0 | 25 | 0 | 0.0 |
| candle_ner | 0 | 25 | 0 | 0.0 |
| bert_onnx | 0 | 25 | 0 | 0.0 |
| w2ner | 0 | 25 | 0 | 0.0 |
| coref_resolver | 0 | 0 | 6 | 0.0 |

