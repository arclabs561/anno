# Eval Report

Total: 258 | ✓: 75 | ⊘: 177 | ✗: 6 | Avg examples: 20 | Avg time: 0ms

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

### Relation Extraction

| Dataset | Backend | Strict | Boundary | N | ms |
|---------|---------|--------|----------|---|----|
| DocRED | gliner2 | ⊘ | no-feature | 0 |
| ReTACRED | gliner2 | ⊘ | no-feature | 0 |

### Named Entity Recognition

| Dataset | Backend | F1 | P | R | N | ms |
|---------|---------|----|----|----|---|----|
| WikiGold | heuristic | 38.0 | 32.2 | 46.3 | 20 | 0 |
| WikiGold | stacked | 37.3 | 31.1 | 46.3 | 20 | 1 |
| MultiNERD | heuristic | 36.4 | 35.7 | 37.0 | 20 | 0 |
| MultiNERD | stacked | 36.4 | 35.7 | 37.0 | 20 | 1 |
| MultiCoNERv2 | heuristic | 28.4 | 27.3 | 29.5 | 20 | 1 |
| MultiCoNERv2 | stacked | 28.2 | 26.9 | 29.5 | 20 | 1 |
| WikiANN | heuristic | 27.3 | 25.0 | 30.0 | 20 | 0 |
| WikiANN | stacked | 27.3 | 25.0 | 30.0 | 20 | 0 |
| CoNLL2003Sample | heuristic | 25.3 | 26.2 | 24.4 | 20 | 0 |
| CoNLL2003Sample | stacked | 23.9 | 23.4 | 24.4 | 20 | 1 |
| OntoNotesSample | stacked | 20.4 | 21.7 | 19.2 | 20 | 0 |
| TweetNER7 | heuristic | 18.4 | 16.5 | 20.8 | 20 | 0 |
| OntoNotesSample | heuristic | 17.8 | 21.1 | 15.4 | 20 | 0 |
| TweetNER7 | stacked | 15.7 | 13.8 | 18.1 | 20 | 1 |
| Wnut17 | heuristic | 15.2 | 9.8 | 33.3 | 20 | 0 |
| BroadTwitterCorpus | heuristic | 14.3 | 9.7 | 27.3 | 20 | 0 |
| PolyglotNER | heuristic | 13.8 | 12.9 | 14.8 | 20 | 0 |
| PolyglotNER | stacked | 13.8 | 12.9 | 14.8 | 20 | 0 |
| BroadTwitterCorpus | stacked | 12.8 | 8.3 | 27.3 | 20 | 0 |
| Wnut17 | stacked | 12.2 | 7.5 | 33.3 | 20 | 0 |
| MultiCoNER | heuristic | 9.5 | 13.0 | 7.5 | 20 | 0 |
| MultiCoNER | stacked | 9.5 | 13.0 | 7.5 | 20 | 0 |
| FewNERD | heuristic | 4.9 | 7.0 | 3.8 | 20 | 0 |
| FewNERD | stacked | 4.9 | 7.0 | 3.8 | 20 | 1 |
| CrossNER | heuristic | 1.1 | 1.3 | 0.9 | 20 | 0 |
| CrossNER | stacked | 1.1 | 1.3 | 0.9 | 20 | 1 |
| WikiGold | pattern | 0.0 | 0.0 | 0.0 | 20 | 23 |
| Wnut17 | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| MitMovie | pattern | 0.0 | 0.0 | 0.0 | 20 | 0 |
| MitMovie | heuristic | 0.0 | 0.0 | 0.0 | 20 | 0 |
| MitMovie | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| MitRestaurant | pattern | 0.0 | 0.0 | 0.0 | 20 | 0 |
| MitRestaurant | heuristic | 0.0 | 0.0 | 0.0 | 20 | 0 |
| MitRestaurant | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| CoNLL2003Sample | pattern | 0.0 | 0.0 | 0.0 | 20 | 0 |
| OntoNotesSample | pattern | 0.0 | 0.0 | 0.0 | 20 | 0 |
| MultiNERD | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| BC5CDR | pattern | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC5CDR | heuristic | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC5CDR | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| NCBIDisease | pattern | 0.0 | 0.0 | 0.0 | 20 | 0 |
| NCBIDisease | heuristic | 0.0 | 0.0 | 0.0 | 20 | 0 |
| NCBIDisease | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| GENIA | pattern | 0.0 | 0.0 | 0.0 | 20 | 0 |
| GENIA | heuristic | 0.0 | 0.0 | 0.0 | 20 | 0 |
| GENIA | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| AnatEM | pattern | 0.0 | 0.0 | 0.0 | 20 | 0 |
| AnatEM | heuristic | 0.0 | 0.0 | 0.0 | 20 | 0 |
| AnatEM | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC2GM | pattern | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC2GM | heuristic | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC2GM | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC4CHEMD | pattern | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC4CHEMD | heuristic | 0.0 | 0.0 | 0.0 | 20 | 0 |
| BC4CHEMD | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| TweetNER7 | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| BroadTwitterCorpus | pattern | 0.0 | 0.0 | 0.0 | 20 | 0 |
| FabNER | pattern | 0.0 | 0.0 | 0.0 | 20 | 0 |
| FabNER | heuristic | 0.0 | 0.0 | 0.0 | 20 | 0 |
| FabNER | stacked | 0.0 | 0.0 | 0.0 | 20 | 0 |
| FewNERD | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| CrossNER | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| UniversalNERBench | pattern | 0.0 | 0.0 | 0.0 | 20 | 0 |
| UniversalNERBench | heuristic | 0.0 | 0.0 | 0.0 | 20 | 0 |
| UniversalNERBench | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| WikiANN | pattern | 0.0 | 0.0 | 0.0 | 20 | 0 |
| MultiCoNER | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| MultiCoNERv2 | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| WikiNeural | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| WikiNeural | heuristic | 0.0 | 0.0 | 0.0 | 20 | 0 |
| WikiNeural | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
| PolyglotNER | pattern | 0.0 | 0.0 | 0.0 | 20 | 0 |
| UniversalNER | pattern | 0.0 | 0.0 | 0.0 | 20 | 1 |
| UniversalNER | heuristic | 0.0 | 0.0 | 0.0 | 20 | 0 |
| UniversalNER | stacked | 0.0 | 0.0 | 0.0 | 20 | 1 |
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

## Backend Summary

| Backend | ✓ | ⊘ | ✗ | Avg F1 |
|---------|---|---|---|--------|
| stacked | 25 | 0 | 0 | 9.7 |
| heuristic | 25 | 0 | 0 | 10.0 |
| pattern | 25 | 0 | 0 | 0.0 |
| gliner_candle | 0 | 25 | 0 | 0.0 |
| nuner | 0 | 25 | 0 | 0.0 |
| gliner2 | 0 | 27 | 0 | 0.0 |
| gliner_onnx | 0 | 25 | 0 | 0.0 |
| candle_ner | 0 | 25 | 0 | 0.0 |
| bert_onnx | 0 | 25 | 0 | 0.0 |
| w2ner | 0 | 25 | 0 | 0.0 |
| coref_resolver | 0 | 0 | 6 | 0.0 |

