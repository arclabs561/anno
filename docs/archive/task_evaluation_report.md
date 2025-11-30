# Comprehensive Task-Dataset-Backend Evaluation

**Total Combinations**: 280
**Successful**: 100
**Failed**: 180

## Tasks Evaluated

- Named Entity Recognition
- Relation Extraction
- Intra-document Coreference

## Datasets Used

- WikiGold
- Wnut17
- MitMovie
- MitRestaurant
- CoNLL2003Sample
- OntoNotesSample
- MultiNERD
- BC5CDR
- NCBIDisease
- GENIA
- AnatEM
- BC2GM
- BC4CHEMD
- TweetNER7
- BroadTwitterCorpus
- FabNER
- FewNERD
- CrossNER
- UniversalNERBench
- WikiANN
- MultiCoNER
- MultiCoNERv2
- WikiNeural
- PolyglotNER
- UniversalNER
- DocRED
- ReTACRED
- GAP
- PreCo
- LitBank

## Backends Tested

- pattern
- heuristic
- stacked
- hybrid
- bert_onnx
- candle_ner
- nuner
- gliner_onnx
- gliner_candle
- gliner2
- w2ner
- coref_resolver

## Results by Task

### Named Entity Recognition

| Dataset | Backend | Success | Examples |
|---------|---------|---------|----------|
| WikiGold | pattern | ✓ | 100 |
| WikiGold | heuristic | ✓ | 100 |
| WikiGold | stacked | ✓ | 100 |
| WikiGold | hybrid | ✓ | 100 |
| WikiGold | bert_onnx | ✗ | 100 |
| WikiGold | candle_ner | ✗ | 100 |
| WikiGold | nuner | ✗ | 100 |
| WikiGold | gliner_onnx | ✗ | 100 |
| WikiGold | gliner_candle | ✗ | 100 |
| WikiGold | gliner2 | ✗ | 100 |
| WikiGold | w2ner | ✗ | 100 |
| Wnut17 | pattern | ✓ | 100 |
| Wnut17 | heuristic | ✓ | 100 |
| Wnut17 | stacked | ✓ | 100 |
| Wnut17 | hybrid | ✓ | 100 |
| Wnut17 | bert_onnx | ✗ | 100 |
| Wnut17 | candle_ner | ✗ | 100 |
| Wnut17 | nuner | ✗ | 100 |
| Wnut17 | gliner_onnx | ✗ | 100 |
| Wnut17 | gliner_candle | ✗ | 100 |
| Wnut17 | gliner2 | ✗ | 100 |
| Wnut17 | w2ner | ✗ | 100 |
| MitMovie | pattern | ✓ | 100 |
| MitMovie | heuristic | ✓ | 100 |
| MitMovie | stacked | ✓ | 100 |
| MitMovie | hybrid | ✓ | 100 |
| MitMovie | bert_onnx | ✗ | 100 |
| MitMovie | candle_ner | ✗ | 100 |
| MitMovie | nuner | ✗ | 100 |
| MitMovie | gliner_onnx | ✗ | 100 |
| MitMovie | gliner_candle | ✗ | 100 |
| MitMovie | gliner2 | ✗ | 100 |
| MitMovie | w2ner | ✗ | 100 |
| MitRestaurant | pattern | ✓ | 100 |
| MitRestaurant | heuristic | ✓ | 100 |
| MitRestaurant | stacked | ✓ | 100 |
| MitRestaurant | hybrid | ✓ | 100 |
| MitRestaurant | bert_onnx | ✗ | 100 |
| MitRestaurant | candle_ner | ✗ | 100 |
| MitRestaurant | nuner | ✗ | 100 |
| MitRestaurant | gliner_onnx | ✗ | 100 |
| MitRestaurant | gliner_candle | ✗ | 100 |
| MitRestaurant | gliner2 | ✗ | 100 |
| MitRestaurant | w2ner | ✗ | 100 |
| CoNLL2003Sample | pattern | ✓ | 100 |
| CoNLL2003Sample | heuristic | ✓ | 100 |
| CoNLL2003Sample | stacked | ✓ | 100 |
| CoNLL2003Sample | hybrid | ✓ | 100 |
| CoNLL2003Sample | bert_onnx | ✗ | 100 |
| CoNLL2003Sample | candle_ner | ✗ | 100 |
| CoNLL2003Sample | nuner | ✗ | 100 |
| CoNLL2003Sample | gliner_onnx | ✗ | 100 |
| CoNLL2003Sample | gliner_candle | ✗ | 100 |
| CoNLL2003Sample | gliner2 | ✗ | 100 |
| CoNLL2003Sample | w2ner | ✗ | 100 |
| OntoNotesSample | pattern | ✓ | 100 |
| OntoNotesSample | heuristic | ✓ | 100 |
| OntoNotesSample | stacked | ✓ | 100 |
| OntoNotesSample | hybrid | ✓ | 100 |
| OntoNotesSample | bert_onnx | ✗ | 100 |
| OntoNotesSample | candle_ner | ✗ | 100 |
| OntoNotesSample | nuner | ✗ | 100 |
| OntoNotesSample | gliner_onnx | ✗ | 100 |
| OntoNotesSample | gliner_candle | ✗ | 100 |
| OntoNotesSample | gliner2 | ✗ | 100 |
| OntoNotesSample | w2ner | ✗ | 100 |
| MultiNERD | pattern | ✓ | 100 |
| MultiNERD | heuristic | ✓ | 100 |
| MultiNERD | stacked | ✓ | 100 |
| MultiNERD | hybrid | ✓ | 100 |
| MultiNERD | bert_onnx | ✗ | 100 |
| MultiNERD | candle_ner | ✗ | 100 |
| MultiNERD | nuner | ✗ | 100 |
| MultiNERD | gliner_onnx | ✗ | 100 |
| MultiNERD | gliner_candle | ✗ | 100 |
| MultiNERD | gliner2 | ✗ | 100 |
| MultiNERD | w2ner | ✗ | 100 |
| BC5CDR | pattern | ✓ | 100 |
| BC5CDR | heuristic | ✓ | 100 |
| BC5CDR | stacked | ✓ | 100 |
| BC5CDR | hybrid | ✓ | 100 |
| BC5CDR | bert_onnx | ✗ | 100 |
| BC5CDR | candle_ner | ✗ | 100 |
| BC5CDR | nuner | ✗ | 100 |
| BC5CDR | gliner_onnx | ✗ | 100 |
| BC5CDR | gliner_candle | ✗ | 100 |
| BC5CDR | gliner2 | ✗ | 100 |
| BC5CDR | w2ner | ✗ | 100 |
| NCBIDisease | pattern | ✓ | 100 |
| NCBIDisease | heuristic | ✓ | 100 |
| NCBIDisease | stacked | ✓ | 100 |
| NCBIDisease | hybrid | ✓ | 100 |
| NCBIDisease | bert_onnx | ✗ | 100 |
| NCBIDisease | candle_ner | ✗ | 100 |
| NCBIDisease | nuner | ✗ | 100 |
| NCBIDisease | gliner_onnx | ✗ | 100 |
| NCBIDisease | gliner_candle | ✗ | 100 |
| NCBIDisease | gliner2 | ✗ | 100 |
| NCBIDisease | w2ner | ✗ | 100 |
| GENIA | pattern | ✓ | 100 |
| GENIA | heuristic | ✓ | 100 |
| GENIA | stacked | ✓ | 100 |
| GENIA | hybrid | ✓ | 100 |
| GENIA | bert_onnx | ✗ | 100 |
| GENIA | candle_ner | ✗ | 100 |
| GENIA | nuner | ✗ | 100 |
| GENIA | gliner_onnx | ✗ | 100 |
| GENIA | gliner_candle | ✗ | 100 |
| GENIA | gliner2 | ✗ | 100 |
| GENIA | w2ner | ✗ | 100 |
| AnatEM | pattern | ✓ | 100 |
| AnatEM | heuristic | ✓ | 100 |
| AnatEM | stacked | ✓ | 100 |
| AnatEM | hybrid | ✓ | 100 |
| AnatEM | bert_onnx | ✗ | 100 |
| AnatEM | candle_ner | ✗ | 100 |
| AnatEM | nuner | ✗ | 100 |
| AnatEM | gliner_onnx | ✗ | 100 |
| AnatEM | gliner_candle | ✗ | 100 |
| AnatEM | gliner2 | ✗ | 100 |
| AnatEM | w2ner | ✗ | 100 |
| BC2GM | pattern | ✓ | 100 |
| BC2GM | heuristic | ✓ | 100 |
| BC2GM | stacked | ✓ | 100 |
| BC2GM | hybrid | ✓ | 100 |
| BC2GM | bert_onnx | ✗ | 100 |
| BC2GM | candle_ner | ✗ | 100 |
| BC2GM | nuner | ✗ | 100 |
| BC2GM | gliner_onnx | ✗ | 100 |
| BC2GM | gliner_candle | ✗ | 100 |
| BC2GM | gliner2 | ✗ | 100 |
| BC2GM | w2ner | ✗ | 100 |
| BC4CHEMD | pattern | ✓ | 100 |
| BC4CHEMD | heuristic | ✓ | 100 |
| BC4CHEMD | stacked | ✓ | 100 |
| BC4CHEMD | hybrid | ✓ | 100 |
| BC4CHEMD | bert_onnx | ✗ | 100 |
| BC4CHEMD | candle_ner | ✗ | 100 |
| BC4CHEMD | nuner | ✗ | 100 |
| BC4CHEMD | gliner_onnx | ✗ | 100 |
| BC4CHEMD | gliner_candle | ✗ | 100 |
| BC4CHEMD | gliner2 | ✗ | 100 |
| BC4CHEMD | w2ner | ✗ | 100 |
| TweetNER7 | pattern | ✓ | 100 |
| TweetNER7 | heuristic | ✓ | 100 |
| TweetNER7 | stacked | ✓ | 100 |
| TweetNER7 | hybrid | ✓ | 100 |
| TweetNER7 | bert_onnx | ✗ | 100 |
| TweetNER7 | candle_ner | ✗ | 100 |
| TweetNER7 | nuner | ✗ | 100 |
| TweetNER7 | gliner_onnx | ✗ | 100 |
| TweetNER7 | gliner_candle | ✗ | 100 |
| TweetNER7 | gliner2 | ✗ | 100 |
| TweetNER7 | w2ner | ✗ | 100 |
| BroadTwitterCorpus | pattern | ✓ | 100 |
| BroadTwitterCorpus | heuristic | ✓ | 100 |
| BroadTwitterCorpus | stacked | ✓ | 100 |
| BroadTwitterCorpus | hybrid | ✓ | 100 |
| BroadTwitterCorpus | bert_onnx | ✗ | 100 |
| BroadTwitterCorpus | candle_ner | ✗ | 100 |
| BroadTwitterCorpus | nuner | ✗ | 100 |
| BroadTwitterCorpus | gliner_onnx | ✗ | 100 |
| BroadTwitterCorpus | gliner_candle | ✗ | 100 |
| BroadTwitterCorpus | gliner2 | ✗ | 100 |
| BroadTwitterCorpus | w2ner | ✗ | 100 |
| FabNER | pattern | ✓ | 100 |
| FabNER | heuristic | ✓ | 100 |
| FabNER | stacked | ✓ | 100 |
| FabNER | hybrid | ✓ | 100 |
| FabNER | bert_onnx | ✗ | 100 |
| FabNER | candle_ner | ✗ | 100 |
| FabNER | nuner | ✗ | 100 |
| FabNER | gliner_onnx | ✗ | 100 |
| FabNER | gliner_candle | ✗ | 100 |
| FabNER | gliner2 | ✗ | 100 |
| FabNER | w2ner | ✗ | 100 |
| FewNERD | pattern | ✓ | 100 |
| FewNERD | heuristic | ✓ | 100 |
| FewNERD | stacked | ✓ | 100 |
| FewNERD | hybrid | ✓ | 100 |
| FewNERD | bert_onnx | ✗ | 100 |
| FewNERD | candle_ner | ✗ | 100 |
| FewNERD | nuner | ✗ | 100 |
| FewNERD | gliner_onnx | ✗ | 100 |
| FewNERD | gliner_candle | ✗ | 100 |
| FewNERD | gliner2 | ✗ | 100 |
| FewNERD | w2ner | ✗ | 100 |
| CrossNER | pattern | ✓ | 100 |
| CrossNER | heuristic | ✓ | 100 |
| CrossNER | stacked | ✓ | 100 |
| CrossNER | hybrid | ✓ | 100 |
| CrossNER | bert_onnx | ✗ | 100 |
| CrossNER | candle_ner | ✗ | 100 |
| CrossNER | nuner | ✗ | 100 |
| CrossNER | gliner_onnx | ✗ | 100 |
| CrossNER | gliner_candle | ✗ | 100 |
| CrossNER | gliner2 | ✗ | 100 |
| CrossNER | w2ner | ✗ | 100 |
| UniversalNERBench | pattern | ✓ | 100 |
| UniversalNERBench | heuristic | ✓ | 100 |
| UniversalNERBench | stacked | ✓ | 100 |
| UniversalNERBench | hybrid | ✓ | 100 |
| UniversalNERBench | bert_onnx | ✗ | 100 |
| UniversalNERBench | candle_ner | ✗ | 100 |
| UniversalNERBench | nuner | ✗ | 100 |
| UniversalNERBench | gliner_onnx | ✗ | 100 |
| UniversalNERBench | gliner_candle | ✗ | 100 |
| UniversalNERBench | gliner2 | ✗ | 100 |
| UniversalNERBench | w2ner | ✗ | 100 |
| WikiANN | pattern | ✓ | 100 |
| WikiANN | heuristic | ✓ | 100 |
| WikiANN | stacked | ✓ | 100 |
| WikiANN | hybrid | ✓ | 100 |
| WikiANN | bert_onnx | ✗ | 100 |
| WikiANN | candle_ner | ✗ | 100 |
| WikiANN | nuner | ✗ | 100 |
| WikiANN | gliner_onnx | ✗ | 100 |
| WikiANN | gliner_candle | ✗ | 100 |
| WikiANN | gliner2 | ✗ | 100 |
| WikiANN | w2ner | ✗ | 100 |
| MultiCoNER | pattern | ✓ | 100 |
| MultiCoNER | heuristic | ✓ | 100 |
| MultiCoNER | stacked | ✓ | 100 |
| MultiCoNER | hybrid | ✓ | 100 |
| MultiCoNER | bert_onnx | ✗ | 100 |
| MultiCoNER | candle_ner | ✗ | 100 |
| MultiCoNER | nuner | ✗ | 100 |
| MultiCoNER | gliner_onnx | ✗ | 100 |
| MultiCoNER | gliner_candle | ✗ | 100 |
| MultiCoNER | gliner2 | ✗ | 100 |
| MultiCoNER | w2ner | ✗ | 100 |
| MultiCoNERv2 | pattern | ✓ | 100 |
| MultiCoNERv2 | heuristic | ✓ | 100 |
| MultiCoNERv2 | stacked | ✓ | 100 |
| MultiCoNERv2 | hybrid | ✓ | 100 |
| MultiCoNERv2 | bert_onnx | ✗ | 100 |
| MultiCoNERv2 | candle_ner | ✗ | 100 |
| MultiCoNERv2 | nuner | ✗ | 100 |
| MultiCoNERv2 | gliner_onnx | ✗ | 100 |
| MultiCoNERv2 | gliner_candle | ✗ | 100 |
| MultiCoNERv2 | gliner2 | ✗ | 100 |
| MultiCoNERv2 | w2ner | ✗ | 100 |
| WikiNeural | pattern | ✓ | 100 |
| WikiNeural | heuristic | ✓ | 100 |
| WikiNeural | stacked | ✓ | 100 |
| WikiNeural | hybrid | ✓ | 100 |
| WikiNeural | bert_onnx | ✗ | 100 |
| WikiNeural | candle_ner | ✗ | 100 |
| WikiNeural | nuner | ✗ | 100 |
| WikiNeural | gliner_onnx | ✗ | 100 |
| WikiNeural | gliner_candle | ✗ | 100 |
| WikiNeural | gliner2 | ✗ | 100 |
| WikiNeural | w2ner | ✗ | 100 |
| PolyglotNER | pattern | ✓ | 100 |
| PolyglotNER | heuristic | ✓ | 100 |
| PolyglotNER | stacked | ✓ | 100 |
| PolyglotNER | hybrid | ✓ | 100 |
| PolyglotNER | bert_onnx | ✗ | 100 |
| PolyglotNER | candle_ner | ✗ | 100 |
| PolyglotNER | nuner | ✗ | 100 |
| PolyglotNER | gliner_onnx | ✗ | 100 |
| PolyglotNER | gliner_candle | ✗ | 100 |
| PolyglotNER | gliner2 | ✗ | 100 |
| PolyglotNER | w2ner | ✗ | 100 |
| UniversalNER | pattern | ✓ | 100 |
| UniversalNER | heuristic | ✓ | 100 |
| UniversalNER | stacked | ✓ | 100 |
| UniversalNER | hybrid | ✓ | 100 |
| UniversalNER | bert_onnx | ✗ | 100 |
| UniversalNER | candle_ner | ✗ | 100 |
| UniversalNER | nuner | ✗ | 100 |
| UniversalNER | gliner_onnx | ✗ | 100 |
| UniversalNER | gliner_candle | ✗ | 100 |
| UniversalNER | gliner2 | ✗ | 100 |
| UniversalNER | w2ner | ✗ | 100 |

### Intra-document Coreference

| Dataset | Backend | Success | Examples |
|---------|---------|---------|----------|
| GAP | coref_resolver | ✗ | 100 |
| PreCo | coref_resolver | ✗ | 100 |
| LitBank | coref_resolver | ✗ | 1 |

### Relation Extraction

| Dataset | Backend | Success | Examples |
|---------|---------|---------|----------|
| DocRED | gliner2 | ✗ | 100 |
| ReTACRED | gliner2 | ✗ | 100 |

