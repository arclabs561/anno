# Evaluation Report

## Executive Summary

- **Total Combinations**: 258
- **Successful**: 75
- **Skipped** (feature not available): 177
- **Failed** (actual errors): 6

⚠️ **Warning**: Average of 20 examples per evaluation. Results may not be statistically significant. Consider running with more examples for reliable metrics.


## Key Insights

### Named Entity Recognition

**Top Performers**:
- 🥇 heuristic: 42.3% F1 (20 examples)
- 🥈 stacked: 42.2% F1 (20 examples)
- 🥉 stacked: 39.5% F1 (20 examples)

### Backend Availability

- **pattern**: 25 successful, 0 skipped (100% available)
- **heuristic**: 25 successful, 0 skipped (100% available)
- **stacked**: 25 successful, 0 skipped (100% available)
- **bert_onnx**: 0 successful, 25 skipped (0% available)
- **candle_ner**: 0 successful, 25 skipped (0% available)
- **gliner2**: 0 successful, 27 skipped (0% available)
- **gliner_onnx**: 0 successful, 25 skipped (0% available)
- **nuner**: 0 successful, 25 skipped (0% available)
- **w2ner**: 0 successful, 25 skipped (0% available)
- **gliner_candle**: 0 successful, 25 skipped (0% available)

## Tasks Evaluated

- Named Entity Recognition
- Named Entity Disambiguation
- Relation Extraction
- Intra-document Coreference
- Inter-document Coreference
- Abstract Anaphora Resolution
- Discontinuous NER
- Event Extraction
- Text Classification
- Hierarchical Structure Extraction

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
- NYTFB
- WEBNLG
- GoogleRE
- BioRED
- GAP
- PreCo
- LitBank
- CADEC

## Backends Tested

- pattern
- heuristic
- stacked
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

| Dataset | Backend | Status | F1 | P | R | Examples | Time (ms) |
|---------|---------|--------|----|----|----|----------|-----------|
| WikiANN | heuristic | ✓ | 42.3% | 37.4% | 48.6% | 20 | 1 |
| WikiANN | stacked | ✓ | 42.2% | 37.3% | 48.6% | 20 | 1 |
| MultiNERD | stacked | ✓ | 39.5% | 39.8% | 39.2% | 20 | 1939 |
| MultiNERD | heuristic | ✓ | 39.1% | 40.0% | 38.2% | 20 | 1711 |
| CoNLL2003Sample | heuristic | ✓ | 36.3% | 36.5% | 36.2% | 20 | 1369 |
| CoNLL2003Sample | stacked | ✓ | 35.9% | 34.7% | 37.3% | 20 | 1617 |
| OntoNotesSample | stacked | ✓ | 34.6% | 33.6% | 35.7% | 20 | 94 |
| OntoNotesSample | heuristic | ✓ | 34.2% | 34.7% | 33.6% | 20 | 73 |
| WikiGold | heuristic | ✓ | 33.3% | 31.1% | 35.8% | 20 | 45 |
| WikiGold | stacked | ✓ | 32.7% | 30.4% | 35.5% | 20 | 56 |
| MultiCoNERv2 | heuristic | ✓ | 31.7% | 30.1% | 33.5% | 20 | 3 |
| MultiCoNERv2 | stacked | ✓ | 31.6% | 29.9% | 33.5% | 20 | 4 |
| PolyglotNER | heuristic | ✓ | 20.0% | 16.9% | 24.4% | 20 | 1 |
| PolyglotNER | stacked | ✓ | 19.8% | 16.7% | 24.4% | 20 | 1 |
| Wnut17 | heuristic | ✓ | 18.8% | 12.1% | 41.7% | 20 | 67 |
| BroadTwitterCorpus | heuristic | ✓ | 18.7% | 12.9% | 34.2% | 20 | 7 |
| BroadTwitterCorpus | stacked | ✓ | 16.3% | 10.7% | 34.2% | 20 | 13 |
| Wnut17 | stacked | ✓ | 14.1% | 8.5% | 41.4% | 20 | 112 |
| TweetNER7 | heuristic | ✓ | 13.9% | 11.7% | 17.3% | 20 | 20 |
| TweetNER7 | stacked | ✓ | 12.5% | 10.4% | 15.7% | 20 | 33 |
| FewNERD | heuristic | ✓ | 1.6% | 2.2% | 1.2% | 20 | 1 |
| FewNERD | stacked | ✓ | 1.3% | 1.9% | 1.1% | 20 | 2 |
| MultiCoNER | heuristic | ✓ | 1.0% | 1.5% | 0.8% | 20 | 1 |
| MultiCoNER | stacked | ✓ | 1.0% | 1.5% | 0.8% | 20 | 2 |
| CrossNER | heuristic | ✓ | 0.9% | 1.0% | 0.8% | 20 | 2 |
| CrossNER | stacked | ✓ | 0.9% | 1.0% | 0.8% | 20 | 3 |
| MitRestaurant | pattern | ✓ | 0.0% | 0.4% | 0.0% | 20 | 33 |
| MitRestaurant | stacked | ✓ | 0.0% | 0.4% | 0.0% | 20 | 38 |
| WikiGold | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 27 |
| Wnut17 | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 43 |
| MitMovie | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 41 |
| MitMovie | heuristic | ✓ | 0.0% | 0.0% | 0.0% | 20 | 25 |
| MitMovie | stacked | ✓ | 0.0% | 0.0% | 0.0% | 20 | 44 |
| MitRestaurant | heuristic | ✓ | 0.0% | 0.0% | 0.0% | 20 | 19 |
| CoNLL2003Sample | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 104 |
| OntoNotesSample | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 18 |
| MultiNERD | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 162 |
| BC5CDR | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 32 |
| BC5CDR | heuristic | ✓ | 0.0% | 0.0% | 0.0% | 20 | 44 |
| BC5CDR | stacked | ✓ | 0.0% | 0.0% | 0.0% | 20 | 65 |
| NCBIDisease | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 36 |
| NCBIDisease | heuristic | ✓ | 0.0% | 0.0% | 0.0% | 20 | 131 |
| NCBIDisease | stacked | ✓ | 0.0% | 0.0% | 0.0% | 20 | 159 |
| GENIA | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 1 |
| GENIA | heuristic | ✓ | 0.0% | 0.0% | 0.0% | 20 | 1 |
| GENIA | stacked | ✓ | 0.0% | 0.0% | 0.0% | 20 | 1 |
| AnatEM | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 1 |
| AnatEM | heuristic | ✓ | 0.0% | 0.0% | 0.0% | 20 | 1 |
| AnatEM | stacked | ✓ | 0.0% | 0.0% | 0.0% | 20 | 1 |
| BC2GM | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 1 |
| BC2GM | heuristic | ✓ | 0.0% | 0.0% | 0.0% | 20 | 0 |
| BC2GM | stacked | ✓ | 0.0% | 0.0% | 0.0% | 20 | 1 |
| BC4CHEMD | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 1 |
| BC4CHEMD | heuristic | ✓ | 0.0% | 0.0% | 0.0% | 20 | 0 |
| BC4CHEMD | stacked | ✓ | 0.0% | 0.0% | 0.0% | 20 | 1 |
| TweetNER7 | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 16 |
| BroadTwitterCorpus | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 7 |
| FabNER | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 1 |
| FabNER | heuristic | ✓ | 0.0% | 0.0% | 0.0% | 20 | 0 |
| FabNER | stacked | ✓ | 0.0% | 0.0% | 0.0% | 20 | 1 |
| FewNERD | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 1 |
| CrossNER | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 2 |
| UniversalNERBench | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 31 |
| UniversalNERBench | heuristic | ✓ | 0.0% | 0.0% | 0.0% | 20 | 67 |
| UniversalNERBench | stacked | ✓ | 0.0% | 0.0% | 0.0% | 20 | 73 |
| WikiANN | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 1 |
| MultiCoNER | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 1 |
| MultiCoNERv2 | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 2 |
| WikiNeural | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 2 |
| WikiNeural | heuristic | ✓ | 0.0% | 0.0% | 0.0% | 20 | 2 |
| WikiNeural | stacked | ✓ | 0.0% | 0.0% | 0.0% | 20 | 3 |
| PolyglotNER | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 1 |
| UniversalNER | pattern | ✓ | 0.0% | 0.0% | 0.0% | 20 | 2 |
| UniversalNER | heuristic | ✓ | 0.0% | 0.0% | 0.0% | 20 | 2 |
| UniversalNER | stacked | ✓ | 0.0% | 0.0% | 0.0% | 20 | 3 |
| WikiGold | bert_onnx | ⊘ | Feature not available | 0 |
| WikiGold | candle_ner | ⊘ | Feature not available | 0 |
| WikiGold | nuner | ⊘ | Feature not available | 0 |
| WikiGold | gliner_onnx | ⊘ | Feature not available | 0 |
| WikiGold | gliner_candle | ⊘ | Feature not available | 0 |
| WikiGold | gliner2 | ⊘ | Feature not available | 0 |
| WikiGold | w2ner | ⊘ | Feature not available | 0 |
| Wnut17 | bert_onnx | ⊘ | Feature not available | 0 |
| Wnut17 | candle_ner | ⊘ | Feature not available | 0 |
| Wnut17 | nuner | ⊘ | Feature not available | 0 |
| Wnut17 | gliner_onnx | ⊘ | Feature not available | 0 |
| Wnut17 | gliner_candle | ⊘ | Feature not available | 0 |
| Wnut17 | gliner2 | ⊘ | Feature not available | 0 |
| Wnut17 | w2ner | ⊘ | Feature not available | 0 |
| MitMovie | bert_onnx | ⊘ | Feature not available | 0 |
| MitMovie | candle_ner | ⊘ | Feature not available | 0 |
| MitMovie | nuner | ⊘ | Feature not available | 0 |
| MitMovie | gliner_onnx | ⊘ | Feature not available | 0 |
| MitMovie | gliner_candle | ⊘ | Feature not available | 0 |
| MitMovie | gliner2 | ⊘ | Feature not available | 0 |
| MitMovie | w2ner | ⊘ | Feature not available | 0 |
| MitRestaurant | bert_onnx | ⊘ | Feature not available | 0 |
| MitRestaurant | candle_ner | ⊘ | Feature not available | 0 |
| MitRestaurant | nuner | ⊘ | Feature not available | 0 |
| MitRestaurant | gliner_onnx | ⊘ | Feature not available | 0 |
| MitRestaurant | gliner_candle | ⊘ | Feature not available | 0 |
| MitRestaurant | gliner2 | ⊘ | Feature not available | 0 |
| MitRestaurant | w2ner | ⊘ | Feature not available | 0 |
| CoNLL2003Sample | bert_onnx | ⊘ | Feature not available | 0 |
| CoNLL2003Sample | candle_ner | ⊘ | Feature not available | 0 |
| CoNLL2003Sample | nuner | ⊘ | Feature not available | 0 |
| CoNLL2003Sample | gliner_onnx | ⊘ | Feature not available | 0 |
| CoNLL2003Sample | gliner_candle | ⊘ | Feature not available | 0 |
| CoNLL2003Sample | gliner2 | ⊘ | Feature not available | 0 |
| CoNLL2003Sample | w2ner | ⊘ | Feature not available | 0 |
| OntoNotesSample | bert_onnx | ⊘ | Feature not available | 0 |
| OntoNotesSample | candle_ner | ⊘ | Feature not available | 0 |
| OntoNotesSample | nuner | ⊘ | Feature not available | 0 |
| OntoNotesSample | gliner_onnx | ⊘ | Feature not available | 0 |
| OntoNotesSample | gliner_candle | ⊘ | Feature not available | 0 |
| OntoNotesSample | gliner2 | ⊘ | Feature not available | 0 |
| OntoNotesSample | w2ner | ⊘ | Feature not available | 0 |
| MultiNERD | bert_onnx | ⊘ | Feature not available | 0 |
| MultiNERD | candle_ner | ⊘ | Feature not available | 0 |
| MultiNERD | nuner | ⊘ | Feature not available | 0 |
| MultiNERD | gliner_onnx | ⊘ | Feature not available | 0 |
| MultiNERD | gliner_candle | ⊘ | Feature not available | 0 |
| MultiNERD | gliner2 | ⊘ | Feature not available | 0 |
| MultiNERD | w2ner | ⊘ | Feature not available | 0 |
| BC5CDR | bert_onnx | ⊘ | Feature not available | 0 |
| BC5CDR | candle_ner | ⊘ | Feature not available | 0 |
| BC5CDR | nuner | ⊘ | Feature not available | 0 |
| BC5CDR | gliner_onnx | ⊘ | Feature not available | 0 |
| BC5CDR | gliner_candle | ⊘ | Feature not available | 0 |
| BC5CDR | gliner2 | ⊘ | Feature not available | 0 |
| BC5CDR | w2ner | ⊘ | Feature not available | 0 |
| NCBIDisease | bert_onnx | ⊘ | Feature not available | 0 |
| NCBIDisease | candle_ner | ⊘ | Feature not available | 0 |
| NCBIDisease | nuner | ⊘ | Feature not available | 0 |
| NCBIDisease | gliner_onnx | ⊘ | Feature not available | 0 |
| NCBIDisease | gliner_candle | ⊘ | Feature not available | 0 |
| NCBIDisease | gliner2 | ⊘ | Feature not available | 0 |
| NCBIDisease | w2ner | ⊘ | Feature not available | 0 |
| GENIA | bert_onnx | ⊘ | Feature not available | 0 |
| GENIA | candle_ner | ⊘ | Feature not available | 0 |
| GENIA | nuner | ⊘ | Feature not available | 0 |
| GENIA | gliner_onnx | ⊘ | Feature not available | 0 |
| GENIA | gliner_candle | ⊘ | Feature not available | 0 |
| GENIA | gliner2 | ⊘ | Feature not available | 0 |
| GENIA | w2ner | ⊘ | Feature not available | 0 |
| AnatEM | bert_onnx | ⊘ | Feature not available | 0 |
| AnatEM | candle_ner | ⊘ | Feature not available | 0 |
| AnatEM | nuner | ⊘ | Feature not available | 0 |
| AnatEM | gliner_onnx | ⊘ | Feature not available | 0 |
| AnatEM | gliner_candle | ⊘ | Feature not available | 0 |
| AnatEM | gliner2 | ⊘ | Feature not available | 0 |
| AnatEM | w2ner | ⊘ | Feature not available | 0 |
| BC2GM | bert_onnx | ⊘ | Feature not available | 0 |
| BC2GM | candle_ner | ⊘ | Feature not available | 0 |
| BC2GM | nuner | ⊘ | Feature not available | 0 |
| BC2GM | gliner_onnx | ⊘ | Feature not available | 0 |
| BC2GM | gliner_candle | ⊘ | Feature not available | 0 |
| BC2GM | gliner2 | ⊘ | Feature not available | 0 |
| BC2GM | w2ner | ⊘ | Feature not available | 0 |
| BC4CHEMD | bert_onnx | ⊘ | Feature not available | 0 |
| BC4CHEMD | candle_ner | ⊘ | Feature not available | 0 |
| BC4CHEMD | nuner | ⊘ | Feature not available | 0 |
| BC4CHEMD | gliner_onnx | ⊘ | Feature not available | 0 |
| BC4CHEMD | gliner_candle | ⊘ | Feature not available | 0 |
| BC4CHEMD | gliner2 | ⊘ | Feature not available | 0 |
| BC4CHEMD | w2ner | ⊘ | Feature not available | 0 |
| TweetNER7 | bert_onnx | ⊘ | Feature not available | 0 |
| TweetNER7 | candle_ner | ⊘ | Feature not available | 0 |
| TweetNER7 | nuner | ⊘ | Feature not available | 0 |
| TweetNER7 | gliner_onnx | ⊘ | Feature not available | 0 |
| TweetNER7 | gliner_candle | ⊘ | Feature not available | 0 |
| TweetNER7 | gliner2 | ⊘ | Feature not available | 0 |
| TweetNER7 | w2ner | ⊘ | Feature not available | 0 |
| BroadTwitterCorpus | bert_onnx | ⊘ | Feature not available | 0 |
| BroadTwitterCorpus | candle_ner | ⊘ | Feature not available | 0 |
| BroadTwitterCorpus | nuner | ⊘ | Feature not available | 0 |
| BroadTwitterCorpus | gliner_onnx | ⊘ | Feature not available | 0 |
| BroadTwitterCorpus | gliner_candle | ⊘ | Feature not available | 0 |
| BroadTwitterCorpus | gliner2 | ⊘ | Feature not available | 0 |
| BroadTwitterCorpus | w2ner | ⊘ | Feature not available | 0 |
| FabNER | bert_onnx | ⊘ | Feature not available | 0 |
| FabNER | candle_ner | ⊘ | Feature not available | 0 |
| FabNER | nuner | ⊘ | Feature not available | 0 |
| FabNER | gliner_onnx | ⊘ | Feature not available | 0 |
| FabNER | gliner_candle | ⊘ | Feature not available | 0 |
| FabNER | gliner2 | ⊘ | Feature not available | 0 |
| FabNER | w2ner | ⊘ | Feature not available | 0 |
| FewNERD | bert_onnx | ⊘ | Feature not available | 0 |
| FewNERD | candle_ner | ⊘ | Feature not available | 0 |
| FewNERD | nuner | ⊘ | Feature not available | 0 |
| FewNERD | gliner_onnx | ⊘ | Feature not available | 0 |
| FewNERD | gliner_candle | ⊘ | Feature not available | 0 |
| FewNERD | gliner2 | ⊘ | Feature not available | 0 |
| FewNERD | w2ner | ⊘ | Feature not available | 0 |
| CrossNER | bert_onnx | ⊘ | Feature not available | 0 |
| CrossNER | candle_ner | ⊘ | Feature not available | 0 |
| CrossNER | nuner | ⊘ | Feature not available | 0 |
| CrossNER | gliner_onnx | ⊘ | Feature not available | 0 |
| CrossNER | gliner_candle | ⊘ | Feature not available | 0 |
| CrossNER | gliner2 | ⊘ | Feature not available | 0 |
| CrossNER | w2ner | ⊘ | Feature not available | 0 |
| UniversalNERBench | bert_onnx | ⊘ | Feature not available | 0 |
| UniversalNERBench | candle_ner | ⊘ | Feature not available | 0 |
| UniversalNERBench | nuner | ⊘ | Feature not available | 0 |
| UniversalNERBench | gliner_onnx | ⊘ | Feature not available | 0 |
| UniversalNERBench | gliner_candle | ⊘ | Feature not available | 0 |
| UniversalNERBench | gliner2 | ⊘ | Feature not available | 0 |
| UniversalNERBench | w2ner | ⊘ | Feature not available | 0 |
| WikiANN | bert_onnx | ⊘ | Feature not available | 0 |
| WikiANN | candle_ner | ⊘ | Feature not available | 0 |
| WikiANN | nuner | ⊘ | Feature not available | 0 |
| WikiANN | gliner_onnx | ⊘ | Feature not available | 0 |
| WikiANN | gliner_candle | ⊘ | Feature not available | 0 |
| WikiANN | gliner2 | ⊘ | Feature not available | 0 |
| WikiANN | w2ner | ⊘ | Feature not available | 0 |
| MultiCoNER | bert_onnx | ⊘ | Feature not available | 0 |
| MultiCoNER | candle_ner | ⊘ | Feature not available | 0 |
| MultiCoNER | nuner | ⊘ | Feature not available | 0 |
| MultiCoNER | gliner_onnx | ⊘ | Feature not available | 0 |
| MultiCoNER | gliner_candle | ⊘ | Feature not available | 0 |
| MultiCoNER | gliner2 | ⊘ | Feature not available | 0 |
| MultiCoNER | w2ner | ⊘ | Feature not available | 0 |
| MultiCoNERv2 | bert_onnx | ⊘ | Feature not available | 0 |
| MultiCoNERv2 | candle_ner | ⊘ | Feature not available | 0 |
| MultiCoNERv2 | nuner | ⊘ | Feature not available | 0 |
| MultiCoNERv2 | gliner_onnx | ⊘ | Feature not available | 0 |
| MultiCoNERv2 | gliner_candle | ⊘ | Feature not available | 0 |
| MultiCoNERv2 | gliner2 | ⊘ | Feature not available | 0 |
| MultiCoNERv2 | w2ner | ⊘ | Feature not available | 0 |
| WikiNeural | bert_onnx | ⊘ | Feature not available | 0 |
| WikiNeural | candle_ner | ⊘ | Feature not available | 0 |
| WikiNeural | nuner | ⊘ | Feature not available | 0 |
| WikiNeural | gliner_onnx | ⊘ | Feature not available | 0 |
| WikiNeural | gliner_candle | ⊘ | Feature not available | 0 |
| WikiNeural | gliner2 | ⊘ | Feature not available | 0 |
| WikiNeural | w2ner | ⊘ | Feature not available | 0 |
| PolyglotNER | bert_onnx | ⊘ | Feature not available | 0 |
| PolyglotNER | candle_ner | ⊘ | Feature not available | 0 |
| PolyglotNER | nuner | ⊘ | Feature not available | 0 |
| PolyglotNER | gliner_onnx | ⊘ | Feature not available | 0 |
| PolyglotNER | gliner_candle | ⊘ | Feature not available | 0 |
| PolyglotNER | gliner2 | ⊘ | Feature not available | 0 |
| PolyglotNER | w2ner | ⊘ | Feature not available | 0 |
| UniversalNER | bert_onnx | ⊘ | Feature not available | 0 |
| UniversalNER | candle_ner | ⊘ | Feature not available | 0 |
| UniversalNER | nuner | ⊘ | Feature not available | 0 |
| UniversalNER | gliner_onnx | ⊘ | Feature not available | 0 |
| UniversalNER | gliner_candle | ⊘ | Feature not available | 0 |
| UniversalNER | gliner2 | ⊘ | Feature not available | 0 |
| UniversalNER | w2ner | ⊘ | Feature not available | 0 |

### Intra-document Coreference

| Dataset | Backend | Status | CoNLL F1 | MUC F1 | B³ F1 | Examples | Time (ms) |
|---------|---------|--------|----------|--------|-------|----------|-----------|
| GAP | coref_resolver | ✗ | Invalid input: Unknown backend: 'core... | 0 |
| PreCo | coref_resolver | ✗ | Invalid input: Unknown backend: 'core... | 0 |
| LitBank | coref_resolver | ✗ | Invalid input: Unknown backend: 'core... | 0 |

### Relation Extraction

| Dataset | Backend | Status | Strict F1 | Boundary F1 | Examples | Time (ms) |
|---------|---------|--------|------------|-------------|----------|-----------|
| DocRED | gliner2 | ⊘ | Feature not available | 0 |
| ReTACRED | gliner2 | ⊘ | Feature not available | 0 |

### Abstract Anaphora Resolution

| Dataset | Backend | Status | CoNLL F1 | MUC F1 | B³ F1 | Examples | Time (ms) |
|---------|---------|--------|----------|--------|-------|----------|-----------|
| GAP | coref_resolver | ✗ | Invalid input: Unknown backend: 'core... | 0 |
| PreCo | coref_resolver | ✗ | Invalid input: Unknown backend: 'core... | 0 |
| LitBank | coref_resolver | ✗ | Invalid input: Unknown backend: 'core... | 0 |


## Recommendations

### Enable More Features
Many backends are skipped due to missing features. Consider enabling:
- `onnx` feature for ONNX-based backends (bert_onnx, nuner, gliner_onnx, gliner2, w2ner)
- `candle` feature for Candle-based backends (candle_ner, gliner_candle)
- `discourse` feature for coreference resolution

### Increase Sample Size
For statistically significant results, run with more examples:
```bash
just eval-full-limit 100  # 100 examples per dataset
# or for full evaluation:
just eval-full
```

### Performance Notes
Average evaluation time: 112ms per combination

---

*Legend: ✓ = Success, ⊘ = Skipped (feature not available), ✗ = Failed*
