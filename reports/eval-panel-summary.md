# Eval panel summary
- panel: `ner_standard`
- mode: `wide`
- max_examples: `5`
- seed: `42`
- cached_only: `True`

## NER averages (macro over rows seen)
| Backend | Avg F1 | n |
|---------|--------|---|
| bert_onnx | 34.9 | 3 |
| heuristic | 16.7 | 1 |
| stacked | 38.9 | 2 |

## NER rows
| Dataset | Backend | F1 |
|---------|---------|----|
| CoNLL2003Sample | bert_onnx | 0.0 |
| WikiGold | bert_onnx | 71.4 |
| WikiGold | heuristic | 16.7 |
| WikiGold | stacked | 66.7 |
| Wnut17 | bert_onnx | 33.3 |
| Wnut17 | stacked | 11.1 |
