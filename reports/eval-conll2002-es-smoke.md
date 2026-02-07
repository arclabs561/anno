# Eval Report

Total: 1 | ✓: 0 | ⊘: 0 | ✗: 1 | Avg examples: -0 | Avg time: -0ms

## Failures

| Task | Dataset | Backend | Error |
|------|---------|---------|-------|
| Named Entity Recognition | CoNLL2002Spanish | bert_onnx | Failed to load dataset: Invalid input: Error downloading https://www.clips.uantwerpen.be/conll2002/ner/data/esp.testa: https://www.clips.uantwerpen.be/conll2002/ner/data/esp.testa: status code 403. Check your internet connection and try again. |

## Error Patterns

- [1x] Failed to load dataset: Invalid input: Error downl...

## Results

**Compatibility Notes**:
- `stacked`: Combines pattern+heuristic, supports structured entities (date/time/money/etc) and named entities (PER/ORG/LOC), but not biomedical types
- `pattern`: Only structured entities (date, time, money, percent, email, URL, phone)
- `heuristic`: Only named entities (Person, Organization, Location)
- `incompatible`: Backend doesn't support dataset entity types (expected for non-zero-shot backends on fine-grained datasets)
- `load-failed`: Dataset failed to download/load (HuggingFace API errors, network issues, etc.)
- `empty-dataset`: Dataset loaded but contains no sentences
- `0.0 F1` with N>0: Backend doesn't support dataset entity types
- `N=0` or `N=1`: Dataset parsing issue or insufficient data

### Named Entity Recognition

| Dataset | Backend | F1 | P | R | N | ms |
|---------|---------|----|----|----|---|----|
| CoNLL2002Spanish | bert_onnx | ✗ | load-failed | - |

## Backend Summary

| Backend | ✓ | ⊘ | ✗ | Avg F1 |
|---------|---|---|---|--------|
| bert_onnx | 0 | 0 | 1 | 0.0 |

