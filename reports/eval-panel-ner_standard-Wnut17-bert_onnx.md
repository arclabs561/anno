# Eval Report

Total: 1 | ✓: 1 | ⊘: 0 | ✗: 0 | Avg examples: 5 | Avg time: 421ms

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
| Wnut17 | bert_onnx | 33.3 | 27.3 | 42.9 | 5 | 421 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| GROUP | 0.00 | [0.00, 0.00] | 1 |
| LOC | 1.00 | [1.00, 1.00] | 1 |
| MISC | 0.00 | [0.00, 0.00] | 1 |
| ORG | 0.00 | [0.00, 0.00] | 2 |
| PER | 0.33 | [0.33, 0.33] | 1 |


**Confidence Intervals (95%)**: F1: [0.00, 0.37], P: [0.00, 0.32], R: [0.00, 0.44]


## Backend Summary

| Backend | ✓ | ⊘ | ✗ | Avg F1 |
|---------|---|---|---|--------|
| bert_onnx | 1 | 0 | 0 | 33.3 |

