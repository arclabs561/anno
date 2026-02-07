# Eval Report

Total: 1 | ✓: 1 | ⊘: 0 | ✗: 0 | Avg examples: 5 | Avg time: 402ms

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
| Wnut17 | stacked | 11.1 | 9.1 | 14.3 | 5 | 402 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| GROUP | 0.00 | [0.00, 0.00] | 1 |
| LOC | 0.00 | [0.00, 0.00] | 1 |
| ORG | 0.00 | [0.00, 0.00] | 2 |
| PER | 0.40 | [0.00, 1.00] | 2 |
| PERCENT | 0.00 | [0.00, 0.00] | 1 |
| URL | 0.00 | [0.00, 0.00] | 3 |


**Confidence Intervals (95%)**: F1: [0.00, 0.30], P: [0.00, 0.30], R: [0.00, 0.30]


## Backend Summary

| Backend | ✓ | ⊘ | ✗ | Avg F1 |
|---------|---|---|---|--------|
| stacked | 1 | 0 | 0 | 11.1 |

