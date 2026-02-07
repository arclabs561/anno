# Eval Report

Total: 1 | ✓: 1 | ⊘: 0 | ✗: 0 | Avg examples: 5 | Avg time: 501ms

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
| WikiGold | stacked | 66.7 | 58.8 | 76.9 | 5 | 501 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| DATE | 0.00 | [0.00, 0.00] | 2 |
| LOC | 0.50 | [0.00, 1.00] | 4 |
| ORG | 0.56 | [0.00, 1.00] | 3 |
| PER | 0.50 | [0.50, 0.50] | 1 |


**Confidence Intervals (95%)**: F1: [0.29, 0.97], P: [0.25, 0.89], R: [0.34, 1.00]


## Backend Summary

| Backend | ✓ | ⊘ | ✗ | Avg F1 |
|---------|---|---|---|--------|
| stacked | 1 | 0 | 0 | 66.7 |

