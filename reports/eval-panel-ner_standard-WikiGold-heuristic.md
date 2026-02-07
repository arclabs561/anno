# Eval Report

Total: 1 | ✓: 1 | ⊘: 0 | ✗: 0 | Avg examples: 5 | Avg time: 0ms

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
| WikiGold | heuristic | 16.7 | 18.2 | 15.4 | 5 | 0 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.12 | [0.00, 0.37] | 4 |
| ORG | 0.20 | [0.00, 0.59] | 2 |
| PER | 0.00 | [0.00, 0.00] | 4 |


**Confidence Intervals (95%)**: F1: [0.00, 0.21], P: [0.00, 0.23], R: [0.00, 0.20]


## Backend Summary

| Backend | ✓ | ⊘ | ✗ | Avg F1 |
|---------|---|---|---|--------|
| heuristic | 1 | 0 | 0 | 16.7 |

