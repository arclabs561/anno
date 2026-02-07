# Eval Report

Total: 1 | ✓: 1 | ⊘: 0 | ✗: 0 | Avg examples: 20 | Avg time: 8990ms

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

### Intra-document Coreference

| Dataset | Backend | CoNLL | MUC | B³ | N | ms |
|---------|---------|-------|-----|----|---|----|
| GAP | coref_resolver | 33.0 | 0.0 | 75.0 | 20 | 8990 |

#### Chain-Length Stratification

| Chain Type | Count | F1 |
|------------|-------|----|
| Long (>10) | 0 | 0.00 |
| Short (2-10) | 18 | 77.78 |
| Singleton (1) | 24 | 100.00 |


## Backend Summary

| Backend | ✓ | ⊘ | ✗ | Avg F1 |
|---------|---|---|---|--------|
| coref_resolver | 1 | 0 | 0 | 33.0 |

