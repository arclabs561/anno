# Eval Report

## Backend macro averages (successful only)

| Task | Backend | Avg primary metric | n |
|------|---------|--------------------|---|
| Named Entity Recognition | bert_onnx | 31.1 | 3 |
| Named Entity Recognition | stacked | 18.6 | 3 |
| Named Entity Recognition | heuristic | 14.7 | 2 |

Total: 9 | ✓: 8 | ⊘: 1 | ✗: 0 | Avg examples: 13 | Avg time: 430ms

## Results

**Note**: 1 combinations skipped (features not enabled or incompatible). Showing successful and failed results only.

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
| WikiGold | bert_onnx | 75.2 | 72.1 | 78.6 | 20 | 546 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.73 | [0.48, 0.99] | 12 |
| MISC | 0.41 | [0.08, 0.73] | 9 |
| ORG | 0.66 | [0.39, 0.93] | 9 |
| PER | 0.73 | [0.44, 1.00] | 9 |


**Confidence Intervals (95%)**: F1: [0.41, 0.76], P: [0.39, 0.74], R: [0.44, 0.79]

| WikiGold | stacked | 49.6 | 44.9 | 55.4 | 20 | 459 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| DATE | 0.00 | [0.00, 0.00] | 5 |
| LAW | 0.00 | [0.00, 0.00] | 2 |
| LOC | 0.61 | [0.32, 0.90] | 11 |
| MISC | 0.00 | [0.00, 0.00] | 6 |
| ORG | 0.56 | [0.28, 0.84] | 11 |
| PER | 0.35 | [0.15, 0.54] | 13 |
| TIME | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.32, 0.62], P: [0.29, 0.57], R: [0.35, 0.70]

| WikiGold | heuristic | 29.3 | 28.3 | 30.4 | 20 | 0 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.38 | [0.14, 0.63] | 13 |
| MISC | 0.00 | [0.00, 0.00] | 6 |
| ORG | 0.10 | [0.00, 0.23] | 9 |
| PER | 0.15 | [0.00, 0.30] | 17 |


**Confidence Intervals (95%)**: F1: [0.07, 0.34], P: [0.07, 0.33], R: [0.08, 0.38]

| Wnut17 | bert_onnx | 18.2 | 16.7 | 20.0 | 20 | 459 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| CREATIVE_WORK | 0.00 | [0.00, 0.00] | 1 |
| GROUP | 0.00 | [0.00, 0.00] | 4 |
| LOC | 0.33 | [0.00, 0.99] | 3 |
| MISC | 0.00 | [0.00, 0.00] | 6 |
| ORG | 0.17 | [0.00, 0.49] | 4 |
| PER | 0.08 | [0.00, 0.25] | 4 |
| PRODUCT | 0.00 | [0.00, 0.00] | 2 |


**Confidence Intervals (95%)**: F1: [0.00, 0.16], P: [0.00, 0.13], R: [0.00, 0.21]

| Wnut17 | stacked | 6.3 | 4.5 | 10.0 | 20 | 499 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| CREATIVE_WORK | 0.00 | [0.00, 0.00] | 1 |
| DATE | 0.00 | [0.00, 0.00] | 1 |
| GROUP | 0.00 | [0.00, 0.00] | 4 |
| LAW | 0.00 | [0.00, 0.00] | 2 |
| LOC | 0.00 | [0.00, 0.00] | 2 |
| ORG | 0.00 | [0.00, 0.00] | 8 |
| PER | 0.14 | [0.00, 0.34] | 9 |
| PERCENT | 0.00 | [0.00, 0.00] | 1 |
| PRODUCT | 0.00 | [0.00, 0.00] | 3 |
| TIME | 0.00 | [0.00, 0.00] | 1 |
| URL | 0.00 | [0.00, 0.00] | 10 |


**Confidence Intervals (95%)**: F1: [0.00, 0.09], P: [0.00, 0.09], R: [0.00, 0.10]

| CoNLL2003Sample | bert_onnx | 0.0 | 0.0 | 0.0 | 1 | 945 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.00 | [0.00, 0.00] | 1 |
| MISC | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| CoNLL2003Sample | stacked | 0.0 | 0.0 | 0.0 | 1 | 532 | (stacked: incompatible types)

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| DATE | 0.00 | [0.00, 0.00] | 1 |
| LOC | 0.00 | [0.00, 0.00] | 1 |
| MISC | 0.00 | [0.00, 0.00] | 1 |
| TIME | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| CoNLL2003Sample | heuristic | 0.0 | 0.0 | 0.0 | 1 | 2 | (heuristic: no PER/ORG/LOC)

**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]


## Backend Summary

| Backend | ✓ | ⊘ | ✗ | Avg F1 |
|---------|---|---|---|--------|
| bert_onnx | 3 | 0 | 0 | 31.1 |
| stacked | 3 | 0 | 0 | 18.6 |
| heuristic | 2 | 1 | 0 | 14.7 |

