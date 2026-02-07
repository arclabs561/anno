# Eval Report

## Backend macro averages (successful only)

| Task | Backend | Avg primary metric | n |
|------|---------|--------------------|---|
| Named Entity Recognition | gliner_onnx | 27.9 | 4 |
| Named Entity Recognition | nuner | 17.1 | 4 |

Total: 8 | ✓: 8 | ⊘: 0 | ✗: 0 | Avg examples: 10 | Avg time: 2967ms

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
| MultiNERD | gliner_onnx | 72.7 | 75.0 | 70.6 | 10 | 719 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.44 | [0.04, 0.85] | 6 |
| MEDIA | 0.00 | [0.00, 0.00] | 1 |
| ORG | 1.00 | [1.00, 1.00] | 2 |
| PER | 0.60 | [0.12, 1.00] | 5 |


**Confidence Intervals (95%)**: F1: [0.33, 0.88], P: [0.32, 0.87], R: [0.36, 0.93]

| MultiNERD | nuner | 68.4 | 61.9 | 76.5 | 10 | 4806 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.50 | [0.06, 0.94] | 6 |
| MEDIA | 0.00 | [0.00, 0.00] | 1 |
| ORG | 0.56 | [0.00, 1.00] | 3 |
| PER | 0.67 | [0.25, 1.00] | 6 |


**Confidence Intervals (95%)**: F1: [0.53, 0.95], P: [0.49, 0.93], R: [0.59, 1.00]

| WikiANN | gliner_onnx | 25.0 | 23.1 | 27.3 | 10 | 770 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.00 | [0.00, 0.00] | 6 |
| ORG | 0.08 | [0.00, 0.25] | 6 |
| PER | 0.40 | [0.00, 0.88] | 5 |


**Confidence Intervals (95%)**: F1: [0.00, 0.50], P: [0.00, 0.48], R: [0.00, 0.60]

| MasakhaNER | gliner_onnx | 13.8 | 13.3 | 14.3 | 10 | 787 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| DATE | 0.00 | [0.00, 0.00] | 2 |
| LOC | 0.22 | [0.00, 0.52] | 4 |
| ORG | 0.25 | [0.00, 0.74] | 4 |
| PER | 0.33 | [0.00, 0.99] | 3 |


**Confidence Intervals (95%)**: F1: [0.02, 0.36], P: [0.00, 0.52], R: [0.03, 0.30]

| WikiANN | nuner | 0.0 | 0.0 | 0.0 | 10 | 4946 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.00 | [0.00, 0.10] | 2 |
| ORG | 0.00 | [0.00, 0.10] | 5 |
| PER | 0.00 | [0.00, 0.10] | 4 |


**Confidence Intervals (95%)**: F1: [0.00, 0.10], P: [0.00, 0.10], R: [0.00, 0.10]

| MasakhaNER | nuner | 0.0 | 0.0 | 0.0 | 10 | 5586 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| DATE | 0.00 | [0.00, 0.10] | 2 |
| LOC | 0.00 | [0.00, 0.10] | 7 |
| ORG | 0.00 | [0.00, 0.10] | 2 |
| PER | 0.00 | [0.00, 0.10] | 3 |


**Confidence Intervals (95%)**: F1: [0.00, 0.10], P: [0.00, 0.10], R: [0.00, 0.10]

| MultiCoNERv2 | nuner | 0.0 | 0.0 | 0.0 | 10 | 5412 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| AEROSPACEMANUFACTURER | 0.00 | [0.00, 0.10] | 10 |
| ARTWORK | 0.00 | [0.00, 0.10] | 1 |
| ORG | 0.00 | [0.00, 0.10] | 1 |
| OTHERPROD | 0.00 | [0.00, 0.10] | 2 |


**Confidence Intervals (95%)**: F1: [0.00, 0.10], P: [0.00, 0.10], R: [0.00, 0.10]

| MultiCoNERv2 | gliner_onnx | 0.0 | 0.0 | 0.0 | 10 | 707 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| AEROSPACEMANUFACTURER | 0.00 | [0.00, 0.00] | 9 |
| ARTWORK | 0.00 | [0.00, 0.00] | 1 |
| ORG | 0.00 | [0.00, 0.00] | 1 |
| OTHERPROD | 0.00 | [0.00, 0.00] | 2 |
| PER | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]


## Backend Summary

| Backend | ✓ | ⊘ | ✗ | Avg F1 |
|---------|---|---|---|--------|
| gliner_onnx | 4 | 0 | 0 | 27.9 |
| nuner | 4 | 0 | 0 | 17.1 |

