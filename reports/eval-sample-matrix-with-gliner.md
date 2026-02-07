# Eval Report

Total: 14 | ✓: 13 | ⊘: 0 | ✗: 1 | Avg examples: 14 | Avg time: 931ms

## Failures

| Task | Dataset | Backend | Error |
|------|---------|---------|-------|
| Named Entity Recognition | Wnut17 | heuristic | incompatible: backend 'heuristic' doesn't support dataset entity types: ["person", "location", "corporation", "product", "creative-work", "group"] |

## Error Patterns

- [1x] incompatible: backend 'heuristic' doesn't support ...

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
| GAP | coref_resolver | 33.3 | 0.0 | 75.0 | 20 | 9473 |

#### Chain-Length Stratification

| Chain Type | Count | F1 |
|------------|-------|----|
| Long (>10) | 0 | 0.00 |
| Short (2-10) | 18 | 77.78 |
| Singleton (1) | 24 | 100.00 |


### Relation Extraction

| Dataset | Backend | Strict | Boundary | N | ms |
|---------|---------|--------|----------|---|----|
| DocRED | tplinker | 0.0 | 0.0 | 20 | 4 |

### Named Entity Recognition

| Dataset | Backend | F1 | P | R | N | ms |
|---------|---------|----|----|----|---|----|
| WikiGold | bert_onnx | 75.2 | 72.1 | 78.6 | 20 | 438 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.73 | [0.48, 0.99] | 12 |
| MISC | 0.41 | [0.08, 0.73] | 9 |
| ORG | 0.66 | [0.39, 0.93] | 9 |
| PER | 0.73 | [0.44, 1.00] | 9 |


**Confidence Intervals (95%)**: F1: [0.41, 0.76], P: [0.39, 0.74], R: [0.44, 0.79]

| WikiGold | stacked | 49.6 | 44.9 | 55.4 | 20 | 429 |

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

| WikiGold | tplinker | 19.5 | 30.8 | 14.3 | 20 | 0 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.38 | [0.14, 0.63] | 13 |
| MISC | 0.00 | [0.00, 0.00] | 6 |
| ORG | 0.10 | [0.00, 0.23] | 9 |
| PER | 0.10 | [0.00, 0.26] | 13 |


**Confidence Intervals (95%)**: F1: [0.08, 0.35], P: [0.13, 0.49], R: [0.05, 0.30]

| Wnut17 | bert_onnx | 18.2 | 16.7 | 20.0 | 20 | 370 |

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

| Wnut17 | tplinker | 6.5 | 9.1 | 5.0 | 20 | 0 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| CREATIVE_WORK | 0.00 | [0.00, 0.00] | 1 |
| GROUP | 0.00 | [0.00, 0.00] | 4 |
| LOC | 0.22 | [0.00, 0.66] | 3 |
| ORG | 0.00 | [0.00, 0.00] | 5 |
| PER | 0.00 | [0.00, 0.00] | 5 |
| PRODUCT | 0.00 | [0.00, 0.00] | 2 |


**Confidence Intervals (95%)**: F1: [0.00, 0.05], P: [0.00, 0.05], R: [0.00, 0.05]

| Wnut17 | stacked | 6.3 | 4.5 | 10.0 | 20 | 372 |

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

| CoNLL2003Sample | heuristic | 0.0 | 0.0 | 0.0 | 1 | 0 | (heuristic: no PER/ORG/LOC)

**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| CoNLL2003Sample | stacked | 0.0 | 0.0 | 0.0 | 1 | 551 | (stacked: incompatible types)

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| DATE | 0.00 | [0.00, 0.00] | 1 |
| LOC | 0.00 | [0.00, 0.00] | 1 |
| MISC | 0.00 | [0.00, 0.00] | 1 |
| TIME | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| CoNLL2003Sample | bert_onnx | 0.0 | 0.0 | 0.0 | 1 | 469 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.00 | [0.00, 0.00] | 1 |
| MISC | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| CoNLL2003Sample | tplinker | 0.0 | 0.0 | 0.0 | 1 | 0 |

**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| Wnut17 | heuristic | ✗ | incompatible | - |

## Backend Summary

| Backend | ✓ | ⊘ | ✗ | Avg F1 |
|---------|---|---|---|--------|
| tplinker | 4 | 0 | 0 | 6.5 |
| bert_onnx | 3 | 0 | 0 | 31.1 |
| stacked | 3 | 0 | 0 | 18.6 |
| heuristic | 2 | 0 | 1 | 14.7 |
| coref_resolver | 1 | 0 | 0 | 33.3 |

