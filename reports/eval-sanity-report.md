# Eval Report

Total: 122 | ✓: 59 | ⊘: 40 | ✗: 23 | Avg examples: 20 | Avg time: 896ms

## Failures

| Task | Dataset | Backend | Error |
|------|---------|---------|-------|
| Named Entity Recognition | WikiGold | deberta_v3 | Retrieval error: DeBERTa-v3 model unavailable: Retrieval error: Failed to download model.onnx: request error: https://huggingface.co/microsoft/deberta-v3-base/resolve/main/onnx/model.onnx: status code 404  Options: 1. Export your own: uv run scripts/export_deberta_ner_to_onnx.py 2. Set DEBERTA_MODEL_PATH to a local model directory 3. Use --model bert-onnx or --model candle-ner instead |
| Named Entity Recognition | WikiGold | albert | Retrieval error: ALBERT model unavailable: Retrieval error: Failed to download model.onnx: request error: https://huggingface.co/albert-base-v2/resolve/main/onnx/model.onnx: status code 404  Options: 1. Export your own ONNX model 2. Set ALBERT_MODEL_PATH to a local model directory 3. Use --model bert-onnx or --model candle-ner instead |
| Named Entity Recognition | Wnut17 | heuristic | incompatible: backend 'heuristic' doesn't support dataset entity types: ["person", "location", "corporation", "product", "creative-work", "group"] |
| Named Entity Recognition | Wnut17 | deberta_v3 | Retrieval error: DeBERTa-v3 model unavailable: Retrieval error: Failed to download model.onnx: request error: https://huggingface.co/microsoft/deberta-v3-base/resolve/main/onnx/model.onnx: status code 404  Options: 1. Export your own: uv run scripts/export_deberta_ner_to_onnx.py 2. Set DEBERTA_MODEL_PATH to a local model directory 3. Use --model bert-onnx or --model candle-ner instead |
| Named Entity Recognition | Wnut17 | albert | Retrieval error: ALBERT model unavailable: Retrieval error: Failed to download model.onnx: request error: https://huggingface.co/albert-base-v2/resolve/main/onnx/model.onnx: status code 404  Options: 1. Export your own ONNX model 2. Set ALBERT_MODEL_PATH to a local model directory 3. Use --model bert-onnx or --model candle-ner instead |
| Named Entity Recognition | MitMovie | heuristic | incompatible: backend 'heuristic' doesn't support dataset entity types: ["Actor", "Director", "Genre", "Title", "Year", "Song", "Character", "Plot", "Rating"] |
| Named Entity Recognition | MitMovie | deberta_v3 | Retrieval error: DeBERTa-v3 model unavailable: Retrieval error: Failed to download model.onnx: request error: https://huggingface.co/microsoft/deberta-v3-base/resolve/main/onnx/model.onnx: status code 404  Options: 1. Export your own: uv run scripts/export_deberta_ner_to_onnx.py 2. Set DEBERTA_MODEL_PATH to a local model directory 3. Use --model bert-onnx or --model candle-ner instead |
| Named Entity Recognition | MitMovie | albert | Retrieval error: ALBERT model unavailable: Retrieval error: Failed to download model.onnx: request error: https://huggingface.co/albert-base-v2/resolve/main/onnx/model.onnx: status code 404  Options: 1. Export your own ONNX model 2. Set ALBERT_MODEL_PATH to a local model directory 3. Use --model bert-onnx or --model candle-ner instead |
| Named Entity Recognition | MitRestaurant | heuristic | incompatible: backend 'heuristic' doesn't support dataset entity types: ["Amenity", "Cuisine", "Dish", "Hours", "Location", "Price", "Rating", "Restaurant_Name"] |
| Named Entity Recognition | MitRestaurant | deberta_v3 | Retrieval error: DeBERTa-v3 model unavailable: Retrieval error: Failed to download model.onnx: request error: https://huggingface.co/microsoft/deberta-v3-base/resolve/main/onnx/model.onnx: status code 404  Options: 1. Export your own: uv run scripts/export_deberta_ner_to_onnx.py 2. Set DEBERTA_MODEL_PATH to a local model directory 3. Use --model bert-onnx or --model candle-ner instead |
| Named Entity Recognition | MitRestaurant | albert | Retrieval error: ALBERT model unavailable: Retrieval error: Failed to download model.onnx: request error: https://huggingface.co/albert-base-v2/resolve/main/onnx/model.onnx: status code 404  Options: 1. Export your own ONNX model 2. Set ALBERT_MODEL_PATH to a local model directory 3. Use --model bert-onnx or --model candle-ner instead |
| Named Entity Recognition | MultiNERD | heuristic | incompatible: backend 'heuristic' doesn't support dataset entity types: ["PER", "LOC", "ORG", "ANIM", "BIO", "CEL", "DIS", "EVE", "FOOD", "INST", "MEDIA", "MYTH", "PLANT", "TIME", "VEHI"] |
| Named Entity Recognition | MultiNERD | deberta_v3 | Retrieval error: DeBERTa-v3 model unavailable: Retrieval error: Failed to download model.onnx: request error: https://huggingface.co/microsoft/deberta-v3-base/resolve/main/onnx/model.onnx: status code 404  Options: 1. Export your own: uv run scripts/export_deberta_ner_to_onnx.py 2. Set DEBERTA_MODEL_PATH to a local model directory 3. Use --model bert-onnx or --model candle-ner instead |
| Named Entity Recognition | MultiNERD | albert | Retrieval error: ALBERT model unavailable: Retrieval error: Failed to download model.onnx: request error: https://huggingface.co/albert-base-v2/resolve/main/onnx/model.onnx: status code 404  Options: 1. Export your own ONNX model 2. Set ALBERT_MODEL_PATH to a local model directory 3. Use --model bert-onnx or --model candle-ner instead |
| Named Entity Recognition | BC5CDR | heuristic | incompatible: backend 'heuristic' doesn't support dataset entity types: ["Chemical", "Disease"] |
| Named Entity Recognition | BC5CDR | deberta_v3 | Retrieval error: DeBERTa-v3 model unavailable: Retrieval error: Failed to download model.onnx: request error: https://huggingface.co/microsoft/deberta-v3-base/resolve/main/onnx/model.onnx: status code 404  Options: 1. Export your own: uv run scripts/export_deberta_ner_to_onnx.py 2. Set DEBERTA_MODEL_PATH to a local model directory 3. Use --model bert-onnx or --model candle-ner instead |
| Named Entity Recognition | BC5CDR | albert | Retrieval error: ALBERT model unavailable: Retrieval error: Failed to download model.onnx: request error: https://huggingface.co/albert-base-v2/resolve/main/onnx/model.onnx: status code 404  Options: 1. Export your own ONNX model 2. Set ALBERT_MODEL_PATH to a local model directory 3. Use --model bert-onnx or --model candle-ner instead |
| Named Entity Recognition | NCBIDisease | heuristic | incompatible: backend 'heuristic' doesn't support dataset entity types: ["Disease"] |
| Named Entity Recognition | NCBIDisease | deberta_v3 | Retrieval error: DeBERTa-v3 model unavailable: Retrieval error: Failed to download model.onnx: request error: https://huggingface.co/microsoft/deberta-v3-base/resolve/main/onnx/model.onnx: status code 404  Options: 1. Export your own: uv run scripts/export_deberta_ner_to_onnx.py 2. Set DEBERTA_MODEL_PATH to a local model directory 3. Use --model bert-onnx or --model candle-ner instead |
| Named Entity Recognition | NCBIDisease | albert | Retrieval error: ALBERT model unavailable: Retrieval error: Failed to download model.onnx: request error: https://huggingface.co/albert-base-v2/resolve/main/onnx/model.onnx: status code 404  Options: 1. Export your own ONNX model 2. Set ALBERT_MODEL_PATH to a local model directory 3. Use --model bert-onnx or --model candle-ner instead |
| Named Entity Recognition | FewNERD | heuristic | incompatible: backend 'heuristic' doesn't support dataset entity types: ["person", "location", "organization", "building", "art", "product", "event", "other"] |
| Named Entity Recognition | FewNERD | deberta_v3 | Retrieval error: DeBERTa-v3 model unavailable: Retrieval error: Failed to download model.onnx: request error: https://huggingface.co/microsoft/deberta-v3-base/resolve/main/onnx/model.onnx: status code 404  Options: 1. Export your own: uv run scripts/export_deberta_ner_to_onnx.py 2. Set DEBERTA_MODEL_PATH to a local model directory 3. Use --model bert-onnx or --model candle-ner instead |
| Named Entity Recognition | FewNERD | albert | Retrieval error: ALBERT model unavailable: Retrieval error: Failed to download model.onnx: request error: https://huggingface.co/albert-base-v2/resolve/main/onnx/model.onnx: status code 404  Options: 1. Export your own ONNX model 2. Set ALBERT_MODEL_PATH to a local model directory 3. Use --model bert-onnx or --model candle-ner instead |

## Error Patterns

- [8x] Retrieval error: DeBERTa-v3 model unavailable: Ret...
- [8x] Retrieval error: ALBERT model unavailable: Retriev...
- [7x] incompatible: backend 'heuristic' doesn't support ...

## Results

**Note**: 40 combinations skipped (features not enabled or incompatible). Showing successful and failed results only.

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
| GAP | mention_ranking | 34.6 | 0.0 | 80.3 | 20 | 8131 |

#### Chain-Length Stratification

| Chain Type | Count | F1 |
|------------|-------|----|
| Long (>10) | 0 | 0.00 |
| Short (2-10) | 18 | 100.00 |
| Singleton (1) | 24 | 100.00 |

| GAP | coref_resolver | 32.3 | 0.0 | 73.2 | 20 | 8598 |

#### Chain-Length Stratification

| Chain Type | Count | F1 |
|------------|-------|----|
| Long (>10) | 0 | 0.00 |
| Short (2-10) | 18 | 80.00 |
| Singleton (1) | 24 | 100.00 |


### Named Entity Recognition

| Dataset | Backend | F1 | P | R | N | ms |
|---------|---------|----|----|----|---|----|
| MultiNERD | bert_onnx | 80.0 | 84.6 | 75.9 | 20 | 475 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ANIM | 0.00 | [0.00, 0.00] | 1 |
| LOC | 1.00 | [1.00, 1.00] | 8 |
| MEDIA | 0.00 | [0.00, 0.00] | 1 |
| MISC | 0.00 | [0.00, 0.00] | 1 |
| ORG | 1.00 | [1.00, 1.00] | 4 |
| PER | 1.00 | [1.00, 1.00] | 7 |
| PLANT | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.67, 0.99], P: [0.67, 0.99], R: [0.67, 0.99]

| WikiGold | bert_onnx | 75.2 | 72.1 | 78.6 | 20 | 536 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.73 | [0.48, 0.99] | 12 |
| MISC | 0.41 | [0.08, 0.73] | 9 |
| ORG | 0.66 | [0.39, 0.93] | 9 |
| PER | 0.73 | [0.44, 1.00] | 9 |


**Confidence Intervals (95%)**: F1: [0.41, 0.76], P: [0.39, 0.74], R: [0.44, 0.79]

| WikiGold | nuner | 70.3 | 70.9 | 69.6 | 20 | 4499 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.72 | [0.47, 0.96] | 12 |
| MISC | 0.00 | [0.00, 0.00] | 6 |
| ORG | 0.53 | [0.27, 0.78] | 14 |
| PER | 0.74 | [0.46, 1.00] | 9 |


**Confidence Intervals (95%)**: F1: [0.42, 0.77], P: [0.44, 0.80], R: [0.42, 0.77]

| MultiNERD | gliner_onnx | 53.1 | 48.6 | 58.6 | 20 | 752 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ANIM | 0.00 | [0.00, 0.00] | 1 |
| DATE | 0.00 | [0.00, 0.00] | 1 |
| LOC | 0.57 | [0.26, 0.88] | 10 |
| MEDIA | 0.00 | [0.00, 0.00] | 1 |
| ORG | 0.40 | [0.00, 0.88] | 5 |
| PER | 0.40 | [0.08, 0.72] | 10 |
| PLANT | 0.33 | [0.33, 0.33] | 1 |


**Confidence Intervals (95%)**: F1: [0.30, 0.69], P: [0.27, 0.66], R: [0.36, 0.78]

| MultiNERD | nuner | 50.0 | 43.6 | 58.6 | 20 | 4300 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ANIM | 0.00 | [0.00, 0.00] | 1 |
| LOC | 0.50 | [0.20, 0.80] | 12 |
| MEDIA | 0.00 | [0.00, 0.00] | 1 |
| ORG | 0.28 | [0.00, 0.63] | 6 |
| PER | 0.50 | [0.17, 0.83] | 10 |
| PLANT | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.37, 0.77], P: [0.35, 0.76], R: [0.39, 0.81]

| WikiGold | stacked | 49.6 | 44.9 | 55.4 | 20 | 417 |

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

| MultiNERD | stacked | 48.6 | 41.5 | 58.6 | 20 | 374 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ANIM | 0.00 | [0.00, 0.00] | 1 |
| DATE | 0.00 | [0.00, 0.00] | 5 |
| EVENT | 0.00 | [0.00, 0.00] | 1 |
| LOC | 0.60 | [0.32, 0.88] | 10 |
| MEDIA | 0.00 | [0.00, 0.00] | 1 |
| ORG | 0.50 | [0.00, 1.00] | 4 |
| PER | 0.40 | [0.12, 0.68] | 12 |
| PLANT | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.30, 0.67], P: [0.26, 0.62], R: [0.36, 0.78]

| WikiGold | gliner_onnx | 46.9 | 54.8 | 41.1 | 20 | 729 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.66 | [0.40, 0.93] | 11 |
| MISC | 0.00 | [0.00, 0.00] | 6 |
| ORG | 0.48 | [0.22, 0.74] | 13 |
| PER | 0.61 | [0.29, 0.93] | 9 |


**Confidence Intervals (95%)**: F1: [0.34, 0.68], P: [0.42, 0.80], R: [0.30, 0.63]

| WikiGold | gliner2 | 46.3 | 48.1 | 44.6 | 20 | 3793 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.59 | [0.33, 0.85] | 12 |
| MISC | 0.12 | [0.00, 0.37] | 8 |
| ORG | 0.51 | [0.24, 0.79] | 12 |
| PER | 0.73 | [0.42, 1.00] | 8 |


**Confidence Intervals (95%)**: F1: [0.36, 0.71], P: [0.39, 0.75], R: [0.35, 0.69]

| MultiNERD | gliner2 | 41.9 | 31.6 | 62.1 | 20 | 3963 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ANIM | 0.00 | [0.00, 0.00] | 1 |
| ANIMAL | 0.00 | [0.00, 0.00] | 1 |
| CREATIVE_WORK | 0.00 | [0.00, 0.00] | 3 |
| DISEASE | 0.00 | [0.00, 0.00] | 1 |
| LOC | 0.57 | [0.29, 0.86] | 11 |
| MEDIA | 0.00 | [0.00, 0.00] | 1 |
| MYTH | 0.00 | [0.00, 0.00] | 1 |
| ORG | 0.25 | [0.00, 0.74] | 4 |
| PER | 0.67 | [0.32, 1.00] | 7 |
| PLANT | 0.33 | [0.33, 0.33] | 1 |
| TIME | 0.00 | [0.00, 0.00] | 8 |
| VEHICLE | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.24, 0.60], P: [0.19, 0.54], R: [0.35, 0.77]

| FewNERD | stacked | 37.3 | 31.8 | 45.2 | 20 | 383 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| BUILDING | 0.00 | [0.00, 0.00] | 1 |
| CREATIVE_WORK | 0.00 | [0.00, 0.00] | 2 |
| DATE | 0.00 | [0.00, 0.00] | 4 |
| EVENT | 0.00 | [0.00, 0.00] | 3 |
| LOC | 0.40 | [0.00, 0.88] | 5 |
| MISC | 0.00 | [0.00, 0.00] | 3 |
| ORG | 0.40 | [0.00, 0.88] | 5 |
| PER | 0.28 | [0.04, 0.53] | 13 |
| PRODUCT | 0.38 | [0.00, 0.83] | 5 |


**Confidence Intervals (95%)**: F1: [0.20, 0.58], P: [0.18, 0.55], R: [0.23, 0.65]

| FewNERD | gliner2 | 36.6 | 32.5 | 41.9 | 20 | 4489 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| BUILDING | 0.00 | [0.00, 0.00] | 1 |
| CREATIVE_WORK | 0.50 | [0.00, 1.00] | 2 |
| EVENT | 0.00 | [0.00, 0.00] | 4 |
| LOC | 0.00 | [0.00, 0.00] | 3 |
| MISC | 0.29 | [0.00, 0.63] | 4 |
| ORG | 0.33 | [0.00, 0.99] | 3 |
| PER | 1.00 | [1.00, 1.00] | 4 |
| PRODUCT | 0.16 | [0.00, 0.47] | 5 |


**Confidence Intervals (95%)**: F1: [0.19, 0.60], P: [0.19, 0.59], R: [0.20, 0.63]

| FewNERD | bert_onnx | 31.0 | 27.5 | 35.5 | 20 | 397 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| BUILDING | 0.00 | [0.00, 0.00] | 1 |
| CREATIVE_WORK | 0.00 | [0.00, 0.00] | 2 |
| EVENT | 0.00 | [0.00, 0.00] | 3 |
| LOC | 0.67 | [0.01, 1.00] | 3 |
| MISC | 0.08 | [0.00, 0.23] | 13 |
| ORG | 0.50 | [0.00, 1.00] | 4 |
| PER | 0.57 | [0.18, 0.97] | 7 |
| PRODUCT | 0.00 | [0.00, 0.00] | 5 |


**Confidence Intervals (95%)**: F1: [0.18, 0.60], P: [0.18, 0.59], R: [0.19, 0.61]

| WikiGold | heuristic | 29.3 | 28.3 | 30.4 | 20 | 0 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.38 | [0.14, 0.63] | 13 |
| MISC | 0.00 | [0.00, 0.00] | 6 |
| ORG | 0.10 | [0.00, 0.23] | 9 |
| PER | 0.15 | [0.00, 0.30] | 17 |


**Confidence Intervals (95%)**: F1: [0.07, 0.34], P: [0.07, 0.33], R: [0.08, 0.38]

| MultiNERD | tplinker | 29.2 | 36.8 | 24.1 | 20 | 2 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ANIM | 0.00 | [0.00, 0.00] | 1 |
| LOC | 0.41 | [0.08, 0.73] | 9 |
| MEDIA | 0.00 | [0.00, 0.00] | 1 |
| ORG | 0.33 | [0.00, 0.75] | 5 |
| PER | 0.30 | [0.00, 0.60] | 10 |
| PLANT | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.18, 0.58], P: [0.21, 0.64], R: [0.17, 0.57]

| Wnut17 | gliner2 | 26.4 | 21.2 | 35.0 | 20 | 3911 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| CREATIVE_WORK | 0.00 | [0.00, 0.00] | 1 |
| GROUP | 0.33 | [0.00, 0.75] | 6 |
| LOC | 0.33 | [0.00, 0.99] | 2 |
| ORG | 0.13 | [0.00, 0.39] | 5 |
| PER | 0.15 | [0.00, 0.35] | 7 |
| PRODUCT | 0.00 | [0.00, 0.00] | 5 |


**Confidence Intervals (95%)**: F1: [0.02, 0.27], P: [0.01, 0.26], R: [0.02, 0.32]

| NCBIDisease | gliner2 | 24.2 | 26.7 | 22.2 | 20 | 3853 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| DISEASE | 0.28 | [0.04, 0.53] | 13 |


**Confidence Intervals (95%)**: F1: [0.02, 0.35], P: [0.02, 0.38], R: [0.01, 0.34]

| WikiGold | tplinker | 19.5 | 30.8 | 14.3 | 20 | 2 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.38 | [0.14, 0.63] | 13 |
| MISC | 0.00 | [0.00, 0.00] | 6 |
| ORG | 0.10 | [0.00, 0.23] | 9 |
| PER | 0.10 | [0.00, 0.26] | 13 |


**Confidence Intervals (95%)**: F1: [0.08, 0.35], P: [0.13, 0.49], R: [0.05, 0.30]

| FewNERD | gliner_onnx | 19.2 | 23.8 | 16.1 | 20 | 875 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| BUILDING | 1.00 | [1.00, 1.00] | 1 |
| CREATIVE_WORK | 0.00 | [0.00, 0.00] | 2 |
| EVENT | 0.00 | [0.00, 0.00] | 3 |
| LOC | 0.67 | [0.01, 1.00] | 3 |
| MISC | 0.00 | [0.00, 0.00] | 3 |
| ORG | 0.67 | [0.01, 1.00] | 3 |
| PER | 0.67 | [0.25, 1.00] | 6 |
| PRODUCT | 0.38 | [0.00, 0.83] | 5 |


**Confidence Intervals (95%)**: F1: [0.30, 0.72], P: [0.33, 0.77], R: [0.28, 0.70]

| Wnut17 | bert_onnx | 18.2 | 16.7 | 20.0 | 20 | 378 |

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

| MitMovie | gliner2 | 18.2 | 19.0 | 17.4 | 20 | 3800 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ACTOR | 0.00 | [0.00, 0.00] | 6 |
| CHARACTER | 0.00 | [0.00, 0.00] | 1 |
| CREATIVE_WORK | 0.00 | [0.00, 0.00] | 5 |
| DATE | 0.00 | [0.00, 0.00] | 5 |
| DIRECTOR | 0.00 | [0.00, 0.00] | 1 |
| GENRE | 0.58 | [0.29, 0.87] | 12 |
| PER | 0.00 | [0.00, 0.00] | 9 |
| PLOT | 0.33 | [0.00, 0.75] | 6 |
| RATING | 0.14 | [0.00, 0.42] | 7 |
| RATINGS_AVERAGE | 0.00 | [0.00, 0.00] | 7 |
| REVIEW | 0.00 | [0.00, 0.00] | 1 |
| SONG | 0.00 | [0.00, 0.00] | 1 |
| TITLE | 0.00 | [0.00, 0.00] | 4 |
| YEAR | 0.00 | [0.00, 0.00] | 5 |


**Confidence Intervals (95%)**: F1: [0.08, 0.33], P: [0.09, 0.38], R: [0.08, 0.31]

| FewNERD | nuner | 16.3 | 22.2 | 12.9 | 20 | 4455 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| BUILDING | 0.00 | [0.00, 0.00] | 1 |
| CREATIVE_WORK | 0.00 | [0.00, 0.00] | 2 |
| EVENT | 0.00 | [0.00, 0.00] | 3 |
| LOC | 0.40 | [0.00, 0.88] | 5 |
| MISC | 0.00 | [0.00, 0.00] | 3 |
| ORG | 0.29 | [0.00, 0.65] | 7 |
| PER | 0.80 | [0.41, 1.00] | 5 |
| PRODUCT | 0.00 | [0.00, 0.00] | 5 |


**Confidence Intervals (95%)**: F1: [0.14, 0.51], P: [0.13, 0.49], R: [0.15, 0.55]

| Wnut17 | nuner | 15.7 | 12.9 | 20.0 | 20 | 4146 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| CREATIVE_WORK | 0.00 | [0.00, 0.00] | 1 |
| GROUP | 0.00 | [0.00, 0.00] | 4 |
| LOC | 0.00 | [0.00, 0.00] | 2 |
| ORG | 0.08 | [0.00, 0.23] | 13 |
| PER | 0.18 | [0.00, 0.37] | 10 |
| PRODUCT | 0.00 | [0.00, 0.00] | 2 |


**Confidence Intervals (95%)**: F1: [0.00, 0.19], P: [0.00, 0.18], R: [0.00, 0.24]

| Wnut17 | gliner_onnx | 11.1 | 12.5 | 10.0 | 20 | 739 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| CREATIVE_WORK | 0.00 | [0.00, 0.00] | 1 |
| GROUP | 0.20 | [0.00, 0.59] | 5 |
| LOC | 0.00 | [0.00, 0.00] | 1 |
| ORG | 0.20 | [0.00, 0.59] | 5 |
| PER | 0.22 | [0.00, 0.50] | 6 |
| PRODUCT | 0.00 | [0.00, 0.00] | 2 |


**Confidence Intervals (95%)**: F1: [0.00, 0.21], P: [0.00, 0.26], R: [0.00, 0.23]

| MitRestaurant | gliner2 | 9.5 | 10.7 | 8.6 | 20 | 3800 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| AMENITY | 0.33 | [0.00, 0.75] | 6 |
| CUISINE | 0.00 | [0.00, 0.00] | 2 |
| DISH | 0.00 | [0.00, 0.00] | 4 |
| HOURS | 0.00 | [0.00, 0.00] | 3 |
| LOC | 0.13 | [0.00, 0.31] | 15 |
| MONEY | 0.00 | [0.00, 0.00] | 1 |
| ORG | 0.00 | [0.00, 0.00] | 4 |
| PRICE | 0.00 | [0.00, 0.00] | 1 |
| RATING | 1.00 | [1.00, 1.00] | 1 |
| RESTAURANT_NAME | 0.00 | [0.00, 0.00] | 8 |
| TIME | 0.00 | [0.00, 0.00] | 4 |


**Confidence Intervals (95%)**: F1: [0.03, 0.30], P: [0.03, 0.32], R: [0.02, 0.31]

| NCBIDisease | gliner_onnx | 9.1 | 25.0 | 5.6 | 20 | 821 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| DISEASE | 0.22 | [0.00, 0.45] | 12 |


**Confidence Intervals (95%)**: F1: [0.00, 0.28], P: [0.00, 0.31], R: [0.00, 0.26]

| Wnut17 | tplinker | 6.5 | 9.1 | 5.0 | 20 | 2 |

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

| Wnut17 | stacked | 6.3 | 4.5 | 10.0 | 20 | 363 |

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

| FewNERD | tplinker | 4.7 | 8.3 | 3.2 | 20 | 2 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| BUILDING | 0.00 | [0.00, 0.00] | 1 |
| CREATIVE_WORK | 0.00 | [0.00, 0.00] | 2 |
| EVENT | 0.00 | [0.00, 0.00] | 3 |
| LOC | 0.67 | [0.01, 1.00] | 3 |
| MISC | 0.00 | [0.00, 0.00] | 3 |
| ORG | 0.20 | [0.00, 0.59] | 5 |
| PER | 0.00 | [0.00, 0.00] | 8 |
| PRODUCT | 0.00 | [0.00, 0.00] | 5 |


**Confidence Intervals (95%)**: F1: [0.00, 0.24], P: [0.00, 0.31], R: [0.00, 0.21]

| MitMovie | gliner_onnx | 3.6 | 11.1 | 2.2 | 20 | 729 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ACTOR | 0.00 | [0.00, 0.00] | 6 |
| CHARACTER | 0.00 | [0.00, 0.00] | 1 |
| DIRECTOR | 0.00 | [0.00, 0.00] | 1 |
| GENRE | 0.17 | [0.00, 0.39] | 12 |
| PER | 0.00 | [0.00, 0.00] | 3 |
| PLOT | 0.00 | [0.00, 0.00] | 6 |
| RATING | 0.00 | [0.00, 0.00] | 3 |
| RATINGS_AVERAGE | 0.00 | [0.00, 0.00] | 7 |
| REVIEW | 0.00 | [0.00, 0.00] | 1 |
| SONG | 0.00 | [0.00, 0.00] | 1 |
| TITLE | 0.00 | [0.00, 0.00] | 4 |
| YEAR | 0.60 | [0.12, 1.00] | 5 |


**Confidence Intervals (95%)**: F1: [0.02, 0.18], P: [0.03, 0.32], R: [0.01, 0.13]

| WikiGold | crf | 0.0 | 0.0 | 0.0 | 20 | 0 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| LOC | 0.05 | [0.00, 0.13] | 11 |
| MISC | 0.00 | [0.00, 0.00] | 6 |
| ORG | 0.00 | [0.00, 0.00] | 8 |
| PER | 0.00 | [0.00, 0.00] | 8 |


**Confidence Intervals (95%)**: F1: [0.00, 0.07], P: [0.00, 0.15], R: [0.00, 0.05]

| Wnut17 | crf | 0.0 | 0.0 | 0.0 | 20 | 0 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| CREATIVE_WORK | 0.00 | [0.00, 0.00] | 1 |
| GROUP | 0.00 | [0.00, 0.00] | 4 |
| LOC | 0.00 | [0.00, 0.00] | 1 |
| ORG | 0.00 | [0.00, 0.00] | 2 |
| PER | 0.00 | [0.00, 0.00] | 4 |
| PRODUCT | 0.00 | [0.00, 0.00] | 2 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| MitMovie | stacked | 0.0 | 0.0 | 0.0 | 20 | 453 | (stacked: incompatible types)

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ACTOR | 0.00 | [0.00, 0.00] | 6 |
| CHARACTER | 0.00 | [0.00, 0.00] | 1 |
| DATE | 0.00 | [0.00, 0.00] | 1 |
| DIRECTOR | 0.00 | [0.00, 0.00] | 1 |
| GENRE | 0.00 | [0.00, 0.00] | 12 |
| ORG | 0.00 | [0.00, 0.00] | 1 |
| PER | 0.00 | [0.00, 0.00] | 11 |
| PLOT | 0.00 | [0.00, 0.00] | 6 |
| RATING | 0.00 | [0.00, 0.00] | 2 |
| RATINGS_AVERAGE | 0.00 | [0.00, 0.00] | 7 |
| REVIEW | 0.00 | [0.00, 0.00] | 1 |
| SONG | 0.00 | [0.00, 0.00] | 1 |
| TITLE | 0.00 | [0.00, 0.00] | 4 |
| YEAR | 0.00 | [0.00, 0.00] | 5 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| MitMovie | crf | 0.0 | 0.0 | 0.0 | 20 | 0 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ACTOR | 0.00 | [0.00, 0.00] | 6 |
| CHARACTER | 0.00 | [0.00, 0.00] | 1 |
| DIRECTOR | 0.00 | [0.00, 0.00] | 1 |
| GENRE | 0.00 | [0.00, 0.00] | 12 |
| PLOT | 0.00 | [0.00, 0.00] | 6 |
| RATING | 0.00 | [0.00, 0.00] | 2 |
| RATINGS_AVERAGE | 0.00 | [0.00, 0.00] | 7 |
| REVIEW | 0.00 | [0.00, 0.00] | 1 |
| SONG | 0.00 | [0.00, 0.00] | 1 |
| TITLE | 0.00 | [0.00, 0.00] | 4 |
| YEAR | 0.00 | [0.00, 0.00] | 5 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| MitMovie | bert_onnx | 0.0 | 0.0 | 0.0 | 20 | 439 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ACTOR | 0.00 | [0.00, 0.00] | 6 |
| CHARACTER | 0.00 | [0.00, 0.00] | 1 |
| DIRECTOR | 0.00 | [0.00, 0.00] | 1 |
| GENRE | 0.00 | [0.00, 0.00] | 12 |
| PER | 0.00 | [0.00, 0.00] | 1 |
| PLOT | 0.00 | [0.00, 0.00] | 6 |
| RATING | 0.00 | [0.00, 0.00] | 2 |
| RATINGS_AVERAGE | 0.00 | [0.00, 0.00] | 7 |
| REVIEW | 0.00 | [0.00, 0.00] | 1 |
| SONG | 0.00 | [0.00, 0.00] | 1 |
| TITLE | 0.00 | [0.00, 0.00] | 4 |
| YEAR | 0.00 | [0.00, 0.00] | 5 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| MitMovie | nuner | 0.0 | 0.0 | 0.0 | 20 | 4596 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ACTOR | 0.00 | [0.00, 0.00] | 6 |
| CHARACTER | 0.00 | [0.00, 0.00] | 1 |
| DIRECTOR | 0.00 | [0.00, 0.00] | 1 |
| GENRE | 0.00 | [0.00, 0.00] | 12 |
| LOC | 0.00 | [0.00, 0.00] | 2 |
| ORG | 0.00 | [0.00, 0.00] | 4 |
| PER | 0.00 | [0.00, 0.00] | 12 |
| PLOT | 0.00 | [0.00, 0.00] | 6 |
| RATING | 0.00 | [0.00, 0.00] | 2 |
| RATINGS_AVERAGE | 0.00 | [0.00, 0.00] | 7 |
| REVIEW | 0.00 | [0.00, 0.00] | 1 |
| SONG | 0.00 | [0.00, 0.00] | 1 |
| TITLE | 0.00 | [0.00, 0.00] | 4 |
| YEAR | 0.00 | [0.00, 0.00] | 5 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| MitMovie | tplinker | 0.0 | 0.0 | 0.0 | 20 | 1 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ACTOR | 0.00 | [0.00, 0.00] | 6 |
| CHARACTER | 0.00 | [0.00, 0.00] | 1 |
| DIRECTOR | 0.00 | [0.00, 0.00] | 1 |
| GENRE | 0.00 | [0.00, 0.00] | 12 |
| PLOT | 0.00 | [0.00, 0.00] | 6 |
| RATING | 0.00 | [0.00, 0.00] | 2 |
| RATINGS_AVERAGE | 0.00 | [0.00, 0.00] | 7 |
| REVIEW | 0.00 | [0.00, 0.00] | 1 |
| SONG | 0.00 | [0.00, 0.00] | 1 |
| TITLE | 0.00 | [0.00, 0.00] | 4 |
| YEAR | 0.00 | [0.00, 0.00] | 5 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| MitRestaurant | stacked | 0.0 | 0.0 | 0.0 | 20 | 369 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| AMENITY | 0.00 | [0.00, 0.00] | 6 |
| CUISINE | 0.00 | [0.00, 0.00] | 2 |
| DISH | 0.00 | [0.00, 0.00] | 4 |
| HOURS | 0.00 | [0.00, 0.00] | 3 |
| LOC | 0.08 | [0.00, 0.23] | 13 |
| PER | 0.00 | [0.00, 0.00] | 3 |
| PRICE | 0.00 | [0.00, 0.00] | 1 |
| RATING | 0.00 | [0.00, 0.00] | 1 |
| RESTAURANT_NAME | 0.00 | [0.00, 0.00] | 8 |
| TIME | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.00, 0.15], P: [0.00, 0.15], R: [0.00, 0.15]

| MitRestaurant | crf | 0.0 | 0.0 | 0.0 | 20 | 0 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| AMENITY | 0.00 | [0.00, 0.00] | 6 |
| CUISINE | 0.00 | [0.00, 0.00] | 2 |
| DISH | 0.00 | [0.00, 0.00] | 4 |
| HOURS | 0.00 | [0.00, 0.00] | 3 |
| LOC | 0.00 | [0.00, 0.00] | 10 |
| PRICE | 0.00 | [0.00, 0.00] | 1 |
| RATING | 0.00 | [0.00, 0.00] | 1 |
| RESTAURANT_NAME | 0.00 | [0.00, 0.00] | 8 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| MitRestaurant | bert_onnx | 0.0 | 0.0 | 0.0 | 20 | 394 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| AMENITY | 0.00 | [0.00, 0.00] | 6 |
| CUISINE | 0.00 | [0.00, 0.00] | 2 |
| DISH | 0.00 | [0.00, 0.00] | 4 |
| HOURS | 0.00 | [0.00, 0.00] | 3 |
| LOC | 0.00 | [0.00, 0.00] | 10 |
| ORG | 0.00 | [0.00, 0.00] | 1 |
| PRICE | 0.00 | [0.00, 0.00] | 1 |
| RATING | 0.00 | [0.00, 0.00] | 1 |
| RESTAURANT_NAME | 0.00 | [0.00, 0.00] | 8 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| MitRestaurant | nuner | 0.0 | 0.0 | 0.0 | 20 | 4319 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| AMENITY | 0.00 | [0.00, 0.00] | 6 |
| CUISINE | 0.00 | [0.00, 0.00] | 2 |
| DISH | 0.00 | [0.00, 0.00] | 4 |
| HOURS | 0.00 | [0.00, 0.00] | 3 |
| LOC | 0.15 | [0.00, 0.36] | 13 |
| ORG | 0.00 | [0.00, 0.00] | 14 |
| PER | 0.00 | [0.00, 0.00] | 3 |
| PRICE | 0.00 | [0.00, 0.00] | 1 |
| RATING | 0.00 | [0.00, 0.00] | 1 |
| RESTAURANT_NAME | 0.00 | [0.00, 0.00] | 8 |


**Confidence Intervals (95%)**: F1: [0.00, 0.13], P: [0.00, 0.12], R: [0.00, 0.17]

| MitRestaurant | gliner_onnx | 0.0 | 0.0 | 0.0 | 20 | 732 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| AMENITY | 0.00 | [0.00, 0.00] | 6 |
| CUISINE | 0.00 | [0.00, 0.00] | 2 |
| DISH | 0.00 | [0.00, 0.00] | 4 |
| HOURS | 0.00 | [0.00, 0.00] | 3 |
| LOC | 0.08 | [0.00, 0.25] | 12 |
| PRICE | 0.00 | [0.00, 0.00] | 1 |
| RATING | 0.00 | [0.00, 0.00] | 1 |
| RESTAURANT_NAME | 0.00 | [0.00, 0.00] | 8 |


**Confidence Intervals (95%)**: F1: [0.00, 0.15], P: [0.00, 0.15], R: [0.00, 0.15]

| MitRestaurant | tplinker | 0.0 | 0.0 | 0.0 | 20 | 2 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| AMENITY | 0.00 | [0.00, 0.00] | 6 |
| CUISINE | 0.00 | [0.00, 0.00] | 2 |
| DISH | 0.00 | [0.00, 0.00] | 4 |
| HOURS | 0.00 | [0.00, 0.00] | 3 |
| LOC | 0.00 | [0.00, 0.00] | 10 |
| PRICE | 0.00 | [0.00, 0.00] | 1 |
| RATING | 0.00 | [0.00, 0.00] | 1 |
| RESTAURANT_NAME | 0.00 | [0.00, 0.00] | 8 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| MultiNERD | crf | 0.0 | 0.0 | 0.0 | 20 | 0 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ANIM | 0.00 | [0.00, 0.00] | 1 |
| LOC | 0.12 | [0.00, 0.37] | 8 |
| MEDIA | 0.00 | [0.00, 0.00] | 1 |
| ORG | 0.00 | [0.00, 0.00] | 4 |
| PER | 0.14 | [0.00, 0.42] | 7 |
| PLANT | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.00, 0.17], P: [0.00, 0.23], R: [0.00, 0.16]

| BC5CDR | stacked | 0.0 | 0.0 | 0.0 | 20 | 454 | (stacked: biomedical not supported)

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ENTITY | 0.00 | [0.00, 0.00] | 20 |
| PER | 0.00 | [0.00, 0.00] | 10 |
| PERCENT | 0.00 | [0.00, 0.00] | 1 |
| PRODUCT | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| BC5CDR | crf | 0.0 | 0.0 | 0.0 | 20 | 0 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ENTITY | 0.00 | [0.00, 0.00] | 20 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| BC5CDR | bert_onnx | 0.0 | 0.0 | 0.0 | 20 | 402 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ENTITY | 0.00 | [0.00, 0.00] | 20 |
| MISC | 0.00 | [0.00, 0.00] | 6 |
| ORG | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| BC5CDR | nuner | 0.0 | 0.0 | 0.0 | 20 | 4451 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ENTITY | 0.00 | [0.00, 0.00] | 20 |
| ORG | 0.00 | [0.00, 0.00] | 2 |
| PER | 0.00 | [0.00, 0.00] | 7 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| BC5CDR | gliner_onnx | 0.0 | 0.0 | 0.0 | 20 | 742 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| CHEMICAL | 0.00 | [0.00, 0.00] | 8 |
| DISEASE | 0.00 | [0.00, 0.00] | 5 |
| ENTITY | 0.00 | [0.00, 0.00] | 20 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| BC5CDR | gliner2 | 0.0 | 0.0 | 0.0 | 20 | 3830 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| CHEMICAL | 0.00 | [0.00, 0.00] | 14 |
| DISEASE | 0.00 | [0.00, 0.00] | 14 |
| ENTITY | 0.00 | [0.00, 0.00] | 20 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| BC5CDR | tplinker | 0.0 | 0.0 | 0.0 | 20 | 1 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| ENTITY | 0.00 | [0.00, 0.00] | 20 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| NCBIDisease | stacked | 0.0 | 0.0 | 0.0 | 20 | 367 | (stacked: biomedical not supported)

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| DISEASE | 0.00 | [0.00, 0.00] | 12 |
| LOC | 0.00 | [0.00, 0.00] | 5 |
| PER | 0.00 | [0.00, 0.00] | 14 |
| PERCENT | 0.00 | [0.00, 0.00] | 2 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| NCBIDisease | crf | 0.0 | 0.0 | 0.0 | 20 | 0 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| DISEASE | 0.00 | [0.00, 0.00] | 12 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| NCBIDisease | bert_onnx | 0.0 | 0.0 | 0.0 | 20 | 379 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| DISEASE | 0.00 | [0.00, 0.00] | 12 |
| LOC | 0.00 | [0.00, 0.00] | 1 |
| MISC | 0.00 | [0.00, 0.00] | 11 |
| PER | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| NCBIDisease | nuner | 0.0 | 0.0 | 0.0 | 20 | 4479 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| DISEASE | 0.00 | [0.00, 0.00] | 12 |
| LOC | 0.00 | [0.00, 0.00] | 3 |
| ORG | 0.00 | [0.00, 0.00] | 7 |
| PER | 0.00 | [0.00, 0.00] | 4 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| NCBIDisease | tplinker | 0.0 | 0.0 | 0.0 | 20 | 2 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| DISEASE | 0.00 | [0.00, 0.00] | 12 |
| LOC | 0.00 | [0.00, 0.00] | 4 |
| PER | 0.00 | [0.00, 0.00] | 1 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| FewNERD | crf | 0.0 | 0.0 | 0.0 | 20 | 0 |

#### Stratified by Entity Type

| Type | F1 | CI 95% | N |
|------|----|--------|---|
| BUILDING | 0.00 | [0.00, 0.00] | 1 |
| CREATIVE_WORK | 0.00 | [0.00, 0.00] | 2 |
| EVENT | 0.00 | [0.00, 0.00] | 3 |
| LOC | 0.00 | [0.00, 0.00] | 2 |
| MISC | 0.00 | [0.00, 0.00] | 3 |
| ORG | 0.00 | [0.00, 0.00] | 3 |
| PER | 0.00 | [0.00, 0.00] | 4 |
| PRODUCT | 0.00 | [0.00, 0.00] | 5 |


**Confidence Intervals (95%)**: F1: [0.00, 0.00], P: [0.00, 0.00], R: [0.00, 0.00]

| WikiGold | deberta_v3 | ✗ | onnx-error | 498 |
| WikiGold | albert | ✗ | onnx-error | 1978 |
| Wnut17 | heuristic | ✗ | incompatible | - |
| Wnut17 | deberta_v3 | ✗ | onnx-error | 277 |
| Wnut17 | albert | ✗ | onnx-error | 230 |
| MitMovie | heuristic | ✗ | incompatible | - |
| MitMovie | deberta_v3 | ✗ | onnx-error | 243 |
| MitMovie | albert | ✗ | onnx-error | 236 |
| MitRestaurant | heuristic | ✗ | incompatible | - |
| MitRestaurant | deberta_v3 | ✗ | onnx-error | 249 |
| MitRestaurant | albert | ✗ | onnx-error | 236 |
| MultiNERD | heuristic | ✗ | incompatible | - |
| MultiNERD | deberta_v3 | ✗ | onnx-error | 233 |
| MultiNERD | albert | ✗ | onnx-error | 227 |
| BC5CDR | heuristic | ✗ | incompatible | - |
| BC5CDR | deberta_v3 | ✗ | onnx-error | 237 |
| BC5CDR | albert | ✗ | onnx-error | 244 |
| NCBIDisease | heuristic | ✗ | incompatible | - |
| NCBIDisease | deberta_v3 | ✗ | onnx-error | 232 |
| NCBIDisease | albert | ✗ | onnx-error | 288 |
| FewNERD | heuristic | ✗ | incompatible | - |
| FewNERD | deberta_v3 | ✗ | onnx-error | 246 |
| FewNERD | albert | ✗ | onnx-error | 240 |

## Backend Summary

| Backend | ✓ | ⊘ | ✗ | Avg F1 |
|---------|---|---|---|--------|
| tplinker | 8 | 0 | 0 | 7.5 |
| gliner_onnx | 8 | 0 | 0 | 17.9 |
| nuner | 8 | 0 | 0 | 19.0 |
| gliner2 | 8 | 0 | 0 | 25.4 |
| stacked | 8 | 0 | 0 | 17.7 |
| bert_onnx | 8 | 0 | 0 | 25.5 |
| crf | 8 | 0 | 0 | 0.0 |
| mention_ranking | 1 | 0 | 0 | 34.6 |
| heuristic | 1 | 0 | 7 | 29.3 |
| coref_resolver | 1 | 0 | 0 | 32.3 |
| w2ner | 0 | 8 | 0 | 0.0 |
| deberta_v3 | 0 | 0 | 8 | 0.0 |
| gliner_candle | 0 | 8 | 0 | 0.0 |
| gliner_poly | 0 | 8 | 0 | 0.0 |
| candle_ner | 0 | 8 | 0 | 0.0 |
| albert | 0 | 0 | 8 | 0.0 |
| universal_ner | 0 | 8 | 0 | 0.0 |

