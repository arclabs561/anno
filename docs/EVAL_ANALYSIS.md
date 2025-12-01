# Evaluation Results Analysis

## Missing Tasks & Datasets

### 1. DiscontinuousNER Task - NOT EVALUATED

**Problem**: CADEC dataset exists and is mapped to DiscontinuousNER, but no DiscontinuousNER section in report.

**Expected**: Should see a "Discontinuous NER" section with CADEC dataset evaluated.

**Root Cause**: Need to check if:
- CADEC is being loaded correctly
- DiscontinuousNER task is being evaluated
- Backend supports DiscontinuousNER (W2NER does, but it's skipped due to no onnx feature)

**Fix**: Verify CADEC loading and ensure DiscontinuousNER task is evaluated.

### 2. Relation Extraction - Missing 4/6 Datasets

**Problem**: Only 2 datasets shown (DocRED, ReTACRED), but 6 are mapped:
- ✅ DocRED (shown)
- ✅ ReTACRED (shown)
- ❌ NYTFB (missing)
- ❌ WEBNLG (missing)
- ❌ GoogleRE (missing)
- ❌ BioRED (missing)

**Root Cause**: These datasets likely:
- Not cached (require download)
- Download failing silently
- Parser not implemented

**Fix**: Check dataset loading/parsing for these 4 datasets.

### 3. Inter-document Coreference - No Datasets

**Status**: Expected - TODO in code says "Add inter-doc coref datasets when available"

**Current**: Task exists but no datasets mapped.

## Curious Scores & Anomalies

### 1. Wnut17: Very Low Precision (5.0 P, 42.9 R)

**Score**: 9.0 F1 | 5.0 P | 42.9 R | 20 examples

**Analysis**: 
- Precision 5.0% = many false positives (over-extracting)
- Recall 42.9% = missing many true entities
- F1 9.0% = poor overall performance

**Possible Causes**:
- Wnut17 is social media (noisy, challenging)
- Stacked backend (heuristic+pattern) struggles with informal text
- Entity type mismatches (Wnut17 has 6 types, stacked only knows PER/ORG/LOC)

**Expected**: Low scores on social media are normal, but precision should be higher if it's just extracting PER/ORG/LOC.

### 2. GAP Coreference: Huge Metric Discrepancy

**Score**: 35.0 CoNLL | 0.3 MUC | 75.2 B³ | 20 examples | 108149ms

**Analysis**:
- **MUC 0.3%**: Extremely low (link-based metric)
- **B³ 75.2%**: Very high (mention-based metric)
- **CoNLL 35.0%**: Average of MUC, B³, CEAF-e

**Interpretation**:
- MUC measures links between mentions (very strict)
- B³ measures mention-level clustering (more lenient)
- Discrepancy suggests: resolver is grouping mentions correctly (B³ high) but missing many links (MUC low)

**Possible Causes**:
- SimpleCorefResolver uses basic matching (exact/substring/pronoun)
- May be creating too many small clusters instead of linking them
- GAP is pronoun resolution (harder than full coreference)

**Expected**: For a simple resolver, B³ > MUC is normal, but 0.3% MUC is suspiciously low.

### 3. PreCo/LitBank Coreference: 0.0 with N=0 or N=1

**Scores**:
- PreCo: 0.0 | N=0 (no examples loaded)
- LitBank: 0.0 | N=1 (only 1 example loaded)

**Analysis**: Dataset loading issues, not evaluation issues.

**Possible Causes**:
- PreCo: `load_coref()` failing, falling back to entity conversion which produces empty docs
- LitBank: Only 1 sentence loaded (dataset may be small or parsing issue)

**Fix**: Check `load_coref()` implementation and dataset parsing.

### 4. Many 0.0 F1 Scores on Domain-Specific Datasets

**Datasets with 0.0 F1**:
- MIT Movie (actor, director, genre, title)
- MIT Restaurant (amenity, cuisine, dish)
- Biomedical (BC5CDR, NCBIDisease, GENIA, AnatEM, BC2GM, BC4CHEMD)
- Domain-specific (FabNER, UniversalNERBench, WikiNeural, UniversalNER)

**Analysis**: Expected - Stacked backend only knows PER/ORG/LOC, not domain-specific types.

**Status**: These should be marked as "incompatible entity types" rather than showing 0.0 F1.

## Task Coverage Summary

| Task | Datasets Mapped | Datasets Evaluated | Status |
|------|----------------|-------------------|--------|
| NER | 25 | 25 | ✅ Complete |
| DiscontinuousNER | 1 (CADEC) | 0 | ❌ Missing |
| Relation Extraction | 6 | 2 | ⚠️ Partial (4 missing) |
| IntraDocCoref | 3 | 3 | ✅ Complete (but 2 have loading issues) |
| InterDocCoref | 0 | 0 | ⚠️ No datasets (expected) |
| AbstractAnaphora | 3 | 3 | ✅ Complete (same as IntraDocCoref) |
| NED | 0 | 0 | ⚠️ No datasets (expected) |
| EventExtraction | 0 | 0 | ⚠️ No datasets (expected) |
| TextClassification | 0 | 0 | ⚠️ No datasets (expected) |
| HierarchicalExtraction | 0 | 0 | ⚠️ No datasets (expected) |

## Backend Coverage

| Backend | Tasks Supported | Evaluated | Status |
|---------|----------------|-----------|--------|
| stacked | NER | ✅ 25 datasets | Working |
| coref_resolver | IntraDocCoref, AbstractAnaphora | ✅ 6 combinations | Working (GAP good, PreCo/LitBank loading issues) |
| gliner2 | NER, RelationExtraction, TextClassification, HierarchicalExtraction | ⊘ Skipped (no onnx) | Expected |
| w2ner | NER, DiscontinuousNER | ⊘ Skipped (no onnx) | Expected |
| Other ML backends | NER | ⊘ Skipped (no onnx/candle) | Expected |

## Root Causes Identified

### 1. CADEC URL is 404
- **URL**: `https://huggingface.co/datasets/KevinSpaghetti/cadec/resolve/main/data/test.jsonl`
- **Status**: Returns 404
- **Fix**: Need to find correct HuggingFace dataset URL or use alternative source

### 2. PreCo Parser Issue
- **Problem**: `parse_preco_jsonl()` only extracts sentences, not coref info
- **Result**: N=0 because sentences have no entities/coref chains
- **Fix**: PreCo should use `load_coref()` which calls `parse_preco_json()`, not `parse_preco_jsonl()`

### 3. Relation Extraction Datasets
- **NYTFB, WEBNLG, GoogleRE, BioRED**: All use CrossRE GitHub URLs
- **Status**: May not be loading due to:
  - Not cached (require download)
  - Download failing silently
  - Parser expecting different format

### 4. LitBank Parser
- **Problem**: Only 1 sentence loaded
- **Possible causes**: 
  - Dataset is small (single document)
  - Parser only extracting first sentence
  - Format mismatch

## Recommendations

### Immediate Fixes

1. **Fix CADEC URL** - Find correct HuggingFace dataset URL or alternative source
2. **Fix PreCo coref loading** - Use `load_coref()` which calls `parse_preco_json()`, not `parse_preco_jsonl()`
3. **Fix relation extraction datasets** - Verify CrossRE URLs work, check parser format
4. **Mark 0.0 F1 as incompatible** - Don't show 0.0 F1 for domain-specific datasets, mark as "incompatible entity types"

### Investigation Needed

1. **GAP MUC score (0.3%)** - Why is link-based metric so low when B³ is 75%?
2. **Wnut17 precision (5.0%)** - Why is precision so low? Is it extracting wrong types?
3. **PreCo N=0** - Why is dataset loading producing 0 examples?
4. **LitBank N=1** - Is dataset really that small, or parsing issue?

### Missing Coverage

1. **DiscontinuousNER** - CADEC exists but not evaluated
2. **Relation Extraction** - 4/6 datasets missing
3. **InterDocCoref** - No datasets (expected, but should document)

## Performance Notes

- **GAP coref**: 108149ms (108s) for 20 examples - very slow, likely due to NER extraction + resolution
- **Most NER**: <2ms per dataset - very fast (stacked backend)
- **PreCo/LitBank**: <2ms but N=0/1 - fast but wrong (loading issue)

