# Missing Backends and Datasets Analysis

Based on MCP research and codebase review, this document identifies backends and datasets that are missing but coherent with `anno`'s purpose as a comprehensive NER, relation extraction, and coreference resolution library.

## Implementation Status (2025-01-XX)

### ✅ Completed
- **All 7 missing datasets added**: SciER, UNER, MSNER, BioMNER, LegNER, MixRED, CovEReD
- **All 5 missing backends implemented**:
  - **TPLinker**: Placeholder implementation with handshaking matrix support
  - **Poly-Encoder GLiNER**: Placeholder wrapping bi-encoder GLiNER (fusion layer pending)
  - **DeBERTa-v3 NER**: Wraps BERT ONNX backend with DeBERTa-v3 models
  - **ALBERT NER**: Wraps BERT ONNX backend with ALBERT models
  - **UniversalNER**: Placeholder for LLM-based NER (LLM integration pending)

### 📋 Future Enhancements
- Full poly-encoder fusion layer implementation (when models available)
- LLM inference infrastructure for UniversalNER (llama.cpp, vLLM integration)
- Full ONNX model support for TPLinker (currently placeholder)

## Missing Backends/Models

### 1. UniversalNER (LLM-based Zero-Shot NER)
- **Status**: Not implemented
- **Why Missing**: LLM-based, requires different architecture
- **Research**: [UniversalNER](https://universal-ner.github.io) - Instruction-tuned LLaMA for open NER
- **Capabilities**: 45 entity types, competitive with ChatGPT
- **Pros**: Very flexible, supports many entity types
- **Cons**: Expensive inference (LLM-based), slower than transformer models
- **Implementation Priority**: Low (expensive, but could be useful for zero-shot scenarios)

### 2. TPLinker (Handshaking Matrix for Joint Entity-Relation)
- **Status**: Partially implemented (HandshakingMatrix trait exists in `inference.rs`)
- **Why Missing**: Full TPLinker backend not implemented
- **Research**: [TPLinker: Single-stage Joint Extraction](https://aclanthology.org/2020.coling-main.138/)
- **Capabilities**: Joint entity-relation extraction using handshaking tagging
- **Pros**: Single-stage extraction, efficient
- **Cons**: Requires specialized architecture
- **Implementation Priority**: Medium (handshaking infrastructure exists)

### 3. Poly-Encoder GLiNER (Advanced GLiNER Architecture)
- **Status**: Not implemented (only bi-encoder GLiNER exists)
- **Why Missing**: Newer architecture variant
- **Research**: [GLiNER Evolution](https://blog.knowledgator.com/meet-the-new-zero-shot-ner-architecture-30ffc2cb1ee0)
- **Capabilities**: Better inter-label understanding, improved performance on complex NER
- **Pros**: Better than bi-encoder for many entity types
- **Cons**: More complex, requires post-fusion step
- **Implementation Priority**: Medium (evolution of existing GLiNER)

### 4. DeBERTa-v3 NER Backend
- **Status**: Not implemented (only BERT-based backends exist)
- **Why Missing**: Different encoder architecture
- **Research**: DeBERTa-v3 with disentangled attention
- **Capabilities**: Better attention mechanism, improved performance
- **Pros**: State-of-the-art encoder, better than BERT
- **Cons**: Requires DeBERTa models
- **Implementation Priority**: Medium (could improve existing BERT backends)

### 5. ALBERT-based NER (Smaller, Efficient)
- **Status**: Not implemented
- **Why Missing**: Different model architecture
- **Research**: ALBERT achieves SOTA on biomedical method NER with only 11MB
- **Capabilities**: Efficient, good for domain-specific tasks
- **Pros**: Small model size, fast inference
- **Cons**: May not generalize as well as larger models
- **Implementation Priority**: Low-Medium (useful for specialized domains)

## Missing Datasets

### 1. SciER (Scientific Document Relation Extraction)
- **Status**: Not implemented
- **Why Missing**: Newer dataset (2024)
- **Source**: [SciER Dataset](https://arxiv.org/abs/2410.21155)
- **Size**: 106 full-text scientific publications, 24,000+ entities, 12,000+ relations
- **Entity Types**: Datasets, methods, tasks in scientific articles
- **Relation Types**: Fine-grained scientific relations
- **Format**: Full-text publications (not just abstracts)
- **Use Case**: Scientific information extraction, research paper analysis
- **Implementation Priority**: High (complements existing scientific datasets)

### 2. UNER (Universal NER - Multilingual)
- **Status**: Not implemented
- **Why Missing**: Newer multilingual framework (2024)
- **Source**: [Universal NER](https://arxiv.org/html/2311.09122v2)
- **Size**: 19 datasets across 13 languages
- **Languages**: English, Spanish, Dutch, Russian, Turkish, Korean, Farsi, German, Chinese, Hindi, Bangla, and more
- **Format**: Cross-lingually consistent annotations
- **Use Case**: Multilingual NER evaluation, cross-lingual transfer studies
- **Implementation Priority**: High (fills multilingual evaluation gap)

### 3. BioMNER (Biomedical Method Entity Recognition)
- **Status**: Not implemented
- **Why Missing**: Specialized biomedical dataset (2024)
- **Source**: [BioMNER Dataset](https://arxiv.org/abs/2406.20038)
- **Size**: Biomedical method entities
- **Entity Types**: Methodological concepts in biomedical literature
- **Use Case**: Biomedical method extraction, literature mining
- **Implementation Priority**: Medium (specialized, but complements existing biomedical datasets)

### 4. LegNER (Legal Domain NER)
- **Status**: Not implemented
- **Why Missing**: Domain-specific legal dataset
- **Source**: [LegNER Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12631292/)
- **Size**: 1,542 manually annotated court cases
- **Entity Types**: PERSON, ORGANIZATION, LAW, CASE_REFERENCE, etc.
- **Use Case**: Legal text processing, anonymization, e-discovery
- **Implementation Priority**: Medium (specialized domain, but useful)

### 5. MSNER (Multilingual Speech NER)
- **Status**: Not implemented
- **Why Missing**: Speech modality dataset (2024)
- **Source**: [MSNER Dataset](https://aclanthology.org/2024.isa-1.2/)
- **Size**: 590 hours silver-annotated + 17 hours manual evaluation
- **Languages**: Dutch, French, German, Spanish
- **Format**: Speech transcripts with NER annotations
- **Use Case**: Spoken language understanding, multilingual speech NER
- **Implementation Priority**: Low-Medium (different modality, but interesting)

### 6. MixRED (Mix-Lingual Relation Extraction)
- **Status**: Not implemented
- **Why Missing**: Code-mixed relation extraction dataset
- **Source**: [MixRED Dataset](https://aclanthology.org/2024.lrec-main.993/)
- **Size**: Human-annotated code-mixed relation extraction
- **Use Case**: Multilingual communities, code-switching scenarios
- **Implementation Priority**: Medium (addresses code-mixing gap)

### 7. CMNEE (Chinese Military News Event Extraction)
- **Status**: Not implemented
- **Why Missing**: Domain-specific event extraction dataset
- **Source**: [CMNEE Dataset](https://aclanthology.org/2024.lrec-main.299/)
- **Size**: 17,000 documents, 29,223 events
- **Event Types**: 8 event types, 11 argument role types
- **Use Case**: Military domain event extraction, Chinese NER
- **Implementation Priority**: Low-Medium (specialized, Chinese-specific)

### 8. CovEReD (Counterfactual RE Dataset)
- **Status**: Not implemented
- **Why Missing**: Counterfactual evaluation dataset (2024)
- **Source**: [CovEReD Dataset](https://arxiv.org/abs/2407.06699)
- **Size**: Counterfactual documents based on DocRED
- **Use Case**: Evaluating factual consistency, robustness testing
- **Implementation Priority**: Medium (important for robustness evaluation)

### 9. Ultra-Fine Entity Typing Dataset
- **Status**: Not implemented
- **Why Missing**: Fine-grained entity typing task
- **Source**: [Ultra-Fine Entity Typing](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html)
- **Size**: Crowdsourced, 429 types for 80% coverage
- **Use Case**: Fine-grained entity classification, entity typing research
- **Implementation Priority**: Low-Medium (different task, but related)

### 10. SLIMER Zero-Shot NER Benchmarks
- **Status**: Not implemented (but similar to existing zero-shot evaluation)
- **Why Missing**: Specific zero-shot evaluation framework
- **Source**: [SLIMER](https://arxiv.org/html/2407.01272v3)
- **Use Case**: Zero-shot NER evaluation with definitions/guidelines
- **Implementation Priority**: Low (evaluation framework, not dataset)

## Implementation Recommendations

### High Priority (Fill Critical Gaps)

1. **SciER Dataset** - Scientific relation extraction is important, complements existing scientific datasets
2. **UNER Dataset** - Multilingual evaluation is critical, fills major gap
3. **TPLinker Backend** - Handshaking infrastructure exists, would enable better joint extraction

### Medium Priority (Enhance Capabilities)

1. **Poly-Encoder GLiNER** - Evolution of existing GLiNER, better performance
2. **MixRED Dataset** - Addresses code-mixing scenarios
3. **LegNER Dataset** - Legal domain is important application area
4. **BioMNER Dataset** - Complements existing biomedical datasets
5. **CovEReD Dataset** - Important for robustness evaluation

### Low Priority (Nice to Have)

1. **UniversalNER Backend** - Expensive LLM-based, but flexible
2. **DeBERTa-v3 Backend** - Could improve existing BERT backends
3. **MSNER Dataset** - Different modality (speech)
4. **CMNEE Dataset** - Very specialized domain
5. **Ultra-Fine Entity Typing** - Different task focus

## Integration Notes

### For New Backends:
- Follow existing `Model` trait pattern
- Implement relevant capability traits (`ZeroShotNER`, `RelationExtractor`, etc.)
- Add to `BackendFactory` in `src/eval/backend_factory.rs`
- Update `BACKEND_CATALOG` in `src/backends/catalog.rs`
- Add tests following existing patterns

### For New Datasets:
- Add `DatasetId` enum variant in `src/eval/loader.rs`
- Implement download URL and parser
- Add to task mappings in `src/eval/task_mapping.rs`
- Update entity type mappings
- Add checksum verification for downloads
- Document in `docs/DATASET_DOWNLOADS.md`

## Research Alignment

All identified backends and datasets align with `anno`'s research focus on:
- **Zero-shot NER**: UniversalNER, Poly-Encoder GLiNER
- **Joint Extraction**: TPLinker, relation extraction datasets
- **Multilingual**: UNER, MixRED, MSNER
- **Domain-Specific**: LegNER, BioMNER, CMNEE, SciER
- **Robustness**: CovEReD, counterfactual evaluation
- **Efficiency**: ALBERT, DeBERTa-v3

These additions would significantly enhance `anno`'s coverage of state-of-the-art NER, relation extraction, and coreference resolution capabilities.

