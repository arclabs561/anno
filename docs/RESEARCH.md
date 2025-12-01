# Research Contributions and Implementations

This document clarifies what is novel in this codebase versus what implements existing research.

## Overview

This library primarily **implements** existing research papers and methods. The main contributions are:

1. **Architectural design** - Unified abstractions for NER, coreference, and evaluation
2. **Integration** - Combining multiple research findings into a cohesive evaluation framework
3. **Rust implementation** - Production-ready implementations of research methods

## Novel Contributions

### 1. Grounded Entity Representation (`src/grounded.rs`)

The **Signal → Track → Identity** hierarchy provides a unified abstraction for entity representation across modalities (text and vision).

- **Signal**: Detection-level entity mentions (where + what)
- **Track**: Document-level coreference chains (same entity within document)
- **Identity**: Knowledge base-level resolution (canonical entity across documents)

This architecture separates concerns that are often conflated in NER systems, enabling:
- Clear separation between detection, clustering, and linking
- Unified treatment of text and visual signals
- Efficient streaming/incremental coreference

**Note**: The isomorphism between vision detection and NER is inspired by DETR and similar work, but the specific three-level hierarchy appears to be original to this codebase.

### 2. Unified Evaluation Framework

The evaluation system integrates multiple recent research findings into a single framework:

- **Chain-length stratification** (arXiv:2401.00238) - Performs stratified analysis by chain length
- **Familiarity computation** (arXiv:2412.10121) - Quantifies label overlap to detect inflated zero-shot claims
- **Temporal stratification** - Analyzes performance across time periods
- **Confidence intervals** - Proper statistical reporting with sample variance

**Novel aspect**: The unified integration of these methods, not the individual methods themselves.

### 3. Trait Architecture

The sealed trait pattern and unified `Model` interface enable:
- Swappable backends without code changes
- Consistent evaluation across different model types
- Type-safe abstractions for zero-shot, relation extraction, etc.

This is architectural design work, not algorithmic research.

## Implementations of Existing Research

### Backend Implementations

| Backend | Paper | What We Implement |
|---------|-------|-------------------|
| **GLiNER** | [arXiv:2311.08526](https://arxiv.org/abs/2311.08526) | Bi-encoder span-label matching for zero-shot NER |
| **NuNER** | NuMind research | Token-based zero-shot NER with BIO tagging |
| **W2NER** | [arXiv:2112.10070](https://arxiv.org/abs/2112.10070) | Word-word relation grid for nested/discontinuous entities |
| **TPLinker** | Original TPLinker paper | Handshaking matrix for relation extraction |
| **UniversalNER** | [arXiv:2308.03279](https://arxiv.org/abs/2308.03279) | Placeholder only - not fully implemented |
| **ModernBERT** | [arXiv:2412.13663](https://arxiv.org/abs/2412.13663) | Encoder implementation |
| **DeBERTaV3** | Original DeBERTa papers | Encoder implementation |

**Credit**: These are implementations of existing research. We do not claim algorithmic contributions.

### Coreference Metrics

All coreference metrics are standard implementations:

- **MUC** (Vilain et al., 1995)
- **B³** (Bagga & Baldwin, 1998)
- **CEAF-e/m** (Luo, 2005)
- **LEA** (Moosavi & Strube, 2016) - [arXiv:1605.06301](https://arxiv.org/abs/1605.06301)
- **BLANC** (Recasens & Hovy, 2011)
- **CoNLL F1** - Average of MUC, B³, CEAF

**Credit**: Standard implementations, no algorithmic changes.

### Evaluation Methods

| Method | Paper | Status |
|--------|-------|--------|
| Chain-length stratification | [arXiv:2401.00238](https://arxiv.org/abs/2401.00238) | Implemented |
| Familiarity computation | [arXiv:2412.10121](https://arxiv.org/abs/2412.10121) | Implemented (string-based, embedding placeholder) |
| Temporal stratification | TempEL, various temporal drift papers | Implemented |
| Robustness testing | RUIE-Bench, adversarial NER papers | Implemented |

**Credit**: These implement research findings. We integrate them but do not propose new methods.

### Dataset Loaders

All datasets are downloaded and parsed from existing sources:

- CoNLL-2003, WikiGold, WNUT-17, MultiNERD, GAP, LitBank, etc.
- No datasets are created by this project
- Proper attribution given in dataset documentation

## Research-Aware Design

The codebase is designed with awareness of recent research findings:

- **Evaluation pitfalls** - Addresses common mistakes identified in research
- **Stratified analysis** - Breaks down metrics by entity type, chain length, temporal strata
- **Statistical rigor** - Proper confidence intervals, sample variance
- **Label shift** - Detects when "zero-shot" evaluations have high label overlap

This is **integration work**, not novel research.

## Placeholders / Not Yet Implemented

Some features are documented but not fully implemented:

- **UniversalNER** - Placeholder only, LLM integration pending
- **Embedding-based familiarity** - Currently uses string similarity fallback
- **ReasoningNER** - Trait documented, not implemented
- **CMAS (multi-agent)** - Documented, not implemented

## Attribution

We strive to:

1. **Cite papers** in code comments and documentation
2. **Link to arXiv** when available
3. **Credit original authors** in relevant modules
4. **Be clear** about what's implementation vs. novel

If you find missing attributions, please open an issue.

## Summary

- **~15-20% novel**: Architectural design (grounded representation, trait system)
- **~70-75% implementation**: Backend implementations, metrics, dataset loaders
- **~10-15% integration**: Research-aware evaluation framework combining multiple findings

The main value is in the **unified architecture** and **production-ready Rust implementations** of research methods, not in proposing new algorithms.

