# Documentation Index

This directory contains documentation for the `anno` crate. This guide helps you navigate the documentation structure.

## Core Documentation

### Getting Started
- **[README.md](../README.md)** - Main project README with quick start and examples
- **[SCOPE.md](SCOPE.md)** - What's implemented, what's not, trait hierarchy, and roadmap
- **[RESEARCH.md](RESEARCH.md)** - Research contributions vs. implementations, attribution

### Evaluation
- **[EVALUATION.md](EVALUATION.md)** - Comprehensive evaluation guide with examples
- **[EVALUATION_CRITIQUE.md](EVALUATION_CRITIQUE.md)** - Research-based evaluation limitations
- **[EVAL_CRITIQUE.md](EVAL_CRITIQUE.md)** - Critique of specific evaluation results
- **[EVALUATION_CRITICAL_REVIEW.md](EVALUATION_CRITICAL_REVIEW.md)** - Implementation review

### Design & Architecture
- **[BACKEND_INTERFACE_REVIEW.md](BACKEND_INTERFACE_REVIEW.md)** - Backend interface design review
- **[ENCODER_TRAIT_DESIGN.md](ENCODER_TRAIT_DESIGN.md)** - Text encoder trait design
- **[INTER_INTRA_DOC_ABSTRACTIONS.md](INTER_INTRA_DOC_ABSTRACTIONS.md)** - Coreference abstractions

## Feature-Specific Documentation

### Caching & Performance
- **[FEATURE_CACHE_DESIGN.md](FEATURE_CACHE_DESIGN.md)** - Feature cache design options
- **[FEATURE_CACHE_CRITIQUE.md](FEATURE_CACHE_CRITIQUE.md)** - Design critique
- **[FEATURE_CACHE_IMPLEMENTATION_PLAN.md](FEATURE_CACHE_IMPLEMENTATION_PLAN.md)** - Implementation plan
- **[ACCIDENTAL_OPTIMIZATION_OPPORTUNITIES.md](ACCIDENTAL_OPTIMIZATION_OPPORTUNITIES.md)** - Reuse opportunities
- **[REUSE_OPPORTUNITIES.md](REUSE_OPPORTUNITIES.md)** - Optimization opportunities
- **[PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md)** - Performance analysis
- **[PROFILING.md](PROFILING.md)** - Profiling results

### Datasets & Models
- **[DATASETS.md](DATASETS.md)** - Supported datasets
- **[DATASET_DOWNLOADS.md](DATASET_DOWNLOADS.md)** - Dataset download information
- **[DATASET_IMPROVEMENTS.md](DATASET_IMPROVEMENTS.md)** - Dataset improvements
- **[MODEL_DOWNLOADS.md](MODEL_DOWNLOADS.md)** - Model download information
- **[MISSING_BACKENDS_AND_DATASETS.md](MISSING_BACKENDS_AND_DATASETS.md)** - Missing implementations

### Advanced Features
- **[LLM_NER_DESIGN.md](LLM_NER_DESIGN.md)** - LLM-based NER design
- **[MODERNBERT.md](MODERNBERT.md)** - ModernBERT implementation
- **[MULTIMODAL_EVAL_DESIGN.md](MULTIMODAL_EVAL_DESIGN.md)** - Multimodal evaluation design
- **[LEXICON_DESIGN.md](LEXICON_DESIGN.md)** - Lexicon design
- **[ABSTRACT_ANAPHORA_RESEARCH.md](ABSTRACT_ANAPHORA_RESEARCH.md)** - Abstract anaphora research

## Development & Testing

### Testing & Quality
- **[TEST_SUMMARY.md](TEST_SUMMARY.md)** - Test coverage summary
- **[TEST_GAPS_ANALYSIS.md](TEST_GAPS_ANALYSIS.md)** - Test gaps analysis
- **[TESTING_GAPS.md](TESTING_GAPS.md)** - Testing gaps
- **[SYSTEM_TESTING_SUMMARY.md](SYSTEM_TESTING_SUMMARY.md)** - System testing summary
- **[FINAL_TEST_REPORT.md](FINAL_TEST_REPORT.md)** - Final test report
- **[FUZZING_OPPORTUNITIES.md](FUZZING_OPPORTUNITIES.md)** - Fuzzing opportunities
- **[BUGS_FIXED.md](BUGS_FIXED.md)** - Bug fixes documentation

### Analysis & Reviews
- **[BENCHMARK_ANALYSIS.md](BENCHMARK_ANALYSIS.md)** - Benchmark analysis
- **[CI_EVALUATION.md](CI_EVALUATION.md)** - CI evaluation
- **[EVAL_ANALYSIS.md](EVAL_ANALYSIS.md)** - Evaluation analysis
- **[REVIEW_FINDINGS.md](REVIEW_FINDINGS.md)** - Review findings
- **[BACKWARD_REVIEW.md](BACKWARD_REVIEW.md)** - Backward compatibility review
- **[ORIGINAL_GOALS_REVIEW.md](ORIGINAL_GOALS_REVIEW.md)** - Original goals review

### Implementation Status
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation summary
- **[HARMONIZATION_COMPLETE.md](HARMONIZATION_COMPLETE.md)** - Harmonization status
- **[HARMONIZATION_PLAN.md](HARMONIZATION_PLAN.md)** - Harmonization plan
- **[HARMONIZATION_SUMMARY.md](HARMONIZATION_SUMMARY.md)** - Harmonization summary
- **[HARMONIZATION_VERIFICATION.md](HARMONIZATION_VERIFICATION.md)** - Harmonization verification

## Maintenance & Cleanup

### Documentation Management
- **[DOCUMENTATION_CRITIQUE.md](DOCUMENTATION_CRITIQUE.md)** - Documentation critique and improvements
- **[DOCS_CLEANUP_ANALYSIS.md](DOCS_CLEANUP_ANALYSIS.md)** - Documentation cleanup analysis
- **[REPO_STATUS.md](REPO_STATUS.md)** - Current repository status
- **[AUDIT_2025.md](AUDIT_2025.md)** - 2025 audit

### Task & Dataset Mapping
- **[TASK_DATASET_MAPPING.md](TASK_DATASET_MAPPING.md)** - Task to dataset mapping
- **[TASKS_MODELS_DATASETS_EVALS_TESTS_MATRIX.md](TASKS_MODELS_DATASETS_EVALS_TESTS_MATRIX.md)** - Comprehensive matrix

## Archived Documentation

Historical and completed documentation is in [`archive/`](archive/):
- Old status reports
- Completed implementation docs
- Historical reviews
- Mutation test results

## Quick Reference

| Need | Document |
|------|----------|
| Quick start | [README.md](../README.md) |
| What's implemented | [SCOPE.md](SCOPE.md) |
| Evaluation guide | [EVALUATION.md](EVALUATION.md) |
| Research attribution | [RESEARCH.md](RESEARCH.md) |
| Available datasets | [DATASETS.md](DATASETS.md) |
| Performance tuning | [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md) |
| Bug fixes | [BUGS_FIXED.md](BUGS_FIXED.md) |
| Test coverage | [TEST_SUMMARY.md](TEST_SUMMARY.md) |

## Contributing

When adding new documentation:
1. Place in appropriate category above
2. Update this README with a brief description
3. Add cross-references to related docs
4. Use consistent formatting (see [DOCUMENTATION_CRITIQUE.md](DOCUMENTATION_CRITIQUE.md))

