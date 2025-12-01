# Pre-Commit Checklist

## Files Status

### ✅ Should Commit (Code)
- `examples/comprehensive_evaluation.rs` - Main evaluation example
- `examples/eval_advanced_features.rs` - Advanced features demo
- `examples/eval_coref_analysis.rs` - Coreference analysis
- `examples/eval_stress_test.rs` - Stress testing
- `examples/eval_comparison.rs` - Comparison analysis
- `src/backends/tplinker.rs` - TPLinker backend
- `src/backends/albert.rs` - Albert backend
- `src/backends/deberta_v3.rs` - DeBERTa v3 backend
- `src/backends/gliner_poly.rs` - GLiNER poly backend
- `src/backends/universal_ner.rs` - Universal NER backend
- `tests/eval_improvements_tests.rs` - New tests
- `tests/task_evaluator_comprehensive.rs` - Comprehensive tests

### ✅ Should Commit (Documentation)
- `docs/BUGS_FIXED.md` - Bug fixes documentation
- `docs/REPO_STATUS.md` - Repository status
- `docs/SYSTEM_TESTING_SUMMARY.md` - Testing summary
- Other `docs/*.md` files (implementation docs, reviews, etc.)

### ❌ Should NOT Commit (Now in .gitignore)
- `*_report.md` - Generated evaluation reports
- `*-seed-*.md` - Generated seed-based reports
- `eval-*.md` - Generated eval reports
- `FIXES_APPLIED.md` - Temporary review doc
- `REVIEW_BACKWARDS.md` - Temporary review doc
- `REVIEW_ISSUES.md` - Temporary review doc

### ✅ Modified Files (Should Commit)
- `src/eval/task_evaluator.rs` - Main evaluation logic (bug fixes, new features)
- `src/eval/coref_metrics.rs` - Chain-length stratification
- `src/eval/types.rs` - New types (ConfidenceIntervals, StratifiedMetrics, etc.)
- `src/eval/loader.rs` - Temporal metadata support
- `src/eval/ner_metrics.rs` - Boundary error documentation
- `src/eval/task_mapping.rs` - Task mapping updates
- `src/backends/catalog.rs` - Backend catalog updates
- `src/backends/mod.rs` - Backend module updates
- `src/backends/nuner.rs` - NUNER backend updates
- `src/eval/backend_factory.rs` - Backend factory updates
- `src/lib.rs` - Library exports
- `tests/integration_eval.rs` - Integration test updates
- `PROBLEMS.md` - Problems documentation (if intentional)
- `eval-sanity-report.md` - Sanity report (if intentional)

## Cleanup Done

- ✅ Deleted `examples/verify_new_features.rs` (replaced by `comprehensive_evaluation.rs`)
- ✅ Updated `.gitignore` to exclude generated reports and temporary docs

## Before Committing

1. ✅ Review modified files - all look intentional
2. ✅ Check untracked files - code and docs should be committed
3. ✅ Generated reports are ignored
4. ✅ Old duplicate files removed
5. ✅ Build passes
6. ✅ Tests pass (711/712 - 1 pre-existing failure)

## Recommended Commit Structure

```bash
# Core evaluation improvements
git add src/eval/*.rs
git add src/backends/*.rs
git add src/lib.rs

# New examples
git add examples/comprehensive_evaluation.rs examples/eval_*.rs

# New tests
git add tests/eval_improvements_tests.rs tests/task_evaluator_comprehensive.rs

# Documentation
git add docs/BUGS_FIXED.md docs/REPO_STATUS.md docs/SYSTEM_TESTING_SUMMARY.md
git add docs/EVALUATION_CRITICAL_REVIEW.md docs/IMPLEMENTATION_*.md

# Configuration
git add .gitignore

# Other intentional changes
git add PROBLEMS.md tests/integration_eval.rs
```

## Notes

- Generated reports (`*_report.md`, `*-seed-*.md`) are now ignored
- Temporary review docs are ignored
- All code changes are intentional and tested
- Documentation is comprehensive and should be committed

