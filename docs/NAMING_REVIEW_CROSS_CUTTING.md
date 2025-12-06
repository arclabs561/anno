# Naming Review: Cross-Cutting Concerns & Nuances

## Status: ✅ HARMONIZED (2025-01-XX)

All naming inconsistencies have been resolved. The codebase now has consistent naming from crate level through CLI commands to documentation.

## Summary of Harmonization

### ✅ Completed Changes

1. **CLI Commands**: 
   - `anno crossdoc` command exists with `coalesce` as visible alias
   - `anno strata` command implemented and available
   - Both commands properly feature-gated under `eval-advanced`

2. **Documentation**:
   - `docs/ARCHITECTURE.md` updated to reflect actual CLI commands
   - All examples use correct command names
   - Tagline "Extract. Coalesce. Stratify." consistent across all docs

3. **Terminology**:
   - Standardized to "Cross-document entity coalescing" (not "coreference")
   - Error messages use "coalesce" terminology consistently
   - Help text updated throughout

4. **Code Consistency**:
   - All imports harmonized
   - Feature gates properly configured
   - Match statements exhaustive
   - Unused imports cleaned up

## Current State

### Crate Names
- ✅ `anno-core` - Shared types
- ✅ `anno` - Main NER library  
- ✅ `anno-coalesce` - Cross-document entity coalescing
- ✅ `anno-strata` - Hierarchical clustering
- ✅ `anno-cli` - Unified CLI (publish = false)

### CLI Commands
- ✅ `anno extract` - NER extraction
- ✅ `anno crossdoc` / `anno coalesce` - Cross-document entity coalescing (alias works)
- ✅ `anno strata` - Hierarchical clustering
- ✅ All 17 commands properly integrated

### Documentation
- ✅ `docs/ARCHITECTURE.md` - Accurate command examples
- ✅ `README.md` - Tagline "Extract. Coalesce. Stratify."
- ✅ `CHANGELOG.md` - Documents workspace refactoring
- ✅ All user-facing text uses consistent terminology

## Verification

All commands verified working:
```bash
# Cross-document entity coalescing (both forms work)
anno crossdoc --help
anno coalesce --help  # alias

# Hierarchical clustering
anno strata --help
```

Build status: ✅ All packages compile successfully with `eval-advanced` feature.

## Notes

- The `coalesce` alias provides backward compatibility and matches crate naming
- `strata` command requires `eval-advanced` feature (as documented)
- All terminology consistently uses "coalescing" rather than "coreference" for cross-document operations
- The tagline "Extract. Coalesce. Stratify." accurately describes the pipeline
