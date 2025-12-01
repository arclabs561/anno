# Documentation Cleanup Analysis

## Files That Shouldn't Be Committed

### 1. Generated Report Files (Root Directory)
These are tracked but should be gitignored:
- `comprehensive_evaluation_report.md` - Generated evaluation report
- `eval-sanity-report.md` - Generated sanity check report
- `eval-seed-*.md` (4 files) - Generated seed-based reports
- `stress_test_report.md` - Generated stress test report
- `verify_new_features_report.md` - Generated verification report

**Action**: Remove from git tracking (they're already in `.gitignore`)

### 2. Potentially Redundant Critique/Review Docs
Multiple critique/review files that might overlap:

- `docs/EVAL_CRITIQUE.md` - Evaluation results critique (specific to results)
- `docs/EVALUATION_CRITIQUE.md` - Evaluation limitations (research-based, general)
- `docs/EVALUATION_CRITICAL_REVIEW.md` - Implementation review (code review)

**Recommendation**: 
- Keep `EVALUATION_CRITIQUE.md` (research-based, general)
- Consider archiving `EVAL_CRITIQUE.md` (specific to old results)
- Keep `EVALUATION_CRITICAL_REVIEW.md` (implementation review, useful)

### 3. Temporary/Status Docs
These might be better in archive or `.github`:

- `docs/COMMIT_CHECKLIST.md` - Dev tool, could move to `.github/`
- `docs/REPO_CLEANUP.md` - Historical status, could archive
- `docs/REPO_STATUS.md` - Current status (keep if actively maintained)

**Recommendation**:
- Move `COMMIT_CHECKLIST.md` to `.github/` or archive
- Archive `REPO_CLEANUP.md` (historical)
- Keep `REPO_STATUS.md` if it's actively updated

### 4. Design/Planning Docs
Multiple related design docs that could be consolidated:

- `FEATURE_CACHE_CRITIQUE.md`
- `FEATURE_CACHE_DESIGN.md`
- `FEATURE_CACHE_IMPLEMENTATION_PLAN.md`

**Recommendation**: These are fine as-is (different phases of design)

### 5. Multiple Review Docs
- `BACKWARD_REVIEW.md` - Review of implementation
- `ORIGINAL_GOALS_REVIEW.md` - Review of original goals
- `BACKEND_INTERFACE_REVIEW.md` - Backend interface review
- `REVIEW_FINDINGS.md` - General review findings

**Recommendation**: These seem to serve different purposes, keep them

## Other Nuances

### 1. Documentation Organization
- `docs/archive/` is properly gitignored ✅
- Main `docs/` has many files but they seem organized
- Consider adding a `docs/README.md` to explain organization

### 2. Duplicate Information
- Some information appears in multiple places (e.g., evaluation modes in both README and EVALUATION.md)
- This is okay for discoverability, but could add cross-references

### 3. Outdated Information
- Some docs might reference old status (check `REPO_STATUS.md` for currency)
- `COMMIT_CHECKLIST.md` might reference old files

### 4. Missing Documentation
- No `docs/README.md` explaining the documentation structure
- No clear "Getting Started" guide (examples are in README but could be expanded)

## Recommended Actions

1. **Remove generated reports from git**:
   ```bash
   git rm --cached comprehensive_evaluation_report.md eval-sanity-report.md eval-seed-*.md stress_test_report.md verify_new_features_report.md
   ```

2. **Archive or move temporary docs**:
   - Move `COMMIT_CHECKLIST.md` to `.github/` or `docs/archive/`
   - Archive `REPO_CLEANUP.md` to `docs/archive/`

3. **Consider consolidating**:
   - Review `EVAL_CRITIQUE.md` vs `EVALUATION_CRITIQUE.md` - keep the more general one

4. **Add documentation index**:
   - Create `docs/README.md` explaining the documentation structure

