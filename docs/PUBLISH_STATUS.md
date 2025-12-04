# Publish Status and Validation

## Current Publish Status

### Published on crates.io
- ✅ **anno** v0.2.0 - Published and available

### Ready to Publish
- ✅ **anno-core** v0.2.0 - Dry-run successful, ready to publish

### Publish Issues (Need Fixes)

#### anno-coalesce
**Issue**: Missing version requirement for `anno-core` dependency
**Error**: `dependency 'anno-core' does not specify a version`
**Fix Required**: Add version requirement to `anno-core` dependency in `coalesce/Cargo.toml`

#### anno-strata  
**Issue**: Missing version requirement for `anno-core` dependency
**Error**: `dependency 'anno-core' does not specify a version`
**Fix Required**: Add version requirement to `anno-core` dependency in `strata/Cargo.toml`

#### anno
**Issue**: Missing version requirements for workspace dependencies
**Error**: `dependency 'anno-coalesce' does not specify a version`
**Dependencies needing fixes**:
- `anno-coalesce` - needs version requirement
- `anno-strata` - needs version requirement (optional dependency)

**Note**: `anno-core` dependency is fine (already published)

### Not for Publication
- ❌ **anno-cli** - Marked `publish = false` (binary-only crate)

## Publish Order (Dependencies First)

1. **anno-core** ✅ (ready, no dependencies on other workspace crates)
2. **anno-coalesce** (depends on anno-core - needs version fix)
3. **anno-strata** (depends on anno-core - needs version fix)
4. **anno** (depends on anno-core, anno-coalesce, anno-strata - needs version fixes)

## Validation Status

### Build Status
- ✅ Workspace builds successfully with `eval-advanced` feature
- ✅ All tests pass
- ⚠️ `anno-cli` has compilation error when built without `eval-advanced` (expected - feature-gated commands)

### Workspace Structure
- ✅ All 5 crates properly configured
- ✅ Workspace dependencies correctly set up
- ✅ Feature flags properly propagated

## Next Steps

To enable publishing of all crates:

1. Add version requirements to workspace dependencies:
   - `coalesce/Cargo.toml`: `anno-core = { version = "0.2.0", path = "../anno-core" }`
   - `strata/Cargo.toml`: `anno-core = { version = "0.2.0", path = "../anno-core" }`
   - `anno/Cargo.toml`: 
     - `anno-coalesce = { version = "0.2.0", path = "../coalesce" }`
     - `anno-strata = { version = "0.2.0", path = "../strata", optional = true }`

2. Publish in dependency order:
   ```bash
   cargo publish -p anno-core
   cargo publish -p anno-coalesce
   cargo publish -p anno-strata
   cargo publish -p anno
   ```

3. After publishing, update dependencies to remove `path` specifications (optional - cargo handles this automatically during publish)

## Notes

- Path dependencies are fine for local development
- When publishing, cargo automatically converts `path` to version requirements
- Version requirements must match the published version on crates.io
- Workspace version is `0.2.0` for all crates

