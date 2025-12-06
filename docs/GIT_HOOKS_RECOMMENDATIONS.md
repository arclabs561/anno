# Git Hooks Recommendations

Based on historical errors and CI failures, here are recommended checks for git hooks.

## Current Hooks

### Pre-commit (`/.git/hooks/pre-commit`)
- ✅ Files with spaces in names
- ✅ Duplicate test files (`* 2.rs`)
- ✅ Format check (`just fmt-check`)
- ✅ Clippy warnings (non-blocking)

### Pre-push (`/.git/hooks/pre-push`)
- ✅ Quick checks (`just check`)

## Recommended Additions

### Pre-commit Enhancements

#### 1. **Compilation Check** (Fast, Catches Syntax Errors)
```bash
cargo check --workspace --all-targets --message-format=short
```
**Why**: Catches compilation errors before commit. Faster than full test suite.
**Cost**: ~5-10 seconds

#### 2. **Test Compilation** (Ensure Tests Compile)
```bash
cargo check --workspace --tests
```
**Why**: Ensures test code compiles even if we don't run tests.
**Cost**: ~3-5 seconds

#### 3. **Secrets Detection** (Prevent API Keys/Passwords)
```bash
# Check for common secret patterns
rg -i "api[_-]?key|password|secret|token|credential" \
   --glob '!*.md' --glob '!*.txt' --glob '!target/**' \
   --glob '!.git/**' --glob '!*.lock' || true
```
**Why**: Prevents accidentally committing secrets.
**Cost**: <1 second

#### 4. **Large File Detection** (Prevent Accidental Large Commits)
```bash
# Warn about files > 1MB
find . -type f -size +1M ! -path "./target/*" ! -path "./.git/*" \
      ! -path "./.cargo/*" ! -path "./*.lock" | head -5
```
**Why**: Prevents accidentally committing large model files, datasets, etc.
**Cost**: <1 second

#### 5. **Documentation Check** (Public APIs Must Be Documented)
```bash
cargo doc --workspace --no-deps --document-private-items 2>&1 | \
  grep -i "missing documentation\|undocumented" || true
```
**Why**: Ensures public APIs are documented (can be non-blocking warning).
**Cost**: ~10-15 seconds

### Commit Message Hook (`/.git/hooks/commit-msg`)

#### 6. **Commit Message Validation**
```bash
# Enforce conventional commits or at least meaningful messages
COMMIT_MSG=$(cat "$1")
if echo "$COMMIT_MSG" | grep -qE "^[a-z]+(\(.+\))?: .{10,}"; then
  exit 0
else
  echo "❌ Commit message should follow: type(scope): description (min 10 chars)"
  echo "Examples: feat(api): add new endpoint"
  echo "          fix: resolve compilation error"
  exit 1
fi
```
**Why**: Ensures meaningful commit messages for better history.
**Cost**: <1 second

### Pre-push Enhancements

#### 7. **Unused Dependencies Check** (Optional, Can Be Slow)
```bash
# Only if cargo-machete is installed
if command -v cargo-machete &> /dev/null; then
  cargo machete --check || echo "⚠️  Unused dependencies found (not blocking)"
fi
```
**Why**: Keeps dependencies clean, but can be slow.
**Cost**: ~30-60 seconds (optional)

#### 8. **License/Security Check** (Optional, Can Be Slow)
```bash
# Only if cargo-deny is installed
if command -v cargo-deny &> /dev/null; then
  cargo deny check --workspace || echo "⚠️  License/security issues found (not blocking)"
fi
```
**Why**: Catches license and security issues early.
**Cost**: ~20-40 seconds (optional)

## Hook Types Available

1. **pre-commit**: Runs before commit is created
   - Best for: Fast checks (format, lint, compilation)
   - Should be: Fast (<30 seconds total)

2. **pre-push**: Runs before push to remote
   - Best for: Medium checks (tests, integration)
   - Should be: Medium (<2 minutes)

3. **commit-msg**: Validates commit message
   - Best for: Message format validation
   - Should be: Very fast (<1 second)

4. **post-commit**: Runs after commit
   - Best for: Notifications, logging
   - Not recommended for blocking checks

5. **pre-rebase**: Runs before rebase
   - Best for: Preventing destructive rebases
   - Rarely needed

## Recommended Implementation Priority

### Phase 1: Critical (Add Now)
1. ✅ Compilation check (`cargo check`)
2. ✅ Secrets detection (basic patterns)
3. ✅ Commit message validation

### Phase 2: Important (Add Soon)
4. Test compilation check
5. Large file detection
6. Documentation warnings (non-blocking)

### Phase 3: Nice to Have (Optional)
7. Unused dependencies (pre-push, non-blocking)
8. License/security checks (pre-push, non-blocking)

## Performance Considerations

- **Pre-commit**: Should complete in <30 seconds total
- **Pre-push**: Can take up to 2 minutes
- **Commit-msg**: Must be <1 second

Use `--message-format=short` and `--quiet` flags where possible to reduce output.

## Example: Enhanced Pre-commit Hook

```bash
#!/bin/bash
set -e

# Fast checks first
echo "🔍 Checking for invalid filenames..."
# ... existing checks ...

echo "🔍 Checking for secrets..."
# ... secrets check ...

echo "🔍 Checking compilation..."
cargo check --workspace --all-targets --message-format=short --quiet

echo "🔍 Checking test compilation..."
cargo check --workspace --tests --message-format=short --quiet

echo "🔍 Checking formatting..."
just fmt-check

echo "✅ Pre-commit checks passed!"
```

