#!/usr/bin/env bash
# Integrate static analysis findings with evaluation framework
# Creative use: uses evaluation framework to validate static analysis rules

set -euo pipefail

echo "=== Static Analysis + Evaluation Framework Integration ==="
echo ""

# This script demonstrates how static analysis can inform evaluation
# and vice versa - a creative integration

INTEGRATION_FILE="static-analysis-eval-integration.md"
cat > "$INTEGRATION_FILE" <<EOF
# Static Analysis + Evaluation Framework Integration

Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Concept

This document explores how static analysis findings can be validated against
actual evaluation results, and how evaluation patterns can inform static analysis rules.

## Integration Points

### 1. Variance Calculation Validation

**Static Analysis Finding**: Check for Bessel's correction (n-1) in variance calculations.

**Evaluation Validation**: Run evaluation and check if confidence intervals are reasonable.
If variance is too small (population variance), CIs will be too narrow.

**Command**:
\`\`\`bash
# Run evaluation
cargo run --example eval_basic --features eval

# Check CI widths in output
# Narrow CIs may indicate population variance instead of sample variance
\`\`\`

### 2. Entity Validation

**Static Analysis Finding**: Check for entity offset validation (start <= end).

**Evaluation Validation**: Run evaluation on real datasets and check for validation errors.
If entities have invalid offsets, evaluation will fail or produce warnings.

**Command**:
\`\`\`bash
# Run evaluation with validation
cargo test --test eval_integration --features eval-advanced

# Check for validation errors in output
\`\`\`

### 3. Backend Performance

**Static Analysis Finding**: Check for backend recreation in loops.

**Evaluation Validation**: Compare evaluation times with/without backend reuse.
If backends are recreated, evaluation will be slower.

**Command**:
\`\`\`bash
# Benchmark evaluation performance
cargo bench --bench evaluation_performance --features eval-parallel

# Compare with/without backend reuse
\`\`\`

### 4. Confidence Score Validation

**Static Analysis Finding**: Check for confidence score range validation (0.0-1.0).

**Evaluation Validation**: Run evaluation and check if any confidence scores
are outside [0.0, 1.0]. This would indicate a bug.

**Command**:
\`\`\`bash
# Run evaluation and extract confidence scores
cargo run --example eval_basic --features eval | grep -i confidence

# Check for values outside [0.0, 1.0]
\`\`\`

## Automated Integration

### Future Enhancement: Evaluation-Driven Rule Validation

1. **Run static analysis** → Get list of potential issues
2. **Run evaluation** → Get actual behavior
3. **Compare** → Validate if static analysis findings match real issues
4. **Refine rules** → Update OpenGrep rules based on false positives/negatives

### Example Workflow

\`\`\`bash
# 1. Run static analysis
just analysis-nlp-ml > static-analysis.txt

# 2. Run evaluation
cargo run --example eval_basic --features eval > evaluation-output.txt

# 3. Cross-reference
# - If static analysis finds "missing offset validation" but evaluation passes,
#   the validation might exist elsewhere (false positive)
# - If evaluation fails with offset errors but static analysis didn't catch it,
#   update rules (false negative)
\`\`\`

## Current Integration

### CI Integration

Static analysis runs in CI alongside evaluation:
- Static analysis catches code patterns
- Evaluation validates runtime behavior
- Both provide complementary insights

### Justfile Integration

\`\`\`bash
# Run both static analysis and evaluation
just analysis-nlp-ml
just eval-quick

# Compare results
\`\`\`

## Benefits

1. **Validation**: Evaluation validates static analysis findings
2. **Refinement**: Evaluation results help refine static analysis rules
3. **Coverage**: Both static and dynamic analysis provide comprehensive coverage
4. **Feedback Loop**: Continuous improvement of both tools

EOF

echo "OK: Integration document generated: $INTEGRATION_FILE"
echo ""
echo "NOTE: This demonstrates how static analysis and evaluation can work together"
echo "   to provide comprehensive code quality assurance."

