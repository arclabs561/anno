#!/usr/bin/env bash
# Check ML backend patterns for common issues
# Creative use: validates ML-specific code patterns

set -euo pipefail

echo "=== ML Backend Pattern Analysis ==="
echo ""

ISSUES=0

# 1. Check for HuggingFace authentication handling
echo "## HuggingFace Authentication"
echo ""
AUTH_CHECKS=$(rg -c "401|Unauthorized|HF_TOKEN|huggingface.*token" --type rust src/backends/ 2>/dev/null || echo "0")
if [ "$AUTH_CHECKS" -lt 2 ]; then
    echo "WARNING:  Limited HuggingFace authentication error handling"
    echo "   Found $AUTH_CHECKS authentication-related checks"
    echo "   Consider adding 401 error detection with helpful hints"
    ((ISSUES++))
else
    echo "OK: HuggingFace authentication handling found ($AUTH_CHECKS checks)"
fi

# 2. Check for model download error context
echo ""
echo "## Model Download Error Messages"
echo ""
DETAILED_ERRORS=$(rg -c "Failed to download.*Hint|model.*download.*error.*context" --type rust src/backends/ 2>/dev/null || echo "0")
if [ "$DETAILED_ERRORS" -lt 3 ]; then
    echo "WARNING:  Some model downloads may lack detailed error context"
    ((ISSUES++))
else
    echo "OK: Detailed error messages found ($DETAILED_ERRORS instances)"
fi

# 3. Check for ONNX session pooling
echo ""
echo "## ONNX Session Management"
echo ""
SESSION_POOL=$(rg -c "SessionPool|session.*pool" --type rust src/backends/ 2>/dev/null || echo "0")
SESSION_CREATION=$(rg -c "Session::builder\(\)|commit_from_file" --type rust src/backends/ 2>/dev/null || echo "0")
if [ "$SESSION_POOL" -eq 0 ] && [ "$SESSION_CREATION" -gt 5 ]; then
    echo "WARNING:  Multiple ONNX session creations without pooling"
    echo "   Consider using SessionPool for better performance"
    ((ISSUES++))
else
    echo "OK: Session pooling or appropriate session management found"
fi

# 4. Check for tokenizer error handling
echo ""
echo "## Tokenizer Error Handling"
echo ""
TOKENIZER_ERRORS=$(rg -c "tokenizer.*error|Failed to load tokenizer" --type rust src/backends/ 2>/dev/null || echo "0")
if [ "$TOKENIZER_ERRORS" -lt 2 ]; then
    echo "WARNING:  Limited tokenizer error handling"
    ((ISSUES++))
else
    echo "OK: Tokenizer error handling found ($TOKENIZER_ERRORS instances)"
fi

# 5. Check for sequence length validation
echo ""
echo "## Sequence Length Validation"
echo ""
MAX_LEN_CHECKS=$(rg -c "max.*length|sequence.*length.*check|ids\.len\(\)\s*>" --type rust src/backends/ 2>/dev/null || echo "0")
if [ "$MAX_LEN_CHECKS" -lt 2 ]; then
    echo "WARNING:  Limited sequence length validation"
    echo "   Long texts may exceed model max sequence length"
    ((ISSUES++))
else
    echo "OK: Sequence length validation found ($MAX_LEN_CHECKS checks)"
fi

# 6. Check for unsafe code in ML backends
echo ""
echo "## Unsafe Code in ML Backends"
echo ""
UNSAFE_BLOCKS=$(rg -c "unsafe\s+\{" --type rust src/backends/ 2>/dev/null || echo "0")
SAFETY_COMMENTS=$(rg -c "// SAFETY:|///.*SAFETY" --type rust src/backends/ 2>/dev/null || echo "0")
if [ "$UNSAFE_BLOCKS" -gt 0 ]; then
    if [ "$SAFETY_COMMENTS" -lt "$UNSAFE_BLOCKS" ]; then
        echo "WARNING:  Some unsafe blocks may lack safety documentation"
        echo "   $UNSAFE_BLOCKS unsafe blocks, $SAFETY_COMMENTS safety comments"
        ((ISSUES++))
    else
        echo "OK: Unsafe blocks have safety documentation ($UNSAFE_BLOCKS blocks, $SAFETY_COMMENTS comments)"
    fi
else
    echo "INFO:  No unsafe blocks found in ML backends"
fi

echo ""
echo "=== Summary ==="
echo "Issues found: $ISSUES"
echo ""

if [ $ISSUES -gt 0 ]; then
    echo "NOTE: Recommendations:"
    echo "   - Add authentication error handling for HuggingFace downloads"
    echo "   - Improve error messages with context and hints"
    echo "   - Consider session pooling for ONNX backends"
    echo "   - Add sequence length validation"
    echo "   - Document unsafe code with safety comments"
    exit 1
else
    echo "OK: ML backend patterns look good"
    exit 0
fi

