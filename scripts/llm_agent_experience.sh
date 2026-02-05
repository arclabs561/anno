#!/bin/bash
# LLM Agent Experience Testing Script
# Tests various anno CLI features that an LLM agent might use
# Run with: bash scripts/llm_agent_experience.sh

set -e
ANNO=${ANNO:-./target/debug/anno}

if [ ! -x "$ANNO" ]; then
  echo "error: anno binary not found at: $ANNO"
  echo "Build with: cargo build -p anno-cli --bin anno"
  exit 1
fi

echo "=================================================================="
echo "LLM Agent Experience Testing for anno CLI"
echo "=================================================================="
echo ""

# Create temp directory for test files
TESTDIR=$(mktemp -d)
trap "rm -rf $TESTDIR" EXIT

echo "Test Directory: $TESTDIR"
echo ""

# ==========================================================================
echo "=== 1. BASIC EXTRACTION ==="
# ==========================================================================

echo "1a. Simple text extraction (default model)"
echo "Input: 'Tim Cook is the CEO of Apple Inc.'"
$ANNO extract 'Tim Cook is the CEO of Apple Inc.'
echo ""

echo "1b. File-based extraction"
echo 'Marie Curie won the Nobel Prize in Physics and Chemistry.' > "$TESTDIR/simple.txt"
$ANNO extract -f "$TESTDIR/simple.txt"
echo ""

echo "1c. Stdin extraction (piping)"
echo 'Elon Musk founded SpaceX in California.' | $ANNO extract
echo ""

# ==========================================================================
echo "=== 2. OUTPUT FORMATS ==="
# ==========================================================================

echo "2a. JSON format"
$ANNO extract --format json 'Barack Obama was the 44th President.'
echo ""

echo "2b. TSV format"
$ANNO extract --format tsv 'Google CEO Sundar Pichai announced Q4 results.'
echo ""

echo "2c. JSONL format"
$ANNO extract --format jsonl 'Jeff Bezos founded Amazon in Seattle.'
echo ""

# ==========================================================================
echo "=== 3. BACKEND COMPARISON ==="
# ==========================================================================

echo "3a. Pattern backend (dates, emails, URLs)"
$ANNO extract --model pattern 'Contact support@company.com by 2024-01-15 for $500 refund.'
echo ""

echo "3b. Heuristic backend (NER)"
$ANNO extract --model heuristic 'Dr. John Smith works at Harvard University.'
echo ""

echo "3c. GLiNER backend (ML)"
if $ANNO extract --help | grep -q "gliner"; then
  $ANNO extract --model gliner 'Angela Merkel led Germany for 16 years.'
else
  echo "(skipped) GLiNER backend not available in this build. Rebuild with: cargo build -p anno-cli --features onnx --bin anno"
fi
echo ""

echo "3d. Stacked backend (combined)"
$ANNO extract --model stacked 'Bill Gates (bill@microsoft.com) donated $1M to WHO on Jan 5, 2024.'
echo ""

echo "3e. Ensemble backend (weighted voting)"
$ANNO extract --model ensemble 'Tim Cook leads Apple Inc. in Cupertino.'
echo ""

echo "3f. Backend comparison table"
$ANNO compare "$TESTDIR/simple.txt" --models --model-list pattern,heuristic,stacked,ensemble --format table
echo ""

# ==========================================================================
echo "=== 4. TYPE FILTERING ==="
# ==========================================================================

echo "4a. Filter by entity type (comma-separated)"
$ANNO extract --types "person,organization" 'Tim Cook is the CEO of Apple Inc.'
echo ""

echo "4b. Filter by type + confidence threshold"
$ANNO extract --types "person,organization" --threshold 0.6 'Tim Cook is the CEO of Apple Inc.'
echo ""

# ==========================================================================
echo "=== 5. CROSS-DOCUMENT ANALYSIS ==="
# ==========================================================================

echo "5. Creating test documents..."
cat > "$TESTDIR/doc1.txt" << 'EOF'
Apple Inc. reported record earnings. CEO Tim Cook praised the team.
EOF
cat > "$TESTDIR/doc2.txt" << 'EOF'
Tim Cook has led Apple since 2011. The Cupertino company continues to innovate.
EOF
cat > "$TESTDIR/doc3.txt" << 'EOF'
Meanwhile, Google's Sundar Pichai announced AI investments. Apple's Cook remained silent.
EOF

echo "5a. Cross-document entity resolution"
if $ANNO --help | grep -q "^  cross-doc"; then
  $ANNO cross-doc "$TESTDIR/" --threshold 0.5 --format summary
else
  echo "(skipped) cross-doc requires eval build. Rebuild with: cargo build -p anno-cli --features eval --bin anno"
fi
echo ""

# ==========================================================================
echo "=== 6. DEBUGGING & INTROSPECTION ==="
# ==========================================================================

echo "6a. Explain entity decisions"
$ANNO explain -t 'Dr. Sarah Chen at MIT studied BRCA1 mutations.' --show-all
echo ""

echo "6b. Domain detection"
$ANNO domain -t 'Patient presented with acute myocardial infarction.'
echo ""

echo "6c. Singleton cluster analysis"
$ANNO singleton -t 'Obama spoke at the White House. The president addressed the nation.'
echo ""

# ==========================================================================
echo "=== 7. PRIVACY & PII ==="
# ==========================================================================

echo "7a. PII detection"
$ANNO privacy -t 'John Smith (555-123-4567) lives at 123 Main St.'
echo ""

echo "7b. PII redaction"
$ANNO privacy -t 'Contact john@company.com or 555-123-4567' --action redact
echo ""

# ==========================================================================
echo "=== 8. BATCH PROCESSING ==="
# ==========================================================================

echo "8. Batch processing directory"
$ANNO batch -d "$TESTDIR/" --format json
echo ""

# ==========================================================================
echo "=== 9. ADVANCED FILTERING ==="
# ==========================================================================

echo "9a. Filter by entity type"
$ANNO extract --label PER 'Tim Cook and Apple are in Cupertino.'
echo ""

echo "9b. Confidence threshold"
$ANNO extract --threshold 0.6 'Maybe John or possibly Mary visited New York.'
echo ""

echo "9c. Expected types validation"
$ANNO extract --expected-types PERSON,MONEY 'Tim Cook visited the Apple Store.'
echo ""

# ==========================================================================
echo "=== 10. EXPORT/IMPORT WORKFLOW ==="
# ==========================================================================

echo "10a. Export to brat format"
echo 'Marie Curie won the Nobel Prize.' > "$TESTDIR/to_export.txt"
$ANNO dataset export --input "$TESTDIR/to_export.txt" --format brat --output "$TESTDIR/export_test" --overwrite
echo "Exported to $TESTDIR/export_test"
ls -la "$TESTDIR/export_test"* 2>/dev/null || echo "(export directory created)"
echo ""

# ==========================================================================
echo "=== 11. WATCH MODE (help only - blocking command) ==="
# ==========================================================================

echo "11. Watch command for live systems"
$ANNO watch --help
echo ""

# ==========================================================================
echo "=== SUMMARY ==="
# ==========================================================================

echo "=================================================================="
echo "LLM Agent Experience Testing Complete"
echo "=================================================================="
echo ""
echo "Working backends:"
echo "  - pattern: Dates, emails, URLs, money, hashtags"
echo "  - heuristic: NER via capitalization patterns"
echo "  - stacked: Combined pattern + heuristic"
echo "  - ensemble: Weighted voting across all backends"
echo "  - gliner: ML-based, label-conditioned types (requires onnx)"
echo ""
echo "Non-working backends (require setup):"
echo "  - gliner2: Model format mismatch"
echo "  - nuner: Index out of range error"  
echo "  - w2ner: Requires HuggingFace authentication"
echo ""
echo "Key commands tested:"
echo "  - extract: Core entity extraction"
echo "  - compare: Multi-backend comparison"
echo "  - cross-doc: Cross-document resolution"
echo "  - explain: Decision introspection"
echo "  - privacy: PII detection/redaction"
echo "  - domain: Domain shift detection"
echo "  - batch: Directory processing"
echo ""

