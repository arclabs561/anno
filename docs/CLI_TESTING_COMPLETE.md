# Complete CLI Testing Results

**Date**: 2025-01-25  
**Status**: Comprehensive testing of critical gaps

## Major Discoveries ✅

### 1. **Eval Command** - ✅ FULLY FUNCTIONAL

**Test**:
```bash
anno eval -t "Apple Inc. was founded by Steve Jobs." -g "Apple Inc:ORG:0:9" -g "Steve Jobs:PER:26:36"
```

**Result**: Perfect!
- Format: `-t "text" -g "entity:TYPE:start:end"`
- Shows precision, recall, F1
- Correctly identifies matches
- Error breakdown included

**Output**:
```
Precision: 100.0%
Recall:    100.0%
F1:        100.0%
+ correct: [ORG] "Apple Inc"
+ correct: [PER] "Steve Jobs"
```

### 2. **JSON Format** - ✅ WORKS PERFECTLY

**Test**:
```bash
anno extract --text "Apple Inc. was founded by Steve Jobs." --format json
```

**Result**: Perfect JSON output!
```json
[
  {
    "confidence": 0.85,
    "end": 9,
    "negated": false,
    "quantifier": null,
    "start": 0,
    "text": "Apple Inc",
    "type": "ORG"
  },
  {
    "confidence": 0.75,
    "end": 36,
    "negated": false,
    "quantifier": null,
    "start": 26,
    "text": "Steve Jobs",
    "type": "PER"
  }
]
```

**Finding**: JSON format works when using `--text` flag explicitly.

### 3. **Export Functionality** - ⚠️ PARTIAL BUG

**Issue**: Export works with `--text` flag but NOT with positional arguments

**Working**:
```bash
anno extract --text "text" --export /tmp/file.json  # ✅ Works
anno extract "text" --export=/tmp/file.json         # ✅ Works (with =)
```

**Not Working**:
```bash
anno extract "text" --export /tmp/file.json        # ❌ Export path included in text
```

**Root Cause**: `trailing_var_arg = true` on positional args consumes everything, including the export path value.

**Workaround**: Use `--text` flag or `--export=/path` (with equals sign)

**Impact**: MEDIUM - Workaround exists, but UX issue

### 4. **Enhance Command** - ✅ WORKS

**Test**:
```bash
anno extract --text "Apple Inc. was founded by Steve Jobs." --export /tmp/doc1.json
anno enhance /tmp/doc1.json --coref --export /tmp/doc1-coref.json
```

**Result**: ✅ **Works perfectly!**
- Reads exported JSON
- Adds coreference resolution
- Creates tracks
- Exports enhanced document

**Output Structure**: Enhanced JSON includes:
- `tracks`: Coreference chains
- `signal_to_track`: Mapping
- `next_track_id`: Track counter

### 5. **Query Command** - ✅ WORKS

**Tests**:
```bash
anno query /tmp/test_export3.json --type ORG
# Found 1 entities: [0:9] Apple Inc (ORG) - 0.85

anno query /tmp/test_export3.json --type PER
# Found 1 entities: [26:36] Steve Jobs (PER) - 0.75

anno query /tmp/test_export3.json --min-confidence 0.8
# Found 1 entities: [0:9] Apple Inc (ORG) - 0.85
```

**Result**: ✅ **Works perfectly!**
- Filters by entity type
- Filters by confidence threshold
- Clear output format

### 6. **Compare Command** - ✅ WORKS

**Test**:
```bash
anno extract --text "Microsoft was founded by Bill Gates." --export /tmp/doc2.json
anno compare /tmp/test_export3.json /tmp/doc2.json
```

**Result**: ✅ **Works!**
- Shows entities only in first document
- Shows entities only in second document
- Shows entities in both
- Clear diff format

**Output**:
```
Only in /tmp/test_export3.json: 2
  + Steve Jobs:PER:0.75
  + Apple Inc:ORG:0.85

Only in /tmp/doc2.json: 2
  - Microsoft:ORG:0.8
  - Bill Gates:PER:0.75

In both: 0
```

### 7. **Pipeline Command** - ✅ WORKS

**Test**:
```bash
echo "Document 1: Apple Inc. was founded by Steve Jobs." > /tmp/pipeline1.txt
echo "Document 2: Microsoft was founded by Bill Gates." > /tmp/pipeline2.txt
anno pipeline --files /tmp/pipeline1.txt /tmp/pipeline2.txt --coref
```

**Result**: ✅ **Works!**
- Processes multiple files
- Extracts entities from each
- Applies coreference when requested
- Shows results per document

### 8. **Analyze Command** - ✅ WORKS

**Test**:
```bash
anno analyze "Apple Inc. was founded by Steve Jobs in 1976." --models stacked
```

**Result**: ✅ **Works!**
- Runs multiple models (pattern, heuristic, stacked)
- Shows entity counts per model
- Shows model agreement
- Performance timing included

**Output**:
```
pattern: 0 entities in 93.5ms
heuristic: 2 entities in 0.6ms
stacked: 2 entities in 0.2ms

Model Agreement:
  Agreed (in stacked from pattern/heuristic): 2 entities
```

### 9. **File Input** - ✅ WORKS

**Test**:
```bash
echo "Test file with entities: Apple Inc. and Steve Jobs." > /tmp/test_input.txt
anno extract --file /tmp/test_input.txt
```

**Result**: ✅ **Works perfectly!**
- Reads from file correctly
- Extracts entities
- No issues

### 10. **Unicode/Emoji** - ✅ HANDLES GRACEFULLY

**Test**:
```bash
anno extract "Café résumé naïve 中文 🎉 emoji test"
```

**Result**: ✅ **No crashes**
- Processes unicode correctly
- Handles emoji
- No errors (no entities found, but that's expected)

## Issues Found

### 🔴 Bug #1: Positional Args with Export

**Symptom**: When using positional text args with `--export`, the export path is included in the text

**Example**:
```bash
anno extract "text" --export /tmp/file.json
# Text becomes: "text --export /tmp/file.json"
```

**Root Cause**: `trailing_var_arg = true` consumes all remaining arguments

**Workaround**: 
- Use `--text` flag: `anno extract --text "text" --export /tmp/file.json`
- Use equals: `anno extract "text" --export=/tmp/file.json`

**Impact**: MEDIUM - Workaround exists but confusing UX

### 🟡 Issue #1: JSON Format with Positional Args

**Symptom**: `--format json` may not work correctly with positional args (same root cause as export bug)

**Workaround**: Use `--text` flag

**Impact**: LOW - Workaround exists

## What Works Excellently ✅

1. **Eval command** - Perfect implementation
2. **JSON output** - Clean, structured, parseable
3. **Enhance command** - Incremental processing works
4. **Query command** - Filtering works perfectly
5. **Compare command** - Clear diff output
6. **Pipeline command** - Multi-file processing works
7. **Analyze command** - Multi-model comparison works
8. **File input** - Robust
9. **Unicode handling** - No issues
10. **Export** - Works with `--text` flag or `--export=/path`

## Integration Workflows Tested ✅

### Workflow 1: Extract → Enhance → Query
```bash
# ✅ Works perfectly
anno extract --text "text" --export doc.json
anno enhance doc.json --coref --export doc-coref.json
anno query doc-coref.json --type PER
```

### Workflow 2: Extract → Compare
```bash
# ✅ Works perfectly
anno extract --text "text1" --export doc1.json
anno extract --text "text2" --export doc2.json
anno compare doc1.json doc2.json
```

### Workflow 3: Pipeline Multi-File
```bash
# ✅ Works perfectly
anno pipeline --files file1.txt file2.txt --coref
```

## Summary Statistics

**Commands Tested**: 10/20 (50%)
**Commands Working**: 9/10 (90%)
**Critical Bugs Found**: 1 (positional args + export)
**Workarounds Available**: Yes (use `--text` or `--export=/path`)

## Recommendations

### Immediate Fixes
1. **Fix positional args parsing** - `trailing_var_arg` shouldn't consume flag values
2. **Document workaround** - Add note about using `--text` with `--export`

### Documentation
1. **Add examples** showing `--text` + `--export` pattern
2. **Document JSON format structure** in help text
3. **Add workflow examples** (extract → enhance → query)

### Testing Remaining
1. **Error handling** - After compilation fixes
2. **Edge cases** - Long text, empty input, special characters
3. **Cross-doc command** - Requires eval-advanced feature
4. **Batch command** - Bulk processing
5. **Dataset command** - Dataset operations
6. **Config command** - Configuration management
7. **Cache command** - Cache operations
8. **URL input** - Web content processing
9. **Validate command** - Annotation validation
10. **Other output formats** - JSONL, TSV, inline, tree, summary, HTML

## Overall Assessment

**Status**: **Very Good** - Most features work well

**Strengths**:
- Core functionality solid
- Integration workflows work
- JSON output is clean
- Commands are intuitive

**Weaknesses**:
- Positional args parsing issue
- Some commands need eval-advanced feature
- Error handling not fully tested

**Recommendation**: Fix positional args bug, then continue testing remaining commands.

