# CLI Gaps and What We Didn't Experience Enough

**Date**: 2025-01-25  
**Purpose**: Honest assessment of what we didn't test thoroughly

## What We Tested ✅

1. **Basic extraction** - Shorthand and explicit commands
2. **Info command** - Model availability
3. **Models command** - List and info
4. **Debug command** - Coreference resolution
5. **Pattern backend** - Dates, emails, phone numbers
6. **Output formats** - Brief look at human-readable output

## What We Didn't Experience Enough ❌

### 1. **Evaluation Command (`eval`)** - CRITICAL GAP
**Why Important**: This is a core feature - evaluating predictions against gold annotations
**What We Missed**:
- How to format gold annotations
- Running actual evaluations
- Understanding evaluation output
- Integration with evaluation framework

**Test Needed**:
```bash
anno eval -t "text" -g "entity:TYPE:start:end"
```

### 2. **JSON Output Formats** - MAJOR GAP
**Why Important**: JSON is essential for programmatic use and pipeline integration
**What We Missed**:
- Structure of JSON output
- Differences between `json`, `jsonl`, `grounded` formats
- How to parse and use JSON output
- Export/import workflows

**Test Needed**:
```bash
anno extract "text" --format json
anno extract "text" --format jsonl
anno extract "text" --format grounded
```

### 3. **File Input/Output** - MAJOR GAP
**Why Important**: Real-world usage involves files, not just command-line text
**What We Missed**:
- Reading from files
- Writing to files
- Batch processing files
- Directory processing

**Test Needed**:
```bash
anno extract --file input.txt
anno extract --file input.txt --output output.json
anno pipeline --dir ./docs
```

### 4. **Pipeline Command** - MAJOR GAP
**Why Important**: This orchestrates the full extraction → coref → KB → cross-doc workflow
**What We Missed**:
- How pipeline combines multiple steps
- Cross-document processing
- Output from pipeline
- Performance characteristics

**Test Needed**:
```bash
anno pipeline "text1" "text2" --coref --link-kb --cross-doc
anno pipeline --dir ./docs --coref
```

### 5. **Cross-Document Coreference** - MAJOR GAP
**Why Important**: This is a unique feature - clustering entities across documents
**What We Missed**:
- How cross-doc clustering works
- Input/output formats
- Threshold tuning
- Cluster visualization

**Test Needed**:
```bash
anno cross-doc /path/to/documents --format json
anno cross-doc /path/to/documents --format tree
```

### 6. **Query Command** - MAJOR GAP
**Why Important**: Essential for exploring and filtering results
**What We Missed**:
- Querying GroundedDocuments
- Filtering by type, confidence
- Querying cross-doc clusters
- Output formats for queries

**Test Needed**:
```bash
anno query doc.json --type PER --min-confidence 0.8
anno query clusters.json --entity "Apple Inc"
```

### 7. **Compare Command** - MODERATE GAP
**Why Important**: Comparing models/documents is useful for analysis
**What We Missed**:
- Comparing two documents
- Comparing models on same text
- Understanding diff output
- Use cases

**Test Needed**:
```bash
anno compare doc1.json doc2.json --format diff
```

### 8. **Enhance Command** - MODERATE GAP
**Why Important**: Incremental processing workflow
**What We Missed**:
- How enhancement works
- Adding coref to existing extraction
- Adding KB linking incrementally
- Export/import workflow

**Test Needed**:
```bash
anno extract "text" --export doc.json
anno enhance doc.json --coref --export doc-coref.json
anno enhance doc-coref.json --link-kb --export doc-full.json
```

### 9. **Analyze Command** - MODERATE GAP
**Why Important**: Multi-model analysis for comparison
**What We Missed**:
- Running multiple models
- Comparing model outputs
- Understanding analysis output
- Performance comparison

**Test Needed**:
```bash
anno analyze "text" --models stacked gliner w2ner
```

### 10. **Validate Command** - MODERATE GAP
**Why Important**: Ensuring annotation quality
**What We Missed**:
- Validation rules
- Error reporting
- Input formats
- Use cases

**Test Needed**:
```bash
anno validate annotations.jsonl
```

### 11. **Dataset Command** - MODERATE GAP
**Why Important**: Working with evaluation datasets
**What We Missed**:
- Listing available datasets
- Downloading datasets
- Dataset information
- Integration with eval

**Test Needed**:
```bash
anno dataset list
anno dataset info wikigold
```

### 12. **Cache Management** - MINOR GAP
**Why Important**: Understanding caching behavior
**What We Missed**:
- What gets cached
- Cache invalidation
- Cache inspection
- Performance implications

**Test Needed**:
```bash
anno cache list
anno cache clear
anno cache invalidate --model gliner
```

### 13. **Error Handling** - CRITICAL GAP
**Why Important**: Understanding failure modes and user experience
**What We Missed**:
- Invalid inputs
- Missing files
- Unsupported backends
- Feature flag requirements
- Error messages quality

**Test Needed**:
```bash
anno extract --file /nonexistent/file.txt
anno extract ""  # empty input
anno extract "text" --backend nonexistent
```

### 14. **Edge Cases** - CRITICAL GAP
**Why Important**: Real-world robustness
**What We Missed**:
- Very long text
- Unicode/emoji handling
- Special characters
- Empty results
- Multiple entity types
- Overlapping entities

**Test Needed**:
```bash
# Long text
anno extract "$(head -c 10000 /dev/urandom | base64)"

# Unicode
anno extract "Café résumé naïve 中文 🎉"

# Empty
anno extract ""
```

### 15. **Integration Workflows** - MAJOR GAP
**Why Important**: Real-world usage patterns
**What We Missed**:
- Extract → Enhance → Cross-doc workflow
- Export → Import patterns
- JSON → Query → Compare workflow
- Batch processing workflows

**Test Needed**:
```bash
# Full workflow
anno extract "text1" --export doc1.json
anno extract "text2" --export doc2.json
anno enhance doc1.json --coref --export doc1-coref.json
anno cross-doc --import doc1-coref.json doc2.json
```

### 16. **Performance Characteristics** - MODERATE GAP
**Why Important**: Understanding scalability
**What We Missed**:
- Processing time for different text lengths
- Memory usage
- Batch processing performance
- Model loading overhead

**Test Needed**:
```bash
time anno extract "$(cat large_file.txt)"
```

### 17. **Output Format Details** - MAJOR GAP
**Why Important**: Understanding what each format provides
**What We Missed**:
- JSON structure details
- JSONL format
- TSV format
- Inline format
- Tree format
- Summary format
- HTML format

**Test Needed**:
```bash
anno extract "text" --format json | jq .
anno extract "text" --format jsonl
anno extract "text" --format tsv
anno extract "text" --format inline
```

### 18. **URL Input** - MINOR GAP
**Why Important**: Web content processing
**What We Missed**:
- URL resolution
- HTML extraction
- Error handling for URLs
- Feature requirements

**Test Needed**:
```bash
anno extract --url https://example.com
```

### 19. **Config Command** - MINOR GAP
**Why Important**: Workflow management
**What We Missed**:
- Saving configurations
- Loading configurations
- Configuration format
- Use cases

**Test Needed**:
```bash
anno config save my-workflow --model gliner --coref
anno pipeline --config my-workflow
```

### 20. **Batch Command** - MODERATE GAP
**Why Important**: Efficient bulk processing
**What We Missed**:
- Batch processing capabilities
- Parallel processing
- Progress tracking
- Input/output handling

**Test Needed**:
```bash
anno batch --dir ./docs --coref --parallel 4 --progress
```

## Priority Ranking

### 🔴 Critical (Must Test)
1. **Eval command** - Core functionality
2. **Error handling** - User experience
3. **Edge cases** - Robustness
4. **JSON output formats** - Programmatic use

### 🟡 High Priority (Should Test)
5. **File input/output** - Real-world usage
6. **Pipeline command** - Full workflow
7. **Cross-doc command** - Unique feature
8. **Query command** - Result exploration
9. **Integration workflows** - End-to-end usage

### 🟢 Medium Priority (Nice to Test)
10. **Compare command** - Analysis tool
11. **Enhance command** - Incremental processing
12. **Analyze command** - Model comparison
13. **Output format details** - Understanding options

### ⚪ Low Priority (Optional)
14. **Cache management** - Implementation detail
15. **Dataset command** - Evaluation-specific
16. **Validate command** - Quality assurance
17. **Config command** - Convenience feature
18. **Batch command** - Performance optimization
19. **URL input** - Niche use case

## Recommendations

1. **Start with Critical**: Test eval, error handling, edge cases, JSON formats
2. **Then High Priority**: File I/O, pipeline, cross-doc, query, integration
3. **Document Findings**: Create test results for each command
4. **Identify Issues**: Note any bugs, UX problems, or missing features
5. **Create Examples**: Build example workflows for common use cases

## Next Steps

1. Test eval command with real gold annotations
2. Examine JSON output structure in detail
3. Test file input/output workflows
4. Test error handling scenarios
5. Test edge cases (unicode, long text, empty input)
6. Test integration workflows (extract → enhance → cross-doc)
7. Document findings and any issues discovered

