# CLI Design Refinement

## Core Principles

1. **Composability over Monoliths**: Small, focused commands that compose well
2. **GroundedDocument as Universal Format**: All commands work with/on GroundedDocument
3. **Transparent Caching**: Cache automatically, invalidate intelligently
4. **Progressive Enhancement**: Start simple, add layers incrementally
5. **Consistent Patterns**: Same input/output patterns across commands

## Command Architecture

### 1. Pipeline Command (Unified Workflow)

**Design Decision**: Single command that orchestrates multiple steps, but each step is also available separately.

```bash
# Full pipeline in one command
anno pipeline "text1" "text2" --coref --link-kb --cross-doc

# Or with files
anno pipeline --files doc1.txt doc2.txt --coref --cross-doc

# Or directory
anno pipeline --dir ./docs --coref --link-kb --cross-doc --output clusters.json
```

**Implementation**: 
- Internally calls `extract`, `enhance`, `cross-doc` in sequence
- Uses temporary files or in-memory GroundedDocuments
- Shows progress for each step
- Can be interrupted and resumed (with caching)

### 2. Enhance Command (Incremental Building)

**Design Decision**: Take existing GroundedDocument, add layers to it.

```bash
# Start with extraction
anno extract "text" --export doc.json

# Add coreference
anno enhance doc.json --coref --export doc-with-coref.json

# Add KB linking
anno enhance doc-with-coref.json --link-kb --export doc-full.json
```

**Implementation**:
- Load GroundedDocument from file/stdin
- Apply requested enhancements (coref, link-kb)
- Output enhanced document
- Preserves existing signals/tracks/identities

### 3. Query Command (Unified Query Interface)

**Design Decision**: Simple query language that works on both single docs and clusters.

```bash
# Query single document
anno query doc.json --type PER --min-confidence 0.8

# Query cross-doc clusters
anno query clusters.json --entity "Apple Inc" --format tree

# Filter clusters
anno query clusters.json --filter "type=ORG AND confidence>0.7 AND cross-doc"
```

**Query Language** (simple, extensible):
- `--type TYPE`: Filter by entity type
- `--entity TEXT`: Find specific entity
- `--min-confidence FLOAT`: Minimum confidence
- `--filter EXPR`: Simple expression language
  - Operators: `=`, `!=`, `>`, `<`, `>=`, `<=`, `AND`, `OR`, `NOT`
  - Fields: `type`, `confidence`, `cross-doc`, `doc-count`, `cluster-size`

### 4. Config Command (Workflow Management)

**Design Decision**: Simple TOML config files, with command-line override.

```toml
# .anno-config.toml
[default]
model = "gliner"
coref = true
link_kb = true
threshold = 0.7

[pipeline]
steps = ["extract", "coref", "link-kb", "cross-doc"]
output_format = "grounded"

[cache]
enabled = true
directory = ".anno-cache"
```

```bash
# Use config
anno pipeline --dir ./docs  # Uses .anno-config.toml

# Override config
anno pipeline --dir ./docs --model stacked  # Overrides model from config

# Create config from current settings
anno config save my-workflow --model gliner --coref --link-kb
```

### 5. Cache Command (Transparent Caching)

**Design Decision**: Automatic caching with smart invalidation.

```bash
# Automatic (transparent)
anno extract "text"  # Caches to .anno-cache/text-hash.json

# Manual cache management
anno cache clear
anno cache list
anno cache stats
anno cache invalidate --model gliner  # Invalidate all gliner results
```

**Cache Strategy**:
- Key: `{model}-{text-hash}-{options-hash}`
- Invalidate on: model change, text change, option change
- Store: GroundedDocument JSON
- Location: `.anno-cache/` (configurable)

### 6. Compare Command (Diff and Comparison)

**Design Decision**: Compare documents, models, or clusters.

```bash
# Compare two documents
anno compare doc1.json doc2.json --format diff

# Compare models on same text
anno compare-models "text" --models stacked gliner --format table

# Compare clusters
anno compare clusters1.json clusters2.json --format summary
```

**Output Formats**:
- `diff`: Show what changed (added/removed/modified entities)
- `table`: Side-by-side comparison
- `summary`: Aggregate statistics

### 7. Batch Command (Efficient Processing)

**Design Decision**: Process multiple documents efficiently with parallel support.

```bash
# Batch process directory
anno batch --dir ./docs --coref --link-kb --parallel 4 --progress

# With caching
anno batch --dir ./docs --cache --parallel 4

# Stream from stdin
cat docs.jsonl | anno batch --stdin --coref
```

**Features**:
- Parallel processing (configurable workers)
- Progress indicators
- Automatic caching
- Resume on failure

## Implementation Strategy

### Phase 1: Core Infrastructure
1. Enhance command (incremental building)
2. Cache infrastructure (transparent caching)
3. Progress indicators (using indicatif)

### Phase 2: Workflow Commands
4. Pipeline command (orchestration)
5. Batch command (parallel processing)
6. Config command (workflow management)

### Phase 3: Analysis Commands
7. Query command (unified querying)
8. Compare command (diff/comparison)

### Phase 4: Refinements
9. Better hierarchy awareness in cross-doc
10. Interactive mode (optional, low priority)

## Design Decisions

### Why Separate Commands vs. One Monolith?

**Separate commands** allow:
- Incremental building (start simple, add complexity)
- Reuse of intermediate results
- Better error handling (fail fast at each step)
- Easier testing (test each command independently)

**Pipeline command** provides:
- Convenience for common workflows
- Single command for full processing
- Better progress feedback across steps

### Why GroundedDocument as Universal Format?

- Already serializable/deserializable
- Contains full hierarchy (signals → tracks → identities)
- Works for both single-doc and cross-doc
- Extensible (can add new fields without breaking)

### Why Simple Query Language?

- Start simple (type, confidence, entity name)
- Extensible (add operators/fields as needed)
- No need for full SQL (overkill for this use case)
- Easy to parse and validate

### Cache Invalidation Strategy

- **Model change**: Invalidate all results for that model
- **Text change**: Invalidate specific text hash
- **Option change**: Invalidate based on option hash
- **Manual**: `anno cache invalidate` command

Cache key format: `{model}-{text-sha256}-{options-sha256}.json`

## Testing Strategy

1. **Unit tests**: Each command function independently
2. **Integration tests**: Full workflows (extract → enhance → cross-doc)
3. **Property tests**: Cache invalidation, query filtering
4. **E2E tests**: Real-world scenarios with actual files

## Migration Path

Existing commands remain unchanged. New commands are additive:
- `extract`, `debug`, `eval` - unchanged
- `enhance` - new, works with existing exports
- `pipeline` - new, orchestrates existing commands
- `query`, `compare`, `cache`, `config` - new utilities

This ensures backward compatibility while adding new capabilities.

