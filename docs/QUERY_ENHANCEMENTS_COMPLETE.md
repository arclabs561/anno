# Query Command Enhancements - Complete

## Summary

Enhanced the `query` command with additional filtering capabilities, leveraging existing `GroundedDocument` library methods for efficient and consistent querying.

## Implemented Features

### 1. Range Queries ✅
- `--start-offset OFFSET`: Filter signals by start character position
- `--end-offset OFFSET`: Filter signals by end character position
- Uses spatial index (`query_signals_in_range_indexed()`) for efficient range queries
- Both offsets must be provided to enable range filtering

**Example**:
```bash
anno query doc.json --start-offset 0 --end-offset 100
```

### 2. Signal Property Filters ✅
- `--negated`: Filter for negated signals only (e.g., "not a doctor")
- `--quantified`: Filter for signals with quantifiers (universal, existential, etc.)

**Example**:
```bash
anno query doc.json --negated
anno query doc.json --quantified
```

### 3. Relationship Filters ✅
- `--untracked`: Filter for signals not in any track (singletons)
- `--linked`: Filter for signals linked to identities (via tracks)
- `--unlinked`: Filter for signals not linked to identities

**Example**:
```bash
anno query doc.json --untracked
anno query doc.json --linked
anno query doc.json --unlinked
```

### 4. Multi-Filter Combinations ✅
All filters can be combined for powerful queries:

```bash
# Complex query: high-confidence, non-negated, linked entities in a specific range
anno query doc.json \
  --type PER \
  --min-confidence 0.8 \
  --start-offset 0 \
  --end-offset 500 \
  --linked \
  --format json
```

## Implementation Details

### Code Changes
- **File**: `src/bin/anno.rs`
- **Struct**: `QueryArgs` - Added 7 new optional fields
- **Function**: `cmd_query()` - Enhanced filtering logic

### Library Methods Used
- `doc.query_signals_in_range_indexed(start, end)` - Efficient range queries
- `doc.track_for_signal(signal_id)` - Check track membership
- `doc.identity_for_signal(signal_id)` - Check identity linkage
- Signal properties: `signal.negated`, `signal.quantifier`

### Performance
- Range queries use spatial index for O(log n) performance
- Relationship filters use efficient hash map lookups
- All filters are applied in sequence (early filtering reduces work)

## Testing

All existing tests pass:
- `test_query_command_single_doc` ✅
- `test_query_command_entity_search` ✅
- All other CLI tests ✅

## Future Enhancements

### Track Queries (Not Yet Implemented)
- `--tracks`: Query tracks instead of signals
- `--canonical TEXT`: Filter tracks by canonical surface
- `--min-signals N`: Filter tracks with minimum signal count
- `--has-kb-identity`: Filter tracks with KB identities

### Identity Queries (Not Yet Implemented)
- `--identities`: Query identities instead of signals
- `--canonical-id ID`: Filter by KB canonical ID
- `--alias TEXT`: Filter by alias
- `--source kb|placeholder`: Filter by identity source

### Text-Based Queries (Not Yet Implemented)
- `--contains TEXT`: Find signals containing text
- `--overlap START:END`: Find overlapping signals
- `--near TEXT --context N`: Find signals near specific text

### Advanced Relationship Queries (Not Yet Implemented)
- `--in-track TRACK_ID`: Filter signals by track membership
- `--has-identity IDENTITY_ID`: Filter by identity
- `--co-occur --range START END`: Find co-occurring entities
- `--same-identity ID`: Find all signals linked to same identity

## Usage Examples

### Basic Filtering
```bash
# Type and confidence
anno query doc.json --type PER --min-confidence 0.7

# Range query
anno query doc.json --start-offset 100 --end-offset 200

# Negated signals
anno query doc.json --negated
```

### Combined Filters
```bash
# High-confidence, linked entities in a range
anno query doc.json \
  --type ORG \
  --min-confidence 0.9 \
  --start-offset 0 \
  --end-offset 1000 \
  --linked \
  --format json > results.json
```

### Relationship Queries
```bash
# Find untracked signals (singletons)
anno query doc.json --untracked

# Find signals with KB identities
anno query doc.json --linked

# Find signals without KB identities
anno query doc.json --unlinked
```

## Benefits

1. **Better Library Utilization**: Uses existing `GroundedDocument` methods instead of manual filtering
2. **More Powerful Queries**: Access to signal properties and relationships
3. **Consistent API**: All queries use library methods
4. **Performance**: Leverages spatial index for range queries
5. **Extensibility**: Easy to add new query types

## Status

✅ **Complete**: Range queries, signal property filters, relationship filters
⏳ **Future**: Track queries, identity queries, text-based queries, advanced relationships

