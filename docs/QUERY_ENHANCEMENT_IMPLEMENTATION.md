# Query Command Enhancement Implementation Plan

## Current State Analysis

The `query` command currently supports:
- ✅ Type filtering (`--type`)
- ✅ Confidence filtering (`--min-confidence`)
- ✅ Entity name filtering (`--entity`)
- ✅ Range queries (`--start-offset`, `--end-offset`) - uses spatial index
- ✅ Output formats (human, json, tree)

## Available Library Methods Not Yet Used

### 1. Track Queries
**Methods Available**:
- `doc.tracks()` - Iterator over all tracks
- `doc.get_track(id)` - Get track by ID
- `doc.linked_tracks()` - Tracks with KB identities
- `doc.unlinked_tracks()` - Tracks without identities
- `doc.track_for_signal(signal_id)` - Get track containing signal

**Enhancement**: Add `--tracks` mode to query tracks instead of signals

### 2. Identity Queries
**Methods Available**:
- `doc.identities()` - Iterator over all identities
- `doc.get_identity(id)` - Get identity by ID
- `doc.identity_for_track(track_id)` - Get identity for track
- `doc.identity_for_signal(signal_id)` - Get identity for signal

**Enhancement**: Add `--identities` mode to query identities

### 3. Signal Property Filters
**Properties Available**:
- `signal.negated` - Boolean
- `signal.quantifier` - Option<Quantifier>
- `signal.hierarchical` - Option<HierarchicalConfidence>

**Methods Available**:
- `doc.negated_signals()` - Get negated signals
- `doc.quantified_signals(quantifier)` - Get signals with quantifier
- `doc.confident_signals(threshold)` - Already used

**Enhancement**: Add `--negated`, `--quantifier`, `--hierarchical` filters

### 4. Text-Based Queries
**Methods Available**:
- `doc.signals_in_range(start, end)` - Signals in text range (linear)
- `doc.query_signals_in_range_indexed(start, end)` - Fast range query (spatial index)
- `doc.query_overlapping_signals_indexed(start, end)` - Overlapping signals
- `doc.overlapping_signals(location)` - Overlapping with location

**Enhancement**: 
- Add `--contains TEXT` to find signals containing text
- Add `--overlap START:END` for overlapping queries
- Improve range query UX

### 5. Relationship Queries
**Methods Available**:
- `doc.track_for_signal(signal_id)` - Get track for signal
- `doc.identity_for_track(track_id)` - Get identity for track
- `doc.identity_for_signal(signal_id)` - Get identity for signal
- `doc.untracked_signals()` - Signals not in tracks
- `doc.linked_tracks()` - Tracks with identities
- `doc.unlinked_tracks()` - Tracks without identities

**Enhancement**:
- Add `--in-track TRACK_ID` to filter signals by track
- Add `--has-identity IDENTITY_ID` to filter by identity
- Add `--untracked` to find signals not in tracks
- Add `--linked` / `--unlinked` for track identity status

## Proposed Implementation

### Phase 1: Signal Property Filters (Easy)
Add flags for existing methods:
- `--negated` → use `doc.negated_signals()`
- `--quantifier universal|existential|none|definite|bare` → use `doc.quantified_signals()`
- `--hierarchical` → filter by `signal.hierarchical.is_some()`

### Phase 2: Track Queries (Medium)
Add `--tracks` mode:
- When `--tracks` is set, query tracks instead of signals
- Support `--canonical TEXT` for canonical surface matching
- Support `--min-signals N` for minimum signal count
- Support `--has-identity` to filter tracks with KB identities

### Phase 3: Identity Queries (Medium)
Add `--identities` mode:
- When `--identities` is set, query identities instead of signals
- Support `--canonical-id ID` for KB ID matching
- Support `--alias TEXT` for alias matching
- Support `--source kb|placeholder` for identity source

### Phase 4: Text-Based Queries (Medium)
Enhance text search:
- `--contains TEXT` → find signals where `surface.contains(TEXT)`
- `--overlap START:END` → use `query_overlapping_signals_indexed()`
- Improve `--start-offset` / `--end-offset` UX with better error messages

### Phase 5: Relationship Queries (Hard)
Add relationship filters:
- `--in-track TRACK_ID` → filter signals by track membership
- `--has-identity IDENTITY_ID` → filter by identity
- `--untracked` → use `doc.untracked_signals()`
- `--linked` / `--unlinked` → filter tracks by identity status

## Implementation Priority

### High Priority (Quick Wins)
1. ✅ Signal property filters (`--negated`, `--quantifier`)
2. ✅ Text-based queries (`--contains`, `--overlap`)
3. ✅ Relationship filters (`--untracked`, `--in-track`)

### Medium Priority
4. ⏳ Track queries (`--tracks` mode)
5. ⏳ Identity queries (`--identities` mode)

### Low Priority
6. ⏳ Complex relationship traversal
7. ⏳ Advanced spatial queries (sentence boundaries)

## Code Structure

Current `cmd_query` structure:
```rust
fn cmd_query(args: QueryArgs) -> Result<(), String> {
    // Load GroundedDocument
    // Filter signals by type, confidence, entity
    // Output results
}
```

Proposed structure:
```rust
fn cmd_query(args: QueryArgs) -> Result<(), String> {
    // Load GroundedDocument
    
    // Determine query mode: signals, tracks, or identities
    match args.mode {
        QueryMode::Signals => {
            // Apply signal filters
            // Use library methods: negated_signals(), quantified_signals(), etc.
        }
        QueryMode::Tracks => {
            // Query tracks
            // Filter by canonical, min-signals, has-identity
        }
        QueryMode::Identities => {
            // Query identities
            // Filter by canonical-id, alias, source
        }
    }
    
    // Output results
}
```

## Benefits

1. **Better Library Utilization**: Use existing methods instead of manual filtering
2. **More Powerful Queries**: Access to tracks and identities
3. **Consistent API**: All queries use GroundedDocument methods
4. **Performance**: Leverage spatial index for range queries
5. **Extensibility**: Easy to add new query types

