# Query Command Enhancement Opportunities

## Current Implementation

The `query` command currently supports:
- Loading GroundedDocument from file
- Filtering by entity type (`--type`)
- Filtering by minimum confidence (`--min-confidence`)
- Range queries with `--start-offset` and `--end-offset` (uses spatial index)

## Available Library Methods Not Yet Used

### 1. Advanced Signal Filtering ✅ (Partially Used)

**Available Methods**:
- `doc.signals()` - Iterator over all signals
- `doc.confident_signals(threshold)` - Signals above confidence threshold
- `doc.signals_with_label(label)` - Signals of specific type
- `doc.get_signal(id)` - Get signal by ID

**Current Usage**: Basic filtering implemented
**Enhancement Opportunity**: 
- Combine multiple filters (type + confidence + range)
- Filter by negation status
- Filter by quantifier presence
- Filter by hierarchical relationships

### 2. Track Queries ⚠️ (Not Used)

**Available Methods**:
- `doc.tracks()` - Iterator over all tracks
- `doc.get_track(id)` - Get track by ID
- `doc.track_ref(id)` - Get TrackRef by ID
- Track properties: `canonical_surface`, `entity_type`, `signals`

**Enhancement Opportunity**:
- Query tracks by canonical surface (fuzzy matching)
- Query tracks by entity type
- Query tracks with minimum signal count
- Query tracks linked to identities

### 3. Identity Queries ⚠️ (Not Used)

**Available Methods**:
- `doc.identities()` - Iterator over all identities
- `doc.get_identity(id)` - Get identity by ID
- Identity properties: `canonical_id`, `aliases`, `entity_type`, `source`

**Enhancement Opportunity**:
- Query identities by canonical ID (KB linking)
- Query identities by alias
- Query identities by source (KB vs placeholder)
- Query identities with linked tracks

### 4. Spatial Index Queries ✅ (Used)

**Current Usage**: Range queries with `--start-offset` and `--end-offset`
**Available Methods**:
- `doc.build_text_index()` - Build spatial index
- `query_signals_in_range_indexed()` - Query signals in text range

**Enhancement Opportunity**:
- Query by text substring (find signals containing text)
- Query by context window (signals near specific text)
- Query by sentence boundaries

### 5. Cross-Entity Relationships ⚠️ (Not Used)

**Available Methods**:
- `doc.signal_to_track` - Map signal to track
- `doc.track_to_identity` - Map track to identity
- Track signals - Get all signals in a track
- Identity tracks - Get all tracks for an identity

**Enhancement Opportunity**:
- Query signals that are part of tracks
- Query tracks that have KB identities
- Query all mentions of a specific entity (track + identity)
- Query co-occurring entities (signals in same range)

## Proposed Enhancements

### Enhancement 1: Multi-Filter Queries

```bash
# Combine multiple filters
anno query doc.json --type PER --min-confidence 0.8 --not-negated

# Filter by track membership
anno query doc.json --in-track T0

# Filter by identity
anno query doc.json --has-identity Q76
```

### Enhancement 2: Track Queries

```bash
# Query tracks
anno query doc.json --tracks --canonical "Barack Obama"

# Query tracks with minimum signals
anno query doc.json --tracks --min-signals 2

# Query tracks with KB identities
anno query doc.json --tracks --has-kb-identity
```

### Enhancement 3: Identity Queries

```bash
# Query identities
anno query doc.json --identities --canonical-id Q76

# Query identities by alias
anno query doc.json --identities --alias "Obama"

# Query KB-linked identities
anno query doc.json --identities --source kb
```

### Enhancement 4: Text-Based Queries

```bash
# Find signals containing text
anno query doc.json --contains "Apple"

# Find signals in context
anno query doc.json --near "founded" --context 50

# Find signals in sentence
anno query doc.json --sentence 0
```

### Enhancement 5: Relationship Queries

```bash
# Find all mentions of an entity
anno query doc.json --entity "Barack Obama"

# Find co-occurring entities
anno query doc.json --co-occur --range 0 100

# Find entities linked to same identity
anno query doc.json --same-identity Q76
```

## Implementation Priority

### High Priority
1. ✅ Multi-filter combinations (type + confidence + range)
2. ✅ Signal property filters (`--negated`, `--quantified`, `--untracked`, `--linked`, `--unlinked`)
3. ✅ Range queries (`--start-offset`, `--end-offset`)
4. ⏳ Track queries (canonical surface, entity type)
5. ⏳ Text-based queries (contains, near)

### Medium Priority
6. ⏳ Identity queries (KB linking, aliases)
7. ⏳ Relationship queries (co-occurrence, same identity)

### Low Priority
8. ⏳ Advanced spatial queries (sentence boundaries, context windows)
9. ⏳ Complex relationship traversal

## Current Status

**Basic Functionality**: ✅ Working
- Type filtering (`--type`)
- Confidence filtering (`--min-confidence`)
- Entity name filtering (`--entity`)
- Range queries (`--start-offset`, `--end-offset`) - uses spatial index
- Signal property filters (`--negated`, `--quantified`)
- Relationship filters (`--untracked`, `--linked`, `--unlinked`)

**Enhancement Opportunities**: ⚠️ Some available
- Track queries (canonical surface, entity type, min signals)
- Identity queries (KB linking, aliases)
- Text-based queries (contains, near)
- Advanced relationship queries (co-occurrence, same identity)

