# MCP Tool Usage Enhancements

## Summary

Explored MCP tools to enhance codebase understanding and identify improvement opportunities.

## Tools Explored

### 1. Context7 - Library Documentation ✅
**Used**: Looked up `clap` documentation for CLI best practices
**Result**: Access to up-to-date API documentation and patterns

### 2. Perplexity - Research ✅
**Used**: Researched Rust CLI best practices and UX patterns
**Result**: Current best practices for error handling and UX

### 3. Codebase Analysis ✅
**Used**: Comprehensive search for enhancement opportunities
**Result**: Identified query command enhancement opportunities

## Key Findings

### Query Command Enhancement Opportunities

**Current Capabilities**:
- Type filtering (`--type`)
- Confidence filtering (`--min-confidence`)
- Range queries (`--start-offset`, `--end-offset`)

**Available But Unused**:
- Track queries (canonical surface, entity type, signal count)
- Identity queries (KB linking, aliases, source)
- Text-based queries (contains, near, context)
- Relationship queries (co-occurrence, same identity)
- Signal properties (negation, quantifiers, hierarchical)

### Library Methods Underutilized

**GroundedDocument Methods**:
- `confident_signals(threshold)` - Used
- `signals_with_label(label)` - Used
- `tracks()` - Not used in query
- `identities()` - Not used in query
- `get_track(id)` - Not used
- `get_identity(id)` - Not used

**Signal Properties**:
- `negated` - Available but not filterable
- `quantifier` - Available but not filterable
- `hierarchical` - Available but not filterable

## Recommendations

### High Priority
1. Add track queries to query command
2. Add identity queries to query command
3. Add text-based queries (contains, near)

### Medium Priority
4. Add multi-filter combinations
5. Add relationship queries
6. Add signal property filters (negation, quantifiers)

### Low Priority
7. Add advanced spatial queries
8. Add complex relationship traversal

## Tool Usage Strategy

### Going Forward
1. **Context7**: Use for library documentation before implementing
2. **Perplexity**: Use for research on best practices
3. **ast-grep**: Use for finding code patterns
4. **GitHub MCP**: Use for finding similar implementations

### Current Status
- ✅ Good foundation of tool usage
- ⚠️ Opportunity to leverage MCP tools more
- ✅ Identified clear enhancement opportunities

