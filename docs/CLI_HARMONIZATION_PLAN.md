# CLI Harmonization Plan

## Goal
Standardize patterns across all CLI commands for consistency, maintainability, and better user experience.

## Areas for Harmonization

### 1. Input Handling ✅
**Current State**: Mixed patterns
- Some commands use `get_input_text()` helper
- Others read files directly with `fs::read_to_string()`
- Inconsistent error messages

**Target State**: 
- All commands use unified input helpers
- Consistent error message format
- Support for URL, file, text, stdin uniformly

### 2. Output Handling ⚠️
**Current State**: Inconsistent
- Some write to file with `fs::write()`
- Others print to stdout
- Inconsistent directory creation
- Mixed error handling

**Target State**:
- Unified `write_output()` helper
- Automatic directory creation
- Consistent error messages

### 3. Format Handling ⚠️
**Current State**: Duplicated logic
- Each command has its own `match args.format` block
- Similar patterns repeated
- Inconsistent format support

**Target State**:
- Shared format serialization helpers
- Consistent format support across commands
- Reusable format conversion functions

### 4. Error Messages ⚠️
**Current State**: Inconsistent style
- Some use `format!("Failed to read {}: {}", file, e)`
- Others use different patterns
- Inconsistent use of color codes

**Target State**:
- Standardized error message format
- Consistent use of `color()` helper
- Clear, actionable error messages

### 5. Quiet/Verbose Flags ⚠️
**Current State**: Inconsistent usage
- Some commands check `args.quiet` before printing
- Others use `eprintln!` directly
- Inconsistent verbose output

**Target State**:
- Unified logging helpers
- Consistent quiet/verbose behavior
- Standardized progress reporting

### 6. JSON Parsing ⚠️
**Current State**: Direct usage
- Commands parse JSON directly with `serde_json::from_str`
- Inconsistent error messages
- No validation helpers

**Target State**:
- Shared JSON parsing helpers
- Consistent error messages
- Validation utilities

## Implementation Strategy

### Phase 1: Create Helper Functions
1. `read_input_file(path: &str) -> Result<String, String>` - Unified file reading
2. `parse_grounded_document(json: &str) -> Result<GroundedDocument, String>` - JSON parsing
3. `write_output(content: &str, path: Option<&str>) -> Result<(), String>` - Output writing
4. `log_info(msg: &str, quiet: bool)` - Quiet-aware logging
5. `log_verbose(msg: &str, verbose: bool)` - Verbose logging
6. `format_error(operation: &str, details: &str) -> String` - Error formatting

### Phase 2: Refactor Commands
1. Update `cmd_extract` to use helpers
2. Update `cmd_debug` to use helpers
3. Update `cmd_query` to use helpers
4. Update `cmd_enhance` to use helpers
5. Update `cmd_pipeline` to use helpers
6. Update remaining commands

### Phase 3: Format Helpers
1. Create `format_signals()` helper
2. Create `format_grounded_document()` helper
3. Create `format_cross_doc_clusters()` helper
4. Standardize format output across commands

## Benefits

1. **Consistency**: All commands behave similarly
2. **Maintainability**: Changes in one place affect all commands
3. **User Experience**: Predictable behavior across commands
4. **Code Quality**: Reduced duplication, better error handling
5. **Testing**: Easier to test common functionality

## Status

- ✅ Input handling: `get_input_text()` exists but not used everywhere
- ⏳ Output handling: Needs unified helper
- ⏳ Format handling: Needs shared helpers
- ⏳ Error messages: Needs standardization
- ⏳ Quiet/verbose: Needs unified logging
- ⏳ JSON parsing: Needs helper functions

