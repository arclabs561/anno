# CLI Harmonization Progress

## Summary

Started harmonizing CLI command patterns to improve consistency, maintainability, and user experience.

## Completed

### 1. Helper Functions Created ✅
- `read_input_file(path: &str) -> Result<String, String>` - Unified file reading with consistent error messages
- `parse_grounded_document(json: &str) -> Result<GroundedDocument, String>` - Unified JSON parsing
- `write_output(content: &str, path: Option<&str>) -> Result<(), String>` - Unified output writing with automatic directory creation
- `format_error(operation: &str, details: &str) -> String` - Consistent error message formatting
- `log_info(msg: &str, quiet: bool)` - Quiet-aware info logging
- `log_verbose(msg: &str, verbose: bool)` - Verbose logging
- `log_success(msg: &str, quiet: bool)` - Success message logging with color

### 2. Commands Updated ✅
- `cmd_query` - Now uses `read_input_file()`, `parse_grounded_document()`, and `write_output()`
- `cmd_enhance` - Now uses helpers for input/output and logging
- `get_input_text()` - Now uses `read_input_file()` internally

### 3. Error Messages Standardized ✅
- Consistent format: "Failed to {operation}: {details}"
- All file operations use unified helpers
- JSON parsing uses unified helper

## In Progress

### Remaining Areas to Harmonize
1. **Output Format Handling** - Still duplicated across commands
2. **File Reading** - Some commands still use `fs::read_to_string()` directly
3. **JSON Serialization** - Some commands still serialize directly
4. **Logging** - Some commands still use `eprintln!` directly
5. **Validation Error Reporting** - Needs unified helper

## Benefits

1. **Consistency**: Commands now use same patterns for common operations
2. **Maintainability**: Changes to error handling/logging affect all commands
3. **User Experience**: Consistent error messages and behavior
4. **Code Quality**: Reduced duplication, better error handling

## Next Steps

1. Continue harmonizing remaining commands (`cmd_pipeline`, `cmd_crossdoc`, etc.)
2. Create format serialization helpers
3. Standardize validation error reporting
4. Add more logging helpers for progress reporting

