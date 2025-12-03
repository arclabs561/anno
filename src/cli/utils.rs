//! Utility functions for CLI commands

use std::io::{self, Read};

/// Get input text from various sources (text arg, file, or stdin)
pub fn get_input_text(
    text: &Option<String>,
    file: &Option<&str>,
    positional: &[String],
) -> Result<String, String> {
    if let Some(t) = text {
        Ok(t.clone())
    } else if let Some(f) = file {
        read_input_file(f)
    } else if !positional.is_empty() {
        Ok(positional.join(" "))
    } else {
        // Read from stdin
        let mut buffer = String::new();
        io::stdin()
            .read_to_string(&mut buffer)
            .map_err(|e| format!("Failed to read from stdin: {}", e))?;
        Ok(buffer)
    }
}

/// Read input from file
pub fn read_input_file(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read file {}: {}", path, e))
}

/// Parse GroundedDocument from JSON
pub fn parse_grounded_document(json: &str) -> Result<anno::grounded::GroundedDocument, String> {
    serde_json::from_str(json)
        .map_err(|e| format!("Failed to parse GroundedDocument JSON: {}", e))
}

