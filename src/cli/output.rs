//! Output formatting utilities for CLI commands

use std::io::{self, Write};

/// Format error message for display
pub fn format_error(operation: &str, details: &str) -> String {
    format!("ERROR: {} - {}", operation, details)
}

/// Log info message (respects quiet flag)
pub fn log_info(msg: &str, quiet: bool) {
    if !quiet {
        eprintln!("{}", msg);
    }
}

/// Write output to file or stdout
pub fn write_output(content: &str, path: Option<&str>) -> Result<(), String> {
    if let Some(path) = path {
        std::fs::write(path, content)
            .map_err(|e| format!("Failed to write to {}: {}", path, e))?;
    } else {
        print!("{}", content);
        io::stdout().flush().map_err(|e| format!("Failed to flush stdout: {}", e))?;
    }
    Ok(())
}

/// Format file size in human-readable format
pub fn format_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    
    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[unit_idx])
    } else {
        format!("{:.2} {}", size, UNITS[unit_idx])
    }
}

