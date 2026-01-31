//! Exit codes for CLI commands - semantic codes for pipeline orchestration
//!
//! Pipeline orchestrators (make, snakemake, airflow) depend on exit codes.
//! These are semantic, not just 0/1.

use std::process::ExitCode;

/// Success - all operations completed
pub const SUCCESS: u8 = 0;

/// General/unknown error
pub const ERROR_GENERAL: u8 = 1;

/// Invalid command-line arguments
pub const ERROR_ARGS: u8 = 2;

/// Input file/directory not found
pub const ERROR_INPUT_NOT_FOUND: u8 = 3;

/// Model or knowledge base not found/failed to load
pub const ERROR_MODEL_NOT_FOUND: u8 = 4;

/// Partial success - some documents/items failed, see stderr
pub const ERROR_PARTIAL: u8 = 5;

/// All documents/items failed
pub const ERROR_ALL_FAILED: u8 = 6;

/// Out of memory / resource exhaustion
pub const ERROR_OOM: u8 = 7;

/// Operation timed out
pub const ERROR_TIMEOUT: u8 = 8;

/// Convert exit code constant to ExitCode
pub fn exit(code: u8) -> ExitCode {
    ExitCode::from(code)
}

/// Result type for CLI operations that tracks partial failures.
///
/// Used by batch commands to accumulate success/failure counts and
/// produce appropriate exit codes for pipeline orchestration.
#[derive(Debug, Default)]
pub struct BatchResult {
    /// Number of items that processed successfully
    pub succeeded: usize,
    /// Number of items that failed processing
    pub failed: usize,
    /// Error messages for failed items (for diagnostics)
    pub errors: Vec<String>,
}

impl BatchResult {
    /// Create a new empty batch result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful item.
    pub fn success(&mut self) {
        self.succeeded += 1;
    }

    /// Record a failed item with an error message.
    pub fn failure(&mut self, error: String) {
        self.failed += 1;
        self.errors.push(error);
    }

    /// Get the appropriate exit code for this result.
    ///
    /// Returns `SUCCESS` if all items succeeded, `ERROR_ALL_FAILED` if none
    /// succeeded, or `ERROR_PARTIAL` for mixed results.
    pub fn exit_code(&self) -> ExitCode {
        if self.failed == 0 {
            exit(SUCCESS)
        } else if self.succeeded == 0 {
            exit(ERROR_ALL_FAILED)
        } else {
            exit(ERROR_PARTIAL)
        }
    }

    /// Print a summary of results to stderr if there were any failures.
    pub fn print_summary(&self) {
        if self.failed > 0 {
            eprintln!(
                "Processed: {} succeeded, {} failed",
                self.succeeded, self.failed
            );
            for (i, err) in self.errors.iter().take(10).enumerate() {
                eprintln!("  [{}] {}", i + 1, err);
            }
            if self.errors.len() > 10 {
                eprintln!("  ... and {} more errors", self.errors.len() - 10);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_result_all_success() {
        let mut result = BatchResult::new();
        result.success();
        result.success();
        assert_eq!(result.exit_code(), ExitCode::from(SUCCESS));
    }

    #[test]
    fn test_batch_result_all_failed() {
        let mut result = BatchResult::new();
        result.failure("error 1".into());
        result.failure("error 2".into());
        assert_eq!(result.exit_code(), ExitCode::from(ERROR_ALL_FAILED));
    }

    #[test]
    fn test_batch_result_partial() {
        let mut result = BatchResult::new();
        result.success();
        result.failure("error".into());
        assert_eq!(result.exit_code(), ExitCode::from(ERROR_PARTIAL));
    }
}
