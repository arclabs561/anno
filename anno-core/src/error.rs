//! Error types for anno-core.

use thiserror::Error;

/// Result type for anno-core operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for anno-core operations.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Invalid input provided.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Parse error.
    #[error("Parse error: {0}")]
    Parse(String),

    /// Corpus operation error.
    #[error("Corpus error: {0}")]
    Corpus(String),

    /// Track reference error.
    #[error("Track reference error: {0}")]
    TrackRef(String),
}

impl Error {
    /// Create a track reference error.
    #[must_use]
    pub fn track_ref(msg: impl Into<String>) -> Self {
        Self::TrackRef(msg.into())
    }

    /// Create a corpus error.
    #[must_use]
    pub fn corpus(msg: impl Into<String>) -> Self {
        Self::Corpus(msg.into())
    }

    /// Create an invalid input error.
    #[must_use]
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create a parse error.
    #[must_use]
    pub fn parse(msg: impl Into<String>) -> Self {
        Self::Parse(msg.into())
    }
}
