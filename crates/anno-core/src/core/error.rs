//! Error types for `anno::core`.

use thiserror::Error;

/// Result type for `anno::core` operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for `anno::core` operations.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = Error::InvalidInput("bad data".into());
        assert_eq!(e.to_string(), "Invalid input: bad data");

        let e = Error::parse("unexpected token");
        assert_eq!(e.to_string(), "Parse error: unexpected token");

        let e = Error::corpus("document not found");
        assert_eq!(e.to_string(), "Corpus error: document not found");

        let e = Error::track_ref("invalid track ID");
        assert_eq!(e.to_string(), "Track reference error: invalid track ID");
    }

    #[test]
    fn test_error_constructors() {
        let e = Error::invalid_input("test");
        assert!(matches!(e, Error::InvalidInput(_)));

        let e = Error::parse("test");
        assert!(matches!(e, Error::Parse(_)));

        let e = Error::corpus("test");
        assert!(matches!(e, Error::Corpus(_)));

        let e = Error::track_ref("test");
        assert!(matches!(e, Error::TrackRef(_)));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
        assert!(err.to_string().contains("file missing"));
    }
}
