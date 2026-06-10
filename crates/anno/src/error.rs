//! Error types for anno.

use thiserror::Error;

/// Result type for anno operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for anno operations.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Model initialization failed.
    #[error("Model initialization failed: {0}")]
    ModelInit(String),

    /// Model inference failed.
    #[error("Inference failed: {0}")]
    Inference(String),

    /// Invalid input provided.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Dataset loading/parsing error.
    #[error("Dataset error: {0}")]
    Dataset(String),

    /// Feature not available.
    #[error("Feature not available: {0}")]
    FeatureNotAvailable(String),

    /// Parse error.
    #[error("Parse error: {0}")]
    Parse(String),

    /// Evaluation error.
    #[error("Evaluation error: {0}")]
    Evaluation(String),

    /// Model retrieval error (downloading from HuggingFace).
    #[error("Retrieval error: {0}")]
    Retrieval(String),

    /// Candle ML error (when candle feature enabled).
    #[cfg(feature = "candle")]
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Corpus operation error.
    #[error("Corpus error: {0}")]
    Corpus(String),

    /// Track reference error.
    #[error("Track reference error: {0}")]
    TrackRef(String),

    /// Error from `anno::core` (stable data-model layer).
    ///
    /// Enables `?` propagation from functions returning `crate::Result`.
    #[error(transparent)]
    Core(crate::core::error::Error),

    /// Backend-local error.
    #[error("backend error: {0}")]
    Backend(String),
}

impl Error {
    /// Create a model initialization error.
    pub fn model_init(msg: impl Into<String>) -> Self {
        Error::ModelInit(msg.into())
    }

    /// Create an inference error.
    pub fn inference(msg: impl Into<String>) -> Self {
        Error::Inference(msg.into())
    }

    /// Create an invalid input error.
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Error::InvalidInput(msg.into())
    }

    /// Create a dataset error.
    pub fn dataset(msg: impl Into<String>) -> Self {
        Error::Dataset(msg.into())
    }

    /// Create a feature not available error.
    pub fn feature_not_available(feature: impl Into<String>) -> Self {
        Error::FeatureNotAvailable(feature.into())
    }

    /// Create a parse error.
    pub fn parse(msg: impl Into<String>) -> Self {
        Error::Parse(msg.into())
    }

    /// Create an evaluation error.
    pub fn evaluation(msg: impl Into<String>) -> Self {
        Error::Evaluation(msg.into())
    }

    /// Create a retrieval error.
    pub fn retrieval(msg: impl Into<String>) -> Self {
        Error::Retrieval(msg.into())
    }

    /// Create a corpus error.
    pub fn corpus(msg: impl Into<String>) -> Self {
        Error::Corpus(msg.into())
    }

    /// Create a track reference error.
    pub fn track_ref(msg: impl Into<String>) -> Self {
        Error::TrackRef(msg.into())
    }
}

/// Convert `anno::core` errors into `anno::Error`, enabling `?` propagation
/// across the module boundary.
impl From<crate::core::error::Error> for Error {
    fn from(err: crate::core::error::Error) -> Self {
        Error::Core(err)
    }
}

/// Convert HuggingFace API errors to our Error type.
/// Only available when hf-hub is in the dependency tree (onnx or candle features).
#[cfg(any(feature = "onnx", feature = "candle"))]
impl From<hf_hub::api::sync::ApiError> for Error {
    fn from(err: hf_hub::api::sync::ApiError) -> Self {
        Error::Retrieval(format!("{}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_constructors() {
        let e = Error::model_init("test model init");
        assert!(e.to_string().contains("Model initialization failed"));
        assert!(e.to_string().contains("test model init"));

        let e = Error::inference("test inference");
        assert!(e.to_string().contains("Inference failed"));

        let e = Error::invalid_input("test input");
        assert!(e.to_string().contains("Invalid input"));

        let e = Error::dataset("test dataset");
        assert!(e.to_string().contains("Dataset error"));

        let e = Error::feature_not_available("test feature");
        assert!(e.to_string().contains("Feature not available"));

        let e = Error::parse("test parse");
        assert!(e.to_string().contains("Parse error"));

        let e = Error::evaluation("test eval");
        assert!(e.to_string().contains("Evaluation error"));

        let e = Error::retrieval("test retrieval");
        assert!(e.to_string().contains("Retrieval error"));

        let e = Error::corpus("test corpus");
        assert!(e.to_string().contains("Corpus error"));

        let e = Error::track_ref("test track");
        assert!(e.to_string().contains("Track reference error"));
    }

    #[test]
    fn test_error_debug_display() {
        let e = Error::ModelInit("debug test".to_string());
        let debug = format!("{:?}", e);
        assert!(debug.contains("ModelInit"));
        assert!(debug.contains("debug test"));

        let display = format!("{}", e);
        assert!(display.contains("Model initialization failed"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let e: Error = io_err.into();
        assert!(e.to_string().contains("IO error"));
    }

    #[test]
    fn test_core_error_conversion() {
        let core_err = crate::core::error::Error::parse("bad token");
        let err: Error = core_err.into();
        assert!(matches!(err, Error::Core(_)));
        assert!(err.to_string().contains("Parse error"));
        assert!(err.to_string().contains("bad token"));
    }

    #[test]
    fn test_core_error_question_mark_propagation() {
        fn inner() -> crate::core::error::Result<()> {
            Err(crate::core::error::Error::corpus("missing doc"))
        }
        fn outer() -> Result<()> {
            inner()?; // exercises From<crate::Error>
            Ok(())
        }
        let err = outer().unwrap_err();
        assert!(err.to_string().contains("Corpus error"));
    }

    #[test]
    fn test_result_type_alias() {
        fn returns_result() -> Result<i32> {
            Ok(42)
        }
        assert_eq!(returns_result().unwrap(), 42);

        fn returns_error() -> Result<i32> {
            Err(Error::invalid_input("bad"))
        }
        assert!(returns_error().is_err());
    }
}
