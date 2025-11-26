//! Error types for anno.

use thiserror::Error;

/// Result type for anno operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for anno operations.
#[derive(Error, Debug)]
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
}

