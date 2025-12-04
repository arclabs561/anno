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

/// Convert HuggingFace API errors to our Error type.
/// Only available when hf-hub is in the dependency tree (onnx or candle features).
#[cfg(any(feature = "onnx", feature = "candle"))]
impl From<hf_hub::api::sync::ApiError> for Error {
    fn from(err: hf_hub::api::sync::ApiError) -> Self {
        Error::Retrieval(format!("{}", err))
    }
}
