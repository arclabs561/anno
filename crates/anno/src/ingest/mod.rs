//! Document ingestion and preparation.
//!
//! Handles fetching content from URLs, cleaning text, and preparing documents
//! for entity extraction.

/// Document format conversion (PDF, DOCX, HTML, etc.).
pub mod converter;
pub mod preprocessor;
pub mod url_resolver;

pub use preprocessor::{DocumentPreprocessor, PreparedDocument};
pub use url_resolver::{CompositeResolver, ResolvedContent, UrlResolver};
