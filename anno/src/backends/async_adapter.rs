//! Async inference adapters for production web servers.
//!
//! ONNX and Candle inference is blocking. In async runtimes (tokio, actix),
//! blocking the executor thread starves other tasks. This module provides
//! adapters that run inference on a blocking thread pool.
//!
//! # Why This Matters
//!
//! ```text
//! WITHOUT spawn_blocking:
//! ─────────────────────────
//! Request 1 → [ONNX inference 100ms] → Response 1
//!                                      Request 2 → [ONNX 100ms] → Response 2
//!                                                                  Request 3 → ...
//! Total: 300ms for 3 requests (serialized!)
//!
//! WITH spawn_blocking:
//! ────────────────────
//! Request 1 → [spawn_blocking → ONNX] → Response 1
//! Request 2 → [spawn_blocking → ONNX] → Response 2   (parallel!)
//! Request 3 → [spawn_blocking → ONNX] → Response 3
//! Total: ~100ms for 3 requests (parallel on blocking pool)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::backends::async_adapter::{AsyncNER, IntoAsync};
//! use anno::GLiNEROnnx;
//!
//! #[tokio::main]
//! async fn main() {
//!     let model = GLiNEROnnx::new("onnx-community/gliner_small-v2.1").unwrap();
//!     let async_model = model.into_async();
//!     
//!     // Now safe to use in async handlers
//!     let entities = async_model.extract_entities("John works at Apple").await.unwrap();
//! }
//! ```
//!
//! # With Axum
//!
//! ```rust,ignore
//! use axum::{Router, Json, extract::State};
//! use anno::backends::async_adapter::AsyncNER;
//! use std::sync::Arc;
//!
//! async fn extract(
//!     State(model): State<Arc<AsyncNER<GLiNEROnnx>>>,
//!     Json(text): Json<String>,
//! ) -> Json<Vec<Entity>> {
//!     let entities = model.extract_entities(&text).await.unwrap();
//!     Json(entities)
//! }
//! ```

#![cfg(feature = "async-inference")]

use crate::{Entity, Error, Model, Result};
use std::sync::Arc;

/// Async wrapper for any synchronous NER model.
///
/// Wraps a model to run inference on tokio's blocking thread pool,
/// preventing executor starvation in async web servers.
///
/// # Thread Safety
///
/// The inner model is wrapped in `Arc` for cheap cloning across tasks.
/// The model itself must be `Send + Sync` (all `anno` models are).
/// Clone is NOT required on the model - the Arc provides shared ownership.
pub struct AsyncNER<M: Model + Send + Sync + 'static> {
    inner: Arc<M>,
}

// Manual Clone impl - clones the Arc, not the model
impl<M: Model + Send + Sync + 'static> Clone for AsyncNER<M> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<M: Model + Send + Sync + 'static> AsyncNER<M> {
    /// Create a new async wrapper around a model.
    pub fn new(model: M) -> Self {
        Self {
            inner: Arc::new(model),
        }
    }

    /// Create from an existing Arc-wrapped model.
    pub fn from_arc(model: Arc<M>) -> Self {
        Self { inner: model }
    }

    /// Extract entities asynchronously.
    ///
    /// Runs inference on tokio's blocking thread pool to avoid
    /// starving the async executor.
    pub async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let model = Arc::clone(&self.inner);
        let text = text.to_string();

        tokio::task::spawn_blocking(move || model.extract_entities(&text, None))
            .await
            .map_err(|e| Error::Parse(format!("Async join error: {}", e)))?
    }

    /// Extract entities with language hint.
    pub async fn extract_entities_with_lang(
        &self,
        text: &str,
        language: &str,
    ) -> Result<Vec<Entity>> {
        let model = Arc::clone(&self.inner);
        let text = text.to_string();
        let lang = language.to_string();

        tokio::task::spawn_blocking(move || model.extract_entities(&text, Some(&lang)))
            .await
            .map_err(|e| Error::Parse(format!("Async join error: {}", e)))?
    }

    /// Get reference to the underlying model.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Get the Arc-wrapped model (for sharing).
    pub fn inner_arc(&self) -> Arc<M> {
        Arc::clone(&self.inner)
    }

    /// Check if model is available.
    pub fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    /// Get model name.
    pub fn name(&self) -> &'static str {
        self.inner.name()
    }
}

/// Extension trait to convert any model to async.
///
/// Note: This does NOT require Clone - the Arc wrapping handles sharing.
pub trait IntoAsync: Model + Send + Sync + Sized + 'static {
    /// Convert this model into an async-safe wrapper.
    fn into_async(self) -> AsyncNER<Self> {
        AsyncNER::new(self)
    }
}

// Implement for all models (no Clone required!)
impl<M: Model + Send + Sync + 'static> IntoAsync for M {}

// =============================================================================
// Batch Async Inference
// =============================================================================

/// Batch multiple texts for parallel async inference.
///
/// Spawns multiple blocking tasks in parallel, maximizing throughput
/// on multi-core systems.
///
/// # Example
///
/// ```rust,ignore
/// let texts = vec!["John works at Apple", "Mary founded Google"];
/// let results = batch_extract(&async_model, &texts).await?;
/// ```
pub async fn batch_extract<M: Model + Send + Sync + 'static>(
    model: &AsyncNER<M>,
    texts: &[&str],
) -> Result<Vec<Vec<Entity>>> {
    let futures: Vec<_> = texts
        .iter()
        .map(|text| model.extract_entities(text))
        .collect();

    let results = futures::future::join_all(futures).await;

    results.into_iter().collect()
}

/// Batch extract with concurrency limit.
///
/// Use this when you have many texts and want to limit memory usage
/// by controlling the number of concurrent inference tasks.
///
/// # Arguments
///
/// * `model` - The async model wrapper
/// * `texts` - Texts to process
/// * `concurrency` - Maximum concurrent inference tasks
///
/// # Example
///
/// ```rust,ignore
/// // Process 1000 texts, 8 at a time
/// let results = batch_extract_limited(&model, &texts, 8).await?;
/// ```
pub async fn batch_extract_limited<M: Model + Send + Sync + 'static>(
    model: &AsyncNER<M>,
    texts: &[&str],
    concurrency: usize,
) -> Result<Vec<Vec<Entity>>> {
    use tokio::sync::Semaphore;

    let semaphore = Arc::new(Semaphore::new(concurrency));
    let model = model.clone(); // AsyncNER is Clone (wraps Arc)

    let futures: Vec<_> = texts
        .iter()
        .map(|text| {
            let sem = Arc::clone(&semaphore);
            let model = model.clone();
            let text = text.to_string();

            async move {
                let _permit = sem
                    .acquire()
                    .await
                    .map_err(|e| Error::Parse(format!("Semaphore error: {}", e)))?;
                model.extract_entities(&text).await
            }
        })
        .collect();

    let results = futures::future::join_all(futures).await;
    results.into_iter().collect()
}

// =============================================================================
// Zero-Shot Async
// =============================================================================

#[cfg(feature = "onnx")]
use crate::backends::inference::ZeroShotNER;

/// Async wrapper specifically for zero-shot NER models.
///
/// Provides async versions of the `ZeroShotNER` trait methods.
#[cfg(feature = "onnx")]
pub struct AsyncZeroShotNER<M: ZeroShotNER + Send + Sync + 'static> {
    inner: Arc<M>,
}

#[cfg(feature = "onnx")]
impl<M: ZeroShotNER + Send + Sync + 'static> AsyncZeroShotNER<M> {
    /// Create a new async zero-shot wrapper.
    pub fn new(model: M) -> Self {
        Self {
            inner: Arc::new(model),
        }
    }

    /// Extract with custom entity types (async).
    pub async fn extract_with_types(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        let model = Arc::clone(&self.inner);
        let text = text.to_string();
        let types: Vec<String> = entity_types.iter().map(|s| s.to_string()).collect();

        tokio::task::spawn_blocking(move || {
            let type_refs: Vec<&str> = types.iter().map(|s| s.as_str()).collect();
            model.extract_with_types(&text, &type_refs, threshold)
        })
        .await
        .map_err(|e| Error::Parse(format!("Async join error: {}", e)))?
    }

    /// Extract with natural language descriptions (async).
    pub async fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        let model = Arc::clone(&self.inner);
        let text = text.to_string();
        let descs: Vec<String> = descriptions.iter().map(|s| s.to_string()).collect();

        tokio::task::spawn_blocking(move || {
            let desc_refs: Vec<&str> = descs.iter().map(|s| s.as_str()).collect();
            model.extract_with_descriptions(&text, &desc_refs, threshold)
        })
        .await
        .map_err(|e| Error::Parse(format!("Async join error: {}", e)))?
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EntityType, MockModel};

    #[tokio::test]
    async fn test_async_model_basic() {
        let mock = MockModel::new("test")
            .with_entities(vec![Entity::new("Test", EntityType::Person, 0, 4, 0.9)])
            .without_validation();

        let async_model = mock.into_async();

        let entities = async_model
            .extract_entities("Test works at Apple")
            .await
            .unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "Test");
    }

    #[tokio::test]
    async fn test_batch_extract() {
        let mock = MockModel::new("test")
            .with_entities(vec![Entity::new("test", EntityType::Person, 0, 4, 0.9)])
            .without_validation();

        let async_model = mock.into_async();

        let texts = vec!["test one", "test two", "test three"];

        let futures: Vec<_> = texts
            .iter()
            .map(|_| async_model.extract_entities("test"))
            .collect();
        let results: Vec<Result<Vec<Entity>>> = futures::future::join_all(futures).await;

        assert_eq!(results.len(), 3);
        for result in results {
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_into_async_trait() {
        let mock = MockModel::new("trait-test");

        // Test the IntoAsync trait
        let async_model: AsyncNER<MockModel> = mock.into_async();

        assert_eq!(async_model.name(), "trait-test");
        assert!(async_model.is_available());
    }

    #[tokio::test]
    async fn test_async_model_clone() {
        let mock = MockModel::new("clone-test")
            .with_entities(vec![Entity::new("test", EntityType::Person, 0, 4, 0.9)])
            .without_validation();
        let async_model = AsyncNER::new(mock);

        // AsyncNER should be Clone even if inner model is not
        let cloned = async_model.clone();

        assert_eq!(cloned.name(), "clone-test");

        // Both should work
        let _entities1 = async_model.extract_entities("test").await.unwrap();
        let _entities2 = cloned.extract_entities("test").await.unwrap();
    }
}
