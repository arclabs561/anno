//! Pipeline middleware for composing and extending NER backends.
//!
//! This module provides a flexible middleware architecture for:
//! - Pre-processing text before entity extraction
//! - Post-processing entities after extraction
//! - Filtering, enriching, or transforming entities
//! - Integrating with external systems
//!
//! # Design
//!
//! The middleware pattern follows a "chain of responsibility" approach where
//! each middleware can:
//! 1. Transform input before passing to the next stage
//! 2. Transform output after receiving from the previous stage
//! 3. Short-circuit the chain by returning early
//!
//! # Example: Basic Pipeline
//!
//! ```rust,ignore
//! use anno::backends::middleware::{Pipeline, Middleware, NormalizeWhitespace, FilterByConfidence};
//! use anno::StackedNER;
//!
//! let pipeline = Pipeline::new(Box::new(StackedNER::default()))
//!     .with(NormalizeWhitespace)
//!     .with(FilterByConfidence(0.5));
//!
//! let entities = pipeline.extract_entities("Hello  world", None)?;
//! ```
//!
//! # Example: Custom Middleware
//!
//! ```rust,ignore
//! use anno::backends::middleware::{Middleware, MiddlewareContext};
//! use anno::{Entity, Result};
//!
//! struct LogEntities;
//!
//! impl Middleware for LogEntities {
//!     fn post_process(&self, ctx: &mut MiddlewareContext, entities: Vec<Entity>) -> Result<Vec<Entity>> {
//!         eprintln!("Found {} entities", entities.len());
//!         Ok(entities)
//!     }
//! }
//! ```
//!
//! # Example: Entity Enrichment
//!
//! ```rust,ignore
//! use anno::backends::middleware::{Middleware, MiddlewareContext};
//! use anno::{Entity, Result};
//!
//! struct AddKnowledgeBaseLinks {
//!     kb_lookup: HashMap<String, String>,
//! }
//!
//! impl Middleware for AddKnowledgeBaseLinks {
//!     fn post_process(&self, ctx: &mut MiddlewareContext, mut entities: Vec<Entity>) -> Result<Vec<Entity>> {
//!         for entity in &mut entities {
//!             if let Some(kb_id) = self.kb_lookup.get(&entity.text.to_lowercase()) {
//!                 entity.kb_id = Some(kb_id.clone());
//!             }
//!         }
//!         Ok(entities)
//!     }
//! }
//! ```

use crate::{Entity, EntityType, Model, Result};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

// =============================================================================
// Middleware Context
// =============================================================================

/// Context passed through the middleware chain.
///
/// Provides access to:
/// - Original and transformed text
/// - Metadata and configuration
/// - Extracted entity types filter
#[derive(Debug, Clone)]
pub struct MiddlewareContext {
    /// Original input text (before any preprocessing).
    pub original_text: String,
    /// Current text (may be transformed by preprocessors).
    pub current_text: String,
    /// Requested entity types filter.
    pub entity_types: Option<Vec<EntityType>>,
    /// Language hint for the text.
    pub language: Option<String>,
    /// Arbitrary metadata (for custom middleware).
    pub metadata: HashMap<String, String>,
}

impl MiddlewareContext {
    /// Create a new context from input text.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            original_text: text.clone(),
            current_text: text,
            entity_types: None,
            language: None,
            metadata: HashMap::new(),
        }
    }

    /// Set language hint.
    #[must_use]
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Set entity types filter.
    #[must_use]
    pub fn with_entity_types(mut self, types: Vec<EntityType>) -> Self {
        self.entity_types = Some(types);
        self
    }

    /// Set metadata value.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

// =============================================================================
// Middleware Trait
// =============================================================================

/// Middleware for transforming input/output in the NER pipeline.
///
/// Implement this trait to create custom middleware that can:
/// - Preprocess text before extraction
/// - Postprocess entities after extraction
/// - Add metadata or enrich entities
pub trait Middleware: Send + Sync {
    /// Preprocess text before entity extraction.
    ///
    /// Returns the (possibly transformed) text to pass to the next stage.
    /// The default implementation passes through unchanged.
    fn pre_process<'a>(&self, ctx: &mut MiddlewareContext, text: &'a str) -> Result<Cow<'a, str>> {
        let _ = ctx;
        Ok(Cow::Borrowed(text))
    }

    /// Postprocess entities after extraction.
    ///
    /// Returns the (possibly transformed) entities to pass to the next stage.
    /// The default implementation passes through unchanged.
    fn post_process(
        &self,
        ctx: &mut MiddlewareContext,
        entities: Vec<Entity>,
    ) -> Result<Vec<Entity>> {
        let _ = ctx;
        Ok(entities)
    }

    /// Name of this middleware (for debugging/logging).
    fn name(&self) -> &'static str {
        "unnamed"
    }
}

// =============================================================================
// Pipeline
// =============================================================================

/// A composable NER pipeline with middleware support.
///
/// The pipeline executes middleware in order:
/// 1. Pre-process: Each middleware transforms the input text
/// 2. Extract: The core backend extracts entities
/// 3. Post-process: Each middleware transforms the entities (in reverse order)
///
/// # Example
///
/// ```rust,ignore
/// use anno::backends::middleware::{Pipeline, NormalizeWhitespace, FilterByConfidence};
/// use anno::StackedNER;
///
/// let pipeline = Pipeline::new(Box::new(StackedNER::default()))
///     .with(NormalizeWhitespace)
///     .with(FilterByConfidence(0.5));
///
/// // Extract entities through the pipeline
/// let entities = pipeline.extract("Hello  world")?;
/// ```
pub struct Pipeline {
    backend: Arc<dyn Model>,
    middleware: Vec<Box<dyn Middleware>>,
}

impl Pipeline {
    /// Create a new pipeline with a backend.
    #[must_use]
    pub fn new(backend: Box<dyn Model>) -> Self {
        Self {
            backend: Arc::from(backend),
            middleware: Vec::new(),
        }
    }

    /// Add middleware to the pipeline.
    #[must_use]
    pub fn with<M: Middleware + 'static>(mut self, middleware: M) -> Self {
        self.middleware.push(Box::new(middleware));
        self
    }

    /// Add middleware conditionally.
    #[must_use]
    pub fn with_if<M: Middleware + 'static>(self, condition: bool, middleware: M) -> Self {
        if condition {
            self.with(middleware)
        } else {
            self
        }
    }

    /// Extract entities through the pipeline.
    pub fn extract(&self, text: &str) -> Result<Vec<Entity>> {
        self.extract_with_context(&mut MiddlewareContext::new(text))
    }

    /// Extract entities with explicit context.
    pub fn extract_with_context(&self, ctx: &mut MiddlewareContext) -> Result<Vec<Entity>> {
        // Pre-process: each middleware transforms the text
        // Clone the text to avoid borrow issues
        let mut current_text = ctx.current_text.clone();
        for mw in &self.middleware {
            let result = mw.pre_process(ctx, &current_text)?;
            current_text = result.into_owned();
        }

        // Update context with final preprocessed text
        ctx.current_text = current_text;

        // Extract entities from the backend
        let mut entities = self
            .backend
            .extract_entities(&ctx.current_text, ctx.language.as_deref())?;

        // Post-process: each middleware transforms entities (reverse order)
        for mw in self.middleware.iter().rev() {
            entities = mw.post_process(ctx, entities)?;
        }

        Ok(entities)
    }

    /// Get the underlying backend.
    #[must_use]
    pub fn backend(&self) -> &dyn Model {
        &*self.backend
    }

    /// List middleware names.
    #[must_use]
    pub fn middleware_names(&self) -> Vec<&'static str> {
        self.middleware.iter().map(|m| m.name()).collect()
    }
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("middleware", &self.middleware_names())
            .finish()
    }
}

// =============================================================================
// Built-in Middleware
// =============================================================================

/// Normalize whitespace in input text.
///
/// - Collapses multiple spaces into single spaces
/// - Trims leading/trailing whitespace
#[derive(Debug, Clone, Copy, Default)]
pub struct NormalizeWhitespace;

impl Middleware for NormalizeWhitespace {
    fn pre_process<'a>(&self, _ctx: &mut MiddlewareContext, text: &'a str) -> Result<Cow<'a, str>> {
        // Check if normalization is needed
        let needs_normalization = text.contains("  ")
            || text.starts_with(char::is_whitespace)
            || text.ends_with(char::is_whitespace);

        if needs_normalization {
            let normalized: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
            Ok(Cow::Owned(normalized))
        } else {
            Ok(Cow::Borrowed(text))
        }
    }

    fn name(&self) -> &'static str {
        "normalize_whitespace"
    }
}

/// Filter entities by minimum confidence threshold.
#[derive(Debug, Clone, Copy)]
pub struct FilterByConfidence(pub f64);

impl Middleware for FilterByConfidence {
    fn post_process(
        &self,
        _ctx: &mut MiddlewareContext,
        entities: Vec<Entity>,
    ) -> Result<Vec<Entity>> {
        let threshold = self.0;
        Ok(entities
            .into_iter()
            .filter(|e| e.confidence >= threshold)
            .collect())
    }

    fn name(&self) -> &'static str {
        "filter_by_confidence"
    }
}

/// Filter entities by entity type.
#[derive(Debug, Clone)]
pub struct FilterByType(pub Vec<EntityType>);

impl Middleware for FilterByType {
    fn post_process(
        &self,
        _ctx: &mut MiddlewareContext,
        entities: Vec<Entity>,
    ) -> Result<Vec<Entity>> {
        Ok(entities
            .into_iter()
            .filter(|e| self.0.contains(&e.entity_type))
            .collect())
    }

    fn name(&self) -> &'static str {
        "filter_by_type"
    }
}

/// Remove overlapping entities, keeping the highest confidence one.
#[derive(Debug, Clone, Copy, Default)]
pub struct RemoveOverlaps;

impl Middleware for RemoveOverlaps {
    fn post_process(
        &self,
        _ctx: &mut MiddlewareContext,
        mut entities: Vec<Entity>,
    ) -> Result<Vec<Entity>> {
        // Sort by confidence descending
        entities.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut result = Vec::with_capacity(entities.len());
        for entity in entities {
            let overlaps = result
                .iter()
                .any(|e: &Entity| entity.start < e.end && entity.end > e.start);
            if !overlaps {
                result.push(entity);
            }
        }

        // Sort back by position
        result.sort_by_key(|e| e.start);
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "remove_overlaps"
    }
}

/// Add provenance information to all entities.
#[derive(Debug, Clone)]
pub struct AddProvenance {
    /// Backend name to set.
    pub backend: String,
    /// Method description.
    pub method: String,
}

impl AddProvenance {
    /// Create a new AddProvenance middleware.
    #[must_use]
    pub fn new(backend: impl Into<String>, method: impl Into<String>) -> Self {
        Self {
            backend: backend.into(),
            method: method.into(),
        }
    }
}

impl Middleware for AddProvenance {
    fn post_process(
        &self,
        _ctx: &mut MiddlewareContext,
        mut entities: Vec<Entity>,
    ) -> Result<Vec<Entity>> {
        use anno_core::Provenance;
        for entity in &mut entities {
            if entity.provenance.is_none() {
                entity.provenance = Some(Provenance::ml(self.backend.clone(), entity.confidence));
            }
        }
        Ok(entities)
    }

    fn name(&self) -> &'static str {
        "add_provenance"
    }
}

/// Merge adjacent entities of the same type.
///
/// Useful for combining split entities like "New" + "York" → "New York".
#[derive(Debug, Clone, Copy)]
pub struct MergeAdjacent {
    /// Maximum gap (in characters) between entities to merge.
    pub max_gap: usize,
}

impl Default for MergeAdjacent {
    fn default() -> Self {
        Self { max_gap: 1 }
    }
}

impl Middleware for MergeAdjacent {
    fn post_process(
        &self,
        ctx: &mut MiddlewareContext,
        mut entities: Vec<Entity>,
    ) -> Result<Vec<Entity>> {
        if entities.len() < 2 {
            return Ok(entities);
        }

        // Sort by position
        entities.sort_by_key(|e| e.start);

        let text = &ctx.current_text;
        let mut merged = Vec::with_capacity(entities.len());
        let mut current: Option<Entity> = None;

        for entity in entities {
            if let Some(prev) = current.take() {
                // Check if should merge
                let gap = entity.start.saturating_sub(prev.end);
                let same_type = prev.entity_type == entity.entity_type;

                if same_type && gap <= self.max_gap {
                    // Merge entities
                    let merged_text = text
                        .chars()
                        .skip(prev.start)
                        .take(entity.end - prev.start)
                        .collect::<String>();
                    let merged_confidence = (prev.confidence + entity.confidence) / 2.0;

                    current = Some(Entity::new(
                        merged_text,
                        prev.entity_type,
                        prev.start,
                        entity.end,
                        merged_confidence,
                    ));
                } else {
                    merged.push(prev);
                    current = Some(entity);
                }
            } else {
                current = Some(entity);
            }
        }

        if let Some(last) = current {
            merged.push(last);
        }

        Ok(merged)
    }

    fn name(&self) -> &'static str {
        "merge_adjacent"
    }
}

/// Callback middleware for custom processing.
///
/// Wraps a closure for simple one-off transformations.
pub struct Callback<F> {
    name: &'static str,
    func: F,
}

impl<F> Callback<F>
where
    F: Fn(&mut MiddlewareContext, Vec<Entity>) -> Result<Vec<Entity>> + Send + Sync,
{
    /// Create a new callback middleware.
    #[must_use]
    pub fn new(name: &'static str, func: F) -> Self {
        Self { name, func }
    }
}

impl<F> Middleware for Callback<F>
where
    F: Fn(&mut MiddlewareContext, Vec<Entity>) -> Result<Vec<Entity>> + Send + Sync,
{
    fn post_process(
        &self,
        ctx: &mut MiddlewareContext,
        entities: Vec<Entity>,
    ) -> Result<Vec<Entity>> {
        (self.func)(ctx, entities)
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

impl<F> std::fmt::Debug for Callback<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Callback")
            .field("name", &self.name)
            .finish()
    }
}

// =============================================================================
// Hook System
// =============================================================================

/// Event types for hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookEvent {
    /// Before extraction starts.
    BeforeExtraction,
    /// After extraction completes.
    AfterExtraction,
    /// When an entity is found.
    EntityFound,
    /// When an error occurs.
    OnError,
}

/// Hook function signature.
pub type HookFn = Box<dyn Fn(HookEvent, &MiddlewareContext, Option<&[Entity]>) + Send + Sync>;

/// Hook registry for pipeline events.
pub struct HookRegistry {
    hooks: HashMap<HookEvent, Vec<HookFn>>,
}

impl HookRegistry {
    /// Create a new hook registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hooks: HashMap::new(),
        }
    }

    /// Register a hook for an event.
    pub fn register(&mut self, event: HookEvent, hook: HookFn) {
        self.hooks.entry(event).or_default().push(hook);
    }

    /// Trigger hooks for an event.
    pub fn trigger(&self, event: HookEvent, ctx: &MiddlewareContext, entities: Option<&[Entity]>) {
        if let Some(hooks) = self.hooks.get(&event) {
            for hook in hooks {
                hook(event, ctx, entities);
            }
        }
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for HookRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookRegistry")
            .field("events", &self.hooks.keys().collect::<Vec<_>>())
            .finish()
    }
}

// =============================================================================
// Hooked Pipeline (with interior mutability)
// =============================================================================

use std::cell::RefCell;

/// A pipeline with hook support using interior mutability.
///
/// This addresses the borrow checker issues with the standard `HookRegistry` by:
/// 1. Using `RefCell` for interior mutability of hook-related state
/// 2. Separating "before" and "after" contexts to avoid simultaneous borrows
/// 3. Using owned data in hook invocations to avoid reference conflicts
///
/// # Design
///
/// The `HookedPipeline` solves the problem where:
/// - `extract_with_context` needs `&mut MiddlewareContext`
/// - Hooks need to be called with context data during extraction
/// - Multiple hooks may need to read entity data while context is borrowed
///
/// By cloning context data before passing to hooks, we avoid borrow conflicts.
///
/// # Example
///
/// ```rust,ignore
/// use anno::backends::middleware::{HookedPipeline, HookEvent};
/// use anno::HeuristicNER;
///
/// let mut pipeline = HookedPipeline::new(Box::new(HeuristicNER::new()));
///
/// pipeline.on(HookEvent::AfterExtraction, |event, text, entities| {
///     if let Some(entities) = entities {
///         println!("Found {} entities in: {}", entities.len(), text);
///     }
/// });
///
/// let entities = pipeline.extract("Hello World")?;
/// ```
pub struct HookedPipeline {
    backend: Arc<dyn Model>,
    middleware: Vec<Box<dyn Middleware>>,
    /// Hook registry with interior mutability for safe hook registration during extraction
    hooks: RefCell<HookRegistry>,
}

impl HookedPipeline {
    /// Create a new hooked pipeline with a backend.
    #[must_use]
    pub fn new(backend: Box<dyn Model>) -> Self {
        Self {
            backend: Arc::from(backend),
            middleware: Vec::new(),
            hooks: RefCell::new(HookRegistry::new()),
        }
    }

    /// Add middleware to the pipeline.
    #[must_use]
    pub fn with<M: Middleware + 'static>(mut self, middleware: M) -> Self {
        self.middleware.push(Box::new(middleware));
        self
    }

    /// Register a hook for an event.
    ///
    /// Hooks receive:
    /// - `event`: The event type
    /// - `text`: The current text being processed (owned copy)
    /// - `entities`: Optional entities (owned copy)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// pipeline.on(HookEvent::EntityFound, |event, text, entities| {
    ///     if let Some(entities) = entities {
    ///         for entity in entities {
    ///             println!("Found: {} ({})", entity.text, entity.entity_type);
    ///         }
    ///     }
    /// });
    /// ```
    pub fn on<F>(&self, event: HookEvent, handler: F)
    where
        F: Fn(HookEvent, &str, Option<&[Entity]>) + Send + Sync + 'static,
    {
        // Wrap the simpler handler in the full hook signature
        let wrapper = Box::new(
            move |evt: HookEvent, ctx: &MiddlewareContext, entities: Option<&[Entity]>| {
                handler(evt, &ctx.current_text, entities);
            },
        );
        self.hooks.borrow_mut().register(event, wrapper);
    }

    /// Register a hook with full context access.
    pub fn on_with_context(&self, event: HookEvent, hook: HookFn) {
        self.hooks.borrow_mut().register(event, hook);
    }

    /// Extract entities through the pipeline with hook support.
    pub fn extract(&self, text: &str) -> Result<Vec<Entity>> {
        let mut ctx = MiddlewareContext::new(text);

        // Trigger before extraction hooks (with cloned context to avoid borrow issues)
        {
            let hooks = self.hooks.borrow();
            hooks.trigger(HookEvent::BeforeExtraction, &ctx, None);
        }

        // Pre-process: each middleware transforms the text
        let mut current_text = ctx.current_text.clone();
        for mw in &self.middleware {
            let result = mw.pre_process(&mut ctx, &current_text)?;
            current_text = result.into_owned();
        }
        ctx.current_text = current_text;

        // Extract entities from the backend
        let entities = match self
            .backend
            .extract_entities(&ctx.current_text, ctx.language.as_deref())
        {
            Ok(entities) => entities,
            Err(e) => {
                // Trigger error hooks
                let hooks = self.hooks.borrow();
                hooks.trigger(HookEvent::OnError, &ctx, None);
                return Err(e);
            }
        };

        // Trigger entity found hooks for each entity
        {
            let hooks = self.hooks.borrow();
            for entity in &entities {
                hooks.trigger(
                    HookEvent::EntityFound,
                    &ctx,
                    Some(std::slice::from_ref(entity)),
                );
            }
        }

        // Post-process: each middleware transforms entities (reverse order)
        let mut entities = entities;
        for mw in self.middleware.iter().rev() {
            entities = mw.post_process(&mut ctx, entities)?;
        }

        // Trigger after extraction hooks
        {
            let hooks = self.hooks.borrow();
            hooks.trigger(HookEvent::AfterExtraction, &ctx, Some(&entities));
        }

        Ok(entities)
    }

    /// Get the underlying backend.
    #[must_use]
    pub fn backend(&self) -> &dyn Model {
        &*self.backend
    }

    /// List middleware names.
    #[must_use]
    pub fn middleware_names(&self) -> Vec<&'static str> {
        self.middleware.iter().map(|m| m.name()).collect()
    }

    /// Get the number of registered hooks.
    #[must_use]
    pub fn hook_count(&self) -> usize {
        self.hooks.borrow().hooks.values().map(|v| v.len()).sum()
    }
}

impl std::fmt::Debug for HookedPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookedPipeline")
            .field("middleware", &self.middleware_names())
            .field("hooks", &*self.hooks.borrow())
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HeuristicNER;

    #[test]
    fn test_normalize_whitespace() {
        let mw = NormalizeWhitespace;
        let mut ctx = MiddlewareContext::new("  hello   world  ");
        let text = ctx.original_text.clone();
        let result = mw
            .pre_process(&mut ctx, &text)
            .expect("pre_process should succeed");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_filter_by_confidence() {
        let mw = FilterByConfidence(0.5);
        let mut ctx = MiddlewareContext::new("test");
        let entities = vec![
            Entity::new("high", EntityType::Person, 0, 4, 0.8),
            Entity::new("low", EntityType::Person, 5, 8, 0.3),
        ];
        let result = mw
            .post_process(&mut ctx, entities)
            .expect("post_process should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "high");
    }

    #[test]
    fn test_pipeline_basic() {
        let pipeline = Pipeline::new(Box::new(HeuristicNER::new()))
            .with(NormalizeWhitespace)
            .with(FilterByConfidence(0.3));

        let _entities = pipeline
            .extract("Hello  World")
            .expect("extraction should succeed");
        // Just verify it runs without error
    }

    #[test]
    fn test_remove_overlaps() {
        let mw = RemoveOverlaps;
        let mut ctx = MiddlewareContext::new("New York City");
        let entities = vec![
            Entity::new("New York", EntityType::Location, 0, 8, 0.9),
            Entity::new("York City", EntityType::Location, 4, 13, 0.7),
        ];
        let result = mw
            .post_process(&mut ctx, entities)
            .expect("post_process should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "New York"); // Higher confidence wins
    }

    #[test]
    fn test_hooked_pipeline_basic() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let pipeline = HookedPipeline::new(Box::new(HeuristicNER::new())).with(NormalizeWhitespace);

        // Track hook invocations
        let before_count = Arc::new(AtomicUsize::new(0));
        let after_count = Arc::new(AtomicUsize::new(0));

        let before_count_clone = Arc::clone(&before_count);
        pipeline.on(HookEvent::BeforeExtraction, move |_, _, _| {
            before_count_clone.fetch_add(1, Ordering::SeqCst);
        });

        let after_count_clone = Arc::clone(&after_count);
        pipeline.on(HookEvent::AfterExtraction, move |_, _, _| {
            after_count_clone.fetch_add(1, Ordering::SeqCst);
        });

        let _entities = pipeline.extract("Hello World").unwrap();

        assert_eq!(before_count.load(Ordering::SeqCst), 1);
        assert_eq!(after_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_hooked_pipeline_entity_found_hook() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let pipeline = HookedPipeline::new(Box::new(HeuristicNER::new()));

        let entity_count = Arc::new(AtomicUsize::new(0));
        let entity_count_clone = Arc::clone(&entity_count);

        pipeline.on(HookEvent::EntityFound, move |_, _, entities| {
            if entities.is_some() {
                entity_count_clone.fetch_add(1, Ordering::SeqCst);
            }
        });

        // HeuristicNER should find capitalized words
        let _entities = pipeline.extract("John Smith went to New York").unwrap();

        // EntityFound should be called for each entity
        assert!(entity_count.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn test_hooked_pipeline_with_middleware() {
        let pipeline = HookedPipeline::new(Box::new(HeuristicNER::new()))
            .with(NormalizeWhitespace)
            .with(FilterByConfidence(0.3));

        let entities = pipeline
            .extract("  John   Smith  ")
            .expect("extraction should succeed");
        // Should normalize whitespace and filter by confidence
        // Just verify it runs without error
        let _ = entities;
    }

    #[test]
    fn test_hooked_pipeline_hook_count() {
        let pipeline = HookedPipeline::new(Box::new(HeuristicNER::new()));

        assert_eq!(pipeline.hook_count(), 0);

        pipeline.on(HookEvent::BeforeExtraction, |_, _, _| {});
        pipeline.on(HookEvent::AfterExtraction, |_, _, _| {});
        pipeline.on(HookEvent::EntityFound, |_, _, _| {});

        assert_eq!(pipeline.hook_count(), 3);
    }
}
