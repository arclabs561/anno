//! Stacked NER.
//!
//! `StackedNER` composes multiple extractors (regex, heuristics, and optionally ML backends)
//! and then resolves overlaps via a small conflict strategy (priority/longest/confidence/union).
//!
//! This module intentionally keeps the API surface small. For user-facing guidance and
//! provenance details, see `docs/BACKENDS.md` and the repo README.

use super::heuristic::HeuristicNER;
use super::regex::RegexNER;
use crate::{Entity, EntityType, Language, Model, Result};
use itertools::Itertools;
use std::borrow::Cow;
use std::sync::Arc;

use crate::backends::method_for_backend_name;

// =============================================================================
// Conflict Resolution
// =============================================================================

/// Strategy for resolving overlapping entity spans.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ConflictStrategy {
    /// First layer to claim a span wins. Simple and predictable.
    #[default]
    Priority,

    /// Longest span wins. Prefers "New York City" over "New York".
    LongestSpan,

    /// Highest confidence score wins.
    HighestConf,

    /// Keep all entities, even if they overlap.
    /// Useful when downstream processing handles disambiguation.
    Union,
}

impl ConflictStrategy {
    /// Resolve a conflict between two overlapping entities.
    ///
    /// # Arguments
    /// * `existing` - Entity already in the result set (from earlier layer)
    /// * `candidate` - New entity from current layer
    ///
    /// # Design Note
    ///
    /// When confidence/length are equal, we prefer `existing` to respect
    /// layer priority (earlier layers have higher priority).
    fn resolve(&self, existing: &Entity, candidate: &Entity) -> Resolution {
        // Subsumption rule: if the candidate fully contains the existing entity
        // and has a more specific structured type (Money, Date, Phone, etc.)
        // while existing is generic (Other/misc), prefer the candidate.
        // Example: "EUR 3.2 billion" (MONEY) subsumes "EUR" (misc).
        if candidate.start <= existing.start
            && candidate.end >= existing.end
            && candidate.end > candidate.start
            && is_structured_type(&candidate.entity_type)
            && is_generic_type(&existing.entity_type)
        {
            return Resolution::Replace;
        }

        match self {
            ConflictStrategy::Priority => Resolution::KeepExisting,

            ConflictStrategy::LongestSpan => {
                let existing_len = existing.end - existing.start;
                let candidate_len = candidate.end - candidate.start;
                if candidate_len > existing_len {
                    Resolution::Replace
                } else if candidate_len < existing_len {
                    Resolution::KeepExisting
                } else {
                    // Equal length: prefer existing (earlier layer has priority)
                    Resolution::KeepExisting
                }
            }

            ConflictStrategy::HighestConf => {
                // Prefer higher confidence, but if equal, prefer existing (earlier layer)
                if candidate.confidence > existing.confidence {
                    Resolution::Replace
                } else if candidate.confidence < existing.confidence {
                    Resolution::KeepExisting
                } else {
                    // Equal confidence: prefer existing (earlier layer has priority)
                    Resolution::KeepExisting
                }
            }

            ConflictStrategy::Union => Resolution::KeepBoth,
        }
    }
}

/// Pattern-detectable structured types that should subsume generic/misc entities.
fn is_structured_type(t: &EntityType) -> bool {
    matches!(
        t,
        EntityType::Money
            | EntityType::Date
            | EntityType::Time
            | EntityType::Percent
            | EntityType::Email
            | EntityType::Url
            | EntityType::Phone
            | EntityType::Quantity
    )
}

/// Generic/misc types that can be subsumed by more specific structured types.
fn is_generic_type(t: &EntityType) -> bool {
    matches!(t, EntityType::Custom { .. })
}

#[derive(Debug)]
enum Resolution {
    KeepExisting,
    Replace,
    KeepBoth,
}

// =============================================================================
// StackedNER
// =============================================================================

/// Composable NER that combines multiple backends.
///
/// `StackedNER` accepts **any backend that implements `Model`**, not just regex and heuristics.
/// You can combine pattern-based, heuristic-based, and ML-based backends in any order.
///
/// # Design
///
/// Different backends excel at different tasks:
///
/// | Backend Type | Best For | Trade-off |
/// |--------------|----------|-----------|
/// | Pattern (`RegexNER`) | Structured entities (dates, money, emails) | Can't do named entities |
/// | Heuristic (`HeuristicNER`) | Named entities (no deps) | Lower accuracy than ML |
/// | ML (`GLiNER`, `NuNER`, `BertNEROnnx`, etc.) | Everything, high accuracy | Heavy dependencies, slower |
///
/// `StackedNER` runs backends in order, merging results according to the
/// configured [`ConflictStrategy`].
///
/// # Default Configuration
///
/// `StackedNER::default()` creates a Pattern + Heuristic configuration:
/// - Layer 1: `RegexNER` (dates, money, emails, etc.)
/// - Layer 2: `HeuristicNER` (person, org, location)
///
/// This provides solid NER coverage with zero ML dependencies.
///
/// # Examples
///
/// Zero-dependency default (Pattern + Heuristic):
///
/// ```rust
/// use anno::{Model, StackedNER};
///
/// let ner = StackedNER::default();
/// let entities = ner.extract_entities("Dr. Smith charges $100/hr", None).unwrap();
/// ```
///
/// Custom stack with pattern + heuristic:
///
/// ```rust
/// use anno::{Model, RegexNER, HeuristicNER, StackedNER};
/// use anno::backends::stacked::ConflictStrategy;
///
/// let ner = StackedNER::builder()
///     .layer(RegexNER::new())
///     .layer(HeuristicNER::new())
///     .strategy(ConflictStrategy::LongestSpan)
///     .build();
/// ```
///
/// **Composing with ML backends** (requires `onnx` or `candle` feature):
///
/// ```rust,no_run
/// #[cfg(feature = "onnx")]
/// {
/// use anno::{Model, StackedNER, GLiNEROnnx, RegexNER, HeuristicNER};
/// use anno::backends::stacked::ConflictStrategy;
///
/// // ML-first: ML runs first, then patterns fill gaps
/// let ner = StackedNER::with_ml_first(
///     Box::new(GLiNEROnnx::new("onnx-community/gliner_small-v2.1").unwrap())
/// );
///
/// // ML-fallback: patterns/heuristics first, ML as fallback
/// let ner = StackedNER::with_ml_fallback(
///     Box::new(GLiNEROnnx::new("onnx-community/gliner_small-v2.1").unwrap())
/// );
///
/// // Custom stack: any combination of backends
/// let ner = StackedNER::builder()
///     .layer(RegexNER::new())           // High-precision structured entities
///     .layer_boxed(Box::new(GLiNEROnnx::new("onnx-community/gliner_small-v2.1").unwrap()))  // ML layer
///     .layer(HeuristicNER::new())       // Quick named entities
///     .strategy(ConflictStrategy::HighestConf)  // Resolve conflicts by confidence
///     .build();
/// }
/// ```
///
/// You can stack multiple ML backends, mix ONNX and Candle backends, or create any
/// combination that fits your use case. The builder accepts any `Model` implementation.
pub struct StackedNER {
    layers: Vec<Arc<dyn Model + Send + Sync>>,
    strategy: ConflictStrategy,
    name: String,
    /// Cached static name (avoids Box::leak on every name() call)
    name_static: std::sync::OnceLock<&'static str>,
}

/// Builder for [`StackedNER`] with fluent configuration.
#[derive(Default)]
pub struct StackedNERBuilder {
    layers: Vec<Box<dyn Model + Send + Sync>>,
    strategy: ConflictStrategy,
}

impl StackedNERBuilder {
    /// Add a layer (order matters: earlier = higher priority).
    #[must_use]
    pub fn layer<M: Model + Send + Sync + 'static>(mut self, model: M) -> Self {
        self.layers.push(Box::new(model));
        self
    }

    /// Add a boxed layer.
    #[must_use]
    pub fn layer_boxed(mut self, model: Box<dyn Model + Send + Sync>) -> Self {
        self.layers.push(model);
        self
    }

    /// Set the conflict resolution strategy.
    #[must_use]
    pub fn strategy(mut self, strategy: ConflictStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Build the configured StackedNER.
    ///
    /// # Panics
    ///
    /// Panics if no layers are provided (empty stack is invalid).
    #[must_use]
    pub fn build(self) -> StackedNER {
        self.try_build().expect(
            "StackedNER requires at least one layer. Use StackedNER::builder().layer(...).build()",
        )
    }

    /// Build the configured StackedNER without panicking.
    ///
    /// This is useful when the stack is assembled dynamically (e.g., from CLI flags)
    /// and an empty stack should be handled as an error instead of aborting.
    pub fn try_build(self) -> crate::Result<StackedNER> {
        if self.layers.is_empty() {
            return Err(crate::Error::InvalidInput(
                "StackedNER requires at least one layer".to_string(),
            ));
        }

        let name = format!(
            "stacked({})",
            self.layers
                .iter()
                .map(|l| l.name())
                .collect::<Vec<_>>()
                .join("+")
        );

        Ok(StackedNER {
            layers: self.layers.into_iter().map(Arc::from).collect(),
            strategy: self.strategy,
            name,
            name_static: std::sync::OnceLock::new(),
        })
    }
}

impl StackedNER {
    /// Create default configuration: Pattern + Statistical layers.
    ///
    /// This provides zero-dependency NER with:
    /// - High-precision structured entity extraction (dates, money, etc.)
    /// - Heuristic named entity extraction (person, org, location)
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for custom configuration.
    #[must_use]
    pub fn builder() -> StackedNERBuilder {
        StackedNERBuilder::default()
    }

    /// Create with explicit layers and default priority strategy.
    #[must_use]
    pub fn with_layers(layers: Vec<Box<dyn Model + Send + Sync>>) -> Self {
        let mut builder = Self::builder().strategy(ConflictStrategy::Priority);
        for layer in layers {
            builder = builder.layer_boxed(layer);
        }
        builder.build()
    }

    /// Create with custom heuristic threshold.
    ///
    /// Higher threshold = fewer but higher confidence heuristic entities.
    /// Note: HeuristicNER does not currently support dynamic thresholding
    /// in constructor, so this method ignores the parameter for now but maintains API compat.
    #[must_use]
    pub fn with_heuristic_threshold(_threshold: f64) -> Self {
        Self::builder()
            .layer(RegexNER::new())
            .layer(HeuristicNER::new())
            .build()
    }

    /// Pattern-only configuration (no heuristic layer).
    ///
    /// Extracts only structured entities: dates, times, money, percentages,
    /// emails, URLs, phone numbers.
    #[must_use]
    pub fn pattern_only() -> Self {
        Self::builder().layer(RegexNER::new()).build()
    }

    /// Heuristic-only configuration (no pattern layer).
    ///
    /// Extracts only named entities: person, organization, location.
    #[must_use]
    pub fn heuristic_only() -> Self {
        Self::builder().layer(HeuristicNER::new()).build()
    }

    /// Add an ML backend as highest priority.
    ///
    /// ML runs first, then Pattern fills structured gaps, then Heuristic.
    #[must_use]
    pub fn with_ml_first(ml_backend: Box<dyn Model + Send + Sync>) -> Self {
        Self::builder()
            .layer_boxed(ml_backend)
            .layer(RegexNER::new())
            .layer(HeuristicNER::new())
            .build()
    }

    /// Add an ML backend as fallback (lowest priority).
    ///
    /// Pattern runs first (high precision), then Heuristic, then ML.
    #[must_use]
    pub fn with_ml_fallback(ml_backend: Box<dyn Model + Send + Sync>) -> Self {
        Self::builder()
            .layer(RegexNER::new())
            .layer(HeuristicNER::new())
            .layer_boxed(ml_backend)
            .build()
    }

    /// Get the number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get layer names in priority order.
    #[must_use]
    pub fn layer_names(&self) -> Vec<String> {
        self.layers
            .iter()
            .map(|l| l.name().to_string())
            .collect_vec()
    }

    /// Get the conflict strategy.
    #[must_use]
    pub fn strategy(&self) -> ConflictStrategy {
        self.strategy
    }

    /// Get statistics about the stack configuration.
    ///
    /// Returns a summary of layer count, strategy, and layer names.
    /// Useful for debugging and monitoring.
    #[must_use]
    pub fn stats(&self) -> StackStats {
        StackStats {
            layer_count: self.layers.len(),
            strategy: self.strategy,
            layer_names: self.layer_names(),
        }
    }
}

/// Statistics about a StackedNER configuration.
///
/// Provides insight into the stack's structure for debugging and monitoring.
#[derive(Debug, Clone)]
pub struct StackStats {
    /// Number of layers in the stack.
    pub layer_count: usize,
    /// Conflict resolution strategy.
    pub strategy: ConflictStrategy,
    /// Names of all layers in priority order (earliest = highest priority).
    pub layer_names: Vec<String>,
}

impl Default for StackedNER {
    /// Default configuration: Best available model stack.
    ///
    /// Tries to include ML backends (GLiNER, BERT) when available, falling back to
    /// Pattern + Heuristic for zero-dependency operation.
    ///
    /// Downloads are allowed by default; opt out by setting `ANNO_NO_DOWNLOADS=1`
    /// (or `HF_HUB_OFFLINE=1` to force HuggingFace offline mode).
    ///
    /// Priority:
    /// 1. BERT ONNX (if `onnx` feature and model available) - strong default for standard NER
    /// 2. GLiNER (if `onnx` feature and model available) - zero-shot, broader label set
    /// 3. Pattern + Heuristic (always available) - zero dependencies
    fn default() -> Self {
        // Try BERT first for standard NER (usually best on PER/ORG/LOC/MISC).
        #[cfg(feature = "onnx")]
        {
            fn no_downloads() -> bool {
                match std::env::var("ANNO_NO_DOWNLOADS") {
                    Ok(v) => matches!(
                        v.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "y" | "on"
                    ),
                    Err(_) => false,
                }
            }

            struct EnvVarGuard {
                key: &'static str,
                prev: Option<String>,
            }

            impl EnvVarGuard {
                fn set(key: &'static str, value: &str) -> Self {
                    let prev = std::env::var(key).ok();
                    std::env::set_var(key, value);
                    Self { key, prev }
                }
            }

            impl Drop for EnvVarGuard {
                fn drop(&mut self) {
                    match &self.prev {
                        Some(v) => std::env::set_var(self.key, v),
                        None => std::env::remove_var(self.key),
                    }
                }
            }

            // Opt-out policy: allow downloads unless explicitly disabled.
            // GLiNER/BERT loaders use `hf_hub`, which honors `HF_HUB_OFFLINE=1`.
            let _offline = no_downloads().then(|| EnvVarGuard::set("HF_HUB_OFFLINE", "1"));

            // Try ML backends independently: each is useful on its own.
            // BERT handles standard NER; NuNER handles lowercase/informal text;
            // GLiNER provides zero-shot coverage. Any combination works.
            use crate::backends::onnx::BertNEROnnx;
            use crate::DEFAULT_BERT_ONNX_MODEL;
            let bert = BertNEROnnx::new(DEFAULT_BERT_ONNX_MODEL).ok();
            // NuNER threshold is higher (0.9) when stacked with other ML
            // backends to avoid false positives on common nouns.
            let nuner = crate::backends::nuner::NuNER::from_pretrained(crate::DEFAULT_NUNER_MODEL)
                .map(|n| n.with_threshold(0.9))
                .ok();

            if bert.is_some() || nuner.is_some() {
                let mut builder = Self::builder();
                if let Some(b) = bert {
                    builder = builder.layer_boxed(Box::new(b));
                }
                if let Some(n) = nuner {
                    builder = builder.layer_boxed(Box::new(n));
                }
                return builder
                    .layer(RegexNER::new())
                    .layer(HeuristicNER::new())
                    .build();
            }

            // Fallback to GLiNER (zero-shot, broader label set).
            use crate::{GLiNEROnnx, DEFAULT_GLINER_MODEL};
            if let Ok(gliner) = GLiNEROnnx::new(DEFAULT_GLINER_MODEL) {
                return Self::builder()
                    .layer_boxed(Box::new(gliner))
                    .layer(RegexNER::new())
                    .layer(HeuristicNER::new())
                    .build();
            }
        }

        // Ultimate fallback: Pattern + Heuristic (zero dependencies)
        Self::builder()
            .layer(RegexNER::new())
            .layer(HeuristicNER::new())
            .build()
    }
}

impl Model for StackedNER {
    #[cfg_attr(feature = "production", tracing::instrument(skip(self, text), fields(text_len = text.len(), num_layers = self.layers.len())))]
    fn extract_entities(&self, text: &str, language: Option<Language>) -> Result<Vec<Entity>> {
        // Performance: Pre-allocate entities vec with estimated capacity
        // Most texts have 0-20 entities, but we'll start with a reasonable default
        let mut entities: Vec<Entity> = Vec::with_capacity(16);
        let mut layer_errors = Vec::new();

        // Performance optimization: Cache text length (O(n) operation, called many times)
        // This is shared across all backends and called in hot loops
        // ROI: High - called once per extract_entities, saves O(n) per entity in loop
        let text_char_count = text.chars().count();

        // Adaptive NuNER: skip NuNER for well-capitalized text where BERT alone suffices.
        // NuNER adds ~2s latency and mainly helps with lowercase entity names.
        let skip_nuner = !text_may_need_nuner(text);

        for layer in &self.layers {
            let layer_name = layer.name();

            // Skip NuNER when text is well-capitalized (saves ~2s latency)
            if skip_nuner && layer_name.to_lowercase().contains("nuner") {
                log::debug!("StackedNER: skipping NuNER (text appears well-capitalized)");
                continue;
            }

            // Try to extract from this layer, but continue on error if other layers succeeded
            let layer_entities = match layer.extract_entities(text, language) {
                Ok(ents) => ents,
                Err(e) => {
                    // Log error but continue with remaining layers.
                    // Only fail after all layers have been tried (see below).
                    layer_errors.push((layer_name.to_string(), e));
                    continue;
                }
            };

            for mut candidate in layer_entities {
                // Defensive: Clamp entity offsets to valid range
                // Some backends may produce out-of-bounds offsets in edge cases (Unicode, control chars)
                // Use cached text_char_count instead of recalculating (performance optimization)
                if candidate.end > text_char_count {
                    log::debug!(
                        "StackedNER: Clamping entity end offset from {} to {} (text length: {})",
                        candidate.end,
                        text_char_count,
                        text_char_count
                    );
                    candidate.end = text_char_count;
                    // Keep `entity.text` consistent with the adjusted span (Unicode-safe).
                    //
                    // This only triggers on buggy/out-of-bounds backends, but when it does,
                    // returning a span/text mismatch is more confusing than truncating text.
                    if candidate.start < candidate.end {
                        candidate.text = crate::offset::TextSpan::from_chars(
                            text,
                            candidate.start,
                            candidate.end,
                        )
                        .extract(text)
                        .to_string();
                    }
                }
                if candidate.start >= candidate.end || candidate.start > text_char_count {
                    // Invalid span - skip this entity
                    log::debug!(
                        "StackedNER: Skipping entity with invalid span: start={}, end={}, text_len={}",
                        candidate.start,
                        candidate.end,
                        text_char_count
                    );
                    continue;
                }

                // Add provenance tracking if not already set
                if candidate.provenance.is_none() {
                    candidate.provenance = Some(anno_core::Provenance {
                        source: Cow::Borrowed(layer_name),
                        method: method_for_backend_name(layer_name),
                        pattern: None,
                        raw_confidence: Some(candidate.confidence),
                        model_version: None,
                        timestamp: None,
                    });
                }

                // Find ALL overlapping entities (not just first)
                //
                // Performance: O(n) per candidate, O(n²) overall for n entities.
                // For large entity sets, consider optimizing with:
                // - Interval tree: O(n log n) construction, O(log n + k) query (k = overlaps)
                // - Sorted intervals with binary search: O(n log n) sort, O(log n + k) query
                // Current implementation prioritizes correctness and simplicity.
                //
                // Note: Entities are sorted at the end, but during conflict resolution
                // we process candidates in layer order, so we can't assume sorted order here.
                let overlapping_indices: Vec<usize> = entities
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, e)| {
                        // Check if candidate overlaps with existing entity
                        // Overlap: !(candidate.end <= e.start || candidate.start >= e.end)
                        if candidate.end > e.start && candidate.start < e.end {
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .collect();

                match overlapping_indices.len() {
                    0 => {
                        // No overlap - add directly
                        entities.push(candidate);
                    }
                    1 => {
                        // Single overlap - resolve normally
                        let idx = overlapping_indices[0];
                        match self.strategy.resolve(&entities[idx], &candidate) {
                            Resolution::KeepExisting => {}
                            Resolution::Replace => {
                                entities[idx] = candidate;
                            }
                            Resolution::KeepBoth => {
                                entities.push(candidate);
                            }
                        }
                    }
                    _ => {
                        // Multiple overlaps - need to handle carefully
                        // Strategy: resolve with the "best" existing entity based on strategy,
                        // then check if candidate should replace it
                        let best_idx = overlapping_indices
                            .iter()
                            .max_by(|&&a, &&b| {
                                // Find the "best" existing entity to compare against
                                match self.strategy {
                                    ConflictStrategy::Priority => {
                                        // Earlier in list = higher priority
                                        a.cmp(&b).reverse()
                                    }
                                    ConflictStrategy::LongestSpan => {
                                        let len_a = entities[a].end - entities[a].start;
                                        let len_b = entities[b].end - entities[b].start;
                                        len_a.cmp(&len_b).then_with(|| b.cmp(&a))
                                    }
                                    ConflictStrategy::HighestConf => entities[a]
                                        .confidence
                                        .partial_cmp(&entities[b].confidence)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                        .then_with(|| b.cmp(&a)),
                                    ConflictStrategy::Union => {
                                        // For union, we'll keep all, so just pick first
                                        a.cmp(&b)
                                    }
                                }
                            })
                            .copied()
                            .unwrap_or(overlapping_indices[0]);

                        match self.strategy {
                            ConflictStrategy::Union => {
                                // Keep candidate and all existing overlapping entities
                                entities.push(candidate);
                            }
                            _ => {
                                // Resolve with best existing entity
                                match self.strategy.resolve(&entities[best_idx], &candidate) {
                                    Resolution::KeepExisting => {
                                        // Remove other overlapping entities (they're subsumed)
                                        // Sort indices descending to remove from end
                                        let mut to_remove: Vec<usize> = overlapping_indices
                                            .into_iter()
                                            .filter(|&idx| idx != best_idx)
                                            .collect();
                                        // Performance: Use unstable sort (we don't need stable sort here)
                                        to_remove.sort_unstable_by(|a, b| b.cmp(a));
                                        for idx in to_remove {
                                            entities.remove(idx);
                                        }
                                    }
                                    Resolution::Replace => {
                                        // Replace best and remove others
                                        let mut to_remove: Vec<usize> = overlapping_indices
                                            .into_iter()
                                            .filter(|&idx| idx != best_idx)
                                            .collect();
                                        // Performance: Use unstable sort (we don't need stable sort here)
                                        to_remove.sort_unstable_by(|a, b| b.cmp(a));

                                        // Adjust best_idx based on how many entities we remove before it
                                        let removed_before_best =
                                            to_remove.iter().filter(|&&idx| idx < best_idx).count();
                                        let adjusted_best_idx = best_idx - removed_before_best;

                                        // Remove entities (in descending order to preserve indices)
                                        for idx in to_remove {
                                            entities.remove(idx);
                                        }

                                        // Now use adjusted index
                                        entities[adjusted_best_idx] = candidate;
                                    }
                                    Resolution::KeepBoth => {
                                        // Remove others, keep best and candidate
                                        let mut to_remove: Vec<usize> = overlapping_indices
                                            .into_iter()
                                            .filter(|&idx| idx != best_idx)
                                            .collect();
                                        // Performance: Use unstable sort (we don't need stable sort here)
                                        to_remove.sort_unstable_by(|a, b| b.cmp(a));
                                        // Remove entities (best_idx remains valid since we don't remove it)
                                        for idx in to_remove {
                                            entities.remove(idx);
                                        }
                                        entities.push(candidate);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort by position (start, then end) with deterministic tie-breaks.
        //
        // We include additional keys so exact-tie cases (same span) produce stable ordering,
        // and so dedup-by-span+type (below) works reliably if duplicates slip through.
        entities.sort_unstable_by(|a, b| {
            let a_ty = a.entity_type.as_label();
            let b_ty = b.entity_type.as_label();
            let a_src = a
                .provenance
                .as_ref()
                .map(|p| p.source.as_ref())
                .unwrap_or("");
            let b_src = b
                .provenance
                .as_ref()
                .map(|p| p.source.as_ref())
                .unwrap_or("");

            (a.start, a.end, a_ty, a_src, a.text.as_str()).cmp(&(
                b.start,
                b.end,
                b_ty,
                b_src,
                b.text.as_str(),
            ))
        });

        // Remove any duplicates that might have been created (defensive)
        // Only deduplicate if not using Union strategy (Union intentionally allows overlaps)
        if self.strategy != ConflictStrategy::Union {
            // Two entities are duplicates if they have same span and type
            // Performance: dedup_by is O(n) and efficient for sorted vec
            entities.dedup_by(|a, b| {
                a.start == b.start && a.end == b.end && a.entity_type == b.entity_type
            });
        }

        // If every layer errored out and we have no entities, surface the last error.
        if entities.is_empty() && layer_errors.len() == self.layers.len() {
            if let Some((_, last_err)) = layer_errors.pop() {
                return Err(last_err);
            }
        }
        // If we had errors but got partial results, log them but return success.
        if !layer_errors.is_empty() && !entities.is_empty() {
            log::warn!(
                "StackedNER: Some layers failed but returning partial results. Errors: {:?}",
                layer_errors
                    .iter()
                    .map(|(n, e)| format!("{n}: {e}"))
                    .collect::<Vec<_>>()
            );
        }

        // Span healing: merge adjacent same-type entities separated by 0-1 chars.
        // This fixes split entities from BERT tokenization misalignment (e.g.,
        // "Bundeskanzler" split into "Bundes" + "kanzler", or "U.S. District Court"
        // split into fragments).
        heal_adjacent_spans(text, &mut entities);
        extend_person_spans(text, &mut entities);
        filter_title_words(&mut entities);

        // Validate final entities (defensive programming)
        // This catches bugs in individual backends that might produce invalid spans
        for entity in &entities {
            if entity.start >= entity.end {
                log::warn!(
                    "StackedNER: Invalid entity span detected: start={}, end={}, text={:?}, type={:?}",
                    entity.start,
                    entity.end,
                    entity.text,
                    entity.entity_type
                );
            }
        }

        Ok(entities)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        // Use itertools for efficient deduplication
        self.layers
            .iter()
            .flat_map(|layer| layer.supported_types())
            .sorted_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)))
            .dedup()
            .collect_vec()
    }

    fn is_available(&self) -> bool {
        self.layers.iter().any(|l| l.is_available())
    }

    fn name(&self) -> &'static str {
        // Use OnceLock to cache the static string, avoiding repeated memory leaks
        self.name_static
            .get_or_init(|| Box::leak(self.name.clone().into_boxed_str()))
    }

    fn description(&self) -> &'static str {
        "Stacked NER (multi-backend composition)"
    }

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities::default()
    }
}

/// Merge adjacent same-type entities separated by 0-1 characters.
///
/// BERT subword tokenization can split compound words or multi-word entities
/// when subword boundaries don't align with word boundaries. This post-process
/// heals those splits by merging adjacent entities of the same type when the
/// gap between them is at most 1 character.
fn heal_adjacent_spans(text: &str, entities: &mut Vec<Entity>) {
    if entities.len() < 2 {
        return;
    }

    // Entities are already sorted by (start, end).
    let mut merged = Vec::with_capacity(entities.len());
    let mut current = entities[0].clone();

    for next in entities.iter().skip(1) {
        // Check: same type, adjacent (gap 0 or 1 char), and gap char (if any)
        // is alphanumeric or whitespace (not a sentence-ending punctuation).
        // Only heal truly adjacent spans (next starts at or just after current ends).
        // Skip overlapping or identical spans (e.g. from Union strategy keeping duplicates).
        let gap = next.start.saturating_sub(current.end);
        let truly_adjacent = next.start >= current.end && next.start > current.start;
        let same_type = current.entity_type == next.entity_type;
        let gap_ok = truly_adjacent
            && gap <= 1
            && (gap == 0
                || text
                    .chars()
                    .nth(current.end)
                    .is_some_and(|c| c.is_alphanumeric() || c == ' '));

        // A very short adjacent fragment (1-3 chars, gap=0) from ONNX
        // subword mislabeling should be absorbed into the preceding entity
        // regardless of type.  Example: "Merkel" tokenized as
        // [PER: "Merk"] + [LOC: "el"] -- the "el" is not a real LOC.
        let next_len = next.end.saturating_sub(next.start);
        let is_fragment = truly_adjacent && gap == 0 && next_len <= 3;

        if (same_type && gap_ok) || is_fragment {
            // Merge: extend current to cover next
            current.end = next.end;
            // Rebuild text from the merged span
            current.text = text
                .chars()
                .skip(current.start)
                .take(current.end - current.start)
                .collect();
            // Keep higher confidence
            if next.confidence > current.confidence {
                current.confidence = next.confidence;
            }
        } else {
            merged.push(current);
            current = next.clone();
        }
    }
    merged.push(current);

    *entities = merged;
}

/// Extend PER entities rightward into adjacent untagged capitalized words.
///
/// When a multi-word person name like "Kishida Fumio" is only partially tagged
/// (e.g., "Kishida" is PER but "Fumio" gets no tag), this extends the PER span
/// to cover the following capitalized word(s) that are likely part of the name.
/// Only extends PER entities, not ORG/LOC, and only into words that are not
/// common sentence starters or role/title words.
fn extend_person_spans(text: &str, entities: &mut [Entity]) {
    // Words that commonly appear capitalized but are not part of person names.
    // This is a focused subset of the heuristic backend's COMMON_SENTENCE_STARTERS
    // and SKIP_WORDS lists, covering the most frequent false-positive triggers.
    const NON_NAME_WORDS: &[&str] = &[
        // Sentence starters / determiners
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "it",
        "he",
        "she",
        "we",
        "they",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "and",
        "but",
        "or",
        "so",
        "if",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "what",
        "where",
        "when",
        "who",
        "why",
        "how",
        "here",
        "about",
        "more",
        "next",
        "back",
        "home",
        "however",
        "meanwhile",
        "furthermore",
        "moreover",
        "therefore",
        "although",
        "indeed",
        "perhaps",
        "certainly",
        "no",
        "yes",
        "some",
        "many",
        "each",
        "every",
        "both",
        "all",
        "few",
        "several",
        "other",
        "another",
        // Day/month names
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        // Job titles / role words
        "ceo",
        "cto",
        "cfo",
        "coo",
        "vp",
        "president",
        "chairman",
        "director",
        "manager",
        "secretary",
        "minister",
        "kanzler",
        "bundeskanzler",
        // Form/field labels
        "phone",
        "fax",
        "mobile",
        "address",
        "website",
        "name",
        "company",
        "contact",
    ];

    let text_chars: Vec<char> = text.chars().collect();
    let text_len = text_chars.len();

    // Build a set of char-offset ranges that are already covered by entities,
    // so we only extend into truly untagged territory.
    let occupied: Vec<(usize, usize)> = entities.iter().map(|e| (e.start, e.end)).collect();

    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..entities.len() {
            if entities[i].entity_type != EntityType::Person {
                continue;
            }

            let end = entities[i].end;
            if end >= text_len {
                continue;
            }

            // Skip whitespace after the entity
            let mut pos = end;
            while pos < text_len && text_chars[pos].is_whitespace() {
                pos += 1;
            }
            if pos >= text_len || pos == end {
                // No whitespace gap (adjacent punctuation etc.) or end of text
                continue;
            }

            // The next char must be uppercase (Latin-script capitalization signal)
            if !text_chars[pos].is_uppercase() {
                continue;
            }

            // Collect the next word
            let word_start = pos;
            while pos < text_len
                && !text_chars[pos].is_whitespace()
                && text_chars[pos] != ','
                && text_chars[pos] != '.'
                && text_chars[pos] != ';'
                && text_chars[pos] != ':'
                && text_chars[pos] != '('
                && text_chars[pos] != ')'
            {
                pos += 1;
            }
            let word_end = pos;

            // Check this word region isn't already occupied by another entity
            let overlaps_existing = occupied.iter().any(|&(s, e)| {
                // Overlap if not disjoint
                word_start < e && word_end > s
            });
            if overlaps_existing {
                continue;
            }

            // Extract the word text and check it's not a non-name word
            let word: String = text_chars[word_start..word_end].iter().collect();
            let word_lower = word
                .trim_end_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if NON_NAME_WORDS.contains(&word_lower.as_str()) {
                continue;
            }

            // Extend the entity span
            entities[i].end = word_end;
            entities[i].text = text_chars[entities[i].start..word_end].iter().collect();
            changed = true;
        }
    }
}

/// Check whether text may contain lowercase entity names that need NuNER.
///
/// Remove single-word entities that are known title/role words or common nouns.
///
/// BERT frequently tags German political titles ("Bundeskanzler") and similar role words
/// as ORG because they appear in organizational contexts in training data. The heuristic
/// backend also tags headline-capitalized common nouns ("Death", "Bus", "Christmas") as PER.
/// This post-processing step catches both categories.
fn filter_title_words(entities: &mut Vec<Entity>) {
    const TITLE_WORDS: &[&str] = &[
        // German political titles
        "bundeskanzler",
        "bundeskanzlerin",
        "kanzler",
        "kanzlerin",
        "bundespraesident",
        "bundespraesidentin",
        "buergermeister",
        "buergermeisterin",
        // English role words
        "president",
        "chairman",
        "chairwoman",
        "director",
        "secretary",
        "minister",
        "chancellor",
        "governor",
        "senator",
        "congressman",
        "congresswoman",
        "mayor",
    ];

    // Common nouns that BERT sometimes tags as PER due to headline capitalization.
    // Kept small -- the primary defense is the heuristic's lowered confidence for
    // single capitalized words (below threshold). This list catches the few that
    // slip through when BERT agrees on a wrong PER classification.
    const COMMON_NOUNS_NOT_PER: &[&str] = &[
        "death",
        "police",
        "military",
        "authorities",
        "officials",
        "analysts",
        "scientists",
        "researchers",
        "experts",
        "voters",
        "residents",
    ];

    entities.retain(|e| {
        // Multi-word entities are likely real (skip filtering)
        if e.text.contains(' ') {
            return true;
        }
        let lower = e.text.to_lowercase();

        // Filter ORG/MISC: title words
        if matches!(
            e.entity_type,
            EntityType::Organization | EntityType::Custom { .. }
        ) && TITLE_WORDS.contains(&lower.as_str())
        {
            return false;
        }

        // Filter PER: common nouns that are never person names
        if matches!(e.entity_type, EntityType::Person)
            && COMMON_NOUNS_NOT_PER.contains(&lower.as_str())
        {
            return false;
        }

        true
    });
}

/// NuNER's main value over BERT is detecting lowercase entities ("tim cook", "apple inc.").
/// For well-capitalized text (news, formal docs), BERT alone handles NER well and NuNER
/// adds ~2s latency without meaningful recall improvement.
///
/// Heuristic: scan for sequences of 2+ consecutive lowercase words (>2 chars each,
/// excluding common English stopwords) -- these suggest informal/untitled entity names.
fn text_may_need_nuner(text: &str) -> bool {
    // Very short texts (under ~30 chars) are cheap to process even with NuNER.
    // Above that, only run NuNER when lowercase entity patterns are detected.
    if text.len() < 30 {
        return true;
    }

    const STOPWORDS: &[&str] = &[
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "is", "are", "was", "were", "be", "been", "has", "have", "had", "that", "this",
        "it", "he", "she", "we", "they", "not", "would", "could", "should", "will", "can", "may",
        "also", "its", "his", "her", "our", "their", "who", "which", "what", "when", "where",
        "how", "than", "then", "into", "over", "about", "after", "before", "between", "under",
        "up", "out", "new", "said", "told", "expects",
    ];

    let mut consecutive_lc = 0u32;
    for word in text.split_whitespace() {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        if clean.len() <= 2 {
            consecutive_lc = 0;
            continue;
        }
        let first = match clean.chars().next() {
            Some(c) if c.is_alphabetic() => c,
            _ => {
                consecutive_lc = 0;
                continue;
            }
        };
        if first.is_lowercase() && !STOPWORDS.contains(&clean.to_lowercase().as_str()) {
            consecutive_lc += 1;
            if consecutive_lc >= 2 {
                return true;
            }
        } else {
            consecutive_lc = 0;
        }
    }
    false
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
