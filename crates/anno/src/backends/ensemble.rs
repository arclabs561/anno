//! Ensemble NER - Multi-backend extraction with unsupervised weighted voting.
//!
//! # Method
//!
//! This is an **unsupervised heuristic** approach (no training data required).
//! Conflict resolution uses hand-tuned weights based on expected backend reliability.
//! For supervised weight learning from labeled data, see `WeightLearner`.
//!
//! # The Core Idea
//!
//! Instead of simple priority-based stacking, `EnsembleNER`:
//! 1. Runs ALL available backends opportunistically (in parallel conceptually)
//! 2. Collects candidate entities with provenance
//! 3. Groups overlapping spans into conflict clusters
//! 4. Resolves conflicts using weighted voting with agreement bonus
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    ENSEMBLE NER ARCHITECTURE                            │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  Input: "Tim Cook, CEO of Apple, met with Sundar Pichai"                │
//! │                                                                         │
//! │  ┌──────────────────────────────────────────────────────────────────┐   │
//! │  │ PHASE 1: OPPORTUNISTIC EXTRACTION (parallel)                     │   │
//! │  │                                                                  │   │
//! │  │  Pattern ──────► [no entities]                                   │   │
//! │  │  Heuristic ────► Tim Cook (PER, 0.75), Apple (ORG, 0.80), ...    │   │
//! │  │  GLiNER ────────► Tim Cook (PER, 0.95), Apple (ORG, 0.87), ...   │   │
//! │  │  Candle ────────► [unavailable, skip]                            │   │
//! │  └──────────────────────────────────────────────────────────────────┘   │
//! │                            │                                            │
//! │                            ▼                                            │
//! │  ┌──────────────────────────────────────────────────────────────────┐   │
//! │  │ PHASE 2: CANDIDATE AGGREGATION                                   │   │
//! │  │                                                                  │   │
//! │  │  Span [0:8] "Tim Cook":                                          │   │
//! │  │    • Heuristic: PER (0.75)                                       │   │
//! │  │    • GLiNER: PER (0.95)                                          │   │
//! │  │    Agreement: 2/2 → HIGH confidence                              │   │
//! │  │                                                                  │   │
//! │  │  Span [17:22] "Apple":                                           │   │
//! │  │    • Heuristic: ORG (0.80)                                       │   │
//! │  │    • GLiNER: ORG (0.87)                                          │   │
//! │  │    Agreement: 2/2 → HIGH confidence                              │   │
//! │  └──────────────────────────────────────────────────────────────────┘   │
//! │                            │                                            │
//! │                            ▼                                            │
//! │  ┌──────────────────────────────────────────────────────────────────┐   │
//! │  │ PHASE 3: CONFLICT RESOLUTION (weighted voting)                   │   │
//! │  │                                                                  │   │
//! │  │  Backend weights (learned or configured):                        │   │
//! │  │    Pattern: 0.99 (when fires, almost always right)               │   │
//! │  │    GLiNER:  0.85 (ML-based, good accuracy)                       │   │
//! │  │    Heuristic: 0.65 (reasonable but noisy)                        │   │
//! │  │                                                                  │   │
//! │  │  For span [0:8]:                                                 │   │
//! │  │    Weighted vote = (0.65 * 0.75) + (0.85 * 0.95) = 1.29          │   │
//! │  │    Normalized confidence = 0.91                                  │   │
//! │  └──────────────────────────────────────────────────────────────────┘   │
//! │                            │                                            │
//! │                            ▼                                            │
//! │  ┌──────────────────────────────────────────────────────────────────┐   │
//! │  │ OUTPUT                                                           │   │
//! │  │                                                                  │   │
//! │  │  Entity { text: "Tim Cook", type: PER, conf: 0.91,               │   │
//! │  │           sources: ["heuristic", "gliner"], agreement: 1.0 }     │   │
//! │  └──────────────────────────────────────────────────────────────────┘   │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Conflict Resolution Strategies
//!
//! ## Weighted Voting (Unsupervised)
//!
//! Each backend has a weight based on its expected reliability:
//! - Pattern backends: high weight (0.95+) when they fire
//! - ML backends: medium-high weight (0.80-0.90)
//! - Heuristic backends: lower weight (0.60-0.70)
//!
//! ## Type-Conditioned Voting
//!
//! Some backends are better at certain types:
//! - Pattern: DATE, MONEY, EMAIL, URL (near-perfect)
//! - GLiNER: PER, ORG (good), LOC (decent)
//! - Heuristic: ORG (good with "Inc", "Corp"), PER (title+name patterns)
//!
//! ## Agreement Bonus
//!
//! When multiple backends agree on type AND span, boost confidence:
//! - 2 backends agree: +0.10 bonus
//! - 3+ backends agree: +0.15 bonus
//!
//! # Example
//!
//! ```rust
//! use anno::{Model, EnsembleNER};
//!
//! let ner = EnsembleNER::new();
//! let entities = ner.extract_entities("Tim Cook leads Apple Inc.", None).unwrap();
//!
//! // Each entity includes provenance and agreement info
//! for e in &entities {
//!     println!("{}: {} (conf: {:.2}, sources: {:?})",
//!              e.entity_type.as_label(), e.text, e.confidence,
//!              e.provenance.as_ref().map(|p| &p.source));
//! }
//! ```

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::{Entity, EntityType, Model, Result};

fn method_for_backend_id(backend_id: &str) -> anno_core::ExtractionMethod {
    match backend_id {
        // Stable IDs used by `EnsembleNER::new()`.
        "regex" => anno_core::ExtractionMethod::Pattern,
        "heuristic" => anno_core::ExtractionMethod::Heuristic,
        // Legacy backend id (deprecated, but still used in tests/compositions).
        "rule" => anno_core::ExtractionMethod::Heuristic,
        // Everything else: treat as neural by default.
        _ => anno_core::ExtractionMethod::Neural,
    }
}

// =============================================================================
// Backend Weights
// =============================================================================

/// Reliability weight for a backend (0.0 to 1.0).
///
/// Higher weight = more trusted when resolving conflicts.
#[derive(Debug, Clone, Copy)]
pub struct BackendWeight {
    /// Overall reliability of this backend
    pub overall: f64,
    /// Type-specific weights (optional overrides)
    pub per_type: Option<TypeWeights>,
}

impl Default for BackendWeight {
    fn default() -> Self {
        Self {
            overall: 0.5,
            per_type: None,
        }
    }
}

/// Type-specific reliability weights.
///
/// Different backends may have different accuracy profiles for different entity types.
/// These weights adjust confidence scores based on the entity type being extracted.
#[derive(Debug, Clone, Copy, Default)]
pub struct TypeWeights {
    /// Weight multiplier for Person entities
    pub person: f64,
    /// Weight multiplier for Organization entities
    pub organization: f64,
    /// Weight multiplier for Location entities
    pub location: f64,
    /// Weight multiplier for Date entities
    pub date: f64,
    /// Weight multiplier for Money entities
    pub money: f64,
    /// Weight multiplier for other/misc entity types
    pub other: f64,
}

impl TypeWeights {
    fn get(&self, entity_type: &EntityType) -> f64 {
        match entity_type {
            EntityType::Person => self.person,
            EntityType::Organization => self.organization,
            EntityType::Location => self.location,
            EntityType::Date => self.date,
            EntityType::Money => self.money,
            _ => self.other,
        }
    }
}

/// Default weights based on empirical observations.
fn default_backend_weights() -> HashMap<&'static str, BackendWeight> {
    let mut weights = HashMap::new();

    // Pattern backends: very high precision when they fire
    weights.insert(
        "regex",
        BackendWeight {
            overall: 0.98,
            per_type: Some(TypeWeights {
                date: 0.99,
                money: 0.99,
                person: 0.50, // Pattern doesn't do NER
                organization: 0.50,
                location: 0.50,
                other: 0.95, // URLs, emails, etc.
            }),
        },
    );

    // GLiNER: good ML-based NER
    weights.insert(
        "gliner",
        BackendWeight {
            overall: 0.85,
            per_type: Some(TypeWeights {
                person: 0.90,
                organization: 0.85,
                location: 0.80,
                date: 0.75,
                money: 0.70,
                other: 0.75,
            }),
        },
    );
    weights.insert(
        "GLiNER-ONNX",
        BackendWeight {
            overall: 0.85,
            per_type: Some(TypeWeights {
                person: 0.90,
                organization: 0.85,
                location: 0.80,
                date: 0.75,
                money: 0.70,
                other: 0.75,
            }),
        },
    );

    // GLiNER Candle
    weights.insert(
        "gliner-candle",
        BackendWeight {
            overall: 0.85,
            per_type: None,
        },
    );

    // BERT NER
    weights.insert(
        "bert-ner-onnx",
        BackendWeight {
            overall: 0.80,
            per_type: None,
        },
    );

    // Heuristic: reasonable but noisy
    weights.insert(
        "heuristic",
        BackendWeight {
            overall: 0.60,
            per_type: Some(TypeWeights {
                person: 0.65,       // Title + Name pattern works well
                organization: 0.70, // "Inc", "Corp" patterns
                location: 0.55,     // Context-dependent
                date: 0.40,         // Better to use pattern
                money: 0.40,
                other: 0.50,
            }),
        },
    );

    weights
}

// =============================================================================
// Candidate Entity (with source tracking)
// =============================================================================

/// An entity candidate from a specific backend.
#[derive(Debug, Clone)]
struct Candidate {
    entity: Entity,
    source: String,
    backend_weight: f64,
}

// =============================================================================
// Span Key (for grouping overlapping entities)
// =============================================================================

/// Key for grouping entities by span.
///
/// Two entities are considered "same span" if they significantly overlap.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SpanKey {
    start: usize,
    end: usize,
}

impl SpanKey {
    fn from_entity(e: &Entity) -> Self {
        Self {
            start: e.start,
            end: e.end,
        }
    }

    /// Check if two spans overlap significantly (>50% of smaller span).
    fn overlaps(&self, other: &SpanKey) -> bool {
        let overlap_start = self.start.max(other.start);
        let overlap_end = self.end.min(other.end);

        if overlap_start >= overlap_end {
            return false;
        }

        let overlap = overlap_end - overlap_start;
        let smaller_span = (self.end - self.start).min(other.end - other.start);

        // Overlap if >50% of smaller span is covered
        (overlap as f64 / smaller_span as f64) > 0.5
    }
}

// =============================================================================
// EnsembleNER
// =============================================================================

/// Ensemble NER that runs ALL backends and resolves conflicts via weighted voting.
///
/// Unlike [`StackedNER`] (priority-based cascade), `EnsembleNER`:
/// 1. Runs ALL backends in parallel (conceptually)
/// 2. Groups overlapping spans into conflict clusters
/// 3. Resolves via weighted voting with type-conditioned weights
/// 4. Applies agreement bonus when multiple backends agree
///
/// # When to Use
///
/// - **EnsembleNER**: Maximum accuracy, latency not critical
/// - **StackedNER**: Production, predictable latency, early exit
///
/// # Example
///
/// ```rust
/// use anno::{EnsembleNER, Model, RegexNER, HeuristicNER};
///
/// // Default: uses all available backends
/// let ensemble = EnsembleNER::new();
///
/// // Custom: specific backends
/// let custom = EnsembleNER::with_backends(vec![
///     Box::new(RegexNER::new()),
///     Box::new(HeuristicNER::new()),
/// ]);
///
/// let entities = custom.extract_entities("Contact us at test@example.com", None)?;
/// # Ok::<(), anno::Error>(())
/// ```
///
/// # Algorithm
///
/// 1. **Collect candidates**: Run each backend, tag results with provenance
/// 2. **Cluster overlaps**: Group candidates with >50% span overlap
/// 3. **Weighted vote**: Score each candidate by `backend_weight * confidence`
/// 4. **Agreement bonus**: Add +0.10 when 2+ backends agree on type
/// 5. **Select winner**: Highest weighted score wins the cluster
///
/// [`StackedNER`]: super::stacked::StackedNER
pub struct EnsembleNER {
    backends: Vec<Arc<dyn Model + Send + Sync>>,
    /// Stable backend IDs used for weighting and source tracking.
    ///
    /// This is intentionally decoupled from `backend.name()`, which is a
    /// human-facing label and may vary across implementations (e.g. "GLiNER-ONNX").
    backend_ids: Vec<String>,
    weights: HashMap<String, BackendWeight>,
    agreement_bonus: f64,
    min_confidence: f64,
    /// Transparent name showing constituent backends (e.g., "ensemble(regex|gliner|heuristic)")
    name: String,
    /// Cached static name (avoids Box::leak on every name() call)
    name_static: std::sync::OnceLock<&'static str>,
}

impl Default for EnsembleNER {
    fn default() -> Self {
        Self::new()
    }
}

impl EnsembleNER {
    /// Create ensemble with all available backends.
    #[must_use]
    pub fn new() -> Self {
        let mut backends: Vec<Arc<dyn Model + Send + Sync>> = Vec::new();
        let mut backend_ids: Vec<&'static str> = Vec::new();

        // Always add pattern (high precision for structured data)
        backends.push(Arc::new(crate::RegexNER::new()));
        backend_ids.push("regex");

        // Add GLiNER if available
        #[cfg(feature = "onnx")]
        {
            use super::GLiNEROnnx;
            use crate::DEFAULT_GLINER_MODEL;
            if let Ok(gliner) = GLiNEROnnx::new(DEFAULT_GLINER_MODEL) {
                backends.push(Arc::new(gliner));
                backend_ids.push("gliner");
            }
        }

        // Add Candle GLiNER if available
        #[cfg(feature = "candle")]
        {
            use super::GLiNERCandle;
            use crate::DEFAULT_GLINER_MODEL;
            if let Ok(candle) = GLiNERCandle::from_pretrained(DEFAULT_GLINER_MODEL) {
                backends.push(Arc::new(candle));
                backend_ids.push("gliner-candle");
            }
        }

        // Always add heuristic as fallback
        backends.push(Arc::new(crate::HeuristicNER::new()));
        backend_ids.push("heuristic");

        // Build transparent name showing constituents
        // Use '|' for parallel weighted voting (no priority ordering)
        let name = format!("ensemble({})", backend_ids.join("|"));

        // Convert default weights to owned strings
        let weights: HashMap<String, BackendWeight> = default_backend_weights()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        Self {
            backends,
            backend_ids: backend_ids.into_iter().map(str::to_string).collect(),
            weights,
            agreement_bonus: 0.10,
            min_confidence: 0.30,
            name,
            name_static: std::sync::OnceLock::new(),
        }
    }

    /// Create with custom backends.
    #[must_use]
    pub fn with_backends(backends: Vec<Box<dyn Model + Send + Sync>>) -> Self {
        // For custom backends, use the backend's reported name as both ID and display string.
        let backend_ids: Vec<String> = backends.iter().map(|b| b.name().to_string()).collect();
        let name = format!("ensemble({})", backend_ids.join("|"));

        let backends: Vec<Arc<dyn Model + Send + Sync>> =
            backends.into_iter().map(Arc::from).collect();

        let weights: HashMap<String, BackendWeight> = default_backend_weights()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        Self {
            backends,
            backend_ids,
            weights,
            agreement_bonus: 0.10,
            min_confidence: 0.30,
            name,
            name_static: std::sync::OnceLock::new(),
        }
    }

    /// Set custom backend weights.
    #[must_use]
    pub fn with_weights(mut self, weights: HashMap<String, BackendWeight>) -> Self {
        self.weights = weights;
        self
    }

    /// Set the agreement bonus (added when multiple backends agree).
    #[must_use]
    pub fn with_agreement_bonus(mut self, bonus: f64) -> Self {
        self.agreement_bonus = bonus;
        self
    }

    /// Set minimum confidence threshold.
    #[must_use]
    pub fn with_min_confidence(mut self, min: f64) -> Self {
        self.min_confidence = min;
        self
    }

    /// Get the weight for a backend and entity type.
    fn get_weight(&self, backend_name: &str, entity_type: &EntityType) -> f64 {
        if let Some(weight) = self.weights.get(backend_name) {
            if let Some(ref type_weights) = weight.per_type {
                type_weights.get(entity_type)
            } else {
                weight.overall
            }
        } else {
            // Unknown backend - use conservative default
            0.50
        }
    }

    /// Resolve overlapping candidates using weighted voting.
    fn resolve_candidates(&self, candidates: Vec<Candidate>) -> Option<Entity> {
        if candidates.is_empty() {
            return None;
        }

        if candidates.len() == 1 {
            // Single candidate - use its confidence directly
            let candidate = candidates
                .into_iter()
                .next()
                .expect("candidates.len() == 1 guarantees next() is Some");
            let mut entity = candidate.entity;
            let original_prov = entity.provenance.clone();
            let original_confidence = entity.confidence;
            // Slight penalty for single-source
            entity.confidence *= 0.95;
            // Set provenance for single-source entities
            entity.provenance = Some(anno_core::Provenance {
                source: std::borrow::Cow::Owned(format!("ensemble({})", candidate.source)),
                // Preserve underlying method/pattern when possible (important for nested ensembles).
                method: original_prov
                    .as_ref()
                    .map(|p| p.method)
                    .unwrap_or_else(|| method_for_backend_id(&candidate.source)),
                pattern: original_prov.as_ref().and_then(|p| p.pattern.clone()),
                raw_confidence: original_prov
                    .as_ref()
                    .and_then(|p| p.raw_confidence)
                    .or(Some(original_confidence)),
                model_version: None,
                timestamp: None,
            });
            return Some(entity);
        }

        // Group by entity type
        let mut type_votes: HashMap<String, Vec<&Candidate>> = HashMap::new();
        for c in &candidates {
            let type_key = c.entity.entity_type.as_label().to_string();
            type_votes.entry(type_key).or_default().push(c);
        }

        // Find the type with highest weighted vote (deterministic tie-breaking).
        //
        // HashMap iteration order can vary across process runs. If two types tie on
        // weighted_sum, we still need a stable selection.
        //
        // Ordering:
        // 1) Higher weighted_sum wins
        // 2) If tied, more candidates (more votes) wins
        // 3) If tied, lexicographically smaller type key wins
        let mut best_type: Option<(String, f64, usize, Vec<&Candidate>)> = None;
        for (type_key, type_candidates) in &type_votes {
            let weighted_sum: f64 = type_candidates
                .iter()
                .map(|c| c.backend_weight * c.entity.confidence)
                .sum();
            let count = type_candidates.len();

            let should_replace = match &best_type {
                None => true,
                Some((best_key, best_sum, best_count, _)) => {
                    if weighted_sum > *best_sum {
                        true
                    } else if weighted_sum < *best_sum {
                        false
                    } else if count > *best_count {
                        true
                    } else if count < *best_count {
                        false
                    } else {
                        type_key < best_key
                    }
                }
            };

            if should_replace {
                best_type = Some((
                    type_key.clone(),
                    weighted_sum,
                    count,
                    type_candidates.clone(),
                ));
            }
        }

        let (_type_key, weighted_sum, _count, winning_candidates) = best_type?;

        // Calculate ensemble confidence
        let num_sources = winning_candidates.len();
        let total_weight: f64 = winning_candidates.iter().map(|c| c.backend_weight).sum();

        let base_confidence = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.5
        };

        // Agreement bonus
        let agreement_bonus = if num_sources >= 3 {
            self.agreement_bonus * 1.5
        } else if num_sources >= 2 {
            self.agreement_bonus
        } else {
            0.0
        };

        let final_confidence = (base_confidence + agreement_bonus).min(1.0);

        // Build merged entity
        // Use the candidate with highest individual confidence as base
        let best_candidate = winning_candidates.iter().max_by(|a, b| {
            a.entity
                .confidence
                .partial_cmp(&b.entity.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;

        let sources: Vec<String> = winning_candidates
            .iter()
            .map(|c| c.source.clone())
            .collect();

        // Calculate hierarchical confidence scores
        // - linkage: How many backends detected an entity here (normalized)
        // - type_score: Agreement on type classification
        // - boundary: Agreement on exact span boundaries
        let total_candidates = candidates.len() as f32;
        let num_winners = winning_candidates.len() as f32;

        // Linkage: ratio of candidates in winning type
        let linkage = if total_candidates > 0.0 {
            (num_winners / total_candidates).min(1.0)
        } else {
            0.5
        };

        // Type score: confidence in the winning type (weighted)
        let type_score = final_confidence as f32;

        // Boundary: agreement on span boundaries
        // Check if all winning candidates have the same start/end
        let reference_span = (best_candidate.entity.start, best_candidate.entity.end);
        let span_agreement_count = winning_candidates
            .iter()
            .filter(|c| c.entity.start == reference_span.0 && c.entity.end == reference_span.1)
            .count();
        let boundary = if num_winners > 0.0 {
            (span_agreement_count as f32 / num_winners).min(1.0)
        } else {
            1.0
        };

        let mut entity = best_candidate.entity.clone();
        entity.confidence = final_confidence;
        entity.hierarchical_confidence = Some(anno_core::HierarchicalConfidence::new(
            linkage, type_score, boundary,
        ));
        entity.provenance = Some(anno_core::Provenance {
            source: Cow::Owned(format!("ensemble({})", sources.join("+"))),
            method: anno_core::ExtractionMethod::Consensus,
            pattern: None,
            raw_confidence: Some(base_confidence),
            model_version: None,
            timestamp: None,
        });

        Some(entity)
    }
}

impl Model for EnsembleNER {
    fn extract_entities(&self, text: &str, language: Option<&str>) -> Result<Vec<Entity>> {
        if self.backends.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 1: Collect candidates from all backends
        let mut all_candidates: Vec<Candidate> = Vec::new();

        for (i, backend) in self.backends.iter().enumerate() {
            let backend_id = self
                .backend_ids
                .get(i)
                .cloned()
                .unwrap_or_else(|| backend.name().to_string());

            match backend.extract_entities(text, language) {
                Ok(entities) => {
                    for entity in entities {
                        let weight = self.get_weight(&backend_id, &entity.entity_type);
                        all_candidates.push(Candidate {
                            entity,
                            source: backend_id.clone(),
                            backend_weight: weight,
                        });
                    }
                }
                Err(e) => {
                    // Log but continue (opportunistic)
                    log::debug!(
                        "EnsembleNER: Backend {} (id={}) failed: {}",
                        backend.name(),
                        backend_id,
                        e
                    );
                }
            }
        }

        if all_candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 2: Group candidates by overlapping spans
        let mut span_groups: Vec<Vec<Candidate>> = Vec::new();

        for candidate in all_candidates {
            let span = SpanKey::from_entity(&candidate.entity);

            // Find existing group with overlapping span
            let mut found_group = false;
            for group in &mut span_groups {
                if let Some(first) = group.first() {
                    let existing_span = SpanKey::from_entity(&first.entity);
                    if span.overlaps(&existing_span) {
                        group.push(candidate.clone());
                        found_group = true;
                        break;
                    }
                }
            }

            if !found_group {
                span_groups.push(vec![candidate]);
            }
        }

        // Phase 3: Resolve each group
        let mut results: Vec<Entity> = Vec::new();

        for group in span_groups {
            if let Some(entity) = self.resolve_candidates(group) {
                if entity.confidence >= self.min_confidence {
                    results.push(entity);
                }
            }
        }

        // Sort by position
        results.sort_by_key(|e| (e.start, e.end));

        Ok(results)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        // Union of all backend types
        let mut types: Vec<EntityType> = Vec::new();
        for backend in &self.backends {
            for t in backend.supported_types() {
                if !types.contains(&t) {
                    types.push(t);
                }
            }
        }
        types
    }

    fn is_available(&self) -> bool {
        // Available if at least one backend is available
        self.backends.iter().any(|b| b.is_available())
    }

    fn name(&self) -> &'static str {
        // Use OnceLock to cache the static string, avoiding repeated memory leaks
        self.name_static
            .get_or_init(|| Box::leak(self.name.clone().into_boxed_str()))
    }

    fn description(&self) -> &'static str {
        "Ensemble NER: weighted voting across multiple backends"
    }
}

// Implement required traits
impl crate::NamedEntityCapable for EnsembleNER {}

impl crate::BatchCapable for EnsembleNER {
    fn optimal_batch_size(&self) -> Option<usize> {
        Some(8) // Reasonable default for ensemble
    }
}

impl crate::StreamingCapable for EnsembleNER {
    fn recommended_chunk_size(&self) -> usize {
        8192
    }
}

// =============================================================================
// Weight Learning
// =============================================================================

/// Training example for weight learning.
#[derive(Debug, Clone)]
pub struct WeightTrainingExample {
    /// Text of the entity
    pub text: String,
    /// True entity type (gold label)
    pub gold_type: EntityType,
    /// Span start
    pub start: usize,
    /// Span end
    pub end: usize,
    /// Predictions from each backend: (backend_name, predicted_type, confidence)
    pub predictions: Vec<(String, EntityType, f64)>,
}

/// Statistics for weight learning.
#[derive(Debug, Clone, Default)]
pub struct BackendStats {
    /// Total correct predictions
    pub correct: usize,
    /// Total predictions made
    pub total: usize,
    /// Per-type statistics: (type, correct, total)
    pub per_type: HashMap<String, (usize, usize)>,
}

impl BackendStats {
    /// Calculate overall precision.
    pub fn precision(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }

    /// Calculate per-type precision.
    pub fn type_precision(&self, entity_type: &str) -> f64 {
        if let Some((correct, total)) = self.per_type.get(entity_type) {
            if *total == 0 {
                0.0
            } else {
                *correct as f64 / *total as f64
            }
        } else {
            0.0
        }
    }
}

/// Weight learner for EnsembleNER.
///
/// Learns optimal backend weights from evaluation data.
///
/// # Example
///
/// ```rust,ignore
/// use anno::backends::ensemble::{EnsembleNER, WeightLearner};
///
/// let mut learner = WeightLearner::new();
///
/// // Add training examples from gold data
/// for (text, gold_entities) in gold_data {
///     learner.add_examples(&text, &gold_entities, &backends);
/// }
///
/// // Learn weights
/// let learned_weights = learner.learn_weights();
///
/// // Create ensemble with learned weights
/// let ensemble = EnsembleNER::new().with_weights(learned_weights);
/// ```
pub struct WeightLearner {
    /// Per-backend statistics
    backend_stats: HashMap<String, BackendStats>,
    /// Smoothing factor for precision (avoid division by zero / overfitting)
    smoothing: f64,
}

impl Default for WeightLearner {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightLearner {
    /// Create a new weight learner.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend_stats: HashMap::new(),
            smoothing: 1.0, // Laplace smoothing
        }
    }

    /// Set smoothing factor.
    #[must_use]
    pub fn with_smoothing(mut self, smoothing: f64) -> Self {
        self.smoothing = smoothing;
        self
    }

    /// Add a training example.
    pub fn add_example(&mut self, example: &WeightTrainingExample) {
        for (backend_name, predicted_type, _confidence) in &example.predictions {
            let stats = self.backend_stats.entry(backend_name.clone()).or_default();

            stats.total += 1;
            let correct = *predicted_type == example.gold_type;
            if correct {
                stats.correct += 1;
            }

            // Per-type stats
            let type_key = example.gold_type.as_label().to_string();
            let type_stats = stats.per_type.entry(type_key).or_insert((0, 0));
            type_stats.1 += 1;
            if correct {
                type_stats.0 += 1;
            }
        }
    }

    /// Add examples from gold entities and backend predictions.
    ///
    /// Runs each backend on the text and compares to gold entities.
    pub fn add_from_backends(
        &mut self,
        text: &str,
        gold_entities: &[Entity],
        backends: &[(&str, &dyn Model)],
    ) {
        // Get predictions from each backend
        let mut backend_preds: HashMap<String, Vec<Entity>> = HashMap::new();
        for (name, backend) in backends {
            if let Ok(entities) = backend.extract_entities(text, None) {
                backend_preds.insert(name.to_string(), entities);
            }
        }

        // Match predictions to gold entities
        for gold in gold_entities {
            let mut example = WeightTrainingExample {
                text: gold.text.clone(),
                gold_type: gold.entity_type.clone(),
                start: gold.start,
                end: gold.end,
                predictions: Vec::new(),
            };

            for (backend_name, entities) in &backend_preds {
                // Find matching prediction (same span)
                for pred in entities {
                    if pred.start == gold.start && pred.end == gold.end {
                        example.predictions.push((
                            backend_name.clone(),
                            pred.entity_type.clone(),
                            pred.confidence,
                        ));
                        break;
                    }
                }
            }

            if !example.predictions.is_empty() {
                self.add_example(&example);
            }
        }
    }

    /// Learn optimal weights from accumulated statistics.
    ///
    /// Uses precision-based weighting with Laplace smoothing.
    pub fn learn_weights(&self) -> HashMap<String, BackendWeight> {
        let mut weights = HashMap::new();

        for (backend_name, stats) in &self.backend_stats {
            // Smoothed precision: (correct + smoothing) / (total + 2*smoothing)
            let smoothed_precision = (stats.correct as f64 + self.smoothing)
                / (stats.total as f64 + 2.0 * self.smoothing);

            // Per-type weights
            let mut type_weights = TypeWeights::default();
            for (type_key, (correct, total)) in &stats.per_type {
                let type_precision =
                    (*correct as f64 + self.smoothing) / (*total as f64 + 2.0 * self.smoothing);

                match type_key.as_str() {
                    "PER" | "PERSON" => type_weights.person = type_precision,
                    "ORG" | "ORGANIZATION" => type_weights.organization = type_precision,
                    "LOC" | "LOCATION" | "GPE" => type_weights.location = type_precision,
                    "DATE" => type_weights.date = type_precision,
                    "MONEY" => type_weights.money = type_precision,
                    _ => type_weights.other = type_precision,
                }
            }

            weights.insert(
                backend_name.clone(),
                BackendWeight {
                    overall: smoothed_precision,
                    per_type: Some(type_weights),
                },
            );
        }

        weights
    }

    /// Get statistics for a backend.
    pub fn get_stats(&self, backend_name: &str) -> Option<&BackendStats> {
        self.backend_stats.get(backend_name)
    }

    /// Get all backend names.
    pub fn backend_names(&self) -> Vec<&String> {
        self.backend_stats.keys().collect()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use anno_core::ExtractionMethod;

    fn fast_ensemble() -> EnsembleNER {
        // Keep unit tests deterministic and fast: do not initialize model-loading backends here.
        EnsembleNER::with_backends(vec![
            Box::new(crate::RegexNER::new()),
            Box::new(crate::HeuristicNER::new()),
        ])
    }

    #[test]
    fn test_new_backend_ids_have_weights() {
        let ner = EnsembleNER::new();

        // For the built-in constructor, we require stable IDs so weights apply as intended.
        assert!(
            !ner.backend_ids.is_empty(),
            "EnsembleNER::new() should have at least one backend"
        );

        for id in &ner.backend_ids {
            assert!(
                ner.weights.contains_key(id),
                "EnsembleNER::new(): missing weight for backend id={:?}. This usually means the ensemble's advertised IDs drifted from default_backend_weights keys.",
                id
            );
        }
    }

    #[test]
    fn test_ensemble_basic() {
        let ner = fast_ensemble();
        let entities = ner
            .extract_entities("Tim Cook is the CEO of Apple Inc.", None)
            .unwrap();

        // Should find at least some entities
        assert!(!entities.is_empty(), "Should extract entities");

        // Check that provenance exists (may or may not say "ensemble" for single-source entities)
        for e in &entities {
            assert!(
                e.provenance.is_some(),
                "All entities should have provenance"
            );
        }
    }

    #[test]
    fn test_span_overlap() {
        // Span1 [0-10], Span2 [5-15]: overlap [5-10] = 5 chars
        // Smaller span = 10 chars, overlap/smaller = 5/10 = 0.5
        // Need >0.5 so this is borderline - adjust test
        let span1 = SpanKey { start: 0, end: 10 };
        let span2 = SpanKey { start: 3, end: 15 }; // overlap [3-10] = 7 chars, 7/10 = 0.7 > 0.5
        let span3 = SpanKey { start: 20, end: 30 };

        assert!(span1.overlaps(&span2), "Overlapping spans should match");
        assert!(
            !span1.overlaps(&span3),
            "Non-overlapping spans should not match"
        );
    }

    #[test]
    fn test_backend_weights() {
        let weights = default_backend_weights();

        // Pattern should have high weight
        assert!(weights["regex"].overall > 0.9);

        // GLiNER should have good weight
        assert!(weights["gliner"].overall > 0.8);

        // Heuristic should have lower weight
        assert!(weights["heuristic"].overall < 0.7);
    }

    #[test]
    fn test_type_specific_weights() {
        let weights = default_backend_weights();

        // Pattern should be best for dates
        let pattern_date = weights["regex"].per_type.as_ref().unwrap().date;
        let heuristic_date = weights["heuristic"].per_type.as_ref().unwrap().date;
        assert!(pattern_date > heuristic_date);

        // Heuristic should be decent for orgs
        let heuristic_org = weights["heuristic"].per_type.as_ref().unwrap().organization;
        assert!(heuristic_org > 0.6);
    }

    #[test]
    fn test_agreement_bonus() {
        let ner = fast_ensemble().with_agreement_bonus(0.15);
        assert!((ner.agreement_bonus - 0.15).abs() < 0.001);
    }

    #[test]
    fn test_weight_learner_basic() {
        let mut learner = WeightLearner::new();

        // Add some training examples
        learner.add_example(&WeightTrainingExample {
            text: "Apple".to_string(),
            gold_type: EntityType::Organization,
            start: 0,
            end: 5,
            predictions: vec![
                ("heuristic".to_string(), EntityType::Organization, 0.8),
                ("gliner".to_string(), EntityType::Organization, 0.9),
            ],
        });

        learner.add_example(&WeightTrainingExample {
            text: "Paris".to_string(),
            gold_type: EntityType::Location,
            start: 0,
            end: 5,
            predictions: vec![
                ("heuristic".to_string(), EntityType::Person, 0.6), // Wrong!
                ("gliner".to_string(), EntityType::Location, 0.85),
            ],
        });

        // Learn weights
        let weights = learner.learn_weights();

        // GLiNER should have higher weight (2/2 correct vs 1/2)
        let gliner_weight = weights.get("gliner").map(|w| w.overall).unwrap_or(0.0);
        let heuristic_weight = weights.get("heuristic").map(|w| w.overall).unwrap_or(0.0);

        assert!(
            gliner_weight > heuristic_weight,
            "GLiNER should have higher weight (was {} vs {})",
            gliner_weight,
            heuristic_weight
        );
    }

    #[test]
    fn test_backend_stats() {
        let mut stats = BackendStats {
            correct: 8,
            total: 10,
            ..Default::default()
        };
        stats.per_type.insert("PER".to_string(), (5, 6));

        assert!((stats.precision() - 0.8).abs() < 0.01);
        assert!((stats.type_precision("PER") - 0.833).abs() < 0.01);
        assert!((stats.type_precision("ORG") - 0.0).abs() < 0.01); // Unknown type
    }

    // =========================================================================
    // Additional Edge Case Tests
    // =========================================================================

    #[test]
    fn test_empty_text() {
        let ner = fast_ensemble();
        let entities = ner.extract_entities("", None).unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_whitespace_only_text() {
        let ner = fast_ensemble();
        let entities = ner.extract_entities("   \t\n   ", None).unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_resolve_candidates_tie_break_is_order_independent() {
        let ner = fast_ensemble();
        let span_text = "Apple";
        let span = (0, 5);

        let e_person = Entity::new(span_text, EntityType::Person, span.0, span.1, 0.5);
        let e_org = Entity::new(span_text, EntityType::Organization, span.0, span.1, 0.5);

        let c1 = Candidate {
            entity: e_person,
            source: "heuristic".to_string(),
            backend_weight: 1.0,
        };
        let c2 = Candidate {
            entity: e_org,
            source: "heuristic".to_string(),
            backend_weight: 1.0,
        };

        let out_a = ner
            .resolve_candidates(vec![c1.clone(), c2.clone()])
            .expect("should resolve");
        let out_b = ner
            .resolve_candidates(vec![c2, c1])
            .expect("should resolve");

        assert_eq!(
            out_a.entity_type, out_b.entity_type,
            "tie resolution should not depend on candidate order"
        );

        let key_a = out_a.entity_type.as_label().to_string();
        let person_key = EntityType::Person.as_label().to_string();
        let org_key = EntityType::Organization.as_label().to_string();
        let expected = std::cmp::min(person_key, org_key);
        assert_eq!(
            key_a, expected,
            "tie-break should choose lexicographically smallest type label"
        );
    }

    #[test]
    fn test_single_source_preserves_underlying_method_and_pattern() {
        // With a single backend, ensemble should preserve the backend's extraction method/pattern
        // (important for explainability and nested composition).
        let ner = EnsembleNER::with_backends(vec![Box::new(crate::RegexNER::new())]);
        let text = "Contact test@email.com on 2024-01-15";
        let entities = ner.extract_entities(text, None).expect("extract");
        assert!(!entities.is_empty());

        let email = entities
            .iter()
            .find(|e| e.text == "test@email.com")
            .expect("email entity should exist");
        let prov = email.provenance.as_ref().expect("provenance");

        assert_eq!(prov.method, ExtractionMethod::Pattern);
        assert!(
            prov.pattern.is_some(),
            "expected to preserve regex pattern name"
        );
    }

    #[test]
    fn test_nested_single_source_preserves_inner_method() {
        // Inner ensemble produces provenance.method = Heuristic; outer should not overwrite it
        // to Neural just because the backend id is "ensemble(...)".
        let inner = EnsembleNER::with_backends(vec![Box::new(crate::HeuristicNER::new())]);
        let outer = EnsembleNER::with_backends(vec![Box::new(inner)]);

        let text = "John Smith visited Paris.";
        let entities = outer.extract_entities(text, None).expect("extract");
        assert!(!entities.is_empty());

        for e in &entities {
            let prov = e.provenance.as_ref().expect("provenance");
            assert_eq!(
                prov.method,
                ExtractionMethod::Heuristic,
                "expected outer to preserve inner method"
            );
        }
    }

    #[test]
    fn test_span_key_self_overlap() {
        let span = SpanKey { start: 0, end: 10 };
        assert!(span.overlaps(&span), "Span should overlap with itself");
    }

    #[test]
    fn test_span_key_adjacent_no_overlap() {
        let span1 = SpanKey { start: 0, end: 10 };
        let span2 = SpanKey { start: 10, end: 20 };
        assert!(!span1.overlaps(&span2), "Adjacent spans should not overlap");
    }

    #[test]
    fn test_span_key_contained() {
        let outer = SpanKey { start: 0, end: 20 };
        let inner = SpanKey { start: 5, end: 15 };
        assert!(outer.overlaps(&inner), "Contained spans should overlap");
        assert!(inner.overlaps(&outer), "Overlap should be symmetric");
    }

    #[test]
    fn test_backend_stats_empty() {
        let stats = BackendStats::default();
        assert!((stats.precision() - 0.0).abs() < 0.001);
        assert!((stats.type_precision("ANY") - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_weight_learner_empty() {
        let learner = WeightLearner::new();
        let weights = learner.learn_weights();
        // Empty learner returns empty weights (caller should use defaults)
        // Just verify it doesn't panic and returns a valid HashMap
        let _ = weights.len();
    }

    #[test]
    fn test_ensemble_with_language() {
        let ner = fast_ensemble();

        // Try with English language hint
        let entities = ner
            .extract_entities("Tim Cook is the CEO of Apple.", Some("en"))
            .unwrap();

        // Should find entities (language hint shouldn't break anything)
        assert!(
            !entities.is_empty(),
            "Should find entities with language hint"
        );
    }

    #[test]
    fn test_type_weights_structure() {
        let weights = TypeWeights {
            person: 0.9,
            location: 0.85,
            organization: 0.88,
            date: 0.95,
            money: 0.8,
            other: 0.7,
        };

        assert!(weights.person > 0.0);
        assert!(weights.date > weights.other);
    }

    #[test]
    fn test_backend_weight_structure() {
        let weight = BackendWeight {
            overall: 0.85,
            per_type: Some(TypeWeights {
                person: 0.9,
                location: 0.88,
                organization: 0.87,
                date: 0.92,
                money: 0.85,
                other: 0.75,
            }),
        };

        assert!(weight.overall > 0.0);
        assert!(weight.per_type.is_some());
    }

    #[test]
    fn test_unicode_extraction() {
        let ner = EnsembleNER::new();
        let entities = ner
            .extract_entities("東京で会議がありました。", None)
            .unwrap();

        // Should not crash on Unicode
        for e in &entities {
            assert!(e.confidence >= 0.0 && e.confidence <= 1.0);
        }
    }

    #[test]
    fn test_ensemble_provenance_tracking() {
        let ner = EnsembleNER::new();
        let entities = ner
            .extract_entities("Barack Obama visited Paris yesterday.", None)
            .unwrap();

        for e in &entities {
            // All entities should have provenance
            assert!(
                e.provenance.is_some(),
                "Entity '{}' ({:?}) at {}..{} has no provenance",
                e.text,
                e.entity_type,
                e.start,
                e.end
            );
            let prov = e.provenance.as_ref().unwrap();
            // Provenance source should not be empty
            assert!(!prov.source.is_empty());
        }
    }
}
