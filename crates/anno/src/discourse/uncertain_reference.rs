//! Uncertain Reference and Deferred Resolution.
//!
//! # The Problem: Premature Commitment
//!
//! Most coreference systems resolve pronouns immediately: see "he", find antecedent,
//! commit. But this fails on:
//!
//! ```text
//! "When she finally arrived, Mary apologized for being late."
//! ```
//!
//! Here "she" appears *before* its antecedent "Mary" (cataphora). An immediate
//! resolver has no candidates; a deferred resolver waits and resolves later.
//!
//! Similarly:
//!
//! ```text
//! "John told Bill that he would be late."
//! ```
//!
//! Is "he" John or Bill? Both are valid. Rather than guess immediately, we can
//! maintain a *distribution* over candidates and wait for disambiguating context.
//!
//! # Why Deferred Resolution?
//!
//! The key insight from Israel (1994) is that reference is often *indeterminate*
//! at the point of mention. The identity of a referent emerges through discourse,
//! not at a single moment. Forcing immediate resolution:
//!
//! 1. **Loses information** — collapsing a distribution to a point estimate
//! 2. **Propagates errors** — wrong early decisions cascade downstream
//! 3. **Misses cataphora** — can't handle forward references at all
//!
//! Deferred resolution maintains uncertainty explicitly, resolving only when
//! forced (end of document, downstream task requires it) or when confidence
//! exceeds a threshold.
//!
//! # Epsilon-Term Semantics
//!
//! Israel (1994) discusses Hilbert's epsilon-terms as an alternative to standard
//! quantifiers. For a predicate `A[x]`, the term `ε_x(A[x])` denotes "some A" without
//! specifying which particular A. This is exactly what we need:
//!
//! - **Not unbound**: We know *something* is referred to
//! - **Not uniquely bound**: We don't know *which* entity yet
//! - **Bound to a set**: Candidates with associated probabilities
//!
//! In coreference resolution, this translates to a model where:
//!
//! 1. **Introduction**: A discourse referent is introduced with uncertain identity
//! 2. **Refinement**: As context arrives, candidates are pruned or reweighted
//! 3. **Resolution**: When disambiguation is required, the best candidate is selected
//!
//! # Use Cases
//!
//! - **Cataphora**: Forward-referencing pronouns ("When *she* arrived, Mary...")
//! - **Ambiguous pronouns**: Multiple valid antecedents ("John told Bill that he...")
//! - **Bridging inference**: Inferrable referents ("the door" after "a room")
//! - **Abstract anaphora**: "This shows that..." where "this" refers to a proposition
//!
//! # Connection to Dynamic Semantics
//!
//! Israel notes that in proof systems, parameters/assumptions are introduced
//! with "indeterminate" identity—their binding is determined by subsequent
//! proof structure. Natural language works similarly: we introduce referents
//! and let later context determine their identity.
//!
//! # Example
//!
//! ```rust
//! use anno::discourse::uncertain_reference::{
//!     UncertainReference, ReferenceCandidate, resolve_uncertain
//! };
//!
//! let mut uncertain = UncertainReference::new("someone who works here");
//!
//! // Add candidates from context
//! uncertain.add_candidate(ReferenceCandidate::new(1, "John", 0.6));
//! uncertain.add_candidate(ReferenceCandidate::new(2, "Mary", 0.4));
//!
//! // Later, new evidence arrives
//! uncertain.update_evidence(1, 0.3); // Boost John
//! uncertain.update_evidence(2, -0.2); // Demote Mary
//!
//! // Resolve when needed
//! let resolved = uncertain.resolve().expect("should resolve");
//! assert_eq!(resolved.entity_id, 1);
//! ```
//!
//! # References
//!
//! - Israel (1994): "The Very Idea of Dynamic Semantics"
//! - Hilbert & Bernays (1939): Grundlagen der Mathematik (epsilon calculus)
//! - Kehler (1997): Current Theories of Centering for Pronoun Interpretation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Core Types
// =============================================================================

/// A candidate referent for uncertain reference resolution.
///
/// Each candidate represents a possible antecedent with an associated weight
/// (in log-odds space for numerical stability) and metadata about how it was
/// identified and which constraints it satisfies or violates.
///
/// # Example
///
/// ```rust
/// use anno::discourse::uncertain_reference::{ReferenceCandidate, CandidateSource};
///
/// let candidate = ReferenceCandidate::new(1, "John Smith", 0.8)
///     .with_source(CandidateSource::Discourse)
///     .satisfies("gender:masculine")
///     .satisfies("number:singular");
///
/// // Check for hard constraint violations
/// let bad_candidate = ReferenceCandidate::new(2, "they", 0.6)
///     .violates("number:singular");  // "they" violates singular constraint
///
/// assert!(!candidate.has_violations());
/// assert!(bad_candidate.has_violations());
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceCandidate {
    /// Entity/cluster ID (links to coreference cluster).
    pub entity_id: u64,
    /// Surface form or description (e.g., "John", "the CEO").
    pub description: String,
    /// Weight in log-odds space. Higher = more likely. Use `update_evidence`
    /// to adjust, and [`probabilities`](UncertainReference::probabilities) to
    /// convert to probabilities via softmax.
    pub weight: f64,
    /// How this candidate was identified (discourse, bridging, world knowledge).
    pub source: CandidateSource,
    /// Constraints this candidate satisfies (e.g., "gender:masculine").
    pub satisfied_constraints: Vec<String>,
    /// Constraints this candidate violates. Hard constraint violations can be
    /// pruned via [`UncertainReference::prune_violations`].
    pub violated_constraints: Vec<String>,
}

/// How a candidate antecedent was identified.
///
/// Different sources may have different reliability or require different
/// handling during resolution.
///
/// # Example
///
/// ```rust
/// use anno::discourse::uncertain_reference::{ReferenceCandidate, CandidateSource};
///
/// // Explicit antecedent from prior text
/// let discourse = ReferenceCandidate::new(1, "John", 0.9)
///     .with_source(CandidateSource::Discourse);
///
/// // Inferrable via bridging ("the door" after "a room")
/// let bridging = ReferenceCandidate::new(2, "door#42", 0.7)
///     .with_source(CandidateSource::Bridging);
///
/// // Known from world knowledge ("the president")
/// let world = ReferenceCandidate::new(3, "POTUS", 0.8)
///     .with_source(CandidateSource::WorldKnowledge);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum CandidateSource {
    /// From explicit antecedent in prior discourse.
    /// E.g., "John" mentioned in previous sentence.
    #[default]
    Discourse,
    /// From world knowledge or common ground.
    /// E.g., "the sun", "the president" — identifiable without prior mention.
    WorldKnowledge,
    /// From bridging inference based on discourse relations.
    /// E.g., "the door" after "John entered a room" — inferred part-whole.
    Bridging,
    /// From accommodation — adding a new entity to the discourse model
    /// based on context (Lewis 1979).
    Accommodation,
    /// From cataphoric (forward) reference resolution.
    /// E.g., "When she arrived, Mary..." — resolved after seeing "Mary".
    Cataphoric,
}

impl ReferenceCandidate {
    /// Create a new candidate.
    #[must_use]
    pub fn new(entity_id: u64, description: impl Into<String>, weight: f64) -> Self {
        Self {
            entity_id,
            description: description.into(),
            weight,
            source: CandidateSource::default(),
            satisfied_constraints: Vec::new(),
            violated_constraints: Vec::new(),
        }
    }

    /// Set the source of this candidate.
    #[must_use]
    pub fn with_source(mut self, source: CandidateSource) -> Self {
        self.source = source;
        self
    }

    /// Add a satisfied constraint.
    #[must_use]
    pub fn satisfies(mut self, constraint: impl Into<String>) -> Self {
        self.satisfied_constraints.push(constraint.into());
        self
    }

    /// Add a violated constraint.
    #[must_use]
    pub fn violates(mut self, constraint: impl Into<String>) -> Self {
        self.violated_constraints.push(constraint.into());
        self
    }

    /// Check if this candidate has violations.
    #[must_use]
    pub fn has_violations(&self) -> bool {
        !self.violated_constraints.is_empty()
    }
}

// =============================================================================
// Uncertain Reference
// =============================================================================

/// An uncertain reference awaiting resolution.
///
/// This is anno's implementation of epsilon-term semantics for coreference.
/// The reference maintains a distribution over possible antecedents that is
/// refined as more context arrives, allowing deferred commitment.
///
/// # Lifecycle
///
/// 1. **Create**: Initialize with descriptive content (the anaphor)
/// 2. **Populate**: Add candidate antecedents via [`add_candidate`](Self::add_candidate)
/// 3. **Refine**: Update evidence as context arrives via [`update_evidence`](Self::update_evidence)
/// 4. **Prune**: Remove unlikely candidates via [`prune`](Self::prune) or
///    [`prune_violations`](Self::prune_violations)
/// 5. **Resolve**: Force resolution via [`resolve`](Self::resolve) when needed
///
/// # Example
///
/// ```rust
/// use anno::discourse::uncertain_reference::{
///     UncertainReference, ReferenceCandidate, ConstraintKind,
/// };
///
/// // Create uncertain reference for "she"
/// let mut reference = UncertainReference::new("she");
///
/// // Add hard constraint: must be feminine
/// reference.add_constraint(ConstraintKind::Gender, "feminine", true);
///
/// // Add candidates
/// reference.add_candidate(ReferenceCandidate::new(1, "Mary", 0.6));
/// reference.add_candidate(
///     ReferenceCandidate::new(2, "John", 0.4)
///         .violates("gender:feminine")
/// );
///
/// // Prune candidates violating hard constraints
/// reference.prune_violations();
/// assert_eq!(reference.candidate_count(), 1);
///
/// // Resolve
/// let resolved = reference.resolve().unwrap();
/// assert_eq!(resolved.entity_id, 1);
/// ```
///
/// # Uncertainty Measures
///
/// - [`entropy`](Self::entropy): Shannon entropy of candidate distribution
/// - [`is_ambiguous`](Self::is_ambiguous): Multiple high-probability candidates?
/// - [`probabilities`](Self::probabilities): Softmax distribution over candidates
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UncertainReference {
    /// Descriptive content — the anaphor text or the "A" in `ε_x A[x]`.
    pub description: String,

    /// Candidate referents with weights (in log-odds space).
    pub candidates: Vec<ReferenceCandidate>,

    /// Whether resolution has been forced via [`resolve`](Self::resolve).
    pub resolved: bool,

    /// The resolved entity ID (set after calling [`resolve`](Self::resolve)).
    pub resolved_entity: Option<u64>,

    /// Constraints from the anaphor (gender, number, animacy, etc.).
    /// Hard constraints filter candidates; soft constraints rank them.
    pub constraints: Vec<ReferenceConstraint>,

    /// Position in discourse (character offset or utterance index).
    pub discourse_position: Option<usize>,

    /// Whether this is cataphoric (forward-referencing).
    /// Cataphoric references are resolved when later context provides the antecedent.
    pub is_cataphoric: bool,
}

/// A constraint on reference resolution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceConstraint {
    /// Constraint type
    pub kind: ConstraintKind,
    /// Expected value
    pub value: String,
    /// Whether this is hard (must satisfy) or soft (preference)
    pub is_hard: bool,
}

/// Types of reference constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintKind {
    /// Gender agreement
    Gender,
    /// Number agreement
    Number,
    /// Person agreement
    Person,
    /// Animacy requirement
    Animacy,
    /// Semantic type requirement
    SemanticType,
    /// Syntactic binding (e.g., Binding Theory)
    Binding,
    /// Topicality/salience
    Salience,
    /// Recency
    Recency,
}

impl UncertainReference {
    /// Create a new uncertain reference.
    #[must_use]
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            candidates: Vec::new(),
            resolved: false,
            resolved_entity: None,
            constraints: Vec::new(),
            discourse_position: None,
            is_cataphoric: false,
        }
    }

    /// Mark as cataphoric (forward-referencing).
    #[must_use]
    pub fn cataphoric(mut self) -> Self {
        self.is_cataphoric = true;
        self
    }

    /// Set discourse position.
    #[must_use]
    pub fn at_position(mut self, position: usize) -> Self {
        self.discourse_position = Some(position);
        self
    }

    /// Add a constraint.
    pub fn add_constraint(
        &mut self,
        kind: ConstraintKind,
        value: impl Into<String>,
        is_hard: bool,
    ) {
        self.constraints.push(ReferenceConstraint {
            kind,
            value: value.into(),
            is_hard,
        });
    }

    /// Add a candidate referent.
    pub fn add_candidate(&mut self, candidate: ReferenceCandidate) {
        // Check if candidate already exists
        if let Some(existing) = self
            .candidates
            .iter_mut()
            .find(|c| c.entity_id == candidate.entity_id)
        {
            // Update weight (combine evidence)
            existing.weight = log_sum_exp(existing.weight, candidate.weight);
        } else {
            self.candidates.push(candidate);
        }
    }

    /// Update evidence for a candidate.
    ///
    /// Positive evidence increases weight, negative decreases.
    pub fn update_evidence(&mut self, entity_id: u64, evidence: f64) {
        if let Some(candidate) = self
            .candidates
            .iter_mut()
            .find(|c| c.entity_id == entity_id)
        {
            candidate.weight += evidence;
        }
    }

    /// Prune candidates below a threshold.
    pub fn prune(&mut self, threshold: f64) {
        self.candidates.retain(|c| c.weight >= threshold);
    }

    /// Prune candidates with hard constraint violations.
    pub fn prune_violations(&mut self) {
        self.candidates.retain(|c| !c.has_violations());
    }

    /// Get candidates sorted by weight (highest first).
    #[must_use]
    pub fn ranked_candidates(&self) -> Vec<&ReferenceCandidate> {
        let mut sorted: Vec<_> = self.candidates.iter().collect();
        sorted.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Get the best candidate (highest weight).
    #[must_use]
    pub fn best_candidate(&self) -> Option<&ReferenceCandidate> {
        self.ranked_candidates().first().copied()
    }

    /// Compute entropy of the candidate distribution.
    ///
    /// Higher entropy = more uncertainty.
    #[must_use]
    pub fn entropy(&self) -> f64 {
        if self.candidates.is_empty() {
            return 0.0;
        }

        let probs = self.probabilities();
        let mut h = 0.0;
        for p in probs.values() {
            if *p > 0.0 {
                h -= p * p.log2();
            }
        }
        h
    }

    /// Convert weights to probabilities (softmax).
    #[must_use]
    pub fn probabilities(&self) -> HashMap<u64, f64> {
        if self.candidates.is_empty() {
            return HashMap::new();
        }

        let max_weight = self
            .candidates
            .iter()
            .map(|c| c.weight)
            .fold(f64::NEG_INFINITY, f64::max);

        let exp_sum: f64 = self
            .candidates
            .iter()
            .map(|c| (c.weight - max_weight).exp())
            .sum();

        self.candidates
            .iter()
            .map(|c| (c.entity_id, (c.weight - max_weight).exp() / exp_sum))
            .collect()
    }

    /// Check if resolution is ambiguous (multiple high-probability candidates).
    #[must_use]
    pub fn is_ambiguous(&self, threshold: f64) -> bool {
        let probs = self.probabilities();
        let high_prob_count = probs.values().filter(|&&p| p >= threshold).count();
        high_prob_count > 1
    }

    /// Force resolution to the best candidate.
    ///
    /// Returns the resolved candidate.
    #[must_use]
    pub fn resolve(&mut self) -> Option<ReferenceCandidate> {
        if self.resolved {
            return self
                .candidates
                .iter()
                .find(|c| Some(c.entity_id) == self.resolved_entity)
                .cloned();
        }

        let best = self.best_candidate()?.clone();
        self.resolved = true;
        self.resolved_entity = Some(best.entity_id);
        Some(best)
    }

    /// Resolve to a specific entity.
    pub fn resolve_to(&mut self, entity_id: u64) {
        self.resolved = true;
        self.resolved_entity = Some(entity_id);
    }

    /// Check if this reference is resolved.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        self.resolved
    }

    /// Get the number of candidates.
    #[must_use]
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    /// Check if there are no candidates (unresolvable).
    #[must_use]
    pub fn is_unresolvable(&self) -> bool {
        self.candidates.is_empty()
    }
}

// =============================================================================
// Deferred Resolution Context
// =============================================================================

/// Context for managing multiple uncertain references across a discourse.
///
/// This is the "virtual machine" state for deferred coreference resolution,
/// tracking pending references and resolving them as context becomes available.
///
/// # Example
///
/// ```rust
/// use anno::discourse::uncertain_reference::{
///     DeferredResolutionContext, UncertainReference, ReferenceCandidate,
/// };
///
/// let mut context = DeferredResolutionContext::new();
///
/// // Encounter a cataphoric pronoun "When she arrived..."
/// let cataphoric = UncertainReference::new("she").cataphoric().at_position(5);
/// context.add_uncertain(cataphoric);
///
/// // Process more text...
/// context.advance();
///
/// // Later, encounter "Mary walked in"
/// context.try_resolve_cataphoric(&[(1, "Mary".to_string(), 0.9)]);
///
/// // At discourse end, resolve all pending
/// context.resolve_all();
///
/// // Check statistics
/// let stats = context.statistics();
/// assert_eq!(stats.resolved, 1);
/// ```
///
/// # Workflow
///
/// 1. Create context at discourse start
/// 2. Add uncertain references as they're encountered
/// 3. Call [`advance`](Self::advance) to track position
/// 4. When new entities appear, call [`try_resolve_cataphoric`](Self::try_resolve_cataphoric)
/// 5. At discourse end, call [`resolve_all`](Self::resolve_all)
/// 6. Inspect [`statistics`](Self::statistics) for analysis
#[derive(Debug, Clone, Default)]
pub struct DeferredResolutionContext {
    /// Uncertain references not yet resolved.
    pub pending: Vec<UncertainReference>,
    /// References that have been resolved (for tracking and analysis).
    pub resolved: Vec<UncertainReference>,
    /// Entity mentions seen so far: entity_id -> list of positions.
    pub entity_mentions: HashMap<u64, Vec<usize>>,
    /// Current discourse position (incremented via [`advance`](Self::advance)).
    pub position: usize,
}

impl DeferredResolutionContext {
    /// Create a new context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new uncertain reference.
    pub fn add_uncertain(&mut self, reference: UncertainReference) {
        self.pending.push(reference);
    }

    /// Record an entity mention at the current position.
    pub fn record_mention(&mut self, entity_id: u64) {
        self.entity_mentions
            .entry(entity_id)
            .or_default()
            .push(self.position);
    }

    /// Advance discourse position.
    pub fn advance(&mut self) {
        self.position += 1;
    }

    /// Try to resolve pending cataphoric references.
    ///
    /// Called when new entities become available.
    pub fn try_resolve_cataphoric(&mut self, new_entities: &[(u64, String, f64)]) {
        for reference in &mut self.pending {
            if reference.is_cataphoric && !reference.is_resolved() {
                for (entity_id, description, weight) in new_entities {
                    reference.add_candidate(
                        ReferenceCandidate::new(*entity_id, description.clone(), *weight)
                            .with_source(CandidateSource::Cataphoric),
                    );
                }
            }
        }
    }

    /// Force resolution of all pending references.
    ///
    /// Used at discourse end or when resolution is required.
    pub fn resolve_all(&mut self) {
        for reference in &mut self.pending {
            let _ = reference.resolve();
        }
        self.resolved.append(&mut self.pending);
    }

    /// Get pending references that are ambiguous.
    #[must_use]
    pub fn ambiguous_references(&self, threshold: f64) -> Vec<&UncertainReference> {
        self.pending
            .iter()
            .filter(|r| r.is_ambiguous(threshold))
            .collect()
    }

    /// Get statistics about resolution.
    #[must_use]
    pub fn statistics(&self) -> ResolutionStatistics {
        let total = self.pending.len() + self.resolved.len();
        let resolved_count = self.resolved.len();
        let ambiguous_count = self.pending.iter().filter(|r| r.is_ambiguous(0.3)).count();
        let unresolvable_count = self.pending.iter().filter(|r| r.is_unresolvable()).count();
        let cataphoric_count = self.pending.iter().filter(|r| r.is_cataphoric).count();

        let avg_entropy = if self.pending.is_empty() {
            0.0
        } else {
            self.pending.iter().map(|r| r.entropy()).sum::<f64>() / self.pending.len() as f64
        };

        ResolutionStatistics {
            total,
            resolved: resolved_count,
            pending: self.pending.len(),
            ambiguous: ambiguous_count,
            unresolvable: unresolvable_count,
            cataphoric: cataphoric_count,
            avg_entropy,
        }
    }
}

/// Statistics about reference resolution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResolutionStatistics {
    /// Total references encountered
    pub total: usize,
    /// Successfully resolved
    pub resolved: usize,
    /// Still pending
    pub pending: usize,
    /// Ambiguous (multiple high-prob candidates)
    pub ambiguous: usize,
    /// Unresolvable (no candidates)
    pub unresolvable: usize,
    /// Cataphoric (forward-referencing)
    pub cataphoric: usize,
    /// Average entropy of pending references
    pub avg_entropy: f64,
}

// =============================================================================
// Resolution Strategies
// =============================================================================

/// Strategy for when to resolve uncertain references.
///
/// Different strategies trade off between:
/// - **Accuracy**: Waiting for more context improves accuracy
/// - **Latency**: Immediate resolution enables faster processing
/// - **Memory**: Maintaining distributions uses more memory
///
/// # Example
///
/// ```rust
/// use anno::discourse::uncertain_reference::{
///     UncertainReference, ReferenceCandidate, ResolutionStrategy, resolve_uncertain,
/// };
///
/// let mut reference = UncertainReference::new("he");
/// reference.add_candidate(ReferenceCandidate::new(1, "John", 2.0));
/// reference.add_candidate(ReferenceCandidate::new(2, "Bill", 0.5));
///
/// // Greedy: resolve immediately
/// let greedy = ResolutionStrategy::Greedy;
/// assert!(greedy.should_resolve(&reference));
///
/// // Confident: only resolve if top candidate has >90% probability
/// let confident = ResolutionStrategy::Confident(90);
/// // With weights 2.0 and 0.5, softmax gives ~82% to John, so won't resolve
/// // (actual behavior depends on softmax computation)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResolutionStrategy {
    /// Resolve immediately to best candidate.
    /// Fast but may be inaccurate if context would help.
    #[default]
    Greedy,
    /// Never resolve automatically; wait until forced.
    /// Use when downstream processing can handle uncertainty.
    Deferred,
    /// Maintain full distribution; don't commit to single candidate.
    /// Useful for probabilistic downstream processing.
    Probabilistic,
    /// Resolve only when top candidate probability exceeds threshold.
    /// The threshold is a percentage (0-100).
    /// E.g., `Confident(90)` resolves only if P(best) >= 90%.
    Confident(u8),
}

impl ResolutionStrategy {
    /// Should resolve now given the reference state?
    #[must_use]
    pub fn should_resolve(&self, reference: &UncertainReference) -> bool {
        match self {
            ResolutionStrategy::Greedy => true,
            ResolutionStrategy::Deferred => false,
            ResolutionStrategy::Probabilistic => false,
            ResolutionStrategy::Confident(threshold) => {
                let probs = reference.probabilities();
                probs.values().any(|&p| p * 100.0 >= *threshold as f64)
            }
        }
    }
}

/// Resolve an uncertain reference using a given strategy.
///
/// This is the main entry point for applying resolution strategies.
/// It respects the strategy's rules about when to resolve.
///
/// # Returns
///
/// - `Some(candidate)` if resolution occurred (or was already resolved)
/// - `None` if the strategy says not to resolve yet, or no candidates exist
///
/// # Example
///
/// ```rust
/// use anno::discourse::uncertain_reference::{
///     UncertainReference, ReferenceCandidate, ResolutionStrategy, resolve_uncertain,
/// };
///
/// let mut reference = UncertainReference::new("the issue");
/// reference.add_candidate(ReferenceCandidate::new(1, "bug #42", 0.9));
///
/// // With Greedy strategy, resolves immediately
/// let result = resolve_uncertain(&mut reference, ResolutionStrategy::Greedy);
/// assert!(result.is_some());
/// assert!(reference.is_resolved());
///
/// // Calling again returns the same result
/// let result2 = resolve_uncertain(&mut reference, ResolutionStrategy::Greedy);
/// assert_eq!(result.map(|r| r.entity_id), result2.map(|r| r.entity_id));
/// ```
pub fn resolve_uncertain(
    reference: &mut UncertainReference,
    strategy: ResolutionStrategy,
) -> Option<ReferenceCandidate> {
    if reference.is_resolved() {
        return reference
            .candidates
            .iter()
            .find(|c| Some(c.entity_id) == reference.resolved_entity)
            .cloned();
    }

    if strategy.should_resolve(reference) {
        reference.resolve()
    } else {
        None
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Log-sum-exp for numerically stable probability combination.
fn log_sum_exp(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertain_reference_basic() {
        let mut reference = UncertainReference::new("the person");

        reference.add_candidate(ReferenceCandidate::new(1, "John", 0.8));
        reference.add_candidate(ReferenceCandidate::new(2, "Mary", 0.6));

        assert_eq!(reference.candidate_count(), 2);
        assert!(!reference.is_resolved());

        let best = reference.best_candidate().unwrap();
        assert_eq!(best.entity_id, 1);
    }

    #[test]
    fn test_evidence_update() {
        let mut reference = UncertainReference::new("the person");

        reference.add_candidate(ReferenceCandidate::new(1, "John", 0.5));
        reference.add_candidate(ReferenceCandidate::new(2, "Mary", 0.5));

        // Initially equal
        assert_eq!(
            reference.candidates[0].weight,
            reference.candidates[1].weight
        );

        // Update evidence
        reference.update_evidence(1, 0.3);
        reference.update_evidence(2, -0.2);

        // Now John should be preferred
        let best = reference.best_candidate().unwrap();
        assert_eq!(best.entity_id, 1);
    }

    #[test]
    fn test_probabilities() {
        let mut reference = UncertainReference::new("test");

        reference.add_candidate(ReferenceCandidate::new(1, "A", 1.0));
        reference.add_candidate(ReferenceCandidate::new(2, "B", 0.0));

        let probs = reference.probabilities();

        // Softmax should give higher probability to higher weight
        assert!(probs[&1] > probs[&2]);

        // Probabilities should sum to 1
        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_entropy() {
        // Equal weights = max entropy for 2 candidates
        let mut equal_ref = UncertainReference::new("test");
        equal_ref.add_candidate(ReferenceCandidate::new(1, "A", 0.0));
        equal_ref.add_candidate(ReferenceCandidate::new(2, "B", 0.0));

        // Very unequal weights = low entropy
        let mut unequal_ref = UncertainReference::new("test");
        unequal_ref.add_candidate(ReferenceCandidate::new(1, "A", 10.0));
        unequal_ref.add_candidate(ReferenceCandidate::new(2, "B", 0.0));

        assert!(equal_ref.entropy() > unequal_ref.entropy());
    }

    #[test]
    fn test_ambiguity_detection() {
        let mut reference = UncertainReference::new("test");

        // Very different weights - not ambiguous
        reference.add_candidate(ReferenceCandidate::new(1, "A", 10.0));
        reference.add_candidate(ReferenceCandidate::new(2, "B", 0.0));

        assert!(!reference.is_ambiguous(0.3));

        // Similar weights - ambiguous
        let mut ambiguous_ref = UncertainReference::new("test");
        ambiguous_ref.add_candidate(ReferenceCandidate::new(1, "A", 0.0));
        ambiguous_ref.add_candidate(ReferenceCandidate::new(2, "B", 0.0));

        assert!(ambiguous_ref.is_ambiguous(0.3));
    }

    #[test]
    fn test_resolution() {
        let mut reference = UncertainReference::new("test");
        reference.add_candidate(ReferenceCandidate::new(1, "John", 0.8));
        reference.add_candidate(ReferenceCandidate::new(2, "Mary", 0.6));

        assert!(!reference.is_resolved());

        let resolved = reference.resolve().unwrap();
        assert_eq!(resolved.entity_id, 1);
        assert!(reference.is_resolved());
        assert_eq!(reference.resolved_entity, Some(1));
    }

    #[test]
    fn test_constraint_violations() {
        let candidate = ReferenceCandidate::new(1, "John", 0.8).violates("gender:feminine");

        assert!(candidate.has_violations());

        let mut reference = UncertainReference::new("she");
        reference.add_candidate(candidate);
        reference.add_candidate(ReferenceCandidate::new(2, "Mary", 0.6));

        reference.prune_violations();
        assert_eq!(reference.candidate_count(), 1);
        assert_eq!(reference.candidates[0].entity_id, 2);
    }

    #[test]
    fn test_cataphoric_resolution() {
        let mut context = DeferredResolutionContext::new();

        // Add cataphoric reference: "When she arrived..."
        let cataphoric = UncertainReference::new("she").cataphoric().at_position(0);
        context.add_uncertain(cataphoric);

        // Later, we see "...Mary walked in"
        context.advance();
        context.try_resolve_cataphoric(&[(1, "Mary".to_string(), 0.9)]);

        // Should now have a candidate
        assert_eq!(context.pending[0].candidate_count(), 1);
    }

    #[test]
    fn test_resolution_strategy() {
        let mut reference = UncertainReference::new("test");
        reference.add_candidate(ReferenceCandidate::new(1, "A", 5.0));
        reference.add_candidate(ReferenceCandidate::new(2, "B", 0.0));

        // Greedy should resolve immediately
        assert!(ResolutionStrategy::Greedy.should_resolve(&reference));

        // Deferred should not resolve
        assert!(!ResolutionStrategy::Deferred.should_resolve(&reference));

        // Confident(90) should resolve if probability > 90%
        // With weights 5.0 and 0.0, softmax gives ~99% to A
        assert!(ResolutionStrategy::Confident(90).should_resolve(&reference));
    }

    #[test]
    fn test_context_statistics() {
        let mut context = DeferredResolutionContext::new();

        // Add various references
        let mut resolved = UncertainReference::new("resolved");
        resolved.add_candidate(ReferenceCandidate::new(1, "A", 0.9));
        let _ = resolved.resolve();
        context.resolved.push(resolved);

        let mut ambiguous = UncertainReference::new("ambiguous");
        ambiguous.add_candidate(ReferenceCandidate::new(2, "B", 0.0));
        ambiguous.add_candidate(ReferenceCandidate::new(3, "C", 0.0));
        context.pending.push(ambiguous);

        let cataphoric = UncertainReference::new("cataphoric").cataphoric();
        context.pending.push(cataphoric);

        let stats = context.statistics();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.resolved, 1);
        assert_eq!(stats.pending, 2);
        assert_eq!(stats.ambiguous, 1);
        assert_eq!(stats.cataphoric, 1);
    }
}
