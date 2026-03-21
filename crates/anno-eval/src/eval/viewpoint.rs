//! Viewpoint-aware entity extraction.
//!
//! # The Attribution Problem
//!
//! Traditional NER extracts entities as objective facts, but information is often:
//! - **Attributed**: "According to Reuters, the CEO resigned"
//! - **Negated**: "The company did not announce layoffs"
//! - **Speculative**: "The merger may happen next quarter"
//! - **Quoted**: "He said 'I will run for president'"
//!
//! This module tracks WHO says WHAT about entities, enabling:
//! - Fact-checking and claim verification
//! - Source credibility assessment
//! - Perspective-aware summarization
//!
//! # Viewpoint Model
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────┐
//! │                        INFORMATION SPACE                                  │
//! │                                                                          │
//! │   ┌─────────┐    asserts    ┌─────────────┐    about    ┌─────────┐     │
//! │   │ SOURCE  │ ───────────► │ PROPOSITION │ ──────────► │ ENTITY  │     │
//! │   └─────────┘              └─────────────┘              └─────────┘     │
//! │       │                          │                                       │
//! │       │                          │                                       │
//! │       ▼                          ▼                                       │
//! │   credibility              epistemic status                              │
//! │   - author type            - factual/asserted                            │
//! │   - publication            - negated                                     │
//! │   - confidence             - hypothetical                                │
//! │                            - reported                                    │
//! │                            - opinion                                     │
//! └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust
//! use anno::eval::viewpoint::{ViewpointExtractor, Attribution, EpistemicStatus};
//!
//! let extractor = ViewpointExtractor::default();
//!
//! let text = "According to John Smith, Apple will acquire Microsoft next year.";
//! let attributed = extractor.extract(text);
//!
//! for attr in &attributed {
//!     println!("Source: {:?}", attr.source);
//!     println!("Status: {:?}", attr.epistemic_status);
//!     println!("Entity: {:?}", attr.entity);
//! }
//! ```
//!
//! # Research Background
//!
//! - Saurí & Pustejovsky (2009): "FactBank: A Corpus Annotated with Event Factuality"
//! - Minard et al. (2016): "MEANTIME: NewsReader Multilingual Event and Time Corpus"
//! - Park et al. (2015): "Epistemic Stance in NLP"

use anno::offset::TextSpan;
use anno::Entity;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Core Types
// =============================================================================

/// Epistemic status of a proposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EpistemicStatus {
    /// Asserted as fact by the author
    #[default]
    Factual,
    /// Explicitly negated
    Negated,
    /// Hypothetical/conditional
    Hypothetical,
    /// Reported speech (attributed to source)
    Reported,
    /// Opinion/belief
    Opinion,
    /// Uncertain/hedged
    Uncertain,
    /// Future/planned
    Future,
    /// Question/interrogative
    Interrogative,
}

impl EpistemicStatus {
    /// Check if this status indicates the proposition is likely true.
    #[must_use]
    pub fn is_positive(&self) -> bool {
        matches!(self, Self::Factual | Self::Reported)
    }

    /// Check if this status indicates uncertainty.
    #[must_use]
    pub fn is_uncertain(&self) -> bool {
        matches!(
            self,
            Self::Hypothetical | Self::Uncertain | Self::Future | Self::Opinion
        )
    }
}

/// Source of attributed information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// Source text (e.g., "John Smith", "Reuters", "the company")
    pub text: String,
    /// Source type
    pub source_type: SourceType,
    /// Character offset in original text
    pub span: Option<(usize, usize)>,
    /// Linked entity (if source is an entity)
    /// NOTE: Not included in Hash/Eq - Entity doesn't implement these traits
    pub entity: Option<Entity>,
}

impl PartialEq for Source {
    fn eq(&self, other: &Self) -> bool {
        self.text == other.text && self.source_type == other.source_type && self.span == other.span
    }
}

impl Eq for Source {}

impl std::hash::Hash for Source {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.text.hash(state);
        self.source_type.hash(state);
        self.span.hash(state);
    }
}

impl Source {
    /// Create a new source.
    pub fn new(text: &str, source_type: SourceType) -> Self {
        Self {
            text: text.to_string(),
            source_type,
            span: None,
            entity: None,
        }
    }

    /// The implicit author/narrator.
    pub fn author() -> Self {
        Self {
            text: "AUTHOR".to_string(),
            source_type: SourceType::Author,
            span: None,
            entity: None,
        }
    }
}

/// Type of information source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceType {
    /// The document author/narrator
    Author,
    /// Named person
    Person,
    /// Organization/institution
    Organization,
    /// News outlet/publication
    Publication,
    /// Generic/anonymous source
    Anonymous,
    /// Document/report
    Document,
}

/// An entity mention with viewpoint attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attribution {
    /// The entity being discussed
    pub entity: Entity,
    /// Who made this claim
    pub source: Source,
    /// Epistemic status of the claim
    pub epistemic_status: EpistemicStatus,
    /// Confidence in attribution detection
    pub confidence: f64,
    /// The proposition/claim text
    pub claim_text: Option<String>,
    /// Nesting level (for recursive attribution)
    pub nesting_level: u8,
}

impl Attribution {
    /// Create a new attribution.
    pub fn new(entity: Entity, source: Source, status: EpistemicStatus) -> Self {
        Self {
            entity,
            source,
            epistemic_status: status,
            confidence: 1.0,
            claim_text: None,
            nesting_level: 0,
        }
    }

    /// Check if this attribution is from the author.
    #[must_use]
    pub fn is_direct(&self) -> bool {
        self.source.source_type == SourceType::Author
    }

    /// Check if this is reported speech.
    #[must_use]
    pub fn is_reported(&self) -> bool {
        self.epistemic_status == EpistemicStatus::Reported
    }
}

// =============================================================================
// Viewpoint Extractor
// =============================================================================

/// Extracts viewpoint/attribution information from text.
#[derive(Debug, Clone)]
pub struct ViewpointExtractor {
    /// Reporting verbs that indicate attribution
    reporting_verbs: Vec<String>,
    /// Negation markers
    negation_markers: Vec<String>,
    /// Hedging markers
    hedging_markers: Vec<String>,
    /// Future markers
    future_markers: Vec<String>,
    /// Opinion markers
    opinion_markers: Vec<String>,
}

impl Default for ViewpointExtractor {
    fn default() -> Self {
        Self {
            reporting_verbs: vec![
                "said",
                "says",
                "told",
                "stated",
                "claimed",
                "reported",
                "announced",
                "declared",
                "confirmed",
                "denied",
                "argued",
                "suggested",
                "explained",
                "noted",
                "added",
                "according to",
                "tweeted",
                "posted",
                "wrote",
                "mentioned",
                "revealed",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            negation_markers: vec![
                "not",
                "no",
                "never",
                "neither",
                "nor",
                "none",
                "nobody",
                "nothing",
                "nowhere",
                "didn't",
                "doesn't",
                "don't",
                "won't",
                "wasn't",
                "weren't",
                "isn't",
                "aren't",
                "haven't",
                "hasn't",
                "cannot",
                "can't",
                "couldn't",
                "wouldn't",
                "shouldn't",
                "failed to",
                "refused to",
                "denied",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            hedging_markers: vec![
                "may",
                "might",
                "could",
                "possibly",
                "perhaps",
                "probably",
                "likely",
                "unlikely",
                "appears",
                "seems",
                "suggests",
                "reportedly",
                "allegedly",
                "rumored",
                "believed to",
                "thought to",
                "expected to",
                "estimated",
                "approximately",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            future_markers: vec![
                "will",
                "shall",
                "going to",
                "plans to",
                "intends to",
                "expected to",
                "scheduled to",
                "set to",
                "about to",
                "next year",
                "next month",
                "in the future",
                "upcoming",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            opinion_markers: vec![
                "think",
                "believe",
                "feel",
                "consider",
                "view",
                "regard",
                "opinion",
                "perspective",
                "in my view",
                "personally",
                "should",
                "ought to",
                "must",
                "best",
                "worst",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        }
    }
}

impl ViewpointExtractor {
    /// Extract viewpoint attributions for entities.
    pub fn extract(&self, text: &str) -> Vec<Attribution> {
        let mut attributions = Vec::new();
        let lower = text.to_lowercase();

        // First, extract entities (simplified - would use actual NER)
        let entities = self.simple_entity_extraction(text);

        // For each entity, determine its attribution context
        for entity in entities {
            let context = self.get_context(text, entity.start(), 100);
            let (source, status) = self.analyze_context(&context, &lower, entity.start());

            let mut attr = Attribution::new(entity, source, status);
            attr.claim_text = Some(context);
            attr.confidence = self.compute_confidence(&attr);

            attributions.push(attr);
        }

        attributions
    }

    /// Extract entities with their viewpoint context.
    pub fn extract_with_entities(&self, text: &str, entities: &[Entity]) -> Vec<Attribution> {
        let lower = text.to_lowercase();

        entities
            .iter()
            .map(|entity| {
                let context = self.get_context(text, entity.start(), 100);
                let (source, status) = self.analyze_context(&context, &lower, entity.start());

                let mut attr = Attribution::new(entity.clone(), source, status);
                attr.claim_text = Some(context);
                attr.confidence = self.compute_confidence(&attr);

                attr
            })
            .collect()
    }

    /// Get context around a position.
    fn get_context(&self, text: &str, pos: usize, window: usize) -> String {
        let chars: Vec<char> = text.chars().collect();
        let start = pos.saturating_sub(window);
        let end = (pos + window).min(chars.len());
        chars[start..end].iter().collect()
    }

    /// Analyze context to determine source and epistemic status.
    fn analyze_context(
        &self,
        context: &str,
        _lower_text: &str,
        _pos: usize,
    ) -> (Source, EpistemicStatus) {
        let context_lower = context.to_lowercase();

        // Check for reporting patterns first
        if let Some(source) = self.find_attribution_source(context) {
            return (source, EpistemicStatus::Reported);
        }

        // Check for negation
        if self.has_negation(&context_lower) {
            return (Source::author(), EpistemicStatus::Negated);
        }

        // Check for hedging
        if self.has_hedging(&context_lower) {
            return (Source::author(), EpistemicStatus::Uncertain);
        }

        // Check for future
        if self.has_future(&context_lower) {
            return (Source::author(), EpistemicStatus::Future);
        }

        // Check for opinion
        if self.has_opinion(&context_lower) {
            return (Source::author(), EpistemicStatus::Opinion);
        }

        // Default: factual assertion by author
        (Source::author(), EpistemicStatus::Factual)
    }

    /// Find attribution source in context.
    fn find_attribution_source(&self, context: &str) -> Option<Source> {
        let context_lower = context.to_lowercase();

        // Pattern: "According to X"
        if let Some(caps) =
            Regex::new(r"(?i)according to ([A-Z][a-z]+ [A-Z][a-z]+|[A-Z][a-z]+|the [a-z]+)")
                .ok()
                .and_then(|re| re.captures(context))
        {
            let source_text = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            return Some(Source::new(source_text, SourceType::Person));
        }

        // Pattern: "X said/told/claimed"
        for verb in &self.reporting_verbs {
            let patterns = [
                format!(r"([A-Z][a-z]+ [A-Z][a-z]+|[A-Z][a-z]+) {}", verb),
                format!(r"([A-Z][a-z]+ [A-Z][a-z]+|[A-Z][a-z]+),? who {},", verb),
            ];

            for pattern in &patterns {
                if let Ok(re) = Regex::new(pattern) {
                    if let Some(caps) = re.captures(context) {
                        let source_text = caps.get(1).map(|m| m.as_str()).unwrap_or("");
                        return Some(Source::new(source_text, SourceType::Person));
                    }
                }
            }
        }

        // Check for reporting verb presence (anonymous source)
        for verb in &self.reporting_verbs {
            if context_lower.contains(verb) {
                return Some(Source::new("sources", SourceType::Anonymous));
            }
        }

        None
    }

    /// Check for negation markers.
    fn has_negation(&self, context: &str) -> bool {
        self.negation_markers.iter().any(|m| context.contains(m))
    }

    /// Check for hedging markers.
    fn has_hedging(&self, context: &str) -> bool {
        self.hedging_markers.iter().any(|m| context.contains(m))
    }

    /// Check for future markers.
    fn has_future(&self, context: &str) -> bool {
        self.future_markers.iter().any(|m| context.contains(m))
    }

    /// Check for opinion markers.
    fn has_opinion(&self, context: &str) -> bool {
        self.opinion_markers.iter().any(|m| context.contains(m))
    }

    /// Compute confidence in attribution.
    fn compute_confidence(&self, attr: &Attribution) -> f64 {
        let mut conf: f64 = 0.5; // Base confidence

        // Higher confidence for explicit attribution
        if attr.source.source_type != SourceType::Author {
            conf += 0.3;
        }

        // Higher confidence for clear epistemic markers
        if attr.epistemic_status != EpistemicStatus::Factual {
            conf += 0.1;
        }

        conf.min(1.0)
    }

    /// Lightweight regex entity extraction for viewpoint analysis (not a full NER backend).
    fn simple_entity_extraction(&self, text: &str) -> Vec<Entity> {
        use anno::EntityType;

        let mut entities = Vec::new();

        // Very simple pattern: Capitalized words (excluding sentence starts)
        let re = Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
            .expect("regex pattern should be valid");

        for caps in re.captures_iter(text) {
            if let Some(m) = caps.get(1) {
                // Skip common non-entities
                let text_match = m.as_str();
                if [
                    "The",
                    "This",
                    "That",
                    "These",
                    "Those",
                    "According",
                    "However",
                ]
                .contains(&text_match)
                {
                    continue;
                }

                let span = TextSpan::from_bytes(text, m.start(), m.end());
                entities.push(Entity::new(
                    text_match,
                    EntityType::Person, // Default to person
                    span.char_start,
                    span.char_end,
                    0.5,
                ));
            }
        }

        entities
    }
}

#[cfg(test)]
mod simple_extraction_tests {
    use super::*;
    use anno::offset::TextSpan;

    #[test]
    fn test_simple_entity_extraction_offsets_are_character_offsets_on_unicode_prefix() {
        let extractor = ViewpointExtractor::default();
        let text = "Müller met John Smith in Paris.";
        let ents = extractor.simple_entity_extraction(text);
        assert!(!ents.is_empty(), "expected at least one extracted entity");

        for e in ents {
            let extracted = TextSpan::from_chars(text, e.start(), e.end()).extract(text);
            assert_eq!(extracted, e.text);
        }
    }
}

// =============================================================================
// Viewpoint Aggregation
// =============================================================================

/// Aggregates viewpoints about an entity across multiple sources.
#[derive(Debug, Clone, Default)]
pub struct ViewpointAggregator {
    /// Minimum agreement for consensus
    pub consensus_threshold: f64,
}

impl ViewpointAggregator {
    /// Create a new aggregator.
    pub fn new(consensus_threshold: f64) -> Self {
        Self {
            consensus_threshold,
        }
    }

    /// Aggregate attributions for the same entity.
    pub fn aggregate(&self, attributions: &[Attribution]) -> ViewpointSummary {
        if attributions.is_empty() {
            return ViewpointSummary::default();
        }

        // Group by source
        let mut by_source: HashMap<String, Vec<&Attribution>> = HashMap::new();
        for attr in attributions {
            by_source
                .entry(attr.source.text.clone())
                .or_default()
                .push(attr);
        }

        // Count epistemic statuses
        let mut status_counts: HashMap<EpistemicStatus, usize> = HashMap::new();
        for attr in attributions {
            *status_counts.entry(attr.epistemic_status).or_insert(0) += 1;
        }

        // Find dominant status
        let dominant_status = status_counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(s, _)| *s)
            .unwrap_or(EpistemicStatus::Factual);

        // Check for conflicting viewpoints
        let has_conflict =
            status_counts.len() > 1 && status_counts.values().filter(|&&c| c > 0).count() > 1;

        // Compute consensus
        let total = attributions.len() as f64;
        let dominant_count = status_counts.get(&dominant_status).copied().unwrap_or(0) as f64;
        let consensus = dominant_count / total;

        ViewpointSummary {
            entity_text: attributions
                .first()
                .map(|a| a.entity.text.clone())
                .unwrap_or_default(),
            sources: by_source.keys().cloned().collect(),
            dominant_status,
            status_distribution: status_counts,
            consensus,
            has_conflict,
            n_mentions: attributions.len(),
        }
    }
}

/// Summary of viewpoints about an entity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ViewpointSummary {
    /// Entity text
    pub entity_text: String,
    /// All sources that mentioned this entity
    pub sources: Vec<String>,
    /// Most common epistemic status
    pub dominant_status: EpistemicStatus,
    /// Distribution of epistemic statuses
    pub status_distribution: HashMap<EpistemicStatus, usize>,
    /// Consensus level (0-1)
    pub consensus: f64,
    /// Whether sources conflict
    pub has_conflict: bool,
    /// Number of mentions
    pub n_mentions: usize,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epistemic_status() {
        assert!(EpistemicStatus::Factual.is_positive());
        assert!(!EpistemicStatus::Negated.is_uncertain());
        assert!(EpistemicStatus::Hypothetical.is_uncertain());
    }

    #[test]
    fn test_attribution_source_detection() {
        let extractor = ViewpointExtractor::default();

        let text = "According to John Smith, Apple will acquire Microsoft.";
        let attrs = extractor.extract(text);

        // Should detect John Smith as source
        let has_john_smith = attrs.iter().any(|a| a.source.text.contains("John Smith"));
        assert!(
            has_john_smith
                || attrs
                    .iter()
                    .any(|a| a.epistemic_status == EpistemicStatus::Reported)
        );
    }

    #[test]
    fn test_negation_detection() {
        let extractor = ViewpointExtractor::default();

        let text = "Apple did not announce any layoffs.";
        let attrs = extractor.extract(text);

        // Should detect negation
        let has_negated = attrs
            .iter()
            .any(|a| a.epistemic_status == EpistemicStatus::Negated);
        assert!(has_negated);
    }

    #[test]
    fn test_hedging_detection() {
        let extractor = ViewpointExtractor::default();

        let text = "Microsoft may acquire Activision next year.";
        let attrs = extractor.extract(text);

        // Should detect uncertainty or future
        let has_uncertain = attrs.iter().any(|a| {
            a.epistemic_status == EpistemicStatus::Uncertain
                || a.epistemic_status == EpistemicStatus::Future
        });
        assert!(has_uncertain);
    }

    #[test]
    fn test_viewpoint_aggregation() {
        use anno::EntityType;

        let entity = Entity::new("Apple", EntityType::Organization, 0, 5, 0.9);

        let attributions = vec![
            Attribution::new(entity.clone(), Source::author(), EpistemicStatus::Factual),
            Attribution::new(
                entity.clone(),
                Source::new("Reuters", SourceType::Publication),
                EpistemicStatus::Reported,
            ),
            Attribution::new(
                entity.clone(),
                Source::new("Bloomberg", SourceType::Publication),
                EpistemicStatus::Reported,
            ),
        ];

        let aggregator = ViewpointAggregator::new(0.5);
        let summary = aggregator.aggregate(&attributions);

        assert_eq!(summary.n_mentions, 3);
        assert_eq!(summary.sources.len(), 3);
        assert!(summary.consensus > 0.0);
    }

    #[test]
    fn test_conflict_detection() {
        use anno::{EntityCategory, EntityType};

        let entity = Entity::new(
            "Merger",
            EntityType::custom("EVENT", EntityCategory::Misc),
            0,
            6,
            0.9,
        );

        let attributions = vec![
            Attribution::new(entity.clone(), Source::author(), EpistemicStatus::Factual),
            Attribution::new(
                entity.clone(),
                Source::new("Analyst", SourceType::Person),
                EpistemicStatus::Negated,
            ),
        ];

        let aggregator = ViewpointAggregator::new(0.5);
        let summary = aggregator.aggregate(&attributions);

        assert!(summary.has_conflict);
    }
}
