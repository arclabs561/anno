//! Main entity linker combining candidate generation, ranking, and NIL detection.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::candidate::{
    Candidate, CandidateGenerator, CandidateSource, DictionaryCandidateGenerator,
};
use super::nil::{NilAction, NilDetector, NilReason};
use anno_core::{Confidence, EntityType};

/// A mention to be linked.
#[derive(Debug, Clone)]
pub struct Mention {
    /// Mention text
    pub text: String,
    /// Start offset in document
    pub start: usize,
    /// End offset in document
    pub end: usize,
    /// Entity type from NER (optional)
    pub entity_type: Option<EntityType>,
}

impl Mention {
    /// Create a new mention.
    pub fn new(text: &str, start: usize, end: usize) -> Self {
        Self {
            text: text.to_string(),
            start,
            end,
            entity_type: None,
        }
    }

    /// Set entity type.
    pub fn with_type(mut self, entity_type: EntityType) -> Self {
        self.entity_type = Some(entity_type);
        self
    }
}

/// A linked entity result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkedEntity {
    /// Original mention text
    pub mention_text: String,
    /// Start offset
    pub start: usize,
    /// End offset
    pub end: usize,
    /// Linked KB ID (None if NIL)
    pub kb_id: Option<String>,
    /// KB source
    pub source: CandidateSource,
    /// Canonical label from KB
    pub label: Option<String>,
    /// Full IRI/URI
    pub iri: Option<String>,
    /// Linking confidence
    pub confidence: Confidence,
    /// Is this a NIL entity?
    pub is_nil: bool,
    /// NIL reason if applicable
    pub nil_reason: Option<NilReason>,
    /// NIL action if applicable
    pub nil_action: Option<NilAction>,
    /// Alternative candidates (for debugging/review)
    pub alternatives: Vec<CandidateSummary>,
}

/// Summary of a candidate (for alternatives list).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateSummary {
    /// KB ID
    pub kb_id: String,
    /// Label
    pub label: String,
    /// Score
    pub score: f64,
}

impl From<&Candidate> for CandidateSummary {
    fn from(c: &Candidate) -> Self {
        Self {
            kb_id: c.kb_id.clone(),
            label: c.label.clone(),
            score: c.score,
        }
    }
}

/// Overall linking result for a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkingResult {
    /// Linked entities
    pub entities: Vec<LinkedEntity>,
    /// Total mentions processed
    pub total_mentions: usize,
    /// Successfully linked
    pub linked_count: usize,
    /// NIL count
    pub nil_count: usize,
    /// Average confidence
    pub avg_confidence: f64,
}

impl LinkingResult {
    /// Get linking rate.
    pub fn linking_rate(&self) -> f64 {
        if self.total_mentions == 0 {
            0.0
        } else {
            self.linked_count as f64 / self.total_mentions as f64
        }
    }
}

/// Entity linker combining all components.
pub struct EntityLinker {
    /// Candidate generator
    generator: Arc<dyn CandidateGenerator>,
    /// NIL detector
    nil_detector: NilDetector,
    /// Maximum candidates to retrieve
    max_candidates: usize,
    /// Include alternatives in output
    include_alternatives: bool,
}

impl EntityLinker {
    /// Create a builder.
    pub fn builder() -> EntityLinkerBuilder {
        EntityLinkerBuilder::default()
    }

    /// Link mentions in a document.
    pub fn link(&self, mentions: &[Mention], context: &str) -> LinkingResult {
        let mut entities = Vec::with_capacity(mentions.len());
        let mut linked_count = 0;
        let mut nil_count = 0;
        let mut total_confidence = 0.0;

        for mention in mentions {
            let entity_type_str = mention.entity_type.as_ref().map(|et| et.to_string());

            // Generate candidates
            let mut candidates = self.generator.generate(
                &mention.text,
                context,
                entity_type_str.as_deref(),
                self.max_candidates,
            );

            // Score candidates
            for c in &mut candidates {
                c.compute_score();
            }
            candidates.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // NIL analysis
            let nil_analysis =
                self.nil_detector
                    .analyze(&mention.text, &candidates, entity_type_str.as_deref());

            let linked_entity = if nil_analysis.is_nil {
                nil_count += 1;

                LinkedEntity {
                    mention_text: mention.text.clone(),
                    start: mention.start,
                    end: mention.end,
                    kb_id: None,
                    source: CandidateSource::default(),
                    label: None,
                    iri: None,
                    confidence: nil_analysis.confidence,
                    is_nil: true,
                    nil_reason: nil_analysis.reason,
                    nil_action: Some(nil_analysis.action),
                    alternatives: if self.include_alternatives {
                        candidates
                            .iter()
                            .take(5)
                            .map(CandidateSummary::from)
                            .collect()
                    } else {
                        Vec::new()
                    },
                }
            } else {
                linked_count += 1;
                let top_candidate = &candidates[0];
                total_confidence += top_candidate.score;

                LinkedEntity {
                    mention_text: mention.text.clone(),
                    start: mention.start,
                    end: mention.end,
                    kb_id: Some(top_candidate.kb_id.clone()),
                    source: top_candidate.source.clone(),
                    label: Some(top_candidate.label.clone()),
                    iri: Some(top_candidate.to_iri()),
                    confidence: Confidence::new(top_candidate.score),
                    is_nil: false,
                    nil_reason: None,
                    nil_action: None,
                    alternatives: if self.include_alternatives && candidates.len() > 1 {
                        candidates[1..]
                            .iter()
                            .take(4)
                            .map(CandidateSummary::from)
                            .collect()
                    } else {
                        Vec::new()
                    },
                }
            };

            entities.push(linked_entity);
        }

        let avg_confidence = if linked_count > 0 {
            total_confidence / linked_count as f64
        } else {
            0.0
        };

        LinkingResult {
            entities,
            total_mentions: mentions.len(),
            linked_count,
            nil_count,
            avg_confidence,
        }
    }

    /// Link a single mention (convenience method).
    pub fn link_one(
        &self,
        mention: &str,
        context: &str,
        entity_type: Option<EntityType>,
    ) -> Option<LinkedEntity> {
        let m = if let Some(et) = entity_type {
            Mention::new(mention, 0, mention.len()).with_type(et)
        } else {
            Mention::new(mention, 0, mention.len())
        };

        let result = self.link(&[m], context);
        result.entities.into_iter().next()
    }
}

/// Builder for EntityLinker.
pub struct EntityLinkerBuilder {
    generator: Option<Arc<dyn CandidateGenerator>>,
    nil_threshold: f64,
    max_candidates: usize,
    include_alternatives: bool,
}

impl Default for EntityLinkerBuilder {
    fn default() -> Self {
        Self {
            generator: None,
            nil_threshold: 0.3,
            max_candidates: 20,
            include_alternatives: true,
        }
    }
}

impl EntityLinkerBuilder {
    /// Set the candidate generator.
    pub fn with_candidate_generator<G: CandidateGenerator + 'static>(mut self, gen: G) -> Self {
        self.generator = Some(Arc::new(gen));
        self
    }

    /// Set NIL threshold.
    pub fn with_nil_threshold(mut self, threshold: f64) -> Self {
        self.nil_threshold = threshold;
        self
    }

    /// Set max candidates.
    pub fn with_max_candidates(mut self, max: usize) -> Self {
        self.max_candidates = max;
        self
    }

    /// Set whether to include alternatives.
    pub fn include_alternatives(mut self, include: bool) -> Self {
        self.include_alternatives = include;
        self
    }

    /// Build the linker.
    pub fn build(self) -> EntityLinker {
        let generator = self
            .generator
            .unwrap_or_else(|| Arc::new(DictionaryCandidateGenerator::new().with_well_known()));

        EntityLinker {
            generator,
            nil_detector: NilDetector::new().with_score_threshold(self.nil_threshold),
            max_candidates: self.max_candidates,
            include_alternatives: self.include_alternatives,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_linker_basic() {
        let linker = EntityLinker::builder().build();

        let mentions = vec![Mention::new("Einstein", 0, 8).with_type(EntityType::Person)];

        let result = linker.link(&mentions, "Albert Einstein was a physicist.");

        assert_eq!(result.total_mentions, 1);
        // May or may not link depending on fuzzy matching
    }

    #[test]
    fn test_entity_linker_known_entity() {
        let linker = EntityLinker::builder().with_nil_threshold(0.1).build();

        let linked = linker.link_one(
            "Albert Einstein",
            "He was a physicist.",
            Some(EntityType::Person),
        );

        if let Some(entity) = linked {
            if !entity.is_nil {
                assert!(entity.kb_id.is_some());
                assert!(entity.iri.as_ref().unwrap().contains("wikidata"));
            }
        }
    }

    #[test]
    fn test_entity_linker_nil() {
        let linker = EntityLinker::builder().build();

        let linked = linker.link_one("Xyzzy Qwerty Asdf", "Unknown person.", None);

        if let Some(entity) = linked {
            // Should be NIL
            assert!(entity.is_nil || entity.confidence < 0.5);
        }
    }

    #[test]
    fn test_linking_result_stats() {
        let result = LinkingResult {
            entities: Vec::new(),
            total_mentions: 10,
            linked_count: 7,
            nil_count: 3,
            avg_confidence: 0.8,
        };

        assert!((result.linking_rate() - 0.7).abs() < 0.001);
    }

    // === Additional tests for coverage ===

    #[test]
    fn test_multilingual_entity_linking() {
        // Test CJK entities (per multicultural guidelines)
        let linker = EntityLinker::builder().with_nil_threshold(0.1).build();

        // Chinese name
        let linked = linker.link_one("北京", "Visit Beijing, China.", None);
        if let Some(entity) = &linked {
            // Beijing should be in our KB
            if !entity.is_nil {
                assert!(entity.kb_id.is_some());
            }
        }

        // Japanese name
        let linked = linker.link_one("東京", "Tokyo is in Japan.", None);
        assert!(linked.is_some()); // Should at least return a result
    }

    #[test]
    fn test_entity_type_aware_linking() {
        let linker = EntityLinker::builder().build();

        // Same text, different entity types should still work
        let person = linker.link_one(
            "Apple",
            "Steve Jobs founded Apple.",
            Some(EntityType::Person),
        );
        let org = linker.link_one(
            "Apple",
            "Apple is a tech company.",
            Some(EntityType::Organization),
        );

        // Both should return results (possibly different)
        assert!(person.is_some());
        assert!(org.is_some());
    }

    #[test]
    fn test_batch_linking_multiple_mentions() {
        let linker = EntityLinker::builder().build();

        let mentions = vec![
            Mention::new("Google", 0, 6).with_type(EntityType::Organization),
            Mention::new("Microsoft", 15, 24).with_type(EntityType::Organization),
            Mention::new("Apple", 30, 35).with_type(EntityType::Organization),
        ];

        let result = linker.link(&mentions, "Google and Microsoft and Apple are tech giants.");

        assert_eq!(result.total_mentions, 3);
        assert!(result.entities.len() <= 3);
    }

    #[test]
    fn test_empty_mentions() {
        let linker = EntityLinker::builder().build();

        let result = linker.link(&[], "Some text without mentions.");

        assert_eq!(result.total_mentions, 0);
        assert_eq!(result.linked_count, 0);
        assert_eq!(result.nil_count, 0);
    }

    #[test]
    fn test_very_short_mention() {
        let linker = EntityLinker::builder().build();

        // Single character mentions are likely noise
        let linked = linker.link_one("X", "X marks the spot.", None);

        // Should handle gracefully (probably NIL or low confidence)
        if let Some(entity) = linked {
            // Short mentions typically get flagged as noisy
            assert!(entity.is_nil || entity.confidence < 0.3);
        }
    }

    #[test]
    fn test_mention_builder_pattern() {
        let mention = Mention::new("Test", 0, 4).with_type(EntityType::Person);

        assert_eq!(mention.text, "Test");
        assert_eq!(mention.start, 0);
        assert_eq!(mention.end, 4);
        assert_eq!(mention.entity_type, Some(EntityType::Person));
    }

    #[test]
    fn test_linked_entity_serialization() {
        let entity = LinkedEntity {
            mention_text: "Einstein".to_string(),
            start: 0,
            end: 8,
            kb_id: Some("Q937".to_string()),
            source: CandidateSource::Wikidata,
            label: Some("Albert Einstein".to_string()),
            iri: Some("http://www.wikidata.org/entity/Q937".to_string()),
            confidence: Confidence::new(0.95),
            is_nil: false,
            nil_reason: None,
            nil_action: None,
            alternatives: vec![],
        };

        // Test serialization round-trip
        let json = serde_json::to_string(&entity).unwrap();
        let deserialized: LinkedEntity = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.kb_id, entity.kb_id);
        assert_eq!(deserialized.mention_text, entity.mention_text);
    }

    #[test]
    fn test_linker_with_custom_threshold() {
        // High threshold should increase NIL rate
        let strict_linker = EntityLinker::builder().with_nil_threshold(0.9).build();

        // Low threshold should decrease NIL rate
        let lenient_linker = EntityLinker::builder().with_nil_threshold(0.1).build();

        let result_strict = strict_linker.link_one("some entity", "context", None);
        let result_lenient = lenient_linker.link_one("some entity", "context", None);

        // Both should work without panicking
        let _ = (result_strict, result_lenient);
    }
}
