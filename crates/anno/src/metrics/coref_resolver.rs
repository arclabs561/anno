//! Coreference resolvers for analysis/evaluation pipelines.
//!
//! The core types ([`SimpleCorefResolver`], [`CorefConfig`]) are defined in
//! [`crate::backends::coref::simple`] and re-exported here for backward compatibility.
//!
//! This module additionally defines the discourse-aware resolver (feature-gated on `discourse`).

#[cfg(feature = "discourse")]
use crate::CanonicalId;
#[cfg(feature = "discourse")]
use crate::Entity;

// Re-export canonical definitions from backends/.
pub use crate::backends::coref::simple::{CorefConfig, SimpleCorefResolver};

// Re-export the canonical trait from `crate::core`.
pub use crate::CoreferenceResolver;

// =============================================================================
// Discourse-aware coreference (optional)
// =============================================================================

#[cfg(feature = "discourse")]
use crate::discourse::{
    classify_shell_noun, DiscourseReferent, DiscourseScope, EventExtractor, EventMention,
    ReferentType,
};

/// Configuration for discourse-aware coreference resolution.
#[cfg(feature = "discourse")]
#[derive(Debug, Clone)]
pub struct DiscourseCorefConfig {
    /// Base coreference config.
    pub base: CorefConfig,
    /// Enable shell noun detection.
    pub detect_shell_nouns: bool,
    /// Maximum sentences back to search for antecedent.
    pub max_sentence_distance: usize,
    /// Prefer clause-level antecedents for "this".
    pub prefer_clause_antecedent: bool,
}

#[cfg(feature = "discourse")]
impl Default for DiscourseCorefConfig {
    fn default() -> Self {
        Self {
            base: CorefConfig::default(),
            detect_shell_nouns: true,
            max_sentence_distance: 3,
            prefer_clause_antecedent: true,
        }
    }
}

/// Discourse-aware coreference resolver.
///
/// This is intended for analysis/eval paths; it does not attempt to be a full abstract-anaphora
/// system. It extracts a small event inventory and uses discourse boundary heuristics to propose
/// clause/sentence antecedent spans for demonstratives and shell-noun phrases.
#[cfg(feature = "discourse")]
#[derive(Debug, Clone)]
pub struct DiscourseAwareResolver {
    config: DiscourseCorefConfig,
    base_resolver: SimpleCorefResolver,
    scope: DiscourseScope,
    text: String,
    events: Vec<EventMention>,
}

#[cfg(feature = "discourse")]
impl DiscourseAwareResolver {
    /// Create a new discourse-aware resolver.
    #[must_use]
    pub fn new(config: DiscourseCorefConfig, text: &str) -> Self {
        let scope = DiscourseScope::analyze(text);
        let extractor = EventExtractor::default();
        let events = extractor.extract(text);

        Self {
            base_resolver: SimpleCorefResolver::new(config.base.clone()),
            config,
            scope,
            text: text.to_string(),
            events,
        }
    }

    /// Get extracted events.
    #[must_use]
    pub fn events(&self) -> &[EventMention] {
        &self.events
    }

    /// Resolve coreference with discourse awareness.
    #[must_use]
    pub fn resolve_entities(&self, entities: &[Entity]) -> Vec<Entity> {
        // First pass: nominal-only coref.
        let mut resolved = self.base_resolver.resolve_entities(entities);

        // Second pass: for unresolved abstract anaphors, attach a synthetic canonical_id and encode
        // antecedent info in `normalized` as a lightweight carrier.
        for (i, entity) in resolved.iter_mut().enumerate() {
            if entity.canonical_id.is_some() {
                continue;
            }
            if self.is_abstract_anaphor(&entity.text) {
                if let Some(antecedent) = self.find_discourse_antecedent(entity) {
                    let cluster_id = CanonicalId::new(10_000 + i as u64);
                    entity.canonical_id = Some(cluster_id);
                    entity.normalized = Some(format!(
                        "DISCOURSE_REF:{:?}@{}-{}",
                        antecedent.referent_type, antecedent.start, antecedent.end
                    ));
                }
            }
        }

        resolved
    }

    /// Resolve coreference with discourse awareness.
    ///
    /// Inherent-method convenience wrapper (so callers don't need to import the
    /// [`CoreferenceResolver`] trait to call `resolver.resolve(&entities)`).
    #[must_use]
    pub fn resolve(&self, entities: &[Entity]) -> Vec<Entity> {
        self.resolve_entities(entities)
    }

    fn is_abstract_anaphor(&self, text: &str) -> bool {
        let lower = text.to_lowercase();

        if matches!(lower.as_str(), "this" | "that" | "it") {
            return true;
        }

        let words: Vec<&str> = lower.split_whitespace().collect();
        if words.len() >= 2 {
            let det = words[0];
            let noun = words
                .last()
                .expect("words has >= 2 elements")
                .trim_matches(|c: char| !c.is_alphanumeric());
            if matches!(det, "this" | "that" | "the" | "such")
                && classify_shell_noun(noun).is_some()
            {
                return true;
            }
        }

        false
    }

    fn find_event_near(&self, offset: usize, max_distance: usize) -> Option<&EventMention> {
        if let Some((clause_start, clause_end)) = self.scope.clause_at(offset) {
            for event in &self.events {
                if event.trigger_start >= clause_start && event.trigger_end <= clause_end {
                    return Some(event);
                }
            }
        }

        self.events
            .iter()
            .filter(|e| {
                let dist = if e.trigger_end <= offset {
                    offset - e.trigger_end
                } else {
                    e.trigger_start.saturating_sub(offset)
                };
                dist <= max_distance
            })
            .min_by_key(|e| {
                if e.trigger_end <= offset {
                    offset - e.trigger_end
                } else {
                    e.trigger_start.saturating_sub(offset)
                }
            })
    }

    fn find_event_clause_span(&self, event: &EventMention) -> (usize, usize) {
        if let Some((start, end)) = self.scope.clause_at(event.trigger_start) {
            return (start, end);
        }
        if let Some((start, end)) = self.scope.sentence_at(event.trigger_start) {
            return (start, end);
        }
        let char_count = self.text.chars().count();
        let context_before = event.trigger_start.saturating_sub(30);
        let context_after = (event.trigger_end + 30).min(char_count);
        (context_before, context_after)
    }

    fn infer_referent_type(&self, text: &str) -> ReferentType {
        let lower = text.to_lowercase();
        for event in &self.events {
            if lower.contains(&event.trigger.to_lowercase()) {
                if let Some(ref event_type) = event.trigger_type {
                    if event_type.starts_with("conflict:")
                        || event_type.starts_with("movement:")
                        || event_type.starts_with("transaction:")
                        || event_type.starts_with("justice:")
                        || event_type.starts_with("personnel:")
                        || event_type.starts_with("life:")
                        || event_type.starts_with("disaster:")
                        || event_type.starts_with("business:")
                    {
                        return ReferentType::Event;
                    }
                    if event_type.starts_with("economic:") {
                        return ReferentType::Situation;
                    }
                }
                return ReferentType::Event;
            }
        }

        let event_indicators = [
            "ed ",
            " was ",
            " were ",
            " had ",
            " did ",
            " happened",
            " occurred",
        ];
        for ind in &event_indicators {
            if lower.contains(ind) {
                return ReferentType::Event;
            }
        }

        let fact_indicators = [" is ", " are ", " equals ", " means "];
        for ind in &fact_indicators {
            if lower.contains(ind) {
                return ReferentType::Fact;
            }
        }

        let prop_indicators = [" might ", " may ", " could ", " would ", " should ", " if "];
        for ind in &prop_indicators {
            if lower.contains(ind) {
                return ReferentType::Proposition;
            }
        }

        let sit_indicators = [" while ", " as ", "ing ", " continues", " remains"];
        for ind in &sit_indicators {
            if lower.contains(ind) {
                return ReferentType::Situation;
            }
        }

        ReferentType::Event
    }

    /// Find a discourse-level antecedent for an abstract anaphor.
    #[must_use]
    pub fn find_discourse_antecedent(&self, anaphor: &Entity) -> Option<DiscourseReferent> {
        if let Some(event) = self.find_event_near(anaphor.start(), 200) {
            let (clause_char_start, clause_char_end) = self.find_event_clause_span(event);
            let span_text: String = self
                .text
                .chars()
                .skip(clause_char_start)
                .take(clause_char_end.saturating_sub(clause_char_start))
                .collect();

            return Some(
                DiscourseReferent::new(ReferentType::Event, clause_char_start, clause_char_end)
                    .with_event(event.clone())
                    .with_text(span_text)
                    .with_confidence(0.85),
            );
        }

        let candidates = self.scope.candidate_antecedent_spans(anaphor.start());
        let shell_class = if self.config.detect_shell_nouns {
            let lower = anaphor.text.to_lowercase();
            let last_word = lower.split_whitespace().last().map(|w| w.to_string());
            last_word
                .as_ref()
                .and_then(|w| classify_shell_noun(w.trim_matches(|c: char| !c.is_alphanumeric())))
        } else {
            None
        };

        for (start, end) in candidates
            .into_iter()
            .take(self.config.max_sentence_distance)
        {
            let span_text = self.scope.extract_span(&self.text, start, end);
            if span_text.trim().len() < 3 {
                continue;
            }
            let ref_type = self.infer_referent_type(span_text);
            if let Some(class) = &shell_class {
                let expected_types = class.typical_antecedent_types();
                if !expected_types.contains(&ref_type) {
                    continue;
                }
            }
            return Some(
                DiscourseReferent::new(ref_type, start, end)
                    .with_text(span_text)
                    .with_confidence(0.7),
            );
        }

        None
    }
}

#[cfg(feature = "discourse")]
impl CoreferenceResolver for DiscourseAwareResolver {
    fn resolve(&self, entities: &[Entity]) -> Vec<Entity> {
        self.resolve_entities(entities)
    }

    fn name(&self) -> &'static str {
        "discourse-aware"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::coref::{entities_to_chains, CorefChain, Mention};
    use crate::metrics::coref_metrics::{
        b_cubed_score, ceaf_e_score, conll_f1, lea_score, muc_score, CorefEvaluation, CorefScores,
    };
    use crate::CanonicalId;
    use crate::{Entity, EntityType};

    // ---- helpers ----

    fn person(text: &str, start: usize, end: usize) -> Entity {
        Entity::new(text, EntityType::Person, start, end, 0.9)
    }

    fn org(text: &str, start: usize, end: usize) -> Entity {
        Entity::new(text, EntityType::Organization, start, end, 0.9)
    }

    fn loc(text: &str, start: usize, end: usize) -> Entity {
        Entity::new(text, EntityType::Location, start, end, 0.9)
    }

    // ---- SimpleCorefResolver: basic clustering ----

    #[test]
    fn resolve_empty_entities() {
        let resolver = SimpleCorefResolver::default();
        let result = resolver.resolve(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn resolve_single_entity_gets_cluster_id() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![person("Alice", 0, 5)];
        let resolved = resolver.resolve(&entities);
        assert_eq!(resolved.len(), 1);
        assert!(
            resolved[0].canonical_id.is_some(),
            "singleton should get a cluster id"
        );
    }

    #[test]
    fn exact_name_match_clusters_together() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("John Smith", 0, 10),
            org("Acme Corp", 15, 24),
            person("John Smith", 30, 40),
        ];
        let resolved = resolver.resolve(&entities);

        // Both "John Smith" mentions should share the same cluster.
        let id0 = resolved[0].canonical_id.unwrap();
        let id2 = resolved[2].canonical_id.unwrap();
        assert_eq!(id0, id2, "exact name matches must share cluster id");

        // "Acme Corp" should be in a different cluster.
        let id1 = resolved[1].canonical_id.unwrap();
        assert_ne!(id0, id1, "different names must be in different clusters");
    }

    #[test]
    fn fuzzy_substring_match() {
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: true,
            ..CorefConfig::default()
        });
        let entities = vec![person("John Smith", 0, 10), person("Smith", 20, 25)];
        let resolved = resolver.resolve(&entities);
        let id0 = resolved[0].canonical_id.unwrap();
        let id1 = resolved[1].canonical_id.unwrap();
        assert_eq!(id0, id1, "substring match should cluster together");
    }

    #[test]
    fn fuzzy_matching_disabled_does_not_merge_substrings() {
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: false,
            proper_containment: false,
            strict_head_match: false,
            proper_head_word_match: false,
            relaxed_head_match: false,
            ..CorefConfig::default()
        });
        let entities = vec![person("John Smith", 0, 10), person("Smith", 20, 25)];
        let resolved = resolver.resolve(&entities);
        let id0 = resolved[0].canonical_id.unwrap();
        let id1 = resolved[1].canonical_id.unwrap();
        assert_ne!(id0, id1, "with fuzzy off, substring should not cluster");
    }

    #[test]
    fn type_mismatch_prevents_clustering() {
        let resolver = SimpleCorefResolver::default();
        // Same text, different entity types.
        let entities = vec![person("Apple", 0, 5), org("Apple", 10, 15)];
        let resolved = resolver.resolve(&entities);
        let id0 = resolved[0].canonical_id.unwrap();
        let id1 = resolved[1].canonical_id.unwrap();
        assert_ne!(
            id0, id1,
            "same text but different EntityType must be separate clusters"
        );
    }

    // ---- Pronoun resolution ----

    #[test]
    fn pronoun_resolves_to_gender_compatible_entity() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("Alice", 0, 5),
            person("Bob", 10, 13),
            person("she", 20, 23),
        ];
        let resolved = resolver.resolve(&entities);
        // With the name gazetteer, "Alice" -> Feminine, "Bob" -> Masculine.
        // "she" is feminine, so it should match "Alice" (gender compatible),
        // skipping "Bob" (masculine, incompatible).
        let she_id = resolved[2].canonical_id.unwrap();
        let alice_id = resolved[0].canonical_id.unwrap();
        assert_eq!(
            she_id, alice_id,
            "pronoun should resolve to gender-compatible entity"
        );
    }

    #[test]
    fn pronoun_it_resolves_to_org() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("Alice", 0, 5),
            org("Acme Corp", 10, 19),
            org("it", 25, 27),
        ];
        let resolved = resolver.resolve(&entities);
        // "it" is compatible with Organization but not Person.
        let it_id = resolved[2].canonical_id.unwrap();
        let acme_id = resolved[1].canonical_id.unwrap();
        assert_eq!(
            it_id, acme_id,
            "\"it\" should resolve to the org, not the person"
        );
    }

    #[test]
    fn pronoun_it_compatible_with_person() {
        // "it/its" can be used as personal pronouns (e.g., some genderqueer individuals).
        let resolver = SimpleCorefResolver::default();
        let entities = vec![person("Alice", 0, 5), person("it", 10, 12)];
        let resolved = resolver.resolve(&entities);
        let alice_id = resolved[0].canonical_id.unwrap();
        let it_id = resolved[1].canonical_id.unwrap();
        assert_eq!(
            alice_id, it_id,
            "\"it\" should resolve to Person antecedent"
        );
    }

    // ---- resolve_to_chains ----

    #[test]
    fn resolve_to_chains_produces_correct_clusters() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("Marie Curie", 0, 11),
            org("Nobel Committee", 15, 30),
            person("Marie Curie", 35, 46),
            person("she", 50, 53),
        ];
        let chains = resolver.resolve_to_chains(&entities);
        // At least one chain should have multiple mentions.
        let multi_mention_chains: Vec<_> = chains.iter().filter(|c| c.len() > 1).collect();
        assert!(
            !multi_mention_chains.is_empty(),
            "coreferent entities should produce multi-mention chains"
        );
    }

    // ---- CorefEvaluation: perfect prediction ----

    #[test]
    fn coref_evaluation_perfect_prediction() {
        let gold = vec![
            CorefChain::new(vec![Mention::new("John", 0, 4), Mention::new("he", 10, 12)]),
            CorefChain::new(vec![
                Mention::new("IBM", 20, 23),
                Mention::new("the company", 30, 41),
            ]),
        ];
        let eval = CorefEvaluation::compute(&gold, &gold);

        assert!(
            (eval.conll_f1 - 1.0).abs() < 1e-9,
            "perfect prediction should yield CoNLL F1 = 1.0, got {}",
            eval.conll_f1
        );
        assert!((eval.muc.f1 - 1.0).abs() < 1e-9);
        assert!((eval.b_cubed.f1 - 1.0).abs() < 1e-9);
        assert!((eval.ceaf_e.f1 - 1.0).abs() < 1e-9);
        assert!((eval.lea.f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn coref_evaluation_no_overlap() {
        let predicted = vec![CorefChain::new(vec![
            Mention::new("Alice", 0, 5),
            Mention::new("she", 10, 13),
        ])];
        let gold = vec![CorefChain::new(vec![
            Mention::new("Bob", 20, 23),
            Mention::new("he", 30, 32),
        ])];
        let eval = CorefEvaluation::compute(&predicted, &gold);
        // No common mentions => metrics should be zero (except BLANC edge case).
        assert!(
            eval.muc.f1.abs() < 1e-9,
            "no overlap should yield MUC F1 = 0, got {}",
            eval.muc.f1
        );
        assert!(eval.b_cubed.f1.abs() < 1e-9);
        assert!(eval.conll_f1.abs() < 1e-9);
    }

    #[test]
    fn coref_evaluation_partial_overlap() {
        // Gold: {A, B, C} in one cluster.
        let gold = vec![CorefChain::new(vec![
            Mention::new("John", 0, 4),
            Mention::new("he", 10, 12),
            Mention::new("him", 20, 23),
        ])];
        // Predicted: splits into two clusters: {A, B} and {C}.
        let predicted = vec![
            CorefChain::new(vec![Mention::new("John", 0, 4), Mention::new("he", 10, 12)]),
            CorefChain::new(vec![Mention::new("him", 20, 23)]),
        ];
        let eval = CorefEvaluation::compute(&predicted, &gold);
        // Should be imperfect but nonzero.
        assert!(
            eval.conll_f1 > 0.0,
            "partial overlap should yield nonzero F1"
        );
        assert!(
            eval.conll_f1 < 1.0,
            "partial overlap should not yield perfect F1"
        );
    }

    // ---- Individual metric functions ----

    #[test]
    fn muc_score_perfect() {
        let chains = vec![CorefChain::new(vec![
            Mention::new("A", 0, 1),
            Mention::new("B", 5, 6),
            Mention::new("C", 10, 11),
        ])];
        let (p, r, f1) = muc_score(&chains, &chains);
        assert!((p - 1.0).abs() < 1e-9);
        assert!((r - 1.0).abs() < 1e-9);
        assert!((f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn b_cubed_singleton_perfect() {
        // Single mention chains: B3 should give perfect scores when predicted == gold.
        let chains = vec![
            CorefChain::new(vec![Mention::new("X", 0, 1)]),
            CorefChain::new(vec![Mention::new("Y", 5, 6)]),
        ];
        let (p, r, f1) = b_cubed_score(&chains, &chains);
        assert!((p - 1.0).abs() < 1e-9);
        assert!((r - 1.0).abs() < 1e-9);
        assert!((f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn lea_score_empty() {
        let (p, r, f1) = lea_score(&[], &[]);
        // Empty inputs: no common mentions.
        assert!(p.abs() < 1e-9);
        assert!(r.abs() < 1e-9);
        assert!(f1.abs() < 1e-9);
    }

    #[test]
    fn conll_f1_is_average_of_three() {
        let gold = vec![CorefChain::new(vec![
            Mention::new("A", 0, 1),
            Mention::new("B", 5, 6),
        ])];
        let predicted = vec![CorefChain::new(vec![
            Mention::new("A", 0, 1),
            Mention::new("B", 5, 6),
        ])];
        let conll = conll_f1(&predicted, &gold);
        let (_, _, muc_f1) = muc_score(&predicted, &gold);
        let (_, _, b3_f1) = b_cubed_score(&predicted, &gold);
        let (_, _, ceaf_f1) = ceaf_e_score(&predicted, &gold);
        let expected = (muc_f1 + b3_f1 + ceaf_f1) / 3.0;
        assert!(
            (conll - expected).abs() < 1e-9,
            "conll_f1 should equal mean of MUC, B3, CEAFe F1"
        );
    }

    // ---- CorefScores ----

    #[test]
    fn coref_scores_f1_computation() {
        let s = CorefScores::new(0.8, 0.6);
        let expected_f1 = 2.0 * 0.8 * 0.6 / (0.8 + 0.6);
        assert!((s.f1 - expected_f1).abs() < 1e-9);
    }

    #[test]
    fn coref_scores_zero_precision_and_recall() {
        let s = CorefScores::new(0.0, 0.0);
        assert!(s.f1.abs() < 1e-9, "0/0 should yield F1 = 0");
    }

    // ---- CorefEvaluation aggregate helpers ----

    #[test]
    fn coref_evaluation_all_f1_scores_len() {
        let gold = vec![CorefChain::new(vec![
            Mention::new("A", 0, 1),
            Mention::new("B", 5, 6),
        ])];
        let eval = CorefEvaluation::compute(&gold, &gold);
        assert_eq!(
            eval.all_f1_scores().len(),
            6,
            "should report 6 metric F1 values"
        );
    }

    #[test]
    fn coref_evaluation_average_f1_upper_bound() {
        // With perfect prediction, average F1 (across 6 metrics) should be high.
        // BLANC can deviate from 1.0 when there is only one cluster (no negative pairs),
        // so we check that average_f1 >= 0.9 rather than == 1.0.
        let gold = vec![
            CorefChain::new(vec![
                Mention::new("A", 0, 1),
                Mention::new("B", 5, 6),
                Mention::new("C", 10, 11),
            ]),
            CorefChain::new(vec![Mention::new("X", 20, 21), Mention::new("Y", 25, 26)]),
        ];
        let eval = CorefEvaluation::compute(&gold, &gold);
        assert!(
            eval.average_f1() > 0.9,
            "perfect prediction with multiple clusters should have high average F1, got {}",
            eval.average_f1()
        );
        // The core three (MUC, B3, CEAFe) used in CoNLL should each be 1.0.
        assert!((eval.muc.f1 - 1.0).abs() < 1e-9);
        assert!((eval.b_cubed.f1 - 1.0).abs() < 1e-9);
        assert!((eval.ceaf_e.f1 - 1.0).abs() < 1e-9);
    }

    // ---- entities_to_chains round-trip ----

    #[test]
    fn entities_to_chains_groups_by_canonical_id() {
        let mut e1 = person("Alice", 0, 5);
        e1.canonical_id = Some(CanonicalId::new(1));
        let mut e2 = person("she", 10, 13);
        e2.canonical_id = Some(CanonicalId::new(1));
        let mut e3 = org("Acme", 20, 24);
        e3.canonical_id = Some(CanonicalId::new(2));

        let chains = entities_to_chains(&[e1, e2, e3]);
        // Cluster 1 should have 2 mentions, cluster 2 should have 1.
        let two_mention = chains.iter().find(|c| c.len() == 2);
        let one_mention = chains.iter().find(|c| c.len() == 1);
        assert!(two_mention.is_some(), "should have a 2-mention chain");
        assert!(one_mention.is_some(), "should have a 1-mention chain");
    }

    #[test]
    fn entities_without_canonical_id_become_singletons() {
        let entities = vec![person("Alice", 0, 5), person("Bob", 10, 13)];
        let chains = entities_to_chains(&entities);
        // Each unresolved entity becomes its own singleton chain.
        assert_eq!(chains.len(), 2);
        assert!(chains.iter().all(|c| c.len() == 1));
    }

    // ---- CoreferenceResolver trait impl ----

    #[test]
    fn simple_resolver_trait_name() {
        let resolver = SimpleCorefResolver::default();
        assert_eq!(CoreferenceResolver::name(&resolver), "simple-rule-based");
    }

    #[test]
    fn it_pronoun_resolves_to_person() {
        // "it/its" should be compatible with Person entities (used as personal pronouns
        // by some genderqueer individuals, e.g., "Alex uses it/its pronouns").
        let resolver = SimpleCorefResolver::default();
        let entities = vec![person("Alex", 0, 4), person("it", 6, 8)];
        let chains = resolver.resolve_to_chains(&entities);
        // Alex and "it" should be in the same chain.
        let alex_chain = chains
            .iter()
            .find(|c| c.mentions.iter().any(|m| m.text == "Alex"));
        assert!(alex_chain.is_some(), "Alex should appear in a chain");
        let alex_chain = alex_chain.unwrap();
        assert!(
            alex_chain.mentions.iter().any(|m| m.text == "it"),
            "\"it\" should corefer with Person entity \"Alex\""
        );
    }

    // ---- End-to-end: resolve then evaluate ----

    #[test]
    fn resolve_then_evaluate_round_trip() {
        let resolver = SimpleCorefResolver::default();

        // Synthetic document: "Alice went to Paris. She loved Paris."
        let entities = vec![
            person("Alice", 0, 5),
            loc("Paris", 14, 19),
            person("she", 21, 24),
            loc("Paris", 31, 36),
        ];

        let predicted_chains = resolver.resolve_to_chains(&entities);
        // Build gold: Alice+she in one cluster, Paris+Paris in another.
        let gold_chains = vec![
            CorefChain::new(vec![
                Mention::new("Alice", 0, 5),
                Mention::new("she", 21, 24),
            ]),
            CorefChain::new(vec![
                Mention::new("Paris", 14, 19),
                Mention::new("Paris", 31, 36),
            ]),
        ];

        let eval = CorefEvaluation::compute(&predicted_chains, &gold_chains);
        // The resolver should get at least partial credit.
        assert!(
            eval.conll_f1 > 0.0,
            "resolve-then-evaluate should produce nonzero CoNLL F1"
        );
    }

    // ====================================================================
    // Audit-driven regression tests
    // ====================================================================

    // ---- Gender-blind pronoun resolution (documenting known limitation) ----

    #[test]
    fn pronoun_resolves_with_gazetteer() {
        // With name gazetteer enabled (default), "She" should resolve to "Alice"
        // (Feminine) instead of "Bob" (Masculine), even though Bob is nearer.
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("Alice", 0, 5),
            person("Bob", 20, 23),
            person("She", 30, 33),
        ];
        let resolved = resolver.resolve(&entities);

        let she_id = resolved[2].canonical_id.unwrap();
        let alice_id = resolved[0].canonical_id.unwrap();
        assert_eq!(
            she_id, alice_id,
            "With gazetteer, 'She' should resolve to 'Alice' (gender match)"
        );
    }

    #[test]
    fn pronoun_gender_blind_without_gazetteer() {
        // Without gazetteer, falls back to nearest compatible entity.
        let resolver = SimpleCorefResolver::new(CorefConfig {
            use_name_gazetteer: false,
            ..CorefConfig::default()
        });
        let entities = vec![
            person("Alice", 0, 5),
            person("Bob", 20, 23),
            person("She", 30, 33),
        ];
        let resolved = resolver.resolve(&entities);

        let she_id = resolved[2].canonical_id.unwrap();
        let bob_id = resolved[1].canonical_id.unwrap();
        // "She" resolves to "Bob" (nearest) because infer_gender("Bob") returns None,
        // which is compatible with Feminine via the (None, Some) match arm.
        assert_eq!(
            she_id, bob_id,
            "Without gazetteer, 'She' resolves to nearest entity 'Bob' (gender-blind)"
        );
    }

    // ---- String matching edge cases ----

    #[test]
    fn names_match_substring_short_name() {
        // "Li" should NOT match "political" via substring containment.
        // Short names (< 3 chars) require word-boundary alignment.
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: true,
            ..CorefConfig::default()
        });
        let entities = vec![person("political", 0, 9), person("Li", 20, 22)];
        let resolved = resolver.resolve(&entities);
        let id0 = resolved[0].canonical_id.unwrap();
        let id1 = resolved[1].canonical_id.unwrap();
        assert_ne!(
            id0, id1,
            "'Li' must not match 'political' -- short-name substring guard"
        );
    }

    #[test]
    fn names_match_legitimate_substring() {
        // "Obama" (5 chars, >= 3) should still match "Barack Obama" via substring.
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: true,
            ..CorefConfig::default()
        });
        let entities = vec![person("Barack Obama", 0, 12), person("Obama", 20, 25)];
        let resolved = resolver.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "'Obama' should match 'Barack Obama' via substring containment"
        );
    }

    #[test]
    fn names_match_short_unrelated() {
        // Two-character name "Al" should not match "gallery".
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: true,
            ..CorefConfig::default()
        });
        let entities = vec![person("gallery", 0, 7), person("Al", 20, 22)];
        let resolved = resolver.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "'Al' must not match 'gallery' -- short-name substring guard"
        );
    }

    #[test]
    fn names_match_short_word_boundary() {
        // Short name "Li" SHOULD match "Li Wei" because "li" is a complete word in "li wei".
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: true,
            ..CorefConfig::default()
        });
        let entities = vec![person("Li Wei", 0, 6), person("Li", 20, 22)];
        let resolved = resolver.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "'Li' should match 'Li Wei' via word-boundary containment"
        );
    }

    #[test]
    fn names_match_last_word() {
        // "John Smith" should match "Smith" via last-word heuristic.
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: true,
            ..CorefConfig::default()
        });
        let entities = vec![person("John Smith", 0, 10), person("Smith", 20, 25)];
        let resolved = resolver.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "'John Smith' and 'Smith' should cluster via last-word match"
        );
    }

    #[test]
    fn names_match_acronym() {
        // Acronym matching: "IBM" matches "International Business Machines".
        let resolver = SimpleCorefResolver::new(CorefConfig {
            acronym_matching: true,
            ..CorefConfig::default()
        });
        let entities = vec![
            org("International Business Machines", 0, 31),
            org("IBM", 40, 43),
        ];
        let resolved = resolver.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "'IBM' should cluster with 'International Business Machines' via acronym matching"
        );
    }

    #[test]
    fn acronym_matching_disabled() {
        // When acronym_matching is disabled, "IBM" does NOT match.
        let resolver = SimpleCorefResolver::new(CorefConfig {
            acronym_matching: false,
            fuzzy_matching: false,
            ..CorefConfig::default()
        });
        let entities = vec![
            org("International Business Machines", 0, 31),
            org("IBM", 40, 43),
        ];
        let resolved = resolver.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "Acronym matching disabled: 'IBM' should not cluster"
        );
    }

    #[test]
    fn acronym_type_mismatch_rejected() {
        // Acronym must share entity type.
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            org("International Business Machines", 0, 31),
            person("IBM", 40, 43),
        ];
        let resolved = resolver.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "Acronym across different entity types should not cluster"
        );
    }

    #[test]
    fn acronym_length_mismatch_rejected() {
        // "UN" (2 letters) does not match "United Nations Organization" (3 words)
        // via acronym matching. Fuzzy matching disabled to isolate the acronym check.
        let config = CorefConfig {
            fuzzy_matching: false,
            ..Default::default()
        };
        let resolver = SimpleCorefResolver::new(config);
        let entities = vec![org("United Nations Organization", 0, 26), org("UN", 30, 32)];
        let resolved = resolver.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "Acronym letter count must match word count"
        );
    }

    #[test]
    fn acronym_un_matches() {
        // "UN" (2 letters) matches "United Nations" (2 words).
        let resolver = SimpleCorefResolver::default();
        let entities = vec![org("United Nations", 0, 14), org("UN", 20, 22)];
        let resolved = resolver.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "'UN' should cluster with 'United Nations'"
        );
    }

    // ---- Relaxed head match ----

    #[test]
    fn relaxed_head_match_shared_last_word() {
        // "President Obama" and "Barack Obama" share head word "Obama".
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: false,
            proper_containment: false,
            ..CorefConfig::default()
        });
        let entities = vec![
            person("President Obama", 0, 15),
            person("Barack Obama", 20, 32),
        ];
        let resolved = resolver.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "'President Obama' and 'Barack Obama' should cluster via shared head 'Obama'"
        );
    }

    #[test]
    fn relaxed_head_match_requires_two_words() {
        // Single-word mentions should NOT match via relaxed head match.
        // Also disable strict_head_match which handles single-word mentions.
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: false,
            proper_containment: false,
            strict_head_match: false,
            proper_head_word_match: false,
            ..CorefConfig::default()
        });
        let entities = vec![person("Obama", 0, 5), person("Barack Obama", 20, 32)];
        let resolved = resolver.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "Relaxed head match requires both mentions to have 2+ words"
        );
    }

    #[test]
    fn relaxed_head_match_type_mismatch() {
        // Different entity types should not match even with same head word.
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: false,
            proper_containment: false,
            ..CorefConfig::default()
        });
        let entities = vec![
            person("President Obama", 0, 15),
            org("Foundation Obama", 20, 36),
        ];
        let resolved = resolver.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "Relaxed head match requires same entity type"
        );
    }

    #[test]
    fn relaxed_head_match_case_insensitive() {
        // Head word comparison should be case-insensitive.
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: false,
            proper_containment: false,
            ..CorefConfig::default()
        });
        let entities = vec![
            person("President OBAMA", 0, 15),
            person("Barack obama", 20, 32),
        ];
        let resolved = resolver.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "Relaxed head match should be case-insensitive"
        );
    }

    #[test]
    fn relaxed_head_match_disabled() {
        // Disable all head-based sieves to isolate relaxed_head_match.
        let resolver = SimpleCorefResolver::new(CorefConfig {
            relaxed_head_match: false,
            fuzzy_matching: false,
            proper_containment: false,
            strict_head_match: false,
            proper_head_word_match: false,
            ..CorefConfig::default()
        });
        let entities = vec![
            person("President Obama", 0, 15),
            person("Barack Obama", 20, 32),
        ];
        let resolved = resolver.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "With relaxed_head_match disabled, should not cluster"
        );
    }

    // ---- Proper noun containment ----

    #[test]
    fn proper_containment_word_boundary() {
        // "Obama" is a complete word in "Barack Obama".
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: false,
            relaxed_head_match: false,
            ..CorefConfig::default()
        });
        let entities = vec![person("Barack Obama", 0, 12), person("Obama", 20, 25)];
        let resolved = resolver.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "'Obama' should cluster with 'Barack Obama' via proper containment"
        );
    }

    #[test]
    fn proper_containment_not_partial_word() {
        // "Nation" is NOT a complete word in "United Nations" (it would be "Nations").
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: false,
            relaxed_head_match: false,
            ..CorefConfig::default()
        });
        let entities = vec![org("United Nations", 0, 14), org("Nation", 20, 26)];
        let resolved = resolver.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "'Nation' is not a word-boundary match in 'United Nations'"
        );
    }

    #[test]
    fn proper_containment_type_mismatch() {
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: false,
            relaxed_head_match: false,
            ..CorefConfig::default()
        });
        let entities = vec![person("Barack Obama", 0, 12), org("Obama", 20, 25)];
        let resolved = resolver.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "Proper containment requires same entity type"
        );
    }

    #[test]
    fn proper_containment_disabled() {
        // Disable all sieves that could match "Barack Obama" / "Obama".
        let resolver = SimpleCorefResolver::new(CorefConfig {
            proper_containment: false,
            fuzzy_matching: false,
            relaxed_head_match: false,
            strict_head_match: false,
            proper_head_word_match: false,
            ..CorefConfig::default()
        });
        let entities = vec![person("Barack Obama", 0, 12), person("Obama", 20, 25)];
        let resolved = resolver.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "With proper_containment disabled, should not cluster"
        );
    }

    #[test]
    fn proper_containment_multi_word_subsequence() {
        // "New York" is contained as a word sequence in "New York City".
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: false,
            relaxed_head_match: false,
            ..CorefConfig::default()
        });
        let entities = vec![loc("New York City", 0, 13), loc("New York", 20, 28)];
        let resolved = resolver.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "'New York' should cluster with 'New York City' via proper containment"
        );
    }

    #[test]
    fn names_match_case_insensitive() {
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: true,
            ..CorefConfig::default()
        });
        let entities = vec![person("alice", 0, 5), person("Alice", 20, 25)];
        let resolved = resolver.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "'alice' and 'Alice' should cluster (canonical_form lowercases)"
        );
    }

    // ---- Missing coref features (negative tests documenting gaps) ----

    #[test]
    fn appositive_adjacent_entities_clustered() {
        // Precise constructs sieve: adjacent entities (gap <= 2) with same type are merged.
        // "Obama" [0,5) and "the president" [7,20) have gap=2, matching the appositive pattern.
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("Obama", 0, 5),
            person("the president", 7, 20),
            person("He", 28, 30),
        ];
        let resolved = resolver.resolve(&entities);
        let obama_id = resolved[0].canonical_id.unwrap();
        let president_id = resolved[1].canonical_id.unwrap();
        assert_eq!(
            obama_id, president_id,
            "Appositive 'Obama, the president' should be clustered via precise constructs sieve"
        );
    }

    #[test]
    fn no_predicate_nominal() {
        // Known gap: "John is a doctor. The doctor left." -- predicate nominal
        // coreference requires understanding copular constructions.
        let resolver = SimpleCorefResolver::default();
        let entities = vec![person("John", 0, 4), person("The doctor", 22, 32)];
        let resolved = resolver.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "Known gap: predicate nominal 'John' / 'The doctor' not clustered"
        );
    }

    #[test]
    fn no_demonstrative_resolution() {
        // Known gap: "This" is not in the pronoun list for SimpleCorefResolver,
        // so abstract demonstrative anaphora is not resolved.
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            org("The company", 0, 11),
            // "This" as demonstrative anaphor
            person("This", 35, 39),
        ];
        let resolved = resolver.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "Known gap: demonstrative 'This' not resolved by SimpleCorefResolver"
        );
    }

    #[test]
    fn no_split_antecedent() {
        // Known gap: "They" cannot resolve to the set {John, Mary}. The data model
        // represents canonical_id as a single ID, not a set of antecedents.
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("John", 0, 4),
            person("Mary", 9, 13),
            person("They", 20, 24),
        ];
        let resolved = resolver.resolve(&entities);
        let they_id = resolved[2].canonical_id.unwrap();
        // "They" will resolve to the nearest compatible entity (Mary), not {John, Mary}.
        let john_id = resolved[0].canonical_id.unwrap();
        let mary_id = resolved[1].canonical_id.unwrap();
        let resolves_to_both = they_id == john_id && they_id == mary_id;
        assert!(
            !resolves_to_both,
            "Known gap: split antecedent -- 'They' cannot resolve to {{John, Mary}}"
        );
    }

    #[test]
    fn no_relative_pronoun() {
        // "who" is correctly NOT in the pronoun list. Relative pronouns should not
        // be rewritten in coreference output.
        let resolver = SimpleCorefResolver::default();
        assert!(
            !resolver.is_pronoun("who"),
            "'who' should not be treated as a resolvable pronoun"
        );
        assert!(!resolver.is_pronoun("which"));
        assert!(!resolver.is_pronoun("that"));
    }

    // ---- Boundary conditions ----

    #[test]
    fn empty_entities_no_crash() {
        let resolver = SimpleCorefResolver::default();
        let result = resolver.resolve(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn single_entity_gets_canonical_id() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![person("Alice", 0, 5)];
        let resolved = resolver.resolve(&entities);
        assert!(resolved[0].canonical_id.is_some());
    }

    #[test]
    fn all_same_type_cluster_by_name() {
        // Multiple Person entities with the same name should all cluster together.
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("Alice", 0, 5),
            person("Alice", 20, 25),
            person("Alice", 40, 45),
        ];
        let resolved = resolver.resolve(&entities);
        let id = resolved[0].canonical_id.unwrap();
        assert!(
            resolved.iter().all(|e| e.canonical_id.unwrap() == id),
            "All 'Alice' entities should be in the same cluster"
        );
    }

    #[test]
    fn max_pronoun_lookback_respected() {
        // With max_pronoun_lookback=1, the pronoun search window is small (1*10 = 10 entities).
        // Place a pronoun far after many intervening entities to test boundary.
        let mut entities = Vec::new();
        entities.push(person("Alice", 0, 5));
        // Insert enough non-pronoun entities to exceed the window.
        for i in 1..=20 {
            entities.push(org(&format!("Corp{}", i), i * 10, i * 10 + 5));
        }
        entities.push(person("she", 300, 303));

        let resolver = SimpleCorefResolver::new(CorefConfig {
            max_pronoun_lookback: 1,
            ..CorefConfig::default()
        });
        let resolved = resolver.resolve(&entities);

        let alice_id = resolved[0].canonical_id.unwrap();
        let she_id = resolved.last().unwrap().canonical_id.unwrap();
        // "she" is beyond the pronoun distance window (1 * 10 = 10 entities back),
        // but it may still find a compatible entity within those 10. The key invariant
        // is that Alice (entity 0) is NOT within the window when there are 20 intervening.
        assert_ne!(
            alice_id, she_id,
            "Pronoun too far from antecedent should stay unresolved or resolve to nearer entity"
        );
    }

    #[test]
    fn fuzzy_matching_disabled_prevents_substring() {
        let resolver = SimpleCorefResolver::new(CorefConfig {
            fuzzy_matching: false,
            proper_containment: false,
            strict_head_match: false,
            proper_head_word_match: false,
            relaxed_head_match: false,
            ..CorefConfig::default()
        });
        let entities = vec![person("John Smith", 0, 10), person("Smith", 20, 25)];
        let resolved = resolver.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id.unwrap(),
            resolved[1].canonical_id.unwrap(),
            "fuzzy_matching=false should prevent substring matching"
        );
    }

    // ---- Task #5 regression tests ----

    #[test]
    fn test_alice_bob_she_resolves_to_alice() {
        // With gazetteer: Alice -> Feminine, Bob -> Masculine.
        // "she" (Feminine) should skip Bob and resolve to Alice.
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("Alice", 0, 5),
            person("Bob", 10, 13),
            person("she", 20, 23),
        ];
        let resolved = resolver.resolve(&entities);
        let she_id = resolved[2].canonical_id.unwrap();
        let alice_id = resolved[0].canonical_id.unwrap();
        assert_eq!(
            she_id, alice_id,
            "With gazetteer, 'she' should resolve to 'Alice' (Feminine), not 'Bob' (Masculine)"
        );
    }

    #[test]
    fn test_gazetteer_disabled_falls_back_to_nearest() {
        let resolver = SimpleCorefResolver::new(CorefConfig {
            use_name_gazetteer: false,
            ..CorefConfig::default()
        });
        let entities = vec![
            person("Alice", 0, 5),
            person("Bob", 10, 13),
            person("she", 20, 23),
        ];
        let resolved = resolver.resolve(&entities);
        let she_id = resolved[2].canonical_id.unwrap();
        let bob_id = resolved[1].canonical_id.unwrap();
        assert_eq!(
            she_id, bob_id,
            "Without gazetteer, 'she' falls back to nearest compatible entity 'Bob'"
        );
    }

    // ---- Property tests ----

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_entity_type() -> impl Strategy<Value = EntityType> {
            prop_oneof![
                Just(EntityType::Person),
                Just(EntityType::Organization),
                Just(EntityType::Location),
            ]
        }

        fn arb_entity() -> impl Strategy<Value = Entity> {
            (
                "[a-zA-Z ]{1,20}", // text
                arb_entity_type(),
                0..1000usize, // start
            )
                .prop_map(|(text, entity_type, start)| {
                    let end = start + text.len();
                    Entity::new(text, entity_type, start, end, 0.9)
                })
        }

        fn arb_entity_vec() -> impl Strategy<Value = Vec<Entity>> {
            proptest::collection::vec(arb_entity(), 0..30)
        }

        proptest! {
            #[test]
            fn resolve_never_panics(entities in arb_entity_vec()) {
                let resolver = SimpleCorefResolver::default();
                let _ = resolver.resolve(&entities);
            }

            #[test]
            fn every_entity_has_canonical_id(entities in arb_entity_vec()) {
                let resolver = SimpleCorefResolver::default();
                let resolved = resolver.resolve(&entities);
                for (i, entity) in resolved.iter().enumerate() {
                    prop_assert!(
                        entity.canonical_id.is_some(),
                        "Entity {} ({:?}) missing canonical_id after resolve",
                        i, entity.text
                    );
                }
            }

            #[test]
            fn canonical_id_reflexive(entities in arb_entity_vec()) {
                // Every entity's canonical_id should point to a cluster that contains
                // at least the entity itself (i.e., some entity in the output shares
                // the same canonical_id).
                let resolver = SimpleCorefResolver::default();
                let resolved = resolver.resolve(&entities);
                for entity in &resolved {
                    if let Some(cid) = entity.canonical_id {
                        let count = resolved.iter()
                            .filter(|e| e.canonical_id == Some(cid))
                            .count();
                        prop_assert!(
                            count >= 1,
                            "canonical_id {:?} has no members in output",
                            cid
                        );
                    }
                }
            }
        }
    }
}
