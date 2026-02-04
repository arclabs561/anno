//! Simple coreference resolution for analysis/evaluation pipelines.
//!
//! Provides minimal resolvers to produce coreference chains from entities, completing the loop
//! between extraction and coreference metric computation.

use super::coref::CorefChain;
use crate::backends::box_embeddings::{BoxCorefConfig, BoxEmbedding};
use crate::{Entity, EntityType};
use anno_core::CanonicalId;
use std::collections::HashMap;

/// Configuration for simple coreference resolver.
#[derive(Debug, Clone)]
pub struct CorefConfig {
    /// Similarity threshold for name matching (0.0-1.0).
    pub similarity_threshold: f64,
    /// Maximum sentence distance for pronoun resolution.
    pub max_pronoun_distance: usize,
    /// Enable fuzzy name matching (e.g., "John Smith" ~ "J. Smith").
    pub fuzzy_matching: bool,
    /// Include singletons in output chains.
    pub include_singletons: bool,
}

impl Default for CorefConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            max_pronoun_distance: 3,
            fuzzy_matching: true,
            include_singletons: true,
        }
    }
}

/// Simple rule-based coreference resolver.
#[derive(Debug, Clone)]
pub struct SimpleCorefResolver {
    config: CorefConfig,
}

impl Default for SimpleCorefResolver {
    fn default() -> Self {
        Self::new(CorefConfig::default())
    }
}

impl SimpleCorefResolver {
    /// Create a new resolver with configuration.
    #[must_use]
    pub fn new(config: CorefConfig) -> Self {
        Self { config }
    }

    /// Resolve coreference for entities, assigning canonical IDs.
    #[must_use]
    pub fn resolve_entities(&self, entities: &[Entity]) -> Vec<Entity> {
        if entities.is_empty() {
            return vec![];
        }

        let mut resolved = entities.to_vec();
        let mut next_cluster_id = CanonicalId::ZERO;

        // Map from canonical form to cluster ID.
        let mut canonical_to_cluster: HashMap<String, CanonicalId> = HashMap::new();

        // Process entities in order.
        for i in 0..resolved.len() {
            let entity = &resolved[i];

            // Skip if already assigned.
            if entity.canonical_id.is_some() {
                continue;
            }

            let cluster_id =
                self.find_matching_cluster(entity, &resolved[..i], &canonical_to_cluster);

            let cluster_id = cluster_id.unwrap_or_else(|| {
                let id = next_cluster_id;
                next_cluster_id += 1;
                id
            });

            resolved[i].canonical_id = Some(cluster_id);

            let canonical = self.canonical_form(&resolved[i].text, &resolved[i].entity_type);
            canonical_to_cluster.insert(canonical, cluster_id);
        }

        resolved
    }

    /// Resolve coreference for entities.
    ///
    /// This is an inherent-method convenience wrapper so callers do not need to import the
    /// [`CoreferenceResolver`] trait to call `resolver.resolve(&entities)`.
    #[must_use]
    pub fn resolve(&self, entities: &[Entity]) -> Vec<Entity> {
        self.resolve_entities(entities)
    }

    /// Convert resolved entities directly to coreference chains.
    #[must_use]
    pub fn resolve_to_chains(&self, entities: &[Entity]) -> Vec<CorefChain> {
        let resolved = self.resolve_entities(entities);
        super::coref::entities_to_chains(&resolved)
    }

    fn find_matching_cluster(
        &self,
        entity: &Entity,
        previous: &[Entity],
        canonical_map: &HashMap<String, CanonicalId>,
    ) -> Option<CanonicalId> {
        // Strategy 1: pronoun resolution.
        if self.is_pronoun(&entity.text) {
            return self.resolve_pronoun(entity, previous);
        }

        // Strategy 2: exact canonical match.
        let canonical = self.canonical_form(&entity.text, &entity.entity_type);
        if let Some(&cluster_id) = canonical_map.get(&canonical) {
            return Some(cluster_id);
        }

        // Strategy 3: substring/fuzzy matching.
        if self.config.fuzzy_matching {
            for (other_canonical, &cluster_id) in canonical_map {
                if self.names_match(&canonical, other_canonical) {
                    return Some(cluster_id);
                }
            }
        }

        None
    }

    fn resolve_pronoun(&self, pronoun: &Entity, previous: &[Entity]) -> Option<CanonicalId> {
        let pronoun_gender = self.infer_gender(&pronoun.text);

        for entity in previous
            .iter()
            .rev()
            .take(self.config.max_pronoun_distance * 10)
        {
            if self.is_pronoun(&entity.text) {
                continue;
            }
            if !self.pronoun_compatible(&pronoun.text, &entity.entity_type) {
                continue;
            }

            let entity_gender = self.infer_gender(&entity.text);
            match (pronoun_gender, entity_gender) {
                (Some('n'), _) => {}
                (_, Some('n')) => {}
                (Some(pg), Some(eg)) => {
                    if pg != eg {
                        continue;
                    }
                }
                (_, None) => {}
                (None, Some(_)) => {}
            }

            return entity.canonical_id;
        }

        None
    }

    fn is_pronoun(&self, text: &str) -> bool {
        matches!(
            text.to_lowercase().as_str(),
            // Traditional gendered pronouns
            "he" | "she" | "him" | "her" | "his" | "hers" | "himself" | "herself" |
            // Singular they
            "they" | "them" | "their" | "theirs" | "themselves" | "themself" |
            // Impersonal pronouns
            "it" | "its" | "itself" |
            // Neopronouns
            "xe" | "xem" | "xyr" | "xyrs" | "xemself" |
            "ze" | "hir" | "zir" | "hirs" | "zirs" | "hirself" | "zirself" |
            "ey" | "em" | "eir" | "eirs" | "emself" |
            "fae" | "faer" | "faers" | "faeself"
        )
    }

    fn pronoun_compatible(&self, pronoun: &str, entity_type: &EntityType) -> bool {
        let lower = pronoun.to_lowercase();
        match entity_type {
            EntityType::Person => matches!(
                lower.as_str(),
                "he" | "she" | "they" | "him" | "her" | "them" |
                "his" | "hers" | "their" | "theirs" |
                "himself" | "herself" | "themselves" | "themself" |
                "xe" | "xem" | "xyr" | "xyrs" | "xemself" |
                "ze" | "hir" | "zir" | "hirs" | "zirs" | "hirself" | "zirself" |
                "ey" | "em" | "eir" | "eirs" | "emself" |
                "fae" | "faer" | "faers" | "faeself"
            ),
            EntityType::Organization => matches!(
                lower.as_str(),
                "it" | "they" | "its" | "their" | "theirs" | "itself" | "themselves"
            ),
            EntityType::Location => matches!(lower.as_str(), "it" | "its" | "itself"),
            _ => matches!(lower.as_str(), "it" | "its" | "itself"),
        }
    }

    fn infer_gender(&self, text: &str) -> Option<char> {
        let lower = text.to_lowercase();
        match lower.as_str() {
            "he" | "him" | "his" | "himself" => Some('m'),
            "she" | "her" | "hers" | "herself" => Some('f'),
            "they" | "them" | "their" | "theirs" | "themselves" | "themself" => Some('n'),
            "xe" | "xem" | "xyr" | "xyrs" | "xemself" => Some('n'),
            "ze" | "hir" | "zir" | "hirs" | "zirs" | "hirself" | "zirself" => Some('n'),
            "ey" | "em" | "eir" | "eirs" | "emself" => Some('n'),
            "fae" | "faer" | "faers" | "faeself" => Some('n'),
            _ => None,
        }
    }

    fn canonical_form(&self, text: &str, entity_type: &EntityType) -> String {
        let normalized = text.to_lowercase().trim().to_string();
        format!("{}:{}", entity_type.as_label(), normalized)
    }

    fn names_match(&self, name1: &str, name2: &str) -> bool {
        let (type1, text1) = name1.split_once(':').unwrap_or(("", name1));
        let (type2, text2) = name2.split_once(':').unwrap_or(("", name2));
        if type1 != type2 {
            return false;
        }

        if text1 == text2 {
            return true;
        }

        if text1.contains(text2) || text2.contains(text1) {
            return true;
        }

        let words1: Vec<&str> = text1.split_whitespace().collect();
        let words2: Vec<&str> = text2.split_whitespace().collect();

        if words1.len() > 1 && words2.len() == 1 && words1.last() == words2.first() {
            return true;
        }
        if words2.len() > 1 && words1.len() == 1 && words2.last() == words1.first() {
            return true;
        }

        false
    }
}

// Re-export the canonical trait from `anno-core`.
pub use anno_core::CoreferenceResolver;

impl CoreferenceResolver for SimpleCorefResolver {
    fn resolve(&self, entities: &[Entity]) -> Vec<Entity> {
        self.resolve_entities(entities)
    }

    fn name(&self) -> &'static str {
        "simple-rule-based"
    }
}

/// Box-based coreference resolver.
pub struct BoxCorefResolver {
    config: BoxCorefConfig,
}

impl BoxCorefResolver {
    /// Create a new box-based coreference resolver.
    #[must_use]
    pub fn new(config: BoxCorefConfig) -> Self {
        Self { config }
    }

    /// Resolve coreference using provided box embeddings.
    ///
    /// # Panics
    /// Panics if `boxes.len() != entities.len()`.
    #[must_use]
    pub fn resolve_with_boxes(&self, entities: &[Entity], boxes: &[BoxEmbedding]) -> Vec<Entity> {
        assert_eq!(
            entities.len(),
            boxes.len(),
            "entities and boxes must have same length"
        );

        if entities.is_empty() {
            return vec![];
        }

        let mut resolved = entities.to_vec();
        let mut next_cluster_id = CanonicalId::ZERO;

        // Union-find clustering.
        let mut parent: Vec<usize> = (0..entities.len()).collect();

        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut [usize], i: usize, j: usize) {
            let pi = find(parent, i);
            let pj = find(parent, j);
            if pi != pj {
                parent[pi] = pj;
            }
        }

        // Pairwise scoring.
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let score = boxes[i].coreference_score(&boxes[j]);

                if score >= self.config.coreference_threshold {
                    if entities[i].entity_type == entities[j].entity_type {
                        if !self.config.enforce_syntactic_constraints
                            || self.check_syntactic_constraints(&entities[i], &entities[j])
                        {
                            union(&mut parent, i, j);
                        }
                    }
                }
            }
        }

        // Assign cluster IDs.
        let mut cluster_map: HashMap<usize, CanonicalId> = HashMap::new();
        for (i, resolved_entity) in resolved.iter_mut().enumerate().take(entities.len()) {
            let root = find(&mut parent, i);
            let cluster_id = *cluster_map.entry(root).or_insert_with(|| {
                let id = next_cluster_id;
                next_cluster_id += 1;
                id
            });
            resolved_entity.canonical_id = Some(cluster_id);
        }

        resolved
    }

    fn check_syntactic_constraints(&self, entity_a: &Entity, entity_b: &Entity) -> bool {
        let distance = if entity_a.end <= entity_b.start {
            entity_b.start - entity_a.end
        } else {
            entity_a.start.saturating_sub(entity_b.end)
        };

        if self.is_simple_pronoun(&entity_a.text) && distance <= self.config.max_local_distance {
            return distance > 50;
        }

        if self.is_rexpression(&entity_a.text) && distance <= self.config.max_local_distance {
            return distance > 20;
        }

        true
    }

    fn is_simple_pronoun(&self, text: &str) -> bool {
        matches!(
            text.to_lowercase().as_str(),
            "he" | "she" | "they" | "him" | "her" | "them" | "it" | "this" | "that"
        )
    }

    fn is_rexpression(&self, text: &str) -> bool {
        text.chars().next().is_some_and(|c| c.is_uppercase()) && text.len() > 1
    }
}

impl CoreferenceResolver for BoxCorefResolver {
    fn resolve(&self, entities: &[Entity]) -> Vec<Entity> {
        // Boxes are provided out-of-band; default is identity.
        entities.to_vec()
    }

    fn name(&self) -> &'static str {
        "box-embedding"
    }
}

/// Convert vector embeddings to box embeddings for coreference resolution.
#[must_use]
pub fn vectors_to_boxes(embeddings: &[f32], hidden_dim: usize, radius: Option<f32>) -> Vec<BoxEmbedding> {
    let radius = radius.unwrap_or(0.1);
    let num_entities = embeddings.len() / hidden_dim;
    let mut boxes = Vec::with_capacity(num_entities);

    for i in 0..num_entities {
        let start = i * hidden_dim;
        let end = start + hidden_dim;
        let vector = &embeddings[start..end];
        boxes.push(BoxEmbedding::from_vector(vector, radius));
    }

    boxes
}

/// Resolve coreference using box embeddings derived from vector embeddings.
#[must_use]
pub fn resolve_with_box_embeddings(
    entities: &[Entity],
    embeddings: &[f32],
    hidden_dim: usize,
    config: BoxCorefConfig,
) -> Vec<Entity> {
    let radius = config.vector_to_box_radius;
    let boxes = vectors_to_boxes(embeddings, hidden_dim, radius);
    let resolver = BoxCorefResolver::new(config);
    resolver.resolve_with_boxes(entities, &boxes)
}

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
            "ed ", " was ", " were ", " had ", " did ", " happened", " occurred",
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
        if let Some(event) = self.find_event_near(anaphor.start, 200) {
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

        let candidates = self.scope.candidate_antecedent_spans(anaphor.start);
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

