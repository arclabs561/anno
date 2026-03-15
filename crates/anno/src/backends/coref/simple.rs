//! Simple coreference resolution backends.
//!
//! Provides rule-based resolvers to produce coreference chains from entities,
//! completing the loop between extraction and coreference metric computation.
//!
//! # Sieve pipeline
//!
//! [`SimpleCorefResolver`] applies nine sieves in precision-ranked order, returning the first match:
//!
//! 1. **Pronoun resolution** -- links pronouns to a compatible antecedent within a lookback window,
//!    using gender inference from a name gazetteer and pronoun-entity-type compatibility rules.
//! 2. **Exact canonical match** -- matches entities with identical `type:lowercased_text` keys.
//! 3. **Precise constructs** -- merges entities in appositive constructions, detected via span
//!    proximity (gap <= 2 chars for ", ") and compatible entity types.
//! 4. **Acronym match** -- links single-token acronyms to multi-word names whose initials match
//!    (e.g., "IBM" ~ "International Business Machines"), same entity type required.
//! 5. **Strict head match** -- links mentions sharing a head word (last word) with compatible
//!    entity type and gender, guarded by an i-within-i constraint (nested spans in the same
//!    sentence are excluded).
//! 6. **Proper head word match** -- for named entities, links mentions where the head word of one
//!    appears as a word in the other (e.g., "President Obama" ~ "Barack Hussein Obama").
//! 7. **Relaxed head match** -- links multi-word mentions sharing a head word (last token) and
//!    entity type (e.g., "President Obama" ~ "Barack Obama").
//! 8. **Proper containment** -- links mentions where one is a contiguous word subsequence of the
//!    other in original text, same entity type required (e.g., "Obama" in "Barack Obama").
//! 9. **Substring/fuzzy match** -- links mentions sharing a substring (>= 3 chars) or a matching
//!    last word, same entity type required.

use crate::{Entity, EntityType};
use anno_core::{CanonicalId, CoreferenceResolver, Gender};
use std::collections::HashMap;

/// Configuration for [`SimpleCorefResolver`].
///
/// All boolean sieve toggles default to `true`. Disable individual sieves to measure their
/// contribution or to reduce false merges on a particular dataset.
#[derive(Debug, Clone)]
pub struct CorefConfig {
    /// Maximum number of preceding entities to search when resolving a pronoun (sieve 1).
    ///
    /// The actual lookback window is `max_pronoun_lookback * 10` entities,
    /// a rough approximation of sentence distance by entity count.
    ///
    /// Default: `3` (i.e., 30 entities).
    pub max_pronoun_lookback: usize,

    /// Enable substring/fuzzy name matching (sieve 6).
    ///
    /// When enabled, two mentions match if one is a substring of the other (>= 3 chars)
    /// or if they share a last word and entity type.
    ///
    /// Default: `true`.
    pub fuzzy_matching: bool,

    /// Include singletons (entities with no coreferent) in output chains.
    ///
    /// Default: `true`.
    pub include_singletons: bool,

    /// Use a built-in name-to-gender gazetteer for pronoun resolution (sieve 1).
    ///
    /// When enabled, common English names (e.g., "Alice" -> Feminine,
    /// "Bob" -> Masculine) are used to infer gender for proper nouns,
    /// improving pronoun resolution accuracy. Disable for non-English
    /// text or when names don't follow English gender conventions.
    ///
    /// Default: `true`.
    pub use_name_gazetteer: bool,

    /// Enable acronym matching (sieve 3).
    ///
    /// Checks if one mention is a single-token acronym whose letters match the
    /// initial letters of the other mention's words. Requires same entity type.
    /// Example: "IBM" matches "International Business Machines".
    ///
    /// Default: `true`.
    pub acronym_matching: bool,

    /// Enable relaxed head match (sieve 4).
    ///
    /// Two mentions corefer if their head words (last word) match and they share the same
    /// entity type. Only applies when both mentions have 2+ words.
    /// Example: "President Obama" ~ "Barack Obama" via shared head "Obama".
    ///
    /// Default: `true`.
    pub relaxed_head_match: bool,

    /// Enable proper noun containment (sieve 8).
    ///
    /// One mention's original text must be a proper subsequence of the other, matching
    /// at word boundaries, and both must share the same entity type. Unlike sieve 9
    /// (fuzzy/substring), this operates on original text (not canonical lowercased form)
    /// and requires complete word-boundary alignment.
    /// Example: "Obama" is contained in "Barack Obama".
    ///
    /// Default: `true`.
    pub proper_containment: bool,

    /// Enable precise constructs sieve (sieve 3).
    ///
    /// Detects appositive constructions where two entity mentions are immediately
    /// adjacent (gap <= 2 characters, allowing for ", ") with compatible entity types.
    /// E.g., "Barack Obama, the president" -- merges the two NPs.
    ///
    /// Default: `true`.
    pub precise_constructs: bool,

    /// Enable strict head matching (sieve 5).
    ///
    /// Two mentions match if they share the same head word (last word), have compatible
    /// entity types and gender, and do not violate the i-within-i constraint (one mention's
    /// span must not contain the other's span).
    ///
    /// Default: `true`.
    pub strict_head_match: bool,

    /// Enable proper head word matching (sieve 6).
    ///
    /// For named entity mentions: the head word (last word) of one mention appears as
    /// any word in the other mention, with same entity type required.
    /// E.g., "President Obama" matches "Barack Hussein Obama" (head "Obama" found in both).
    ///
    /// Default: `true`.
    pub proper_head_word_match: bool,
}

impl Default for CorefConfig {
    fn default() -> Self {
        Self {
            max_pronoun_lookback: 3,
            fuzzy_matching: true,
            include_singletons: true,
            use_name_gazetteer: true,
            acronym_matching: true,
            relaxed_head_match: true,
            proper_containment: true,
            precise_constructs: true,
            strict_head_match: true,
            proper_head_word_match: true,
        }
    }
}

/// Simple rule-based coreference resolver.
///
/// Applies the nine-sieve pipeline described in the [module documentation](self) to assign
/// [`CanonicalId`]s to a sequence of [`Entity`] values. Entities that corefer receive the
/// same id; singletons receive a unique id.
///
/// # Example
///
/// ```rust,no_run
/// use anno::Entity;
/// use anno::EntityType;
/// use anno::backends::coref::simple::{SimpleCorefResolver, CorefConfig};
///
/// let resolver = SimpleCorefResolver::new(CorefConfig::default());
///
/// let entities = vec![
///     Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
///     Entity::new("he", EntityType::Person, 15, 17, 0.8),
///     Entity::new("John Smith", EntityType::Person, 30, 40, 0.9),
/// ];
///
/// let resolved = resolver.resolve(&entities);
/// // All three mentions share the same canonical_id.
/// assert_eq!(resolved[0].canonical_id, resolved[1].canonical_id);
/// assert_eq!(resolved[0].canonical_id, resolved[2].canonical_id);
/// ```
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
    pub fn resolve_to_chains(&self, entities: &[Entity]) -> Vec<crate::eval::coref::CorefChain> {
        let resolved = self.resolve_entities(entities);
        crate::eval::coref::entities_to_chains(&resolved)
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

        // Strategy 3: precise constructs (appositives via span proximity).
        if self.config.precise_constructs {
            for prev in previous.iter().rev() {
                if let Some(cluster_id) = prev.canonical_id {
                    if self.is_precise_construct(entity, prev) {
                        return Some(cluster_id);
                    }
                }
            }
        }

        // Strategy 4: acronym matching (e.g., "IBM" ~ "International Business Machines").
        if self.config.acronym_matching {
            for (other_canonical, &cluster_id) in canonical_map {
                if self.is_acronym_match(&canonical, other_canonical) {
                    return Some(cluster_id);
                }
            }
        }

        // Strategy 5: strict head match (head word + type + gender + i-within-i guard).
        if self.config.strict_head_match {
            for prev in previous.iter().rev() {
                if let Some(cluster_id) = prev.canonical_id {
                    if self.is_strict_head_match(entity, prev) {
                        return Some(cluster_id);
                    }
                }
            }
        }

        // Guard: For wildcard entity types (Other/Custom), skip fuzzy sieves 6-9.
        // These catch-all types defeat the entity-type guard in the matching functions,
        // causing spurious merges (e.g., "Nobel" clustered with "Emmanuelle" because
        // both are type "proper" and fuzzy containment matches on short substrings).
        let is_wildcard_type = matches!(entity.entity_type, EntityType::Custom { .. });

        if !is_wildcard_type {
            // Strategy 6: proper head word match (head word of one found in the other).
            if self.config.proper_head_word_match {
                for prev in previous.iter().rev() {
                    if let Some(cluster_id) = prev.canonical_id {
                        if self.is_proper_head_word_match(entity, prev) {
                            return Some(cluster_id);
                        }
                    }
                }
            }

            // Strategy 7: relaxed head match (e.g., "President Obama" ~ "Barack Obama").
            if self.config.relaxed_head_match {
                for prev in previous.iter().rev() {
                    if let Some(cluster_id) = prev.canonical_id {
                        if self.is_relaxed_head_match(entity, prev) {
                            return Some(cluster_id);
                        }
                    }
                }
            }

            // Strategy 8: proper noun containment (e.g., "Obama" in "Barack Obama").
            if self.config.proper_containment {
                for prev in previous.iter().rev() {
                    if let Some(cluster_id) = prev.canonical_id {
                        if self.is_proper_containment(entity, prev) {
                            return Some(cluster_id);
                        }
                    }
                }
            }

            // Strategy 9: substring/fuzzy matching.
            if self.config.fuzzy_matching {
                for (other_canonical, &cluster_id) in canonical_map {
                    if self.names_match(&canonical, other_canonical) {
                        return Some(cluster_id);
                    }
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
            .take(self.config.max_pronoun_lookback * 10)
        {
            if self.is_pronoun(&entity.text) {
                continue;
            }
            if !self.pronoun_compatible(&pronoun.text, &entity.entity_type) {
                continue;
            }

            let entity_gender = self.infer_gender(&entity.text);
            if let (Some(pg), Some(eg)) = (pronoun_gender, entity_gender) {
                if !pg.is_compatible(&eg) {
                    continue;
                }
            }

            return entity.canonical_id;
        }

        None
    }

    /// # Multilingual Note
    ///
    /// English-only. This resolver is used in the eval pipeline where English
    /// datasets (OntoNotes, PreCo, LitBank) dominate. For multilingual eval,
    /// use model-based mention detection or extend with language-specific lists.
    pub(crate) fn is_pronoun(&self, text: &str) -> bool {
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
            "fae" | "faer" | "faers" | "faeself" | "faerself"
        )
    }

    /// # Multilingual Note
    ///
    /// English-only. Pronoun-entity type compatibility rules are language-specific
    /// (e.g., French gendered articles, German grammatical gender on nouns).
    fn pronoun_compatible(&self, pronoun: &str, entity_type: &EntityType) -> bool {
        let lower = pronoun.to_lowercase();
        match entity_type {
            EntityType::Person => matches!(
                lower.as_str(),
                "he" | "she"
                    | "they"
                    | "him"
                    | "her"
                    | "them"
                    | "his"
                    | "hers"
                    | "their"
                    | "theirs"
                    | "himself"
                    | "herself"
                    | "themselves"
                    | "themself"
                    // it/its as personal pronouns (used by some genderqueer individuals)
                    | "it"
                    | "its"
                    | "itself"
                    | "xe"
                    | "xem"
                    | "xyr"
                    | "xyrs"
                    | "xemself"
                    | "ze"
                    | "hir"
                    | "zir"
                    | "hirs"
                    | "zirs"
                    | "hirself"
                    | "zirself"
                    | "ey"
                    | "em"
                    | "eir"
                    | "eirs"
                    | "emself"
                    | "fae"
                    | "faer"
                    | "faers"
                    | "faeself"
            ),
            EntityType::Organization => matches!(
                lower.as_str(),
                "it" | "they" | "its" | "their" | "theirs" | "itself" | "themselves"
            ),
            EntityType::Location => matches!(lower.as_str(), "it" | "its" | "itself"),
            _ => matches!(lower.as_str(), "it" | "its" | "itself"),
        }
    }

    /// # Multilingual Note
    ///
    /// English-only. Delegates to `Gender::from_pronoun()` which covers English
    /// pronouns and neopronouns. Falls back to a name gazetteer (if enabled)
    /// for common English proper nouns. Other languages need language-specific
    /// gender inference (grammatical gender, morphological agreement, etc.).
    fn infer_gender(&self, text: &str) -> Option<Gender> {
        if let Some(g) = Gender::from_pronoun(text) {
            return Some(g);
        }
        if self.config.use_name_gazetteer {
            return gender_from_name(text);
        }
        None
    }

    fn canonical_form(&self, text: &str, entity_type: &EntityType) -> String {
        let normalized = text.to_lowercase().trim().to_string();
        format!("{}:{}", entity_type.as_label(), normalized)
    }

    /// Check if one mention is an acronym of the other.
    ///
    /// "IBM" matches "International Business Machines" because the first letter of
    /// each word in the full name matches the acronym characters. Both mentions must
    /// share the same entity type (enforced by the canonical form prefix).
    fn is_acronym_match(&self, name1: &str, name2: &str) -> bool {
        let (type1, text1) = name1.split_once(':').unwrap_or(("", name1));
        let (type2, text2) = name2.split_once(':').unwrap_or(("", name2));
        if type1 != type2 {
            return false;
        }

        // Canonical form lowercases text, so we can't check for uppercase.
        // Instead: one side must be a single "word" (no spaces) with length > 1,
        // the other must be multi-word with word count == single-word char count.
        let words1: Vec<&str> = text1.split_whitespace().collect();
        let words2: Vec<&str> = text2.split_whitespace().collect();

        let (acronym, words) = if words1.len() == 1 && words2.len() > 1 {
            (text1, &words2)
        } else if words2.len() == 1 && words1.len() > 1 {
            (text2, &words1)
        } else {
            return false;
        };

        let acronym_chars: Vec<char> = acronym.chars().collect();
        // Acronym must be at least 2 chars and match word count.
        if acronym_chars.len() < 2 || acronym_chars.len() != words.len() {
            return false;
        }

        // Each acronym letter must match the first letter of the corresponding word.
        acronym_chars
            .iter()
            .zip(words.iter())
            .all(|(&ac, word)| word.starts_with(ac))
    }

    /// Detect appositive constructions via span proximity.
    ///
    /// Two mentions form an appositive when one immediately follows the other
    /// (gap <= 2 characters, allowing for ", " between them) and they share the
    /// same entity type. This is a high-precision signal: "Barack Obama, the president"
    /// produces entities "Barack Obama" [0,12) and "the president" [14,27), with a
    /// gap of 2 characters for ", ".
    ///
    /// Guards:
    /// - Same entity type required.
    /// - Neither mention is a pronoun.
    /// - Gap must be 0-2 characters (covers adjacency, comma, comma+space).
    fn is_precise_construct(&self, a: &Entity, b: &Entity) -> bool {
        if a.entity_type != b.entity_type {
            return false;
        }
        if self.is_pronoun(&a.text) || self.is_pronoun(&b.text) {
            return false;
        }
        // Order-independent gap check: one must immediately follow the other.
        let gap = if a.start() >= b.end() {
            a.start() - b.end()
        } else if b.start() >= a.end() {
            b.start() - a.end()
        } else {
            // Overlapping spans -- not an appositive.
            return false;
        };
        gap <= 2
    }

    /// Strict head matching: head word + entity type + gender compatibility + i-within-i guard.
    ///
    /// Two mentions match if:
    /// 1. Same entity type.
    /// 2. Same head word (last word, case-insensitive).
    /// 3. Compatible gender (inferred from mention text via gazetteer/pronoun rules).
    /// 4. No i-within-i violation: one mention's span must not contain the other's span.
    ///    Nested mentions in the same clause are likely distinct referents
    ///    (e.g., "the president of [the company]" -- "the company" is not the president).
    ///
    /// Unlike `is_relaxed_head_match`, this sieve:
    /// - Applies to single-word mentions too (head = the word itself).
    /// - Requires gender compatibility.
    /// - Enforces i-within-i.
    fn is_strict_head_match(&self, a: &Entity, b: &Entity) -> bool {
        if a.entity_type != b.entity_type {
            return false;
        }
        // I-within-i: reject if one span is nested inside the other.
        if (a.start() >= b.start() && a.end() <= b.end())
            || (b.start() >= a.start() && b.end() <= a.end())
        {
            return false;
        }
        let head_a = Self::head_word(&a.text);
        let head_b = Self::head_word(&b.text);
        if !head_a.eq_ignore_ascii_case(head_b) {
            return false;
        }
        // Gender compatibility check.
        let gender_a = self.infer_gender(&a.text);
        let gender_b = self.infer_gender(&b.text);
        match (gender_a, gender_b) {
            (Some(ga), Some(gb)) => ga.is_compatible(&gb),
            _ => true, // unknown gender is permissive
        }
    }

    /// Proper head word match for named entities.
    ///
    /// The head word (last word) of one mention appears as any word in the other mention,
    /// with same entity type required. Both mentions must have 2+ words to avoid
    /// degenerate single-word matches (those are handled by exact/containment sieves).
    ///
    /// Example: "President Obama" (head "Obama") matches "Barack Hussein Obama"
    /// because "Obama" appears as a word in the other mention.
    fn is_proper_head_word_match(&self, a: &Entity, b: &Entity) -> bool {
        if a.entity_type != b.entity_type {
            return false;
        }
        let words_a: Vec<&str> = a.text.split_whitespace().collect();
        let words_b: Vec<&str> = b.text.split_whitespace().collect();
        // Both must be multi-word to avoid trivial matches.
        if words_a.len() < 2 || words_b.len() < 2 {
            return false;
        }
        let head_a = Self::head_word(&a.text);
        let head_b = Self::head_word(&b.text);
        // Head of A found in B's words, or head of B found in A's words.
        let head_a_in_b = words_b.iter().any(|w| w.eq_ignore_ascii_case(head_a));
        let head_b_in_a = words_a.iter().any(|w| w.eq_ignore_ascii_case(head_b));
        head_a_in_b || head_b_in_a
    }

    /// Extract the head word from a mention (last whitespace-delimited token).
    ///
    /// For NPs, the head is typically the rightmost noun. Without a full parse,
    /// the last word is a reasonable approximation for English named entities.
    fn head_word(text: &str) -> &str {
        text.split_whitespace().next_back().unwrap_or(text)
    }

    /// Check if two mentions share a head word (last word) and entity type.
    ///
    /// Only applies when both mentions have 2+ words, so single-word mentions
    /// are not spuriously matched (those are handled by exact/fuzzy sieves).
    fn is_relaxed_head_match(&self, a: &Entity, b: &Entity) -> bool {
        if a.entity_type != b.entity_type {
            return false;
        }
        let words_a: Vec<&str> = a.text.split_whitespace().collect();
        let words_b: Vec<&str> = b.text.split_whitespace().collect();
        if words_a.len() < 2 || words_b.len() < 2 {
            return false;
        }
        // Head word = last word, compared case-insensitively.
        words_a
            .last()
            .unwrap()
            .eq_ignore_ascii_case(words_b.last().unwrap())
    }

    /// Check if one mention's original text is properly contained in the other
    /// at word boundaries, with matching entity type.
    ///
    /// "Obama" is properly contained in "Barack Obama" because "Obama" appears
    /// as a complete word. Unlike `names_match`, this uses the original text
    /// (not canonical lowercased form) and requires word-boundary alignment.
    fn is_proper_containment(&self, a: &Entity, b: &Entity) -> bool {
        if a.entity_type != b.entity_type {
            return false;
        }
        let text_a = a.text.trim();
        let text_b = b.text.trim();
        if text_a.is_empty() || text_b.is_empty() || text_a.eq_ignore_ascii_case(text_b) {
            return false; // must be a *proper* subsequence, not equal
        }

        let (shorter, longer) = if text_a.len() < text_b.len() {
            (text_a, text_b)
        } else if text_b.len() < text_a.len() {
            (text_b, text_a)
        } else {
            return false; // same length but different text => not contained
        };

        // Check if shorter appears as a complete word sequence in longer.
        let longer_lower = longer.to_lowercase();
        let shorter_lower = shorter.to_lowercase();
        let longer_words: Vec<&str> = longer_lower.split_whitespace().collect();
        let shorter_words: Vec<&str> = shorter_lower.split_whitespace().collect();

        if shorter_words.is_empty() || shorter_words.len() >= longer_words.len() {
            return false;
        }

        // Check if shorter_words appears as a contiguous subsequence of longer_words.
        longer_words
            .windows(shorter_words.len())
            .any(|window| window == shorter_words.as_slice())
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

        // Substring containment -- guarded against false positives.
        // Require >= 5 chars AND a minimum length ratio to prevent spurious merges
        // (e.g., "CEO" matching "CEOVILLE", or 3-char substrings causing transitive chains).
        // For shorter substrings, require exact word-boundary match.
        let (shorter, longer) = if text1.len() <= text2.len() {
            (text1, text2)
        } else {
            (text2, text1)
        };
        if longer.contains(shorter) {
            let shorter_char_count = shorter.chars().count();
            let longer_char_count = longer.chars().count();
            let ratio = shorter_char_count as f64 / longer_char_count as f64;
            if shorter_char_count >= 5 && ratio > 0.3 {
                return true;
            }
            // For shorter strings, require exact word-boundary match.
            if longer.split_whitespace().any(|word| word == shorter) {
                return true;
            }
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

/// Infer gender from a proper noun using a small English name gazetteer.
///
/// Returns `None` for unrecognized names. Case-insensitive on the first word
/// of the input (handles "Alice Smith" by checking "alice").
pub(crate) fn gender_from_name(text: &str) -> Option<Gender> {
    let first_word = text.split_whitespace().next()?;
    let lower = first_word.to_lowercase();
    // ~100 common English given names, covering the most frequent names
    // in US/UK census data. This is intentionally small and conservative.
    match lower.as_str() {
        // Masculine
        "james" | "john" | "robert" | "michael" | "david" | "william" | "richard" | "joseph"
        | "thomas" | "charles" | "christopher" | "daniel" | "matthew" | "anthony" | "mark"
        | "donald" | "steven" | "paul" | "andrew" | "joshua" | "kenneth" | "kevin" | "brian"
        | "george" | "timothy" | "ronald" | "edward" | "jason" | "jeffrey" | "ryan" | "jacob"
        | "gary" | "nicholas" | "eric" | "jonathan" | "stephen" | "larry" | "justin" | "scott"
        | "brandon" | "benjamin" | "samuel" | "raymond" | "gregory" | "frank" | "alexander"
        | "patrick" | "jack" | "dennis" | "peter" | "bob" | "jim" | "tom" | "mike" | "bill"
        | "joe" | "dan" | "matt" | "steve" | "chris" | "nick" | "ben" | "sam" | "jake" | "adam"
        | "henry" | "nathan" | "philip" | "carl" | "ahmed" | "ahmad" | "mohammed" | "muhammad"
        | "omar" | "ali" | "hassan" | "hussein" | "khalid" | "ibrahim" => Some(Gender::Masculine),
        // Feminine
        "mary" | "patricia" | "jennifer" | "linda" | "barbara" | "elizabeth" | "susan"
        | "jessica" | "sarah" | "karen" | "lisa" | "nancy" | "betty" | "margaret" | "sandra"
        | "ashley" | "dorothy" | "kimberly" | "emily" | "donna" | "michelle" | "carol"
        | "amanda" | "melissa" | "deborah" | "stephanie" | "rebecca" | "sharon" | "laura"
        | "cynthia" | "kathleen" | "amy" | "angela" | "shirley" | "anna" | "brenda" | "pamela"
        | "emma" | "nicole" | "helen" | "samantha" | "katherine" | "christine" | "debra"
        | "rachel" | "carolyn" | "janet" | "catherine" | "maria" | "heather" | "diane" | "ruth"
        | "julie" | "olivia" | "joyce" | "virginia" | "victoria" | "kelly" | "lauren"
        | "christina" | "joan" | "evelyn" | "judith" | "alice" | "ann" | "anne" | "jane"
        | "jean" | "marie" | "rose" | "grace" | "fatima" | "aisha" | "maryam" | "nour"
        | "layla" | "hana" => Some(Gender::Feminine),
        _ => None,
    }
}

impl CoreferenceResolver for SimpleCorefResolver {
    fn resolve(&self, entities: &[Entity]) -> Vec<Entity> {
        self.resolve_entities(entities)
    }

    fn name(&self) -> &'static str {
        "simple-rule-based"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EntityCategory;

    fn resolver() -> SimpleCorefResolver {
        SimpleCorefResolver::new(CorefConfig::default())
    }

    // =========================================================================
    // Precise constructs sieve
    // =========================================================================

    #[test]
    fn precise_construct_appositive_adjacent() {
        // "Barack Obama, the president" -- two entities separated by ", " (gap=2).
        let r = resolver();
        let entities = vec![
            Entity::new("Barack Obama", EntityType::Person, 0, 12, 0.9),
            Entity::new("the president", EntityType::Person, 14, 27, 0.85),
        ];
        let resolved = r.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id, resolved[1].canonical_id,
            "appositive entities should corefer"
        );
    }

    #[test]
    fn precise_construct_rejects_distant_entities() {
        // Two Person entities far apart should not match via precise constructs.
        let entities = vec![
            Entity::new("Barack Obama", EntityType::Person, 0, 12, 0.9),
            Entity::new("the president", EntityType::Person, 50, 63, 0.85),
        ];
        // They may still match via other sieves, but not via precise constructs.
        // Disable other sieves to isolate.
        let cfg = CorefConfig {
            fuzzy_matching: false,
            relaxed_head_match: false,
            proper_containment: false,
            strict_head_match: false,
            proper_head_word_match: false,
            acronym_matching: false,
            ..Default::default()
        };
        let r = SimpleCorefResolver::new(cfg);
        let resolved = r.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id, resolved[1].canonical_id,
            "distant entities should not match via precise constructs"
        );
    }

    #[test]
    fn precise_construct_rejects_different_types() {
        // Adjacent but different entity types should not match.
        let a = Entity::new("Acme Corp", EntityType::Organization, 0, 9, 0.9);
        let b = Entity::new("New York", EntityType::Location, 11, 19, 0.85);
        assert!(
            !resolver().is_precise_construct(&a, &b),
            "different entity types should not match"
        );
    }

    // =========================================================================
    // Strict head matching sieve
    // =========================================================================

    #[test]
    fn strict_head_match_same_head_compatible_gender() {
        let r = resolver();
        // "John Smith" and "Robert Smith" -- same head "Smith", both Person.
        let a = Entity::new("John Smith", EntityType::Person, 0, 10, 0.9);
        let b = Entity::new("Robert Smith", EntityType::Person, 50, 62, 0.9);
        assert!(
            r.is_strict_head_match(&a, &b),
            "same head word + compatible gender should match"
        );
    }

    #[test]
    fn strict_head_match_i_within_i_rejection() {
        let r = resolver();
        // "the company" [15,26) is nested inside "the president of the company" [0,30).
        let outer = Entity::new(
            "the president of the company",
            EntityType::Person,
            0,
            30,
            0.9,
        );
        let inner = Entity::new("the company", EntityType::Person, 15, 26, 0.9);
        assert!(
            !r.is_strict_head_match(&outer, &inner),
            "nested spans (i-within-i) should be rejected"
        );
    }

    #[test]
    fn strict_head_match_gender_incompatible() {
        let r = resolver();
        // "John Smith" (masculine via gazetteer) vs "Mary Smith" (feminine).
        // Same head "Smith" but incompatible gender.
        let a = Entity::new("John Smith", EntityType::Person, 0, 10, 0.9);
        let b = Entity::new("Mary Smith", EntityType::Person, 50, 60, 0.9);
        assert!(
            !r.is_strict_head_match(&a, &b),
            "gender-incompatible mentions should not match"
        );
    }

    #[test]
    fn strict_head_match_single_word() {
        let r = resolver();
        // Single-word mentions with the same text should match (head = the word itself).
        let a = Entity::new("Obama", EntityType::Person, 0, 5, 0.9);
        let b = Entity::new("Obama", EntityType::Person, 50, 55, 0.9);
        assert!(
            r.is_strict_head_match(&a, &b),
            "identical single-word mentions should match via strict head"
        );
    }

    // =========================================================================
    // Proper head word match sieve
    // =========================================================================

    #[test]
    fn proper_head_word_match_cross_reference() {
        let r = resolver();
        // "President Obama" (head "Obama") and "Barack Obama" (head "Obama").
        // Head of each is found in the other.
        let a = Entity::new("President Obama", EntityType::Person, 0, 15, 0.9);
        let b = Entity::new("Barack Obama", EntityType::Person, 50, 62, 0.9);
        assert!(
            r.is_proper_head_word_match(&a, &b),
            "head 'Obama' found in both mentions"
        );
    }

    #[test]
    fn proper_head_word_match_head_in_longer() {
        let r = resolver();
        // "President Obama" (head "Obama") matches "Barack Hussein Obama"
        // because "Obama" appears as a word in both.
        let a = Entity::new("President Obama", EntityType::Person, 0, 15, 0.9);
        let b = Entity::new("Barack Hussein Obama", EntityType::Person, 50, 70, 0.9);
        assert!(
            r.is_proper_head_word_match(&a, &b),
            "head 'Obama' from A found in B"
        );
    }

    #[test]
    fn proper_head_word_rejects_single_word() {
        let r = resolver();
        // Single-word mentions are excluded (handled by other sieves).
        let a = Entity::new("Obama", EntityType::Person, 0, 5, 0.9);
        let b = Entity::new("Barack Obama", EntityType::Person, 50, 62, 0.9);
        assert!(
            !r.is_proper_head_word_match(&a, &b),
            "single-word mention should not match via proper head word sieve"
        );
    }

    #[test]
    fn proper_head_word_rejects_different_types() {
        let r = resolver();
        let a = Entity::new("New York Times", EntityType::Organization, 0, 14, 0.9);
        let b = Entity::new("New York", EntityType::Location, 50, 58, 0.9);
        assert!(
            !r.is_proper_head_word_match(&a, &b),
            "different entity types should not match"
        );
    }

    // =========================================================================
    // Integration: end-to-end resolve with new sieves
    // =========================================================================

    #[test]
    fn integration_strict_head_clusters_smiths() {
        let r = resolver();
        let entities = vec![
            Entity::new("Acme Corp", EntityType::Organization, 0, 9, 0.9),
            Entity::new("Acme Corporation", EntityType::Organization, 50, 66, 0.9),
        ];
        let resolved = r.resolve(&entities);
        // "Acme Corp" and "Acme Corporation" don't share a head word (Corp vs Corporation),
        // but they should still match via substring/fuzzy sieve.
        assert_eq!(
            resolved[0].canonical_id, resolved[1].canonical_id,
            "should corefer via fuzzy matching"
        );
    }

    #[test]
    fn integration_new_sieves_do_not_break_existing() {
        // Existing exact-match and pronoun resolution still work.
        let r = resolver();
        let entities = vec![
            Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
            Entity::new("he", EntityType::Person, 15, 17, 0.8),
            Entity::new("John Smith", EntityType::Person, 30, 40, 0.9),
        ];
        let resolved = r.resolve(&entities);
        assert_eq!(resolved[0].canonical_id, resolved[1].canonical_id);
        assert_eq!(resolved[0].canonical_id, resolved[2].canonical_id);
    }

    // =========================================================================
    // head_word helper
    // =========================================================================

    #[test]
    fn head_word_extraction() {
        assert_eq!(SimpleCorefResolver::head_word("President Obama"), "Obama");
        assert_eq!(SimpleCorefResolver::head_word("Obama"), "Obama");
        assert_eq!(
            SimpleCorefResolver::head_word("the United States"),
            "States"
        );
        assert_eq!(SimpleCorefResolver::head_word(""), "");
    }

    // =========================================================================
    // Fix 7: Tightened fuzzy matching (names_match)
    // =========================================================================

    /// Short substrings (< 5 chars) should NOT cause substring matches.
    #[test]
    fn fuzzy_rejects_short_substrings() {
        let r = resolver();
        // "CEO" (3 chars) should not match "CEOVILLE" via substring
        assert!(
            !r.names_match("ORG:ceo", "ORG:ceoville"),
            "3-char substring 'ceo' should not match 'ceoville'"
        );
        // "the" should not match "other"
        assert!(
            !r.names_match("PER:the", "PER:other"),
            "'the' should not match 'other' via substring"
        );
        // "art" should not match "article"
        assert!(
            !r.names_match("ORG:art", "ORG:article"),
            "'art' should not match 'article'"
        );
    }

    /// Substrings >= 5 chars with sufficient ratio should match.
    #[test]
    fn fuzzy_accepts_long_substrings() {
        let r = resolver();
        // "obama" (5 chars) in "barack obama" -- ratio 5/12 = 0.41 > 0.3
        assert!(
            r.names_match("PER:obama", "PER:barack obama"),
            "'obama' should match 'barack obama'"
        );
        // "smith" (5 chars) in "john smith" -- ratio 5/10 = 0.5 > 0.3
        assert!(
            r.names_match("PER:smith", "PER:john smith"),
            "'smith' should match 'john smith'"
        );
    }

    /// Substrings >= 5 chars but with low ratio should NOT match (unless word boundary).
    #[test]
    fn fuzzy_rejects_low_ratio_non_word_substrings() {
        let r = resolver();
        // "angel" (5 chars) appears as a substring of "los angeles international airport"
        // but NOT as a complete word. ratio 5/32 = 0.15 < 0.3.
        assert!(
            !r.names_match("ORG:angel", "ORG:los angeles international airport"),
            "Low-ratio non-word substring should not match"
        );
    }

    /// Substrings >= 5 chars that ARE complete words still match via word-boundary fallback.
    #[test]
    fn fuzzy_accepts_word_boundary_even_low_ratio() {
        let r = resolver();
        // "march" appears as a complete word in the longer string, so it matches
        // via word-boundary even though the ratio is low.
        assert!(
            r.names_match("ORG:march", "ORG:march of the penguins documentary film"),
            "Word-boundary match should still work for complete words"
        );
    }

    /// Word-boundary matching should still work for short strings.
    #[test]
    fn fuzzy_accepts_word_boundary_match() {
        let r = resolver();
        // "ceo" as a complete word in "ceo john" should match
        assert!(
            r.names_match("PER:ceo", "PER:ceo john"),
            "'ceo' should match 'ceo john' at word boundary"
        );
    }

    /// Last-word matching should still work.
    #[test]
    fn fuzzy_last_word_match() {
        let r = resolver();
        assert!(
            r.names_match("PER:obama", "PER:barack obama"),
            "Last word 'obama' should match multi-word"
        );
        assert!(
            r.names_match("PER:barack obama", "PER:obama"),
            "Last word match should be symmetric"
        );
    }

    /// Different entity types should never match.
    #[test]
    fn fuzzy_rejects_different_types() {
        let r = resolver();
        assert!(
            !r.names_match("PER:obama", "ORG:obama"),
            "Different entity types should not match"
        );
    }

    /// Exact match still works.
    #[test]
    fn fuzzy_exact_match() {
        let r = resolver();
        assert!(r.names_match("PER:john", "PER:john"));
    }

    /// Transitive chain prevention: short substrings should not create
    /// chains like "CEO" -> "Thursday" -> "Shuntaro Furukawa".
    #[test]
    fn fuzzy_no_transitive_chain_via_short_substrings() {
        let r = resolver();
        // Each pair individually should NOT match via substring
        assert!(
            !r.names_match("PER:ceo", "PER:thursday"),
            "CEO should not match Thursday"
        );
        assert!(
            !r.names_match("PER:thursday", "PER:shuntaro furukawa"),
            "Thursday should not match Shuntaro Furukawa"
        );
        // And the endpoints definitely should not
        assert!(
            !r.names_match("PER:ceo", "PER:shuntaro furukawa"),
            "CEO should not match Shuntaro Furukawa"
        );
    }

    /// Integration: tightened fuzzy matching with full resolve pipeline.
    #[test]
    fn integration_no_spurious_merge_short_names() {
        // Disable all sieves except fuzzy to isolate the fix
        let cfg = CorefConfig {
            relaxed_head_match: false,
            proper_containment: false,
            strict_head_match: false,
            proper_head_word_match: false,
            precise_constructs: false,
            acronym_matching: false,
            ..Default::default()
        };
        let r = SimpleCorefResolver::new(cfg);

        let entities = vec![
            Entity::new("CEO", EntityType::Person, 0, 3, 0.9),
            Entity::new("Furukawa", EntityType::Person, 20, 28, 0.9),
        ];
        let resolved = r.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id, resolved[1].canonical_id,
            "CEO and Furukawa should NOT be in the same cluster via fuzzy matching"
        );
    }

    // =========================================================================
    // Wildcard type guard (QA regression)
    // =========================================================================

    #[test]
    fn proper_entities_not_spuriously_merged() {
        // Two unrelated proper-noun entities with wildcard type should NOT be
        // clustered together by fuzzy sieves.
        let r = resolver();
        let entities = vec![
            Entity::new(
                "Nobel",
                EntityType::custom("proper", EntityCategory::Misc),
                0,
                5,
                0.8,
            ),
            Entity::new(
                "Emmanuelle",
                EntityType::custom("proper", EntityCategory::Misc),
                20,
                30,
                0.8,
            ),
        ];
        let resolved = r.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id, resolved[1].canonical_id,
            "wildcard-type entities 'Nobel' and 'Emmanuelle' should NOT be merged"
        );
    }

    #[test]
    fn coref_does_not_merge_distinct_people() {
        // Two different PER entities should not be merged just because they
        // share words or are close together.
        let r = resolver();
        let entities = vec![
            Entity::new("Jennifer Doudna", EntityType::Person, 0, 15, 0.9),
            Entity::new("Emmanuelle Charpentier", EntityType::Person, 20, 42, 0.9),
        ];
        let resolved = r.resolve(&entities);
        assert_ne!(
            resolved[0].canonical_id, resolved[1].canonical_id,
            "Doudna and Charpentier should NOT be in the same cluster"
        );
    }

    #[test]
    fn coref_exact_match_still_works_for_wildcard() {
        // Exact canonical match (sieve 2) should still work for wildcard types.
        let r = resolver();
        let entities = vec![
            Entity::new(
                "Nobel",
                EntityType::custom("proper", EntityCategory::Misc),
                0,
                5,
                0.8,
            ),
            Entity::new(
                "Nobel",
                EntityType::custom("proper", EntityCategory::Misc),
                20,
                25,
                0.8,
            ),
        ];
        let resolved = r.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id, resolved[1].canonical_id,
            "identical wildcard-type entities should still be merged via exact match"
        );
    }
}
