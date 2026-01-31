//! Hard negative (confusable) entity mining for disambiguation.
//!
//! Based on the approach from "Contrastive Entity Coreference and Disambiguation for
//! Historical Texts" (Arora et al. 2024), which introduces WikiConfusables:
//!
//! > We source hard negative examples from Wikipedia disambiguation pages, which
//! > list entities that share the same name but refer to different real-world
//! > referents (e.g., "John Smith" could be a footballer, politician, or musician).
//!
//! # Hard Negative Sources
//!
//! 1. **Wikipedia Disambiguation Pages**: "John Smith" → multiple Q-IDs
//! 2. **Name Collisions**: Same surface form, different entities
//! 3. **Transliteration Variants**: "Beijing"/"Peking", "Moscow"/"Moskva"
//! 4. **OCR Confusables**: "Gnome"/"Gnorne" (historical documents)
//! 5. **Temporal Confusables**: "President Bush" (41 vs 43)
//!
//! # Usage
//!
//! ```rust
//! use anno::linking::confusables::{ConfusableSet, ConfusableReason};
//!
//! let mut confusables = ConfusableSet::new("John Smith");
//!
//! // Add from disambiguation page
//! confusables.add("Q123456", ConfusableReason::DisambiguationPage,
//!     Some("American politician"));
//! confusables.add("Q789012", ConfusableReason::DisambiguationPage,
//!     Some("English footballer"));
//!
//! // Use for contrastive training
//! let pairs = confusables.to_training_pairs("Q123456");
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Reason why two entities are confusable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConfusableReason {
    /// From Wikipedia/Wikidata disambiguation page
    DisambiguationPage,
    /// Share the same surface form/name
    NameCollision,
    /// Transliteration or spelling variant
    TransliterationVariant,
    /// OCR error pattern (common in historical docs)
    OcrError,
    /// Same title/role at different times
    TemporalAmbiguity,
    /// Parent/child or family with same name
    FamilialAmbiguity,
    /// Organization name changed over time
    NameChange,
    /// Fictional vs real entity with same name
    FictionalVsReal,
}

/// A confusable entity candidate (hard negative).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusableEntity {
    /// Knowledge base ID
    pub kb_id: String,
    /// Canonical label
    pub label: String,
    /// Short description to distinguish
    pub description: Option<String>,
    /// Why this is confusable
    pub reason: ConfusableReason,
    /// Type information (human, organization, etc.)
    pub entity_type: Option<String>,
    /// Difficulty score (0-1, higher = harder to distinguish)
    pub difficulty: f64,
}

impl ConfusableEntity {
    /// Create a new confusable entity.
    pub fn new(kb_id: &str, reason: ConfusableReason) -> Self {
        Self {
            kb_id: kb_id.to_string(),
            label: String::new(),
            description: None,
            reason,
            entity_type: None,
            difficulty: 0.5,
        }
    }

    /// Set label.
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = label.to_string();
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Set entity type.
    pub fn with_entity_type(mut self, entity_type: &str) -> Self {
        self.entity_type = Some(entity_type.to_string());
        self
    }

    /// Set difficulty.
    pub fn with_difficulty(mut self, difficulty: f64) -> Self {
        self.difficulty = difficulty.clamp(0.0, 1.0);
        self
    }
}

/// A set of confusable entities for a given surface form.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfusableSet {
    /// The surface form (e.g., "John Smith")
    pub surface_form: String,
    /// All confusable entities
    pub entities: Vec<ConfusableEntity>,
}

impl ConfusableSet {
    /// Create a new confusable set.
    pub fn new(surface_form: &str) -> Self {
        Self {
            surface_form: surface_form.to_string(),
            entities: Vec::new(),
        }
    }

    /// Add a confusable entity.
    pub fn add(&mut self, kb_id: &str, reason: ConfusableReason, description: Option<&str>) {
        let mut entity = ConfusableEntity::new(kb_id, reason);
        if let Some(desc) = description {
            entity = entity.with_description(desc);
        }
        self.entities.push(entity);
    }

    /// Add a full confusable entity.
    pub fn add_entity(&mut self, entity: ConfusableEntity) {
        self.entities.push(entity);
    }

    /// Generate training pairs for contrastive learning.
    ///
    /// Given a positive entity ID, returns (positive, negative) pairs
    /// for all other confusable entities.
    pub fn to_training_pairs(&self, positive_kb_id: &str) -> Vec<TrainingPair> {
        self.entities
            .iter()
            .filter(|e| e.kb_id != positive_kb_id)
            .map(|negative| TrainingPair {
                surface_form: self.surface_form.clone(),
                positive_kb_id: positive_kb_id.to_string(),
                negative_kb_id: negative.kb_id.clone(),
                negative_description: negative.description.clone(),
                difficulty: negative.difficulty,
            })
            .collect()
    }

    /// Number of confusable entities.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Get entities by reason.
    pub fn filter_by_reason(&self, reason: &ConfusableReason) -> Vec<&ConfusableEntity> {
        self.entities
            .iter()
            .filter(|e| &e.reason == reason)
            .collect()
    }
}

/// A training pair for contrastive learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPair {
    /// Surface form (anchor)
    pub surface_form: String,
    /// Positive entity KB ID
    pub positive_kb_id: String,
    /// Negative entity KB ID (hard negative)
    pub negative_kb_id: String,
    /// Description of negative (for context)
    pub negative_description: Option<String>,
    /// Difficulty score
    pub difficulty: f64,
}

/// Registry of confusable entities.
///
/// Maps surface forms to their confusable entity sets.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfusableRegistry {
    /// Map from normalized surface form to confusable set
    entries: HashMap<String, ConfusableSet>,
}

impl ConfusableRegistry {
    /// Create a new registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or merge a confusable set.
    pub fn add(&mut self, set: ConfusableSet) {
        let key = set.surface_form.to_lowercase();
        self.entries
            .entry(key)
            .and_modify(|existing| {
                for entity in &set.entities {
                    if !existing.entities.iter().any(|e| e.kb_id == entity.kb_id) {
                        existing.entities.push(entity.clone());
                    }
                }
            })
            .or_insert(set);
    }

    /// Look up confusables for a surface form.
    pub fn get(&self, surface_form: &str) -> Option<&ConfusableSet> {
        self.entries.get(&surface_form.to_lowercase())
    }

    /// Number of surface forms tracked.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Load well-known confusables for testing/demo.
    ///
    /// These are famous disambiguation cases that frequently appear
    /// in historical and modern documents.
    pub fn with_well_known(mut self) -> Self {
        // "John Smith" - extremely common name
        let mut john_smith = ConfusableSet::new("John Smith");
        john_smith.add_entity(
            ConfusableEntity::new("Q217557", ConfusableReason::DisambiguationPage)
                .with_label("John Smith")
                .with_description("English soldier and explorer, founder of Jamestown")
                .with_entity_type("human")
                .with_difficulty(0.8),
        );
        john_smith.add_entity(
            ConfusableEntity::new("Q556859", ConfusableReason::DisambiguationPage)
                .with_label("John Smith")
                .with_description("British Labour Party leader")
                .with_entity_type("human")
                .with_difficulty(0.7),
        );
        john_smith.add_entity(
            ConfusableEntity::new("Q23489016", ConfusableReason::FictionalVsReal)
                .with_label("John Smith")
                .with_description("Doctor Who character")
                .with_entity_type("fictional character")
                .with_difficulty(0.6),
        );
        self.add(john_smith);

        // "George Bush" - temporal ambiguity
        let mut george_bush = ConfusableSet::new("George Bush");
        george_bush.add_entity(
            ConfusableEntity::new("Q207", ConfusableReason::TemporalAmbiguity)
                .with_label("George W. Bush")
                .with_description("43rd President of the United States")
                .with_entity_type("human")
                .with_difficulty(0.9),
        );
        george_bush.add_entity(
            ConfusableEntity::new("Q23505", ConfusableReason::TemporalAmbiguity)
                .with_label("George H. W. Bush")
                .with_description("41st President of the United States")
                .with_entity_type("human")
                .with_difficulty(0.9),
        );
        self.add(george_bush);

        // "President Bush" - same issue
        let mut president_bush = ConfusableSet::new("President Bush");
        president_bush.add_entity(
            ConfusableEntity::new("Q207", ConfusableReason::TemporalAmbiguity)
                .with_label("George W. Bush")
                .with_description("43rd President of the United States")
                .with_entity_type("human")
                .with_difficulty(0.95),
        );
        president_bush.add_entity(
            ConfusableEntity::new("Q23505", ConfusableReason::TemporalAmbiguity)
                .with_label("George H. W. Bush")
                .with_description("41st President of the United States")
                .with_entity_type("human")
                .with_difficulty(0.95),
        );
        self.add(president_bush);

        // "Michael Jackson" - different famous people
        let mut michael_jackson = ConfusableSet::new("Michael Jackson");
        michael_jackson.add_entity(
            ConfusableEntity::new("Q2831", ConfusableReason::DisambiguationPage)
                .with_label("Michael Jackson")
                .with_description("American singer-songwriter (1958-2009)")
                .with_entity_type("human")
                .with_difficulty(0.7),
        );
        michael_jackson.add_entity(
            ConfusableEntity::new("Q318029", ConfusableReason::DisambiguationPage)
                .with_label("Michael Jackson")
                .with_description("English beer and whisky writer")
                .with_entity_type("human")
                .with_difficulty(0.6),
        );
        self.add(michael_jackson);

        // "Apple" - company vs fruit vs records
        let mut apple = ConfusableSet::new("Apple");
        apple.add_entity(
            ConfusableEntity::new("Q312", ConfusableReason::NameCollision)
                .with_label("Apple Inc.")
                .with_description("American technology company")
                .with_entity_type("organization")
                .with_difficulty(0.7),
        );
        apple.add_entity(
            ConfusableEntity::new("Q213710", ConfusableReason::NameCollision)
                .with_label("Apple Records")
                .with_description("Record label founded by the Beatles")
                .with_entity_type("organization")
                .with_difficulty(0.6),
        );
        self.add(apple);

        // "Paris" - city vs person vs mythological figure
        let mut paris = ConfusableSet::new("Paris");
        paris.add_entity(
            ConfusableEntity::new("Q90", ConfusableReason::DisambiguationPage)
                .with_label("Paris")
                .with_description("Capital city of France")
                .with_entity_type("city")
                .with_difficulty(0.5),
        );
        paris.add_entity(
            ConfusableEntity::new("Q167646", ConfusableReason::DisambiguationPage)
                .with_label("Paris Hilton")
                .with_description("American socialite and businesswoman")
                .with_entity_type("human")
                .with_difficulty(0.4),
        );
        paris.add_entity(
            ConfusableEntity::new("Q167491", ConfusableReason::FictionalVsReal)
                .with_label("Paris")
                .with_description("Trojan prince in Greek mythology")
                .with_entity_type("mythological figure")
                .with_difficulty(0.6),
        );
        self.add(paris);

        // Transliteration variants
        let mut beijing = ConfusableSet::new("Beijing");
        beijing.add_entity(
            ConfusableEntity::new("Q956", ConfusableReason::TransliterationVariant)
                .with_label("Beijing")
                .with_description("Capital of China (modern transliteration)")
                .with_entity_type("city")
                .with_difficulty(0.3),
        );
        self.add(beijing);

        let mut peking = ConfusableSet::new("Peking");
        peking.add_entity(
            ConfusableEntity::new("Q956", ConfusableReason::TransliterationVariant)
                .with_label("Beijing")
                .with_description("Capital of China (historical transliteration)")
                .with_entity_type("city")
                .with_difficulty(0.3),
        );
        self.add(peking);

        // Historical OCR-style confusables (common in digitized newspapers)
        let mut washington = ConfusableSet::new("Washington");
        washington.add_entity(
            ConfusableEntity::new("Q23", ConfusableReason::DisambiguationPage)
                .with_label("George Washington")
                .with_description("1st President of the United States")
                .with_entity_type("human")
                .with_difficulty(0.7),
        );
        washington.add_entity(
            ConfusableEntity::new("Q61", ConfusableReason::DisambiguationPage)
                .with_label("Washington, D.C.")
                .with_description("Capital of the United States")
                .with_entity_type("city")
                .with_difficulty(0.6),
        );
        washington.add_entity(
            ConfusableEntity::new("Q1223", ConfusableReason::DisambiguationPage)
                .with_label("Washington")
                .with_description("State in the Pacific Northwest")
                .with_entity_type("administrative region")
                .with_difficulty(0.5),
        );
        self.add(washington);

        self
    }

    /// Generate all training pairs for contrastive learning.
    ///
    /// For each confusable set, generates pairs where one entity is positive
    /// and all others are negatives.
    pub fn generate_all_training_pairs(&self) -> Vec<TrainingPair> {
        let mut pairs = Vec::new();
        for set in self.entries.values() {
            for positive in &set.entities {
                pairs.extend(set.to_training_pairs(&positive.kb_id));
            }
        }
        pairs
    }

    /// Filter to entities of a specific type.
    pub fn filter_by_type(&self, entity_type: &str) -> Vec<&ConfusableEntity> {
        self.entries
            .values()
            .flat_map(|set| &set.entities)
            .filter(|e| {
                e.entity_type
                    .as_ref()
                    .is_some_and(|t| t.to_lowercase() == entity_type.to_lowercase())
            })
            .collect()
    }
}

/// OCR error patterns common in historical documents.
///
/// Based on the observation from Arora et al. that historical documents
/// "are replete with individuals not remembered in contemporary knowledgebases"
/// and suffer from OCR noise.
#[derive(Debug, Clone, Default)]
pub struct OcrConfusables {
    /// Character substitution patterns
    substitutions: HashMap<char, Vec<char>>,
}

impl OcrConfusables {
    /// Create with common OCR error patterns.
    pub fn new() -> Self {
        let mut subs = HashMap::new();

        // Common OCR confusions
        subs.insert('m', vec!['r', 'n']); // "m" → "rn"
        subs.insert('l', vec!['1', 'I', '|']);
        subs.insert('O', vec!['0', 'Q']);
        subs.insert('I', vec!['l', '1', '|']);
        subs.insert('S', vec!['5', '$']);
        subs.insert('B', vec!['8', '3']);
        subs.insert('G', vec!['6', 'C']);
        subs.insert('Z', vec!['2']);
        subs.insert('o', vec!['0', 'c']);
        subs.insert('c', vec!['o', 'e']);
        subs.insert('e', vec!['c', 'o']);
        subs.insert('h', vec!['b', 'n']);
        subs.insert('u', vec!['v', 'n']);
        subs.insert('v', vec!['u', 'w']);
        subs.insert('w', ["vv", "uu"].iter().flat_map(|s| s.chars()).collect());

        Self {
            substitutions: subs,
        }
    }

    /// Generate OCR variants of a string.
    ///
    /// Returns possible OCR-corrupted versions of the input.
    pub fn generate_variants(&self, text: &str) -> Vec<String> {
        let mut variants = Vec::new();
        let chars: Vec<char> = text.chars().collect();

        // Single character substitutions
        for (i, c) in chars.iter().enumerate() {
            if let Some(subs) = self.substitutions.get(c) {
                for sub in subs {
                    let mut variant: String = chars[..i].iter().collect();
                    variant.push(*sub);
                    variant.extend(&chars[i + 1..]);
                    variants.push(variant);
                }
            }
        }

        // Common pattern: "rn" → "m"
        let text_lower = text.to_lowercase();
        if text_lower.contains("rn") {
            variants.push(text.replace("rn", "m"));
        }
        if text_lower.contains("m") {
            variants.push(text.replace('m', "rn"));
        }

        variants
    }

    /// Check if two strings might be OCR variants of each other.
    pub fn might_be_variants(&self, a: &str, b: &str) -> bool {
        let a_variants = self.generate_variants(a);
        if a_variants.iter().any(|v| v.eq_ignore_ascii_case(b)) {
            return true;
        }
        let b_variants = self.generate_variants(b);
        b_variants.iter().any(|v| v.eq_ignore_ascii_case(a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusable_set() {
        let mut set = ConfusableSet::new("John Smith");
        set.add("Q1", ConfusableReason::DisambiguationPage, Some("Explorer"));
        set.add(
            "Q2",
            ConfusableReason::DisambiguationPage,
            Some("Politician"),
        );

        assert_eq!(set.len(), 2);

        let pairs = set.to_training_pairs("Q1");
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].negative_kb_id, "Q2");
    }

    #[test]
    fn test_registry() {
        let registry = ConfusableRegistry::new().with_well_known();

        // Check George Bush disambiguation
        let bush = registry.get("george bush");
        assert!(bush.is_some());
        let bush = bush.unwrap();
        assert!(bush.len() >= 2);

        // Check training pairs
        let pairs = registry.generate_all_training_pairs();
        assert!(!pairs.is_empty());
    }

    #[test]
    fn test_temporal_ambiguity() {
        let registry = ConfusableRegistry::new().with_well_known();

        let bush = registry.get("president bush").unwrap();
        let temporal = bush.filter_by_reason(&ConfusableReason::TemporalAmbiguity);
        assert!(temporal.len() >= 2);
    }

    #[test]
    fn test_ocr_confusables() {
        let ocr = OcrConfusables::new();

        // "rn" ↔ "m" is a classic OCR confusion
        let variants = ocr.generate_variants("Gnome");
        assert!(variants.contains(&"Gnorne".to_string()));

        assert!(ocr.might_be_variants("Gnome", "Gnorne"));
    }

    #[test]
    fn test_transliteration() {
        let registry = ConfusableRegistry::new().with_well_known();

        // Beijing and Peking should both map to Q956
        let beijing = registry.get("beijing");
        let peking = registry.get("peking");

        assert!(beijing.is_some());
        assert!(peking.is_some());

        let beijing_ids: Vec<_> = beijing.unwrap().entities.iter().map(|e| &e.kb_id).collect();
        let peking_ids: Vec<_> = peking.unwrap().entities.iter().map(|e| &e.kb_id).collect();

        assert!(beijing_ids.contains(&&"Q956".to_string()));
        assert!(peking_ids.contains(&&"Q956".to_string()));
    }
}
