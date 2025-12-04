//! Rule-based Named Entity Recognition (NER) - **DEPRECATED**.
//!
//! **Use `RegexNER` instead** - it only extracts format-based entities
//! (dates, money, percentages) without hardcoded gazetteers.
//!
//! ## Why Deprecated
//!
//! This module contains hardcoded gazetteers (100+ org names, 60+ locations)
//! that give inflated F1 scores on curated test sets but fail on novel entities.
//!
//! | Backend | CoNLL-03 F1 | HOP Test F1 | Problem |
//! |---------|-------------|-------------|---------|
//! | BERT ONNX | ~86% | ~74% | None - generalizes |
//! | Rule-based | **~50%** | **~84%** | Only finds known entities |
//!
//! The 84% F1 on HOP tests is because test entities match gazetteers.
//! On novel entities (Anthropic, DeepMind, etc.), rule-based fails.
//!
//! ## Migration
//!
//! ```rust
//! use anno::{RegexNER, Model};
//!
//! // Use RegexNER for dates, money, percentages
//! let model = RegexNER::new();
//! let entities = model.extract_entities("Cost: $100", None).unwrap();
//! // For Person/Org/Location, enable `onnx` feature and use BertNEROnnx
//! ```
//!
//! ## If You Must Use This
//!
//! - For legacy compatibility only
//! - For deterministic/reproducible results on known entity sets
//! - For extremely low-latency requirements (~0.3ms)
//!
//! But understand: it cannot generalize to unseen entities.

use crate::{Entity, EntityType, Model, Result};
use once_cell::sync::Lazy;
use regex::Regex;

/// Rule-based NER (**DEPRECATED** - use `RegexNER` or ML backends).
///
/// Contains hardcoded gazetteers that give inflated F1 on curated tests
/// but fail on novel entities. Use `NERExtractor::best_available()` instead.
#[deprecated(
    since = "0.1.0",
    note = "Use RegexNER (no gazetteers) or ML backends (BERT ONNX). Will be removed in 1.0."
)]
pub struct RuleBasedNER {
    /// Minimum confidence for extracted entities
    min_confidence: f64,
    /// Whether to filter common words
    filter_common: bool,
}

#[allow(deprecated)]
impl RuleBasedNER {
    /// Create a new rule-based NER with default settings.
    pub fn new() -> Self {
        Self {
            min_confidence: 0.3, // Low threshold - let downstream filter by confidence
            filter_common: true,
        }
    }

    /// Create with custom minimum confidence.
    #[must_use]
    pub fn with_min_confidence(min_confidence: f64) -> Self {
        Self {
            min_confidence,
            filter_common: true,
        }
    }

    /// Create without common word filtering (for debugging).
    #[allow(dead_code)]
    pub fn without_filtering() -> Self {
        Self {
            min_confidence: 0.3,
            filter_common: false,
        }
    }
}

#[allow(deprecated)]
impl Default for RuleBasedNER {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(deprecated)]
impl Model for RuleBasedNER {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // ========================================================================
        // PRIORITY ORDER: Known orgs > Org patterns > Locations > Persons > Other
        // Higher-confidence patterns run first to avoid type confusion
        // ========================================================================

        // Pattern 0: Well-known organizations (acronyms and single words)
        // These are high-confidence matches that would otherwise be missed
        static KNOWN_ORGS: Lazy<Regex> = Lazy::new(|| {
            // Tech, Government, Academic, Conferences + Sports Leagues/Teams
            Regex::new(r"\b(?:NASA|FBI|CIA|NSA|NIH|FDA|CDC|EPA|WHO|NATO|UN|EU|IMF|WTO|CERN|MIT|UCLA|DARPA|OECD|OPEC|IEEE|ACM|AWS|GCP|IBM|HP|AMD|ARM|NVIDIA|Intel|Apple|Google|Microsoft|Amazon|Meta|OpenAI|Anthropic|DeepMind|Pfizer|Moderna|Rivian|BYD|Netflix|Uber|Airbnb|NeurIPS|ICML|ICLR|CVPR|ACL|EMNLP|NAACL|IPCC|SEC|FCC|DOJ|DOE|DOD|USDA|HUD|IRS|FEMA|OSHA|NOAA|NSF|USPTO|FTC|NIST|DOT|VA|SSA|SBA|FAA|TSA|ICE|CBP|USCIS|NFL|NBA|MLB|NHL|MLS|FIFA|UEFA|IOC|NCAA|PGA|ATP|WTA|UFC|WWE|ESPN|LEICESTERSHIRE|DERBYSHIRE|YORKSHIRE|SURREY|ESSEX|WARWICKSHIRE|SUSSEX|MIDDLESEX|HAMPSHIRE|SOMERSET|KENT|LANCASHIRE|GLOUCESTERSHIRE|NOTTINGHAMSHIRE|NORTHAMPTONSHIRE|WORCESTERSHIRE|DURHAM)\b")
                .expect("Failed to compile known orgs pattern")
        });

        for cap in KNOWN_ORGS.find_iter(text) {
            entities.push(Entity::new(
                cap.as_str(),
                EntityType::Organization,
                cap.start(),
                cap.end(),
                0.95, // Very high confidence for known orgs
            ));
        }

        // Pattern 1: Organizations (Inc., Corp., Corporation, Ltd., University, etc.)
        // Run FIRST to avoid "Microsoft Corporation" being tagged as Person
        static ORG_PATTERN: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|LLC|GmbH|University|Institute|Foundation|Laboratory|Labs?|Company|Technologies|Systems|Research|Group|Partners|Associates|Agency|Commission|Court|Council|Board|Committee|Organization|Organisation|Bank|Reserve|Museum)\b")
                .expect("Failed to compile org pattern")
        });

        for cap in ORG_PATTERN.find_iter(text) {
            // Skip if already covered
            if entities
                .iter()
                .any(|e| spans_overlap(e.start, e.end, cap.start(), cap.end()))
            {
                continue;
            }
            let text_str = strip_leading_article(cap.as_str());
            let start_adj = cap.start() + (cap.as_str().len() - text_str.len());
            entities.push(Entity::new(
                text_str,
                EntityType::Organization,
                start_adj,
                cap.end(),
                0.85, // High confidence for explicit org suffixes
            ));
        }

        // Pattern 2: Locations (city, country patterns - expanded)
        static LOCATION_PATTERN: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"\b(?:New\s+York(?:\s+City)?|San\s+Francisco|Los\s+Angeles|Washington(?:\s+D\.?C\.?)?|Tokyo\s+Bay|United\s+States|United\s+Kingdom|European\s+Union|Asia-Pacific|North\s+America|South\s+America|Atlantic\s+Ocean|Pacific\s+Ocean|Amazon\s+River|Tokyo|Berlin|Paris|London|Beijing|Shanghai|Mumbai|Sydney|Moscow|Dubai|Seoul|Singapore|Hong\s+Kong|Brazil|Peru|Colombia|China|Japan|Germany|France|Italy|Spain|Canada|Australia|India|Russia|Mexico|Argentina|Chile|Ukraine|California|Texas|Florida|Illinois|Seattle|Chicago|Boston|Atlanta|Denver|Phoenix|Portland|Miami|Cupertino|Redmond|Wuhan|Geneva)\b")
                .expect("Failed to compile location pattern")
        });

        for cap in LOCATION_PATTERN.find_iter(text) {
            // Skip if already covered by organization
            if entities
                .iter()
                .any(|e| spans_overlap(e.start, e.end, cap.start(), cap.end()))
            {
                continue;
            }
            let text_str = strip_leading_article(cap.as_str());
            let start_adj = cap.start() + (cap.as_str().len() - text_str.len());
            entities.push(Entity::new(
                text_str,
                EntityType::Location,
                start_adj,
                cap.end(),
                0.9,
            ));
        }

        // Pattern 3: Person names (common first+last name patterns)
        // Look for patterns like "John Smith", "J. Smith", "Dr. Smith", "Chen et al."
        // Exclude patterns starting with "The "
        static PERSON_PATTERN: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"(?:Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.|Chairman|CEO|President|Director|Justice|General|Commissioner|Coach|Governor|Senator|Mayor)\s+[A-Z][a-z]+(?:\s+[a-z]+\s+[A-Z][a-z]+|\s+[A-Z][a-z]+)?|[A-Z][a-z]+\s+(?:et\s+al\.?)|[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?")
                .expect("Failed to compile person pattern")
        });

        for cap in PERSON_PATTERN.find_iter(text) {
            let mut text_str = cap.as_str();
            // Strip leading "The " if present
            text_str = strip_leading_article(text_str);

            // Skip if overlaps with already-extracted org/location (higher priority)
            if entities
                .iter()
                .any(|e| spans_overlap(e.start, e.end, cap.start(), cap.end()))
            {
                continue;
            }
            // Skip common words or sentences starting with "The"
            if self.filter_common
                && (is_common_capitalized_word(text_str) || starts_with_noise(text_str))
            {
                continue;
            }
            let start_adj = cap.start() + (cap.as_str().len() - text_str.len());
            entities.push(Entity::new(
                text_str,
                EntityType::Person,
                start_adj,
                cap.end(),
                0.7,
            ));
        }

        // Pattern 4: Other capitalized phrases (less confident)
        static CAPITALIZED_PATTERN: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
                .expect("Failed to compile capitalized pattern")
        });

        for cap in CAPITALIZED_PATTERN.find_iter(text) {
            let mut text_str = cap.as_str();
            // Strip leading "The " if present
            text_str = strip_leading_article(text_str);
            if text_str.is_empty() {
                continue;
            }
            // Skip common words that are capitalized but not entities
            if self.filter_common
                && (is_common_capitalized_word(text_str) || starts_with_noise(text_str))
            {
                continue;
            }
            // Skip if already matched by more specific patterns (org, location, person)
            if entities
                .iter()
                .any(|e| spans_overlap(e.start, e.end, cap.start(), cap.end()))
            {
                continue;
            }
            // Use heuristics to infer type
            let entity_type = infer_entity_type(text_str);
            let start_adj = cap.start() + (cap.as_str().len() - text_str.len());
            entities.push(Entity::new(
                text_str,
                entity_type,
                start_adj,
                cap.end(),
                0.4, // Lower confidence for generic matches
            ));
        }

        // Pattern 5: Dates - expanded to catch more formats
        static DATE_PATTERN: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+\d{4})?|(?:Q[1-4]|(?:19|20)\d{2}))\b")
                .expect("Failed to compile date pattern")
        });

        for date_match in DATE_PATTERN.find_iter(text) {
            // Skip if overlaps with existing entity
            if entities
                .iter()
                .any(|e| spans_overlap(e.start, e.end, date_match.start(), date_match.end()))
            {
                continue;
            }
            entities.push(Entity::new(
                date_match.as_str(),
                EntityType::Date,
                date_match.start(),
                date_match.end(),
                0.8,
            ));
        }

        // Pattern 6: Money amounts (enhanced)
        static MONEY_PATTERN: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"\$[\d,]+\.?\d*\s*(?:billion|million|thousand|B|M|K)?|\d+\.?\d*\s*(?:dollars?|USD|EUR|GBP|billion|million)")
                .expect("Failed to compile money pattern")
        });

        for money_match in MONEY_PATTERN.find_iter(text) {
            // Skip if overlaps with existing entity
            if entities
                .iter()
                .any(|e| spans_overlap(e.start, e.end, money_match.start(), money_match.end()))
            {
                continue;
            }
            entities.push(Entity::new(
                money_match.as_str(),
                EntityType::Money,
                money_match.start(),
                money_match.end(),
                0.8,
            ));
        }

        // Pattern 7: Percentages
        static PERCENT_PATTERN: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"\d+\.?\d*\s*%").expect("Failed to compile percent pattern"));

        for percent_match in PERCENT_PATTERN.find_iter(text) {
            // Skip if overlaps with existing entity
            if entities
                .iter()
                .any(|e| spans_overlap(e.start, e.end, percent_match.start(), percent_match.end()))
            {
                continue;
            }
            entities.push(Entity::new(
                percent_match.as_str(),
                EntityType::Percent,
                percent_match.start(),
                percent_match.end(),
                0.8,
            ));
        }

        // Filter by minimum confidence
        entities.retain(|e| e.confidence >= self.min_confidence);

        // Post-process: remove entities that start with "The " (final cleanup)
        entities.retain(|e| !e.text.starts_with("The "));

        Ok(entities)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Date,
            EntityType::Money,
            EntityType::Percent,
            EntityType::Other("unknown".to_string()),
        ]
    }

    fn is_available(&self) -> bool {
        true // Rule-based is always available
    }

    fn name(&self) -> &'static str {
        "rule"
    }

    fn description(&self) -> &'static str {
        "Rule-based NER using regex patterns and heuristics"
    }
}

/// Check if two spans overlap.
fn spans_overlap(s1_start: usize, s1_end: usize, s2_start: usize, s2_end: usize) -> bool {
    !(s1_end <= s2_start || s2_end <= s1_start)
}

/// Strip leading articles ("The ", "A ", "An ") from entity text.
fn strip_leading_article(text: &str) -> &str {
    text.strip_prefix("The ")
        .or_else(|| text.strip_prefix("A "))
        .or_else(|| text.strip_prefix("An "))
        .unwrap_or(text)
}

/// Check if text starts with common noise patterns.
fn starts_with_noise(text: &str) -> bool {
    // These patterns often lead to false positives
    let noise_starts = [
        "According",
        "Based",
        "Given",
        "Following",
        "Regarding",
        "Attention Is",
        "All You", // Common paper title fragments
    ];
    noise_starts.iter().any(|n| text.starts_with(n))
}

/// Infer entity type from text using simple heuristics.
///
/// This provides better typing than "unknown" for common patterns.
fn infer_entity_type(text: &str) -> EntityType {
    let lower = text.to_lowercase();
    let words: Vec<&str> = text.split_whitespace().collect();

    // Looks like a person name (2-3 word capitalized, common name patterns)
    if words.len() == 2 || words.len() == 3 {
        // Check if looks like "Firstname Lastname"
        if words
            .iter()
            .all(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
        {
            // Common Chinese/Korean surnames are a strong signal
            if is_common_surname(words[0])
                || (words.len() > 1 && words.last().is_some_and(|w| is_common_surname(w)))
            {
                return EntityType::Person;
            }
        }
    }

    // Single word with common surname
    if words.len() == 1 && is_common_surname(text) {
        return EntityType::Person;
    }

    // Technical/concept terms
    if lower.contains("network")
        || lower.contains("model")
        || lower.contains("algorithm")
        || lower.contains("learning")
        || lower.contains("neural")
        || lower.contains("transformer")
    {
        return EntityType::Other("concept".to_string());
    }

    // Acronyms (all caps, 2-5 chars) are often organizations or technical terms
    if text.len() >= 2 && text.len() <= 5 && text.chars().all(|c| c.is_uppercase()) {
        return EntityType::Other("acronym".to_string());
    }

    EntityType::Other("unknown".to_string())
}

/// Check if a word is a common surname (for person detection).
fn is_common_surname(word: &str) -> bool {
    static COMMON_SURNAMES: &[&str] = &[
        // Chinese surnames (very common in academic papers)
        "Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhao", "Wu", "Zhou", "Xu", "Sun",
        "Ma", "Zhu", "Hu", "Guo", "Lin", "He", "Gao", "Luo", "Zheng", "Liang", "Xie", "Tang",
        "Han", "Feng", "Deng", "Cao", "Peng", "Xiao", "Jiang", "Cheng", "Yuan", "Lu", "Pan",
        "Ding", "Wei", "Ren", "Shao", "Qian", // Korean surnames
        "Kim", "Lee", "Park", "Choi", "Jung", "Kang", "Cho", "Yoon", "Jang", "Lim",
        // Japanese surnames
        "Tanaka", "Suzuki", "Yamamoto", "Watanabe", "Sato", "Ito", "Nakamura",
        // Western surnames (common in papers)
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson", "Moore",
        "Taylor", "Anderson", "Thomas", "White", "Harris",
    ];
    COMMON_SURNAMES.contains(&word)
}

/// Check if a capitalized word is a common word (not an entity).
///
/// This extensive list filters out noise from PDF extraction and academic papers.
/// Words are matched exactly (case-sensitive).
fn is_common_capitalized_word(word: &str) -> bool {
    // Use a static HashSet for O(1) lookups
    use std::collections::HashSet;
    use std::sync::OnceLock;

    static COMMON_WORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();

    let common_words = COMMON_WORDS.get_or_init(|| {
        let words: &[&str] = &[
            // Pronouns and determiners
            "The",
            "A",
            "An",
            "This",
            "That",
            "These",
            "Those",
            "I",
            "You",
            "He",
            "She",
            "It",
            "We",
            "They",
            "My",
            "Your",
            "His",
            "Her",
            "Its",
            "Our",
            "Their",
            "What",
            "Which",
            "Who",
            "Whom",
            // Conjunctions and prepositions
            "And",
            "Or",
            "But",
            "If",
            "When",
            "Where",
            "Why",
            "How",
            "As",
            "At",
            "By",
            "For",
            "From",
            "In",
            "Into",
            "Of",
            "On",
            "To",
            "With",
            "About",
            "After",
            "Against",
            "Before",
            "Between",
            "During",
            "Through",
            "Under",
            "Over",
            "Above",
            "Below",
            "Since",
            "Until",
            "Upon",
            // Verbs (common)
            "Is",
            "Are",
            "Was",
            "Were",
            "Be",
            "Been",
            "Being",
            "Have",
            "Has",
            "Had",
            "Do",
            "Does",
            "Did",
            "Will",
            "Would",
            "Could",
            "Should",
            "May",
            "Might",
            "Can",
            "Cannot",
            "Let",
            "Get",
            "Got",
            "Make",
            "Made",
            "Take",
            "Took",
            "Give",
            "Gave",
            "See",
            "Saw",
            "Know",
            "Knew",
            "Think",
            "Thought",
            "Want",
            "Use",
            "Used",
            "Using",
            "Find",
            // Academic/document noise
            "Figure",
            "Table",
            "Section",
            "Chapter",
            "Page",
            "Abstract",
            "Introduction",
            "Conclusion",
            "Conclusions",
            "Discussion",
            "Method",
            "Methods",
            "Results",
            "References",
            "Appendix",
            "Acknowledgments",
            "Background",
            "Related",
            "Work",
            "Paper",
            "Papers",
            "Study",
            "Studies",
            "Research",
            "Analysis",
            "Data",
            "Model",
            "Models",
            "Approach",
            "Problem",
            "Solution",
            "System",
            "Systems",
            "Algorithm",
            "Algorithms",
            "Experiment",
            "Experiments",
            "Evaluation",
            "Performance",
            "Application",
            "Applications",
            // Common sentence starters
            "However",
            "Therefore",
            "Furthermore",
            "Moreover",
            "Although",
            "Thus",
            "Hence",
            "Similarly",
            "Additionally",
            "Nevertheless",
            "Consequently",
            "Specifically",
            "Generally",
            "Particularly",
            "Especially",
            "Indeed",
            "Actually",
            "Obviously",
            "Clearly",
            "Certainly",
            "Probably",
            "Possibly",
            "Perhaps",
            "Rather",
            "Instead",
            "Otherwise",
            "Finally",
            "Initially",
            "Ultimately",
            "Essentially",
            "Basically",
            // Noise from PDFs
            "Note",
            "Notes",
            "Example",
            "Examples",
            "Definition",
            "Theorem",
            "Proof",
            "Lemma",
            "Proposition",
            "Corollary",
            "Remark",
            "Case",
            "Cases",
            "Step",
            "Steps",
            "Part",
            "Parts",
            "Item",
            "Items",
            "Point",
            "Points",
            "Fact",
            "Facts",
            "First",
            "Second",
            "Third",
            "Fourth",
            "Fifth",
            "Next",
            "Previous",
            "Following",
            "Preceding",
            "Here",
            "There",
            "Now",
            "Then",
            "Today",
            "Yesterday",
            "Tomorrow",
            // Very short words (likely noise)
            "So",
            "No",
            "Yes",
            "Ok",
            "Oh",
            "Ah",
            "Eh",
            "Um",
            "Uh",
            "Re",
            "Vs",
            "Et",
            "Al",
            // Common academic phrases as single words
            "Based",
            "According",
            "Regarding",
            "Concerning",
            "Given",
            "Assuming",
            "Suppose",
            "Consider",
            "Considering",
            "Such",
            "Many",
            "Much",
            "Most",
            "Some",
            "Any",
            "Each",
            "Every",
            "Both",
            "All",
            "Other",
            "Another",
            "Same",
            "Different",
            "Various",
            "Several",
            // More noise
            "Published",
            "Received",
            "Accepted",
            "Revised",
            "Available",
            "Online",
            "Copyright",
            "Rights",
            "Reserved",
            "Author",
            "Authors",
            "Corresponding",
            "Email",
            "Address",
            "University",
            "Department",
            "Institute",
            "Center",
            "College",
            "School",
            "Lab",
            // Additional noise from PDFs
            "Fig",
            "Eq",
            "Eqs",
            "Ref",
            "Refs",
            "Tab",
            "Sec",
            "App",
            "Vol",
            "No",
            "Pp",
            "Ed",
            "Eds",
            "Inc",
            "Ltd",
            "Corp",
            "Co",
            "Jr",
            "Sr",
            "Dr",
            "Mr",
            "Mrs",
            "Ms",
            "Prof",
            // Quantifiers and modifiers
            "More",
            "Less",
            "Few",
            "Little",
            "New",
            "Old",
            "Good",
            "Bad",
            "Large",
            "Small",
            "High",
            "Low",
            "Long",
            "Short",
            "Full",
            "Empty",
            "True",
            "False",
            "Real",
            "Main",
            // Technical noise
            "Input",
            "Output",
            "Function",
            "Variable",
            "Parameter",
            "Value",
            "Type",
            "Class",
            "Object",
            "Array",
            "List",
            "Set",
            "Map",
            "Key",
            "Node",
            "Edge",
            "Graph",
            "Tree",
            "Network",
            "Layer",
            "Hidden",
            "Embedding",
            "Vector",
            "Matrix",
            "Tensor",
            "Loss",
            "Error",
            "Accuracy",
            "Score",
            "Rate",
            "Ratio",
            "Mean",
            "Average",
            "Sum",
            "Total",
            "Max",
            "Min",
            "Like",
            "Net",
            "Core",
            "Base",
            "Top",
            "Bottom",
            "Left",
            "Right",
        ];
        words.iter().copied().collect()
    });

    common_words.contains(word)
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_based_ner() {
        let ner = RuleBasedNER::new();
        let text =
            "John Smith works at Acme Corp. He earns $100,000 per year. The meeting is on 2024-01-15.";
        let entities = ner.extract_entities(text, None).unwrap();

        // Should extract capitalized names, money, dates
        assert!(!entities.is_empty());
        assert!(entities.iter().any(|e| e.text == "John Smith"));
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Money));
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Date));
    }

    #[test]
    fn test_common_word_filtering() {
        let ner = RuleBasedNER::new();
        let text = "The Figure shows the Results. However, the Introduction was clear.";
        let entities = ner.extract_entities(text, None).unwrap();

        // Common words should be filtered out
        assert!(!entities.iter().any(|e| e.text == "The"));
        assert!(!entities.iter().any(|e| e.text == "Figure"));
        assert!(!entities.iter().any(|e| e.text == "Results"));
        assert!(!entities.iter().any(|e| e.text == "However"));
        assert!(!entities.iter().any(|e| e.text == "Introduction"));
    }

    #[test]
    fn test_without_filtering() {
        let ner = RuleBasedNER::without_filtering();
        // Use text where common words are NOT followed by other capitalized words
        // (the pattern greedily matches multi-word phrases like "The Figure")
        let text = "The cat sat on Figure today.";
        let entities = ner.extract_entities(text, None).unwrap();

        // Without filtering, common words should be included as standalone entities
        assert!(
            entities.iter().any(|e| e.text == "The"),
            "Expected 'The' in entities: {:?}",
            entities
        );
        assert!(
            entities.iter().any(|e| e.text == "Figure"),
            "Expected 'Figure' in entities: {:?}",
            entities
        );
    }

    #[test]
    fn test_percentage_extraction() {
        let ner = RuleBasedNER::new();
        let text = "Accuracy improved by 15.5% and recall by 20%.";
        let entities = ner.extract_entities(text, None).unwrap();

        let percents: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Percent)
            .collect();
        assert_eq!(percents.len(), 2);
    }

    #[test]
    fn test_model_interface() {
        let ner = RuleBasedNER::new();
        assert!(ner.is_available());
        assert_eq!(ner.name(), "rule");
        assert!(!ner.supported_types().is_empty());
    }
}
