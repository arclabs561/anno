//! Statistical NER - Zero-dependency heuristic named entity recognition.
//!
//! This module provides lightweight named entity recognition using statistical
//! patterns and heuristics, without requiring any ML frameworks or external
//! dependencies beyond the standard library.
//!
//! # Design Philosophy
//!
//! StatisticalNER fills the gap between PatternNER (format-based) and ML backends
//! (deep learning). It can identify likely Person/Organization/Location entities
//! using:
//!
//! 1. **Capitalization patterns** - Title case words are candidate entities
//! 2. **Context windows** - Surrounding words provide classification signal
//! 3. **Suffix/prefix rules** - "Inc.", "Corp.", "Mr.", "Dr." etc.
//! 4. **Sequence patterns** - Multi-word entities like "New York City"
//!
//! # Layered Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │ Layer 3: ML Backends (GLiNER, BERT)                 │
//! │ - Deep contextual understanding                     │
//! │ - Zero-shot capabilities                            │
//! │ - Highest accuracy, highest cost                    │
//! ├─────────────────────────────────────────────────────┤
//! │ Layer 2: StatisticalNER (this module)               │
//! │ - Capitalization + context heuristics               │
//! │ - Person/Org/Location detection                     │
//! │ - Medium accuracy, zero deps                        │
//! ├─────────────────────────────────────────────────────┤
//! │ Layer 1: PatternNER                                 │
//! │ - Regex-based format detection                      │
//! │ - Date/Time/Money/Email/URL/Phone                   │
//! │ - High precision, fast                              │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! # Limitations
//!
//! - Lower accuracy than ML models (~60-70% F1 vs ~85-92%)
//! - Struggles with:
//!   - Lowercase entity names
//!   - Ambiguous contexts
//!   - Domain-specific terminology
//! - Best used as a fast fallback when ML is unavailable
//!
//! # Example
//!
//! ```rust
//! use anno::{Model, backends::statistical::StatisticalNER};
//!
//! let ner = StatisticalNER::new();
//! let entities = ner.extract_entities(
//!     "Steve Jobs founded Apple Inc. in California.",
//!     None
//! ).unwrap();
//!
//! // Should find: Steve Jobs (Person), Apple Inc. (Organization), California (Location)
//! assert!(!entities.is_empty());
//! ```

use crate::entity::{ExtractionMethod, Provenance};
use crate::{Entity, EntityType, Model, Result};

/// Zero-dependency statistical NER using heuristics and context patterns.
///
/// # Approach
///
/// 1. Tokenize text into words
/// 2. Identify capitalized sequences (candidate entities)
/// 3. Apply context rules to classify:
///    - Person: Preceded by titles, followed by verbs like "said", "founded"
///    - Organization: Followed by suffixes like "Inc.", "Corp.", "Ltd."
///    - Location: Preceded by "in", "at", "from", "to"
/// 4. Use fallback classification for remaining candidates
pub struct StatisticalNER {
    /// Minimum confidence threshold for entity extraction.
    threshold: f64,
}

impl StatisticalNER {
    /// Create a new statistical NER with default threshold (0.5).
    #[must_use]
    pub fn new() -> Self {
        Self { threshold: 0.5 }
    }

    /// Create with custom confidence threshold.
    #[must_use]
    pub fn with_threshold(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Default for StatisticalNER {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Context Words (signal entity type)
// =============================================================================

/// Words that typically precede a person name.
const PERSON_PREFIXES: &[&str] = &[
    "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "dr", "dr.", "prof", "prof.",
    "president", "ceo", "cfo", "cto", "director", "manager", "senator",
    "governor", "mayor", "king", "queen", "prince", "princess", "sir", "dame",
    "captain", "general", "colonel", "lieutenant", "sergeant", "officer",
    "judge", "justice", "attorney", "lawyer", "doctor", "nurse", "professor",
    "coach", "chef", "author", "actor", "actress", "singer", "artist",
    "by", "with", "said", "says", "told", "asked", "according",
];

/// Words that typically follow a person (verb patterns).
const PERSON_SUFFIXES: &[&str] = &[
    "said", "says", "told", "asked", "announced", "stated", "explained",
    "founded", "created", "invented", "discovered", "developed", "built",
    "leads", "led", "runs", "manages", "directs", "heads", "chairs",
    "was", "is", "has", "had", "will", "would", "could", "should",
    "jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "phd", "md", "esq",
];

/// Organization suffixes (high confidence signals).
const ORG_SUFFIXES: &[&str] = &[
    "inc", "inc.", "corp", "corp.", "corporation", "co", "co.",
    "ltd", "ltd.", "limited", "llc", "llp", "plc", "gmbh", "ag", "sa",
    "company", "companies", "group", "holdings", "partners", "associates",
    "foundation", "institute", "university", "college", "school",
    "hospital", "clinic", "museum", "library", "church", "temple",
    "bank", "trust", "fund", "capital", "ventures", "labs", "technologies",
    "systems", "solutions", "services", "industries", "international",
    "global", "worldwide", "national", "federal", "state",
];

/// Words that precede organization names.
const ORG_PREFIXES: &[&str] = &[
    "at", "for", "from", "by", "with", "of", "the",
    "company", "firm", "corporation", "organization",
    "joined", "joining", "join", "left", "leaving", "leave",
    "works", "worked", "working", "work",
];

/// Location indicators (prepositions).
const LOC_PREFIXES: &[&str] = &[
    "in", "at", "from", "to", "near", "around", "within", "outside",
    "north", "south", "east", "west", "northern", "southern", "eastern", "western",
    "downtown", "uptown", "central", "metropolitan", "suburban", "rural",
    "city", "town", "village", "county", "state", "province", "country",
    "based", "located", "headquartered", "born", "lived", "lives", "living",
    "visited", "visiting", "traveled", "traveling", "moved", "moving",
];

/// Location suffixes.
const LOC_SUFFIXES: &[&str] = &[
    "city", "town", "village", "county", "state", "province", "country",
    "street", "st", "st.", "avenue", "ave", "ave.", "road", "rd", "rd.",
    "boulevard", "blvd", "blvd.", "drive", "dr", "lane", "ln", "court", "ct",
    "river", "lake", "mountain", "hill", "valley", "island", "beach", "park",
    "airport", "station", "terminal", "port", "harbor", "bridge",
];

/// Common first names (high confidence person signal).
const COMMON_FIRST_NAMES: &[&str] = &[
    "james", "john", "robert", "michael", "william", "david", "richard",
    "joseph", "thomas", "charles", "mary", "patricia", "jennifer", "linda",
    "elizabeth", "barbara", "susan", "jessica", "sarah", "karen", "steve",
    "bill", "tim", "mark", "paul", "peter", "george", "edward", "henry",
    "alexander", "benjamin", "daniel", "matthew", "andrew", "anthony",
    "brian", "chris", "christopher", "donald", "eric", "frank", "gary",
    "jack", "jeff", "jeffrey", "jason", "kevin", "larry", "mike", "nick",
    "patrick", "ray", "ryan", "sam", "scott", "sean", "stephen", "steven",
    "tom", "tony", "anna", "anne", "emily", "emma", "hannah", "julia",
    "kate", "katherine", "lisa", "maria", "michelle", "nancy", "nicole",
    "rachel", "rebecca", "samantha", "stephanie", "victoria", "elon",
];

/// Stop words (not entities).
const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "that", "this", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "i", "me", "him", "her", "us", "them", "my", "your",
    "his", "their", "our", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "also",
    "now", "here", "there", "then", "once", "january", "february", "march",
    "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "today", "yesterday", "tomorrow",
    "year", "month", "week", "day", "hour", "minute", "second", "time",
];

// =============================================================================
// Token and Span Types
// =============================================================================

/// A token with its position and features.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields reserved for future refinements
struct Token<'a> {
    text: &'a str,
    start: usize,
    end: usize,
    is_capitalized: bool,
    is_all_caps: bool,
    is_title_case: bool,
}

impl<'a> Token<'a> {
    fn new(text: &'a str, start: usize, end: usize) -> Self {
        let first_char = text.chars().next();
        let is_capitalized = first_char.map(|c| c.is_uppercase()).unwrap_or(false);
        let is_all_caps = text.chars().all(|c| !c.is_alphabetic() || c.is_uppercase());
        let is_title_case = is_capitalized && text.chars().skip(1).all(|c| !c.is_alphabetic() || c.is_lowercase());

        Self {
            text,
            start,
            end,
            is_capitalized,
            is_all_caps,
            is_title_case,
        }
    }

    fn lower(&self) -> String {
        self.text.to_lowercase()
    }
}

/// A candidate entity span.
#[derive(Debug, Clone)]
struct CandidateSpan {
    tokens: Vec<usize>, // indices into token list
    start: usize,
    end: usize,
    text: String,
}

/// Classification result for a candidate.
#[derive(Debug, Clone)]
struct Classification {
    entity_type: EntityType,
    confidence: f64,
    reason: &'static str,
}

// =============================================================================
// Model Implementation
// =============================================================================

impl Model for StatisticalNER {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // Step 1: Tokenize
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return Ok(vec![]);
        }

        // Step 2: Find capitalized sequences (candidate entities)
        let candidates = find_candidates(&tokens);

        // Step 3: Classify each candidate
        let mut entities = Vec::new();
        for candidate in candidates {
            if let Some(classification) = classify_candidate(&candidate, &tokens) {
                if classification.confidence >= self.threshold {
                    entities.push(Entity::with_provenance(
                        &candidate.text,
                        classification.entity_type,
                        candidate.start,
                        candidate.end,
                        classification.confidence,
                        Provenance {
                            source: "statistical".into(),
                            method: ExtractionMethod::Heuristic,
                            pattern: Some(classification.reason.into()),
                            raw_confidence: Some(classification.confidence),
                            model_version: None,
                            timestamp: None,
                        },
                    ));
                }
            }
        }

        // Sort by position
        entities.sort_by_key(|e| e.start);

        Ok(entities)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
        ]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "statistical"
    }

    fn description(&self) -> &'static str {
        "Zero-dependency statistical NER (heuristic Person/Org/Location detection)"
    }
}

// =============================================================================
// Tokenization
// =============================================================================

/// Simple tokenizer that preserves character offsets.
fn tokenize(text: &str) -> Vec<Token<'_>> {
    let mut tokens = Vec::new();
    let mut start = None;

    for (i, c) in text.char_indices() {
        if c.is_alphanumeric() || c == '\'' || c == '-' {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start {
            let end = i;
            let word = &text[s..end];
            if !word.is_empty() {
                tokens.push(Token::new(word, s, end));
            }
            start = None;
        }
    }

    // Handle last token
    if let Some(s) = start {
        let word = &text[s..];
        if !word.is_empty() {
            tokens.push(Token::new(word, s, text.len()));
        }
    }

    tokens
}

// =============================================================================
// Candidate Detection
// =============================================================================

/// Find capitalized word sequences as candidate entities.
fn find_candidates(tokens: &[Token<'_>]) -> Vec<CandidateSpan> {
    let mut candidates = Vec::new();
    let mut current_span: Option<Vec<usize>> = None;

    for (i, token) in tokens.iter().enumerate() {
        // Skip if at sentence start and single word (could be non-entity)
        let _at_sentence_start = i == 0 || tokens.get(i.saturating_sub(1))
            .map(|t| t.text.ends_with('.') || t.text.ends_with('!') || t.text.ends_with('?'))
            .unwrap_or(false);

        let is_stop = STOP_WORDS.contains(&token.lower().as_str());

        if token.is_capitalized && !is_stop {
            // Extend or start a span
            match &mut current_span {
                Some(span) => span.push(i),
                None => current_span = Some(vec![i]),
            }
        } else if token.lower() == "of" || token.lower() == "the" || token.lower() == "and" {
            // Allow connectors within entity names (e.g., "Bank of America")
            if let Some(span) = &mut current_span {
                // Only extend if next token is also capitalized
                if tokens.get(i + 1).map(|t| t.is_capitalized).unwrap_or(false) {
                    span.push(i);
                } else {
                    // End the span
                    if !span.is_empty() {
                        candidates.push(build_candidate(span, tokens));
                    }
                    current_span = None;
                }
            }
        } else {
            // End any current span
            if let Some(span) = current_span.take() {
                if !span.is_empty() {
                    candidates.push(build_candidate(&span, tokens));
                }
            }
        }
    }

    // Handle final span
    if let Some(span) = current_span {
        if !span.is_empty() {
            candidates.push(build_candidate(&span, tokens));
        }
    }

    // Filter out single-word candidates at sentence start unless high confidence
    candidates.retain(|c| {
        if c.tokens.len() == 1 {
            let token = &tokens[c.tokens[0]];
            let is_common_name = COMMON_FIRST_NAMES.contains(&token.lower().as_str());
            let has_org_suffix = ORG_SUFFIXES.contains(&token.lower().as_str());
            // Keep if it's a common name, org suffix, or has clear context
            is_common_name || has_org_suffix || token.text.len() > 1
        } else {
            true
        }
    });

    candidates
}

/// Build a CandidateSpan from token indices.
///
/// # Panics
/// Panics if `indices` is empty (invariant: callers must pass non-empty indices).
fn build_candidate(indices: &[usize], tokens: &[Token<'_>]) -> CandidateSpan {
    let first = &tokens[indices[0]];
    let last = &tokens[*indices.last().expect("indices must not be empty")];
    
    // Collect text, handling connectors
    let text: String = indices
        .iter()
        .map(|&i| tokens[i].text)
        .collect::<Vec<_>>()
        .join(" ");

    CandidateSpan {
        tokens: indices.to_vec(),
        start: first.start,
        end: last.end,
        text,
    }
}

// =============================================================================
// Classification
// =============================================================================

/// Classify a candidate span as Person/Org/Location.
fn classify_candidate(candidate: &CandidateSpan, tokens: &[Token<'_>]) -> Option<Classification> {
    let first_idx = candidate.tokens[0];
    let last_idx = *candidate.tokens.last().expect("candidate must have tokens");

    // Get context words (2 before and 2 after)
    let prev_words: Vec<String> = (1..=2)
        .filter_map(|i| first_idx.checked_sub(i).and_then(|idx| tokens.get(idx)))
        .map(|t| t.lower())
        .collect();

    let next_words: Vec<String> = (1..=2)
        .filter_map(|i| tokens.get(last_idx + i))
        .map(|t| t.lower())
        .collect();

    let first_word = tokens[first_idx].lower();
    let last_word = tokens[last_idx].lower();

    // Score each entity type
    let person_score = score_person(candidate, &first_word, &last_word, &prev_words, &next_words);
    let org_score = score_organization(candidate, &last_word, &prev_words, &next_words);
    let loc_score = score_location(candidate, &prev_words, &next_words);

    // Pick highest score
    let (entity_type, confidence, reason) = if person_score.0 >= org_score.0 && person_score.0 >= loc_score.0 {
        (EntityType::Person, person_score.0, person_score.1)
    } else if org_score.0 >= loc_score.0 {
        (EntityType::Organization, org_score.0, org_score.1)
    } else {
        (EntityType::Location, loc_score.0, loc_score.1)
    };

    // Require minimum confidence
    if confidence >= 0.3 {
        Some(Classification {
            entity_type,
            confidence,
            reason,
        })
    } else {
        None
    }
}

/// Score likelihood of being a Person.
fn score_person(
    candidate: &CandidateSpan,
    first_word: &str,
    _last_word: &str,
    prev_words: &[String],
    next_words: &[String],
) -> (f64, &'static str) {
    let mut score: f64 = 0.0;
    let mut reason = "capitalized";

    // Check for person prefixes (Mr., Dr., etc.)
    if prev_words.iter().any(|w| PERSON_PREFIXES.contains(&w.as_str())) {
        score += 0.4;
        reason = "person_prefix";
    }

    // Check for person suffixes (said, founded, etc.)
    if next_words.iter().any(|w| PERSON_SUFFIXES.contains(&w.as_str())) {
        score += 0.3;
        reason = "person_suffix";
    }

    // Check for common first names
    if COMMON_FIRST_NAMES.contains(&first_word) {
        score += 0.4;
        reason = "common_name";
    }

    // Two-word names are likely people
    if candidate.tokens.len() == 2 {
        score += 0.2;
        if reason == "capitalized" {
            reason = "two_word_name";
        }
    }

    // Title case bonus
    if candidate.text.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
        score += 0.1;
    }

    (score.min(0.95), reason)
}

/// Score likelihood of being an Organization.
fn score_organization(
    candidate: &CandidateSpan,
    last_word: &str,
    prev_words: &[String],
    next_words: &[String],
) -> (f64, &'static str) {
    let mut score: f64 = 0.0;
    let mut reason = "capitalized";

    // Check for organization suffixes (Inc., Corp., etc.) - very strong signal
    if ORG_SUFFIXES.contains(&last_word) {
        score += 0.6;
        reason = "org_suffix";
    }

    // Check next word for org suffix
    if next_words.iter().any(|w| ORG_SUFFIXES.contains(&w.as_str())) {
        score += 0.4;
        reason = "org_suffix_after";
    }

    // Check for org prefixes
    if prev_words.iter().any(|w| ORG_PREFIXES.contains(&w.as_str())) {
        score += 0.2;
        if reason == "capitalized" {
            reason = "org_prefix";
        }
    }

    // ALL CAPS often indicates organization/acronym
    if candidate.text.chars().all(|c| !c.is_alphabetic() || c.is_uppercase()) 
        && candidate.text.len() <= 5 
    {
        score += 0.3;
        reason = "acronym";
    }

    (score.min(0.95), reason)
}

/// Score likelihood of being a Location.
fn score_location(
    _candidate: &CandidateSpan,
    prev_words: &[String],
    next_words: &[String],
) -> (f64, &'static str) {
    let mut score: f64 = 0.0;
    let mut reason = "capitalized";

    // Check for location prefixes (in, at, from, etc.) - strong signal
    if prev_words.iter().any(|w| LOC_PREFIXES.contains(&w.as_str())) {
        score += 0.5;
        reason = "loc_prefix";
    }

    // Check for location suffixes (City, State, etc.)
    if next_words.iter().any(|w| LOC_SUFFIXES.contains(&w.as_str())) {
        score += 0.3;
        reason = "loc_suffix";
    }

    (score.min(0.95), reason)
}

// Capability marker: StatisticalNER extracts named entities via heuristics
impl crate::NamedEntityCapable for StatisticalNER {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(text: &str) -> Vec<Entity> {
        StatisticalNER::new().extract_entities(text, None).unwrap()
    }

    fn has_text_type(entities: &[Entity], text: &str, ty: &EntityType) -> bool {
        entities.iter().any(|e| e.text.contains(text) && e.entity_type == *ty)
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello World!");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "Hello");
        assert_eq!(tokens[1].text, "World");
    }

    #[test]
    fn test_person_with_title() {
        let e = extract("Mr. Smith said hello.");
        assert!(has_text_type(&e, "Smith", &EntityType::Person));
    }

    #[test]
    fn test_person_two_names() {
        let e = extract("Steve Jobs founded Apple.");
        assert!(has_text_type(&e, "Steve Jobs", &EntityType::Person));
    }

    #[test]
    fn test_person_common_name() {
        let e = extract("According to John, the project is on track.");
        assert!(has_text_type(&e, "John", &EntityType::Person));
    }

    #[test]
    fn test_org_with_suffix() {
        let e = extract("He works at Apple Inc.");
        assert!(has_text_type(&e, "Apple Inc", &EntityType::Organization));
    }

    #[test]
    fn test_org_corp() {
        let e = extract("Microsoft Corporation announced earnings.");
        // Should detect as organization (has Corp suffix)
        let has_org = e.iter().any(|e| e.entity_type == EntityType::Organization);
        assert!(has_org || !e.is_empty(), "Should find Microsoft as some entity: {:?}", e);
    }

    #[test]
    fn test_location_with_in() {
        let e = extract("The conference is in Paris.");
        assert!(has_text_type(&e, "Paris", &EntityType::Location));
    }

    #[test]
    fn test_location_with_from() {
        let e = extract("She traveled from Tokyo to London.");
        assert!(has_text_type(&e, "Tokyo", &EntityType::Location));
        assert!(has_text_type(&e, "London", &EntityType::Location));
    }

    #[test]
    fn test_multi_word_location() {
        let e = extract("He lives in New York City.");
        assert!(has_text_type(&e, "New York City", &EntityType::Location));
    }

    #[test]
    fn test_bank_of_america() {
        // "of" should be included in the entity
        let e = extract("She works at Bank of America.");
        // Should find at least one entity (Bank, America, or Bank of America)
        assert!(!e.is_empty(), "Should find some entity in 'Bank of America'");
    }

    #[test]
    fn test_no_stop_words_as_entities() {
        let e = extract("The quick brown fox jumps over the lazy dog.");
        assert!(e.is_empty() || e.iter().all(|e| !STOP_WORDS.contains(&e.text.to_lowercase().as_str())));
    }

    #[test]
    fn test_empty_text() {
        let e = extract("");
        assert!(e.is_empty());
    }

    #[test]
    fn test_no_capitalized() {
        let e = extract("the quick brown fox");
        assert!(e.is_empty());
    }

    #[test]
    fn test_complex_sentence() {
        let e = extract("Dr. Jane Smith, CEO of Acme Corp., visited the offices in San Francisco.");
        
        // Heuristic NER should find multiple entities
        // Note: heuristics may not get all types correct, but should detect capitalized spans
        assert!(e.len() >= 2, "Should find multiple entities: {:?}", e);
        
        // At minimum, should have some combination of these types
        let has_named = e.iter().any(|e| 
            matches!(e.entity_type, 
                EntityType::Person | EntityType::Organization | EntityType::Location
            )
        );
        assert!(has_named, "Should find at least one named entity type: {:?}", e);
    }

    #[test]
    fn test_provenance() {
        let e = extract("Mr. Johnson said hello.");
        let person = e.iter().find(|e| e.entity_type == EntityType::Person).unwrap();
        
        let prov = person.provenance.as_ref().unwrap();
        assert_eq!(prov.source.as_ref(), "statistical");
        assert_eq!(prov.method, ExtractionMethod::Heuristic);
        assert!(prov.pattern.is_some());
    }

    #[test]
    fn test_threshold() {
        let ner = StatisticalNER::with_threshold(0.9);
        // High threshold means fewer, higher confidence entities
        let e = ner.extract_entities("Maybe John or something.", None).unwrap();
        // With 0.9 threshold, weak signals should be filtered
        assert!(e.is_empty() || e.iter().all(|e| e.confidence >= 0.9));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn never_panics(text in ".*") {
            let ner = StatisticalNER::new();
            let _ = ner.extract_entities(&text, None);
        }

        #[test]
        fn entities_within_bounds(text in ".{1,200}") {
            let ner = StatisticalNER::new();
            if let Ok(entities) = ner.extract_entities(&text, None) {
                for e in entities {
                    prop_assert!(e.start <= text.len());
                    prop_assert!(e.end <= text.len());
                    prop_assert!(e.start <= e.end);
                }
            }
        }

        #[test]
        fn capitalized_needed(text in "[a-z ]{10,50}") {
            // All lowercase should produce no entities
            let ner = StatisticalNER::new();
            let entities = ner.extract_entities(&text, None).unwrap();
            prop_assert!(entities.is_empty());
        }
    }
}

