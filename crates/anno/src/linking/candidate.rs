//! Candidate generation for entity linking.
//!
//! High-recall retrieval of potential KB entries for a mention.
//!
//! # Similarity Metrics
//!
//! The candidate generator supports multiple string similarity metrics
//! for fuzzy matching. The default is Jaccard (word + trigram), but
//! edit distance variants are useful for:
//!
//! - **Typo correction**: Damerau-Levenshtein handles transpositions
//! - **OCR/damaged text**: Wildcard edit distance matches partial patterns
//! - **Cross-script matching**: Edit distance works character-by-character
//!
//! # Research Context
//!
//! Edit distance with wildcards is particularly important for ancient/historical
//! text processing where inscriptions may be damaged or illegible.
//! See Tamburini (2025) on decipherment of ancient scripts.

use crate::edit_distance;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Source of a candidate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum CandidateSource {
    /// Wikidata - Most comprehensive, actively maintained
    #[default]
    Wikidata,
    /// YAGO - Wikipedia + WordNet + GeoNames ontology
    YAGO,
    /// DBpedia - Wikipedia infobox extraction
    DBpedia,
    /// Wikipedia - Direct article links
    Wikipedia,
    /// Freebase - Legacy (deprecated 2016, mapped to Wikidata)
    Freebase,
    /// UMLS - Unified Medical Language System (biomedical)
    UMLS,
    /// GeoNames - Geographic entities
    GeoNames,
    /// Custom knowledge base
    Custom(String),
}

// =============================================================================
// Similarity Metrics
// =============================================================================

/// Similarity metric for candidate matching.
///
/// Different metrics are suited to different use cases:
///
/// | Metric | Best For | Speed |
/// |--------|----------|-------|
/// | Jaccard | General text, multi-word entities | Fast |
/// | EditDistance | Typo correction, single words | Medium |
/// | DamerauLevenshtein | Keyboard typos (transpositions) | Medium |
/// | EditDistanceWildcard | OCR/damaged text | Slower |
///
/// # Research Context
///
/// The wildcard edit distance (Tamburini 2025) is particularly useful for
/// computational philology and ancient language processing where texts
/// may have illegible portions marked with `?` (single char) or `*` (multi char).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SimilarityMetric {
    /// Jaccard similarity on words + character trigrams (default).
    ///
    /// Best for general text with multi-word entity names.
    /// Fast but doesn't handle character-level edits well.
    #[default]
    Jaccard,

    /// Normalized Levenshtein edit distance.
    ///
    /// Counts insertions, deletions, substitutions.
    /// Better for single-word matching and typo detection.
    EditDistance,

    /// Damerau-Levenshtein distance.
    ///
    /// Like edit distance but counts adjacent transpositions
    /// (e.g., "teh" → "the") as single edits.
    /// Better for keyboard typo correction.
    DamerauLevenshtein,

    /// Edit distance with wildcards.
    ///
    /// Supports `?` (match one char) and `*` (match zero or more).
    /// Essential for damaged/OCR'd historical text.
    ///
    /// **Note**: Wildcards only work in the **mention** (query), not the candidate.
    EditDistanceWildcard,
}

impl SimilarityMetric {
    /// Compute similarity score between two strings using this metric.
    ///
    /// Returns a value in [0.0, 1.0] where 1.0 = identical.
    #[must_use]
    pub fn compute(&self, a: &str, b: &str) -> f64 {
        match self {
            SimilarityMetric::Jaccard => string_similarity(a, b),
            SimilarityMetric::EditDistance => edit_distance::edit_similarity(a, b),
            SimilarityMetric::DamerauLevenshtein => {
                // Normalize Damerau-Levenshtein to [0, 1] similarity
                let dist = edit_distance::damerau_levenshtein(a, b);
                let max_len = a.chars().count().max(b.chars().count());
                if max_len == 0 {
                    1.0
                } else {
                    1.0 - (dist as f64 / max_len as f64)
                }
            }
            SimilarityMetric::EditDistanceWildcard => {
                edit_distance::edit_similarity_wildcards(a, b)
            }
        }
    }

    /// Get a human-readable name for this metric.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            SimilarityMetric::Jaccard => "jaccard",
            SimilarityMetric::EditDistance => "edit-distance",
            SimilarityMetric::DamerauLevenshtein => "damerau-levenshtein",
            SimilarityMetric::EditDistanceWildcard => "edit-distance-wildcard",
        }
    }

    /// Parse from string (for CLI).
    ///
    /// Note: This is not the standard `FromStr::from_str` trait method.
    pub fn parse_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "jaccard" | "jac" => Some(SimilarityMetric::Jaccard),
            "edit-distance" | "edit" | "levenshtein" | "lev" => {
                Some(SimilarityMetric::EditDistance)
            }
            "damerau-levenshtein" | "damerau" | "dl" => Some(SimilarityMetric::DamerauLevenshtein),
            "edit-distance-wildcard" | "wildcard" | "edw" => {
                Some(SimilarityMetric::EditDistanceWildcard)
            }
            _ => None,
        }
    }
}

/// A candidate KB entry for a mention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    /// KB identifier (e.g., "Q937" for Wikidata)
    pub kb_id: String,
    /// Source knowledge base
    pub source: CandidateSource,
    /// Canonical name/label
    pub label: String,
    /// Aliases/alternate names
    pub aliases: Vec<String>,
    /// Description/gloss
    pub description: Option<String>,
    /// Entity type from KB (e.g., "human", "organization")
    pub kb_type: Option<String>,
    /// Wikipedia sitelink count (popularity proxy)
    pub sitelinks: Option<u32>,
    /// Prior probability (if known)
    pub prior: f64,
    /// String similarity to mention
    pub string_sim: f64,
    /// Type compatibility score
    pub type_score: f64,
    /// Overall candidate score (for ranking)
    pub score: f64,
    /// Temporal validity start (ISO 8601 date string).
    ///
    /// For people: birth date. For organizations: founding date.
    /// Critical for historical document disambiguation where
    /// "President Bush" could refer to different people depending
    /// on the document date (Arora et al. 2024).
    pub valid_from: Option<String>,
    /// Temporal validity end (ISO 8601 date string).
    ///
    /// For people: death date. For organizations: dissolution date.
    pub valid_until: Option<String>,
}

impl Candidate {
    /// Create a new candidate.
    pub fn new(kb_id: &str, source: CandidateSource, label: &str) -> Self {
        Self {
            kb_id: kb_id.to_string(),
            source,
            label: label.to_string(),
            aliases: Vec::new(),
            description: None,
            kb_type: None,
            sitelinks: None,
            prior: 0.0,
            string_sim: 0.0,
            type_score: 1.0,
            score: 0.0,
            valid_from: None,
            valid_until: None,
        }
    }

    /// Set temporal validity start.
    pub fn with_valid_from(mut self, date: &str) -> Self {
        self.valid_from = Some(date.to_string());
        self
    }

    /// Set temporal validity end.
    pub fn with_valid_until(mut self, date: &str) -> Self {
        self.valid_until = Some(date.to_string());
        self
    }

    /// Add an alias.
    pub fn with_alias(mut self, alias: &str) -> Self {
        self.aliases.push(alias.to_string());
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Set KB type.
    pub fn with_kb_type(mut self, kb_type: &str) -> Self {
        self.kb_type = Some(kb_type.to_string());
        self
    }

    /// Set prior.
    pub fn with_prior(mut self, prior: f64) -> Self {
        self.prior = prior;
        self
    }

    /// Compute overall score.
    pub fn compute_score(&mut self) {
        // Weighted combination of signals
        self.score = 0.4 * self.string_sim
            + 0.3 * self.prior
            + 0.2 * self.type_score
            + 0.1
                * self
                    .sitelinks
                    .map(|s| (s as f64).log10() / 7.0)
                    .unwrap_or(0.0);
    }

    /// Compute score with temporal context.
    ///
    /// For historical documents, temporal compatibility is critical for
    /// disambiguation. "President Bush" in 1990 refers to George H.W. Bush,
    /// while in 2005 it refers to George W. Bush.
    ///
    /// # Arguments
    /// * `document_date` - ISO 8601 date string (e.g., "1990-01-15")
    pub fn compute_score_with_temporal(&mut self, document_date: Option<&str>) {
        // Base score
        self.compute_score();

        // Apply temporal penalty if document date is known
        if let Some(doc_date) = document_date {
            let temporal_score = self.temporal_compatibility(doc_date);
            // Temporal compatibility can reduce score by up to 50%
            self.score *= 0.5 + 0.5 * temporal_score;
        }
    }

    /// Check temporal compatibility with a document date.
    ///
    /// Returns 1.0 if the candidate is valid at the document date,
    /// 0.0 if clearly invalid, and intermediate values for uncertainty.
    pub fn temporal_compatibility(&self, document_date: &str) -> f64 {
        // Parse document date (just year for simplicity)
        let doc_year = parse_year(document_date);

        // Check if entity was "active" at document date
        let from_year = self.valid_from.as_deref().and_then(parse_year);
        let until_year = self.valid_until.as_deref().and_then(parse_year);

        match (from_year, until_year, doc_year) {
            // Can't determine temporal compatibility
            (None, None, _) | (_, _, None) => 1.0,

            // Entity not yet "born" at document date
            (Some(from), _, Some(doc)) if doc < from => {
                // Graduated penalty: 10 years before birth = 0.5
                let years_before = from - doc;
                (1.0 - years_before as f64 / 20.0).max(0.1)
            }

            // Entity "dead" before document date
            (_, Some(until), Some(doc)) if doc > until => {
                // Graduated penalty: 10 years after death = 0.5
                let years_after = doc - until;
                (1.0 - years_after as f64 / 20.0).max(0.1)
            }

            // Entity active at document date
            _ => 1.0,
        }
    }

    /// Get IRI/URI for this candidate.
    pub fn to_iri(&self) -> String {
        match &self.source {
            CandidateSource::Wikidata => {
                format!("http://www.wikidata.org/entity/{}", self.kb_id)
            }
            CandidateSource::YAGO => {
                format!("http://yago-knowledge.org/resource/{}", self.kb_id)
            }
            CandidateSource::DBpedia => {
                format!("http://dbpedia.org/resource/{}", self.kb_id)
            }
            CandidateSource::Wikipedia => {
                format!("https://en.wikipedia.org/wiki/{}", self.kb_id)
            }
            CandidateSource::Freebase => {
                format!("http://rdf.freebase.com/ns/{}", self.kb_id)
            }
            CandidateSource::UMLS => {
                format!("https://uts.nlm.nih.gov/uts/umls/concept/{}", self.kb_id)
            }
            CandidateSource::GeoNames => {
                format!("https://sws.geonames.org/{}/", self.kb_id)
            }
            CandidateSource::Custom(name) => {
                format!("{}:{}", name, self.kb_id)
            }
        }
    }

    /// Get CURIE (Compact URI) for this candidate.
    pub fn to_curie(&self) -> String {
        let prefix = match &self.source {
            CandidateSource::Wikidata => "wd",
            CandidateSource::YAGO => "yago",
            CandidateSource::DBpedia => "dbr",
            CandidateSource::Wikipedia => "wp",
            CandidateSource::Freebase => "fb",
            CandidateSource::UMLS => "umls",
            CandidateSource::GeoNames => "gn",
            CandidateSource::Custom(name) => name,
        };
        format!("{}:{}", prefix, self.kb_id)
    }
}

/// Trait for candidate generators.
pub trait CandidateGenerator: Send + Sync {
    /// Generate candidates for a mention.
    ///
    /// # Parameters
    /// - `mention`: The mention text
    /// - `context`: Surrounding text context
    /// - `entity_type`: Optional NER type constraint
    /// - `limit`: Maximum candidates to return
    fn generate(
        &self,
        mention: &str,
        context: &str,
        entity_type: Option<&str>,
        limit: usize,
    ) -> Vec<Candidate>;

    /// Name of this generator.
    fn name(&self) -> &'static str;
}

/// In-memory candidate generator using a preloaded dictionary.
///
/// Suitable for small KBs or testing.
///
/// # Similarity Metrics
///
/// The generator supports multiple similarity metrics via `with_metric()`:
///
/// ```rust
/// use anno::linking::{DictionaryCandidateGenerator, SimilarityMetric};
///
/// let gen = DictionaryCandidateGenerator::new()
///     .with_metric(SimilarityMetric::EditDistance)
///     .with_well_known();
///
/// // For OCR'd text with damage markers:
/// let gen_ocr = DictionaryCandidateGenerator::new()
///     .with_metric(SimilarityMetric::EditDistanceWildcard)
///     .with_well_known();
/// ```
#[derive(Debug, Clone, Default)]
pub struct DictionaryCandidateGenerator {
    /// Map from normalized surface form to candidates
    entries: HashMap<String, Vec<Candidate>>,
    /// Similarity metric for fuzzy matching
    metric: SimilarityMetric,
}

impl DictionaryCandidateGenerator {
    /// Create a new dictionary generator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the similarity metric for fuzzy matching.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno::linking::{DictionaryCandidateGenerator, SimilarityMetric};
    ///
    /// // For damaged historical text with wildcards
    /// let gen = DictionaryCandidateGenerator::new()
    ///     .with_metric(SimilarityMetric::EditDistanceWildcard);
    /// ```
    pub fn with_metric(mut self, metric: SimilarityMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Get the current similarity metric.
    #[must_use]
    pub fn metric(&self) -> SimilarityMetric {
        self.metric
    }

    /// Add an entry to the dictionary.
    pub fn add_entry(&mut self, surface: &str, candidate: Candidate) {
        let normalized = surface.to_lowercase();
        self.entries.entry(normalized).or_default().push(candidate);
    }

    /// Load well-known entities (for demo/testing).
    ///
    /// Includes diverse entities across cultures and languages per multilingual guidelines.
    /// Coverage: Western, East Asian, Middle Eastern, African, South Asian, Latin American.
    pub fn with_well_known(mut self) -> Self {
        let well_known = [
            // === Scientists (diverse origins) ===
            ("albert einstein", "Q937", "theoretical physicist"),
            ("marie curie", "Q7186", "physicist and chemist"),
            (
                "tu youyou",
                "Q546079",
                "Chinese pharmacologist, Nobel laureate",
            ),
            ("屠呦呦", "Q546079", "Chinese pharmacologist"),
            ("c.v. raman", "Q201010", "Indian physicist, Nobel laureate"),
            (
                "abdus salam",
                "Q108365",
                "Pakistani physicist, Nobel laureate",
            ),
            (
                "wangari maathai",
                "Q180728",
                "Kenyan environmentalist, Nobel laureate",
            ),
            // === Political figures (global) ===
            ("barack obama", "Q76", "44th President of the United States"),
            ("angela merkel", "Q567", "Chancellor of Germany"),
            ("習近平", "Q15031", "General Secretary of CCP"),
            ("xi jinping", "Q15031", "General Secretary of CCP"),
            ("narendra modi", "Q1058", "Prime Minister of India"),
            ("नरेन्द्र मोदी", "Q1058", "Prime Minister of India"),
            ("محمد بن سلمان", "Q6889872", "Crown Prince of Saudi Arabia"),
            (
                "mohammed bin salman",
                "Q6889872",
                "Crown Prince of Saudi Arabia",
            ),
            ("cyril ramaphosa", "Q312910", "President of South Africa"),
            ("lula da silva", "Q37181", "President of Brazil"),
            // === Technology companies (global) ===
            ("google", "Q95", "American technology company"),
            ("apple", "Q312", "American technology company"),
            ("microsoft", "Q2283", "American technology company"),
            ("alibaba", "Q306717", "Chinese technology company"),
            ("阿里巴巴", "Q306717", "Chinese technology company"),
            ("tencent", "Q860580", "Chinese technology company"),
            ("腾讯", "Q860580", "Chinese technology company"),
            ("samsung", "Q20718", "South Korean conglomerate"),
            ("삼성", "Q20718", "South Korean conglomerate"),
            ("tata", "Q752289", "Indian conglomerate"),
            ("infosys", "Q723418", "Indian technology company"),
            // === Cities (global coverage) ===
            ("new york", "Q60", "city in New York State"),
            ("london", "Q84", "capital of the United Kingdom"),
            ("paris", "Q90", "capital of France"),
            ("berlin", "Q64", "capital of Germany"),
            ("tokyo", "Q1490", "capital of Japan"),
            ("東京", "Q1490", "capital of Japan"),
            ("beijing", "Q956", "capital of China"),
            ("北京", "Q956", "capital of China"),
            ("mumbai", "Q1156", "financial capital of India"),
            ("मुंबई", "Q1156", "financial capital of India"),
            ("cairo", "Q85", "capital of Egypt"),
            ("القاهرة", "Q85", "capital of Egypt"),
            ("são paulo", "Q174", "largest city in Brazil"),
            ("lagos", "Q8673", "largest city in Nigeria"),
            ("москва", "Q649", "capital of Russia"),
            ("moscow", "Q649", "capital of Russia"),
            ("dubai", "Q612", "city in UAE"),
            ("دبي", "Q612", "city in UAE"),
            ("singapore", "Q334", "city-state in Southeast Asia"),
            ("新加坡", "Q334", "city-state in Southeast Asia"),
            // === International organizations ===
            ("united nations", "Q1065", "international organization"),
            ("european union", "Q458", "political and economic union"),
            (
                "world health organization",
                "Q7817",
                "UN specialized agency",
            ),
            ("who", "Q7817", "World Health Organization"),
            ("nato", "Q7184", "North Atlantic Treaty Organization"),
            (
                "african union",
                "Q7159",
                "continental union of African states",
            ),
            ("asean", "Q7768", "Association of Southeast Asian Nations"),
            (
                "opec",
                "Q7795",
                "Organization of the Petroleum Exporting Countries",
            ),
            // === Historical figures (diverse) ===
            ("confucius", "Q4604", "Chinese philosopher"),
            ("孔子", "Q4604", "Chinese philosopher"),
            ("mahatma gandhi", "Q1001", "Indian independence leader"),
            ("महात्मा गांधी", "Q1001", "Indian independence leader"),
            (
                "nelson mandela",
                "Q8023",
                "South African anti-apartheid leader",
            ),
            ("cleopatra", "Q635", "last Pharaoh of Egypt"),
            ("genghis khan", "Q720", "founder of Mongol Empire"),
            ("成吉思汗", "Q720", "founder of Mongol Empire"),
            // === Cultural figures (diverse) ===
            ("pelé", "Q12897", "Brazilian footballer"),
            ("shakira", "Q34424", "Colombian singer"),
            ("bts", "Q485927", "South Korean boy band"),
            ("방탄소년단", "Q485927", "South Korean boy band"),
            ("宮崎駿", "Q55400", "Japanese animator"),
            ("hayao miyazaki", "Q55400", "Japanese animator"),
        ];

        for (surface, qid, desc) in well_known {
            let candidate = Candidate::new(qid, CandidateSource::Wikidata, surface)
                .with_description(desc)
                .with_prior(0.5);
            self.add_entry(surface, candidate);
        }

        self
    }
}

impl CandidateGenerator for DictionaryCandidateGenerator {
    fn generate(
        &self,
        mention: &str,
        _context: &str,
        _entity_type: Option<&str>,
        limit: usize,
    ) -> Vec<Candidate> {
        let normalized = mention.to_lowercase();

        // Exact match (skip wildcards for exact matching)
        if !mention.contains('?') && !mention.contains('*') {
            if let Some(candidates) = self.entries.get(&normalized) {
                return candidates.iter().take(limit).cloned().collect();
            }
        }

        // Fuzzy match using configured similarity metric
        let mut results: Vec<Candidate> =
            self.entries.iter().flat_map(|(_, v)| v.clone()).collect();

        // Score by configured similarity metric
        for c in &mut results {
            c.string_sim = self.metric.compute(mention, &c.label);
            c.compute_score();
        }

        // Filter out very low similarity matches
        results.retain(|c| c.string_sim > 0.1);

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        results
    }

    fn name(&self) -> &'static str {
        "dictionary"
    }
}

/// Compute simple string similarity (Jaccard on words + char trigrams).
pub fn string_similarity(a: &str, b: &str) -> f64 {
    // Keep this module's policy (weights) local, but reuse shared primitives.
    //
    // Default: delegate to `textprep` (self-contained within the `anno` repo).
    // Opt-in: `--features gramdex` uses `gramdex::trigram_jaccard` for the trigram
    // component so we can compare performance and semantics in benches.
    #[cfg(feature = "gramdex")]
    {
        string_similarity_gramdex(a, b)
    }
    #[cfg(not(feature = "gramdex"))]
    {
        string_similarity_textprep(a, b)
    }
}

/// `textprep`-backed implementation of [`string_similarity`].
#[must_use]
pub fn string_similarity_textprep(a: &str, b: &str) -> f64 {
    textprep::similarity::weighted_word_char_ngram_jaccard(a, b, 3, 0.6, 0.4)
}

/// `gramdex`-backed implementation of [`string_similarity`].
#[cfg(feature = "gramdex")]
#[must_use]
pub fn string_similarity_gramdex(a: &str, b: &str) -> f64 {
    // Mirror the `textprep` weighted policy (word Jaccard + char trigram Jaccard).
    //
    // Note: `gramdex::trigram_jaccard`'s convention is 1.0 when both inputs have
    // zero trigrams (length < 3). We keep this behind a feature until we decide
    // whether we want to match `textprep` semantics exactly for short strings.
    let word = textprep::similarity::word_jaccard(a, b);
    let tri = gramdex::trigram_jaccard(a, b) as f64;
    0.6 * word + 0.4 * tri
}

/// Parse year from an ISO 8601 date string.
///
/// Handles common formats:
/// - "1990-01-15" → Some(1990)
/// - "1990" → Some(1990)
/// - "-0044" → Some(-44) (BCE dates)
fn parse_year(date: &str) -> Option<i32> {
    let trimmed = date.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Handle BCE dates (negative years)
    let (sign, rest) = if let Some(rest) = trimmed.strip_prefix('-') {
        (-1, rest)
    } else {
        (1, trimmed)
    };

    // Extract year portion (first 4 digits or until first dash)
    let year_str = rest.split('-').next()?;
    let year: i32 = year_str.parse().ok()?;
    Some(sign * year)
}

/// Type compatibility scoring.
pub fn type_compatibility(ner_type: Option<&str>, kb_type: Option<&str>) -> f64 {
    match (ner_type, kb_type) {
        (None, _) | (_, None) => 1.0, // No constraint
        (Some(n), Some(k)) => {
            let n_lower = n.to_lowercase();
            let k_lower = k.to_lowercase();

            // Direct match
            if n_lower == k_lower {
                return 1.0;
            }

            // Person type compatibility
            if (n_lower.contains("person") || n_lower == "per")
                && (k_lower.contains("human") || k_lower.contains("person"))
            {
                return 0.95;
            }

            // Organization type compatibility
            if (n_lower.contains("org") || n_lower == "organization")
                && (k_lower.contains("organization")
                    || k_lower.contains("company")
                    || k_lower.contains("institution"))
            {
                return 0.9;
            }

            // Location type compatibility
            if (n_lower.contains("loc") || n_lower.contains("gpe") || n_lower == "location")
                && (k_lower.contains("city")
                    || k_lower.contains("country")
                    || k_lower.contains("place")
                    || k_lower.contains("location"))
            {
                return 0.9;
            }

            // Mismatch
            0.3
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_generator() {
        let gen = DictionaryCandidateGenerator::new().with_well_known();

        // Exact match
        let candidates = gen.generate("albert einstein", "", None, 5);
        assert!(!candidates.is_empty());
        assert!(candidates[0].kb_id == "Q937");

        // Partial match might not work depending on fuzzy logic
        // The generator is designed for exact/close matches
        let partial = gen.generate("Einstein", "", None, 5);
        // Partial may or may not match - that's OK for dictionary-based
        let _ = partial; // Just verify it doesn't panic
    }

    #[test]
    fn test_string_similarity() {
        // Keep the baseline behavior stable even if `--features gramdex` is enabled.
        assert!(string_similarity_textprep("Albert Einstein", "Einstein") > 0.3);
        assert!(string_similarity_textprep("Albert Einstein", "Albert Einstein") > 0.99);
        assert!(string_similarity_textprep("New York", "New York City") > 0.5);
    }

    #[cfg(feature = "gramdex")]
    #[test]
    fn test_string_similarity_gramdex_bounds() {
        let sim = string_similarity_gramdex("Albert Einstein", "Einstein");
        assert!((0.0..=1.0).contains(&sim));
    }

    #[test]
    fn test_type_compatibility() {
        assert!(type_compatibility(Some("PERSON"), Some("human")) > 0.9);
        assert!(type_compatibility(Some("ORG"), Some("company")) > 0.8);
        assert!(type_compatibility(Some("PERSON"), Some("city")) < 0.5);
    }

    #[test]
    fn test_candidate_iri() {
        let c = Candidate::new("Q937", CandidateSource::Wikidata, "Einstein");
        assert_eq!(c.to_iri(), "http://www.wikidata.org/entity/Q937");
    }

    #[test]
    fn test_parse_year() {
        assert_eq!(parse_year("1990-01-15"), Some(1990));
        assert_eq!(parse_year("1990"), Some(1990));
        assert_eq!(parse_year("-0044"), Some(-44)); // Julius Caesar
        assert_eq!(parse_year(""), None);
    }

    #[test]
    fn test_temporal_compatibility() {
        // George H.W. Bush: 1924-2018
        let bush_sr = Candidate::new("Q23505", CandidateSource::Wikidata, "George H. W. Bush")
            .with_valid_from("1924-06-12")
            .with_valid_until("2018-11-30");

        // George W. Bush: 1946-present
        let bush_jr = Candidate::new("Q207", CandidateSource::Wikidata, "George W. Bush")
            .with_valid_from("1946-07-06");

        // In 1990, both were alive - no temporal penalty
        assert!(bush_sr.temporal_compatibility("1990-01-01") > 0.9);
        assert!(bush_jr.temporal_compatibility("1990-01-01") > 0.9);

        // In 2020, Bush Sr. had died 2 years prior - slight penalty
        let sr_compat_2020 = bush_sr.temporal_compatibility("2020-01-01");
        assert!(sr_compat_2020 < 1.0);
        assert!(sr_compat_2020 > 0.5);

        // Bush Jr. still alive - no penalty
        assert!(bush_jr.temporal_compatibility("2020-01-01") > 0.9);
    }

    #[test]
    fn test_compute_score_with_temporal() {
        // Historical figure: Julius Caesar
        let mut caesar = Candidate::new("Q1048", CandidateSource::Wikidata, "Julius Caesar")
            .with_valid_from("-0100-07-12")
            .with_valid_until("-0044-03-15")
            .with_prior(0.9);
        caesar.string_sim = 0.9;

        // Score without temporal context
        caesar.compute_score();
        let base_score = caesar.score;

        // Score with document from 50 BCE (Caesar was alive)
        caesar.compute_score_with_temporal(Some("-0050-01-01"));
        let ancient_score = caesar.score;

        // Score with modern document (Caesar long dead)
        caesar.compute_score_with_temporal(Some("2024-01-01"));
        let modern_score = caesar.score;

        // Ancient document should have higher score than modern
        assert!(ancient_score > modern_score);
        // Both should be below or equal to base
        assert!(ancient_score <= base_score || (ancient_score - base_score).abs() < 0.01);
    }

    // -------------------------------------------------------------------------
    // Similarity Metric Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_similarity_metric_jaccard() {
        let metric = SimilarityMetric::Jaccard;
        assert!(metric.compute("hello world", "hello world") > 0.99);
        assert!(metric.compute("hello world", "hello") > 0.3);
    }

    #[test]
    fn test_similarity_metric_edit_distance() {
        let metric = SimilarityMetric::EditDistance;
        assert!(metric.compute("Einstein", "Einstein") > 0.99);
        assert!(metric.compute("Einstein", "Einstien") > 0.7); // Typo
        assert!(metric.compute("Einstein", "Newton") < 0.5);
    }

    #[test]
    fn test_similarity_metric_damerau() {
        let metric = SimilarityMetric::DamerauLevenshtein;
        // Transpositions are common typos
        assert!(metric.compute("teh", "the") > 0.6);
        assert!(metric.compute("recieve", "receive") > 0.8);
    }

    #[test]
    fn test_similarity_metric_wildcard() {
        let metric = SimilarityMetric::EditDistanceWildcard;

        // Wildcards in query should match
        assert!(metric.compute("R?ma", "Roma") > 0.99);
        assert!(metric.compute("Ein*", "Einstein") > 0.99);
        assert!(metric.compute("*stein", "Einstein") > 0.99);

        // Damaged inscription pattern
        assert!(metric.compute("???TOR", "CASTOR") > 0.99);
    }

    #[test]
    fn test_similarity_metric_from_str() {
        assert_eq!(
            SimilarityMetric::parse_str("jaccard"),
            Some(SimilarityMetric::Jaccard)
        );
        assert_eq!(
            SimilarityMetric::parse_str("edit-distance"),
            Some(SimilarityMetric::EditDistance)
        );
        assert_eq!(
            SimilarityMetric::parse_str("lev"),
            Some(SimilarityMetric::EditDistance)
        );
        assert_eq!(
            SimilarityMetric::parse_str("wildcard"),
            Some(SimilarityMetric::EditDistanceWildcard)
        );
        assert_eq!(SimilarityMetric::parse_str("unknown"), None);
    }

    #[test]
    fn test_generator_with_edit_distance() {
        let gen = DictionaryCandidateGenerator::new()
            .with_metric(SimilarityMetric::EditDistance)
            .with_well_known();

        // Test exact match first
        let candidates = gen.generate("Albert Einstein", "", None, 5);
        assert!(!candidates.is_empty());
        assert!(candidates
            .iter()
            .any(|c| c.label.to_lowercase().contains("einstein")));

        // Fuzzy match with typo - may not return results with small dictionaries
        // due to similarity threshold filtering
        let typo_candidates = gen.generate("Einstien", "", None, 5);
        // Just verify it doesn't panic - results depend on dictionary size and threshold
        let _ = typo_candidates;
    }

    #[test]
    fn test_generator_with_wildcard() {
        let gen = DictionaryCandidateGenerator::new()
            .with_metric(SimilarityMetric::EditDistanceWildcard)
            .with_well_known();

        // Wildcards for damaged text: "M?r?e C*" should match "Marie Curie"
        let candidates = gen.generate("marie c*", "", None, 10);
        assert!(!candidates.is_empty());
        assert!(candidates
            .iter()
            .any(|c| c.label.to_lowercase().contains("curie")));
    }

    #[test]
    fn test_similarity_metric_cjk() {
        // Test edit distance handles CJK correctly
        let metric = SimilarityMetric::EditDistance;

        // 北京 (Beijing) vs 北平 (old name) - 1 char diff
        let sim = metric.compute("北京", "北平");
        assert!(sim > 0.4 && sim < 0.9, "CJK similarity: {}", sim);

        // Identical CJK
        assert!(metric.compute("東京", "東京") > 0.99);
    }
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // Strategy for short strings (for expensive operations)
    fn arb_short_string() -> impl Strategy<Value = String> {
        prop::string::string_regex("[a-zA-Z0-9 ]{0,30}").unwrap()
    }

    // Strategy for entity-like strings
    fn arb_entity_name() -> impl Strategy<Value = String> {
        prop::string::string_regex("[A-Z][a-z]+ [A-Z][a-z]+")
            .unwrap()
            .prop_filter("non-empty", |s| !s.is_empty())
    }

    // Strategy for similarity metrics
    fn arb_metric() -> impl Strategy<Value = SimilarityMetric> {
        prop_oneof![
            Just(SimilarityMetric::Jaccard),
            Just(SimilarityMetric::EditDistance),
            Just(SimilarityMetric::DamerauLevenshtein),
            Just(SimilarityMetric::EditDistanceWildcard),
        ]
    }

    // -------------------------------------------------------------------------
    // SimilarityMetric Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// All metrics return values in [0, 1]
        #[test]
        fn prop_metric_bounds(metric in arb_metric(), a in arb_short_string(), b in arb_short_string()) {
            let sim = metric.compute(&a, &b);
            prop_assert!(
                (0.0..=1.0).contains(&sim),
                "Similarity {} out of [0,1] for {:?}",
                sim,
                metric
            );
        }

        /// All metrics give 1.0 for identical strings
        #[test]
        fn prop_metric_identity(metric in arb_metric(), s in arb_short_string()) {
            let sim = metric.compute(&s, &s);
            prop_assert!(
                (sim - 1.0).abs() < 1e-10,
                "Identity similarity should be 1.0, got {} for {:?}", sim, metric
            );
        }

        /// Jaccard, EditDistance, DamerauLevenshtein are symmetric
        /// (EditDistanceWildcard is NOT symmetric due to pattern matching)
        #[test]
        fn prop_symmetric_metrics_symmetric(a in arb_short_string(), b in arb_short_string()) {
            for metric in [
                SimilarityMetric::Jaccard,
                SimilarityMetric::EditDistance,
                SimilarityMetric::DamerauLevenshtein,
            ] {
                let sim1 = metric.compute(&a, &b);
                let sim2 = metric.compute(&b, &a);
                prop_assert!(
                    (sim1 - sim2).abs() < 1e-10,
                    "{:?} not symmetric: ({},{})={} vs ({},{})={}", metric, a, b, sim1, b, a, sim2
                );
            }
        }

        /// Metric name round-trips through from_str
        #[test]
        fn prop_metric_name_roundtrip(metric in arb_metric()) {
            let name = metric.name();
            if let Some(recovered) = SimilarityMetric::parse_str(name) {
                prop_assert_eq!(metric, recovered);
            }
            // Some names might not round-trip exactly (aliases)
        }

        /// Empty string has similarity 1.0 with itself
        #[test]
        fn prop_metric_empty_identity(metric in arb_metric()) {
            let sim = metric.compute("", "");
            prop_assert!(
                (sim - 1.0).abs() < 1e-10,
                "Empty string identity should be 1.0, got {} for {:?}", sim, metric
            );
        }
    }

    // -------------------------------------------------------------------------
    // Candidate Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Candidate score is in [0, 1]
        #[test]
        fn prop_candidate_score_bounds(
            kb_id in "[A-Z][0-9]+",
            label in arb_entity_name(),
            string_sim in 0.0f64..1.0,
            prior in 0.0f64..1.0
        ) {
            let mut candidate = Candidate::new(&kb_id, CandidateSource::Wikidata, &label);
            candidate.string_sim = string_sim;
            candidate.prior = prior;
            candidate.compute_score();

            prop_assert!(
                candidate.score >= 0.0 && candidate.score <= 1.0,
                "Score {} out of [0,1]", candidate.score
            );
        }

        /// Candidate kb_id is deterministic
        #[test]
        fn prop_candidate_kb_id_deterministic(
            kb_id in "[A-Z][0-9]+",
            label in arb_entity_name()
        ) {
            let c1 = Candidate::new(&kb_id, CandidateSource::Wikidata, &label);
            let c2 = Candidate::new(&kb_id, CandidateSource::Wikidata, &label);
            prop_assert_eq!(c1.kb_id, c2.kb_id);
        }

        /// Candidate serde round-trip
        #[test]
        fn prop_candidate_serde_roundtrip(
            kb_id in "[A-Z][0-9]+",
            label in arb_entity_name()
        ) {
            let candidate = Candidate::new(&kb_id, CandidateSource::Wikidata, &label);
            let json = serde_json::to_string(&candidate).unwrap();
            let recovered: Candidate = serde_json::from_str(&json).unwrap();

            prop_assert_eq!(candidate.kb_id, recovered.kb_id);
            prop_assert_eq!(candidate.label, recovered.label);
        }
    }

    // -------------------------------------------------------------------------
    // DictionaryCandidateGenerator Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Generator always returns <= limit candidates
        #[test]
        fn prop_generator_respects_limit(
            mention in arb_entity_name(),
            limit in 1usize..20
        ) {
            let gen = DictionaryCandidateGenerator::new().with_well_known();
            let candidates = gen.generate(&mention, "", None, limit);
            prop_assert!(
                candidates.len() <= limit,
                "Got {} candidates but limit was {}", candidates.len(), limit
            );
        }

        /// Generator name is consistent
        #[test]
        fn prop_generator_name_consistent(metric in arb_metric()) {
            let gen = DictionaryCandidateGenerator::new().with_metric(metric);
            let name = gen.name();
            prop_assert!(!name.is_empty());
        }

        /// Generator with metric round-trips metric correctly
        #[test]
        fn prop_generator_metric_set(metric in arb_metric()) {
            let gen = DictionaryCandidateGenerator::new().with_metric(metric);
            prop_assert_eq!(gen.metric(), metric);
        }

        /// Candidates are sorted by score descending
        #[test]
        fn prop_candidates_sorted_descending(mention in arb_entity_name()) {
            let gen = DictionaryCandidateGenerator::new().with_well_known();
            let candidates = gen.generate(&mention, "", None, 10);

            for i in 1..candidates.len() {
                prop_assert!(
                    candidates[i-1].score >= candidates[i].score,
                    "Candidates not sorted: {} < {} at positions {}-{}",
                    candidates[i-1].score, candidates[i].score, i-1, i
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // CandidateSource Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// CandidateSource serde round-trip
        #[test]
        fn prop_source_serde_roundtrip(source in prop_oneof![
            Just(CandidateSource::Wikidata),
            Just(CandidateSource::YAGO),
            Just(CandidateSource::DBpedia),
            Just(CandidateSource::Wikipedia),
            Just(CandidateSource::Freebase),
            Just(CandidateSource::UMLS),
            Just(CandidateSource::GeoNames),
        ]) {
            let json = serde_json::to_string(&source).unwrap();
            let recovered: CandidateSource = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(source, recovered);
        }

        /// Custom source round-trips
        #[test]
        fn prop_custom_source_roundtrip(name in "[a-z]+") {
            let source = CandidateSource::Custom(name.clone());
            let json = serde_json::to_string(&source).unwrap();
            let recovered: CandidateSource = serde_json::from_str(&json).unwrap();

            if let CandidateSource::Custom(n) = recovered {
                prop_assert_eq!(name, n);
            } else {
                prop_assert!(false, "Expected Custom variant");
            }
        }
    }
}
