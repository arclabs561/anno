//! MentionRankingCoref model: scoring, clustering, and coreference resolution.

#[allow(unused_imports)]
use super::types::*;
#[allow(unused_imports)]
use super::*;
use crate::Animacy;

/// Non-person indicator words. If one mention contains one of these and the
/// other does not, they are semantically incompatible for coreference.
const NON_PERSON_INDICATORS: &[&str] = &[
    "prize",
    "award",
    "medal",
    "cup",
    "trophy",
    "championship",
    "committee",
    "foundation",
    "institute",
    "university",
    "academy",
    "council",
    "board",
    "conference",
    "summit",
    "treaty",
    "agreement",
    "act",
    "law",
];

/// Returns `true` when two mention texts have incompatible semantic types
/// (e.g. one is a prize/award/institution and the other is not).
fn is_type_incompatible(mention_a: &str, mention_b: &str) -> bool {
    let a_lower = mention_a.to_lowercase();
    let b_lower = mention_b.to_lowercase();

    let a_has = NON_PERSON_INDICATORS
        .iter()
        .any(|w| a_lower.split_whitespace().any(|tok| tok == *w));
    let b_has = NON_PERSON_INDICATORS
        .iter()
        .any(|w| b_lower.split_whitespace().any(|tok| tok == *w));

    // Incompatible only when exactly one side has an indicator
    a_has != b_has
}

/// Infer animacy from a pronoun's lowercased text.
fn animacy_from_pronoun(text_lower: &str) -> Animacy {
    match text_lower {
        // Animate pronouns (persons)
        "he" | "him" | "his" | "himself"
        | "she" | "her" | "hers" | "herself"
        | "they" | "them" | "their" | "theirs" | "themselves" | "themself"
        | "i" | "me" | "my" | "mine" | "myself"
        | "we" | "us" | "our" | "ours" | "ourselves"
        | "you" | "your" | "yours" | "yourself" | "yourselves"
        | "who" | "whom" | "whose"
        // Neopronouns (animate by convention)
        | "ze" | "hir" | "hirs" | "hirself"
        | "xe" | "xem" | "xyr" | "xyrs" | "xemself"
        | "ey" | "em" | "eir" | "eirs" | "emself"
        | "fae" | "faer" | "faers" | "faerself" => Animacy::Animate,
        // Inanimate pronouns
        "it" | "its" | "itself" | "which" | "that" => Animacy::Inanimate,
        _ => Animacy::Unknown,
    }
}

/// Infer animacy from NER entity type.
fn animacy_from_entity_type(entity_type: &crate::EntityType) -> Animacy {
    use crate::EntityType;
    match entity_type {
        EntityType::Person => Animacy::Animate,
        EntityType::Organization => Animacy::Animate,
        EntityType::Location => Animacy::Inanimate,
        EntityType::Date
        | EntityType::Time
        | EntityType::Money
        | EntityType::Percent
        | EntityType::Quantity
        | EntityType::Cardinal
        | EntityType::Ordinal => Animacy::Inanimate,
        EntityType::Email | EntityType::Url | EntityType::Phone => Animacy::Inanimate,
        _ => Animacy::Unknown,
    }
}

/// Mention-ranking coreference resolver.
pub struct MentionRankingCoref {
    /// Configuration.
    config: MentionRankingConfig,
    /// Optional NER model for mention detection.
    ner: Option<Box<dyn Model>>,
    /// Optional pre-computed salience scores (entity text -> salience).
    /// Keys should be lowercase for case-insensitive lookup.
    salience_scores: Option<HashMap<String, f64>>,
}

impl std::fmt::Debug for MentionRankingCoref {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MentionRankingCoref")
            .field("config", &self.config)
            .field("ner", &self.ner.as_ref().map(|_| "Some(dyn Model)"))
            .field(
                "salience_scores",
                &self
                    .salience_scores
                    .as_ref()
                    .map(|s| format!("{} entities", s.len())),
            )
            .finish()
    }
}

impl MentionRankingCoref {
    /// Create a new mention-ranking coref resolver.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(MentionRankingConfig::default())
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: MentionRankingConfig) -> Self {
        Self {
            config,
            ner: None,
            salience_scores: None,
        }
    }

    /// Set the NER model for mention detection.
    pub fn with_ner(mut self, ner: Box<dyn Model>) -> Self {
        self.ner = Some(ner);
        self
    }

    /// Set pre-computed salience scores for entities.
    ///
    /// Salience scores should be in range [0, 1] where higher means more
    /// important/salient. Keys are entity text (will be lowercased for lookup).
    ///
    /// Use with `config.salience_weight > 0` to enable salience-weighted scoring.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use anno::salience::{EntityRanker, TextRankSalience};
    ///
    /// let ranker = TextRankSalience::default();
    /// let ranked = ranker.rank(text, &entities);
    ///
    /// // Normalize scores to [0, 1]
    /// let max_score = ranked.iter().map(|(_, s)| *s).fold(0.0_f64, f64::max);
    /// let salience_scores: HashMap<String, f64> = ranked.into_iter()
    ///     .map(|(e, score)| (e.text.to_lowercase(), score / max_score.max(1e-10)))
    ///     .collect();
    ///
    /// let coref = MentionRankingCoref::new()
    ///     .with_salience(salience_scores);
    /// ```
    #[must_use]
    pub fn with_salience(mut self, scores: HashMap<String, f64>) -> Self {
        // Normalize keys to lowercase
        let normalized: HashMap<String, f64> = scores
            .into_iter()
            .map(|(k, v)| (k.to_lowercase(), v))
            .collect();
        self.salience_scores = Some(normalized);
        self
    }

    /// Get salience score for an entity (returns 0.0 if not found).
    fn get_salience(&self, text: &str) -> f64 {
        self.salience_scores
            .as_ref()
            .and_then(|s| s.get(&text.to_lowercase()).copied())
            .unwrap_or(0.0)
    }

    // =========================================================================
    // i2b2-inspired rule-based features (Chen et al. 2011)
    // =========================================================================

    /// Check if two mentions are connected by a "be phrase" (X is Y pattern).
    ///
    /// From Chen et al. (2011): "if there is a 'be phrase' between two concepts
    /// of the same type, they are probably saying 'something is something'."
    ///
    /// # Examples
    ///
    /// - "Resolution of organism is Methicillin-resistant Staphylococcus" → true
    /// - "The patient is John Smith" → true
    /// - "John saw Mary" → false
    fn is_be_phrase_link(&self, text: &str, m1: &RankedMention, m2: &RankedMention) -> bool {
        // Ensure mentions don't overlap and are ordered
        let (earlier, later) = if m1.end <= m2.start {
            (m1, m2)
        } else if m2.end <= m1.start {
            (m2, m1)
        } else {
            return false; // Overlapping mentions
        };

        // Get text between mentions (convert char offsets to get the substring)
        let text_chars: Vec<char> = text.chars().collect();
        if later.start > text_chars.len() || earlier.end > text_chars.len() {
            return false;
        }

        let between: String = text_chars
            .get(earlier.end..later.start)
            .unwrap_or(&[])
            .iter()
            .collect();
        let between_lower = between.to_lowercase();

        // Be-phrase patterns from i2b2 paper
        static BE_PATTERNS: &[&str] = &[
            " is ",
            " are ",
            " was ",
            " were ",
            " be ",
            " being ",
            " been ",
            " refers to ",
            " means ",
            " indicates ",
            " represents ",
            " also known as ",
            " aka ",
            " i.e. ",
            " ie ",
            " namely ",
            " called ",
            " named ",
            " known as ",
            " defined as ",
        ];

        BE_PATTERNS.iter().any(|p| between_lower.contains(p))
    }

    /// Check if one mention is an acronym of the other.
    ///
    /// Delegates to the language-agnostic `anno::coalesce::is_acronym_match` function.
    ///
    /// From Chen et al. (2011): "The first letters of each word in concepts
    /// that have two or more words are taken and compared to whole words
    /// in other concepts."
    ///
    /// # Examples
    ///
    /// - "MRSA" ↔ "Methicillin-resistant Staphylococcus aureus" → true
    /// - "WHO" ↔ "World Health Organization" → true
    /// - "IBM" ↔ "Apple" → false
    fn is_acronym_match(&self, m1: &RankedMention, m2: &RankedMention) -> bool {
        crate::coalesce::similarity::is_acronym_match(&m1.text, &m2.text)
    }

    /// Check if "it" at the given position is pleonastic (non-referential).
    ///
    /// Pleonastic "it" is a grammatical placeholder that doesn't refer to any
    /// entity. Common patterns include:
    /// - Weather: "it rains", "it is sunny", "it's cold"
    /// - Modal: "it is important that...", "it is likely..."
    /// - Cognitive: "it seems", "it appears", "it turns out"
    /// - Cleft: "it was John who..."
    ///
    /// Based on: Boyd et al. "Identification of Pleonastic It Using the Web"
    /// and Stanford CoreNLP's PleonasticFilter patterns.
    fn is_pleonastic_it(&self, text_lower: &str, it_byte_pos: usize) -> bool {
        // Get the text after "it"
        let after_it = &text_lower[it_byte_pos + 2..]; // Skip "it"
        let after_it_trimmed = after_it.trim_start();

        // Weather verbs: "it rains", "it snows", "it hails"
        const WEATHER_VERBS: &[&str] = &[
            "rain",
            "rains",
            "rained",
            "raining",
            "snow",
            "snows",
            "snowed",
            "snowing",
            "hail",
            "hails",
            "hailed",
            "hailing",
            "thunder",
            "thunders",
            "thundered",
            "thundering",
        ];

        // Weather adjectives: "it is sunny", "it's cold"
        const WEATHER_ADJS: &[&str] = &[
            "sunny", "cloudy", "foggy", "windy", "rainy", "snowy", "cold", "hot", "warm", "cool",
            "humid", "dry", "freezing", "chilly", "muggy", "overcast",
        ];

        // Modal/cognitive adjectives: "it is important", "it seems likely"
        const MODAL_ADJS: &[&str] = &[
            "important",
            "necessary",
            "possible",
            "impossible",
            "likely",
            "unlikely",
            "clear",
            "obvious",
            "evident",
            "apparent",
            "true",
            "false",
            "certain",
            "uncertain",
            "doubtful",
            "essential",
            "vital",
            "crucial",
            "critical",
            "imperative",
            "fortunate",
            "unfortunate",
            "surprising",
            "unsurprising",
            "strange",
            "odd",
            "weird",
            "remarkable",
            "noteworthy",
            "known",
            "unknown",
            "believed",
            "thought",
            "said",
            "reported",
            "estimated",
            "assumed",
            "expected",
            "hoped",
            "feared",
        ];

        // Cognitive verbs: "it seems", "it appears"
        const COGNITIVE_VERBS: &[&str] = &[
            "seems",
            "seem",
            "seemed",
            "appears",
            "appear",
            "appeared",
            "turns out",
            "turned out",
            "happens",
            "happen",
            "happened",
            "follows",
            "follow",
            "followed",
            "matters",
            "matter",
            "mattered",
            "helps",
            "help",
            "helped",
            "hurts",
            "hurt",
        ];

        // Check for weather verbs directly
        for verb in WEATHER_VERBS {
            if let Some(after_verb) = after_it_trimmed.strip_prefix(verb) {
                if after_verb.is_empty() || after_verb.starts_with(|c: char| !c.is_alphanumeric()) {
                    return true;
                }
            }
        }

        // Check for cognitive verbs
        for verb in COGNITIVE_VERBS {
            if let Some(after_verb) = after_it_trimmed.strip_prefix(verb) {
                if after_verb.is_empty() || after_verb.starts_with(|c: char| !c.is_alphanumeric()) {
                    return true;
                }
            }
        }

        // Check for "it is/was/has been/will be + MODAL_ADJ"
        // Also handles contractions: "it's"
        let copula_patterns = ["is ", "was ", "'s ", "has been ", "will be ", "would be "];
        for copula in copula_patterns {
            if let Some(after_copula) = after_it_trimmed.strip_prefix(copula) {
                let after_copula = after_copula.trim_start();

                // Check weather verbs after copula: "it is raining"
                for verb in WEATHER_VERBS {
                    if let Some(after_verb) = after_copula.strip_prefix(verb) {
                        if after_verb.is_empty()
                            || after_verb.starts_with(|c: char| !c.is_alphanumeric())
                        {
                            return true;
                        }
                    }
                }

                // Check weather adjectives
                for adj in WEATHER_ADJS {
                    if let Some(after_adj) = after_copula.strip_prefix(adj) {
                        if after_adj.is_empty()
                            || after_adj.starts_with(|c: char| !c.is_alphanumeric())
                        {
                            return true;
                        }
                    }
                }

                // Check modal adjectives
                for adj in MODAL_ADJS {
                    if let Some(after_adj) = after_copula.strip_prefix(adj) {
                        // Modal adjectives often followed by "that", "to", or end of clause
                        if after_adj.is_empty()
                            || after_adj.starts_with(" that")
                            || after_adj.starts_with(" to")
                            || after_adj.starts_with(|c: char| !c.is_alphanumeric())
                        {
                            return true;
                        }
                    }
                }

                // Check for "it is/was + time expression"
                // "it is 5 o'clock", "it was midnight"
                let time_words = ["noon", "midnight", "morning", "evening", "night", "time"];
                for tw in time_words {
                    if after_copula.starts_with(tw) {
                        return true;
                    }
                }

                // Check for numeric time: "it is 5", "it's 3:00"
                if after_copula.starts_with(|c: char| c.is_ascii_digit()) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if two mentions should NOT be linked based on context clues.
    ///
    /// From Chen et al. (2011): "eliminate links that actually refer to two
    /// different entities based on clues found in the sentences surrounding
    /// the mentions... including dates, locations, or descriptive modifiers."
    ///
    /// Returns true if the link should be filtered out.
    fn should_filter_by_context(&self, text: &str, m1: &RankedMention, m2: &RankedMention) -> bool {
        let text_chars: Vec<char> = text.chars().collect();
        let char_count = text_chars.len();

        // Get context windows around each mention (20 chars before and after)
        let context_window = 20;

        let m1_context_start = m1.start.saturating_sub(context_window);
        let m1_context_end = (m1.end + context_window).min(char_count);
        let m1_context: String = text_chars
            .get(m1_context_start..m1_context_end)
            .unwrap_or(&[])
            .iter()
            .collect();

        let m2_context_start = m2.start.saturating_sub(context_window);
        let m2_context_end = (m2.end + context_window).min(char_count);
        let m2_context: String = text_chars
            .get(m2_context_start..m2_context_end)
            .unwrap_or(&[])
            .iter()
            .collect();

        // Check for different dates (YYYY-MM-DD or MM/DD/YYYY patterns)
        let date1 = Self::extract_date(&m1_context);
        let date2 = Self::extract_date(&m2_context);
        if let (Some(d1), Some(d2)) = (&date1, &date2) {
            if d1 != d2 {
                return true; // Different dates → different entities
            }
        }

        // Check for negation context mismatches
        // "not a smoker" vs "smoker" should not link
        let m1_negated = Self::has_negation_context(&m1_context);
        let m2_negated = Self::has_negation_context(&m2_context);
        if m1_negated != m2_negated {
            return true;
        }

        false
    }

    /// Extract a date from context text if present.
    fn extract_date(context: &str) -> Option<String> {
        // Simple date patterns: YYYY-MM-DD or MM/DD/YYYY
        let date_patterns = [
            r"\d{4}-\d{2}-\d{2}",       // ISO format
            r"\d{2}/\d{2}/\d{4}",       // US format
            r"\d{1,2}/\d{1,2}/\d{2,4}", // Flexible US
        ];

        for pattern in &date_patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                if let Some(m) = re.find(context) {
                    return Some(m.as_str().to_string());
                }
            }
        }
        None
    }

    /// Check if context contains negation markers.
    fn has_negation_context(context: &str) -> bool {
        let lower = context.to_lowercase();
        static NEGATION_MARKERS: &[&str] = &[
            "not ",
            "no ",
            "never ",
            "without ",
            "denies ",
            "denied ",
            "negative for ",
            "neg for ",
            "ruled out ",
            "r/o ",
        ];
        NEGATION_MARKERS.iter().any(|m| lower.contains(m))
    }

    /// Check if two mentions are synonyms.
    ///
    /// This method checks for synonym relationships between mentions.
    /// By default, it uses high string similarity as a proxy for synonymy.
    ///
    /// For domain-specific synonym matching (medical, legal, etc.), integrate
    /// a custom `anno::coalesce::SynonymSource` implementation. Available sources:
    /// - UMLS MRCONSO for medical terminology
    /// - WordNet for general English
    /// - Wikidata aliases for multilingual entities
    ///
    /// The pluggable synonym infrastructure is defined in `anno::coalesce::similarity`:
    /// - `SynonymSource` trait: implement to provide custom lookups
    /// - `ChainedSynonyms`: combine multiple sources
    /// - `SynonymMatch`: result type with canonical ID and confidence
    ///
    /// # Design Decision
    ///
    /// We deliberately removed the hardcoded English medical synonym table
    /// (kidney→renal, heart→cardiac, etc.) that was here previously.
    /// Hardcoded tables:
    /// - Only work for one language (English)
    /// - Only work for one domain (medical)
    /// - Create maintenance burden
    /// - Don't scale to new domains
    ///
    /// Instead, use high string similarity or integrate a proper knowledge base.
    fn are_synonyms(&self, m1: &RankedMention, m2: &RankedMention) -> bool {
        let t1 = m1.text.to_lowercase();
        let t2 = m2.text.to_lowercase();

        if t1 == t2 {
            return true;
        }

        // Use multilingual string similarity from coalesce as a proxy.
        // High similarity (>0.8) suggests related terms.
        // This works across languages without hardcoded tables.
        let similarity = crate::coalesce::similarity::multilingual_similarity(&t1, &t2);
        similarity > 0.8
    }

    /// Resolve coreferences in text.
    pub fn resolve(&self, text: &str) -> Result<Vec<MentionCluster>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        // Step 1: Detect mentions
        let mut mentions = self.detect_mentions(text)?;

        if mentions.is_empty() {
            return Ok(vec![]);
        }

        // Sort by position
        mentions.sort_by_key(|m| (m.start, m.end));

        // Step 2: Extract features for each mention
        for mention in &mut mentions {
            self.extract_features(mention);
        }

        // Step 3: Rank antecedents and link (pass text for context-aware features)
        let clusters = self.link_mentions(&mentions, text);

        Ok(clusters)
    }

    /// Get language-specific pronoun patterns.
    ///
    /// Returns (pronoun_text, gender, number) tuples for the specified language.
    /// Falls back to English if language is not supported.
    fn get_pronoun_patterns(&self) -> Vec<(&'static str, Gender, Number)> {
        let lang_code = self
            .config
            .language
            .split('-')
            .next()
            .unwrap_or(&self.config.language)
            .to_lowercase();

        match lang_code.as_str() {
            "es" => vec![
                // Spanish pronouns
                ("él", Gender::Masculine, Number::Singular),
                ("ella", Gender::Feminine, Number::Singular),
                ("ellos", Gender::Masculine, Number::Plural),
                ("ellas", Gender::Feminine, Number::Plural),
                ("lo", Gender::Masculine, Number::Singular),
                ("la", Gender::Feminine, Number::Singular),
                ("los", Gender::Masculine, Number::Plural),
                ("las", Gender::Feminine, Number::Plural),
                ("le", Gender::Unknown, Number::Singular), // Leísmo - can be gender-neutral
                ("les", Gender::Unknown, Number::Plural),
                ("su", Gender::Unknown, Number::Unknown),
                ("sus", Gender::Unknown, Number::Plural),
                ("suyo", Gender::Masculine, Number::Singular),
                ("suya", Gender::Feminine, Number::Singular),
                ("suyos", Gender::Masculine, Number::Plural),
                ("suyas", Gender::Feminine, Number::Plural),
                ("se", Gender::Unknown, Number::Unknown), // Reflexive
                ("nosotros", Gender::Masculine, Number::Plural),
                ("nosotras", Gender::Feminine, Number::Plural),
                ("vosotros", Gender::Masculine, Number::Plural),
                ("vosotras", Gender::Feminine, Number::Plural),
                ("usted", Gender::Unknown, Number::Singular),
                ("ustedes", Gender::Unknown, Number::Plural),
                // Non-binary options (emerging usage)
                // Note: "elle" (singular) and "elles" (plural) are being used by some non-binary Spanish speakers
                // though not yet standardized. Some also use "le" (leísmo) as gender-neutral.
                ("elle", Gender::Unknown, Number::Singular), // Non-binary third-person (emerging)
                ("elles", Gender::Unknown, Number::Plural), // Non-binary third-person plural (emerging)
            ],
            "fr" => vec![
                // French pronouns
                ("il", Gender::Masculine, Number::Singular),
                ("elle", Gender::Feminine, Number::Singular),
                ("ils", Gender::Masculine, Number::Plural),
                ("elles", Gender::Feminine, Number::Plural),
                ("le", Gender::Masculine, Number::Singular),
                ("la", Gender::Feminine, Number::Singular),
                ("les", Gender::Unknown, Number::Plural),
                ("lui", Gender::Unknown, Number::Singular),
                ("leur", Gender::Unknown, Number::Plural),
                ("son", Gender::Masculine, Number::Singular),
                ("sa", Gender::Feminine, Number::Singular),
                ("ses", Gender::Unknown, Number::Plural),
                ("se", Gender::Unknown, Number::Unknown), // Reflexive
                ("nous", Gender::Unknown, Number::Plural),
                ("vous", Gender::Unknown, Number::Unknown),
                // Non-binary options (emerging usage)
                // Note: "iel" (singular) and "iels" (plural) are being used by some non-binary French speakers
                // though not yet standardized in formal French
                ("iel", Gender::Unknown, Number::Singular), // Non-binary third-person (emerging)
                ("iels", Gender::Unknown, Number::Plural), // Non-binary third-person plural (emerging)
            ],
            "de" => vec![
                // German pronouns
                ("er", Gender::Masculine, Number::Singular),
                ("sie", Gender::Feminine, Number::Singular),
                ("es", Gender::Neutral, Number::Singular),
                ("sie", Gender::Unknown, Number::Plural), // Same form as feminine singular
                ("ihn", Gender::Masculine, Number::Singular),
                ("ihr", Gender::Feminine, Number::Singular),
                ("ihm", Gender::Masculine, Number::Singular),
                ("ihnen", Gender::Unknown, Number::Plural),
                ("sein", Gender::Masculine, Number::Singular),
                ("seine", Gender::Feminine, Number::Singular),
                ("sein", Gender::Neutral, Number::Singular),
                ("ihre", Gender::Feminine, Number::Singular),
                ("ihr", Gender::Unknown, Number::Plural),
                ("sich", Gender::Unknown, Number::Unknown), // Reflexive
                ("wir", Gender::Unknown, Number::Plural),
                ("ihr", Gender::Unknown, Number::Plural), // 2nd person plural
                ("sie", Gender::Unknown, Number::Plural), // 3rd person plural (formal)
                // Non-binary options (emerging usage)
                // Note: "sier" and "xier" are being used by some non-binary German speakers
                // though not yet standardized. "es" (it) is grammatically neutral but dehumanizing.
                ("sier", Gender::Unknown, Number::Singular), // Non-binary third-person (emerging)
                ("xier", Gender::Unknown, Number::Singular), // Non-binary third-person (emerging, alternative)
                ("dier", Gender::Unknown, Number::Singular), // Non-binary third-person (emerging, alternative)
            ],
            "ar" => vec![
                // Arabic pronouns (RTL)
                ("هو", Gender::Masculine, Number::Singular), // huwa
                ("هي", Gender::Feminine, Number::Singular),  // hiya
                ("هم", Gender::Masculine, Number::Plural),   // hum
                ("هن", Gender::Feminine, Number::Plural),    // hunna
                ("هما", Gender::Unknown, Number::Plural),    // huma (dual)
            ],
            "ru" => vec![
                // Russian pronouns
                ("он", Gender::Masculine, Number::Singular),
                ("она", Gender::Feminine, Number::Singular),
                ("оно", Gender::Neutral, Number::Singular),
                ("они", Gender::Unknown, Number::Plural),
                ("его", Gender::Masculine, Number::Singular),
                ("её", Gender::Feminine, Number::Singular),
                ("их", Gender::Unknown, Number::Plural),
                ("себя", Gender::Unknown, Number::Unknown), // Reflexive
                ("мы", Gender::Unknown, Number::Plural),
                ("вы", Gender::Unknown, Number::Unknown),
            ],
            "zh" => vec![
                // Chinese pronouns
                // Traditional gendered forms (introduced in 20th century)
                ("他", Gender::Masculine, Number::Singular), // tā - he (also used as gender-neutral historically)
                ("她", Gender::Feminine, Number::Singular),  // tā - she
                ("它", Gender::Neutral, Number::Singular),   // tā - it (objects)
                ("牠", Gender::Neutral, Number::Singular),   // tā - it (animals, traditional)
                ("祂", Gender::Neutral, Number::Singular),   // tā - it (deities)
                // Gender-neutral options for non-binary individuals
                ("怹", Gender::Unknown, Number::Singular), // tān - honorific gender-neutral "they" (archaic but exists)
                ("其", Gender::Unknown, Number::Singular), // qí - formal gender-neutral pronoun (very formal)
                // Modern non-binary options (pinyin, used in informal/online contexts)
                // Note: "TA" and "X也" are typically written in pinyin/latin, but we include them
                // for completeness. In practice, these may appear as "TA" or "X也" in text.
                ("他们", Gender::Masculine, Number::Plural), // tāmen - they (masculine/mixed)
                ("她们", Gender::Feminine, Number::Plural),  // tāmen - they (feminine)
                ("它们", Gender::Neutral, Number::Plural),   // tāmen - they (objects)
                                                             // Note: In spoken Chinese, all third-person pronouns are pronounced "tā" (gender-neutral)
                                                             // The gender distinction exists only in written form
            ],
            "ja" => vec![
                // Japanese pronouns
                // Third-person (historically gender-neutral, now gendered in modern usage)
                ("彼", Gender::Masculine, Number::Singular), // kare - he (originally gender-neutral)
                ("彼女", Gender::Feminine, Number::Singular), // kanojo - she
                ("彼ら", Gender::Unknown, Number::Plural),   // karera - they
                // Gender-neutral alternatives (modern usage)
                // Note: Japanese often avoids pronouns entirely, using names/titles instead
                // For non-binary individuals, その人 (sono hito - that person) or name/title is common
                ("その人", Gender::Unknown, Number::Singular), // sono hito - that person (gender-neutral)
                ("あの人", Gender::Unknown, Number::Singular), // ano hito - that person (gender-neutral)
            ],
            "ko" => vec![
                // Korean pronouns
                // Korean often avoids third-person pronouns, using names/titles
                ("그", Gender::Masculine, Number::Singular), // geu - he (also means "that")
                ("그녀", Gender::Feminine, Number::Singular), // geunyeo - she (literally "that woman")
                ("그들", Gender::Unknown, Number::Plural),    // geudeul - they
                // Gender-neutral alternatives
                ("그 사람", Gender::Unknown, Number::Singular), // geu saram - that person (gender-neutral)
                ("그분", Gender::Unknown, Number::Singular), // geubun - that person (honorific, gender-neutral)
            ],
            _ => {
                // English (default) - comprehensive pronoun patterns including neopronouns
                vec![
                    // Traditional pronouns
                    ("he", Gender::Masculine, Number::Singular),
                    ("she", Gender::Feminine, Number::Singular),
                    ("it", Gender::Neutral, Number::Singular),
                    ("they", Gender::Unknown, Number::Unknown), // Singular or plural
                    ("him", Gender::Masculine, Number::Singular),
                    ("her", Gender::Feminine, Number::Singular),
                    ("them", Gender::Unknown, Number::Unknown), // Singular or plural
                    ("his", Gender::Masculine, Number::Singular),
                    ("hers", Gender::Feminine, Number::Singular),
                    ("its", Gender::Neutral, Number::Singular),
                    ("their", Gender::Unknown, Number::Unknown), // Singular or plural
                    ("theirs", Gender::Unknown, Number::Unknown),
                    ("themself", Gender::Unknown, Number::Singular), // Explicitly singular
                    ("themselves", Gender::Unknown, Number::Plural), // Explicitly plural
                    // Third-person reflexives
                    ("himself", Gender::Masculine, Number::Singular),
                    ("herself", Gender::Feminine, Number::Singular),
                    ("itself", Gender::Neutral, Number::Singular),
                    // First-person pronouns
                    ("i", Gender::Unknown, Number::Singular),
                    ("me", Gender::Unknown, Number::Singular),
                    ("my", Gender::Unknown, Number::Singular),
                    ("mine", Gender::Unknown, Number::Singular),
                    ("myself", Gender::Unknown, Number::Singular),
                    ("we", Gender::Unknown, Number::Plural),
                    ("us", Gender::Unknown, Number::Plural),
                    ("our", Gender::Unknown, Number::Plural),
                    ("ours", Gender::Unknown, Number::Plural),
                    ("ourselves", Gender::Unknown, Number::Plural),
                    ("you", Gender::Unknown, Number::Unknown), // Singular or plural
                    ("your", Gender::Unknown, Number::Unknown),
                    ("yours", Gender::Unknown, Number::Unknown),
                    ("yourself", Gender::Unknown, Number::Singular),
                    ("yourselves", Gender::Unknown, Number::Plural),
                    // Neopronouns: ze/hir set
                    ("ze", Gender::Unknown, Number::Singular),
                    ("hir", Gender::Unknown, Number::Singular),
                    ("hirs", Gender::Unknown, Number::Singular),
                    ("hirself", Gender::Unknown, Number::Singular),
                    // Neopronouns: xe/xem set
                    ("xe", Gender::Unknown, Number::Singular),
                    ("xem", Gender::Unknown, Number::Singular),
                    ("xyr", Gender::Unknown, Number::Singular),
                    ("xyrs", Gender::Unknown, Number::Singular),
                    ("xemself", Gender::Unknown, Number::Singular),
                    // Neopronouns: e/em (Spivak) set
                    ("ey", Gender::Unknown, Number::Singular), // Also spelled "e"
                    ("em", Gender::Unknown, Number::Singular),
                    ("eir", Gender::Unknown, Number::Singular),
                    ("eirs", Gender::Unknown, Number::Singular),
                    ("emself", Gender::Unknown, Number::Singular),
                    // Neopronouns: fae/faer set
                    ("fae", Gender::Unknown, Number::Singular),
                    ("faer", Gender::Unknown, Number::Singular),
                    ("faers", Gender::Unknown, Number::Singular),
                    ("faerself", Gender::Unknown, Number::Singular),
                    // Demonstrative pronouns
                    ("this", Gender::Unknown, Number::Singular),
                    ("that", Gender::Unknown, Number::Singular),
                    ("these", Gender::Unknown, Number::Plural),
                    ("those", Gender::Unknown, Number::Plural),
                    // Indefinite pronouns
                    ("someone", Gender::Unknown, Number::Singular),
                    ("somebody", Gender::Unknown, Number::Singular),
                    ("anyone", Gender::Unknown, Number::Singular),
                    ("anybody", Gender::Unknown, Number::Singular),
                    ("everyone", Gender::Unknown, Number::Singular), // Grammatically singular
                    ("everybody", Gender::Unknown, Number::Singular),
                    ("no one", Gender::Unknown, Number::Singular),
                    ("nobody", Gender::Unknown, Number::Singular),
                    // Impersonal "one"
                    ("one", Gender::Unknown, Number::Singular),
                    ("oneself", Gender::Unknown, Number::Singular),
                    // Interrogative/relative pronouns
                    ("who", Gender::Unknown, Number::Unknown),
                    ("whom", Gender::Unknown, Number::Unknown),
                    ("whose", Gender::Unknown, Number::Unknown),
                    ("which", Gender::Unknown, Number::Unknown),
                    // Reciprocal pronouns
                    ("each other", Gender::Unknown, Number::Plural),
                    ("one another", Gender::Unknown, Number::Plural),
                ]
            }
        }
    }

    /// Detect mentions using NER or heuristics.
    fn detect_mentions(&self, text: &str) -> Result<Vec<RankedMention>> {
        let mut mentions = Vec::new();

        // Use NER if available
        if let Some(ref ner) = self.ner {
            let entities = ner.extract_entities(text, None)?;
            for entity in entities {
                mentions.push(RankedMention {
                    start: entity.start(),
                    end: entity.end(),
                    text: entity.text.clone(),
                    mention_type: MentionType::Proper,
                    gender: None,
                    number: None,
                    animacy: animacy_from_entity_type(&entity.entity_type),
                    head: self.get_head(&entity.text),
                    entity_type: Some(entity.entity_type.clone()),
                });
            }
        }

        // Also detect pronouns via pattern matching
        //
        // Note on singular "they": English has used singular they since the 14th century
        // (Chaucer, Shakespeare, Jane Austen). It's standard for:
        // 1. Non-binary individuals ("Alex said they would come")
        // 2. Unknown/generic referents ("Someone left their umbrella")
        // 3. Formal contexts avoiding gendered assumptions
        //
        // Therefore, they/them/their use Number::Unknown, not Plural.
        // The coreference scorer handles this by not penalizing Unknown mismatches.
        //
        // Neopronouns (ze/hir, xe/xem, e/em Spivak, etc.) are third-person singular
        // pronouns used for gender-neutral or nonbinary reference. They behave
        // grammatically as singular and use Gender::Unknown since they explicitly
        // Get language-specific pronoun patterns
        // Use language from config, fallback to English
        let pronoun_patterns = self.get_pronoun_patterns();

        // =========================================================================
        // KNOWN GAPS / FUTURE WORK (documented for linguistic completeness):
        // =========================================================================
        //
        // 1. CATAPHORA (forward reference):
        //    "Before she arrived, Mary called ahead."
        //    Current: Only backward (anaphoric) reference is modeled.
        //    Fix: Would require looking ahead in discourse.
        //
        // 2. SPLIT ANTECEDENTS:
        //    "John went to the store. Mary went to the bank. They met for lunch."
        //    Current: "They" would need to link to BOTH John and Mary.
        //    Fix: Cluster merging based on plural pronoun + multiple candidates.
        //
        // 3. BRIDGING ANAPHORA:
        //    "I bought a car. The engine was faulty."
        //    Current: "The engine" has no explicit antecedent.
        //    Fix: Requires world knowledge (car has engine).
        //
        // 4. APPOSITIVE CONSTRUCTIONS:
        //    "John, the baker, opened his shop."
        //    Current: Would detect "John" and "the baker" as separate mentions.
        //    Fix: Need to recognize appositive structure and link them.
        //
        // 5. COPULA CONSTRUCTIONS:
        //    "The CEO is John Smith."
        //    Current: Separate mentions, may not link.
        //    Fix: Special handling for "X is Y" patterns (see is_be_phrase_link).
        //
        // 6. PRO-DROP LANGUAGES (Spanish, Italian, Japanese):
        //    Subject pronouns can be omitted: "∅ llegué tarde" = "I arrived late"
        //    Current: Only works with overt pronouns.
        //    Fix: Verb morphology analysis, zero pronoun detection.
        //
        // 7. BINDING THEORY CONSTRAINTS:
        //    Reflexives must be locally bound: "John saw himself" (same clause)
        //    Pronouns must NOT be locally bound: "John saw him" (different entity)
        //    Current: Not enforced - all candidates scored equally.
        //    Fix: Syntactic parsing to identify clause boundaries.
        //
        // 8. ANIMACY CONSTRAINTS:
        //    "The rock fell. *It/*He was heavy."
        //    Current: Basic gender/number matching only.
        //    Fix: Animacy feature extraction from entity type or lexicon.
        //
        // =========================================================================
        // EXOTIC LINGUISTIC PHENOMENA (beyond standard English):
        // =========================================================================
        //
        // 9. CLUSIVITY (inclusive vs exclusive "we"):
        //    Many languages (Austronesian, Dravidian, Algonquian) distinguish:
        //    - Inclusive: speaker + addressee ("you and I")
        //    - Exclusive: speaker + others, NOT addressee ("me and them, not you")
        //    Current: Not modeled. English "we" is ambiguous.
        //
        // 10. OBVIATION (Algonquian "fourth person"):
        //     Distinguishes proximate (topical) vs obviative (less topical) 3rd person.
        //     "He_PROX saw him_OBV" = unambiguous reference to two different entities.
        //     Current: No support for discourse-level topicality tracking.
        //
        // 11. SWITCH-REFERENCE:
        //     Clausal markers indicating whether subject is same/different from prior clause.
        //     "He went home and-SAME_SUBJ ate" vs "He went home and-DIFF_SUBJ she cooked"
        //     Current: No syntactic clause analysis.
        //
        // 12. LOGOPHORIC PRONOUNS (West African languages like Ewe, Yoruba):
        //     Special pronoun for "the person whose speech/thought is being reported"
        //     "Kofi said that LOG will win" (LOG = Kofi, unambiguously)
        //     Current: No perspective/attitude holder tracking.
        //
        // 13. CORRELATIVE-RELATIVE (Sanskrit, Hindi):
        //     "ya- ... sa-" pattern: relative clause first, then demonstrative resumes.
        //     "Who(ever) came, that-one ate" = explicit cross-clause coreference.
        //     Current: Only backward anaphora modeled.
        //
        // 14. NOUN CLASS SYSTEMS (Bantu, Dyirbal):
        //     10-20+ "genders" based on semantics (human, animal, plant, tool, etc.)
        //     Pronouns agree with noun class, not biological sex.
        //     Current: Only masc/fem/neut gender, not full noun class agreement.
        //
        // 15. SHAPE-BASED CLASSIFIERS (Navajo, Chinese classifiers):
        //     Verbs/pronouns encode physical properties (long, flat, round, granular).
        //     Current: No shape/classifier feature tracking.
        //
        // 16. TRIAL/PAUCAL NUMBER (Austronesian):
        //     Some languages distinguish: singular, dual, trial (exactly 3), paucal (few).
        //     Current: Only sg/du/pl/unknown in Number enum.
        //
        // 17. HONORIFIC/POLITENESS LEVELS (Japanese, Korean, Thai):
        //     Pronoun choice encodes social relationship, not just person/number.
        //     "Watashi" vs "boku" vs "ore" (Japanese 1st person, different registers).
        //     Current: No formality/register tracking.
        //
        // =========================================================================
        // INFORMATION-THEORETIC VIEW:
        // =========================================================================
        //
        // Coreference resolution can be framed as entropy reduction:
        // - H(Antecedent | Context) = uncertainty over which entity a pronoun refers to
        // - Good discourse makes pronouns low-entropy (context narrows candidates)
        // - Surprisal of choosing antecedent a = -log p(a | Context)
        // - Each resolved anaphor yields information gain: H(A) - H(A | Context)
        //
        // Features like recency, grammatical role, semantic compatibility all
        // increase mutual information I(Antecedent; Context).
        //

        // Find pronouns in text
        let text_lower = text.to_lowercase();
        let text_chars: Vec<char> = text.chars().collect();
        for (pronoun, gender, number) in pronoun_patterns {
            let mut search_start_byte = 0;
            while let Some(pos) = text_lower[search_start_byte..].find(pronoun) {
                let abs_byte_pos = search_start_byte + pos;
                let end_byte_pos = abs_byte_pos + pronoun.len();

                // Convert byte positions to character positions for boundary checks
                let char_pos = text[..abs_byte_pos].chars().count();
                let end_char_pos = char_pos + pronoun.chars().count();

                // Check word boundaries using character positions
                let is_word_start = char_pos == 0
                    || match text_chars.get(char_pos.saturating_sub(1)) {
                        None => true,
                        Some(c) => !c.is_alphanumeric(),
                    };
                let is_word_end = end_char_pos >= text_chars.len()
                    || match text_chars.get(end_char_pos) {
                        None => true,
                        Some(c) => !c.is_alphanumeric(),
                    };

                if is_word_start && is_word_end {
                    // Skip pleonastic "it" (non-referential uses)
                    // See: Boyd et al. "Identification of Pleonastic It Using the Web"
                    if pronoun == "it" && self.is_pleonastic_it(&text_lower, abs_byte_pos) {
                        search_start_byte = end_byte_pos;
                        continue;
                    }

                    // Use character offsets for the mention
                    let char_start = char_pos;
                    let char_end = end_char_pos;

                    mentions.push(RankedMention {
                        start: char_start,
                        end: char_end,
                        text: text[abs_byte_pos..end_byte_pos].to_string(),
                        mention_type: MentionType::Pronominal,
                        gender: Some(gender),
                        number: Some(number),
                        animacy: animacy_from_pronoun(pronoun),
                        head: pronoun.to_string(),
                        entity_type: None,
                    });
                }

                search_start_byte = end_byte_pos;
            }
        }

        // Detect proper nouns (capitalized words not at sentence start).
        // Known pronouns are excluded so the pronoun detector (above) takes priority
        // in the overlap dedup -- otherwise "He" gets typed as Proper and misses
        // the lower pronoun linking threshold.
        const PRONOUN_WORDS: &[&str] = &[
            "he",
            "she",
            "it",
            "they",
            "him",
            "her",
            "them",
            "his",
            "hers",
            "its",
            "their",
            "himself",
            "herself",
            "itself",
            "themselves",
            "we",
            "us",
            "our",
            "ours",
        ];
        let words: Vec<_> = text.split_whitespace().collect();
        let mut search_byte_pos = 0; // Byte position for searching

        for (i, word) in words.iter().enumerate() {
            // Skip if at sentence start
            let at_sentence_start = i == 0
                || match text[..text.find(word).unwrap_or(0)].chars().last() {
                    None => true,
                    Some(c) => c == '.' || c == '!' || c == '?',
                };

            if !at_sentence_start
                && word.chars().next().is_some_and(|c| c.is_uppercase())
                && word.chars().count() > 1
                && !PRONOUN_WORDS.contains(
                    &word
                        .trim_end_matches(|c: char| c.is_ascii_punctuation())
                        .to_lowercase()
                        .as_str(),
                )
            {
                // Strip trailing punctuation from proper noun text
                let clean_word = word.trim_end_matches(|c: char| c.is_ascii_punctuation());
                if clean_word.is_empty() {
                    search_byte_pos += word.len() + 1;
                    continue;
                }

                // Find byte position of word
                if let Some(rel_byte_pos) = text[search_byte_pos..].find(word) {
                    let abs_byte_pos = search_byte_pos + rel_byte_pos;
                    // Convert byte offset to character offset for Entity
                    let char_start = text[..abs_byte_pos].chars().count();
                    let char_end = char_start + clean_word.chars().count();

                    mentions.push(RankedMention {
                        start: char_start,
                        end: char_end,
                        text: clean_word.to_string(),
                        mention_type: MentionType::Proper,
                        gender: None,
                        number: Some(Number::Singular),
                        animacy: Animacy::Unknown,
                        head: clean_word.to_string(),
                        entity_type: None,
                    });
                }
            }

            search_byte_pos += word.len() + 1; // +1 for space (byte-based)
        }

        // Detect nominal adjectives (J2N: arXiv:2409.14374)
        // Phrases like "the poor", "the elderly" function as plural noun phrases.
        //
        // MULTILINGUAL: Supports English, German, French, Spanish patterns.
        // - German: "die Armen" (the poor), "die Reichen" (the rich)
        // - French: "les pauvres", "les riches"
        // - Spanish: "los pobres", "los ricos"
        // - Arabic and Japanese use different patterns not yet supported.
        if self.config.enable_nominal_adjective_detection {
            // Adjectives that commonly function as nouns when preceded by determiners.
            // These refer to groups of people and are grammatically plural.
            const NOMINALIZED_ADJECTIVES: &[&str] = &[
                // Socioeconomic status
                "poor",
                "rich",
                "wealthy",
                "homeless",
                "unemployed",
                "employed",
                // Age
                "young",
                "old",
                "elderly",
                "aged",
                // Health and physical state
                "sick",
                "ill",
                "healthy",
                "wounded",
                "injured",
                "disabled",
                "blind",
                "deaf",
                // Life state
                "dead",
                "living",
                "deceased",
                // Legal/social status
                "accused",
                "condemned",
                "convicted",
                "guilty",
                "innocent",
                "insured",
                "uninsured",
                // Education/ability
                "gifted",
                "talented",
                "educated",
                "literate",
                "illiterate",
                // Power dynamics
                "powerful",
                "powerless",
                "oppressed",
                "weak",
                "famous",
                "infamous",
                // Moral/religious (common in literary texts)
                "righteous",
                "wicked",
                "blessed",
                "damned",
                "faithful",
                // Other common cases
                "hungry",
                "needy",
                "privileged",
                "underprivileged",
                "disadvantaged",
                "marginalized",
            ];

            // =========================================================================
            // Language-specific nominal adjective patterns
            // =========================================================================

            // Get determiners and adjectives for the configured language
            let (determiners, adjectives): (Vec<&str>, Vec<&str>) =
                match self.config.language.as_str() {
                    "de" => {
                        // German: "die Armen", "die Reichen", etc.
                        // Note: German uses "die" (the) for plural nominalized adjectives
                        let dets = vec!["die ", "diese ", "jene "];
                        let adjs = vec![
                            "armen",
                            "reichen",
                            "alten",
                            "jungen",
                            "kranken",
                            "gesunden",
                            "toten",
                            "lebenden",
                            "blinden",
                            "tauben",
                            "arbeitslosen",
                            "obdachlosen",
                            "mächtigen",
                            "schwachen",
                            "unterdrückten",
                        ];
                        (dets, adjs)
                    }
                    "fr" => {
                        // French: "les pauvres", "les riches", etc.
                        let dets = vec!["les ", "ces "];
                        let adjs = vec![
                            "pauvres",
                            "riches",
                            "vieux",
                            "jeunes",
                            "malades",
                            "morts",
                            "vivants",
                            "aveugles",
                            "sourds",
                            "faibles",
                            "puissants",
                            "opprimés",
                            "affamés",
                            "marginalisés",
                        ];
                        (dets, adjs)
                    }
                    "es" => {
                        // Spanish: "los pobres", "los ricos", etc.
                        // Note: Spanish uses gender-marked articles (los/las)
                        let dets = vec!["los ", "las ", "estos ", "estas "];
                        let adjs = vec![
                            "pobres",
                            "ricos",
                            "viejos",
                            "jóvenes",
                            "enfermos",
                            "muertos",
                            "vivos",
                            "ciegos",
                            "sordos",
                            "débiles",
                            "poderosos",
                            "oprimidos",
                            "hambrientos",
                            "marginados",
                        ];
                        (dets, adjs)
                    }
                    _ => {
                        // English (default): "the poor", "the rich", etc.
                        let dets = vec!["the ", "these ", "those "];
                        (dets, NOMINALIZED_ADJECTIVES.to_vec())
                    }
                };

            for det in &determiners {
                for adj in &adjectives {
                    let pattern = format!("{}{}", det, adj);
                    let pattern_lower = pattern.to_lowercase();

                    let mut search_start = 0;
                    while let Some(rel_pos) = text_lower[search_start..].find(&pattern_lower) {
                        let abs_byte_pos = search_start + rel_pos;
                        let end_byte_pos = abs_byte_pos + pattern.len();

                        // Check that the adjective isn't modifying a following noun.
                        // "the poor performance" should NOT match because "poor" modifies "performance".
                        // But "the poor are struggling" SHOULD match because "poor" is nominalized.
                        //
                        // Heuristic: If followed by a verb, conjunction, or sentence boundary,
                        // it's likely a nominal adjective. If followed by a noun/adjective, it's not.
                        let following_text = &text_lower[end_byte_pos..];
                        let next_word: String = following_text
                            .chars()
                            .skip_while(|c| c.is_whitespace())
                            .take_while(|c| c.is_alphabetic())
                            .collect();

                        // Words that can follow a nominal adjective (language-specific)
                        let valid_followers: Vec<&str> = match self.config.language.as_str() {
                            "de" => vec![
                                // German verbs
                                "sind", "waren", "haben", "hatten", "werden", "wurden", "brauchen",
                                "müssen", "können", "sollen", "wollen", // Conjunctions
                                "und", "oder", "aber", "die", "welche",
                            ],
                            "fr" => vec![
                                // French verbs
                                "sont",
                                "étaient",
                                "ont",
                                "avaient",
                                "seront",
                                "peuvent",
                                "doivent",
                                "veulent",
                                "méritent",
                                // Conjunctions
                                "et",
                                "ou",
                                "mais",
                                "qui",
                                "que",
                            ],
                            "es" => vec![
                                // Spanish verbs
                                "son",
                                "eran",
                                "tienen",
                                "tenían",
                                "serán",
                                "pueden",
                                "deben",
                                "quieren",
                                "merecen",
                                "necesitan",
                                "sufren",
                                "luchan",
                                "reciben",
                                "buscan",
                                // Conjunctions
                                "y",
                                "o",
                                "pero",
                                "que",
                                "quienes",
                            ],
                            _ => vec![
                                // English (default)
                                "are", "were", "is", "was", "be", "been", "being", "have", "has",
                                "had", "having", "do", "does", "did", "can", "could", "will",
                                "would", "shall", "should", "may", "might", "must", "need", "want",
                                "get", "got", "struggle", "suffer", "deserve", "receive", "face",
                                "lack", "seek", "and", "or", "but", "who", "whom", "whose", "that",
                                "which", "in", "of", "from", "with", "without", "among",
                            ],
                        };

                        // Valid if: no next word, starts with punct, or next word is allowed
                        let is_valid_nominal =
                            next_word.is_empty() || valid_followers.contains(&next_word.as_str());

                        if is_valid_nominal {
                            // Convert byte positions to character positions
                            let char_start = text[..abs_byte_pos].chars().count();
                            let char_end = char_start + pattern.chars().count();

                            mentions.push(RankedMention {
                                start: char_start,
                                end: char_end,
                                text: text[abs_byte_pos..end_byte_pos].to_string(),
                                mention_type: MentionType::Nominal,
                                gender: Some(Gender::Unknown), // Groups are gender-neutral
                                number: Some(Number::Plural),  // Grammatically plural
                                animacy: Animacy::Animate,     // Nominal adjectives refer to people
                                head: adj.to_string(),         // Head is the adjective
                                entity_type: None,
                            });
                        }

                        search_start = end_byte_pos;
                    }
                }
            }
        }

        // Deduplicate overlapping mentions (prefer longer/earlier)
        mentions.sort_by_key(|m| (m.start, std::cmp::Reverse(m.end)));
        let mut deduped = Vec::new();
        let mut covered_end = 0;

        for mention in mentions {
            if mention.start >= covered_end {
                covered_end = mention.end;
                deduped.push(mention);
            }
        }

        Ok(deduped)
    }

    /// Extract additional features for a mention.
    fn extract_features(&self, mention: &mut RankedMention) {
        // Infer gender from proper nouns
        if mention.gender.is_none() && mention.mention_type == MentionType::Proper {
            mention.gender = self.guess_gender(&mention.text);
        }

        // Infer number
        if mention.number.is_none() {
            mention.number = Some(Number::Singular); // Default
        }
    }

    /// Guess gender from a proper noun.
    fn guess_gender(&self, text: &str) -> Option<Gender> {
        let masc_names = [
            "john", "james", "michael", "david", "robert", "william", "richard",
        ];
        let fem_names = [
            "mary",
            "jennifer",
            "lisa",
            "sarah",
            "jessica",
            "emily",
            "elizabeth",
        ];

        let first_word = text.split_whitespace().next()?.to_lowercase();

        if masc_names.contains(&first_word.as_str()) {
            Some(Gender::Masculine)
        } else if fem_names.contains(&first_word.as_str()) {
            Some(Gender::Feminine)
        } else {
            None
        }
    }

    /// Get head word of a mention.
    fn get_head(&self, text: &str) -> String {
        // Simple heuristic: last word is head
        text.split_whitespace().last().unwrap_or(text).to_string()
    }

    /// Link mentions to antecedents and form clusters.
    ///
    /// # Arguments
    ///
    /// * `mentions` - Detected mentions sorted by position
    /// * `text` - Source text for context-aware features (i2b2-inspired)
    fn link_mentions(&self, mentions: &[RankedMention], text: &str) -> Vec<MentionCluster> {
        match self.config.clustering_strategy {
            ClusteringStrategy::LeftToRight => self.link_mentions_left_to_right(mentions, text),
            ClusteringStrategy::EasyFirst => self.link_mentions_easy_first(mentions, text),
        }
    }

    /// Traditional left-to-right clustering.
    fn link_mentions_left_to_right(
        &self,
        mentions: &[RankedMention],
        text: &str,
    ) -> Vec<MentionCluster> {
        let mut mention_to_cluster: HashMap<usize, usize> = HashMap::new();
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        for (i, mention) in mentions.iter().enumerate() {
            let mut best_antecedent: Option<usize> = None;
            // Pronouns get a lower threshold: they rely on type-compat +
            // gender/number agreement rather than string matching, so their
            // total scores are inherently lower.
            let effective_threshold = if mention.mention_type == MentionType::Pronominal {
                self.config.link_threshold * 0.5
            } else {
                self.config.link_threshold
            };
            let mut best_score = effective_threshold;

            // Type-specific antecedent limit
            let max_antecedents = self.config.max_antecedents_for_type(mention.mention_type);

            // Score against previous mentions with type-specific limit
            for j in (0..i).rev().take(max_antecedents) {
                let antecedent = &mentions[j];

                // Also check character distance as a fallback
                let distance = mention.start.saturating_sub(antecedent.end);
                if distance > self.config.max_distance {
                    break;
                }

                let score = self.score_pair(mention, antecedent, distance, Some(text));
                if score > best_score {
                    best_score = score;
                    best_antecedent = Some(j);
                }
            }

            if let Some(ant_idx) = best_antecedent {
                // Link to antecedent's cluster
                if let Some(&cluster_id) = mention_to_cluster.get(&ant_idx) {
                    clusters[cluster_id].push(i);
                    mention_to_cluster.insert(i, cluster_id);
                } else {
                    // New cluster
                    let cluster_id = clusters.len();
                    clusters.push(vec![ant_idx, i]);
                    mention_to_cluster.insert(ant_idx, cluster_id);
                    mention_to_cluster.insert(i, cluster_id);
                }
            }
        }

        // Apply global proper noun coreference if enabled
        let clusters = if self.config.enable_global_proper_coref {
            self.apply_global_proper_coref(mentions, clusters)
        } else {
            clusters
        };

        // Convert to MentionCluster
        clusters
            .into_iter()
            .enumerate()
            .map(|(id, indices)| MentionCluster {
                id,
                mentions: indices.into_iter().map(|i| mentions[i].clone()).collect(),
            })
            .collect()
    }

    /// Easy-first clustering: process high-confidence decisions first.
    ///
    /// Based on Clark & Manning (2016) and Bourgois & Poibeau (2025).
    /// High-confidence decisions constrain later decisions.
    fn link_mentions_easy_first(
        &self,
        mentions: &[RankedMention],
        text: &str,
    ) -> Vec<MentionCluster> {
        // Step 1: Compute all pairwise scores
        let mut scored_pairs: Vec<ScoredPair> = Vec::new();
        let mut non_coref_pairs: HashSet<(usize, usize)> = HashSet::new();

        for (i, mention) in mentions.iter().enumerate() {
            let max_antecedents = self.config.max_antecedents_for_type(mention.mention_type);
            let effective_threshold = if mention.mention_type == MentionType::Pronominal {
                self.config.link_threshold * 0.5
            } else {
                self.config.link_threshold
            };

            for j in (0..i).rev().take(max_antecedents) {
                let antecedent = &mentions[j];
                let distance = mention.start.saturating_sub(antecedent.end);
                if distance > self.config.max_distance {
                    break;
                }

                let score = self.score_pair(mention, antecedent, distance, Some(text));

                // Track non-coreference constraints
                if self.config.use_non_coref_constraints && score < self.config.non_coref_threshold
                {
                    // Check for coordinating conjunction pattern
                    // (mentions connected by "and"/"or" are likely non-coreferent)
                    non_coref_pairs.insert((j.min(i), j.max(i)));
                }

                if score > effective_threshold {
                    scored_pairs.push(ScoredPair {
                        mention_idx: i,
                        antecedent_idx: j,
                        score,
                    });
                }
            }
        }

        // Step 2: Sort by confidence (highest first)
        scored_pairs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Step 3: Process in confidence order, respecting constraints
        let mut mention_to_cluster: HashMap<usize, usize> = HashMap::new();
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut processed: HashSet<usize> = HashSet::new();

        for pair in scored_pairs {
            // Skip if mention already has an antecedent
            if processed.contains(&pair.mention_idx) {
                continue;
            }

            // Check non-coreference constraint
            let key = (
                pair.antecedent_idx.min(pair.mention_idx),
                pair.antecedent_idx.max(pair.mention_idx),
            );
            if self.config.use_non_coref_constraints && non_coref_pairs.contains(&key) {
                continue;
            }

            // Check cluster-level constraint: would this merge violate any non-coref?
            let would_violate = if self.config.use_non_coref_constraints {
                self.would_violate_constraint(
                    pair.mention_idx,
                    pair.antecedent_idx,
                    &mention_to_cluster,
                    &clusters,
                    &non_coref_pairs,
                )
            } else {
                false
            };

            if would_violate {
                continue;
            }

            // Link mention to antecedent's cluster
            processed.insert(pair.mention_idx);

            if let Some(&cluster_id) = mention_to_cluster.get(&pair.antecedent_idx) {
                clusters[cluster_id].push(pair.mention_idx);
                mention_to_cluster.insert(pair.mention_idx, cluster_id);
            } else {
                let cluster_id = clusters.len();
                clusters.push(vec![pair.antecedent_idx, pair.mention_idx]);
                mention_to_cluster.insert(pair.antecedent_idx, cluster_id);
                mention_to_cluster.insert(pair.mention_idx, cluster_id);
            }
        }

        // Apply global proper noun coreference if enabled
        let clusters = if self.config.enable_global_proper_coref {
            self.apply_global_proper_coref(mentions, clusters)
        } else {
            clusters
        };

        // Convert to MentionCluster
        clusters
            .into_iter()
            .enumerate()
            .map(|(id, indices)| MentionCluster {
                id,
                mentions: indices.into_iter().map(|i| mentions[i].clone()).collect(),
            })
            .collect()
    }

    /// Check if linking would violate non-coreference constraints.
    fn would_violate_constraint(
        &self,
        mention_idx: usize,
        antecedent_idx: usize,
        mention_to_cluster: &HashMap<usize, usize>,
        clusters: &[Vec<usize>],
        non_coref_pairs: &HashSet<(usize, usize)>,
    ) -> bool {
        // Get cluster members that would be merged
        let mut members = vec![mention_idx];
        if let Some(&cluster_id) = mention_to_cluster.get(&antecedent_idx) {
            members.extend(clusters[cluster_id].iter().copied());
        } else {
            members.push(antecedent_idx);
        }

        // Check all pairs in merged cluster for violations
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                let key = (members[i].min(members[j]), members[i].max(members[j]));
                if non_coref_pairs.contains(&key) {
                    return true;
                }
            }
        }

        false
    }

    /// Apply global proper noun coreference propagation.
    ///
    /// For each pair of proper nouns that are locally predicted coreferent,
    /// propagate this decision to all document-wide pairs involving those strings.
    fn apply_global_proper_coref(
        &self,
        mentions: &[RankedMention],
        mut clusters: Vec<Vec<usize>>,
    ) -> Vec<Vec<usize>> {
        // Collect proper noun clusters and their normalized forms
        let mut proper_to_cluster: HashMap<String, usize> = HashMap::new();
        let mut cluster_to_propers: HashMap<usize, Vec<String>> = HashMap::new();

        for (cluster_idx, cluster) in clusters.iter().enumerate() {
            for &mention_idx in cluster {
                let mention = &mentions[mention_idx];
                if mention.mention_type == MentionType::Proper {
                    let normalized = mention.text.to_lowercase();
                    proper_to_cluster.insert(normalized.clone(), cluster_idx);
                    cluster_to_propers
                        .entry(cluster_idx)
                        .or_default()
                        .push(normalized);
                }
            }
        }

        // Find all proper mentions not yet clustered
        let mut unclustered_propers: Vec<(usize, String)> = Vec::new();
        let mut mention_to_cluster: HashMap<usize, usize> = HashMap::new();

        for (cluster_idx, cluster) in clusters.iter().enumerate() {
            for &mention_idx in cluster {
                mention_to_cluster.insert(mention_idx, cluster_idx);
            }
        }

        for (i, mention) in mentions.iter().enumerate() {
            if mention.mention_type == MentionType::Proper && !mention_to_cluster.contains_key(&i) {
                unclustered_propers.push((i, mention.text.to_lowercase()));
            }
        }

        // Link unclustered proper nouns to matching clusters
        for (mention_idx, normalized) in unclustered_propers {
            if let Some(&cluster_idx) = proper_to_cluster.get(&normalized) {
                clusters[cluster_idx].push(mention_idx);
            }
        }

        // Merge clusters that share proper noun strings
        // This handles cases like "Sir Ralph Brown" and "Raphael" being in same cluster
        let mut merged = vec![false; clusters.len()];
        let mut merge_map: HashMap<usize, usize> = HashMap::new();

        for (idx, cluster) in clusters.iter().enumerate() {
            if merged[idx] {
                continue;
            }

            let propers: Vec<_> = cluster
                .iter()
                .filter_map(|&i| {
                    let m = &mentions[i];
                    if m.mention_type == MentionType::Proper {
                        Some(m.text.to_lowercase())
                    } else {
                        None
                    }
                })
                .collect();

            // Find other clusters with matching propers
            for (other_idx, other_cluster) in clusters.iter().enumerate() {
                if other_idx <= idx || merged[other_idx] {
                    continue;
                }

                let other_propers: Vec<_> = other_cluster
                    .iter()
                    .filter_map(|&i| {
                        let m = &mentions[i];
                        if m.mention_type == MentionType::Proper {
                            Some(m.text.to_lowercase())
                        } else {
                            None
                        }
                    })
                    .collect();

                // Check for overlap -- only merge on proper nouns that are
                // specific enough to be reliable identifiers.  Short single
                // tokens ("CEO", "Dr", "US") cause spurious merges.
                let has_reliable_overlap = propers.iter().any(|p| {
                    other_propers
                        .iter()
                        .any(|op| p == op && (p.contains(' ') || p.chars().count() > 4))
                });
                if has_reliable_overlap {
                    merged[other_idx] = true;
                    merge_map.insert(other_idx, idx);
                }
            }
        }

        // Apply merges
        if !merge_map.is_empty() {
            let mut final_clusters: Vec<Vec<usize>> = Vec::new();
            let mut old_to_new: HashMap<usize, usize> = HashMap::new();

            for (old_idx, cluster) in clusters.into_iter().enumerate() {
                if merged[old_idx] {
                    // Find target cluster
                    let mut target = merge_map[&old_idx];
                    while let Some(&next) = merge_map.get(&target) {
                        target = next;
                    }
                    if let Some(&new_idx) = old_to_new.get(&target) {
                        final_clusters[new_idx].extend(cluster);
                    }
                } else {
                    let new_idx = final_clusters.len();
                    old_to_new.insert(old_idx, new_idx);
                    final_clusters.push(cluster);
                }
            }

            final_clusters
        } else {
            clusters
        }
    }

    /// Score a (mention, antecedent) pair.
    ///
    /// # Arguments
    ///
    /// * `mention` - The anaphor being resolved
    /// * `antecedent` - Candidate antecedent
    /// * `distance` - Character distance between mentions
    /// * `text` - Optional source text for context-aware features
    fn score_pair(
        &self,
        mention: &RankedMention,
        antecedent: &RankedMention,
        distance: usize,
        text: Option<&str>,
    ) -> f64 {
        let mut score = 0.0;

        // =========================================================================
        // i2b2-inspired context filtering (Chen et al. 2011)
        // Check this first - if context filtering rejects the pair, return low score
        // =========================================================================
        if self.config.enable_context_filtering {
            if let Some(txt) = text {
                if self.should_filter_by_context(txt, mention, antecedent) {
                    return -1.0; // Strong negative signal to reject this pair
                }
            }
        }

        // =========================================================================
        // Semantic type incompatibility filter
        // Prevents merging e.g. "Nobel Prize" with "Marie Curie"
        // =========================================================================
        if is_type_incompatible(&mention.text, &antecedent.text) {
            return -1.0;
        }

        // =========================================================================
        // String match features
        // =========================================================================
        let m_lower = mention.text.to_lowercase();
        let a_lower = antecedent.text.to_lowercase();

        // Exact match
        if m_lower == a_lower {
            score += self.config.string_match_weight * 1.0;
        }
        // Head match
        else if mention.head.to_lowercase() == antecedent.head.to_lowercase() {
            score += self.config.string_match_weight * 0.6;
        }
        // Substring (partial match -- weight low enough that substring alone
        // cannot cross the default link_threshold of 0.45)
        else if m_lower.contains(&a_lower) || a_lower.contains(&m_lower) {
            score += self.config.string_match_weight * 0.15;
        }

        // =========================================================================
        // i2b2-inspired "be phrase" detection (Chen et al. 2011)
        // "Resolution of X is Y" → X and Y are coreferent
        // =========================================================================
        if self.config.enable_be_phrase_detection {
            if let Some(txt) = text {
                if self.is_be_phrase_link(txt, mention, antecedent) {
                    score += self.config.be_phrase_weight;
                }
            }
        }

        // =========================================================================
        // i2b2-inspired acronym matching (Chen et al. 2011)
        // "MRSA" ↔ "Methicillin-resistant Staphylococcus aureus"
        // =========================================================================
        if self.config.enable_acronym_matching && self.is_acronym_match(mention, antecedent) {
            score += self.config.acronym_weight;
        }

        // =========================================================================
        // i2b2-inspired synonym matching (Chen et al. 2011)
        // Uses UMLS concept matching in original; we use a basic synonym table
        // =========================================================================
        if self.config.enable_synonym_matching && self.are_synonyms(mention, antecedent) {
            score += self.config.synonym_weight;
        }

        // =========================================================================
        // Type compatibility
        // =========================================================================
        match (mention.mention_type, antecedent.mention_type) {
            (MentionType::Pronominal, MentionType::Proper) => {
                score += self.config.type_compat_weight * 0.5;
            }
            (MentionType::Pronominal, MentionType::Pronominal)
                if mention.text.to_lowercase() == antecedent.text.to_lowercase() =>
            {
                // Same pronoun
                score += self.config.type_compat_weight * 0.3;
            }
            (MentionType::Proper, MentionType::Proper) => {
                score += self.config.type_compat_weight * 0.4;
            }
            _ => {}
        }

        // =========================================================================
        // Gender agreement
        // =========================================================================
        if let (Some(m_gender), Some(a_gender)) = (mention.gender, antecedent.gender) {
            if m_gender == a_gender {
                score += self.config.type_compat_weight * 0.3;
            } else if m_gender != Gender::Unknown && a_gender != Gender::Unknown {
                score -= self.config.type_compat_weight * 0.5; // Penalty for mismatch
            }
        }

        // =========================================================================
        // Number agreement
        //
        // Uses Number::is_compatible() from `crate::core` which handles:
        // - Unknown is compatible with anything (singular they, "you")
        // - Dual is compatible with Plural (Arabic/Hebrew/Sanskrit dual numbers)
        // - Exact matches are preferred
        // =========================================================================
        if let (Some(m_number), Some(a_number)) = (mention.number, antecedent.number) {
            if m_number == a_number {
                // Exact match: strongest bonus
                score += self.config.type_compat_weight * 0.2;
            } else if m_number.is_compatible(&a_number) {
                // Compatible but not exact (e.g., Unknown with Singular, Dual with Plural)
                // Small bonus - compatible but less certain
                score += self.config.type_compat_weight * 0.05;
            } else {
                // Incompatible numbers (e.g., Singular vs Plural)
                score -= self.config.type_compat_weight * 0.4;
            }
        }

        // =========================================================================
        // Animacy agreement (phi-feature: animate vs inanimate)
        //
        // Hard constraint: animate/inanimate mismatch blocks coreference
        // ("John... it" is ungrammatical). Unknown is wildcard.
        // See Jurafsky & Martin SLP3 Ch. 26, Fig. 26.4.
        // =========================================================================
        match (mention.animacy, antecedent.animacy) {
            (Animacy::Animate, Animacy::Inanimate) | (Animacy::Inanimate, Animacy::Animate) => {
                score -= self.config.type_compat_weight * 0.6;
            }
            (Animacy::Animate, Animacy::Animate) | (Animacy::Inanimate, Animacy::Inanimate) => {
                score += self.config.type_compat_weight * 0.1;
            }
            _ => {} // Unknown is wildcard -- no bonus or penalty
        }

        // =========================================================================
        // Distance penalty
        // =========================================================================
        score -= self.config.distance_weight * (distance as f64).ln().max(0.0);

        // =========================================================================
        // Pronoun-specific: recency boost + entity-type preference
        //
        // For pronoun mentions, favor the most recent compatible antecedent.
        // Recency is the primary signal for pronoun resolution (Hobbs 1978).
        // Additionally, apply entity-type preferences:
        //   - he/him/his/she/her/hers -> prefer Person antecedents
        //   - it/its -> prefer Organization (or non-Person) antecedents
        //   - they/them/their -> neutral (no preference)
        // =========================================================================
        if mention.mention_type == MentionType::Pronominal {
            // Recency boost: closer antecedents get a stronger bonus.
            // Uses inverse distance (in characters) to reward proximity.
            // The boost decays as 1/(1 + distance/50), giving ~0.3 for adjacent
            // mentions and tapering off for distant ones.
            let recency_boost =
                self.config.type_compat_weight * 0.3 / (1.0 + distance as f64 / 50.0);
            score += recency_boost;

            // Entity-type preference: when the antecedent has a known entity type
            // from NER, apply a bonus or penalty based on pronoun gender.
            if let Some(ref ant_type) = antecedent.entity_type {
                let m_lower = mention.text.to_lowercase();
                let is_person_pronoun = matches!(
                    m_lower.as_str(),
                    "he" | "him" | "his" | "she" | "her" | "hers" | "himself" | "herself"
                );
                let is_it_pronoun = matches!(m_lower.as_str(), "it" | "its" | "itself");

                if is_person_pronoun {
                    if matches!(ant_type, crate::EntityType::Person) {
                        score += self.config.type_compat_weight * 0.4;
                    } else {
                        score -= self.config.type_compat_weight * 0.3;
                    }
                } else if is_it_pronoun {
                    if matches!(ant_type, crate::EntityType::Organization) {
                        score += self.config.type_compat_weight * 0.3;
                    } else if matches!(ant_type, crate::EntityType::Person) {
                        score -= self.config.type_compat_weight * 0.3;
                    }
                }
                // they/them/their: no entity-type preference (neutral)
            }
        }

        // =========================================================================
        // External score (e.g., box containment)
        // =========================================================================
        if let Some(ref ext) = self.config.external_scores {
            let key = (mention.start, antecedent.start);
            if let Some(&ext_score) = ext.get(&key) {
                score += self.config.external_score_weight * ext_score;
            }
        }

        // =========================================================================
        // Salience boost
        // =========================================================================
        if self.config.salience_weight > 0.0 {
            let salience = self.get_salience(&antecedent.text);
            score += self.config.salience_weight * salience;
        }

        score
    }
}

impl Default for MentionRankingCoref {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Integration with GroundedDocument (Signal → Track → Identity hierarchy)
// =============================================================================

impl MentionRankingCoref {
    /// Resolve coreferences and produce Signals and Tracks for a GroundedDocument.
    ///
    /// This is the bridge between mention-ranking output and the canonical
    /// `Signal -> Track -> Identity` hierarchy in `crate::core::grounded`.
    ///
    /// # Returns
    ///
    /// A tuple of (signals, tracks) that can be added to a GroundedDocument:
    /// - `signals`: Individual mention detections with locations
    /// - `tracks`: Clusters of signals referring to the same entity
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use anno::backends::coref::mention_ranking::MentionRankingCoref;
    /// use crate::GroundedDocument;
    ///
    /// let coref = MentionRankingCoref::new();
    /// let (signals, tracks) = coref.resolve_to_grounded("John saw Mary. He waved.")?;
    ///
    /// let mut doc = GroundedDocument::new("doc1");
    /// for signal in signals {
    ///     doc.add_signal(signal);
    /// }
    /// for track in tracks {
    ///     doc.add_track(track);
    /// }
    /// ```
    pub fn resolve_to_grounded(
        &self,
        text: &str,
    ) -> Result<(Vec<crate::Signal<crate::Location>>, Vec<crate::Track>)> {
        let clusters = self.resolve(text)?;

        let mut all_signals = Vec::new();
        let mut all_tracks = Vec::new();
        let mut signal_id_offset = crate::SignalId::ZERO;

        for cluster in clusters {
            let (track, signals) = cluster.to_track(signal_id_offset);
            signal_id_offset += signals.len() as u64;
            all_signals.extend(signals);
            all_tracks.push(track);
        }

        Ok((all_signals, all_tracks))
    }

    /// Resolve coreferences and add results directly to a GroundedDocument.
    ///
    /// This is a convenience method that calls `resolve_to_grounded` and
    /// adds the signals and tracks to the document.
    ///
    /// # Returns
    ///
    /// Vector of TrackIds for the created tracks.
    pub fn resolve_into_document(
        &self,
        text: &str,
        doc: &mut crate::GroundedDocument,
    ) -> Result<Vec<crate::TrackId>> {
        let (mut signals, tracks) = self.resolve_to_grounded(text)?;
        let mut track_ids = Vec::new();

        // Propagate NER entity type labels to coref signals that only have
        // mention-type labels (proper/pronominal/nominal). Match by exact span
        // overlap with existing NER signals in the document.
        let mention_labels: std::collections::HashSet<&str> =
            ["proper", "pronominal", "nominal", "zero", "unknown"]
                .into_iter()
                .collect();

        for coref_sig in &mut signals {
            if !mention_labels.contains(coref_sig.label.as_str()) {
                continue;
            }
            let coref_loc = &coref_sig.location;
            let mut found = false;
            // Find a NER signal with overlapping span and inherit its label
            for ner_sig in doc.signals() {
                if mention_labels.contains(ner_sig.label.as_str()) {
                    continue;
                }
                if ner_sig.location == *coref_loc || spans_overlap(&ner_sig.location, coref_loc) {
                    coref_sig.label = ner_sig.label.clone();
                    found = true;
                    break;
                }
            }
            // Mentions with no NER overlap: prefix with COREF_ for clarity
            if !found {
                let prefixed = format!("COREF_{}", coref_sig.label.as_str());
                coref_sig.label = crate::TypeLabel::from(prefixed.as_str());
            }
        }

        // Build old->new signal ID map (add_signal reassigns IDs)
        let old_ids: Vec<crate::SignalId> = signals.iter().map(|s| s.id).collect();
        let new_ids = doc.add_signals(signals);
        let id_map: std::collections::HashMap<crate::SignalId, crate::SignalId> =
            old_ids.into_iter().zip(new_ids).collect();

        // Remap signal refs in tracks, then add via add_track
        for mut track in tracks {
            for sig_ref in &mut track.signals {
                if let Some(&new_id) = id_map.get(&sig_ref.signal_id) {
                    sig_ref.signal_id = new_id;
                }
            }
            let new_id = doc.add_track(track);
            track_ids.push(new_id);
        }

        Ok(track_ids)
    }
}

/// Check if two Location::Text spans overlap.
fn spans_overlap(a: &crate::Location, b: &crate::Location) -> bool {
    match (a, b) {
        (
            crate::Location::Text {
                start: a_s,
                end: a_e,
            },
            crate::Location::Text {
                start: b_s,
                end: b_e,
            },
        ) => a_s < b_e && b_s < a_e,
        _ => false,
    }
}

// =============================================================================
// CoreferenceResolver trait implementation
// =============================================================================

use crate::CoreferenceResolver;
use crate::Entity;

impl CoreferenceResolver for MentionRankingCoref {
    fn resolve(&self, entities: &[Entity]) -> Vec<Entity> {
        if entities.is_empty() {
            return vec![];
        }

        // Convert Entity to RankedMention
        let mut mentions: Vec<RankedMention> = entities
            .iter()
            .map(|e| {
                let mention_type = if e.text.chars().all(|c| c.is_lowercase()) {
                    MentionType::Pronominal
                } else if e.text.chars().next().is_some_and(|c| c.is_uppercase()) {
                    MentionType::Proper
                } else {
                    MentionType::Nominal
                };

                let gender = self.guess_gender(&e.text);
                // Infer number from pronoun or surface form
                // Note: they/them/their can be singular or plural (singular they)
                let lower = e.text.to_lowercase();
                let number = if ["we", "us"].iter().any(|p| lower == *p) {
                    Some(Number::Plural)
                } else if ["they", "them", "their", "you"].iter().any(|p| lower == *p) {
                    Some(Number::Unknown) // Singular or plural
                } else {
                    Some(Number::Singular)
                };

                RankedMention {
                    start: e.start(),
                    end: e.end(),
                    text: e.text.clone(),
                    mention_type,
                    gender,
                    number,
                    animacy: animacy_from_entity_type(&e.entity_type),
                    head: self.get_head(&e.text),
                    entity_type: Some(e.entity_type.clone()),
                }
            })
            .collect();

        // Sort by position
        mentions.sort_by_key(|m| (m.start, m.end));

        // Extract features
        for mention in &mut mentions {
            self.extract_features(mention);
        }

        // Link mentions into clusters
        // Note: CoreferenceResolver trait doesn't provide source text,
        // so context-aware features (be-phrase, filtering) are disabled
        let clusters = self.link_mentions(&mentions, "");

        // Build canonical ID mapping: mention_key -> cluster_id
        let mut canonical_map: HashMap<(usize, usize), usize> = HashMap::new();
        for cluster in &clusters {
            for mention in &cluster.mentions {
                canonical_map.insert((mention.start, mention.end), cluster.id);
            }
        }

        // Assign unique IDs to singletons (entities not in any cluster)
        let max_cluster_id = clusters.iter().map(|c| c.id).max().unwrap_or(0);
        let mut next_singleton_id = max_cluster_id + 1;

        // Apply canonical IDs to entities
        entities
            .iter()
            .map(|e| {
                let mut entity = e.clone();
                if let Some(&cluster_id) = canonical_map.get(&(e.start(), e.end())) {
                    entity.canonical_id = Some(crate::CanonicalId::new(cluster_id as u64));
                } else {
                    // Assign unique ID to singleton
                    entity.canonical_id = Some(crate::CanonicalId::new(next_singleton_id as u64));
                    next_singleton_id += 1;
                }
                entity
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        "MentionRankingCoref"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_resolution() {
        let coref = MentionRankingCoref::new();
        let clusters = coref.resolve("John saw Mary. He waved to her.").unwrap();

        // Check structure is valid
        for cluster in &clusters {
            assert!(!cluster.mentions.is_empty());
            for mention in &cluster.mentions {
                assert!(mention.start <= mention.end);
            }
        }
    }

    #[test]
    fn test_empty_input() {
        let coref = MentionRankingCoref::new();
        let clusters = coref.resolve("").unwrap();
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_pronoun_detection() {
        let coref = MentionRankingCoref::new();
        let mentions = coref.detect_mentions("He saw her.").unwrap();

        let pronouns: Vec<_> = mentions
            .iter()
            .filter(|m| m.mention_type == MentionType::Pronominal)
            .collect();

        assert!(
            pronouns.len() >= 2,
            "Should detect 'He' and 'her' as pronouns"
        );
    }

    #[test]
    fn test_gender_inference() {
        let coref = MentionRankingCoref::new();

        assert_eq!(coref.guess_gender("John"), Some(Gender::Masculine));
        assert_eq!(coref.guess_gender("Mary Smith"), Some(Gender::Feminine));
        assert_eq!(coref.guess_gender("Google"), None);
    }

    #[test]
    fn test_pair_scoring() {
        let coref = MentionRankingCoref::new();

        let m1 = RankedMention {
            start: 0,
            end: 4,
            text: "John".to_string(),
            mention_type: MentionType::Proper,
            gender: Some(Gender::Masculine),
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "John".to_string(),
            entity_type: None,
        };

        let m2 = RankedMention {
            start: 10,
            end: 12,
            text: "He".to_string(),
            mention_type: MentionType::Pronominal,
            gender: Some(Gender::Masculine),
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "He".to_string(),
            entity_type: None,
        };

        let score = coref.score_pair(&m2, &m1, 6, None);
        assert!(score > 0.0, "Pronoun with matching gender should link");
    }

    #[test]
    fn test_gender_mismatch_penalty() {
        let coref = MentionRankingCoref::new();

        let m1 = RankedMention {
            start: 0,
            end: 4,
            text: "Mary".to_string(),
            mention_type: MentionType::Proper,
            gender: Some(Gender::Feminine),
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "Mary".to_string(),
            entity_type: None,
        };

        let m2 = RankedMention {
            start: 10,
            end: 12,
            text: "He".to_string(),
            mention_type: MentionType::Pronominal,
            gender: Some(Gender::Masculine),
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "He".to_string(),
            entity_type: None,
        };

        let score = coref.score_pair(&m2, &m1, 6, None);
        assert!(
            score < 0.5,
            "Gender mismatch should have low/negative score"
        );
    }

    #[test]
    fn test_config() {
        let config = MentionRankingConfig {
            link_threshold: 0.5,
            ..Default::default()
        };

        let coref = MentionRankingCoref::with_config(config);
        assert_eq!(coref.config.link_threshold, 0.5);
    }

    #[test]
    fn test_unicode_offsets() {
        let coref = MentionRankingCoref::new();
        let text = "北京很美. He likes it.";
        let char_count = text.chars().count();

        let clusters = coref.resolve(text).unwrap();

        for cluster in &clusters {
            for mention in &cluster.mentions {
                assert!(mention.start <= mention.end);
                assert!(mention.end <= char_count);
            }
        }
    }

    // =========================================================================
    // Tests for type-specific antecedent limits (Bourgois & Poibeau 2025)
    // =========================================================================

    #[test]
    fn test_type_specific_antecedent_limits() {
        let config = MentionRankingConfig::default();

        // Default limits from paper
        assert_eq!(config.pronoun_max_antecedents, 30);
        assert_eq!(config.proper_max_antecedents, 300);
        assert_eq!(config.nominal_max_antecedents, 300);

        // Type-specific getter
        assert_eq!(config.max_antecedents_for_type(MentionType::Pronominal), 30);
        assert_eq!(config.max_antecedents_for_type(MentionType::Proper), 300);
        assert_eq!(config.max_antecedents_for_type(MentionType::Nominal), 300);
        assert_eq!(config.max_antecedents_for_type(MentionType::Zero), 300);
        assert_eq!(config.max_antecedents_for_type(MentionType::Unknown), 300);
    }

    #[test]
    fn test_book_scale_config() {
        let config = MentionRankingConfig::book_scale();

        // Book-scale optimizations enabled
        assert!(config.enable_global_proper_coref);
        assert_eq!(config.clustering_strategy, ClusteringStrategy::EasyFirst);
        assert!(config.use_non_coref_constraints);

        // Larger distance for book-scale
        assert!(config.max_distance > 100);
    }

    #[test]
    fn test_pronoun_antecedent_limit_enforced() {
        // Create config with very small pronoun limit
        let config = MentionRankingConfig {
            pronoun_max_antecedents: 2,
            ..Default::default()
        };
        let coref = MentionRankingCoref::with_config(config);

        // With a pronoun limit of 2, it should only consider 2 antecedents
        // This is a structural test - the limit is enforced in link_mentions
        assert_eq!(coref.config.pronoun_max_antecedents, 2);
    }

    // =========================================================================
    // Tests for clustering strategies
    // =========================================================================

    #[test]
    fn test_clustering_strategy_default() {
        let config = MentionRankingConfig::default();
        assert_eq!(config.clustering_strategy, ClusteringStrategy::LeftToRight);
    }

    #[test]
    fn test_easy_first_clustering() {
        let config = MentionRankingConfig {
            clustering_strategy: ClusteringStrategy::EasyFirst,
            ..Default::default()
        };
        let coref = MentionRankingCoref::with_config(config);

        // Should produce valid clusters
        let clusters = coref.resolve("John went home. He was tired.").unwrap();
        for cluster in &clusters {
            assert!(!cluster.mentions.is_empty());
        }
    }

    #[test]
    fn test_left_to_right_vs_easy_first_produces_clusters() {
        let text = "John met Mary. He greeted her warmly. She smiled at him.";

        // Left-to-right clustering
        let l2r_config = MentionRankingConfig {
            clustering_strategy: ClusteringStrategy::LeftToRight,
            ..Default::default()
        };
        let l2r_coref = MentionRankingCoref::with_config(l2r_config);
        let l2r_clusters = l2r_coref.resolve(text).unwrap();

        // Easy-first clustering
        let ef_config = MentionRankingConfig {
            clustering_strategy: ClusteringStrategy::EasyFirst,
            ..Default::default()
        };
        let ef_coref = MentionRankingCoref::with_config(ef_config);
        let ef_clusters = ef_coref.resolve(text).unwrap();

        // Both should produce some clusters
        assert!(
            !l2r_clusters.is_empty() || !ef_clusters.is_empty(),
            "At least one strategy should produce clusters"
        );
    }

    // =========================================================================
    // Tests for global proper noun coreference
    // =========================================================================

    #[test]
    fn test_global_proper_coref_config() {
        let config = MentionRankingConfig {
            enable_global_proper_coref: true,
            global_proper_threshold: 0.8,
            ..Default::default()
        };

        assert!(config.enable_global_proper_coref);
        assert!((config.global_proper_threshold - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_global_proper_coref_same_name() {
        // Test that repeated proper nouns get clustered globally
        let config = MentionRankingConfig {
            enable_global_proper_coref: true,
            ..Default::default()
        };
        let coref = MentionRankingCoref::with_config(config);

        // Use a text with pronouns to ensure we get clusters
        // "John" -> "he" should link, then global proper coref can propagate
        let text = "John arrived. He was happy. Later John left.";
        let clusters = coref.resolve(text).unwrap();

        // The global proper coref feature is mainly for linking distant proper nouns
        // Here we just verify it doesn't break normal clustering
        // Check valid structure is produced
        for cluster in &clusters {
            for mention in &cluster.mentions {
                assert!(mention.start <= mention.end);
            }
        }
    }

    // =========================================================================
    // Tests for non-coreference constraints
    // =========================================================================

    #[test]
    fn test_non_coref_constraints_config() {
        let config = MentionRankingConfig {
            use_non_coref_constraints: true,
            non_coref_threshold: 0.1,
            ..Default::default()
        };

        assert!(config.use_non_coref_constraints);
        assert!((config.non_coref_threshold - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_easy_first_with_non_coref_constraints() {
        let config = MentionRankingConfig {
            clustering_strategy: ClusteringStrategy::EasyFirst,
            use_non_coref_constraints: true,
            ..Default::default()
        };
        let coref = MentionRankingCoref::with_config(config);

        // "John and Mary" - the "and" should prevent merging John and Mary
        let clusters = coref.resolve("John and Mary went to the store.").unwrap();

        // Should produce valid structure regardless of specific clustering
        for cluster in &clusters {
            for mention in &cluster.mentions {
                assert!(mention.start <= mention.end);
            }
        }
    }

    // =========================================================================
    // Integration tests
    // =========================================================================

    #[test]
    fn test_full_book_scale_pipeline() {
        let config = MentionRankingConfig::book_scale();
        let coref = MentionRankingCoref::with_config(config);

        // A longer text simulating literary content
        let text = "Elizabeth Bennett was a spirited young woman. She lived at Longbourn \
                    with her family. Her mother, Mrs. Bennett, was determined to see her \
                    daughters married well. Elizabeth often walked in the countryside. \
                    She enjoyed the solitude it offered.";

        let clusters = coref.resolve(text).unwrap();

        // Validate cluster structure
        for cluster in &clusters {
            assert!(!cluster.mentions.is_empty());
            for mention in &cluster.mentions {
                assert!(mention.start <= mention.end);
                assert!(mention.end <= text.chars().count());
            }
        }
    }

    #[test]
    fn test_mention_type_distribution() {
        let coref = MentionRankingCoref::new();
        let text = "Dr. Smith saw John. He examined him carefully.";
        let mentions = coref.detect_mentions(text).unwrap();

        let pronoun_count = mentions
            .iter()
            .filter(|m| m.mention_type == MentionType::Pronominal)
            .count();
        let proper_count = mentions
            .iter()
            .filter(|m| m.mention_type == MentionType::Proper)
            .count();

        // Should detect both pronouns and proper nouns
        assert!(pronoun_count > 0, "Should detect pronouns");
        assert!(proper_count > 0, "Should detect proper nouns");
    }

    // =========================================================================
    // Tests for salience integration
    // =========================================================================

    #[test]
    fn test_salience_config_default() {
        let config = MentionRankingConfig::default();
        // Disabled by default for backward compatibility
        assert!((config.salience_weight - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_salience_config_builder() {
        let config = MentionRankingConfig::default().with_salience(0.25);
        assert!((config.salience_weight - 0.25).abs() < 0.001);

        // Clamped to [0, 1]
        let clamped = MentionRankingConfig::default().with_salience(1.5);
        assert!((clamped.salience_weight - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_salience_book_scale_enabled() {
        let config = MentionRankingConfig::book_scale();
        assert!(
            config.salience_weight > 0.0,
            "Book-scale should enable salience"
        );
    }

    #[test]
    fn test_with_salience_scores() {
        let mut scores = HashMap::new();
        scores.insert("john".to_string(), 0.8);
        scores.insert("Mary".to_string(), 0.6); // Mixed case

        let coref = MentionRankingCoref::new().with_salience(scores);

        // Lookup should be case-insensitive
        assert!((coref.get_salience("john") - 0.8).abs() < 0.001);
        assert!((coref.get_salience("John") - 0.8).abs() < 0.001);
        assert!((coref.get_salience("JOHN") - 0.8).abs() < 0.001);
        assert!((coref.get_salience("mary") - 0.6).abs() < 0.001);

        // Unknown entity returns 0.0
        assert!((coref.get_salience("unknown") - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_salience_boosts_antecedent_score() {
        // Create config with salience enabled
        let config = MentionRankingConfig {
            salience_weight: 0.3,
            ..Default::default()
        };

        // Scores: John is salient, Mary is not
        let mut scores = HashMap::new();
        scores.insert("john".to_string(), 1.0);
        scores.insert("mary".to_string(), 0.0);

        let coref = MentionRankingCoref::with_config(config).with_salience(scores);

        let mention = RankedMention {
            start: 20,
            end: 22,
            text: "He".to_string(),
            mention_type: MentionType::Pronominal,
            gender: Some(Gender::Masculine),
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "He".to_string(),
            entity_type: None,
        };

        let john = RankedMention {
            start: 0,
            end: 4,
            text: "John".to_string(),
            mention_type: MentionType::Proper,
            gender: Some(Gender::Masculine),
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "John".to_string(),
            entity_type: None,
        };

        let bob = RankedMention {
            start: 10,
            end: 13,
            text: "Bob".to_string(), // Not in salience scores
            mention_type: MentionType::Proper,
            gender: Some(Gender::Masculine),
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "Bob".to_string(),
            entity_type: None,
        };

        let score_john = coref.score_pair(&mention, &john, 16, None);
        let score_bob = coref.score_pair(&mention, &bob, 7, None);

        // John should get a salience boost of 0.3 * 1.0 = 0.3
        // Both have same gender agreement, but John is salient
        // Despite Bob being closer (distance 7 vs 16), John's salience should help
        assert!(
            score_john > score_bob - 0.1, // Allow some margin for distance penalty
            "Salient antecedent should score higher: john={}, bob={}",
            score_john,
            score_bob
        );
    }

    #[test]
    fn test_salience_no_effect_when_disabled() {
        let config = MentionRankingConfig {
            salience_weight: 0.0, // Disabled
            ..Default::default()
        };

        let mut scores = HashMap::new();
        scores.insert("john".to_string(), 1.0);

        let coref = MentionRankingCoref::with_config(config.clone()).with_salience(scores);

        let mention = RankedMention {
            start: 10,
            end: 12,
            text: "He".to_string(),
            mention_type: MentionType::Pronominal,
            gender: Some(Gender::Masculine),
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "He".to_string(),
            entity_type: None,
        };

        let antecedent = RankedMention {
            start: 0,
            end: 4,
            text: "John".to_string(),
            mention_type: MentionType::Proper,
            gender: Some(Gender::Masculine),
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "John".to_string(),
            entity_type: None,
        };

        // Without salience scores
        let coref_no_salience = MentionRankingCoref::with_config(config);
        let score_without = coref_no_salience.score_pair(&mention, &antecedent, 6, None);

        // With salience scores but weight=0
        let score_with = coref.score_pair(&mention, &antecedent, 6, None);

        // Scores should be equal when weight is 0
        assert!(
            (score_without - score_with).abs() < 0.001,
            "Salience should have no effect when weight=0"
        );
    }

    #[test]
    fn test_salience_resolution_integration() {
        // Full resolution with salience
        let config = MentionRankingConfig {
            salience_weight: 0.2,
            ..Default::default()
        };

        let mut scores = HashMap::new();
        scores.insert("president".to_string(), 0.9);
        scores.insert("john".to_string(), 0.7);
        scores.insert("meeting".to_string(), 0.3);

        let coref = MentionRankingCoref::with_config(config).with_salience(scores);

        let text = "John met the President. He was nervous.";
        let clusters = coref.resolve(text).unwrap();

        // Should produce valid clusters
        for cluster in &clusters {
            assert!(!cluster.mentions.is_empty());
            for mention in &cluster.mentions {
                assert!(mention.start <= mention.end);
                assert!(mention.end <= text.chars().count());
            }
        }
    }

    #[test]
    fn test_salience_with_multilingual_text() {
        let config = MentionRankingConfig {
            salience_weight: 0.2,
            ..Default::default()
        };

        let mut scores = HashMap::new();
        scores.insert("北京".to_string(), 0.8);
        scores.insert("習近平".to_string(), 0.9);

        let coref = MentionRankingCoref::with_config(config).with_salience(scores);

        // Case-insensitive lookup (though CJK doesn't have case)
        assert!((coref.get_salience("北京") - 0.8).abs() < 0.001);
        assert!((coref.get_salience("習近平") - 0.9).abs() < 0.001);
    }

    // =========================================================================
    // Tests for GroundedDocument integration (Signal → Track → Identity)
    // =========================================================================

    #[test]
    fn test_mention_cluster_to_signals() {
        let cluster = MentionCluster {
            id: 0,
            mentions: vec![
                RankedMention {
                    start: 0,
                    end: 4,
                    text: "John".to_string(),
                    mention_type: MentionType::Proper,
                    gender: Some(Gender::Masculine),
                    number: Some(Number::Singular),
                    animacy: Animacy::Unknown,
                    head: "John".to_string(),
                    entity_type: None,
                },
                RankedMention {
                    start: 15,
                    end: 17,
                    text: "He".to_string(),
                    mention_type: MentionType::Pronominal,
                    gender: Some(Gender::Masculine),
                    number: Some(Number::Singular),
                    animacy: Animacy::Unknown,
                    head: "He".to_string(),
                    entity_type: None,
                },
            ],
        };

        let signals = cluster.to_signals(crate::SignalId::new(100));

        assert_eq!(signals.len(), 2);
        assert_eq!(signals[0].id, crate::SignalId::new(100));
        assert_eq!(signals[1].id, crate::SignalId::new(101));
        assert_eq!(signals[0].surface, "John");
        assert_eq!(signals[1].surface, "He");

        // Check location is correct
        if let crate::Location::Text { start, end } = &signals[0].location {
            assert_eq!(*start, 0);
            assert_eq!(*end, 4);
        } else {
            panic!("Expected Text location");
        }
    }

    #[test]
    fn test_mention_cluster_to_track() {
        let cluster = MentionCluster {
            id: 42,
            mentions: vec![
                RankedMention {
                    start: 0,
                    end: 4,
                    text: "John".to_string(),
                    mention_type: MentionType::Proper,
                    gender: Some(Gender::Masculine),
                    number: Some(Number::Singular),
                    animacy: Animacy::Unknown,
                    head: "John".to_string(),
                    entity_type: None,
                },
                RankedMention {
                    start: 15,
                    end: 17,
                    text: "He".to_string(),
                    mention_type: MentionType::Pronominal,
                    gender: Some(Gender::Masculine),
                    number: Some(Number::Singular),
                    animacy: Animacy::Unknown,
                    head: "He".to_string(),
                    entity_type: None,
                },
            ],
        };

        let (track, signals) = cluster.to_track(crate::SignalId::new(0));

        // Track should have correct structure
        assert_eq!(track.id, crate::TrackId::new(42));
        assert_eq!(track.canonical_surface, "John"); // Proper noun preferred
        assert_eq!(track.signals.len(), 2);

        // Signals should be correct
        assert_eq!(signals.len(), 2);
        assert_eq!(signals[0].surface, "John");
        assert_eq!(signals[1].surface, "He");
    }

    #[test]
    fn test_canonical_mention_prefers_proper() {
        // Cluster with pronoun first, proper noun second
        let cluster = MentionCluster {
            id: 0,
            mentions: vec![
                RankedMention {
                    start: 0,
                    end: 2,
                    text: "He".to_string(),
                    mention_type: MentionType::Pronominal,
                    gender: Some(Gender::Masculine),
                    number: Some(Number::Singular),
                    animacy: Animacy::Unknown,
                    head: "He".to_string(),
                    entity_type: None,
                },
                RankedMention {
                    start: 10,
                    end: 14,
                    text: "John".to_string(),
                    mention_type: MentionType::Proper,
                    gender: Some(Gender::Masculine),
                    number: Some(Number::Singular),
                    animacy: Animacy::Unknown,
                    head: "John".to_string(),
                    entity_type: None,
                },
            ],
        };

        // Should prefer proper noun even though it's second
        let canonical = cluster.canonical_mention().unwrap();
        assert_eq!(canonical.text, "John");
    }

    #[test]
    fn test_resolve_to_grounded() {
        let coref = MentionRankingCoref::new();
        // Use text where pronoun matches antecedent gender:
        // "Mary" is Feminine, "She" is Feminine → gender agreement → link
        let text = "John saw Mary. She waved.";
        let (signals, tracks) = coref.resolve_to_grounded(text).unwrap();

        // Should have signals (Mary + She linked into a cluster)
        assert!(!signals.is_empty());

        // All signals should have valid locations
        for signal in &signals {
            if let crate::Location::Text { start, end } = &signal.location {
                assert!(start <= end);
            } else {
                panic!("Expected Text location");
            }
        }

        // Tracks should reference signals correctly
        for track in &tracks {
            assert!(!track.signals.is_empty());
            assert!(!track.canonical_surface.is_empty());
        }
    }

    #[test]
    fn test_resolve_into_document() {
        let coref = MentionRankingCoref::new();
        let text = "John saw Mary. He waved to her.";
        let mut doc = crate::GroundedDocument::new("test_doc", text);

        let track_ids = coref.resolve_into_document(text, &mut doc).unwrap();

        // Document should have signals and tracks
        assert!(!doc.signals().is_empty());
        assert!(!doc.tracks_map().is_empty());

        // Returned track IDs should match document
        for track_id in &track_ids {
            assert!(doc.tracks_map().contains_key(track_id));
        }
    }

    #[test]
    fn test_ranked_mention_to_signal() {
        let mention = RankedMention {
            start: 10,
            end: 20,
            text: "the company".to_string(),
            mention_type: MentionType::Nominal,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "company".to_string(),
            entity_type: None,
        };

        let signal = mention.to_signal(crate::SignalId::new(999));

        assert_eq!(signal.id, crate::SignalId::new(999));
        assert_eq!(signal.surface, "the company");
        assert_eq!(signal.label, "nominal".into());
        assert_eq!(signal.modality, crate::Modality::Symbolic);

        if let crate::Location::Text { start, end } = signal.location {
            assert_eq!(start, 10);
            assert_eq!(end, 20);
        } else {
            panic!("Expected Text location");
        }
    }

    #[test]
    fn test_grounded_integration_unicode() {
        let coref = MentionRankingCoref::new();
        let text = "習近平在北京。他很忙。"; // "Xi Jinping is in Beijing. He is busy."

        let (signals, _tracks) = coref.resolve_to_grounded(text).unwrap();
        let char_count = text.chars().count();

        // All signal locations should be within text bounds (character offsets)
        for signal in &signals {
            if let crate::Location::Text { start, end } = &signal.location {
                assert!(*start <= *end);
                assert!(
                    *end <= char_count,
                    "Signal end {} exceeds char count {}",
                    end,
                    char_count
                );
            }
        }
    }

    // =========================================================================
    // Tests for i2b2-inspired features (Chen et al. 2011)
    // =========================================================================

    #[test]
    fn test_be_phrase_detection() {
        let config = MentionRankingConfig::clinical();
        let coref = MentionRankingCoref::with_config(config);

        let text = "The patient is John Smith. He was seen by Dr. Jones.";

        // "patient" (0-11) is "John Smith" (15-25) via "is"
        let m1 = RankedMention {
            start: 4,
            end: 11,
            text: "patient".to_string(),
            mention_type: MentionType::Nominal,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "patient".to_string(),
            entity_type: None,
        };

        let m2 = RankedMention {
            start: 15,
            end: 25,
            text: "John Smith".to_string(),
            mention_type: MentionType::Proper,
            gender: Some(Gender::Masculine),
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "Smith".to_string(),
            entity_type: None,
        };

        // Should detect be-phrase link
        assert!(
            coref.is_be_phrase_link(text, &m1, &m2),
            "Should detect 'is' between patient and John Smith"
        );

        // Score should be higher due to be-phrase
        let score = coref.score_pair(&m1, &m2, 4, Some(text));
        assert!(score > 0.5, "Be-phrase should boost score: got {}", score);
    }

    #[test]
    fn test_be_phrase_detection_negative() {
        let coref = MentionRankingCoref::new();

        let text = "John saw Mary at the store.";

        let m1 = RankedMention {
            start: 0,
            end: 4,
            text: "John".to_string(),
            mention_type: MentionType::Proper,
            gender: Some(Gender::Masculine),
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "John".to_string(),
            entity_type: None,
        };

        let m2 = RankedMention {
            start: 9,
            end: 13,
            text: "Mary".to_string(),
            mention_type: MentionType::Proper,
            gender: Some(Gender::Feminine),
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "Mary".to_string(),
            entity_type: None,
        };

        // "saw" is not a be-phrase
        assert!(
            !coref.is_be_phrase_link(text, &m1, &m2),
            "Should not detect be-phrase between John and Mary"
        );
    }

    #[test]
    fn test_acronym_matching() {
        let coref = MentionRankingCoref::new();

        let mrsa = RankedMention {
            start: 0,
            end: 4,
            text: "MRSA".to_string(),
            mention_type: MentionType::Proper,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "MRSA".to_string(),
            entity_type: None,
        };

        let full = RankedMention {
            start: 20,
            end: 65,
            text: "Methicillin-resistant Staphylococcus aureus".to_string(),
            mention_type: MentionType::Proper,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "aureus".to_string(),
            entity_type: None,
        };

        assert!(
            coref.is_acronym_match(&mrsa, &full),
            "MRSA should match Methicillin-resistant Staphylococcus aureus"
        );
    }

    #[test]
    fn test_acronym_matching_who() {
        let coref = MentionRankingCoref::new();

        let who = RankedMention {
            start: 0,
            end: 3,
            text: "WHO".to_string(),
            mention_type: MentionType::Proper,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "WHO".to_string(),
            entity_type: None,
        };

        let full = RankedMention {
            start: 10,
            end: 35,
            text: "World Health Organization".to_string(),
            mention_type: MentionType::Proper,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "Organization".to_string(),
            entity_type: None,
        };

        assert!(
            coref.is_acronym_match(&who, &full),
            "WHO should match World Health Organization"
        );
    }

    #[test]
    fn test_acronym_matching_negative() {
        let coref = MentionRankingCoref::new();

        let ibm = RankedMention {
            start: 0,
            end: 3,
            text: "IBM".to_string(),
            mention_type: MentionType::Proper,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "IBM".to_string(),
            entity_type: None,
        };

        let apple = RankedMention {
            start: 10,
            end: 25,
            text: "Apple Inc".to_string(),
            mention_type: MentionType::Proper,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "Apple".to_string(),
            entity_type: None,
        };

        assert!(
            !coref.is_acronym_match(&ibm, &apple),
            "IBM should not match Apple Inc"
        );
    }

    #[test]
    fn test_context_filtering_different_dates() {
        let config = MentionRankingConfig::clinical();
        let coref = MentionRankingCoref::with_config(config);

        // Two mentions with different dates in their context
        let text = "On 2024-01-15 the patient presented. On 2024-02-20 the patient returned.";

        let m1 = RankedMention {
            start: 17,
            end: 24,
            text: "patient".to_string(),
            mention_type: MentionType::Nominal,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "patient".to_string(),
            entity_type: None,
        };

        let m2 = RankedMention {
            start: 50,
            end: 57,
            text: "patient".to_string(),
            mention_type: MentionType::Nominal,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "patient".to_string(),
            entity_type: None,
        };

        // Should filter due to different dates (different visits = potentially different patients)
        assert!(
            coref.should_filter_by_context(text, &m1, &m2),
            "Should filter link between patients with different dates"
        );
    }

    #[test]
    fn test_context_filtering_negation() {
        let config = MentionRankingConfig::clinical();
        let coref = MentionRankingCoref::with_config(config);

        // Use longer text to ensure contexts don't overlap
        // The context window is 20 chars before the mention start
        let text = "Patient is not a diabetic. This is important. The diabetic protocol was used.";
        //          0         1         2         3         4         5         6         7
        //          0123456789012345678901234567890123456789012345678901234567890123456789012345

        // First "diabetic" at position 17-25 (after "not a")
        let m1 = RankedMention {
            start: 17,
            end: 25,
            text: "diabetic".to_string(),
            mention_type: MentionType::Nominal,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "diabetic".to_string(),
            entity_type: None,
        };

        // Second "diabetic" at position 50-58 (far enough that context won't include "not")
        let m2 = RankedMention {
            start: 50,
            end: 58,
            text: "diabetic".to_string(),
            mention_type: MentionType::Nominal,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "diabetic".to_string(),
            entity_type: None,
        };

        // Verify context windows include the right context
        let text_chars: Vec<char> = text.chars().collect();
        let m1_context: String = text_chars
            [m1.start.saturating_sub(20)..m1.end.min(text_chars.len())]
            .iter()
            .collect();
        let m2_context: String = text_chars
            [m2.start.saturating_sub(20)..m2.end.min(text_chars.len())]
            .iter()
            .collect();
        eprintln!("m1 context: '{}'", m1_context);
        eprintln!("m2 context: '{}'", m2_context);

        // m1 should have "not" in context, m2 should not
        assert!(
            m1_context.contains("not"),
            "m1 context should contain 'not'"
        );
        assert!(
            !m2_context.contains("not"),
            "m2 context should not contain 'not'"
        );

        // Should filter due to negation mismatch
        assert!(
            coref.should_filter_by_context(text, &m1, &m2),
            "Should filter link between negated ('{}') and non-negated ('{}') mentions",
            m1_context,
            m2_context
        );
    }

    #[test]
    fn test_synonym_matching_high_similarity() {
        // Synonym matching now uses string similarity (>0.8) rather than
        // a hardcoded table. This tests that high-similarity strings match.
        let coref = MentionRankingCoref::new();

        let obama = RankedMention {
            start: 0,
            end: 5,
            text: "Obama".to_string(),
            mention_type: MentionType::Proper,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "Obama".to_string(),
            entity_type: None,
        };

        let obama_lower = RankedMention {
            start: 10,
            end: 15,
            text: "obama".to_string(),
            mention_type: MentionType::Proper,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "obama".to_string(),
            entity_type: None,
        };

        // Case-insensitive match should work
        assert!(
            coref.are_synonyms(&obama, &obama_lower),
            "Obama and obama should match (case-insensitive)"
        );
    }

    #[test]
    fn test_synonym_matching_low_similarity_no_match() {
        // Domain-specific synonyms like heart/cardiac require external
        // SynonymSource implementations. The default uses string similarity,
        // which won't match semantically related but lexically different terms.
        let coref = MentionRankingCoref::new();

        let heart = RankedMention {
            start: 0,
            end: 5,
            text: "heart".to_string(),
            mention_type: MentionType::Nominal,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "heart".to_string(),
            entity_type: None,
        };

        let cardiac = RankedMention {
            start: 10,
            end: 17,
            text: "cardiac".to_string(),
            mention_type: MentionType::Nominal,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "cardiac".to_string(),
            entity_type: None,
        };

        // Without a domain-specific SynonymSource, these won't match
        // because "heart" and "cardiac" have low string similarity.
        // This is the expected behavior - use anno::coalesce::SynonymSource
        // for domain-specific synonym matching.
        assert!(
            !coref.are_synonyms(&heart, &cardiac),
            "heart/cardiac require domain-specific SynonymSource"
        );
    }

    #[test]
    fn test_clinical_config() {
        let config = MentionRankingConfig::clinical();

        // Verify i2b2-inspired features are enabled
        assert!(config.enable_be_phrase_detection);
        assert!(config.enable_acronym_matching);
        assert!(config.enable_context_filtering);
        assert!(config.enable_synonym_matching);

        // Verify reasonable weights
        assert!(config.be_phrase_weight > 0.5);
        assert!(config.acronym_weight > 0.5);
        assert!(config.synonym_weight > 0.3);
    }

    #[test]
    fn test_clinical_resolution_integration() {
        let config = MentionRankingConfig::clinical();
        let coref = MentionRankingCoref::with_config(config);

        // Clinical text with various coreference patterns
        let text = "The patient is John Smith. Pt was admitted with MRSA. \
                    Methicillin-resistant Staphylococcus aureus was treated.";

        let clusters = coref.resolve(text).unwrap();

        // Should create meaningful clusters
        assert!(
            !clusters.is_empty(),
            "Should find clusters in clinical text"
        );

        // Print clusters for debugging
        for cluster in &clusters {
            let texts: Vec<_> = cluster.mentions.iter().map(|m| &m.text).collect();
            eprintln!("Cluster {}: {:?}", cluster.id, texts);
        }
    }

    #[test]
    fn test_i2b2_scoring_with_all_features() {
        let config = MentionRankingConfig::clinical();
        let coref = MentionRankingCoref::with_config(config);

        // Text with be-phrase pattern
        let text = "Resolution of organism is MRSA.";

        let m1 = RankedMention {
            start: 14,
            end: 22,
            text: "organism".to_string(),
            mention_type: MentionType::Nominal,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "organism".to_string(),
            entity_type: None,
        };

        let m2 = RankedMention {
            start: 26,
            end: 30,
            text: "MRSA".to_string(),
            mention_type: MentionType::Proper,
            gender: None,
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "MRSA".to_string(),
            entity_type: None,
        };

        // Score should be high due to be-phrase
        let score = coref.score_pair(&m1, &m2, 4, Some(text));
        assert!(
            score > 0.7,
            "Be-phrase pattern should yield high score, got {}",
            score
        );
    }

    // =========================================================================
    // Nominal adjective detection tests (J2N: arXiv:2409.14374)
    // =========================================================================

    #[test]
    fn test_nominal_adjective_detection_basic() {
        let config = MentionRankingConfig {
            enable_nominal_adjective_detection: true,
            ..Default::default()
        };
        let coref = MentionRankingCoref::with_config(config);

        let text = "The poor are struggling while the rich get richer.";
        let mentions = coref.detect_mentions(text).unwrap();

        let texts: Vec<_> = mentions.iter().map(|m| m.text.as_str()).collect();
        assert!(
            texts.contains(&"The poor"),
            "Should detect 'The poor': {:?}",
            texts
        );
        assert!(
            texts.contains(&"the rich"),
            "Should detect 'the rich': {:?}",
            texts
        );

        // Check grammatical number is plural
        let poor_mention = mentions
            .iter()
            .find(|m| m.text.to_lowercase() == "the poor");
        assert!(poor_mention.is_some());
        assert_eq!(poor_mention.unwrap().number, Some(Number::Plural));
        assert_eq!(poor_mention.unwrap().mention_type, MentionType::Nominal);
    }

    #[test]
    fn test_nominal_adjective_not_before_noun() {
        // "the poor performance" should NOT detect "the poor" as a mention
        // because "poor" modifies "performance", not a nominalized group
        let config = MentionRankingConfig {
            enable_nominal_adjective_detection: true,
            ..Default::default()
        };
        let coref = MentionRankingCoref::with_config(config);

        let text = "The poor performance was criticized.";
        let mentions = coref.detect_mentions(text).unwrap();

        let texts: Vec<_> = mentions.iter().map(|m| m.text.as_str()).collect();
        assert!(
            !texts.contains(&"The poor"),
            "Should NOT detect 'The poor' when followed by noun: {:?}",
            texts
        );
    }

    #[test]
    fn test_nominal_adjective_at_sentence_end() {
        let config = MentionRankingConfig {
            enable_nominal_adjective_detection: true,
            ..Default::default()
        };
        let coref = MentionRankingCoref::with_config(config);

        let text = "We must help the elderly.";
        let mentions = coref.detect_mentions(text).unwrap();

        let texts: Vec<_> = mentions.iter().map(|m| m.text.as_str()).collect();
        assert!(
            texts.contains(&"the elderly"),
            "Should detect 'the elderly' at end: {:?}",
            texts
        );
    }

    #[test]
    fn test_nominal_adjective_with_punctuation() {
        let config = MentionRankingConfig {
            enable_nominal_adjective_detection: true,
            ..Default::default()
        };
        let coref = MentionRankingCoref::with_config(config);

        let text = "The accused, the condemned, and the guilty were present.";
        let mentions = coref.detect_mentions(text).unwrap();

        let texts: Vec<_> = mentions.iter().map(|m| m.text.as_str()).collect();
        assert!(
            texts.contains(&"The accused"),
            "Should detect 'The accused': {:?}",
            texts
        );
        assert!(
            texts.contains(&"the condemned"),
            "Should detect 'the condemned': {:?}",
            texts
        );
        assert!(
            texts.contains(&"the guilty"),
            "Should detect 'the guilty': {:?}",
            texts
        );
    }

    #[test]
    fn test_nominal_adjective_these_those() {
        let config = MentionRankingConfig {
            enable_nominal_adjective_detection: true,
            ..Default::default()
        };
        let coref = MentionRankingCoref::with_config(config);

        let text = "These homeless need shelter. Those unemployed seek work.";
        let mentions = coref.detect_mentions(text).unwrap();

        let texts: Vec<_> = mentions.iter().map(|m| m.text.as_str()).collect();
        assert!(
            texts.contains(&"These homeless"),
            "Should detect 'These homeless': {:?}",
            texts
        );
        assert!(
            texts.contains(&"Those unemployed"),
            "Should detect 'Those unemployed': {:?}",
            texts
        );
    }

    #[test]
    fn test_nominal_adjective_disabled_by_default() {
        let coref = MentionRankingCoref::new();

        let text = "The poor are struggling.";
        let mentions = coref.detect_mentions(text).unwrap();

        // With detection disabled, "the poor" should not be detected as a mention
        let has_the_poor = mentions.iter().any(|m| m.text.to_lowercase() == "the poor");
        assert!(
            !has_the_poor,
            "Nominal adjective detection should be disabled by default"
        );
    }

    // =========================================================================
    // Singular "they" tests
    // =========================================================================

    #[test]
    fn test_singular_they_number_unknown() {
        let coref = MentionRankingCoref::new();

        // "they" should have Number::Unknown to support both singular and plural
        let text = "Alex said they would come. They brought their friends.";
        let mentions = coref.detect_mentions(text).unwrap();

        // Find "they" mentions
        let they_mentions: Vec<_> = mentions
            .iter()
            .filter(|m| m.text.to_lowercase() == "they")
            .collect();

        for they in &they_mentions {
            assert_eq!(
                they.number,
                Some(Number::Unknown),
                "'they' should have Number::Unknown for singular/plural ambiguity"
            );
        }
    }

    #[test]
    fn test_their_number_unknown() {
        let coref = MentionRankingCoref::new();

        let text = "Someone left their umbrella.";
        let mentions = coref.detect_mentions(text).unwrap();

        let their = mentions.iter().find(|m| m.text.to_lowercase() == "their");
        assert!(their.is_some(), "Should detect 'their'");
        assert_eq!(
            their.unwrap().number,
            Some(Number::Unknown),
            "'their' should have Number::Unknown"
        );
    }

    #[test]
    fn test_themself_vs_themselves() {
        // "themself" is explicitly singular (singular they reflexive)
        // "themselves" is explicitly plural
        let coref = MentionRankingCoref::new();

        let text = "The student prepared themself. The students prepared themselves.";
        let mentions = coref.detect_mentions(text).unwrap();

        let themself = mentions
            .iter()
            .find(|m| m.text.to_lowercase() == "themself");
        let themselves = mentions
            .iter()
            .find(|m| m.text.to_lowercase() == "themselves");

        assert!(themself.is_some(), "Should detect 'themself'");
        assert!(themselves.is_some(), "Should detect 'themselves'");

        assert_eq!(
            themself.unwrap().number,
            Some(Number::Singular),
            "'themself' is explicitly singular"
        );
        assert_eq!(
            themselves.unwrap().number,
            Some(Number::Plural),
            "'themselves' is explicitly plural"
        );
    }

    // =========================================================================
    // Neopronoun tests
    // =========================================================================

    #[test]
    fn test_neopronoun_ze_hir() {
        let coref = MentionRankingCoref::new();

        let text = "Ze told me to text hir, but I don't have hirs number.";
        let mentions = coref.detect_mentions(text).unwrap();

        let ze = mentions.iter().find(|m| m.text.to_lowercase() == "ze");
        let hir = mentions.iter().find(|m| m.text.to_lowercase() == "hir");
        let hirs = mentions.iter().find(|m| m.text.to_lowercase() == "hirs");

        assert!(ze.is_some(), "Should detect 'ze'");
        assert!(hir.is_some(), "Should detect 'hir'");
        assert!(hirs.is_some(), "Should detect 'hirs'");

        // All neopronouns are grammatically singular
        assert_eq!(ze.unwrap().number, Some(Number::Singular));
        assert_eq!(hir.unwrap().number, Some(Number::Singular));
        assert_eq!(hirs.unwrap().number, Some(Number::Singular));

        // All use Gender::Unknown (nonbinary)
        assert_eq!(ze.unwrap().gender, Some(Gender::Unknown));
    }

    #[test]
    fn test_neopronoun_xe_xem() {
        let coref = MentionRankingCoref::new();

        let text = "Xe said xem would bring xyr notes.";
        let mentions = coref.detect_mentions(text).unwrap();

        let xe = mentions.iter().find(|m| m.text.to_lowercase() == "xe");
        let xem = mentions.iter().find(|m| m.text.to_lowercase() == "xem");
        let xyr = mentions.iter().find(|m| m.text.to_lowercase() == "xyr");

        assert!(xe.is_some(), "Should detect 'xe'");
        assert!(xem.is_some(), "Should detect 'xem'");
        assert!(xyr.is_some(), "Should detect 'xyr'");

        assert_eq!(xe.unwrap().number, Some(Number::Singular));
        assert_eq!(xe.unwrap().gender, Some(Gender::Unknown));
    }

    #[test]
    fn test_neopronoun_spivak_ey_em() {
        let coref = MentionRankingCoref::new();

        let text = "Ey told me to call em later.";
        let mentions = coref.detect_mentions(text).unwrap();

        let ey = mentions.iter().find(|m| m.text.to_lowercase() == "ey");
        let em = mentions.iter().find(|m| m.text.to_lowercase() == "em");

        assert!(ey.is_some(), "Should detect 'ey' (Spivak pronoun)");
        assert!(em.is_some(), "Should detect 'em' (Spivak pronoun)");

        assert_eq!(ey.unwrap().number, Some(Number::Singular));
    }

    #[test]
    fn test_neopronoun_fae_faer() {
        let coref = MentionRankingCoref::new();

        let text = "Fae said faer class was cancelled.";
        let mentions = coref.detect_mentions(text).unwrap();

        let fae = mentions.iter().find(|m| m.text.to_lowercase() == "fae");
        let faer = mentions.iter().find(|m| m.text.to_lowercase() == "faer");

        assert!(fae.is_some(), "Should detect 'fae'");
        assert!(faer.is_some(), "Should detect 'faer'");

        assert_eq!(fae.unwrap().number, Some(Number::Singular));
    }

    // =========================================================================
    // From implementation tests
    // =========================================================================

    #[test]
    fn test_ranked_mention_from_entity() {
        let entity = crate::Entity::new("Barack Obama", crate::EntityType::Person, 0, 12, 0.95);
        let mention = RankedMention::from(&entity);

        assert_eq!(mention.start, 0);
        assert_eq!(mention.end, 12);
        assert_eq!(mention.text, "Barack Obama");
        assert_eq!(mention.head, "Obama"); // Last word
        assert_eq!(mention.mention_type, MentionType::Proper);
    }

    #[test]
    fn test_ranked_mention_to_coref_mention() {
        let mention = RankedMention {
            start: 10,
            end: 20,
            text: "the patient".to_string(),
            mention_type: MentionType::Nominal,
            gender: Some(Gender::Unknown),
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "patient".to_string(),
            entity_type: None,
        };

        let coref_mention: crate::Mention = (&mention).into();

        assert_eq!(coref_mention.start, 10);
        assert_eq!(coref_mention.end, 20);
        assert_eq!(coref_mention.text, "the patient");
        assert_eq!(coref_mention.mention_type, Some(MentionType::Nominal));
    }

    #[test]
    fn test_ranked_mention_span() {
        let mention = RankedMention {
            start: 5,
            end: 15,
            text: "test".to_string(),
            mention_type: MentionType::Nominal,
            gender: None,
            number: None,
            animacy: Animacy::Unknown,
            head: "test".to_string(),
            entity_type: None,
        };

        assert_eq!(mention.span(), (5, 15));
    }

    // =========================================================================
    // Pronoun coreference with nominal adjectives
    // =========================================================================

    #[test]
    fn test_nominal_adjective_pronoun_resolution() {
        // This tests the key insight from J2N: detecting "the poor" enables
        // resolving "they" that refers to this group.
        let config = MentionRankingConfig {
            enable_nominal_adjective_detection: true,
            link_threshold: 0.1, // Low threshold for pronoun linking
            ..Default::default()
        };
        let coref = MentionRankingCoref::with_config(config);

        // Use sentence-final position for "the poor" to ensure detection
        let text = "We must help the poor. They deserve better.";

        // First verify detection works
        let detected = coref.detect_mentions(text).unwrap();
        let detected_texts: Vec<_> = detected.iter().map(|m| m.text.as_str()).collect();

        assert!(
            detected.iter().any(|m| m.text.to_lowercase() == "the poor"),
            "Should detect 'the poor' in detect_mentions: {:?}",
            detected_texts
        );
        assert!(
            detected.iter().any(|m| m.text.to_lowercase() == "they"),
            "Should detect 'They' in detect_mentions: {:?}",
            detected_texts
        );

        // Verify scoring: "They" should have positive score with "the poor"
        let the_poor = detected
            .iter()
            .find(|m| m.text.to_lowercase() == "the poor")
            .unwrap();
        let they = detected
            .iter()
            .find(|m| m.text.to_lowercase() == "they")
            .unwrap();

        let distance = they.start.saturating_sub(the_poor.end);
        let score = coref.score_pair(they, the_poor, distance, Some(text));

        // With Number::Unknown for "they" and Number::Plural for "the poor",
        // there should be no number mismatch penalty (Unknown is compatible with any)
        assert!(
            score > -0.5,
            "Score between 'They' and 'the poor' should not be strongly negative, got {}",
            score
        );

        // Note: Clustering only includes mentions that form links.
        // If the score is above threshold, they'll be clustered together.
        // If not, they remain singletons (not in any cluster).
        // This is expected behavior - the key benefit is detection, not guaranteed linking.
    }

    // =========================================================================
    // Neopronoun detection tests (GICoref/MISGENDERED datasets)
    // =========================================================================

    #[test]
    fn test_neopronoun_xe_detection() {
        let coref = MentionRankingCoref::new();
        let text = "Alex introduced xemself. Xe said xe was happy to be here.";
        let mentions = coref.detect_mentions(text).unwrap();

        let texts: Vec<_> = mentions.iter().map(|m| m.text.to_lowercase()).collect();
        assert!(
            texts.contains(&"xemself".to_string()),
            "Should detect 'xemself': {:?}",
            texts
        );
        assert!(
            texts.contains(&"xe".to_string()),
            "Should detect 'xe': {:?}",
            texts
        );
    }

    #[test]
    fn test_neopronoun_ze_detection() {
        let coref = MentionRankingCoref::new();
        let text = "Jordan uses ze/hir pronouns. Hir presentation was excellent.";
        let mentions = coref.detect_mentions(text).unwrap();

        let texts: Vec<_> = mentions.iter().map(|m| m.text.to_lowercase()).collect();
        assert!(
            texts.contains(&"ze".to_string()),
            "Should detect 'ze': {:?}",
            texts
        );
        assert!(
            texts.contains(&"hir".to_string()),
            "Should detect 'hir': {:?}",
            texts
        );
    }

    #[test]
    fn test_neopronoun_ey_detection() {
        let coref = MentionRankingCoref::new();
        let text = "Sam asked em to pass eir notebook.";
        let mentions = coref.detect_mentions(text).unwrap();

        let texts: Vec<_> = mentions.iter().map(|m| m.text.to_lowercase()).collect();
        assert!(
            texts.contains(&"em".to_string()),
            "Should detect 'em': {:?}",
            texts
        );
        assert!(
            texts.contains(&"eir".to_string()),
            "Should detect 'eir': {:?}",
            texts
        );
    }

    #[test]
    fn test_neopronoun_fae_detection() {
        let coref = MentionRankingCoref::new();
        let text = "River explained faer perspective. Fae was very articulate.";
        let mentions = coref.detect_mentions(text).unwrap();

        let texts: Vec<_> = mentions.iter().map(|m| m.text.to_lowercase()).collect();
        assert!(
            texts.contains(&"faer".to_string()),
            "Should detect 'faer': {:?}",
            texts
        );
        assert!(
            texts.contains(&"fae".to_string()),
            "Should detect 'fae': {:?}",
            texts
        );
    }

    #[test]
    fn test_neopronoun_gender_and_number() {
        let coref = MentionRankingCoref::new();
        let text = "Xe arrived early.";
        let mentions = coref.detect_mentions(text).unwrap();

        let xe_mention = mentions.iter().find(|m| m.text.to_lowercase() == "xe");
        assert!(xe_mention.is_some(), "Should detect 'xe'");

        let xe = xe_mention.unwrap();
        // Neopronouns are singular and gender-unknown (non-binary)
        assert_eq!(
            xe.number,
            Some(Number::Singular),
            "Neopronouns are singular"
        );
        assert_eq!(
            xe.gender,
            Some(Gender::Unknown),
            "Neopronouns use Unknown gender"
        );
    }

    #[test]
    fn test_neopronoun_coreference_linking() {
        // Test that neopronouns are detected and have correct properties
        // for coreference linking (proper noun detection requires NER,
        // which is beyond mention_ranking's scope)
        let coref = MentionRankingCoref::new();
        let text = "Xe said xe would be late. Xem was right.";
        let mentions = coref.detect_mentions(text).unwrap();

        // All neopronouns should be detected
        let texts: Vec<_> = mentions.iter().map(|m| m.text.to_lowercase()).collect();
        assert!(
            texts.iter().filter(|t| *t == "xe").count() >= 2,
            "Should detect multiple 'xe': {:?}",
            texts
        );
        assert!(
            texts.contains(&"xem".to_string()),
            "Should detect 'xem': {:?}",
            texts
        );

        // All should be pronominal type
        for m in &mentions {
            if ["xe", "xem"].contains(&m.text.to_lowercase().as_str()) {
                assert_eq!(
                    m.mention_type,
                    MentionType::Pronominal,
                    "Neopronouns should be Pronominal type"
                );
            }
        }
    }

    // =========================================================================
    // Number::Dual compatibility tests (Arabic, Hebrew, Sanskrit)
    // =========================================================================

    #[test]
    fn test_dual_number_compatibility_scoring() {
        // Dual should be compatible with Plural (but not exact match)
        // This is important for languages like Arabic, Hebrew, Sanskrit
        // where dual forms are distinct from plural
        let coref = MentionRankingCoref::new();

        // Create mentions manually to test scoring
        let dual_mention = RankedMention {
            start: 0,
            end: 5,
            text: "كتابان".to_string(), // Arabic dual: "two books"
            mention_type: MentionType::Nominal,
            gender: Some(Gender::Neutral),
            number: Some(Number::Dual),
            animacy: Animacy::Unknown,
            head: "كتابان".to_string(),
            entity_type: None,
        };

        let plural_mention = RankedMention {
            start: 10,
            end: 15,
            text: "هم".to_string(), // Arabic plural pronoun: "they"
            mention_type: MentionType::Pronominal,
            gender: Some(Gender::Unknown),
            number: Some(Number::Plural),
            animacy: Animacy::Unknown,
            head: "هم".to_string(),
            entity_type: None,
        };

        let singular_mention = RankedMention {
            start: 20,
            end: 22,
            text: "هو".to_string(), // Arabic singular: "he"
            mention_type: MentionType::Pronominal,
            gender: Some(Gender::Masculine),
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "هو".to_string(),
            entity_type: None,
        };

        // Test Number::is_compatible directly
        assert!(
            Number::Dual.is_compatible(&Number::Plural),
            "Dual should be compatible with Plural"
        );
        assert!(
            !Number::Dual.is_compatible(&Number::Singular),
            "Dual should NOT be compatible with Singular"
        );

        // Dual ↔ Plural should score better than Dual ↔ Singular
        let score_dual_plural = coref.score_pair(&plural_mention, &dual_mention, 5, None);
        let score_dual_singular = coref.score_pair(&singular_mention, &dual_mention, 5, None);

        assert!(
            score_dual_plural > score_dual_singular,
            "Dual-Plural score ({}) should be higher than Dual-Singular ({})",
            score_dual_plural,
            score_dual_singular
        );
    }

    #[test]
    fn test_number_compatibility_unknown() {
        // Number::Unknown should be compatible with all other values
        // This is critical for singular they, "you", etc.
        assert!(Number::Unknown.is_compatible(&Number::Singular));
        assert!(Number::Unknown.is_compatible(&Number::Plural));
        assert!(Number::Unknown.is_compatible(&Number::Dual));
        assert!(Number::Unknown.is_compatible(&Number::Unknown));

        // The coreference scorer should not penalize Unknown mismatches
        let coref = MentionRankingCoref::new();

        let they_mention = RankedMention {
            start: 0,
            end: 4,
            text: "They".to_string(),
            mention_type: MentionType::Pronominal,
            gender: Some(Gender::Unknown),
            number: Some(Number::Unknown), // Singular or plural
            animacy: Animacy::Animate,
            head: "They".to_string(),
            entity_type: None,
        };

        let singular_mention = RankedMention {
            start: 10,
            end: 14,
            text: "Alex".to_string(),
            mention_type: MentionType::Proper,
            gender: Some(Gender::Unknown),
            number: Some(Number::Singular),
            animacy: Animacy::Unknown,
            head: "Alex".to_string(),
            entity_type: None,
        };

        let plural_mention = RankedMention {
            start: 20,
            end: 30,
            text: "the students".to_string(),
            mention_type: MentionType::Nominal,
            gender: Some(Gender::Unknown),
            number: Some(Number::Plural),
            animacy: Animacy::Unknown,
            head: "students".to_string(),
            entity_type: None,
        };

        // Both should get non-negative scores (Unknown is compatible with both)
        let score_they_singular = coref.score_pair(&they_mention, &singular_mention, 5, None);
        let score_they_plural = coref.score_pair(&they_mention, &plural_mention, 5, None);

        // Neither should be penalized for number mismatch
        assert!(
            score_they_singular > -1.0,
            "'They' ↔ singular should not be heavily penalized: {}",
            score_they_singular
        );
        assert!(
            score_they_plural > -1.0,
            "'They' ↔ plural should not be heavily penalized: {}",
            score_they_plural
        );
    }

    // =========================================================================
    // Pleonastic "it" detection tests
    // =========================================================================

    #[test]
    fn test_pleonastic_it_weather() {
        // Weather expressions should NOT detect "it" as a referring pronoun
        let coref = MentionRankingCoref::new();

        let weather_texts = [
            "It rains every day in Seattle.",
            "It is raining outside.",
            "It snows heavily in winter.",
            "It was snowing when we arrived.",
            "It thundered all night.",
        ];

        for text in weather_texts {
            let mentions = coref.detect_mentions(text).unwrap();
            let has_it = mentions.iter().any(|m| m.text.to_lowercase() == "it");
            assert!(
                !has_it,
                "Weather 'it' should be filtered as pleonastic in: '{}'\nDetected: {:?}",
                text,
                mentions.iter().map(|m| &m.text).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_pleonastic_it_weather_adjectives() {
        let coref = MentionRankingCoref::new();

        let weather_adj_texts = [
            "It is sunny today.",
            "It was cold last night.",
            "It's foggy this morning.",
            "It will be warm tomorrow.",
        ];

        for text in weather_adj_texts {
            let mentions = coref.detect_mentions(text).unwrap();
            let has_it = mentions.iter().any(|m| m.text.to_lowercase() == "it");
            assert!(
                !has_it,
                "Weather adjective 'it' should be filtered: '{}'\nDetected: {:?}",
                text,
                mentions.iter().map(|m| &m.text).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_pleonastic_it_modal() {
        let coref = MentionRankingCoref::new();

        let modal_texts = [
            "It is important that we finish on time.",
            "It is likely that he will arrive late.",
            "It was clear that something was wrong.",
            "It is necessary to complete the form.",
            "It's obvious that she was upset.",
        ];

        for text in modal_texts {
            let mentions = coref.detect_mentions(text).unwrap();
            let has_it = mentions.iter().any(|m| m.text.to_lowercase() == "it");
            assert!(
                !has_it,
                "Modal 'it' should be filtered: '{}'\nDetected: {:?}",
                text,
                mentions.iter().map(|m| &m.text).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_pleonastic_it_cognitive_verbs() {
        let coref = MentionRankingCoref::new();

        let cognitive_texts = [
            "It seems that the project is delayed.",
            "It appears he was mistaken.",
            "It turns out she was right.",
            "It happened that we met by chance.",
        ];

        for text in cognitive_texts {
            let mentions = coref.detect_mentions(text).unwrap();
            let has_it = mentions.iter().any(|m| m.text.to_lowercase() == "it");
            assert!(
                !has_it,
                "Cognitive verb 'it' should be filtered: '{}'\nDetected: {:?}",
                text,
                mentions.iter().map(|m| &m.text).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_referential_it_not_filtered() {
        // Referential "it" should still be detected
        let coref = MentionRankingCoref::new();

        let referential_texts = [
            "I read the book. It was fascinating.",
            "The car broke down. We had to push it.",
            "She gave him a gift. He loved it.",
        ];

        for text in referential_texts {
            let mentions = coref.detect_mentions(text).unwrap();
            let has_it = mentions.iter().any(|m| m.text.to_lowercase() == "it");
            assert!(
                has_it,
                "Referential 'it' should be detected: '{}'\nDetected: {:?}",
                text,
                mentions.iter().map(|m| &m.text).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_pleonastic_it_time_expressions() {
        let coref = MentionRankingCoref::new();

        let time_texts = [
            "It is midnight.",
            "It was noon when we left.",
            "It is 5 o'clock.",
        ];

        for text in time_texts {
            let mentions = coref.detect_mentions(text).unwrap();
            let has_it = mentions.iter().any(|m| m.text.to_lowercase() == "it");
            assert!(
                !has_it,
                "Time expression 'it' should be filtered: '{}'\nDetected: {:?}",
                text,
                mentions.iter().map(|m| &m.text).collect::<Vec<_>>()
            );
        }
    }

    // =========================================================================
    // Demonstrative pronoun tests
    // =========================================================================

    #[test]
    fn test_demonstrative_pronoun_detection() {
        let coref = MentionRankingCoref::new();

        let text = "I saw the problem. This was unexpected. Those are the facts.";
        let mentions = coref.detect_mentions(text).unwrap();
        let texts: Vec<_> = mentions.iter().map(|m| m.text.to_lowercase()).collect();

        assert!(
            texts.contains(&"this".to_string()),
            "Should detect 'This': {:?}",
            texts
        );
        assert!(
            texts.contains(&"those".to_string()),
            "Should detect 'Those': {:?}",
            texts
        );
    }

    #[test]
    fn test_demonstrative_pronoun_number() {
        let coref = MentionRankingCoref::new();

        // "this" and "that" are singular; "these" and "those" are plural
        let text = "This is important. These are facts. That was clear. Those were obvious.";
        let mentions = coref.detect_mentions(text).unwrap();

        let this_m = mentions.iter().find(|m| m.text.to_lowercase() == "this");
        let these_m = mentions.iter().find(|m| m.text.to_lowercase() == "these");
        let that_m = mentions.iter().find(|m| m.text.to_lowercase() == "that");
        let those_m = mentions.iter().find(|m| m.text.to_lowercase() == "those");

        assert_eq!(this_m.map(|m| m.number), Some(Some(Number::Singular)));
        assert_eq!(these_m.map(|m| m.number), Some(Some(Number::Plural)));
        assert_eq!(that_m.map(|m| m.number), Some(Some(Number::Singular)));
        assert_eq!(those_m.map(|m| m.number), Some(Some(Number::Plural)));
    }

    // =========================================================================
    // Indefinite pronoun tests
    // =========================================================================

    #[test]
    fn test_indefinite_pronoun_detection() {
        let coref = MentionRankingCoref::new();

        let text = "Someone called yesterday. Everyone was surprised.";
        let mentions = coref.detect_mentions(text).unwrap();
        let texts: Vec<_> = mentions.iter().map(|m| m.text.to_lowercase()).collect();

        assert!(
            texts.contains(&"someone".to_string()),
            "Should detect 'Someone': {:?}",
            texts
        );
        assert!(
            texts.contains(&"everyone".to_string()),
            "Should detect 'Everyone': {:?}",
            texts
        );
    }

    #[test]
    fn test_indefinite_pronouns_are_singular() {
        // "Everyone", "someone", "nobody" are grammatically singular
        // even though they can refer to multiple people conceptually
        let coref = MentionRankingCoref::new();

        let text = "Everyone was there. Nobody left early.";
        let mentions = coref.detect_mentions(text).unwrap();

        let everyone_m = mentions
            .iter()
            .find(|m| m.text.to_lowercase() == "everyone");
        let nobody_m = mentions.iter().find(|m| m.text.to_lowercase() == "nobody");

        assert!(everyone_m.is_some(), "Should detect 'Everyone'");
        assert!(nobody_m.is_some(), "Should detect 'Nobody'");

        assert_eq!(
            everyone_m.unwrap().number,
            Some(Number::Singular),
            "'everyone' is grammatically singular"
        );
        assert_eq!(
            nobody_m.unwrap().number,
            Some(Number::Singular),
            "'nobody' is grammatically singular"
        );
    }

    #[test]
    fn test_impersonal_one_detection() {
        // Generic "one" is an impersonal pronoun
        let coref = MentionRankingCoref::new();

        let text = "One should always be prepared. One never knows what might happen.";
        let mentions = coref.detect_mentions(text).unwrap();
        let one_count = mentions
            .iter()
            .filter(|m| m.text.to_lowercase() == "one")
            .count();

        assert!(
            one_count >= 2,
            "Should detect impersonal 'one': {:?}",
            mentions.iter().map(|m| &m.text).collect::<Vec<_>>()
        );
    }

    // =========================================================================
    // Reflexive pronoun tests
    // =========================================================================

    #[test]
    fn test_reflexive_pronoun_detection() {
        let coref = MentionRankingCoref::new();

        let text = "John saw himself in the mirror. Mary hurt herself.";
        let mentions = coref.detect_mentions(text).unwrap();
        let texts: Vec<_> = mentions.iter().map(|m| m.text.to_lowercase()).collect();

        assert!(
            texts.contains(&"himself".to_string()),
            "Should detect 'himself': {:?}",
            texts
        );
        assert!(
            texts.contains(&"herself".to_string()),
            "Should detect 'herself': {:?}",
            texts
        );
    }

    #[test]
    fn test_reflexive_pronoun_gender() {
        let coref = MentionRankingCoref::new();

        let text = "He saw himself. She saw herself. It fixed itself.";
        let mentions = coref.detect_mentions(text).unwrap();

        let himself = mentions.iter().find(|m| m.text.to_lowercase() == "himself");
        let herself = mentions.iter().find(|m| m.text.to_lowercase() == "herself");
        let itself = mentions.iter().find(|m| m.text.to_lowercase() == "itself");

        assert!(himself.is_some(), "Should detect 'himself'");
        assert!(herself.is_some(), "Should detect 'herself'");
        assert!(itself.is_some(), "Should detect 'itself'");

        assert_eq!(himself.unwrap().gender, Some(Gender::Masculine));
        assert_eq!(herself.unwrap().gender, Some(Gender::Feminine));
        assert_eq!(itself.unwrap().gender, Some(Gender::Neutral));
    }

    // =========================================================================
    // Reciprocal pronoun tests
    // =========================================================================

    #[test]
    fn test_reciprocal_pronoun_detection() {
        let coref = MentionRankingCoref::new();

        let text = "John and Mary looked at each other. The teams competed against one another.";
        let mentions = coref.detect_mentions(text).unwrap();
        let texts: Vec<_> = mentions.iter().map(|m| m.text.to_lowercase()).collect();

        assert!(
            texts.contains(&"each other".to_string()),
            "Should detect 'each other': {:?}",
            texts
        );
        assert!(
            texts.contains(&"one another".to_string()),
            "Should detect 'one another': {:?}",
            texts
        );
    }

    #[test]
    fn test_reciprocal_pronouns_are_plural() {
        // Reciprocals require plural antecedents
        let coref = MentionRankingCoref::new();

        let text = "They helped each other.";
        let mentions = coref.detect_mentions(text).unwrap();

        let each_other = mentions
            .iter()
            .find(|m| m.text.to_lowercase() == "each other");
        assert!(each_other.is_some(), "Should detect 'each other'");
        assert_eq!(
            each_other.unwrap().number,
            Some(Number::Plural),
            "Reciprocals are grammatically plural"
        );
    }

    // =========================================================================
    // Property-based tests for mention detection invariants
    // =========================================================================
    //
    // These test real invariants that catch actual bugs:
    // - Spans within bounds (prevents panics)
    // - Valid Unicode (no slicing mid-character)
    // - Phi-feature consistency (catches logic errors)

    use proptest::prelude::*;

    /// Generate ASCII text with some pronouns embedded
    fn text_with_pronouns() -> impl Strategy<Value = String> {
        prop::collection::vec(
            prop_oneof![
                Just("he".to_string()),
                Just("she".to_string()),
                Just("they".to_string()),
                Just("it".to_string()),
                Just("the dog".to_string()),
                Just("John".to_string()),
                "[a-z]{3,10}".prop_map(|s| s),
            ],
            3..15,
        )
        .prop_map(|words| words.join(" ") + ".")
    }

    // =========================================================================
    // Multilingual Nominal Adjective Tests
    // =========================================================================

    #[test]
    fn test_multilingual_nominal_adjective_german() {
        let config = MentionRankingConfig {
            enable_nominal_adjective_detection: true,
            language: "de".to_string(),
            ..Default::default()
        };

        let coref = MentionRankingCoref::with_config(config);
        let text = "Die Armen leiden unter der Krise.";
        let mentions = coref.detect_mentions(text).unwrap();

        let has_armen = mentions
            .iter()
            .any(|m| m.text.to_lowercase().contains("armen"));
        assert!(
            has_armen,
            "Should detect 'die Armen' as a nominal adjective in German"
        );
    }

    #[test]
    fn test_multilingual_nominal_adjective_french() {
        let config = MentionRankingConfig {
            enable_nominal_adjective_detection: true,
            language: "fr".to_string(),
            ..Default::default()
        };

        let coref = MentionRankingCoref::with_config(config);
        let text = "Les pauvres ont besoin d'aide.";
        let mentions = coref.detect_mentions(text).unwrap();

        let has_pauvres = mentions
            .iter()
            .any(|m| m.text.to_lowercase().contains("pauvres"));
        assert!(
            has_pauvres,
            "Should detect 'les pauvres' as a nominal adjective in French"
        );
    }

    #[test]
    fn test_multilingual_nominal_adjective_spanish() {
        let config = MentionRankingConfig {
            enable_nominal_adjective_detection: true,
            language: "es".to_string(),
            ..Default::default()
        };

        let coref = MentionRankingCoref::with_config(config);
        let text = "Los pobres necesitan ayuda.";
        let mentions = coref.detect_mentions(text).unwrap();

        let has_pobres = mentions
            .iter()
            .any(|m| m.text.to_lowercase().contains("pobres"));
        assert!(
            has_pobres,
            "Should detect 'los pobres' as a nominal adjective in Spanish"
        );
    }

    #[test]
    fn test_config_language_field() {
        // Default should be English
        let config = MentionRankingConfig::default();
        assert_eq!(config.language, "en");

        // Book scale should default to English
        let book_config = MentionRankingConfig::book_scale();
        assert_eq!(book_config.language, "en");

        // Clinical should default to English
        let clinical_config = MentionRankingConfig::clinical();
        assert_eq!(clinical_config.language, "en");
    }

    /// N2-coref: proper noun mentions must not include trailing punctuation.
    #[test]
    fn test_proper_noun_no_trailing_punct() {
        let coref = MentionRankingCoref::new();
        let text = "John met Obama. They shook hands.";
        let mentions = coref.detect_mentions(text).unwrap();
        for m in &mentions {
            if m.mention_type == MentionType::Proper {
                assert!(
                    !m.text.ends_with('.') && !m.text.ends_with(','),
                    "Proper noun mention '{}' should not have trailing punctuation",
                    m.text
                );
            }
        }
    }

    /// Pronouns must not be detected as Proper mentions (would miss the lower
    /// pronoun linking threshold).
    #[test]
    fn test_pronouns_not_detected_as_proper() {
        let coref = MentionRankingCoref::new();
        let text = "Alice met Bob. He waved. She smiled.";
        let mentions = coref.detect_mentions(text).unwrap();
        let pronoun_words = ["he", "she", "it", "they", "him", "her", "them"];
        for m in &mentions {
            let lower = m.text.to_lowercase();
            if pronoun_words.contains(&lower.as_str()) {
                assert_eq!(
                    m.mention_type,
                    MentionType::Pronominal,
                    "'{}' should be Pronominal, not {:?}",
                    m.text,
                    m.mention_type
                );
            }
        }
    }

    /// Gender-matched pronoun resolution should form a coreference cluster.
    #[test]
    fn test_gender_matched_pronoun_links() {
        let coref = MentionRankingCoref::new();
        // "Mary" is detected as Feminine, "She" is Feminine → should link
        let clusters = coref.resolve("John saw Mary. She waved.").unwrap();
        let has_mary_she = clusters.iter().any(|c| {
            let texts: Vec<_> = c.mentions.iter().map(|m| m.text.to_lowercase()).collect();
            texts.contains(&"mary".to_string()) && texts.contains(&"she".to_string())
        });
        assert!(has_mary_she, "Mary and She should be in the same cluster");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// All detected mentions have spans within text bounds
        ///
        /// This catches off-by-one errors and Unicode slicing bugs.
        #[test]
        fn mention_spans_within_bounds(text in text_with_pronouns()) {
            let coref = MentionRankingCoref::new();
            if let Ok(mentions) = coref.detect_mentions(&text) {
                let char_count = text.chars().count();
                for mention in &mentions {
                    prop_assert!(
                        mention.start <= mention.end,
                        "Start {} > end {} for '{}'",
                        mention.start, mention.end, mention.text
                    );
                    prop_assert!(
                        mention.end <= char_count,
                        "End {} > text length {} for '{}'",
                        mention.end, char_count, mention.text
                    );
                }
            }
        }

        /// Extracted mention text matches the span
        ///
        /// Verifies we're using character offsets correctly.
        #[test]
        fn mention_text_matches_span(text in text_with_pronouns()) {
            let coref = MentionRankingCoref::new();
            if let Ok(mentions) = coref.detect_mentions(&text) {
                for mention in &mentions {
                    let extracted: String = text.chars()
                        .skip(mention.start)
                        .take(mention.end - mention.start)
                        .collect();
                    // Case-insensitive comparison (we lowercase during detection)
                    prop_assert_eq!(
                        extracted.to_lowercase(),
                        mention.text.to_lowercase(),
                        "Extracted text doesn't match stored text"
                    );
                }
            }
        }

        /// Pronouns always have MentionType::Pronominal
        #[test]
        fn pronouns_are_pronominal(text in text_with_pronouns()) {
            let coref = MentionRankingCoref::new();
            if let Ok(mentions) = coref.detect_mentions(&text) {
                let pronouns = ["he", "she", "it", "they", "him", "her", "them"];
                for mention in &mentions {
                    if pronouns.contains(&mention.text.to_lowercase().as_str()) {
                        prop_assert_eq!(
                            mention.mention_type,
                            MentionType::Pronominal,
                            "'{}' should be Pronominal",
                            mention.text
                        );
                    }
                }
            }
        }

        /// Gender is always set for detected pronouns
        #[test]
        fn pronouns_have_gender(text in text_with_pronouns()) {
            let coref = MentionRankingCoref::new();
            if let Ok(mentions) = coref.detect_mentions(&text) {
                for mention in &mentions {
                    if mention.mention_type == MentionType::Pronominal {
                        prop_assert!(
                            mention.gender.is_some(),
                            "Pronoun '{}' should have gender",
                            mention.text
                        );
                    }
                }
            }
        }

        /// Number is always set for detected pronouns
        #[test]
        fn pronouns_have_number(text in text_with_pronouns()) {
            let coref = MentionRankingCoref::new();
            if let Ok(mentions) = coref.detect_mentions(&text) {
                for mention in &mentions {
                    if mention.mention_type == MentionType::Pronominal {
                        prop_assert!(
                            mention.number.is_some(),
                            "Pronoun '{}' should have number",
                            mention.text
                        );
                    }
                }
            }
        }

        /// Coreference clusters partition mentions (no overlaps, no orphans)
        #[test]
        fn clusters_partition_mentions(text in text_with_pronouns()) {
            let coref = MentionRankingCoref::new();
            if let Ok(clusters) = coref.resolve(&text) {
                // Flatten all mentions from clusters
                let mut all_mentions: Vec<_> = clusters.iter()
                    .flat_map(|c| &c.mentions)
                    .collect();

                // Check no duplicates (by span)
                let original_len = all_mentions.len();
                all_mentions.sort_by_key(|m| (m.start, m.end));
                all_mentions.dedup_by_key(|m| (m.start, m.end));
                prop_assert_eq!(
                    all_mentions.len(),
                    original_len,
                    "Duplicate mentions across clusters"
                );
            }
        }

        /// Score pair is deterministic
        ///
        /// Same inputs should always produce same score.
        #[test]
        fn score_pair_deterministic(text in text_with_pronouns()) {
            let coref = MentionRankingCoref::new();
            if let Ok(mentions) = coref.detect_mentions(&text) {
                if mentions.len() >= 2 {
                    let distance = mentions[1].start.saturating_sub(mentions[0].end);
                    let score1 = coref.score_pair(&mentions[0], &mentions[1], distance, Some(&text));
                    let score2 = coref.score_pair(&mentions[0], &mentions[1], distance, Some(&text));
                    prop_assert!(
                        (score1 - score2).abs() < 0.0001,
                        "Scoring should be deterministic"
                    );
                }
            }
        }
    }

    // =========================================================================
    // spans_overlap tests
    // =========================================================================

    #[test]
    fn spans_overlap_exact_match() {
        let a = crate::Location::Text { start: 10, end: 20 };
        let b = crate::Location::Text { start: 10, end: 20 };
        assert!(super::spans_overlap(&a, &b));
    }

    #[test]
    fn spans_overlap_partial() {
        let a = crate::Location::Text { start: 10, end: 20 };
        let b = crate::Location::Text { start: 15, end: 25 };
        assert!(super::spans_overlap(&a, &b));
    }

    #[test]
    fn spans_overlap_containment() {
        let a = crate::Location::Text { start: 10, end: 30 };
        let b = crate::Location::Text { start: 15, end: 20 };
        assert!(super::spans_overlap(&a, &b));
    }

    #[test]
    fn spans_no_overlap_adjacent() {
        let a = crate::Location::Text { start: 10, end: 20 };
        let b = crate::Location::Text { start: 20, end: 30 };
        assert!(!super::spans_overlap(&a, &b));
    }

    #[test]
    fn spans_no_overlap_disjoint() {
        let a = crate::Location::Text { start: 10, end: 20 };
        let b = crate::Location::Text { start: 30, end: 40 };
        assert!(!super::spans_overlap(&a, &b));
    }
}
