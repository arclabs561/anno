//! Feature extraction for mention-ranking coreference.
//!
//! Contains type-compatibility checks, animacy inference, pronoun-pattern tables,
//! and all other feature-extraction helpers used during mention detection and
//! pair scoring.

use super::*;

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
pub(super) fn is_type_incompatible(mention_a: &str, mention_b: &str) -> bool {
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
pub(super) fn animacy_from_pronoun(text_lower: &str) -> Animacy {
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
pub(super) fn animacy_from_entity_type(entity_type: &crate::EntityType) -> Animacy {
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

impl MentionRankingCoref {
    /// Get language-specific pronoun patterns.
    ///
    /// Returns (pronoun_text, gender, number) tuples for the specified language.
    /// Falls back to English if language is not supported.
    pub(super) fn get_pronoun_patterns(&self) -> Vec<(&'static str, Gender, Number)> {
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

    /// Extract additional features for a mention.
    pub(super) fn extract_features(&self, mention: &mut RankedMention) {
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
    pub(super) fn guess_gender(&self, text: &str) -> Option<Gender> {
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
    pub(super) fn get_head(&self, text: &str) -> String {
        // Simple heuristic: last word is head
        text.split_whitespace().last().unwrap_or(text).to_string()
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
    pub(super) fn is_be_phrase_link(
        &self,
        text: &str,
        m1: &RankedMention,
        m2: &RankedMention,
    ) -> bool {
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
    pub(super) fn is_acronym_match(&self, m1: &RankedMention, m2: &RankedMention) -> bool {
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
    pub(super) fn is_pleonastic_it(&self, text_lower: &str, it_byte_pos: usize) -> bool {
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
    pub(super) fn should_filter_by_context(
        &self,
        text: &str,
        m1: &RankedMention,
        m2: &RankedMention,
    ) -> bool {
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
    pub(super) fn extract_date(context: &str) -> Option<String> {
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
    pub(super) fn has_negation_context(context: &str) -> bool {
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
    pub(super) fn are_synonyms(&self, m1: &RankedMention, m2: &RankedMention) -> bool {
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
}
