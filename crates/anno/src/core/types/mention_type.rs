//! Mention type classification for coreference resolution.
//!
//! This is the canonical `MentionType` enum used throughout anno for:
//! - Coreference resolution mention classification
//! - Mention ranking algorithms
//! - Linguistic analysis of referring expressions
//!
//! # Linguistic Background
//!
//! The classification follows the **Accessibility Hierarchy** (Ariel 1990):
//!
//! ```text
//! Full names > Descriptions > Pronouns > Zero
//!   (low accessibility)    →    (high accessibility)
//! ```
//!
//! More accessible antecedents (recent, topical, salient) can use reduced forms
//! (pronouns, zeros). Less accessible antecedents need fuller descriptions.
//!
//! # Cross-Linguistic Variation
//!
//! | Language | Zero Pronouns | Nominal Articles | Honorific Pronouns |
//! |----------|---------------|------------------|-------------------|
//! | English | Rare | the/a | No |
//! | Spanish | Very common | el/la/un/una | usted (formal) |
//! | Japanese | Very common | None | Many levels |
//! | Arabic | Common (in verbs) | al- (definite only) | No |
//! | Chinese | Common | None | Some |
//!
//! # Binding Theory Implications
//!
//! - **Proper/Nominal**: Can be antecedents; "R-expressions" must be free
//! - **Pronominal**: Must be free in local domain (Principle B)
//! - **Reflexive** (subset): Must be bound in local domain (Principle A)
//! - **Zero**: Anaphoric; requires antecedent in discourse

use serde::{Deserialize, Serialize};

/// Type of referring expression in coreference.
///
/// This classification is fundamental to coreference resolution:
/// - **Proper** nouns are typically antecedents (first mention)
/// - **Nominal** mentions provide descriptive information
/// - **Pronominal** mentions require resolution to an antecedent
///
/// # Examples
///
/// ```rust
/// use crate::MentionType;
///
/// let mention_type = MentionType::classify("John Smith");
/// assert_eq!(mention_type, MentionType::Proper);
///
/// let mention_type = MentionType::classify("the president");
/// assert_eq!(mention_type, MentionType::Nominal);
///
/// let mention_type = MentionType::classify("he");
/// assert_eq!(mention_type, MentionType::Pronominal);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MentionType {
    /// Proper name ("John Smith", "Microsoft", "New York")
    ///
    /// Typically the first or canonical mention of an entity.
    /// Usually capitalized in English.
    ///
    /// Also known as "Named" in NER terminology. Use [`MentionType::is_named()`]
    /// or the [`MentionType::NAMED`] constant for compatibility with code
    /// using that terminology.
    Proper,

    /// Common noun phrase ("the company", "a dog", "the president")
    ///
    /// Descriptive mentions that provide semantic information.
    /// May be definite ("the") or indefinite ("a/an").
    #[default]
    Nominal,

    /// Pronoun ("he", "she", "it", "they", "this", "that")
    ///
    /// Anaphoric expressions that must be resolved to an antecedent.
    /// Includes personal, demonstrative, and relative pronouns.
    Pronominal,

    /// Zero pronoun (pro-drop languages: Arabic, Spanish, Japanese, Korean, Chinese)
    ///
    /// A dropped argument with no surface realization. The subject or object
    /// is grammatically required but omitted in the text. Common in:
    /// - Arabic: verb conjugation encodes subject ("ذهب" = "\[he\] went")
    /// - Spanish: "Vino a casa" = "\[He/She\] came home"
    /// - Japanese: topic/subject frequently omitted
    /// - Chinese: arguments recoverable from context
    ///
    /// Zero mentions have an anchor position (where they "would be") but no text span.
    /// They carry phi-features (person, gender, number) from morphology or context.
    Zero,

    /// Unknown or unclassified mention type.
    Unknown,
}

/// Ordering for MentionType based on canonical selection priority.
///
/// Order: `Zero < Pronominal < Unknown < Nominal < Proper`
///
/// This ordering is useful for canonical mention selection: when choosing
/// a representative mention from a cluster, higher-ranked types are preferred.
/// Proper nouns are most informative, zeros are least.
impl PartialOrd for MentionType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MentionType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority().cmp(&other.priority())
    }
}

impl MentionType {
    /// Alias for [`MentionType::Proper`] using NER terminology.
    ///
    /// In NER, "named entity" is the standard term. In coreference literature,
    /// "proper noun/name" is preferred. Both refer to the same concept.
    pub const NAMED: MentionType = MentionType::Proper;

    /// Check if this is a named/proper mention.
    ///
    /// Returns `true` for [`MentionType::Proper`]. This is an alias using
    /// NER terminology for code that prefers "named" over "proper".
    #[must_use]
    pub fn is_named(&self) -> bool {
        matches!(self, MentionType::Proper)
    }

    /// Get the ordering priority for canonical selection.
    ///
    /// Higher values = preferred for canonical mention.
    /// Order: Zero(0) < Pronominal(1) < Unknown(2) < Nominal(3) < Proper(4)
    #[must_use]
    fn priority(&self) -> u8 {
        match self {
            MentionType::Zero => 0,
            MentionType::Pronominal => 1,
            MentionType::Unknown => 2,
            MentionType::Nominal => 3,
            MentionType::Proper => 4,
        }
    }

    /// Classify a mention string by its type.
    ///
    /// Uses heuristics based on:
    /// - Pronoun list matching
    /// - Capitalization patterns
    /// - Article presence
    ///
    /// # Multilingual Note
    ///
    /// English-only heuristics. The pronoun list, article prefixes ("the", "a", "an"),
    /// and capitalization-based proper noun detection assume English text. For other
    /// languages, prefer model-based mention type classification.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::MentionType;
    ///
    /// assert_eq!(MentionType::classify("Barack Obama"), MentionType::Proper);
    /// assert_eq!(MentionType::classify("the former president"), MentionType::Nominal);
    /// assert_eq!(MentionType::classify("he"), MentionType::Pronominal);
    /// ```
    #[must_use]
    pub fn classify(mention: &str) -> Self {
        let lower = mention.to_lowercase();
        let trimmed = mention.trim();

        // Check for pronouns first
        if Self::is_pronoun(&lower) {
            return MentionType::Pronominal;
        }

        // Check for articles (indicates nominal)
        if lower.starts_with("the ")
            || lower.starts_with("a ")
            || lower.starts_with("an ")
            || lower.starts_with("this ")
            || lower.starts_with("that ")
            || lower.starts_with("these ")
            || lower.starts_with("those ")
            || lower.starts_with("some ")
            || lower.starts_with("any ")
        {
            return MentionType::Nominal;
        }

        // Check for capitalization (indicates proper noun)
        // All-caps words (NATO, IBM, FBI) are proper nouns (acronyms),
        // not emphasis, as long as they aren't pronouns (already handled above).
        let words: Vec<&str> = trimmed.split_whitespace().collect();
        if !words.is_empty() {
            let first_char = words[0].chars().next();
            if let Some(c) = first_char {
                if c.is_uppercase() {
                    return MentionType::Proper;
                }
            }
        }

        // Default to nominal
        MentionType::Nominal
    }

    /// Check if a string is a pronoun.
    /// Check if a string is an English pronoun.
    ///
    /// # Multilingual Note
    ///
    /// Currently English-only. For multilingual pronoun detection:
    /// - Use morphological analyzers (e.g., spaCy, Stanza)
    /// - Pro-drop languages (Arabic, Spanish, Japanese) need zero pronoun detection
    /// - CJK languages have different pronoun inventories (e.g., 彼/她/它 in Chinese)
    ///
    /// For multilingual NER, prefer model-based mention detection over heuristics.
    fn is_pronoun(s: &str) -> bool {
        matches!(
            s.trim(),
            // Personal pronouns
            "i" | "me" | "my" | "mine" | "myself"
                | "you" | "your" | "yours" | "yourself" | "yourselves"
                | "he" | "him" | "his" | "himself"
                | "she" | "her" | "hers" | "herself"
                | "it" | "its" | "itself"
                | "we" | "us" | "our" | "ours" | "ourselves"
                | "they" | "them" | "their" | "theirs" | "themselves" | "themself"
                // Demonstrative pronouns
                | "this" | "these" | "those"
                // Relative/interrogative pronouns
                | "who" | "whom" | "whose" | "which" | "that" | "what"
                // Indefinite pronouns (subset)
                | "one" | "ones" | "someone" | "anyone" | "everyone" | "no one"
                | "somebody" | "anybody" | "everybody" | "nobody"
                | "something" | "anything" | "everything" | "nothing"
                // Neopronouns
                | "xe" | "xem" | "xyr" | "xyrs" | "xemself"
                | "ze" | "hir" | "zir" | "hirs" | "zirs" | "hirself" | "zirself"
                | "ey" | "em" | "eir" | "eirs" | "emself"
                | "fae" | "faer" | "faers" | "faeself" | "faerself"
        )
    }

    /// Get the typical salience weight for this mention type.
    ///
    /// Proper nouns are most salient, pronouns and zeros least (they depend on context).
    /// Used in mention ranking algorithms.
    #[must_use]
    pub fn salience_weight(&self) -> f64 {
        match self {
            MentionType::Proper => 1.0,
            MentionType::Nominal => 0.7,
            MentionType::Pronominal => 0.3,
            MentionType::Zero => 0.2, // Even less salient than overt pronouns
            MentionType::Unknown => 0.5,
        }
    }

    /// Check if this mention type typically introduces new entities.
    ///
    /// Proper nouns often introduce entities; pronouns and zeros almost never do.
    #[must_use]
    pub fn can_introduce_entity(&self) -> bool {
        match self {
            MentionType::Proper => true,
            MentionType::Nominal => true, // Indefinite nominals can introduce
            MentionType::Pronominal => false,
            MentionType::Zero => false, // Zero pronouns are always anaphoric
            MentionType::Unknown => true, // Conservative default
        }
    }

    /// Check if this mention type requires an antecedent.
    #[must_use]
    pub fn requires_antecedent(&self) -> bool {
        match self {
            MentionType::Proper => false,
            MentionType::Nominal => false, // Definite nominals often do, but not always
            MentionType::Pronominal => true,
            MentionType::Zero => true, // Zero pronouns always need resolution
            MentionType::Unknown => false,
        }
    }

    /// Check if this mention type has a surface form in the text.
    ///
    /// Returns `false` for zero pronouns (pro-drop).
    #[must_use]
    pub fn has_surface_form(&self) -> bool {
        !matches!(self, MentionType::Zero)
    }

    /// Check if this is a zero mention (pro-drop).
    #[must_use]
    pub fn is_zero(&self) -> bool {
        matches!(self, MentionType::Zero)
    }

    /// Get a string label for this mention type.
    ///
    /// Returns a lowercase string suitable for use as a classification label.
    #[must_use]
    pub fn as_label(&self) -> &'static str {
        match self {
            MentionType::Proper => "proper",
            MentionType::Nominal => "nominal",
            MentionType::Pronominal => "pronominal",
            MentionType::Zero => "zero",
            MentionType::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for MentionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MentionType::Proper => write!(f, "proper"),
            MentionType::Nominal => write!(f, "nominal"),
            MentionType::Pronominal => write!(f, "pronominal"),
            MentionType::Zero => write!(f, "zero"),
            MentionType::Unknown => write!(f, "unknown"),
        }
    }
}

impl std::str::FromStr for MentionType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "proper" | "nam" | "name" | "propernoun" => Ok(MentionType::Proper),
            "nominal" | "nom" | "common" | "commonnoun" => Ok(MentionType::Nominal),
            "pronominal" | "pro" | "pronoun" | "pron" => Ok(MentionType::Pronominal),
            "zero" | "zero_pronoun" | "zeropronoun" | "*pro*" | "dropped" => Ok(MentionType::Zero),
            "unknown" | "?" | "" => Ok(MentionType::Unknown),
            _ => Err(format!("Unknown mention type: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_proper() {
        assert_eq!(MentionType::classify("John Smith"), MentionType::Proper);
        assert_eq!(MentionType::classify("Microsoft"), MentionType::Proper);
        assert_eq!(MentionType::classify("New York City"), MentionType::Proper);
        assert_eq!(MentionType::classify("Dr. Jane Doe"), MentionType::Proper);
    }

    #[test]
    fn test_classify_nominal() {
        assert_eq!(MentionType::classify("the president"), MentionType::Nominal);
        assert_eq!(MentionType::classify("a company"), MentionType::Nominal);
        assert_eq!(MentionType::classify("this document"), MentionType::Nominal);
        assert_eq!(MentionType::classify("some people"), MentionType::Nominal);
    }

    #[test]
    fn test_classify_pronominal() {
        assert_eq!(MentionType::classify("he"), MentionType::Pronominal);
        assert_eq!(MentionType::classify("she"), MentionType::Pronominal);
        assert_eq!(MentionType::classify("they"), MentionType::Pronominal);
        assert_eq!(MentionType::classify("it"), MentionType::Pronominal);
        assert_eq!(MentionType::classify("this"), MentionType::Pronominal);
        // Interrogative pronouns
        assert_eq!(MentionType::classify("who"), MentionType::Pronominal);
        assert_eq!(MentionType::classify("what"), MentionType::Pronominal);
        assert_eq!(MentionType::classify("which"), MentionType::Pronominal);
    }

    #[test]
    fn test_salience() {
        assert!(MentionType::Proper.salience_weight() > MentionType::Nominal.salience_weight());
        assert!(MentionType::Nominal.salience_weight() > MentionType::Pronominal.salience_weight());
    }

    #[test]
    fn test_serde_roundtrip() {
        let mt = MentionType::Pronominal;
        let json = serde_json::to_string(&mt).expect("serialize MentionType");
        assert_eq!(json, "\"pronominal\"");
        let recovered: MentionType = serde_json::from_str(&json).expect("deserialize MentionType");
        assert_eq!(mt, recovered);
    }

    #[test]
    fn test_from_str() {
        assert_eq!(
            "proper".parse::<MentionType>().expect("parse 'proper'"),
            MentionType::Proper
        );
        assert_eq!(
            "nom".parse::<MentionType>().expect("parse 'nom'"),
            MentionType::Nominal
        );
        assert_eq!(
            "pronoun".parse::<MentionType>().expect("parse 'pronoun'"),
            MentionType::Pronominal
        );
        assert_eq!(
            "zero".parse::<MentionType>().expect("parse 'zero'"),
            MentionType::Zero
        );
        assert_eq!(
            "*pro*".parse::<MentionType>().expect("parse '*pro*'"),
            MentionType::Zero
        );
    }

    #[test]
    fn test_zero_properties() {
        let zero = MentionType::Zero;

        // Zero pronouns have no surface form
        assert!(!zero.has_surface_form());
        assert!(zero.is_zero());

        // They require antecedents
        assert!(zero.requires_antecedent());

        // They cannot introduce entities
        assert!(!zero.can_introduce_entity());

        // Low salience (less than overt pronouns)
        assert!(zero.salience_weight() < MentionType::Pronominal.salience_weight());
    }

    #[test]
    fn test_overt_mentions_have_surface_form() {
        assert!(MentionType::Proper.has_surface_form());
        assert!(MentionType::Nominal.has_surface_form());
        assert!(MentionType::Pronominal.has_surface_form());
        assert!(MentionType::Unknown.has_surface_form());

        // Only Zero has no surface form
        assert!(!MentionType::Zero.has_surface_form());
    }

    #[test]
    fn test_zero_serde_roundtrip() {
        let mt = MentionType::Zero;
        let json = serde_json::to_string(&mt).expect("serialize MentionType::Zero");
        assert_eq!(json, "\"zero\"");
        let recovered: MentionType =
            serde_json::from_str(&json).expect("deserialize MentionType::Zero");
        assert_eq!(mt, recovered);
    }

    // =========================================================================
    // Linguistic invariant tests - these encode theoretical constraints
    // =========================================================================

    /// The Accessibility Hierarchy (Ariel 1990): reduced forms require
    /// more accessible (salient/recent) antecedents.
    ///
    /// Proper > Nominal > Pronominal > Zero
    ///
    /// This ordering is used for canonical mention selection.
    #[test]
    fn test_accessibility_hierarchy_ordering() {
        // Proper names are most informative (lowest accessibility requirement)
        assert!(MentionType::Proper > MentionType::Nominal);
        assert!(MentionType::Nominal > MentionType::Pronominal);
        assert!(MentionType::Pronominal > MentionType::Zero);

        // Full chain
        assert!(MentionType::Proper > MentionType::Zero);
    }

    /// Anaphoric types require antecedents; referential types can introduce entities.
    ///
    /// This is a fundamental constraint in discourse:
    /// - "He arrived" (who?) - requires prior context
    /// - "John arrived" - introduces an entity
    #[test]
    fn test_anaphoricity_constraint() {
        // Anaphoric types: require antecedent, cannot introduce
        assert!(MentionType::Pronominal.requires_antecedent());
        assert!(MentionType::Zero.requires_antecedent());
        assert!(!MentionType::Pronominal.can_introduce_entity());
        assert!(!MentionType::Zero.can_introduce_entity());

        // Referential types: can introduce, don't require antecedent
        assert!(!MentionType::Proper.requires_antecedent());
        assert!(MentionType::Proper.can_introduce_entity());

        // Nominals are mixed - definite require antecedent, indefinite introduce
        // So the type doesn't strictly require antecedent
        assert!(!MentionType::Nominal.requires_antecedent());
        assert!(MentionType::Nominal.can_introduce_entity());
    }

    /// Salience weights should reflect accessibility: fuller forms = higher salience.
    ///
    /// When ranking mentions in a cluster for canonical selection,
    /// proper nouns should be preferred over pronouns.
    #[test]
    fn test_salience_reflects_accessibility() {
        let proper = MentionType::Proper.salience_weight();
        let nominal = MentionType::Nominal.salience_weight();
        let pronominal = MentionType::Pronominal.salience_weight();
        let zero = MentionType::Zero.salience_weight();

        assert!(proper > nominal, "Proper nouns more salient than nominals");
        assert!(nominal > pronominal, "Nominals more salient than pronouns");
        assert!(pronominal > zero, "Pronouns more salient than zeros");

        // All should be in valid range
        for w in [proper, nominal, pronominal, zero] {
            assert!(w > 0.0 && w <= 1.0, "Salience weight in (0, 1]");
        }
    }

    /// Zero pronouns are the only type without surface realization.
    ///
    /// This distinguishes pro-drop languages (Arabic, Spanish, Japanese)
    /// from non-pro-drop languages (English, French, German).
    #[test]
    fn test_zero_is_unique_surfaceless() {
        let all_types = [
            MentionType::Proper,
            MentionType::Nominal,
            MentionType::Pronominal,
            MentionType::Zero,
            MentionType::Unknown,
        ];

        let surfaceless: Vec<_> = all_types.iter().filter(|t| !t.has_surface_form()).collect();

        assert_eq!(surfaceless.len(), 1, "Only one surfaceless type");
        assert_eq!(*surfaceless[0], MentionType::Zero);
    }

    /// The default mention type should be Nominal (most common in text).
    #[test]
    fn test_default_is_nominal() {
        assert_eq!(MentionType::default(), MentionType::Nominal);
    }

    /// All-caps acronyms (NATO, IBM, FBI) should be classified as Proper,
    /// not fall through to Nominal.
    #[test]
    fn test_classify_nato_is_proper() {
        assert_eq!(MentionType::classify("NATO"), MentionType::Proper);
        assert_eq!(MentionType::classify("IBM"), MentionType::Proper);
        assert_eq!(MentionType::classify("FBI"), MentionType::Proper);
        assert_eq!(MentionType::classify("UN"), MentionType::Proper);
    }

    // =========================================================================
    // Audit-driven regression tests
    // =========================================================================

    /// "IBM" is an all-caps acronym and should be Proper.
    #[test]
    fn test_classify_ibm_is_proper() {
        assert_eq!(MentionType::classify("IBM"), MentionType::Proper);
    }

    /// "FBI" is an all-caps acronym and should be Proper.
    #[test]
    fn test_classify_fbi_is_proper() {
        assert_eq!(MentionType::classify("FBI"), MentionType::Proper);
    }

    /// "I" is a pronoun and should be Pronominal, not Proper.
    ///
    /// The pronoun check must run before the capitalization check,
    /// otherwise "I" looks like a capitalized word -> Proper.
    #[test]
    fn test_classify_i_is_pronominal() {
        assert_eq!(
            MentionType::classify("I"),
            MentionType::Pronominal,
            "'I' is a pronoun, not a proper noun"
        );
    }

    /// Empty string should produce a deterministic result (Nominal default).
    #[test]
    fn test_classify_empty_is_nominal() {
        // Empty string: no words, no pronoun match, no capitalization -> Nominal
        assert_eq!(MentionType::classify(""), MentionType::Nominal);
    }

    /// Whitespace-only string should produce a deterministic result.
    #[test]
    fn test_classify_whitespace_only() {
        // Whitespace-only: trimmed is empty, split_whitespace yields nothing -> Nominal
        assert_eq!(MentionType::classify("  "), MentionType::Nominal);
        assert_eq!(MentionType::classify("\t"), MentionType::Nominal);
    }

    /// "iPhone" starts lowercase but has internal caps. Since the first
    /// character is lowercase, classify falls through to Nominal.
    ///
    /// This documents current behavior; a smarter heuristic could detect
    /// camelCase brand names as Proper.
    #[test]
    fn test_classify_mixed_case_acronym() {
        // "iPhone" starts with lowercase 'i', so it doesn't match the
        // "first char is uppercase" check -> Nominal
        assert_eq!(
            MentionType::classify("iPhone"),
            MentionType::Nominal,
            "iPhone starts lowercase -> Nominal (current heuristic)"
        );
    }

    /// All standard English personal pronouns should be recognized by is_pronoun.
    #[test]
    fn test_is_pronoun_completeness() {
        let standard_pronouns = [
            // Personal: subject, object, possessive det, possessive pro, reflexive
            "i",
            "me",
            "my",
            "mine",
            "myself",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "we",
            "us",
            "our",
            "ours",
            "ourselves",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "themself",
        ];
        for pronoun in standard_pronouns {
            assert_eq!(
                MentionType::classify(pronoun),
                MentionType::Pronominal,
                "'{}' should be classified as Pronominal",
                pronoun
            );
        }
    }

    /// Neopronoun sets (xe, ze, fae) should all be recognized as pronouns.
    #[test]
    fn test_is_pronoun_neopronouns() {
        let neopronoun_sets = [
            // xe/xem set
            "xe", "xem", "xyr", "xyrs", "xemself", // ze/hir/zir set
            "ze", "hir", "zir", "hirs", "zirs", "hirself", "zirself",
            // ey/em set (Spivak)
            "ey", "em", "eir", "eirs", "emself", // fae/faer set
            "fae", "faer", "faers", "faeself", "faerself",
        ];
        for pronoun in neopronoun_sets {
            assert_eq!(
                MentionType::classify(pronoun),
                MentionType::Pronominal,
                "neopronoun '{}' should be classified as Pronominal",
                pronoun
            );
        }
    }
}
