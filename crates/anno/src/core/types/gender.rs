//! Gender type for coreference resolution and demographic analysis.
//!
//! This is the canonical `Gender` enum used throughout anno for:
//! - Pronoun resolution in coreference
//! - Literary character analysis
//! - Demographic bias evaluation
//!
//! # Design Notes
//!
//! The enum is intentionally simple with four variants to cover most NLP use cases.
//! For more nuanced gender representation (e.g., social vs grammatical gender),
//! consider domain-specific extensions.
//!
//! # Cross-Linguistic Perspective
//!
//! English's masc/fem/neut system is actually unusual. Other systems include:
//!
//! | Language Family | System | Example |
//! |----------------|--------|---------|
//! | Bantu | 10-20 noun classes | Human, animal, plant, tool, abstract... |
//! | Dyirbal | 4 semantic classes | Men; women/fire/danger; edible plants; other |
//! | Algonquian | Animate/inanimate | Rocks can be "animate" if culturally significant |
//! | Navajo | Shape classifiers | Long-slender, flat, round, granular... |
//!
//! For these languages, our `Gender` enum is insufficient. Future work may need
//! a more general `NounClass` system.
//!
//! # Grammatical vs Social Gender
//!
//! This enum conflates grammatical gender (agreement class) with social gender
//! (identity). In many languages these differ:
//! - German "das Mädchen" (girl) is grammatically neuter
//! - Arabic has grammatical gender for all nouns, not just animate ones
//! - Some languages have no grammatical gender but complex honorific systems

use serde::{Deserialize, Serialize};

/// Gender classification for NLP tasks.
///
/// Used for pronoun resolution, coreference clustering, and bias analysis.
///
/// # Examples
///
/// ```rust
/// use crate::Gender;
///
/// let gender = Gender::from_pronoun("she");
/// assert_eq!(gender, Some(Gender::Feminine));
///
/// let gender = Gender::from_pronoun("they");
/// assert_eq!(gender, Some(Gender::Neutral));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Gender {
    /// Masculine (he/him/his)
    Masculine,
    /// Feminine (she/her/hers)
    Feminine,
    /// Neutral or non-binary (they/them/theirs, it/its).
    ///
    /// Encompasses both grammatical neutrality (singular "they"/"it" as convenience
    /// or for inanimate referents) and personal pronoun choice (they/it as identity,
    /// used by some genderqueer individuals).
    Neutral,
    /// Unknown or unspecified (default).
    ///
    /// Absence of information about gender. Compatible with all other genders
    /// (acts as a wildcard in coreference). Distinct from Neutral, which is a
    /// positive assertion ("uses they/it").
    #[default]
    Unknown,
}

impl Gender {
    /// Infer gender from a pronoun.
    ///
    /// Returns `None` for non-pronoun inputs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::Gender;
    ///
    /// assert_eq!(Gender::from_pronoun("he"), Some(Gender::Masculine));
    /// assert_eq!(Gender::from_pronoun("She"), Some(Gender::Feminine));
    /// assert_eq!(Gender::from_pronoun("they"), Some(Gender::Neutral));
    /// assert_eq!(Gender::from_pronoun("it"), Some(Gender::Neutral));
    /// assert_eq!(Gender::from_pronoun("John"), None);
    /// ```
    #[must_use]
    pub fn from_pronoun(pronoun: &str) -> Option<Self> {
        match pronoun.to_lowercase().as_str() {
            // Traditional gendered pronouns
            "he" | "him" | "his" | "himself" => Some(Gender::Masculine),
            "she" | "her" | "hers" | "herself" => Some(Gender::Feminine),
            // Singular they / plural they (Neutral)
            "they" | "them" | "their" | "theirs" | "themself" | "themselves" => {
                Some(Gender::Neutral)
            }
            // Neutral/inanimate (also used as personal pronouns by some genderqueer individuals)
            "it" | "its" | "itself" => Some(Gender::Neutral),
            // Neopronouns (Unknown - explicitly non-binary, distinct from neutral "they")
            // xe/xem set
            "xe" | "xem" | "xyr" | "xyrs" | "xemself" => Some(Gender::Unknown),
            // ze/hir/zir set
            "ze" | "hir" | "zir" | "hirs" | "zirs" | "hirself" | "zirself" => Some(Gender::Unknown),
            // ey/em set (Spivak pronouns)
            "ey" | "em" | "eir" | "eirs" | "emself" => Some(Gender::Unknown),
            // fae/faer set
            "fae" | "faer" | "faers" | "faeself" | "faerself" => Some(Gender::Unknown),
            _ => None,
        }
    }

    /// Get typical subject pronoun for this gender (English only).
    ///
    /// Returns English pronouns unconditionally. For other languages, use
    /// language-specific pronoun generation (e.g., French il/elle, German er/sie/es).
    #[must_use]
    pub fn subject_pronoun(&self) -> &'static str {
        match self {
            Gender::Masculine => "he",
            Gender::Feminine => "she",
            Gender::Neutral | Gender::Unknown => "they",
        }
    }

    /// Get typical object pronoun for this gender (English only).
    ///
    /// Returns English pronouns unconditionally. See [`subject_pronoun`](Self::subject_pronoun).
    #[must_use]
    pub fn object_pronoun(&self) -> &'static str {
        match self {
            Gender::Masculine => "him",
            Gender::Feminine => "her",
            Gender::Neutral | Gender::Unknown => "them",
        }
    }

    /// Get typical possessive pronoun for this gender (English only).
    ///
    /// Returns English pronouns unconditionally. See [`subject_pronoun`](Self::subject_pronoun).
    #[must_use]
    pub fn possessive_pronoun(&self) -> &'static str {
        match self {
            Gender::Masculine => "his",
            Gender::Feminine => "her",
            Gender::Neutral | Gender::Unknown => "their",
        }
    }

    /// Check if this gender is compatible with another for coreference.
    ///
    /// Unknown is compatible with everything. Neutral is compatible with
    /// everything (singular they can refer to any gender).
    #[must_use]
    pub fn is_compatible(&self, other: &Gender) -> bool {
        match (self, other) {
            // Unknown matches anything
            (Gender::Unknown, _) | (_, Gender::Unknown) => true,
            // Neutral (they) can refer to any gender
            (Gender::Neutral, _) | (_, Gender::Neutral) => true,
            // Same gender matches
            (a, b) => a == b,
        }
    }
}

impl std::fmt::Display for Gender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Gender::Masculine => write!(f, "masculine"),
            Gender::Feminine => write!(f, "feminine"),
            Gender::Neutral => write!(f, "neutral"),
            Gender::Unknown => write!(f, "unknown"),
        }
    }
}

impl std::str::FromStr for Gender {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "masculine" | "male" | "m" => Ok(Gender::Masculine),
            "feminine" | "female" | "f" => Ok(Gender::Feminine),
            "neutral" | "nonbinary" | "non-binary" | "nb" | "n" => Ok(Gender::Neutral),
            "unknown" | "?" | "" => Ok(Gender::Unknown),
            _ => Err(format!("Unknown gender: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_pronoun() {
        assert_eq!(Gender::from_pronoun("he"), Some(Gender::Masculine));
        assert_eq!(Gender::from_pronoun("She"), Some(Gender::Feminine));
        assert_eq!(Gender::from_pronoun("THEY"), Some(Gender::Neutral));
        assert_eq!(Gender::from_pronoun("it"), Some(Gender::Neutral));
        assert_eq!(Gender::from_pronoun("John"), None);
    }

    #[test]
    fn test_compatibility() {
        assert!(Gender::Masculine.is_compatible(&Gender::Masculine));
        assert!(!Gender::Masculine.is_compatible(&Gender::Feminine));
        assert!(Gender::Unknown.is_compatible(&Gender::Masculine));
        assert!(Gender::Neutral.is_compatible(&Gender::Feminine));
    }

    #[test]
    fn test_serde_roundtrip() {
        let gender = Gender::Feminine;
        let json = serde_json::to_string(&gender).expect("serialize Gender");
        assert_eq!(json, "\"feminine\"");
        let recovered: Gender = serde_json::from_str(&json).expect("deserialize Gender");
        assert_eq!(gender, recovered);
    }

    #[test]
    fn test_from_str() {
        assert_eq!(
            "masculine".parse::<Gender>().expect("parse 'masculine'"),
            Gender::Masculine
        );
        assert_eq!(
            "female".parse::<Gender>().expect("parse 'female'"),
            Gender::Feminine
        );
        assert_eq!("nb".parse::<Gender>().expect("parse 'nb'"), Gender::Neutral);
        assert_eq!(
            "".parse::<Gender>().expect("parse empty gender"),
            Gender::Unknown
        );
    }

    #[test]
    fn test_pronouns() {
        assert_eq!(Gender::Masculine.subject_pronoun(), "he");
        assert_eq!(Gender::Masculine.object_pronoun(), "him");
        assert_eq!(Gender::Masculine.possessive_pronoun(), "his");

        assert_eq!(Gender::Feminine.subject_pronoun(), "she");
        assert_eq!(Gender::Feminine.object_pronoun(), "her");
        assert_eq!(Gender::Feminine.possessive_pronoun(), "her");

        assert_eq!(Gender::Neutral.subject_pronoun(), "they");
        assert_eq!(Gender::Neutral.object_pronoun(), "them");
        assert_eq!(Gender::Neutral.possessive_pronoun(), "their");
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Gender::Masculine), "masculine");
        assert_eq!(format!("{}", Gender::Feminine), "feminine");
        assert_eq!(format!("{}", Gender::Neutral), "neutral");
        assert_eq!(format!("{}", Gender::Unknown), "unknown");
    }

    #[test]
    fn test_all_pronoun_forms() {
        // Masculine
        for pronoun in &["he", "him", "his", "himself"] {
            assert_eq!(Gender::from_pronoun(pronoun), Some(Gender::Masculine));
        }
        // Feminine
        for pronoun in &["she", "her", "hers", "herself"] {
            assert_eq!(Gender::from_pronoun(pronoun), Some(Gender::Feminine));
        }
        // Neutral
        for pronoun in &[
            "they",
            "them",
            "their",
            "theirs",
            "themself",
            "themselves",
            "it",
            "its",
            "itself",
        ] {
            assert_eq!(Gender::from_pronoun(pronoun), Some(Gender::Neutral));
        }
    }

    #[test]
    fn test_neopronoun_detection() {
        // Neopronouns should return Gender::Unknown (explicitly non-binary)
        // xe/xem set
        for pronoun in &["xe", "xem", "xyr", "xyrs", "xemself"] {
            assert_eq!(
                Gender::from_pronoun(pronoun),
                Some(Gender::Unknown),
                "xe/xem set: {}",
                pronoun
            );
        }
        // ze/hir set
        for pronoun in &["ze", "hir", "hirs", "hirself"] {
            assert_eq!(
                Gender::from_pronoun(pronoun),
                Some(Gender::Unknown),
                "ze/hir set: {}",
                pronoun
            );
        }
        // ey/em set (Spivak pronouns)
        for pronoun in &["ey", "em", "eir", "eirs", "emself"] {
            assert_eq!(
                Gender::from_pronoun(pronoun),
                Some(Gender::Unknown),
                "ey/em set: {}",
                pronoun
            );
        }
        // fae/faer set
        for pronoun in &["fae", "faer", "faers", "faerself"] {
            assert_eq!(
                Gender::from_pronoun(pronoun),
                Some(Gender::Unknown),
                "fae/faer set: {}",
                pronoun
            );
        }
    }

    // =========================================================================
    // Linguistic invariant tests - encoding theoretical constraints
    // =========================================================================

    /// Neutral gender (they/it) is compatible with all genders.
    ///
    /// This models English singular "they" which can refer to any gender:
    /// - "The doctor said they would call back" (gender unknown)
    /// - "Alex brought their laptop" (Alex could be any gender)
    ///
    /// Also models "it" for inanimate referents and as a personal pronoun
    /// (used by some genderqueer individuals, e.g., "Alex uses it/its pronouns").
    #[test]
    fn test_neutral_is_universally_compatible() {
        for gender in [
            Gender::Masculine,
            Gender::Feminine,
            Gender::Neutral,
            Gender::Unknown,
        ] {
            assert!(
                Gender::Neutral.is_compatible(&gender),
                "Neutral should be compatible with {:?}",
                gender
            );
            assert!(
                gender.is_compatible(&Gender::Neutral),
                "{:?} should be compatible with Neutral",
                gender
            );
        }
    }

    /// Masculine and Feminine are mutually exclusive.
    ///
    /// This is the core binary gender constraint in agreement systems.
    /// "John... she" and "Mary... he" are typically ungrammatical.
    #[test]
    fn test_binary_gender_exclusion() {
        assert!(!Gender::Masculine.is_compatible(&Gender::Feminine));
        assert!(!Gender::Feminine.is_compatible(&Gender::Masculine));
    }

    /// Unknown gender is compatible with everything.
    ///
    /// Used for:
    /// - Neopronouns (ze, xe, fae) that explicitly reject binary gender
    /// - Names without clear gender marking ("Alex", "Jordan", "Morgan")
    /// - Entities whose gender is not yet established in discourse
    #[test]
    fn test_unknown_is_maximally_permissive() {
        for gender in [
            Gender::Masculine,
            Gender::Feminine,
            Gender::Neutral,
            Gender::Unknown,
        ] {
            assert!(
                Gender::Unknown.is_compatible(&gender),
                "Unknown should be compatible with {:?}",
                gender
            );
        }
    }

    /// Pronoun paradigms should be complete (subject, object, possessive, reflexive).
    ///
    /// Each gender should provide all forms needed for English sentence generation.
    #[test]
    fn test_pronoun_paradigms_complete() {
        for gender in [
            Gender::Masculine,
            Gender::Feminine,
            Gender::Neutral,
            Gender::Unknown,
        ] {
            // All forms should be non-empty
            assert!(!gender.subject_pronoun().is_empty());
            assert!(!gender.object_pronoun().is_empty());
            assert!(!gender.possessive_pronoun().is_empty());
        }
    }

    /// Neopronouns should map to Unknown, not Neutral.
    ///
    /// This distinction matters because:
    /// - Neutral (they/it) is a grammatical category that can refer to any gender
    /// - Unknown marks explicitly non-binary identity (neopronouns)
    ///
    /// The difference is semantic, not just grammatical.
    #[test]
    fn test_neopronoun_vs_neutral_distinction() {
        // Traditional neutral pronouns → Neutral
        assert_eq!(Gender::from_pronoun("they"), Some(Gender::Neutral));
        assert_eq!(Gender::from_pronoun("it"), Some(Gender::Neutral));

        // Neopronouns → Unknown (explicitly non-binary identity)
        assert_eq!(Gender::from_pronoun("xe"), Some(Gender::Unknown));
        assert_eq!(Gender::from_pronoun("ze"), Some(Gender::Unknown));
        assert_eq!(Gender::from_pronoun("fae"), Some(Gender::Unknown));

        // This means: Neutral is a grammatical convenience,
        // Unknown is a semantic marker of gender identity
    }

    // =========================================================================
    // Audit-driven regression tests
    // =========================================================================

    /// Gender::default() must be Unknown, not Neutral or Masculine.
    ///
    /// Unknown means "no information yet"; Neutral means "uses they/it".
    /// Getting this wrong causes all ungendered entities to appear non-binary
    /// instead of simply unresolved.
    #[test]
    fn test_gender_default_is_unknown() {
        assert_eq!(Gender::default(), Gender::Unknown);
        // Confirm it is NOT Neutral (the audit's key concern)
        assert_ne!(Gender::default(), Gender::Neutral);
    }

    /// Neutral is a positive assertion: the referent uses they/it pronouns.
    ///
    /// This is distinct from Unknown (no gender info). Conflating them causes
    /// false coreference links between ungendered mentions and they/it-using
    /// individuals.
    #[test]
    fn test_neutral_is_positive_assertion() {
        // Neutral comes from explicit they/it pronouns
        assert_eq!(Gender::from_pronoun("they"), Some(Gender::Neutral));
        assert_eq!(Gender::from_pronoun("it"), Some(Gender::Neutral));

        // Neutral is NOT the default
        assert_ne!(Gender::default(), Gender::Neutral);

        // Neutral is compatible with everything (singular "they" can refer to anyone)
        assert!(Gender::Neutral.is_compatible(&Gender::Masculine));
        assert!(Gender::Neutral.is_compatible(&Gender::Feminine));
    }

    /// "her" appears as both object and possessive in English.
    /// Both forms should map to Feminine.
    #[test]
    fn test_from_pronoun_her_is_feminine() {
        assert_eq!(Gender::from_pronoun("her"), Some(Gender::Feminine));
        assert_eq!(Gender::from_pronoun("hers"), Some(Gender::Feminine));
        assert_eq!(Gender::from_pronoun("herself"), Some(Gender::Feminine));
    }

    /// from_pronoun must be case-insensitive for all recognized forms.
    #[test]
    fn test_from_pronoun_case_insensitive_all_forms() {
        let cases = [
            ("HE", Gender::Masculine),
            ("Him", Gender::Masculine),
            ("HIS", Gender::Masculine),
            ("HIMSELF", Gender::Masculine),
            ("SHE", Gender::Feminine),
            ("Her", Gender::Feminine),
            ("HERS", Gender::Feminine),
            ("HERSELF", Gender::Feminine),
            ("THEY", Gender::Neutral),
            ("Them", Gender::Neutral),
            ("THEIR", Gender::Neutral),
            ("THEIRS", Gender::Neutral),
            ("IT", Gender::Neutral),
            ("Its", Gender::Neutral),
            ("XE", Gender::Unknown),
            ("Ze", Gender::Unknown),
            ("FAE", Gender::Unknown),
            ("Ey", Gender::Unknown),
        ];
        for (pronoun, expected) in cases {
            assert_eq!(
                Gender::from_pronoun(pronoun),
                Some(expected),
                "from_pronoun({:?}) should be {:?}",
                pronoun,
                expected
            );
        }
    }

    /// Unknown = identity signal (no info), Neutral = grammatical convenience (they/it).
    ///
    /// Neopronouns map to Unknown because they explicitly reject the binary
    /// but don't map to the grammatical "they/it" category. This distinction
    /// prevents false conflation in coreference chains.
    #[test]
    fn test_neopronoun_unknown_not_neutral_semantic() {
        // Neopronouns -> Unknown (identity signal: explicitly non-binary)
        for neo in ["xe", "ze", "fae", "ey"] {
            assert_eq!(
                Gender::from_pronoun(neo),
                Some(Gender::Unknown),
                "{} should be Unknown (identity signal), not Neutral",
                neo
            );
        }
        // Traditional neutral -> Neutral (grammatical convenience)
        for trad in ["they", "them", "it", "its"] {
            assert_eq!(
                Gender::from_pronoun(trad),
                Some(Gender::Neutral),
                "{} should be Neutral (grammatical), not Unknown",
                trad
            );
        }
    }

    /// Compatibility is NOT transitive: Masc~Unknown and Unknown~Fem,
    /// but Masc is NOT compatible with Fem.
    ///
    /// This is intentional: Unknown is a wildcard, not a bridge.
    #[test]
    fn test_compatibility_not_transitive() {
        // Masc is compatible with Unknown
        assert!(Gender::Masculine.is_compatible(&Gender::Unknown));
        // Unknown is compatible with Fem
        assert!(Gender::Unknown.is_compatible(&Gender::Feminine));
        // But Masc is NOT compatible with Fem (transitivity does not hold)
        assert!(!Gender::Masculine.is_compatible(&Gender::Feminine));
    }

    /// from_pronoun returns None for names, not Some(Unknown).
    #[test]
    fn test_from_pronoun_returns_none_for_names() {
        assert_eq!(Gender::from_pronoun("Alice"), None);
        assert_eq!(Gender::from_pronoun("Bob"), None);
        assert_eq!(Gender::from_pronoun("NATO"), None);
        assert_eq!(Gender::from_pronoun("London"), None);
        assert_eq!(Gender::from_pronoun("the"), None);
    }

    /// "em" is a Spivak pronoun (ey/em/eir), not an HTML <em> tag.
    ///
    /// This test documents that from_pronoun expects tokenized text
    /// (individual words), not raw HTML. In tokenized text, "em" is
    /// the Spivak object pronoun.
    #[test]
    fn test_em_is_spivak_not_html() {
        assert_eq!(
            Gender::from_pronoun("em"),
            Some(Gender::Unknown),
            "'em' is the Spivak object pronoun; expects tokenized text, not HTML"
        );
    }
}
