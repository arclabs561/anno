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
/// ```rust,ignore
/// use anno_core::types::Gender;
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
    /// Neutral or non-binary (they/them/theirs, it/its)
    #[default]
    Neutral,
    /// Unknown or unspecified
    Unknown,
}

impl Gender {
    /// Infer gender from a pronoun.
    ///
    /// Returns `None` for non-pronoun inputs.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use anno_core::types::Gender;
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
            // Inanimate (Neutral)
            "it" | "its" | "itself" => Some(Gender::Neutral),
            // Neopronouns (Unknown - explicitly non-binary, distinct from neutral "they")
            // xe/xem set
            "xe" | "xem" | "xyr" | "xyrs" | "xemself" => Some(Gender::Unknown),
            // ze/hir set
            "ze" | "hir" | "hirs" | "hirself" => Some(Gender::Unknown),
            // ey/em set (Spivak pronouns)
            "ey" | "em" | "eir" | "eirs" | "emself" => Some(Gender::Unknown),
            // fae/faer set
            "fae" | "faer" | "faers" | "faerself" => Some(Gender::Unknown),
            _ => None,
        }
    }

    /// Get typical subject pronoun for this gender.
    #[must_use]
    pub fn subject_pronoun(&self) -> &'static str {
        match self {
            Gender::Masculine => "he",
            Gender::Feminine => "she",
            Gender::Neutral | Gender::Unknown => "they",
        }
    }

    /// Get typical object pronoun for this gender.
    #[must_use]
    pub fn object_pronoun(&self) -> &'static str {
        match self {
            Gender::Masculine => "him",
            Gender::Feminine => "her",
            Gender::Neutral | Gender::Unknown => "them",
        }
    }

    /// Get typical possessive pronoun for this gender.
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
    fn test_default() {
        assert_eq!(Gender::default(), Gender::Neutral);
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
    /// Also models "it" for inanimate referents that have grammatical gender
    /// in other languages (e.g., German "das Mädchen" = neuter "girl").
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
}
