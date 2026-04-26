//! Phi-features (φ-features) for morphological agreement.
//!
//! In linguistics, phi-features (from Greek φ) are the grammatical features
//! that govern syntactic agreement between words. They're the reason verbs
//! conjugate differently for "I run" vs "she runs" in English, and why
//! adjectives must match nouns in gender and number in Spanish.
//!
//! # The Three Phi-Features
//!
//! | Feature | Values | Example (English) |
//! |---------|--------|-------------------|
//! | **Person** | 1st, 2nd, 3rd | "I" vs "you" vs "she" |
//! | **Number** | singular, dual, plural | "cat" vs "cats" |
//! | **Gender** | masculine, feminine, neuter | "he" vs "she" vs "it" |
//!
//! # Why This Matters for NLP
//!
//! Phi-features are critical for:
//!
//! 1. **Zero pronoun resolution** - In pro-drop languages (Arabic, Spanish,
//!    Japanese, Korean, Chinese), subjects/objects are routinely dropped.
//!    Phi-features from verb morphology let us recover what was omitted.
//!
//! 2. **Coreference constraints** - "John... she" is unlikely to corefer
//!    because of gender mismatch. Phi-features formalize these constraints.
//!
//! 3. **Joint entity models** - Factor graphs can use phi-feature compatibility
//!    as soft constraints between mention pairs.
//!
//! # Arabic Example
//!
//! Arabic is a canonical pro-drop language where the verb encodes subject features:
//!
//! ```text
//! ذَهَبَ إِلَى الْبَيْتِ
//! dhahaba ʾilā al-bayti
//! [he]-went to the-house
//! "He went to the house"
//! ```
//!
//! The verb "ذَهَبَ" (dhahaba) encodes:
//! - Person: 3rd (not the speaker, not the listener)
//! - Number: Singular (one person)
//! - Gender: Masculine (specifically "he", not "she")
//!
//! This allows us to create a zero mention with known phi-features even though
//! no pronoun appears in the text.
//!
//! # Cross-Linguistic Variation
//!
//! | Language | Dual Number | Verb Agreement | Pro-drop |
//! |----------|-------------|----------------|----------|
//! | Arabic | Yes | Person+Number+Gender | Yes |
//! | Spanish | No | Person+Number | Yes |
//! | Japanese | No | None (verbs don't conjugate) | Yes |
//! | English | No | Number only (3rd sg) | No |
//! | Sanskrit | Yes | Person+Number+Gender | Yes |
//!
//! # References
//!
//! - Chomsky (1981): Lectures on Government and Binding
//! - Aloraini et al. (2025): "A Survey of Coreference and Zeros Resolution for Arabic"

use super::Gender;
use serde::{Deserialize, Serialize};

/// Animacy classification for coreference constraints.
///
/// Animacy distinguishes animate referents (people, animals) from inanimate
/// ones (objects, abstractions). This matters for coreference: "John... it"
/// is typically ungrammatical because animate entities resist inanimate pronouns.
///
/// # Cross-Linguistic Significance
///
/// Animacy plays a role in many languages:
///
/// | Language | Animacy Effect |
/// |----------|---------------|
/// | English | Pronoun choice: "he/she" (animate) vs "it" (inanimate) |
/// | Algonquian | Noun class: animate vs inanimate determines morphology |
/// | Japanese | Verb choice: "iru" (animate) vs "aru" (inanimate) |
/// | Slavic | Case system: animacy affects accusative in masculine nouns |
///
/// # References
///
/// - Silverstein (1976): Hierarchy of Features and Ergativity
/// - Comrie (1989): Language Universals and Linguistic Typology
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Animacy {
    /// Animate: people, animals, personified entities
    Animate,

    /// Inanimate: objects, abstractions, places
    Inanimate,

    /// Unknown or unspecified animacy (default)
    ///
    /// Used when animacy cannot be determined from context, or when the
    /// language/annotation scheme does not mark animacy.
    #[default]
    Unknown,
}

impl Animacy {
    /// Check if this animacy is compatible with another for coreference.
    ///
    /// Unknown is compatible with anything (permissive).
    #[must_use]
    pub fn is_compatible(&self, other: &Animacy) -> bool {
        match (self, other) {
            (a, b) if a == b => true,
            (Animacy::Unknown, _) | (_, Animacy::Unknown) => true,
            _ => false,
        }
    }
}

impl std::fmt::Display for Animacy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Animacy::Animate => write!(f, "animate"),
            Animacy::Inanimate => write!(f, "inanimate"),
            Animacy::Unknown => write!(f, "?anim"),
        }
    }
}

impl std::str::FromStr for Animacy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "animate" | "anim" | "a" => Ok(Animacy::Animate),
            "inanimate" | "inanim" | "i" => Ok(Animacy::Inanimate),
            "?" | "?anim" | "unknown" | "unk" => Ok(Animacy::Unknown),
            _ => Err(format!("Unknown animacy: {}", s)),
        }
    }
}

/// Grammatical person (1st, 2nd, 3rd).
///
/// Person distinguishes the speaker (1st), the listener (2nd), and everyone
/// else (3rd). This is one of the most universal grammatical categories,
/// appearing in nearly all human languages.
///
/// # Examples by Language
///
/// | Person | English | Arabic | Spanish |
/// |--------|---------|--------|---------|
/// | 1st sg | I | أنا (ʾanā) | yo |
/// | 2nd sg | you | أنتَ (ʾanta) | tú |
/// | 3rd sg | he/she | هو/هي (huwa/hiya) | él/ella |
///
/// # Coreference Implications
///
/// Person is a hard constraint in coreference: "I went to the store. She
/// bought milk" cannot have "I" and "She" coreferring (different persons).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Person {
    /// First person: the speaker(s)
    ///
    /// English: "I", "we", "me", "us", "my", "our"
    /// Arabic: أنا (ʾanā), نحن (naḥnu)
    First,

    /// Second person: the listener(s)
    ///
    /// English: "you", "your", "yours"
    /// Arabic: أنتَ/أنتِ (ʾanta/ʾanti), أنتم (ʾantum)
    Second,

    /// Third person: everyone else (default for zero pronouns)
    ///
    /// English: "he", "she", "it", "they", "him", "her", "them"
    /// Arabic: هو (huwa), هي (hiya), هم (hum)
    ///
    /// This is the default because most zero pronouns in narrative text
    /// refer to third-person entities.
    #[default]
    Third,

    /// Unknown or unspecified person
    ///
    /// Used when person cannot be determined from context. Compatible with
    /// all other person values (acts as a wildcard).
    Unknown,
}

impl Person {
    /// Check if this person is compatible with another for coreference.
    ///
    /// Unknown is compatible with everything (wildcard).
    #[must_use]
    pub fn is_compatible(&self, other: &Person) -> bool {
        match (self, other) {
            (Person::Unknown, _) | (_, Person::Unknown) => true,
            (a, b) => a == b,
        }
    }
}

impl std::fmt::Display for Person {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Person::First => write!(f, "1st"),
            Person::Second => write!(f, "2nd"),
            Person::Third => write!(f, "3rd"),
            Person::Unknown => write!(f, "?per"),
        }
    }
}

impl std::str::FromStr for Person {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "1" | "1st" | "first" => Ok(Person::First),
            "2" | "2nd" | "second" => Ok(Person::Second),
            "3" | "3rd" | "third" => Ok(Person::Third),
            "?" | "?per" | "unknown" | "unk" => Ok(Person::Unknown),
            _ => Err(format!("Unknown person: {}", s)),
        }
    }
}

/// Grammatical number (singular, dual, plural).
///
/// Number indicates how many entities are being referred to. While English
/// has a simple singular/plural distinction, many languages (including Arabic,
/// Hebrew, Sanskrit, and Ancient Greek) have a **dual** form for exactly two.
///
/// # The Dual Number
///
/// The dual is particularly important for Arabic NLP:
///
/// ```text
/// كِتَابٌ      (kitābun)    - one book (singular)
/// كِتَابَانِ   (kitābāni)   - two books (dual)
/// كُتُبٌ      (kutubun)    - three+ books (plural)
/// ```
///
/// For coreference, we treat dual as compatible with plural (a pair of
/// entities can be referred to with plural pronouns in many contexts).
///
/// # Cross-Linguistic Distribution
///
/// | Language | Has Dual? | Example |
/// |----------|-----------|---------|
/// | Arabic | Yes | كِتَابَانِ (two books) |
/// | Hebrew | Yes | יָדַיִם (two hands) |
/// | Sanskrit | Yes | द्वौ (two) |
/// | English | No* | "both" (lexical, not grammatical) |
/// | Japanese | No | 本 (hon) + counter |
///
/// *English lost the dual around 1000 CE; Old English had it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Number {
    /// Singular: exactly one entity
    ///
    /// The default for most zero pronouns in Arabic, where the dropped
    /// subject typically refers to a single previously-mentioned entity.
    #[default]
    Singular,

    /// Dual: exactly two entities
    ///
    /// Important for Arabic, Hebrew, Sanskrit, and other Semitic/Indo-European
    /// languages. In Arabic, verb conjugations and noun endings differ for
    /// dual vs plural.
    Dual,

    /// Plural: more than one (or more than two in dual languages)
    ///
    /// In languages with dual, plural specifically means "three or more".
    /// In languages without dual, plural means "two or more".
    Plural,

    /// Unknown or ambiguous number
    ///
    /// Used when number cannot be determined, e.g., "you" in English
    /// (ambiguous singular/plural), or "they" (singular they vs plural).
    Unknown,
}

impl Number {
    /// Check if this number is compatible with another for coreference.
    ///
    /// Unknown is compatible with anything (permissive).
    /// Dual is compatible with plural in some contexts.
    #[must_use]
    pub fn is_compatible(&self, other: &Number) -> bool {
        match (self, other) {
            (a, b) if a == b => true,
            // Unknown is compatible with anything
            (Number::Unknown, _) | (_, Number::Unknown) => true,
            // Dual can sometimes be treated as plural
            (Number::Dual, Number::Plural) | (Number::Plural, Number::Dual) => true,
            _ => false,
        }
    }
}

impl std::fmt::Display for Number {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Number::Singular => write!(f, "sg"),
            Number::Dual => write!(f, "du"),
            Number::Plural => write!(f, "pl"),
            Number::Unknown => write!(f, "?"),
        }
    }
}

impl std::str::FromStr for Number {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sg" | "singular" | "sing" | "1" => Ok(Number::Singular),
            "du" | "dual" | "2" => Ok(Number::Dual),
            "pl" | "plural" | "plur" | "3+" => Ok(Number::Plural),
            "?" | "unknown" | "unk" => Ok(Number::Unknown),
            _ => Err(format!("Unknown number: {}", s)),
        }
    }
}

/// A bundle of phi-features (person, number, gender) for morphological agreement.
///
/// This struct packages the three core phi-features into a single unit that can
/// be attached to mentions (especially zero mentions) and used for coreference
/// constraint checking.
///
/// # Use Cases
///
/// 1. **Zero pronoun representation**: When Arabic "ذهب" (dhahaba, "he went")
///    drops its subject, we create a zero mention with `PhiFeatures::third_sg_masc()`.
///
/// 2. **Coreference filtering**: Two mentions can only corefer if their phi-features
///    are compatible. "John... she" fails the gender check.
///
/// 3. **Factor graph constraints**: In joint entity models, phi-feature compatibility
///    can be a soft factor between mention pairs.
///
/// # Example: Arabic Zero Pronoun Resolution
///
/// ```rust
/// use anno::{PhiFeatures, Person, Number, Gender};
///
/// // Arabic verb "katabat" = "[she] wrote"
/// // The -at suffix encodes: 3rd person, singular, feminine
/// let verb_features = PhiFeatures::new(
///     Person::Third,
///     Number::Singular,
///     Gender::Feminine
/// );
///
/// // Candidate antecedent: "Maryam" (a feminine name)
/// let maryam_features = PhiFeatures::new(
///     Person::Third,
///     Number::Singular,
///     Gender::Feminine
/// );
///
/// // Candidate antecedent: "Ahmad" (a masculine name)
/// let ahmad_features = PhiFeatures::third_sg_masc();
///
/// // Maryam is compatible (same features), Ahmad is not (gender mismatch)
/// assert!(verb_features.is_compatible(&maryam_features));
/// assert!(!verb_features.is_compatible(&ahmad_features));
/// ```
///
/// # Parsing from Strings
///
/// Phi-features can be parsed from compact notation:
///
/// ```rust
/// use anno::PhiFeatures;
///
/// let phi = PhiFeatures::parse("3sgm").unwrap();  // 3rd singular masculine
/// let phi = PhiFeatures::parse("1plf").unwrap();  // 1st plural feminine
/// let phi = PhiFeatures::parse("2du").unwrap();   // 2nd dual (gender unspecified)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct PhiFeatures {
    /// Grammatical person (1st/2nd/3rd)
    ///
    /// Indicates the discourse role: speaker, listener, or other.
    pub person: Person,

    /// Grammatical number (singular/dual/plural)
    ///
    /// Indicates quantity. Languages like Arabic distinguish dual from plural.
    pub number: Number,

    /// Grammatical gender (masculine/feminine/neutral)
    ///
    /// In languages like Arabic, gender is grammatically assigned to all nouns,
    /// not just animate entities.
    pub gender: Gender,

    /// Animacy (animate/inanimate)
    ///
    /// Distinguishes people and animals from objects and abstractions.
    /// Defaults to Unknown when not specified.
    pub animacy: Animacy,
}

impl PhiFeatures {
    /// Create a new phi-features bundle with explicit values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno::{PhiFeatures, Person, Number, Gender};
    ///
    /// // Spanish "Vino" = "[He/She] came" - 3rd singular, gender unknown
    /// let phi = PhiFeatures::new(Person::Third, Number::Singular, Gender::Unknown);
    /// ```
    #[must_use]
    pub fn new(person: Person, number: Number, gender: Gender) -> Self {
        Self {
            person,
            number,
            gender,
            animacy: Animacy::Unknown,
        }
    }

    /// Create 3rd person singular masculine.
    ///
    /// This is the most common phi-feature combination for zero pronouns in
    /// Arabic narrative text, where the default subject is often a previously
    /// mentioned male character.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno::PhiFeatures;
    ///
    /// // Arabic "dhahaba" = "[he] went"
    /// let phi = PhiFeatures::third_sg_masc();
    /// ```
    #[must_use]
    pub fn third_sg_masc() -> Self {
        Self {
            person: Person::Third,
            number: Number::Singular,
            gender: Gender::Masculine,
            animacy: Animacy::Unknown,
        }
    }

    /// Create 3rd person singular feminine.
    ///
    /// Used for verbs with feminine subject agreement.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno::PhiFeatures;
    ///
    /// // Arabic "dhahabat" = "[she] went" - note the -t suffix
    /// let phi = PhiFeatures::third_sg_fem();
    /// ```
    #[must_use]
    pub fn third_sg_fem() -> Self {
        Self {
            person: Person::Third,
            number: Number::Singular,
            gender: Gender::Feminine,
            animacy: Animacy::Unknown,
        }
    }

    /// Create 3rd person plural with neutral/unspecified gender.
    ///
    /// In Arabic, plural verbs can use masculine or feminine forms depending
    /// on the referent. This constructor uses neutral for cases where gender
    /// is not recoverable from morphology.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno::PhiFeatures;
    ///
    /// // Spanish "Vinieron" = "[They] came" - plural, gender unspecified
    /// let phi = PhiFeatures::third_plural();
    /// ```
    #[must_use]
    pub fn third_plural() -> Self {
        Self {
            person: Person::Third,
            number: Number::Plural,
            gender: Gender::Unknown,
            animacy: Animacy::Unknown,
        }
    }

    /// Return a copy with the given animacy.
    ///
    /// Builder-style method for setting animacy on an existing `PhiFeatures`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno::{PhiFeatures, Animacy};
    ///
    /// let phi = PhiFeatures::third_sg_masc().with_animacy(Animacy::Animate);
    /// assert_eq!(phi.animacy, Animacy::Animate);
    /// ```
    #[must_use]
    pub fn with_animacy(mut self, animacy: Animacy) -> Self {
        self.animacy = animacy;
        self
    }

    /// Check if these phi-features are compatible with another set.
    ///
    /// Compatibility is checked per-feature:
    /// - Person must match exactly
    /// - Number allows dual↔plural (since pairs can be referred to as plural)
    /// - Gender uses the rules from [`Gender::is_compatible`]
    ///
    /// This is a soft constraint for coreference: incompatible phi-features
    /// make coreference unlikely but not impossible (errors, metaphor, etc.).
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno::PhiFeatures;
    ///
    /// let he = PhiFeatures::third_sg_masc();
    /// let she = PhiFeatures::third_sg_fem();
    /// let they = PhiFeatures::third_plural();
    ///
    /// assert!(!he.is_compatible(&she));  // Gender mismatch
    /// assert!(!he.is_compatible(&they)); // Number mismatch
    /// ```
    #[must_use]
    pub fn is_compatible(&self, other: &PhiFeatures) -> bool {
        self.person.is_compatible(&other.person)
            && self.number.is_compatible(&other.number)
            && self.gender.is_compatible(&other.gender)
            && self.animacy.is_compatible(&other.animacy)
    }

    /// Parse phi-features from a compact string notation.
    ///
    /// Accepts formats like:
    /// - `"3sgm"` - 3rd singular masculine
    /// - `"1plf"` - 1st plural feminine
    /// - `"2du"` - 2nd dual (gender unspecified)
    ///
    /// Returns `None` if the string cannot be parsed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno::{PhiFeatures, Person, Number, Gender};
    ///
    /// let phi = PhiFeatures::parse("3sgm").unwrap();
    /// assert_eq!(phi.person, Person::Third);
    /// assert_eq!(phi.number, Number::Singular);
    /// assert_eq!(phi.gender, Gender::Masculine);
    /// ```
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let lower = s.to_lowercase();

        // Try to parse formats like "3sgm", "3sgf", "3plm", etc.
        let person = if lower.starts_with('1') {
            Person::First
        } else if lower.starts_with('2') {
            Person::Second
        } else if lower.starts_with('3') {
            Person::Third
        } else {
            return None;
        };

        let rest = &lower[1..];
        let number = if rest.contains("sg") || rest.contains("sing") {
            Number::Singular
        } else if rest.contains("du") {
            Number::Dual
        } else if rest.contains("pl") {
            Number::Plural
        } else {
            Number::Singular // default
        };

        // Gender detection: use precise token matching to avoid false positives.
        // E.g. "3sing" should NOT match neutral via the 'n' in "sing".
        let gender_rest = if rest.contains("sing") {
            rest.replace("sing", "")
        } else if rest.contains("sg") {
            rest.replace("sg", "")
        } else if rest.contains("du") {
            rest.replace("du", "")
        } else if rest.contains("plur") {
            rest.replace("plur", "")
        } else if rest.contains("pl") {
            rest.replace("pl", "")
        } else {
            rest.to_string()
        };
        let gender = if gender_rest.contains('m') && !gender_rest.contains("fem") {
            Gender::Masculine
        } else if gender_rest.contains('f') || gender_rest.contains("fem") {
            Gender::Feminine
        } else if gender_rest.contains('n') || gender_rest.contains("neut") {
            Gender::Neutral
        } else {
            Gender::Unknown
        };

        Some(Self {
            person,
            number,
            gender,
            animacy: Animacy::Unknown,
        })
    }
}

impl std::fmt::Display for PhiFeatures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.person, self.number, self.gender)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_features_creation() {
        let phi = PhiFeatures::new(Person::Third, Number::Singular, Gender::Masculine);
        assert_eq!(phi.person, Person::Third);
        assert_eq!(phi.number, Number::Singular);
        assert_eq!(phi.gender, Gender::Masculine);
    }

    #[test]
    fn test_phi_features_compatibility() {
        let phi1 = PhiFeatures::third_sg_masc();
        let phi2 = PhiFeatures::third_sg_masc();
        assert!(phi1.is_compatible(&phi2));

        let phi3 = PhiFeatures::third_sg_fem();
        assert!(!phi1.is_compatible(&phi3)); // Gender mismatch

        let phi4 = PhiFeatures::third_plural();
        assert!(!phi1.is_compatible(&phi4)); // Number mismatch
    }

    #[test]
    fn test_phi_features_parse() {
        let phi = PhiFeatures::parse("3sgm").expect("parse '3sgm'");
        assert_eq!(phi.person, Person::Third);
        assert_eq!(phi.number, Number::Singular);
        assert_eq!(phi.gender, Gender::Masculine);

        let phi = PhiFeatures::parse("3plf").expect("parse '3plf'");
        assert_eq!(phi.person, Person::Third);
        assert_eq!(phi.number, Number::Plural);
        assert_eq!(phi.gender, Gender::Feminine);
    }

    #[test]
    fn test_person_display() {
        assert_eq!(format!("{}", Person::First), "1st");
        assert_eq!(format!("{}", Person::Second), "2nd");
        assert_eq!(format!("{}", Person::Third), "3rd");
    }

    #[test]
    fn test_number_display() {
        assert_eq!(format!("{}", Number::Singular), "sg");
        assert_eq!(format!("{}", Number::Dual), "du");
        assert_eq!(format!("{}", Number::Plural), "pl");
        assert_eq!(format!("{}", Number::Unknown), "?");
    }

    #[test]
    fn test_number_from_str() {
        assert_eq!(
            "sg".parse::<Number>().expect("parse 'sg'"),
            Number::Singular
        );
        assert_eq!(
            "singular".parse::<Number>().expect("parse 'singular'"),
            Number::Singular
        );
        assert_eq!("du".parse::<Number>().expect("parse 'du'"), Number::Dual);
        assert_eq!(
            "dual".parse::<Number>().expect("parse 'dual'"),
            Number::Dual
        );
        assert_eq!("pl".parse::<Number>().expect("parse 'pl'"), Number::Plural);
        assert_eq!(
            "plural".parse::<Number>().expect("parse 'plural'"),
            Number::Plural
        );
        assert_eq!("?".parse::<Number>().expect("parse '?'"), Number::Unknown);
        assert_eq!(
            "unknown".parse::<Number>().expect("parse 'unknown'"),
            Number::Unknown
        );
        assert_eq!(
            "unk".parse::<Number>().expect("parse 'unk'"),
            Number::Unknown
        );
    }

    #[test]
    fn test_number_compatibility() {
        // Exact matches
        assert!(Number::Singular.is_compatible(&Number::Singular));
        assert!(Number::Dual.is_compatible(&Number::Dual));
        assert!(Number::Plural.is_compatible(&Number::Plural));
        assert!(Number::Unknown.is_compatible(&Number::Unknown));

        // Unknown is compatible with everything
        assert!(Number::Unknown.is_compatible(&Number::Singular));
        assert!(Number::Unknown.is_compatible(&Number::Dual));
        assert!(Number::Unknown.is_compatible(&Number::Plural));
        assert!(Number::Singular.is_compatible(&Number::Unknown));
        assert!(Number::Dual.is_compatible(&Number::Unknown));
        assert!(Number::Plural.is_compatible(&Number::Unknown));

        // Dual is compatible with Plural (Semitic/Sanskrit languages)
        assert!(Number::Dual.is_compatible(&Number::Plural));
        assert!(Number::Plural.is_compatible(&Number::Dual));

        // Singular is NOT compatible with Plural or Dual
        assert!(!Number::Singular.is_compatible(&Number::Plural));
        assert!(!Number::Singular.is_compatible(&Number::Dual));
        assert!(!Number::Plural.is_compatible(&Number::Singular));
        assert!(!Number::Dual.is_compatible(&Number::Singular));
    }

    #[test]
    fn test_phi_features_display() {
        let phi = PhiFeatures::third_sg_masc();
        assert_eq!(format!("{}", phi), "3rd.sg.masculine");
    }

    #[test]
    fn test_serde_roundtrip() {
        let phi = PhiFeatures::third_sg_masc();
        let json = serde_json::to_string(&phi).expect("serialize PhiFeatures");
        let recovered: PhiFeatures = serde_json::from_str(&json).expect("deserialize PhiFeatures");
        assert_eq!(phi, recovered);
    }

    // =========================================================================
    // Linguistic invariant tests - encoding theoretical constraints
    // =========================================================================

    /// Person is a hard constraint: 1st/2nd/3rd cannot corefer.
    ///
    /// "I went to the store. She bought milk." - "I" and "She" cannot corefer.
    /// This is absolute in all known human languages.
    #[test]
    fn test_person_is_hard_constraint() {
        assert!(Person::First.is_compatible(&Person::First));
        assert!(Person::Second.is_compatible(&Person::Second));
        assert!(Person::Third.is_compatible(&Person::Third));

        // Cross-person is never compatible
        assert!(!Person::First.is_compatible(&Person::Second));
        assert!(!Person::First.is_compatible(&Person::Third));
        assert!(!Person::Second.is_compatible(&Person::Third));
    }

    /// Dual number is compatible with plural in most contexts.
    ///
    /// In Arabic/Hebrew/Sanskrit, a pair of entities (dual) can often be
    /// referred to with plural pronouns. This is a cross-linguistic pattern.
    ///
    /// Example (Arabic):
    /// - الولدان ذهبا (al-waladān dhahabā) - "The two boys went" (dual)
    /// - هم ذهبوا (hum dhahabū) - "They went" (plural can refer to the pair)
    #[test]
    fn test_dual_plural_compatibility_is_symmetric() {
        assert!(Number::Dual.is_compatible(&Number::Plural));
        assert!(Number::Plural.is_compatible(&Number::Dual));
    }

    /// Unknown number/gender should be compatible with anything.
    ///
    /// This handles:
    /// - Singular "they" in English (number ambiguous)
    /// - Epicene nouns (doctor, teacher - gender unknown without context)
    /// - Generic "you" (singular or plural)
    #[test]
    fn test_unknown_is_permissive() {
        // Unknown number is compatible with all numbers
        for number in [
            Number::Singular,
            Number::Dual,
            Number::Plural,
            Number::Unknown,
        ] {
            assert!(
                Number::Unknown.is_compatible(&number),
                "Unknown should be compatible with {:?}",
                number
            );
        }
    }

    /// PhiFeatures compatibility is conjunction of component compatibility.
    ///
    /// Two mentions can corefer only if ALL phi-features are compatible.
    /// This models agreement constraints in syntax.
    #[test]
    fn test_phi_compatibility_is_conjunction() {
        let he = PhiFeatures::third_sg_masc();
        let she = PhiFeatures::third_sg_fem();
        let they = PhiFeatures::third_plural();

        // Same features = compatible
        assert!(he.is_compatible(&he));

        // Gender mismatch = incompatible
        assert!(!he.is_compatible(&she));

        // Number mismatch = incompatible
        assert!(!he.is_compatible(&they));

        // Person mismatch would also be incompatible
        let i = PhiFeatures::new(Person::First, Number::Singular, Gender::Unknown);
        assert!(!i.is_compatible(&he));
    }

    /// The parsing format "3sgm" should round-trip correctly.
    ///
    /// This format is used in linguistic annotations and should be stable.
    #[test]
    fn test_parse_format_stability() {
        let cases = [
            ("3sgm", Person::Third, Number::Singular, Gender::Masculine),
            ("3sgf", Person::Third, Number::Singular, Gender::Feminine),
            ("3plm", Person::Third, Number::Plural, Gender::Masculine),
            ("1sg", Person::First, Number::Singular, Gender::Unknown),
            ("2du", Person::Second, Number::Dual, Gender::Unknown),
        ];

        for (input, expected_person, expected_number, expected_gender) in cases {
            let phi =
                PhiFeatures::parse(input).unwrap_or_else(|| panic!("Should parse: {}", input));
            assert_eq!(phi.person, expected_person, "Person for {}", input);
            assert_eq!(phi.number, expected_number, "Number for {}", input);
            assert_eq!(phi.gender, expected_gender, "Gender for {}", input);
            // parse always produces Unknown animacy
            assert_eq!(phi.animacy, Animacy::Unknown, "Animacy for {}", input);
        }
    }

    // =========================================================================
    // Animacy tests
    // =========================================================================

    #[test]
    fn test_animacy_default() {
        assert_eq!(Animacy::default(), Animacy::Unknown);
    }

    #[test]
    fn test_animacy_compatibility() {
        // Exact matches
        assert!(Animacy::Animate.is_compatible(&Animacy::Animate));
        assert!(Animacy::Inanimate.is_compatible(&Animacy::Inanimate));
        assert!(Animacy::Unknown.is_compatible(&Animacy::Unknown));

        // Unknown is compatible with everything
        assert!(Animacy::Unknown.is_compatible(&Animacy::Animate));
        assert!(Animacy::Unknown.is_compatible(&Animacy::Inanimate));
        assert!(Animacy::Animate.is_compatible(&Animacy::Unknown));
        assert!(Animacy::Inanimate.is_compatible(&Animacy::Unknown));

        // Animate and Inanimate are mutually exclusive
        assert!(!Animacy::Animate.is_compatible(&Animacy::Inanimate));
        assert!(!Animacy::Inanimate.is_compatible(&Animacy::Animate));
    }

    #[test]
    fn test_animacy_display() {
        assert_eq!(format!("{}", Animacy::Animate), "animate");
        assert_eq!(format!("{}", Animacy::Inanimate), "inanimate");
        assert_eq!(format!("{}", Animacy::Unknown), "?anim");
    }

    #[test]
    fn test_animacy_from_str() {
        assert_eq!(
            "animate".parse::<Animacy>().expect("parse 'animate'"),
            Animacy::Animate
        );
        assert_eq!(
            "anim".parse::<Animacy>().expect("parse 'anim'"),
            Animacy::Animate
        );
        assert_eq!(
            "inanimate".parse::<Animacy>().expect("parse 'inanimate'"),
            Animacy::Inanimate
        );
        assert_eq!(
            "inanim".parse::<Animacy>().expect("parse 'inanim'"),
            Animacy::Inanimate
        );
        assert_eq!(
            "unknown".parse::<Animacy>().expect("parse 'unknown'"),
            Animacy::Unknown
        );
        assert!("bogus".parse::<Animacy>().is_err());
    }

    #[test]
    fn test_animacy_serde_roundtrip() {
        for animacy in [Animacy::Animate, Animacy::Inanimate, Animacy::Unknown] {
            let json = serde_json::to_string(&animacy).expect("serialize Animacy");
            let recovered: Animacy = serde_json::from_str(&json).expect("deserialize Animacy");
            assert_eq!(animacy, recovered);
        }
    }

    #[test]
    fn test_with_animacy_builder() {
        let phi = PhiFeatures::third_sg_masc().with_animacy(Animacy::Animate);
        assert_eq!(phi.person, Person::Third);
        assert_eq!(phi.number, Number::Singular);
        assert_eq!(phi.gender, Gender::Masculine);
        assert_eq!(phi.animacy, Animacy::Animate);
    }

    /// Animacy mismatch blocks coreference even when other features match.
    ///
    /// "John (animate)... it (inanimate)" should be incompatible.
    #[test]
    fn test_animacy_blocks_coreference() {
        let john = PhiFeatures::third_sg_masc().with_animacy(Animacy::Animate);
        let table = PhiFeatures::third_sg_masc().with_animacy(Animacy::Inanimate);

        // Same person/number/gender but different animacy
        assert!(!john.is_compatible(&table));
    }

    /// Unknown animacy does not block coreference.
    #[test]
    fn test_unknown_animacy_is_permissive() {
        let known = PhiFeatures::third_sg_masc().with_animacy(Animacy::Animate);
        let unknown = PhiFeatures::third_sg_masc(); // animacy defaults to Unknown

        assert!(known.is_compatible(&unknown));
        assert!(unknown.is_compatible(&known));
    }

    /// PhiFeatures default has Unknown animacy.
    #[test]
    fn test_phi_features_default_animacy() {
        let phi = PhiFeatures::default();
        assert_eq!(phi.animacy, Animacy::Unknown);
    }

    // =========================================================================
    // Exhaustive compatibility matrix: Animacy x Animacy (3x3 = 9 cases)
    // =========================================================================

    /// Full 3x3 Animacy compatibility matrix.
    ///
    /// Expected truth table:
    /// ```text
    ///              Animate  Inanimate  Unknown
    /// Animate        T         F         T
    /// Inanimate      F         T         T
    /// Unknown        T         T         T
    /// ```
    #[test]
    fn test_animacy_exhaustive_compatibility_matrix() {
        use Animacy::*;
        let all = [Animate, Inanimate, Unknown];
        // Encode expected compatibility as a 3x3 bool matrix (row = self, col = other)
        let expected: [[bool; 3]; 3] = [
            // Animate x {Animate, Inanimate, Unknown}
            [true, false, true],
            // Inanimate x {Animate, Inanimate, Unknown}
            [false, true, true],
            // Unknown x {Animate, Inanimate, Unknown}
            [true, true, true],
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(
                    a.is_compatible(b),
                    expected[i][j],
                    "{:?}.is_compatible({:?}) should be {}",
                    a,
                    b,
                    expected[i][j]
                );
            }
        }
    }

    // =========================================================================
    // PhiFeatures: mixed Unknown/specific fields act as wildcards
    // =========================================================================

    /// When one PhiFeatures has Unknown gender and the other has a specific
    /// gender, they should still be compatible (Unknown acts as wildcard).
    #[test]
    fn test_phi_unknown_gender_wildcard() {
        let specific = PhiFeatures::new(Person::Third, Number::Singular, Gender::Masculine);
        let wildcard = PhiFeatures::new(Person::Third, Number::Singular, Gender::Unknown);
        assert!(specific.is_compatible(&wildcard));
        assert!(wildcard.is_compatible(&specific));
    }

    /// When one PhiFeatures has Unknown number, it wildcards.
    #[test]
    fn test_phi_unknown_number_wildcard() {
        let specific = PhiFeatures::new(Person::First, Number::Plural, Gender::Feminine);
        let wildcard = PhiFeatures {
            person: Person::First,
            number: Number::Unknown,
            gender: Gender::Feminine,
            animacy: Animacy::Unknown,
        };
        assert!(specific.is_compatible(&wildcard));
        assert!(wildcard.is_compatible(&specific));
    }

    /// All-Unknown PhiFeatures is compatible with any concrete PhiFeatures.
    #[test]
    fn test_phi_all_unknown_is_universal_wildcard() {
        let all_unknown = PhiFeatures {
            person: Person::Unknown,
            number: Number::Unknown,
            gender: Gender::Unknown,
            animacy: Animacy::Unknown,
        };
        // Should be compatible with any features (all unknown = universal wildcard)
        let cases = [
            PhiFeatures::third_sg_masc(),
            PhiFeatures::third_sg_fem(),
            PhiFeatures::third_plural(),
            PhiFeatures::new(Person::Third, Number::Dual, Gender::Neutral),
        ];
        for (i, phi) in cases.iter().enumerate() {
            assert!(
                all_unknown.is_compatible(phi),
                "all_unknown should be compatible with case {}: {:?}",
                i,
                phi
            );
            assert!(
                phi.is_compatible(&all_unknown),
                "case {} should be compatible with all_unknown: {:?}",
                i,
                phi
            );
        }
        // With Person::Unknown, all_unknown is compatible with any person
        let first = PhiFeatures::new(Person::First, Number::Unknown, Gender::Unknown);
        assert!(all_unknown.is_compatible(&first));
    }

    /// A single incompatible field is sufficient to block compatibility.
    #[test]
    fn test_phi_single_field_mismatch_blocks() {
        let base = PhiFeatures::third_sg_masc().with_animacy(Animacy::Animate);

        // Person mismatch only
        let diff_person = PhiFeatures {
            person: Person::First,
            ..base
        };
        assert!(!base.is_compatible(&diff_person));

        // Number mismatch only
        let diff_number = PhiFeatures {
            number: Number::Plural,
            ..base
        };
        assert!(!base.is_compatible(&diff_number));

        // Gender mismatch only
        let diff_gender = PhiFeatures {
            gender: Gender::Feminine,
            ..base
        };
        assert!(!base.is_compatible(&diff_gender));

        // Animacy mismatch only
        let diff_animacy = PhiFeatures {
            animacy: Animacy::Inanimate,
            ..base
        };
        assert!(!base.is_compatible(&diff_animacy));
    }

    // Entity phi_features field was removed (zero consumers outside tests).
    // PhiFeatures types remain available for coref backends.

    // =========================================================================
    // Property tests (proptest)
    // =========================================================================

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_person() -> impl Strategy<Value = Person> {
            prop_oneof![
                Just(Person::First),
                Just(Person::Second),
                Just(Person::Third),
                Just(Person::Unknown),
            ]
        }

        fn arb_number() -> impl Strategy<Value = Number> {
            prop_oneof![
                Just(Number::Singular),
                Just(Number::Dual),
                Just(Number::Plural),
                Just(Number::Unknown),
            ]
        }

        fn arb_gender() -> impl Strategy<Value = Gender> {
            prop_oneof![
                Just(Gender::Masculine),
                Just(Gender::Feminine),
                Just(Gender::Neutral),
                Just(Gender::Unknown),
            ]
        }

        fn arb_animacy() -> impl Strategy<Value = Animacy> {
            prop_oneof![
                Just(Animacy::Animate),
                Just(Animacy::Inanimate),
                Just(Animacy::Unknown),
            ]
        }

        fn arb_phi() -> impl Strategy<Value = PhiFeatures> {
            (arb_person(), arb_number(), arb_gender(), arb_animacy()).prop_map(
                |(person, number, gender, animacy)| PhiFeatures {
                    person,
                    number,
                    gender,
                    animacy,
                },
            )
        }

        proptest! {
            /// PhiFeatures::is_compatible is reflexive: a.is_compatible(&a) == true.
            #[test]
            fn phi_compatible_reflexive(phi in arb_phi()) {
                prop_assert!(phi.is_compatible(&phi),
                    "PhiFeatures {:?} should be compatible with itself", phi);
            }

            /// PhiFeatures::is_compatible is symmetric.
            #[test]
            fn phi_compatible_symmetric(a in arb_phi(), b in arb_phi()) {
                prop_assert_eq!(
                    a.is_compatible(&b),
                    b.is_compatible(&a),
                    "Symmetry violated: {:?} vs {:?}", a, b
                );
            }

            /// Animacy::is_compatible is reflexive.
            #[test]
            fn animacy_compatible_reflexive(a in arb_animacy()) {
                prop_assert!(a.is_compatible(&a),
                    "Animacy {:?} should be compatible with itself", a);
            }

            /// Animacy::is_compatible is symmetric.
            #[test]
            fn animacy_compatible_symmetric(a in arb_animacy(), b in arb_animacy()) {
                prop_assert_eq!(
                    a.is_compatible(&b),
                    b.is_compatible(&a),
                    "Animacy symmetry violated: {:?} vs {:?}", a, b
                );
            }

            /// Unknown animacy is compatible with everything.
            #[test]
            fn animacy_unknown_universal(a in arb_animacy()) {
                prop_assert!(Animacy::Unknown.is_compatible(&a),
                    "Unknown should be compatible with {:?}", a);
                prop_assert!(a.is_compatible(&Animacy::Unknown),
                    "{:?} should be compatible with Unknown", a);
            }

            /// Number::is_compatible is reflexive.
            #[test]
            fn number_compatible_reflexive(n in arb_number()) {
                prop_assert!(n.is_compatible(&n));
            }

            /// Number::is_compatible is symmetric.
            #[test]
            fn number_compatible_symmetric(a in arb_number(), b in arb_number()) {
                prop_assert_eq!(
                    a.is_compatible(&b),
                    b.is_compatible(&a),
                    "Number symmetry violated: {:?} vs {:?}", a, b
                );
            }

            /// Person::is_compatible is reflexive.
            #[test]
            fn person_compatible_reflexive(p in arb_person()) {
                prop_assert!(p.is_compatible(&p));
            }

            /// PhiFeatures serde round-trip preserves equality.
            #[test]
            fn phi_serde_roundtrip(phi in arb_phi()) {
                let json = serde_json::to_string(&phi).unwrap();
                let recovered: PhiFeatures = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(phi, recovered);
            }

            /// Animacy parse round-trip: Display output can be re-parsed.
            #[test]
            fn animacy_display_parse_roundtrip(a in arb_animacy()) {
                let displayed = format!("{}", a);
                let parsed: Animacy = displayed.parse().unwrap();
                prop_assert_eq!(a, parsed);
            }

            /// Number parse round-trip: Display output can be re-parsed.
            #[test]
            fn number_display_parse_roundtrip(n in arb_number()) {
                let displayed = format!("{}", n);
                let parsed: Number = displayed.parse().unwrap();
                prop_assert_eq!(n, parsed);
            }

            /// Person parse round-trip: Display output can be re-parsed.
            #[test]
            fn person_display_parse_roundtrip(p in arb_person()) {
                let displayed = format!("{}", p);
                let parsed: Person = displayed.parse().unwrap();
                prop_assert_eq!(p, parsed);
            }
        }
    }

    // =========================================================================
    // Audit-driven regression tests
    // =========================================================================

    /// Person::Unknown is compatible with First, Second, Third, and Unknown.
    #[test]
    fn test_person_unknown_compatible_with_all() {
        for p in [
            Person::First,
            Person::Second,
            Person::Third,
            Person::Unknown,
        ] {
            assert!(
                Person::Unknown.is_compatible(&p),
                "Person::Unknown should be compatible with {:?}",
                p
            );
        }
    }

    /// Unknown.is_compatible(x) == x.is_compatible(Unknown) for all Person values.
    #[test]
    fn test_person_unknown_wildcard_symmetric() {
        for p in [
            Person::First,
            Person::Second,
            Person::Third,
            Person::Unknown,
        ] {
            assert_eq!(
                Person::Unknown.is_compatible(&p),
                p.is_compatible(&Person::Unknown),
                "Person::Unknown symmetry broken for {:?}",
                p
            );
        }
    }

    /// parse("3sing") should produce gender Unknown, not Neutral.
    ///
    /// The 'n' in "sing" must not be misinterpreted as a gender marker.
    /// This was a real bug: the parser matched 'n' in "sing" as Neutral.
    #[test]
    fn test_parse_3sing_no_false_gender() {
        let phi = PhiFeatures::parse("3sing").expect("should parse '3sing'");
        assert_eq!(phi.person, Person::Third);
        assert_eq!(phi.number, Number::Singular);
        assert_eq!(
            phi.gender,
            Gender::Unknown,
            "3sing should have Unknown gender, not Neutral (the 'n' is part of 'sing')"
        );
    }

    /// parse("3sgn") should produce gender Neutral (explicit neutral marker).
    #[test]
    fn test_parse_3sgn_is_neutral() {
        let phi = PhiFeatures::parse("3sgn").expect("should parse '3sgn'");
        assert_eq!(phi.person, Person::Third);
        assert_eq!(phi.number, Number::Singular);
        assert_eq!(phi.gender, Gender::Neutral);
    }

    /// parse("3fem") should produce Third, Singular (default), Feminine.
    #[test]
    fn test_parse_3fem_defaults_singular() {
        let phi = PhiFeatures::parse("3fem").expect("should parse '3fem'");
        assert_eq!(phi.person, Person::Third);
        assert_eq!(
            phi.number,
            Number::Singular,
            "no number marker -> default Singular"
        );
        assert_eq!(phi.gender, Gender::Feminine);
    }

    /// For each valid combo, Display -> parse -> compare field-by-field.
    ///
    /// Display format is "Xth.YY.gender" which doesn't round-trip through
    /// PhiFeatures::parse (different format), so we test Display -> FromStr
    /// on individual components instead.
    #[test]
    fn test_parse_roundtrip_all_combinations() {
        // Test all parse-format strings that should round-trip
        let cases = [
            ("1sgm", Person::First, Number::Singular, Gender::Masculine),
            ("1sgf", Person::First, Number::Singular, Gender::Feminine),
            ("1sgn", Person::First, Number::Singular, Gender::Neutral),
            ("1sg", Person::First, Number::Singular, Gender::Unknown),
            ("2dum", Person::Second, Number::Dual, Gender::Masculine),
            ("2duf", Person::Second, Number::Dual, Gender::Feminine),
            ("3plm", Person::Third, Number::Plural, Gender::Masculine),
            ("3plf", Person::Third, Number::Plural, Gender::Feminine),
            ("3pln", Person::Third, Number::Plural, Gender::Neutral),
            ("3pl", Person::Third, Number::Plural, Gender::Unknown),
        ];
        for (input, exp_p, exp_n, exp_g) in cases {
            let phi =
                PhiFeatures::parse(input).unwrap_or_else(|| panic!("should parse '{}'", input));
            assert_eq!(phi.person, exp_p, "Person for {}", input);
            assert_eq!(phi.number, exp_n, "Number for {}", input);
            assert_eq!(phi.gender, exp_g, "Gender for {}", input);
            assert_eq!(phi.animacy, Animacy::Unknown, "Animacy for {}", input);
        }
    }

    /// All 3 Animacy variants roundtrip through Display -> FromStr.
    #[test]
    fn test_animacy_display_parse_roundtrip_all() {
        for animacy in [Animacy::Animate, Animacy::Inanimate, Animacy::Unknown] {
            let displayed = format!("{}", animacy);
            let parsed: Animacy = displayed.parse().unwrap_or_else(|e| {
                panic!(
                    "Animacy::{:?}  display='{}' failed to parse: {}",
                    animacy, displayed, e
                )
            });
            assert_eq!(
                animacy, parsed,
                "Animacy roundtrip failed for {:?}",
                animacy
            );
        }
    }

    /// Animacy::Unknown is compatible with all animacy values (wildcard).
    #[test]
    fn test_animacy_unknown_compatible_with_all() {
        for a in [Animacy::Animate, Animacy::Inanimate, Animacy::Unknown] {
            assert!(
                Animacy::Unknown.is_compatible(&a),
                "Animacy::Unknown should be compatible with {:?}",
                a
            );
            assert!(
                a.is_compatible(&Animacy::Unknown),
                "{:?} should be compatible with Animacy::Unknown",
                a
            );
        }
    }

    /// PhiFeatures compatibility is NOT transitive (same pattern as Gender).
    ///
    /// 3sg_masc ~ 3sg_unknown and 3sg_unknown ~ 3sg_fem,
    /// but 3sg_masc is NOT compatible with 3sg_fem.
    #[test]
    fn test_phi_compatibility_not_transitive() {
        let masc = PhiFeatures::third_sg_masc();
        let fem = PhiFeatures::third_sg_fem();
        let unknown_gender = PhiFeatures::new(Person::Third, Number::Singular, Gender::Unknown);

        assert!(masc.is_compatible(&unknown_gender), "masc ~ unknown");
        assert!(unknown_gender.is_compatible(&fem), "unknown ~ fem");
        assert!(
            !masc.is_compatible(&fem),
            "masc NOT ~ fem (transitivity must not hold)"
        );
    }

    /// Dual is compatible with Plural (Arabic linguistics).
    ///
    /// In Arabic, dual nouns can be referred to with plural pronouns:
    /// "The two boys... they went" is grammatical.
    #[test]
    fn test_number_dual_plural_compatible() {
        assert!(Number::Dual.is_compatible(&Number::Plural));
        assert!(Number::Plural.is_compatible(&Number::Dual));

        // But Dual is NOT compatible with Singular
        assert!(!Number::Dual.is_compatible(&Number::Singular));
        assert!(!Number::Singular.is_compatible(&Number::Dual));
    }
}
