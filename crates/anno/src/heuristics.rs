//! Small, dependency-light heuristics shared across the repo.
//!
//! These helpers are intentionally conservative and are primarily used to:
//! - enrich extraction outputs (negation / quantifiers) in the CLI
//! - provide lightweight baselines when heavier backends are unavailable
//!
//! They operate on **character offsets** (not byte offsets), matching `anno`'s span contract.
//!
//! ## Language / domain agnosticism
//!
//! This module is intentionally **language-agnostic by default**:
//! - the generic entrypoints accept caller-provided cue lexicons
//! - language-specific defaults live under `heuristics::lexicons`

use crate::Quantifier;

use crate::lang::{detect_language, Language};

/// Detect language in a window around a character offset.
///
/// This is more robust than calling `detect_language(text)` on the entire document when the input
/// is code-switched or contains long quoted spans in another script.
fn detect_language_near(text: &str, char_offset: usize) -> Language {
    // Fast path: if the character at the offset is from a distinctive script,
    // we can classify without looking at a window (robust for short code-switch spans).
    if let Some(c) = text.chars().nth(char_offset) {
        match c {
            '\u{4e00}'..='\u{9fff}' => return Language::Chinese,
            '\u{3040}'..='\u{30ff}' => return Language::Japanese,
            '\u{ac00}'..='\u{d7af}' => return Language::Korean,
            '\u{0600}'..='\u{06ff}' => return Language::Arabic,
            '\u{0590}'..='\u{05ff}' => return Language::Hebrew,
            '\u{0400}'..='\u{04ff}' => return Language::Russian,
            _ => {}
        }
    }

    // Keep this fairly small: we want the local "surface" language.
    //
    // Use an *asymmetric* window around the offset:
    // - a small prefix to catch short markers ("not", "kein", etc.)
    // - a larger suffix so a long Latin prefix doesn't drown out a short CJK/RTL segment.
    const PRE_CHARS: usize = 32;
    const POST_CHARS: usize = 160;
    let window_start = char_offset.saturating_sub(PRE_CHARS);
    let window: String = text
        .chars()
        .skip(window_start)
        .take(PRE_CHARS + POST_CHARS)
        .collect();

    // If the local window is empty (very short text), fall back to whole-text detection.
    if window.is_empty() {
        detect_language(text)
    } else {
        detect_language(&window)
    }
}

/// Detect whether an entity mention is likely negated, using an explicit cue lexicon.
///
/// `entity_start` is a **character** offset into `text`.
#[must_use]
pub fn is_negated_with_cues(text: &str, entity_start: usize, cues: &[&str]) -> bool {
    // Negation cues are typically local (a few words before the entity), so scan a bounded window.
    const WINDOW_CHARS: usize = 200;
    let window_start = entity_start.saturating_sub(WINDOW_CHARS);
    let prefix: String = text
        .chars()
        .skip(window_start)
        .take(entity_start.saturating_sub(window_start))
        .collect();
    let words: Vec<&str> = prefix.split_whitespace().collect();
    let last_words: Vec<&str> = words.iter().rev().take(3).copied().collect();

    for word in &last_words {
        // We only lower-case a handful of tokens; keep it simple and Unicode-safe.
        if cues.contains(&word.to_lowercase().as_str()) {
            return true;
        }
    }

    false
}

/// Detect negation using substring cues (best-effort; useful for scripts without whitespace).
///
/// `entity_start` is a **character** offset into `text`.
#[must_use]
pub fn is_negated_with_substrings(text: &str, entity_start: usize, cues: &[&str]) -> bool {
    const WINDOW_CHARS: usize = 200;
    let window_start = entity_start.saturating_sub(WINDOW_CHARS);
    let prefix: String = text
        .chars()
        .skip(window_start)
        .take(entity_start.saturating_sub(window_start))
        .collect();

    cues.iter().any(|cue| prefix.contains(cue))
}

/// Detect a quantifier immediately before an entity mention, using explicit cue lexicons.
///
/// `entity_start` is a **character** offset into `text`.
#[must_use]
pub fn detect_quantifier_with_cues(
    text: &str,
    entity_start: usize,
    universal: &[&str],
    existential: &[&str],
    none: &[&str],
    definite: &[&str],
) -> Option<Quantifier> {
    // Quantifiers are almost always adjacent; keep the scan tight.
    const WINDOW_CHARS: usize = 80;
    let window_start = entity_start.saturating_sub(WINDOW_CHARS);
    let prefix: String = text
        .chars()
        .skip(window_start)
        .take(entity_start.saturating_sub(window_start))
        .collect();
    let words: Vec<&str> = prefix.split_whitespace().collect();

    words.last().and_then(|word| {
        let w = word.to_lowercase();
        let w = w.as_str();
        // Prefer "none" before "universal/any" to avoid accidental overlaps in cue sets.
        if none.contains(&w) {
            Some(Quantifier::None)
        } else if universal.contains(&w) {
            Some(Quantifier::Universal)
        } else if existential.contains(&w) {
            Some(Quantifier::Existential)
        } else if definite.contains(&w) {
            Some(Quantifier::Definite)
        } else {
            None
        }
    })
}

/// Detect a quantifier using substring cues (best-effort; useful for scripts without whitespace).
///
/// `entity_start` is a **character** offset into `text`.
#[must_use]
pub fn detect_quantifier_with_substrings(
    text: &str,
    entity_start: usize,
    universal: &[&str],
    existential: &[&str],
    none: &[&str],
    definite: &[&str],
) -> Option<Quantifier> {
    // Quantifiers are usually adjacent; scan a small window.
    const WINDOW_CHARS: usize = 64;
    let window_start = entity_start.saturating_sub(WINDOW_CHARS);
    let prefix: String = text
        .chars()
        .skip(window_start)
        .take(entity_start.saturating_sub(window_start))
        .collect();

    // Check in an order that minimizes false positives.
    if none.iter().any(|cue| prefix.contains(cue)) {
        Some(Quantifier::None)
    } else if universal.iter().any(|cue| prefix.contains(cue)) {
        Some(Quantifier::Universal)
    } else if existential.iter().any(|cue| prefix.contains(cue)) {
        Some(Quantifier::Existential)
    } else if definite.iter().any(|cue| prefix.contains(cue)) {
        Some(Quantifier::Definite)
    } else {
        None
    }
}

/// Built-in language-specific cue lexicons.
///
/// Keep these small and clearly labeled; they are convenience defaults, not a universal model.
pub mod lexicons {
    /// English cue words for negation detection.
    pub const EN_NEGATION_WORDS: &[&str] = &[
        "not",
        "no",
        "never",
        "none",
        "neither",
        "nor",
        "without",
        "negative",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "wouldn't",
        "couldn't",
        "shouldn't",
    ];

    /// English cue words for quantifier detection, grouped by quantifier class.
    pub const EN_UNIVERSAL: &[&str] = &["every", "all", "each", "any"];
    /// English cue words suggesting existential quantification.
    pub const EN_EXISTENTIAL: &[&str] = &["some", "certain", "a", "an"];
    /// English cue words suggesting "none / no" quantification.
    pub const EN_NONE: &[&str] = &["no", "none"];
    /// English cue words suggesting definiteness / demonstratives.
    pub const EN_DEFINITE: &[&str] = &["the", "this", "that", "these", "those"];
    /// English cue phrases suggesting approximate quantification.
    pub const EN_APPROXIMATE: &[&str] = &[
        "approximately",
        "about",
        "roughly",
        "nearly",
        "around",
        "over",
        "more than",
        "fewer than",
        "less than",
    ];
    /// English cue phrases suggesting a lower bound ("at least N").
    pub const EN_MIN_BOUND: &[&str] = &["at least", "no fewer than"];
    /// English cue phrases suggesting an upper bound ("at most N").
    pub const EN_MAX_BOUND: &[&str] = &["at most", "no more than", "up to"];

    /// German cue words for negation detection.
    pub const DE_NEGATION_WORDS: &[&str] = &["nicht", "kein", "keine", "keinen", "nie", "ohne"];
    /// French cue words for negation detection.
    pub const FR_NEGATION_WORDS: &[&str] = &["pas", "jamais", "aucun", "aucune", "sans"];
    /// Spanish cue words for negation detection.
    pub const ES_NEGATION_WORDS: &[&str] = &["no", "nunca", "ningun", "ningún", "ninguna", "sin"];
    /// Italian cue words for negation detection.
    pub const IT_NEGATION_WORDS: &[&str] = &["non", "mai", "nessun", "nessuna", "senza"];
    /// Portuguese cue words for negation detection.
    pub const PT_NEGATION_WORDS: &[&str] = &["não", "nao", "nunca", "nenhum", "nenhuma", "sem"];
    /// Russian cue words for negation detection.
    pub const RU_NEGATION_WORDS: &[&str] = &["не", "нет", "никогда", "без"];

    /// German cue words for quantifier detection.
    pub const DE_UNIVERSAL: &[&str] = &["alle", "jeder", "jede", "jedes"];
    /// German cue words suggesting existential quantification.
    pub const DE_EXISTENTIAL: &[&str] = &["ein", "eine", "einen", "einige", "manche"];
    /// German cue words suggesting "none / no" quantification.
    pub const DE_NONE: &[&str] = &["kein", "keine", "keinen", "keinem", "keiner", "keins"];
    /// German cue words suggesting definiteness / demonstratives.
    pub const DE_DEFINITE: &[&str] = &[
        "der", "die", "das", "diese", "dieser", "dieses", "jener", "jene", "jenes",
    ];

    /// French cue words for quantifier detection.
    pub const FR_UNIVERSAL: &[&str] = &["tous", "toutes", "chaque"];
    /// French cue words suggesting existential quantification.
    pub const FR_EXISTENTIAL: &[&str] = &["un", "une", "des", "quelques", "certains", "certaines"];
    /// French cue words suggesting "none / no" quantification.
    pub const FR_NONE: &[&str] = &["aucun", "aucune"];
    /// French cue words suggesting definiteness / demonstratives.
    pub const FR_DEFINITE: &[&str] = &["le", "la", "les", "ce", "cette", "ces", "cet"];

    /// Spanish cue words for quantifier detection.
    pub const ES_UNIVERSAL: &[&str] = &["todos", "todas", "cada", "cualquier"];
    /// Spanish cue words suggesting existential quantification.
    pub const ES_EXISTENTIAL: &[&str] = &[
        "un", "una", "unos", "unas", "algún", "alguna", "algunos", "algunas",
    ];
    /// Spanish cue words suggesting "none / no" quantification.
    pub const ES_NONE: &[&str] = &[
        "ningún", "ninguna", "ninguno", "ningunos", "ningunas", "ningun",
    ];
    /// Spanish cue words suggesting definiteness / demonstratives.
    pub const ES_DEFINITE: &[&str] = &[
        "el", "la", "los", "las", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    ];

    /// Italian cue words for quantifier detection.
    pub const IT_UNIVERSAL: &[&str] = &["tutti", "tutte", "ogni", "qualsiasi"];
    /// Italian cue words suggesting existential quantification.
    pub const IT_EXISTENTIAL: &[&str] = &[
        "un", "una", "uno", "alcuni", "alcune", "qualche", "certi", "certe",
    ];
    /// Italian cue words suggesting "none / no" quantification.
    pub const IT_NONE: &[&str] = &["nessun", "nessuna", "nessuno"];
    /// Italian cue words suggesting definiteness / demonstratives.
    pub const IT_DEFINITE: &[&str] = &[
        "il", "lo", "la", "i", "gli", "le", "questo", "questa", "questi", "queste", "quello",
        "quella",
    ];

    /// Portuguese cue words for quantifier detection.
    pub const PT_UNIVERSAL: &[&str] = &["todos", "todas", "cada", "qualquer"];
    /// Portuguese cue words suggesting existential quantification.
    pub const PT_EXISTENTIAL: &[&str] = &[
        "um", "uma", "uns", "umas", "algum", "alguma", "alguns", "algumas",
    ];
    /// Portuguese cue words suggesting "none / no" quantification.
    pub const PT_NONE: &[&str] = &["nenhum", "nenhuma", "nenhuns", "nenhumas"];
    /// Portuguese cue words suggesting definiteness / demonstratives.
    pub const PT_DEFINITE: &[&str] = &[
        "o", "a", "os", "as", "este", "esta", "estes", "estas", "esse", "essa", "esses", "essas",
    ];

    /// Russian cue words for quantifier detection.
    pub const RU_UNIVERSAL: &[&str] = &["все", "каждый", "каждая", "каждое"];
    /// Russian cue words suggesting existential quantification.
    pub const RU_EXISTENTIAL: &[&str] = &["некоторые", "один", "одна", "одно"];
    /// Russian cue words suggesting "none / no" quantification.
    pub const RU_NONE: &[&str] = &["никакой", "никакая", "никакие", "нет"];
    /// Russian cue words suggesting definiteness / demonstratives.
    pub const RU_DEFINITE: &[&str] = &["этот", "эта", "это", "эти", "тот", "та", "то", "те"];

    /// Chinese negation cues (substring-based).
    pub const ZH_NEGATION_CUES: &[&str] = &["不", "没", "沒有", "没有", "無", "无"];
    /// Japanese negation cues (substring-based).
    pub const JA_NEGATION_CUES: &[&str] = &["ない", "ません", "無い", "ず"];
    /// Korean negation cues (substring-based).
    pub const KO_NEGATION_CUES: &[&str] = &["안", "않", "못", "없"];

    /// Chinese quantifier cues (substring-based).
    pub const ZH_UNIVERSAL: &[&str] = &["每", "所有", "全部"];
    /// Chinese cues suggesting existential quantification.
    pub const ZH_EXISTENTIAL: &[&str] = &["一些", "某些", "有些"];
    /// Chinese cues suggesting "none / no" quantification.
    pub const ZH_NONE: &[&str] = &["没有", "沒有", "无", "無", "没"];
    /// Chinese cues suggesting definiteness / demonstratives.
    pub const ZH_DEFINITE: &[&str] = &["这", "那", "这些", "那些", "该"];

    /// Japanese quantifier cues (substring-based).
    pub const JA_UNIVERSAL: &[&str] = &["全て", "すべて", "毎", "各"];
    /// Japanese cues suggesting existential quantification.
    pub const JA_EXISTENTIAL: &[&str] = &["いくつか", "ある"];
    /// Japanese cues suggesting "none / no" quantification.
    pub const JA_NONE: &[&str] = &["無い", "無し", "ない"];
    /// Japanese cues suggesting definiteness / demonstratives.
    pub const JA_DEFINITE: &[&str] = &["この", "その", "あの"];

    /// Korean quantifier cues (substring-based).
    pub const KO_UNIVERSAL: &[&str] = &["모든", "각", "매"];
    /// Korean cues suggesting existential quantification.
    pub const KO_EXISTENTIAL: &[&str] = &["몇몇", "일부", "어떤", "어느"];
    /// Korean cues suggesting "none / no" quantification.
    pub const KO_NONE: &[&str] = &["없", "아무도", "아무것도"];
    /// Korean cues suggesting definiteness / demonstratives.
    pub const KO_DEFINITE: &[&str] = &["이", "그", "저"];

    /// Arabic cue words (token-based, minimal).
    pub const AR_UNIVERSAL: &[&str] = &["كل"];
    /// Arabic cues suggesting existential quantification.
    pub const AR_EXISTENTIAL: &[&str] = &["بعض", "أحد", "احد"];
    /// Arabic cues suggesting "none / no" quantification.
    pub const AR_NONE: &[&str] = &["لا", "ليس", "بدون"];
    /// Arabic cues suggesting definiteness / demonstratives.
    pub const AR_DEFINITE: &[&str] = &["هذا", "هذه", "ذلك", "تلك", "هؤلاء"];

    /// Hebrew cue words (token-based, minimal).
    pub const HE_UNIVERSAL: &[&str] = &["כל"];
    /// Hebrew cues suggesting existential quantification.
    pub const HE_EXISTENTIAL: &[&str] = &["כמה"];
    /// Hebrew cues suggesting "none / no" quantification.
    pub const HE_NONE: &[&str] = &["אין", "לא"];
    /// Hebrew cues suggesting definiteness / demonstratives.
    pub const HE_DEFINITE: &[&str] = &["זה", "זאת", "אלה", "האלו"];
}

/// Convenience wrapper: English negation detection.
#[must_use]
pub fn is_negated_en(text: &str, entity_start: usize) -> bool {
    is_negated_with_cues(text, entity_start, lexicons::EN_NEGATION_WORDS)
}

/// Best-effort negation detection for a specific language.
///
/// This is intentionally conservative: for languages/scripts where we don't have reliable
/// tokenization/lexicons, this may return `false` rather than guessing.
#[must_use]
pub fn is_negated_lang(text: &str, entity_start: usize, lang: Language) -> bool {
    match lang {
        Language::English => is_negated_en(text, entity_start),
        Language::German => is_negated_with_cues(text, entity_start, lexicons::DE_NEGATION_WORDS),
        Language::French => is_negated_with_cues(text, entity_start, lexicons::FR_NEGATION_WORDS),
        Language::Spanish => is_negated_with_cues(text, entity_start, lexicons::ES_NEGATION_WORDS),
        Language::Italian => is_negated_with_cues(text, entity_start, lexicons::IT_NEGATION_WORDS),
        Language::Portuguese => {
            is_negated_with_cues(text, entity_start, lexicons::PT_NEGATION_WORDS)
        }
        Language::Russian => is_negated_with_cues(text, entity_start, lexicons::RU_NEGATION_WORDS),
        Language::Chinese => {
            is_negated_with_substrings(text, entity_start, lexicons::ZH_NEGATION_CUES)
        }
        Language::Japanese => {
            is_negated_with_substrings(text, entity_start, lexicons::JA_NEGATION_CUES)
        }
        Language::Korean => {
            is_negated_with_substrings(text, entity_start, lexicons::KO_NEGATION_CUES)
        }
        Language::Arabic | Language::Hebrew | Language::Other => false,
    }
}

/// Convenience wrapper: language detection + best-effort negation.
#[must_use]
pub fn is_negated_auto(text: &str, entity_start: usize) -> bool {
    is_negated_lang(text, entity_start, detect_language_near(text, entity_start))
}

/// Convenience wrapper: English quantifier detection.
#[must_use]
pub fn detect_quantifier_en(text: &str, entity_start: usize) -> Option<Quantifier> {
    detect_quantifier_with_cues(
        text,
        entity_start,
        lexicons::EN_UNIVERSAL,
        lexicons::EN_EXISTENTIAL,
        lexicons::EN_NONE,
        lexicons::EN_DEFINITE,
    )
    .or_else(|| detect_approximate_quantifier(text, entity_start))
}

/// Best-effort quantifier detection for a specific language.
///
/// This is intentionally conservative: for languages/scripts where we don't have reliable
/// tokenization/lexicons, this returns `None` rather than guessing.
#[must_use]
pub fn detect_quantifier_lang(
    text: &str,
    entity_start: usize,
    lang: Language,
) -> Option<Quantifier> {
    match lang {
        Language::English => detect_quantifier_en(text, entity_start),
        Language::German => detect_quantifier_with_cues(
            text,
            entity_start,
            lexicons::DE_UNIVERSAL,
            lexicons::DE_EXISTENTIAL,
            lexicons::DE_NONE,
            lexicons::DE_DEFINITE,
        ),
        Language::French => detect_quantifier_with_cues(
            text,
            entity_start,
            lexicons::FR_UNIVERSAL,
            lexicons::FR_EXISTENTIAL,
            lexicons::FR_NONE,
            lexicons::FR_DEFINITE,
        ),
        Language::Spanish => detect_quantifier_with_cues(
            text,
            entity_start,
            lexicons::ES_UNIVERSAL,
            lexicons::ES_EXISTENTIAL,
            lexicons::ES_NONE,
            lexicons::ES_DEFINITE,
        ),
        Language::Italian => detect_quantifier_with_cues(
            text,
            entity_start,
            lexicons::IT_UNIVERSAL,
            lexicons::IT_EXISTENTIAL,
            lexicons::IT_NONE,
            lexicons::IT_DEFINITE,
        ),
        Language::Portuguese => detect_quantifier_with_cues(
            text,
            entity_start,
            lexicons::PT_UNIVERSAL,
            lexicons::PT_EXISTENTIAL,
            lexicons::PT_NONE,
            lexicons::PT_DEFINITE,
        ),
        Language::Russian => detect_quantifier_with_cues(
            text,
            entity_start,
            lexicons::RU_UNIVERSAL,
            lexicons::RU_EXISTENTIAL,
            lexicons::RU_NONE,
            lexicons::RU_DEFINITE,
        ),
        Language::Chinese => detect_quantifier_with_substrings(
            text,
            entity_start,
            lexicons::ZH_UNIVERSAL,
            lexicons::ZH_EXISTENTIAL,
            lexicons::ZH_NONE,
            lexicons::ZH_DEFINITE,
        ),
        Language::Japanese => detect_quantifier_with_substrings(
            text,
            entity_start,
            lexicons::JA_UNIVERSAL,
            lexicons::JA_EXISTENTIAL,
            lexicons::JA_NONE,
            lexicons::JA_DEFINITE,
        ),
        Language::Korean => detect_quantifier_with_substrings(
            text,
            entity_start,
            lexicons::KO_UNIVERSAL,
            lexicons::KO_EXISTENTIAL,
            lexicons::KO_NONE,
            lexicons::KO_DEFINITE,
        ),
        Language::Arabic => {
            // Token cues + a minimal definite-article prefix check (`ال...`).
            let q = detect_quantifier_with_cues(
                text,
                entity_start,
                lexicons::AR_UNIVERSAL,
                lexicons::AR_EXISTENTIAL,
                lexicons::AR_NONE,
                lexicons::AR_DEFINITE,
            );
            if q.is_some() {
                return q;
            }
            // If the immediate preceding token starts with the Arabic definite article, treat as
            // definite. This is intentionally minimal and may miss clitics/diacritics.
            let window_start = entity_start.saturating_sub(40);
            let prefix: String = text
                .chars()
                .skip(window_start)
                .take(entity_start.saturating_sub(window_start))
                .collect();
            let last = prefix.split_whitespace().last().unwrap_or("");
            if last.starts_with("ال") {
                Some(Quantifier::Definite)
            } else {
                None
            }
        }
        Language::Hebrew => {
            // Token cues + a minimal definite-article prefix check (`ה...`).
            let q = detect_quantifier_with_cues(
                text,
                entity_start,
                lexicons::HE_UNIVERSAL,
                lexicons::HE_EXISTENTIAL,
                lexicons::HE_NONE,
                lexicons::HE_DEFINITE,
            );
            if q.is_some() {
                return q;
            }
            let window_start = entity_start.saturating_sub(40);
            let prefix: String = text
                .chars()
                .skip(window_start)
                .take(entity_start.saturating_sub(window_start))
                .collect();
            let last = prefix.split_whitespace().last().unwrap_or("");
            if last.starts_with('ה') {
                Some(Quantifier::Definite)
            } else {
                None
            }
        }
        Language::Other => None,
    }
}

/// Detect approximate quantifiers by scanning the prefix for multi-word cue phrases.
///
/// Unlike the standard quantifier detection (which only checks the last word),
/// approximate quantifiers like "at least", "more than", "approximately" often
/// appear several words before the entity (e.g., "approximately 500 employees").
#[must_use]
pub fn detect_approximate_quantifier(text: &str, entity_start: usize) -> Option<Quantifier> {
    const WINDOW_CHARS: usize = 40;
    let window_start = entity_start.saturating_sub(WINDOW_CHARS);
    let prefix: String = text
        .chars()
        .skip(window_start)
        .take(entity_start.saturating_sub(window_start))
        .collect();
    let lower = prefix.to_lowercase();
    if lexicons::EN_MIN_BOUND.iter().any(|cue| lower.contains(cue)) {
        Some(Quantifier::MinBound)
    } else if lexicons::EN_MAX_BOUND.iter().any(|cue| lower.contains(cue)) {
        Some(Quantifier::MaxBound)
    } else if lexicons::EN_APPROXIMATE
        .iter()
        .any(|cue| lower.contains(cue))
    {
        Some(Quantifier::Approximate)
    } else {
        Option::None
    }
}

/// Convenience wrapper: language detection + best-effort quantifier detection.
#[must_use]
pub fn detect_quantifier_auto(text: &str, entity_start: usize) -> Option<Quantifier> {
    detect_quantifier_lang(text, entity_start, detect_language_near(text, entity_start))
        .or_else(|| detect_approximate_quantifier(text, entity_start))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language_near_code_switching() {
        // Mixed-script text: ensure local windows determine language.
        let text = "This is English. 这是中文。Back to English.";

        // Near the start: Latin → English.
        assert_eq!(detect_language_near(text, 5), Language::English);

        // Near the Chinese segment: Han → Chinese.
        let chinese_start = "This is English. ".chars().count();
        assert_eq!(
            detect_language_near(text, chinese_start + 1),
            Language::Chinese
        );
    }

    #[test]
    fn test_is_negated_basic() {
        assert!(is_negated_en("He is not a doctor", 12)); // "doctor" at 12
        assert!(is_negated_en("I never saw John", 11)); // "John" at 11
        assert!(!is_negated_en("He is a doctor", 8)); // "doctor" at 8
        assert!(!is_negated_en("The quick brown fox", 4)); // "quick" at 4
    }

    #[test]
    fn test_is_negated_multilingual_examples() {
        // German: "Arzt" starts at 13 (character offsets)
        assert!(is_negated_lang("Er ist nicht Arzt", 13, Language::German));

        // French: "médecin" starts at 6
        assert!(is_negated_lang("pas médecin", 4, Language::French));

        // Spanish: "médico" starts at 6
        assert!(is_negated_lang("no médico", 3, Language::Spanish));

        // Chinese (substring cues): "医生" starts at 3
        assert!(is_negated_lang("他不是医生", 3, Language::Chinese));
    }

    #[test]
    fn test_detect_quantifier_basic() {
        assert_eq!(
            detect_quantifier_en("Every student passed", 6),
            Some(Quantifier::Universal)
        );
        assert_eq!(
            detect_quantifier_en("Some students failed", 5),
            Some(Quantifier::Existential)
        );
        assert_eq!(
            detect_quantifier_en("No student failed", 3),
            Some(Quantifier::None)
        );
        assert_eq!(
            detect_quantifier_en("The student passed", 4),
            Some(Quantifier::Definite)
        );
        assert_eq!(detect_quantifier_en("Student passed", 0), None);
    }

    #[test]
    fn test_detect_quantifier_with_cues_prefers_none_over_universal_on_overlap() {
        // If a cue appears in multiple sets, "none" must win to avoid accidental inflation
        // from broad cues like "any".
        let text = "y z";
        // "z" begins at char offset 2, so the immediately preceding token is "y".
        let q = detect_quantifier_with_cues(text, 2, &["y"], &[], &["y"], &[]);
        assert_eq!(q, Some(Quantifier::None));
    }

    #[test]
    fn test_detect_quantifier_auto_degrades_safely() {
        assert_eq!(
            detect_quantifier_lang("kein Arzt", 5, Language::German),
            Some(Quantifier::None)
        );
        assert_eq!(
            detect_quantifier_lang("aucun médecin", 6, Language::French),
            Some(Quantifier::None)
        );
        assert_eq!(
            detect_quantifier_lang("ningún médico", 6, Language::Spanish),
            Some(Quantifier::None)
        );
        assert_eq!(
            detect_quantifier_lang("每个 医生", 3, Language::Chinese),
            Some(Quantifier::Universal)
        );
    }

    // --- New tests below ---

    #[test]
    fn test_negation_contraction_forms() {
        // Contractions should match after lowercasing.
        assert!(is_negated_en("She doesn't like cats", 18)); // "cats" at 18
        assert!(is_negated_en("They won't attend meetings", 17)); // "meetings" at 17
        assert!(is_negated_en("He couldn't find keys", 16)); // "keys" at 16
    }

    #[test]
    fn test_negation_case_insensitive() {
        // Cue matching lowercases; "NOT" and "Never" should still match.
        assert!(is_negated_en("He is NOT a doctor", 12));
        assert!(is_negated_en("I Never saw John", 11));
    }

    #[test]
    fn test_negation_outside_three_word_window() {
        // The negation cue "not" is more than 3 words before the entity, so it should NOT match.
        let text = "not one of the many doctors";
        let entity_start = "not one of the many ".chars().count();
        assert!(!is_negated_en(text, entity_start));
    }

    #[test]
    fn test_negation_entity_at_start() {
        // Entity at offset 0: no prefix to scan, should be false.
        assert!(!is_negated_en("Doctor is here", 0));
    }

    #[test]
    fn test_negation_substring_chinese() {
        // "没有" (negation) immediately before entity.
        assert!(is_negated_with_substrings(
            "他没有钱",
            3,
            lexicons::ZH_NEGATION_CUES
        ));
        // No negation cue present.
        assert!(!is_negated_with_substrings(
            "他有钱",
            2,
            lexicons::ZH_NEGATION_CUES
        ));
    }

    #[test]
    fn test_quantifier_all_four_classes_en() {
        // Verify each quantifier class returns the correct variant.
        assert_eq!(
            detect_quantifier_en("all dogs", 4),
            Some(Quantifier::Universal)
        );
        assert_eq!(
            detect_quantifier_en("a dog", 2),
            Some(Quantifier::Existential)
        );
        assert_eq!(detect_quantifier_en("no dogs", 3), Some(Quantifier::None));
        assert_eq!(
            detect_quantifier_en("these dogs", 6),
            Some(Quantifier::Definite)
        );
    }

    #[test]
    fn test_quantifier_case_insensitive() {
        // Cue matching lowercases the last word, so "EVERY" should still match Universal.
        assert_eq!(
            detect_quantifier_en("EVERY student", 6),
            Some(Quantifier::Universal)
        );
        assert_eq!(
            detect_quantifier_en("The cat", 4),
            Some(Quantifier::Definite)
        );
    }

    #[test]
    fn test_quantifier_no_prefix_returns_none() {
        // When entity is at offset 0 there is no preceding token.
        assert_eq!(detect_quantifier_en("dogs run", 0), None);
    }

    #[test]
    fn test_quantifier_substring_japanese() {
        // Japanese uses substring matching. "全ての" contains "全て" (universal).
        assert_eq!(
            detect_quantifier_with_substrings(
                "全ての学生",
                3,
                lexicons::JA_UNIVERSAL,
                &[],
                &[],
                &[]
            ),
            Some(Quantifier::Universal)
        );
        // "この" (definite) before entity.
        assert_eq!(
            detect_quantifier_with_substrings("この本", 2, &[], &[], &[], lexicons::JA_DEFINITE),
            Some(Quantifier::Definite)
        );
    }

    #[test]
    fn test_detect_language_near_script_fast_path() {
        // Characters in distinctive scripts should trigger the fast path.
        let text = "Hello 你好世界 and more";
        let zh_offset = "Hello ".chars().count(); // points to '你'
        assert_eq!(detect_language_near(text, zh_offset), Language::Chinese);

        // Japanese hiragana
        let text2 = "abc あいう xyz";
        let ja_offset = "abc ".chars().count(); // points to 'あ'
        assert_eq!(detect_language_near(text2, ja_offset), Language::Japanese);

        // Korean Hangul
        let text3 = "abc 한국어 xyz";
        let ko_offset = "abc ".chars().count();
        assert_eq!(detect_language_near(text3, ko_offset), Language::Korean);
    }

    #[test]
    fn test_negation_lang_other_returns_false() {
        // Language::Other and Language::Arabic should conservatively return false.
        assert!(!is_negated_lang("no doctor", 3, Language::Other));
        assert!(!is_negated_lang("no doctor", 3, Language::Arabic));
    }

    #[test]
    fn test_quantifier_lang_other_returns_none() {
        // Language::Other should always return None.
        assert_eq!(
            detect_quantifier_lang("every dog", 6, Language::Other),
            None
        );
    }
}
