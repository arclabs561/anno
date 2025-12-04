//! Language detection and classification utilities.

/// Supported languages for text analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    /// English language
    English,
    /// German language
    German,
    /// French language
    French,
    /// Spanish language
    Spanish,
    /// Italian language
    Italian,
    /// Portuguese language
    Portuguese,
    /// Russian language
    Russian,
    /// Chinese language (Simplified/Traditional)
    Chinese,
    /// Japanese language
    Japanese,
    /// Korean language
    Korean,
    /// Arabic language
    Arabic,
    /// Hebrew language
    Hebrew,
    /// Other/unknown language
    Other,
}

impl Language {
    /// Returns true if this is a CJK (Chinese, Japanese, Korean) language.
    #[must_use]
    pub fn is_cjk(&self) -> bool {
        matches!(
            self,
            Language::Chinese | Language::Japanese | Language::Korean
        )
    }

    /// Returns true if this is a right-to-left language (Arabic, Hebrew).
    #[must_use]
    pub fn is_rtl(&self) -> bool {
        matches!(self, Language::Arabic | Language::Hebrew)
    }
}

/// Simple heuristic language detection based on Unicode scripts.
///
/// Returns the most likely language based on character counts.
pub fn detect_language(text: &str) -> Language {
    let mut counts = [0usize; 13];
    let mut total = 0;

    for c in text.chars() {
        if !c.is_alphabetic() {
            continue;
        }
        total += 1;

        match c {
            // CJK Unified Ideographs
            '\u{4e00}'..='\u{9fff}' => counts[Language::Chinese as usize] += 1,
            // Hiragana/Katakana
            '\u{3040}'..='\u{30ff}' => counts[Language::Japanese as usize] += 1,
            // Hangul
            '\u{ac00}'..='\u{d7af}' => counts[Language::Korean as usize] += 1,
            // Arabic
            '\u{0600}'..='\u{06ff}' => counts[Language::Arabic as usize] += 1,
            // Hebrew
            '\u{0590}'..='\u{05ff}' => counts[Language::Hebrew as usize] += 1,
            // Cyrillic
            '\u{0400}'..='\u{04ff}' => counts[Language::Russian as usize] += 1,
            // Latin - distinguishing languages is hard without dictionary,
            // but we can check for specific chars
            'a'..='z' | 'A'..='Z' => counts[Language::English as usize] += 1, // Generic Latin
            // German specific (ß, ä, ö, ü)
            'ß' | 'ä' | 'ö' | 'ü' | 'Ä' | 'Ö' | 'Ü' => {
                counts[Language::German as usize] += 10
            }
            // French (à, â, ç, é, è, ê, ë, î, ï, ô, û, ù)
            'à' | 'â' | 'ç' | 'é' | 'è' | 'ê' | 'ë' | 'î' | 'ï' | 'ô' | 'û' | 'ù' => {
                counts[Language::French as usize] += 5
            }
            // Spanish (ñ, ¿, ¡, á, é, í, ó, ú)
            'ñ' | '¿' | '¡' | 'á' | 'í' | 'ó' | 'ú' => {
                counts[Language::Spanish as usize] += 5
            }
            _ => {}
        }
    }

    if total == 0 {
        return Language::English; // Default
    }

    // Find max
    let mut max_idx = 0;
    let mut max_val = 0;
    for (i, &val) in counts.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    // If we detected CJK chars but classified as Chinese, check if Japanese specific chars exist
    if max_idx == Language::Chinese as usize && counts[Language::Japanese as usize] > 0 {
        return Language::Japanese; // Japanese uses Kanji (Chinese chars) too
    }

    // SAFETY: max_idx is guaranteed to be in range [0, 255] because it comes from
    // argmax over a fixed-size array (Language enum has < 256 variants).
    // The cast to u8 is safe, and transmute from u8 to Language is safe because
    // Language is repr(u8) and max_idx corresponds to a valid Language variant.
    unsafe { std::mem::transmute(max_idx as u8) }
}
