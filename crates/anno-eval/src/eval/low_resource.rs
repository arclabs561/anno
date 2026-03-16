//! Low-resource and morphologically complex language evaluation metrics.
//!
//! This module provides specialized evaluation tools for:
//! - Indigenous/Native American languages (Quechua, Cherokee, Navajo, etc.)
//! - Polysynthetic languages with complex morphology
//! - Languages with orthographic variation
//! - Low-resource scenarios with limited training data
//!
//! # Key Metrics
//!
//! - **Morpheme-level F1**: Evaluation at morpheme boundaries (important for polysynthetic languages)
//! - **Character-level F1**: Robust to tokenization differences
//! - **Normalized Entity Ratio**: Compares entity density across languages
//! - **Transfer Efficiency**: Measures how well high-resource models transfer
//! - **Orthographic Robustness**: Handles spelling variations
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::eval::low_resource::{LowResourceEvaluator, MorphemeConfig};
//!
//! let evaluator = LowResourceEvaluator::new()
//!     .with_morpheme_boundaries(true)
//!     .with_orthographic_normalization(true);
//!
//! let results = evaluator.evaluate(&model, &quechua_dataset)?;
//! println!("Morpheme F1: {:.3}", results.morpheme_f1);
//! println!("Transfer efficiency: {:.3}", results.transfer_efficiency);
//! ```
//!
//! # References
//!
//! - qxoRef: Galarreta et al., AmericasNLP 2021 (Quechua coreference)
//! - AmericasNLI: Ebrahimi et al., EMNLP 2022 (Indigenous NLI)
//! - CorefUD 1.3: Nedoluzhko et al., 2022 (Multilingual coreference)

use anno::{Error, Model, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for morpheme-level evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphemeConfig {
    /// Character used as morpheme boundary marker
    pub boundary_char: char,
    /// Whether to use character-level fallback when morpheme boundaries unavailable
    pub char_level_fallback: bool,
    /// Minimum morpheme length to count
    pub min_morpheme_len: usize,
}

impl Default for MorphemeConfig {
    fn default() -> Self {
        Self {
            boundary_char: '-',
            char_level_fallback: true,
            min_morpheme_len: 1,
        }
    }
}

/// Configuration for orthographic normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrthographicConfig {
    /// Enable Unicode normalization (NFC)
    pub unicode_normalize: bool,
    /// Case-insensitive matching
    pub case_insensitive: bool,
    /// Diacritic-insensitive matching
    pub ignore_diacritics: bool,
    /// Custom character mappings (e.g., for non-standard orthographies)
    pub char_mappings: HashMap<char, char>,
}

impl Default for OrthographicConfig {
    fn default() -> Self {
        Self {
            unicode_normalize: true,
            case_insensitive: false,
            ignore_diacritics: false,
            char_mappings: HashMap::new(),
        }
    }
}

/// Results from low-resource language evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowResourceResults {
    /// Standard token-level F1
    pub token_f1: f64,
    /// Morpheme-level F1 (for polysynthetic languages)
    pub morpheme_f1: Option<f64>,
    /// Character-level F1 (robust to tokenization)
    pub char_f1: f64,
    /// Entity density ratio compared to English baseline
    pub entity_density_ratio: f64,
    /// Transfer efficiency (F1 / English F1 baseline)
    pub transfer_efficiency: Option<f64>,
    /// Per-entity-type breakdown
    pub per_type: HashMap<String, TypeMetrics>,
    /// Orthographic normalization impact
    pub normalization_impact: Option<NormalizationImpact>,
    /// Language-specific metadata
    pub metadata: LowResourceMetadata,
}

/// Per-type metrics for low-resource evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeMetrics {
    /// Precision score (0-1)
    pub precision: f64,
    /// Recall score (0-1)
    pub recall: f64,
    /// F1 score (0-1)
    pub f1: f64,
    /// Number of examples for this type
    pub support: usize,
}

/// Impact of orthographic normalization on results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationImpact {
    /// F1 without normalization
    pub raw_f1: f64,
    /// F1 with normalization
    pub normalized_f1: f64,
    /// Improvement from normalization
    pub improvement: f64,
    /// Number of entities affected
    pub entities_affected: usize,
}

/// Metadata about the low-resource evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowResourceMetadata {
    /// ISO 639-3 language code
    pub language_code: String,
    /// Language family
    pub language_family: Option<String>,
    /// Whether language is polysynthetic
    pub is_polysynthetic: bool,
    /// Whether language has standardized orthography
    pub has_standard_orthography: bool,
    /// Estimated speaker population
    pub speaker_population: Option<u64>,
    /// UNESCO endangerment level
    pub endangerment_level: Option<EndangermentLevel>,
    /// Number of training examples available
    pub training_examples: Option<usize>,
}

/// UNESCO language endangerment levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndangermentLevel {
    /// Language is used by all ages
    Safe,
    /// Most children speak the language
    Vulnerable,
    /// Children speak at home but not in school
    DefinitelyEndangered,
    /// Only spoken by grandparents
    SeverelyEndangered,
    /// Only spoken by a few elderly
    CriticallyEndangered,
    /// No living speakers
    Extinct,
}

/// Evaluator for low-resource and morphologically complex languages.
pub struct LowResourceEvaluator {
    morpheme_config: Option<MorphemeConfig>,
    orthographic_config: Option<OrthographicConfig>,
    english_baseline_f1: Option<f64>,
}

impl LowResourceEvaluator {
    /// Create a new low-resource evaluator with default settings.
    pub fn new() -> Self {
        Self {
            morpheme_config: None,
            orthographic_config: None,
            english_baseline_f1: None,
        }
    }

    /// Enable morpheme-level evaluation.
    pub fn with_morpheme_boundaries(mut self, config: MorphemeConfig) -> Self {
        self.morpheme_config = Some(config);
        self
    }

    /// Enable orthographic normalization.
    pub fn with_orthographic_normalization(mut self, config: OrthographicConfig) -> Self {
        self.orthographic_config = Some(config);
        self
    }

    /// Set English baseline F1 for transfer efficiency calculation.
    pub fn with_english_baseline(mut self, f1: f64) -> Self {
        self.english_baseline_f1 = Some(f1);
        self
    }

    /// Evaluate model on low-resource dataset.
    pub fn evaluate(
        &self,
        model: &dyn Model,
        test_cases: &[(String, Vec<super::GoldEntity>)],
        metadata: LowResourceMetadata,
    ) -> Result<LowResourceResults> {
        if test_cases.is_empty() {
            return Err(Error::InvalidInput("Empty test cases".to_string()));
        }

        // Calculate standard token-level metrics
        let standard_results = super::evaluate_ner_model(model, test_cases)?;

        // Calculate character-level F1
        let char_f1 = self.calculate_char_f1(model, test_cases)?;

        // Calculate morpheme-level F1 if configured
        let morpheme_f1 = if self.morpheme_config.is_some() && metadata.is_polysynthetic {
            Some(self.calculate_morpheme_f1(model, test_cases)?)
        } else {
            None
        };

        // Calculate entity density ratio
        let total_chars: usize = test_cases.iter().map(|(text, _)| text.len()).sum();
        let total_entities: usize = test_cases.iter().map(|(_, entities)| entities.len()).sum();
        let entity_density = if total_chars > 0 {
            total_entities as f64 / total_chars as f64
        } else {
            0.0
        };
        // English baseline entity density (approximate from CoNLL-2003)
        let english_baseline_density = 0.05;
        let entity_density_ratio = entity_density / english_baseline_density;

        // Calculate transfer efficiency
        let transfer_efficiency = self
            .english_baseline_f1
            .map(|baseline| standard_results.f1 / baseline);

        // Calculate normalization impact if configured
        let normalization_impact = if self.orthographic_config.is_some() {
            Some(self.calculate_normalization_impact(model, test_cases)?)
        } else {
            None
        };

        // Convert per-type metrics
        let per_type: HashMap<String, TypeMetrics> = standard_results
            .per_type
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    TypeMetrics {
                        precision: v.precision,
                        recall: v.recall,
                        f1: v.f1,
                        support: v.expected,
                    },
                )
            })
            .collect();

        Ok(LowResourceResults {
            token_f1: standard_results.f1,
            morpheme_f1,
            char_f1,
            entity_density_ratio,
            transfer_efficiency,
            per_type,
            normalization_impact,
            metadata,
        })
    }

    /// Calculate character-level F1.
    ///
    /// This is more robust to tokenization differences across languages.
    fn calculate_char_f1(
        &self,
        model: &dyn Model,
        test_cases: &[(String, Vec<super::GoldEntity>)],
    ) -> Result<f64> {
        let mut total_gold_chars = 0;
        let mut total_pred_chars = 0;
        let mut total_correct_chars = 0;

        for (text, gold_entities) in test_cases {
            // Get predictions
            let predictions = model.extract_entities(text, None)?;

            // Create character-level gold mask
            let text_char_len = text.chars().count();
            let mut gold_mask = vec![false; text_char_len];
            for entity in gold_entities {
                let start = entity.start.min(text_char_len);
                let end = entity.end.min(text_char_len);
                for slot in gold_mask.iter_mut().take(end).skip(start) {
                    *slot = true;
                }
            }

            // Create character-level prediction mask
            let mut pred_mask = vec![false; text_char_len];
            for entity in &predictions {
                let start = entity.start().min(text_char_len);
                let end = entity.end().min(text_char_len);
                for slot in pred_mask.iter_mut().take(end).skip(start) {
                    *slot = true;
                }
            }

            // Count matches
            for i in 0..text_char_len {
                if gold_mask[i] {
                    total_gold_chars += 1;
                }
                if pred_mask[i] {
                    total_pred_chars += 1;
                }
                if gold_mask[i] && pred_mask[i] {
                    total_correct_chars += 1;
                }
            }
        }

        let precision = if total_pred_chars > 0 {
            total_correct_chars as f64 / total_pred_chars as f64
        } else {
            0.0
        };
        let recall = if total_gold_chars > 0 {
            total_correct_chars as f64 / total_gold_chars as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Ok(f1)
    }

    /// Calculate morpheme-level F1 for polysynthetic languages.
    fn calculate_morpheme_f1(
        &self,
        model: &dyn Model,
        test_cases: &[(String, Vec<super::GoldEntity>)],
    ) -> Result<f64> {
        let config = self.morpheme_config.as_ref().ok_or_else(|| {
            Error::evaluation(
                "morpheme-level evaluation requested without MorphemeConfig (call with_morpheme_boundaries(true))",
            )
        })?;

        let mut total_gold_morphemes = 0;
        let mut total_pred_morphemes = 0;
        let mut total_correct_morphemes = 0;

        for (text, gold_entities) in test_cases {
            let predictions = model.extract_entities(text, None)?;

            // Count morphemes in gold entities
            for entity in gold_entities {
                let morpheme_count = entity
                    .text
                    .split(config.boundary_char)
                    .filter(|m| m.len() >= config.min_morpheme_len)
                    .count()
                    .max(1);
                total_gold_morphemes += morpheme_count;
            }

            // Count morphemes in predicted entities
            for entity in &predictions {
                // Note: entity.start/end are CHARACTER offsets, not byte offsets
                let char_count = text.chars().count();
                let entity_text: String = text
                    .chars()
                    .skip(entity.start())
                    .take(entity.end().min(char_count).saturating_sub(entity.start()))
                    .collect();
                let morpheme_count = entity_text
                    .split(config.boundary_char)
                    .filter(|m| m.len() >= config.min_morpheme_len)
                    .count()
                    .max(1);
                total_pred_morphemes += morpheme_count;

                // Check if this prediction matches a gold entity
                for gold in gold_entities {
                    if entity.start() == gold.start && entity.end() == gold.end {
                        total_correct_morphemes += morpheme_count;
                        break;
                    }
                }
            }
        }

        let precision = if total_pred_morphemes > 0 {
            total_correct_morphemes as f64 / total_pred_morphemes as f64
        } else {
            0.0
        };
        let recall = if total_gold_morphemes > 0 {
            total_correct_morphemes as f64 / total_gold_morphemes as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Ok(f1)
    }

    /// Calculate impact of orthographic normalization.
    fn calculate_normalization_impact(
        &self,
        model: &dyn Model,
        test_cases: &[(String, Vec<super::GoldEntity>)],
    ) -> Result<NormalizationImpact> {
        let config = self.orthographic_config.as_ref().ok_or_else(|| {
            Error::evaluation(
                "normalization impact requested without OrthographicConfig (call with_orthographic_normalization(true))",
            )
        })?;

        // Evaluate without normalization
        let raw_results = super::evaluate_ner_model(model, test_cases)?;

        // Apply normalization to test cases
        let normalized_cases: Vec<(String, Vec<super::GoldEntity>)> = test_cases
            .iter()
            .map(|(text, entities)| {
                let normalized_text = self.normalize_text(text, config);
                let normalized_entities: Vec<super::GoldEntity> = entities
                    .iter()
                    .map(|e| super::GoldEntity {
                        text: self.normalize_text(&e.text, config),
                        entity_type: e.entity_type.clone(),
                        original_label: e.original_label.clone(),
                        start: e.start,
                        end: e.end,
                    })
                    .collect();
                (normalized_text, normalized_entities)
            })
            .collect();

        // Evaluate with normalization
        let normalized_results = super::evaluate_ner_model(model, &normalized_cases)?;

        // Count affected entities
        let mut entities_affected = 0;
        for ((orig_text, _), (norm_text, _)) in test_cases.iter().zip(normalized_cases.iter()) {
            if orig_text != norm_text {
                entities_affected += 1;
            }
        }

        Ok(NormalizationImpact {
            raw_f1: raw_results.f1,
            normalized_f1: normalized_results.f1,
            improvement: normalized_results.f1 - raw_results.f1,
            entities_affected,
        })
    }

    /// Apply orthographic normalization to text.
    fn normalize_text(&self, text: &str, config: &OrthographicConfig) -> String {
        let mut result = text.to_string();

        // Unicode normalization (NFC)
        if config.unicode_normalize {
            use unicode_normalization::UnicodeNormalization;
            result = result.nfc().collect();
        }

        // Case normalization
        if config.case_insensitive {
            result = result.to_lowercase();
        }

        // Diacritic removal
        if config.ignore_diacritics {
            result = remove_diacritics(&result);
        }

        // Custom character mappings
        for (from, to) in &config.char_mappings {
            result = result.replace(*from, &to.to_string());
        }

        result
    }
}

impl Default for LowResourceEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Remove diacritics from text.
fn remove_diacritics(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.nfd()
        .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
        .collect()
}

/// Create metadata for common Indigenous American languages.
pub fn language_metadata(language_code: &str) -> Option<LowResourceMetadata> {
    match language_code {
        // Quechua (Conchucos dialect - qxoRef)
        "qxo" => Some(LowResourceMetadata {
            language_code: "qxo".to_string(),
            language_family: Some("Quechuan".to_string()),
            is_polysynthetic: false, // Quechua is agglutinative, not polysynthetic
            has_standard_orthography: false,
            speaker_population: Some(200_000),
            endangerment_level: Some(EndangermentLevel::Vulnerable),
            training_examples: Some(12), // qxoRef has 12 documents
        }),
        // Cherokee
        "chr" => Some(LowResourceMetadata {
            language_code: "chr".to_string(),
            language_family: Some("Iroquoian".to_string()),
            is_polysynthetic: true,
            has_standard_orthography: true, // Cherokee syllabary is standardized
            speaker_population: Some(2_000),
            endangerment_level: Some(EndangermentLevel::SeverelyEndangered),
            training_examples: None,
        }),
        // Navajo
        "nav" => Some(LowResourceMetadata {
            language_code: "nav".to_string(),
            language_family: Some("Na-Dené".to_string()),
            is_polysynthetic: true,
            has_standard_orthography: true,
            speaker_population: Some(170_000),
            endangerment_level: Some(EndangermentLevel::Vulnerable),
            training_examples: None,
        }),
        // Guarani
        "gn" | "grn" => Some(LowResourceMetadata {
            language_code: "grn".to_string(),
            language_family: Some("Tupian".to_string()),
            is_polysynthetic: false,
            has_standard_orthography: true,
            speaker_population: Some(6_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: None,
        }),
        // Nahuatl
        "nah" => Some(LowResourceMetadata {
            language_code: "nah".to_string(),
            language_family: Some("Uto-Aztecan".to_string()),
            is_polysynthetic: true,
            has_standard_orthography: false, // Multiple competing orthographies
            speaker_population: Some(1_700_000),
            endangerment_level: Some(EndangermentLevel::Vulnerable),
            training_examples: None,
        }),
        // Shipibo-Konibo
        "shp" => Some(LowResourceMetadata {
            language_code: "shp".to_string(),
            language_family: Some("Panoan".to_string()),
            is_polysynthetic: false,
            has_standard_orthography: true,
            speaker_population: Some(35_000),
            endangerment_level: Some(EndangermentLevel::Vulnerable),
            training_examples: None,
        }),

        // =========================================================================
        // African Languages (MasakhaNER 2.0 languages)
        // =========================================================================

        // Swahili (Kiswahili) - East Africa's lingua franca
        "sw" | "swa" => Some(LowResourceMetadata {
            language_code: "swa".to_string(),
            language_family: Some("Atlantic-Congo (Bantu)".to_string()),
            is_polysynthetic: false, // Agglutinative, not polysynthetic
            has_standard_orthography: true,
            speaker_population: Some(100_000_000), // Including L2 speakers
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(9_418), // MasakhaNER 2.0 train+dev+test
        }),

        // Yoruba - Tonal language with diacritics
        "yo" | "yor" => Some(LowResourceMetadata {
            language_code: "yor".to_string(),
            language_family: Some("Atlantic-Congo (Volta-Niger)".to_string()),
            is_polysynthetic: false,
            has_standard_orthography: true, // Standard with tone marks
            speaker_population: Some(45_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(9_824), // MasakhaNER 2.0
        }),

        // Hausa - Major West African trade language
        "ha" | "hau" => Some(LowResourceMetadata {
            language_code: "hau".to_string(),
            language_family: Some("Afro-Asiatic (Chadic)".to_string()),
            is_polysynthetic: false,
            has_standard_orthography: true, // Latin (Boko) standard
            speaker_population: Some(80_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(8_165), // MasakhaNER 2.0
        }),

        // Amharic - Ethiopian Semitic with Ge'ez script
        "am" | "amh" => Some(LowResourceMetadata {
            language_code: "amh".to_string(),
            language_family: Some("Afro-Asiatic (Semitic)".to_string()),
            is_polysynthetic: false,
            has_standard_orthography: true, // Ge'ez (Ethiopic) script
            speaker_population: Some(57_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(1_750), // MasakhaNER 1.0
        }),

        // Igbo - Tonal language of Nigeria
        "ig" | "ibo" => Some(LowResourceMetadata {
            language_code: "ibo".to_string(),
            language_family: Some("Atlantic-Congo (Volta-Niger)".to_string()),
            is_polysynthetic: false,
            has_standard_orthography: true, // Önwu alphabet
            speaker_population: Some(45_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(10_905), // MasakhaNER 2.0
        }),

        // Kinyarwanda - Rwanda/Burundi
        "rw" | "kin" => Some(LowResourceMetadata {
            language_code: "kin".to_string(),
            language_family: Some("Atlantic-Congo (Bantu)".to_string()),
            is_polysynthetic: false, // Agglutinative
            has_standard_orthography: true,
            speaker_population: Some(12_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(11_178), // MasakhaNER 2.0
        }),

        // Nigerian Pidgin - English-based creole
        "pcm" => Some(LowResourceMetadata {
            language_code: "pcm".to_string(),
            language_family: Some("English Creole".to_string()),
            is_polysynthetic: false,
            has_standard_orthography: false, // No standardized spelling
            speaker_population: Some(100_000_000), // L2 speakers
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(7_746), // MasakhaNER 2.0
        }),

        // Wolof - Senegal/Gambia
        "wo" | "wol" => Some(LowResourceMetadata {
            language_code: "wol".to_string(),
            language_family: Some("Atlantic-Congo (Atlantic)".to_string()),
            is_polysynthetic: false,
            has_standard_orthography: true,
            speaker_population: Some(12_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(6_561), // MasakhaNER 2.0
        }),

        // Zulu (isiZulu) - South Africa
        "zu" | "zul" => Some(LowResourceMetadata {
            language_code: "zul".to_string(),
            language_family: Some("Atlantic-Congo (Bantu/Nguni)".to_string()),
            is_polysynthetic: false, // Agglutinative
            has_standard_orthography: true,
            speaker_population: Some(27_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(8_354), // MasakhaNER 2.0
        }),

        // Xhosa (isiXhosa) - South Africa, with clicks
        "xh" | "xho" => Some(LowResourceMetadata {
            language_code: "xho".to_string(),
            language_family: Some("Atlantic-Congo (Bantu/Nguni)".to_string()),
            is_polysynthetic: false, // Agglutinative
            has_standard_orthography: true,
            speaker_population: Some(19_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(8_168), // MasakhaNER 2.0
        }),

        // Luganda - Uganda
        "lg" | "lug" => Some(LowResourceMetadata {
            language_code: "lug".to_string(),
            language_family: Some("Atlantic-Congo (Bantu)".to_string()),
            is_polysynthetic: false, // Agglutinative
            has_standard_orthography: true,
            speaker_population: Some(10_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(7_060), // MasakhaNER 2.0
        }),

        // Luo (Dholuo) - Kenya/Tanzania
        "luo" => Some(LowResourceMetadata {
            language_code: "luo".to_string(),
            language_family: Some("Nilo-Saharan (Nilotic)".to_string()),
            is_polysynthetic: false,
            has_standard_orthography: true,
            speaker_population: Some(6_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(7_372), // MasakhaNER 2.0
        }),

        // Twi (Akan) - Ghana
        "tw" | "twi" | "aka" => Some(LowResourceMetadata {
            language_code: "twi".to_string(),
            language_family: Some("Atlantic-Congo (Kwa)".to_string()),
            is_polysynthetic: false,
            has_standard_orthography: true, // Tonal diacritics
            speaker_population: Some(11_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(6_056), // MasakhaNER 2.0
        }),

        // Shona (chiShona) - Zimbabwe
        "sn" | "sna" => Some(LowResourceMetadata {
            language_code: "sna".to_string(),
            language_family: Some("Atlantic-Congo (Bantu)".to_string()),
            is_polysynthetic: false, // Agglutinative
            has_standard_orthography: true,
            speaker_population: Some(15_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(8_867), // MasakhaNER 2.0
        }),

        // Tigrinya - Eritrea/Ethiopia, Ge'ez script
        "ti" | "tir" => Some(LowResourceMetadata {
            language_code: "tir".to_string(),
            language_family: Some("Afro-Asiatic (Semitic)".to_string()),
            is_polysynthetic: false,
            has_standard_orthography: true, // Ge'ez script
            speaker_population: Some(9_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: None, // Not in MasakhaNER, but in AfriSenti
        }),

        // Bambara - Mali
        "bm" | "bam" => Some(LowResourceMetadata {
            language_code: "bam".to_string(),
            language_family: Some("Atlantic-Congo (Mande)".to_string()),
            is_polysynthetic: false,
            has_standard_orthography: true, // N'Ko or Latin
            speaker_population: Some(14_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(6_375), // MasakhaNER 2.0
        }),

        // Ewe - Ghana/Togo
        "ee" | "ewe" => Some(LowResourceMetadata {
            language_code: "ewe".to_string(),
            language_family: Some("Atlantic-Congo (Kwa)".to_string()),
            is_polysynthetic: false,
            has_standard_orthography: true, // Tonal diacritics
            speaker_population: Some(7_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(5_007), // MasakhaNER 2.0
        }),

        // Fon - Benin
        "fon" => Some(LowResourceMetadata {
            language_code: "fon".to_string(),
            language_family: Some("Atlantic-Congo (Kwa)".to_string()),
            is_polysynthetic: false,
            has_standard_orthography: true, // Tonal diacritics
            speaker_population: Some(2_200_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(6_204), // MasakhaNER 2.0
        }),

        // Mossi (Mooré) - Burkina Faso
        "mos" => Some(LowResourceMetadata {
            language_code: "mos".to_string(),
            language_family: Some("Atlantic-Congo (Gur)".to_string()),
            is_polysynthetic: false,
            has_standard_orthography: true,
            speaker_population: Some(8_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(6_793), // MasakhaNER 2.0
        }),

        // Setswana - Botswana/South Africa
        "tn" | "tsn" => Some(LowResourceMetadata {
            language_code: "tsn".to_string(),
            language_family: Some("Atlantic-Congo (Bantu)".to_string()),
            is_polysynthetic: false, // Agglutinative
            has_standard_orthography: true,
            speaker_population: Some(8_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(4_784), // MasakhaNER 2.0
        }),

        // Chichewa (Nyanja) - Malawi
        "ny" | "nya" => Some(LowResourceMetadata {
            language_code: "nya".to_string(),
            language_family: Some("Atlantic-Congo (Bantu)".to_string()),
            is_polysynthetic: false, // Agglutinative
            has_standard_orthography: true,
            speaker_population: Some(15_000_000),
            endangerment_level: Some(EndangermentLevel::Safe),
            training_examples: Some(8_928), // MasakhaNER 2.0
        }),

        // Ghomala - Cameroon (lower resource)
        "bbj" => Some(LowResourceMetadata {
            language_code: "bbj".to_string(),
            language_family: Some("Atlantic-Congo (Grassfields Bantu)".to_string()),
            is_polysynthetic: false,
            has_standard_orthography: false, // Developing
            speaker_population: Some(1_000_000),
            endangerment_level: Some(EndangermentLevel::Vulnerable),
            training_examples: Some(4_833), // MasakhaNER 2.0
        }),

        _ => None,
    }
}

/// MasakhaNER 2.0 language codes.
///
/// These codes can be used to load specific language splits from MasakhaNER 2.0.
/// Example: `load_dataset("masakhane/masakhaner2", "yor")` for Yoruba.
pub const MASAKHANER2_LANGUAGES: &[(&str, &str)] = &[
    ("bam", "Bambara"),
    ("bbj", "Ghomala"),
    ("ewe", "Ewe"),
    ("fon", "Fon"),
    ("hau", "Hausa"),
    ("ibo", "Igbo"),
    ("kin", "Kinyarwanda"),
    ("lug", "Luganda"),
    ("luo", "Dholuo"),
    ("mos", "Mossi"),
    ("nya", "Chichewa"),
    ("pcm", "Nigerian Pidgin"),
    ("sna", "Shona"),
    ("swa", "Swahili"),
    ("tsn", "Setswana"),
    ("twi", "Twi"),
    ("wol", "Wolof"),
    ("xho", "Xhosa"),
    ("yor", "Yoruba"),
    ("zul", "Zulu"),
];

/// Get language name from MasakhaNER 2.0 code.
pub fn masakhaner2_language_name(code: &str) -> Option<&'static str> {
    MASAKHANER2_LANGUAGES
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, name)| *name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_metadata() {
        let quechua = language_metadata("qxo").unwrap();
        assert_eq!(quechua.language_family, Some("Quechuan".to_string()));
        assert!(!quechua.is_polysynthetic);

        let cherokee = language_metadata("chr").unwrap();
        assert!(cherokee.is_polysynthetic);
        assert_eq!(
            cherokee.endangerment_level,
            Some(EndangermentLevel::SeverelyEndangered)
        );
    }

    #[test]
    fn test_orthographic_normalization() {
        let config = OrthographicConfig {
            unicode_normalize: true,
            case_insensitive: true,
            ignore_diacritics: true,
            char_mappings: HashMap::new(),
        };

        let evaluator = LowResourceEvaluator::new();
        let normalized = evaluator.normalize_text("Café", &config);
        assert_eq!(normalized, "cafe");
    }

    #[test]
    fn test_evaluator_creation() {
        let evaluator = LowResourceEvaluator::new()
            .with_morpheme_boundaries(MorphemeConfig::default())
            .with_orthographic_normalization(OrthographicConfig::default())
            .with_english_baseline(0.92);

        assert!(evaluator.morpheme_config.is_some());
        assert!(evaluator.orthographic_config.is_some());
        assert_eq!(evaluator.english_baseline_f1, Some(0.92));
    }
}
