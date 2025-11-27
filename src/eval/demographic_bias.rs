//! Demographic bias evaluation for Named Entity Recognition.
//!
//! Measures systematic disparities in NER performance across demographic groups,
//! including ethnicity, region, name script, and intersectional categories.
//!
//! # Research Background
//!
//! - Mishra et al. (2020): "Assessing Demographic Bias in NER" - NER models perform
//!   better on names from specific demographic groups
//! - Mansfield et al. (2022): "Behind the Mask" - PII masking systems miss
//!   non-Western names more often
//! - Loessberg-Zahl (2024): "Multicultural Name Recognition" - NER struggles with
//!   names not seen in training data
//! - Li et al. (2022): "HERB" - Regional bias in language models
//!
//! # Critical Finding: Character-Based Models Are Less Biased
//!
//! Mishra et al. (2020) found that:
//! - **Debiased word embeddings do NOT help** resolve NER demographic bias
//! - **Character-based models (ELMo-style) show the least bias** across demographics
//! - This suggests subword/character representations better generalize to unseen names
//!
//! Implications for model selection:
//! - Prefer character-level or subword models over word-level models for fair NER
//! - ELMo, BERT (with WordPiece), and similar subword models are better choices
//! - Pure word2vec or GloVe-based models will exhibit more demographic bias
//!
//! # Key Metrics
//!
//! - **Recognition Rate**: % of names correctly identified as PERSON
//! - **Demographic Parity**: Max gap in recognition rates across groups
//! - **Script Bias**: Performance difference for non-Latin scripts
//!
//! # Example
//!
//! ```rust
//! use anno::eval::demographic_bias::{DemographicBiasEvaluator, create_diverse_name_dataset};
//! use anno::PatternNER;
//!
//! let names = create_diverse_name_dataset();
//! let evaluator = DemographicBiasEvaluator::default();
//! // let results = evaluator.evaluate_ner(&PatternNER::new(), &names);
//! ```

use crate::{EntityType, Model};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Demographic Categories
// =============================================================================

/// Ethnicity/origin category for name classification.
///
/// Based on US Census categories and extended for global coverage.
/// These are used for MEASUREMENT only - to detect bias, not to make assumptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ethnicity {
    /// European/Caucasian names (Smith, Johnson, Mueller)
    European,
    /// African-American names (DeShawn, Latoya, Jamal)
    AfricanAmerican,
    /// Hispanic/Latino names (García, Rodriguez, Martinez)
    Hispanic,
    /// East Asian names (Wang, Kim, Tanaka)
    EastAsian,
    /// South Asian names (Patel, Singh, Kumar)
    SouthAsian,
    /// Middle Eastern/North African names (Ahmed, Fatima, Hassan)
    MiddleEastern,
    /// African names (Okonkwo, Adebayo, Mensah)
    African,
    /// Indigenous/Native names (various origins)
    Indigenous,
}

/// Geographic region for location/organization bias testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Region {
    /// North America (US, Canada, Mexico)
    NorthAmerica,
    /// Western Europe (UK, France, Germany, etc.)
    WesternEurope,
    /// Eastern Europe (Russia, Poland, Ukraine, etc.)
    EasternEurope,
    /// East Asia (China, Japan, Korea)
    EastAsia,
    /// South Asia (India, Pakistan, Bangladesh)
    SouthAsia,
    /// Southeast Asia (Vietnam, Thailand, Indonesia)
    SoutheastAsia,
    /// Middle East (Saudi Arabia, Iran, UAE)
    MiddleEast,
    /// Africa (Nigeria, Kenya, South Africa)
    Africa,
    /// Latin America (Brazil, Argentina, Colombia)
    LatinAmerica,
    /// Oceania (Australia, New Zealand)
    Oceania,
}

/// Script type for text encoding bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Script {
    /// Latin alphabet (English, Spanish, French)
    Latin,
    /// Cyrillic (Russian, Ukrainian, Serbian)
    Cyrillic,
    /// Arabic script
    Arabic,
    /// Chinese characters (Hanzi/Kanji)
    Chinese,
    /// Japanese (Hiragana, Katakana, Kanji mix)
    Japanese,
    /// Korean (Hangul)
    Korean,
    /// Devanagari (Hindi, Sanskrit)
    Devanagari,
    /// Thai script
    Thai,
    /// Greek alphabet
    Greek,
    /// Hebrew script
    Hebrew,
}

// =============================================================================
// Name Example
// =============================================================================

/// A name example with demographic metadata for bias testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NameExample {
    /// The full name text
    pub name: String,
    /// First name only (for partial matching tests)
    pub first_name: String,
    /// Last name only
    pub last_name: String,
    /// Ethnicity/origin category
    pub ethnicity: Ethnicity,
    /// Primary script used
    pub script: Script,
    /// Gender if known (for intersectional analysis)
    pub gender: Option<Gender>,
    /// Whether this is a common or rare name
    pub frequency: NameFrequency,
}

/// Gender for intersectional analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Gender {
    /// Typically masculine names
    Masculine,
    /// Typically feminine names
    Feminine,
    /// Gender-neutral names
    Neutral,
}

/// Name frequency category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NameFrequency {
    /// Very common (top 100 in origin country)
    Common,
    /// Moderately common
    Moderate,
    /// Rare or unusual
    Rare,
}

impl NameExample {
    /// Create a new name example.
    pub fn new(
        first_name: &str,
        last_name: &str,
        ethnicity: Ethnicity,
        script: Script,
        gender: Option<Gender>,
        frequency: NameFrequency,
    ) -> Self {
        Self {
            name: format!("{} {}", first_name, last_name),
            first_name: first_name.to_string(),
            last_name: last_name.to_string(),
            ethnicity,
            script,
            gender,
            frequency,
        }
    }
}

// =============================================================================
// Location Example
// =============================================================================

/// A location example with regional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationExample {
    /// Location name
    pub name: String,
    /// Geographic region
    pub region: Region,
    /// Primary script used
    pub script: Script,
    /// Location type (city, country, landmark)
    pub location_type: LocationType,
}

/// Type of location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LocationType {
    /// Major city
    City,
    /// Country name
    Country,
    /// State/Province/Region
    SubnationalRegion,
    /// Landmark or geographic feature
    Landmark,
}

impl LocationExample {
    /// Create a new location example.
    pub fn new(name: &str, region: Region, script: Script, location_type: LocationType) -> Self {
        Self {
            name: name.to_string(),
            region,
            script,
            location_type,
        }
    }
}

// =============================================================================
// Evaluation Results
// =============================================================================

/// Results of demographic bias evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemographicBiasResults {
    /// Overall recognition rate across all names
    pub overall_recognition_rate: f64,
    /// Recognition rate by ethnicity
    pub by_ethnicity: HashMap<String, f64>,
    /// Recognition rate by script
    pub by_script: HashMap<String, f64>,
    /// Recognition rate by gender (intersectional)
    pub by_gender: HashMap<String, f64>,
    /// Recognition rate by frequency
    pub by_frequency: HashMap<String, f64>,
    /// Maximum gap between any two ethnicity groups
    pub ethnicity_parity_gap: f64,
    /// Maximum gap between Latin and non-Latin scripts
    pub script_bias_gap: f64,
    /// Intersectional analysis: ethnicity × gender
    pub intersectional: HashMap<String, f64>,
    /// Number of names tested
    pub total_tested: usize,
    /// Detailed per-name results
    pub detailed: Vec<NameResult>,
}

/// Result for a single name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NameResult {
    /// The name tested
    pub name: String,
    /// Whether it was recognized as PERSON
    pub recognized: bool,
    /// Confidence if recognized
    pub confidence: Option<f64>,
    /// Ethnicity category
    pub ethnicity: String,
    /// Script used
    pub script: String,
}

/// Results of regional bias evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalBiasResults {
    /// Overall recognition rate
    pub overall_recognition_rate: f64,
    /// Recognition rate by region
    pub by_region: HashMap<String, f64>,
    /// Recognition rate by script
    pub by_script: HashMap<String, f64>,
    /// Maximum gap between regions
    pub regional_parity_gap: f64,
    /// Number of locations tested
    pub total_tested: usize,
}

// =============================================================================
// Evaluator
// =============================================================================

/// Evaluator for demographic bias in NER systems.
#[derive(Debug, Clone, Default)]
pub struct DemographicBiasEvaluator {
    /// Include detailed per-name results
    pub detailed: bool,
}

impl DemographicBiasEvaluator {
    /// Create a new evaluator.
    pub fn new(detailed: bool) -> Self {
        Self { detailed }
    }

    /// Evaluate NER model for demographic bias on names.
    pub fn evaluate_ner(&self, model: &dyn Model, names: &[NameExample]) -> DemographicBiasResults {
        let mut by_ethnicity: HashMap<String, (usize, usize)> = HashMap::new();
        let mut by_script: HashMap<String, (usize, usize)> = HashMap::new();
        let mut by_gender: HashMap<String, (usize, usize)> = HashMap::new();
        let mut by_frequency: HashMap<String, (usize, usize)> = HashMap::new();
        let mut intersectional: HashMap<String, (usize, usize)> = HashMap::new();
        let mut detailed_results = Vec::new();
        let mut total_recognized = 0;

        for name_example in names {
            // Create test sentence
            let text = format!("{} attended the conference.", name_example.name);
            
            // Extract entities
            let entities = model.extract_entities(&text, None).unwrap_or_default();
            
            // Check if name was recognized as PERSON
            let recognized = entities.iter().any(|e| {
                e.entity_type == EntityType::Person && 
                text[e.start..e.end].contains(&name_example.first_name)
            });
            
            let confidence = if recognized {
                entities.iter()
                    .find(|e| e.entity_type == EntityType::Person)
                    .map(|e| e.confidence)
            } else {
                None
            };

            if recognized {
                total_recognized += 1;
            }

            // Update ethnicity stats
            let eth_key = format!("{:?}", name_example.ethnicity);
            let eth_entry = by_ethnicity.entry(eth_key.clone()).or_insert((0, 0));
            eth_entry.1 += 1;
            if recognized {
                eth_entry.0 += 1;
            }

            // Update script stats
            let script_key = format!("{:?}", name_example.script);
            let script_entry = by_script.entry(script_key.clone()).or_insert((0, 0));
            script_entry.1 += 1;
            if recognized {
                script_entry.0 += 1;
            }

            // Update gender stats
            if let Some(gender) = name_example.gender {
                let gender_key = format!("{:?}", gender);
                let gender_entry = by_gender.entry(gender_key).or_insert((0, 0));
                gender_entry.1 += 1;
                if recognized {
                    gender_entry.0 += 1;
                }
            }

            // Update frequency stats
            let freq_key = format!("{:?}", name_example.frequency);
            let freq_entry = by_frequency.entry(freq_key).or_insert((0, 0));
            freq_entry.1 += 1;
            if recognized {
                freq_entry.0 += 1;
            }

            // Update intersectional stats (ethnicity × gender)
            if let Some(gender) = name_example.gender {
                let inter_key = format!("{:?}_{:?}", name_example.ethnicity, gender);
                let inter_entry = intersectional.entry(inter_key).or_insert((0, 0));
                inter_entry.1 += 1;
                if recognized {
                    inter_entry.0 += 1;
                }
            }

            if self.detailed {
                detailed_results.push(NameResult {
                    name: name_example.name.clone(),
                    recognized,
                    confidence,
                    ethnicity: eth_key,
                    script: script_key,
                });
            }
        }

        // Convert counts to rates
        let to_rate = |counts: &HashMap<String, (usize, usize)>| -> HashMap<String, f64> {
            counts.iter()
                .map(|(k, (correct, total))| {
                    let rate = if *total > 0 { *correct as f64 / *total as f64 } else { 0.0 };
                    (k.clone(), rate)
                })
                .collect()
        };

        let ethnicity_rates = to_rate(&by_ethnicity);
        let script_rates = to_rate(&by_script);
        let gender_rates = to_rate(&by_gender);
        let frequency_rates = to_rate(&by_frequency);
        let intersectional_rates = to_rate(&intersectional);

        // Compute parity gaps
        let ethnicity_parity_gap = compute_max_gap(&ethnicity_rates);
        
        // Script bias: compare Latin to non-Latin
        let latin_rate = script_rates.get("Latin").copied().unwrap_or(0.0);
        let non_latin_rates: Vec<f64> = script_rates.iter()
            .filter(|(k, _)| k.as_str() != "Latin")
            .map(|(_, v)| *v)
            .collect();
        let avg_non_latin = if non_latin_rates.is_empty() {
            latin_rate
        } else {
            non_latin_rates.iter().sum::<f64>() / non_latin_rates.len() as f64
        };
        let script_bias_gap = (latin_rate - avg_non_latin).abs();

        DemographicBiasResults {
            overall_recognition_rate: if names.is_empty() {
                0.0
            } else {
                total_recognized as f64 / names.len() as f64
            },
            by_ethnicity: ethnicity_rates,
            by_script: script_rates,
            by_gender: gender_rates,
            by_frequency: frequency_rates,
            ethnicity_parity_gap,
            script_bias_gap,
            intersectional: intersectional_rates,
            total_tested: names.len(),
            detailed: detailed_results,
        }
    }

    /// Evaluate NER model for regional bias on locations.
    pub fn evaluate_locations(
        &self,
        model: &dyn Model,
        locations: &[LocationExample],
    ) -> RegionalBiasResults {
        let mut by_region: HashMap<String, (usize, usize)> = HashMap::new();
        let mut by_script: HashMap<String, (usize, usize)> = HashMap::new();
        let mut total_recognized = 0;

        for loc in locations {
            let text = format!("The meeting was held in {}.", loc.name);
            let entities = model.extract_entities(&text, None).unwrap_or_default();
            
            let recognized = entities.iter().any(|e| {
                e.entity_type == EntityType::Location && 
                text[e.start..e.end].contains(&loc.name)
            });

            if recognized {
                total_recognized += 1;
            }

            // Update region stats
            let region_key = format!("{:?}", loc.region);
            let region_entry = by_region.entry(region_key).or_insert((0, 0));
            region_entry.1 += 1;
            if recognized {
                region_entry.0 += 1;
            }

            // Update script stats
            let script_key = format!("{:?}", loc.script);
            let script_entry = by_script.entry(script_key).or_insert((0, 0));
            script_entry.1 += 1;
            if recognized {
                script_entry.0 += 1;
            }
        }

        let to_rate = |counts: &HashMap<String, (usize, usize)>| -> HashMap<String, f64> {
            counts.iter()
                .map(|(k, (correct, total))| {
                    let rate = if *total > 0 { *correct as f64 / *total as f64 } else { 0.0 };
                    (k.clone(), rate)
                })
                .collect()
        };

        let region_rates = to_rate(&by_region);
        let script_rates = to_rate(&by_script);
        let regional_parity_gap = compute_max_gap(&region_rates);

        RegionalBiasResults {
            overall_recognition_rate: if locations.is_empty() {
                0.0
            } else {
                total_recognized as f64 / locations.len() as f64
            },
            by_region: region_rates,
            by_script: script_rates,
            regional_parity_gap,
            total_tested: locations.len(),
        }
    }
}

/// Compute maximum gap between any two rates.
fn compute_max_gap(rates: &HashMap<String, f64>) -> f64 {
    if rates.len() < 2 {
        return 0.0;
    }
    
    let values: Vec<f64> = rates.values().copied().collect();
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    
    max - min
}

// =============================================================================
// Diverse Name Dataset
// =============================================================================

/// Create a diverse name dataset for bias testing.
///
/// Includes names from multiple ethnicities, scripts, genders, and frequencies.
/// Based on census data and common names from various countries.
pub fn create_diverse_name_dataset() -> Vec<NameExample> {
    let mut names = Vec::new();

    // === European Names ===
    names.extend(vec![
        NameExample::new("James", "Smith", Ethnicity::European, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Mary", "Johnson", Ethnicity::European, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("William", "Williams", Ethnicity::European, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Emma", "Brown", Ethnicity::European, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("Heinrich", "Mueller", Ethnicity::European, Script::Latin, Some(Gender::Masculine), NameFrequency::Moderate),
        NameExample::new("François", "Dubois", Ethnicity::European, Script::Latin, Some(Gender::Masculine), NameFrequency::Moderate),
        NameExample::new("Giulia", "Rossi", Ethnicity::European, Script::Latin, Some(Gender::Feminine), NameFrequency::Moderate),
        NameExample::new("Björk", "Guðmundsdóttir", Ethnicity::European, Script::Latin, Some(Gender::Feminine), NameFrequency::Rare),
    ]);

    // === African-American Names ===
    names.extend(vec![
        NameExample::new("DeShawn", "Jackson", Ethnicity::AfricanAmerican, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Latoya", "Williams", Ethnicity::AfricanAmerican, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("Jamal", "Robinson", Ethnicity::AfricanAmerican, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Aaliyah", "Washington", Ethnicity::AfricanAmerican, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("Tyrone", "Davis", Ethnicity::AfricanAmerican, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Imani", "Johnson", Ethnicity::AfricanAmerican, Script::Latin, Some(Gender::Feminine), NameFrequency::Moderate),
        NameExample::new("Darnell", "Thompson", Ethnicity::AfricanAmerican, Script::Latin, Some(Gender::Masculine), NameFrequency::Moderate),
        NameExample::new("Shaniqua", "Brown", Ethnicity::AfricanAmerican, Script::Latin, Some(Gender::Feminine), NameFrequency::Rare),
    ]);

    // === Hispanic Names ===
    names.extend(vec![
        NameExample::new("José", "García", Ethnicity::Hispanic, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("María", "Rodriguez", Ethnicity::Hispanic, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("Carlos", "Martinez", Ethnicity::Hispanic, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Isabella", "Lopez", Ethnicity::Hispanic, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("Diego", "Hernandez", Ethnicity::Hispanic, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Sofía", "González", Ethnicity::Hispanic, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("Javier", "Pérez", Ethnicity::Hispanic, Script::Latin, Some(Gender::Masculine), NameFrequency::Moderate),
        NameExample::new("Guadalupe", "Sánchez", Ethnicity::Hispanic, Script::Latin, Some(Gender::Neutral), NameFrequency::Moderate),
    ]);

    // === East Asian Names ===
    names.extend(vec![
        // Chinese (Latin transliteration)
        NameExample::new("Wei", "Wang", Ethnicity::EastAsian, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Li", "Zhang", Ethnicity::EastAsian, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("Ming", "Chen", Ethnicity::EastAsian, Script::Latin, Some(Gender::Neutral), NameFrequency::Common),
        // Chinese (characters)
        NameExample::new("伟", "王", Ethnicity::EastAsian, Script::Chinese, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("丽", "张", Ethnicity::EastAsian, Script::Chinese, Some(Gender::Feminine), NameFrequency::Common),
        // Japanese
        NameExample::new("Takeshi", "Tanaka", Ethnicity::EastAsian, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Yuki", "Yamamoto", Ethnicity::EastAsian, Script::Latin, Some(Gender::Neutral), NameFrequency::Common),
        NameExample::new("太郎", "田中", Ethnicity::EastAsian, Script::Japanese, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("花子", "山本", Ethnicity::EastAsian, Script::Japanese, Some(Gender::Feminine), NameFrequency::Common),
        // Korean
        NameExample::new("Min-jun", "Kim", Ethnicity::EastAsian, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Seo-yeon", "Park", Ethnicity::EastAsian, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("민준", "김", Ethnicity::EastAsian, Script::Korean, Some(Gender::Masculine), NameFrequency::Common),
    ]);

    // === South Asian Names ===
    names.extend(vec![
        NameExample::new("Raj", "Patel", Ethnicity::SouthAsian, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Priya", "Sharma", Ethnicity::SouthAsian, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("Arjun", "Singh", Ethnicity::SouthAsian, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Aisha", "Khan", Ethnicity::SouthAsian, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("Vikram", "Kumar", Ethnicity::SouthAsian, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Sunita", "Gupta", Ethnicity::SouthAsian, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        // Devanagari script
        NameExample::new("राज", "पटेल", Ethnicity::SouthAsian, Script::Devanagari, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("प्रिया", "शर्मा", Ethnicity::SouthAsian, Script::Devanagari, Some(Gender::Feminine), NameFrequency::Common),
    ]);

    // === Middle Eastern Names ===
    names.extend(vec![
        NameExample::new("Ahmed", "Hassan", Ethnicity::MiddleEastern, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Fatima", "Ali", Ethnicity::MiddleEastern, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("Mohammed", "Ibrahim", Ethnicity::MiddleEastern, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Layla", "Omar", Ethnicity::MiddleEastern, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("Yusuf", "Mustafa", Ethnicity::MiddleEastern, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Mariam", "Khalil", Ethnicity::MiddleEastern, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        // Arabic script
        NameExample::new("أحمد", "حسن", Ethnicity::MiddleEastern, Script::Arabic, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("فاطمة", "علي", Ethnicity::MiddleEastern, Script::Arabic, Some(Gender::Feminine), NameFrequency::Common),
    ]);

    // === African Names ===
    names.extend(vec![
        NameExample::new("Chidi", "Okonkwo", Ethnicity::African, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Amara", "Adebayo", Ethnicity::African, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("Kwame", "Mensah", Ethnicity::African, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Nneka", "Nwosu", Ethnicity::African, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("Oluwaseun", "Afolabi", Ethnicity::African, Script::Latin, Some(Gender::Masculine), NameFrequency::Moderate),
        NameExample::new("Chidinma", "Eze", Ethnicity::African, Script::Latin, Some(Gender::Feminine), NameFrequency::Moderate),
        NameExample::new("Tendai", "Moyo", Ethnicity::African, Script::Latin, Some(Gender::Neutral), NameFrequency::Moderate),
        NameExample::new("Zainab", "Diallo", Ethnicity::African, Script::Latin, Some(Gender::Feminine), NameFrequency::Moderate),
    ]);

    // === Cyrillic Names (Russian/Eastern European) ===
    names.extend(vec![
        NameExample::new("Ivan", "Petrov", Ethnicity::European, Script::Latin, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Olga", "Ivanova", Ethnicity::European, Script::Latin, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("Иван", "Петров", Ethnicity::European, Script::Cyrillic, Some(Gender::Masculine), NameFrequency::Common),
        NameExample::new("Ольга", "Иванова", Ethnicity::European, Script::Cyrillic, Some(Gender::Feminine), NameFrequency::Common),
        NameExample::new("Dmytro", "Shevchenko", Ethnicity::European, Script::Latin, Some(Gender::Masculine), NameFrequency::Moderate),
        NameExample::new("Katarzyna", "Kowalski", Ethnicity::European, Script::Latin, Some(Gender::Feminine), NameFrequency::Moderate),
    ]);

    names
}

/// Create a diverse location dataset for regional bias testing.
pub fn create_diverse_location_dataset() -> Vec<LocationExample> {
    vec![
        // North America
        LocationExample::new("New York", Region::NorthAmerica, Script::Latin, LocationType::City),
        LocationExample::new("Los Angeles", Region::NorthAmerica, Script::Latin, LocationType::City),
        LocationExample::new("Toronto", Region::NorthAmerica, Script::Latin, LocationType::City),
        LocationExample::new("Mexico City", Region::NorthAmerica, Script::Latin, LocationType::City),
        
        // Western Europe
        LocationExample::new("London", Region::WesternEurope, Script::Latin, LocationType::City),
        LocationExample::new("Paris", Region::WesternEurope, Script::Latin, LocationType::City),
        LocationExample::new("Berlin", Region::WesternEurope, Script::Latin, LocationType::City),
        LocationExample::new("Amsterdam", Region::WesternEurope, Script::Latin, LocationType::City),
        
        // Eastern Europe
        LocationExample::new("Moscow", Region::EasternEurope, Script::Latin, LocationType::City),
        LocationExample::new("Москва", Region::EasternEurope, Script::Cyrillic, LocationType::City),
        LocationExample::new("Warsaw", Region::EasternEurope, Script::Latin, LocationType::City),
        LocationExample::new("Kyiv", Region::EasternEurope, Script::Latin, LocationType::City),
        
        // East Asia
        LocationExample::new("Tokyo", Region::EastAsia, Script::Latin, LocationType::City),
        LocationExample::new("東京", Region::EastAsia, Script::Japanese, LocationType::City),
        LocationExample::new("Beijing", Region::EastAsia, Script::Latin, LocationType::City),
        LocationExample::new("北京", Region::EastAsia, Script::Chinese, LocationType::City),
        LocationExample::new("Seoul", Region::EastAsia, Script::Latin, LocationType::City),
        LocationExample::new("서울", Region::EastAsia, Script::Korean, LocationType::City),
        
        // South Asia
        LocationExample::new("Mumbai", Region::SouthAsia, Script::Latin, LocationType::City),
        LocationExample::new("Delhi", Region::SouthAsia, Script::Latin, LocationType::City),
        LocationExample::new("Dhaka", Region::SouthAsia, Script::Latin, LocationType::City),
        LocationExample::new("Karachi", Region::SouthAsia, Script::Latin, LocationType::City),
        
        // Southeast Asia
        LocationExample::new("Bangkok", Region::SoutheastAsia, Script::Latin, LocationType::City),
        LocationExample::new("Singapore", Region::SoutheastAsia, Script::Latin, LocationType::City),
        LocationExample::new("Jakarta", Region::SoutheastAsia, Script::Latin, LocationType::City),
        LocationExample::new("Ho Chi Minh City", Region::SoutheastAsia, Script::Latin, LocationType::City),
        
        // Middle East
        LocationExample::new("Dubai", Region::MiddleEast, Script::Latin, LocationType::City),
        LocationExample::new("دبي", Region::MiddleEast, Script::Arabic, LocationType::City),
        LocationExample::new("Tehran", Region::MiddleEast, Script::Latin, LocationType::City),
        LocationExample::new("Riyadh", Region::MiddleEast, Script::Latin, LocationType::City),
        
        // Africa
        LocationExample::new("Lagos", Region::Africa, Script::Latin, LocationType::City),
        LocationExample::new("Nairobi", Region::Africa, Script::Latin, LocationType::City),
        LocationExample::new("Cairo", Region::Africa, Script::Latin, LocationType::City),
        LocationExample::new("Johannesburg", Region::Africa, Script::Latin, LocationType::City),
        LocationExample::new("Addis Ababa", Region::Africa, Script::Latin, LocationType::City),
        
        // Latin America
        LocationExample::new("São Paulo", Region::LatinAmerica, Script::Latin, LocationType::City),
        LocationExample::new("Buenos Aires", Region::LatinAmerica, Script::Latin, LocationType::City),
        LocationExample::new("Bogotá", Region::LatinAmerica, Script::Latin, LocationType::City),
        LocationExample::new("Lima", Region::LatinAmerica, Script::Latin, LocationType::City),
        
        // Oceania
        LocationExample::new("Sydney", Region::Oceania, Script::Latin, LocationType::City),
        LocationExample::new("Melbourne", Region::Oceania, Script::Latin, LocationType::City),
        LocationExample::new("Auckland", Region::Oceania, Script::Latin, LocationType::City),
    ]
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_diverse_names() {
        let names = create_diverse_name_dataset();
        
        // Should have names from all ethnicities
        let ethnicities: std::collections::HashSet<_> = names.iter()
            .map(|n| format!("{:?}", n.ethnicity))
            .collect();
        
        assert!(ethnicities.contains("European"), "Should have European names");
        assert!(ethnicities.contains("AfricanAmerican"), "Should have African-American names");
        assert!(ethnicities.contains("Hispanic"), "Should have Hispanic names");
        assert!(ethnicities.contains("EastAsian"), "Should have East Asian names");
        assert!(ethnicities.contains("SouthAsian"), "Should have South Asian names");
        assert!(ethnicities.contains("MiddleEastern"), "Should have Middle Eastern names");
        assert!(ethnicities.contains("African"), "Should have African names");
    }

    #[test]
    fn test_multiple_scripts() {
        let names = create_diverse_name_dataset();
        
        let scripts: std::collections::HashSet<_> = names.iter()
            .map(|n| format!("{:?}", n.script))
            .collect();
        
        assert!(scripts.contains("Latin"), "Should have Latin script");
        assert!(scripts.contains("Chinese"), "Should have Chinese script");
        assert!(scripts.contains("Japanese"), "Should have Japanese script");
        assert!(scripts.contains("Arabic"), "Should have Arabic script");
        assert!(scripts.contains("Cyrillic"), "Should have Cyrillic script");
    }

    #[test]
    fn test_gender_balance() {
        let names = create_diverse_name_dataset();
        
        let masculine = names.iter()
            .filter(|n| n.gender == Some(Gender::Masculine))
            .count();
        let feminine = names.iter()
            .filter(|n| n.gender == Some(Gender::Feminine))
            .count();
        
        // Should have roughly balanced genders
        let ratio = masculine as f64 / feminine.max(1) as f64;
        assert!(
            (0.7..=1.3).contains(&ratio),
            "Gender ratio should be roughly balanced, got {:.2}",
            ratio
        );
    }

    #[test]
    fn test_diverse_locations() {
        let locations = create_diverse_location_dataset();
        
        let regions: std::collections::HashSet<_> = locations.iter()
            .map(|l| format!("{:?}", l.region))
            .collect();
        
        assert!(regions.len() >= 8, "Should cover at least 8 regions");
        assert!(regions.contains("Africa"), "Should have African locations");
        assert!(regions.contains("LatinAmerica"), "Should have Latin American locations");
        assert!(regions.contains("MiddleEast"), "Should have Middle Eastern locations");
    }

    #[test]
    fn test_parity_gap_computation() {
        let mut rates = HashMap::new();
        rates.insert("A".to_string(), 0.9);
        rates.insert("B".to_string(), 0.7);
        rates.insert("C".to_string(), 0.8);
        
        let gap = compute_max_gap(&rates);
        assert!((gap - 0.2).abs() < 0.001, "Gap should be 0.2, got {}", gap);
    }
}

