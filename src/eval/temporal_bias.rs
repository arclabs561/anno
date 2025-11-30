//! Temporal bias evaluation for Named Entity Recognition.
//!
//! Measures performance differences on names popular in different time periods.
//! Models trained primarily on contemporary data may struggle with historical names,
//! and vice versa.
//!
//! # Research Background
//!
//! - U.S. Social Security Administration baby name data (1880-present)
//! - Temporal distribution shift in training data
//! - "Ethel" (peaked 1900s) vs "Jayden" (peaked 2000s)
//!
//! # Key Metrics
//!
//! - **Decade Recognition Rate**: Recognition accuracy per decade of name popularity
//! - **Temporal Parity Gap**: Max difference across decades
//! - **Historical-Modern Gap**: Difference between pre-1950 and post-2000 names
//!
//! # Example
//!
//! ```rust
//! use anno::eval::temporal_bias::{TemporalBiasEvaluator, create_temporal_name_dataset};
//!
//! let names = create_temporal_name_dataset();
//! let evaluator = TemporalBiasEvaluator::default();
//! // let results = evaluator.evaluate(&PatternNER::new(), &names);
//! ```

use crate::{EntityType, Model};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Temporal Categories
// =============================================================================

/// Decade when a name was most popular.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum Decade {
    /// Pre-1900 (Victorian era names)
    Pre1900,
    /// 1900-1909
    D1900s,
    /// 1910-1919
    D1910s,
    /// 1920-1929
    D1920s,
    /// 1930-1939
    D1930s,
    /// 1940-1949
    D1940s,
    /// 1950-1959
    D1950s,
    /// 1960-1969
    D1960s,
    /// 1970-1979
    D1970s,
    /// 1980-1989
    D1980s,
    /// 1990-1999
    D1990s,
    /// 2000-2009
    D2000s,
    /// 2010-2019
    D2010s,
    /// 2020-present
    D2020s,
}

impl Decade {
    /// Returns whether this is a historical (pre-1950) decade.
    pub fn is_historical(&self) -> bool {
        matches!(
            self,
            Decade::Pre1900
                | Decade::D1900s
                | Decade::D1910s
                | Decade::D1920s
                | Decade::D1930s
                | Decade::D1940s
        )
    }

    /// Returns whether this is a modern (post-2000) decade.
    pub fn is_modern(&self) -> bool {
        matches!(self, Decade::D2000s | Decade::D2010s | Decade::D2020s)
    }

    /// Returns approximate midpoint year of the decade.
    pub fn midpoint_year(&self) -> u16 {
        match self {
            Decade::Pre1900 => 1890,
            Decade::D1900s => 1905,
            Decade::D1910s => 1915,
            Decade::D1920s => 1925,
            Decade::D1930s => 1935,
            Decade::D1940s => 1945,
            Decade::D1950s => 1955,
            Decade::D1960s => 1965,
            Decade::D1970s => 1975,
            Decade::D1980s => 1985,
            Decade::D1990s => 1995,
            Decade::D2000s => 2005,
            Decade::D2010s => 2015,
            Decade::D2020s => 2022,
        }
    }
}

// =============================================================================
// Temporal Name Example
// =============================================================================

/// A name example with temporal metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalNameExample {
    /// First name
    pub first_name: String,
    /// Last name
    pub last_name: String,
    /// Full name
    pub full_name: String,
    /// Decade of peak popularity
    pub peak_decade: Decade,
    /// Gender associated with name
    pub gender: TemporalGender,
    /// Whether this is a "classic" name (consistent popularity) vs trendy
    pub is_classic: bool,
}

/// Gender for temporal name analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemporalGender {
    /// Traditionally masculine names
    Masculine,
    /// Traditionally feminine names
    Feminine,
    /// Gender-neutral names
    Neutral,
}

impl TemporalNameExample {
    /// Create a new temporal name example.
    pub fn new(
        first_name: &str,
        last_name: &str,
        peak_decade: Decade,
        gender: TemporalGender,
        is_classic: bool,
    ) -> Self {
        Self {
            first_name: first_name.to_string(),
            last_name: last_name.to_string(),
            full_name: format!("{} {}", first_name, last_name),
            peak_decade,
            gender,
            is_classic,
        }
    }
}

// =============================================================================
// Evaluation Results
// =============================================================================

/// Results of temporal bias evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalBiasResults {
    /// Overall recognition rate
    pub overall_recognition_rate: f64,
    /// Recognition rate by decade
    pub by_decade: HashMap<String, f64>,
    /// Recognition rate for historical (pre-1950) names
    pub historical_rate: f64,
    /// Recognition rate for modern (post-2000) names
    pub modern_rate: f64,
    /// Gap between historical and modern: |historical - modern|
    pub historical_modern_gap: f64,
    /// Maximum gap between any two decades
    pub temporal_parity_gap: f64,
    /// Recognition rate by gender
    pub by_gender: HashMap<String, f64>,
    /// Recognition rate for classic names (consistent popularity across decades)
    pub classic_rate: f64,
    /// Recognition rate for trendy names (peaked in specific decade)
    pub trendy_rate: f64,
    /// Total names tested
    pub total_tested: usize,
}

// =============================================================================
// Evaluator
// =============================================================================

/// Evaluator for temporal bias in NER systems.
#[derive(Debug, Clone, Default)]
pub struct TemporalBiasEvaluator {
    /// Include detailed per-name results
    pub detailed: bool,
}

impl TemporalBiasEvaluator {
    /// Create a new evaluator.
    pub fn new(detailed: bool) -> Self {
        Self { detailed }
    }

    /// Evaluate NER model for temporal bias.
    pub fn evaluate(
        &self,
        model: &dyn Model,
        names: &[TemporalNameExample],
    ) -> TemporalBiasResults {
        let mut by_decade: HashMap<String, (usize, usize)> = HashMap::new();
        let mut by_gender: HashMap<String, (usize, usize)> = HashMap::new();
        let mut historical_count = (0usize, 0usize);
        let mut modern_count = (0usize, 0usize);
        let mut classic_count = (0usize, 0usize);
        let mut trendy_count = (0usize, 0usize);
        let mut total_recognized = 0;

        for name in names {
            // Create test sentence
            let text = format!("{} attended the conference in Chicago.", name.full_name);

            // Extract entities
            let entities = model.extract_entities(&text, None).unwrap_or_default();

            // Check if name was recognized as PERSON
            let recognized = entities.iter().any(|e| {
                e.entity_type == EntityType::Person
                    && e.extract_text(&text).contains(&name.first_name)
            });

            if recognized {
                total_recognized += 1;
            }

            // Update decade stats
            let decade_key = format!("{:?}", name.peak_decade);
            let decade_entry = by_decade.entry(decade_key).or_insert((0, 0));
            decade_entry.1 += 1;
            if recognized {
                decade_entry.0 += 1;
            }

            // Update historical/modern stats
            if name.peak_decade.is_historical() {
                historical_count.1 += 1;
                if recognized {
                    historical_count.0 += 1;
                }
            }
            if name.peak_decade.is_modern() {
                modern_count.1 += 1;
                if recognized {
                    modern_count.0 += 1;
                }
            }

            // Update gender stats
            let gender_key = format!("{:?}", name.gender);
            let gender_entry = by_gender.entry(gender_key).or_insert((0, 0));
            gender_entry.1 += 1;
            if recognized {
                gender_entry.0 += 1;
            }

            // Update classic/trendy stats
            if name.is_classic {
                classic_count.1 += 1;
                if recognized {
                    classic_count.0 += 1;
                }
            } else {
                trendy_count.1 += 1;
                if recognized {
                    trendy_count.0 += 1;
                }
            }
        }

        // Convert counts to rates
        let to_rate = |counts: &HashMap<String, (usize, usize)>| -> HashMap<String, f64> {
            counts
                .iter()
                .map(|(k, (correct, total))| {
                    let rate = if *total > 0 {
                        *correct as f64 / *total as f64
                    } else {
                        0.0
                    };
                    (k.clone(), rate)
                })
                .collect()
        };

        let count_to_rate = |c: (usize, usize)| -> f64 {
            if c.1 > 0 {
                c.0 as f64 / c.1 as f64
            } else {
                0.0
            }
        };

        let decade_rates = to_rate(&by_decade);
        let gender_rates = to_rate(&by_gender);
        let historical_rate = count_to_rate(historical_count);
        let modern_rate = count_to_rate(modern_count);
        let classic_rate = count_to_rate(classic_count);
        let trendy_rate = count_to_rate(trendy_count);

        // Compute parity gap
        let temporal_parity_gap = compute_max_gap(&decade_rates);
        let historical_modern_gap = (historical_rate - modern_rate).abs();

        TemporalBiasResults {
            overall_recognition_rate: if names.is_empty() {
                0.0
            } else {
                total_recognized as f64 / names.len() as f64
            },
            by_decade: decade_rates,
            historical_rate,
            modern_rate,
            historical_modern_gap,
            temporal_parity_gap,
            by_gender: gender_rates,
            classic_rate,
            trendy_rate,
            total_tested: names.len(),
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
// Temporal Name Dataset
// =============================================================================

/// Create a dataset of names popular in different decades.
///
/// Based on U.S. Social Security Administration baby name data.
/// Names are selected to represent peak popularity in each decade.
pub fn create_temporal_name_dataset() -> Vec<TemporalNameExample> {
    let mut names = Vec::new();

    // Generic last names to pair with first names
    let last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones"];

    // Pre-1900 (Victorian era)
    let pre1900 = [
        ("Gertrude", TemporalGender::Feminine),
        ("Clarence", TemporalGender::Masculine),
        ("Mildred", TemporalGender::Feminine),
        ("Herbert", TemporalGender::Masculine),
        ("Bertha", TemporalGender::Feminine),
    ];

    // 1900s
    let d1900s = [
        ("Ethel", TemporalGender::Feminine),
        ("Harold", TemporalGender::Masculine),
        ("Pearl", TemporalGender::Feminine),
        ("Clarence", TemporalGender::Masculine),
        ("Minnie", TemporalGender::Feminine),
    ];

    // 1910s
    let d1910s = [
        ("Dorothy", TemporalGender::Feminine),
        ("Earl", TemporalGender::Masculine),
        ("Gladys", TemporalGender::Feminine),
        ("Howard", TemporalGender::Masculine),
        ("Thelma", TemporalGender::Feminine),
    ];

    // 1920s
    let d1920s = [
        ("Betty", TemporalGender::Feminine),
        ("Donald", TemporalGender::Masculine),
        ("Doris", TemporalGender::Feminine),
        ("Raymond", TemporalGender::Masculine),
        ("Shirley", TemporalGender::Feminine),
    ];

    // 1930s
    let d1930s = [
        ("Barbara", TemporalGender::Feminine),
        ("Robert", TemporalGender::Masculine),
        ("Patricia", TemporalGender::Feminine),
        ("Richard", TemporalGender::Masculine),
        ("Carol", TemporalGender::Feminine),
    ];

    // 1940s
    let d1940s = [
        ("Linda", TemporalGender::Feminine),
        ("Gary", TemporalGender::Masculine),
        ("Sandra", TemporalGender::Feminine),
        ("Larry", TemporalGender::Masculine),
        ("Sharon", TemporalGender::Feminine),
    ];

    // 1950s
    let d1950s = [
        ("Deborah", TemporalGender::Feminine),
        ("Dennis", TemporalGender::Masculine),
        ("Debra", TemporalGender::Feminine),
        ("Timothy", TemporalGender::Masculine),
        ("Pamela", TemporalGender::Feminine),
    ];

    // 1960s
    let d1960s = [
        ("Lisa", TemporalGender::Feminine),
        ("Mark", TemporalGender::Masculine),
        ("Kimberly", TemporalGender::Feminine),
        ("Kevin", TemporalGender::Masculine),
        ("Michelle", TemporalGender::Feminine),
    ];

    // 1970s
    let d1970s = [
        ("Jennifer", TemporalGender::Feminine),
        ("Jason", TemporalGender::Masculine),
        ("Amy", TemporalGender::Feminine),
        ("Brian", TemporalGender::Masculine),
        ("Heather", TemporalGender::Feminine),
    ];

    // 1980s
    let d1980s = [
        ("Jessica", TemporalGender::Feminine),
        ("Michael", TemporalGender::Masculine),
        ("Amanda", TemporalGender::Feminine),
        ("Christopher", TemporalGender::Masculine),
        ("Ashley", TemporalGender::Feminine),
    ];

    // 1990s
    let d1990s = [
        ("Brittany", TemporalGender::Feminine),
        ("Tyler", TemporalGender::Masculine),
        ("Taylor", TemporalGender::Neutral),
        ("Brandon", TemporalGender::Masculine),
        ("Megan", TemporalGender::Feminine),
    ];

    // 2000s
    let d2000s = [
        ("Madison", TemporalGender::Feminine),
        ("Aiden", TemporalGender::Masculine),
        ("Emma", TemporalGender::Feminine),
        ("Ethan", TemporalGender::Masculine),
        ("Chloe", TemporalGender::Feminine),
    ];

    // 2010s
    let d2010s = [
        ("Sophia", TemporalGender::Feminine),
        ("Liam", TemporalGender::Masculine),
        ("Olivia", TemporalGender::Feminine),
        ("Noah", TemporalGender::Masculine),
        ("Ava", TemporalGender::Feminine),
    ];

    // 2020s
    let d2020s = [
        ("Luna", TemporalGender::Feminine),
        ("Ezra", TemporalGender::Masculine),
        ("Charlotte", TemporalGender::Feminine),
        ("Oliver", TemporalGender::Masculine),
        ("Amelia", TemporalGender::Feminine),
    ];

    // Classic names (popular across many decades)
    let classics = [
        ("James", TemporalGender::Masculine, true),
        ("Elizabeth", TemporalGender::Feminine, true),
        ("William", TemporalGender::Masculine, true),
        ("Mary", TemporalGender::Feminine, true),
        ("John", TemporalGender::Masculine, true),
    ];

    // Helper to add names from a decade
    let add_decade = |names: &mut Vec<TemporalNameExample>,
                      decade_names: &[(&str, TemporalGender)],
                      decade: Decade,
                      last_names: &[&str]| {
        for (i, (first, gender)) in decade_names.iter().enumerate() {
            let last = last_names[i % last_names.len()];
            names.push(TemporalNameExample::new(
                first, last, decade, *gender, false,
            ));
        }
    };

    add_decade(&mut names, &pre1900, Decade::Pre1900, &last_names);
    add_decade(&mut names, &d1900s, Decade::D1900s, &last_names);
    add_decade(&mut names, &d1910s, Decade::D1910s, &last_names);
    add_decade(&mut names, &d1920s, Decade::D1920s, &last_names);
    add_decade(&mut names, &d1930s, Decade::D1930s, &last_names);
    add_decade(&mut names, &d1940s, Decade::D1940s, &last_names);
    add_decade(&mut names, &d1950s, Decade::D1950s, &last_names);
    add_decade(&mut names, &d1960s, Decade::D1960s, &last_names);
    add_decade(&mut names, &d1970s, Decade::D1970s, &last_names);
    add_decade(&mut names, &d1980s, Decade::D1980s, &last_names);
    add_decade(&mut names, &d1990s, Decade::D1990s, &last_names);
    add_decade(&mut names, &d2000s, Decade::D2000s, &last_names);
    add_decade(&mut names, &d2010s, Decade::D2010s, &last_names);
    add_decade(&mut names, &d2020s, Decade::D2020s, &last_names);

    // Add classic names (spread across different "peak" decades but marked as classic)
    for (i, (first, gender, _is_classic)) in classics.iter().enumerate() {
        let last = last_names[i % last_names.len()];
        // Classic names get D1950s as nominal decade but marked as classic
        names.push(TemporalNameExample::new(
            first,
            last,
            Decade::D1950s,
            *gender,
            true,
        ));
    }

    names
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_temporal_dataset() {
        let names = create_temporal_name_dataset();

        // Should have names from multiple decades
        let decades: std::collections::HashSet<_> = names
            .iter()
            .map(|n| format!("{:?}", n.peak_decade))
            .collect();

        assert!(decades.len() >= 10, "Should cover at least 10 decades");
        assert!(
            decades.contains("Pre1900"),
            "Should have pre-1900 (Victorian) names"
        );
        assert!(decades.contains("D2020s"), "Should have 2020s names");
    }

    #[test]
    fn test_historical_vs_modern() {
        let names = create_temporal_name_dataset();

        let historical = names
            .iter()
            .filter(|n| n.peak_decade.is_historical())
            .count();
        let modern = names.iter().filter(|n| n.peak_decade.is_modern()).count();

        assert!(historical > 0, "Should have historical names");
        assert!(modern > 0, "Should have modern names");
    }

    #[test]
    fn test_classic_names_marked() {
        let names = create_temporal_name_dataset();

        let classics: Vec<_> = names.iter().filter(|n| n.is_classic).collect();

        assert!(!classics.is_empty(), "Should have classic names");
        assert!(
            classics.iter().any(|n| n.first_name == "James"),
            "James should be a classic"
        );
        assert!(
            classics.iter().any(|n| n.first_name == "Elizabeth"),
            "Elizabeth should be a classic"
        );
    }

    #[test]
    fn test_decade_ordering() {
        assert!(Decade::Pre1900 < Decade::D1900s);
        assert!(Decade::D1900s < Decade::D2020s);
        assert!(Decade::D1980s.midpoint_year() == 1985);
    }

    #[test]
    fn test_gender_distribution() {
        let names = create_temporal_name_dataset();

        let masculine = names
            .iter()
            .filter(|n| n.gender == TemporalGender::Masculine)
            .count();
        let feminine = names
            .iter()
            .filter(|n| n.gender == TemporalGender::Feminine)
            .count();

        // Should have reasonable gender distribution
        assert!(masculine > 20, "Should have substantial masculine names");
        assert!(feminine > 20, "Should have substantial feminine names");
    }
}
