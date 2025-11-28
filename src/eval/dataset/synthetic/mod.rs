//! Synthetic NER datasets organized by domain.
//!
//! # Research Context
//!
//! Synthetic data has known limitations (arXiv:2505.16814 "Does Synthetic Data Help NER"):
//!
//! | Issue | Mitigation |
//! |-------|------------|
//! | Entity type skew | Stratified sampling |
//! | Clean annotations | Add noise injection |
//! | Domain gap | Mix with real data |
//! | Label shift | Track via metrics |
//!
//! # What This Dataset IS Good For
//!
//! - **Unit testing**: Does the code work at all?
//! - **Pattern coverage**: Are regex patterns correct?
//! - **Edge cases**: Unicode, boundaries, special chars
//! - **Fast iteration**: Runs in <1s, no network
//!
//! # What This Dataset IS NOT Good For
//!
//! - **Zero-shot claims**: Label overlap with training ≈ 100%
//! - **Real-world performance**: Synthetic ≠ domain-specific noise
//! - **Model comparison**: Needs WikiGold/CoNLL/WNUT for fair eval
//!
//! # Domain Modules
//!
//! Each domain module provides a `dataset()` function returning `Vec<AnnotatedExample>`.

mod biomedical;
mod entertainment;
mod financial;
mod legal;
mod misc;
mod news;
mod scientific;
mod social_media;
mod specialized;

pub use biomedical::dataset as biomedical_dataset;
pub use entertainment::dataset as entertainment_dataset;
pub use financial::dataset as financial_dataset;
pub use legal::dataset as legal_dataset;
pub use misc::{
    adversarial_dataset, conversational_dataset, historical_dataset, structured_dataset,
};
pub use news::dataset as news_dataset;
pub use scientific::dataset as scientific_dataset;
pub use social_media::dataset as social_media_dataset;
pub use specialized::{
    academic_dataset, cybersecurity_dataset, ecommerce_dataset, food_dataset,
    globally_diverse_dataset, hard_domain_examples, multilingual_dataset, politics_dataset,
    real_estate_dataset, sports_dataset, travel_dataset, weather_dataset,
};

use super::types::{AnnotatedExample, Difficulty, Domain};
use std::collections::HashMap;

/// Get all synthetic datasets combined.
///
/// This is the primary function for comprehensive testing.
/// Returns examples from all domains and difficulty levels.
pub fn all_datasets() -> Vec<AnnotatedExample> {
    let mut all = Vec::with_capacity(500);

    // Core domains
    all.extend(news::dataset());
    all.extend(social_media::dataset());
    all.extend(biomedical::dataset());
    all.extend(financial::dataset());
    all.extend(legal::dataset());
    all.extend(scientific::dataset());
    all.extend(entertainment::dataset());

    // Miscellaneous
    all.extend(misc::adversarial_dataset());
    all.extend(misc::structured_dataset());
    all.extend(misc::conversational_dataset());
    all.extend(misc::historical_dataset());

    // Specialized domains
    all.extend(specialized::sports_dataset());
    all.extend(specialized::politics_dataset());
    all.extend(specialized::ecommerce_dataset());
    all.extend(specialized::travel_dataset());
    all.extend(specialized::weather_dataset());
    all.extend(specialized::academic_dataset());
    all.extend(specialized::food_dataset());
    all.extend(specialized::real_estate_dataset());
    all.extend(specialized::cybersecurity_dataset());
    all.extend(specialized::multilingual_dataset());
    all.extend(specialized::globally_diverse_dataset());
    all.extend(specialized::hard_domain_examples());

    all
}

/// Get datasets for a specific domain.
pub fn by_domain(domain: Domain) -> Vec<AnnotatedExample> {
    all_datasets()
        .into_iter()
        .filter(|ex| ex.domain == domain)
        .collect()
}

/// Get datasets for a specific difficulty level.
pub fn by_difficulty(difficulty: Difficulty) -> Vec<AnnotatedExample> {
    all_datasets()
        .into_iter()
        .filter(|ex| ex.difficulty == difficulty)
        .collect()
}

/// Get dataset statistics.
pub fn stats() -> SyntheticStats {
    let all = all_datasets();
    let total_entities: usize = all.iter().map(|ex| ex.entities.len()).sum();

    let mut domains = HashMap::new();
    let mut difficulties = HashMap::new();

    for ex in &all {
        *domains.entry(format!("{:?}", ex.domain)).or_insert(0) += 1;
        *difficulties
            .entry(format!("{:?}", ex.difficulty))
            .or_insert(0) += 1;
    }

    SyntheticStats {
        total_examples: all.len(),
        total_entities,
        domains,
        difficulties,
    }
}

/// Statistics about synthetic datasets.
#[derive(Debug, Clone)]
pub struct SyntheticStats {
    /// Total number of examples across all datasets.
    pub total_examples: usize,
    /// Total number of entities across all examples.
    pub total_entities: usize,
    /// Count of examples per domain.
    pub domains: HashMap<String, usize>,
    /// Count of examples per difficulty level.
    pub difficulties: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_datasets_not_empty() {
        let all = all_datasets();
        assert!(!all.is_empty(), "Should have synthetic examples");
        assert!(all.len() >= 100, "Should have at least 100 examples");
    }

    #[test]
    fn test_by_domain() {
        let news = by_domain(Domain::News);
        assert!(!news.is_empty());
        for ex in &news {
            assert_eq!(ex.domain, Domain::News);
        }
    }

    #[test]
    fn test_by_difficulty() {
        let easy = by_difficulty(Difficulty::Easy);
        assert!(!easy.is_empty());
        for ex in &easy {
            assert_eq!(ex.difficulty, Difficulty::Easy);
        }
    }

    #[test]
    fn test_stats() {
        let s = stats();
        assert!(s.total_examples > 0);
        assert!(s.total_entities > 0);
        assert!(!s.domains.is_empty());
    }

    #[test]
    fn test_entity_offsets_valid() {
        for example in all_datasets() {
            let text_chars: Vec<char> = example.text.chars().collect();

            for entity in &example.entities {
                assert!(
                    entity.end <= text_chars.len(),
                    "Entity '{}' end {} exceeds char count {} in: {}",
                    entity.text,
                    entity.end,
                    text_chars.len(),
                    example.text
                );

                let actual_text: String = text_chars[entity.start..entity.end].iter().collect();
                assert_eq!(
                    actual_text, entity.text,
                    "Entity text mismatch at [{}, {}): expected '{}', found '{}' in: {}",
                    entity.start, entity.end, entity.text, actual_text, example.text
                );
            }
        }
    }

    #[test]
    fn test_no_overlapping_entities() {
        for example in all_datasets() {
            let mut spans: Vec<(usize, usize, &str)> = example
                .entities
                .iter()
                .map(|e| (e.start, e.end, e.text.as_str()))
                .collect();
            spans.sort_by_key(|(start, _, _)| *start);

            for window in spans.windows(2) {
                let (_, end1, text1) = window[0];
                let (start2, _, text2) = window[1];
                assert!(
                    end1 <= start2,
                    "Overlapping entities '{}' and '{}' in: {}",
                    text1,
                    text2,
                    example.text
                );
            }
        }
    }
}

