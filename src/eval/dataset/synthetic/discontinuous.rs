//! Synthetic discontinuous NER examples.
//!
//! # Overview
//!
//! Discontinuous entities span non-contiguous text regions, common in:
//! - **Coordination structures**: "New York and Los Angeles airports"
//! - **Biomedical text**: "left and right ventricle"
//! - **Legal documents**: "paragraphs 2(a), 3(b), and 4(c)"
//!
//! # Research Alignment
//!
//! From W2NER (arXiv:2112.10070):
//! > "Discontinuous NER remains challenging because entities can be
//! > scattered across non-adjacent text positions."
//!
//! Benchmark datasets: CADEC (~70% F1), ShARe13 (~80% F1), ShARe14 (~85% F1)

use crate::eval::discontinuous::DiscontinuousGold;

/// A synthetic example with discontinuous entities.
#[derive(Debug, Clone)]
pub struct DiscontinuousExample {
    /// The text
    pub text: String,
    /// Gold standard entities (may be discontinuous)
    pub entities: Vec<DiscontinuousGold>,
    /// Difficulty level
    pub difficulty: Difficulty,
    /// Domain
    pub domain: Domain,
}

/// Difficulty level for discontinuous examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Difficulty {
    /// Simple coordination: "X and Y Z"
    Easy,
    /// Multiple coordinations: "X, Y, and Z W"
    Medium,
    /// Nested/complex: "X and Y of Z and W"
    Hard,
}

/// Domain for discontinuous examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// General news/web text
    General,
    /// Biomedical/clinical text
    Biomedical,
    /// Legal documents
    Legal,
    /// Scientific text
    Scientific,
}

/// Generate all discontinuous synthetic examples.
pub fn dataset() -> Vec<DiscontinuousExample> {
    let mut examples = Vec::new();

    // Easy: Simple coordination structures
    examples.extend(easy_coordination());

    // Medium: Multiple coordinations
    examples.extend(medium_coordination());

    // Hard: Nested and complex structures
    examples.extend(hard_structures());

    // Biomedical domain
    examples.extend(biomedical_domain());

    // Legal domain
    examples.extend(legal_domain());

    examples
}

/// Easy: Simple "X and Y Z" patterns.
fn easy_coordination() -> Vec<DiscontinuousExample> {
    vec![
        DiscontinuousExample {
            text: "New York and Los Angeles airports have increased security.".to_string(),
            entities: vec![
                // "New York airports" (discontinuous)
                DiscontinuousGold::new(
                    vec![(0, 8), (25, 33)],  // "New York" + "airports"
                    "LOC",
                    "New York airports",
                ),
                // "Los Angeles airports" (discontinuous)
                DiscontinuousGold::new(
                    vec![(13, 24), (25, 33)], // "Los Angeles" + "airports"
                    "LOC",
                    "Los Angeles airports",
                ),
            ],
            difficulty: Difficulty::Easy,
            domain: Domain::General,
        },
        DiscontinuousExample {
            text: "Apple and Microsoft stocks rose sharply.".to_string(),
            entities: vec![
                DiscontinuousGold::new(
                    vec![(0, 5), (20, 26)],
                    "ORG",
                    "Apple stocks",
                ),
                DiscontinuousGold::new(
                    vec![(10, 19), (20, 26)],
                    "ORG",
                    "Microsoft stocks",
                ),
            ],
            difficulty: Difficulty::Easy,
            domain: Domain::General,
        },
        DiscontinuousExample {
            text: "John and Mary Smith attended the conference.".to_string(),
            entities: vec![
                DiscontinuousGold::new(
                    vec![(0, 4), (14, 19)],  // "John" + "Smith"
                    "PER",
                    "John Smith",
                ),
                DiscontinuousGold::new(
                    vec![(9, 13), (14, 19)], // "Mary" + "Smith"
                    "PER",
                    "Mary Smith",
                ),
            ],
            difficulty: Difficulty::Easy,
            domain: Domain::General,
        },
        DiscontinuousExample {
            text: "The red and blue cars were parked outside.".to_string(),
            entities: vec![
                DiscontinuousGold::new(
                    vec![(4, 7), (17, 21)],  // "red" + "cars"
                    "MISC",
                    "red cars",
                ),
                DiscontinuousGold::new(
                    vec![(12, 16), (17, 21)], // "blue" + "cars"
                    "MISC",
                    "blue cars",
                ),
            ],
            difficulty: Difficulty::Easy,
            domain: Domain::General,
        },
    ]
}

/// Medium: Multiple coordinations and longer structures.
fn medium_coordination() -> Vec<DiscontinuousExample> {
    vec![
        DiscontinuousExample {
            text: "Paris, London, and Berlin museums are world-renowned.".to_string(),
            entities: vec![
                DiscontinuousGold::new(
                    vec![(0, 5), (27, 34)],
                    "LOC",
                    "Paris museums",
                ),
                DiscontinuousGold::new(
                    vec![(7, 13), (27, 34)],
                    "LOC",
                    "London museums",
                ),
                DiscontinuousGold::new(
                    vec![(19, 25), (27, 34)],
                    "LOC",
                    "Berlin museums",
                ),
            ],
            difficulty: Difficulty::Medium,
            domain: Domain::General,
        },
        DiscontinuousExample {
            text: "CEO and CFO positions at Google and Meta are highly competitive.".to_string(),
            entities: vec![
                DiscontinuousGold::new(
                    vec![(0, 3), (14, 23)],  // "CEO" + "positions"
                    "MISC",
                    "CEO positions",
                ),
                DiscontinuousGold::new(
                    vec![(8, 11), (14, 23)], // "CFO" + "positions"
                    "MISC",
                    "CFO positions",
                ),
                DiscontinuousGold::contiguous(27, 33, "ORG", "Google"),
                DiscontinuousGold::contiguous(38, 42, "ORG", "Meta"),
            ],
            difficulty: Difficulty::Medium,
            domain: Domain::General,
        },
        DiscontinuousExample {
            text: "The first, second, and third quarters of 2024 showed growth.".to_string(),
            entities: vec![
                DiscontinuousGold::new(
                    vec![(4, 9), (32, 40), (44, 48)],
                    "DATE",
                    "first quarters of 2024",
                ),
                DiscontinuousGold::new(
                    vec![(11, 17), (32, 40), (44, 48)],
                    "DATE",
                    "second quarters of 2024",
                ),
                DiscontinuousGold::new(
                    vec![(23, 28), (32, 40), (44, 48)],
                    "DATE",
                    "third quarters of 2024",
                ),
            ],
            difficulty: Difficulty::Medium,
            domain: Domain::General,
        },
    ]
}

/// Hard: Nested and complex discontinuous structures.
fn hard_structures() -> Vec<DiscontinuousExample> {
    vec![
        DiscontinuousExample {
            text: "North and South American countries signed the treaty.".to_string(),
            entities: vec![
                // "North American countries" (discontinuous nested)
                DiscontinuousGold::new(
                    vec![(0, 5), (16, 24), (25, 34)],
                    "LOC",
                    "North American countries",
                ),
                // "South American countries"
                DiscontinuousGold::new(
                    vec![(10, 15), (16, 24), (25, 34)],
                    "LOC",
                    "South American countries",
                ),
            ],
            difficulty: Difficulty::Hard,
            domain: Domain::General,
        },
        DiscontinuousExample {
            text: "Sections 2(a), 3(b), and 4(c) of the agreement shall apply.".to_string(),
            entities: vec![
                DiscontinuousGold::new(
                    vec![(0, 8), (9, 13), (31, 48)],
                    "MISC",
                    "Sections 2(a) of the agreement",
                ),
                DiscontinuousGold::new(
                    vec![(0, 8), (15, 19), (31, 48)],
                    "MISC",
                    "Sections 3(b) of the agreement",
                ),
                DiscontinuousGold::new(
                    vec![(0, 8), (25, 29), (31, 48)],
                    "MISC",
                    "Sections 4(c) of the agreement",
                ),
            ],
            difficulty: Difficulty::Hard,
            domain: Domain::Legal,
        },
    ]
}

/// Biomedical domain examples.
fn biomedical_domain() -> Vec<DiscontinuousExample> {
    vec![
        DiscontinuousExample {
            text: "The left and right ventricle showed abnormal function.".to_string(),
            entities: vec![
                DiscontinuousGold::new(
                    vec![(4, 8), (19, 28)],  // "left" + "ventricle"
                    "ANATOMY",
                    "left ventricle",
                ),
                DiscontinuousGold::new(
                    vec![(13, 18), (19, 28)], // "right" + "ventricle"
                    "ANATOMY",
                    "right ventricle",
                ),
            ],
            difficulty: Difficulty::Easy,
            domain: Domain::Biomedical,
        },
        DiscontinuousExample {
            text: "Pain in the upper and lower back was reported.".to_string(),
            entities: vec![
                DiscontinuousGold::new(
                    vec![(12, 17), (28, 32)],  // "upper" + "back"
                    "SYMPTOM",
                    "upper back",
                ),
                DiscontinuousGold::new(
                    vec![(22, 27), (28, 32)],  // "lower" + "back"
                    "SYMPTOM",
                    "lower back",
                ),
            ],
            difficulty: Difficulty::Easy,
            domain: Domain::Biomedical,
        },
        DiscontinuousExample {
            text: "Aspirin and ibuprofen tablets were administered.".to_string(),
            entities: vec![
                DiscontinuousGold::new(
                    vec![(0, 7), (20, 27)],
                    "DRUG",
                    "Aspirin tablets",
                ),
                DiscontinuousGold::new(
                    vec![(12, 21), (20, 27)],
                    "DRUG",
                    "ibuprofen tablets",
                ),
            ],
            difficulty: Difficulty::Easy,
            domain: Domain::Biomedical,
        },
        DiscontinuousExample {
            text: "Type 1 and type 2 diabetes mellitus require different treatments.".to_string(),
            entities: vec![
                DiscontinuousGold::new(
                    vec![(0, 6), (18, 35)],  // "Type 1" + "diabetes mellitus"
                    "DISEASE",
                    "Type 1 diabetes mellitus",
                ),
                DiscontinuousGold::new(
                    vec![(11, 17), (18, 35)], // "type 2" + "diabetes mellitus"
                    "DISEASE",
                    "type 2 diabetes mellitus",
                ),
            ],
            difficulty: Difficulty::Medium,
            domain: Domain::Biomedical,
        },
    ]
}

/// Legal domain examples.
fn legal_domain() -> Vec<DiscontinuousExample> {
    vec![
        DiscontinuousExample {
            text: "Paragraphs 5 and 7 of Article III shall govern.".to_string(),
            entities: vec![
                DiscontinuousGold::new(
                    vec![(0, 10), (11, 12), (19, 33)],
                    "LEGAL_REF",
                    "Paragraphs 5 of Article III",
                ),
                DiscontinuousGold::new(
                    vec![(0, 10), (17, 18), (19, 33)],
                    "LEGAL_REF",
                    "Paragraphs 7 of Article III",
                ),
            ],
            difficulty: Difficulty::Medium,
            domain: Domain::Legal,
        },
        DiscontinuousExample {
            text: "The plaintiff and defendant counsel filed motions.".to_string(),
            entities: vec![
                DiscontinuousGold::new(
                    vec![(4, 13), (28, 35)],  // "plaintiff" + "counsel"
                    "LEGAL_ROLE",
                    "plaintiff counsel",
                ),
                DiscontinuousGold::new(
                    vec![(18, 27), (28, 35)], // "defendant" + "counsel"
                    "LEGAL_ROLE",
                    "defendant counsel",
                ),
            ],
            difficulty: Difficulty::Easy,
            domain: Domain::Legal,
        },
    ]
}

/// Get statistics about the discontinuous dataset.
pub fn stats() -> DiscontinuousStats {
    let all = dataset();
    let total_entities: usize = all.iter().map(|ex| ex.entities.len()).sum();
    let discontinuous_count = all
        .iter()
        .flat_map(|ex| &ex.entities)
        .filter(|e| !e.is_contiguous())
        .count();

    DiscontinuousStats {
        total_examples: all.len(),
        total_entities,
        discontinuous_entities: discontinuous_count,
        contiguous_entities: total_entities - discontinuous_count,
    }
}

/// Statistics about discontinuous dataset.
#[derive(Debug, Clone)]
pub struct DiscontinuousStats {
    /// Total number of examples.
    pub total_examples: usize,
    /// Total entities.
    pub total_entities: usize,
    /// Discontinuous entities.
    pub discontinuous_entities: usize,
    /// Contiguous entities.
    pub contiguous_entities: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_not_empty() {
        let examples = dataset();
        assert!(!examples.is_empty());
        assert!(examples.len() >= 10, "Should have at least 10 examples");
    }

    #[test]
    fn test_has_discontinuous_entities() {
        let examples = dataset();
        let has_discontinuous = examples
            .iter()
            .flat_map(|ex| &ex.entities)
            .any(|e| !e.is_contiguous());
        assert!(has_discontinuous, "Should have discontinuous entities");
    }

    #[test]
    fn test_entity_spans_valid() {
        for example in dataset() {
            for entity in &example.entities {
                for (start, end) in &entity.spans {
                    assert!(
                        *end <= example.text.len(),
                        "Entity span ({}, {}) exceeds text length {} in: {}",
                        start,
                        end,
                        example.text.len(),
                        example.text
                    );
                    assert!(
                        start < end,
                        "Invalid span ({}, {}) in: {}",
                        start,
                        end,
                        example.text
                    );
                }
            }
        }
    }

    #[test]
    fn test_stats() {
        let s = stats();
        assert!(s.total_examples > 0);
        assert!(s.total_entities > 0);
        assert!(s.discontinuous_entities > 0);
    }
}

