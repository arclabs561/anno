//! Synthetic relation extraction examples.
//!
//! # Overview
//!
//! Relation extraction identifies semantic relationships between entity pairs:
//! - **Employment**: Person WORKS_FOR Organization
//! - **Foundation**: Person FOUNDED Organization  
//! - **Location**: Entity LOCATED_IN Location
//! - **Family**: Person SIBLING/PARENT_OF Person
//!
//! # Research Alignment
//!
//! From DocRED (arXiv:1906.06127):
//! > "Document-level relation extraction requires integrating
//! > information across sentences."
//!
//! Benchmark datasets: TACRED (~70% F1), DocRED (~60% F1), SciERC (~45% F1)

use crate::eval::relation::RelationGold;

/// A synthetic example with entities and relations.
#[derive(Debug, Clone)]
pub struct RelationExample {
    /// The text
    pub text: String,
    /// Gold standard relations
    pub relations: Vec<RelationGold>,
    /// Difficulty level
    pub difficulty: Difficulty,
    /// Domain
    pub domain: Domain,
}

/// Difficulty level for relation examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Difficulty {
    /// Single relation, clear trigger
    Easy,
    /// Multiple relations in one sentence
    Medium,
    /// Implicit relations, long distance
    Hard,
}

/// Domain for relation examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// General news
    General,
    /// Business/corporate
    Business,
    /// Scientific/academic
    Scientific,
    /// Biographical
    Biography,
}

/// Generate all relation extraction synthetic examples.
pub fn dataset() -> Vec<RelationExample> {
    let mut examples = Vec::new();

    // Easy: Single relation with clear trigger
    examples.extend(easy_relations());

    // Medium: Multiple relations
    examples.extend(medium_relations());

    // Hard: Implicit and long-distance
    examples.extend(hard_relations());

    // Domain-specific
    examples.extend(business_domain());
    examples.extend(scientific_domain());
    examples.extend(biography_domain());

    examples
}

/// Easy: Single relation with explicit trigger word.
fn easy_relations() -> Vec<RelationExample> {
    vec![
        RelationExample {
            text: "Steve Jobs founded Apple in 1976.".to_string(),
            relations: vec![RelationGold::new(
                (0, 10),
                "PER",
                "Steve Jobs",
                (19, 24),
                "ORG",
                "Apple",
                "FOUNDED",
            )],
            difficulty: Difficulty::Easy,
            domain: Domain::Business,
        },
        RelationExample {
            text: "Mary works for Google in California.".to_string(),
            relations: vec![
                RelationGold::new(
                    (0, 4),
                    "PER",
                    "Mary",
                    (15, 21),
                    "ORG",
                    "Google",
                    "WORKS_FOR",
                ),
                RelationGold::new(
                    (15, 21),
                    "ORG",
                    "Google",
                    (25, 35),
                    "LOC",
                    "California",
                    "LOCATED_IN",
                ),
            ],
            difficulty: Difficulty::Easy,
            domain: Domain::Business,
        },
        RelationExample {
            text: "The Eiffel Tower is located in Paris, France.".to_string(),
            relations: vec![
                RelationGold::new(
                    (4, 16),
                    "LOC",
                    "Eiffel Tower",
                    (31, 36),
                    "LOC",
                    "Paris",
                    "LOCATED_IN",
                ),
                RelationGold::new(
                    (31, 36),
                    "LOC",
                    "Paris",
                    (38, 44),
                    "LOC",
                    "France",
                    "LOCATED_IN",
                ),
            ],
            difficulty: Difficulty::Easy,
            domain: Domain::General,
        },
        RelationExample {
            text: "Tim Cook is the CEO of Apple Inc.".to_string(),
            relations: vec![RelationGold::new(
                (0, 8),
                "PER",
                "Tim Cook",
                (23, 32),
                "ORG",
                "Apple Inc",
                "CEO_OF",
            )],
            difficulty: Difficulty::Easy,
            domain: Domain::Business,
        },
        RelationExample {
            text: "Amazon acquired Whole Foods in 2017.".to_string(),
            relations: vec![RelationGold::new(
                (0, 6),
                "ORG",
                "Amazon",
                (16, 27),
                "ORG",
                "Whole Foods",
                "ACQUIRED",
            )],
            difficulty: Difficulty::Easy,
            domain: Domain::Business,
        },
    ]
}

/// Medium: Multiple relations or more complex triggers.
fn medium_relations() -> Vec<RelationExample> {
    vec![
        RelationExample {
            text: "Bill Gates and Paul Allen co-founded Microsoft in Seattle.".to_string(),
            relations: vec![
                RelationGold::new(
                    (0, 10),
                    "PER",
                    "Bill Gates",
                    (37, 46),
                    "ORG",
                    "Microsoft",
                    "FOUNDED",
                ),
                RelationGold::new(
                    (15, 25),
                    "PER",
                    "Paul Allen",
                    (37, 46),
                    "ORG",
                    "Microsoft",
                    "FOUNDED",
                ),
                RelationGold::new(
                    (37, 46),
                    "ORG",
                    "Microsoft",
                    (50, 57),
                    "LOC",
                    "Seattle",
                    "LOCATED_IN",
                ),
            ],
            difficulty: Difficulty::Medium,
            domain: Domain::Business,
        },
        RelationExample {
            text: "Dr. Smith at Harvard published research with Dr. Jones from MIT.".to_string(),
            relations: vec![
                RelationGold::new(
                    (0, 9),
                    "PER",
                    "Dr. Smith",
                    (13, 20),
                    "ORG",
                    "Harvard",
                    "AFFILIATED_WITH",
                ),
                RelationGold::new(
                    (45, 54),
                    "PER",
                    "Dr. Jones",
                    (60, 63),
                    "ORG",
                    "MIT",
                    "AFFILIATED_WITH",
                ),
                RelationGold::new(
                    (0, 9),
                    "PER",
                    "Dr. Smith",
                    (45, 54),
                    "PER",
                    "Dr. Jones",
                    "COLLABORATED_WITH",
                ),
            ],
            difficulty: Difficulty::Medium,
            domain: Domain::Scientific,
        },
        RelationExample {
            text: "Alice, Bob's sister, married Charlie, who works at IBM.".to_string(),
            relations: vec![
                RelationGold::new((0, 5), "PER", "Alice", (7, 10), "PER", "Bob", "SIBLING_OF"),
                RelationGold::new(
                    (0, 5),
                    "PER",
                    "Alice",
                    (29, 36),
                    "PER",
                    "Charlie",
                    "SPOUSE_OF",
                ),
                RelationGold::new(
                    (29, 36),
                    "PER",
                    "Charlie",
                    (51, 54),
                    "ORG",
                    "IBM",
                    "WORKS_FOR",
                ),
            ],
            difficulty: Difficulty::Medium,
            domain: Domain::Biography,
        },
    ]
}

/// Hard: Implicit relations or long-distance dependencies.
fn hard_relations() -> Vec<RelationExample> {
    vec![
        RelationExample {
            // Implicit relation - no trigger word
            text: "Sundar Pichai, born in India, leads Google's AI efforts.".to_string(),
            relations: vec![
                RelationGold::new(
                    (0, 13), "PER", "Sundar Pichai",
                    (23, 28), "LOC", "India",
                    "BORN_IN",
                ),
                RelationGold::new(
                    (0, 13), "PER", "Sundar Pichai",
                    (36, 42), "ORG", "Google",
                    "WORKS_FOR",
                ),
            ],
            difficulty: Difficulty::Hard,
            domain: Domain::Biography,
        },
        RelationExample {
            // Long-distance relation
            text: "The company, which was established in 1998 by Larry Page and Sergey Brin, is headquartered in Mountain View.".to_string(),
            relations: vec![
                RelationGold::new(
                    (45, 55), "PER", "Larry Page",
                    (0, 11), "ORG", "The company",
                    "FOUNDED",
                ),
                RelationGold::new(
                    (60, 71), "PER", "Sergey Brin",
                    (0, 11), "ORG", "The company",
                    "FOUNDED",
                ),
                RelationGold::new(
                    (0, 11), "ORG", "The company",
                    (92, 105), "LOC", "Mountain View",
                    "LOCATED_IN",
                ),
            ],
            difficulty: Difficulty::Hard,
            domain: Domain::Business,
        },
    ]
}

/// Business domain examples.
fn business_domain() -> Vec<RelationExample> {
    vec![
        RelationExample {
            text: "Nvidia, led by Jensen Huang, designs chips in Santa Clara.".to_string(),
            relations: vec![
                RelationGold::new(
                    (15, 27),
                    "PER",
                    "Jensen Huang",
                    (0, 6),
                    "ORG",
                    "Nvidia",
                    "CEO_OF",
                ),
                RelationGold::new(
                    (0, 6),
                    "ORG",
                    "Nvidia",
                    (46, 57),
                    "LOC",
                    "Santa Clara",
                    "LOCATED_IN",
                ),
            ],
            difficulty: Difficulty::Medium,
            domain: Domain::Business,
        },
        RelationExample {
            text: "Netflix is headquartered in Los Gatos, California.".to_string(),
            relations: vec![
                RelationGold::new(
                    (0, 7),
                    "ORG",
                    "Netflix",
                    (28, 37),
                    "LOC",
                    "Los Gatos",
                    "LOCATED_IN",
                ),
                RelationGold::new(
                    (28, 37),
                    "LOC",
                    "Los Gatos",
                    (39, 49),
                    "LOC",
                    "California",
                    "LOCATED_IN",
                ),
            ],
            difficulty: Difficulty::Easy,
            domain: Domain::Business,
        },
    ]
}

/// Scientific domain examples.
fn scientific_domain() -> Vec<RelationExample> {
    vec![
        RelationExample {
            text: "CRISPR was developed by Jennifer Doudna at UC Berkeley.".to_string(),
            relations: vec![
                RelationGold::new(
                    (24, 39),
                    "PER",
                    "Jennifer Doudna",
                    (0, 6),
                    "MISC",
                    "CRISPR",
                    "DEVELOPED",
                ),
                RelationGold::new(
                    (24, 39),
                    "PER",
                    "Jennifer Doudna",
                    (43, 54),
                    "ORG",
                    "UC Berkeley",
                    "AFFILIATED_WITH",
                ),
            ],
            difficulty: Difficulty::Medium,
            domain: Domain::Scientific,
        },
        RelationExample {
            text: "Einstein published the theory of relativity while at the Swiss Patent Office."
                .to_string(),
            relations: vec![
                RelationGold::new(
                    (0, 8),
                    "PER",
                    "Einstein",
                    (23, 44),
                    "MISC",
                    "theory of relativity",
                    "AUTHORED",
                ),
                RelationGold::new(
                    (0, 8),
                    "PER",
                    "Einstein",
                    (58, 77),
                    "ORG",
                    "Swiss Patent Office",
                    "WORKS_FOR",
                ),
            ],
            difficulty: Difficulty::Medium,
            domain: Domain::Scientific,
        },
    ]
}

/// Biographical domain examples.
fn biography_domain() -> Vec<RelationExample> {
    vec![
        RelationExample {
            text: "Barack Obama, born in Honolulu, served as the 44th President.".to_string(),
            relations: vec![RelationGold::new(
                (0, 12),
                "PER",
                "Barack Obama",
                (22, 30),
                "LOC",
                "Honolulu",
                "BORN_IN",
            )],
            difficulty: Difficulty::Easy,
            domain: Domain::Biography,
        },
        RelationExample {
            text: "Marie Curie and Pierre Curie, her husband, won the Nobel Prize.".to_string(),
            relations: vec![RelationGold::new(
                (0, 11),
                "PER",
                "Marie Curie",
                (16, 28),
                "PER",
                "Pierre Curie",
                "SPOUSE_OF",
            )],
            difficulty: Difficulty::Medium,
            domain: Domain::Biography,
        },
    ]
}

/// Get statistics about the relation dataset.
pub fn stats() -> RelationStats {
    let all = dataset();
    let total_relations: usize = all.iter().map(|ex| ex.relations.len()).sum();

    let mut relation_types = std::collections::HashMap::new();
    for ex in &all {
        for rel in &ex.relations {
            *relation_types.entry(rel.relation_type.clone()).or_insert(0) += 1;
        }
    }

    RelationStats {
        total_examples: all.len(),
        total_relations,
        relation_types,
    }
}

/// Statistics about relation dataset.
#[derive(Debug, Clone)]
pub struct RelationStats {
    /// Total number of examples.
    pub total_examples: usize,
    /// Total relations.
    pub total_relations: usize,
    /// Count per relation type.
    pub relation_types: std::collections::HashMap<String, usize>,
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
    fn test_has_multiple_relation_types() {
        let s = stats();
        assert!(
            s.relation_types.len() >= 5,
            "Should have at least 5 relation types"
        );
    }

    #[test]
    fn test_entity_spans_valid() {
        for example in dataset() {
            for rel in &example.relations {
                assert!(
                    rel.head_span.1 <= example.text.len(),
                    "Head span exceeds text length in: {}",
                    example.text
                );
                assert!(
                    rel.tail_span.1 <= example.text.len(),
                    "Tail span exceeds text length in: {}",
                    example.text
                );
                assert!(
                    rel.head_span.0 < rel.head_span.1,
                    "Invalid head span in: {}",
                    example.text
                );
                assert!(
                    rel.tail_span.0 < rel.tail_span.1,
                    "Invalid tail span in: {}",
                    example.text
                );
            }
        }
    }

    #[test]
    fn test_stats() {
        let s = stats();
        assert!(s.total_examples > 0);
        assert!(s.total_relations > 0);
        assert!(!s.relation_types.is_empty());
    }
}
