//! Biomedical/healthcare domain synthetic data.

use super::super::types::helpers::{chemical, disease, drug, entity, gene};
use super::super::types::{AnnotatedExample, Difficulty, Domain};
use anno_core::EntityType;

/// Biomedical/healthcare domain dataset.
pub fn dataset() -> Vec<AnnotatedExample> {
    vec![
        // Easy: Organizations
        AnnotatedExample {
            text: "Pfizer's COVID-19 vaccine Comirnaty received FDA approval for boosters.".into(),
            entities: vec![
                entity("Pfizer", EntityType::Organization, 0),
                entity("FDA", EntityType::Organization, 45),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Researchers at Johns Hopkins University published findings on Alzheimer's treatment."
                .into(),
            entities: vec![entity("Johns Hopkins University", EntityType::Organization, 15)],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The Mayo Clinic and Cleveland Clinic collaborated on the heart disease study.".into(),
            entities: vec![
                entity("Mayo Clinic", EntityType::Organization, 4),
                entity("Cleveland Clinic", EntityType::Organization, 20),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Medium,
        },
        // Medium: Diseases
        AnnotatedExample {
            text: "The patient was diagnosed with Type 2 diabetes mellitus and hypertension.".into(),
            entities: vec![
                disease("Type 2 diabetes mellitus", 31),
                disease("hypertension", 60),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Metformin is the first-line treatment for diabetes. Lisinopril helps control blood pressure."
                .into(),
            entities: vec![
                drug("Metformin", 0),
                disease("diabetes", 42),
                drug("Lisinopril", 52),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Medium,
        },
        // Hard: Genes and chemicals
        AnnotatedExample {
            text: "Mutations in BRCA1 and BRCA2 genes increase breast cancer risk significantly.".into(),
            entities: vec![
                gene("BRCA1", 13),
                gene("BRCA2", 23),
                disease("breast cancer", 44),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "The p53 tumor suppressor gene regulates cell cycle and apoptosis.".into(),
            entities: vec![gene("p53", 4)],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "Acetylsalicylic acid (aspirin) inhibits COX-1 and COX-2 enzymes.".into(),
            entities: vec![
                chemical("Acetylsalicylic acid", 0),
                drug("aspirin", 22),
                gene("COX-1", 40),
                gene("COX-2", 50),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Hard,
        },
        // Adversarial: Complex medical terminology
        AnnotatedExample {
            text: "The TP53-mutated non-small cell lung carcinoma showed response to pembrolizumab immunotherapy."
                .into(),
            entities: vec![
                gene("TP53", 4),
                disease("non-small cell lung carcinoma", 17),
                drug("pembrolizumab", 66),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Adversarial,
        },
        AnnotatedExample {
            text: "Rheumatoid arthritis patients receiving tocilizumab showed decreased IL-6 levels.".into(),
            entities: vec![
                disease("Rheumatoid arthritis", 0),
                drug("tocilizumab", 40),
                chemical("IL-6", 69),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Adversarial,
        },
        AnnotatedExample {
            text: "EGFR T790M mutation confers resistance to first-generation EGFR-TKIs like gefitinib."
                .into(),
            entities: vec![
                gene("EGFR", 0),
                gene("EGFR-TKIs", 59),
                drug("gefitinib", 74),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Adversarial,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biomedical_dataset_not_empty() {
        assert!(!dataset().is_empty());
    }

    #[test]
    fn test_all_biomedical_domain() {
        for ex in dataset() {
            assert_eq!(ex.domain, Domain::Biomedical);
        }
    }
}
