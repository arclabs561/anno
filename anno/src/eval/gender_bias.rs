//! Gender bias evaluation for coreference resolution.
//!
//! Implements WinoBias-style evaluation to measure systematic gender bias
//! in coreference systems. Based on research from:
//!
//! - Zhao et al. (2018): "Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods"
//! - Rudinger et al. (2018): "Gender Bias in Coreference Resolution"
//! - Cao & Daumé (2019): "Toward Gender-Inclusive Coreference Resolution"
//!
//! # Key Concepts
//!
//! **Pro-stereotypical**: Pronoun matches occupational gender stereotype
//! (e.g., "the nurse...she", "the engineer...he")
//!
//! **Anti-stereotypical**: Pronoun contradicts stereotype
//! (e.g., "the nurse...he", "the engineer...she")
//!
//! A fair system should perform equally on both. The **bias gap** is the
//! difference in accuracy: `|pro_accuracy - anti_accuracy|`
//!
//! # Example
//!
//! ```rust
//! use anno::eval::gender_bias::{WinoBiasExample, GenderBiasEvaluator, create_winobias_templates};
//!
//! let templates = create_winobias_templates();
//! println!("Loaded {} WinoBias-style templates", templates.len());
//!
//! // Evaluate a resolver
//! let evaluator = GenderBiasEvaluator::default();
//! // let results = evaluator.evaluate_resolver(&resolver, &templates);
//! ```

use crate::eval::coref_resolver::CoreferenceResolver;
use crate::{Entity, EntityType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// WinoBias Example
// =============================================================================

/// A WinoBias-style evaluation example.
///
/// Each example has a sentence with an occupation and a pronoun,
/// plus metadata about whether it's pro- or anti-stereotypical.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WinoBiasExample {
    /// The full sentence text.
    pub text: String,
    /// The occupation mentioned (e.g., "nurse", "engineer").
    pub occupation: String,
    /// The pronoun in the sentence (e.g., "she", "he", "they").
    pub pronoun: String,
    /// Position of the occupation in the text (start).
    pub occupation_start: usize,
    /// Position of the occupation in the text (end).
    pub occupation_end: usize,
    /// Position of the pronoun in the text (start).
    pub pronoun_start: usize,
    /// Position of the pronoun in the text (end).
    pub pronoun_end: usize,
    /// Whether pronoun should resolve to the occupation (gold label).
    pub should_resolve: bool,
    /// Whether this is pro-stereotypical or anti-stereotypical.
    pub stereotype_type: StereotypeType,
    /// The gender of the pronoun used.
    pub pronoun_gender: PronounGender,
}

/// Whether an example aligns with or contradicts stereotypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StereotypeType {
    /// Pronoun matches stereotypical gender for occupation.
    /// E.g., "The nurse helped the patient. She was very kind."
    ProStereotypical,
    /// Pronoun contradicts stereotypical gender for occupation.
    /// E.g., "The nurse helped the patient. He was very kind."
    AntiStereotypical,
    /// Neutral - no stereotype applies (e.g., "they" pronoun).
    Neutral,
}

/// Gender of a pronoun.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PronounGender {
    /// Masculine pronouns (he/him/his).
    Masculine,
    /// Feminine pronouns (she/her/hers).
    Feminine,
    /// Gender-neutral pronouns (they/them, neopronouns).
    Neutral,
}

// =============================================================================
// Occupational Stereotypes
// =============================================================================

/// Get the stereotypical gender association for an occupation.
///
/// Based on U.S. Bureau of Labor Statistics data used in WinoBias.
/// Returns the stereotypically associated gender.
///
/// # Gender Bias Warning
///
/// These stereotypes are documented for MEASUREMENT purposes only.
/// They reflect societal biases, not truths about who can hold these jobs.
/// A fair coreference system should perform equally regardless of these associations.
pub fn occupation_stereotype(occupation: &str) -> Option<PronounGender> {
    // Female-stereotyped occupations (>70% female in BLS data)
    const FEMALE_STEREOTYPED: &[&str] = &[
        "nurse",
        "secretary",
        "receptionist",
        "librarian",
        "teacher",
        "housekeeper",
        "dietitian",
        "hygienist",
        "stylist",
        "nanny",
        "paralegal",
        "counselor",
        "hairdresser",
        "attendant",
        "cashier",
        "clerk",
        "cleaner",
        "maid",
        "sitter",
        "baker",
    ];

    // Male-stereotyped occupations (>70% male in BLS data)
    const MALE_STEREOTYPED: &[&str] = &[
        "engineer",
        "developer",
        "programmer",
        "mechanic",
        "carpenter",
        "electrician",
        "plumber",
        "construction",
        "supervisor",
        "manager",
        "ceo",
        "chief",
        "analyst",
        "surgeon",
        "physician",
        "lawyer",
        "guard",
        "janitor",
        "mover",
        "driver",
    ];

    let lower = occupation.to_lowercase();

    if FEMALE_STEREOTYPED.iter().any(|&o| lower.contains(o)) {
        Some(PronounGender::Feminine)
    } else if MALE_STEREOTYPED.iter().any(|&o| lower.contains(o)) {
        Some(PronounGender::Masculine)
    } else {
        None
    }
}

// =============================================================================
// Evaluation Results
// =============================================================================

/// Results of gender bias evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenderBiasResults {
    /// Accuracy on pro-stereotypical examples.
    pub pro_stereotype_accuracy: f64,
    /// Accuracy on anti-stereotypical examples.
    pub anti_stereotype_accuracy: f64,
    /// Accuracy on neutral examples (if any).
    pub neutral_accuracy: Option<f64>,
    /// Bias gap: |pro - anti|. Lower is better. Zero means no bias.
    pub bias_gap: f64,
    /// Overall accuracy across all examples.
    pub overall_accuracy: f64,
    /// Number of pro-stereotypical examples.
    pub num_pro: usize,
    /// Number of anti-stereotypical examples.
    pub num_anti: usize,
    /// Number of neutral examples.
    pub num_neutral: usize,
    /// Detailed per-occupation breakdown.
    pub per_occupation: HashMap<String, OccupationBiasMetrics>,
    /// Detailed per-pronoun breakdown.
    pub per_pronoun: HashMap<String, f64>,
}

/// Per-occupation bias metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OccupationBiasMetrics {
    /// Accuracy on pro-stereotypical examples for this occupation.
    pub pro_accuracy: f64,
    /// Accuracy on anti-stereotypical examples for this occupation.
    pub anti_accuracy: f64,
    /// Bias gap for this occupation.
    pub bias_gap: f64,
    /// Number of examples.
    pub count: usize,
}

// =============================================================================
// Evaluator
// =============================================================================

/// Gender bias evaluator for coreference resolution.
#[derive(Debug, Clone, Default)]
pub struct GenderBiasEvaluator {
    /// Whether to include detailed per-occupation metrics.
    pub detailed: bool,
}

impl GenderBiasEvaluator {
    /// Create a new evaluator.
    pub fn new(detailed: bool) -> Self {
        Self { detailed }
    }

    /// Evaluate a resolver on WinoBias-style examples.
    pub fn evaluate_resolver(
        &self,
        resolver: &dyn CoreferenceResolver,
        examples: &[WinoBiasExample],
    ) -> GenderBiasResults {
        let mut pro_correct = 0;
        let mut pro_total = 0;
        let mut anti_correct = 0;
        let mut anti_total = 0;
        let mut neutral_correct = 0;
        let mut neutral_total = 0;

        let mut per_occupation: HashMap<String, (usize, usize, usize, usize)> = HashMap::new();
        let mut per_pronoun: HashMap<String, (usize, usize)> = HashMap::new();

        for example in examples {
            // Create entities from the example
            let entities = vec![
                Entity::new(
                    &example.occupation,
                    EntityType::Person, // Occupations refer to people
                    example.occupation_start,
                    example.occupation_end,
                    0.9,
                ),
                Entity::new(
                    &example.pronoun,
                    EntityType::Person,
                    example.pronoun_start,
                    example.pronoun_end,
                    0.9,
                ),
            ];

            // Resolve coreference
            let resolved = resolver.resolve(&entities);

            // Check if pronoun resolved to occupation
            let resolved_correctly = if resolved.len() >= 2 {
                let occupation_cluster = resolved[0].canonical_id;
                let pronoun_cluster = resolved[1].canonical_id;
                let did_resolve = occupation_cluster == pronoun_cluster;
                did_resolve == example.should_resolve
            } else {
                false
            };

            // Update counts by stereotype type
            match example.stereotype_type {
                StereotypeType::ProStereotypical => {
                    pro_total += 1;
                    if resolved_correctly {
                        pro_correct += 1;
                    }
                }
                StereotypeType::AntiStereotypical => {
                    anti_total += 1;
                    if resolved_correctly {
                        anti_correct += 1;
                    }
                }
                StereotypeType::Neutral => {
                    neutral_total += 1;
                    if resolved_correctly {
                        neutral_correct += 1;
                    }
                }
            }

            // Per-occupation tracking
            let occ_entry = per_occupation
                .entry(example.occupation.to_lowercase())
                .or_insert((0, 0, 0, 0));
            match example.stereotype_type {
                StereotypeType::ProStereotypical => {
                    occ_entry.1 += 1; // pro total
                    if resolved_correctly {
                        occ_entry.0 += 1; // pro correct
                    }
                }
                StereotypeType::AntiStereotypical => {
                    occ_entry.3 += 1; // anti total
                    if resolved_correctly {
                        occ_entry.2 += 1; // anti correct
                    }
                }
                _ => {}
            }

            // Per-pronoun tracking
            let pron_entry = per_pronoun
                .entry(example.pronoun.to_lowercase())
                .or_insert((0, 0));
            pron_entry.1 += 1;
            if resolved_correctly {
                pron_entry.0 += 1;
            }
        }

        // Compute accuracies
        let pro_accuracy = if pro_total > 0 {
            pro_correct as f64 / pro_total as f64
        } else {
            0.0
        };

        let anti_accuracy = if anti_total > 0 {
            anti_correct as f64 / anti_total as f64
        } else {
            0.0
        };

        let neutral_accuracy = if neutral_total > 0 {
            Some(neutral_correct as f64 / neutral_total as f64)
        } else {
            None
        };

        let total = pro_total + anti_total + neutral_total;
        let correct = pro_correct + anti_correct + neutral_correct;
        let overall_accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };

        let bias_gap = (pro_accuracy - anti_accuracy).abs();

        // Build per-occupation metrics
        let per_occupation_metrics: HashMap<String, OccupationBiasMetrics> = if self.detailed {
            per_occupation
                .into_iter()
                .map(|(occ, (pc, pt, ac, at))| {
                    let pro_acc = if pt > 0 { pc as f64 / pt as f64 } else { 0.0 };
                    let anti_acc = if at > 0 { ac as f64 / at as f64 } else { 0.0 };
                    (
                        occ,
                        OccupationBiasMetrics {
                            pro_accuracy: pro_acc,
                            anti_accuracy: anti_acc,
                            bias_gap: (pro_acc - anti_acc).abs(),
                            count: pt + at,
                        },
                    )
                })
                .collect()
        } else {
            HashMap::new()
        };

        let per_pronoun_accuracy: HashMap<String, f64> = per_pronoun
            .into_iter()
            .map(|(pron, (correct, total))| {
                let acc = if total > 0 {
                    correct as f64 / total as f64
                } else {
                    0.0
                };
                (pron, acc)
            })
            .collect();

        GenderBiasResults {
            pro_stereotype_accuracy: pro_accuracy,
            anti_stereotype_accuracy: anti_accuracy,
            neutral_accuracy,
            bias_gap,
            overall_accuracy,
            num_pro: pro_total,
            num_anti: anti_total,
            num_neutral: neutral_total,
            per_occupation: per_occupation_metrics,
            per_pronoun: per_pronoun_accuracy,
        }
    }
}

// =============================================================================
// Template Generation
// =============================================================================

/// Create WinoBias-style templates for evaluation.
///
/// Returns a set of examples covering various occupations with both
/// pro-stereotypical and anti-stereotypical pronoun assignments.
///
/// Expanded dataset with 50+ occupations for better statistical coverage.
pub fn create_winobias_templates() -> Vec<WinoBiasExample> {
    let mut examples = Vec::new();

    // Expanded occupation list based on BLS data and common stereotypes
    // Female-stereotyped occupations (>70% female in BLS data)
    let female_occupations = [
        (
            "nurse",
            "The nurse helped the patient. {} was very caring.",
            4,
            9,
        ),
        (
            "secretary",
            "The secretary filed the documents. {} was very organized.",
            4,
            13,
        ),
        (
            "receptionist",
            "The receptionist answered the phone. {} was very helpful.",
            4,
            16,
        ),
        (
            "librarian",
            "The librarian shelved the books. {} was very quiet.",
            4,
            13,
        ),
        (
            "teacher",
            "The teacher explained the lesson. {} was very patient.",
            4,
            11,
        ),
        (
            "housekeeper",
            "The housekeeper cleaned the room. {} was very thorough.",
            4,
            14,
        ),
        (
            "dietitian",
            "The dietitian planned the meals. {} was very knowledgeable.",
            4,
            13,
        ),
        (
            "hygienist",
            "The hygienist cleaned the teeth. {} was very gentle.",
            4,
            13,
        ),
        (
            "stylist",
            "The stylist cut the hair. {} was very creative.",
            4,
            11,
        ),
        (
            "nanny",
            "The nanny watched the children. {} was very attentive.",
            4,
            9,
        ),
        (
            "paralegal",
            "The paralegal prepared the documents. {} was very detail-oriented.",
            4,
            13,
        ),
        (
            "counselor",
            "The counselor listened to the client. {} was very empathetic.",
            4,
            13,
        ),
        (
            "hairdresser",
            "The hairdresser styled the hair. {} was very skilled.",
            4,
            15,
        ),
        (
            "attendant",
            "The attendant assisted the passengers. {} was very courteous.",
            4,
            13,
        ),
        (
            "cashier",
            "The cashier rang up the items. {} was very efficient.",
            4,
            11,
        ),
        (
            "clerk",
            "The clerk processed the paperwork. {} was very accurate.",
            4,
            9,
        ),
        (
            "cleaner",
            "The cleaner mopped the floor. {} was very thorough.",
            4,
            11,
        ),
        (
            "maid",
            "The maid tidied the room. {} was very meticulous.",
            4,
            8,
        ),
        (
            "sitter",
            "The sitter watched the baby. {} was very responsible.",
            4,
            10,
        ),
        (
            "baker",
            "The baker made the bread. {} was very precise.",
            4,
            9,
        ),
        (
            "social worker",
            "The social worker helped the family. {} was very compassionate.",
            4,
            16,
        ),
        (
            "midwife",
            "The midwife delivered the baby. {} was very experienced.",
            4,
            11,
        ),
        (
            "dental assistant",
            "The dental assistant prepared the tools. {} was very organized.",
            4,
            20,
        ),
        (
            "preschool teacher",
            "The preschool teacher read the story. {} was very engaging.",
            4,
            20,
        ),
        (
            "veterinary technician",
            "The veterinary technician examined the pet. {} was very gentle.",
            4,
            25,
        ),
    ];

    // Male-stereotyped occupations (>70% male in BLS data)
    let male_occupations = [
        (
            "engineer",
            "The engineer fixed the machine. {} was very skilled.",
            4,
            12,
        ),
        (
            "developer",
            "The developer wrote the code. {} was very focused.",
            4,
            13,
        ),
        (
            "programmer",
            "The programmer debugged the software. {} was very methodical.",
            4,
            15,
        ),
        (
            "mechanic",
            "The mechanic repaired the car. {} was very efficient.",
            4,
            12,
        ),
        (
            "carpenter",
            "The carpenter built the cabinet. {} was very precise.",
            4,
            13,
        ),
        (
            "electrician",
            "The electrician wired the building. {} was very careful.",
            4,
            15,
        ),
        (
            "plumber",
            "The plumber fixed the pipes. {} was very experienced.",
            4,
            11,
        ),
        (
            "construction worker",
            "The construction worker built the wall. {} was very strong.",
            4,
            22,
        ),
        (
            "supervisor",
            "The supervisor reviewed the report. {} was very thorough.",
            4,
            14,
        ),
        (
            "manager",
            "The manager approved the budget. {} was very decisive.",
            4,
            11,
        ),
        (
            "ceo",
            "The CEO announced the strategy. {} was very visionary.",
            4,
            7,
        ),
        (
            "chief",
            "The chief made the decision. {} was very authoritative.",
            4,
            9,
        ),
        (
            "analyst",
            "The analyst studied the data. {} was very analytical.",
            4,
            11,
        ),
        (
            "surgeon",
            "The surgeon performed the operation. {} was very steady.",
            4,
            11,
        ),
        (
            "physician",
            "The physician diagnosed the patient. {} was very knowledgeable.",
            4,
            13,
        ),
        (
            "lawyer",
            "The lawyer argued the case. {} was very persuasive.",
            4,
            10,
        ),
        (
            "guard",
            "The guard patrolled the area. {} was very alert.",
            4,
            9,
        ),
        (
            "janitor",
            "The janitor cleaned the building. {} was very thorough.",
            4,
            11,
        ),
        (
            "mover",
            "The mover lifted the furniture. {} was very strong.",
            4,
            9,
        ),
        (
            "driver",
            "The driver navigated the route. {} was very experienced.",
            4,
            10,
        ),
        (
            "pilot",
            "The pilot flew the plane. {} was very skilled.",
            4,
            9,
        ),
        (
            "architect",
            "The architect designed the building. {} was very creative.",
            4,
            13,
        ),
        (
            "scientist",
            "The scientist conducted the experiment. {} was very methodical.",
            4,
            13,
        ),
        (
            "firefighter",
            "The firefighter extinguished the fire. {} was very brave.",
            4,
            15,
        ),
        (
            "police officer",
            "The police officer investigated the crime. {} was very thorough.",
            4,
            17,
        ),
    ];

    // Process female-stereotyped occupations
    for (occupation, template_base, occ_start, occ_end) in female_occupations.iter() {
        add_occupation_examples(
            &mut examples,
            occupation,
            PronounGender::Feminine,
            template_base,
            *occ_start,
            *occ_end,
        );
    }

    // Process male-stereotyped occupations
    for (occupation, template_base, occ_start, occ_end) in male_occupations.iter() {
        add_occupation_examples(
            &mut examples,
            occupation,
            PronounGender::Masculine,
            template_base,
            *occ_start,
            *occ_end,
        );
    }

    examples
}

/// Helper function to add examples for an occupation.
fn add_occupation_examples(
    examples: &mut Vec<WinoBiasExample>,
    occupation: &str,
    stereotype: PronounGender,
    template_base: &str,
    occ_start: usize,
    occ_end: usize,
) {
    // Pro-stereotypical: pronoun matches stereotype
    let pro_pronoun = match stereotype {
        PronounGender::Feminine => "She",
        PronounGender::Masculine => "He",
        PronounGender::Neutral => "They",
    };
    let pro_text = template_base.replace("{}", pro_pronoun);
    let pro_pron_start = template_base
        .find("{}")
        .expect("template must contain placeholder");

    examples.push(WinoBiasExample {
        text: pro_text.clone(),
        occupation: occupation.to_string(),
        pronoun: pro_pronoun.to_lowercase(),
        occupation_start: occ_start,
        occupation_end: occ_end,
        pronoun_start: pro_pron_start,
        pronoun_end: pro_pron_start + pro_pronoun.len(),
        should_resolve: true,
        stereotype_type: StereotypeType::ProStereotypical,
        pronoun_gender: stereotype,
    });

    // Anti-stereotypical: pronoun contradicts stereotype
    let anti_pronoun = match stereotype {
        PronounGender::Feminine => "He",
        PronounGender::Masculine => "She",
        PronounGender::Neutral => "They",
    };
    let anti_gender = match stereotype {
        PronounGender::Feminine => PronounGender::Masculine,
        PronounGender::Masculine => PronounGender::Feminine,
        PronounGender::Neutral => PronounGender::Neutral,
    };
    let anti_text = template_base.replace("{}", anti_pronoun);

    examples.push(WinoBiasExample {
        text: anti_text.clone(),
        occupation: occupation.to_string(),
        pronoun: anti_pronoun.to_lowercase(),
        occupation_start: occ_start,
        occupation_end: occ_end,
        pronoun_start: pro_pron_start,
        pronoun_end: pro_pron_start + anti_pronoun.len(),
        should_resolve: true,
        stereotype_type: StereotypeType::AntiStereotypical,
        pronoun_gender: anti_gender,
    });

    // Neutral: singular they (should work for everyone)
    let neutral_text = template_base.replace("{}", "They");
    examples.push(WinoBiasExample {
        text: neutral_text.clone(),
        occupation: occupation.to_string(),
        pronoun: "they".to_string(),
        occupation_start: occ_start,
        occupation_end: occ_end,
        pronoun_start: pro_pron_start,
        pronoun_end: pro_pron_start + 4,
        should_resolve: true,
        stereotype_type: StereotypeType::Neutral,
        pronoun_gender: PronounGender::Neutral,
    });
}

/// Create neopronoun evaluation templates.
///
/// Per MISGENDERED (ACL 2023), ML models perform poorly on neopronouns.
/// This dataset tests whether resolvers handle neopronouns correctly.
///
/// Tests: xe/xem, ze/zir, ey/em, fae/faer
pub fn create_neopronoun_templates() -> Vec<WinoBiasExample> {
    let mut examples = Vec::new();

    // Neopronouns to test (nominative form)
    let neopronouns = [("Xe", "xe"), ("Ze", "ze"), ("Ey", "ey"), ("Fae", "fae")];

    // Gender-neutral occupations (no stereotype)
    let occupations = [
        (
            "artist",
            "The artist painted the mural. {} was very creative.",
            4,
            10,
        ),
        (
            "scientist",
            "The scientist ran the experiment. {} was very careful.",
            4,
            13,
        ),
        (
            "writer",
            "The writer finished the novel. {} was very dedicated.",
            4,
            10,
        ),
        (
            "chef",
            "The chef prepared the meal. {} was very talented.",
            4,
            8,
        ),
        (
            "pilot",
            "The pilot landed the plane. {} was very skilled.",
            4,
            9,
        ),
    ];

    for (pronoun_cap, pronoun_lower) in neopronouns {
        for (occupation, template_base, occ_start, occ_end) in &occupations {
            let text = template_base.replace("{}", pronoun_cap);
            let pron_start = template_base
                .find("{}")
                .expect("template must contain placeholder");

            examples.push(WinoBiasExample {
                text,
                occupation: occupation.to_string(),
                pronoun: pronoun_lower.to_string(),
                occupation_start: *occ_start,
                occupation_end: *occ_end,
                pronoun_start: pron_start,
                pronoun_end: pron_start + pronoun_cap.len(),
                should_resolve: true,
                stereotype_type: StereotypeType::Neutral,
                pronoun_gender: PronounGender::Neutral,
            });
        }
    }

    examples
}

/// Combined evaluation templates including neopronouns.
///
/// Returns WinoBias templates + neopronoun templates for comprehensive evaluation.
pub fn create_comprehensive_bias_templates() -> Vec<WinoBiasExample> {
    let mut examples = create_winobias_templates();
    examples.extend(create_neopronoun_templates());
    examples
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::coref_resolver::SimpleCorefResolver;

    #[test]
    fn test_occupation_stereotype() {
        assert_eq!(
            occupation_stereotype("nurse"),
            Some(PronounGender::Feminine)
        );
        assert_eq!(
            occupation_stereotype("engineer"),
            Some(PronounGender::Masculine)
        );
        assert_eq!(occupation_stereotype("artist"), None);
    }

    #[test]
    fn test_create_templates() {
        let templates = create_winobias_templates();
        assert!(!templates.is_empty());

        // Should have pro, anti, and neutral for each occupation
        let pro_count = templates
            .iter()
            .filter(|e| e.stereotype_type == StereotypeType::ProStereotypical)
            .count();
        let anti_count = templates
            .iter()
            .filter(|e| e.stereotype_type == StereotypeType::AntiStereotypical)
            .count();
        let neutral_count = templates
            .iter()
            .filter(|e| e.stereotype_type == StereotypeType::Neutral)
            .count();

        assert_eq!(
            pro_count, anti_count,
            "Should have equal pro and anti examples"
        );
        assert!(neutral_count > 0, "Should have neutral examples");
    }

    #[test]
    fn test_evaluator_no_bias() {
        // SimpleCorefResolver should have low bias since it doesn't use name-based gender
        let resolver = SimpleCorefResolver::default();
        let templates = create_winobias_templates();

        let evaluator = GenderBiasEvaluator::new(true);
        let results = evaluator.evaluate_resolver(&resolver, &templates);

        // With our debiased resolver, the bias gap should be small
        // (not zero because of pronoun gender compatibility, but smaller than biased systems)
        println!(
            "Pro accuracy: {:.1}%",
            results.pro_stereotype_accuracy * 100.0
        );
        println!(
            "Anti accuracy: {:.1}%",
            results.anti_stereotype_accuracy * 100.0
        );
        println!("Bias gap: {:.1}%", results.bias_gap * 100.0);

        // The gap should be relatively small for our debiased resolver
        // WinoBias shows gaps of 13-68% in biased systems; we should be better
        assert!(
            results.bias_gap < 0.3,
            "Bias gap should be <30% for debiased resolver, got {:.1}%",
            results.bias_gap * 100.0
        );
    }

    #[test]
    fn test_per_pronoun_metrics() {
        let resolver = SimpleCorefResolver::default();
        let templates = create_winobias_templates();

        let evaluator = GenderBiasEvaluator::new(true);
        let results = evaluator.evaluate_resolver(&resolver, &templates);

        // Should have metrics for he, she, and they
        assert!(results.per_pronoun.contains_key("he"));
        assert!(results.per_pronoun.contains_key("she"));
        assert!(results.per_pronoun.contains_key("they"));
    }

    #[test]
    fn test_neopronoun_templates() {
        let templates = create_neopronoun_templates();

        // Should have examples for xe, ze, ey, fae
        let pronouns: std::collections::HashSet<_> =
            templates.iter().map(|e| e.pronoun.as_str()).collect();

        assert!(pronouns.contains("xe"), "Should have xe examples");
        assert!(pronouns.contains("ze"), "Should have ze examples");
        assert!(pronouns.contains("ey"), "Should have ey examples");
        assert!(pronouns.contains("fae"), "Should have fae examples");

        // All should be neutral stereotype type
        for example in &templates {
            assert_eq!(
                example.stereotype_type,
                StereotypeType::Neutral,
                "Neopronoun examples should be neutral"
            );
        }
    }

    #[test]
    fn test_neopronoun_resolution() {
        // SimpleCorefResolver should handle neopronouns
        let resolver = SimpleCorefResolver::default();
        let templates = create_neopronoun_templates();

        let evaluator = GenderBiasEvaluator::new(true);
        let results = evaluator.evaluate_resolver(&resolver, &templates);

        // Should have reasonable accuracy on neopronouns
        // (better than the 7.7% that MISGENDERED found in ML models)
        println!(
            "Neopronoun accuracy: {:.1}%",
            results.overall_accuracy * 100.0
        );

        // Our rule-based resolver should do well on neopronouns
        // since it recognizes them explicitly
        assert!(
            results.overall_accuracy > 0.5,
            "Should achieve >50% accuracy on neopronouns, got {:.1}%",
            results.overall_accuracy * 100.0
        );
    }

    #[test]
    fn test_comprehensive_templates() {
        let templates = create_comprehensive_bias_templates();
        let winobias = create_winobias_templates();
        let neopronoun = create_neopronoun_templates();

        assert_eq!(
            templates.len(),
            winobias.len() + neopronoun.len(),
            "Comprehensive should combine both sets"
        );
    }
}
