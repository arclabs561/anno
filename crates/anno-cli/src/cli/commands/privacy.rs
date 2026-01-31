//! Privacy command - Detect and redact/pseudonymize PII for compliance
//!
//! Entity extraction is dual-use: the same tool that builds knowledge graphs
//! can de-anonymize documents. This command helps identify and manage PII.

use clap::{Parser, ValueEnum};
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use super::super::output::color;
use super::super::parser::ModelBackend;
use anno_core::Entity;

/// Privacy analysis and redaction
#[derive(Parser, Debug)]
pub struct PrivacyArgs {
    /// Input file or text
    #[arg(short, long)]
    pub input: Option<PathBuf>,

    /// Input text directly
    #[arg(short, long)]
    pub text: Option<String>,

    /// Action to perform
    #[arg(short, long, default_value = "report")]
    pub action: PrivacyAction,

    /// Output file (for redact/pseudonymize)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Model backend
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Export PII map for re-identification (with pseudonymize)
    #[arg(long)]
    pub export_map: Option<PathBuf>,

    /// PII types to target (default: all)
    #[arg(long, value_delimiter = ',')]
    pub types: Vec<String>,

    /// Quiet mode
    #[arg(short, long)]
    pub quiet: bool,
}

/// Privacy action
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum PrivacyAction {
    /// Report PII detected (no modification)
    #[default]
    Report,
    /// Replace PII with type tokens (\[PERSON_1\], \[DATE_3\], etc.)
    Redact,
    /// Replace with consistent fake values
    Pseudonymize,
}

/// PII detection result.
///
/// Summary of personally identifiable information found in text.
#[derive(Debug, Clone)]
pub struct PIIReport {
    /// Count of person name entities
    pub person_count: usize,
    /// Count of date/time entities
    pub date_count: usize,
    /// Count of location entities
    pub location_count: usize,
    /// Count of contact info (email, phone, etc.)
    pub contact_count: usize,
    /// Count of ID numbers (SSN, passport, etc.)
    pub id_number_count: usize,
    /// Detailed list of PII entities
    pub entities: Vec<PIIEntity>,
    /// Assessment of re-identification risk
    pub k_anonymity_risk: String,
}

/// A detected PII entity.
#[derive(Debug, Clone)]
pub struct PIIEntity {
    /// The PII text
    pub text: String,
    /// Type of PII (PERSON, DATE, EMAIL, etc.)
    pub pii_type: String,
    /// Start byte offset
    pub start: usize,
    /// End byte offset (exclusive)
    pub end: usize,
    /// Risk level (low, medium, high)
    pub risk_level: String,
}

/// Run the privacy command.
pub fn run(args: PrivacyArgs) -> Result<(), String> {
    // Get input text
    let text = if let Some(path) = &args.input {
        fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?
    } else if let Some(t) = &args.text {
        t.clone()
    } else {
        return Err("No input provided. Use --input or --text".into());
    };

    // Create model and extract entities
    let model = args.model.create_model()?;
    let entities = model
        .extract_entities(&text, None)
        .map_err(|e| format!("Extraction failed: {}", e))?;

    // Classify entities as PII
    let pii_entities: Vec<PIIEntity> = entities
        .iter()
        .filter_map(classify_pii)
        .filter(|pii| {
            args.types.is_empty()
                || args
                    .types
                    .iter()
                    .any(|t| t.eq_ignore_ascii_case(&pii.pii_type))
        })
        .collect();

    // Generate report
    let report = generate_pii_report(&pii_entities);

    match args.action {
        PrivacyAction::Report => {
            print_pii_report(&report, args.quiet);
        }
        PrivacyAction::Redact => {
            let redacted = redact_text(&text, &pii_entities);
            if let Some(path) = &args.output {
                fs::write(path, &redacted).map_err(|e| format!("Failed to write: {}", e))?;
                if !args.quiet {
                    eprintln!(
                        "{} Redacted {} PII entities, saved to {:?}",
                        color("32", "✓"),
                        pii_entities.len(),
                        path
                    );
                }
            } else {
                println!("{}", redacted);
            }
        }
        PrivacyAction::Pseudonymize => {
            let (pseudonymized, mapping) = pseudonymize_text(&text, &pii_entities);
            if let Some(path) = &args.output {
                fs::write(path, &pseudonymized).map_err(|e| format!("Failed to write: {}", e))?;
                if !args.quiet {
                    eprintln!(
                        "{} Pseudonymized {} PII entities, saved to {:?}",
                        color("32", "✓"),
                        pii_entities.len(),
                        path
                    );
                }
            } else {
                println!("{}", pseudonymized);
            }

            // Export mapping if requested
            if let Some(map_path) = &args.export_map {
                let map_content = mapping
                    .iter()
                    .map(|(orig, fake)| format!("{}\t{}", orig, fake))
                    .collect::<Vec<_>>()
                    .join("\n");
                fs::write(map_path, map_content)
                    .map_err(|e| format!("Failed to write map: {}", e))?;
                if !args.quiet {
                    eprintln!(
                        "{} Exported {} mappings to {:?}",
                        color("32", "✓"),
                        mapping.len(),
                        map_path
                    );
                }
            }
        }
    }

    Ok(())
}

/// Classify an entity as PII
fn classify_pii(entity: &Entity) -> Option<PIIEntity> {
    let label = entity.entity_type.as_label();
    let text = &entity.text;

    let (pii_type, risk_level) = match label {
        "PER" | "PERSON" => ("PERSON", assess_person_risk(text)),
        "DATE" => {
            // DOB patterns are high risk
            if looks_like_dob(text) {
                ("DOB", "HIGH")
            } else {
                return None; // Regular dates aren't PII
            }
        }
        "LOC" | "GPE" | "LOCATION" => {
            // Addresses are PII, general locations are not
            if looks_like_address(text) {
                ("ADDRESS", "HIGH")
            } else {
                return None;
            }
        }
        "EMAIL" => ("CONTACT", "HIGH"),
        "PHONE" => ("CONTACT", "HIGH"),
        "URL" => return None,   // URLs generally not PII
        "MONEY" => return None, // Money amounts generally not PII
        _ => {
            // Check for ID patterns
            if looks_like_id_number(text) {
                ("ID_NUMBER", "CRITICAL")
            } else {
                return None;
            }
        }
    };

    Some(PIIEntity {
        text: text.clone(),
        pii_type: pii_type.to_string(),
        start: entity.start,
        end: entity.end,
        risk_level: risk_level.to_string(),
    })
}

fn assess_person_risk(text: &str) -> &'static str {
    // Full names with titles are higher risk
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() >= 3 {
        "HIGH"
    } else if words.len() == 2 {
        "MEDIUM"
    } else {
        "LOW"
    }
}

fn looks_like_dob(text: &str) -> bool {
    // Simple heuristic: dates with year in 1900-2010 range could be DOBs
    if let Ok(year_pattern) = Regex::new(r"19[0-9]{2}|20[0-1][0-9]") {
        year_pattern.is_match(text)
    } else {
        false
    }
}

fn looks_like_address(text: &str) -> bool {
    // Contains number + street indicator
    let has_number = text.chars().any(|c| c.is_numeric());
    let street_indicators = [
        "St", "Street", "Ave", "Avenue", "Rd", "Road", "Blvd", "Dr", "Lane", "Ln",
    ];
    let has_street = street_indicators.iter().any(|ind| text.contains(ind));
    has_number && has_street
}

fn looks_like_id_number(text: &str) -> bool {
    // SSN pattern: XXX-XX-XXXX
    if let Ok(ssn_pattern) = Regex::new(r"\d{3}-\d{2}-\d{4}") {
        if ssn_pattern.is_match(text) {
            return true;
        }
    }
    // MRN pattern: alphanumeric 6-10 chars
    if text.len() >= 6 && text.len() <= 10 && text.chars().all(|c| c.is_alphanumeric()) {
        return true;
    }
    false
}

fn generate_pii_report(entities: &[PIIEntity]) -> PIIReport {
    let mut person_count = 0;
    let mut date_count = 0;
    let mut location_count = 0;
    let mut contact_count = 0;
    let mut id_number_count = 0;

    for e in entities {
        match e.pii_type.as_str() {
            "PERSON" => person_count += 1,
            "DOB" => date_count += 1,
            "ADDRESS" => location_count += 1,
            "CONTACT" => contact_count += 1,
            "ID_NUMBER" => id_number_count += 1,
            _ => {}
        }
    }

    // Simple k-anonymity risk assessment
    let unique_names: std::collections::HashSet<_> = entities
        .iter()
        .filter(|e| e.pii_type == "PERSON")
        .map(|e| e.text.to_lowercase())
        .collect();

    let k_anonymity_risk = if id_number_count > 0 {
        "CRITICAL (direct identifiers present)"
    } else if unique_names.len() > 5 && date_count > 0 && location_count > 0 {
        "HIGH (quasi-identifier combination)"
    } else if unique_names.len() > 3 {
        "MEDIUM (multiple names)"
    } else {
        "LOW"
    };

    PIIReport {
        person_count,
        date_count,
        location_count,
        contact_count,
        id_number_count,
        entities: entities.to_vec(),
        k_anonymity_risk: k_anonymity_risk.to_string(),
    }
}

fn print_pii_report(report: &PIIReport, quiet: bool) {
    if quiet {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            report.person_count,
            report.date_count,
            report.location_count,
            report.contact_count,
            report.id_number_count,
            report.k_anonymity_risk
        );
        return;
    }

    println!("{}", color("1;36", "PII Detection Report"));
    println!();

    println!("{}:", color("1;33", "Summary"));
    println!("  PERSON:     {} names", report.person_count);
    println!("  DATE (DOB): {} potential DOBs", report.date_count);
    println!("  ADDRESS:    {} addresses", report.location_count);
    println!("  CONTACT:    {} emails/phones", report.contact_count);
    println!("  ID_NUMBER:  {} identifiers", report.id_number_count);
    println!();

    let risk_color = if report.k_anonymity_risk.starts_with("CRITICAL") {
        "31"
    } else if report.k_anonymity_risk.starts_with("HIGH") {
        "33"
    } else {
        "32"
    };
    println!(
        "{}: {}",
        color("1;33", "k-Anonymity Risk"),
        color(risk_color, &report.k_anonymity_risk)
    );
    println!();

    if !report.entities.is_empty() {
        println!("{}:", color("1;33", "Entities"));
        for e in &report.entities {
            let risk_col = match e.risk_level.as_str() {
                "CRITICAL" => "31",
                "HIGH" => "33",
                "MEDIUM" => "35",
                _ => "90",
            };
            println!(
                "  {} \"{}\" @{}:{} [{}]",
                color("36", &e.pii_type),
                e.text,
                e.start,
                e.end,
                color(risk_col, &e.risk_level)
            );
        }
    }

    println!();
    println!(
        "Actions: {} | {} | {}",
        color("90", "--action redact"),
        color("90", "--action pseudonymize"),
        color("90", "--export-map")
    );
}

fn redact_text(text: &str, entities: &[PIIEntity]) -> String {
    let mut result = text.to_string();
    let mut type_counts: HashMap<&str, usize> = HashMap::new();

    // Sort by start position descending to preserve offsets
    let mut sorted: Vec<_> = entities.iter().collect();
    sorted.sort_by(|a, b| b.start.cmp(&a.start));

    for entity in sorted {
        let count = type_counts.entry(&entity.pii_type).or_insert(0);
        *count += 1;
        let replacement = format!("[{}_{}]", entity.pii_type, count);
        result.replace_range(entity.start..entity.end, &replacement);
    }

    result
}

fn pseudonymize_text(text: &str, entities: &[PIIEntity]) -> (String, HashMap<String, String>) {
    let mut result = text.to_string();
    let mut mapping: HashMap<String, String> = HashMap::new();
    let mut name_counter = 0;
    let mut date_counter = 0;
    let mut addr_counter = 0;

    // Fake names pool
    let fake_names = [
        "John Smith",
        "Jane Doe",
        "Alex Johnson",
        "Sam Williams",
        "Chris Brown",
        "Pat Davis",
        "Jordan Miller",
        "Taylor Wilson",
        "Morgan Lee",
        "Casey Martinez",
    ];

    // Sort by start position descending
    let mut sorted: Vec<_> = entities.iter().collect();
    sorted.sort_by(|a, b| b.start.cmp(&a.start));

    for entity in sorted {
        let fake = if let Some(existing) = mapping.get(&entity.text) {
            existing.clone()
        } else {
            let fake = match entity.pii_type.as_str() {
                "PERSON" => {
                    let name = fake_names[name_counter % fake_names.len()];
                    name_counter += 1;
                    name.to_string()
                }
                "DOB" => {
                    date_counter += 1;
                    format!("1990-01-{:02}", (date_counter % 28) + 1)
                }
                "ADDRESS" => {
                    addr_counter += 1;
                    format!("{} Main St", 100 + addr_counter)
                }
                "CONTACT" => "contact@example.com".to_string(),
                "ID_NUMBER" => "XXX-XX-XXXX".to_string(),
                _ => "[REDACTED]".to_string(),
            };
            mapping.insert(entity.text.clone(), fake.clone());
            fake
        };

        result.replace_range(entity.start..entity.end, &fake);
    }

    (result, mapping)
}
