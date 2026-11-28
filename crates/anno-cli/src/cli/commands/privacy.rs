//! Privacy command - Detect and redact/pseudonymize PII for compliance
//!
//! Entity extraction is dual-use: the same tool that builds knowledge graphs
//! can de-anonymize documents. This command helps identify and manage PII.

use clap::{Parser, ValueEnum};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::super::output::color;
use super::super::parser::ModelBackend;
use anno::pii::{
    classify_entity as classify_pii, scan_patterns as scan_structured_pii, PiiEntity as PIIEntity,
    PiiReport as PIIReport,
};
use anno::Model;
#[cfg(test)]
use anno::{
    pii::{looks_like_address, looks_like_id_number},
    Entity,
};

/// Privacy analysis and redaction
#[derive(Parser, Debug)]
#[command(group(clap::ArgGroup::new("input_source").args(["text", "input"]).multiple(false)))]
pub struct PrivacyArgs {
    /// Read UTF-8 text from this file
    #[arg(short, long, value_name = "PATH")]
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

    /// Model backend. Defaults to stacked, or prepared GLiNER2-PII plus
    /// local contact rules for an explicit CONTACT-only request.
    #[arg(short, long)]
    pub model: Option<ModelBackend>,

    /// Export PII map for re-identification (with pseudonymize).
    /// The map must be separate from both the input and the pseudonymized output.
    /// On Unix, maps use mode 0600 and symlink targets are rejected.
    /// Existing access-control lists are not changed.
    #[arg(long)]
    pub export_map: Option<PathBuf>,

    /// Comma-separated PII types to target (default: all). CONTACT includes email and phone.
    #[arg(long, value_delimiter = ',', ignore_case = true,
          value_parser = ["PERSON", "DOB", "ADDRESS", "CONTACT", "ID_NUMBER"])]
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
    /// Replace PII with type tokens such as `[CONTACT_1]` and `[ID_NUMBER_1]`
    Redact,
    /// Replace with consistent fake values
    Pseudonymize,
}

/// Run the privacy command.
pub fn run(args: PrivacyArgs) -> Result<(), String> {
    let mut input_id = None;
    let text = if let Some(path) = &args.input {
        let mut file = fs::File::open(path).map_err(|e| format!("Failed to open input: {e}"))?;
        let text =
            std::io::read_to_string(&mut file).map_err(|e| format!("Failed to read file: {e}"))?;
        if matches!(args.action, PrivacyAction::Pseudonymize) && args.export_map.is_some() {
            input_id = Some(same_file::Handle::from_file(file).map_err(|e| e.to_string())?);
        }
        text
    } else if let Some(t) = &args.text {
        t.clone()
    } else {
        return Err("No input provided. Use --input or --text".into());
    };

    let pii_entities = if super::model_routing::automatic_contact_rules(args.model, &args.types) {
        default_contacts(&text, args.quiet)?
    } else {
        let model = args.model.unwrap_or_default().create_model()?;
        let entities = model
            .extract_entities(&text, None)
            .map_err(|e| format!("Extraction failed: {e}"))?;
        let mut pii = scan_structured_pii(&text);
        for entity in entities.iter().filter_map(classify_pii) {
            if !pii
                .iter()
                .any(|existing| existing.start <= entity.start && existing.end >= entity.end)
            {
                pii.push(entity);
            }
        }
        pii.sort_by_key(|entity| entity.start);
        pii.retain(|entity| {
            args.types.is_empty()
                || args
                    .types
                    .iter()
                    .any(|t| t.eq_ignore_ascii_case(&entity.pii_type))
        });
        pii
    };

    // Generate report
    let report = anno::pii::report(&pii_entities);

    match args.action {
        PrivacyAction::Report => {
            print_pii_report(&report, args.quiet);
        }
        PrivacyAction::Redact => {
            let redacted = redact_text(&text, pii_entities);
            if let Some(path) = &args.output {
                fs::write(path, &redacted).map_err(|e| format!("Failed to write: {}", e))?;
                if !args.quiet {
                    eprintln!(
                        "{} Redacted {} PII entities, saved to {:?}",
                        color("32", "✓"),
                        report.entities.len(),
                        path
                    );
                }
            } else {
                println!("{}", redacted);
            }
        }
        PrivacyAction::Pseudonymize => {
            let (pseudonymized, mapping) = pseudonymize_text(&text, &pii_entities);
            if let (Some(output), Some(map)) = (&args.output, &args.export_map) {
                if super::import::same_entry(output, map)? {
                    return Err("--output and --export-map must be different files".into());
                }
            }
            let map_file = args
                .export_map
                .as_deref()
                .map(open_pii_map)
                .transpose()
                .map_err(|e| format!("Failed to open map: {e}"))?;
            let identity =
                |file: &fs::File| file.try_clone().and_then(same_file::Handle::from_file);
            let map_id = map_file
                .as_ref()
                .map(identity)
                .transpose()
                .map_err(|e| e.to_string())?;
            if let (Some(input), Some(map)) = (&input_id, &map_id) {
                if input == map {
                    return Err("--input and --export-map must be different files".into());
                }
            }
            let output = args
                .output
                .as_deref()
                .map(|path| {
                    fs::OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(false)
                        .open(path)
                })
                .transpose()
                .map_err(|e| format!("Failed to open output: {e}"))?;
            if let Some(map) = &map_id {
                let output_id = match &output {
                    Some(file) => identity(file),
                    None => same_file::Handle::stdout(),
                }
                .map_err(|e| e.to_string())?;
                if &output_id == map {
                    return Err(
                        "Pseudonymized output and --export-map must be different files".into(),
                    );
                }
            }
            if let (Some(path), Some(mut output)) = (&args.output, output) {
                if output.metadata().map_err(|e| e.to_string())?.is_file() {
                    output
                        .set_len(0)
                        .map_err(|e| format!("Failed to truncate output: {e}"))?;
                }
                output
                    .write_all(pseudonymized.as_bytes())
                    .map_err(|e| format!("Failed to write: {e}"))?;
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
            if let (Some(map_path), Some(map_file)) = (&args.export_map, map_file) {
                let map_content = mapping
                    .iter()
                    .map(|(orig, fake)| format!("{}\t{}", orig, fake))
                    .collect::<Vec<_>>()
                    .join("\n");
                write_pii_map(map_file, &map_content)
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

fn open_pii_map(path: &Path) -> std::io::Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Do not follow a symlink or block opening a FIFO supplied as a map path.
        options
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "PII map destination must be a regular file",
        ));
    }
    Ok(file)
}

fn write_pii_map(mut file: fs::File, content: &str) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Creation mode does not change an existing file's permissions.
        // Restrict the opened file before truncating or writing sensitive data.
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    file.set_len(0)?;
    file.write_all(content.as_bytes())
}

/// Contact-only defaults combine local rule protections with any prepared
/// specialist. Unioning overlapping spans preserves all source characters
/// protected by the rules when the model predicts a shorter or partial span.
fn default_contacts(text: &str, quiet: bool) -> Result<Vec<PIIEntity>, String> {
    #[cfg(not(feature = "gliner2-fastino"))]
    let _ = quiet;
    let rule_entities = anno::RegexNER::new()
        .extract_entities(text, None)
        .map_err(|e| format!("Contact extraction failed: {e}"))?;
    let mut contacts = scan_structured_pii(text);
    contacts.extend(rule_entities.iter().filter_map(classify_pii));
    contacts.retain(|entity| entity.pii_type == "CONTACT");
    #[cfg(feature = "gliner2-fastino")]
    {
        use anno::backends::gliner2_fastino::GLiNER2Fastino;
        let extracted = GLiNER2Fastino::from_prepared_pii()
            .and_then(|model| model.map(|m| m.extract_entities(text, None)).transpose());
        match extracted {
            Ok(Some(entities)) => {
                contacts.extend(
                    entities
                        .iter()
                        .filter_map(classify_pii)
                        .filter(|entity| entity.pii_type == "CONTACT"),
                );
            }
            Ok(None) => {}
            Err(error) if !quiet => {
                eprintln!(
                    "warning: prepared GLiNER2-PII failed; using local contact rules: {error}"
                );
            }
            Err(_) => {}
        }
    }
    Ok(replacement_spans(text, &contacts))
}

/// Convert a character-offset span to byte offsets suitable for string slicing.
fn char_span_to_byte_range(text: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    if start > end {
        return None;
    }
    let char_count = text.chars().count();
    if end > char_count {
        return None;
    }
    let converter = anno::offset::SpanConverter::new(text);
    Some((converter.char_to_byte(start), converter.char_to_byte(end)))
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
            report.identifier_risk
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

    let risk_color = if report.identifier_risk.starts_with("CRITICAL") {
        "31"
    } else if report.identifier_risk.starts_with("HIGH") {
        "33"
    } else {
        "32"
    };
    println!(
        "{}: {}",
        color("1;33", "Identifier Risk"),
        color(risk_color, &report.identifier_risk)
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

/// Merge overlapping detections before replacement, including partial overlaps from
/// different backends. The union prevents a second edit from slicing a replacement.
fn replacement_spans(text: &str, entities: &[PIIEntity]) -> Vec<PIIEntity> {
    let mut sorted = entities.to_vec();
    sorted.sort_by_key(|entity| (entity.start, std::cmp::Reverse(entity.end)));
    let mut spans: Vec<PIIEntity> = Vec::new();
    for entity in sorted {
        if entity.start >= entity.end
            || char_span_to_byte_range(text, entity.start, entity.end).is_none()
        {
            continue;
        }
        if let Some(previous) = spans.last_mut() {
            if entity.start < previous.end {
                previous.end = previous.end.max(entity.end);
                previous.text = text
                    .chars()
                    .skip(previous.start)
                    .take(previous.end - previous.start)
                    .collect();
                continue;
            }
        }
        spans.push(entity);
    }
    spans
}

fn redact_text(text: &str, mut entities: Vec<PIIEntity>) -> String {
    // CLI detectors may return invalid coordinates. Keep the CLI's skip
    // behavior before handing valid spans to the shared redactor.
    let char_count = text.chars().count();
    entities.retain(|e| e.start < e.end && e.end <= char_count);
    anno::pii::redact(text, &entities)
}

fn pseudonymize_text(text: &str, entities: &[PIIEntity]) -> (String, HashMap<String, String>) {
    let mut result = text.to_string();
    let mut mapping: HashMap<String, String> = HashMap::new();
    let mut counters = HashMap::new();

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

    let spans = replacement_spans(text, entities);
    for entity in spans.iter().rev() {
        let Some((start, end)) = char_span_to_byte_range(text, entity.start, entity.end) else {
            continue;
        };
        let fake = if let Some(existing) = mapping.get(&entity.text) {
            existing.clone()
        } else {
            let kind = match entity.pii_type.as_str() {
                "CONTACT" if entity.text.contains('@') => "EMAIL",
                other => other,
            };
            let counter = counters.entry(kind).or_insert(0usize);
            let fake = loop {
                *counter += 1;
                let n = *counter;
                let candidate = match kind {
                    "PERSON" if n <= fake_names.len() => fake_names[n - 1].to_string(),
                    "PERSON" => format!("Person {n}"),
                    "DOB" => format!("{}-01-{:02}", 1990 + n / 28, n % 28 + 1),
                    "ADDRESS" => format!("{} Main St", 100 + n),
                    "EMAIL" if n == 1 => "contact@example.com".to_string(),
                    "EMAIL" => format!("contact{n}@example.com"),
                    "CONTACT" => format!("555-000-{:04}", 1000 + n),
                    "ID_NUMBER" if n == 1 => "XXX-XX-XXXX".to_string(),
                    "ID_NUMBER" => format!("XXX-XX-{n:04}"),
                    _ => format!("[{kind}_{n}]"),
                };
                if !text.contains(&candidate) {
                    break candidate;
                }
            };
            mapping.insert(entity.text.clone(), fake.clone());
            fake
        };

        result.replace_range(start..end, &fake);
    }

    (result, mapping)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssn_detected_by_pre_scan() {
        let pii = scan_structured_pii("My SSN is 123-45-6789 and that's it.");
        assert!(
            pii.iter().any(|p| p.text == "123-45-6789"),
            "SSN should be detected: {:?}",
            pii
        );
    }

    #[test]
    fn credit_card_detected_by_pre_scan() {
        let pii = scan_structured_pii("Card: 4111-1111-1111-1111 on file.");
        assert!(
            pii.iter().any(|p| p.text == "4111-1111-1111-1111"),
            "Credit card should be detected: {:?}",
            pii
        );
    }

    #[test]
    fn credit_card_no_separators() {
        let pii = scan_structured_pii("Card: 4111111111111111 on file.");
        assert!(
            pii.iter()
                .any(|p| p.pii_type == "ID_NUMBER" && p.text.contains("4111")),
            "Credit card without separators should be detected: {:?}",
            pii
        );
    }

    #[test]
    fn common_word_not_id_number() {
        assert!(
            !looks_like_id_number("Chemistry"),
            "Chemistry should NOT be flagged as ID number"
        );
    }

    #[test]
    fn mrn_with_digit_detected() {
        assert!(
            looks_like_id_number("ABC123"),
            "ABC123 should be detected as MRN"
        );
    }

    #[test]
    fn email_detected_by_pre_scan() {
        let pii = scan_structured_pii("Contact me at bob@example.com please.");
        assert!(
            pii.iter().any(|p| p.pii_type == "CONTACT"),
            "Email should be detected as CONTACT: {:?}",
            pii
        );
    }

    #[test]
    fn overlapping_unicode_detections_are_replaced_once() {
        let text = "éé Alice Smith!";
        let person = |start, end| PIIEntity {
            text: text.chars().skip(start).take(end - start).collect(),
            pii_type: "PERSON".into(),
            start,
            end,
            risk_level: "HIGH".into(),
        };
        let detections = vec![person(3, 8), person(6, 14), person(3, 8)];
        assert_eq!(redact_text(text, detections.clone()), "éé [PERSON_1]!");
        let (pseudonymized, mapping) = pseudonymize_text(text, &detections);
        assert_eq!(pseudonymized, "éé John Smith!");
        assert_eq!(mapping.len(), 1);
        assert_eq!(mapping["Alice Smith"], "John Smith");
    }

    #[test]
    fn intersecting_contact_detections_protect_the_whole_original_number() {
        let text = "📞 +49 1234567!";
        let contacts = [(2, 10), (3, 13), (4, 8)].map(|(start, end)| PIIEntity {
            pii_type: "CONTACT".into(),
            text: text.chars().skip(start).take(end - start).collect(),
            start,
            end,
            risk_level: "HIGH".into(),
        });
        let merged = replacement_spans(text, &contacts);
        assert_eq!(merged.len(), 1);
        assert_eq!((merged[0].start, merged[0].end), (2, 13));
        assert_eq!(redact_text(text, merged.clone()), "📞 [CONTACT_1]!");
        let (pseudonymized, mapping) = pseudonymize_text(text, &merged);
        assert_eq!(pseudonymized, "📞 555-000-1001!");
        assert_eq!(mapping["+49 1234567"], "555-000-1001");
    }

    #[test]
    fn structured_scan_uses_character_offsets_after_unicode_prefix() {
        let text = "東京の連絡先: bob@example.com";
        let entity = scan_structured_pii(text)
            .into_iter()
            .find(|entity| entity.text == "bob@example.com")
            .expect("email should be detected");

        assert_eq!(
            entity.start,
            text[..text.find("bob@example.com").unwrap()]
                .chars()
                .count()
        );
        assert_eq!(entity.end, text.chars().count());
    }

    #[test]
    fn redaction_preserves_unicode_prefix() {
        let text = "東京の連絡先: bob@example.com";
        let entities = scan_structured_pii(text);

        assert_eq!(redact_text(text, entities), "東京の連絡先: [CONTACT_1]");
    }

    #[test]
    fn pseudonymization_preserves_unicode_prefix() {
        let text = "東京の連絡先: bob@example.com";
        let entities = scan_structured_pii(text);
        let (result, mapping) = pseudonymize_text(text, &entities);

        assert_eq!(result, "東京の連絡先: contact@example.com");
        assert_eq!(
            mapping.get("bob@example.com"),
            Some(&"contact@example.com".to_string())
        );
    }

    #[test]
    fn redact_skips_invalid_offsets_and_only_classifies_whole_address_indicators() {
        let source = "😀 Studio 54 or 123 Main St.";
        let venue = Entity::new("Studio 54", anno::EntityType::Location, 2, 11, 0.95);
        assert!(classify_pii(&venue).is_none());
        let address = Entity::new("123 Main St.", anno::EntityType::Location, 15, 27, 0.95);
        let valid = classify_pii(&address).unwrap();
        let spans = [(8, 8), (11, 1), (0, 28), (15, 27)].map(|(start, end)| PIIEntity {
            start,
            end,
            ..valid.clone()
        });
        assert_eq!(
            redact_text(source, spans.to_vec()),
            "😀 Studio 54 or [ADDRESS_1]"
        );
    }

    #[test]
    fn address_with_zip_detected() {
        assert!(
            looks_like_address("1234 Elm Street, Springfield, IL 62704"),
            "Address with street and ZIP should be detected"
        );
    }

    #[test]
    fn address_zip_and_state_only() {
        assert!(
            looks_like_address("Springfield, IL 62704"),
            "ZIP + state abbreviation should be detected as address"
        );
    }

    #[test]
    fn pseudonymize_phone_gets_phone_replacement() {
        let entities = vec![
            PIIEntity {
                text: "bob@example.com".to_string(),
                pii_type: "CONTACT".to_string(),
                start: 0,
                end: 15,
                risk_level: "HIGH".to_string(),
            },
            PIIEntity {
                text: "555-867-5309".to_string(),
                pii_type: "CONTACT".to_string(),
                start: 20,
                end: 32,
                risk_level: "HIGH".to_string(),
            },
        ];
        let text = "bob@example.com --- 555-867-5309";
        let (result, mapping) = pseudonymize_text(text, &entities);

        // Email should get email-like replacement
        let email_replacement = mapping.get("bob@example.com").unwrap();
        assert!(
            email_replacement.contains('@'),
            "Email should get email-like replacement, got: {}",
            email_replacement
        );

        // Phone should get phone-like replacement (555-000-XXXX)
        let phone_replacement = mapping.get("555-867-5309").unwrap();
        assert!(
            phone_replacement.starts_with("555-000-"),
            "Phone should get phone-like replacement, got: {}",
            phone_replacement
        );
        assert!(
            !phone_replacement.contains('@'),
            "Phone replacement should not look like email, got: {}",
            phone_replacement
        );
        // Verify the replacement actually appears in the output
        assert!(
            result.contains(phone_replacement),
            "Output should contain phone replacement"
        );
    }

    #[test]
    fn iban_detected() {
        assert!(
            looks_like_id_number("DE89370400440532013000"),
            "IBAN should be detected as ID number"
        );
    }
}
