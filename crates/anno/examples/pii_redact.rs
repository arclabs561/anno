/// Detect and redact personally identifiable information.
///
/// Combines NER-based PII classification with regex pattern scanning
/// to find names, SSNs, emails, phone numbers, and other PII.
///
/// ```sh
/// cargo run --example pii_redact
/// ```
///
/// Example output:
///
/// ```text
/// Redacted: [PERSON_1]'s SSN is [ID_NUMBER_1]. Email: [CONTACT_1].
/// ```
fn main() -> anno::Result<()> {
    use anno::{pii, Model, StackedNER};

    let text = "John Smith's SSN is 123-45-6789. Email: john@example.com.";
    let m = StackedNER::default();

    // One-liner: scan + redact
    let redacted = pii::scan_and_redact(text, &m)?;
    println!("Redacted: {}", redacted);

    // Or step by step for more control:
    let entities = m.extract_entities(text, None)?;
    let mut pii_entities: Vec<pii::PiiEntity> =
        entities.iter().filter_map(pii::classify_entity).collect();
    pii_entities.extend(pii::scan_patterns(text));

    println!("\nFound {} PII entities:", pii_entities.len());
    for p in &pii_entities {
        println!(
            "  {} \"{}\" [{}] ({},{})",
            p.pii_type, p.text, p.risk_level, p.start, p.end
        );
    }

    // redact() handles overlapping/duplicate spans automatically
    let redacted = pii::redact(text, &pii_entities);
    println!("\nRedacted: {}", redacted);

    Ok(())
}
