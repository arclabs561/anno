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
/// Found 3 PII entities:
///   PERSON "John Smith" [HIGH] (0,10)
///   ID_NUMBER "123-45-6789" [CRITICAL] (22,33)
///   CONTACT "john@example.com" [MEDIUM] (45,61)
///
/// Redacted: [REDACTED]'s SSN is [REDACTED]. Email: [REDACTED].
/// ```
fn main() -> anno::Result<()> {
    use anno::{pii, Model, StackedNER};

    let text = "John Smith's SSN is 123-45-6789. Email: john@example.com.";

    // Step 1: Extract entities with NER
    let m = StackedNER::default();
    let entities = m.extract_entities(text, None)?;

    // Step 2: Classify NER entities as PII
    let mut pii_entities: Vec<pii::PiiEntity> =
        entities.iter().filter_map(pii::classify_entity).collect();

    // Step 3: Scan for structured PII patterns (SSN, email, phone, etc.)
    pii_entities.extend(pii::scan_patterns(text));

    // Step 4: Report
    println!("Found {} PII entities:", pii_entities.len());
    for p in &pii_entities {
        println!(
            "  {} \"{}\" [{}] ({},{})",
            p.pii_type, p.text, p.risk_level, p.start, p.end
        );
    }

    // Step 5: Redact
    let redacted = pii::redact(text, &pii_entities);
    println!("\nRedacted: {}", redacted);

    Ok(())
}
