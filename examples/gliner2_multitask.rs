//! GLiNER2 Multi-Task Information Extraction Example
//!
//! Demonstrates GLiNER2's capabilities for:
//! - Zero-shot Named Entity Recognition
//! - Text Classification (single & multi-label)
//! - Hierarchical Structure Extraction
//! - Task Composition (multiple tasks in one pass)
//!
//! Run with: `cargo run --example gliner2_multitask --features candle`

#[cfg(any(feature = "onnx", feature = "candle"))]
use anno::backends::gliner2::{FieldType, StructureTask, TaskSchema};
#[cfg(any(feature = "onnx", feature = "candle"))]
use anno::Result;
#[cfg(any(feature = "onnx", feature = "candle"))]
use std::collections::HashMap;

#[cfg(any(feature = "onnx", feature = "candle"))]
fn main() -> Result<()> {
    println!("=== GLiNER2 Multi-Task Information Extraction ===\n");

    // =========================================================================
    // Example 1: Financial News Analysis
    // =========================================================================
    println!("--- Example 1: Financial News Analysis ---\n");

    let financial_text = r#"
        Apple Inc. announced quarterly revenue of $89.5 billion on November 2, 2023.
        CEO Tim Cook stated that iPhone sales exceeded expectations in China.
        The company plans to invest $430 million in new manufacturing facilities.
        Analysts at Goldman Sachs maintain a Buy rating with a price target of $200.
    "#;

    // Build a multi-task schema for financial analysis
    let financial_schema = TaskSchema::new()
        // NER: Extract financial entities with descriptions
        .with_entities_described({
            let mut desc = HashMap::new();
            desc.insert(
                "company".to_string(),
                "Business organization or corporation".to_string(),
            );
            desc.insert(
                "person".to_string(),
                "Individual human, executive, or analyst".to_string(),
            );
            desc.insert(
                "money".to_string(),
                "Monetary amount with currency".to_string(),
            );
            desc.insert(
                "date".to_string(),
                "Specific date or time reference".to_string(),
            );
            desc.insert(
                "product".to_string(),
                "Commercial product or service".to_string(),
            );
            desc
        })
        // Classification: Determine sentiment (single-label)
        .with_classification("sentiment", &["bullish", "bearish", "neutral"], false)
        // Classification: Identify topics (multi-label)
        .with_classification(
            "topics",
            &[
                "earnings",
                "product_launch",
                "executive_commentary",
                "investment",
                "analyst_coverage",
            ],
            true, // multi-label
        )
        // Structure: Extract key financial facts
        .with_structure(
            StructureTask::new("financial_fact")
                .with_field("company", FieldType::String)
                .with_field("metric", FieldType::String)
                .with_field("value", FieldType::String)
                .with_field("period", FieldType::String),
        );

    println!("Financial Text:\n{}\n", financial_text.trim());
    println!("Schema Configuration:");
    println!(
        "  Entity Types: {:?}",
        financial_schema.entities.as_ref().map(|e| &e.types)
    );
    println!(
        "  Classifications: {:?}",
        financial_schema
            .classifications
            .iter()
            .map(|c| &c.name)
            .collect::<Vec<_>>()
    );
    println!(
        "  Structures: {:?}",
        financial_schema
            .structures
            .iter()
            .map(|s| &s.name)
            .collect::<Vec<_>>()
    );
    println!();

    // =========================================================================
    // Example 2: Medical Case Report Analysis
    // =========================================================================
    println!("--- Example 2: Medical Case Report Analysis ---\n");

    let medical_text = r#"
        Patient John Doe, 65-year-old male, presented with chest pain on December 5th.
        Physical examination revealed elevated blood pressure (180/95 mmHg).
        ECG showed ST-segment elevation. Troponin I was 2.5 ng/mL (normal <0.04).
        Diagnosis: Acute myocardial infarction. Started on aspirin 325mg and heparin drip.
        Patient transferred to Dr. Sarah Chen at Boston Medical Center for PCI.
    "#;

    let medical_schema = TaskSchema::new()
        .with_entities(&[
            "patient",
            "doctor",
            "hospital",
            "diagnosis",
            "medication",
            "vital_sign",
            "lab_result",
        ])
        .with_classification("urgency", &["emergency", "urgent", "routine"], false)
        .with_structure(
            StructureTask::new("clinical_finding")
                .with_field("finding", FieldType::String)
                .with_field("value", FieldType::String)
                .with_field("unit", FieldType::String)
                .with_choice_field("abnormal", &["yes", "no"]),
        );

    println!("Medical Text:\n{}\n", medical_text.trim());
    println!("Schema Configuration:");
    println!(
        "  Entity Types: {:?}",
        medical_schema.entities.as_ref().map(|e| &e.types)
    );
    println!(
        "  Urgency Classification Labels: {:?}",
        medical_schema.classifications.first().map(|c| &c.labels)
    );
    println!();

    // =========================================================================
    // Example 3: Legal Contract Analysis
    // =========================================================================
    println!("--- Example 3: Legal Contract Analysis ---\n");

    let legal_text = r#"
        This Service Agreement ("Agreement") is entered into as of January 15, 2024,
        by and between Acme Corporation, a Delaware corporation ("Provider"),
        and TechStart Inc., a California corporation ("Client").
        
        Provider agrees to deliver software development services for a total fee
        of $150,000, payable in three installments of $50,000 each.
        The term of this Agreement shall be 12 months from the Effective Date.
        
        Either party may terminate this Agreement with 30 days written notice.
        Governing law: State of New York.
    "#;

    let legal_schema = TaskSchema::new()
        .with_entities(&[
            "party",
            "jurisdiction",
            "contract_type",
            "monetary_amount",
            "date",
            "duration",
        ])
        .with_classification("contract_risk", &["low", "medium", "high"], false)
        .with_structure(
            StructureTask::new("contract_term")
                .with_field("term_type", FieldType::String)
                .with_field("value", FieldType::String)
                .with_choice_field("party", &["Provider", "Client", "Both"]),
        )
        .with_structure(
            StructureTask::new("payment_schedule")
                .with_field("amount", FieldType::String)
                .with_field("frequency", FieldType::String)
                .with_field("milestones", FieldType::List), // List field for multiple milestones
        );

    println!("Legal Text:\n{}\n", legal_text.trim());
    println!("Schema Configuration:");
    println!(
        "  Entity Types: {:?}",
        legal_schema.entities.as_ref().map(|e| &e.types)
    );
    println!(
        "  Structure Types: {:?}",
        legal_schema
            .structures
            .iter()
            .map(|s| &s.name)
            .collect::<Vec<_>>()
    );
    println!();

    // =========================================================================
    // Example 4: Cybersecurity Incident Report
    // =========================================================================
    println!("--- Example 4: Cybersecurity Incident Report ---\n");

    let security_text = r#"
        On 2024-01-10 at 14:32 UTC, the SOC detected suspicious network activity
        originating from IP address 192.168.1.105 targeting production servers.
        Initial analysis indicates a potential ransomware attack using Lockbit 3.0.
        
        Affected systems: web-server-01, db-server-02, file-server-03.
        Malware hash: a1b2c3d4e5f6... (SHA-256).
        
        Incident classified as P1 Critical. CISO Jane Smith notified.
        Containment measures initiated: isolated affected network segment,
        disabled compromised user accounts (admin@company.com, backup@company.com).
    "#;

    let security_schema = TaskSchema::new()
        .with_entities(&[
            "ip_address",
            "hostname",
            "malware",
            "hash",
            "email",
            "person",
            "timestamp",
        ])
        .with_classification(
            "severity",
            &["P1 Critical", "P2 High", "P3 Medium", "P4 Low"],
            false,
        )
        .with_classification(
            "attack_type",
            &[
                "ransomware",
                "phishing",
                "data_exfiltration",
                "lateral_movement",
                "privilege_escalation",
            ],
            true, // multi-label for attack types
        )
        .with_structure(
            StructureTask::new("ioc") // Indicator of Compromise
                .with_field("type", FieldType::String)
                .with_field("value", FieldType::String)
                .with_choice_field("status", &["active", "contained", "resolved"]),
        );

    println!("Security Text:\n{}\n", security_text.trim());
    println!("Schema Configuration:");
    println!(
        "  Entity Types: {:?}",
        security_schema.entities.as_ref().map(|e| &e.types)
    );
    println!(
        "  Attack Type Labels: {:?}",
        security_schema
            .classifications
            .iter()
            .find(|c| c.name == "attack_type")
            .map(|c| &c.labels)
    );
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("=== Schema Summary ===\n");
    println!("GLiNER2 enables flexible multi-task IE through composable schemas:");
    println!();
    println!("1. ENTITY EXTRACTION:");
    println!("   - Zero-shot: Define any entity type without training");
    println!("   - With descriptions: Improve accuracy with semantic hints");
    println!();
    println!("2. TEXT CLASSIFICATION:");
    println!("   - Single-label: Mutually exclusive categories (e.g., sentiment)");
    println!("   - Multi-label: Non-exclusive tags (e.g., topics)");
    println!();
    println!("3. STRUCTURE EXTRACTION:");
    println!("   - Define hierarchical schemas with typed fields");
    println!("   - Support for String, List, and Choice field types");
    println!("   - Extract multiple instances from single document");
    println!();
    println!("4. TASK COMPOSITION:");
    println!("   - Combine NER + Classification + Structure in one pass");
    println!("   - Efficient inference with shared encoder");
    println!();

    // Note: Actual inference requires loading the model (network access)
    // This example demonstrates schema construction
    println!("Note: Run with a loaded model to perform actual extraction.");
    println!("      Model loading requires network access to HuggingFace.");

    Ok(())
}

#[cfg(not(any(feature = "onnx", feature = "candle")))]
fn main() {
    eprintln!("This example requires the 'onnx' or 'candle' feature.");
    eprintln!("Run with: cargo run --example gliner2_multitask --features candle");
}
