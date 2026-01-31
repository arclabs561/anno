//! Domain shift detection command - warn when input may not match model training domain
//!
//! NER models are trained on specific domains (news, biomedical, legal, etc.)
//! This command warns users when input text appears to differ from training data.

use clap::Parser;
use std::fs;
use std::path::PathBuf;

use super::super::output::color;
use super::super::parser::ModelBackend;

/// Analyze text for potential domain shift
#[derive(Parser, Debug)]
pub struct DomainArgs {
    /// Input file or text
    #[arg(short, long)]
    pub input: Option<PathBuf>,

    /// Input text directly
    #[arg(short, long)]
    pub text: Option<String>,

    /// Model backend to check against
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Output format (human, json)
    #[arg(long, default_value = "human")]
    pub format: String,

    /// Quiet mode
    #[arg(short, long)]
    pub quiet: bool,
}

/// Domain detection result.
///
/// Contains analysis of input text to detect potential domain shift
/// from the model's training distribution.
#[derive(Debug, Clone)]
pub struct DomainReport {
    /// Detected domain of input text
    pub detected_domain: String,
    /// Confidence in domain detection (0.0-1.0)
    pub confidence: f32,
    /// Domains the model was trained on
    pub training_domains: Vec<String>,
    /// Risk level for domain shift
    pub shift_risk: ShiftRisk,
    /// Specific indicators found in text
    pub indicators: Vec<DomainIndicator>,
    /// Suggested actions for user
    pub recommendations: Vec<String>,
}

/// Risk level for domain shift.
///
/// Higher risk indicates greater mismatch between input and training data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShiftRisk {
    /// Low risk - input matches training distribution
    Low,
    /// Medium risk - some mismatch detected
    Medium,
    /// High risk - significant mismatch, consider domain-specific model
    High,
    /// Unknown - insufficient signals to assess
    Unknown,
}

/// Indicator of domain characteristics.
#[derive(Debug, Clone)]
pub struct DomainIndicator {
    /// Indicator name (e.g., "vocabulary", "entity_types")
    pub name: String,
    /// Observed value
    pub value: String,
    /// Impact description
    pub impact: String,
}

/// Run the domain-specific extraction command.
pub fn run(args: DomainArgs) -> Result<(), String> {
    // Get input text
    let text = if let Some(path) = &args.input {
        fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?
    } else if let Some(t) = &args.text {
        t.clone()
    } else {
        return Err("No input provided. Use --input or --text".into());
    };

    // Analyze domain
    let report = analyze_domain(&text, &args.model);

    // Output results
    if args.format == "json" {
        print_json_report(&report);
    } else {
        print_human_report(&report, args.quiet);
    }

    // Return non-zero on high risk
    if report.shift_risk == ShiftRisk::High {
        eprintln!(
            "{} High domain shift risk detected",
            color("33", "warning:")
        );
    }

    Ok(())
}

fn analyze_domain(text: &str, model: &ModelBackend) -> DomainReport {
    let mut indicators = Vec::new();

    // Lexical indicators
    let words: Vec<&str> = text.split_whitespace().collect();
    let avg_word_len: f32 =
        words.iter().map(|w| w.len() as f32).sum::<f32>() / words.len().max(1) as f32;

    // Check for technical/domain vocabulary
    let biomedical_terms = [
        // Common medical terms
        "patient",
        "diagnosis",
        "treatment",
        "symptoms",
        "clinical",
        "mg",
        "dosage",
        "ml",
        "prognosis",
        "therapy",
        "medication",
        "drug",
        "dose",
        "prescription",
        "chronic",
        "acute",
        // Cardiology
        "cardiac",
        "myocardial",
        "infarction",
        "coronary",
        "artery",
        "heart",
        "vascular",
        "hypertension",
        "cardiovascular",
        "intervention",
        "percutaneous",
        "stent",
        // Oncology
        "tumor",
        "cancer",
        "oncology",
        "metastasis",
        "malignant",
        "benign",
        "chemotherapy",
        "radiotherapy",
        // General medicine
        "pathology",
        "etiology",
        "comorbidity",
        "contraindication",
        "adverse",
        "efficacy",
        "syndrome",
        "disease",
        "disorder",
        "inflammation",
        "infection",
        "antibiotic",
        "surgical",
        "procedure",
    ];
    let legal_terms = [
        "plaintiff",
        "defendant",
        "court",
        "jurisdiction",
        "pursuant",
        "hereby",
        "whereas",
        "appellant",
        "statute",
        "liability",
        "damages",
        "breach",
        "contract",
        "negligence",
        "tort",
        "verdict",
        "judgment",
        "testimony",
        "affidavit",
        "litigation",
        "counsel",
        "attorney",
    ];
    let financial_terms = [
        "revenue",
        "earnings",
        "fiscal",
        "quarter",
        "margin",
        "equity",
        "dividend",
        "securities",
        "portfolio",
        "investment",
        "stock",
        "bond",
        "asset",
        "liability",
        "capital",
        "roi",
        "ebitda",
        "valuation",
        "acquisition",
        "merger",
    ];
    let scientific_terms = [
        "hypothesis",
        "methodology",
        "experiment",
        "results",
        "analysis",
        "data",
        "significant",
        "p-value",
        "correlation",
        "regression",
        "variable",
        "sample",
        "cohort",
        "control",
        "randomized",
        "placebo",
        "statistically",
        "confidence interval",
    ];

    let text_lower = text.to_lowercase();
    let bio_count = biomedical_terms
        .iter()
        .filter(|t| text_lower.contains(*t))
        .count();
    let legal_count = legal_terms
        .iter()
        .filter(|t| text_lower.contains(*t))
        .count();
    let financial_count = financial_terms
        .iter()
        .filter(|t| text_lower.contains(*t))
        .count();
    let scientific_count = scientific_terms
        .iter()
        .filter(|t| text_lower.contains(*t))
        .count();

    // Detect primary domain
    let domain_scores = [
        ("biomedical", bio_count),
        ("legal", legal_count),
        ("financial", financial_count),
        ("scientific", scientific_count),
        ("general", 0),
    ];

    let (detected_domain, max_score) = domain_scores
        .iter()
        .max_by_key(|(_, score)| score)
        .map(|(d, s)| (d.to_string(), *s))
        .unwrap_or(("general".to_string(), 0));

    let detected_domain = if max_score > 2 {
        detected_domain
    } else {
        "general".to_string()
    };

    // Add domain-specific indicators
    indicators.push(DomainIndicator {
        name: "avg_word_length".into(),
        value: format!("{:.1}", avg_word_len),
        impact: if avg_word_len > 7.0 {
            "High (technical/specialized)".into()
        } else {
            "Normal".into()
        },
    });

    indicators.push(DomainIndicator {
        name: "biomedical_terms".into(),
        value: bio_count.to_string(),
        impact: if bio_count > 2 {
            "Strong indicator".into()
        } else {
            "Weak".into()
        },
    });

    indicators.push(DomainIndicator {
        name: "legal_terms".into(),
        value: legal_count.to_string(),
        impact: if legal_count > 2 {
            "Strong indicator".into()
        } else {
            "Weak".into()
        },
    });

    // Check for unusual patterns
    let numeric_ratio =
        text.chars().filter(|c| c.is_numeric()).count() as f32 / text.len().max(1) as f32;
    indicators.push(DomainIndicator {
        name: "numeric_ratio".into(),
        value: format!("{:.1}%", numeric_ratio * 100.0),
        impact: if numeric_ratio > 0.1 {
            "High (data-heavy)".into()
        } else {
            "Normal".into()
        },
    });

    // Model training domains
    let training_domains = match model {
        ModelBackend::Stacked | ModelBackend::Heuristic | ModelBackend::Pattern => {
            vec!["news".into(), "general web".into(), "wikipedia".into()]
        }
        #[cfg(feature = "onnx")]
        ModelBackend::Gliner => vec!["news".into(), "web".into(), "multi-domain".into()],
        _ => vec!["general".into()],
    };

    // Assess shift risk
    let shift_risk = if detected_domain == "general" || training_domains.contains(&detected_domain)
    {
        ShiftRisk::Low
    } else if detected_domain == "biomedical" || detected_domain == "legal" {
        ShiftRisk::High
    } else {
        ShiftRisk::Medium
    };

    // Generate recommendations
    let mut recommendations = Vec::new();
    if shift_risk == ShiftRisk::High {
        recommendations.push(format!(
            "Consider using a domain-specific model for {} text",
            detected_domain
        ));
        if detected_domain == "biomedical" {
            recommendations.push("Try: scispacy, pubmedbert, or biobert models".into());
        } else if detected_domain == "legal" {
            recommendations.push("Try: legal-bert or cuad models".into());
        }
        recommendations.push("Validate outputs carefully - expect lower accuracy".into());
    } else if shift_risk == ShiftRisk::Medium {
        recommendations.push("Consider reviewing a sample of extractions manually".into());
    }

    DomainReport {
        detected_domain,
        confidence: if max_score > 3 { 0.8 } else { 0.5 },
        training_domains,
        shift_risk,
        indicators,
        recommendations,
    }
}

fn print_human_report(report: &DomainReport, quiet: bool) {
    if quiet {
        println!(
            "{}\t{}\t{:?}",
            report.detected_domain, report.confidence, report.shift_risk
        );
        return;
    }

    println!("{}", color("1;36", "Domain Analysis Report"));
    println!();

    println!(
        "{}: {} ({:.0}% confident)",
        color("1;33", "Detected Domain"),
        report.detected_domain,
        report.confidence * 100.0
    );

    println!(
        "{}: {:?}",
        color("1;33", "Model Training Domains"),
        report.training_domains
    );

    let risk_color = match report.shift_risk {
        ShiftRisk::High => "31",
        ShiftRisk::Medium => "33",
        ShiftRisk::Low => "32",
        ShiftRisk::Unknown => "90",
    };
    println!(
        "{}: {}",
        color("1;33", "Domain Shift Risk"),
        color(risk_color, &format!("{:?}", report.shift_risk))
    );

    println!();
    println!("{}:", color("1;33", "Indicators"));
    for ind in &report.indicators {
        println!("  {} = {} ({})", ind.name, ind.value, ind.impact);
    }

    if !report.recommendations.is_empty() {
        println!();
        println!("{}:", color("1;33", "Recommendations"));
        for rec in &report.recommendations {
            println!("  • {}", rec);
        }
    }
}

fn print_json_report(report: &DomainReport) {
    let json = serde_json::json!({
        "detected_domain": report.detected_domain,
        "confidence": report.confidence,
        "training_domains": report.training_domains,
        "shift_risk": format!("{:?}", report.shift_risk),
        "indicators": report.indicators.iter().map(|i| {
            serde_json::json!({
                "name": i.name,
                "value": i.value,
                "impact": i.impact,
            })
        }).collect::<Vec<_>>(),
        "recommendations": report.recommendations,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&json).unwrap_or_default()
    );
}
