//! Singleton cluster analysis command - identify entities with no coreference links
//!
//! Singleton coreference clusters (entities that don't refer to anything else)
//! can reveal model limitations, genuine unique entities, or missed coreference links.

use clap::Parser;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use super::super::output::color;
use super::super::parser::ModelBackend;

/// Analyze singleton coreference clusters
#[derive(Parser, Debug)]
pub struct SingletonArgs {
    /// Input file or text
    #[arg(short, long)]
    pub input: Option<PathBuf>,

    /// Input text directly
    #[arg(short, long)]
    pub text: Option<String>,

    /// Model backend
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Output format (human, json, tsv)
    #[arg(long, default_value = "human")]
    pub format: String,

    /// Show detailed analysis
    #[arg(short, long)]
    pub verbose: bool,

    /// Quiet mode
    #[arg(short, long)]
    pub quiet: bool,
}

/// Singleton analysis report.
///
/// Summarizes coreference analysis to identify entities that don't cluster with others.
#[derive(Debug, Clone)]
pub struct SingletonReport {
    /// Total entities found in the document
    pub total_entities: usize,
    /// Number of singleton (unclustered) entities
    pub singleton_count: usize,
    /// Number of entities in coreference clusters
    pub clustered_count: usize,
    /// Ratio of singletons to total (0.0-1.0)
    pub singleton_ratio: f32,
    /// Singletons grouped by entity type
    pub singletons_by_type: HashMap<String, Vec<SingletonEntity>>,
    /// Singletons likely due to missed coreference links
    pub likely_missed: Vec<SingletonEntity>,
    /// Singletons that are likely genuine (unique mentions)
    pub likely_genuine: Vec<SingletonEntity>,
}

/// A singleton entity with diagnostic information.
#[derive(Debug, Clone)]
pub struct SingletonEntity {
    /// Entity surface text
    pub text: String,
    /// Entity type label
    pub entity_type: String,
    /// Start byte offset
    pub start: usize,
    /// End byte offset (exclusive)
    pub end: usize,
    /// Extraction confidence
    pub confidence: f32,
    /// Hypothesized reason for being a singleton
    pub reason: SingletonReason,
}

/// Why an entity might be a singleton
#[derive(Debug, Clone)]
pub enum SingletonReason {
    /// First mention with no subsequent references
    FirstMentionOnly,
    /// Unique proper noun (name of specific thing)
    UniqueProperNoun,
    /// Generic reference ("a person", "some company")
    GenericReference,
    /// Likely missed coreference (similar to another entity)
    LikelyMissed {
        /// Entity this is similar to
        similar_to: String,
        /// Similarity score
        similarity: f32,
    },
    /// Part of a compound ("CEO of Apple" - singleton "CEO")
    PartOfCompound,
}

/// Run the singleton analysis command.
pub fn run(args: SingletonArgs) -> Result<(), String> {
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

    // Analyze for singletons
    let report = analyze_singletons(&entities, &text);

    // Output results
    match args.format.as_str() {
        "json" => print_json_report(&report),
        "tsv" => print_tsv_report(&report),
        _ => print_human_report(&report, &text, args.verbose, args.quiet),
    }

    Ok(())
}

fn analyze_singletons(entities: &[anno::Entity], text: &str) -> SingletonReport {
    let mut singletons_by_type: HashMap<String, Vec<SingletonEntity>> = HashMap::new();
    let mut likely_missed = Vec::new();
    let mut likely_genuine = Vec::new();

    // For now, all entities are "singletons" since we're not doing coreference
    // In a full implementation, this would use the coreference module
    let total_entities = entities.len();
    let singleton_count = entities.len(); // Without coref, all are singletons
    let clustered_count = 0;

    // Classify each entity
    for entity in entities {
        let entity_type = entity.entity_type.as_label().to_string();
        let reason = classify_singleton(entity, entities, text);

        let singleton = SingletonEntity {
            text: entity.text.clone(),
            entity_type: entity_type.clone(),
            start: entity.start(),
            end: entity.end(),
            confidence: f32::from(entity.confidence),
            reason: reason.clone(),
        };

        // Add to type bucket
        singletons_by_type
            .entry(entity_type)
            .or_default()
            .push(singleton.clone());

        // Categorize
        match &reason {
            SingletonReason::LikelyMissed { .. } => {
                likely_missed.push(singleton);
            }
            SingletonReason::UniqueProperNoun | SingletonReason::FirstMentionOnly => {
                likely_genuine.push(singleton);
            }
            _ => {}
        }
    }

    SingletonReport {
        total_entities,
        singleton_count,
        clustered_count,
        singleton_ratio: if total_entities > 0 {
            singleton_count as f32 / total_entities as f32
        } else {
            0.0
        },
        singletons_by_type,
        likely_missed,
        likely_genuine,
    }
}

fn classify_singleton(
    entity: &anno::Entity,
    all_entities: &[anno::Entity],
    text: &str,
) -> SingletonReason {
    let entity_text = entity.text.to_lowercase();
    let entity_words: Vec<&str> = entity.text.split_whitespace().collect();

    // Check if this is a generic reference (starts with "a", "an", "the", "some")
    if entity_words
        .first()
        .is_some_and(|w| ["a", "an", "the", "some", "any"].contains(&w.to_lowercase().as_str()))
    {
        return SingletonReason::GenericReference;
    }

    // Check for similar entities (potential missed coreference)
    for other in all_entities {
        if std::ptr::eq(entity, other) {
            continue;
        }

        let other_text = other.text.to_lowercase();

        // Exact match elsewhere (definite missed coref)
        if entity_text == other_text && entity.start() != other.start() {
            return SingletonReason::LikelyMissed {
                similar_to: other.text.clone(),
                similarity: 1.0,
            };
        }

        // Check for substring match (e.g., "John" and "John Smith")
        if entity_text.contains(&other_text) || other_text.contains(&entity_text) {
            let similarity = entity_text.len().min(other_text.len()) as f32
                / entity_text.len().max(other_text.len()) as f32;
            if similarity > 0.5 {
                return SingletonReason::LikelyMissed {
                    similar_to: other.text.clone(),
                    similarity,
                };
            }
        }

        // Check for shared last name (for PERSON entities)
        if entity.entity_type.as_label() == "PER"
            && other.entity_type.as_label() == "PER"
            && entity_words.len() > 1
        {
            let other_words: Vec<&str> = other.text.split_whitespace().collect();
            if other_words.len() > 1 && entity_words.last() == other_words.last() {
                return SingletonReason::LikelyMissed {
                    similar_to: other.text.clone(),
                    similarity: 0.7,
                };
            }
        }
    }

    // Check if it's part of a compound (has "of", "for", etc. nearby)
    // Note: entity.start is a CHARACTER offset, not byte offset
    let before_context: String = text
        .chars()
        .skip(entity.start().saturating_sub(10))
        .take(10.min(entity.start()))
        .collect();
    if before_context.contains(" of ")
        || before_context.contains(" for ")
        || before_context.contains(" at ")
        || before_context.contains("'s ")
    {
        return SingletonReason::PartOfCompound;
    }

    // Check if it's a proper noun (capitalized and not at sentence start)
    if entity.start() > 0 && entity.text.chars().next().is_some_and(|c| c.is_uppercase()) {
        let prev_char = text.chars().nth(entity.start() - 1);
        if prev_char.is_some_and(|c| c != '.' && c != '!' && c != '?') {
            return SingletonReason::UniqueProperNoun;
        }
    }

    // Default: first mention only
    SingletonReason::FirstMentionOnly
}

fn print_human_report(report: &SingletonReport, _text: &str, verbose: bool, quiet: bool) {
    if quiet {
        println!(
            "{}\t{}\t{:.1}%",
            report.singleton_count,
            report.clustered_count,
            report.singleton_ratio * 100.0
        );
        return;
    }

    println!("{}", color("1;36", "Singleton Analysis Report"));
    println!();

    println!("{}:", color("1;33", "Summary"));
    println!("  Total entities:  {}", report.total_entities);
    println!("  Singletons:      {}", report.singleton_count);
    println!("  Clustered:       {}", report.clustered_count);
    println!("  Singleton ratio: {:.1}%", report.singleton_ratio * 100.0);
    println!();

    // By type breakdown
    println!("{}:", color("1;33", "By Entity Type"));
    for (entity_type, singletons) in &report.singletons_by_type {
        println!("  {}: {} singletons", entity_type, singletons.len());
    }
    println!();

    // Likely missed coreferences
    if !report.likely_missed.is_empty() {
        println!("{}:", color("1;31", "Likely Missed Coreferences"));
        for s in &report.likely_missed {
            if let SingletonReason::LikelyMissed {
                similar_to,
                similarity,
            } = &s.reason
            {
                println!(
                    "  \"{}\" ↔ \"{}\" ({:.0}% similar)",
                    s.text,
                    similar_to,
                    similarity * 100.0
                );
            }
        }
        println!();
    }

    // Likely genuine singletons
    if verbose && !report.likely_genuine.is_empty() {
        println!("{}:", color("1;32", "Likely Genuine Singletons"));
        for s in &report.likely_genuine {
            let reason_str = match &s.reason {
                SingletonReason::UniqueProperNoun => "unique proper noun",
                SingletonReason::FirstMentionOnly => "first mention only",
                _ => "other",
            };
            println!("  \"{}\" [{}] - {}", s.text, s.entity_type, reason_str);
        }
        println!();
    }

    // Verbose: all singletons
    if verbose {
        println!("{}:", color("1;33", "All Singletons"));
        for (entity_type, singletons) in &report.singletons_by_type {
            for s in singletons {
                let reason_str = match &s.reason {
                    SingletonReason::FirstMentionOnly => "first_mention",
                    SingletonReason::UniqueProperNoun => "unique_proper",
                    SingletonReason::GenericReference => "generic_ref",
                    SingletonReason::LikelyMissed { .. } => "likely_missed",
                    SingletonReason::PartOfCompound => "compound_part",
                };
                println!(
                    "  {} \"{}\" @{}:{} [{}]",
                    entity_type, s.text, s.start, s.end, reason_str
                );
            }
        }
    }
}

fn print_json_report(report: &SingletonReport) {
    let json = serde_json::json!({
        "total_entities": report.total_entities,
        "singleton_count": report.singleton_count,
        "clustered_count": report.clustered_count,
        "singleton_ratio": report.singleton_ratio,
        "by_type": report.singletons_by_type.iter().map(|(t, s)| {
            (t.clone(), serde_json::json!({
                "count": s.len(),
                "entities": s.iter().map(|e| serde_json::json!({
                    "text": e.text,
                    "start": e.start,
                    "end": e.end,
                    "confidence": e.confidence,
                    "reason": format!("{:?}", e.reason),
                })).collect::<Vec<_>>()
            }))
        }).collect::<HashMap<_, _>>(),
        "likely_missed": report.likely_missed.iter().map(|e| {
            serde_json::json!({
                "text": e.text,
                "entity_type": e.entity_type,
                "reason": format!("{:?}", e.reason),
            })
        }).collect::<Vec<_>>(),
        "likely_genuine": report.likely_genuine.len(),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&json).unwrap_or_default()
    );
}

fn print_tsv_report(report: &SingletonReport) {
    println!("text\ttype\tstart\tend\tconfidence\treason");
    for singletons in report.singletons_by_type.values() {
        for s in singletons {
            let reason_str = match &s.reason {
                SingletonReason::FirstMentionOnly => "first_mention",
                SingletonReason::UniqueProperNoun => "unique_proper",
                SingletonReason::GenericReference => "generic_ref",
                SingletonReason::LikelyMissed { similar_to, .. } => {
                    &format!("missed:{}", similar_to)
                }
                SingletonReason::PartOfCompound => "compound_part",
            };
            println!(
                "{}\t{}\t{}\t{}\t{:.2}\t{}",
                s.text, s.entity_type, s.start, s.end, s.confidence, reason_str
            );
        }
    }
}
