//! CLI tool for quick NER evaluation.
//!
//! Usage:
//!   anno-eval [OPTIONS] <COMMAND>
//!
//! Commands:
//!   quick    Run quick evaluation with synthetic data
//!   bio      Convert BIO tags to entities or validate sequences
//!   overlap  Calculate overlap metrics between spans
//!
//! Examples:
//!   anno-eval quick --model pattern
//!   anno-eval bio validate "B-PER I-PER O B-ORG"
//!   anno-eval overlap 0 10 5 15

use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }
    
    match args[1].as_str() {
        "quick" | "q" => run_quick(&args[2..]),
        "bio" | "b" => run_bio(&args[2..]),
        "overlap" | "o" => run_overlap(&args[2..]),
        "help" | "-h" | "--help" => print_usage(),
        "version" | "-V" | "--version" => {
            println!("anno-eval {}", env!("CARGO_PKG_VERSION"));
        }
        other => {
            eprintln!("Unknown command: {}", other);
            print_usage();
            process::exit(1);
        }
    }
}

fn print_usage() {
    println!(
        r#"anno-eval - NER evaluation CLI

USAGE:
    anno-eval <COMMAND> [OPTIONS]

COMMANDS:
    quick, q       Run quick evaluation with synthetic data
    bio, b         BIO tag utilities (validate, convert, repair)
    overlap, o     Calculate span overlap metrics
    help           Print this help message
    version        Print version

EXAMPLES:
    # Quick evaluation with pattern model
    anno-eval quick
    
    # Validate BIO sequence
    anno-eval bio validate "B-PER I-PER O B-ORG"
    
    # Calculate overlap IoU between spans
    anno-eval overlap 0 10 5 15
    
    # Repair invalid BIO sequence
    anno-eval bio repair "O I-PER I-PER O"
"#
    );
}

fn run_quick(args: &[String]) {
    use anno::eval::modes::MultiModeResults;
    use anno::eval::GoldEntity;
    use anno::{Entity, EntityType, Model, PatternNER};
    
    let model = PatternNER::new();
    println!("Running quick evaluation with: {}", model.name());
    println!();
    
    // Simple test cases
    let test_cases = [
        ("Meeting on January 15, 2024 at 3:00 PM", vec![
            ("January 15, 2024", EntityType::Date, 11),
            ("3:00 PM", EntityType::Time, 31),
        ]),
        ("Contact john@example.com or call 555-123-4567", vec![
            ("john@example.com", EntityType::Email, 8),
            ("555-123-4567", EntityType::Phone, 33),
        ]),
        ("Price: $99.99 for 50% off", vec![
            ("$99.99", EntityType::Money, 7),
            ("50%", EntityType::Percent, 18),
        ]),
    ];
    
    let mut all_predicted: Vec<Entity> = Vec::new();
    let mut all_gold: Vec<GoldEntity> = Vec::new();
    
    for (text, expected) in &test_cases {
        println!("Text: {}", text);
        
        let predicted = model.extract_entities(text, None).unwrap_or_default();
        for p in &predicted {
            println!("  Predicted: {} [{}] ({}-{})", 
                p.text, p.entity_type.as_label(), p.start, p.end);
        }
        
        let gold: Vec<GoldEntity> = expected
            .iter()
            .map(|(t, ty, start)| GoldEntity::new(*t, ty.clone(), *start))
            .collect();
        
        for g in &gold {
            println!("  Expected:  {} [{}] ({}-{})", 
                g.text, g.entity_type.as_label(), g.start, g.end);
        }
        
        let mode_results = MultiModeResults::compute(&predicted, &gold);
        println!("  Strict F1: {:.1}%", mode_results.strict.f1 * 100.0);
        println!();
        
        all_predicted.extend(predicted);
        all_gold.extend(gold);
    }
    
    // Skip summary if using non-default args
    if !args.is_empty() {
        return;
    }
    
    println!("=== Summary ===");
    let summary = MultiModeResults::compute(&all_predicted, &all_gold);
    println!("Total entities: {} predicted, {} gold", all_predicted.len(), all_gold.len());
    println!();
    summary.print_summary();
}

fn run_bio(args: &[String]) {
    use anno::eval::bio_adapter::{
        bio_to_entities, repair_bio_sequence, validate_bio_sequence,
        BioScheme, RepairStrategy,
    };
    
    if args.is_empty() {
        eprintln!("Usage: anno-eval bio <validate|repair|convert> <tags>");
        process::exit(1);
    }
    
    let subcommand = &args[0];
    let tags_str = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let tags: Vec<&str> = tags_str.split_whitespace().collect();
    
    if tags.is_empty() {
        eprintln!("No tags provided");
        process::exit(1);
    }
    
    match subcommand.as_str() {
        "validate" | "v" => {
            let errors = validate_bio_sequence(&tags, BioScheme::IOB2);
            if errors.is_empty() {
                println!("Valid IOB2 sequence");
            } else {
                println!("Invalid IOB2 sequence:");
                for e in &errors {
                    println!("  - {}", e);
                }
                process::exit(1);
            }
        }
        "repair" | "r" => {
            let repaired = repair_bio_sequence(&tags, BioScheme::IOB2, RepairStrategy::PromoteToBegin);
            println!("Original: {}", tags.join(" "));
            println!("Repaired: {}", repaired.join(" "));
        }
        "convert" | "c" => {
            // Create dummy tokens from tags
            let tokens: Vec<String> = (0..tags.len())
                .map(|i| format!("tok{}", i))
                .collect();
            let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
            
            match bio_to_entities(&token_refs, &tags, BioScheme::IOB2) {
                Ok(entities) => {
                    if entities.is_empty() {
                        println!("No entities found");
                    } else {
                        for e in &entities {
                            println!("{}: {} ({}-{})", 
                                e.entity_type.as_label(), e.text, e.start, e.end);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    process::exit(1);
                }
            }
        }
        other => {
            eprintln!("Unknown bio subcommand: {}", other);
            eprintln!("Use: validate, repair, or convert");
            process::exit(1);
        }
    }
}

fn run_overlap(args: &[String]) {
    use anno::eval::modes::overlap_ratio;
    
    if args.len() < 4 {
        eprintln!("Usage: anno-eval overlap <start1> <end1> <start2> <end2>");
        eprintln!("Example: anno-eval overlap 0 10 5 15");
        process::exit(1);
    }
    
    let parse = |s: &str| -> usize {
        s.parse().unwrap_or_else(|_| {
            eprintln!("Invalid number: {}", s);
            process::exit(1);
        })
    };
    
    let start1 = parse(&args[0]);
    let end1 = parse(&args[1]);
    let start2 = parse(&args[2]);
    let end2 = parse(&args[3]);
    
    let iou = overlap_ratio(start1, end1, start2, end2);
    
    println!("Span 1: [{}, {})", start1, end1);
    println!("Span 2: [{}, {})", start2, end2);
    println!();
    
    if iou == 0.0 {
        println!("No overlap");
    } else {
        let intersection_start = start1.max(start2);
        let intersection_end = end1.min(end2);
        let intersection = intersection_end.saturating_sub(intersection_start);
        let union = (end1 - start1) + (end2 - start2) - intersection;
        
        println!("Intersection: [{}, {}) = {} chars", intersection_start, intersection_end, intersection);
        println!("Union: {} chars", union);
        println!("IoU: {:.3} ({:.1}%)", iou, iou * 100.0);
    }
}

