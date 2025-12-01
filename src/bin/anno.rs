//! anno - Information Extraction CLI
//!
//! A unified toolkit for named entity recognition, coreference resolution,
//! relation extraction, and entity linking.
//!
//! # Capabilities
//!
//! - **NER**: Named Entity Recognition (persons, organizations, locations, etc.)
//! - **Coreference**: Link mentions to the same entity ("She" → "Marie Curie")  
//! - **Relations**: Extract (head, relation, tail) triples
//! - **Entity Linking**: Connect entities to knowledge bases (Wikidata)
//! - **Events**: Discourse-level event extraction
//!
//! # Signal → Track → Identity Hierarchy
//!
//! ```text
//! Level 1 (Signal)   : Raw detections with spans  
//! Level 2 (Track)    : Within-document coreference chains
//! Level 3 (Identity) : Cross-document KB-linked entities
//! ```
//!
//! # Usage
//!
//! ```bash
//! # Basic NER extraction
//! anno extract "Marie Curie won the Nobel Prize."
//!
//! # Debug with coreference and KB linking
//! anno debug --coref --link-kb -t "Barack Obama met Angela Merkel. He praised her."
//!
//! # Evaluate against gold annotations
//! anno eval -t "..." -g "Marie Curie:PER:0:11"
//!
//! # Validate annotation files
//! anno validate file.jsonl
//!
//! # Show available models and features
//! anno info
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};
use std::process::ExitCode;
use std::time::Instant;

use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{generate, Shell};
use is_terminal::IsTerminal;

use anno::grounded::{
    render_document_html, render_eval_html, EvalComparison, EvalMatch, GroundedDocument, Identity,
    Location, Modality, Quantifier, Signal, SignalValidationError,
};
use anno::{AutoNER, Entity, HeuristicNER, Model, PatternNER, StackedNER};

#[cfg(feature = "onnx")]
// GLiNER exports available when onnx feature is enabled
#[allow(unused_imports)]
use anno::{DEFAULT_GLINER2_MODEL, DEFAULT_GLINER_MODEL};

// ============================================================================
// CLI Structure
// ============================================================================

/// Information Extraction CLI - NER, Coreference, Relations, Entity Linking
#[derive(Parser)]
#[command(name = "anno")]
#[command(
    author,
    version,
    about = "Information Extraction CLI - NER, Coreference, Relations, Entity Linking",
    long_about = r#"
anno - A unified information extraction toolkit

CAPABILITIES:
  • Named Entity Recognition (NER) - detect persons, orgs, locations, etc.
  • Coreference Resolution - link mentions to same entity ("She" → "Marie Curie")
  • Relation Extraction - extract (head, relation, tail) triples
  • Entity Linking - connect entities to knowledge bases (Wikidata)
  • Event Extraction - discourse-level events and shell nouns

SIGNAL → TRACK → IDENTITY HIERARCHY:
  Level 1 (Signal)   : Raw detections/mentions with spans
  Level 2 (Track)    : Within-document coreference chains  
  Level 3 (Identity) : Cross-document KB-linked entities

BACKENDS:
  • pattern    - High-precision patterns (dates, money, emails)
  • heuristic (alias: statistical) - Capitalization + context heuristics
  • gliner     - Zero-shot NER via ONNX (any entity type)
  • w2ner      - Nested/discontinuous entities

EXAMPLES:
  anno extract "Marie Curie won the Nobel Prize."
  anno debug --coref --link-kb -t "Barack Obama met Angela Merkel. He discussed NATO."
  anno eval -t "..." -g "Marie Curie:PER:0:11"
  anno info
"#
)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Text to extract entities from (shorthand for `anno extract`)
    #[arg(trailing_var_arg = true)]
    text: Vec<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Extract entities from text
    #[command(visible_alias = "x")]
    Extract(ExtractArgs),

    /// Generate HTML debug visualization
    #[command(visible_alias = "d")]
    Debug(DebugArgs),

    /// Evaluate predictions against gold annotations
    #[command(visible_alias = "e")]
    Eval(EvalArgs),

    /// Validate JSONL annotation files
    #[command(visible_alias = "v")]
    Validate(ValidateArgs),

    /// Deep analysis with multiple models
    #[command(visible_alias = "a")]
    Analyze(AnalyzeArgs),

    /// Work with NER datasets
    #[command(visible_alias = "ds")]
    Dataset(DatasetArgs),

    /// Comprehensive evaluation across all task-dataset-backend combinations
    #[command(visible_alias = "bench")]
    #[cfg(feature = "eval-advanced")]
    Benchmark(BenchmarkArgs),

    /// Show model and version info
    #[command(visible_alias = "i")]
    Info,

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

// ============================================================================
// Shared Types
// ============================================================================

/// Model backend selection
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum ModelBackend {
    /// Pattern matching only (dates, emails, etc.)
    Pattern,
    /// Heuristic NER (persons, orgs, locs via capitalization + context)
    #[value(alias = "statistical")]
    Heuristic,
    /// Minimal heuristic (low complexity experiment)
    Minimal,
    /// Automatic (Language-detected routing)
    Auto,
    /// Stacked: Pattern + Heuristic (default)
    #[default]
    Stacked,
    /// GLiNER via ONNX (requires --features onnx)
    #[cfg(feature = "onnx")]
    Gliner,
    /// NuNER (requires --features onnx)
    #[cfg(feature = "onnx")]
    Nuner,
    /// W2NER for nested entities (requires --features onnx)
    #[cfg(feature = "onnx")]
    W2ner,
    /// GLiNER via Candle (requires --features candle)
    #[cfg(feature = "candle")]
    GlinerCandle,
}

impl ModelBackend {
    fn create_model(self) -> Result<Box<dyn Model>, String> {
        match self {
            Self::Pattern => Ok(Box::new(PatternNER::new())),
            Self::Heuristic => Ok(Box::new(HeuristicNER::new())),
            // Minimal was merged into HeuristicNER
            Self::Minimal => Ok(Box::new(HeuristicNER::new())),
            Self::Auto => {
                // AutoNER just routes to default (StackedNER), doesn't combine models
                Ok(Box::new(AutoNER::new()))
            }
            Self::Stacked => Ok(Box::new(StackedNER::default())),
            #[cfg(feature = "onnx")]
            Self::Gliner => anno::GLiNEROnnx::new(anno::DEFAULT_GLINER_MODEL)
                .map(|m| Box::new(m) as Box<dyn Model>)
                .map_err(|e| format!("Failed to load GLiNER: {}", e)),
            #[cfg(feature = "onnx")]
            Self::Nuner => Err("NuNER not yet implemented in CLI".to_string()),
            #[cfg(feature = "onnx")]
            Self::W2ner => Err("W2NER not yet implemented in CLI".to_string()),
            #[cfg(feature = "candle")]
            Self::GlinerCandle => Err("GLiNER Candle not yet implemented in CLI".to_string()),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Pattern => "pattern",
            Self::Heuristic => "heuristic",
            Self::Minimal => "minimal",
            Self::Auto => "auto",
            Self::Stacked => "stacked",
            #[cfg(feature = "onnx")]
            Self::Gliner => "gliner",
            #[cfg(feature = "onnx")]
            Self::Nuner => "nuner",
            #[cfg(feature = "onnx")]
            Self::W2ner => "w2ner",
            #[cfg(feature = "candle")]
            Self::GlinerCandle => "gliner-candle",
        }
    }
}

/// Output format selection
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum OutputFormat {
    /// Human-readable colored output
    #[default]
    Human,
    /// JSON array
    Json,
    /// JSON lines (one object per line)
    Jsonl,
    /// Tab-separated values
    Tsv,
    /// Inline annotations in text
    Inline,
    /// Full GroundedDocument as JSON
    Grounded,
}

/// Evaluation task type
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum EvalTask {
    /// Named Entity Recognition
    #[default]
    Ner,
    /// Coreference Resolution
    Coref,
    /// Relation Extraction
    Relation,
}

// ============================================================================
// Command Arguments
// ============================================================================

#[derive(Parser)]
struct ExtractArgs {
    /// Input text to process
    #[arg(short, long)]
    text: Option<String>,

    /// Read input from file
    #[arg(short, long, value_name = "PATH")]
    file: Option<String>,

    /// Model backend to use
    #[arg(short, long, default_value = "stacked")]
    model: ModelBackend,

    /// Filter to specific entity types (repeatable)
    #[arg(short, long = "label", value_name = "TYPE")]
    labels: Vec<String>,

    /// Output format
    #[arg(long, default_value = "human")]
    format: OutputFormat,

    /// Detect negated entities ("not John Smith")
    #[arg(long)]
    negation: bool,

    /// Detect quantified entities ("every employee")
    #[arg(long)]
    quantifiers: bool,

    /// Show context around entities
    #[arg(short, long)]
    verbose: bool,

    /// Minimal output
    #[arg(short, long)]
    quiet: bool,

    /// Positional text argument
    #[arg(trailing_var_arg = true)]
    positional: Vec<String>,
}

#[derive(Parser)]
struct DebugArgs {
    /// Input text to process
    #[arg(short, long)]
    text: Option<String>,

    /// Read input from file
    #[arg(short, long, value_name = "PATH")]
    file: Option<String>,

    /// Model backend to use
    #[arg(short, long, default_value = "stacked")]
    model: ModelBackend,

    /// Output as HTML (default: text)
    #[arg(long)]
    html: bool,

    /// Write output to file (default: stdout)
    #[arg(short, long, value_name = "PATH")]
    output: Option<String>,

    /// Run coreference resolution to form tracks
    #[arg(long)]
    coref: bool,

    /// Link tracks to KB identities (creates placeholder Wikidata IDs)
    #[arg(long)]
    link_kb: bool,

    /// Suppress status messages
    #[arg(short, long)]
    quiet: bool,
}

#[derive(Parser)]
struct EvalArgs {
    /// Input text to process
    #[arg(short, long)]
    text: Option<String>,

    /// Read input from file
    #[arg(short, long, value_name = "PATH")]
    file: Option<String>,

    /// Model backend to use
    #[arg(short, long, default_value = "stacked")]
    model: ModelBackend,

    /// Gold annotation: "text:label:start:end" (repeatable)
    #[arg(short, long = "gold", value_name = "SPEC")]
    gold_specs: Vec<String>,

    /// Load gold annotations from JSONL file
    #[arg(long, value_name = "PATH")]
    gold_file: Option<String>,

    /// Write HTML report to file
    #[arg(short, long, value_name = "PATH")]
    output: Option<String>,

    /// Output as JSON
    #[arg(long)]
    json: bool,

    /// Output as HTML report
    #[arg(long)]
    html: bool,

    /// Show detailed match info
    #[arg(short, long)]
    verbose: bool,

    /// Minimal output
    #[arg(short, long)]
    quiet: bool,

    /// Positional text argument
    #[arg(trailing_var_arg = true)]
    positional: Vec<String>,
}

#[derive(Parser)]
struct ValidateArgs {
    /// JSONL files to validate
    #[arg(required = true)]
    files: Vec<String>,
}

#[derive(Parser)]
struct AnalyzeArgs {
    /// Input text to process
    #[arg(short, long)]
    text: Option<String>,

    /// Read input from file
    #[arg(short, long, value_name = "PATH")]
    file: Option<String>,

    /// Positional text argument
    #[arg(trailing_var_arg = true)]
    positional: Vec<String>,
}

#[derive(Parser)]
struct DatasetArgs {
    /// Action to perform
    #[command(subcommand)]
    action: DatasetAction,
}

#[derive(Subcommand)]
enum DatasetAction {
    /// List available datasets
    #[command(visible_alias = "ls")]
    List,

    /// Show dataset statistics
    Info {
        /// Dataset name
        #[arg(short, long)]
        dataset: String,
    },

    /// Evaluate model on dataset
    Eval {
        /// Dataset name
        #[arg(short, long, default_value = "synthetic")]
        dataset: String,

        /// Model backend to use
        #[arg(short, long, default_value = "stacked")]
        model: ModelBackend,

        /// Task type: ner, coref, or relation
        #[arg(short, long, default_value = "ner")]
        task: EvalTask,
    },
}

#[cfg(feature = "eval-advanced")]
#[derive(Parser)]
struct BenchmarkArgs {
    /// Tasks to evaluate (comma-separated: ner,coref,relation). Default: all
    #[arg(short, long, value_delimiter = ',')]
    tasks: Option<Vec<String>>,

    /// Datasets to use (comma-separated). Default: all suitable datasets
    #[arg(short, long, value_delimiter = ',')]
    datasets: Option<Vec<String>>,

    /// Backends to test (comma-separated). Default: all compatible backends
    #[arg(short, long, value_delimiter = ',')]
    backends: Option<Vec<String>>,

    /// Maximum examples per dataset (for quick testing)
    #[arg(short, long)]
    max_examples: Option<usize>,

    /// Random seed for sampling (for reproducibility and varied testing)
    #[arg(long)]
    seed: Option<u64>,

    /// Only use cached datasets (skip downloads)
    #[arg(long)]
    cached_only: bool,

    /// Output file for markdown report (default: stdout)
    #[arg(short, long)]
    output: Option<String>,
}

#[cfg(feature = "eval")]
use anno::eval::loader::DatasetId;

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() -> ExitCode {
    let cli = Cli::parse();

    let result = match cli.command {
        Some(Commands::Extract(args)) => cmd_extract(args),
        Some(Commands::Debug(args)) => cmd_debug(args),
        Some(Commands::Eval(args)) => cmd_eval(args),
        Some(Commands::Validate(args)) => cmd_validate(args),
        Some(Commands::Analyze(args)) => cmd_analyze(args),
        Some(Commands::Dataset(args)) => cmd_dataset(args),
        #[cfg(feature = "eval-advanced")]
        Some(Commands::Benchmark(args)) => cmd_benchmark(args),
        Some(Commands::Info) => cmd_info(),
        Some(Commands::Completions { shell }) => {
            generate(shell, &mut Cli::command(), "anno", &mut io::stdout());
            Ok(())
        }
        None => {
            // No subcommand: treat positional args as text to extract
            if cli.text.is_empty() {
                eprintln!("No input provided. Run `anno --help` for usage.");
                return ExitCode::FAILURE;
            }
            let text = cli.text.join(" ");
            cmd_extract(ExtractArgs {
                text: Some(text),
                file: None,
                model: ModelBackend::default(),
                labels: vec![],
                format: OutputFormat::default(),
                negation: false,
                quantifiers: false,
                verbose: false,
                quiet: false,
                positional: vec![],
            })
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{} {}", color("31", "error:"), e);
            ExitCode::FAILURE
        }
    }
}

// ============================================================================
// Commands
// ============================================================================

fn cmd_extract(args: ExtractArgs) -> Result<(), String> {
    let text = get_input_text(&args.text, args.file.as_deref(), &args.positional)?;
    let model = args.model.create_model()?;

    let start = Instant::now();
    let entities = model
        .extract_entities(&text, None)
        .map_err(|e| format!("Extraction failed: {}", e))?;
    let elapsed = start.elapsed();

    // Filter by labels if specified
    let entities: Vec<_> = if args.labels.is_empty() {
        entities
    } else {
        entities
            .into_iter()
            .filter(|e| {
                args.labels
                    .iter()
                    .any(|l| e.entity_type.as_label().eq_ignore_ascii_case(l))
            })
            .collect()
    };

    // Build grounded document with validation
    let mut doc = GroundedDocument::new("extract", &text);
    let mut validation_errors: Vec<SignalValidationError> = Vec::new();

    for e in &entities {
        let mut signal = Signal::new(
            0,
            Location::text(e.start, e.end),
            &e.text,
            e.entity_type.as_label(),
            e.confidence as f32,
        )
        .with_modality(Modality::Symbolic);

        // Detect negation
        if args.negation && is_negated(&text, e.start) {
            signal = signal.negated();
        }

        // Detect quantification
        if args.quantifiers {
            if let Some(q) = detect_quantifier(&text, e.start) {
                signal = signal.with_quantifier(q);
            }
        }

        // Validate before adding
        if let Some(err) = signal.validate_against(&text) {
            validation_errors.push(err);
        } else {
            doc.add_signal(signal);
        }
    }

    // Report validation errors
    if !validation_errors.is_empty() && !args.quiet {
        eprintln!(
            "{} {} validation errors:",
            color("33", "warning:"),
            validation_errors.len()
        );
        for err in &validation_errors {
            eprintln!("  - {}", err);
        }
    }

    // Output
    match args.format {
        OutputFormat::Json => {
            let output: Vec<_> = doc
                .signals()
                .iter()
                .map(|s| {
                    let (start, end) = s.text_offsets().unwrap_or((0, 0));
                    serde_json::json!({
                        "text": s.surface(),
                        "type": s.label(),
                        "start": start,
                        "end": end,
                        "confidence": s.confidence,
                        "negated": s.negated,
                        "quantifier": s.quantifier.map(|q| format!("{:?}", q)),
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&output).unwrap_or_default()
            );
        }
        OutputFormat::Jsonl => {
            for s in doc.signals() {
                let (start, end) = s.text_offsets().unwrap_or((0, 0));
                let obj = serde_json::json!({
                    "text": s.surface(),
                    "type": s.label(),
                    "start": start,
                    "end": end,
                    "confidence": s.confidence,
                });
                println!("{}", obj);
            }
        }
        OutputFormat::Tsv => {
            println!("start\tend\ttype\tconfidence\tnegated\ttext");
            for s in doc.signals() {
                let (start, end) = s.text_offsets().unwrap_or((0, 0));
                println!(
                    "{}\t{}\t{}\t{:.2}\t{}\t{}",
                    start,
                    end,
                    s.label(),
                    s.confidence,
                    s.negated,
                    s.surface()
                );
            }
        }
        OutputFormat::Grounded => {
            println!("{}", serde_json::to_string_pretty(&doc).unwrap_or_default());
        }
        OutputFormat::Inline => {
            print_annotated_signals(&text, doc.signals());
        }
        OutputFormat::Human => {
            if args.quiet {
                for s in doc.signals() {
                    let (start, end) = s.text_offsets().unwrap_or((0, 0));
                    let neg = if s.negated { " [NEG]" } else { "" };
                    let quant = s
                        .quantifier
                        .map(|q| format!(" [{:?}]", q))
                        .unwrap_or_default();
                    println!(
                        "[{},{})\t{}\t{}{}{}",
                        start,
                        end,
                        s.label(),
                        s.surface(),
                        neg,
                        quant
                    );
                }
            } else {
                println!();
                println!(
                    "{} extracted {} entities in {:.1}ms (model: {})",
                    color("32", "ok:"),
                    doc.signals().len(),
                    elapsed.as_secs_f64() * 1000.0,
                    args.model.name()
                );
                println!();

                if doc.signals().is_empty() {
                    println!("  (no entities found)");
                } else {
                    print_signals(&doc, &text, args.verbose);
                }
                println!();
                print_annotated_signals(&text, doc.signals());
            }
        }
    }

    Ok(())
}

fn cmd_debug(args: DebugArgs) -> Result<(), String> {
    let text = get_input_text(&args.text, args.file.as_deref(), &[])?;
    let model = args.model.create_model()?;

    let entities = model
        .extract_entities(&text, None)
        .map_err(|e| format!("Extraction failed: {}", e))?;

    // Build grounded document with validated signals
    // Always use actual offsets from model - don't re-find text (which would always find first occurrence)
    let mut doc = GroundedDocument::new("debug", &text);
    let mut signal_ids: Vec<u64> = Vec::new();

    for e in &entities {
        let signal = Signal::new(
            0,
            Location::text(e.start, e.end),
            &e.text,
            e.entity_type.as_label(),
            e.confidence as f32,
        );
        let id = doc.add_signal(signal);
        signal_ids.push(id);
    }

    // Run coreference resolution if requested
    if args.coref {
        resolve_coreference(&mut doc, &text, &signal_ids);
    }

    // Link tracks to KB identities if requested
    if args.link_kb {
        link_tracks_to_kb(&mut doc);
    }

    // Build spatial index and validate
    let index = doc.build_text_index();
    let errors = doc.validate();

    if !errors.is_empty() && !args.quiet {
        eprintln!(
            "{} {} validation errors:",
            color("33", "warning:"),
            errors.len()
        );
        for e in &errors {
            eprintln!("  - {}", e);
        }
    }

    // Show stats
    if !args.quiet {
        let stats = doc.stats();
        println!();
        println!("{}", color("1;36", "Document Analysis"));
        println!("  Text length: {} chars", text.len());
        println!("  Signals: {}", stats.signal_count);
        println!("  Tracks: {}", stats.track_count);
        println!("  Identities: {}", stats.identity_count);
        println!("  Spatial index nodes: {}", index.len());
        println!(
            "  Validation: {}",
            if errors.is_empty() {
                color("32", "valid")
            } else {
                color("31", &format!("{} errors", errors.len()))
            }
        );
        println!();
    }

    // Output format
    if args.html
        || args
            .output
            .as_ref()
            .map(|p| p.ends_with(".html"))
            .unwrap_or(false)
    {
        // Generate HTML
        let html = render_document_html(&doc);

        if let Some(path) = &args.output {
            fs::write(path, &html).map_err(|e| format!("Failed to write {}: {}", path, e))?;
            if !args.quiet {
                println!("{} HTML written to: {}", color("32", "ok:"), path);
            }
        } else {
            println!("{}", html);
        }
    } else {
        // Text output (default)
        if doc.signals().is_empty() {
            println!("  (no entities found)");
        } else {
            print_signals(&doc, &text, false);
        }
        println!();
        print_annotated_signals(&text, doc.signals());

        // Show tracks if coref was run
        if args.coref {
            let tracks: Vec<_> = doc.tracks().collect();
            if !tracks.is_empty() {
                println!();
                println!("{}", color("1;36", "Coreference Tracks"));
                for track in tracks {
                    let entity_type = track.entity_type.as_deref().unwrap_or("-");
                    let signals: Vec<String> = track
                        .signals
                        .iter()
                        .filter_map(|s| doc.get_signal(s.signal_id))
                        .map(|s| format!("\"{}\"", s.surface()))
                        .collect();
                    println!(
                        "  T{}: {} [{}] ({})",
                        track.id,
                        track.canonical_surface,
                        entity_type,
                        signals.join(", ")
                    );
                }
            }
        }

        // Show identities if KB linking was run
        if args.link_kb {
            let identities: Vec<_> = doc.identities().collect();
            if !identities.is_empty() {
                println!();
                println!("{}", color("1;36", "KB-Linked Identities"));
                for identity in identities {
                    let kb_id = identity.kb_id.as_deref().unwrap_or("-");
                    println!(
                        "  I{}: {} ({})",
                        identity.id, identity.canonical_name, kb_id
                    );
                }
            }
        }

        // Note: Text output always goes to stdout
        // Use --html --output file.html for HTML file output
    }

    Ok(())
}

fn cmd_eval(args: EvalArgs) -> Result<(), String> {
    let text = get_input_text(&args.text, args.file.as_deref(), &args.positional)?;

    // Load gold from file or args
    let gold = if let Some(gold_file) = &args.gold_file {
        load_gold_from_file(gold_file)?
    } else if !args.gold_specs.is_empty() {
        args.gold_specs
            .iter()
            .filter_map(|s| parse_gold_spec(s))
            .collect()
    } else {
        return Err(
            "No gold annotations. Use -g 'text:label:start:end' or --gold-file path.jsonl"
                .to_string(),
        );
    };

    if gold.is_empty() {
        return Err("No valid gold annotations found".to_string());
    }

    let model = args.model.create_model()?;

    let start = Instant::now();
    let entities = model
        .extract_entities(&text, None)
        .map_err(|e| format!("Extraction failed: {}", e))?;
    let elapsed = start.elapsed();

    // Build signals
    let gold_signals: Vec<Signal<Location>> = gold
        .iter()
        .enumerate()
        .map(|(i, g)| {
            Signal::new(
                i as u64,
                Location::text(g.start, g.end),
                &g.text,
                &g.label,
                1.0,
            )
        })
        .collect();

    let pred_signals: Vec<Signal<Location>> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| {
            Signal::new(
                i as u64,
                Location::text(e.start, e.end),
                &e.text,
                e.entity_type.as_label(),
                e.confidence as f32,
            )
        })
        .collect();

    let cmp = EvalComparison::compare(&text, gold_signals, pred_signals);

    // Detailed analysis with eval feature
    #[cfg(feature = "eval")]
    let detailed_analysis = {
        use anno::eval::analysis::ErrorAnalysis;
        use anno::eval::GoldEntity;
        use anno::EntityType;

        let gold_entities: Vec<GoldEntity> = gold
            .iter()
            .map(|g| GoldEntity {
                text: g.text.clone(),
                entity_type: EntityType::Other(g.label.clone()),
                original_label: g.label.clone(),
                start: g.start,
                end: g.end,
            })
            .collect();

        Some(ErrorAnalysis::analyze(&text, &entities, &gold_entities))
    };
    #[cfg(not(feature = "eval"))]
    let _detailed_analysis: Option<()> = None;

    // Output
    if args.json {
        let mut output = serde_json::json!({
            "model": args.model.name(),
            "elapsed_ms": elapsed.as_secs_f64() * 1000.0,
            "gold_count": cmp.gold.len(),
            "predicted_count": cmp.predicted.len(),
            "correct": cmp.correct_count(),
            "errors": cmp.error_count(),
            "precision": cmp.precision(),
            "recall": cmp.recall(),
            "f1": cmp.f1(),
        });

        let matches: Vec<_> = cmp
            .matches
            .iter()
            .map(|m| match m {
                EvalMatch::Correct { gold_id, pred_id } => serde_json::json!({
                    "type": "correct",
                    "gold_id": gold_id,
                    "pred_id": pred_id,
                }),
                EvalMatch::TypeMismatch {
                    gold_id,
                    pred_id,
                    gold_label,
                    pred_label,
                } => serde_json::json!({
                    "type": "type_mismatch",
                    "gold_id": gold_id,
                    "pred_id": pred_id,
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                }),
                EvalMatch::BoundaryError {
                    gold_id,
                    pred_id,
                    iou,
                } => serde_json::json!({
                    "type": "boundary_error",
                    "gold_id": gold_id,
                    "pred_id": pred_id,
                    "iou": iou,
                }),
                EvalMatch::Spurious { pred_id } => serde_json::json!({
                    "type": "false_positive",
                    "pred_id": pred_id,
                }),
                EvalMatch::Missed { gold_id } => serde_json::json!({
                    "type": "false_negative",
                    "gold_id": gold_id,
                }),
            })
            .collect();
        output["matches"] = serde_json::Value::Array(matches);

        println!(
            "{}",
            serde_json::to_string_pretty(&output).unwrap_or_default()
        );
    } else if args.html {
        let html = render_eval_html(&cmp);
        if let Some(path) = &args.output {
            fs::write(path, &html).map_err(|e| format!("Write failed: {}", e))?;
            if !args.quiet {
                println!("{} HTML written to: {}", color("32", "ok:"), path);
            }
        } else {
            println!("{}", html);
        }
    } else {
        // Human readable
        println!();
        println!(
            "{}",
            color(
                "1;36",
                "======================================================================="
            )
        );
        println!(
            "  {}  model={}  time={:.1}ms",
            color("1;36", "EVALUATION"),
            args.model.name(),
            elapsed.as_secs_f64() * 1000.0
        );
        println!(
            "  gold={}  pred={}  correct={}  errors={}",
            cmp.gold.len(),
            cmp.predicted.len(),
            cmp.correct_count(),
            cmp.error_count()
        );
        println!(
            "{}",
            color(
                "1;36",
                "======================================================================="
            )
        );
        println!();

        let p = cmp.precision() * 100.0;
        let r = cmp.recall() * 100.0;
        let f1 = cmp.f1() * 100.0;

        println!("  Precision: {}%", metric_colored(p));
        println!("  Recall:    {}%", metric_colored(r));
        println!("  F1:        {}%", metric_colored(f1));
        println!();

        print_matches(&cmp, args.verbose);

        #[cfg(feature = "eval")]
        if let Some(analysis) = detailed_analysis {
            println!();
            println!("{}:", color("1;33", "Error Breakdown"));
            for (err_type, count) in &analysis.counts {
                println!("  {:?}: {}", err_type, count);
            }
        }

        println!();
    }

    Ok(())
}

fn cmd_validate(args: ValidateArgs) -> Result<(), String> {
    let mut total_errors = 0;
    let mut total_warnings = 0;
    let mut total_entries = 0;

    for file in &args.files {
        let content =
            fs::read_to_string(file).map_err(|e| format!("Failed to read {}: {}", file, e))?;

        for (line_num, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }

            total_entries += 1;

            let entry: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| format!("{}:{}: Invalid JSON: {}", file, line_num + 1, e))?;

            let text = entry["text"]
                .as_str()
                .ok_or_else(|| format!("{}:{}: Missing 'text' field", file, line_num + 1))?;

            let entities = entry["entities"]
                .as_array()
                .ok_or_else(|| format!("{}:{}: Missing 'entities' array", file, line_num + 1))?;

            let mut doc = GroundedDocument::new(format!("{}:{}", file, line_num + 1), text);

            for (i, ent) in entities.iter().enumerate() {
                // Check for missing required fields
                let start = match ent["start"].as_u64() {
                    Some(v) => v as usize,
                    None => {
                        eprintln!(
                            "{} {}:{}:entity[{}]: missing 'start' field",
                            color("33", "warn"),
                            file,
                            line_num + 1,
                            i
                        );
                        total_warnings += 1;
                        0
                    }
                };
                let end = match ent["end"].as_u64() {
                    Some(v) => v as usize,
                    None => {
                        eprintln!(
                            "{} {}:{}:entity[{}]: missing 'end' field",
                            color("33", "warn"),
                            file,
                            line_num + 1,
                            i
                        );
                        total_warnings += 1;
                        0
                    }
                };
                let ent_text = ent["text"].as_str().unwrap_or("");
                let ent_type = ent["type"]
                    .as_str()
                    .or(ent["label"].as_str())
                    .unwrap_or("UNK");

                let signal = Signal::new(
                    i as u64,
                    Location::text(start, end),
                    ent_text,
                    ent_type,
                    1.0,
                );

                if let Some(err) = signal.validate_against(text) {
                    match err {
                        SignalValidationError::OutOfBounds { .. }
                        | SignalValidationError::InvalidSpan { .. } => {
                            eprintln!(
                                "{} {}:{}:entity[{}]: {}",
                                color("31", "error"),
                                file,
                                line_num + 1,
                                i,
                                err
                            );
                            total_errors += 1;
                        }
                        SignalValidationError::TextMismatch { .. } => {
                            eprintln!(
                                "{} {}:{}:entity[{}]: {}",
                                color("33", "warn"),
                                file,
                                line_num + 1,
                                i,
                                err
                            );
                            total_warnings += 1;
                        }
                    }
                }

                doc.add_signal(signal);
            }
        }
    }

    println!();
    println!(
        "Validated {} entries in {} file(s)",
        total_entries,
        args.files.len()
    );
    if total_errors > 0 {
        println!("{} {} errors", color("31", "x"), total_errors);
    }
    if total_warnings > 0 {
        println!("{} {} warnings", color("33", "!"), total_warnings);
    }
    if total_errors == 0 && total_warnings == 0 {
        println!("{} All valid", color("32", "ok:"));
    }

    if total_errors > 0 {
        return Err(format!("{} validation errors", total_errors));
    }

    Ok(())
}

fn cmd_analyze(args: AnalyzeArgs) -> Result<(), String> {
    let text = get_input_text(&args.text, args.file.as_deref(), &args.positional)?;

    println!();
    println!(
        "{}",
        color(
            "1;36",
            "======================================================================="
        )
    );
    println!("  {}", color("1;36", "DEEP ANALYSIS"));
    println!(
        "{}",
        color(
            "1;36",
            "======================================================================="
        )
    );
    println!();

    let backends = [
        ModelBackend::Pattern,
        ModelBackend::Heuristic,
        ModelBackend::Stacked,
    ];

    let mut all_results: HashMap<String, Vec<Entity>> = HashMap::new();

    for backend in &backends {
        let model = backend.create_model()?;
        let start = Instant::now();
        let entities = model.extract_entities(&text, None).unwrap_or_default();
        let elapsed = start.elapsed();

        println!("{}:", color("1;33", backend.name()));
        println!(
            "  {} entities in {:.1}ms",
            entities.len(),
            elapsed.as_secs_f64() * 1000.0
        );

        if !entities.is_empty() {
            let mut by_type: HashMap<String, usize> = HashMap::new();
            for e in &entities {
                *by_type
                    .entry(e.entity_type.as_label().to_string())
                    .or_default() += 1;
            }
            for (t, c) in &by_type {
                println!("    {}: {}", t, c);
            }
        }
        println!();

        all_results.insert(backend.name().to_string(), entities);
    }

    // Find disagreements
    println!("{}:", color("1;33", "Model Agreement"));

    let stacked = all_results.get("stacked").cloned().unwrap_or_default();
    let pattern = all_results.get("pattern").cloned().unwrap_or_default();
    let heuristic = all_results.get("heuristic").cloned().unwrap_or_default();

    let mut all_found: Vec<&Entity> = Vec::new();
    let mut only_stacked: Vec<&Entity> = Vec::new();

    for e in &stacked {
        let in_pattern = pattern.iter().any(|p| p.start == e.start && p.end == e.end);
        let in_heuristic = heuristic
            .iter()
            .any(|s| s.start == e.start && s.end == e.end);

        if in_pattern || in_heuristic {
            all_found.push(e);
        } else {
            only_stacked.push(e);
        }
    }

    // Count entities unique to each model
    let pattern_only_count = pattern
        .iter()
        .filter(|p| !stacked.iter().any(|s| s.start == p.start && s.end == p.end))
        .count();
    let heuristic_only_count = heuristic
        .iter()
        .filter(|h| !stacked.iter().any(|s| s.start == h.start && s.end == h.end))
        .count();

    println!(
        "  Agreed (in stacked from pattern/heuristic): {} entities",
        all_found.len()
    );
    println!(
        "  Pattern-only (not in stacked): {} entities",
        pattern_only_count
    );
    println!(
        "  Heuristic-only (not in stacked): {} entities",
        heuristic_only_count
    );
    println!(
        "  Stacked-only (novel combinations): {} entities",
        only_stacked.len()
    );
    println!();

    // Show annotated text
    println!("{}:", color("1;33", "Annotated Text"));
    print_annotated_entities(&text, &stacked);
    println!();

    Ok(())
}

fn cmd_dataset(args: DatasetArgs) -> Result<(), String> {
    match args.action {
        DatasetAction::List => {
            println!();
            println!("{}", color("1;36", "Available Datasets"));
            println!();

            #[cfg(feature = "eval-advanced")]
            {
                println!("  Downloadable (with --features eval-advanced):");
                println!("    - wikigold    : WikiGold NER corpus");
                println!("    - wnut17      : WNUT 2017 emerging entities");
                println!("    - conll2003   : CoNLL 2003 (requires manual download)");
            }

            println!();
            println!("  Synthetic (always available):");
            println!("    - synthetic   : Generated test cases");
            println!("    - robustness  : Adversarial perturbations");
            println!();
        }
        DatasetAction::Info { dataset } => {
            println!("Dataset: {}", dataset);
            // TODO: Show stats about the dataset
        }
        DatasetAction::Eval {
            dataset,
            model,
            task,
        } => {
            #[cfg(feature = "eval")]
            {
                let m = model.create_model()?;

                let (name, test_cases) = if dataset == "synthetic" {
                    (
                        "synthetic".to_string(),
                        vec![
                            (
                                "Marie Curie won the Nobel Prize.".to_string(),
                                vec![
                                    anno::eval::GoldEntity {
                                        text: "Marie Curie".to_string(),
                                        original_label: "PER".to_string(),
                                        entity_type: anno::EntityType::Person,
                                        start: 0,
                                        end: 11,
                                    },
                                    anno::eval::GoldEntity {
                                        text: "Nobel Prize".to_string(),
                                        original_label: "MISC".to_string(),
                                        entity_type: anno::EntityType::Other("MISC".to_string()),
                                        start: 20,
                                        end: 31,
                                    },
                                ],
                            ),
                            (
                                "Apple Inc. is based in California.".to_string(),
                                vec![
                                    anno::eval::GoldEntity {
                                        text: "Apple Inc.".to_string(),
                                        original_label: "ORG".to_string(),
                                        entity_type: anno::EntityType::Organization,
                                        start: 0,
                                        end: 10,
                                    },
                                    anno::eval::GoldEntity {
                                        text: "California".to_string(),
                                        original_label: "LOC".to_string(),
                                        entity_type: anno::EntityType::Location,
                                        start: 24,
                                        end: 34,
                                    },
                                ],
                            ),
                            (
                                "Contact john@example.com for help.".to_string(),
                                vec![anno::eval::GoldEntity {
                                    text: "john@example.com".to_string(),
                                    original_label: "EMAIL".to_string(),
                                    entity_type: anno::EntityType::Other("EMAIL".to_string()),
                                    start: 8,
                                    end: 24,
                                }],
                            ),
                        ],
                    )
                } else {
                    // Parse dataset ID
                    let dataset_id: DatasetId = dataset
                        .parse::<DatasetId>()
                        .map_err(|e| format!("Invalid dataset '{}': {}", dataset, e))?;

                    #[cfg(not(feature = "eval-advanced"))]
                    {
                        let _ = dataset_id; // Suppress unused warning
                        return Err(
                            "Loading real datasets requires --features eval-advanced".to_string()
                        );
                    }

                    #[cfg(feature = "eval-advanced")]
                    {
                        use anno::eval::loader::DatasetLoader;

                        let loader = DatasetLoader::new()
                            .map_err(|e| format!("Failed to init dataset loader: {}", e))?;

                        println!(
                            "Loading {} (may download if not cached)...",
                            dataset_id.name()
                        );
                        let ds = loader
                            .load_or_download(dataset_id)
                            .map_err(|e| format!("Failed to load dataset: {}", e))?;

                        // Only warn if evaluating NER on non-NER dataset (not for coref/relation tasks)
                        if matches!(task, EvalTask::Ner)
                            && (dataset_id.is_coreference() || dataset_id.is_relation_extraction())
                        {
                            println!("{} Warning: Evaluating NER on non-NER dataset. Results may be empty.", color("33", "!"));
                        }

                        (ds.stats().name, ds.to_test_cases())
                    }
                };

                // Route to appropriate evaluation based on task
                match task {
                    EvalTask::Ner => {
                        #[cfg(feature = "eval-advanced")]
                        let type_mapper: Option<anno::TypeMapper> = if dataset != "synthetic" {
                            dataset
                                .parse::<DatasetId>()
                                .ok()
                                .and_then(|id| id.type_mapper())
                        } else {
                            None
                        };
                        #[cfg(not(feature = "eval-advanced"))]
                        let type_mapper: Option<anno::TypeMapper> = None;

                        println!();
                        println!("Evaluating {} on {} dataset (NER)...", model.name(), name);
                        if type_mapper.is_some() {
                            println!(
                                "  {} Using type mapping for domain-specific dataset",
                                color("33", "!")
                            );
                        }
                        println!("  Sentences: {}", test_cases.len());
                        println!();

                        let mut total_gold = 0;
                        let mut total_pred = 0;
                        let mut total_correct = 0;

                        let start_time = Instant::now();

                        // Validate gold annotations before evaluation (warn but continue)
                        #[cfg(feature = "eval-advanced")]
                        {
                            use anno::eval::validation::validate_ground_truth_entities;
                            for (text, gold) in &test_cases {
                                let validation = validate_ground_truth_entities(text, gold, false);
                                if !validation.is_valid {
                                    eprintln!(
                                        "{} Invalid gold annotations: {}",
                                        color("33", "warning:"),
                                        validation.errors.join("; ")
                                    );
                                }
                                // Note: Warnings are typically non-critical (e.g., overlapping entities)
                                // Only show first few warnings to avoid spam
                                if !validation.warnings.is_empty() && validation.warnings.len() <= 3
                                {
                                    for warning in validation.warnings.iter().take(3) {
                                        eprintln!("{} {}", color("33", "warning:"), warning);
                                    }
                                }
                            }
                        }

                        for (text, gold) in &test_cases {
                            let entities = m.extract_entities(text, None).unwrap_or_default();

                            total_gold += gold.len();
                            total_pred += entities.len();

                            // Track which predictions have been matched to prevent double-counting
                            let mut matched_pred = vec![false; entities.len()];

                            for gold_entity in gold {
                                // Apply type mapping if available
                                let gold_type = if let Some(ref mapper) = type_mapper {
                                    mapper.normalize(&gold_entity.original_label)
                                } else {
                                    anno::EntityType::from_label(&gold_entity.original_label)
                                };

                                // Match: exact span + type match (with flexible type matching)
                                // Find first unmatched prediction that matches
                                let matched = entities.iter().enumerate().any(|(i, e)| {
                                    if matched_pred[i] {
                                        return false; // Already matched
                                    }

                                    let span_match =
                                        e.start == gold_entity.start && e.end == gold_entity.end;
                                    if !span_match {
                                        return false;
                                    }

                                    // Type match with flexible matching
                                    let pred_type_str = e.entity_type.as_label();
                                    let gold_type_str = gold_type.as_label();

                                    // Exact match or flexible match
                                    let type_matches = pred_type_str == gold_type_str
                                        || types_match_flexible(pred_type_str, gold_type_str);

                                    if type_matches {
                                        matched_pred[i] = true; // Mark as matched
                                        return true;
                                    }

                                    false
                                });

                                if matched {
                                    total_correct += 1;
                                }
                            }
                        }

                        let elapsed = start_time.elapsed();

                        let p = if total_pred > 0 {
                            total_correct as f64 / total_pred as f64
                        } else {
                            0.0
                        };
                        let r = if total_gold > 0 {
                            total_correct as f64 / total_gold as f64
                        } else {
                            0.0
                        };
                        let f1 = if p + r > 0.0 {
                            2.0 * p * r / (p + r)
                        } else {
                            0.0
                        };

                        println!("Results:");
                        println!(
                            "  Gold: {}  Predicted: {}  Correct: {}",
                            total_gold, total_pred, total_correct
                        );
                        println!(
                            "  P: {:.1}%  R: {:.1}%  F1: {:.1}%",
                            p * 100.0,
                            r * 100.0,
                            f1 * 100.0
                        );
                        let ms_per_sent = if !test_cases.is_empty() {
                            elapsed.as_secs_f64() * 1000.0 / test_cases.len() as f64
                        } else {
                            0.0
                        };
                        println!(
                            "  Time: {:.1}s ({:.1}ms/sent)",
                            elapsed.as_secs_f64(),
                            ms_per_sent
                        );
                        println!();
                    }
                    EvalTask::Coref => {
                        #[cfg(not(feature = "eval-advanced"))]
                        {
                            return Err("Coreference evaluation requires --features eval-advanced"
                                .to_string());
                        }
                        #[cfg(feature = "eval-advanced")]
                        {
                            use anno::eval::coref_resolver::SimpleCorefResolver;
                            use anno::eval::loader::DatasetLoader;

                            if dataset == "synthetic" {
                                return Err("Coreference evaluation requires a real dataset (e.g., gap, preco, litbank)".to_string());
                            }

                            let dataset_id: DatasetId = dataset
                                .parse::<DatasetId>()
                                .map_err(|e| format!("Invalid dataset '{}': {}", dataset, e))?;

                            if !dataset_id.is_coreference() {
                                return Err(format!("Dataset '{}' is not a coreference dataset. Use: gap, preco, or litbank", dataset));
                            }

                            let loader = DatasetLoader::new()
                                .map_err(|e| format!("Failed to init dataset loader: {}", e))?;

                            println!();
                            println!(
                                "Evaluating coreference resolution on {} dataset...",
                                dataset_id.name()
                            );
                            println!("Loading dataset (may download if not cached)...");

                            let gold_docs =
                                loader.load_or_download_coref(dataset_id).map_err(|e| {
                                    format!("Failed to load coreference dataset: {}", e)
                                })?;

                            println!("  Documents: {}", gold_docs.len());
                            println!();

                            let resolver = SimpleCorefResolver::default();
                            let mut all_pred_chains: Vec<Vec<anno::eval::coref::CorefChain>> =
                                Vec::new();
                            let mut all_gold_chains: Vec<&[anno::eval::coref::CorefChain]> =
                                Vec::new();
                            let start_time = Instant::now();

                            for doc in &gold_docs {
                                let text = doc.text.as_str();
                                all_gold_chains.push(&doc.chains);

                                // Extract entities using NER
                                let entities = m.extract_entities(text, None).unwrap_or_default();

                                // Resolve coreference
                                let pred_chains = resolver.resolve_to_chains(&entities);
                                all_pred_chains.push(pred_chains);
                            }

                            let elapsed = start_time.elapsed();

                            // Build document pairs
                            let document_pairs: Vec<_> = all_pred_chains
                                .iter()
                                .zip(all_gold_chains.iter())
                                .map(|(pred, gold)| (pred.as_slice(), *gold))
                                .collect();

                            // Compute aggregate metrics
                            let results =
                                anno::eval::coref_metrics::AggregateCorefEvaluation::compute(
                                    &document_pairs,
                                );

                            println!("Results:");
                            println!("  CoNLL F1: {:.3}", results.mean.conll_f1);
                            println!(
                                "  MUC: P={:.3} R={:.3} F1={:.3}",
                                results.mean.muc.precision,
                                results.mean.muc.recall,
                                results.mean.muc.f1
                            );
                            println!(
                                "  B³: P={:.3} R={:.3} F1={:.3}",
                                results.mean.b_cubed.precision,
                                results.mean.b_cubed.recall,
                                results.mean.b_cubed.f1
                            );
                            println!(
                                "  CEAF-e: P={:.3} R={:.3} F1={:.3}",
                                results.mean.ceaf_e.precision,
                                results.mean.ceaf_e.recall,
                                results.mean.ceaf_e.f1
                            );
                            println!(
                                "  LEA: P={:.3} R={:.3} F1={:.3}",
                                results.mean.lea.precision,
                                results.mean.lea.recall,
                                results.mean.lea.f1
                            );
                            println!(
                                "  BLANC: P={:.3} R={:.3} F1={:.3}",
                                results.mean.blanc.precision,
                                results.mean.blanc.recall,
                                results.mean.blanc.f1
                            );
                            println!("  Documents: {}", results.num_documents);
                            println!("  Time: {:.1}s", elapsed.as_secs_f64());
                            println!();
                        }
                    }
                    EvalTask::Relation => {
                        #[cfg(not(feature = "eval-advanced"))]
                        {
                            return Err(
                                "Relation extraction evaluation requires --features eval-advanced"
                                    .to_string(),
                            );
                        }
                        #[cfg(feature = "eval-advanced")]
                        {
                            use anno::backends::inference::RelationExtractor;
                            use anno::eval::loader::DatasetLoader;
                            use anno::eval::relation::{
                                evaluate_relations, RelationEvalConfig, RelationPrediction,
                            };

                            if dataset == "synthetic" {
                                return Err("Relation extraction evaluation requires a real dataset (e.g., docred, retacred)".to_string());
                            }

                            let dataset_id: DatasetId = dataset
                                .parse::<DatasetId>()
                                .map_err(|e| format!("Invalid dataset '{}': {}", dataset, e))?;

                            if !dataset_id.is_relation_extraction() {
                                return Err(format!("Dataset '{}' is not a relation extraction dataset. Use: docred or retacred", dataset));
                            }

                            let loader = DatasetLoader::new()
                                .map_err(|e| format!("Failed to init dataset loader: {}", e))?;

                            println!();
                            println!(
                                "Evaluating relation extraction on {} dataset...",
                                dataset_id.name()
                            );
                            println!("Loading dataset (may download if not cached)...");

                            let gold_docs = loader
                                .load_or_download_relation(dataset_id)
                                .map_err(|e| format!("Failed to load relation dataset: {}", e))?;

                            println!("  Documents: {}", gold_docs.len());
                            println!();

                            // Try to use RelationExtractor if available (e.g., GLiNER2)
                            // Otherwise fall back to entity-pair heuristic

                            // Collect entity types and relation types from gold data
                            let mut entity_types = std::collections::HashSet::new();
                            let mut relation_types = std::collections::HashSet::new();
                            for doc in &gold_docs {
                                for rel in &doc.relations {
                                    entity_types.insert(rel.head_type.clone());
                                    entity_types.insert(rel.tail_type.clone());
                                    relation_types.insert(rel.relation_type.clone());
                                }
                            }

                            let entity_types_vec: Vec<&str> =
                                entity_types.iter().map(|s| s.as_str()).collect();
                            let relation_types_vec: Vec<&str> =
                                relation_types.iter().map(|s| s.as_str()).collect();

                            println!("  Entity types: {}", entity_types_vec.join(", "));
                            println!(
                                "  Relation types: {} ({} total)",
                                relation_types_vec.len(),
                                relation_types_vec
                                    .iter()
                                    .take(5)
                                    .cloned()
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            );
                            println!();

                            // Check if we can use GLiNER2 for relation extraction
                            let use_relation_extractor: Option<Box<dyn RelationExtractor>> = {
                                #[cfg(feature = "onnx")]
                                {
                                    // Try to create GLiNER2 multitask model for relation extraction
                                    if let Ok(gliner2) =
                                        anno::backends::gliner2::GLiNER2Onnx::from_pretrained(
                                            "onnx-community/gliner-multitask-large-v0.5",
                                        )
                                    {
                                        Some(Box::new(gliner2) as Box<dyn RelationExtractor>)
                                    } else {
                                        None
                                    }
                                }
                                #[cfg(not(feature = "onnx"))]
                                {
                                    None
                                }
                            };

                            let mut all_gold = Vec::new();
                            let mut all_pred = Vec::new();
                            let start_time = Instant::now();

                            if let Some(ref rel_extractor) = use_relation_extractor {
                                println!("{} Using GLiNER2 RelationExtractor (heuristic-based pattern matching)", color("32", "✓"));
                                println!("  Note: This uses pattern matching on text, not a neural relation model.",);
                                println!();

                                for doc in &gold_docs {
                                    let text = doc.text.as_str();
                                    all_gold.extend(doc.relations.clone());

                                    // Use RelationExtractor
                                    match rel_extractor.extract_with_relations(
                                        text,
                                        &entity_types_vec,
                                        &relation_types_vec,
                                        0.5,
                                    ) {
                                        Ok(result) => {
                                            // Convert RelationTriples to RelationPredictions
                                            for triple in &result.relations {
                                                if let Some(pred) =
                                                    RelationPrediction::from_triple_with_entities(
                                                        triple,
                                                        &result.entities,
                                                    )
                                                {
                                                    all_pred.push(pred);
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!(
                                                "{} Relation extraction failed: {}",
                                                color("33", "warning:"),
                                                e
                                            );
                                            // Fall back to entity-pair heuristic for this document
                                            let entities =
                                                m.extract_entities(text, None).unwrap_or_default();
                                            all_pred.extend(create_entity_pair_relations(
                                                &entities,
                                                text,
                                                &relation_types_vec,
                                            ));
                                        }
                                    }
                                }
                            } else {
                                println!("{} Using entity-pair heuristic (GLiNER2 multitask not available)", color("33", "!"));
                                println!();

                                for doc in &gold_docs {
                                    let text = doc.text.as_str();
                                    all_gold.extend(doc.relations.clone());

                                    // Extract entities using NER
                                    let entities =
                                        m.extract_entities(text, None).unwrap_or_default();

                                    // Create relation predictions from entity pairs
                                    all_pred.extend(create_entity_pair_relations(
                                        &entities,
                                        text,
                                        &relation_types_vec,
                                    ));
                                }
                            }

                            let elapsed = start_time.elapsed();

                            // Evaluate relations
                            // Note: require_entity_type_match=false because entity types may differ
                            // (e.g., gold uses "person" but pred uses "Person", or "PER" vs "PERSON")
                            let config = RelationEvalConfig {
                                overlap_threshold: 0.5,
                                require_entity_type_match: false, // More lenient for evaluation
                                directed_relations: true,
                            };
                            let metrics = evaluate_relations(&all_gold, &all_pred, &config);

                            // Output results (human-readable by default)
                            println!();
                            println!("{}", color("1;36", "======================================================================="));
                            println!(
                                "  {}  model={}  time={:.1}s",
                                color("1;36", "RELATION EXTRACTION EVALUATION"),
                                model.name(),
                                elapsed.as_secs_f64()
                            );
                            println!("{}", color("1;36", "======================================================================="));
                            println!();
                            println!("{}", metrics.to_string_human(false)); // verbose=false for now
                            println!();
                        }
                    }
                }
            }
            #[cfg(not(feature = "eval"))]
            {
                let _ = (dataset, model, task);
                return Err("Dataset evaluation requires --features eval".to_string());
            }
        }
    }

    Ok(())
}

#[cfg(feature = "eval-advanced")]
fn cmd_benchmark(args: BenchmarkArgs) -> Result<(), String> {
    use anno::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
    use anno::eval::task_mapping::Task;
    use std::fs;

    println!("=== Comprehensive Task-Dataset-Backend Evaluation ===\n");

    // Parse tasks
    let tasks = if let Some(task_strs) = args.tasks {
        let mut parsed = Vec::new();
        for t in task_strs {
            match t.to_lowercase().as_str() {
                "ner" | "ner_task" => parsed.push(Task::NER),
                "coref" | "coreference" | "intradoc_coref" => parsed.push(Task::IntraDocCoref),
                "relation" | "relation_extraction" => parsed.push(Task::RelationExtraction),
                other => {
                    return Err(format!(
                        "Unknown task: {}. Use: ner, coref, relation",
                        other
                    ));
                }
            }
        }
        parsed
    } else {
        Task::all().to_vec()
    };

    // Parse datasets
    let datasets = if let Some(dataset_strs) = args.datasets {
        let mut parsed = Vec::new();
        for d in dataset_strs {
            let dataset_id: DatasetId = d
                .parse()
                .map_err(|e| format!("Invalid dataset '{}': {}", d, e))?;
            parsed.push(dataset_id);
        }
        parsed
    } else {
        vec![] // Empty = use all suitable datasets
    };

    // Parse backends
    let backends = args.backends.unwrap_or_default();

    // Create evaluator
    let evaluator =
        TaskEvaluator::new().map_err(|e| format!("Failed to create evaluator: {}", e))?;

    // Configure evaluation
    let config = TaskEvalConfig {
        tasks,
        datasets,
        backends,
        max_examples: args.max_examples,
        seed: args.seed,
        require_cached: args.cached_only,
    };

    println!("Running comprehensive evaluation...");
    println!("Tasks: {:?}", config.tasks);
    if !config.datasets.is_empty() {
        println!("Datasets: {:?}", config.datasets);
    } else {
        println!("Datasets: all suitable datasets");
    }
    if !config.backends.is_empty() {
        println!("Backends: {:?}", config.backends);
    } else {
        println!("Backends: all compatible backends");
    }
    if let Some(max) = config.max_examples {
        println!("Max examples per dataset: {}", max);
    }
    if let Some(seed) = config.seed {
        println!("Random seed: {}", seed);
    }
    println!();

    // Run evaluation
    let results = evaluator
        .evaluate_all(config)
        .map_err(|e| format!("Evaluation failed: {}", e))?;

    // Print summary
    println!("=== Evaluation Summary ===");
    println!("Total combinations: {}", results.summary.total_combinations);
    println!("Successful: {}", results.summary.successful);
    println!(
        "Skipped (feature not available): {}",
        results.summary.skipped
    );
    println!("Failed (actual errors): {}", results.summary.failed);
    println!("\nTasks evaluated: {}", results.summary.tasks.len());
    println!("Datasets used: {}", results.summary.datasets.len());
    println!("Backends tested: {}", results.summary.backends.len());
    println!();

    // Generate markdown report
    let report = results.to_markdown();

    // Output report
    if let Some(output_path) = &args.output {
        fs::write(output_path, &report)
            .map_err(|e| format!("Failed to write report to {}: {}", output_path, e))?;
        println!("Report saved to: {}", output_path);
    } else {
        println!("=== Markdown Report ===");
        println!("{}", report);
    }

    Ok(())
}

/// Create relation predictions from entity pairs using heuristics.
///
/// # Bugs Fixed:
/// - Character vs byte offset: Now uses character offsets consistently
/// - Bounds validation: Validates entity spans are within text bounds
/// - Distance limit: Configurable (default 200 chars) to catch cross-sentence relations
#[cfg(feature = "eval-advanced")]
fn create_entity_pair_relations(
    entities: &[Entity],
    text: &str,
    relation_types: &[&str],
) -> Vec<anno::eval::relation::RelationPrediction> {
    use anno::eval::relation::RelationPrediction;

    let text_char_len = text.chars().count();
    let max_distance = 200; // Increased from 100 to catch cross-sentence relations

    let mut pred_relations = Vec::new();

    // Validate entities first to avoid panics
    let valid_entities: Vec<&Entity> = entities
        .iter()
        .filter(|e| e.start < e.end && e.end <= text_char_len && e.start < text_char_len)
        .collect();

    // Limit to avoid O(n²) explosion with many entities
    // Only consider pairs from first 50 entities to keep it tractable
    let max_entities = 50.min(valid_entities.len());

    for i in 0..max_entities {
        for j in (i + 1)..max_entities {
            let head = valid_entities[i];
            let tail = valid_entities[j];

            // Calculate distance using character offsets
            let distance = if tail.start >= head.end {
                tail.start - head.end
            } else if head.start >= tail.end {
                head.start - tail.end
            } else {
                // Overlapping entities - skip (they can't have a relation)
                continue;
            };

            if distance > max_distance {
                continue;
            }

            // Extract text between entities using character offsets (not byte offsets)
            let between_text = if head.end <= tail.start {
                text.chars()
                    .skip(head.end)
                    .take(tail.start - head.end)
                    .collect::<String>()
            } else {
                text.chars()
                    .skip(tail.end)
                    .take(head.start - tail.end)
                    .collect::<String>()
            };

            // Simple pattern matching for common relations
            let between_lower = between_text.to_lowercase();
            let rel_type = if between_lower.contains("founded") || between_lower.contains("founder")
            {
                "FOUNDED"
            } else if between_lower.contains("works for")
                || between_lower.contains("employee")
                || between_lower.contains("employed")
            {
                "WORKS_FOR"
            } else if between_lower.contains("located in")
                || between_lower.contains("based in")
                || between_lower.contains("in ")
            {
                "LOCATED_IN"
            } else if between_lower.contains("born in") {
                "BORN_IN"
            } else {
                // Use first relation type from gold data as fallback, or "RELATED"
                relation_types.first().copied().unwrap_or("RELATED")
            };

            pred_relations.push(RelationPrediction {
                head_span: (head.start, head.end),
                head_type: head.entity_type.as_label().to_string(),
                tail_span: (tail.start, tail.end),
                tail_type: tail.entity_type.as_label().to_string(),
                relation_type: rel_type.to_string(),
                confidence: 0.5,
            });
        }
    }

    pred_relations
}

fn cmd_info() -> Result<(), String> {
    println!();
    println!("{}", color("1;36", "anno"));
    println!("  Information Extraction: NER + Coreference + Relations + Entity Linking");
    println!();
    println!("{}:", color("1;33", "Version"));
    println!("  {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("{}:", color("1;33", "Available Models"));

    let models: &[(&str, &str, bool)] = &[
        ("pattern", "Date/Time/Money/Email/URL/Phone patterns", true),
        (
            "heuristic",
            "Person/Org/Location heuristics (HeuristicNER)",
            true,
        ),
        ("stacked", "Pattern + Heuristic (default)", true),
        ("gliner", "GLiNER zero-shot NER", cfg!(feature = "onnx")),
        ("nuner", "NuNER token classifier", cfg!(feature = "onnx")),
        ("w2ner", "W2NER nested entities", cfg!(feature = "onnx")),
        (
            "gliner-candle",
            "GLiNER via Candle",
            cfg!(feature = "candle"),
        ),
    ];

    for (name, desc, available) in models {
        let status = if *available {
            color("32", "+")
        } else {
            color("90", "-")
        };
        println!("  {} {} - {}", status, name, desc);
    }
    println!();

    let model = StackedNER::default();
    println!("{}:", color("1;33", "Supported Entity Types (stacked)"));
    for t in model.supported_types() {
        let color_code = type_color(t.as_label());
        println!("  {} {}", color(color_code, "*"), t.as_label());
    }
    println!();

    println!("{}:", color("1;33", "Enabled Features"));
    let features: &[(&str, &str, bool)] = &[
        ("eval", "Evaluation framework", cfg!(feature = "eval")),
        (
            "eval-advanced",
            "Robustness, calibration, datasets",
            cfg!(feature = "eval-advanced"),
        ),
        ("onnx", "ONNX Runtime backend", cfg!(feature = "onnx")),
        ("candle", "Candle backend", cfg!(feature = "candle")),
        (
            "discourse",
            "Event/anaphora analysis",
            cfg!(feature = "discourse"),
        ),
    ];

    for (name, desc, enabled) in features {
        let status = if *enabled {
            color("32", "+")
        } else {
            color("90", "-")
        };
        println!("  {} {} - {}", status, name, desc);
    }
    println!();

    Ok(())
}

// ============================================================================
// Helpers
// ============================================================================

fn get_input_text(
    text: &Option<String>,
    file: Option<&str>,
    positional: &[String],
) -> Result<String, String> {
    // Check explicit text arg
    if let Some(t) = text {
        return Ok(t.clone());
    }

    // Check file arg
    if let Some(f) = file {
        return fs::read_to_string(f).map_err(|e| format!("Failed to read {}: {}", f, e));
    }

    // Check positional args
    if !positional.is_empty() {
        return Ok(positional.join(" "));
    }

    // Try stdin
    if !io::stdin().is_terminal() {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("Failed to read stdin: {}", e))?;
        if !buf.is_empty() {
            return Ok(buf);
        }
    }

    Err("No input text provided. Use -t 'text' or -f file or pipe via stdin".to_string())
}

#[derive(Debug, Clone)]
struct GoldSpec {
    text: String,
    label: String,
    start: usize,
    end: usize,
}

/// Parse gold spec with format: "text:label:start:end"
/// Uses rsplit to handle text containing colons (like URLs)
fn parse_gold_spec(s: &str) -> Option<GoldSpec> {
    // Split from right to handle colons in text
    let parts: Vec<&str> = s.rsplitn(4, ':').collect();
    if parts.len() < 4 {
        return None;
    }

    let end: usize = parts[0].parse().ok()?;
    let start: usize = parts[1].parse().ok()?;
    let label = parts[2].to_string();
    let text = parts[3].to_string();

    Some(GoldSpec {
        text,
        label,
        start,
        end,
    })
}

fn load_gold_from_file(path: &str) -> Result<Vec<GoldSpec>, String> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("Failed to read {}: {}", path, e))?;
    let mut gold = Vec::new();
    let mut warnings = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        let entry: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| format!("Invalid JSON in gold file at line {}: {}", line_num + 1, e))?;

        if let Some(entities) = entry["entities"].as_array() {
            for (i, ent) in entities.iter().enumerate() {
                // Validate required fields are present
                let start = match ent["start"].as_u64() {
                    Some(v) => v as usize,
                    None => {
                        warnings.push(format!(
                            "{}:{}: entity[{}] missing 'start' field, defaulting to 0",
                            path,
                            line_num + 1,
                            i
                        ));
                        0
                    }
                };
                let end = match ent["end"].as_u64() {
                    Some(v) => v as usize,
                    None => {
                        warnings.push(format!(
                            "{}:{}: entity[{}] missing 'end' field, defaulting to 0",
                            path,
                            line_num + 1,
                            i
                        ));
                        0
                    }
                };

                gold.push(GoldSpec {
                    text: ent["text"].as_str().unwrap_or("").to_string(),
                    label: ent["type"]
                        .as_str()
                        .or(ent["label"].as_str())
                        .unwrap_or("UNK")
                        .to_string(),
                    start,
                    end,
                });
            }
        }
    }

    // Report warnings to stderr
    for warning in &warnings {
        eprintln!("{} {}", color("33", "warning:"), warning);
    }

    Ok(gold)
}

/// Detect if entity at position is negated
fn is_negated(text: &str, entity_start: usize) -> bool {
    let prefix: String = text.chars().take(entity_start).collect();
    let words: Vec<&str> = prefix.split_whitespace().collect();
    let last_words: Vec<&str> = words.iter().rev().take(3).copied().collect();

    const NEGATION_WORDS: &[&str] = &[
        "not",
        "no",
        "never",
        "none",
        "neither",
        "nor",
        "without",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "wouldn't",
        "couldn't",
        "shouldn't",
    ];

    for word in &last_words {
        if NEGATION_WORDS.contains(&word.to_lowercase().as_str()) {
            return true;
        }
    }

    false
}

/// Detect quantifier before entity
fn detect_quantifier(text: &str, entity_start: usize) -> Option<Quantifier> {
    let prefix: String = text.chars().take(entity_start).collect();
    let words: Vec<&str> = prefix.split_whitespace().collect();

    words
        .last()
        .and_then(|word| match word.to_lowercase().as_str() {
            "every" | "all" | "each" | "any" => Some(Quantifier::Universal),
            "some" | "certain" | "a" | "an" => Some(Quantifier::Existential),
            "no" | "none" => Some(Quantifier::None),
            "the" | "this" | "that" | "these" | "those" => Some(Quantifier::Definite),
            _ => None,
        })
}

/// Flexible type matching for evaluation (handles PER/PERSON, LOC/LOCATION, etc.)
fn types_match_flexible(pred: &str, gold: &str) -> bool {
    let pred = pred.to_uppercase();
    let gold = gold.to_uppercase();

    if pred == gold {
        return true;
    }

    // Allow common mappings
    match (pred.as_str(), gold.as_str()) {
        // Person
        ("PERSON", "PER") | ("PER", "PERSON") => true,
        // Location
        ("LOCATION", "LOC") | ("LOC", "LOCATION") | ("LOCATION", "GPE") | ("GPE", "LOCATION") => {
            true
        }
        // Organization
        ("ORGANIZATION", "ORG") | ("ORG", "ORGANIZATION") => true,
        // Date/Time
        ("DATE", "YEAR") | ("YEAR", "DATE") | ("DATE", "HOURS") => true,
        _ => false,
    }
}

fn color(code: &str, text: &str) -> String {
    if io::stdout().is_terminal() {
        format!("\x1b[{}m{}\x1b[0m", code, text)
    } else {
        text.to_string()
    }
}

/// Resolve coreference by grouping signals into tracks.
///
/// This uses a simple rule-based approach:
/// 1. Named entities of the same type that overlap in their canonical form
/// 2. Pronouns detected in text and linked to nearest compatible antecedent
fn resolve_coreference(doc: &mut GroundedDocument, text: &str, signal_ids: &[u64]) {
    // Pronouns by gender
    let male_pronouns = ["he", "him", "his"];
    let female_pronouns = ["she", "her", "hers"];
    let neutral_pronouns = ["they", "them", "their", "theirs"]; // Can refer to any
    let org_pronouns = ["it", "its"];

    // First, detect pronouns in text that weren't found by NER
    // (pronoun_id, pronoun_type: "male", "female", "org", "any")
    let mut pronoun_signals: Vec<(u64, &str)> = Vec::new();

    // Build byte-to-char offset mapping for proper conversion
    let byte_to_char: Vec<usize> = text.char_indices().map(|(byte_idx, _)| byte_idx).collect();
    let char_count = text.chars().count();

    // Helper to convert byte offset to char offset
    let byte_to_char_offset = |byte_offset: usize| -> usize {
        byte_to_char
            .iter()
            .position(|&b| b == byte_offset)
            .unwrap_or_else(|| {
                // If exact match not found, it's at the end or between chars
                if byte_offset >= text.len() {
                    char_count
                } else {
                    // Find the char that contains this byte
                    byte_to_char
                        .iter()
                        .take_while(|&&b| b < byte_offset)
                        .count()
                }
            })
    };

    // Find pronouns in text and add them as signals
    let text_lower = text.to_lowercase();
    let chars: Vec<char> = text.chars().collect();

    for (pronouns, ptype) in [
        (&male_pronouns[..], "male"),
        (&female_pronouns[..], "female"),
        (&org_pronouns[..], "org"),
        (&neutral_pronouns[..], "any"),
    ] {
        for &pronoun in pronouns {
            // Find all occurrences (byte offsets from find())
            let mut byte_start = 0;
            while let Some(pos) = text_lower[byte_start..].find(pronoun) {
                let abs_byte_start = byte_start + pos;
                let abs_byte_end = abs_byte_start + pronoun.len();

                // Convert to character offsets for Location::Text
                let char_start = byte_to_char_offset(abs_byte_start);
                let char_end = byte_to_char_offset(abs_byte_end);

                // Check word boundaries using character indices
                let is_word_start = char_start == 0
                    || !chars
                        .get(char_start.saturating_sub(1))
                        .map(|c| c.is_alphanumeric())
                        .unwrap_or(false);
                let is_word_end = char_end >= char_count
                    || !chars
                        .get(char_end)
                        .map(|c| c.is_alphanumeric())
                        .unwrap_or(false);

                if is_word_start && is_word_end {
                    // Check if this position already has a signal (using char offsets)
                    let already_exists = doc.signals().iter().any(|s| {
                        if let Location::Text {
                            start: s_start,
                            end: s_end,
                        } = &s.location
                        {
                            *s_start == char_start && *s_end == char_end
                        } else {
                            false
                        }
                    });

                    if !already_exists {
                        // Add pronoun as a signal (use byte slice for surface text)
                        let surface = &text[abs_byte_start..abs_byte_end];
                        let signal = Signal::new(
                            0,
                            Location::text(char_start, char_end), // Use char offsets
                            surface,
                            "PRON", // Special label for pronouns
                            0.9,
                        );
                        let sig_id = doc.add_signal(signal);
                        pronoun_signals.push((sig_id, ptype));
                    }
                }

                // Advance by one byte to find next occurrence
                byte_start = abs_byte_start + 1;
            }
        }
    }

    // Group NER signals by type
    let mut per_signals: Vec<u64> = Vec::new();
    let mut org_signals: Vec<u64> = Vec::new();
    let mut loc_signals: Vec<u64> = Vec::new();

    for &sig_id in signal_ids {
        if let Some(sig) = doc.get_signal(sig_id) {
            let label_lower = sig.label.to_lowercase();
            match label_lower.as_str() {
                "per" | "person" => per_signals.push(sig_id),
                "org" | "organization" => org_signals.push(sig_id),
                "loc" | "location" | "gpe" => loc_signals.push(sig_id),
                _ => {}
            }
        }
    }

    // Create tracks from named entities (group same-type entities by canonical form)
    let mut track_assignments: HashMap<u64, u64> = HashMap::new(); // signal_id -> track_id

    // For each entity type, create tracks by grouping similar surface forms
    for signals in [&per_signals, &org_signals, &loc_signals] {
        if signals.is_empty() {
            continue;
        }

        // Simple grouping: each unique entity gets its own track
        let mut canonical_groups: HashMap<String, Vec<u64>> = HashMap::new();

        for &sig_id in signals {
            if let Some(sig) = doc.get_signal(sig_id) {
                // Use lowercase canonical form for grouping
                let canonical = normalize_entity_name(&sig.surface);
                canonical_groups.entry(canonical).or_default().push(sig_id);
            }
        }

        // Create a track for each group
        for (canonical, group_signals) in canonical_groups {
            let track_id = doc.create_track_from_signals(&canonical, &group_signals);
            if let Some(tid) = track_id {
                for &sig_id in &group_signals {
                    track_assignments.insert(sig_id, tid);
                }
            }
        }
    }

    // Link pronouns to nearest compatible antecedent's track
    for (pronoun_id, pronoun_type) in &pronoun_signals {
        let pronoun_sig = match doc.get_signal(*pronoun_id) {
            Some(s) => s.clone(),
            None => continue,
        };

        let pronoun_start = match &pronoun_sig.location {
            Location::Text { start, .. } => *start,
            _ => continue,
        };

        // For person pronouns, we need to filter by gender compatibility
        let compatible_signals: Vec<u64> = match *pronoun_type {
            "male" => {
                // Filter to likely male names
                per_signals
                    .iter()
                    .filter(|&&id| {
                        doc.get_signal(id)
                            .map(|s| is_likely_male(&s.surface))
                            .unwrap_or(false)
                    })
                    .cloned()
                    .collect()
            }
            "female" => {
                // Filter to likely female names
                per_signals
                    .iter()
                    .filter(|&&id| {
                        doc.get_signal(id)
                            .map(|s| is_likely_female(&s.surface))
                            .unwrap_or(false)
                    })
                    .cloned()
                    .collect()
            }
            "org" => org_signals.clone(),
            "any" => per_signals
                .iter()
                .chain(org_signals.iter())
                .cloned()
                .collect(),
            _ => continue,
        };

        let mut nearest: Option<(u64, usize)> = None; // (signal_id, distance)

        for sig_id in &compatible_signals {
            if let Some(sig) = doc.get_signal(*sig_id) {
                if let Location::Text { end, .. } = &sig.location {
                    if *end < pronoun_start {
                        let distance = pronoun_start - end;
                        if nearest.map_or(true, |(_, prev_dist)| distance < prev_dist) {
                            nearest = Some((*sig_id, distance));
                        }
                    }
                }
            }
        }

        // Add pronoun to the track of its antecedent
        if let Some((antecedent_id, _)) = nearest {
            if let Some(&track_id) = track_assignments.get(&antecedent_id) {
                // Get position (number of signals already in track)
                let position = doc
                    .get_track(track_id)
                    .map(|t| t.signals.len() as u32)
                    .unwrap_or(0);
                // Add pronoun signal to this track (updates index)
                if doc.add_signal_to_track(*pronoun_id, track_id, position) {
                    track_assignments.insert(*pronoun_id, track_id);
                }
            }
        }
    }
}

/// Simple heuristic for determining if a name is likely male.
fn is_likely_male(name: &str) -> bool {
    // Get first name (first word)
    let first_name = name.split_whitespace().next().unwrap_or("").to_lowercase();

    // Common male first names
    let male_names = [
        "james", "john", "robert", "michael", "william", "david", "richard", "joseph", "thomas",
        "charles", "barack", "donald", "joe", "george", "bill", "vladimir", "emmanuel", "boris",
        "xi", "narendra", "justin", "elon", "jeff", "mark", "steve", "tim", "satya", "sundar",
        "albert", "isaac", "stephen", "neil", "peter", "paul", "matthew", "andrew", "philip",
        "simon",
    ];

    male_names.contains(&first_name.as_str())
}

/// Simple heuristic for determining if a name is likely female.
fn is_likely_female(name: &str) -> bool {
    // Get first name (first word)
    let first_name = name.split_whitespace().next().unwrap_or("").to_lowercase();

    // Common female first names
    let female_names = [
        "mary",
        "patricia",
        "jennifer",
        "linda",
        "elizabeth",
        "angela",
        "marie",
        "susan",
        "margaret",
        "dorothy",
        "hillary",
        "nancy",
        "kamala",
        "michelle",
        "melania",
        "jill",
        "theresa",
        "ursula",
        "christine",
        "sanna",
        "jacinda",
        "oprah",
        "beyonce",
        "taylor",
        "sheryl",
        "marissa",
        "susan",
        "ginni",
        "diana",
        "catherine",
        "anne",
        "victoria",
        "queen",
        "jane",
        "sarah",
    ];

    female_names.contains(&first_name.as_str())
}

/// Link tracks to KB identities.
///
/// Creates placeholder Wikidata-style identities for each track.
/// In a production system, this would query a real KB like Wikidata.
fn link_tracks_to_kb(doc: &mut GroundedDocument) {
    // Well-known entities with Wikidata IDs
    let known_entities: HashMap<&str, (&str, &str)> = [
        (
            "barack obama",
            ("Q76", "44th President of the United States"),
        ),
        ("angela merkel", ("Q567", "Chancellor of Germany 2005-2021")),
        ("berlin", ("Q64", "Capital of Germany")),
        ("nato", ("Q7184", "North Atlantic Treaty Organization")),
        (
            "donald trump",
            ("Q22686", "45th President of the United States"),
        ),
        (
            "joe biden",
            ("Q6279", "46th President of the United States"),
        ),
        ("vladimir putin", ("Q7747", "President of Russia")),
        ("emmanuel macron", ("Q3052772", "President of France")),
        ("elon musk", ("Q317521", "CEO of Tesla and SpaceX")),
        ("marie curie", ("Q7186", "Physicist and chemist")),
        ("albert einstein", ("Q937", "Theoretical physicist")),
        ("new york", ("Q60", "City in New York State")),
        ("london", ("Q84", "Capital of the United Kingdom")),
        ("paris", ("Q90", "Capital of France")),
        ("google", ("Q95", "American technology company")),
        ("apple", ("Q312", "American technology company")),
        ("microsoft", ("Q2283", "American technology company")),
        ("united nations", ("Q1065", "International organization")),
        ("european union", ("Q458", "Political and economic union")),
    ]
    .into_iter()
    .collect();

    // Collect track IDs first to avoid borrow issues
    let track_ids: Vec<u64> = doc.tracks().map(|t| t.id).collect();

    for track_id in track_ids {
        let (canonical, entity_type) = {
            let track = match doc.get_track(track_id) {
                Some(t) => t,
                None => continue,
            };
            (track.canonical_surface.clone(), track.entity_type.clone())
        };

        let canonical_lower = canonical.to_lowercase();

        // Look up in known entities
        if let Some(&(qid, description)) = known_entities.get(canonical_lower.as_str()) {
            // Create identity from KB
            let mut identity = Identity::from_kb(
                0, // Will be assigned by add_identity
                &canonical, "wikidata", qid,
            );
            identity.aliases.push(description.to_string());
            if let Some(etype) = &entity_type {
                identity.entity_type = Some(etype.clone());
            }

            let identity_id = doc.add_identity(identity);
            doc.link_track_to_identity(track_id, identity_id);
        } else {
            // Create placeholder identity without KB link
            let identity = Identity::new(0, &canonical);
            let identity_id = doc.add_identity(identity);
            doc.link_track_to_identity(track_id, identity_id);
        }
    }
}

/// Normalize an entity name for grouping (lowercase, trim)
fn normalize_entity_name(name: &str) -> String {
    name.to_lowercase().trim().to_string()
}

fn type_color(typ: &str) -> &'static str {
    match typ.to_lowercase().as_str() {
        "person" | "per" => "1;34",
        "organization" | "org" => "1;32",
        "location" | "loc" | "gpe" => "1;33",
        "date" | "time" => "1;35",
        "money" | "percent" => "1;36",
        "email" | "url" | "phone" => "36",
        _ => "1;37",
    }
}

fn metric_colored(value: f64) -> String {
    let code = if value >= 90.0 {
        "1;32"
    } else if value >= 70.0 {
        "1;33"
    } else if value >= 50.0 {
        "33"
    } else {
        "1;31"
    };
    color(code, &format!("{:5.1}", value))
}

fn confidence_bar(conf: f32) -> String {
    // Clamp to valid range to prevent underflow if conf > 1.0
    let filled = ((conf * 10.0).round() as usize).min(10);
    let empty = 10 - filled;
    let code = if conf >= 0.9 {
        "32"
    } else if conf >= 0.7 {
        "33"
    } else {
        "31"
    };
    format!(
        "{}{} {:3.0}%",
        color(code, &"#".repeat(filled)),
        color("90", &".".repeat(empty)),
        conf * 100.0
    )
}

fn print_signals(doc: &GroundedDocument, text: &str, verbose: bool) {
    let mut by_type: HashMap<String, Vec<&Signal<Location>>> = HashMap::new();
    for s in doc.signals() {
        by_type.entry(s.label().to_string()).or_default().push(s);
    }

    for (typ, signals) in &by_type {
        let col = type_color(typ);
        println!("  {} ({}):", color(col, typ), signals.len());
        for s in signals {
            let (start, end) = s.text_offsets().unwrap_or((0, 0));
            let neg = if s.negated {
                color("31", " [NEG]")
            } else {
                String::new()
            };
            let quant = s
                .quantifier
                .map(|q| color("35", &format!(" [{:?}]", q)))
                .unwrap_or_default();

            println!(
                "    [{:3},{:3}) {} \"{}\"{}{}",
                start,
                end,
                confidence_bar(s.confidence),
                s.surface(),
                neg,
                quant
            );

            if verbose {
                let ctx_start = start.saturating_sub(15);
                let ctx_end = (end + 15).min(text.chars().count());
                let before: String = text
                    .chars()
                    .skip(ctx_start)
                    .take(start - ctx_start)
                    .collect();
                let entity: String = text.chars().skip(start).take(end - start).collect();
                let after: String = text.chars().skip(end).take(ctx_end - end).collect();
                println!(
                    "           {}{}{}{}{}",
                    color("90", "..."),
                    color("90", &before),
                    color("1;33", &entity),
                    color("90", &after),
                    color("90", "...")
                );
            }
        }
    }
}

fn print_annotated_entities(text: &str, entities: &[Entity]) {
    let mut sorted: Vec<&Entity> = entities.iter().collect();
    sorted.sort_by_key(|e| e.start);

    let chars: Vec<char> = text.chars().collect();
    let char_len = chars.len();
    let mut result = String::new();
    let mut last_end = 0;

    for e in sorted {
        if e.start >= char_len || e.end > char_len || e.start >= e.end {
            continue;
        }
        if e.start < last_end {
            continue;
        }

        if e.start > last_end {
            let before: String = chars[last_end..e.start].iter().collect();
            result.push_str(&before);
        }

        let col = type_color(e.entity_type.as_label());
        let entity_text: String = chars[e.start..e.end].iter().collect();
        result.push_str(&color(
            col,
            &format!("[{}: {}]", e.entity_type.as_label(), entity_text),
        ));
        last_end = e.end;
    }

    if last_end < char_len {
        let after: String = chars[last_end..].iter().collect();
        result.push_str(&after);
    }

    println!();
    for line in result.lines() {
        println!("  {}", line);
    }
}

fn print_annotated_signals(text: &str, signals: &[Signal<Location>]) {
    let mut sorted: Vec<&Signal<Location>> = signals.iter().collect();
    sorted.sort_by_key(|s| s.text_offsets().map(|(start, _)| start).unwrap_or(0));

    let chars: Vec<char> = text.chars().collect();
    let char_len = chars.len();
    let mut result = String::new();
    let mut last_end = 0;

    for s in sorted {
        let (start, end) = match s.text_offsets() {
            Some((start, end)) => (start, end),
            None => continue,
        };

        if start >= char_len || end > char_len || start >= end {
            continue;
        }
        if start < last_end {
            continue;
        }

        if start > last_end {
            let before: String = chars[last_end..start].iter().collect();
            result.push_str(&before);
        }

        let col = type_color(s.label());
        let entity_text: String = chars[start..end].iter().collect();
        result.push_str(&color(col, &format!("[{}: {}]", s.label(), entity_text)));
        last_end = end;
    }

    if last_end < char_len {
        let after: String = chars[last_end..].iter().collect();
        result.push_str(&after);
    }

    println!();
    for line in result.lines() {
        println!("  {}", line);
    }
}

fn print_matches(cmp: &EvalComparison, _verbose: bool) {
    for m in &cmp.matches {
        match m {
            EvalMatch::Correct { gold_id, .. } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                println!(
                    "  {} {}: [{}] \"{}\"",
                    color("32", "+"),
                    color("32", "correct"),
                    g.map(|s| s.label.as_str()).unwrap_or("?"),
                    g.map(|s| s.surface()).unwrap_or("?")
                );
            }
            EvalMatch::TypeMismatch {
                gold_id,
                gold_label,
                pred_label,
                ..
            } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                println!(
                    "  {} {}: \"{}\" ({} -> {})",
                    color("33", "!"),
                    color("33", "type mismatch"),
                    g.map(|s| s.surface()).unwrap_or("?"),
                    gold_label,
                    pred_label
                );
            }
            EvalMatch::BoundaryError {
                gold_id,
                pred_id,
                iou,
            } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                let p = cmp.predicted.iter().find(|s| s.id == *pred_id);
                println!(
                    "  {} {}: gold=\"{}\" pred=\"{}\" (IoU={:.2})",
                    color("33", "!"),
                    color("33", "boundary"),
                    g.map(|s| s.surface()).unwrap_or("?"),
                    p.map(|s| s.surface()).unwrap_or("?"),
                    iou
                );
            }
            EvalMatch::Spurious { pred_id } => {
                let p = cmp.predicted.iter().find(|s| s.id == *pred_id);
                println!(
                    "  {} {}: [{}] \"{}\"",
                    color("31", "x"),
                    color("31", "false positive"),
                    p.map(|s| s.label.as_str()).unwrap_or("?"),
                    p.map(|s| s.surface()).unwrap_or("?")
                );
            }
            EvalMatch::Missed { gold_id } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                println!(
                    "  {} {}: [{}] \"{}\"",
                    color("31", "x"),
                    color("31", "false negative"),
                    g.map(|s| s.label.as_str()).unwrap_or("?"),
                    g.map(|s| s.surface()).unwrap_or("?")
                );
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gold_spec_simple() {
        let spec =
            parse_gold_spec("Marie Curie:PER:0:11").expect("Test gold spec should parse correctly");
        assert_eq!(spec.text, "Marie Curie");
        assert_eq!(spec.label, "PER");
        assert_eq!(spec.start, 0);
        assert_eq!(spec.end, 11);
    }

    #[test]
    fn test_parse_gold_spec_with_colon_in_text() {
        // URL containing colons
        let spec = parse_gold_spec("https://example.com:URL:0:19")
            .expect("Test gold spec should parse correctly");
        assert_eq!(spec.text, "https://example.com");
        assert_eq!(spec.label, "URL");
        assert_eq!(spec.start, 0);
        assert_eq!(spec.end, 19);
    }

    #[test]
    fn test_parse_gold_spec_invalid() {
        assert!(parse_gold_spec("invalid").is_none());
        assert!(parse_gold_spec("text:label").is_none());
        assert!(parse_gold_spec("text:label:notanumber:10").is_none());
    }

    #[test]
    fn test_is_negated() {
        assert!(is_negated("He is not a doctor", 10));
        assert!(is_negated("Never trust John", 12));
        assert!(!is_negated("Trust John", 6));
    }

    #[test]
    fn test_detect_quantifier() {
        assert_eq!(
            detect_quantifier("every employee", 6),
            Some(Quantifier::Universal)
        );
        assert_eq!(
            detect_quantifier("some people", 5),
            Some(Quantifier::Existential)
        );
        assert_eq!(
            detect_quantifier("the manager", 4),
            Some(Quantifier::Definite)
        );
        assert_eq!(detect_quantifier("John Smith", 0), None);
    }

    #[test]
    fn test_model_backend_names() {
        assert_eq!(ModelBackend::Pattern.name(), "pattern");
        assert_eq!(ModelBackend::Heuristic.name(), "heuristic");
        assert_eq!(ModelBackend::Stacked.name(), "stacked");
    }

    #[test]
    fn test_confidence_bar_normal() {
        // Normal cases
        let bar = confidence_bar(0.5);
        assert!(bar.contains("50%"));

        let bar = confidence_bar(1.0);
        assert!(bar.contains("100%"));

        let bar = confidence_bar(0.0);
        assert!(bar.contains("0%"));
    }

    #[test]
    fn test_confidence_bar_clamping() {
        // Edge case: confidence slightly over 1.0 should not panic
        let bar = confidence_bar(1.01);
        assert!(bar.contains("101%")); // Display shows actual value
                                       // But the bar itself should be clamped to 10 filled chars (not panic)

        // Edge case: confidence at exactly 1.0
        let bar = confidence_bar(1.0);
        assert!(bar.contains("100%"));
    }

    #[test]
    fn test_is_negated_unicode() {
        // Test with Unicode text (character offsets, not byte offsets)
        // "café" has 4 chars but 5 bytes (é is 2 bytes in UTF-8)
        assert!(!is_negated("café John", 5)); // "John" starts at char 5
        assert!(is_negated("not café John", 9)); // "not" is in the prefix
    }

    #[test]
    fn test_detect_quantifier_unicode() {
        // Test with Unicode text
        // "every café employee" - "employee" starts at char index 11
        assert_eq!(
            detect_quantifier("every café employee", 11),
            None // "café" is not a quantifier
        );
        // "every employee" still works
        assert_eq!(
            detect_quantifier("every employee", 6),
            Some(Quantifier::Universal)
        );
    }

    #[test]
    fn test_normalize_entity_name() {
        assert_eq!(normalize_entity_name("  John Smith  "), "john smith");
        assert_eq!(normalize_entity_name("MARIE CURIE"), "marie curie");
        assert_eq!(normalize_entity_name("Test"), "test");
    }

    #[test]
    fn test_is_likely_male() {
        assert!(is_likely_male("John Smith"));
        assert!(is_likely_male("Barack Obama"));
        assert!(!is_likely_male("Marie Curie"));
        assert!(!is_likely_male("Unknown Person"));
    }

    #[test]
    fn test_is_likely_female() {
        assert!(is_likely_female("Marie Curie"));
        assert!(is_likely_female("Hillary Clinton"));
        assert!(!is_likely_female("John Smith"));
        assert!(!is_likely_female("Unknown Person"));
    }

    #[test]
    fn test_type_color() {
        assert_eq!(type_color("PER"), "1;34");
        assert_eq!(type_color("person"), "1;34");
        assert_eq!(type_color("ORG"), "1;32");
        assert_eq!(type_color("LOC"), "1;33");
        assert_eq!(type_color("UNKNOWN"), "1;37");
    }

    #[test]
    fn test_metric_colored() {
        // High score (>= 90)
        let result = metric_colored(95.0);
        assert!(result.contains("95.0"));

        // Medium score (>= 70)
        let result = metric_colored(75.0);
        assert!(result.contains("75.0"));

        // Low score (< 50)
        let result = metric_colored(30.0);
        assert!(result.contains("30.0"));
    }

    #[test]
    fn test_color_function() {
        // When not in a terminal, color() should return plain text
        // This test verifies the function doesn't panic
        let result = color("32", "test");
        assert!(result.contains("test"));
    }
}
