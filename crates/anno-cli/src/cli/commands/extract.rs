//! Extract command - Level 1 (Signal): Raw entity extraction

use clap::Parser;
use std::fs;
use std::time::Instant;

use super::super::output::{color, log_info, print_annotated_signals, print_signals};
use super::super::parser::{ModelBackend, OutputFormat};
use super::super::utils::{detect_quantifier, get_input_text, is_negated};

#[cfg(feature = "eval-advanced")]
use anno::ingest::url_resolver::CompositeResolver;
use anno::ingest::DocumentPreprocessor;
use anno_core::core::graph::{GraphDocument, GraphExportFormat};
use anno_core::core::grounded::{
    GroundedDocument, Location, Modality, Signal, SignalId, SignalValidationError,
};

use crate::cli::CliError;

use xxhash_rust::xxh3::xxh3_64;

/// Extract entities from text
#[derive(Parser, Debug)]
pub struct ExtractArgs {
    /// Input text to process
    #[arg(short, long)]
    pub text: Option<String>,

    /// Read input from file
    #[arg(short, long, value_name = "PATH")]
    pub file: Option<String>,

    /// Model backend to use
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Filter to specific entity types (repeatable)
    #[arg(short, long = "label", value_name = "TYPE")]
    pub labels: Vec<String>,

    /// Filter to specific entity types (comma-separated, e.g., "PER,ORG,DATE")
    #[arg(long, value_name = "CSV")]
    pub types: Option<String>,

    /// Zero-shot entity types to extract (comma-separated).
    /// Only works with zero-shot capable models (e.g., gliner).
    /// Example: --extract-types "SCIENTIST,ELEMENT,AWARD"
    #[arg(long, value_name = "CSV")]
    pub extract_types: Option<String>,

    /// Minimum confidence threshold (0.0-1.0).
    ///
    /// Entities with confidence below this value are discarded before output.
    #[arg(long, value_name = "FLOAT")]
    pub threshold: Option<f64>,

    /// Warn if expected entity types are not present in the output (comma-separated).
    ///
    /// Example: `--expected-types PER,ORG,DATE,MONEY`
    #[arg(long, value_name = "CSV")]
    pub expected_types: Option<String>,

    /// Output format
    #[arg(long, default_value = "human")]
    pub format: OutputFormat,

    /// Include a character context window around each extracted entity (adds `context_before` / `context_after`)
    #[arg(long, value_name = "CHARS")]
    pub context_window: Option<usize>,

    /// Include the containing sentence for each extracted entity (adds `sentence`)
    #[arg(long)]
    pub include_sentence: bool,

    /// Export GroundedDocument JSON to file
    #[arg(long, value_name = "PATH")]
    pub export: Option<String>,

    /// Export to graph format (neo4j, networkx, jsonld)
    #[arg(long, value_name = "FORMAT")]
    pub export_graph: Option<String>,

    /// URL to fetch content from (requires eval-advanced feature)
    #[arg(long, value_name = "URL")]
    pub url: Option<String>,

    /// Clean and normalize text before extraction
    #[arg(long)]
    pub clean: bool,

    /// Normalize Unicode
    #[arg(long)]
    pub normalize: bool,

    /// Detect and record language
    #[arg(long)]
    pub detect_lang: bool,

    /// Export format when using --export (full, signals, minimal)
    #[arg(long, default_value = "full", value_name = "FORMAT")]
    pub export_format: String,

    /// Detect negated entities
    #[arg(long)]
    pub negation: bool,

    /// Detect quantified entities
    #[arg(long)]
    pub quantifiers: bool,

    /// Verbose output (repeat for more detail: -v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Minimal output
    #[arg(short, long)]
    pub quiet: bool,

    /// Positional text argument
    /// Positional text input (alternative to --text or --file)
    pub positional: Vec<String>,
}

/// Execute the extract command.
pub fn run(args: ExtractArgs) -> Result<(), CliError> {
    // Level 1 (Signal): Raw entity extraction from single document
    // This is the foundation for all other commands:
    // - `debug` adds Level 2 (Track) via coreference resolution
    // - `debug --link-kb` attaches demo identities (debug-only; offline)
    // - `crossdoc`/`coalesce` clusters Level 1 entities across multiple documents

    // Resolve input: URL, file, text, or stdin
    let mut raw_text = if let Some(url) = &args.url {
        #[cfg(feature = "eval-advanced")]
        {
            use anno::ingest::UrlResolver;
            let resolver = CompositeResolver::new();
            let resolved = resolver
                .resolve(url)
                .map_err(|e| CliError::from(format!("Failed to fetch URL {}: {}", url, e)))?;
            resolved.text
        }
        #[cfg(not(feature = "eval-advanced"))]
        {
            #[allow(unused_variables)]
            let _url = url;
            return Err(CliError::from("URL resolution requires 'eval-advanced' feature. Enable with: cargo build --features eval-advanced"));
        }
    } else {
        get_input_text(&args.text, args.file.as_deref(), &args.positional)
            .map_err(CliError::from)?
    };

    // Preprocess text if requested
    if args.clean || args.normalize || args.detect_lang {
        let preprocessor = DocumentPreprocessor {
            clean_whitespace: args.clean,
            normalize_unicode: args.normalize,
            detect_language: args.detect_lang,
            chunk_size: None,
        };
        let prepared = preprocessor.prepare(&raw_text);
        raw_text = prepared.text;
        if args.verbose >= 1 && !prepared.metadata.is_empty() {
            log_info(
                &format!("Preprocessing metadata: {:?}", prepared.metadata),
                args.quiet,
            );
        }
    }

    let text = raw_text;

    // Parse zero-shot extract types if provided
    let extract_types: Option<Vec<String>> = args.extract_types.as_ref().map(|csv| {
        csv.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    });

    let start = Instant::now();
    let entities = if let Some(ref custom_types) = extract_types {
        // Zero-shot extraction with custom types
        extract_with_custom_types(&args.model, &text, custom_types, args.threshold, args.quiet)?
    } else {
        // Standard extraction
        let model = args.model.create_model().map_err(CliError::from)?;
        model
            .extract_entities(&text, None)
            .map_err(|e| CliError::from(format!("Extraction failed: {}", e)))?
    };
    let elapsed = start.elapsed();

    let mut entities = entities;

    // Apply confidence threshold if requested
    if let Some(threshold) = args.threshold {
        entities.retain(|e| e.confidence >= threshold);
    }

    // Filter by labels/types if specified
    let mut type_filters: Vec<String> = Vec::new();
    type_filters.extend(args.labels.iter().cloned());
    if let Some(csv) = &args.types {
        for part in csv.split(',') {
            let t = part.trim();
            if !t.is_empty() {
                type_filters.push(t.to_string());
            }
        }
    }

    if !type_filters.is_empty() {
        let normalized: std::collections::HashSet<String> = type_filters
            .iter()
            .map(|t| anno::EntityType::from_label(t).as_label().to_string())
            .collect();

        entities.retain(|e| normalized.contains(e.entity_type.as_label()));
    }

    // Warn about missing expected types (best-effort, non-fatal)
    if let Some(csv) = &args.expected_types {
        let mut expected: Vec<String> = Vec::new();
        for part in csv.split(',') {
            let t = part.trim();
            if !t.is_empty() {
                expected.push(t.to_string());
            }
        }

        if !expected.is_empty() {
            let present: std::collections::HashSet<String> = entities
                .iter()
                .map(|e| e.entity_type.as_label().to_string())
                .collect();

            let mut missing: Vec<String> = expected
                .iter()
                .filter_map(|t| {
                    let normalized = anno::EntityType::from_label(t).as_label().to_string();
                    (!present.contains(&normalized)).then_some(normalized)
                })
                .collect();

            missing.sort();
            missing.dedup();

            if !missing.is_empty() && !args.quiet {
                eprintln!(
                    "{} Expected types not found: {}",
                    color("33", "warning:"),
                    missing.join(", ")
                );
            }
        }
    }

    // Build grounded document with validation using library method
    let mut doc = GroundedDocument::new("extract", &text);
    let mut validation_errors: Vec<SignalValidationError> = Vec::new();

    for e in &entities {
        let mut signal = Signal::new(
            SignalId::ZERO,
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

        // Use library validation method for consistent error handling
        match doc.add_signal_validated(signal) {
            Ok(_) => {
                // Signal added successfully
            }
            Err(err) => {
                validation_errors.push(err);
            }
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
            let entities_out: Vec<serde_json::Value> = doc
                .signals()
                .iter()
                .map(|s| {
                    let (start, end) = s.text_offsets().unwrap_or((0, 0));
                    let mut obj = serde_json::json!({
                        "id": compute_entity_id(&text, s.surface(), s.label(), start, end),
                        "text": s.surface(),
                        "type": s.label(),
                        "start": start,
                        "end": end,
                        "confidence": s.confidence,
                        "negated": s.negated,
                        "quantifier": s.quantifier.map(|q| format!("{:?}", q)),
                    });

                    if let Some(window) = args.context_window {
                        let (before, after) = get_context_window(&text, start, end, window);
                        obj["context_before"] = serde_json::Value::String(before);
                        obj["context_after"] = serde_json::Value::String(after);
                    }

                    if args.include_sentence {
                        let sent = get_sentence_for_span(&text, start, end);
                        obj["sentence"] = serde_json::Value::String(sent);
                    }

                    obj
                })
                .collect();

            let provenance = build_provenance(&text, args.model.name(), &entities_out, elapsed);
            let output = serde_json::json!({
                "provenance": provenance,
                "entities": entities_out,
            });

            println!(
                "{}",
                serde_json::to_string_pretty(&output).unwrap_or_default()
            );
        }
        OutputFormat::Jsonl => {
            let entities_out: Vec<serde_json::Value> = doc
                .signals()
                .iter()
                .map(|s| {
                    let (start, end) = s.text_offsets().unwrap_or((0, 0));
                    let mut obj = serde_json::json!({
                        "id": compute_entity_id(&text, s.surface(), s.label(), start, end),
                        "text": s.surface(),
                        "type": s.label(),
                        "start": start,
                        "end": end,
                        "confidence": s.confidence,
                        "negated": s.negated,
                    });

                    if let Some(window) = args.context_window {
                        let (before, after) = get_context_window(&text, start, end, window);
                        obj["context_before"] = serde_json::Value::String(before);
                        obj["context_after"] = serde_json::Value::String(after);
                    }

                    if args.include_sentence {
                        let sent = get_sentence_for_span(&text, start, end);
                        obj["sentence"] = serde_json::Value::String(sent);
                    }

                    obj
                })
                .collect();

            let provenance = build_provenance(&text, args.model.name(), &entities_out, elapsed);
            println!("{}", serde_json::json!({ "provenance": provenance }));
            for obj in entities_out {
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
            // Validate signals
            let errors = doc.validate();
            if !errors.is_empty() {
                return Err(CliError::from(format!(
                    "Signal validation failed with {} errors:\n{}",
                    errors.len(),
                    errors
                        .iter()
                        .take(5)
                        .map(|e| format!("  - {}", e))
                        .collect::<Vec<_>>()
                        .join("\n")
                )));
            }
        }
        OutputFormat::Grounded => {
            println!(
                "{}",
                serde_json::to_string_pretty(&doc).map_err(CliError::from)?
            );
        }
        OutputFormat::Html => {
            return Err(CliError::from(
                "HTML format not supported for extract command. Use 'debug --format html' instead.",
            ));
        }
        OutputFormat::Tree | OutputFormat::Summary => {
            return Err(CliError::from(
                "Tree/Summary formats are only available for cross-doc command.",
            ));
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
                // print_signals already handles statistics at level 2+ and metadata at level 3+
                // Debug: Check if verbose is being passed correctly

                print_signals(&doc, &text, args.verbose);

                // Level 3+: Additional metadata (timing, model, document ID) - only if not already shown
                if args.verbose >= 3 {
                    let stats = doc.stats();
                    let text_len = text.chars().count();
                    println!();
                    println!("  {}:", color("90", "Metadata"));
                    println!("    document: {}", doc.id);
                    println!("    model: {}", args.model.name());
                    println!("    timing: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
                    println!("    text length: {} chars", text_len);
                    if stats.signal_count > 0 {
                        println!(
                            "    signals/track: {:.1}",
                            stats.signal_count as f32 / stats.track_count.max(1) as f32
                        );
                    }
                }
            }
        }
    }

    // Export to file if requested
    if let Some(export_path) = &args.export {
        let export_data = match args.export_format.as_str() {
            "full" => serde_json::to_value(&doc).map_err(CliError::from)?,
            "signals" => {
                let signals: Vec<_> = doc.signals().to_vec();
                serde_json::json!({
                    "id": doc.id,
                    "text": doc.text,
                    "signals": signals
                })
            }
            "minimal" => {
                let signals: Vec<_> = doc
                    .signals()
                    .iter()
                    .map(|s| {
                        let (start, end) = s.text_offsets().unwrap_or((0, 0));
                        serde_json::json!({
                            "surface": s.surface(),
                            "label": s.label(),
                            "start": start,
                            "end": end,
                            "confidence": s.confidence
                        })
                    })
                    .collect();
                serde_json::json!({
                    "id": doc.id,
                    "text": doc.text,
                    "signals": signals
                })
            }
            _ => {
                return Err(CliError::from(format!(
                    "Invalid export format '{}'. Use: full, signals, or minimal",
                    args.export_format
                )));
            }
        };

        let json = serde_json::to_string_pretty(&export_data)
            .map_err(|e| CliError::from(format!("Failed to serialize export data: {}", e)))?;

        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(&export_path).parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    CliError::from(format!(
                        "Failed to create directory for export file '{}': {}",
                        export_path, e
                    ))
                })?;
            }
        }

        fs::write(export_path, json).map_err(|e| {
            CliError::from(format!(
                "Failed to write export file '{}': {}",
                export_path, e
            ))
        })?;
        if !args.quiet {
            eprintln!(
                "{} Exported {} format to {}",
                color("32", "✓"),
                args.export_format,
                export_path
            );
        }
    }

    // Export to graph format if requested
    if let Some(graph_format_str) = args.export_graph {
        let graph_format = match graph_format_str.to_lowercase().as_str() {
            "neo4j" | "cypher" => GraphExportFormat::Cypher,
            "networkx" | "nx" => GraphExportFormat::NetworkXJson,
            "jsonld" | "json-ld" => GraphExportFormat::JsonLd,
            _ => {
                return Err(CliError::from(format!(
                    "Invalid graph format '{}'. Use: neo4j, networkx, or jsonld",
                    graph_format_str
                )));
            }
        };

        let graph = GraphDocument::from_grounded_document(&doc);
        let graph_output = graph.export(graph_format);

        // Output graph to stdout (always print to stdout for graph export)
        // Note: If user wants to save to file, they can use shell redirection: --export-graph neo4j > output.cypher
        if !args.quiet {
            eprintln!(
                "{} Exported graph ({} nodes, {} edges) in {} format",
                color("32", "✓"),
                graph.node_count(),
                graph.edge_count(),
                graph_format_str
            );
        }
        println!("{}", graph_output);
    }

    Ok(())
}

/// Extract entities using zero-shot models with custom entity types.
fn extract_with_custom_types(
    backend: &super::super::parser::ModelBackend,
    text: &str,
    custom_types: &[String],
    threshold: Option<f64>,
    quiet: bool,
) -> Result<Vec<anno::Entity>, CliError> {
    use super::super::output::color;

    let threshold = threshold.unwrap_or(0.5) as f32;
    let type_refs: Vec<&str> = custom_types.iter().map(|s| s.as_str()).collect();

    // Try to use zero-shot capable backends
    #[cfg(feature = "onnx")]
    {
        use super::super::parser::ModelBackend;
        use anno::backends::inference::ZeroShotNER;

        match backend {
            ModelBackend::Gliner => {
                let model = anno::GLiNEROnnx::new(anno::DEFAULT_GLINER_MODEL)
                    .map_err(|e| CliError::from(format!("Failed to create GLiNER model: {}", e)))?;
                return model
                    .extract_with_types(text, &type_refs, threshold)
                    .map_err(|e| CliError::from(format!("Zero-shot extraction failed: {}", e)));
            }
            ModelBackend::Gliner2 => {
                let model = anno::backends::gliner2::GLiNER2Onnx::from_pretrained(
                    anno::DEFAULT_GLINER2_MODEL,
                )
                .map_err(|e| CliError::from(format!("Failed to create GLiNER2 model: {}", e)))?;
                return model
                    .extract_with_types(text, &type_refs, threshold)
                    .map_err(|e| CliError::from(format!("Zero-shot extraction failed: {}", e)));
            }
            _ => {}
        }
    }

    // Fallback: warn and use standard extraction
    if !quiet {
        eprintln!(
            "{} --extract-types requires --model gliner or --model gliner2. Ignoring custom types.",
            color("33", "warning:")
        );
    }

    let model = backend.create_model().map_err(CliError::from)?;
    model
        .extract_entities(text, None)
        .map_err(|e| CliError::from(format!("Extraction failed: {}", e)))
}

fn get_context_window(text: &str, start: usize, end: usize, window: usize) -> (String, String) {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    if start > len || end > len || start > end {
        return (String::new(), String::new());
    }

    let ctx_start = start.saturating_sub(window);
    let ctx_end = (end + window).min(len);

    let before: String = chars[ctx_start..start].iter().collect();
    let after: String = chars[end..ctx_end].iter().collect();
    (before, after)
}

fn get_sentence_for_span(text: &str, start: usize, end: usize) -> String {
    // Heuristic: find the nearest sentence boundary punctuation around the span.
    // Unicode-safe: operate in character space.
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    if start > len || end > len || start > end {
        return String::new();
    }
    if len == 0 {
        return String::new();
    }

    let is_boundary = |c: char| matches!(c, '.' | '!' | '?' | '。' | '！' | '？' | '\n');

    // Search backward for a boundary; start the sentence after it.
    let mut sent_start = 0usize;
    for i in (0..start.min(len)).rev() {
        if is_boundary(chars[i]) {
            sent_start = (i + 1).min(len);
            break;
        }
    }

    // Search forward for a boundary; end the sentence at it (inclusive).
    let mut sent_end = len;
    for (i, &ch) in chars.iter().enumerate().skip(end.min(len)) {
        if is_boundary(ch) {
            sent_end = (i + 1).min(len);
            break;
        }
    }

    chars[sent_start..sent_end]
        .iter()
        .collect::<String>()
        .trim()
        .to_string()
}

fn compute_entity_id(text: &str, surface: &str, label: &str, start: usize, end: usize) -> String {
    // Content-addressed entity id: stable across runs, sensitive to (text + span + label).
    // Note: include full text so the same span indices in different docs can't collide.
    let mut data = Vec::new();
    data.extend_from_slice(text.as_bytes());
    data.extend_from_slice(surface.as_bytes());
    data.extend_from_slice(label.as_bytes());
    data.extend_from_slice(&start.to_le_bytes());
    data.extend_from_slice(&end.to_le_bytes());
    format!("xxh3:{:016x}", xxh3_64(&data))
}

fn build_provenance(
    text: &str,
    model: &str,
    entities: &[serde_json::Value],
    elapsed: std::time::Duration,
) -> serde_json::Value {
    // Result hash mirrors eval/property_tests.rs: sorted, order-independent, deterministic.
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct HashEnt {
        text: String,
        entity_type: String,
        start: usize,
        end: usize,
    }

    let mut ents: Vec<HashEnt> = entities
        .iter()
        .filter_map(|e| {
            Some(HashEnt {
                text: e.get("text")?.as_str()?.to_string(),
                entity_type: e.get("type")?.as_str()?.to_string(),
                start: e.get("start")?.as_u64()? as usize,
                end: e.get("end")?.as_u64()? as usize,
            })
        })
        .collect();

    ents.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then_with(|| a.end.cmp(&b.end))
            .then_with(|| a.entity_type.cmp(&b.entity_type))
            .then_with(|| a.text.cmp(&b.text))
    });

    let mut data = Vec::new();
    data.extend_from_slice(text.as_bytes());
    for e in &ents {
        data.extend_from_slice(e.text.as_bytes());
        data.extend_from_slice(e.entity_type.as_bytes());
        data.extend_from_slice(&e.start.to_le_bytes());
        data.extend_from_slice(&e.end.to_le_bytes());
    }
    let result_hash = format!("xxh3:{:016x}", xxh3_64(&data));

    let confs: Vec<f64> = entities
        .iter()
        .filter_map(|e| e.get("confidence").and_then(|v| v.as_f64()))
        .collect();
    let confidence_stats = compute_confidence_stats(&confs);

    serde_json::json!({
        "model": model,
        "text_chars": text.chars().count(),
        "entity_count": ents.len(),
        "elapsed_ms": (elapsed.as_secs_f64() * 1000.0),
        "result_hash": result_hash,
        "confidence_stats": confidence_stats,
    })
}

fn compute_confidence_stats(confs: &[f64]) -> serde_json::Value {
    if confs.is_empty() {
        return serde_json::json!({
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std_dev": 0.0,
            "min": 0.0,
            "max": 0.0,
        });
    }

    let mut sorted = confs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let count = sorted.len();
    let sum: f64 = sorted.iter().sum();
    let mean = sum / count as f64;
    let median = if count % 2 == 1 {
        sorted[count / 2]
    } else {
        (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
    };
    let var = sorted
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / count as f64;
    let std_dev = var.sqrt();

    serde_json::json!({
        "count": count,
        "mean": mean,
        "median": median,
        "std_dev": std_dev,
        "min": sorted.first().copied().unwrap_or(0.0),
        "max": sorted.last().copied().unwrap_or(0.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_context_window_basic() {
        let text = "Hello world, this is a test.";
        // "world" is at indices 6..11
        // window=5 means: before = chars[1..6] = "ello ", after = chars[11..16] = ", thi"
        let (before, after) = get_context_window(text, 6, 11, 5);
        assert_eq!(before, "ello ");
        assert_eq!(after, ", thi");
    }

    #[test]
    fn test_get_context_window_at_start() {
        let text = "Hello world";
        let (before, after) = get_context_window(text, 0, 5, 10);
        assert_eq!(before, "");
        assert_eq!(after, " world");
    }

    #[test]
    fn test_get_context_window_at_end() {
        let text = "Hello world";
        let (before, after) = get_context_window(text, 6, 11, 10);
        assert_eq!(before, "Hello ");
        assert_eq!(after, "");
    }

    #[test]
    fn test_get_sentence_for_span() {
        let text = "First sentence. Second sentence. Third.";
        let sentence = get_sentence_for_span(text, 16, 22);
        assert_eq!(sentence, "Second sentence.");
    }

    #[test]
    fn test_get_sentence_for_span_at_boundary() {
        let text = "Only one sentence here";
        let sentence = get_sentence_for_span(text, 5, 8);
        assert_eq!(sentence, "Only one sentence here");
    }

    #[test]
    fn test_compute_entity_id_deterministic() {
        let text = "Test text";
        let id1 = compute_entity_id(text, "Test", "PER", 0, 4);
        let id2 = compute_entity_id(text, "Test", "PER", 0, 4);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_compute_entity_id_different_for_different_spans() {
        let text = "Test text";
        let id1 = compute_entity_id(text, "Test", "PER", 0, 4);
        let id2 = compute_entity_id(text, "text", "PER", 5, 9);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_confidence_stats_empty() {
        let stats = compute_confidence_stats(&[]);
        assert_eq!(stats["count"], 0);
        assert_eq!(stats["mean"], 0.0);
    }

    #[test]
    fn test_confidence_stats_single() {
        let stats = compute_confidence_stats(&[0.8]);
        assert_eq!(stats["count"], 1);
        assert_eq!(stats["mean"], 0.8);
        assert_eq!(stats["median"], 0.8);
    }

    #[test]
    fn test_confidence_stats_multiple() {
        let stats = compute_confidence_stats(&[0.5, 0.7, 0.9]);
        assert_eq!(stats["count"], 3);
        let mean = stats["mean"].as_f64().unwrap();
        assert!((mean - 0.7).abs() < 0.001);
        assert_eq!(stats["median"], 0.7);
    }

    #[test]
    fn test_extract_args_parse_extract_types() {
        // Verify the extract_types field parses correctly from CSV
        let csv = "DRUG,SYMPTOM,CONDITION";
        let types: Vec<String> = csv
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        assert_eq!(types, vec!["DRUG", "SYMPTOM", "CONDITION"]);
    }

    #[test]
    fn test_extract_args_parse_extract_types_with_spaces() {
        let csv = " DRUG , SYMPTOM , CONDITION ";
        let types: Vec<String> = csv
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        assert_eq!(types, vec!["DRUG", "SYMPTOM", "CONDITION"]);
    }

    #[test]
    fn test_extract_args_parse_extract_types_empty_parts() {
        let csv = "DRUG,,SYMPTOM,";
        let types: Vec<String> = csv
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        assert_eq!(types, vec!["DRUG", "SYMPTOM"]);
    }
}
