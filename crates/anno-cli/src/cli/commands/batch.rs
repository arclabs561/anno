//! Batch command - Batch processing

use super::super::parser::{ModelBackend, OutputFormat};
use clap::Parser;

/// Batch processing
#[derive(Parser, Debug)]
pub struct BatchArgs {
    /// Process directory of files
    #[arg(short, long, value_name = "DIR")]
    pub dir: Option<String>,

    /// Read from stdin (JSONL format)
    #[arg(long)]
    pub stdin: bool,

    /// Model backend to use
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Run coreference resolution
    #[arg(long)]
    pub coref: bool,

    /// Link tracks to KB identities
    #[arg(long)]
    pub link_kb: bool,

    /// Number of parallel workers (currently ignored)
    #[arg(short, long, default_value = "1")]
    pub parallel: usize,

    /// Show progress bar
    #[arg(long)]
    pub progress: bool,

    /// Enable caching (currently ignored)
    #[arg(long)]
    pub cache: bool,

    /// Output directory for results
    #[arg(short, long, value_name = "DIR")]
    pub output: Option<String>,

    /// Output format
    #[arg(long, default_value = "grounded")]
    pub format: OutputFormat,

    /// Suppress status messages
    #[arg(short, long)]
    pub quiet: bool,
}

/// Execute the batch processing command.
pub fn run(args: BatchArgs) -> Result<(), String> {
    use super::super::parser::OutputFormat;
    use super::pipeline::PipelineArgs;
    use std::io::{self, BufRead};
    use std::path::{Path, PathBuf};

    // Validate input source
    if args.dir.is_none() && !args.stdin {
        return Err("Either --dir <DIR> or --stdin must be specified".to_string());
    }

    if args.dir.is_some() && args.stdin {
        return Err("Cannot use both --dir and --stdin. Choose one.".to_string());
    }

    // Handle stdin (JSONL format)
    let mut texts: Vec<(String, String)> = Vec::new();

    if args.stdin {
        if !args.quiet {
            eprintln!("Reading JSONL from stdin...");
        }
        let stdin = io::stdin();
        let reader = stdin.lock();
        for (line_num, line) in reader.lines().enumerate() {
            let line =
                line.map_err(|e| format!("Failed to read stdin line {}: {}", line_num + 1, e))?;
            if line.trim().is_empty() {
                continue;
            }
            // Parse JSONL: each line should be {"id": "...", "text": "..."} or just {"text": "..."}
            let json: serde_json::Value = serde_json::from_str(&line).map_err(|e| {
                format!("Failed to parse stdin line {} as JSON: {}", line_num + 1, e)
            })?;
            let doc_id = json
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("stdin:{}", line_num + 1));
            let text = json
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("Missing 'text' field in stdin line {}", line_num + 1))?;
            texts.push((doc_id, text.to_string()));
        }
    }

    // Note: parallel and cache flags are placeholders for future optimization
    // They don't affect behavior yet but are accepted for API compatibility
    if args.parallel > 1 && !args.quiet {
        eprintln!("Note: Parallel processing (--parallel) is not yet implemented");
    }
    if args.cache && !args.quiet {
        eprintln!("Note: Caching (--cache) is not yet implemented");
    }

    // For stdin, we need to manually process since pipeline expects files/dir
    if args.stdin {
        use super::super::utils::{link_tracks_to_kb, resolve_coreference};
        use anno_core::{GroundedDocument, Location, Signal, SignalId};

        let model = args.model.create_model()?;
        let mut documents: Vec<GroundedDocument> = Vec::new();

        let pb = if args.progress && !args.quiet {
            use indicatif::{ProgressBar, ProgressStyle};
            let pb = ProgressBar::new(texts.len() as u64);
            let style = ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .expect("Progress bar template should be valid");
            pb.set_style(style.progress_chars("#>-"));
            Some(pb)
        } else {
            None
        };

        for (doc_id, text) in &texts {
            if let Some(ref pb) = pb {
                pb.set_message(format!("Processing {}", doc_id));
            }

            let entities = model
                .extract_entities(text, None)
                .map_err(|e| format!("Extraction failed for {}: {}", doc_id, e))?;

            let mut doc = GroundedDocument::new(doc_id, text);
            let mut signal_ids: Vec<SignalId> = Vec::new();

            for e in &entities {
                let signal = Signal::new(
                    SignalId::ZERO,
                    Location::text(e.start, e.end),
                    &e.text,
                    e.entity_type.as_label(),
                    e.confidence as f32,
                );
                let id = doc.add_signal(signal);
                signal_ids.push(id);
            }

            if args.coref {
                resolve_coreference(&mut doc, text, &signal_ids);
            }

            if args.link_kb {
                link_tracks_to_kb(&mut doc);
            }

            documents.push(doc);

            if let Some(ref pb) = pb {
                pb.inc(1);
            }
        }

        // Output results
        use super::super::output::{color, print_signals};
        for doc in &documents {
            if !args.quiet {
                println!("\n{}", color("1;36", &format!("Document: {}", doc.id)));
            }
            match args.format {
                OutputFormat::Json | OutputFormat::Grounded => {
                    let output = serde_json::to_string_pretty(doc)
                        .map_err(|e| format!("Failed to serialize document: {}", e))?;
                    if let Some(ref output_path) = args.output {
                        std::fs::write(output_path, output)
                            .map_err(|e| format!("Failed to write output: {}", e))?;
                    } else {
                        println!("{}", output);
                    }
                }
                _ => {
                    // Human-readable output for all other formats
                    print_signals(doc, &doc.text, 0);
                }
            }
        }

        Ok(())
    } else {
        // Directory mode: batch differs from pipeline in that `--output` is a directory.
        // We materialize per-document outputs under that directory.
        let dir = args
            .dir
            .as_ref()
            .expect("validated: args.dir is Some when not stdin");

        let mut pipeline_args = PipelineArgs {
            text: vec![],
            files: vec![],
            dir: Some(dir.clone()),
            model: args.model,
            coref: args.coref,
            link_kb: args.link_kb,
            cross_doc: false, // Batch doesn't support cross-doc yet
            threshold: 0.6,
            format: args.format,
            output: None, // handled below
            progress: args.progress,
            quiet: args.quiet,
        };

        // If no output directory was requested, just delegate to pipeline (stdout).
        let Some(out_dir) = args.output.as_ref() else {
            return super::pipeline::run(pipeline_args);
        };

        let out_dir: PathBuf = PathBuf::from(out_dir);
        if out_dir.exists() && !out_dir.is_dir() {
            return Err(format!(
                "--output must be a directory for `anno batch`, but '{}' is not a directory",
                out_dir.display()
            ));
        }
        std::fs::create_dir_all(&out_dir)
            .map_err(|e| format!("Failed to create output dir '{}': {}", out_dir.display(), e))?;

        // Run the pipeline once (in-memory), then write one file per document.
        // This keeps batch semantics stable even if pipeline gains new transforms.
        pipeline_args.format = OutputFormat::Grounded;
        // Use pipeline's stdout path by capturing documents via the same logic here.
        // (Pipeline writes to a single file; batch writes per-doc.)
        //
        // Re-implement the minimal loop from `pipeline::run` for dir input so we can
        // materialize per-doc outputs without changing pipeline’s contract.
        use super::super::utils::{link_tracks_to_kb, resolve_coreference};
        use anno_core::{GroundedDocument, Location, Signal, SignalId};

        let model = args.model.create_model()?;
        let mut documents: Vec<GroundedDocument> = Vec::new();

        // Load directory texts (txt/md only; matches pipeline behavior).
        let dir_path = Path::new(dir);
        let entries = std::fs::read_dir(dir_path)
            .map_err(|e| format!("Failed to read directory {}: {}", dir, e))?;

        let mut inputs: Vec<(String, String)> = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let ext_ok = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e == "txt" || e == "md")
                .unwrap_or(false);
            if !ext_ok {
                continue;
            }
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
            let doc_id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("doc{}", inputs.len() + 1));
            inputs.push((doc_id, text));
        }

        if inputs.is_empty() {
            return Err(format!(
                "No input files found under '{}' (expected .txt/.md)",
                dir
            ));
        }

        for (doc_id, text) in &inputs {
            let entities = model
                .extract_entities(text, None)
                .map_err(|e| format!("Extraction failed for {}: {}", doc_id, e))?;

            let mut doc = GroundedDocument::new(doc_id, text);
            let mut signal_ids: Vec<SignalId> = Vec::new();
            for e in &entities {
                let signal = Signal::new(
                    SignalId::ZERO,
                    Location::text(e.start, e.end),
                    &e.text,
                    e.entity_type.as_label(),
                    e.confidence as f32,
                );
                let id = doc.add_signal(signal);
                signal_ids.push(id);
            }

            if args.coref {
                resolve_coreference(&mut doc, text, &signal_ids);
            }
            if args.link_kb {
                link_tracks_to_kb(&mut doc);
            }

            documents.push(doc);
        }

        // Write per-document outputs.
        match args.format {
            OutputFormat::Json | OutputFormat::Grounded => {
                for doc in &documents {
                    let out_path = out_dir.join(format!("{}.json", doc.id));
                    let payload = serde_json::to_string_pretty(doc)
                        .map_err(|e| format!("Failed to serialize {}: {}", doc.id, e))?;
                    std::fs::write(&out_path, payload)
                        .map_err(|e| format!("Failed to write {}: {}", out_path.display(), e))?;
                }
            }
            OutputFormat::Jsonl => {
                for doc in &documents {
                    let out_path = out_dir.join(format!("{}.jsonl", doc.id));
                    let payload = serde_json::to_string(doc)
                        .map_err(|e| format!("Failed to serialize {}: {}", doc.id, e))?;
                    std::fs::write(&out_path, payload + "\n")
                        .map_err(|e| format!("Failed to write {}: {}", out_path.display(), e))?;
                }
            }
            other => {
                return Err(format!(
                    "Batch output directory mode does not support format '{:?}'. Use --format grounded|json|jsonl, or omit --output to print to stdout.",
                    other
                ));
            }
        }

        if !args.quiet {
            eprintln!(
                "Wrote {} document(s) under {}",
                documents.len(),
                out_dir.display()
            );
        }

        Ok(())
    }
}
