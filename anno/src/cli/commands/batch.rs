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

    /// Number of parallel workers
    #[arg(short, long, default_value = "1")]
    pub parallel: usize,

    /// Show progress bar
    #[arg(long)]
    pub progress: bool,

    /// Enable caching
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

pub fn run(args: BatchArgs) -> Result<(), String> {
    use super::super::parser::OutputFormat;
    use super::pipeline::PipelineArgs;
    use std::io::{self, BufRead};

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

    // Convert BatchArgs to PipelineArgs
    let output_clone = args.output.clone();
    let pipeline_args = PipelineArgs {
        text: if args.stdin {
            vec![]
        } else {
            texts.iter().map(|(_, t)| t.clone()).collect()
        },
        files: vec![],
        dir: args.dir,
        model: args.model,
        coref: args.coref,
        link_kb: args.link_kb,
        cross_doc: false, // Batch doesn't support cross-doc yet
        threshold: 0.6,
        format: args.format,
        output: output_clone,
        progress: args.progress,
        quiet: args.quiet,
    };

    // For stdin, we need to manually process since pipeline expects files/dir
    if args.stdin {
        use super::super::utils::{link_tracks_to_kb, resolve_coreference};
        use anno_core::{GroundedDocument, Location, Signal};

        let model = args.model.create_model()?;
        let mut documents: Vec<GroundedDocument> = Vec::new();

        #[cfg(all(feature = "cli", feature = "eval"))]
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

        #[cfg(not(all(feature = "cli", feature = "eval")))]
        let pb: Option<()> = None;

        for (doc_id, text) in &texts {
            #[cfg(all(feature = "cli", feature = "eval"))]
            if let Some(ref pb) = pb {
                pb.set_message(format!("Processing {}", doc_id));
            }

            let entities = model
                .extract_entities(text, None)
                .map_err(|e| format!("Extraction failed for {}: {}", doc_id, e))?;

            let mut doc = GroundedDocument::new(doc_id, text);
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

            if args.coref {
                resolve_coreference(&mut doc, text, &signal_ids);
            }

            if args.link_kb {
                link_tracks_to_kb(&mut doc);
            }

            documents.push(doc);

            #[cfg(all(feature = "cli", feature = "eval"))]
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
                    print_signals(doc, &doc.text, false);
                }
            }
        }

        Ok(())
    } else {
        // Delegate to pipeline implementation for dir-based processing
        super::pipeline::run(pipeline_args)
    }
}
