//! Watch command - Monitor directory for new/changed files (incremental processing)
//!
//! For development and live systems that need to react to incoming documents.

use clap::Parser;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::super::output::color;
use super::super::parser::{ModelBackend, OutputFormat};

/// Watch a directory and process new/changed files
#[derive(Parser, Debug)]
pub struct WatchArgs {
    /// Directory to watch
    #[arg(short, long, value_name = "DIR")]
    pub directory: PathBuf,

    /// Output directory for processed files
    #[arg(short, long, value_name = "DIR")]
    pub output: Option<PathBuf>,

    /// File extensions to watch (comma-separated)
    #[arg(long, default_value = "txt,md,pdf", value_delimiter = ',')]
    pub extensions: Vec<String>,

    /// Model backend to use
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Output format for results
    #[arg(long, default_value = "jsonl")]
    pub format: OutputFormat,

    /// Poll interval in seconds
    #[arg(long, default_value = "2")]
    pub interval: u64,

    /// Process existing files on startup
    #[arg(long)]
    pub initial: bool,

    /// Quiet mode - only output results, no status messages
    #[arg(short, long)]
    pub quiet: bool,

    /// Maximum number of files to process (0 = unlimited)
    #[arg(long, default_value = "0")]
    pub max_files: usize,
}

/// File state for change detection
#[derive(Debug, Clone)]
struct FileState {
    modified: SystemTime,
    size: u64,
    processed: bool,
}

/// Run the watch command.
pub fn run(args: WatchArgs) -> Result<(), String> {
    // Validate directory
    if !args.directory.exists() {
        return Err(format!("Directory not found: {:?}", args.directory));
    }
    if !args.directory.is_dir() {
        return Err(format!("Not a directory: {:?}", args.directory));
    }

    // Create output directory if specified
    if let Some(ref out_dir) = args.output {
        if !out_dir.exists() {
            fs::create_dir_all(out_dir)
                .map_err(|e| format!("Failed to create output directory: {}", e))?;
        }
    }

    // Create model
    let model = args.model.create_model()?;

    // Track file states
    let mut file_states: HashMap<PathBuf, FileState> = HashMap::new();
    let mut files_processed = 0;

    if !args.quiet {
        eprintln!(
            "{} Watching {:?} for {} files (poll: {}s)",
            color("32", "[watch]"),
            args.directory,
            args.extensions.join(","),
            args.interval
        );
    }

    // Initial scan
    let initial_files = scan_directory(&args.directory, &args.extensions)?;
    for path in &initial_files {
        if let Ok(metadata) = fs::metadata(path) {
            let state = FileState {
                modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                size: metadata.len(),
                processed: !args.initial, // Mark as processed unless --initial
            };
            file_states.insert(path.clone(), state);
        }
    }

    if args.initial && !args.quiet {
        eprintln!(
            "{} Processing {} existing files",
            color("33", "[init]"),
            initial_files.len()
        );
    }

    // Main watch loop
    loop {
        let current_files = scan_directory(&args.directory, &args.extensions)?;

        for path in &current_files {
            let metadata = match fs::metadata(path) {
                Ok(m) => m,
                Err(_) => continue,
            };

            let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            let size = metadata.len();

            let should_process = match file_states.get(path) {
                Some(state) => {
                    // File changed
                    !state.processed || state.modified != modified || state.size != size
                }
                None => {
                    // New file
                    true
                }
            };

            if should_process {
                // Check max files limit
                if args.max_files > 0 && files_processed >= args.max_files {
                    if !args.quiet {
                        eprintln!(
                            "{} Reached max files limit ({}), stopping",
                            color("33", "[limit]"),
                            args.max_files
                        );
                    }
                    return Ok(());
                }

                // Process the file
                match process_file(path, &*model, &args) {
                    Ok(entity_count) => {
                        files_processed += 1;
                        let timestamp = chrono::Local::now().format("%H:%M:%S");
                        if !args.quiet {
                            let action = if file_states.contains_key(path) {
                                "changed"
                            } else {
                                "new"
                            };
                            eprintln!(
                                "[{}] {}: {:?} → {} entities",
                                timestamp,
                                color("32", action),
                                path.file_name().unwrap_or_default(),
                                entity_count
                            );
                        }

                        // Update state
                        file_states.insert(
                            path.clone(),
                            FileState {
                                modified,
                                size,
                                processed: true,
                            },
                        );
                    }
                    Err(e) => {
                        if !args.quiet {
                            eprintln!(
                                "{} {:?}: {}",
                                color("31", "[error]"),
                                path.file_name().unwrap_or_default(),
                                e
                            );
                        }
                    }
                }
            }
        }

        // Check for deleted files
        let deleted: Vec<PathBuf> = file_states
            .keys()
            .filter(|p| !current_files.contains(p))
            .cloned()
            .collect();

        for path in deleted {
            if !args.quiet {
                let timestamp = chrono::Local::now().format("%H:%M:%S");
                eprintln!(
                    "[{}] {}: {:?} removed from index",
                    timestamp,
                    color("33", "deleted"),
                    path.file_name().unwrap_or_default()
                );
            }
            file_states.remove(&path);
        }

        // Sleep before next poll
        std::thread::sleep(Duration::from_secs(args.interval));
    }
}

/// Scan directory for matching files
fn scan_directory(dir: &PathBuf, extensions: &[String]) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();

    let entries = fs::read_dir(dir).map_err(|e| format!("Failed to read directory: {}", e))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if extensions.iter().any(|e| *e == *ext.to_string_lossy()) {
                    files.push(path);
                }
            }
        }
    }

    Ok(files)
}

/// Process a single file
fn process_file(path: &Path, model: &dyn anno::Model, args: &WatchArgs) -> Result<usize, String> {
    // Read file content
    let content = fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;

    // Extract entities
    let entities = model
        .extract_entities(&content, None)
        .map_err(|e| format!("Extraction failed: {}", e))?;

    let entity_count = entities.len();

    // Output results
    if let Some(ref out_dir) = args.output {
        let out_filename = path
            .file_name()
            .map(|n| format!("{}.json", n.to_string_lossy()))
            .unwrap_or_else(|| "output.json".into());
        let out_path = out_dir.join(out_filename);

        let output = serde_json::json!({
            "source": path.to_string_lossy(),
            "entity_count": entity_count,
            "entities": entities.iter().map(|e| {
                serde_json::json!({
                    "text": e.text,
                    "type": e.entity_type.as_label(),
                    "start": e.start(),
                    "end": e.end(),
                    "confidence": e.confidence,
                })
            }).collect::<Vec<_>>()
        });

        fs::write(
            &out_path,
            serde_json::to_string_pretty(&output).unwrap_or_default(),
        )
        .map_err(|e| format!("Failed to write output: {}", e))?;
    } else {
        // Print to stdout in JSONL format
        let output = serde_json::json!({
            "source": path.to_string_lossy(),
            "entity_count": entity_count,
            "entities": entities.iter().map(|e| {
                serde_json::json!({
                    "text": e.text,
                    "type": e.entity_type.as_label(),
                    "start": e.start(),
                    "end": e.end(),
                    "confidence": e.confidence,
                })
            }).collect::<Vec<_>>()
        });
        println!("{}", output);
    }

    Ok(entity_count)
}
