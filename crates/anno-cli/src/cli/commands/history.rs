//! History command - Query evaluation history

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[cfg(feature = "eval")]
use anno_eval::eval::history::{EvalHistory, HistoryStats};

/// Query evaluation history
#[derive(Parser, Debug)]
#[command(about = "Query and analyze evaluation history")]
pub struct HistoryArgs {
    /// Path to evaluation history JSONL file
    #[arg(long, env = "ANNO_EVAL_HISTORY")]
    pub history_file: Option<PathBuf>,

    /// Subcommand
    #[command(subcommand)]
    pub action: HistoryAction,
}

/// History query actions
#[derive(Subcommand, Debug)]
pub enum HistoryAction {
    /// Show statistics about evaluation history
    Stats,
    /// Query recent results for a backend
    Recent {
        /// Backend name
        backend: String,
        /// Number of results to return
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    /// Query best results (by F1 score)
    Best {
        /// Backend name
        backend: String,
        /// Dataset name (optional)
        #[arg(short, long)]
        dataset: Option<String>,
        /// Number of results to return
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    /// Query results by date range
    Range {
        /// Start date (ISO 8601 format, e.g., 2024-01-01T00:00:00Z)
        start: String,
        /// End date (ISO 8601 format)
        end: String,
        /// Backend name (optional)
        #[arg(short, long)]
        backend: Option<String>,
    },
    /// Compare two backends
    Compare {
        /// First backend
        backend1: String,
        /// Second backend
        backend2: String,
        /// Dataset name (optional)
        #[arg(short, long)]
        dataset: Option<String>,
    },
    /// List all backends in history
    Backends,
    /// List all datasets in history
    Datasets,
    /// Rebuild SQLite index from JSONL
    Rebuild,
}

/// Execute the history command.
#[cfg(feature = "eval")]
pub fn run(args: HistoryArgs) -> Result<(), String> {
    use dirs::cache_dir;

    // Determine history file path
    let history_path = args.history_file.unwrap_or_else(|| {
        cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("anno")
            .join("eval-results.jsonl")
    });

    let history = EvalHistory::new(&history_path).map_err(|e| {
        format!(
            "Failed to open history file {}: {}",
            history_path.display(),
            e
        )
    })?;

    match args.action {
        HistoryAction::Stats => {
            let stats = history
                .stats()
                .map_err(|e| format!("Failed to get stats: {}", e))?;
            print_stats(&stats);
        }
        HistoryAction::Backends => {
            let backends = history
                .backends()
                .map_err(|e| format!("Failed to get backends: {}", e))?;
            println!("=== Backends in History ===\n");
            if backends.is_empty() {
                println!("No backends found.");
            } else {
                for backend in backends {
                    println!("  {}", backend);
                }
            }
        }
        HistoryAction::Datasets => {
            let datasets = history
                .datasets()
                .map_err(|e| format!("Failed to get datasets: {}", e))?;
            println!("=== Datasets in History ===\n");
            if datasets.is_empty() {
                println!("No datasets found.");
            } else {
                for dataset in datasets {
                    println!("  {}", dataset);
                }
            }
        }
        HistoryAction::Recent { backend, limit } => {
            let entries = history
                .query_recent(&backend, limit)
                .map_err(|e| format!("Failed to query recent results: {}", e))?;
            print_entries(&entries, "Recent Results");
        }
        HistoryAction::Best {
            backend,
            dataset,
            limit,
        } => {
            let entries = history
                .query_best(&backend, dataset.as_deref(), limit)
                .map_err(|e| format!("Failed to query best results: {}", e))?;
            print_entries(&entries, "Best Results");
        }
        HistoryAction::Range {
            start,
            end,
            backend,
        } => {
            let entries = history
                .query_by_date_range(&start, &end, backend.as_deref())
                .map_err(|e| format!("Failed to query date range: {}", e))?;
            print_entries(&entries, "Date Range Results");
        }
        HistoryAction::Compare {
            backend1,
            backend2,
            dataset,
        } => {
            let entries = history
                .compare_backends(&backend1, &backend2, dataset.as_deref())
                .map_err(|e| format!("Failed to compare backends: {}", e))?;
            print_entries(&entries, "Backend Comparison");
        }
        HistoryAction::Rebuild => {
            history
                .rebuild_index()
                .map_err(|e| format!("Failed to rebuild index: {}", e))?;
            println!("✓ SQLite index rebuilt successfully");
        }
    }

    Ok(())
}

#[cfg(not(feature = "eval"))]
/// Stub implementation when `eval` is disabled.
pub fn run(_args: HistoryArgs) -> Result<(), String> {
    Err("History command requires 'eval' feature".to_string())
}

#[cfg(feature = "eval")]
fn print_stats(stats: &HistoryStats) {
    println!("=== Evaluation History Statistics ===\n");
    println!("Total entries: {}", stats.total_entries);

    if let Some(avg_f1) = stats.avg_f1 {
        println!("Average F1: {:.2}%", avg_f1 * 100.0);
    }

    if !stats.by_backend.is_empty() {
        println!("\nBy backend:");
        let mut backends: Vec<_> = stats.by_backend.iter().collect();
        backends.sort_by(|a, b| b.1.cmp(a.1));
        for (backend, count) in backends {
            println!("  {}: {} entries", backend, count);
        }
    }

    if !stats.by_dataset.is_empty() {
        println!("\nBy dataset:");
        let mut datasets: Vec<_> = stats.by_dataset.iter().collect();
        datasets.sort_by(|a, b| b.1.cmp(a.1));
        for (dataset, count) in datasets {
            println!("  {}: {} entries", dataset, count);
        }
    }
}

#[cfg(feature = "eval")]
fn print_entries(entries: &[anno_eval::eval::history::EvalHistoryEntry], title: &str) {
    println!("=== {} ===\n", title);

    if entries.is_empty() {
        println!("No results found.");
        return;
    }

    println!(
        "{:<15} {:<20} {:<10} {:<8} {:<8} {:<8} {:<10}",
        "Backend", "Dataset", "Task", "F1", "Prec", "Recall", "Examples"
    );
    println!("{}", "-".repeat(90));

    for entry in entries {
        let f1_str = entry
            .f1
            .map(|f| format!("{:.2}%", f * 100.0))
            .unwrap_or_else(|| "N/A".to_string());
        let prec_str = entry
            .precision
            .map(|f| format!("{:.2}%", f * 100.0))
            .unwrap_or_else(|| "N/A".to_string());
        let recall_str = entry
            .recall
            .map(|f| format!("{:.2}%", f * 100.0))
            .unwrap_or_else(|| "N/A".to_string());

        println!(
            "{:<15} {:<20} {:<10} {:<8} {:<8} {:<8} {:<10}",
            entry.backend, entry.dataset, entry.task, f1_str, prec_str, recall_str, entry.n
        );
    }

    println!("\nTotal: {} entries", entries.len());
}
