//! Cache command - Cache management

use clap::{Parser, Subcommand};
use std::fs;

use super::super::output::{color, format_size};
use super::super::utils::get_cache_dir;

/// Cache management
#[derive(Parser, Debug)]
pub struct CacheArgs {
    /// Action to perform
    #[command(subcommand)]
    pub action: CacheAction,
}

#[derive(Subcommand, Debug)]
pub enum CacheAction {
    /// List cached results
    #[command(visible_alias = "ls")]
    List,

    /// Clear all cache
    Clear,

    /// Show cache statistics
    Stats,

    /// Invalidate cache entries
    Invalidate {
        /// Invalidate entries for specific model
        #[arg(long, value_name = "MODEL")]
        model: Option<String>,

        /// Invalidate entries for specific file
        #[arg(long, value_name = "FILE")]
        file: Option<String>,
    },
}

pub fn run(args: CacheArgs) -> Result<(), String> {
    let cache_dir = get_cache_dir()?;

    match args.action {
        CacheAction::List => {
            if !cache_dir.exists() {
                println!("Cache directory does not exist: {}", cache_dir.display());
                return Ok(());
            }

            let entries = fs::read_dir(&cache_dir)
                .map_err(|e| format!("Failed to read cache directory: {}", e))?;

            let mut files: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_file())
                .collect();
            files.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).ok());

            println!("Cached results ({} files):", files.len());
            for entry in files {
                if let Ok(metadata) = entry.metadata() {
                    let size = metadata.len();
                    let modified = if let Ok(modified_time) = metadata.modified() {
                        if let Ok(duration) = modified_time.duration_since(std::time::UNIX_EPOCH) {
                            if let Some(dt) = chrono::DateTime::<chrono::Utc>::from_timestamp(
                                duration.as_secs() as i64,
                                0,
                            ) {
                                dt.format("%Y-%m-%d %H:%M:%S").to_string()
                            } else {
                                "unknown".to_string()
                            }
                        } else {
                            "unknown".to_string()
                        }
                    } else {
                        "unknown".to_string()
                    };

                    println!(
                        "  {} ({}) - {}",
                        entry.file_name().to_string_lossy(),
                        format_size(size),
                        modified
                    );
                }
            }
        }
        CacheAction::Clear => {
            if cache_dir.exists() {
                fs::remove_dir_all(&cache_dir)
                    .map_err(|e| format!("Failed to clear cache: {}", e))?;
                println!("{} Cache cleared", color("32", "✓"));
            } else {
                println!("Cache directory does not exist");
            }
        }
        CacheAction::Stats => {
            if !cache_dir.exists() {
                println!("Cache directory does not exist");
                return Ok(());
            }

            let entries = fs::read_dir(&cache_dir)
                .map_err(|e| format!("Failed to read cache directory: {}", e))?;

            let mut total_size = 0u64;
            let mut count = 0usize;

            for entry in entries {
                if let Ok(entry) = entry {
                    if let Ok(metadata) = entry.metadata() {
                        total_size += metadata.len();
                        count += 1;
                    }
                }
            }

            println!("Cache Statistics:");
            println!("  Files: {}", count);
            println!("  Total size: {}", format_size(total_size));
        }
        CacheAction::Invalidate { model, file } => {
            if !cache_dir.exists() {
                println!("Cache directory does not exist");
                return Ok(());
            }

            let entries = fs::read_dir(&cache_dir)
                .map_err(|e| format!("Failed to read cache directory: {}", e))?;

            let mut removed = 0usize;

            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

                    let should_remove = if let Some(ref m) = model {
                        name.starts_with(&format!("{}-", m))
                    } else if let Some(ref f) = file {
                        name.contains(f)
                    } else {
                        false
                    };

                    if should_remove {
                        if fs::remove_file(&path).is_ok() {
                            removed += 1;
                        }
                    }
                }
            }

            println!("{} Removed {} cache entries", color("32", "✓"), removed);
        }
    }

    Ok(())
}
