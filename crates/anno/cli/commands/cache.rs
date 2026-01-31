//! Cache command - Cache management

use clap::{Parser, Subcommand};
use std::fs;

#[cfg(feature = "eval")]
use crate::eval::prediction_cache::PredictionCache;

use super::super::output::{color, format_size};
use super::super::utils::get_cache_dir;

/// Cache management
#[derive(Parser, Debug)]
pub struct CacheArgs {
    /// Action to perform
    #[command(subcommand)]
    pub action: CacheAction,
}

/// Cache subcommand actions.
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

    /// Upload cached datasets to S3 (best-effort).
    ///
    /// This is useful when you have a warm local cache and want to periodically refresh the
    /// shared S3 cache used by CI.
    #[cfg(feature = "eval-advanced")]
    SyncS3 {
        /// S3 bucket name (defaults to `ANNO_S3_BUCKET` or `arc-anno-data`).
        #[arg(long)]
        bucket: Option<String>,

        /// Optional comma-separated list of dataset IDs to upload.
        ///
        /// Examples: `--datasets Wnut17` or `--datasets Wnut17,DocRED`
        #[arg(long, value_delimiter = ',')]
        datasets: Vec<String>,

        /// Maximum number of datasets to upload (sorted by cache filename).
        #[arg(long)]
        limit: Option<usize>,

        /// Dry run (print what would be uploaded, but don't call AWS).
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
}

/// Execute the cache management command.
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

            #[cfg(feature = "eval")]
            {
                let pred_path = PredictionCache::default_path();
                let pred_cache = PredictionCache::load_or_create(&pred_path);
                if let Err(e) = pred_cache.clear() {
                    eprintln!("Warning: Failed to clear prediction cache: {}", e);
                } else {
                    println!("{} Prediction cache cleared", color("32", "✓"));
                }
            }
        }
        CacheAction::Stats => {
            if !cache_dir.exists() {
                println!("Cache directory does not exist");
            } else {
                let entries = fs::read_dir(&cache_dir)
                    .map_err(|e| format!("Failed to read cache directory: {}", e))?;

                let mut total_size = 0u64;
                let mut count = 0usize;

                for entry in entries {
                    if let Ok(entry) = entry.and_then(|e| e.metadata().map(|m| (e, m))) {
                        total_size += entry.1.len();
                        count += 1;
                    }
                }

                println!("File Cache Statistics:");
                println!("  Files: {}", count);
                println!("  Total size: {}", format_size(total_size));
            }

            #[cfg(feature = "eval")]
            {
                let pred_path = PredictionCache::default_path();
                let pred_cache = PredictionCache::load_or_create(&pred_path);
                if pred_cache.is_enabled() {
                    let stats = pred_cache.stats();
                    println!("\nPrediction Cache ({})", pred_path.display());
                    println!("  Total predictions: {}", stats.total_entries);
                    if !stats.by_backend.is_empty() {
                        println!("  By backend:");
                        // Sort backends for consistent output
                        let mut backends: Vec<_> = stats.by_backend.iter().collect();
                        backends.sort_by_key(|(k, _)| *k);
                        for (backend, count) in backends {
                            println!("    {}: {}", backend, count);
                        }
                    }
                } else {
                    println!("\nPrediction Cache: <empty>");
                }
            }
        }
        CacheAction::Invalidate { model, file } => {
            if !cache_dir.exists() {
                println!("Cache directory does not exist");
                return Ok(());
            }

            let entries = fs::read_dir(&cache_dir)
                .map_err(|e| format!("Failed to read cache directory: {}", e))?;

            let mut removed = 0usize;

            for entry in entries.flatten() {
                let path = entry.path();
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

                let should_remove = if let Some(ref m) = model {
                    name.starts_with(&format!("{}-", m))
                } else if let Some(ref f) = file {
                    name.contains(f)
                } else {
                    false
                };

                if should_remove && fs::remove_file(&path).is_ok() {
                    removed += 1;
                }
            }

            println!("{} Removed {} cache entries", color("32", "✓"), removed);
        }

        #[cfg(feature = "eval-advanced")]
        CacheAction::SyncS3 {
            bucket,
            datasets,
            limit,
            dry_run,
        } => {
            use crate::eval::loader::{DatasetId, DatasetLoader};
            use std::process::Command;

            // Resolve bucket name (prefer explicit arg, then env, then default).
            let bucket = bucket
                .or_else(|| {
                    std::env::var("ANNO_S3_BUCKET")
                        .ok()
                        .filter(|s| !s.trim().is_empty())
                })
                .unwrap_or_else(|| "arc-anno-data".to_string());

            // Loader owns the manifest + knows the dataset cache dir.
            // Note: it uses ANNO_CACHE_DIR if set; otherwise platform cache.
            let loader = DatasetLoader::new().map_err(|e| e.to_string())?;
            let mut entries = loader.cached_manifest_entries();

            // Optional: filter to explicit dataset list.
            if !datasets.is_empty() {
                let mut wanted: Vec<DatasetId> = Vec::new();
                for ds in datasets {
                    let id: DatasetId = ds
                        .parse()
                        .map_err(|e| format!("Invalid dataset id '{}': {}", ds, e))?;
                    wanted.push(id);
                }
                let wanted_keys: std::collections::BTreeSet<&'static str> =
                    wanted.iter().map(|d| d.cache_filename()).collect();
                let available_keys: std::collections::BTreeSet<&str> =
                    entries.iter().map(|e| e.dataset_id.as_str()).collect();
                let mut missing: Vec<&'static str> = wanted_keys
                    .iter()
                    .copied()
                    .filter(|k| !available_keys.contains(*k))
                    .collect();
                missing.sort();
                if !missing.is_empty() {
                    eprintln!(
                        "Warning: {} requested datasets are not cached locally (missing manifest entries): {:?}",
                        missing.len(),
                        missing
                    );
                }
                entries.retain(|e| wanted_keys.contains(e.dataset_id.as_str()));
            }

            if let Some(lim) = limit {
                entries.truncate(lim.max(0));
            }

            if entries.is_empty() {
                println!("No cached datasets found in manifest; nothing to upload.");
                return Ok(());
            }

            println!("S3 bucket: {}", bucket);
            println!("Cached datasets in manifest: {}", entries.len());
            if dry_run {
                println!("Dry run: true");
            }

            let mut ok = 0usize;
            let mut skipped = 0usize;
            let mut failed = 0usize;

            if !dry_run {
                // Fail fast on obviously-missing AWS credentials.
                let out = Command::new("aws")
                    .args(["sts", "get-caller-identity"])
                    .output()
                    .map_err(|e| format!("Failed to run aws sts get-caller-identity: {}", e))?;
                if !out.status.success() {
                    return Err(format!(
                        "AWS credentials not available (aws sts get-caller-identity failed): {}",
                        String::from_utf8_lossy(&out.stderr)
                    ));
                }
            }

            for e in entries {
                let Some(id) = DatasetId::from_cache_filename(&e.dataset_id) else {
                    skipped += 1;
                    eprintln!(
                        "Skip: cannot map cache filename to DatasetId: {}",
                        e.dataset_id
                    );
                    continue;
                };

                if dry_run {
                    println!("Would upload: {}", id.cache_filename());
                    ok += 1;
                    continue;
                }

                match loader.upload_cached_dataset_to_s3(&bucket, id) {
                    Ok(()) => ok += 1,
                    Err(err) => {
                        failed += 1;
                        eprintln!("Upload failed for {}: {}", id.cache_filename(), err);
                    }
                }
            }

            println!(
                "{} Uploaded {} datasets (skipped {}, failed {})",
                color("32", "✓"),
                ok,
                skipped,
                failed
            );
            if failed > 0 {
                return Err(format!("S3 sync had {} failures", failed));
            }
        }
    }

    Ok(())
}
