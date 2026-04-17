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

/// Cache subcommand actions.
#[derive(Subcommand, Debug)]
pub enum CacheAction {
    /// List cached results
    #[command(visible_alias = "ls")]
    List,

    /// Clear all cache (requires --confirm)
    Clear {
        /// Confirm deletion (required to prevent accidental data loss)
        #[arg(long)]
        confirm: bool,
    },

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
    #[cfg(feature = "eval")]
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

/// Walk a directory tree, calling `f` for every regular file found.
fn walk_files(dir: &std::path::Path, f: &mut impl FnMut(&std::path::Path)) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            f(&path);
        } else if path.is_dir() {
            walk_files(&path, f);
        }
    }
}

/// Execute the cache management command.
pub fn run(args: CacheArgs) -> Result<(), String> {
    let cache_dir = get_cache_dir()?;

    // Result cache lives under {cache_dir}/results/
    let result_cache_dir = cache_dir.join("results");

    match args.action {
        CacheAction::List => {
            if !cache_dir.exists() {
                println!("Cache directory does not exist: {}", cache_dir.display());
                return Ok(());
            }

            // Top-level files (model downloads, datasets, etc.)
            let top_files: Vec<_> = fs::read_dir(&cache_dir)
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .filter(|e| e.path().is_file())
                        .collect()
                })
                .unwrap_or_default();

            if !top_files.is_empty() {
                println!("Model / dataset cache ({} files):", top_files.len());
                let mut sorted = top_files;
                sorted.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).ok());
                for entry in sorted {
                    if let Ok(meta) = entry.metadata() {
                        println!(
                            "  {} ({})",
                            entry.file_name().to_string_lossy(),
                            format_size(meta.len())
                        );
                    }
                }
            }

            // Result cache: group by model segment
            if result_cache_dir.exists() {
                let mut result_count = 0usize;
                let mut result_size = 0u64;
                let mut by_model: std::collections::BTreeMap<String, (usize, u64)> =
                    std::collections::BTreeMap::new();

                walk_files(&result_cache_dir, &mut |path| {
                    let size = path.metadata().map(|m| m.len()).unwrap_or(0);
                    result_count += 1;
                    result_size += size;
                    // Path pattern: results/{model-version}/{shard}/{hash}.json
                    if let Some(model_seg) = path
                        .strip_prefix(&result_cache_dir)
                        .ok()
                        .and_then(|p| p.components().next())
                        .and_then(|c| c.as_os_str().to_str())
                    {
                        let e = by_model.entry(model_seg.to_string()).or_default();
                        e.0 += 1;
                        e.1 += size;
                    }
                });

                println!(
                    "\nResult cache ({} entries, {}):",
                    result_count,
                    format_size(result_size)
                );
                for (model, (count, size)) in &by_model {
                    println!("  {} — {} entries, {}", model, count, format_size(*size));
                }
            }
        }
        CacheAction::Clear { confirm } => {
            if !cache_dir.exists() {
                println!("Cache directory does not exist");
                return Ok(());
            }
            if !confirm {
                eprintln!(
                    "This will delete all cached models and results in {}",
                    cache_dir.display()
                );
                eprintln!("Run with --confirm to proceed: anno cache clear --confirm");
                return Err("cache clear requires --confirm".into());
            }
            fs::remove_dir_all(&cache_dir).map_err(|e| format!("Failed to clear cache: {}", e))?;
            println!(
                "{} Cache cleared (model cache + result cache)",
                color("32", "✓")
            );
        }
        CacheAction::Stats => {
            if !cache_dir.exists() {
                println!("Cache directory does not exist");
                return Ok(());
            }

            let mut model_count = 0usize;
            let mut model_size = 0u64;
            let mut result_count = 0usize;
            let mut result_size = 0u64;

            walk_files(&cache_dir, &mut |path| {
                let size = path.metadata().map(|m| m.len()).unwrap_or(0);
                if path.starts_with(&result_cache_dir) {
                    result_count += 1;
                    result_size += size;
                } else {
                    model_count += 1;
                    model_size += size;
                }
            });

            println!("Cache statistics:");
            println!(
                "  Model / dataset cache: {} files, {}",
                model_count,
                format_size(model_size)
            );
            println!(
                "  Result cache (--batch --cache): {} entries, {}",
                result_count,
                format_size(result_size)
            );
            println!(
                "  Total: {} files, {}",
                model_count + result_count,
                format_size(model_size + result_size)
            );
        }
        CacheAction::Invalidate { model, file } => {
            if !cache_dir.exists() {
                println!("Cache directory does not exist");
                return Ok(());
            }

            let mut removed = 0usize;

            walk_files(&cache_dir, &mut |path| {
                let path_str = path.to_string_lossy();
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default();

                let should_remove = if let Some(ref m) = model {
                    // Match both top-level (model download) and result cache path segments
                    name.starts_with(&format!("{}-", m)) || path_str.contains(m)
                } else if let Some(ref f) = file {
                    name.contains(f) || path_str.contains(f)
                } else {
                    false
                };

                if should_remove && fs::remove_file(path).is_ok() {
                    removed += 1;
                }
            });

            println!("{} Removed {} cache entries", color("32", "✓"), removed);
        }

        #[cfg(feature = "eval")]
        CacheAction::SyncS3 {
            bucket,
            datasets,
            limit,
            dry_run,
        } => {
            use anno_eval::eval::loader::{DatasetId, DatasetLoader};
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
                entries.truncate(lim);
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
