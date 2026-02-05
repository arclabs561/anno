//! Repeat muxer runs across seeds and aggregate.
//!
//! This is a Rust-native replacement for ad-hoc shell/Python wrappers.
//!
//! Usage:
//! - `cargo run -p anno-eval --features eval --bin muxer_repeat -- --runs 10 --seed-base 0 --log .generated/muxer_repeat.jsonl --agg .generated/muxer_repeat_agg.json`
//!
//! This binary supports a small set of CLI flags and otherwise defers to the same env vars as the
//! harness (`matrix_muxer_ci.rs`).

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut runs: u64 = 10;
    let mut seed_base: u64 = 0;
    let mut log_path: PathBuf = PathBuf::from(".generated/muxer_repeat.jsonl");
    let mut agg_path: Option<PathBuf> = Some(PathBuf::from(".generated/muxer_repeat_agg.json"));
    let mut mode: Option<String> = None;
    let mut task: Option<String> = None;
    let mut max_examples: Option<u64> = None;
    let mut datasets_per_run: Option<u64> = None;
    let mut backends_per_run: Option<u64> = None;
    let mut require_cached: bool = false;
    let mut try_download_on_empty: bool = false;
    let mut fixed_datasets: Option<String> = None;
    let mut fixed_backend: Option<String> = None;
    let mut pin_lang: Option<String> = None;
    let mut pin_domain: Option<String> = None;
    let mut pin_backend: Option<String> = None;
    let mut verbose: bool = false;

    while let Some(a) = args.first().cloned() {
        args.remove(0);
        match a.as_str() {
            "--runs" => {
                if let Some(v) = args.first().cloned() {
                    args.remove(0);
                    runs = v.parse()?;
                }
            }
            "--seed-base" => {
                if let Some(v) = args.first().cloned() {
                    args.remove(0);
                    seed_base = v.parse()?;
                }
            }
            "--log" => {
                if let Some(v) = args.first().cloned() {
                    args.remove(0);
                    log_path = PathBuf::from(v);
                }
            }
            "--agg" => {
                if let Some(v) = args.first().cloned() {
                    args.remove(0);
                    agg_path = Some(PathBuf::from(v));
                }
            }
            "--no-agg" => {
                agg_path = None;
            }
            "--mode" => {
                if let Some(v) = args.first().cloned() {
                    args.remove(0);
                    mode = Some(v);
                }
            }
            "--task" => {
                if let Some(v) = args.first().cloned() {
                    args.remove(0);
                    task = Some(v);
                }
            }
            "--max-examples" => {
                if let Some(v) = args.first().cloned() {
                    args.remove(0);
                    max_examples = Some(v.parse()?);
                }
            }
            "--datasets-per-run" => {
                if let Some(v) = args.first().cloned() {
                    args.remove(0);
                    datasets_per_run = Some(v.parse()?);
                }
            }
            "--backends-per-run" => {
                if let Some(v) = args.first().cloned() {
                    args.remove(0);
                    backends_per_run = Some(v.parse()?);
                }
            }
            "--require-cached" => {
                require_cached = true;
            }
            "--try-download-on-empty" => {
                try_download_on_empty = true;
            }
            "--fixed-datasets" => {
                if let Some(v) = args.first().cloned() {
                    args.remove(0);
                    fixed_datasets = Some(v);
                }
            }
            "--fixed-backend" => {
                if let Some(v) = args.first().cloned() {
                    args.remove(0);
                    fixed_backend = Some(v);
                }
            }
            "--pin-lang" => {
                if let Some(v) = args.first().cloned() {
                    args.remove(0);
                    pin_lang = Some(v);
                }
            }
            "--pin-domain" => {
                if let Some(v) = args.first().cloned() {
                    args.remove(0);
                    pin_domain = Some(v);
                }
            }
            "--pin-backend" => {
                if let Some(v) = args.first().cloned() {
                    args.remove(0);
                    pin_backend = Some(v);
                }
            }
            "--verbose" => {
                verbose = true;
            }
            _ => {}
        }
    }

    if runs == 0 {
        return Err("runs must be > 0".into());
    }

    if let Some(parent) = log_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    if log_path.exists() {
        std::fs::remove_file(&log_path)?;
    }

    // Apply CLI flags as env vars used by the harness.
    if let Some(mode) = mode {
        std::env::set_var("ANNO_MUXER_MODE", mode);
    } else if std::env::var("ANNO_MUXER_MODE").is_err() {
        // Default to measure mode; this can still be overridden by the user's env.
        std::env::set_var("ANNO_MUXER_MODE", "measure");
    }
    if let Some(task) = task {
        std::env::set_var("ANNO_MATRIX_TASK", task);
    }
    if let Some(n) = max_examples {
        std::env::set_var("ANNO_MAX_EXAMPLES", n.to_string());
    }
    if let Some(n) = datasets_per_run {
        std::env::set_var("ANNO_MUXER_DATASETS_PER_RUN", n.to_string());
    }
    if let Some(n) = backends_per_run {
        std::env::set_var("ANNO_MUXER_BACKENDS_PER_RUN", n.to_string());
    }
    if require_cached {
        std::env::set_var("ANNO_MATRIX_REQUIRE_CACHED", "1");
    }
    if try_download_on_empty {
        std::env::set_var("ANNO_MATRIX_TRY_DOWNLOAD_ON_EMPTY", "1");
    }
    if let Some(v) = fixed_datasets {
        std::env::set_var("ANNO_MUXER_FIXED_DATASETS", v);
    }
    if let Some(v) = fixed_backend {
        std::env::set_var("ANNO_MUXER_FIXED_BACKEND", v);
    }
    if let Some(v) = pin_lang {
        std::env::set_var("ANNO_MUXER_PIN_LANG", v);
    }
    if let Some(v) = pin_domain {
        std::env::set_var("ANNO_MUXER_PIN_DOMAIN", v);
    }
    if let Some(v) = pin_backend {
        std::env::set_var("ANNO_MUXER_PIN_BACKEND", v);
    }
    if verbose {
        std::env::set_var("ANNO_MUXER_VERBOSE", "1");
    }

    std::env::set_var(
        "ANNO_MUXER_DECISIONS_FILE",
        log_path.to_string_lossy().to_string(),
    );

    for i in 0..runs {
        let seed = seed_base + i;
        anno_eval::muxer_matrix::run_randomized_matrix_sample_with_seed(seed);
    }

    if let Some(out) = agg_path {
        if let Some(parent) = out.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let v = anno_eval::muxer_agg_lib::aggregate_jsonl_paths(&[log_path.clone()])
            .map_err(|e| format!("muxer_repeat: aggregate failed: {e}"))?;
        std::fs::write(out, serde_json::to_string_pretty(&v)?)?;
    }

    Ok(())
}
