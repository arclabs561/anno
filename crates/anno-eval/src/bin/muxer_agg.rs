//! Aggregate muxer decision/outcome logs into group-by summaries.
//!
//! This is intentionally lightweight: no clap, no network, no side effects beyond optional output.
//!
//! Usage:
//! - `cargo run -p anno-eval --features eval-advanced --bin muxer_agg -- <path.jsonl> [more.jsonl ...]`
//! - Or set `ANNO_MUXER_AGG_INPUT=<path.jsonl>` (arg wins).
//!
//! Output:
//! - JSON summary to stdout, or to `--out <path>` / `ANNO_MUXER_AGG_OUTPUT=<path>`.

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut out_path: Option<String> = None;
    let mut inputs: Vec<String> = Vec::new();
    while let Some(a) = args.first().cloned() {
        args.remove(0);
        match a.as_str() {
            "--out" => {
                if let Some(p) = args.first().cloned() {
                    args.remove(0);
                    out_path = Some(p);
                }
            }
            "--input" => {
                if let Some(p) = args.first().cloned() {
                    args.remove(0);
                    inputs.push(p);
                }
            }
            _ => {
                // Positional inputs.
                inputs.push(a);
            }
        }
    }
    if inputs.is_empty() {
        if let Ok(p) = std::env::var("ANNO_MUXER_AGG_INPUT") {
            let t = p.trim();
            if !t.is_empty() {
                inputs.push(t.to_string());
            }
        }
    }
    if inputs.is_empty() {
        return Err("missing input path(s) (args or ANNO_MUXER_AGG_INPUT)"
            .to_string()
            .into());
    }
    let out_path = out_path.or_else(|| std::env::var("ANNO_MUXER_AGG_OUTPUT").ok());

    let paths: Vec<PathBuf> = inputs.iter().map(|s| PathBuf::from(s)).collect();
    let out = anno_eval::muxer_agg_lib::aggregate_jsonl_paths(&paths)
        .map_err(|e| format!("muxer_agg: {e}"))?;
    let s = serde_json::to_string_pretty(&out)?;

    match out_path {
        None => {
            println!("{s}");
        }
        Some(p) => {
            if let Some(parent) = std::path::Path::new(&p).parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            std::fs::write(&p, s)?;
        }
    }
    Ok(())
}
