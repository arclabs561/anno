//! Batch command — multi-document extraction with optional parallelism and result caching.
//!
//! ## Parallelism (`--parallel N`)
//!
//! Extraction is delegated to [`anno::Model::extract_batch`], allowing a backend to use its
//! native inference batch. When `N > 1`, bounded extraction batches and CLI-only enrichment
//! use Rayon thread pools capped at `N` workers.
//!
//! ## Caching (`--cache`)
//!
//! Cacheable deterministic results are persisted to
//! `{cache_dir}/results/v2/{model}-{version}/{shard}/{hash}.json`. The key includes
//! the document id and text, CLI release and executable digest, selected backend and runtime
//! model version, and KB-linking.
//! Artifact-backed, remote, dynamic, and coreference runs bypass the result cache because
//! their effective model/configuration identity is not available through the public model API.
//! Cache entries are never evicted automatically; use `anno cache clear` to flush.

use super::super::parser::{ModelBackend, OutputFormat};
use clap::Parser;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

/// Batch processing
#[derive(Parser, Debug)]
pub struct BatchArgs {
    /// Process directory of text files (.txt, .md, .html, .htm, .pdf)
    #[arg(short, long, value_name = "DIR")]
    pub dir: Option<String>,

    /// Read from stdin (JSONL: one `{"id":"…","text":"…"}` object per line)
    #[arg(long)]
    pub stdin: bool,

    /// Model backend to use
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Run coreference resolution on each document
    #[arg(long)]
    pub coref: bool,

    /// Link tracks to KB identities
    #[arg(long)]
    pub link_kb: bool,

    /// Number of parallel batch and enrichment workers (1 = sequential)
    #[arg(short, long, default_value = "1")]
    pub parallel: usize,

    /// Maximum documents sent to one model batch (must be at least 1)
    #[arg(long, default_value = "32")]
    pub batch_size: NonZeroUsize,

    /// Show progress bar
    #[arg(long)]
    pub progress: bool,

    /// Cache eligible deterministic extraction results
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

// Cache helpers

const RESULT_CACHE_LAYOUT_VERSION: &str = "v2";

/// Known inputs that can change a cache-eligible serialized batch result.
///
/// `model_name` records the CLI-selected backend while `runtime_model_name` and
/// `runtime_model_version` come from the instantiated model. Keeping both avoids
/// aliases such as `auto` and `stacked` sharing a cache namespace by accident.
#[derive(Clone, Copy)]
struct ResultCacheIdentity<'a> {
    document_id: &'a str,
    text: &'a str,
    cli_version: &'a str,
    executable_fingerprint: &'a str,
    model_name: &'a str,
    runtime_model_name: &'a str,
    runtime_model_version: &'a str,
    link_kb: bool,
}

impl ResultCacheIdentity<'_> {
    /// Produce an unambiguous, length-prefixed cache digest.
    ///
    /// Delimiters alone allow ambiguous tuples (for example `("ab", "c")`
    /// versus `("a", "bc")`). Length-prefixing each field keeps the hash input
    /// injective for the values represented here.
    fn digest(&self) -> String {
        use xxhash_rust::xxh3::xxh3_64;

        let mut bytes = Vec::with_capacity(
            self.document_id.len()
                + self.text.len()
                + self.cli_version.len()
                + self.executable_fingerprint.len()
                + self.model_name.len()
                + self.runtime_model_name.len()
                + self.runtime_model_version.len()
                + 48,
        );
        for field in [
            self.document_id,
            self.text,
            self.cli_version,
            self.executable_fingerprint,
            self.model_name,
            self.runtime_model_name,
            self.runtime_model_version,
        ] {
            bytes.extend_from_slice(&(field.len() as u64).to_le_bytes());
            bytes.extend_from_slice(field.as_bytes());
        }
        bytes.push(u8::from(self.link_kb));
        format!("{:016x}", xxh3_64(&bytes))
    }
}

/// Hash the executable that determines local deterministic backend behavior.
fn executable_fingerprint() -> Result<String, String> {
    use std::io::Read;
    use xxhash_rust::xxh3::Xxh3;

    let path = std::env::current_exe()
        .map_err(|error| format!("cannot locate current executable: {error}"))?;
    let mut executable = std::fs::File::open(&path)
        .map_err(|error| format!("cannot read executable '{}': {error}", path.display()))?;
    let mut hasher = Xxh3::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let bytes = executable
            .read(&mut buffer)
            .map_err(|error| format!("cannot hash executable '{}': {error}", path.display()))?;
        if bytes == 0 {
            break;
        }
        hasher.update(&buffer[..bytes]);
    }
    Ok(format!("{:016x}", hasher.digest()))
}

/// Return why a result cache cannot safely represent this run.
///
/// The `Model` trait exposes a name and a logical version but no artifact digest
/// or complete backend configuration. Admit only local, deterministic backends
/// whose output is fully identified by the CLI selection and current binary.
fn cache_ineligibility_reason(backend: ModelBackend, coref: bool) -> Option<&'static str> {
    if coref {
        return Some(
            "coreference can load a separate model artifact whose identity is not cache-keyed",
        );
    }

    match backend {
        ModelBackend::Pattern | ModelBackend::Heuristic | ModelBackend::Minimal => None,
        #[cfg(feature = "heuristic-fr")]
        ModelBackend::HeuristicFr => None,
        _ => Some(
            "the selected backend may depend on model artifacts, runtime configuration, or a remote service that is not cache-keyed",
        ),
    }
}

/// Return half-open input ranges for bounded model batches.
fn batch_ranges(total: usize, batch_size: usize) -> Vec<std::ops::Range<usize>> {
    assert!(batch_size > 0, "clap validates --batch-size >= 1");
    (0..total)
        .step_by(batch_size)
        .map(|start| start..(start + batch_size).min(total))
        .collect()
}

/// Extract bounded chunks while preserving the original text order.
///
/// Each backend call can use model-native batching. Independent calls are
/// concurrent only when requested; the resulting chunk order is restored before
/// individual results are associated with document IDs.
fn extract_bounded_batches(
    model: &dyn anno::Model,
    texts: &[&str],
    batch_size: NonZeroUsize,
    parallel: usize,
    on_batch_completed: impl Fn(usize) + Sync,
) -> Result<Vec<anno::Result<Vec<anno::Entity>>>, String> {
    // A small input set must still make use of requested parallelism when the
    // maximum batch size is larger than the entire input set.
    let chunk_size = if parallel > 1 {
        batch_size.get().min((texts.len() / parallel).max(1))
    } else {
        batch_size.get()
    };
    let ranges = batch_ranges(texts.len(), chunk_size);
    let extract_one = |(batch_index, range): (usize, std::ops::Range<usize>)| {
        let batch = &texts[range.clone()];
        let results = model.extract_batch(batch, None);
        if results.len() != batch.len() {
            return Err(format!(
                "Model '{}' returned {} results for batch {} with {} documents",
                model.name(),
                results.len(),
                batch_index + 1,
                batch.len(),
            ));
        }
        on_batch_completed(batch.len());
        Ok((range.start, results))
    };

    let mut chunks: Vec<(usize, Vec<anno::Result<Vec<anno::Entity>>>)> = if parallel > 1 {
        use rayon::prelude::*;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(parallel)
            .build()
            .map_err(|error| format!("Failed to build thread pool: {error}"))?;
        pool.install(|| {
            ranges
                .into_par_iter()
                .enumerate()
                .map(&extract_one)
                .collect::<Result<_, _>>()
        })?
    } else {
        ranges
            .into_iter()
            .enumerate()
            .map(&extract_one)
            .collect::<Result<_, _>>()?
    };
    chunks.sort_unstable_by_key(|(start, _)| *start);
    Ok(chunks
        .into_iter()
        .flat_map(|(_, results)| results)
        .collect())
}

/// Derive a filesystem path for a cached document result.
///
/// Layout: `{cache_root}/results/v2/{model}-{version}/{first_2_hex}/{full_hash}.json`.
/// The versioned layout cleanly invalidates the legacy text-only cache format.
fn result_cache_path(cache_root: &Path, identity: &ResultCacheIdentity<'_>) -> PathBuf {
    let hash = identity.digest();
    let shard = &hash[..2];
    let segment = format!(
        "{}-{}",
        identity.runtime_model_name.replace(['/', '\\', ':'], "_"),
        identity
            .runtime_model_version
            .replace(['/', '\\', ':'], "_"),
    );
    cache_root
        .join("results")
        .join(RESULT_CACHE_LAYOUT_VERSION)
        .join(segment)
        .join(shard)
        .join(format!("{}.json", hash))
}

fn try_load_cached(path: &Path, document_id: &str, text: &str) -> Option<anno::GroundedDocument> {
    let bytes = std::fs::read(path).ok()?;
    let doc: anno::GroundedDocument = serde_json::from_slice(&bytes).ok()?;
    (doc.id() == document_id && doc.text() == text).then_some(doc)
}

fn store_cached(path: &Path, doc: &anno::GroundedDocument) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string(doc) {
        let _ = std::fs::write(path, json);
    }
}

// Per-document extraction

struct DocOpts {
    coref: bool,
    link_kb: bool,
    cache_path: Option<PathBuf>,
}

struct ProcessedDocument {
    document: anno::GroundedDocument,
    cache_hit: bool,
}

fn finalize_document(
    mut document: anno::GroundedDocument,
    text: &str,
    opts: &DocOpts,
) -> ProcessedDocument {
    use super::super::utils::{link_tracks_to_kb, resolve_coreference};

    if opts.coref {
        // `resolve_coreference` keeps this parameter for compatibility but does
        // not inspect it; the grounded document already contains all signals.
        resolve_coreference(&mut document, text, &[]);
    }
    if opts.link_kb {
        link_tracks_to_kb(&mut document);
    }

    if let Some(ref path) = opts.cache_path {
        store_cached(path, &document);
    }

    ProcessedDocument {
        document,
        cache_hit: false,
    }
}

// Main entry point

/// Execute the batch processing command.
pub fn run(args: BatchArgs) -> Result<(), String> {
    use std::io::{self, BufRead};

    if args.dir.is_none() && !args.stdin {
        return Err("Either --dir <DIR> or --stdin must be specified".to_string());
    }
    if args.dir.is_some() && args.stdin {
        return Err("Cannot use both --dir and --stdin. Choose one.".to_string());
    }

    // Resolve the cache root only for runs whose effective extraction identity
    // is fully known to this command. See `cache_ineligibility_reason`.
    let (cache_root, executable_fingerprint): (Option<PathBuf>, Option<String>) = if args.cache {
        if let Some(reason) = cache_ineligibility_reason(args.model, args.coref) {
            if !args.quiet {
                eprintln!("[batch] cache disabled: {reason}");
            }
            (None, None)
        } else {
            match executable_fingerprint() {
                Ok(fingerprint) => (
                    Some(super::super::utils::get_cache_dir()?),
                    Some(fingerprint),
                ),
                Err(reason) => {
                    if !args.quiet {
                        eprintln!("[batch] cache disabled: {reason}");
                    }
                    (None, None)
                }
            }
        }
    } else {
        (None, None)
    };

    // Build once; extraction below delegates to `Model::extract_batch` so a
    // backend can use native inference batching.
    let model = args.model.create_model()?;
    let selected_model_name = args.model.name();
    let model_name = model.name().to_string();
    let model_version = model.version();

    // Collect (doc_id, text) pairs from the chosen input source.
    let inputs: Vec<(String, String)> = if args.stdin {
        if !args.quiet {
            eprintln!("Reading JSONL from stdin...");
        }
        let stdin = io::stdin();
        let mut out = Vec::new();
        for (i, line) in stdin.lock().lines().enumerate() {
            let line = line.map_err(|e| format!("Failed to read stdin line {}: {}", i + 1, e))?;
            if line.trim().is_empty() {
                continue;
            }
            let json: serde_json::Value = serde_json::from_str(&line)
                .map_err(|e| format!("Failed to parse stdin line {} as JSON: {}", i + 1, e))?;
            let doc_id = json
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("stdin:{}", i + 1));
            let text = json
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("Missing 'text' field in stdin line {}", i + 1))?
                .to_string();
            out.push((doc_id, text));
        }
        out
    } else {
        let dir = args.dir.as_ref().expect("validated above");
        let dir_path = Path::new(dir);
        let entries = std::fs::read_dir(dir_path)
            .map_err(|e| format!("Failed to read directory '{}': {}", dir, e))?;

        let mut out = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let ext_ok = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| {
                    matches!(
                        e,
                        "txt" | "md" | "html" | "htm" | "xhtml" | "pdf" | "rst" | "text"
                    )
                })
                .unwrap_or(false);
            if !ext_ok {
                continue;
            }
            let path_str = path.to_string_lossy();
            let text = crate::cli::utils::read_input_file(&path_str)
                .map_err(|e| format!("Failed to read '{}': {}", path.display(), e))?;
            let doc_id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("doc{}", out.len() + 1));
            out.push((doc_id, text));
        }

        if out.is_empty() {
            return Err(format!(
                "No input files found under '{}' (expected .txt, .md, .html, .htm, .pdf, .rst)",
                args.dir.as_deref().unwrap_or("")
            ));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    };

    if !args.quiet {
        let workers = if args.parallel > 1 {
            format!("{} workers", args.parallel)
        } else {
            "sequential".to_string()
        };
        let cache_note = if cache_root.is_some() {
            ", cache on"
        } else {
            ""
        };
        eprintln!(
            "[batch] {} documents, model={}, {}{}",
            inputs.len(),
            model_name,
            workers,
            cache_note,
        );
    }

    // Progress bar setup (indicatif ProgressBar is Arc-backed, safe to clone for rayon).
    let pb = if args.progress && !args.quiet {
        use indicatif::{ProgressBar, ProgressStyle};
        let pb = ProgressBar::new(inputs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .expect("valid template")
                .progress_chars("#>-"),
        );
        Some(pb)
    } else {
        None
    };

    // Build per-document cache paths once (deterministic, parallel-safe).
    let cache_paths: Vec<Option<PathBuf>> = inputs
        .iter()
        .map(|(doc_id, text)| {
            cache_root.as_ref().map(|root| {
                let identity = ResultCacheIdentity {
                    document_id: doc_id,
                    text,
                    cli_version: env!("CARGO_PKG_VERSION"),
                    executable_fingerprint: executable_fingerprint
                        .as_deref()
                        .expect("cache root requires executable fingerprint"),
                    model_name: selected_model_name,
                    runtime_model_name: &model_name,
                    runtime_model_version: &model_version,
                    link_kb: args.link_kb,
                };
                result_cache_path(root, &identity)
            })
        })
        .collect();

    // Read cache hits before invoking the backend. A cached document is checked
    // against its source id/text as defense-in-depth against manual corruption
    // or an exceedingly unlikely digest collision.
    let mut processed: Vec<Option<ProcessedDocument>> =
        std::iter::repeat_with(|| None).take(inputs.len()).collect();
    let mut misses = Vec::new();
    for (index, ((doc_id, text), cache_path)) in inputs.iter().zip(&cache_paths).enumerate() {
        match cache_path
            .as_deref()
            .and_then(|path| try_load_cached(path, doc_id, text))
        {
            Some(document) => {
                processed[index] = Some(ProcessedDocument {
                    document,
                    cache_hit: true,
                });
                if let Some(ref pb) = pb {
                    pb.inc(1);
                }
            }
            None => misses.push(index),
        }
    }

    // Only misses reach the backend. Preserve the existing CLI document shape:
    // extraction initially creates signals only, then --coref/--link-kb add
    // tracks and identities explicitly below.
    let miss_texts: Vec<&str> = misses
        .iter()
        .map(|&index| inputs[index].1.as_str())
        .collect();
    let progress_for_batches = pb.clone();
    let extracted = extract_bounded_batches(
        model.as_ref(),
        &miss_texts,
        args.batch_size,
        args.parallel,
        move |completed| {
            if let Some(ref pb) = progress_for_batches {
                pb.inc(completed as u64);
            }
        },
    )?;

    let fresh: Vec<(usize, anno::GroundedDocument, Option<PathBuf>)> = misses
        .into_iter()
        .zip(extracted)
        .map(|(index, result)| {
            result
                .map(|entities| {
                    (
                        index,
                        anno::GroundedDocument::from_entity_signals(
                            &inputs[index].0,
                            &inputs[index].1,
                            &entities,
                        ),
                        cache_paths[index].clone(),
                    )
                })
                .map_err(|error| format!("Extraction failed for '{}': {}", inputs[index].0, error))
        })
        .collect::<Result<_, _>>()?;

    // Coreference/linking are CLI-only enrichment steps. Keep them parallelizable
    // without replacing model-native extraction batching above.
    let finalized: Vec<(usize, ProcessedDocument)> = if args.parallel > 1 {
        use rayon::prelude::*;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(args.parallel)
            .build()
            .map_err(|e| format!("Failed to build thread pool: {}", e))?;

        pool.install(|| {
            fresh
                .into_par_iter()
                .map(|(index, document, cache_path)| {
                    let opts = DocOpts {
                        coref: args.coref,
                        link_kb: args.link_kb,
                        cache_path,
                    };
                    (index, finalize_document(document, &inputs[index].1, &opts))
                })
                .collect()
        })
    } else {
        fresh
            .into_iter()
            .map(|(index, document, cache_path)| {
                let opts = DocOpts {
                    coref: args.coref,
                    link_kb: args.link_kb,
                    cache_path,
                };
                (index, finalize_document(document, &inputs[index].1, &opts))
            })
            .collect()
    };

    for (index, document) in finalized {
        processed[index] = Some(document);
    }
    let cache_hits = processed
        .iter()
        .filter(|doc| doc.as_ref().is_some_and(|doc| doc.cache_hit))
        .count();
    let documents: Vec<anno::GroundedDocument> = processed
        .into_iter()
        .enumerate()
        .map(|(index, document)| {
            document
                .map(|document| document.document)
                .ok_or_else(|| format!("Batch processing did not produce '{}'.", inputs[index].0))
        })
        .collect::<Result<_, _>>()?;

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    // Write outputs.
    write_outputs(&documents, &args)?;

    if !args.quiet {
        if cache_root.is_some() && cache_hits > 0 {
            eprintln!(
                "[batch] {} cache hits, {} computed",
                cache_hits,
                documents.len() - cache_hits
            );
        }
        if let Some(ref out) = args.output {
            eprintln!("[batch] wrote {} document(s) to {}", documents.len(), out);
        }
    }

    Ok(())
}

// Output writing

/// Convert a GroundedDocument to the clean JSON schema used by `extract --format json`.
///
/// Schema: `{id, entities: [{text, type, start, end, confidence, negated, quantifier}], tracks}`
/// This matches the extract command's output so consumers get a consistent schema.
fn doc_to_clean_json(doc: &anno::GroundedDocument, model_name: &str) -> serde_json::Value {
    let entities: Vec<serde_json::Value> = doc
        .signals()
        .iter()
        .map(|s| {
            let (start, end) = s.text_offsets().unwrap_or((0, 0));
            serde_json::json!({
                "text": s.surface(),
                "type": s.label(),
                "start": start,
                "end": end,
                "confidence": s.confidence,
                "negated": s.negated,
                "quantifier": s.quantifier.map(|q| format!("{:?}", q)),
            })
        })
        .collect();

    let tracks: Vec<serde_json::Value> = doc
        .tracks()
        .map(|t| {
            let mentions: Vec<serde_json::Value> = t
                .signals
                .iter()
                .filter_map(|sr| {
                    let sig = doc.get_signal(sr.signal_id)?;
                    let (start, end) = sig.text_offsets().unwrap_or((0, 0));
                    Some(serde_json::json!({
                        "text": sig.surface(),
                        "type": sig.label(),
                        "start": start,
                        "end": end,
                    }))
                })
                .collect();
            serde_json::json!({
                "canonical": t.canonical_surface,
                "mentions": mentions,
            })
        })
        .collect();

    let mut obj = serde_json::json!({
        "id": doc.id(),
        "model": model_name,
        "text_length": doc.text().chars().count(),
        "entity_count": entities.len(),
        "entities": entities,
    });
    if !tracks.is_empty() {
        obj["tracks"] = serde_json::json!(tracks);
    }
    obj
}

fn write_outputs(documents: &[anno::GroundedDocument], args: &BatchArgs) -> Result<(), String> {
    use super::super::output::{color, print_signals};

    let model_name = args.model.name();

    let Some(ref out_dir_str) = args.output else {
        // No output directory: print to stdout
        match args.format {
            OutputFormat::Json => {
                // Clean schema matching `extract --format json`
                let clean: Vec<serde_json::Value> = documents
                    .iter()
                    .map(|d| doc_to_clean_json(d, model_name))
                    .collect();
                let output = serde_json::to_string_pretty(&clean)
                    .map_err(|e| format!("Failed to serialize batch output: {}", e))?;
                println!("{}", output);
            }
            OutputFormat::Grounded => {
                // Raw GroundedDocument for pipeline integration
                let output = serde_json::to_string_pretty(documents)
                    .map_err(|e| format!("Failed to serialize batch output: {}", e))?;
                println!("{}", output);
            }
            OutputFormat::Jsonl => {
                // One clean JSON object per line
                for doc in documents {
                    let clean = doc_to_clean_json(doc, model_name);
                    let line = serde_json::to_string(&clean)
                        .map_err(|e| format!("Failed to serialize '{}': {}", doc.id(), e))?;
                    println!("{}", line);
                }
            }
            _ => {
                for doc in documents {
                    if !args.quiet {
                        println!("\n{}", color("1;36", &format!("Document: {}", doc.id())));
                    }
                    print_signals(doc, doc.text(), 0);
                }
            }
        }
        return Ok(());
    };

    let out_dir = PathBuf::from(out_dir_str);
    if out_dir.exists() && !out_dir.is_dir() {
        return Err(format!(
            "--output must be a directory for `anno batch`, but '{}' is not",
            out_dir.display()
        ));
    }
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("Failed to create output dir '{}': {}", out_dir.display(), e))?;

    for doc in documents {
        match args.format {
            OutputFormat::Json => {
                let path = out_dir.join(format!("{}.json", doc.id()));
                let clean = doc_to_clean_json(doc, model_name);
                let payload = serde_json::to_string_pretty(&clean)
                    .map_err(|e| format!("Failed to serialize '{}': {}", doc.id(), e))?;
                std::fs::write(&path, payload)
                    .map_err(|e| format!("Failed to write '{}': {}", path.display(), e))?;
            }
            OutputFormat::Jsonl => {
                let path = out_dir.join(format!("{}.jsonl", doc.id()));
                let clean = doc_to_clean_json(doc, model_name);
                let payload = serde_json::to_string(&clean)
                    .map_err(|e| format!("Failed to serialize '{}': {}", doc.id(), e))?;
                std::fs::write(&path, payload + "\n")
                    .map_err(|e| format!("Failed to write '{}': {}", path.display(), e))?;
            }
            _ => {
                // Grounded and other formats: raw GroundedDocument
                let path = out_dir.join(format!("{}.json", doc.id()));
                let payload = serde_json::to_string_pretty(doc)
                    .map_err(|e| format!("Failed to serialize '{}': {}", doc.id(), e))?;
                std::fs::write(&path, payload)
                    .map_err(|e| format!("Failed to write '{}': {}", path.display(), e))?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use std::fs;
    use std::num::NonZeroUsize;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::time::{Duration, Instant};

    #[test]
    fn batch_ranges_preserve_order_and_bound_each_model_call() {
        assert_eq!(super::batch_ranges(65, 32), vec![0..32, 32..64, 64..65]);
        assert!(super::batch_ranges(0, 32).is_empty());
    }

    #[test]
    fn batch_size_must_be_positive() {
        assert!(
            super::BatchArgs::try_parse_from(["anno", "--stdin", "--batch-size", "0"]).is_err()
        );
    }

    #[test]
    fn bounded_batches_run_concurrently_without_exceeding_the_worker_cap() {
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let model = {
            let active = Arc::clone(&active);
            let peak = Arc::clone(&peak);
            anno::AnyModel::new("counting", "test model", vec![], move |_, _| {
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(current, Ordering::SeqCst);

                // Wait briefly for another worker, but never use a barrier that
                // would deadlock if parallel dispatch regressed to sequential.
                let deadline = Instant::now() + Duration::from_secs(1);
                while active.load(Ordering::SeqCst) < 2 && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(1));
                }
                active.fetch_sub(1, Ordering::SeqCst);
                Ok(Vec::new())
            })
        };
        let texts = ["a", "b", "c", "d"];

        let results = super::extract_bounded_batches(
            &model,
            &texts,
            NonZeroUsize::new(32).unwrap(),
            2,
            |_| {},
        )
        .unwrap();

        assert_eq!(results.len(), texts.len());
        assert!(
            peak.load(Ordering::SeqCst) > 1,
            "expected concurrent batches"
        );
        assert!(
            peak.load(Ordering::SeqCst) <= 2,
            "must not exceed the requested worker cap"
        );
    }

    #[test]
    fn bounded_batches_keep_order_and_individual_errors() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let model = anno::AnyModel::new("ordered", "test model", vec![], |text, _| {
            if text == "failure" {
                return Err(anno::Error::Inference("document failed".into()));
            }
            Ok(vec![anno::Entity::new(
                text,
                anno::EntityType::Person,
                0,
                text.chars().count(),
                1.0,
            )])
        });
        let completed = AtomicUsize::new(0);
        let results = super::extract_bounded_batches(
            &model,
            &["Zoë", "failure", "Grace"],
            NonZeroUsize::new(1).unwrap(),
            2,
            |count| {
                completed.fetch_add(count, Ordering::SeqCst);
            },
        )
        .unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].as_ref().unwrap()[0].text, "Zoë");
        assert!(results[1].is_err());
        assert_eq!(results[2].as_ref().unwrap()[0].text, "Grace");
        assert_eq!(completed.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn result_cache_identity_covers_every_cache_eligible_input() {
        let dir = tempfile::tempdir().unwrap();
        let identity = super::ResultCacheIdentity {
            document_id: "document-a",
            text: "Alice met Bob.",
            cli_version: "0.11.0",
            executable_fingerprint: "executable-a",
            model_name: "pattern",
            runtime_model_name: "regex",
            runtime_model_version: "1",
            link_kb: false,
        };
        let baseline = super::result_cache_path(dir.path(), &identity);

        assert!(baseline.starts_with(dir.path().join("results").join("v2")));
        assert_ne!(
            baseline,
            super::result_cache_path(
                dir.path(),
                &super::ResultCacheIdentity {
                    document_id: "document-b",
                    ..identity
                },
            ),
            "the serialized document id is result data and must not cross cache entries"
        );
        assert_ne!(
            baseline,
            super::result_cache_path(
                dir.path(),
                &super::ResultCacheIdentity {
                    model_name: "heuristic",
                    runtime_model_name: "heuristic",
                    ..identity
                },
            ),
            "CLI backend selection is execution configuration"
        );
        assert_ne!(
            baseline,
            super::result_cache_path(
                dir.path(),
                &super::ResultCacheIdentity {
                    runtime_model_version: "2",
                    ..identity
                },
            ),
            "runtime model version changes extraction output"
        );
        assert_ne!(
            baseline,
            super::result_cache_path(
                dir.path(),
                &super::ResultCacheIdentity {
                    link_kb: true,
                    ..identity
                },
            ),
            "KB linking changes the serialized document"
        );
        assert_ne!(
            baseline,
            super::result_cache_path(
                dir.path(),
                &super::ResultCacheIdentity {
                    cli_version: "0.12.0",
                    ..identity
                },
            ),
            "a CLI release can change deterministic extraction behavior"
        );
        assert_ne!(
            baseline,
            super::result_cache_path(
                dir.path(),
                &super::ResultCacheIdentity {
                    executable_fingerprint: "executable-b",
                    ..identity
                },
            ),
            "a local rebuild can change deterministic extraction behavior"
        );
    }

    #[test]
    fn cache_is_limited_to_deterministic_local_backends_without_coreference() {
        for backend in [
            super::ModelBackend::Pattern,
            super::ModelBackend::Heuristic,
            super::ModelBackend::Minimal,
        ] {
            assert!(super::cache_ineligibility_reason(backend, false).is_none());
            assert!(super::cache_ineligibility_reason(backend, true).is_some());
        }
        assert!(super::cache_ineligibility_reason(super::ModelBackend::Stacked, false).is_some());
        assert!(
            super::cache_ineligibility_reason(super::ModelBackend::UniversalNer, false).is_some()
        );
    }

    #[test]
    fn cached_extraction_preserves_exact_output() {
        use anno::Model;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("entry.json");
        let text = "Zoë: mira@example.org";
        let entities = anno::RegexNER::new().extract_entities(text, None).unwrap();
        assert!(!entities.is_empty());
        let fresh = anno::GroundedDocument::from_entity_signals("unicode", text, &entities);
        super::store_cached(&path, &fresh);
        let cached = super::try_load_cached(&path, "unicode", text).unwrap();

        assert_eq!(
            serde_json::to_value(&fresh).unwrap(),
            serde_json::to_value(&cached).unwrap(),
            "cache reuse must preserve the complete extraction, including confidence"
        );
        assert_eq!(
            super::doc_to_clean_json(&fresh, "pattern"),
            super::doc_to_clean_json(&cached, "pattern")
        );
    }

    #[test]
    fn cached_document_must_match_its_keyed_source() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("entry.json");
        let doc = anno::GroundedDocument::new("document-a", "Alice met Bob.");
        super::store_cached(&path, &doc);

        assert!(super::try_load_cached(&path, "document-a", "Alice met Bob.").is_some());
        assert!(super::try_load_cached(&path, "document-b", "Alice met Bob.").is_none());
        assert!(super::try_load_cached(&path, "document-a", "Alice met Carol.").is_none());
    }

    /// Batch should accept .html files alongside .txt and .md
    #[test]
    fn dir_scan_accepts_html_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "Alice met Bob.").unwrap();
        fs::write(dir.path().join("b.md"), "Charlie met Dave.").unwrap();
        fs::write(
            dir.path().join("c.html"),
            "<html><body><p>Eve met Frank.</p></body></html>",
        )
        .unwrap();
        fs::write(dir.path().join("d.csv"), "should,be,ignored").unwrap();

        let mut found = Vec::new();
        for entry in fs::read_dir(dir.path()).unwrap() {
            let path = entry.unwrap().path();
            if !path.is_file() {
                continue;
            }
            let ext_ok = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| {
                    matches!(
                        e,
                        "txt" | "md" | "html" | "htm" | "xhtml" | "pdf" | "rst" | "text"
                    )
                })
                .unwrap_or(false);
            if ext_ok {
                found.push(path.file_name().unwrap().to_string_lossy().to_string());
            }
        }
        found.sort();
        assert_eq!(found, vec!["a.txt", "b.md", "c.html"]);
    }

    /// HTML files in batch dir should be stripped to text via read_input_file
    #[test]
    fn html_file_stripped_in_batch() {
        let dir = tempfile::tempdir().unwrap();
        let html = r#"<!DOCTYPE html>
        <html><body>
        <nav>Navigation</nav>
        <p>Jensen Huang announced the Blackwell GPU.</p>
        <footer>Footer text</footer>
        </body></html>"#;
        fs::write(dir.path().join("test.html"), html).unwrap();

        let text =
            crate::cli::utils::read_input_file(dir.path().join("test.html").to_str().unwrap())
                .unwrap();
        assert!(text.contains("Jensen Huang"), "should extract article text");
        assert!(!text.contains("<nav>"), "should not contain raw HTML tags");
    }

    /// Batch output files should be sorted deterministically by doc_id
    #[test]
    fn output_sorted_by_doc_id() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("zulu.txt"), "Zulu text.").unwrap();
        fs::write(dir.path().join("alpha.txt"), "Alpha text.").unwrap();
        fs::write(dir.path().join("mike.txt"), "Mike text.").unwrap();

        let mut docs = Vec::new();
        for entry in fs::read_dir(dir.path()).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) == Some("txt") {
                let id = path.file_stem().unwrap().to_str().unwrap().to_string();
                let text = fs::read_to_string(&path).unwrap();
                docs.push((id, text));
            }
        }
        docs.sort_by(|a, b| a.0.cmp(&b.0));

        let ids: Vec<&str> = docs.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, vec!["alpha", "mike", "zulu"]);
    }

    /// Empty file should not cause panic
    #[test]
    fn empty_file_handled() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("empty.txt"), "").unwrap();

        let text =
            crate::cli::utils::read_input_file(dir.path().join("empty.txt").to_str().unwrap())
                .unwrap();
        assert!(text.is_empty());
    }

    /// No matching files should produce error, not silently succeed
    #[test]
    fn no_matching_files_detected() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("data.csv"), "a,b,c").unwrap();

        let mut found = Vec::new();
        for entry in fs::read_dir(dir.path()).unwrap() {
            let path = entry.unwrap().path();
            let ext_ok = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| {
                    matches!(
                        e,
                        "txt" | "md" | "html" | "htm" | "xhtml" | "pdf" | "rst" | "text"
                    )
                })
                .unwrap_or(false);
            if ext_ok {
                found.push(path);
            }
        }
        assert!(found.is_empty());
    }

    /// doc_to_clean_json produces the expected schema with correct field names
    #[test]
    fn clean_json_schema() {
        let mut doc = anno::GroundedDocument::new("test_doc", "Alice met Bob in Paris.");
        let signal = anno::Signal::new(
            anno::SignalId::ZERO,
            anno::Location::Text { start: 0, end: 5 },
            "Alice".to_string(),
            anno::TypeLabel::from("PER"),
            0.95,
        );
        doc.add_signal(signal);

        let json = super::doc_to_clean_json(&doc, "bert-onnx");
        assert_eq!(json["id"], "test_doc");
        assert_eq!(json["model"], "bert-onnx");
        assert_eq!(json["entity_count"], 1);
        assert_eq!(json["text_length"], 23);

        let entity = &json["entities"][0];
        assert_eq!(entity["text"], "Alice");
        assert_eq!(entity["type"], "PER");
        assert_eq!(entity["start"], 0);
        assert_eq!(entity["end"], 5);
        let conf = entity["confidence"].as_f64().unwrap();
        assert!(
            (conf - 0.95).abs() < 0.01,
            "confidence should be ~0.95, got {}",
            conf
        );
        assert_eq!(entity["negated"], false);

        // Should NOT contain GroundedDocument-specific fields
        assert!(json.get("signals").is_none());
        assert!(json.get("identities").is_none());
        assert!(
            json.get("text").is_none(),
            "full text not included in clean json"
        );
    }
}
