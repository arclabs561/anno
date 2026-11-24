//! CrossDoc command - Cross-document entity coalescing: cluster entities across multiple documents

#[cfg(feature = "eval")]
use std::collections::HashMap;
#[cfg(feature = "eval")]
use std::fs;
#[cfg(feature = "eval")]
use std::io::{self, BufRead};
#[cfg(feature = "eval")]
use std::path::{Path, PathBuf};

#[cfg(feature = "eval")]
use anno::coalesce::Resolver;
#[cfg(feature = "eval")]
use anno::offset::TextSpan;
#[cfg(feature = "eval")]
use anno::{Corpus, GroundedDocument};
#[cfg(feature = "eval")]
use anno_eval::cdcr::{CDCRConfig, CDCRResolver, CrossDocCluster, Document};

#[cfg(feature = "eval")]
use super::super::output::color;
use super::super::parser::{ModelBackend, OutputFormat};

#[cfg(feature = "eval")]
use glob::glob;

#[derive(Clone, Copy, Debug, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CandidateMode {
    /// Compare every pair of mentions.
    Exhaustive,
    /// Compare only pairs proposed by locality-sensitive hashing of mention text.
    Lsh,
}

/// Cross-document entity coalescing: cluster entities across multiple documents
#[derive(clap::Parser, Debug)]
pub struct CrossDocArgs {
    /// Directory containing text files to process (optional if --import is used)
    #[arg(value_name = "DIR")]
    pub directory: Option<String>,

    /// Model backend to use for entity extraction
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Similarity threshold for clustering (0.0-1.0)
    #[arg(short, long, default_value = "0.6")]
    pub threshold: f64,

    /// Raw directory candidate generation (default: exhaustive up to 100 documents, then lsh)
    #[arg(long, value_enum, conflicts_with_all = ["import", "stdin"])]
    pub candidates: Option<CandidateMode>,

    /// Require entity type match for clustering
    #[arg(long)]
    pub require_type_match: bool,

    /// Output format
    #[arg(short, long, default_value = "json")]
    pub format: OutputFormat,

    /// Import pre-processed GroundedDocument JSON file(s) instead of processing directory
    #[arg(long, value_name = "PATH")]
    pub import: Vec<String>,

    /// Read input from stdin (JSONL format, one GroundedDocument per line)
    #[arg(long)]
    pub stdin: bool,

    /// File extensions to process (comma-separated)
    #[arg(long, default_value = "txt,md")]
    pub extensions: String,

    /// Recursively search subdirectories
    #[arg(short, long)]
    pub recursive: bool,

    /// Minimum cluster size to include in output
    #[arg(long, default_value = "1")]
    pub min_cluster_size: usize,

    /// Filter to only cross-document clusters (appears in 2+ docs)
    #[arg(long)]
    pub cross_doc_only: bool,

    /// Filter by entity type (repeatable, e.g., --type PER --type ORG)
    #[arg(long = "type", value_name = "TYPE")]
    pub entity_types: Vec<String>,

    /// Maximum number of clusters to output (0 = unlimited)
    #[arg(long, default_value = "0")]
    pub max_clusters: usize,

    /// Output file path (if not specified, prints to stdout)
    #[arg(short = 'o', long)]
    pub output: Option<String>,

    /// Show progress and detailed cluster information
    #[arg(short, long)]
    pub verbose: bool,
}

#[cfg(feature = "eval")]
struct ContextSnippet<'a> {
    before_marker: &'static str,
    before: &'a str,
    entity_text: &'a str,
    after: &'a str,
    after_marker: &'static str,
}

#[cfg(feature = "eval")]
fn context_snippet<'a>(
    text: &'a str,
    entity_start_char: usize,
    entity_end_char: usize,
    context_window_chars: usize,
) -> ContextSnippet<'a> {
    let text_char_count = text.chars().count();
    let start = entity_start_char.min(text_char_count);
    let mut end = entity_end_char.min(text_char_count);
    if end < start {
        end = start;
    }

    let context_start = start.saturating_sub(context_window_chars);
    let context_end = (end + context_window_chars).min(text_char_count);

    let before = TextSpan::from_chars(text, context_start, start).extract(text);
    let entity_text = TextSpan::from_chars(text, start, end).extract(text);
    let after = TextSpan::from_chars(text, end, context_end).extract(text);

    ContextSnippet {
        // Only show ellipses when the context is clipped, not merely because there is
        // non-empty text before/after the entity.
        before_marker: if context_start > 0 { "..." } else { "" },
        before,
        entity_text,
        after,
        after_marker: if context_end < text_char_count {
            "..."
        } else {
            ""
        },
    }
}

/// Build the import view from current track assignments. Entity indices in
/// CLI output refer to the signal's position in the document's flat projection.
#[cfg(feature = "eval")]
pub(super) fn corpus_clusters(corpus: &Corpus) -> Vec<CrossDocCluster> {
    #[derive(Hash, PartialEq, Eq)]
    enum Key<'a> {
        Kb(&'a str, &'a str),
        Shared(anno::IdentityId),
        Local(&'a str, anno::IdentityId),
    }
    let mut indices = HashMap::new();
    let mut clusters: Vec<CrossDocCluster> = Vec::new();
    let mut documents: Vec<_> = corpus.documents().collect();
    documents.sort_by_key(|doc| doc.id());

    // The output format uses numeric IDs. Reserve corpus IDs and assign local-only
    // keys unused report IDs in deterministic order.
    let mut report_ids = (0u64..).filter(|id| corpus.get_identity((*id).into()).is_none());
    for doc in documents {
        let mut tracks: Vec<_> = doc.tracks().collect();
        tracks.sort_by_key(|track| track.id);
        let mut track_indices = HashMap::new();
        for track in tracks {
            let Some(identity) = doc.identity_for_track(track.id) else {
                continue;
            };
            let shared = corpus.is_shared_identity(identity);
            let key = match identity.qualified_kb_ref() {
                Some((namespace, accession)) => Key::Kb(namespace, accession),
                None if shared => Key::Shared(identity.id),
                None => Key::Local(doc.id(), identity.id),
            };
            let index = *indices.entry(key).or_insert_with(|| {
                let id = if shared {
                    identity.id.get()
                } else {
                    report_ids.next().expect("available report ID")
                };
                let mut cluster = CrossDocCluster::new(id, &identity.canonical_name);
                cluster.kb_id = identity.kb_id.clone();
                cluster.kb_name = identity.kb_name.clone();
                cluster.confidence = identity.confidence.value();
                cluster.minimum_pair_similarity = identity.minimum_pair_similarity;
                cluster.entity_type = identity
                    .entity_type
                    .as_ref()
                    .map(|label| label.to_entity_type());
                clusters.push(cluster);
                clusters.len() - 1
            });
            // A pairwise score only describes its original membership. A report-level
            // group joined through matching KB IDs has no cross-document measurement.
            let cluster = &mut clusters[index];
            if !shared || cluster.id != identity.id.get() {
                cluster.minimum_pair_similarity = None;
                cluster.confidence = 0.0;
            }
            track_indices.insert(track.id, index);
        }
        for (index, signal) in doc.signals().iter().enumerate() {
            if let Some(cluster_idx) = doc
                .track_for_signal(signal.id)
                .and_then(|track| track_indices.get(&track.id))
            {
                clusters[*cluster_idx].add_mention(doc.id(), index);
            }
        }
    }
    clusters
}

#[cfg(feature = "eval")]
/// Execute the crossdoc command.
pub fn run(args: CrossDocArgs) -> Result<(), String> {
    if args.verbose {
        if let Some(ref dir) = args.directory {
            eprintln!("Scanning directory: {}", dir);
        }
    }

    // Collect text files
    let extensions: Vec<&str> = args.extensions.split(',').map(|s| s.trim()).collect();
    let mut files = Vec::new();

    fn collect_files(
        dir: &Path,
        extensions: &[&str],
        recursive: bool,
        files: &mut Vec<PathBuf>,
    ) -> Result<(), String> {
        let entries = fs::read_dir(dir)
            .map_err(|e| format!("Failed to read directory {}: {}", dir.display(), e))?;

        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();

            if path.is_dir() && recursive {
                collect_files(&path, extensions, recursive, files)?;
            } else if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    if extensions.iter().any(|&e| e == ext_str) {
                        files.push(path);
                    }
                }
            }
        }
        Ok(())
    }

    // Check if import mode is enabled - use Corpus for better architecture
    let mut doc_paths: HashMap<String, String> = HashMap::new(); // doc_id -> file_path
    let mut use_corpus = false;
    let mut corpus = Corpus::new();
    let mut documents: Vec<Document> = Vec::new(); // Only used in normal mode, but declared here for output formatting
    let candidate_mode;

    let clusters_from_corpus: Option<Vec<CrossDocCluster>> = if !args.import.is_empty()
        || args.stdin
    {
        // Import mode: use Corpus for proper inter-doc coref with GroundedDocuments
        use_corpus = true;
        let mut import_files = Vec::new();

        if args.stdin {
            // Read from stdin (JSONL format)
            if args.verbose {
                eprintln!("Reading GroundedDocuments from stdin (JSONL format)...");
            }
            let stdin = io::stdin();
            let reader = stdin.lock();
            for (line_num, line) in reader.lines().enumerate() {
                let line =
                    line.map_err(|e| format!("Failed to read stdin line {}: {}", line_num + 1, e))?;
                if line.trim().is_empty() {
                    continue;
                }
                let doc: GroundedDocument = serde_json::from_str(&line)
                    .map_err(|e| format!("Failed to parse stdin line {}: {}", line_num + 1, e))?;
                if args.verbose {
                    eprintln!(
                        "  Imported {} signals, {} tracks from stdin line {}",
                        doc.signals().len(),
                        doc.tracks_map().len(),
                        line_num + 1
                    );
                }
                let id = corpus.add_document(doc);
                doc_paths.insert(id, format!("stdin:{}", line_num + 1));
            }
        } else {
            // Collect files from import paths (support glob patterns)
            for import_pattern in &args.import {
                if import_pattern == "-" {
                    // Special case: read from stdin
                    let stdin = io::stdin();
                    let reader = stdin.lock();
                    for (line_num, line) in reader.lines().enumerate() {
                        let line = line.map_err(|e| {
                            format!("Failed to read stdin line {}: {}", line_num + 1, e)
                        })?;
                        if line.trim().is_empty() {
                            continue;
                        }
                        let doc: GroundedDocument = serde_json::from_str(&line).map_err(|e| {
                            format!("Failed to parse stdin line {}: {}", line_num + 1, e)
                        })?;
                        let doc_id = doc.id().to_owned();
                        corpus.add_document(doc);
                        doc_paths.insert(doc_id.clone(), format!("stdin:{}", line_num + 1));
                        if args.verbose {
                            if let Some(d) = corpus.get_document(&doc_id) {
                                let stats = d.stats();
                                eprintln!(
                                    "  Imported {} signals, {} tracks from stdin line {}",
                                    stats.signal_count,
                                    stats.track_count,
                                    line_num + 1
                                );
                            }
                        }
                    }
                } else if import_pattern.contains('*')
                    || import_pattern.contains('?')
                    || import_pattern.contains('[')
                {
                    // Glob pattern
                    if args.verbose {
                        eprintln!("Expanding glob pattern: {}", import_pattern);
                    }
                    let matches = glob(import_pattern)
                        .map_err(|e| format!("Invalid glob pattern '{}': {}", import_pattern, e))?;
                    for entry in matches {
                        match entry {
                            Ok(path) => {
                                if path.is_file() {
                                    import_files.push(path);
                                }
                            }
                            Err(e) => {
                                if args.verbose {
                                    eprintln!("  Warning: glob match error: {}", e);
                                }
                            }
                        }
                    }
                } else {
                    // Regular file path
                    let path = Path::new(import_pattern);
                    if path.exists() && path.is_file() {
                        import_files.push(path.to_path_buf());
                    } else {
                        return Err(format!("Import file not found: {}", import_pattern));
                    }
                }
            }

            // Load all collected files
            if args.verbose && !import_files.is_empty() {
                eprintln!(
                    "Importing {} GroundedDocument file(s)...",
                    import_files.len()
                );
            }

            for (idx, file_path) in import_files.iter().enumerate() {
                if args.verbose {
                    eprint!(
                        "\r  Loading {}/{}: {}...",
                        idx + 1,
                        import_files.len(),
                        file_path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("?")
                    );
                    use std::io::Write;
                    io::stderr().flush().ok();
                }

                let json_content = fs::read_to_string(file_path).map_err(|e| {
                    format!(
                        "Failed to read import file '{}': {}",
                        file_path.display(),
                        e
                    )
                })?;

                let doc: GroundedDocument = serde_json::from_str(&json_content).map_err(|e| {
                    format!(
                        "Failed to parse GroundedDocument JSON from '{}': {}",
                        file_path.display(),
                        e
                    )
                })?;

                let entity_count = doc.signals().len();
                let id = corpus.add_document(doc);
                doc_paths.insert(id, file_path.display().to_string());

                if args.verbose {
                    eprintln!(
                        "\r  Loaded {} entities from {}",
                        entity_count,
                        file_path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("?")
                    );
                }
            }
        }

        let doc_count = corpus.documents().count();
        if doc_count == 0 {
            return Err(
                "No GroundedDocuments imported. Check import paths or stdin input.".to_string(),
            );
        }

        if args.verbose {
            let total_signals: usize = corpus.documents().map(|d| d.stats().signal_count).sum();
            let total_tracks: usize = corpus.documents().map(|d| d.stats().track_count).sum();
            eprintln!(
                "Imported {} documents with {} signals, {} tracks",
                doc_count, total_signals, total_tracks
            );
        }

        // Use Corpus for inter-doc coref resolution (much cleaner than CDCR conversion)
        if args.verbose {
            eprintln!(
                "Resolving inter-document coreference (threshold: {}, require_type_match: {})...",
                args.threshold, args.require_type_match
            );
        }

        candidate_mode = CandidateMode::Exhaustive;
        let resolver = Resolver::new()
            .with_threshold(args.threshold as f32)
            .require_type_match(args.require_type_match);
        let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);

        if args.verbose {
            eprintln!(
                "Resolved {} identities from inter-doc coref",
                identity_ids.len()
            );
        }

        let mut clusters = corpus_clusters(&corpus);
        clusters.retain(|cluster| {
            cluster.len() >= args.min_cluster_size
                && (!args.cross_doc_only || cluster.doc_count() > 1)
                && (args.entity_types.is_empty()
                    || cluster.entity_type.as_ref().is_some_and(|entity_type| {
                        let label = entity_type.as_label().to_uppercase();
                        args.entity_types
                            .iter()
                            .any(|query| query.to_uppercase() == label)
                    }))
        });

        // Sort by importance
        clusters.sort_by(|a, b| {
            b.doc_count()
                .cmp(&a.doc_count())
                .then_with(|| b.len().cmp(&a.len()))
                .then_with(|| b.canonical_name.cmp(&a.canonical_name))
        });

        // Limit output
        if args.max_clusters > 0 {
            clusters.truncate(args.max_clusters);
        }

        Some(clusters)
    } else {
        // Normal mode: extract from text files, use CDCRResolver (legacy path)
        // Normal mode: extract entities from text files
        // Directory is required in normal mode
        let dir = if let Some(ref dir_str) = args.directory {
            Path::new(dir_str)
        } else {
            return Err("Directory is required when --import is not used. Use: anno cross-doc <DIR> or anno cross-doc --import <FILE>".to_string());
        };

        collect_files(dir, &extensions, args.recursive, &mut files)?;

        if files.is_empty() {
            return Err(format!(
                "No files found with extensions: {}",
                args.extensions
            ));
        }

        files.sort();
        let mut doc_ids = Vec::with_capacity(files.len());
        for (idx, path) in files.iter().enumerate() {
            let id = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_owned)
                .unwrap_or_else(|| format!("doc{idx}"));
            super::super::utils::register_document_source(
                &mut doc_paths,
                &id,
                path.display().to_string(),
            )?;
            doc_ids.push(id);
        }

        let model = args.model.create_model()?;
        if args.verbose {
            eprintln!("Found {} files", files.len());
            eprintln!("Extracting entities...");
        }

        // NOTE: Currently operates on raw entities (Level 1: Signal)
        // With --import, can use Level 2 (Tracks) and Level 3 (Identities) from pre-processed docs
        let total_files = files.len();
        for (idx, (file_path, doc_id)) in files.iter().zip(&doc_ids).enumerate() {
            if args.verbose {
                eprint!(
                    "\r  Processing {}/{}: {}...",
                    idx + 1,
                    total_files,
                    file_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("?")
                );
                use std::io::Write;
                io::stderr().flush().ok();
            }

            let text = fs::read_to_string(file_path)
                .map_err(|e| format!("Failed to read {}: {}", file_path.display(), e))?;

            let entities = model
                .extract_entities(&text, None)
                .map_err(|e| format!("Failed to extract entities from {}: {}", doc_id, e))?;

            documents.push(Document::new(doc_id, &text).with_entities(entities));
        }

        if args.verbose {
            eprintln!("\r  Processed {} files successfully", total_files);
        }

        if args.verbose {
            eprintln!(
                "Clustering {} entities across {} documents...",
                documents
                    .iter()
                    .map(|doc| doc.entities.len())
                    .sum::<usize>(),
                documents.len()
            );
        }

        // Configure and run cross-doc coref using CDCRResolver (for raw text files)
        candidate_mode = args.candidates.unwrap_or(if documents.len() > 100 {
            CandidateMode::Lsh
        } else {
            CandidateMode::Exhaustive
        });
        let config = CDCRConfig {
            min_similarity: args.threshold,
            require_type_match: args.require_type_match,
            use_lsh: matches!(candidate_mode, CandidateMode::Lsh),
            ..Default::default()
        };

        let resolver = CDCRResolver::with_config(config);
        let clusters = resolver.resolve(&documents);

        // Filter clusters
        let mut filtered_clusters: Vec<_> = clusters
            .into_iter()
            .filter(|c| {
                // Minimum size filter
                if c.len() < args.min_cluster_size {
                    return false;
                }
                // Cross-doc only filter
                if args.cross_doc_only && c.doc_count() <= 1 {
                    return false;
                }
                // Entity type filter
                if !args.entity_types.is_empty() {
                    if let Some(ref entity_type) = c.entity_type {
                        let type_label = entity_type.as_label().to_uppercase();
                        if !args
                            .entity_types
                            .iter()
                            .any(|t| t.to_uppercase() == type_label)
                        {
                            return false;
                        }
                    } else {
                        return false; // Skip clusters without type if filtering by type
                    }
                }
                true
            })
            .collect();

        // Sort by importance
        filtered_clusters.sort_by(|a, b| {
            b.doc_count()
                .cmp(&a.doc_count())
                .then_with(|| b.len().cmp(&a.len()))
                .then_with(|| b.canonical_name.cmp(&a.canonical_name))
        });

        // Limit output
        let clusters: Vec<_> = if args.max_clusters > 0 {
            filtered_clusters
                .into_iter()
                .take(args.max_clusters)
                .collect()
        } else {
            filtered_clusters
        };

        // Store clusters for later use
        Some(clusters)
    };

    // Use clusters from Corpus if available, otherwise use CDCR clusters
    let final_clusters: Vec<CrossDocCluster> = clusters_from_corpus
        .ok_or_else(|| "No clusters generated. This should not happen.".to_string())?;

    let (doc_count, total_entities) = if use_corpus {
        (
            corpus.documents().count(),
            corpus
                .documents()
                .map(|doc| doc.signals().len())
                .sum::<usize>(),
        )
    } else {
        (
            documents.len(),
            documents
                .iter()
                .map(|doc| doc.entities.len())
                .sum::<usize>(),
        )
    };

    // Prepare output
    let output_text = match args.format {
        OutputFormat::Json => {
            // Enhanced JSON with metadata
            let mut output = serde_json::Map::new();
            output.insert("metadata".to_string(), serde_json::json!({
                "resolver": if use_corpus { "grounded_tracks" } else { "flat_mentions" },
                "candidate_mode": candidate_mode,
                "documents_processed": doc_count,
                "total_entities": total_entities,
                "clusters_found": final_clusters.len(),
                "cross_document_clusters": final_clusters.iter().filter(|c| c.doc_count() > 1).count(),
                "threshold": args.threshold,
                "require_type_match": args.require_type_match,
                "filters": {
                "min_cluster_size": args.min_cluster_size,
                "cross_doc_only": args.cross_doc_only,
                "entity_types": args.entity_types,
                "max_clusters": args.max_clusters,
                    }
                }));
            output.insert(
                "clusters".to_string(),
                serde_json::to_value(&final_clusters)
                    .map_err(|e| format!("Failed to serialize clusters: {}", e))?,
            );
            serde_json::to_string_pretty(&output)
                .map_err(|e| format!("Failed to serialize output: {}", e))?
        }
        OutputFormat::Jsonl => {
            let mut lines = Vec::new();
            for cluster in &final_clusters {
                let json = serde_json::to_string(cluster)
                    .map_err(|e| format!("Failed to serialize cluster: {}", e))?;
                lines.push(json);
            }
            lines.join("\n")
        }
        OutputFormat::Tree => {
            // Build document index for O(1) lookups (only needed for CDCR path)
            let doc_index: HashMap<&str, &Document> = if !use_corpus {
                documents.iter().map(|d| (d.id.as_str(), d)).collect()
            } else {
                HashMap::new() // Not needed for Corpus path
            };

            let mut output = String::new();
            // Sort clusters by importance (doc count, then mention count)
            let mut sorted_clusters: Vec<_> = final_clusters.iter().collect();
            sorted_clusters.sort_by(|a, b| {
                b.doc_count()
                    .cmp(&a.doc_count())
                    .then_with(|| b.len().cmp(&a.len()))
                    .then_with(|| b.canonical_name.cmp(&a.canonical_name))
            });

            // Simplified header - less visual noise
            output.push_str(&format!(
                "{}\n",
                color("1;36", "Cross-Document Entity Coalescing Results")
            ));
            output.push('\n');

            // Summary header
            let cross_doc_clusters = final_clusters.iter().filter(|c| c.doc_count() > 1).count();
            let singleton_clusters = final_clusters.len() - cross_doc_clusters;

            output.push_str(&format!("{}\n", color("1;33", "Summary")));
            output.push_str(&format!("  Documents: {}\n", doc_count));
            output.push_str(&format!("  Entities: {}\n", total_entities));
            output.push_str(&format!(
                "  Clusters: {} ({} cross-doc, {} singleton)\n",
                final_clusters.len(),
                color("32", &cross_doc_clusters.to_string()),
                singleton_clusters
            ));
            if !args.entity_types.is_empty() {
                output.push_str(&format!(
                    "  Filtered by: {}\n",
                    args.entity_types.join(", ")
                ));
            }
            output.push('\n');

            // Entity type breakdown
            let mut type_counts: HashMap<String, usize> = HashMap::new();
            for cluster in &final_clusters {
                if let Some(ref entity_type) = cluster.entity_type {
                    let label: &str = entity_type.as_label();
                    *type_counts.entry(label.to_string()).or_insert(0) += 1;
                }
            }
            if !type_counts.is_empty() {
                output.push_str(&format!("{}\n", color("1;33", "Entity Types")));
                let mut type_vec: Vec<_> = type_counts.iter().collect();
                type_vec.sort_by(|a, b| b.1.cmp(a.1));
                for (etype, count) in type_vec {
                    output.push_str(&format!("  {}: {}\n", etype, count));
                }
                output.push('\n');
            }

            output.push_str(&format!("{}\n", color("1;36", "Clusters")));
            output.push('\n');

            // Determine display limit
            let display_limit = if args.max_clusters > 0 {
                args.max_clusters
            } else if !args.verbose {
                50 // Default limit for non-verbose
            } else {
                sorted_clusters.len() // No limit in verbose mode
            };

            for cluster in sorted_clusters.iter().take(display_limit) {
                let is_cross_doc = cluster.doc_count() > 1;
                let prefix = if is_cross_doc {
                    color("32", "●")
                } else {
                    color("90", "○")
                };

                // Cluster header: prefix + name + type
                let mut header = format!("{} {}", prefix, color("1", &cluster.canonical_name));
                if let Some(ref entity_type) = cluster.entity_type {
                    let label: &str = entity_type.as_label();
                    header.push_str(&format!(" ({})", label));
                }
                if is_cross_doc {
                    header.push_str(&format!(" {}", color("32", "[cross-doc]")));
                }
                output.push_str(&format!("{}\n", header));

                // Metadata line
                let mut meta_parts = Vec::new();
                meta_parts.push(format!("{} mentions", cluster.len()));
                meta_parts.push(format!(
                    "{} doc{}",
                    cluster.doc_count(),
                    if cluster.doc_count() == 1 { "" } else { "s" }
                ));
                if let Some(similarity) = cluster.minimum_pair_similarity {
                    meta_parts.push(format!("min pair similarity: {similarity:.2}"));
                } else if !use_corpus && cluster.confidence < 1.0 {
                    meta_parts.push(format!("conf: {:.2}", cluster.confidence));
                }
                output.push_str(&format!("  {}\n", meta_parts.join(" • ")));

                if let Some(ref kb_id) = cluster.kb_id {
                    let reference = cluster
                        .kb_name
                        .as_ref()
                        .map_or_else(|| kb_id.clone(), |name| format!("{name}/{kb_id}"));
                    output.push_str(&format!("  KB: {}\n", color("36", &reference)));
                }

                // Show documents with paths (truncate if too many)
                if !cluster.documents.is_empty() {
                    let max_docs_to_show = if args.verbose { 20 } else { 5 };
                    let doc_list: Vec<String> = cluster
                        .documents
                        .iter()
                        .take(max_docs_to_show)
                        .map(|doc_id: &String| {
                            let path = doc_paths
                                .get(doc_id)
                                .map(|p| format!("{} ({})", doc_id, p))
                                .unwrap_or_else(|| doc_id.clone());
                            color("36", &path)
                        })
                        .collect();
                    let doc_count = cluster.documents.len();
                    if doc_count > max_docs_to_show {
                        output.push_str(&format!(
                            "  Docs: {} (and {} more)\n",
                            doc_list.join(", "),
                            doc_count - max_docs_to_show
                        ));
                    } else {
                        output.push_str(&format!("  Docs: {}\n", doc_list.join(", ")));
                    }
                }

                // Show mentions - always show sample, verbose adds context
                if !cluster.mentions.is_empty() {
                    let sample_size = if args.verbose {
                        cluster.mentions.len()
                    } else {
                        cluster.mentions.len().min(3)
                    };

                    for (doc_id, entity_idx) in cluster.mentions.iter().take(sample_size) {
                        let mention = if use_corpus {
                            corpus.get_document(doc_id).and_then(|doc| {
                                let signal = doc.signals().get(*entity_idx)?;
                                let (start, end) = signal.text_offsets()?;
                                Some((doc.text(), signal.surface(), start, end))
                            })
                        } else {
                            doc_index.get(doc_id.as_str()).and_then(|doc| {
                                let entity = doc.entities.get(*entity_idx)?;
                                Some((
                                    doc.text.as_str(),
                                    entity.text.as_str(),
                                    entity.start(),
                                    entity.end(),
                                ))
                            })
                        };
                        let Some((text, surface, start, end)) = mention else {
                            continue;
                        };
                        if args.verbose {
                            let snippet = context_snippet(text, start, end, 50);
                            output.push_str(&format!(
                                "    {} {}: {}{}[{}]{}{}\n",
                                color("90", "•"),
                                color("36", doc_id),
                                snippet.before_marker,
                                snippet.before,
                                color("1;32", snippet.entity_text),
                                snippet.after,
                                snippet.after_marker
                            ));
                        } else {
                            output.push_str(&format!(
                                "    {} {}: \"{}\"\n",
                                color("90", "•"),
                                color("36", doc_id),
                                surface
                            ));
                        }
                    }

                    if cluster.mentions.len() > sample_size {
                        output.push_str(&format!(
                            "    {} ... and {} more\n",
                            color("90", "•"),
                            cluster.mentions.len() - sample_size
                        ));
                    }
                }

                output.push('\n');
            }

            // Show limit message if applicable
            if sorted_clusters.len() > display_limit {
                let more_count = sorted_clusters.len() - display_limit;
                let message = format!(
                    "... {} more cluster{} (use --max-clusters {} or --verbose to see all)",
                    more_count,
                    if more_count == 1 { "" } else { "s" },
                    sorted_clusters.len()
                );
                output.push_str(&format!("{}\n", color("90", &message)));
            }
            output
        }
        OutputFormat::Summary => {
            let cross_doc_clusters = final_clusters.iter().filter(|c| c.doc_count() > 1).count();
            let singleton_clusters = final_clusters.len() - cross_doc_clusters;
            let avg_cluster_size = if final_clusters.is_empty() {
                0.0
            } else {
                final_clusters.iter().map(|c| c.len()).sum::<usize>() as f64
                    / final_clusters.len() as f64
            };
            let max_cluster_size = final_clusters.iter().map(|c| c.len()).max().unwrap_or(0);
            let max_doc_count = final_clusters
                .iter()
                .map(|c| c.doc_count())
                .max()
                .unwrap_or(0);

            // Entity type distribution
            let mut type_counts: HashMap<String, usize> = HashMap::new();
            for cluster in &final_clusters {
                if let Some(ref entity_type) = cluster.entity_type {
                    let label: &str = entity_type.as_label();
                    *type_counts.entry(label.to_string()).or_insert(0) += 1;
                }
            }

            let mut output = String::new();
            output.push_str(&format!(
                "{}\n",
                color(
                    "1;36",
                    "═══════════════════════════════════════════════════════════"
                )
            ));
            output.push_str(&format!(
                "{}\n",
                color("1;36", "  Cross-Document Entity Coalescing Summary")
            ));
            output.push_str(&format!(
                "{}\n",
                color(
                    "1;36",
                    "═══════════════════════════════════════════════════════════"
                )
            ));
            output.push('\n');
            output.push_str(&format!("{}\n", color("1;33", "Document Statistics:")));
            output.push_str(&format!("  Documents processed: {}\n", doc_count));
            output.push_str(&format!("  Total entities extracted: {}\n", total_entities));
            output.push_str(&format!(
                "  Average entities per document: {:.1}\n",
                if doc_count == 0 {
                    0.0
                } else {
                    total_entities as f64 / doc_count as f64
                }
            ));
            output.push('\n');
            output.push_str(&format!("{}\n", color("1;33", "Cluster Statistics:")));
            output.push_str(&format!("  Total clusters: {}\n", final_clusters.len()));
            output.push_str(&format!(
                "  Cross-document clusters: {} ({:.1}%)\n",
                cross_doc_clusters,
                if final_clusters.is_empty() {
                    0.0
                } else {
                    cross_doc_clusters as f64 / final_clusters.len() as f64 * 100.0
                }
            ));
            output.push_str(&format!("  Singleton clusters: {}\n", singleton_clusters));
            output.push_str(&format!(
                "  Average cluster size: {:.2} mentions\n",
                avg_cluster_size
            ));
            output.push_str(&format!(
                "  Largest cluster: {} mentions\n",
                max_cluster_size
            ));
            output.push_str(&format!(
                "  Most documents per cluster: {}\n",
                max_doc_count
            ));
            output.push('\n');
            if !type_counts.is_empty() {
                output.push_str(&format!("{}\n", color("1;33", "Entity Type Distribution:")));
                let mut type_vec: Vec<_> = type_counts.iter().collect();
                type_vec.sort_by(|a, b| b.1.cmp(a.1));
                for (etype, count) in type_vec {
                    let percentage = if final_clusters.is_empty() {
                        0.0
                    } else {
                        *count as f64 / final_clusters.len() as f64 * 100.0
                    };
                    output.push_str(&format!("  {}: {} ({:.1}%)\n", etype, count, percentage));
                }
            }
            output
        }
        other => {
            return Err(format!("Format '{:?}' not supported for cross-doc command. Use: json, jsonl, tree, or summary.", other));
        }
    };

    // Write output to file or stdout
    if let Some(output_path) = args.output {
        fs::write(&output_path, &output_text)
            .map_err(|e| format!("Failed to write output to {}: {}", output_path, e))?;
        if args.verbose {
            eprintln!("Output written to: {}", output_path);
        }
    } else {
        print!("{}", output_text);
    }

    Ok(())
}

#[cfg(all(test, feature = "eval"))]
mod tests {
    use super::context_snippet;
    use anno::offset::TextSpan;

    #[test]
    fn imported_clusters_use_flat_indices_once_after_enrichment() {
        use super::{corpus_clusters, Corpus, GroundedDocument, Resolver};
        use anno::{Location, Signal};

        let mut doc = GroundedDocument::new("d", "x Ada Ada");
        doc.add_signal(Signal::new(0, Location::text(0, 1), "x", "OTHER", 0.5));
        let first = doc.add_signal(Signal::new(0, Location::text(2, 5), "Ada", "PER", 0.9));
        let second = doc.add_signal(Signal::new(0, Location::text(6, 9), "Ada", "PER", 0.9));
        let track = doc
            .create_track_from_signals("Ada", &[first, second])
            .unwrap();
        let track_ref = doc.track_ref(track).unwrap();
        let mut corpus = Corpus::new();
        corpus.add_document(doc);
        Resolver::new().resolve_inter_doc_coref(&mut corpus, None, None);
        corpus
            .link_track_to_kb(&track_ref, "catalog", "1", "Ada")
            .unwrap();

        let clusters = corpus_clusters(&corpus);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].documents, ["d"]);
        assert_eq!(clusters[0].kb_id.as_deref(), Some("1"));
        assert_eq!(clusters[0].mentions, [("d".into(), 1), ("d".into(), 2)]);
    }

    fn document_with_mention(id: &str, name: &str) -> (anno::GroundedDocument, anno::TrackRef) {
        use anno::{GroundedDocument, Location, Signal};
        let mut doc = GroundedDocument::new(id, name);
        let span = Location::text(0, name.chars().count());
        let mention = doc.add_signal(Signal::new(0, span, name, "PER", 1.0));
        let track = doc.create_track_from_signals(name, &[mention]).unwrap();
        let member = doc.track_ref(track).unwrap();
        (doc, member)
    }

    fn linked_document(id: &str, namespace: &str) -> anno::GroundedDocument {
        let (mut doc, member) = document_with_mention(id, "Ada");
        let identity = doc.add_identity(anno::Identity::from_kb(0, "Ada", namespace, "42"));
        doc.link_track_to_identity(member.track_id, identity);
        doc
    }

    #[test]
    fn reviewed_reassignment_splits_the_catalog_report_and_preserves_coreference() {
        use super::{corpus_clusters, Corpus, Resolver};
        let mut corpus = Corpus::new();
        let (doc_a, a) = document_with_mention("a", "Orr");
        let (doc_b, b) = document_with_mention("b", "Orr");
        corpus.add_document(doc_a);
        corpus.add_document(doc_b);
        Resolver::new().resolve_inter_doc_coref(&mut corpus, None, None);
        let shared = corpus
            .link_track_to_kb(&a, "fixture", "old", "Orr")
            .unwrap();

        let (doc_c, c) = document_with_mention("c", "Orr");
        corpus.add_document(doc_c);
        let independent = corpus
            .link_track_to_kb(&c, "fixture", "old", "Orr")
            .unwrap();
        assert_ne!(shared, independent);
        let members = [a.clone(), b];
        let partition = corpus.current_coref_clusters();
        assert_eq!(partition, [(shared, members.to_vec())]);

        let before = corpus_clusters(&corpus);
        assert_eq!(before.len(), 1);
        assert_eq!(before[0].kb_name.as_deref(), Some("fixture"));
        assert_eq!(before[0].kb_id.as_deref(), Some("old"));
        assert_eq!(
            before[0].mentions,
            [("a".into(), 0), ("b".into(), 0), ("c".into(), 0)]
        );
        assert_eq!(before[0].minimum_pair_similarity, None);

        let preview = corpus
            .preview_kb_assignment(&a, "fixture", "new", "Orr")
            .unwrap();
        assert_eq!(preview.assignment().affected_tracks, members);
        let receipt = corpus.apply_kb_assignment(preview).unwrap();
        assert_eq!(receipt.identity_id, shared);
        assert_eq!(receipt.assignment.affected_tracks, members);
        assert_eq!(corpus.current_coref_clusters(), partition);

        let after = corpus_clusters(&corpus);
        assert_eq!(after.len(), 2);
        let group = |key| {
            after
                .iter()
                .find(|cluster| {
                    cluster.kb_name.as_deref() == Some("fixture")
                        && cluster.kb_id.as_deref() == Some(key)
                })
                .unwrap()
        };
        assert_eq!(group("new").mentions, [("a".into(), 0), ("b".into(), 0)]);
        assert_eq!(group("old").mentions, [("c".into(), 0)]);
    }

    #[test]
    fn import_groups_only_qualified_kb_links_across_documents() {
        use super::{corpus_clusters, Corpus, Resolver};
        let mut corpus = Corpus::new();
        for (id, namespace) in [
            ("a", "foo"),
            ("b", "foo"),
            ("c", "bar"),
            ("d", "unknown"),
            ("e", "unknown"),
        ] {
            corpus.add_document(linked_document(id, namespace));
        }
        Resolver::new().resolve_inter_doc_coref(&mut corpus, None, None);
        assert!(corpus.current_coref_clusters().is_empty());
        let clusters = corpus_clusters(&corpus);
        assert_eq!(clusters.len(), 4);
        assert_eq!(clusters[0].kb_name.as_deref(), Some("foo"));
        assert_eq!(clusters[0].mentions, [("a".into(), 0), ("b".into(), 0)]);
        assert_eq!(clusters[0].minimum_pair_similarity, None);
        assert_eq!(clusters[1].kb_name.as_deref(), Some("bar"));
        for (cluster, doc) in clusters[1..].iter().zip(["c", "d", "e"]) {
            assert_eq!(cluster.documents, [doc]);
        }
        assert_eq!(
            clusters
                .iter()
                .map(|cluster| cluster.id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            4
        );
    }

    #[test]
    fn summary_uses_imported_document_and_mention_counts() {
        use clap::Parser;
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("doc.json");
        let output = dir.path().join("summary.txt");
        std::fs::write(
            &input,
            serde_json::to_vec(&linked_document("doc", "test")).unwrap(),
        )
        .unwrap();
        let args = super::CrossDocArgs::parse_from([
            "cross-doc",
            "--model",
            "pattern",
            "--format",
            "summary",
            "--import",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ]);
        super::run(args).unwrap();
        let summary = std::fs::read_to_string(output).unwrap();
        assert!(summary.contains("Documents processed: 1"));
        assert!(summary.contains("Total entities extracted: 1"));
        assert!(summary.contains("Average entities per document: 1.0"));
    }

    #[test]
    fn test_context_snippet_uses_char_offsets_and_is_unicode_safe() {
        let text = "🎉 東京に行った。Barack Obama visited São Paulo. التقى محمد في الرياض.";
        let needle = "São Paulo";
        let start_byte = text.find(needle).expect("needle should exist");
        let end_byte = start_byte + needle.len();
        let span = TextSpan::from_bytes(text, start_byte, end_byte);

        // Ensure our offsets are truly character offsets.
        assert_eq!(
            TextSpan::from_chars(text, span.char_start, span.char_end).extract(text),
            needle
        );

        let snippet = context_snippet(text, span.char_start, span.char_end, 5);
        assert_eq!(snippet.entity_text, needle);
    }
}

/// Execute the cross-document command (stub when `eval` is disabled).
#[cfg(not(feature = "eval"))]
pub fn run(_args: CrossDocArgs) -> Result<(), String> {
    Err("Cross-document entity coalescing requires 'eval' feature. Build with: cargo build -p anno-cli --features eval".to_string())
}
