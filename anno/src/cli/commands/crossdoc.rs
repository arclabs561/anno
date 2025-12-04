//! CrossDoc command - Cross-document coreference: cluster entities across multiple documents



use super::super::parser::{ModelBackend, OutputFormat};

/// Cross-document coreference: cluster entities across multiple documents
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

#[cfg(feature = "eval-advanced")]
pub fn run(args: CrossDocArgs) -> Result<(), String> {
    #[cfg(not(feature = "eval"))]
    return Err("Cross-document coreference requires 'eval' feature. Build with: cargo build --features eval".to_string());

    #[cfg(feature = "eval")]
    {
        // Create model
        let model = args.model.create_model()?;

        if args.verbose && args.directory.is_some() {
            eprintln!("Scanning directory: {}", args.directory.as_ref().unwrap());
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
        let mut clusters_from_corpus: Option<Vec<CrossDocCluster>> = None;
        let mut documents: Vec<Document> = Vec::new(); // Only used in normal mode, but declared here for output formatting

        // Helper function to convert Identity to CrossDocCluster with proper mention extraction
        fn identity_to_cluster(identity: &Identity, corpus: &Corpus) -> CrossDocCluster {
            let mut cluster = CrossDocCluster::new(identity.id, &identity.canonical_name);
            cluster.kb_id = identity.kb_id.clone();
            cluster.confidence = identity.confidence as f64;
            if let Some(ref entity_type) = identity.entity_type {
                cluster.entity_type = Some(EntityType::from_label(entity_type));
            }

            // Extract mentions from TrackRefs if available
            if let Some(IdentitySource::CrossDocCoref { ref track_refs }) = &identity.source {
                let mut doc_set = HashSet::new();
                for track_ref in track_refs {
                    // Get the track and extract signal IDs
                    if let Some(doc) = corpus.get_document(&track_ref.doc_id) {
                        if let Some(track) = doc.get_track(track_ref.track_id) {
                            // For each signal in the track, we need to find its entity index
                            // Since we're converting from GroundedDocument, we need to map signals to entities
                            // For now, use signal positions as entity indices (approximation)
                            for (pos, signal_ref) in track.signals.iter().enumerate() {
                                if let Some(signal) = doc.get_signal(signal_ref.signal_id) {
                                    // Find entity index by matching signal text and position
                                    // This is approximate - in a perfect world, we'd track the mapping
                                    let entity_idx = pos; // Use position as approximation
                                    cluster
                                        .mentions
                                        .push((track_ref.doc_id.clone(), entity_idx));
                                    doc_set.insert(track_ref.doc_id.clone());
                                }
                            }
                        }
                    }
                }
                cluster.documents = doc_set.into_iter().collect();
            }

            cluster
        }

        // Legacy helper for CDCR Document conversion (used when not using Corpus)
        fn load_grounded_doc_legacy(
            doc: &GroundedDocument,
            _source_path: &str,
        ) -> (Document, usize) {
            // Prefer tracks if available (Level 2), otherwise use signals (Level 1)
            let tracks_vec: Vec<_> = doc.tracks().collect();
            let entities: Vec<_> = if !tracks_vec.is_empty() {
                // Use tracks: each track represents a within-doc coreference chain
                // Extract canonical mention from each track
                tracks_vec
                    .iter()
                    .filter_map(|track| {
                        // Get the first signal in the track as the canonical mention
                        let signal_ids: Vec<_> =
                            track.signals.iter().map(|sr| sr.signal_id).collect();
                        signal_ids
                            .first()
                            .and_then(|signal_id| {
                                doc.get_signal(*signal_id).map(|signal| {
                                    let (start, end) = signal.text_offsets().unwrap_or((0, 0));
                                    Entity::new(
                                        signal.surface(),
                                        EntityType::from_label(signal.label()),
                                        start,
                                        end,
                                        signal.confidence as f64,
                                    )
                                })
                            })
                            .or_else(|| {
                                // Fallback: create entity from track canonical
                                Some(Entity::new(
                                    &track.canonical_surface,
                                    track
                                        .entity_type
                                        .as_ref()
                                        .map(|t| EntityType::from_label(t))
                                        .unwrap_or(EntityType::Other("UNKNOWN".into())),
                                    0,
                                    0,
                                    track.cluster_confidence as f64,
                                ))
                            })
                    })
                    .collect()
            } else {
                // Fallback to signals
                doc.signals()
                    .iter()
                    .map(|s| {
                        let (start, end) = s.text_offsets().unwrap_or((0, 0));
                        Entity::new(
                            s.surface(),
                            EntityType::from_label(s.label()),
                            start,
                            end,
                            s.confidence as f64,
                        )
                    })
                    .collect()
            };

            let entity_count = entities.len();
            let cdcr_doc = Document::new(&doc.id, &doc.text).with_entities(entities);
            (cdcr_doc, entity_count)
        }

        if !args.import.is_empty() || args.stdin {
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
                    let line = line.map_err(|e| {
                        format!("Failed to read stdin line {}: {}", line_num + 1, e)
                    })?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    let mut doc: GroundedDocument = serde_json::from_str(&line).map_err(|e| {
                        format!("Failed to parse stdin line {}: {}", line_num + 1, e)
                    })?;
                    // Ensure tracks exist - if not, create them from signals for better clustering
                    if doc.tracks.is_empty() && !doc.signals.is_empty() {
                        // Could run within-doc coref here, but for now just use signals
                        // The Corpus will cluster based on signals if no tracks
                    }
                    corpus.add_document(doc);
                    doc_paths.insert(
                        corpus
                            .get_document(&format!("stdin:{}", line_num + 1))
                            .map(|d| d.id.clone())
                            .unwrap_or_else(|| format!("stdin:{}", line_num + 1)),
                        format!("stdin:{}", line_num + 1),
                    );
                    if args.verbose {
                        let stats = corpus
                            .get_document(&format!("stdin:{}", line_num + 1))
                            .map(|d| d.stats())
                            .unwrap_or_default();
                        eprintln!(
                            "  Imported {} signals, {} tracks from stdin line {}",
                            stats.signal_count,
                            stats.track_count,
                            line_num + 1
                        );
                    }
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
                            let mut doc: GroundedDocument =
                                serde_json::from_str(&line).map_err(|e| {
                                    format!("Failed to parse stdin line {}: {}", line_num + 1, e)
                                })?;
                            let doc_id = doc.id.clone();
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
                        let matches = glob(import_pattern).map_err(|e| {
                            format!("Invalid glob pattern '{}': {}", import_pattern, e)
                        })?;
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

                    let doc: GroundedDocument =
                        serde_json::from_str(&json_content).map_err(|e| {
                            format!(
                                "Failed to parse GroundedDocument JSON from '{}': {}",
                                file_path.display(),
                                e
                            )
                        })?;

                    let (_cdcr_doc, entity_count) =
                        load_grounded_doc_legacy(&doc, &file_path.display().to_string());
                    corpus.add_document(doc);
                    doc_paths.insert(
                        corpus
                            .get_document(&file_path.display().to_string())
                            .map(|d| d.id.clone())
                            .unwrap_or_else(|| file_path.display().to_string()),
                        file_path.display().to_string(),
                    );

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
                    args.threshold,
                    args.require_type_match
                );
            }

            let resolver = Resolver::new()
                .with_threshold(args.threshold as f32)
                .require_type_match(args.require_type_match);
            let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);

            if args.verbose {
                eprintln!(
                    "Created {} identities from inter-doc coref",
                    identity_ids.len()
                );
            }

            // Convert identities to CrossDocCluster for output compatibility
            // Extract mentions from TrackRefs in Identity
            let mut clusters: Vec<CrossDocCluster> = Vec::new();
            for &id in &identity_ids {
                if let Some(identity) = corpus.get_identity(id) {
                    let mut cluster = identity_to_cluster(identity, &corpus);

                    // Populate mentions from TrackRefs
                    if let Some(IdentitySource::CrossDocCoref { ref track_refs }) = &identity.source
                    {
                        let mut doc_set = HashSet::new();
                        for track_ref in track_refs {
                            if let Some(doc) = corpus.get_document(&track_ref.doc_id) {
                                if let Some(track) = doc.get_track(track_ref.track_id) {
                                    // Add each signal in the track as a mention
                                    // Use signal position as entity index (approximation)
                                    for (pos, _signal_ref) in track.signals.iter().enumerate() {
                                        cluster.mentions.push((track_ref.doc_id.clone(), pos));
                                        doc_set.insert(track_ref.doc_id.clone());
                                    }
                                }
                            }
                        }
                        cluster.documents = doc_set.into_iter().collect();
                    }

                    // Apply filters
                    if cluster.len() >= args.min_cluster_size
                        && (!args.cross_doc_only || cluster.doc_count() > 1)
                        && (args.entity_types.is_empty()
                            || cluster
                                .entity_type
                                .as_ref()
                                .map(|et| {
                                    let type_label = et.as_label().to_uppercase();
                                    args.entity_types
                                        .iter()
                                        .any(|t| t.to_uppercase() == type_label)
                                })
                                .unwrap_or(false))
                    {
                        clusters.push(cluster);
                    }
                }
            }

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

            clusters_from_corpus = Some(clusters);
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

            if args.verbose {
                eprintln!("Found {} files", files.len());
                eprintln!("Extracting entities...");
            }

            // NOTE: Currently operates on raw entities (Level 1: Signal)
            // With --import, can use Level 2 (Tracks) and Level 3 (Identities) from pre-processed docs
            let total_files = files.len();
            for (idx, file_path) in files.iter().enumerate() {
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

                let doc_id = file_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("doc{}", idx));

                // Store file path for later display
                doc_paths.insert(doc_id.clone(), file_path.display().to_string());

                let entities = model
                    .extract_entities(&text, None)
                    .map_err(|e| format!("Failed to extract entities from {}: {}", doc_id, e))?;

                // Build GroundedDocument and optionally run coreference for better clustering
                // This enables using tracks (Level 2) instead of just raw signals (Level 1)
                let mut _grounded_doc = GroundedDocument::new(&doc_id, &text);
                let mut _signal_ids: Vec<u64> = Vec::new();

                for e in &entities {
                    let signal = Signal::new(
                        0,
                        Location::text(e.start, e.end),
                        &e.text,
                        e.entity_type.as_label(),
                        e.confidence as f32,
                    );
                    let id = _grounded_doc.add_signal(signal);
                    _signal_ids.push(id);
                }

                // If we have tracks from import, use them; otherwise use signals
                // For now, convert to CDCR Document using signals (can be enhanced to use tracks)
                documents.push(Document::new(&doc_id, &text).with_entities(entities));
            }

            if args.verbose {
                eprintln!("\r  Processed {} files successfully", total_files);
            }

            if args.verbose {
                let total_entities: usize = documents.iter().map(|d| d.entities.len()).sum();
                eprintln!(
                    "Clustering {} entities across {} documents...",
                    total_entities,
                    documents.len()
                );
            }

            // Configure and run cross-doc coref using CDCRResolver (for raw text files)
            let config = CDCRConfig {
                min_similarity: args.threshold,
                require_type_match: args.require_type_match,
                use_lsh: documents.len() > 100, // Use LSH for large document sets
                ..Default::default()
            };

            let resolver = CDCRResolver::with_config(config);
            let mut clusters = resolver.resolve(&documents);

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
            clusters_from_corpus = Some(clusters);
        }

        // Use clusters from Corpus if available, otherwise use CDCR clusters
        let final_clusters: Vec<CrossDocCluster> = clusters_from_corpus
            .ok_or_else(|| "No clusters generated. This should not happen.".to_string())?;

        // Prepare output
        let output_text = match args.format {
            OutputFormat::Json => {
                // Enhanced JSON with metadata
                let mut output = serde_json::Map::new();
                let doc_count = if use_corpus {
                    corpus.documents().count()
                } else {
                    documents.len()
                };
                let total_entities = if use_corpus {
                    corpus
                        .documents()
                        .map(|d| d.stats().signal_count)
                        .sum::<usize>()
                } else {
                    documents.iter().map(|d| d.entities.len()).sum::<usize>()
                };
                output.insert("metadata".to_string(), serde_json::json!({
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
                    color("1;36", "Cross-Document Entity Clusters")
                ));
                output.push_str("\n");

                // Summary header
                let doc_count = if use_corpus {
                    corpus.documents().count()
                } else {
                    documents.len()
                };
                let total_entities = if use_corpus {
                    corpus
                        .documents()
                        .map(|d| d.stats().signal_count)
                        .sum::<usize>()
                } else {
                    documents.iter().map(|d| d.entities.len()).sum::<usize>()
                };
                let cross_doc_clusters =
                    final_clusters.iter().filter(|c| c.doc_count() > 1).count();
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
                output.push_str("\n");

                // Entity type breakdown
                let mut type_counts: HashMap<String, usize> = HashMap::new();
                for cluster in &final_clusters {
                    if let Some(ref entity_type) = cluster.entity_type {
                        *type_counts
                            .entry(entity_type.as_label().to_string())
                            .or_insert(0) += 1;
                    }
                }
                if !type_counts.is_empty() {
                    output.push_str(&format!("{}\n", color("1;33", "Entity Types")));
                    let mut type_vec: Vec<_> = type_counts.iter().collect();
                    type_vec.sort_by(|a, b| b.1.cmp(a.1));
                    for (etype, count) in type_vec {
                        output.push_str(&format!("  {}: {}\n", etype, count));
                    }
                    output.push_str("\n");
                }

                output.push_str(&format!("{}\n", color("1;36", "Clusters")));
                output.push_str("\n");

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
                        header.push_str(&format!(" ({})", entity_type.as_label()));
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
                    if cluster.confidence < 1.0 {
                        meta_parts.push(format!("conf: {:.2}", cluster.confidence));
                    }
                    output.push_str(&format!("  {}\n", meta_parts.join(" • ")));

                    if let Some(ref kb_id) = cluster.kb_id {
                        output.push_str(&format!("  KB: {}\n", color("36", kb_id)));
                    }

                    // Show documents with paths (truncate if too many)
                    if !cluster.documents.is_empty() {
                        let max_docs_to_show = if args.verbose { 20 } else { 5 };
                        let doc_list: Vec<String> = cluster
                            .documents
                            .iter()
                            .take(max_docs_to_show)
                            .map(|doc_id| {
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
                            if let Some(doc) = doc_index.get(doc_id.as_str()) {
                                if let Some(entity) = doc.entities.get(*entity_idx) {
                                    if args.verbose {
                                        // Extract context safely (50 chars before/after)
                                        let context_window = 50;

                                        // Find character boundaries for entity
                                        let entity_start_char = doc
                                            .text
                                            .char_indices()
                                            .position(|(byte_idx, _)| byte_idx >= entity.start)
                                            .unwrap_or(0);
                                        let entity_end_char = doc
                                            .text
                                            .char_indices()
                                            .position(|(byte_idx, _)| byte_idx >= entity.end)
                                            .unwrap_or(doc.text.chars().count());

                                        // Calculate context character range
                                        let context_start_char =
                                            entity_start_char.saturating_sub(context_window);
                                        let context_end_char = (entity_end_char + context_window)
                                            .min(doc.text.chars().count());

                                        // Convert back to byte positions
                                        let safe_start = doc
                                            .text
                                            .char_indices()
                                            .nth(context_start_char)
                                            .map(|(byte_idx, _)| byte_idx)
                                            .unwrap_or(0);
                                        let safe_end = doc
                                            .text
                                            .char_indices()
                                            .nth(context_end_char)
                                            .map(|(byte_idx, _)| byte_idx)
                                            .unwrap_or(doc.text.len());

                                        let context = &doc.text[safe_start..safe_end];

                                        // Ensure entity positions are at character boundaries
                                        let entity_start_byte = doc
                                            .text
                                            .char_indices()
                                            .find(|&(byte_idx, _)| byte_idx >= entity.start)
                                            .map(|(byte_idx, _)| byte_idx)
                                            .unwrap_or(entity.start);
                                        let entity_end_byte = doc
                                            .text
                                            .char_indices()
                                            .find(|&(byte_idx, _)| byte_idx >= entity.end)
                                            .map(|(byte_idx, _)| byte_idx)
                                            .unwrap_or(entity.end);
                                        let entity_text =
                                            &doc.text[entity_start_byte..entity_end_byte];

                                        // Calculate offsets within the safe context window (in bytes)
                                        let before_len = entity.start.saturating_sub(safe_start);
                                        let after_start =
                                            (entity.end - safe_start).min(context.len());

                                        // Find character boundaries for safe slicing
                                        let before = if safe_start < entity.start && before_len > 0
                                        {
                                            // Find the character boundary closest to before_len
                                            let target_byte = before_len.min(context.len());
                                            let char_boundary = context
                                                .char_indices()
                                                .find(|&(byte_idx, _)| byte_idx >= target_byte)
                                                .map(|(byte_idx, _)| byte_idx)
                                                .unwrap_or(context.len());
                                            &context[..char_boundary.min(context.len())]
                                        } else {
                                            ""
                                        };

                                        let after = if after_start < context.len() {
                                            // Find the character boundary at after_start
                                            let char_boundary = context
                                                .char_indices()
                                                .find(|&(byte_idx, _)| byte_idx >= after_start)
                                                .map(|(byte_idx, _)| byte_idx)
                                                .unwrap_or(context.len());
                                            &context[char_boundary.min(context.len())..]
                                        } else {
                                            ""
                                        };

                                        let before_marker =
                                            if safe_start < entity.start { "..." } else { "" };
                                        let after_marker =
                                            if entity.end < safe_end { "..." } else { "" };

                                        output.push_str(&format!(
                                            "    {} {}: {}{}[{}]{}{}\n",
                                            color("90", "•"),
                                            color("36", doc_id),
                                            before_marker,
                                            before,
                                            color("1;32", entity_text),
                                            after,
                                            after_marker
                                        ));
                                    } else {
                                        // Non-verbose: just show entity text
                                        output.push_str(&format!(
                                            "    {} {}: \"{}\"\n",
                                            color("90", "•"),
                                            color("36", doc_id),
                                            entity.text
                                        ));
                                    }
                                }
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

                    output.push_str("\n");
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
                let total_entities: usize = documents.iter().map(|d| d.entities.len()).sum();
                let cross_doc_clusters =
                    final_clusters.iter().filter(|c| c.doc_count() > 1).count();
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
                        *type_counts
                            .entry(entity_type.as_label().to_string())
                            .or_insert(0) += 1;
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
                    color("1;36", "  Cross-Document Coreference Summary")
                ));
                output.push_str(&format!(
                    "{}\n",
                    color(
                        "1;36",
                        "═══════════════════════════════════════════════════════════"
                    )
                ));
                output.push_str("\n");
                output.push_str(&format!("{}\n", color("1;33", "Document Statistics:")));
                let doc_count = if use_corpus {
                    corpus.documents().count()
                } else {
                    documents.len()
                };
                output.push_str(&format!("  Documents processed: {}\n", doc_count));
                output.push_str(&format!("  Total entities extracted: {}\n", total_entities));
                output.push_str(&format!(
                    "  Average entities per document: {:.1}\n",
                    if documents.is_empty() {
                        0.0
                    } else {
                        total_entities as f64 / documents.len() as f64
                    }
                ));
                output.push_str("\n");
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
                output.push_str("\n");
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
            OutputFormat::Human
            | OutputFormat::Tsv
            | OutputFormat::Inline
            | OutputFormat::Grounded
            | OutputFormat::Html => {
                return Err(format!("Format '{}' not supported for cross-doc command. Use: json, jsonl, tree, or summary.", 
                    match args.format {
                        OutputFormat::Human => "human",
                        OutputFormat::Tsv => "tsv",
                        OutputFormat::Inline => "inline",
                        OutputFormat::Grounded => "grounded",
                        OutputFormat::Html => "html",
                        _ => unreachable!(),
                    }
                ));
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
}

#[cfg(not(feature = "eval-advanced"))]
pub fn run(_args: CrossDocArgs) -> Result<(), String> {
    Err("Cross-document coreference requires 'eval-advanced' feature. Build with: cargo build --features eval-advanced".to_string())
}
