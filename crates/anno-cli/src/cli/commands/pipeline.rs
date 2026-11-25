//! Pipeline command - Unified pipeline command

use clap::Parser;
use std::fs;

use super::super::output::{color, print_signals};
use super::super::parser::{ModelBackend, OutputFormat};
use super::super::utils::{link_tracks_to_kb, resolve_coreference};
use anno::backends::coref::{CorefBackend, MaverickCoref};
use anno::{GroundedDocument, SignalId};

/// Unified pipeline command
#[derive(Parser, Debug)]
pub struct PipelineArgs {
    /// Input text(s) to process (positional)
    pub text: Vec<String>,

    /// Read input from file(s)
    #[arg(short, long, value_name = "PATH", visible_alias = "input")]
    pub files: Vec<String>,

    /// Process directory of text files
    #[arg(short, long, value_name = "DIR", visible_alias = "input-dir")]
    pub dir: Option<String>,

    /// Model backend to use
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Run coreference resolution
    #[arg(long)]
    pub coref: bool,

    /// Resolve with Maverick using prepared MES safetensors or a locally prepared official model ID (optional Python runtime).
    /// Set MAVERICK_PYTHON to a Python executable with maverick-coref installed.
    #[arg(long, requires = "coref", value_name = "CHECKPOINT")]
    pub maverick_model: Option<String>,

    /// Select speaker-only coreference (English first-person singular) in place of general coreference.
    /// General name and third-person chains are omitted.
    /// Each turn has speaker, text, start and end (character offsets in one input document).
    #[cfg(feature = "discourse")]
    #[arg(
        long,
        requires = "coref",
        conflicts_with = "maverick_model",
        value_name = "JSON"
    )]
    pub speaker_turns: Option<std::path::PathBuf>,

    /// Attach offline demo KB identifiers to tracks (including generated names).
    #[arg(long)]
    pub link_kb: bool,

    /// Resolve tracks across inputs. General coreference also includes unmatched detections.
    /// Grounded output preserves source documents and assignments. JSON emits cluster rows.
    #[arg(long)]
    pub cross_doc: bool,

    /// Similarity threshold for cross-doc clustering
    #[arg(long, default_value = "0.6", value_parser = super::parse_similarity_threshold)]
    pub threshold: f64,

    /// Require compatible known track types when comparing tracks across documents.
    #[arg(long, requires = "cross_doc")]
    pub require_type_match: bool,

    /// Output format
    #[arg(long, default_value = "human")]
    pub format: OutputFormat,

    /// Export results to file
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<String>,

    /// Show progress
    #[arg(long)]
    pub progress: bool,

    /// Suppress status messages
    #[arg(short, long)]
    pub quiet: bool,
}

/// Execute the pipeline command.
pub fn run(args: PipelineArgs) -> Result<(), String> {
    #[cfg(not(feature = "eval"))]
    if args.cross_doc {
        return Err("Cross-document clustering requires 'eval' feature".into());
    }

    let mut sources = std::collections::HashMap::new();
    let mut texts = Vec::new();
    let mut add_input = |id: String, text: String, source: String| -> Result<(), String> {
        if args.cross_doc {
            super::super::utils::register_document_source(&mut sources, &id, source)?;
        }
        texts.push((id, text));
        Ok(())
    };
    for (idx, text) in args.text.iter().enumerate() {
        let number = idx + 1;
        add_input(
            format!("text{number}"),
            text.clone(),
            format!("positional input {number}"),
        )?;
    }

    let mut paths: Vec<std::path::PathBuf> = args.files.iter().map(Into::into).collect();
    if let Some(dir) = &args.dir {
        let mut directory_paths = Vec::new();
        for entry in
            fs::read_dir(dir).map_err(|e| format!("Failed to read directory {dir}: {e}"))?
        {
            let path = entry
                .map_err(|e| format!("Failed to read entry: {e}"))?
                .path();
            if path.is_file()
                && matches!(
                    path.extension().and_then(|s| s.to_str()),
                    Some("txt" | "md")
                )
            {
                directory_paths.push(path);
            }
        }
        directory_paths.sort();
        paths.extend(directory_paths);
    }
    for path in paths {
        let source = path.display().to_string();
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&source)
            .to_owned();
        let text =
            fs::read_to_string(&path).map_err(|e| format!("Failed to read {source}: {e}"))?;
        add_input(id, text, source)?;
    }

    if texts.is_empty() {
        return Err(
            "No input provided. Use positional TEXT, --files/--input, or --dir/--input-dir."
                .to_string(),
        );
    }

    #[cfg(feature = "discourse")]
    let speaker_clusters = if let Some(path) = &args.speaker_turns {
        let [(_, text)] = texts.as_slice() else {
            return Err("--speaker-turns requires exactly one input document".into());
        };
        let file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let turns: Vec<anno::discourse::dialogue::DialogueTurn> =
            serde_json::from_reader(file).map_err(|e| format!("{}: {e}", path.display()))?;
        Some(
            anno::discourse::dialogue::resolve_speaker_mentions(text, &turns)
                .map_err(|e| format!("{}: {e}", path.display()))?,
        )
    } else {
        None
    };

    let maverick = args
        .maverick_model
        .as_ref()
        .map(|checkpoint| {
            let python = std::env::var_os("MAVERICK_PYTHON").unwrap_or_else(|| "python3".into());
            MaverickCoref::new(python, checkpoint, false).map_err(|e| e.to_string())
        })
        .transpose()?;

    // Process each document
    let model = args.model.create_model()?;
    let mut documents: Vec<GroundedDocument> = Vec::new();

    let pb = if args.progress && !args.quiet {
        use indicatif::{ProgressBar, ProgressStyle};
        let pb = ProgressBar::new(texts.len() as u64);
        // Template is a constant string, so unwrap is safe, but we handle it explicitly
        let style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .expect("Progress bar template should be valid");
        pb.set_style(style.progress_chars("#>-"));
        Some(pb)
    } else {
        None
    };

    for (doc_id, text) in &texts {
        if let Some(ref pb) = pb {
            pb.set_message(format!("Processing {}", doc_id));
        }

        let entities = model
            .extract_entities(text, None)
            .map_err(|e| format!("Extraction failed for {}: {}", doc_id, e))?;
        let mut doc = if args.cross_doc && !args.coref {
            GroundedDocument::from_entities(doc_id, text, &entities)
        } else {
            GroundedDocument::from_entity_signals(doc_id, text, &entities)
        };

        doc.record_source_map(anno::SourceMap::identity(text))?;

        // Apply enhancements
        if args.coref {
            let signal_ids: Vec<SignalId> = doc.signals().iter().map(|signal| signal.id).collect();
            let selected = maverick
                .as_ref()
                .map(|coref| coref.resolve(text).map(|clusters| (coref.name(), clusters)))
                .transpose()
                .map_err(|e| format!("Maverick failed for {doc_id}: {e}"))?;
            let selected = selected
                .as_ref()
                .map(|(backend, clusters)| (*backend, clusters));
            #[cfg(feature = "discourse")]
            let selected = selected.or_else(|| {
                speaker_clusters
                    .as_ref()
                    .map(|groups| ("speaker-turns", groups))
            });
            if let Some((backend, clusters)) = selected {
                let tracks_added = super::super::utils::append_coref_clusters(&mut doc, clusters)?;
                doc.record_coref_run(anno::CorefRunReport::Completed {
                    backend: backend.into(),
                    tracks_added,
                    prior_failures: Vec::new(),
                });
            } else {
                resolve_coreference(&mut doc, text, &signal_ids);
            }

            let include_detections = args.cross_doc;
            #[cfg(feature = "discourse")]
            let include_detections = include_detections && speaker_clusters.is_none();
            if include_detections {
                seed_uncovered_detections(&mut doc, &signal_ids)?;
            }
        }

        if args.link_kb {
            link_tracks_to_kb(&mut doc);
        }

        documents.push(doc);

        if let Some(ref pb) = pb {
            pb.inc(1);
        }
    }

    if let Some(ref pb) = pb {
        pb.finish_with_message("Processing complete");
    }

    #[cfg(feature = "eval")]
    if args.cross_doc {
        return output_cross_doc(&args, documents);
    }

    match args.format {
        OutputFormat::Json | OutputFormat::Grounded => {
            write_json(&documents, args.output.as_deref())?
        }
        _ => {
            // Human-readable output
            for doc in &documents {
                println!();
                println!("{}", color("1;36", &format!("Document: {}", doc.id())));
                if let Some(backend) = doc.coref_backend() {
                    println!("Coreference backend: {backend}");
                }
                print_signals(doc, doc.text(), 0);
            }
        }
    }

    Ok(())
}

/// General coreference can emit a separate signal at the same source occurrence
/// as a detection. Keep both signals and count that occurrence once in tracks.
fn seed_uncovered_detections(
    doc: &mut GroundedDocument,
    original_ids: &[SignalId],
) -> Result<(), String> {
    use anno::{Location, Signal, Track, TrackId};
    use std::collections::HashSet;

    #[derive(Hash, Eq, PartialEq)]
    enum Extent<'a> {
        Single(usize, usize),
        Multiple(&'a [(usize, usize)]),
    }

    fn occurrence(signal: &Signal) -> Option<(&str, Extent<'_>)> {
        let extent = match &signal.location {
            Location::Text { start, end } if start < end => Extent::Single(*start, *end),
            Location::Discontinuous { segments } => match segments.as_slice() {
                [] => return None,
                &[(start, end)] => Extent::Single(start, end),
                segments => Extent::Multiple(segments),
            },
            _ => return None,
        };
        Some((signal.surface(), extent))
    }

    let mut covered: HashSet<_> = doc
        .signals()
        .iter()
        .filter(|signal| doc.track_for_signal(signal.id).is_some())
        .filter_map(occurrence)
        .collect();
    let original_ids: HashSet<_> = original_ids.iter().copied().collect();
    let mut seeds = Vec::new();
    for signal in doc
        .signals()
        .iter()
        .filter(|s| original_ids.contains(&s.id))
    {
        if doc.track_for_signal(signal.id).is_some()
            || occurrence(signal).is_some_and(|key| !covered.insert(key))
        {
            continue;
        }
        let mut track = Track::new(TrackId::ZERO, signal.surface());
        track.entity_type = Some(signal.label.clone());
        track.add_signal(signal.id, 0);
        seeds.push(track);
    }
    for track in seeds {
        doc.try_add_track(track)
            .map_err(|e| format!("Failed to seed a cross-document track: {e}"))?;
    }
    Ok(())
}

fn write_json<T: serde::Serialize>(value: &T, path: Option<&str>) -> Result<(), String> {
    let output = serde_json::to_string_pretty(value)
        .map_err(|e| format!("Failed to serialize output: {e}"))?;
    if let Some(path) = path {
        fs::write(path, output).map_err(|e| format!("Failed to write {path}: {e}"))?;
    } else {
        println!("{output}");
    }
    Ok(())
}

#[cfg(feature = "eval")]
fn output_cross_doc(args: &PipelineArgs, documents: Vec<GroundedDocument>) -> Result<(), String> {
    use anno::{coalesce::Resolver, Corpus};

    let mut corpus = Corpus::new();
    let ids: Vec<_> = documents.iter().map(|doc| doc.id().to_owned()).collect();
    for doc in documents {
        corpus.add_document(doc);
    }
    Resolver::new()
        .with_threshold(args.threshold as f32)
        .require_type_match(args.require_type_match)
        .resolve_inter_doc_coref(&mut corpus, None, None);
    if matches!(args.format, OutputFormat::Grounded) {
        let docs: Vec<_> = ids
            .iter()
            .map(|id| corpus.get_document(id).expect("inserted document"))
            .collect();
        return write_json(&docs, args.output.as_deref());
    }
    let clusters = super::crossdoc::corpus_clusters(&corpus);
    match args.format {
        OutputFormat::Json => write_json(&clusters, args.output.as_deref())?,
        OutputFormat::Tree => {
            for cluster in &clusters {
                println!("Cluster {}: {}", cluster.id, cluster.canonical_name);
                for (doc_id, entity_idx) in &cluster.mentions {
                    let mention = corpus
                        .get_document(doc_id)
                        .and_then(|doc| doc.signals().get(*entity_idx))
                        .map(|signal| signal.surface().to_owned())
                        .unwrap_or_else(|| format!("entity_{entity_idx}"));
                    println!("  - {mention} (doc: {doc_id})");
                }
                println!();
            }
        }
        _ => {
            println!(
                "\n{} Cross-document clusters: {}",
                color("1;36", "Found"),
                clusters.len()
            );
            for cluster in &clusters {
                println!(
                    "  {}: {} mentions across {} documents",
                    cluster.canonical_name,
                    cluster.mentions.len(),
                    cluster.doc_count()
                );
            }
        }
    }
    Ok(())
}
