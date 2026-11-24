//! Coreference dataset loading and parsing.
//!
//! Provides specialized loaders for coreference datasets that return
//! `CorefDocument` structures rather than NER annotations.
//!
//! # Supported Datasets
//!
//! | Dataset | Format | Size | Features |
//! |---------|--------|------|----------|
//! | GAP | TSV | 8,908 pairs | Gender-balanced pronoun resolution |
//! | PreCo | JSON | ~38k docs | Large-scale, includes singletons |
//! | Synthetic | Generated | Configurable | For testing metrics |
//!
//! # Example
//!
//! ```rust,ignore
//! use anno_eval::eval::coref_loader::{CorefLoader, synthetic_coref_dataset};
//!
//! // Load GAP development set (requires eval feature for download)
//! let loader = CorefLoader::new().unwrap();
//! let docs = loader.load_gap().unwrap();
//!
//! // Or generate synthetic data for testing
//! let synthetic = synthetic_coref_dataset(10);
//! ```

use super::coref::{CorefChain, CorefDocument, Mention, MentionType};
use super::loader::DatasetId;
use anno::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::PathBuf;

// =============================================================================
// CorefUD (CoNLL-U + Entity brackets in MISC)
// =============================================================================

/// Parse CorefUD CoNLL-U format into coreference documents.
///
/// CorefUD encodes coreference in the CoNLL-U MISC column via the `Entity` attribute.
/// The value of `Entity=` is a bracketed stream of mention boundary markers:
/// - Opening marker at the first token of a mention: `Entity=(e5-person-...)`
/// - Closing marker at the last token of a mention: `Entity=e5)`
/// - One-token mentions can be encoded as a self-contained marker: `Entity=(e8-place-1)`
/// - Multiple markers can be concatenated in one value: `Entity=(e8-place-1)e9)`
/// - Discontinuous mentions may include part tags after the cluster id: `e10[1/2]`
///
/// This parser is intentionally conservative:
/// - It extracts clusters and **contiguous character spans**.
/// - It preserves `MentionType::Zero` for mentions that are empty nodes (ID with `.`).
/// - It ignores bridging/split antecedents and other CorefUD extras.
///
/// Note: CorefUD supports discontinuous mentions. `anno` currently represents mentions
/// as contiguous character spans, so discontinuous mentions are approximated by their
/// minimal bounding span in surface character space.
pub fn parse_corefud_conllu(content: &str) -> Result<Vec<CorefDocument>> {
    #[derive(Debug, Clone)]
    struct TokenSpan {
        is_empty_node: bool,
        entity_value: Option<String>,
    }

    #[derive(Debug, Clone)]
    struct OpenMention {
        start_char: usize,
        entity_type: Option<String>,
        is_empty_node: bool,
    }

    #[derive(Debug, Clone)]
    enum EntityMark {
        Open {
            cluster: String,
            entity_type: Option<String>,
            self_close: bool,
        },
        Close {
            cluster: String,
        },
    }

    fn parse_misc_entity(misc: &str) -> Option<String> {
        if misc.trim().is_empty() || misc.trim() == "_" {
            return None;
        }
        for part in misc.split('|') {
            if let Some(rest) = part.strip_prefix("Entity=") {
                if !rest.is_empty() && rest != "_" {
                    return Some(rest.to_string());
                }
            }
        }
        None
    }

    fn parse_space_after_no(misc: &str) -> bool {
        if misc.trim().is_empty() || misc.trim() == "_" {
            return false;
        }
        misc.split('|').any(|p| p == "SpaceAfter=No")
    }

    fn split_cluster_and_type(open_descriptor: &str) -> (String, Option<String>) {
        // Descriptor example: "e5-person-1-1,2,4-new-coref"
        // We only reliably extract:
        // - cluster id: first field before '-'
        // - entity type: second field (if present)
        let mut parts = open_descriptor.splitn(3, '-');
        let raw_cluster = parts.next().unwrap_or("").to_string();
        let etype = parts.next().map(|s| s.to_string());

        // Handle discontinuous tags like e10[1/2] by stripping bracket suffix.
        let cluster = if let Some((base, _rest)) = raw_cluster.split_once('[') {
            base.to_string()
        } else {
            raw_cluster
        };
        (cluster, etype)
    }

    fn split_cluster(close_descriptor: &str) -> String {
        // Close descriptor example: "e9" (from "e9)")
        // May include discontinuous tag suffix e10[1/2]
        if let Some((base, _)) = close_descriptor.split_once('[') {
            base.to_string()
        } else {
            close_descriptor.to_string()
        }
    }

    fn parse_entity_marks(entity_value: &str) -> Vec<EntityMark> {
        // Parse a stream of markers from left-to-right.
        //
        // Markers are either:
        // - '(' + descriptor + ')'        => Open marker that self-closes (one-token mention)
        // - '(' + descriptor (no ')')     => Open marker (mention continues)
        // - descriptor + ')'              => Close marker
        //
        // The format allows concatenation: "(e8-place-1)e9)e7)"
        let mut marks = Vec::new();
        let mut i = 0usize;
        let bytes = entity_value.as_bytes();

        while i < bytes.len() {
            match bytes[i] as char {
                '(' => {
                    i += 1;
                    let start = i;
                    // Descriptor ends at ')' (self-contained) OR at the next '(' (multiple opens).
                    while i < bytes.len() && (bytes[i] as char) != ')' && (bytes[i] as char) != '('
                    {
                        i += 1;
                    }
                    if i < bytes.len() && (bytes[i] as char) == ')' {
                        // Self-contained marker "(...)" (one-token mention)
                        let descriptor = entity_value[start..i].to_string();
                        let (cluster, etype) = split_cluster_and_type(&descriptor);
                        marks.push(EntityMark::Open {
                            cluster,
                            entity_type: etype,
                            self_close: true,
                        });
                        i += 1;
                    } else {
                        // No closing ')': open marker continues.
                        // If we stopped because we hit another '(', allow parsing the next marker too.
                        let descriptor = entity_value[start..i].to_string();
                        let (cluster, etype) = split_cluster_and_type(&descriptor);
                        marks.push(EntityMark::Open {
                            cluster,
                            entity_type: etype,
                            self_close: false,
                        });
                        if i >= bytes.len() {
                            break;
                        }
                    }
                }
                'e' => {
                    // Parse closing marker: "e123...)" possibly repeated.
                    let start = i;
                    while i < bytes.len() && (bytes[i] as char) != ')' {
                        if (bytes[i] as char) == '(' {
                            break;
                        }
                        i += 1;
                    }
                    if i < bytes.len() && (bytes[i] as char) == ')' {
                        let raw = entity_value[start..i].to_string();
                        marks.push(EntityMark::Close {
                            cluster: split_cluster(&raw),
                        });
                        i += 1;
                    } else {
                        break;
                    }
                }
                _ => i += 1,
            }
        }

        marks
    }

    fn extract_span_text(text: &str, start: usize, end: usize) -> String {
        if end <= start {
            return String::new();
        }
        text.chars().skip(start).take(end - start).collect()
    }

    // Document accumulators
    let mut docs: Vec<CorefDocument> = Vec::new();
    let mut doc_idx: usize = 0;
    let mut current_doc_id: Option<String> = None;
    let mut text = String::new();
    let mut text_char_len: usize = 0;

    let mut tokens: Vec<TokenSpan> = Vec::new();
    let mut clusters: HashMap<String, Vec<Mention>> = HashMap::new();
    let mut open: HashMap<String, Vec<OpenMention>> = HashMap::new();

    let mut prev_space_after_no = false;

    let flush_doc = |docs: &mut Vec<CorefDocument>,
                     doc_idx: &mut usize,
                     current_doc_id: &mut Option<String>,
                     text: &mut String,
                     clusters: &mut HashMap<String, Vec<Mention>>,
                     open: &mut HashMap<String, Vec<OpenMention>>|
     -> Result<()> {
        if text.is_empty() && clusters.is_empty() {
            *current_doc_id = None;
            open.clear();
            return Ok(());
        }

        if open.values().any(|stk| !stk.is_empty()) {
            return Err(Error::InvalidInput(
                "CorefUD parse error: document ended with unclosed Entity brackets".to_string(),
            ));
        }

        // Build chains.
        let mut coref_chains: Vec<CorefChain> = Vec::new();
        for (cluster_id, mut mentions) in std::mem::take(clusters).into_iter() {
            // Fill mention texts now that doc text is finalized.
            for m in &mut mentions {
                if m.mention_type == Some(MentionType::Zero) {
                    m.text = String::new();
                } else {
                    m.text = extract_span_text(text, m.start, m.end);
                }
            }

            // Convert "e123" -> 123 if possible.
            let numeric_id = cluster_id.strip_prefix('e').and_then(|rest| {
                rest.chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse::<u64>()
                    .ok()
            });

            if let Some(cid) = numeric_id {
                coref_chains.push(CorefChain::with_id(mentions, cid));
            } else {
                coref_chains.push(CorefChain::new(mentions));
            }
        }

        if coref_chains.is_empty() {
            return Err(Error::InvalidInput(
                "CorefUD CoNLL-U contains no coreference chains".to_string(),
            ));
        }

        let doc_id = current_doc_id
            .clone()
            .unwrap_or_else(|| format!("corefud_doc_{}", *doc_idx));
        *doc_idx += 1;

        docs.push(CorefDocument::with_id(
            std::mem::take(text),
            doc_id,
            coref_chains,
        ));

        // Reset
        *current_doc_id = None;
        open.clear();
        Ok(())
    };

    for raw_line in content.lines() {
        let line = raw_line.trim_end();

        // Document boundary
        if line.starts_with("# newdoc") {
            flush_doc(
                &mut docs,
                &mut doc_idx,
                &mut current_doc_id,
                &mut text,
                &mut clusters,
                &mut open,
            )?;
            tokens.clear();
            text_char_len = 0;
            prev_space_after_no = false;

            // Parse optional doc id
            if let Some(pos) = line.find("id") {
                let maybe = line[pos..].split('=').nth(1).map(|s| s.trim());
                if let Some(id) = maybe {
                    if !id.is_empty() {
                        current_doc_id = Some(id.to_string());
                    }
                }
            }
            continue;
        }

        // Comments
        if line.starts_with('#') {
            continue;
        }

        // Sentence boundary
        if line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 2 {
            continue;
        }

        let id_field = fields[0];
        // Skip multi-word token lines (e.g., 1-2)
        if id_field.contains('-') {
            continue;
        }

        let is_empty_node = id_field.contains('.');
        let form = fields.get(1).copied().unwrap_or("_");
        let misc = fields.get(9).copied().unwrap_or("_");
        let entity_value = parse_misc_entity(misc);
        let space_after_no = parse_space_after_no(misc);

        let (char_start, char_end) = if is_empty_node {
            (text_char_len, text_char_len)
        } else {
            if !text.is_empty() && !prev_space_after_no {
                text.push(' ');
                text_char_len += 1;
            }
            let start = text_char_len;
            text.push_str(form);
            text_char_len += form.chars().count();
            (start, text_char_len)
        };

        if !is_empty_node {
            prev_space_after_no = space_after_no;
        }

        let token_idx = tokens.len();
        tokens.push(TokenSpan {
            is_empty_node,
            entity_value,
        });

        // Apply entity boundary markers to build mentions.
        if let Some(ref ev) = tokens[token_idx].entity_value {
            let marks = parse_entity_marks(ev);
            for mark in marks {
                match mark {
                    EntityMark::Open {
                        cluster,
                        entity_type,
                        self_close,
                    } => {
                        if self_close {
                            let mut m = Mention::new("", char_start, char_end);
                            if tokens[token_idx].is_empty_node {
                                m.mention_type = Some(MentionType::Zero);
                            }
                            if let Some(et) = entity_type {
                                m.entity_type = Some(et);
                            }
                            clusters.entry(cluster).or_default().push(m);
                        } else {
                            open.entry(cluster).or_default().push(OpenMention {
                                start_char: char_start,
                                entity_type,
                                is_empty_node: tokens[token_idx].is_empty_node,
                            });
                        }
                    }
                    EntityMark::Close { cluster } => {
                        let Some(stack) = open.get_mut(&cluster) else {
                            return Err(Error::InvalidInput(format!(
                                "CorefUD parse error: closing Entity for {} with no open mention",
                                cluster
                            )));
                        };
                        let Some(opened) = stack.pop() else {
                            return Err(Error::InvalidInput(format!(
                                "CorefUD parse error: closing Entity for {} with empty stack",
                                cluster
                            )));
                        };

                        let mut m = Mention::new("", opened.start_char, char_end);
                        if opened.is_empty_node
                            && tokens[token_idx].is_empty_node
                            && opened.start_char == char_end
                        {
                            m.mention_type = Some(MentionType::Zero);
                        }
                        if let Some(et) = opened.entity_type {
                            m.entity_type = Some(et);
                        }
                        clusters.entry(cluster).or_default().push(m);
                    }
                }
            }
        }
    }

    // Final flush
    flush_doc(
        &mut docs,
        &mut doc_idx,
        &mut current_doc_id,
        &mut text,
        &mut clusters,
        &mut open,
    )?;

    if docs.is_empty() {
        return Err(Error::InvalidInput(
            "CorefUD CoNLL-U contains no documents".to_string(),
        ));
    }

    Ok(docs)
}

// =============================================================================
// GAP Dataset Structures
// =============================================================================

/// A single GAP example (pronoun-name pair).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapExample {
    /// Unique identifier
    pub id: String,
    /// Full text context
    pub text: String,
    /// The pronoun to resolve
    pub pronoun: String,
    /// Character offset of pronoun
    pub pronoun_offset: usize,
    /// First candidate name (A)
    pub name_a: String,
    /// Character offset of name A
    pub offset_a: usize,
    /// Whether pronoun refers to A
    pub coref_a: bool,
    /// Second candidate name (B)
    pub name_b: String,
    /// Character offset of name B
    pub offset_b: usize,
    /// Whether pronoun refers to B
    pub coref_b: bool,
    /// Source URL (Wikipedia)
    pub url: Option<String>,
}

impl GapExample {
    /// Convert to coreference document.
    ///
    /// Creates chains based on the coreference labels.
    #[must_use]
    pub fn to_coref_document(&self) -> CorefDocument {
        let mut chains = Vec::new();

        // Create mention for pronoun
        let pronoun_mention = Mention::with_type(
            &self.pronoun,
            self.pronoun_offset,
            self.pronoun_offset + self.pronoun.chars().count(),
            MentionType::Pronominal,
        );

        // Create mentions for names
        let mention_a = Mention::with_type(
            &self.name_a,
            self.offset_a,
            self.offset_a + self.name_a.chars().count(),
            MentionType::Proper,
        );

        let mention_b = Mention::with_type(
            &self.name_b,
            self.offset_b,
            self.offset_b + self.name_b.chars().count(),
            MentionType::Proper,
        );

        // Build chains based on coreference labels
        if self.coref_a {
            // Pronoun refers to A
            chains.push(CorefChain::new(vec![mention_a, pronoun_mention.clone()]));
            chains.push(CorefChain::singleton(mention_b));
        } else if self.coref_b {
            // Pronoun refers to B
            chains.push(CorefChain::singleton(mention_a));
            chains.push(CorefChain::new(vec![mention_b, pronoun_mention.clone()]));
        } else {
            // Neither (pronoun doesn't refer to either candidate)
            chains.push(CorefChain::singleton(mention_a));
            chains.push(CorefChain::singleton(mention_b));
            chains.push(CorefChain::singleton(pronoun_mention));
        }

        CorefDocument::with_id(&self.text, &self.id, chains)
    }
}

// =============================================================================
// PreCo Dataset Structures
// =============================================================================

/// A PreCo document with coreference annotations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreCoDocument {
    /// Document ID
    pub id: String,
    /// Sentences as token arrays
    pub sentences: Vec<Vec<String>>,
    /// Coreference mentions: (sentence_idx, start_token, end_token, cluster_id)
    pub mentions: Vec<(usize, usize, usize, usize)>,
}

impl PreCoDocument {
    /// Convert to coreference document.
    #[must_use]
    pub fn to_coref_document(&self) -> CorefDocument {
        // Reconstruct text from sentences
        let mut text = String::new();
        let mut token_offsets: Vec<Vec<(usize, usize)>> = Vec::new();

        for sentence in &self.sentences {
            let mut sent_offsets = Vec::new();

            for (i, token) in sentence.iter().enumerate() {
                if i > 0 {
                    text.push(' ');
                }
                let start = text.len();
                text.push_str(token);
                let end = text.len();
                sent_offsets.push((start, end));
            }
            text.push(' ');
            token_offsets.push(sent_offsets);
        }

        // Group mentions by cluster
        let mut clusters: HashMap<usize, Vec<Mention>> = HashMap::new();

        for &(sent_idx, start_tok, end_tok, cluster_id) in &self.mentions {
            if sent_idx >= token_offsets.len() {
                continue;
            }
            let sent_tokens = &token_offsets[sent_idx];
            if start_tok >= sent_tokens.len() || end_tok > sent_tokens.len() {
                continue;
            }

            // Note: sent_tokens stores byte offsets (from text.len())
            let byte_start = sent_tokens[start_tok].0;
            let byte_end = sent_tokens[end_tok.saturating_sub(1).max(start_tok)].1;
            let mention_text = text[byte_start..byte_end].to_string();

            // Convert byte offsets to character offsets for Mention (which expects char offsets)
            let char_start = text[..byte_start].chars().count();
            let char_end = char_start + mention_text.chars().count();

            let mention = Mention::new(mention_text, char_start, char_end);
            clusters.entry(cluster_id).or_default().push(mention);
        }

        // Convert clusters to chains
        let chains: Vec<CorefChain> = clusters
            .into_iter()
            .map(|(id, mentions)| CorefChain::with_id(mentions, id as u64))
            .collect();

        let mut doc = CorefDocument::with_id(text, &self.id, chains);
        doc.includes_singletons = true;
        doc
    }
}

// =============================================================================
// Coreference Loader (delegates to DatasetLoader)
// =============================================================================

/// Loader for coreference datasets.
///
/// This is a thin wrapper around `DatasetLoader` that provides coreference-specific
/// loading methods. For most use cases, you can use `DatasetLoader::load_coref()` directly.
pub struct CorefLoader {
    inner: super::loader::DatasetLoader,
}

impl CorefLoader {
    /// Create a new loader with default cache directory.
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: super::loader::DatasetLoader::new()?,
        })
    }

    /// Create loader with custom cache directory.
    pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> Result<Self> {
        Ok(Self {
            inner: super::loader::DatasetLoader::with_cache_dir(cache_dir)?,
        })
    }

    /// Load GAP dataset as coreference documents.
    pub fn load_gap(&self) -> Result<Vec<CorefDocument>> {
        self.inner.load_coref(DatasetId::GAP)
    }

    /// Load GAP as raw examples (for detailed analysis).
    pub fn load_gap_examples(&self) -> Result<Vec<GapExample>> {
        let gap = super::loader::LoadableDatasetId::try_from(DatasetId::GAP)?;
        let cache_path = self.inner.cache_path(gap);

        if !cache_path.exists() {
            return Err(Error::InvalidInput(format!(
                "GAP dataset not cached at {:?}",
                cache_path
            )));
        }

        let content = fs::read_to_string(&cache_path)
            .map_err(|e| Error::InvalidInput(format!("Failed to read {:?}: {}", cache_path, e)))?;

        parse_gap_tsv(&content)
    }

    /// Load PreCo dataset as coreference documents.
    pub fn load_preco(&self) -> Result<Vec<CorefDocument>> {
        self.inner.load_coref(DatasetId::PreCo)
    }

    /// Read one LitBank BRAT annotation file and its same-stem .txt without using the cache.
    /// The document ID is the annotation file stem. Call once per selected book.
    pub fn load_litbank_from_path(
        &self,
        annotation_path: impl AsRef<std::path::Path>,
    ) -> Result<CorefDocument> {
        let path = annotation_path.as_ref();
        if path.extension().and_then(|e| e.to_str()) != Some("ann") {
            return Err(Error::InvalidInput(
                "LitBank requires a .ann path with a matching .txt file".into(),
            ));
        }
        let annotations = fs::read_to_string(path)
            .map_err(|e| Error::InvalidInput(format!("Failed to read {}: {e}", path.display())))?;
        let mut doc = super::litbank::SourcePair::from_annotations(path, annotations)?.coref()?;
        doc.doc_id = path
            .file_stem()
            .map(|name| name.to_string_lossy().into_owned());
        Ok(doc)
    }

    /// Load a CorefUD CoNLL-U file from an explicit local path (no caching, no network).
    ///
    /// This is the easiest way to run CorefUD experiments without wiring up downloads:
    /// download/extract the desired `*.conllu` locally and point this method at it.
    pub fn load_corefud_from_path(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Vec<CorefDocument>> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)
            .map_err(|e| Error::InvalidInput(format!("Failed to read {:?}: {}", path, e)))?;
        parse_corefud_conllu(&content)
    }

    /// Check if a coreference dataset is cached.
    #[must_use]
    pub fn is_cached(&self, id: DatasetId) -> bool {
        match super::loader::LoadableDatasetId::try_from(id) {
            Ok(loadable) => self.inner.is_cached(loadable),
            Err(_) => false,
        }
    }

    /// Get the underlying DatasetLoader.
    #[must_use]
    pub fn dataset_loader(&self) -> &super::loader::DatasetLoader {
        &self.inner
    }
}

impl Default for CorefLoader {
    fn default() -> Self {
        Self::new().expect("Failed to create CorefLoader")
    }
}

// =============================================================================
// Parsers
// =============================================================================

/// Parse GAP TSV format.
///
/// Format: `ID\tText\tPronoun\tPronoun-offset\tA\tA-offset\tA-coref\tB\tB-offset\tB-coref\tURL`
///
/// This function is public for use by `DatasetLoader`.
pub fn parse_gap_tsv(content: &str) -> Result<Vec<GapExample>> {
    let mut examples = Vec::new();
    let mut first_line = true;

    for line in content.lines() {
        // Skip header
        if first_line {
            first_line = false;
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 10 {
            continue;
        }

        let id = parts[0].to_string();
        let text = parts[1].to_string();
        let pronoun = parts[2].to_string();
        let pronoun_offset: usize = parts[3].parse().unwrap_or(0);
        let name_a = parts[4].to_string();
        let offset_a: usize = parts[5].parse().unwrap_or(0);
        let coref_a = parts[6].to_lowercase() == "true";
        let name_b = parts[7].to_string();
        let offset_b: usize = parts[8].parse().unwrap_or(0);
        let coref_b = parts[9].to_lowercase() == "true";
        let url = parts.get(10).map(|s| s.to_string());

        examples.push(GapExample {
            id,
            text,
            pronoun,
            pronoun_offset,
            name_a,
            offset_a,
            coref_a,
            name_b,
            offset_b,
            coref_b,
            url,
        });
    }

    Ok(examples)
}

#[derive(Deserialize)]
struct PreCoRecord {
    id: Option<String>,
    sentences: Vec<Vec<String>>,
    mention_clusters: Option<Vec<Vec<[usize; 3]>>>,
}

/// Both the generic sentence loader and coreference scorer must see the same records.
pub(super) fn preco_records<T: serde::de::DeserializeOwned>(content: &str) -> Result<Vec<T>> {
    let invalid =
        |err: serde_json::Error| Error::InvalidInput(format!("Invalid PreCo JSON: {err}"));
    let mut records = Vec::new();
    for item in serde_json::Deserializer::from_str(content).into_iter::<serde_json::Value>() {
        let value = item.map_err(invalid)?;
        match value {
            serde_json::Value::Array(array) => {
                for doc in array {
                    records.push(serde_json::from_value(doc).map_err(invalid)?);
                }
            }
            mut envelope if envelope.get("rows").is_some() => {
                super::loader::DatasetLoader::validate_hf_dataset_viewer_envelope(&envelope)?;
                let rows = envelope["rows"].as_array_mut().ok_or_else(|| {
                    Error::InvalidInput("PreCo Dataset Viewer payload requires a rows array".into())
                })?;
                for (idx, entry) in rows.iter_mut().enumerate() {
                    if entry
                        .get("truncated_cells")
                        .and_then(|v| v.as_array())
                        .is_some_and(|v| !v.is_empty())
                    {
                        return Err(Error::InvalidInput(format!(
                            "PreCo Dataset Viewer row {idx} contains truncated cells"
                        )));
                    }
                    let row = entry.get_mut("row").ok_or_else(|| {
                        Error::InvalidInput(format!("PreCo Dataset Viewer entry {idx} has no row"))
                    })?;
                    records.push(serde_json::from_value(row.take()).map_err(invalid)?);
                }
            }
            doc => records.push(serde_json::from_value(doc).map_err(invalid)?),
        }
    }
    if records.is_empty() {
        return Err(Error::InvalidInput(
            "PreCo JSON contains no documents".into(),
        ));
    }
    Ok(records)
}

/// Parse PreCo JSONL, arrays, and Hugging Face Dataset Viewer records with gold coreference.
pub fn parse_preco_json(content: &str) -> Result<Vec<PreCoDocument>> {
    let mut docs = Vec::new();
    for (idx, record) in preco_records::<PreCoRecord>(content)?
        .into_iter()
        .enumerate()
    {
        let id = record.id.unwrap_or_else(|| format!("doc_{idx}"));
        if !record
            .sentences
            .iter()
            .flatten()
            .any(|token| !token.is_empty())
        {
            return Err(Error::InvalidInput(format!("PreCo {id}: empty document")));
        }
        let clusters = record
            .mention_clusters
            .ok_or_else(|| Error::InvalidInput(format!("PreCo {id}: missing mention_clusters")))?;
        let mut mentions = Vec::new();
        let mut span_owners = HashMap::new();
        for (cluster_id, cluster) in clusters.into_iter().enumerate() {
            for [sent, start, end] in cluster {
                if start >= end || record.sentences.get(sent).is_none_or(|s| end > s.len()) {
                    return Err(Error::InvalidInput(format!(
                        "PreCo {id}: invalid mention [{sent}, {start}, {end}] in cluster {cluster_id}"
                    )));
                }
                if let Some(owner) = span_owners.insert((sent, start, end), cluster_id) {
                    return Err(Error::InvalidInput(format!(
                        "PreCo {id}: duplicate mention [{sent}, {start}, {end}] in clusters {owner} and {cluster_id}"
                    )));
                }
                mentions.push((sent, start, end, cluster_id));
            }
        }
        docs.push(PreCoDocument {
            id,
            sentences: record.sentences,
            mentions,
        });
    }
    Ok(docs)
}

// =============================================================================
// Synthetic Coreference Data
// =============================================================================

/// Generate synthetic coreference documents for testing.
///
/// Useful for validating metrics without downloading real datasets.
#[must_use]
pub fn synthetic_coref_dataset(num_docs: usize) -> Vec<CorefDocument> {
    let templates = [
        // Simple pronoun resolution
        (
            "John Smith went to the store. He bought some milk.",
            vec![
                ("John Smith", 0, 10, 0),
                ("He", 35, 37, 0),
            ],
        ),
        // Multiple entities
        (
            "Mary called Bob. She asked him about the meeting.",
            vec![
                ("Mary", 0, 4, 0),
                ("She", 17, 20, 0),
                ("Bob", 12, 15, 1),
                ("him", 27, 30, 1),
            ],
        ),
        // Longer chain
        (
            "The CEO announced the merger. She said the company would benefit. The executive was confident.",
            vec![
                ("The CEO", 0, 7, 0),
                ("She", 30, 33, 0),
                ("The executive", 68, 81, 0),
            ],
        ),
        // Nested mentions (company and its parts)
        (
            "Apple released a new iPhone. The tech giant's device sold well.",
            vec![
                ("Apple", 0, 5, 0),
                ("The tech giant", 29, 43, 0),
                ("iPhone", 21, 27, 1),
                ("device", 46, 52, 1),
            ],
        ),
        // Singletons
        (
            "The weather was nice. Sarah went for a walk in the park.",
            vec![
                ("The weather", 0, 11, 0),
                ("Sarah", 22, 27, 1),
                ("the park", 47, 55, 2),
            ],
        ),
    ];

    let mut docs = Vec::new();

    for i in 0..num_docs {
        let (text, mentions) = &templates[i % templates.len()];

        // Group mentions by cluster
        let mut clusters: HashMap<usize, Vec<Mention>> = HashMap::new();
        for &(mention_text, start, end, cluster_id) in mentions {
            let mention = Mention::new(mention_text, start, end);
            clusters.entry(cluster_id).or_default().push(mention);
        }

        let chains: Vec<CorefChain> = clusters
            .into_iter()
            .map(|(id, mentions)| CorefChain::with_id(mentions, id as u64))
            .collect();

        docs.push(CorefDocument::with_id(
            *text,
            format!("synthetic_{}", i),
            chains,
        ));
    }

    docs
}

/// Generate domain-specific synthetic coreference documents.
#[must_use]
pub fn domain_specific_coref_dataset(domain: &str) -> Vec<CorefDocument> {
    match domain {
        "biomedical" => biomedical_coref_examples(),
        "legal" => legal_coref_examples(),
        "news" => news_coref_examples(),
        _ => synthetic_coref_dataset(5),
    }
}

/// Biomedical coreference examples.
fn biomedical_coref_examples() -> Vec<CorefDocument> {
    vec![
        CorefDocument::with_id(
            "BRCA1 is a tumor suppressor gene. It plays a role in DNA repair. The gene is frequently mutated in breast cancer.",
            "bio_1",
            vec![CorefChain::new(vec![
                Mention::new("BRCA1", 0, 5),
                Mention::new("It", 34, 36),
                Mention::new("The gene", 62, 70),
            ])],
        ),
        CorefDocument::with_id(
            "The patient presented with chest pain. She was diagnosed with myocardial infarction. The woman received immediate treatment.",
            "bio_2",
            vec![
                CorefChain::new(vec![
                    Mention::new("The patient", 0, 11),
                    Mention::new("She", 39, 42),
                    Mention::new("The woman", 85, 94),
                ]),
                CorefChain::singleton(Mention::new("myocardial infarction", 62, 83)),
            ],
        ),
        CorefDocument::with_id(
            "Aspirin inhibits COX-1 and COX-2. The drug reduces inflammation. It is commonly used for pain relief.",
            "bio_3",
            vec![
                CorefChain::new(vec![
                    Mention::new("Aspirin", 0, 7),
                    Mention::new("The drug", 35, 43),
                    Mention::new("It", 65, 67),
                ]),
                CorefChain::singleton(Mention::new("COX-1", 17, 22)),
                CorefChain::singleton(Mention::new("COX-2", 27, 32)),
            ],
        ),
    ]
}

/// Legal coreference examples.
fn legal_coref_examples() -> Vec<CorefDocument> {
    vec![
        CorefDocument::with_id(
            "The defendant entered into a contract with the plaintiff. He failed to deliver the goods. The accused claimed force majeure.",
            "legal_1",
            vec![
                CorefChain::new(vec![
                    Mention::new("The defendant", 0, 13),
                    Mention::new("He", 58, 60),
                    Mention::new("The accused", 89, 100),
                ]),
                CorefChain::singleton(Mention::new("the plaintiff", 43, 56)),
            ],
        ),
        CorefDocument::with_id(
            "Article 5 of the Treaty governs this matter. It states that parties must negotiate in good faith. The provision has been interpreted broadly.",
            "legal_2",
            vec![CorefChain::new(vec![
                Mention::new("Article 5 of the Treaty", 0, 23),
                Mention::new("It", 45, 47),
                Mention::new("The provision", 99, 112),
            ])],
        ),
    ]
}

/// News coreference examples.
fn news_coref_examples() -> Vec<CorefDocument> {
    vec![
        CorefDocument::with_id(
            "President Biden met with Chancellor Scholz. The American leader discussed trade. He emphasized cooperation. Biden later held a press conference.",
            "news_1",
            vec![
                CorefChain::new(vec![
                    Mention::new("President Biden", 0, 14),
                    Mention::new("The American leader", 44, 63),
                    Mention::new("He", 81, 83),
                    Mention::new("Biden", 107, 112),
                ]),
                CorefChain::singleton(Mention::new("Chancellor Scholz", 25, 42)),
            ],
        ),
        CorefDocument::with_id(
            "Nvidia announced record quarterly earnings. The chipmaker exceeded expectations. Its stock rose 5% in after-hours trading.",
            "news_2",
            vec![
                CorefChain::new(vec![
                    Mention::new("Nvidia", 0, 6),
                    Mention::new("The chipmaker", 44, 57),
                    Mention::new("Its", 80, 83),
                ]),
            ],
        ),
        CorefDocument::with_id(
            "The hurricane made landfall in Florida. It caused widespread damage. The storm was Category 4. Authorities ordered evacuations before it arrived.",
            "news_3",
            vec![
                CorefChain::new(vec![
                    Mention::new("The hurricane", 0, 13),
                    Mention::new("It", 40, 42),
                    Mention::new("The storm", 68, 77),
                    Mention::new("it", 133, 135),
                ]),
            ],
        ),
    ]
}

/// Generate adversarial coreference examples.
///
/// These stress-test edge cases in coreference metrics.
#[must_use]
pub fn adversarial_coref_examples() -> Vec<(CorefDocument, CorefDocument, &'static str)> {
    vec![
        // Over-clustering: system merges two distinct entities
        (
            CorefDocument::new(
                "John saw Mary. He waved.",
                vec![
                    CorefChain::new(vec![Mention::new("John", 0, 4), Mention::new("He", 15, 17)]),
                    CorefChain::singleton(Mention::new("Mary", 9, 13)),
                ],
            ),
            CorefDocument::new(
                "John saw Mary. He waved.",
                vec![CorefChain::new(vec![
                    Mention::new("John", 0, 4),
                    Mention::new("Mary", 9, 13),
                    Mention::new("He", 15, 17),
                ])],
            ),
            "over-clustering",
        ),
        // Under-clustering: system splits one entity
        (
            CorefDocument::new(
                "Barack Obama gave a speech. The president was eloquent. Obama smiled.",
                vec![CorefChain::new(vec![
                    Mention::new("Barack Obama", 0, 12),
                    Mention::new("The president", 28, 41),
                    Mention::new("Obama", 56, 61),
                ])],
            ),
            CorefDocument::new(
                "Barack Obama gave a speech. The president was eloquent. Obama smiled.",
                vec![
                    CorefChain::new(vec![
                        Mention::new("Barack Obama", 0, 12),
                        Mention::new("Obama", 56, 61),
                    ]),
                    CorefChain::singleton(Mention::new("The president", 28, 41)),
                ],
            ),
            "under-clustering",
        ),
        // Missed mention: system finds fewer mentions
        (
            CorefDocument::new(
                "The dog ran. It was fast. The animal stopped.",
                vec![CorefChain::new(vec![
                    Mention::new("The dog", 0, 7),
                    Mention::new("It", 13, 15),
                    Mention::new("The animal", 26, 36),
                ])],
            ),
            CorefDocument::new(
                "The dog ran. It was fast. The animal stopped.",
                vec![CorefChain::new(vec![
                    Mention::new("The dog", 0, 7),
                    Mention::new("It", 13, 15),
                ])], // Missing "The animal"
            ),
            "missed-mention",
        ),
        // All singletons vs all in one cluster
        (
            CorefDocument::new(
                "A B C",
                vec![
                    CorefChain::singleton(Mention::new("A", 0, 1)),
                    CorefChain::singleton(Mention::new("B", 2, 3)),
                    CorefChain::singleton(Mention::new("C", 4, 5)),
                ],
            ),
            CorefDocument::new(
                "A B C",
                vec![CorefChain::new(vec![
                    Mention::new("A", 0, 1),
                    Mention::new("B", 2, 3),
                    Mention::new("C", 4, 5),
                ])],
            ),
            "singletons-vs-one-cluster",
        ),
    ]
}

// =============================================================================
// BookCoref Support
// =============================================================================

/// Parse BookCoref JSON/JSONL format.
///
/// BookCoref (Martinelli et al. 2025) provides book-scale coreference data.
///
/// The format follows OntoNotes-style with character metadata:
/// ```json
/// {
///   "doc_key": "pride_and_prejudice_1342",
///   "gutenberg_key": "1342",
///   "sentences": [["CHAPTER", "I."], ["It", "is", "a", "truth", ...], ...],
///   "clusters": [[[79,80], [81,82], ...], [[2727,2728], ...], ...],
///   "characters": [{"name": "Mr Bennet", "cluster": [[79,80], ...]}, ...]
/// }
/// ```
///
/// - `sentences`: nested arrays of tokens (word-tokenized)
/// - `clusters`: list of clusters, each cluster is list of [start, end] token spans (inclusive)
/// - `characters`: optional character metadata (not used for coreference eval)
///
/// # Errors
///
/// Returns error if JSON is malformed or has invalid structure.
pub fn parse_bookcoref_json(content: &str) -> Result<Vec<CorefDocument>> {
    let mut documents = Vec::new();

    // Parse as JSONL (one JSON object per line) or JSON array
    if content.trim().starts_with('[') {
        // JSON array format
        let parsed: Vec<serde_json::Value> = serde_json::from_str(content).map_err(|e| {
            Error::InvalidInput(format!("Failed to parse BookCoref JSON array: {}", e))
        })?;
        for item in parsed {
            if let Some(doc) = parse_bookcoref_item(&item)? {
                documents.push(doc);
            }
        }
    } else {
        // JSONL format - one JSON per line
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let item: serde_json::Value = serde_json::from_str(line).map_err(|e| {
                Error::InvalidInput(format!("Failed to parse BookCoref JSONL: {}", e))
            })?;

            if let Some(doc) = parse_bookcoref_item(&item)? {
                documents.push(doc);
            }
        }
    }

    if documents.is_empty() {
        return Err(Error::InvalidInput(
            "BookCoref content contains no valid documents".to_string(),
        ));
    }

    Ok(documents)
}

/// Parse a single BookCoref item.
fn parse_bookcoref_item(item: &serde_json::Value) -> Result<Option<CorefDocument>> {
    // Get sentences array
    let sentences = match item.get("sentences").and_then(|v| v.as_array()) {
        Some(s) => s,
        None => return Ok(None),
    };

    // Get clusters array
    let clusters = match item.get("clusters").and_then(|v| v.as_array()) {
        Some(c) => c,
        None => return Ok(None),
    };

    // Flatten sentences to get tokens and build token-to-char offset map
    let mut tokens: Vec<String> = Vec::new();
    for sentence in sentences {
        if let Some(sent_tokens) = sentence.as_array() {
            for token in sent_tokens {
                if let Some(t) = token.as_str() {
                    tokens.push(t.to_string());
                }
            }
        }
    }

    if tokens.is_empty() {
        return Ok(None);
    }

    // Build text and token offset map
    // We need to reconstruct text from tokens (space-separated)
    let mut text = String::new();
    let mut token_char_starts: Vec<usize> = Vec::new();
    let mut token_char_ends: Vec<usize> = Vec::new();

    for (i, token) in tokens.iter().enumerate() {
        if i > 0 {
            text.push(' ');
        }
        let start = text.chars().count();
        text.push_str(token);
        let end = text.chars().count();
        token_char_starts.push(start);
        token_char_ends.push(end);
    }

    // Parse clusters - each cluster is a list of [start_token, end_token] spans (inclusive)
    let mut coref_chains = Vec::new();
    for cluster in clusters {
        if let Some(spans) = cluster.as_array() {
            let mut mentions = Vec::new();
            for span in spans {
                if let Some(span_arr) = span.as_array() {
                    if span_arr.len() >= 2 {
                        let start_tok = span_arr[0].as_u64().unwrap_or(0) as usize;
                        let end_tok = span_arr[1].as_u64().unwrap_or(0) as usize;

                        // Convert token indices to char offsets
                        if start_tok < token_char_starts.len() && end_tok < token_char_ends.len() {
                            let char_start = token_char_starts[start_tok];
                            let char_end = token_char_ends[end_tok];

                            // Extract mention text
                            let mention_text: String = text
                                .chars()
                                .skip(char_start)
                                .take(char_end - char_start)
                                .collect();

                            mentions.push(Mention::new(&mention_text, char_start, char_end));
                        }
                    }
                }
            }

            if !mentions.is_empty() {
                coref_chains.push(CorefChain::new(mentions));
            }
        }
    }

    Ok(Some(CorefDocument::new(&text, coref_chains)))
}

// =============================================================================
// ECB+ Coreference Parser
// =============================================================================

/// Read the legacy ECB+ token-and-chain CSV export.
/// For original XML bytes use [`parse_ecb_plus_zip`]. The three-column published
/// sentence index supplies a selection to [`parse_ecb_plus_selected_gold_zip`].
pub fn parse_ecb_plus_coref(content: &str) -> Result<Vec<CorefDocument>> {
    // Check for ZIP magic bytes at the start
    if content.as_bytes().starts_with(b"PK\x03\x04") {
        return parse_ecb_plus_zip(content.as_bytes());
    }
    parse_ecb_plus_token_csv(content)
}

/// An ECB+ chain with its original cross-document identifier, when explicitly provided.
#[derive(Debug, Clone)]
pub struct EcbPlusChain {
    /// Opaque instance identifier. Compare it only within the same known ECB+ topic.
    pub cross_document_id: Option<String>,
    /// Mention text joins the actual anchored tokens in source order. Offsets enclose
    /// the first through last anchor, including any intervening unanchored words.
    pub chain: CorefChain,
}

/// Gold coreference annotations from one ECB+ XML document.
#[derive(Debug, Clone)]
pub struct EcbPlusDocument {
    /// Topic number from the archive directory or document filename, if available.
    pub topic: Option<String>,
    /// Document identifier used by the generic coreference loader.
    pub doc_id: String,
    /// Text reconstructed from the XML tokens.
    pub text: String,
    /// Cross-document, intra-document and singleton chains.
    pub chains: Vec<EcbPlusChain>,
}

impl EcbPlusDocument {
    /// Convert to the generic document representation, which only supports numeric cluster IDs.
    pub fn into_coref_document(self) -> CorefDocument {
        let mut doc = CorefDocument::with_id(
            self.text,
            self.doc_id,
            self.chains
                .into_iter()
                .map(|annotated| annotated.chain)
                .collect(),
        );
        doc.includes_singletons = true;
        doc
    }
}

/// Parse the original ECB+ XML ZIP into generic within-document chains.
/// Use [`parse_ecb_plus_gold_zip`] to retain opaque cross-document IDs.
pub fn parse_ecb_plus_zip(data: &[u8]) -> Result<Vec<CorefDocument>> {
    Ok(parse_ecb_plus_gold_zip(data)?
        .into_iter()
        .map(EcbPlusDocument::into_coref_document)
        .collect())
}

/// Read the original XML ZIP, including all anchored markables from all sentences.
/// The upstream selected-sentence CSV supplies a separate benchmark filter.
pub fn parse_ecb_plus_gold_zip(data: &[u8]) -> Result<Vec<EcbPlusDocument>> {
    read_ecb_plus_gold_zip(data, None)
}

type EcbSentenceIndex = BTreeMap<(String, String), BTreeSet<u32>>;

/// Project the original XML annotations onto the supplied ECB+ sentence-index CSV.
/// Document text and character offsets still refer to the whole document. A mention
/// is included only when every anchored token belongs to a selected sentence.
pub fn parse_ecb_plus_selected_gold_zip(
    data: &[u8],
    sentence_index: &str,
) -> Result<Vec<EcbPlusDocument>> {
    let mut lines = sentence_index.trim_start_matches('\u{feff}').lines();
    if lines.next().map(str::trim) != Some("Topic,File,Sentence Number") {
        return Err(Error::InvalidInput(
            "ECB+ sentence index requires the header Topic,File,Sentence Number".into(),
        ));
    }
    let mut index = EcbSentenceIndex::new();
    for (line_number, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut fields = line.split(',').map(str::trim);
        let invalid = || {
            Error::InvalidInput(format!(
                "Invalid ECB+ sentence-index row {}",
                line_number + 2
            ))
        };
        let topic = fields
            .next()
            .filter(|field| !field.is_empty() && field.bytes().all(|c| c.is_ascii_digit()))
            .ok_or_else(invalid)?;
        let file = fields
            .next()
            .filter(|field| !field.is_empty() && field.bytes().all(|c| c.is_ascii_alphanumeric()))
            .ok_or_else(invalid)?;
        let sentence = fields
            .next()
            .and_then(|field| field.parse::<u32>().ok())
            .ok_or_else(invalid)?;
        if fields.next().is_some() {
            return Err(invalid());
        }
        let doc_name = format!("{topic}_{file}");
        if !index
            .entry((topic.to_owned(), doc_name))
            .or_default()
            .insert(sentence)
        {
            return Err(Error::InvalidInput(format!(
                "Duplicate ECB+ sentence-index coordinate {topic},{file},{sentence}"
            )));
        }
    }
    if index.is_empty() {
        return Err(Error::InvalidInput(
            "ECB+ sentence index contains no selected sentences".into(),
        ));
    }
    read_ecb_plus_gold_zip(data, Some(&index))
}

fn read_ecb_plus_gold_zip(
    data: &[u8],
    selection: Option<&EcbSentenceIndex>,
) -> Result<Vec<EcbPlusDocument>> {
    use std::io::Cursor;

    let cursor = Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| Error::InvalidInput(format!("Failed to open ECB+ zip: {}", e)))?;

    let mut all_docs = Vec::new();
    let mut selected_docs = BTreeSet::new();

    for i in 0..archive.len() {
        let Ok(mut file) = archive.by_index(i) else {
            continue;
        };
        let name = match file.name() {
            Ok(name) if name.ends_with(".xml") => name.into_owned(),
            _ => continue,
        };

        let mut components = name.split('/').rev();
        let doc_name = components.next().unwrap_or(&name).trim_end_matches(".xml");
        let is_topic =
            |topic: &&str| !topic.is_empty() && topic.bytes().all(|c| c.is_ascii_digit());
        let topic = components.next().filter(is_topic).or_else(|| {
            doc_name
                .split_once('_')
                .map(|(topic, _)| topic)
                .filter(is_topic)
        });

        let selected_sentences = if let Some(index) = selection {
            let Some(topic) = topic else { continue };
            let key = (topic.to_owned(), doc_name.to_owned());
            let Some(sentences) = index.get(&key) else {
                continue;
            };
            if !selected_docs.insert(key) {
                return Err(Error::InvalidInput(format!(
                    "Duplicate selected ECB+ document {name}"
                )));
            }
            Some(sentences)
        } else {
            None
        };
        let mut xml_content = String::new();
        std::io::Read::read_to_string(&mut file, &mut xml_content)
            .map_err(|e| Error::InvalidInput(format!("Failed to read XML {}: {}", name, e)))?;

        match parse_ecb_plus_xml(&xml_content, topic, doc_name, selected_sentences) {
            Ok(doc) => all_docs.push(doc),
            Err(e) if selection.is_some() => return Err(e),
            Err(e) => log::debug!("Skipping {}: {}", name, e),
        }
    }

    if let Some(index) = selection {
        if let Some((topic, name)) = index.keys().find(|key| !selected_docs.contains(*key)) {
            return Err(Error::InvalidInput(format!(
                "Missing selected ECB+ document {topic}/{name}"
            )));
        }
    }

    if all_docs.is_empty() {
        return Err(Error::InvalidInput(
            "ECB+ zip contains no parseable XML documents".to_string(),
        ));
    }

    // Sort by (topic, doc) for deterministic order
    all_docs.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
    Ok(all_docs)
}

/// Decode the markable IDs, relation IDs and cross-document instance IDs in their own scopes.
fn parse_ecb_plus_xml(
    xml: &str,
    topic: Option<&str>,
    doc_name: &str,
    selection: Option<&BTreeSet<u32>>,
) -> Result<EcbPlusDocument> {
    use quick_xml::events::{BytesStart, Event};
    use quick_xml::Reader;
    use std::collections::{BTreeMap, BTreeSet, HashSet};

    fn attr(e: &BytesStart<'_>, name: &[u8]) -> Option<String> {
        e.try_get_attribute(name)
            .ok()??
            .normalized_value(quick_xml::XmlVersion::Implicit1_0)
            .ok()
            .map(|value| value.into_owned())
    }
    fn id(e: &BytesStart<'_>, name: &[u8]) -> Option<u32> {
        attr(e, name)?.parse().ok()
    }
    fn label(value: Option<String>) -> Option<String> {
        value.filter(|value| !value.trim().is_empty())
    }
    fn entity_type(tag: &[u8]) -> Option<&'static str> {
        if tag.starts_with(b"ACTION_") || tag.starts_with(b"NEG_ACTION_") {
            Some("EVENT")
        } else if tag.starts_with(b"TIME_") {
            Some("TIME")
        } else if tag.starts_with(b"LOC_") {
            Some("LOC")
        } else if tag == b"HUMAN_PART_PER" {
            Some("PERSON")
        } else if tag == b"HUMAN_PART_ORG" {
            Some("ORG")
        } else if tag.starts_with(b"HUMAN_PART") || tag.starts_with(b"NON_HUMAN_PART") {
            Some("PARTICIPANT")
        } else {
            None
        }
    }

    #[derive(Default)]
    struct Relation {
        cross: bool,
        note: Option<String>,
        target: Option<u32>,
        members: BTreeSet<u32>,
    }

    let mut tokens = BTreeMap::new();
    let mut seen_sentences = BTreeSet::new();
    let mut selected_tokens = HashSet::new();
    let mut token: Option<(u32, String)> = None;
    let mut mentions = BTreeMap::new();
    let mut markable: Option<(u32, &'static str, Vec<u32>)> = None;
    let mut descriptors = HashMap::new();
    let mut relations = Vec::new();
    let mut relation: Option<Relation> = None;

    let mut reader = Reader::from_str(xml);
    // Expand self-closing tags so anchored markables and empty descriptors have the same lifecycle.
    reader.config_mut().expand_empty_elements = true;
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                let name = e.name();
                let tag = name.as_ref();
                match tag {
                    b"token" => {
                        token = id(&e, b"t_id").map(|tid| (tid, String::new()));
                        if let (Some(index), Some((tid, _))) = (selection, &token) {
                            let sentence = id(&e, b"sentence").ok_or_else(|| {
                                Error::InvalidInput(format!(
                                    "Missing sentence number for ECB+ token {tid} in {doc_name}"
                                ))
                            })?;
                            seen_sentences.insert(sentence);
                            if index.contains(&sentence) {
                                selected_tokens.insert(*tid);
                            }
                        }
                    }
                    b"token_anchor" => {
                        if let (Some((_, _, anchors)), Some(tid)) = (&mut markable, id(&e, b"t_id"))
                        {
                            anchors.push(tid);
                        }
                    }
                    b"CROSS_DOC_COREF" | b"INTRA_DOC_COREF" => {
                        relation = Some(Relation {
                            cross: tag == b"CROSS_DOC_COREF",
                            note: label(attr(&e, b"note")),
                            ..Relation::default()
                        });
                    }
                    b"source" | b"target" => {
                        if let (Some(r), Some(mid)) = (&mut relation, id(&e, b"m_id")) {
                            r.members.insert(mid);
                            if tag == b"target" {
                                r.target = Some(mid);
                            }
                        }
                    }
                    _ => {
                        if let Some(mid) = id(&e, b"m_id") {
                            if let Some(instance) = label(attr(&e, b"instance_id")) {
                                descriptors.insert(mid, instance);
                            }
                            if let Some(kind) = entity_type(tag) {
                                markable = Some((mid, kind, Vec::new()));
                            }
                        }
                    }
                }
            }
            Ok(Event::Text(e)) => {
                if let Some((_, text)) = &mut token {
                    text.push_str(&e.xml10_content().unwrap_or_default());
                }
            }
            Ok(Event::GeneralRef(e)) => {
                if let Some((_, text)) = &mut token {
                    if let Ok(Some(ch)) = e.resolve_char_ref() {
                        text.push(ch);
                    } else if let Some(expanded) = quick_xml::escape::resolve_predefined_entity(
                        &e.decode().unwrap_or_default(),
                    ) {
                        text.push_str(expanded);
                    }
                }
            }
            Ok(Event::End(e)) => match e.name().as_ref() {
                b"token" => {
                    if let Some((tid, value)) = token.take() {
                        tokens.insert(tid, value);
                    }
                }
                b"CROSS_DOC_COREF" | b"INTRA_DOC_COREF" => {
                    if let Some(r) = relation.take() {
                        relations.push(r);
                    }
                }
                tag if entity_type(tag).is_some() => {
                    if let Some((mid, kind, mut anchors)) = markable.take() {
                        if !anchors.is_empty() {
                            anchors.sort_unstable();
                            anchors.dedup();
                            mentions.insert(mid, (kind, anchors));
                        }
                    }
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(Error::InvalidInput(format!(
                    "XML parse error in {doc_name}: {e}"
                )))
            }
            _ => {}
        }
    }
    if tokens.is_empty() {
        return Err(Error::InvalidInput(format!(
            "ECB+ XML {doc_name} contains no tokens"
        )));
    }

    if let Some(index) = selection {
        if let Some(sentence) = index.difference(&seen_sentences).next() {
            return Err(Error::InvalidInput(format!(
                "Missing selected ECB+ sentence {sentence} in {doc_name}"
            )));
        }
        mentions.retain(|_, (_, anchors)| anchors.iter().all(|tid| selected_tokens.contains(tid)));
    }

    let mut text = String::new();
    let mut offsets = HashMap::new();
    let mut char_end = 0;
    for (tid, word) in tokens {
        if !text.is_empty() {
            text.push(' ');
            char_end += 1;
        }
        let char_start = char_end;
        let byte_start = text.len();
        char_end += word.chars().count();
        text.push_str(&word);
        offsets.insert(tid, (char_start, char_end, byte_start, text.len()));
    }

    let mut shared: BTreeMap<String, BTreeSet<u32>> = BTreeMap::new();
    let mut local = Vec::new();
    let mut linked = HashSet::new();
    for r in relations {
        linked.extend(r.members.iter().copied());
        // r_id and m_id refer to this XML document only. Only an explicit cross-document
        // relation's note or target descriptor identifies an instance shared by other files.
        let instance = if r.cross {
            r.note.or_else(|| descriptors.get(&r.target?).cloned())
        } else {
            None
        };
        if let Some(instance) = instance {
            shared.entry(instance).or_default().extend(r.members);
        } else {
            local.push(r.members);
        }
    }
    for &mid in mentions.keys() {
        if !linked.contains(&mid) {
            local.push(BTreeSet::from([mid]));
        }
    }

    let mut chains = Vec::new();
    for (cross_document_id, members) in shared
        .into_iter()
        .map(|(id, m)| (Some(id), m))
        .chain(local.into_iter().map(|m| (None, m)))
    {
        let mut spans = Vec::new();
        for mid in members {
            let Some((kind, anchors)) = mentions.get(&mid) else {
                continue;
            };
            let mut anchored = anchors.iter().filter_map(|tid| offsets.get(tid));
            let Some(&(start, mut end, first, last)) = anchored.next() else {
                continue;
            };
            let mut surface = text[first..last].to_owned();
            for &(_, token_end, first, last) in anchored {
                surface.push(' ');
                surface.push_str(&text[first..last]);
                end = token_end;
            }
            let mut mention = Mention::new(surface, start, end);
            mention.entity_type = Some((*kind).into());
            spans.push(mention);
        }
        if !spans.is_empty() {
            let mut chain = CorefChain::new(spans);
            chain.cluster_id = cross_document_id
                .as_deref()
                .and_then(|s| s.parse::<u64>().ok())
                .map(Into::into);
            chains.push(EcbPlusChain {
                cross_document_id,
                chain,
            });
        }
    }
    Ok(EcbPlusDocument {
        doc_id: format!("{}_{doc_name}", topic.unwrap_or("unknown")),
        topic: topic.map(str::to_owned),
        text,
        chains,
    })
}

/// Parse a legacy token-level export. The supported columns are
/// `Topic,File,Sentence Number,Token Number,Token,Lemma,Event Mention,Coreference Chain`.
///
/// Documents are identified by `(topic, file)` pairs. Tokens are grouped into
/// sentences, and coreference chains are built from the `Coreference Chain` column.
/// A non-empty chain value indicates the token belongs to that coref chain.
///
/// Returns one `CorefDocument` per (topic, file).
fn parse_ecb_plus_token_csv(content: &str) -> Result<Vec<CorefDocument>> {
    // (topic, file) -> DocAcc
    let mut docs: HashMap<(String, String), ecb_plus_acc::DocAcc> = HashMap::new();

    let mut lines = content.lines();

    // Skip header line
    if let Some(header) = lines.next() {
        // Validate it looks like an ECB+ header
        let lower = header.to_lowercase();
        if !lower.contains("token") && !lower.contains("topic") {
            // Not a header -- could be data; we'll be lenient and try parsing it below
            // by not consuming it (but we already called next(), so re-parse this line)
            parse_ecb_plus_line(header, &mut docs);
        }
    }

    for line in lines {
        parse_ecb_plus_line(line, &mut docs);
    }

    if docs.is_empty() {
        return Err(Error::InvalidInput(
            "ECB+ CSV contains no valid token rows".to_string(),
        ));
    }

    // Convert accumulated data into CorefDocuments
    let mut result: Vec<CorefDocument> = Vec::new();

    // Sort by key for deterministic order
    let mut doc_keys: Vec<_> = docs.keys().cloned().collect();
    doc_keys.sort();

    for key in doc_keys {
        let acc = docs.remove(&key).unwrap();
        let (topic, file) = &key;

        // Reconstruct text from tokens in order
        let mut text = String::new();
        let mut token_char_offsets: HashMap<(u32, u32), (usize, usize)> = HashMap::new();

        for (&(sent, tok), token_text) in &acc.tokens {
            let start = text.chars().count();
            text.push_str(token_text);
            let end = text.chars().count();
            token_char_offsets.insert((sent, tok), (start, end));
            text.push(' ');
        }

        // Build coref chains from chain mappings
        let mut coref_chains: Vec<CorefChain> = Vec::new();
        let mut chain_ids: Vec<_> = acc.chains.keys().cloned().collect();
        chain_ids.sort();

        for chain_id in chain_ids {
            let token_positions = &acc.chains[&chain_id];
            let mentions: Vec<Mention> = token_positions
                .iter()
                .filter_map(|pos| {
                    let (start, end) = token_char_offsets.get(pos)?;
                    let token_text = acc.tokens.get(pos)?;
                    Some(Mention {
                        text: token_text.clone(),
                        start: *start,
                        end: *end,
                        head_start: None,
                        head_end: None,
                        entity_type: Some("EVENT".to_string()),
                        mention_type: Some(MentionType::Proper),
                    })
                })
                .collect();

            if !mentions.is_empty() {
                coref_chains.push(CorefChain::new(mentions));
            }
        }

        // Encode topic in doc_id so downstream can extract it via split('_')
        let doc = CorefDocument::with_id(&text, format!("{}_{}", topic, file), coref_chains);
        result.push(doc);
    }

    Ok(result)
}

/// Parse a single ECB+ CSV line into the document accumulators.
fn parse_ecb_plus_line(line: &str, docs: &mut HashMap<(String, String), ecb_plus_acc::DocAcc>) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }

    // CSV split (simple: ECB+ doesn't have quoted fields with commas)
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 5 {
        return;
    }

    let topic = parts[0].trim().to_string();
    let file = parts[1].trim().to_string();
    let sent_num: u32 = match parts[2].trim().parse() {
        Ok(n) => n,
        Err(_) => return, // Skip non-numeric (e.g. malformed rows)
    };
    let tok_num: u32 = match parts[3].trim().parse() {
        Ok(n) => n,
        Err(_) => return,
    };
    let token = parts[4].trim().to_string();

    // Coreference chain is typically the last column (index 7), but may vary.
    // Look for a non-empty chain value in columns after the token.
    let chain_id = parts
        .get(7)
        .or_else(|| parts.get(6))
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && *s != "-" && *s != "_")
        .map(|s| s.to_string());

    let acc = docs.entry((topic, file)).or_default();
    acc.tokens.insert((sent_num, tok_num), token);
    if let Some(cid) = chain_id {
        acc.chains.entry(cid).or_default().push((sent_num, tok_num));
    }
}

/// Internal types for ECB+ accumulation (avoids polluting module namespace).
mod ecb_plus_acc {
    use std::collections::{BTreeMap, HashMap};

    #[derive(Default)]
    pub struct DocAcc {
        pub tokens: BTreeMap<(u32, u32), String>,
        pub chains: HashMap<String, Vec<(u32, u32)>>,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preco_mentions_have_one_owner_but_may_nest() {
        let outer = [0, 0, 2];
        let inner = [0, 1, 2];
        for (clusters, duplicate) in [
            (vec![vec![outer], vec![inner]], false),
            (vec![vec![outer, outer], vec![inner]], true),
            (vec![vec![outer], vec![inner, outer]], true),
        ] {
            let record =
                serde_json::json!({"sentences":[["Mayor","Jo"]], "mention_clusters":clusters});
            let parsed = parse_preco_json(&record.to_string());
            if duplicate {
                assert!(parsed
                    .unwrap_err()
                    .to_string()
                    .contains("duplicate mention"));
            } else {
                assert_eq!(parsed.unwrap()[0].mentions.len(), 2);
            }
        }
    }

    /// GAP offsets are character offsets, so a mention ends at offset plus the
    /// name's character count, not its byte length.
    #[test]
    fn gap_mention_end_uses_character_length() {
        let ex = GapExample {
            id: "t".into(),
            text: "Zoë met Ana. She smiled.".into(),
            pronoun: "She".into(),
            pronoun_offset: 13,
            name_a: "Zoë".into(),
            offset_a: 0,
            coref_a: true,
            name_b: "Ana".into(),
            offset_b: 8,
            coref_b: false,
            url: None,
        };
        let doc = ex.to_coref_document();
        let zoe = doc
            .chains
            .iter()
            .flat_map(|c| c.mentions.iter())
            .find(|m| m.text == "Zoë")
            .unwrap();
        assert_eq!((zoe.start, zoe.end), (0, 3));
    }

    #[test]
    fn test_gap_example_to_coref() {
        let example = GapExample {
            id: "test-1".to_string(),
            text: "John saw Mary. He waved.".to_string(),
            pronoun: "He".to_string(),
            pronoun_offset: 15,
            name_a: "John".to_string(),
            offset_a: 0,
            coref_a: true,
            name_b: "Mary".to_string(),
            offset_b: 9,
            coref_b: false,
            url: None,
        };

        let doc = example.to_coref_document();
        assert_eq!(doc.mention_count(), 3);
        assert_eq!(doc.chain_count(), 2); // John+He, Mary
    }

    #[test]
    fn test_gap_example_mention_types() {
        use crate::eval::coref::MentionType;

        let example = GapExample {
            id: "test-2".to_string(),
            text: "Alice met Bob. She smiled.".to_string(),
            pronoun: "She".to_string(),
            pronoun_offset: 15,
            name_a: "Alice".to_string(),
            offset_a: 0,
            coref_a: true,
            name_b: "Bob".to_string(),
            offset_b: 10,
            coref_b: false,
            url: None,
        };

        let doc = example.to_coref_document();

        // Verify mention types are correctly assigned
        let all_mentions: Vec<_> = doc.chains.iter().flat_map(|c| &c.mentions).collect();
        assert_eq!(all_mentions.len(), 3);

        // Proper nouns: Alice, Bob
        let proper_count = all_mentions
            .iter()
            .filter(|m| m.mention_type == Some(MentionType::Proper))
            .count();
        assert_eq!(
            proper_count, 2,
            "Should have 2 proper noun mentions (Alice, Bob)"
        );

        // Pronominal: She
        let pronominal_count = all_mentions
            .iter()
            .filter(|m| m.mention_type == Some(MentionType::Pronominal))
            .count();
        assert_eq!(
            pronominal_count, 1,
            "Should have 1 pronominal mention (She)"
        );
    }

    #[test]
    fn test_synthetic_coref_dataset() {
        let docs = synthetic_coref_dataset(5);
        assert_eq!(docs.len(), 5);

        for doc in &docs {
            assert!(!doc.text.is_empty());
            assert!(!doc.chains.is_empty());
        }
    }

    #[test]
    fn test_adversarial_examples() {
        let examples = adversarial_coref_examples();
        assert!(!examples.is_empty());

        for (gold, pred, name) in &examples {
            assert!(!gold.chains.is_empty(), "Gold chains empty for {}", name);
            assert!(!pred.chains.is_empty(), "Pred chains empty for {}", name);
        }
    }

    #[test]
    fn test_gap_tsv_parsing() {
        let tsv = "ID\tText\tPronoun\tPronoun-offset\tA\tA-offset\tA-coref\tB\tB-offset\tB-coref\tURL\n\
                   1\tJohn saw Mary. He waved.\tHe\t15\tJohn\t0\tTRUE\tMary\t9\tFALSE\thttps://example.com";

        let examples = parse_gap_tsv(tsv).unwrap();
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].id, "1");
        assert!(examples[0].coref_a);
        assert!(!examples[0].coref_b);
    }

    #[test]
    fn test_bookcoref_json_parsing() {
        // BookCoref format: nested sentences, token-span clusters
        // Use single line to test JSONL parsing
        let json = r#"{"doc_key": "test_book_1", "gutenberg_key": "1", "sentences": [["Alice", "met", "Bob", "."], ["She", "waved", "."]], "clusters": [[[0, 0], [4, 4]], [[2, 2]]], "characters": [{"name": "Alice", "cluster": [[0, 0], [4, 4]]}]}"#;

        let docs = parse_bookcoref_json(json).unwrap();
        assert_eq!(docs.len(), 1);

        let doc = &docs[0];
        // Text should be tokens joined by space
        assert!(doc.text.contains("Alice"));
        assert!(doc.text.contains("She"));

        // Should have 2 clusters: Alice+She, Bob
        assert_eq!(doc.chain_count(), 2);

        // First cluster should have 2 mentions (Alice, She)
        let alice_cluster = doc
            .chains
            .iter()
            .find(|c| c.mentions.iter().any(|m| m.text == "Alice"));
        assert!(alice_cluster.is_some());
        assert_eq!(alice_cluster.unwrap().mentions.len(), 2);

        // Second cluster should have 1 mention (Bob)
        let bob_cluster = doc
            .chains
            .iter()
            .find(|c| c.mentions.iter().any(|m| m.text == "Bob"));
        assert!(bob_cluster.is_some());
        assert_eq!(bob_cluster.unwrap().mentions.len(), 1);
    }

    #[test]
    fn test_bookcoref_json_array_parsing() {
        // Test JSON array format
        let json_array = r#"[{"doc_key": "book1", "sentences": [["He", "ran", "."]], "clusters": [[[0, 0]]]}, {"doc_key": "book2", "sentences": [["She", "walked", "."]], "clusters": [[[0, 0]]]}]"#;

        let docs = parse_bookcoref_json(json_array).unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_bookcoref_jsonl_parsing() {
        // Test JSONL format (one JSON per line)
        let jsonl = r#"{"doc_key": "book1", "sentences": [["He", "ran", "."]], "clusters": [[[0, 0]]]}
{"doc_key": "book2", "sentences": [["She", "walked", "."]], "clusters": [[[0, 0]]]}"#;

        let docs = parse_bookcoref_json(jsonl).unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_corefud_conllu_parsing_multilingual_and_zero() {
        // A tiny CorefUD-like CoNLL-U with:
        // - multiple documents via # newdoc id
        // - multi-script tokens
        // - one empty node (ID with .) representing a zero mention
        //
        // Entity bracketing follows CorefUD 1.0/1.2 format:
        // - open: Entity=(eX-type-...)
        // - close: Entity=eX)
        // - one-token mention: Entity=(eX-type-1)
        let conllu = r#"# newdoc id = doc_en
# sent_id = 1
1	Marie	_	PROPN	_	_	0	root	_	Entity=(e1-person-1
2	Curie	_	PROPN	_	_	1	flat	_	Entity=e1)
3	met	_	VERB	_	_	1	dep	_	_
4	Cher	_	PROPN	_	_	3	obj	_	Entity=(e2-person-1)
5	.	_	PUNCT	_	_	3	punct	_	_

# newdoc id = doc_multi
# sent_id = 1
1	習近平	_	PROPN	_	_	0	root	_	Entity=(e10-person-1)
2	在	_	ADP	_	_	1	case	_	_
3	北京	_	PROPN	_	_	1	obl	_	Entity=(e11-place-1)
4	會見	_	VERB	_	_	1	dep	_	_
5	了	_	AUX	_	_	4	aux	_	_
6	普京	_	PROPN	_	_	4	obj	_	Entity=(e12-person-1)
7	。	_	PUNCT	_	_	4	punct	_	_
8.1	_	_	_	_	_	_	_	_	Entity=(e13-person-1)

# newdoc id = doc_ar
# sent_id = 1
1	محمد	_	PROPN	_	_	0	root	_	Entity=(e20-person-1
2	بن	_	PART	_	_	1	flat	_	SpaceAfter=No
3	سلمان	_	PROPN	_	_	1	flat	_	Entity=e20)
4	.	_	PUNCT	_	_	1	punct	_	_

# newdoc id = doc_ru
# sent_id = 1
1	Путин	_	PROPN	_	_	0	root	_	Entity=(e30-person-1)
2	встретился	_	VERB	_	_	1	dep	_	_
3	с	_	ADP	_	_	1	case	_	_
4	Си	_	PROPN	_	_	1	obl	_	Entity=(e31-person-1
5	Цзиньпином	_	PROPN	_	_	4	flat	_	Entity=e31)
6	.	_	PUNCT	_	_	1	punct	_	_

# newdoc id = doc_hi
# sent_id = 1
1	प्रधानमंत्री	_	NOUN	_	_	0	root	_	Entity=(e40-person-1
2	शर्मा	_	PROPN	_	_	1	flat	_	Entity=e40)
3	दिल्ली	_	PROPN	_	_	1	obl	_	Entity=(e41-place-1)
4	में	_	ADP	_	_	3	case	_	_
5	थे	_	AUX	_	_	1	cop	_	_
6	।	_	PUNCT	_	_	1	punct	_	_
"#;

        let docs = parse_corefud_conllu(conllu).unwrap();
        assert_eq!(docs.len(), 5);

        // doc_en: spans should be valid char offsets.
        let doc_en = docs
            .iter()
            .find(|d| d.doc_id.as_deref() == Some("doc_en"))
            .unwrap();
        let char_len = doc_en.text.chars().count();
        for m in doc_en.all_mentions() {
            assert!(m.start <= m.end);
            assert!(m.end <= char_len);
        }

        // doc_multi: should include a zero mention (empty node 8.1)
        let doc_multi = docs
            .iter()
            .find(|d| d.doc_id.as_deref() == Some("doc_multi"))
            .unwrap();
        let zeros: Vec<_> = doc_multi
            .all_mentions()
            .into_iter()
            .filter(|m| m.mention_type == Some(MentionType::Zero) || (m.start == m.end))
            .collect();
        assert!(
            !zeros.is_empty(),
            "Expected at least one zero/empty mention in doc_multi"
        );

        // doc_ar: SpaceAfter=No should glue بن + سلمان without inserting a space after بن.
        let doc_ar = docs
            .iter()
            .find(|d| d.doc_id.as_deref() == Some("doc_ar"))
            .unwrap();
        assert!(
            doc_ar.text.contains("محمد بنسلمان") || doc_ar.text.contains("محمد بن سلمان"),
            "Arabic spacing should be Unicode-safe; got: {:?}",
            doc_ar.text
        );
    }

    #[test]
    fn test_parse_ecb_plus_coref() {
        let csv = "\
Topic,File,Sentence Number,Token Number,Token,Lemma,Event Mention,Coreference Chain
1,1ecb,0,0,The,the,,
1,1ecb,0,1,earthquake,earthquake,ACT,1
1,1ecb,0,2,struck,strike,ACT,2
1,1ecb,0,3,at,at,,
1,1ecb,0,4,dawn,dawn,,
1,1ecb,1,0,A,a,,
1,1ecb,1,1,tremor,tremor,ACT,2
1,1ecb,1,2,was,be,,
1,1ecb,1,3,felt,feel,ACT,
1,2ecb,0,0,The,the,,
1,2ecb,0,1,quake,quake,ACT,1
1,2ecb,0,2,damaged,damage,ACT,3
1,2ecb,0,3,buildings,building,,
";
        let docs = parse_ecb_plus_coref(csv).unwrap();

        // Should produce 2 documents: (1, 1ecb) and (1, 2ecb)
        assert_eq!(docs.len(), 2);

        let doc1 = docs
            .iter()
            .find(|d| d.doc_id.as_deref() == Some("1_1ecb"))
            .unwrap();
        let doc2 = docs
            .iter()
            .find(|d| d.doc_id.as_deref() == Some("1_2ecb"))
            .unwrap();

        // doc1 has chains for chain IDs "1" and "2"
        assert_eq!(doc1.chains.len(), 2);
        // doc2 has chains for chain IDs "1" and "3"
        assert_eq!(doc2.chains.len(), 2);

        // Chain "1" in doc1 should contain "earthquake"
        let chain1 = doc1
            .chains
            .iter()
            .find(|c| c.mentions.iter().any(|m| m.text == "earthquake"));
        assert!(chain1.is_some());

        // Chain "2" in doc1 should contain "struck" and "tremor"
        let chain2 = doc1
            .chains
            .iter()
            .find(|c| c.mentions.iter().any(|m| m.text == "struck"));
        assert!(chain2.is_some());
        assert!(chain2.unwrap().mentions.iter().any(|m| m.text == "tremor"));

        // Chain "1" in doc2 should contain "quake" (cross-doc coreferent with "earthquake")
        let chain1_doc2 = doc2
            .chains
            .iter()
            .find(|c| c.mentions.iter().any(|m| m.text == "quake"));
        assert!(chain1_doc2.is_some());
    }

    #[test]
    fn ecb_plus_zip_reads_each_member_when_decoded_names_match() {
        use std::io::{Cursor, Write};
        use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};

        let utf8 = ("ECB+/1/café.xml", "Alice");
        let cp437 = ("ECB+/1/caf~.xml", "Bob");
        for entries in [[utf8, cp437], [cp437, utf8]] {
            let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
            let options =
                SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
            for (name, token) in entries.into_iter().chain([
                ("ECB+/2/control.xml", "Control"),
                ("ECB+/2/notes.txt", "Ignored"),
            ]) {
                zip.start_file(name, options).unwrap();
                write!(zip, r#"<Document><token t_id="1" sentence="0" number="0">{token}</token></Document>"#).unwrap();
            }
            let mut bytes = zip.finish().unwrap().into_inner();
            // The writer takes UTF-8 names. Encode é in CP437 in both ZIP headers.
            let placeholder = b"caf~.xml";
            while let Some(offset) = bytes
                .windows(placeholder.len())
                .position(|s| s == placeholder)
            {
                bytes[offset + 3] = 0x82;
            }
            let docs = parse_ecb_plus_zip(&bytes).unwrap();
            let parsed: Vec<_> = docs
                .iter()
                .map(|doc| (doc.doc_id.as_deref(), doc.text.as_str()))
                .collect();
            assert_eq!(
                parsed,
                [
                    (Some("1_café"), entries[0].1),
                    (Some("1_café"), entries[1].1),
                    (Some("2_control"), "Control"),
                ]
            );
        }
    }

    #[test]
    fn ecb_plus_gold_reads_scoped_labels_and_retains_local_chains() {
        use std::io::{Cursor, Write};
        use zip::{write::SimpleFileOptions, ZipWriter};

        // Independently authored XML. Reuse m_id, r_id and local chain order across files.
        let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
        for (name, label_attr) in [
            ("ECB+/4/4_2ecb.xml", r#"note="ACT-demo""#),
            ("4_1ecb.xml", ""), // the target's instance_id also identifies the cross-document instance
            ("ECB+/5/5_1ecb.xml", r#"note="ACT-demo""#),
            ("unknown.xml", r#"note="ACT-demo""#),
        ] {
            zip.start_file(name, SimpleFileOptions::default()).unwrap();
            write!(zip, r#"<Document>
                <token t_id="1">Zoë</token><token t_id="2">répare</token>
                <token t_id="3">la</token><token t_id="4">roue</token>
                <token t_id="5">vite</token><token t_id="6">ici</token>
                <Markables>
                <HUMAN_PART_PER m_id="0"><token_anchor t_id="1"/></HUMAN_PART_PER>
                <ACTION_OCCURRENCE m_id="1"><token_anchor t_id="2"/></ACTION_OCCURRENCE>
                <NON_HUMAN_PART m_id="2"><token_anchor t_id="3"/></NON_HUMAN_PART>
                <NON_HUMAN_PART m_id="3"><token_anchor t_id="4"/></NON_HUMAN_PART>
                <TIME_DURATION m_id="4"><token_anchor t_id="5"/></TIME_DURATION>
                <LOC_OTHER m_id="5"><token_anchor t_id="6"/></LOC_OTHER>
                <UNKNOWN_INSTANCE_TAG m_id="20" TAG_DESCRIPTOR="repair" instance_id="ACT-demo"/>
                <NON_HUMAN_PART m_id="21" TAG_DESCRIPTOR=""/>
                </Markables>
                <Relations>
                <CROSS_DOC_COREF r_id="0" {label_attr}><source m_id="1"/><target m_id="20"/></CROSS_DOC_COREF>
                <INTRA_DOC_COREF r_id="1" note="ACT-demo"><source m_id="2"/><source m_id="3"/><target m_id="21"/></INTRA_DOC_COREF>
                <CROSS_DOC_COREF r_id="2"><source m_id="0"/></CROSS_DOC_COREF>
                <CROSS_DOC_COREF r_id="3" note="0"><source m_id="4"/></CROSS_DOC_COREF>
                </Relations></Document>"#).unwrap();
        }
        let bytes = zip.finish().unwrap().into_inner();
        let docs = parse_ecb_plus_gold_zip(&bytes).unwrap();
        assert_eq!(
            docs.iter().map(|d| d.topic.as_deref()).collect::<Vec<_>>(),
            [Some("4"), Some("4"), Some("5"), None]
        );
        for doc in docs {
            assert_eq!(doc.chains.len(), 5);
            let info: Vec<_> = doc
                .chains
                .iter()
                .map(|c| {
                    let mentions: Vec<_> = c
                        .chain
                        .mentions
                        .iter()
                        .map(|m| (m.text.as_str(), m.entity_type.as_deref().unwrap()))
                        .collect();
                    (c.cross_document_id.as_deref(), mentions)
                })
                .collect();
            assert_eq!(
                info,
                [
                    (Some("0"), vec![("vite", "TIME")]),
                    (Some("ACT-demo"), vec![("répare", "EVENT")]),
                    (None, vec![("la", "PARTICIPANT"), ("roue", "PARTICIPANT")]),
                    (None, vec![("Zoë", "PERSON")]),
                    (None, vec![("ici", "LOC")]),
                ]
            );
            let action = &doc.chains[1].chain.mentions[0];
            assert_eq!((action.start, action.end), (4, 10));
            let generic = doc.into_coref_document();
            assert!(generic.includes_singletons);
            assert_eq!(generic.chains[0].cluster_id.unwrap().get(), 0);
            assert!(generic.chains[1..]
                .iter()
                .all(|chain| chain.cluster_id.is_none()));
        }
    }

    #[test]
    fn ecb_plus_discontinuous_mentions_join_only_anchored_tokens() {
        let xml = r#"<Document><token t_id="1">Zoë</token><token t_id="2">left</token>
            <token t_id="3">quietly</token><token t_id="4">town</token>
            <Markables><ACTION_OCCURRENCE m_id="1"><token_anchor t_id="4"/>
            <token_anchor t_id="2"/><token_anchor t_id="2"/></ACTION_OCCURRENCE></Markables></Document>"#;
        let doc = parse_ecb_plus_xml(xml, Some("1"), "1_1ecb", None).unwrap();
        let mention = &doc.chains[0].chain.mentions[0];
        assert_eq!(mention.text, "left town");
        assert_eq!((mention.start, mention.end), (4, 21));
        assert_eq!(
            doc.text
                .chars()
                .skip(mention.start)
                .take(mention.end - mention.start)
                .collect::<String>(),
            "left quietly town"
        );
    }

    #[test]
    fn ecb_plus_selected_gold_validates_coordinates_and_keeps_full_document_offsets() {
        use std::io::{Cursor, Write};
        use zip::{write::SimpleFileOptions, ZipWriter};

        let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
        for name in ["ECB+/4/4_1ecb.xml", "ECB+/4/4_1ecbplus.xml"] {
            zip.start_file(name, SimpleFileOptions::default()).unwrap();
            zip.write_all(r#"<Document>
                <token t_id="1" sentence="0">Zoë</token><token t_id="2" sentence="2">fixes</token>
                <token t_id="3" sentence="2">it</token><token t_id="4" sentence="3">today</token>
                <Markables>
                <HUMAN_PART_PER m_id="1"><token_anchor t_id="1"/></HUMAN_PART_PER>
                <ACTION_OCCURRENCE m_id="2"><token_anchor t_id="2"/></ACTION_OCCURRENCE>
                <TIME_DATE m_id="3"><token_anchor t_id="3"/><token_anchor t_id="4"/></TIME_DATE>
                </Markables>
                <Relations><CROSS_DOC_COREF note="A17"><source m_id="1"/><source m_id="2"/></CROSS_DOC_COREF></Relations>
                </Document>"#.as_bytes()).unwrap();
        }
        let bytes = zip.finish().unwrap().into_inner();
        let header = "Topic,File,Sentence Number";
        let selected = format!("\u{feff}{header}\r\n4,1ecb, 2 \r\n");
        let docs = parse_ecb_plus_selected_gold_zip(&bytes, &selected).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].doc_id, "4_4_1ecb");
        assert_eq!(docs[0].text, "Zoë fixes it today");
        assert_eq!(docs[0].chains.len(), 1);
        let chain = &docs[0].chains[0];
        assert_eq!(chain.cross_document_id.as_deref(), Some("A17"));
        let mentions = &chain.chain.mentions;
        assert_eq!(mentions.len(), 1);
        assert_eq!(
            (
                mentions[0].text.as_str(),
                mentions[0].start,
                mentions[0].end
            ),
            ("fixes", 4, 9)
        );

        for (csv, diagnostic) in [
            ("Bad header".to_owned(), "header"),
            (header.to_owned(), "no selected sentences"),
            (format!("{header}\n4,1ecb,bad"), "row 2"),
            (format!("{header}\n4,1ecb,2,extra"), "row 2"),
            (
                format!("{header}\n5,1ecb,2"),
                "Missing selected ECB+ document",
            ),
            (
                format!("{header}\n4,1ecb,1"),
                "Missing selected ECB+ sentence",
            ),
            (format!("{header}\n4,1ecb,2\n4,1ecb,2 "), "Duplicate"),
        ] {
            let error = parse_ecb_plus_selected_gold_zip(&bytes, &csv)
                .unwrap_err()
                .to_string();
            assert!(error.contains(diagnostic), "{error}");
        }
    }

    /// Opt-in fixture for the original cltl/ecbPlus LREC 2014 XML release.
    #[test]
    #[ignore = "set ANNO_ECB_PLUS_ARCHIVE to ECB+.zip beside ECBplus_coreference_sentences.csv"]
    fn ecb_plus_published_archive_gold_counts() {
        use std::collections::HashSet;
        let archive = PathBuf::from(
            std::env::var_os("ANNO_ECB_PLUS_ARCHIVE").expect("ANNO_ECB_PLUS_ARCHIVE"),
        );
        let bytes = std::fs::read(&archive).unwrap();
        let index =
            std::fs::read_to_string(archive.with_file_name("ECBplus_coreference_sentences.csv"))
                .unwrap();
        for (docs, expected) in [
            (
                parse_ecb_plus_gold_zip(&bytes).unwrap(),
                (982, 2319, 17856, 14441),
            ),
            (
                parse_ecb_plus_selected_gold_zip(&bytes, &index).unwrap(),
                (976, 1832, 11892, 3230),
            ),
        ] {
            let mut scoped_ids = HashSet::new();
            let mut cross_mentions = 0;
            let mut local_mentions = 0;
            for doc in &docs {
                let topic = doc.topic.as_ref().unwrap();
                for chain in &doc.chains {
                    if let Some(id) = &chain.cross_document_id {
                        scoped_ids.insert((topic, id));
                        cross_mentions += chain.chain.len();
                    } else {
                        local_mentions += chain.chain.len();
                    }
                }
            }
            assert_eq!(
                (docs.len(), scoped_ids.len(), cross_mentions, local_mentions),
                expected
            );
        }
    }

    #[test]
    fn ecb_plus_alphanumeric_notes_remain_separate_chains_without_a_numeric_id() {
        // Two annotated mentions and their source IDs from ECB+/1/1_10ecb.xml.
        let xml = r#"<Document>
          <token t_id="7" sentence="0" number="6">herself</token>
          <token t_id="15" sentence="0" number="14">told</token>
          <Markables>
            <HUMAN_PART_PER m_id="21"><token_anchor t_id="7"/></HUMAN_PART_PER>
            <ACTION_REPORTING m_id="34"><token_anchor t_id="15"/></ACTION_REPORTING>
          </Markables>
          <Relations>
            <CROSS_DOC_COREF r_id="37612" note="ACT16235311629112331"><source m_id="34"/><target m_id="41"/></CROSS_DOC_COREF>
            <CROSS_DOC_COREF r_id="37614" note="HUM16236184328979740"><source m_id="21"/><target m_id="43"/></CROSS_DOC_COREF>
          </Relations>
        </Document>"#;
        let doc = parse_ecb_plus_xml(xml, Some("1"), "1_10ecb", None)
            .unwrap()
            .into_coref_document();
        assert_eq!(doc.chains.len(), 2);
        assert_eq!(doc.chains[0].mentions[0].text, "told");
        assert_eq!(doc.chains[1].mentions[0].text, "herself");
        assert!(doc.chains.iter().all(|c| c.cluster_id.is_none()));
    }

    #[test]
    fn test_parse_ecb_plus_xml() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<Document doc_name="1_1ecb" doc_id="1_1ecb.xml.xml">
  <token t_id="1" sentence="0" number="0">The</token>
  <token t_id="2" sentence="0" number="1">earthquake</token>
  <token t_id="3" sentence="0" number="2">struck</token>
  <token t_id="4" sentence="0" number="3">at</token>
  <token t_id="5" sentence="0" number="4">dawn</token>
  <token t_id="6" sentence="1" number="0">A</token>
  <token t_id="7" sentence="1" number="1">tremor</token>
  <token t_id="8" sentence="1" number="2">was</token>
  <token t_id="9" sentence="1" number="3">felt</token>
  <token t_id="10" sentence="1" number="4">A&amp;B</token>
  <Markables>
    <ACTION_OCCURRENCE m_id="30">
      <token_anchor t_id="2"/>
    </ACTION_OCCURRENCE>
    <ACTION_OCCURRENCE m_id="31">
      <token_anchor t_id="3"/>
    </ACTION_OCCURRENCE>
    <ACTION_OCCURRENCE m_id="32">
      <token_anchor t_id="7"/>
    </ACTION_OCCURRENCE>
  </Markables>
  <Relations>
    <CROSS_DOC_COREF r_id="1" note="30001">
      <source m_id="30"/>
      <target m_id="32"/>
    </CROSS_DOC_COREF>
    <CROSS_DOC_COREF r_id="2" note="30002">
      <source m_id="31"/>
    </CROSS_DOC_COREF>
  </Relations>
</Document>"#;

        let doc = parse_ecb_plus_xml(xml, Some("1"), "1_1ecb", None)
            .unwrap()
            .into_coref_document();
        assert_eq!(doc.doc_id.as_deref(), Some("1_1_1ecb"));
        assert!(doc.text.contains("earthquake"));
        assert!(doc.text.contains("tremor"));
        assert!(doc.text.contains("A&B"));

        // Cluster 30001 should have mentions for "earthquake" (t_id=2) and "tremor" (t_id=7)
        let cluster_30001 = doc.chains.iter().find(|c| {
            c.mentions.iter().any(|m| m.text == "earthquake")
                && c.mentions.iter().any(|m| m.text == "tremor")
        });
        assert!(
            cluster_30001.is_some(),
            "Expected cross-doc cluster linking earthquake and tremor"
        );

        // Cluster 30002 should have "struck"
        let cluster_30002 = doc
            .chains
            .iter()
            .find(|c| c.mentions.iter().any(|m| m.text == "struck"));
        assert_eq!(cluster_30002.unwrap().cluster_id.unwrap().get(), 30002);
    }
}
