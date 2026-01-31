//! Unified provenance tracking for the annotation pipeline.
//!
//! Provenance captures the full lineage of annotations:
//! - **Source**: Where did the document come from? (URL, file, API)
//! - **Ingest**: How was it converted to text? (pandoc, pdftotext, direct)
//! - **Preprocessing**: What transformations were applied? (normalization, chunking)
//! - **Extraction**: Which models produced annotations? (StackedNER, GLiNER)
//! - **Mentions**: Individual entity extractions with per-entity provenance
//! - **Tracks**: Coreference chains linking mentions
//!
//! This enables:
//! - Reproducibility (re-run the same pipeline)
//! - Debugging (trace errors back to source)
//! - Auditing (who/what/when for compliance)
//! - Confidence calibration (different pipelines have different accuracy)
//!
//! # Example
//!
//! ```rust
//! use anno_core::core::provenance::{DocumentProvenance, SourceInfo, IngestInfo};
//!
//! let provenance = DocumentProvenance::builder()
//!     .source(SourceInfo::url("https://example.com/doc.pdf"))
//!     .ingest(IngestInfo::converter("pdftotext", "0.86.1"))
//!     .preprocessor(&["whitespace_normalized", "unicode_nfc"])
//!     .build();
//! ```

use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::time::SystemTime;

/// Complete provenance for a document annotation pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentProvenance {
    /// Unique identifier for this provenance chain
    pub id: String,
    /// Source of the document
    pub source: SourceInfo,
    /// How the document was ingested/converted
    pub ingest: IngestInfo,
    /// Preprocessing steps applied
    pub preprocessing: Vec<PreprocessingStep>,
    /// Extraction pipelines used
    pub extraction: Vec<ExtractionPipeline>,
    /// When annotation started
    pub started_at: Option<String>,
    /// When annotation completed
    pub completed_at: Option<String>,
    /// Tool and version
    pub tool: ToolInfo,
    /// Additional metadata
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

/// Source information - where did the document come from?
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SourceInfo {
    /// Document from a URL
    Url {
        /// The URL of the document.
        url: String,
        /// ISO 8601 timestamp when the document was fetched.
        fetched_at: Option<String>,
        /// HTTP status code returned when fetching.
        http_status: Option<u16>,
        /// Content-Type header from response.
        content_type: Option<String>,
        /// ETag header for caching/validation.
        etag: Option<String>,
    },
    /// Document from local file
    File {
        /// Path to the file on disk.
        path: String,
        /// ISO 8601 timestamp when the file was last modified.
        modified_at: Option<String>,
        /// Size of the file in bytes.
        size_bytes: Option<u64>,
        /// Checksum (e.g., SHA256) for integrity validation.
        checksum: Option<String>,
    },
    /// Document from API
    Api {
        /// API endpoint URL.
        endpoint: String,
        /// Unique request identifier for tracing.
        request_id: Option<String>,
        /// Response time in milliseconds.
        response_time_ms: Option<u64>,
    },
    /// Raw text input (no external source)
    Raw {
        /// Length of the raw text in characters.
        length: usize,
        /// Checksum (e.g., SHA256) for integrity validation.
        checksum: Option<String>,
    },
    /// Dataset sample
    Dataset {
        /// Name of the dataset (e.g., "conll2003").
        name: String,
        /// Split of the dataset (e.g., "train", "test", "dev").
        split: String,
        /// Index of the sample within the split.
        index: Option<usize>,
    },
    /// Unknown/unspecified source
    Unknown,
}

impl SourceInfo {
    /// Create URL source.
    #[must_use]
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url {
            url: url.into(),
            fetched_at: None,
            http_status: None,
            content_type: None,
            etag: None,
        }
    }

    /// Create file source.
    #[must_use]
    pub fn file(path: impl Into<String>) -> Self {
        Self::File {
            path: path.into(),
            modified_at: None,
            size_bytes: None,
            checksum: None,
        }
    }

    /// Create raw text source.
    #[must_use]
    pub fn raw(text: &str) -> Self {
        Self::Raw {
            length: text.len(),
            checksum: None,
        }
    }

    /// Create dataset source.
    #[must_use]
    pub fn dataset(name: impl Into<String>, split: impl Into<String>) -> Self {
        Self::Dataset {
            name: name.into(),
            split: split.into(),
            index: None,
        }
    }
}

/// Ingest information - how was the document converted to text?
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method")]
pub enum IngestInfo {
    /// Direct text (no conversion needed)
    Direct,
    /// Converted via external tool
    Converter {
        /// Name of the conversion tool (e.g., "pdftotext", "pandoc").
        tool: String,
        /// Version of the tool used.
        version: Option<String>,
        /// Input format/MIME type.
        input_format: Option<String>,
        /// Output format/MIME type (typically "text/plain").
        output_format: String,
    },
    /// Converted via library
    Library {
        /// Name of the library (e.g., "pdf-extract", "docx-rust").
        name: String,
        /// Version of the library used.
        version: Option<String>,
    },
    /// Custom/unknown conversion
    Custom {
        /// Human-readable description of the conversion method.
        description: String,
    },
}

impl IngestInfo {
    /// Create converter ingest info.
    #[must_use]
    pub fn converter(tool: impl Into<String>, version: impl Into<String>) -> Self {
        Self::Converter {
            tool: tool.into(),
            version: Some(version.into()),
            input_format: None,
            output_format: "plain_text".to_string(),
        }
    }

    /// Create direct ingest (no conversion).
    #[must_use]
    pub fn direct() -> Self {
        Self::Direct
    }
}

/// A preprocessing step applied to the document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreprocessingStep {
    /// Name of the step
    pub name: String,
    /// Parameters/configuration
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub params: HashMap<String, String>,
    /// Order in the pipeline (0 = first)
    pub order: u32,
}

impl PreprocessingStep {
    /// Create a new preprocessing step.
    #[must_use]
    pub fn new(name: impl Into<String>, order: u32) -> Self {
        Self {
            name: name.into(),
            params: HashMap::new(),
            order,
        }
    }

    /// Add a parameter.
    #[must_use]
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }
}

/// An extraction pipeline used to produce annotations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractionPipeline {
    /// Pipeline identifier (e.g., "ner", "coref", "rel")
    pub task: String,
    /// Model/backend name
    pub model: String,
    /// Model version
    pub version: Option<String>,
    /// Entity/relation types extracted
    pub types_extracted: Vec<String>,
    /// Configuration parameters
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub config: HashMap<String, String>,
    /// Sub-backends (for stacked/ensemble)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sub_backends: Vec<String>,
}

impl ExtractionPipeline {
    /// Create a new extraction pipeline.
    #[must_use]
    pub fn new(task: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            task: task.into(),
            model: model.into(),
            version: None,
            types_extracted: Vec::new(),
            config: HashMap::new(),
            sub_backends: Vec::new(),
        }
    }

    /// Set version.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Add extracted type.
    #[must_use]
    pub fn with_type(mut self, entity_type: impl Into<String>) -> Self {
        self.types_extracted.push(entity_type.into());
        self
    }

    /// Add sub-backend.
    #[must_use]
    pub fn with_sub_backend(mut self, backend: impl Into<String>) -> Self {
        self.sub_backends.push(backend.into());
        self
    }
}

/// Tool information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolInfo {
    /// Tool name
    pub name: Cow<'static, str>,
    /// Tool version
    pub version: Cow<'static, str>,
    /// Git commit hash (optional)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
}

impl Default for ToolInfo {
    fn default() -> Self {
        Self {
            name: Cow::Borrowed("anno"),
            version: Cow::Borrowed(env!("CARGO_PKG_VERSION")),
            commit: option_env!("GIT_HASH").map(String::from),
        }
    }
}

/// Builder for DocumentProvenance.
#[derive(Debug, Default)]
pub struct ProvenanceBuilder {
    source: Option<SourceInfo>,
    ingest: Option<IngestInfo>,
    preprocessing: Vec<PreprocessingStep>,
    extraction: Vec<ExtractionPipeline>,
    metadata: HashMap<String, String>,
}

impl ProvenanceBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the source.
    #[must_use]
    pub fn source(mut self, source: SourceInfo) -> Self {
        self.source = Some(source);
        self
    }

    /// Set the ingest method.
    #[must_use]
    pub fn ingest(mut self, ingest: IngestInfo) -> Self {
        self.ingest = Some(ingest);
        self
    }

    /// Add a preprocessing step.
    #[must_use]
    pub fn preprocessing(mut self, step: PreprocessingStep) -> Self {
        self.preprocessing.push(step);
        self
    }

    /// Add preprocessing steps by name.
    #[must_use]
    pub fn preprocessor(mut self, names: &[&str]) -> Self {
        for (i, name) in names.iter().enumerate() {
            self.preprocessing
                .push(PreprocessingStep::new(*name, i as u32));
        }
        self
    }

    /// Add an extraction pipeline.
    #[must_use]
    pub fn extraction(mut self, pipeline: ExtractionPipeline) -> Self {
        self.extraction.push(pipeline);
        self
    }

    /// Add metadata.
    #[must_use]
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build the provenance.
    #[must_use]
    pub fn build(self) -> DocumentProvenance {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Generate ID from content
        let mut hasher = DefaultHasher::new();
        if let Some(ref source) = self.source {
            format!("{:?}", source).hash(&mut hasher);
        }
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
            .hash(&mut hasher);

        let id = format!("prov:{:x}", hasher.finish());
        let now = chrono::Utc::now().to_rfc3339();

        DocumentProvenance {
            id,
            source: self.source.unwrap_or(SourceInfo::Unknown),
            ingest: self.ingest.unwrap_or(IngestInfo::Direct),
            preprocessing: self.preprocessing,
            extraction: self.extraction,
            started_at: Some(now.clone()),
            completed_at: None,
            tool: ToolInfo::default(),
            metadata: self.metadata,
        }
    }
}

impl DocumentProvenance {
    /// Create a new builder.
    #[must_use]
    pub fn builder() -> ProvenanceBuilder {
        ProvenanceBuilder::new()
    }

    /// Mark as completed.
    pub fn complete(&mut self) {
        self.completed_at = Some(chrono::Utc::now().to_rfc3339());
    }

    /// Add extraction pipeline.
    pub fn add_extraction(&mut self, pipeline: ExtractionPipeline) {
        self.extraction.push(pipeline);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provenance_builder() {
        let prov = DocumentProvenance::builder()
            .source(SourceInfo::url("https://example.com/doc.pdf"))
            .ingest(IngestInfo::converter("pdftotext", "0.86.1"))
            .preprocessor(&["whitespace", "unicode_nfc"])
            .extraction(
                ExtractionPipeline::new("ner", "stacked")
                    .with_sub_backend("pattern")
                    .with_sub_backend("heuristic"),
            )
            .build();

        assert!(prov.id.starts_with("prov:"));
        assert!(matches!(prov.source, SourceInfo::Url { .. }));
        assert!(matches!(prov.ingest, IngestInfo::Converter { .. }));
        assert_eq!(prov.preprocessing.len(), 2);
        assert_eq!(prov.extraction.len(), 1);
    }

    #[test]
    fn test_source_info_variants() {
        let url = SourceInfo::url("https://example.com");
        assert!(matches!(url, SourceInfo::Url { .. }));

        let file = SourceInfo::file("/path/to/doc.txt");
        assert!(matches!(file, SourceInfo::File { .. }));

        let raw = SourceInfo::raw("Hello, world!");
        assert!(matches!(raw, SourceInfo::Raw { length: 13, .. }));
    }

    #[test]
    fn test_serde_roundtrip() {
        let prov = DocumentProvenance::builder()
            .source(SourceInfo::url("https://example.com"))
            .ingest(IngestInfo::direct())
            .metadata("key", "value")
            .build();

        let json = serde_json::to_string(&prov).expect("serialize DocumentProvenance");
        let recovered: DocumentProvenance =
            serde_json::from_str(&json).expect("deserialize DocumentProvenance");

        assert_eq!(prov.id, recovered.id);
        assert_eq!(prov.metadata.get("key"), recovered.metadata.get("key"));
    }

    #[test]
    fn test_extraction_pipeline_builder() {
        let pipeline = ExtractionPipeline::new("ner", "stacked")
            .with_version("0.2.0")
            .with_type("PER")
            .with_type("ORG")
            .with_sub_backend("pattern")
            .with_sub_backend("heuristic");

        assert_eq!(pipeline.task, "ner");
        assert_eq!(pipeline.model, "stacked");
        assert_eq!(pipeline.version, Some("0.2.0".to_string()));
        assert_eq!(pipeline.types_extracted, vec!["PER", "ORG"]);
        assert_eq!(pipeline.sub_backends, vec!["pattern", "heuristic"]);
    }
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // Strategy for generating URLs
    fn arb_url() -> impl Strategy<Value = String> {
        prop::string::string_regex("https?://[a-z]+\\.[a-z]{2,4}/[a-z0-9/]*")
            .expect("valid URL regex for proptest")
            .prop_filter("valid url", |s| !s.is_empty())
    }

    // Strategy for generating file paths
    fn arb_path() -> impl Strategy<Value = String> {
        prop::string::string_regex("/[a-z]+(/[a-z0-9]+)*\\.txt")
            .expect("valid path regex for proptest")
            .prop_filter("valid path", |s| !s.is_empty())
    }

    // Strategy for generating alphanumeric identifiers
    fn arb_identifier() -> impl Strategy<Value = String> {
        prop::string::string_regex("[a-zA-Z][a-zA-Z0-9_]{0,20}")
            .expect("valid identifier regex for proptest")
    }

    // -------------------------------------------------------------------------
    // SourceInfo Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// URL source round-trips through serde
        #[test]
        fn prop_source_url_roundtrip(url in arb_url()) {
            let source = SourceInfo::url(&url);
            let json = serde_json::to_string(&source).expect("serialize SourceInfo::Url");
            let recovered: SourceInfo = serde_json::from_str(&json).expect("deserialize SourceInfo");

            if let SourceInfo::Url { url: recovered_url, .. } = recovered {
                prop_assert_eq!(url, recovered_url);
            } else {
                prop_assert!(false, "Expected Url variant");
            }
        }

        /// File source round-trips through serde
        #[test]
        fn prop_source_file_roundtrip(path in arb_path()) {
            let source = SourceInfo::file(&path);
            let json = serde_json::to_string(&source).expect("serialize SourceInfo::File");
            let recovered: SourceInfo = serde_json::from_str(&json).expect("deserialize SourceInfo");

            if let SourceInfo::File { path: recovered_path, .. } = recovered {
                prop_assert_eq!(path, recovered_path);
            } else {
                prop_assert!(false, "Expected File variant");
            }
        }

        /// Raw source captures correct length
        #[test]
        fn prop_source_raw_length(text in ".*") {
            let source = SourceInfo::raw(&text);
            if let SourceInfo::Raw { length, .. } = source {
                prop_assert_eq!(length, text.len());
            } else {
                prop_assert!(false, "Expected Raw variant");
            }
        }

        /// Dataset source round-trips
        #[test]
        fn prop_source_dataset_roundtrip(
            name in arb_identifier(),
            split in prop::sample::select(vec!["train", "test", "dev"])
        ) {
            let source = SourceInfo::dataset(&name, split);
            let json = serde_json::to_string(&source).expect("serialize SourceInfo::Dataset");
            let recovered: SourceInfo = serde_json::from_str(&json).expect("deserialize SourceInfo");

            if let SourceInfo::Dataset { name: n, split: _, .. } = recovered {
                prop_assert_eq!(name, n);
            } else {
                prop_assert!(false, "Expected Dataset variant");
            }
        }
    }

    // -------------------------------------------------------------------------
    // IngestInfo Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Converter ingest round-trips
        #[test]
        fn prop_ingest_converter_roundtrip(tool in arb_identifier(), version in "[0-9]+\\.[0-9]+\\.[0-9]+") {
            let ingest = IngestInfo::converter(&tool, &version);
            let json = serde_json::to_string(&ingest).expect("serialize IngestInfo::Converter");
            let recovered: IngestInfo = serde_json::from_str(&json).expect("deserialize IngestInfo");

            if let IngestInfo::Converter { tool: t, version: v, .. } = recovered {
                prop_assert_eq!(tool, t);
                prop_assert_eq!(Some(version), v);
            } else {
                prop_assert!(false, "Expected Converter variant");
            }
        }

        /// Direct ingest round-trips
        #[test]
        fn prop_ingest_direct_roundtrip(_unused in Just(())) {
            let ingest = IngestInfo::direct();
            let json = serde_json::to_string(&ingest).expect("serialize IngestInfo::Direct");
            let recovered: IngestInfo = serde_json::from_str(&json).expect("deserialize IngestInfo");
            prop_assert!(matches!(recovered, IngestInfo::Direct));
        }
    }

    // -------------------------------------------------------------------------
    // PreprocessingStep Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Preprocessing step round-trips
        #[test]
        fn prop_preprocessing_roundtrip(name in arb_identifier(), order in 0u32..100) {
            let step = PreprocessingStep::new(&name, order);
            let json = serde_json::to_string(&step).expect("serialize PreprocessingStep");
            let recovered: PreprocessingStep =
                serde_json::from_str(&json).expect("deserialize PreprocessingStep");

            prop_assert_eq!(name, recovered.name);
            prop_assert_eq!(order, recovered.order);
        }

        /// Preprocessing step with params round-trips
        #[test]
        fn prop_preprocessing_with_params_roundtrip(
            name in arb_identifier(),
            order in 0u32..100,
            param_key in arb_identifier(),
            param_value in arb_identifier()
        ) {
            let step = PreprocessingStep::new(&name, order)
                .with_param(&param_key, &param_value);
            let json = serde_json::to_string(&step).expect("serialize PreprocessingStep");
            let recovered: PreprocessingStep =
                serde_json::from_str(&json).expect("deserialize PreprocessingStep");

            prop_assert_eq!(step.params.get(&param_key), recovered.params.get(&param_key));
        }
    }

    // -------------------------------------------------------------------------
    // ExtractionPipeline Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Extraction pipeline round-trips
        #[test]
        fn prop_extraction_roundtrip(task in arb_identifier(), model in arb_identifier()) {
            let pipeline = ExtractionPipeline::new(&task, &model);
            let json = serde_json::to_string(&pipeline).expect("serialize ExtractionPipeline");
            let recovered: ExtractionPipeline =
                serde_json::from_str(&json).expect("deserialize ExtractionPipeline");

            prop_assert_eq!(task, recovered.task);
            prop_assert_eq!(model, recovered.model);
        }

        /// Pipeline with version round-trips
        #[test]
        fn prop_extraction_version_roundtrip(
            task in arb_identifier(),
            model in arb_identifier(),
            version in "[0-9]+\\.[0-9]+\\.[0-9]+"
        ) {
            let pipeline = ExtractionPipeline::new(&task, &model)
                .with_version(&version);
            let json = serde_json::to_string(&pipeline).expect("serialize ExtractionPipeline");
            let recovered: ExtractionPipeline =
                serde_json::from_str(&json).expect("deserialize ExtractionPipeline");

            prop_assert_eq!(Some(version), recovered.version);
        }

        /// Pipeline with types round-trips
        #[test]
        fn prop_extraction_types_roundtrip(
            task in arb_identifier(),
            model in arb_identifier(),
            types in prop::collection::vec(arb_identifier(), 0..5)
        ) {
            let mut pipeline = ExtractionPipeline::new(&task, &model);
            for t in &types {
                pipeline = pipeline.with_type(t);
            }
            let json = serde_json::to_string(&pipeline).expect("serialize ExtractionPipeline");
            let recovered: ExtractionPipeline =
                serde_json::from_str(&json).expect("deserialize ExtractionPipeline");

            prop_assert_eq!(types, recovered.types_extracted);
        }
    }

    // -------------------------------------------------------------------------
    // DocumentProvenance Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Full provenance round-trips
        #[test]
        fn prop_provenance_roundtrip(url in arb_url()) {
            let prov = DocumentProvenance::builder()
                .source(SourceInfo::url(&url))
                .ingest(IngestInfo::direct())
                .build();

            let json = serde_json::to_string(&prov).expect("serialize DocumentProvenance");
            let recovered: DocumentProvenance =
                serde_json::from_str(&json).expect("deserialize DocumentProvenance");

            prop_assert_eq!(prov.id, recovered.id);

            if let (SourceInfo::Url { url: u1, .. }, SourceInfo::Url { url: u2, .. }) =
                (&prov.source, &recovered.source)
            {
                prop_assert_eq!(u1, u2);
            }
        }

        /// Provenance ID starts with "prov:"
        #[test]
        fn prop_provenance_id_prefix(url in arb_url()) {
            let prov = DocumentProvenance::builder()
                .source(SourceInfo::url(&url))
                .build();

            prop_assert!(prov.id.starts_with("prov:"));
        }

        /// Provenance has started_at timestamp
        #[test]
        fn prop_provenance_has_timestamp(url in arb_url()) {
            let prov = DocumentProvenance::builder()
                .source(SourceInfo::url(&url))
                .build();

            prop_assert!(prov.started_at.is_some());
        }

        /// Metadata round-trips correctly
        #[test]
        fn prop_provenance_metadata(
            key in arb_identifier(),
            value in arb_identifier()
        ) {
            let prov = DocumentProvenance::builder()
                .source(SourceInfo::Unknown)
                .metadata(&key, &value)
                .build();

            let json = serde_json::to_string(&prov).expect("serialize DocumentProvenance");
            let recovered: DocumentProvenance =
                serde_json::from_str(&json).expect("deserialize DocumentProvenance");

            prop_assert_eq!(prov.metadata.get(&key), recovered.metadata.get(&key));
        }
    }
}
