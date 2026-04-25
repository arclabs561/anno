//! Dataset Registry: Single Source of Truth
//!
//! This module defines ALL dataset metadata in a single declarative macro.
//! The macro generates:
//! - The `DatasetId` enum
//! - All accessor methods (name, description, url, etc.)
//! - Category membership (is_coreference, is_biomedical, etc.)
//! - Group functions (all_ner, all_coref, etc.)
//!
//! # Design Philosophy
//!
//! > "A single point of truth means a single point of maintenance."
//!
//! Instead of maintaining both a Rust enum AND a TOML file with a fragile
//! `rust_variant` mapping, we define everything here. The TOML file is
//! now a *derived artifact* generated from this registry for documentation.
//!
//! # Tasks vs Categories
//!
//! These two fields serve different purposes:
//!
//! ## `tasks` - What you can DO with the dataset (NLP capabilities)
//!
//! | Task | Description |
//! |------|-------------|
//! | `ner` | Named entity recognition |
//! | `coref` | Coreference resolution |
//! | `re` | Relation extraction |
//! | `el` | Entity linking |
//! | `event_coref` | Event coreference |
//! | `slot_filling` | Slot/intent filling |
//! | `temporal` | Temporal information extraction |
//! | `cdcr` | Cross-document coreference |
//!
//! ## `categories` - How to FIND relevant datasets (discovery/filtering)
//!
//! | Category | What it means |
//! |----------|---------------|
//! | **Domain** | |
//! | `biomedical` | Medical/clinical/life science text |
//! | `legal` | Legal documents, contracts, court cases |
//! | `scientific` | Academic papers, technical reports |
//! | `social_media` | Twitter, Reddit, informal text |
//! | `literary` | Fiction, novels, poetry |
//! | `news` | News articles, journalism |
//! | `dialogue` | Conversations, meetings, interviews |
//! | `gaming` | Game content, D&D, fantasy |
//! | `arcane_domain` | Very niche domains (cuneiform, mythology) |
//! | **Language** | |
//! | `multilingual` | Covers multiple languages |
//! | `low_resource` | Under-resourced languages |
//! | `code_switching` | Mixed-language text |
//! | `indigenous` | Indigenous/endangered languages |
//! | `historical` | Ancient/historical languages |
//! | **Annotation** | |
//! | `nested` | Nested entity annotations |
//! | `discontinuous` | Discontinuous entity spans |
//! | `long_document` | Documents > 2k tokens |
//! | **Evaluation** | |
//! | `adversarial` | Adversarial/robustness testing |
//! | `bias_evaluation` | Gender/demographic bias testing |
//! | `few_shot` | Few-shot learning evaluation |
//!
//! ### Example: Same dataset, different views
//!
//! ```text
//! LitBank:
//!   tasks: ["ner", "coref", "event_coref"]  ← What can you do with it?
//!   categories: [literary, long_document]    ← How to discover it
//!
//! BC5CDR:
//!   tasks: ["ner", "re"]                     ← NER + relation extraction
//!   categories: [biomedical]                 ← Domain filtering
//!
//! GICoref:
//!   tasks: ["coref"]                         ← Coreference task
//!   categories: [bias_evaluation]            ← For bias evaluation
//! ```
//!
//! ### Migration Note
//!
//! Some datasets still have task-like categories (e.g., `ner`, `coref`).
//! These are being migrated to use `tasks` field instead. When adding new
//! datasets, prefer using `tasks` for NLP tasks and `categories` for
//! domain/property discovery.
//!
//! # Schema Fields
//!
//! Required:
//! - `name`: Human-readable name
//! - `description`: Brief description with historical context where relevant
//! - `url`: Download URL (empty if requires license)
//! - `entity_types`: Array of entity type strings
//! - `language`: ISO 639-1/3 code (en, de, grc, multi, etc.)
//! - `domain`: Domain category (news, biomedical, literature, etc.)
//! - `categories`: Category flags for filtering
//!
//! Optional metadata (highly recommended):
//! - `license`: SPDX identifier or common name (CC-BY-4.0, MIT, LDC, Research)
//! - `citation`: "Author et al. (YYYY)" format for quick reference
//! - `paper_url`: DOI or ACL Anthology URL preferred
//! - `year`: Publication year (u16) for temporal sorting
//! - `format`: Data format (CoNLL, JSONL, TSV, BIO, BRAT, XML)
//! - `annotation_scheme`: Annotation scheme (BIO, BIOES, IOB2, standoff)
//! - `size_hint`: Approximate size ("~1500 docs", "~40k tokens")
//! - `notes`: Historical motivation, cultural context, known issues
//! - `tasks`: NLP tasks this dataset supports (see Tasks vs Categories above)
//!
//! # License Values
//!
//! Use SPDX identifiers where possible:
//! - `CC-BY-4.0`, `CC-BY-SA-4.0`, `CC-BY-NC-4.0`, `CC-BY-NC-SA-4.0`
//! - `MIT`, `Apache-2.0`, `GPL-3.0`
//! - `LDC` - Requires Linguistic Data Consortium membership
//! - `Research` - Academic use only, contact authors
//! - `Public` - Public domain or no restrictions
//!
//! # Adding a New Dataset
//!
//! 1. Add an entry in `define_datasets!` below
//! 2. Run tests to verify: `cargo test -p anno-eval`
//! 3. Regenerate exports:
//!    - `cargo test -p anno-eval generate_datasets_json -- --ignored`
//!    - `cargo test -p anno-eval generate_datasets_jsonl -- --ignored`
//!    - `cargo test -p anno-eval generate_datasets_markdown -- --ignored`
//! 4. If the dataset should be *loadable*, ensure `eval::loader::LoadableDatasetId`
//!    maps it to a parsing plan.
//!
//! Note:
//! - `DatasetId` here is the canonical catalog + metadata.
//! - Some catalog datasets are intentionally not loadable (yet).
//!
//! # Example
//!
//! ```rust,ignore
//! use anno_eval::eval::DatasetId;
//!
//! let id = DatasetId::WikiGold;
//! assert_eq!(id.name(), "WikiGold");
//! assert!(!id.is_coreference());
//! assert!(DatasetId::all_ner().contains(&id));
//! assert_eq!(id.year(), Some(2009));
//! ```

/// Data format for dataset files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DataFormat {
    /// CoNLL column format (space/tab separated)
    CoNLL,
    /// CoNLL-U (Universal Dependencies format)
    CoNLLU,
    /// JSON Lines (one JSON object per line)
    JSONL,
    /// Tab-separated values
    TSV,
    /// Comma-separated values
    CSV,
    /// BIO/IOB tagged format
    BIO,
    /// BRAT standoff annotation
    BRAT,
    /// XML-based format
    XML,
    /// HuggingFace datasets format
    HuggingFace,
    /// Custom or mixed format
    Custom,
}

impl DataFormat {
    /// Get format name as string
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CoNLL => "CoNLL",
            Self::CoNLLU => "CoNLL-U",
            Self::JSONL => "JSONL",
            Self::TSV => "TSV",
            Self::CSV => "CSV",
            Self::BIO => "BIO",
            Self::BRAT => "BRAT",
            Self::XML => "XML",
            Self::HuggingFace => "HuggingFace",
            Self::Custom => "Custom",
        }
    }
}

/// Annotation scheme used in the dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AnnotationScheme {
    /// BIO (Begin, Inside, Outside)
    BIO,
    /// BIOES (Begin, Inside, Outside, End, Single)
    BIOES,
    /// IOB2 (variant of IOB)
    IOB2,
    /// Standoff annotation (character offsets)
    Standoff,
    /// CoNLL coref (cluster IDs)
    CoNLLCoref,
    /// Custom scheme
    Custom,
}

impl AnnotationScheme {
    /// Get scheme name as string
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::BIO => "BIO",
            Self::BIOES => "BIOES",
            Self::IOB2 => "IOB2",
            Self::Standoff => "Standoff",
            Self::CoNLLCoref => "CoNLL-Coref",
            Self::Custom => "Custom",
        }
    }
}

/// Dataset accessibility status.
///
/// Indicates how easy it is to obtain the dataset data files.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default,
)]
pub enum DatasetAccessibility {
    /// Fully public, direct download available
    #[default]
    Public,
    /// Available on HuggingFace datasets hub
    HuggingFace,
    /// Available locally in testdata/ or cache
    Local,
    /// Requires registration but free (e.g., LDC for academics)
    Registration,
    /// Requires license agreement or contact with authors
    ContactAuthors,
    /// Data not yet released, only paper available
    NotYetReleased,
    /// Requires access to another dataset first (e.g., GCDC needs Yahoo L6)
    DependsOnOther,
    /// Dataset has been deprecated or is no longer available
    Deprecated,
}

impl DatasetAccessibility {
    /// Get accessibility as string
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::HuggingFace => "huggingface",
            Self::Local => "local",
            Self::Registration => "registration",
            Self::ContactAuthors => "contact_authors",
            Self::NotYetReleased => "not_yet_released",
            Self::DependsOnOther => "depends_on_other",
            Self::Deprecated => "deprecated",
        }
    }

    /// Check if data can be downloaded without human intervention
    #[must_use]
    pub fn is_automatable(&self) -> bool {
        matches!(self, Self::Public | Self::HuggingFace | Self::Local)
    }

    /// Check if dataset is actually obtainable (eventually)
    #[must_use]
    pub fn is_obtainable(&self) -> bool {
        !matches!(self, Self::NotYetReleased | Self::Deprecated)
    }
}

/// Macro to define all datasets in a single place.
///
/// Each dataset entry has:
/// - `variant`: The enum variant name (e.g., `WikiGold`)
/// - `name`: Human-readable name (e.g., "WikiGold")
/// - `description`: Brief description with historical context
/// - `url`: Download URL (empty string if requires license)
/// - `entity_types`: Slice of entity type strings
/// - `language`: Primary language code (ISO 639-1/3). Use `"mul"` for multilingual datasets.
/// - `domain`: Domain category
/// - `categories`: List of category flags (coref, biomedical, etc.)
///
/// Optional fields (add after domain, before categories):
/// - `license`: SPDX identifier or common name
/// - `citation`: "Author et al. (YYYY)" format
/// - `paper_url`: DOI or ACL Anthology URL
/// - `year`: Publication year (u16)
/// - `format`: Data format string
/// - `annotation_scheme`: BIO, BIOES, IOB2, standoff
/// - `size_hint`: Approximate size description
/// - `example`: Short example snippet showing annotation style
/// - `notes`: Historical context, known issues, cultural significance
/// - `splits`: Available splits ("train", "dev", "test", "all")
/// - `tasks`: Supported NLP tasks ("ner", "coref", "re", "el", etc.)
/// - `expected_docs`: Expected doc count for validation (u32)
/// - `sha256`: Content hash for integrity verification
/// - `access_status`: DatasetAccessibility value (public, huggingface, contact_authors, etc.)
/// - `mirror_url`: Alternative download URL if primary is unavailable
/// - `depends_on`: Dataset ID this depends on (e.g., GCDC depends on Yahoo L6)
#[macro_export]
macro_rules! define_datasets {
    (
        $(
            $variant:ident {
                name: $name:literal,
                description: $description:literal,
                url: $url:literal,
                entity_types: [$($etype:literal),* $(,)?],
                language: $language:literal,
                domain: $domain:literal,
                $(license: $license:literal,)?
                $(citation: $citation:literal,)?
                $(paper_url: $paper_url:literal,)?
                $(year: $year:literal,)?
                $(format: $format:literal,)?
                $(annotation_scheme: $ann_scheme:literal,)?
                $(size_hint: $size_hint:literal,)?
                $(example: $example:literal,)?
                $(notes: $notes:literal,)?
                $(splits: [$($split:literal),* $(,)?],)?
                $(tasks: [$($task:literal),* $(,)?],)?
                $(expected_docs: $expected_docs:literal,)?
                $(sha256: $sha256:literal,)?
                $(hf_id: $hf_id:literal,)?
                $(hf_config: $hf_config:literal,)?
                $(access_status: $access_status:ident,)?
                $(mirror_url: $mirror_url:literal,)?
                $(depends_on: $depends_on:literal,)?
                categories: [$($cat:ident),* $(,)?] $(,)?
            }
        ),* $(,)?
    ) => {
        /// Supported dataset identifiers.
        ///
        /// Each dataset has a known download URL, format, and expected entity types.
        /// Use `DatasetId::all()` to iterate over all available datasets.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
        #[non_exhaustive]
        pub enum DatasetId {
            $(
                #[doc = $description]
                $variant,
            )*
        }

        #[allow(clippy::too_many_lines, clippy::type_complexity, clippy::cognitive_complexity)]
        impl DatasetId {
            /// Get the download URL for this dataset.
            #[must_use]
            pub fn download_url(&self) -> &'static str {
                match self {
                    $(Self::$variant => $url,)*
                }
            }

            /// Get the human-readable name.
            #[must_use]
            pub fn name(&self) -> &'static str {
                match self {
                    $(Self::$variant => $name,)*
                }
            }

            /// Get the description.
            #[must_use]
            pub fn description(&self) -> &'static str {
                match self {
                    $(Self::$variant => $description,)*
                }
            }

            /// Get the primary language.
            #[must_use]
            pub fn language(&self) -> &'static str {
                match self {
                    $(Self::$variant => $language,)*
                }
            }

            /// Get the domain category.
            #[must_use]
            pub fn domain(&self) -> &'static str {
                match self {
                    $(Self::$variant => $domain,)*
                }
            }

            /// Get the category tags for this dataset.
            ///
            /// Categories are used for discovery/filtering (domain, language family, annotation
            /// properties). They are not intended to encode benchmark outcomes.
            #[must_use]
            pub fn categories(&self) -> &'static [&'static str] {
                match self {
                    $(Self::$variant => &[$(stringify!($cat)),*],)*
                }
            }

            /// Get the entity types for this dataset.
            #[must_use]
            pub fn entity_types(&self) -> &'static [&'static str] {
                match self {
                    $(Self::$variant => &[$($etype),*],)*
                }
            }

            /// Get the license for this dataset.
            #[must_use]
            pub fn license(&self) -> Option<&'static str> {
                match self {
                    $(Self::$variant => $crate::optional_field!($($license)?),)*
                }
            }

            /// Get the citation for this dataset.
            #[must_use]
            pub fn citation(&self) -> Option<&'static str> {
                match self {
                    $(Self::$variant => $crate::optional_field!($($citation)?),)*
                }
            }

            /// Get the paper URL for this dataset.
            #[must_use]
            pub fn paper_url(&self) -> Option<&'static str> {
                match self {
                    $(Self::$variant => $crate::optional_field!($($paper_url)?),)*
                }
            }

            /// Get additional notes for this dataset.
            #[must_use]
            pub fn notes(&self) -> Option<&'static str> {
                match self {
                    $(Self::$variant => $crate::optional_field!($($notes)?),)*
                }
            }

            /// Get the publication year for this dataset.
            #[must_use]
            pub fn year(&self) -> Option<u16> {
                match self {
                    $(Self::$variant => $crate::optional_field_num!($($year)?),)*
                }
            }

            /// Get the data format for this dataset.
            #[must_use]
            pub fn format(&self) -> Option<&'static str> {
                match self {
                    $(Self::$variant => $crate::optional_field!($($format)?),)*
                }
            }

            /// Get the annotation scheme for this dataset.
            #[must_use]
            pub fn annotation_scheme(&self) -> Option<&'static str> {
                match self {
                    $(Self::$variant => $crate::optional_field!($($ann_scheme)?),)*
                }
            }

            /// Get the size hint for this dataset.
            #[must_use]
            pub fn size_hint(&self) -> Option<&'static str> {
                match self {
                    $(Self::$variant => $crate::optional_field!($($size_hint)?),)*
                }
            }

            /// Get an example snippet from this dataset.
            #[must_use]
            pub fn example(&self) -> Option<&'static str> {
                match self {
                    $(Self::$variant => $crate::optional_field!($($example)?),)*
                }
            }

            /// Get available data splits (train, dev, test, etc.).
            #[must_use]
            pub fn splits(&self) -> &'static [&'static str] {
                match self {
                    $(Self::$variant => &[$($($split),*)?],)*
                }
            }

            /// Get supported NLP tasks (ner, coref, re, el, etc.).
            /// Datasets often support multiple tasks via their annotations.
            #[must_use]
            pub fn tasks(&self) -> &'static [&'static str] {
                match self {
                    $(Self::$variant => &[$($($task),*)?],)*
                }
            }

            /// Get supported tasks as typed `Task` values (best-effort).
            ///
            /// This keeps the registry as the single source of truth (strings), while giving
            /// downstream code a type-safe view for control flow and compatibility checks.
            ///
            /// Unknown/out-of-scope task strings are ignored.
            #[must_use]
            pub fn tasks_typed(&self) -> Vec<$crate::eval::task_mapping::Task> {
                let mut out = Vec::new();
                for task_str in self.tasks_or_inferred() {
                    let Some(task) = $crate::eval::task_mapping::Task::from_code(task_str) else {
                        continue;
                    };
                    if !out.contains(&task) {
                        out.push(task);
                    }
                }
                out
            }

            /// Get expected document count for validation.
            #[must_use]
            pub fn expected_docs(&self) -> Option<u32> {
                match self {
                    $(Self::$variant => $crate::optional_field_num!($($expected_docs)?),)*
                }
            }

            /// Get SHA256 hash for integrity verification.
            #[must_use]
            pub fn sha256(&self) -> Option<&'static str> {
                match self {
                    $(Self::$variant => $crate::optional_field!($($sha256)?),)*
                }
            }

            /// Get HuggingFace dataset ID for automated download.
            #[must_use]
            pub fn hf_id(&self) -> Option<&'static str> {
                match self {
                    $(Self::$variant => $crate::optional_field!($($hf_id)?),)*
                }
            }

            /// Get HuggingFace dataset config/subset name.
            #[must_use]
            pub fn hf_config(&self) -> Option<&'static str> {
                match self {
                    $(Self::$variant => $crate::optional_field!($($hf_config)?),)*
                }
            }

            /// Get the access status for this dataset.
            ///
            /// Indicates how to obtain the dataset: public download, HuggingFace,
            /// contact authors, not yet released, etc.
            #[must_use]
            pub fn access_status(&self) -> $crate::eval::dataset_registry::DatasetAccessibility {
                match self {
                    $(Self::$variant => $crate::optional_access_status!($($access_status)?),)*
                }
            }

            /// Get mirror URL for this dataset (alternative download location).
            #[must_use]
            pub fn mirror_url(&self) -> Option<&'static str> {
                match self {
                    $(Self::$variant => $crate::optional_field!($($mirror_url)?),)*
                }
            }

            /// Get dependency dataset name (e.g., GCDC depends on "Yahoo L6 corpus").
            #[must_use]
            pub fn depends_on(&self) -> Option<&'static str> {
                match self {
                    $(Self::$variant => $crate::optional_field!($($depends_on)?),)*
                }
            }

            /// Check if this dataset can be automatically downloaded.
            #[must_use]
            pub fn is_automatable(&self) -> bool {
                self.access_status().is_automatable()
            }

            /// Check if this dataset requires a license agreement (no public URL).
            #[must_use]
            pub fn requires_license(&self) -> bool {
                self.download_url().is_empty()
            }

            /// Check if this dataset has SPDX-compatible license.
            #[must_use]
            pub fn has_spdx_license(&self) -> bool {
                matches!(self.license(), Some(l) if
                    l.starts_with("CC-") ||
                    l == "MIT" ||
                    l == "Apache-2.0" ||
                    l == "GPL-3.0" ||
                    l == "Public")
            }

            /// Get the cache filename (derived from variant name).
            #[must_use]
            pub fn cache_filename(&self) -> &'static str {
                match self {
                    $(Self::$variant => concat!(stringify!($variant), ".cache"),)*
                }
            }

            /// Parse a `DatasetId` from a cache filename (like `Wnut17.cache`).
            ///
            /// This is the inverse of `cache_filename()` and is intended for tooling that
            /// iterates the local cache manifest or filesystem and needs to recover the enum.
            #[must_use]
            pub fn from_cache_filename(s: &str) -> Option<Self> {
                match s {
                    $(concat!(stringify!($variant), ".cache") => Some(Self::$variant),)*
                    _ => None,
                }
            }

            /// Get all dataset IDs.
            #[must_use]
            pub fn all() -> &'static [DatasetId] {
                &[$(Self::$variant,)*]
            }

            // Category checks are generated via helper macro
            $crate::impl_category_checks!($($variant => [$($cat),*]),*);
        }

        impl std::fmt::Display for DatasetId {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.name())
            }
        }

        impl std::str::FromStr for DatasetId {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                // Try exact match first
                match s {
                    $(stringify!($variant) => return Ok(Self::$variant),)*
                    _ => {}
                }
                // Try name match
                match s {
                    $($name => return Ok(Self::$variant),)*
                    _ => {}
                }
                // Try case-insensitive
                let lower = s.to_lowercase();

                // Common aliases (human-friendly / legacy spellings)
                match lower.as_str() {
                    "masakhane-ner" | "masakhane_ner" | "masakhane ner" => return Ok(Self::MasakhaNER),
                    "masakhane-news" | "masakhane_news" | "masakhane news" => return Ok(Self::MasakhaNEWS),
                    "chinese-historical-ie" | "chinese_historical_ie" | "ancient-chinese-ner" | "ancient_chinese_ner" => {
                        return Ok(Self::CHisIEC);
                    }
                    _ => {}
                }
                $(
                    if stringify!($variant).to_lowercase() == lower {
                        return Ok(Self::$variant);
                    }
                )*

                // Try normalized matching (handles `wnut-17`, `mit_movie`, `ch-is-iec`, etc.)
                // NOTE: We intentionally only strip separators/punctuation; we do not attempt
                // fuzzy/approximate matching.
                fn normalize(inp: &str) -> String {
                    inp.chars()
                        .filter(|c| c.is_ascii_alphanumeric())
                        .flat_map(|c| c.to_lowercase())
                        .collect()
                }

                let needle = normalize(s);
                if needle.is_empty() {
                    return Err("Unknown dataset: <empty>".to_string());
                }

                let mut matches = Vec::new();
                $(
                    if normalize(stringify!($variant)) == needle || normalize($name) == needle {
                        matches.push(Self::$variant);
                    }
                )*

                match matches.as_slice() {
                    [only] => Ok(*only),
                    [] => Err(format!("Unknown dataset: {}", s)),
                    _ => Err(format!("Ambiguous dataset id '{}': {:?}", s, matches)),
                }
            }
        }
    };
}

/// Helper macro to implement category checking methods.
///
/// This generates `is_X()` methods and `all_X()` functions for each category.
#[macro_export]
macro_rules! impl_category_checks {
    ($($variant:ident => [$($cat:ident),*]),*) => {
        // Individual category checks
        /// Check if this dataset is for coreference resolution.
        #[must_use]
        pub fn is_coreference(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], coref),)*
            }
        }

        /// Check if this dataset is biomedical domain.
        #[must_use]
        pub fn is_biomedical(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], biomedical),)*
            }
        }

        /// Check if this dataset is social media domain.
        #[must_use]
        pub fn is_social_media(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], social_media),)*
            }
        }

        /// Check if this dataset is for relation extraction.
        #[must_use]
        pub fn is_relation_extraction(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], relation_extraction),)*
            }
        }

        /// Check if this dataset is multilingual.
        #[must_use]
        pub fn is_multilingual(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], multilingual),)*
            }
        }

        /// Check if this dataset is for historical texts.
        #[must_use]
        pub fn is_historical(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], historical),)*
            }
        }

        /// Check if this dataset is for bias evaluation.
        #[must_use]
        pub fn is_bias_evaluation(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], bias_evaluation),)*
            }
        }

        /// Check if this dataset is indigenous language.
        #[must_use]
        pub fn is_indigenous(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], indigenous),)*
            }
        }

        /// Check if this dataset is an African language dataset.
        #[must_use]
        pub fn is_african_language(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], african_language),)*
            }
        }

        /// Check if this dataset is for literary/fiction texts.
        #[must_use]
        pub fn is_literary(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], literary),)*
            }
        }

        /// Check if this dataset has nested NER annotations.
        #[must_use]
        pub fn is_nested_ner(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], nested_ner),)*
            }
        }

        /// Check if this dataset has discontinuous NER annotations.
        #[must_use]
        pub fn is_discontinuous_ner(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], discontinuous_ner),)*
            }
        }

        /// Check if this dataset is dialogue/conversational.
        #[must_use]
        pub fn is_dialogue(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], dialogue),)*
            }
        }

        /// Check if this dataset is NER (not coreference).
        #[must_use]
        pub fn is_ner(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], ner),)*
            }
        }

        /// Check if this dataset is for cross-document event coreference.
        #[must_use]
        pub fn is_event_coref(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], event_coref),)*
            }
        }

        /// Check if this dataset is for ancient/classical languages.
        #[must_use]
        pub fn is_ancient(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], ancient),)*
            }
        }

        /// Check if this dataset is for abstract anaphora/discourse deixis.
        #[must_use]
        pub fn is_abstract_anaphora(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], abstract_anaphora),)*
            }
        }

        /// Check if this dataset is for low-resource/endangered languages.
        #[must_use]
        pub fn is_low_resource(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], low_resource),)*
            }
        }

        /// Check if this dataset is for constructed languages.
        #[must_use]
        pub fn is_constructed(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], constructed),)*
            }
        }

        /// Check if this dataset is for arcane/specialized domains.
        #[must_use]
        pub fn is_arcane_domain(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], arcane_domain),)*
            }
        }

        /// Check if this dataset is for adversarial/robustness evaluation.
        #[must_use]
        pub fn is_adversarial(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], adversarial),)*
            }
        }

        /// Check if this dataset is for speech/audio NER.
        #[must_use]
        pub fn is_speech(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], speech),)*
            }
        }

        /// Check if this dataset is for entity linking / named entity disambiguation.
        #[must_use]
        pub fn is_entity_linking(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], entity_linking),)*
            }
        }

        /// Check if this dataset is for long-document processing (>10k tokens).
        #[must_use]
        pub fn is_long_document(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], long_document),)*
            }
        }

        /// Check if this dataset is clinical/medical domain.
        #[must_use]
        pub fn is_clinical(&self) -> bool {
            match self {
                $(Self::$variant => $crate::has_category!([$($cat),*], clinical),)*
            }
        }

        // Group accessors
        /// Get all NER datasets.
        pub fn all_ner() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_ner()).collect()
        }

        /// Get all coreference datasets.
        pub fn all_coref() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_coreference()).collect()
        }

        /// Get all biomedical datasets.
        pub fn all_biomedical() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_biomedical()).collect()
        }

        /// Get all multilingual datasets.
        pub fn all_multilingual() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_multilingual()).collect()
        }

        /// Get all historical datasets.
        pub fn all_historical() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_historical()).collect()
        }

        /// Get all indigenous language datasets.
        pub fn all_indigenous() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_indigenous()).collect()
        }

        /// Get all literary/fiction datasets.
        pub fn all_literary() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_literary()).collect()
        }

        /// Get all nested NER datasets.
        pub fn all_nested_ner() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_nested_ner()).collect()
        }

        /// Get all relation extraction datasets.
        pub fn all_relation_extraction() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_relation_extraction()).collect()
        }

        /// Get all event coreference datasets.
        pub fn all_event_coref() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_event_coref()).collect()
        }

        /// Get all ancient/classical language datasets.
        pub fn all_ancient() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_ancient()).collect()
        }

        /// Get all abstract anaphora datasets.
        pub fn all_abstract_anaphora() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_abstract_anaphora()).collect()
        }

        /// Get all low-resource language datasets.
        pub fn all_low_resource() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_low_resource()).collect()
        }

        /// Get all constructed language datasets.
        pub fn all_constructed() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_constructed()).collect()
        }

        /// Get all arcane domain datasets.
        pub fn all_arcane_domain() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_arcane_domain()).collect()
        }

        /// Get all adversarial/robustness datasets.
        pub fn all_adversarial() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_adversarial()).collect()
        }

        /// Get all speech NER datasets.
        pub fn all_speech() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_speech()).collect()
        }

        /// Get all social media datasets.
        pub fn all_social_media() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_social_media()).collect()
        }

        /// Get all discontinuous NER datasets.
        pub fn all_discontinuous_ner() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_discontinuous_ner()).collect()
        }

        /// Get all bias evaluation datasets.
        pub fn all_bias_evaluation() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_bias_evaluation()).collect()
        }

        /// Get all entity linking / NED datasets.
        pub fn all_entity_linking() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_entity_linking()).collect()
        }

        /// Get all long-document datasets.
        pub fn all_long_document() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_long_document()).collect()
        }

        /// Get all clinical datasets.
        pub fn all_clinical() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_clinical()).collect()
        }

        /// Get all dialogue/conversational datasets.
        pub fn all_dialogue() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| d.is_dialogue()).collect()
        }

        /// Get all African language datasets.
        pub fn all_african_languages() -> Vec<&'static DatasetId> {
            Self::all().iter().filter(|d| (*d).is_african_language()).collect()
        }

        /// Quick evaluation set (fast downloads, common benchmarks).
        pub fn quick() -> Vec<&'static DatasetId> {
            Self::all().iter()
                .filter(|d| matches!(d,
                    DatasetId::WikiGold |
                    DatasetId::Wnut17 |
                    DatasetId::GAP
                ))
                .collect()
        }

        /// Medium evaluation set (moderate size, diverse domains).
        pub fn medium() -> Vec<&'static DatasetId> {
            Self::all().iter()
                .filter(|d| matches!(d,
                    DatasetId::WikiGold |
                    DatasetId::Wnut17 |
                    DatasetId::MitMovie |
                    DatasetId::MitRestaurant |
                    DatasetId::CoNLL2003Sample |
                    DatasetId::BC5CDR |
                    DatasetId::GAP |
                    DatasetId::PreCo
                ))
                .collect()
        }
    };
}

/// Helper macro to check if a list of categories contains a specific category.
#[macro_export]
macro_rules! has_category {
    ([], $target:ident) => { false };
    ([$first:ident $(, $rest:ident)*], $target:ident) => {
        stringify!($first) == stringify!($target) || $crate::has_category!([$($rest),*], $target)
    };
}

/// Helper macro to handle optional string fields in dataset definitions.
#[macro_export]
macro_rules! optional_field {
    () => {
        None
    };
    ($value:literal) => {
        Some($value)
    };
}

/// Helper macro to handle optional numeric fields in dataset definitions.
#[macro_export]
macro_rules! optional_field_num {
    () => {
        None
    };
    ($value:literal) => {
        Some($value)
    };
}

/// Helper macro to handle optional access_status fields in dataset definitions.
/// Returns DatasetAccessibility::Public as the default when not specified.
#[macro_export]
macro_rules! optional_access_status {
    () => {
        $crate::eval::dataset_registry::DatasetAccessibility::Public
    };
    (Public) => {
        $crate::eval::dataset_registry::DatasetAccessibility::Public
    };
    (HuggingFace) => {
        $crate::eval::dataset_registry::DatasetAccessibility::HuggingFace
    };
    (Local) => {
        $crate::eval::dataset_registry::DatasetAccessibility::Local
    };
    (Registration) => {
        $crate::eval::dataset_registry::DatasetAccessibility::Registration
    };
    (ContactAuthors) => {
        $crate::eval::dataset_registry::DatasetAccessibility::ContactAuthors
    };
    (NotYetReleased) => {
        $crate::eval::dataset_registry::DatasetAccessibility::NotYetReleased
    };
    (DependsOnOther) => {
        $crate::eval::dataset_registry::DatasetAccessibility::DependsOnOther
    };
    (Deprecated) => {
        $crate::eval::dataset_registry::DatasetAccessibility::Deprecated
    };
}

// Re-export for use in other modules
pub use define_datasets;
pub use has_category;
pub use impl_category_checks;
pub use optional_field;
pub use optional_field_num;

// =============================================================================
// The Dataset Registry - Single Source of Truth
// =============================================================================
//
// To add a dataset:
// 1. Add an entry below with all metadata
// 2. Run tests: `cargo test -p anno-eval`
// 3. The TOML is generated automatically for documentation
//
// Categories available:
// - ner: Standard NER dataset
// - coref: Coreference resolution
// - biomedical: Medical/biological domain
// - clinical: Clinical/medical notes domain
// - social_media: Twitter, Reddit, etc.
// - relation_extraction: Entity relations
// - entity_linking: Entity linking / named entity disambiguation
// - multilingual: Multiple languages
// - historical: Historical texts
// - bias_evaluation: Bias benchmarks
// - indigenous: Indigenous languages
// - literary: Fiction/literature
// - nested_ner: Nested entity annotations
// - discontinuous_ner: Non-contiguous spans
// - dialogue: Conversational text
// - event_coref: Cross-document event coreference
// - ancient: Ancient/classical languages (Latin, Greek, Sumerian)
// - abstract_anaphora: Shell nouns, discourse deixis, bridging
// - low_resource: Low-resource/endangered languages
// - constructed: Constructed languages (Esperanto, Klingon)
// - arcane_domain: Specialized domains (astronomy, archaeology)
// - adversarial: Robustness/adversarial evaluation
// - speech: Audio/speech NER
// - long_document: Long-document (>10k tokens) processing

define_datasets! {
    // =========================================================================
    // Core NER Datasets
    // =========================================================================
    WikiGold {
        name: "WikiGold",
        description: "Wikipedia-based NER (PER, LOC, ORG, MISC). Historically significant as early Wikipedia NER resource.",
        url: "https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/wikigold/CONLL-format/data/wikigold.conll.txt",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "wikipedia",
        license: "CC-BY-4.0",
        citation: "Balasuriya et al. (2009)",
        paper_url: "https://aclanthology.org/U09-1001/",
        year: 2009,
        format: "CoNLL",
        annotation_scheme: "IOB2",
        size_hint: "~40k tokens, ~3,500 entities",
        example: "Japan B-LOC\n's O\nMinister O\nShinzo B-PER\nAbe I-PER\nvisited O\nthe O\nUnited B-LOC\nStates I-LOC\n. O",
        splits: ["all"],
        tasks: ["ner"],
        expected_docs: 145,
        categories: [ner],
    },
    Wnut17 {
        name: "WNUT-17",
        description: "Social media NER with emerging entities. Created to evaluate models on rare/emerging entities in noisy social text.",
        url: "https://raw.githubusercontent.com/leondz/emerging_entities_17/master/emerging.test.annotated",
        entity_types: ["person", "location", "corporation", "product", "creative-work", "group"],
        language: "en",
        domain: "social_media",
        license: "CC-BY-4.0",
        citation: "Derczynski et al. (2017)",
        paper_url: "https://aclanthology.org/W17-4418/",
        year: 2017,
        format: "CoNLL",
        annotation_scheme: "BIO",
        size_hint: "~65k tokens, 1,000 tweets",
        notes: "89% unseen entities in test set - excellent for OOD evaluation; shared task at W-NUT workshop",
        tasks: ["ner"],
        hf_id: "leondz/wnut_17",
        access_status: HuggingFace,
        categories: [ner, social_media],
    },
    MitMovie {
        name: "MIT Movie",
        description: "Movie domain slot filling NER. Created at MIT SLS for spoken language understanding research.",
        url: "https://groups.csail.mit.edu/sls/downloads/movie/engtest.bio",
        entity_types: ["Actor", "Director", "Genre", "Title", "Year", "Song", "Character", "Plot", "Rating"],
        language: "en",
        domain: "entertainment",
        license: "Research",
        citation: "Liu et al. (2013)",
        paper_url: "https://groups.csail.mit.edu/sls/publications/2013/Liu_ASRU_2013.pdf",
        year: 2013,
        format: "BIO",
        annotation_scheme: "BIO",
        size_hint: "~12k utterances",
        example: "show O\nme O\naction B-Genre\nmovies O\ndirected O\nby O\nsteven B-Director\nspielberg I-Director",
        tasks: ["ner", "slot_filling"],
        categories: [ner],
    },
    MitRestaurant {
        name: "MIT Restaurant",
        description: "Restaurant domain slot filling NER. Part of MIT SLS spoken dialogue systems research.",
        url: "https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttest.bio",
        entity_types: ["Amenity", "Cuisine", "Dish", "Hours", "Location", "Price", "Rating", "Restaurant_Name"],
        language: "en",
        domain: "restaurant",
        license: "Research",
        citation: "Liu et al. (2013)",
        paper_url: "https://groups.csail.mit.edu/sls/publications/2013/Liu_ASRU_2013.pdf",
        year: 2013,
        format: "BIO",
        annotation_scheme: "BIO",
        size_hint: "~8k utterances",
        example: "find O\nitalian B-Cuisine\nrestaurants O\nin O\nboston B-Location\nwith O\noutdoor B-Amenity\nseating I-Amenity",
        tasks: ["ner", "slot_filling"],
        categories: [ner],
    },
    CoNLL2003Sample {
        name: "CoNLL-2003 Sample",
        description: "Classic news NER benchmark from Reuters Corpus. Foundational dataset that established modern NER evaluation standards.",
        url: "https://huggingface.co/datasets/tner/conll2003",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "news",
        license: "Research",
        citation: "Tjong Kim Sang & De Meulder (2003)",
        paper_url: "https://aclanthology.org/W03-0419/",
        year: 2003,
        format: "CoNLL",
        annotation_scheme: "IOB2",
        size_hint: "~300k tokens, ~35k entities",
        example: "EU B-ORG\nrejects O\nGerman B-MISC\ncall O\nto O\nboycott O\nBritish B-MISC\nlamb O\n. O",
        notes: "Known annotation noise; see CleanCoNLL (2023) for one audit/correction pass",
        tasks: ["ner"],
        hf_id: "tner/conll2003",
        access_status: HuggingFace,
        categories: [ner],
    },
    OntoNotesSample {
        name: "OntoNotes Sample",
        description: "Multi-genre 18-type NER from OntoNotes 5.0. Rich annotation including coreference, parsing, and PropBank.",
        url: "https://huggingface.co/datasets/tner/ontonotes5",
        entity_types: ["PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", "PERCENT", "NORP", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "QUANTITY", "ORDINAL", "CARDINAL"],
        language: "en",
        domain: "news",
        license: "LDC",
        citation: "Weischedel et al. (2013)",
        paper_url: "https://catalog.ldc.upenn.edu/LDC2013T19",
        year: 2013,
        format: "CoNLL",
        annotation_scheme: "IOB2",
        size_hint: "~1.6M tokens, ~128k entities",
        example: "The B-ORG\nEuropean I-ORG\nUnion I-ORG\nannounced O\nMonday B-DATE\nthat O\nthe O\n$ B-MONEY\n10 I-MONEY\nmillion I-MONEY\nwill O\ngo O\nto O\nUkraine B-GPE\n. O",
        notes: "Full corpus requires LDC license; sample available on HuggingFace; includes 7 genres",
        hf_id: "tner/ontonotes5",
        access_status: HuggingFace,
        categories: [ner],
    },
    MultiNERD {
        name: "MultiNERD",
        description: "Large multilingual NER covering 10 languages. Created to address scarcity of multilingual fine-grained NER data.",
        url: "https://huggingface.co/datasets/Babelscape/multinerd/resolve/main/test/test_en.jsonl",
        entity_types: ["PER", "LOC", "ORG", "ANIM", "BIO", "CEL", "DIS", "EVE", "FOOD", "INST", "MEDIA", "MYTH", "PLANT", "TIME", "VEHI"],
        language: "en",
        domain: "wikipedia",
        license: "CC-BY-SA-4.0",
        citation: "Tedeschi & Navigli (2022)",
        paper_url: "https://aclanthology.org/2022.findings-naacl.60/",
        year: 2022,
        format: "JSONL",
        annotation_scheme: "BIO",
        size_hint: "~1M sentences across 10 languages",
        example: "Marie Curie (PER) discovered radium at the University of Paris (ORG) in France (LOC).",
        tasks: ["ner"],
        hf_id: "Babelscape/multinerd",
        access_status: HuggingFace,
        categories: [ner, multilingual],
    },

    FewNERD {
        name: "Few-NERD",
        description: "Fine-grained NER with 66 types in 8 coarse categories. Designed for few-shot learning evaluation.",
        url: "https://huggingface.co/datasets/DFKI-SLT/few-nerd",
        entity_types: ["person", "location", "organization", "building", "art", "product", "event", "other"],
        language: "en",
        domain: "wikipedia",
        license: "CC-BY-SA-4.0",
        citation: "Ding et al. (2021)",
        paper_url: "https://aclanthology.org/2021.acl-long.248/",
        year: 2021,
        format: "TSV",
        size_hint: "188k sentences, 66 fine-grained types",
        example: "Jensen Huang (person-entrepreneur) founded NVIDIA (organization-company) in Santa Clara (location-city), California.",
        notes: "Hierarchical type system; benchmark for few-shot and fine-grained NER. Note: HuggingFace API may return 422 errors; consider downloading locally.",
        tasks: ["ner"],
        hf_id: "DFKI-SLT/few-nerd",
        access_status: HuggingFace,
        categories: [ner],
    },

    CrossNER {
        name: "CrossNER",
        description: "Cross-domain NER across 5 domains: politics, science, music, literature, AI. Tests domain transfer.",
        url: "https://huggingface.co/datasets/DFKI-SLT/cross_ner",
        entity_types: ["PER", "ORG", "LOC", "MISC", "Domain-specific"],
        language: "en",
        domain: "multi-domain",
        license: "MIT",
        citation: "Liu et al. (2021)",
        paper_url: "https://aclanthology.org/2021.aaai.main.672/",
        year: 2021,
        format: "CoNLL",
        size_hint: "5 domains, ~10k sentences each",
        notes: "Tests cross-domain transfer; domain-specific entity types. Use HuggingFace datasets library to load.",
        hf_id: "DFKI-SLT/cross_ner",
        hf_config: "politics",
        access_status: HuggingFace,
        categories: [ner],
    },

    FabNER {
        name: "FabNER",
        description: "Manufacturing domain NER. 12 entity types for Industry 4.0 applications.",
        url: "https://huggingface.co/datasets/DFKI-SLT/fabner",
        entity_types: ["Material", "Process", "Machine", "Product", "Property"],
        language: "en",
        domain: "manufacturing",
        license: "CC-BY-4.0",
        citation: "Kumar et al. (2022)",
        paper_url: "https://aclanthology.org/2022.lrec-1.227/",
        year: 2022,
        format: "CoNLL",
        size_hint: "~14k sentences, 12 entity types",
        notes: "Specialized manufacturing/engineering domain; Industry 4.0",
        hf_id: "DFKI-SLT/fabner",
        access_status: HuggingFace,
        categories: [ner],
    },

    BroadTwitterCorpus {
        name: "Broad Twitter Corpus",
        description: "Twitter NER across multiple time periods. Tests temporal robustness of NER systems.",
        url: "https://huggingface.co/datasets/tner/btc",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "social_media",
        license: "CC-BY-4.0",
        citation: "Derczynski et al. (2016)",
        paper_url: "https://aclanthology.org/C16-1111/",
        year: 2016,
        format: "BIO",
        size_hint: "~9k tweets, stratified by time period",
        notes: "Temporal stratification; tests model robustness to language evolution",
        hf_id: "tner/btc",
        access_status: HuggingFace,
        categories: [ner, social_media],
    },

    WikiNeural {
        name: "WikiNeural",
        description: "Silver-standard multilingual NER from Wikipedia. 9 languages with automatic annotation.",
        url: "https://huggingface.co/datasets/Babelscape/wikineural",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "mul",
        domain: "wikipedia",
        license: "CC-BY-SA-4.0",
        citation: "Tedeschi et al. (2021)",
        paper_url: "https://aclanthology.org/2021.findings-emnlp.215/",
        year: 2021,
        format: "CoNLL",
        size_hint: "9 languages, ~100k sentences each",
        notes: "Automatically generated silver annotations; useful for pre-training",
        hf_id: "Babelscape/wikineural",
        hf_config: "en",
        access_status: HuggingFace,
        categories: [ner, multilingual],
    },

    PolyglotNER {
        name: "Polyglot-NER",
        description: "Massively multilingual NER. 40 languages with silver annotations from Wikipedia.",
        url: "https://huggingface.co/datasets/rmyeid/polyglot_ner",
        entity_types: ["PER", "LOC", "ORG"],
        language: "mul",
        domain: "wikipedia",
        license: "Research",
        citation: "Al-Rfou et al. (2015)",
        paper_url: "https://aclanthology.org/C14-1078/",
        year: 2015,
        format: "CoNLL",
        size_hint: "40 languages, silver annotations",
        notes: "Largest language coverage; silver annotations via Wikipedia links",
        tasks: ["ner"],
        hf_id: "rmyeid/polyglot_ner",
        access_status: HuggingFace,
        categories: [ner, multilingual],
    },

    UniversalNERBench {
        name: "Universal NER",
        description: "Cross-lingual NER benchmark spanning 13 diverse languages. Tests zero-shot transfer.",
        url: "https://github.com/UniversalNER/uner_code",
        entity_types: ["PER", "LOC", "ORG"],
        language: "mul",
        domain: "mixed",
        license: "CC-BY-4.0",
        citation: "Malmasi et al. (2022)",
        paper_url: "https://aclanthology.org/2022.emnlp-main.13/",
        year: 2022,
        format: "CoNLL",
        size_hint: "13 languages, gold annotations",
        notes: "Tests cross-lingual zero-shot transfer; diverse language families",
        categories: [ner, multilingual],
    },

    CoNLL2002 {
        name: "CoNLL-2002",
        description: "Spanish and Dutch NER from CoNLL 2002 shared task. Multi-language NER benchmark.",
        // NOTE: Do not point this at a HuggingFace dataset *page* unless the repo exposes
        // raw data files or datasets-server rows. `eriktks/conll2002` is a dataset-script
        // repo (e.g. `conll2002.py`) and does not expose raw data via the Hub API, so our
        // loader cannot fetch it as a plain CoNLL file.
        //
        // Use the language-specific, directly-downloadable splits instead:
        // - `CoNLL2002Spanish`
        // - `CoNLL2002Dutch`
        url: "",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "mul",
        domain: "news",
        license: "Research",
        citation: "Tjong Kim Sang (2002)",
        paper_url: "https://aclanthology.org/W02-2024/",
        year: 2002,
        format: "CoNLL",
        annotation_scheme: "BIO",
        size_hint: "Spanish + Dutch news articles",
        notes: "Meta entry only; use `CoNLL2002Spanish` / `CoNLL2002Dutch` for loadable data.",
        tasks: ["blocked"],
        access_status: Deprecated,
        categories: [ner, multilingual],
    },

    TweetNER7 {
        name: "TweetNER7",
        description: "Twitter NER across 7 entity types. Fine-grained social media NER with temporal annotations.",
        url: "https://huggingface.co/datasets/tner/tweetner7",
        entity_types: ["person", "location", "corporation", "product", "creative_work", "group", "event"],
        language: "en",
        domain: "social_media",
        license: "CC-BY-4.0",
        citation: "Ushio et al. (2022)",
        paper_url: "https://aclanthology.org/2022.findings-emnlp.304/",
        year: 2022,
        format: "JSONL",
        size_hint: "~12k tweets",
        notes: "Temporal distribution shift; tests robustness to evolving language",
        hf_id: "tner/tweetner7",
        hf_config: "tweetner7",
        access_status: HuggingFace,
        categories: [ner, social_media],
    },

    GoogleRE {
        name: "Google-RE",
        description: "Google Relation Extraction dataset. Wikipedia sentences with relation annotations.",
        url: "https://raw.githubusercontent.com/google-research-datasets/relation-extraction-corpus/master/20130403-place_of_birth.json",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "wikipedia",
        license: "CC-BY-4.0",
        citation: "Levy et al. (2017)",
        paper_url: "https://aclanthology.org/D17-1004/",
        year: 2017,
        format: "JSON",
        size_hint: "~60k relation triples",
        notes: "Clean relation extraction; commonly used for zero-shot RE evaluation; using place_of_birth subset",
        tasks: ["re"],
        access_status: Public,
        categories: [relation_extraction],
    },

    NYTFB {
        name: "NYT-FB",
        description: "New York Times with Freebase relations. Distant supervision relation extraction.",
        url: "https://github.com/thunlp/OpenNRE",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "news",
        license: "Research",
        citation: "Riedel et al. (2010)",
        paper_url: "https://aclanthology.org/N10-1114/",
        year: 2010,
        format: "JSONL",
        size_hint: "~570k sentences, 53 relations",
        notes: "Classic distant supervision RE; noisy but large-scale",
        categories: [relation_extraction],
    },

    REBEL {
        name: "REBEL",
        description: "Relation Extraction By End-to-end Language generation. Large-scale RE dataset.",
        url: "https://huggingface.co/datasets/Babelscape/rebel-dataset",
        entity_types: ["PER", "LOC", "ORG", "Event"],
        language: "en",
        domain: "wikipedia",
        license: "CC-BY-SA-4.0",
        citation: "Huguet Cabot & Navigli (2021)",
        paper_url: "https://aclanthology.org/2021.findings-emnlp.204/",
        year: 2021,
        format: "JSONL",
        size_hint: "~6M triples from Wikipedia",
        notes: "Large-scale; generative RE approach; 220 relation types",
        categories: [relation_extraction],
    },

    MultiCoNER {
        name: "MultiCoNER",
        description: "Multilingual Complex NER. 11 languages with fine-grained and complex entities.",
        url: "https://huggingface.co/datasets/samanjoy2/multiconer_v1",
        entity_types: ["PER", "LOC", "CORP", "GRP", "PROD", "CW"],
        language: "mul",
        domain: "mixed",
        license: "CC-BY-4.0",
        citation: "Malmasi et al. (2022)",
        paper_url: "https://aclanthology.org/2022.semeval-1.196/",
        year: 2022,
        format: "CoNLL",
        size_hint: "11 languages, ~1.1M tokens",
        notes: "SemEval-2022 shared task; complex entities from diverse sources. Community mirror of original.",
        hf_id: "samanjoy2/multiconer_v1",
        access_status: HuggingFace,
        categories: [ner, multilingual],
    },

    MultiCoNERv2 {
        name: "MultiCoNER v2",
        description: "MultiCoNER v2 with expanded languages and fine-grained types.",
        url: "https://huggingface.co/datasets/MultiCoNER/multiconer_v2",
        entity_types: ["PER", "LOC", "CORP", "GRP", "PROD", "CW", "Medical", "Scientist"],
        language: "mul",
        domain: "mixed",
        license: "CC-BY-4.0",
        citation: "Fetahu et al. (2023)",
        paper_url: "https://aclanthology.org/2023.semeval-1.43/",
        year: 2023,
        format: "CoNLL",
        size_hint: "12 languages, fine-grained types",
        notes: "SemEval-2023 shared task; expanded from v1 with more types.",
        tasks: ["ner"],
        hf_id: "MultiCoNER/multiconer_v2",
        access_status: HuggingFace,
        categories: [ner, multilingual],
    },

    // =========================================================================
    // Biomedical NER Datasets
    // =========================================================================
    BC5CDR {
        name: "BC5CDR",
        description: "Biomedical NER for diseases and chemicals. Created for BioCreative V CDR task, a major biomedical NLP benchmark.",
        url: "https://huggingface.co/datasets/tner/bc5cdr",
        entity_types: ["Chemical", "Disease"],
        language: "en",
        domain: "biomedical",
        license: "Public",
        citation: "Li et al. (2016)",
        paper_url: "https://academic.oup.com/database/article/doi/10.1093/database/baw068/2630414",
        year: 2016,
        format: "BIO",
        annotation_scheme: "BIO",
        size_hint: "~1500 PubMed abstracts, ~14k mentions",
        example: "Aspirin B-Chemical\ninduced O\nhepatotoxicity B-Disease\nwas O\nobserved O\n. O",
        tasks: ["ner"],
        hf_id: "tner/bc5cdr",
        access_status: HuggingFace,
        categories: [ner, biomedical],
    },
    NCBIDisease {
        name: "NCBI Disease",
        description: "NCBI disease mentions corpus. Foundational resource for disease NER from NIH.",
        url: "https://raw.githubusercontent.com/shreyashub/BioFLAIR/master/data/ner/NCBI-disease/test.txt",
        entity_types: ["Disease"],
        language: "en",
        domain: "biomedical",
        license: "Public",
        citation: "Dogan et al. (2014)",
        paper_url: "https://www.sciencedirect.com/science/article/pii/S1532046413001974",
        year: 2014,
        format: "BIO",
        annotation_scheme: "BIO",
        size_hint: "~800 PubMed abstracts, ~6k mentions",
        example: "The O\npatient O\nwas O\ndiagnosed O\nwith O\ntype B-Disease\n2 I-Disease\ndiabetes I-Disease\n. O",
        tasks: ["ner"],
        hf_id: "ncbi_disease",
        access_status: HuggingFace,
        categories: [ner, biomedical],
    },
    GENIA {
        name: "GENIA",
        description: "Biomedical NER for molecular biology. First large-scale biomedical NER corpus; historically significant.",
        url: "https://huggingface.co/datasets/chufangao/GENIA-NER",
        entity_types: ["DNA", "RNA", "protein", "cell_line", "cell_type"],
        language: "en",
        domain: "biomedical",
        license: "GENIA Project License",
        citation: "Kim et al. (2003)",
        paper_url: "https://academic.oup.com/bioinformatics/article/19/suppl_1/i180/227927",
        year: 2003,
        format: "XML",
        annotation_scheme: "Standoff",
        size_hint: "2000 MEDLINE abstracts, ~100k entities",
        example: "The B-protein\nNF-kappa B I-protein\nprotein I-protein\nbinds O\nto O\nthe B-DNA\nkappa B I-DNA\nbinding I-DNA\nsite I-DNA\n. O",
        notes: "Nested entities common; requires special handling; pioneered biomedical NER",
        hf_id: "chufangao/GENIA-NER",
        access_status: HuggingFace,
        categories: [ner, biomedical],
    },

    AnatEM {
        name: "AnatEM",
        description: "Anatomical entity mention corpus. 1,212 PubMed abstracts with anatomical structures.",
        url: "https://huggingface.co/datasets/disi-unibo-nlp/AnatEM",
        entity_types: ["Anatomy"],
        language: "en",
        domain: "biomedical",
        license: "CC-BY-4.0",
        citation: "Ohta et al. (2012)",
        paper_url: "https://aclanthology.org/W12-2402/",
        year: 2012,
        format: "Standoff",
        size_hint: "1,212 abstracts, ~7k entity mentions",
        notes: "Fine-grained anatomical mentions; standalone or nested within other entities",
        hf_id: "disi-unibo-nlp/AnatEM",
        access_status: HuggingFace,
        categories: [ner, biomedical],
    },

    BC2GM {
        name: "BC2GM",
        description: "BioCreative II Gene Mention recognition. Gold-standard gene/protein name tagging.",
        url: "https://huggingface.co/datasets/spyysalo/bc2gm_corpus",
        entity_types: ["Gene", "Protein"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "Smith et al. (2008)",
        paper_url: "https://genomebiology.biomedcentral.com/articles/10.1186/gb-2008-9-s2-s2",
        year: 2008,
        format: "IOB2",
        size_hint: "20k sentences, ~24k gene mentions",
        example: "The B-Gene\np53 I-Gene\nprotein I-Gene\nregulates O\ncell O\ncycle O\narrest O\n. O",
        notes: "Classic benchmark for gene/protein NER; BioCreative shared task",
        tasks: ["ner"],
        hf_id: "spyysalo/bc2gm_corpus",
        access_status: HuggingFace,
        categories: [ner, biomedical],
    },

    BC4CHEMD {
        name: "BC4CHEMD",
        description: "BioCreative IV Chemical Entity Mention Detection. Drug and chemical name recognition.",
        url: "https://huggingface.co/datasets/chintagunta85/bc4chemd",
        entity_types: ["Chemical"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "Krallinger et al. (2015)",
        paper_url: "https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S2",
        year: 2015,
        format: "IOB2",
        size_hint: "10k PubMed abstracts, ~84k chemical mentions",
        example: "Treatment O\nwith O\nB-Chemical\naspirin I-Chemical\nreduced O\ninflammation O\n. O",
        notes: "Chemical NER benchmark; includes IUPAC names, trivial names, abbreviations",
        tasks: ["ner"],
        hf_id: "chintagunta85/bc4chemd",
        access_status: HuggingFace,
        categories: [ner, biomedical],
    },

    // =========================================================================
    // Coreference Datasets
    // =========================================================================
    GAP {
        name: "GAP",
        description: "Gender Ambiguous Pronoun resolution. Google's benchmark for exposing gender bias in coreference systems.",
        url: "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv",
        entity_types: ["PER"],
        language: "en",
        domain: "wikipedia",
        license: "Apache-2.0",
        citation: "Webster et al. (2018)",
        paper_url: "https://aclanthology.org/Q18-1042/",
        year: 2018,
        format: "TSV",
        size_hint: "8,908 pronoun-name pairs",
        example: "ID\tText\tPronoun\tA\tB\tA-coref\ntest-1\tZoe met Alice and she waved.\tshe\tZoe\tAlice\tFALSE",
        notes: "Designed to expose gender bias; Kaggle shared task; balanced male/female",
        tasks: ["coref"],
        hf_id: "google-gap-coreference/gap",
        access_status: HuggingFace,
        categories: [coref, bias_evaluation],
    },
    PreCo {
        name: "PreCo",
        description: "Large-scale coreference from PreCo reading comprehension corpus. 10x larger than OntoNotes.",
        url: "https://huggingface.co/datasets/coref-data/preco_raw",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "reading_comprehension",
        license: "CC-BY-4.0",
        citation: "Chen et al. (2018)",
        paper_url: "https://aclanthology.org/D18-1016/",
        year: 2018,
        format: "JSONL",
        annotation_scheme: "CoNLLCoref",
        size_hint: "38k documents, includes singletons",
        notes: "Preschool vocabulary for cleaner evaluation; largest public coref corpus",
        splits: ["train", "dev", "test"],
        tasks: ["coref"],
        hf_id: "coref-data/preco_raw",
        access_status: HuggingFace,
        categories: [coref],
    },
    LitBank {
        name: "LitBank",
        description: "Literary coreference. 100 public-domain English fiction works (1719-1922) with ACE-style entities.",
        url: "https://raw.githubusercontent.com/dbamman/litbank/master/coref/brat/1023_bleak_house_brat.ann",
        entity_types: ["PER", "LOC", "ORG", "GPE", "FAC", "VEH"],
        language: "en",
        domain: "literature",
        license: "CC-BY-4.0",
        citation: "Bamman et al. (2019)",
        paper_url: "https://aclanthology.org/P19-1353/",
        year: 2019,
        format: "BRAT",
        annotation_scheme: "Standoff",
        size_hint: "100 novels, ~2k tokens each",
        notes: "Focus on character coreference; includes event coref; public domain texts",
        splits: ["all"],
        tasks: ["ner", "coref", "event_coref"],
        expected_docs: 100,
        categories: [coref, literary],
    },
    ECBPlus {
        name: "ECB+",
        description: "Event Coreference Bank Plus. Standard benchmark for cross-document event coreference resolution.",
        url: "https://raw.githubusercontent.com/cltl/ecbPlus/master/ECB%2B_LREC2014/ECBplus_coreference_sentences.csv",
        entity_types: ["EVENT", "TIME", "LOC", "PARTICIPANT"],
        language: "en",
        domain: "news",
        license: "CC-BY-3.0",
        citation: "Cybulska & Vossen (2014)",
        paper_url: "https://aclanthology.org/L14-1646/",
        year: 2014,
        format: "Custom",
        size_hint: "43 topics, 982 docs, ~7k events",
        example: "Doc1: 'The earthquake [struck] at 3am.' Doc2: 'The [tremor] caused damage.'\nEvents: struck_1, tremor_2 -> coreferent (same event)",
        notes: "De facto CDCR standard; topic-clustered structure may cause overfitting",
        splits: ["train", "dev", "test"],
        tasks: ["coref", "event_coref", "cdcr"],
        expected_docs: 982,
        categories: [coref, event_coref],
    },

    OntoNotesCoref {
        name: "OntoNotes Coreference",
        description: "OntoNotes 5.0 coreference annotations. Gold-standard multi-genre coref including WSJ, broadcast, web.",
        url: "https://catalog.ldc.upenn.edu/LDC2013T19",
        entity_types: ["PER", "ORG", "GPE", "NORP"],
        language: "en",
        domain: "mixed",
        license: "LDC",
        citation: "Pradhan et al. (2012)",
        paper_url: "https://aclanthology.org/W12-4501/",
        year: 2012,
        format: "CoNLL",
        annotation_scheme: "CoNLLCoref",
        size_hint: "3,493 documents, ~1.6M tokens",
        notes: "De facto standard for within-document coreference evaluation",
        categories: [coref],
    },

    WikiCoref {
        name: "WikiCoref",
        description: "Wikipedia coreference corpus. 30 documents with full coreference annotation.",
        url: "",
        entity_types: ["PER", "ORG", "LOC"],
        language: "en",
        domain: "wikipedia",
        license: "CC-BY-SA-4.0",
        citation: "Ghaddar & Langlais (2016)",
        paper_url: "https://aclanthology.org/C16-1252/",
        year: 2016,
        format: "CoNLL",
        size_hint: "30 documents, ~60k tokens",
        notes: "Long documents averaging 2k tokens; challenging for span-based models. Prior download URL has an expired TLS cert; needs a fresh mirror.",
        access_status: Deprecated,
        categories: [coref],
    },

    ARRAU3 {
        name: "ARRAU 3.0",
        description: "Anaphora Resolution and Underspecification corpus version 3. Multi-genre with rich annotation.",
        url: "https://aclanthology.org/2024.codi-1.12/",
        entity_types: ["PER", "ORG", "LOC", "Event"],
        language: "en",
        domain: "mixed",
        license: "Research",
        citation: "Uryupina et al. (2024)",
        paper_url: "https://aclanthology.org/2024.codi-1.12/",
        year: 2024,
        format: "MMAX2",
        annotation_scheme: "ARRAU",
        size_hint: "~350k tokens across multiple genres",
        notes: "Rich annotation including bridging, discourse deixis, and ambiguity",
        categories: [coref],
    },

    AMIMeeting {
        name: "AMI Meeting",
        description: "Meeting transcripts with coreference and dialogue act annotation.",
        url: "https://groups.inf.ed.ac.uk/ami/download/",
        entity_types: ["PER", "ORG", "LOC"],
        language: "en",
        domain: "dialogue",
        license: "CC-BY-4.0",
        citation: "Carletta et al. (2005)",
        paper_url: "https://groups.inf.ed.ac.uk/ami/icsi/",
        year: 2005,
        format: "XML",
        size_hint: "100 hours of meetings",
        notes: "Multi-party dialogue; includes prosody and head gestures",
        categories: [coref, dialogue],
    },

    CLEFClinicalCoref {
        name: "CLEF Clinical Coreference",
        description: "Clinical coreference from ShARe/CLEF eHealth. Patient records with coref.",
        url: "",
        entity_types: ["Disorder", "Drug", "Procedure"],
        language: "en",
        domain: "clinical",
        license: "PhysioNet",
        citation: "Suominen et al. (2013)",
        paper_url: "https://clef2013.clef-initiative.eu/index.php?page=pages/proceedings.php",
        year: 2013,
        format: "Standoff",
        size_hint: "298 discharge summaries",
        notes: "Clinical concept coreference; disorder mentions across sentences. Was hosted on PhysioNet; URL appears gone/renamed; requires controlled access.",
        access_status: Registration,
        categories: [coref, biomedical],
    },

    RSTDT {
        name: "RST Discourse Treebank",
        description: "Penn Discourse Treebank with RST annotations. Discourse relations and structure.",
        url: "https://catalog.ldc.upenn.edu/LDC2002T07",
        entity_types: [],
        language: "en",
        domain: "news",
        license: "LDC",
        citation: "Carlson et al. (2001)",
        paper_url: "https://aclanthology.org/A00-1036/",
        year: 2001,
        format: "Custom",
        size_hint: "385 WSJ articles",
        notes: "RST discourse structure; useful for discourse-aware coreference",
        categories: [coref],
    },

    WinoBias {
        name: "WinoBias",
        description: "Coreference bias benchmark. Winograd-schema sentences testing occupational gender stereotypes.",
        url: "https://raw.githubusercontent.com/uclanlp/corefBias/master/WinoBias/wino/data/anti_stereotyped_type1.txt.dev",
        entity_types: ["PER"],
        language: "en",
        domain: "evaluation",
        license: "MIT",
        citation: "Zhao et al. (2018)",
        paper_url: "https://aclanthology.org/N18-2003/",
        year: 2018,
        format: "Custom",
        size_hint: "3,160 sentences",
        notes: "Type 1 (syntactic) and Type 2 (semantic) splits; tests BLS occupational stats",
        categories: [coref, bias_evaluation],
    },

    // =========================================================================
    // Indigenous Language Datasets
    // =========================================================================
    QxoRef {
        name: "qxoRef",
        description: "First coreference corpus for Conchucos Quechua. Historically significant as first indigenous coref resource.",
        url: "https://raw.githubusercontent.com/elizabethpankratz/qxoRef/f0eb5716573b3f428bfcfdda923b195d0e7967b8/qxoRef_AZ23.conll",
        entity_types: ["PER", "LOC", "ORG"],
        language: "qxo",
        domain: "narrative",
        license: "CC-BY-NC-SA-4.0",
        citation: "Rios (2021)",
        paper_url: "https://aclanthology.org/2021.americasnlp-1.1/",
        year: 2021,
        format: "CoNLL",
        annotation_scheme: "CoNLLCoref",
        size_hint: "12 docs, 332 mentions",
        notes: "First indigenous coreference corpus; pro-drop language; agglutinative morphology",
        categories: [coref, indigenous, low_resource],
    },
    AmericasNLI {
        name: "AmericasNLI",
        description: "NLI for 10 Indigenous American languages (Quechua, Guaraní, Nahuatl, etc.).",
        url: "https://raw.githubusercontent.com/nala-cub/AmericasNLI/be3c351b7e1ae69936c61bfde3e24f30757db9ac/test.tsv",
        entity_types: [],
        language: "mul",
        domain: "general",
        license: "CC-BY-4.0",
        citation: "Ebrahimi et al. (2022)",
        paper_url: "https://aclanthology.org/2022.acl-long.435/",
        year: 2022,
        format: "TSV",
        notes: "Tests zero-shot transfer from multilingual models; 10 indigenous languages",
        categories: [indigenous, multilingual, low_resource],
    },
    CherokeeNER {
        name: "Cherokee NER",
        description: "Cherokee-English parallel corpus for NER transfer. Uses Syllabary script.",
        url: "",
        entity_types: ["PER", "LOC", "ORG"],
        language: "chr",
        domain: "indigenous",
        license: "Research",
        citation: "Zhang et al. (2020)",
        paper_url: "https://aclanthology.org/2020.findings-emnlp.464/",
        year: 2020,
        format: "Custom",
        notes: "Syllabary script (85 characters); polysynthetic language; ~7k speakers. Prior URL is dead; needs a fresh mirror.",
        access_status: Deprecated,
        categories: [ner, indigenous, low_resource],
    },
    NahuatlNER {
        name: "Nahuatl NER",
        description: "Named entity recognition for Nahuatl (Aztec language). Colonial-era texts and modern usage.",
        url: "",
        entity_types: ["PER", "LOC", "ORG"],
        language: "nah",
        domain: "historical",
        license: "CC-BY-4.0",
        citation: "Gutierrez-Vasquez et al. (2023)",
        year: 2023,
        format: "CoNLL",
        notes: "Polysynthetic Uto-Aztecan language; ~1.7M speakers; includes colonial manuscripts. Prior URL is dead; needs a fresh mirror.",
        access_status: Deprecated,
        categories: [ner, indigenous, low_resource, historical],
    },
    MaoriNER {
        name: "Māori NER",
        description: "Named entity recognition for Te Reo Māori. New Zealand indigenous language corpus.",
        url: "",
        entity_types: ["PER", "LOC", "ORG"],
        language: "mi",
        domain: "indigenous",
        license: "Research",
        citation: "Te Hiku Media (2022)",
        year: 2022,
        format: "JSONL",
        notes: "Polynesian language; ~50k fluent speakers; limited training data available",
        access_status: ContactAuthors,
        categories: [ner, indigenous, low_resource],
    },
    WelshNER {
        name: "Welsh NER",
        description: "Named entity recognition for Welsh (Cymraeg). Celtic language NER corpus.",
        url: "",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "cy",
        domain: "news",
        license: "CC-BY-4.0",
        citation: "Roberts et al. (2021)",
        year: 2021,
        format: "CoNLL",
        notes: "Celtic language; ~900k speakers; supports Welsh-specific entity types. Prior URL is dead; needs a fresh mirror.",
        access_status: Deprecated,
        categories: [ner, indigenous, low_resource],
    },
    BasqueNER {
        name: "Basque NER",
        description: "Named entity recognition for Basque (Euskara). Language isolate NER corpus.",
        url: "",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "eu",
        domain: "news",
        license: "CC-BY-4.0",
        citation: "Alegria et al. (2019)",
        year: 2019,
        format: "CoNLL",
        size_hint: "~80k tokens",
        notes: "Language isolate; agglutinative morphology; ~750k speakers; ergative-absolutive alignment. Prior URL is dead; needs a fresh mirror.",
        access_status: Deprecated,
        categories: [ner, indigenous, low_resource],
    },

    // =========================================================================
    // Historical NER Datasets
    // =========================================================================
    HIPE2022 {
        name: "HIPE-2022",
        description: "Multilingual Historical NER. 6 datasets across 11 languages including Latin.",
        url: "https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/147f5bc3c7fb7e5c6b024a9ffd6503cd019fb9ea/data/v2.1/hipe2020/de/HIPE-2022-v2.1-hipe2020-test-de.tsv",
        entity_types: ["PER", "LOC", "ORG", "PROD"],
        language: "mul",
        domain: "historical",
        license: "CC-BY-NC-4.0",
        citation: "Ehrmann et al. (2022)",
        paper_url: "https://ceur-ws.org/Vol-3180/paper-83.pdf",
        year: 2022,
        format: "TSV",
        annotation_scheme: "IOB2",
        notes: "CLEF-HIPE shared task; includes Latin and Classical commentary; OCR noise",
        splits: ["test"],
        tasks: ["ner"],
        categories: [ner, historical, multilingual],
    },
    HistNERo {
        name: "HistNERo",
        description: "Romanian historical newspaper NER. First Romanian historical NER corpus from four regions.",
        url: "https://github.com/avramandrei/histnero",
        entity_types: ["PER", "LOC", "ORG", "DATE", "MISC"],
        language: "ro",
        domain: "historical",
        license: "CC-BY-4.0",
        citation: "HistNERo Team (2024)",
        paper_url: "https://arxiv.org/abs/2405.00155",
        year: 2024,
        format: "CoNLL",
        size_hint: "~323k tokens, 19th-20th century newspapers",
        notes: "Four historical Romanian regions (Bessarabia, Moldavia, Transylvania, Wallachia); diachronic benchmark",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner, historical, low_resource],
    },
    QuaeroOldPress {
        name: "Quaero Old Press",
        description: "French historical newspaper NER from 1890. OCR-corrected with manual NE annotations.",
        url: "",
        entity_types: ["PER", "LOC", "ORG", "TIME", "PROD"],
        language: "fr",
        domain: "historical",
        license: "Research",
        citation: "Galibert et al. (2012)",
        year: 2012,
        format: "XML",
        size_hint: "295 pages, 1890 newspapers",
        notes: "French historical NER benchmark; manual OCR corrections; reasonably clean historical text",
        splits: ["test"],
        tasks: ["ner"],
        access_status: ContactAuthors,
        categories: [ner, historical],
    },
    HistoricalChineseNER {
        name: "Historical Chinese NER",
        description: "Multi-task historical Chinese corpus. NER + entity linking + coreference + relations.",
        url: "",
        entity_types: ["PER", "LOC", "ORG", "TIME", "OFFICIAL"],
        language: "zh",
        domain: "historical",
        license: "Research",
        citation: "LREC-COLING (2024)",
        paper_url: "https://aclanthology.org/2024.lrec-main.35.pdf",
        year: 2024,
        format: "JSONL",
        size_hint: "Historical Chinese newspapers + documents",
        notes: "LREC-COLING 2024; multi-task historical IE benchmark; cross-genre historical Chinese",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "el", "coref", "re"],
        access_status: ContactAuthors,
        categories: [ner, entity_linking, coref, historical, multilingual],
    },
    CHisIEC {
        name: "CHisIEC",
        description: "Chinese Historical Information Extraction Corpus. Ancient Chinese NER + RE with 12 relation types.",
        url: "https://raw.githubusercontent.com/tangxuemei1995/CHisIEC/main/data/re/coling_test.json",
        entity_types: ["PER", "LOC", "OFI", "BOOK"],
        language: "lzh",
        domain: "historical",
        license: "Research",
        citation: "Tang et al. (2024)",
        paper_url: "https://aclanthology.org/2024.lrec-main.283/",
        year: 2024,
        format: "JSON",
        size_hint: "3,891 paragraphs, 13,520 entities, 8,228 relations",
        notes: "Ancient Chinese dynastic histories (24史); 12 domain-specific relations for historical socio-political structures; pre-modern Chinese (文言文)",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "re"],
        categories: [ner, relation_extraction, historical, ancient],
    },

    // =========================================================================
    // Relation Extraction Datasets
    // =========================================================================
    DocRED {
        name: "DocRED",
        description: "Document-level relation extraction. 96 relation types from Wikipedia.",
        url: "https://raw.githubusercontent.com/mainlp/CrossRE/main/crossre_data/ai-test.json",
        entity_types: ["PER", "LOC", "ORG", "TIME", "NUM"],
        language: "en",
        domain: "wikipedia",
        license: "MIT",
        citation: "Yao et al. (2019)",
        paper_url: "https://aclanthology.org/P19-1074/",
        year: 2019,
        format: "JSONL",
        size_hint: "5,053 docs, 132k entities, 56k relations",
        splits: ["train", "dev", "test"],
        tasks: ["re"],
        expected_docs: 5053,
        hf_id: "docred",
        access_status: HuggingFace,
        categories: [relation_extraction],
    },
    ReTACRED {
        name: "Re-TACRED",
        description: "Large-scale relation extraction. 41 relation types + no_relation. Cleaned TACRED.",
        // NOTE: Re-TACRED is distributed as a delta/cleaned relabeling that depends on the
        // original TACRED distribution (LDC). We intentionally do not provide a direct URL
        // here to avoid accidentally pointing at unrelated files.
        url: "",
        entity_types: ["PER", "ORG", "LOC", "DATE", "NUM"],
        language: "en",
        domain: "news",
        license: "LDC",
        citation: "Stoica et al. (2021)",
        paper_url: "https://aclanthology.org/2021.acl-long.359/",
        year: 2021,
        format: "JSONL",
        size_hint: "~106k relations",
        notes: "Cleaned version of TACRED with relabeling; requires original TACRED",
        access_status: DependsOnOther,
        categories: [relation_extraction],
    },

    // =========================================================================
    // Nested/Discontinuous NER Datasets
    // =========================================================================
    ACE2004 {
        name: "ACE 2004",
        description: "Nested entity recognition benchmark. Influential early corpus for nested NER research.",
        url: "",  // Requires LDC license
        entity_types: ["PER", "ORG", "GPE", "LOC", "FAC", "WEA", "VEH"],
        language: "en",
        domain: "news",
        license: "LDC",
        citation: "Doddington et al. (2004)",
        paper_url: "https://aclanthology.org/L04-1011/",
        year: 2004,
        format: "XML",
        annotation_scheme: "Standoff",
        notes: "Requires LDC license; includes entity relations; includes some nested entities",
        access_status: Registration,
        categories: [ner, nested_ner],
    },
    CADEC {
        name: "CADEC",
        description: "Clinical Adverse Drug Events. Benchmark for discontinuous NER from AskaPatient.",
        url: "https://huggingface.co/datasets/KevinSpaghetti/cadec",
        entity_types: ["ADR", "Drug", "Disease", "Symptom"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "Karimi et al. (2015)",
        paper_url: "https://pubmed.ncbi.nlm.nih.gov/25817970/",
        year: 2015,
        format: "BRAT",
        annotation_scheme: "Standoff",
        size_hint: "~1,250 posts",
        example: "'severe [pain]...in my [legs]' -> ADR spans [0:10, 20:24] (discontinuous)",
        notes: "Discontinuous spans common; requires special handling; patient-written text",
        // Note: HF export endpoints currently fail (rows=422) and repo stores parquet shards.
        // Keep as HF reference but treat as non-automatable until we add parquet support or a stable jsonl mirror.
        splits: ["train", "dev", "test"],
        tasks: ["ner", "discontinuous_ner"],
        hf_id: "KevinSpaghetti/cadec",
        access_status: HuggingFace,
        categories: [ner, biomedical, discontinuous_ner],
    },

    // =========================================================================
    // Bias Evaluation Datasets
    // =========================================================================
    WinoQueer {
        name: "WinoQueer",
        description: "Anti-LGBTQ+ bias benchmark. Community-in-the-loop design for queer representation.",
        url: "https://raw.githubusercontent.com/katyfelkner/winoqueer/main/data/winoqueer_final.csv",
        entity_types: ["PER"],
        language: "en",
        domain: "evaluation",
        license: "MIT",
        citation: "Felkner et al. (2023)",
        paper_url: "https://aclanthology.org/2023.acl-long.507/",
        year: 2023,
        format: "CSV",
        size_hint: "45,540 sentence pairs, ~4.8MB",
        notes: "Community-designed; tests queer stereotypes in LLMs; Winograd-schema style",
        splits: ["all"],
        tasks: ["bias_evaluation"],
        categories: [bias_evaluation],
    },
    BBQ {
        name: "BBQ",
        description: "Bias Benchmark for QA. Tests 9 social bias categories including sexual orientation.",
        url: "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/Gender_identity.jsonl",
        entity_types: [],
        language: "en",
        domain: "evaluation",
        license: "CC-BY-4.0",
        citation: "Parrish et al. (2022)",
        paper_url: "https://aclanthology.org/2022.findings-acl.165/",
        year: 2022,
        format: "JSONL",
        size_hint: "~58k QA pairs across 11 categories",
        notes: "Hand-built ambiguous contexts; age, disability, nationality, religion, etc.",
        splits: ["all"],
        tasks: ["bias_evaluation", "qa"],
        categories: [bias_evaluation],
    },
    GICoref {
        name: "GICoref",
        description: "Gender-inclusive coreference. Written by/about trans and non-binary individuals.",
        url: "https://raw.githubusercontent.com/TristaCao/into_inclusivecoref/master/GICoref/coref.combo.conll",
        entity_types: ["PER"],
        language: "en",
        domain: "evaluation",
        license: "CC-BY-4.0",
        citation: "Cao & Daume III (2020)",
        paper_url: "https://aclanthology.org/2020.acl-main.418/",
        year: 2020,
        format: "CoNLL",
        annotation_scheme: "CoNLLCoref",
        size_hint: "95 docs, 470KB",
        notes: "Includes neopronouns (ze/hir, xe/xem); singular they; first gender-inclusive coref corpus",
        splits: ["all"],
        tasks: ["coref"],
        categories: [coref, bias_evaluation],
    },

    // =========================================================================
    // Multilingual Coreference
    // =========================================================================
    CorefUD {
        name: "CorefUD",
        description: "Multilingual coreference (17 languages, 22 datasets). CRAC shared task standard.",
        url: "http://hdl.handle.net/11234/1-5987",
        entity_types: ["PER", "LOC", "ORG"],
        language: "mul",
        domain: "general",
        license: "CC-BY-NC-SA-4.0",
        citation: "Nedoluzhko et al. (2022)",
        paper_url: "https://aclanthology.org/2022.lrec-1.581/",
        year: 2022,
        format: "CoNLLU",
        annotation_scheme: "CoNLLCoref",
        notes: "CRAC shared task standard; includes zero anaphora; harmonized across treebanks",
        categories: [coref, multilingual],
    },
    TransMuCoRes {
        name: "TransMuCoRes",
        description: "Coreference in 31 South Asian languages. Silver annotations via NLLB-200 translation.",
        url: "",  // HuggingFace gated
        entity_types: ["PER", "LOC", "ORG"],
        language: "mul",
        domain: "general",
        license: "Research",
        citation: "Verma et al. (2024)",
        paper_url: "https://arxiv.org/abs/2402.13571",
        year: 2024,
        format: "JSONL",
        notes: "Silver annotations via translation; fine-tuned mBERT models available",
        access_status: ContactAuthors,
        categories: [coref, multilingual],
    },
    MGAP {
        name: "mGAP",
        description: "Multilingual Gender-Ambiguous Pronouns. 27 South Asian languages.",
        url: "",  // Research access required
        entity_types: ["PER"],
        language: "mul",
        domain: "evaluation",
        license: "Research",
        citation: "Verma et al. (2025)",
        paper_url: "https://aclanthology.org/2025.chipsal-1.10/",
        year: 2025,
        format: "TSV",
        size_hint: "8,908 pronoun-name pairs",
        notes: "Cross-attention improves results; extension of GAP to South Asian languages",
        access_status: ContactAuthors,
        categories: [coref, multilingual, bias_evaluation],
    },
    CrowSPairs {
        name: "CrowS-Pairs",
        description: "Crowdsourced stereotype pairs benchmark. 9 bias categories for language models.",
        url: "https://github.com/nyu-mll/crows-pairs",
        entity_types: [],
        language: "en",
        domain: "evaluation",
        license: "CC-BY-SA-4.0",
        citation: "Nangia et al. (2020)",
        paper_url: "https://aclanthology.org/2020.emnlp-main.154/",
        year: 2020,
        format: "CSV",
        size_hint: "~1.5k sentence pairs",
        notes: "Tests stereotypical associations; gender, race, religion, age, nationality, etc.",
        splits: ["all"],
        tasks: ["bias_evaluation"],
        categories: [bias_evaluation],
    },
    StereoSet {
        name: "StereoSet",
        description: "Measuring stereotypical bias in language models. 4 target domains.",
        url: "https://github.com/moinnadeem/StereoSet",
        entity_types: [],
        language: "en",
        domain: "evaluation",
        license: "MIT",
        citation: "Nadeem et al. (2020)",
        paper_url: "https://aclanthology.org/2021.acl-long.416/",
        year: 2020,
        format: "JSONL",
        size_hint: "~17k instances",
        notes: "Intrasentence and intersentence evaluation; gender, profession, race, religion",
        splits: ["all"],
        tasks: ["bias_evaluation"],
        categories: [bias_evaluation],
    },
    RealToxicityPrompts {
        name: "RealToxicityPrompts",
        description: "100k prompts for measuring toxicity generation in language models.",
        url: "https://huggingface.co/datasets/allenai/real-toxicity-prompts",
        entity_types: [],
        language: "en",
        domain: "evaluation",
        license: "Apache-2.0",
        citation: "Gehman et al. (2020)",
        paper_url: "https://aclanthology.org/2020.findings-emnlp.301/",
        year: 2020,
        format: "JSONL",
        size_hint: "~100k prompts",
        notes: "Tests toxicity generation; perspectives API scores; diverse prompt styles",
        splits: ["all"],
        tasks: ["bias_evaluation"],
        categories: [bias_evaluation, adversarial],
    },
    BoldBias {
        name: "BOLD",
        description: "Bias in Open-ended Language Generation Dataset. Wikipedia-based prompts.",
        url: "https://github.com/amazon-science/bold",
        entity_types: [],
        language: "en",
        domain: "evaluation",
        license: "CC-BY-4.0",
        citation: "Dhamala et al. (2021)",
        paper_url: "https://aclanthology.org/2021.findings-acl.311/",
        year: 2021,
        format: "JSONL",
        size_hint: "~23k prompts",
        notes: "Tests generation bias; profession, gender, race, religion, political ideology",
        splits: ["all"],
        tasks: ["bias_evaluation"],
        categories: [bias_evaluation],
    },

    // =========================================================================
    // Literary Coreference
    // =========================================================================
    DROC {
        name: "DROC",
        description: "German novel coreference. 90 German novels from DTA (Deutsches Textarchiv).",
        url: "",
        entity_types: ["PER"],
        language: "de",
        domain: "literature",
        license: "CC-BY-4.0",
        citation: "Krug et al. (2018)",
        paper_url: "https://aclanthology.org/L18-1045/",
        year: 2018,
        format: "JSONL",
        notes: "First public German literary coreference dataset. Prior URL is dead; needs a fresh mirror.",
        splits: ["all"],
        tasks: ["coref"],
        access_status: Deprecated,
        categories: [coref, literary],
    },
    FantasyCoref {
        name: "FantasyCoref",
        description: "Fantasy fiction coreference. Handles entity transformations. 211 Grimms' stories.",
        url: "https://github.com/emorynlp/FantasyCoref/archive/refs/heads/master.zip",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "literature",
        license: "Research",
        citation: "Han et al. (2021)",
        paper_url: "https://aclanthology.org/2021.emnlp-main.672/",
        year: 2021,
        format: "JSONL",
        notes: "Shape-shifting, possession, disguise - unique challenges. Prior GitHub URL is currently 404; treat as non-automatable until mirrored.",
        tasks: ["coref"],
        access_status: ContactAuthors,
        categories: [coref, literary],
    },

    // =========================================================================
    // Book-Scale / Long-Document Coreference
    // =========================================================================

    BookCoref {
        name: "BOOKCOREF",
        description: "Book-scale coreference. First benchmark with 200k+ tokens/doc average. Character coreference on 53 Project Gutenberg novels.",
        url: "https://huggingface.co/datasets/sapienzanlp/bookcoref",
        entity_types: ["PER"],
        language: "en",
        domain: "literature",
        license: "CC-BY-NC-SA-4.0",
        citation: "Martinelli et al. (2025)",
        paper_url: "https://aclanthology.org/2025.acl-long.1197/",
        year: 2025,
        format: "JSONL",
        annotation_scheme: "CoNLLCoref",
        size_hint: "53 books, ~10.8M tokens silver, 229k tokens gold",
        example: "doc_key: pride_and_prejudice_1342\nsentences: [[CHAPTER, I.], [It, is, a, truth, ...]]\nclusters: [[[79,80], [81,82], ...], ...]\ncharacters: [{name: Mr Bennet, cluster: [[79,80]]}]",
        notes: "Gold test set: 3 books (Animal Farm, Siddhartha, Pride & Prejudice). Silver train: 45 books. Unprecedented 73k avg mention distance. Requires HF datasets Python lib to download; export to JSONL for anno loader.",
        splits: ["train", "validation", "test"],
        tasks: ["coref"],
        expected_docs: 53,
        hf_id: "sapienzanlp/bookcoref",
        access_status: DependsOnOther,
        categories: [coref, literary, long_document],
    },
    BookCorefSplit {
        name: "BOOKCOREF (Split)",
        description: "BOOKCOREF split into 1500-token windows for comparison with standard benchmarks.",
        url: "https://huggingface.co/datasets/sapienzanlp/bookcoref",
        entity_types: ["PER"],
        language: "en",
        domain: "literature",
        license: "CC-BY-NC-SA-4.0",
        citation: "Martinelli et al. (2025)",
        paper_url: "https://aclanthology.org/2025.acl-long.1197/",
        year: 2025,
        format: "JSONL",
        annotation_scheme: "CoNLLCoref",
        size_hint: "7544 train, 398 val, 152 test windows",
        notes: "Same data as BOOKCOREF but windowed. Enables fair comparison: Maverickxl gets 82.2 CoNLL F1 on split vs 61.0 on full books. Requires HF datasets Python lib to download.",
        splits: ["train", "validation", "test"],
        tasks: ["coref"],
        hf_id: "sapienzanlp/bookcoref",
        hf_config: "split",
        access_status: DependsOnOther,
        categories: [coref, literary],
    },
    LongtoNotes {
        name: "LongtoNotes",
        description: "OntoNotes with merged coreference chains. Manually merges split OntoNotes documents back into full documents.",
        url: "https://docs.google.com/forms/d/e/1FAIpQLScoWkBOgJ1HH_phtvTJ4_hGvQw6f0W6K7kw74sUKCDTG8P2iA/viewform",
        entity_types: ["PER", "ORG", "GPE", "NORP"],
        language: "en",
        domain: "mixed",
        license: "CC-BY-4.0",
        citation: "Shridhar et al. (2023)",
        paper_url: "https://aclanthology.org/2023.findings-eacl.105/",
        year: 2023,
        format: "CoNLL",
        annotation_scheme: "CoNLLCoref",
        size_hint: "2,415 documents, up to 8x longer than OntoNotes",
        notes: "Requires OntoNotes access. Documents up to 8x OntoNotes length, 2x LitBank. Multi-genre (WSJ, broadcast, web).",
        splits: ["train", "dev", "test"],
        tasks: ["coref"],
        categories: [coref, long_document],
    },
    MovieCoref {
        name: "MovieCoref",
        description: "Screenplay coreference. Character coreference in movie screenplays with unique structural challenges.",
        url: "https://aclanthology.org/attachments/2021.findings-acl.176.OptionalSupplementaryMaterial.gz",
        entity_types: ["PER"],
        language: "en",
        domain: "literature",
        license: "Research",
        citation: "Baruah et al. (2021)",
        paper_url: "https://aclanthology.org/2021.findings-acl.176/",
        year: 2021,
        format: "CoNLL",
        annotation_scheme: "CoNLLCoref",
        size_hint: "9 screenplays (~22k tokens/doc avg), 6 full + 3 excerpts",
        notes: "Screenplay structure (scene headings, character names, parentheticals) differs significantly from prose. Focus on character coreference only.",
        splits: ["all"],
        tasks: ["coref"],
        categories: [coref, literary, long_document],
    },
    // =========================================================================
    // Dialogue / Conversational Coreference
    // =========================================================================
    TwiConv {
        name: "TwiConv",
        description: "Twitter conversational coreference.",
        url: "https://raw.githubusercontent.com/berfingit/TwiConv/main/conll_skeleton/001_940791133357199360.branch7._with_boundaries_gold_conll",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "social_media",
        license: "Research",
        citation: "Aktaş et al. (2020)",
        paper_url: "https://aclanthology.org/2020.lrec-1.835/",
        format: "CoNLL",
        notes: "Turn-taking dynamics; speaker grounding",
        categories: [coref, dialogue, social_media],
    },
    MuDoCo {
        name: "MuDoCo",
        description: "Multi-domain document-level coreference. Dialog-based.",
        url: "https://raw.githubusercontent.com/facebookresearch/mudoco/main/mudoco_calling.json",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "dialogue",
        license: "MIT",
        citation: "Raghunathan et al. (2020)",
        paper_url: "https://arxiv.org/abs/2005.00816",
        format: "JSON",
        notes: "Multi-domain dialog coreference with speaker tracking",
        categories: [coref, dialogue],
    },
    DialogRE {
        name: "DialogRE",
        description: "Dialogue-based relation extraction. Multi-turn conversations requiring entity tracking across turns.",
        url: "https://github.com/nlpdata/dialogre",
        entity_types: ["PER", "ORG", "LOC"],
        language: "en",
        domain: "dialogue",
        license: "CC-BY-NC-SA-4.0",
        citation: "Yu et al. (2020)",
        paper_url: "https://aclanthology.org/2020.acl-main.444/",
        year: 2020,
        format: "JSONL",
        size_hint: "~1.8k dialogues, 36 relation types",
        notes: "Based on Friends TV show transcripts; requires tracking entities across dialogue turns",
        splits: ["train", "dev", "test"],
        tasks: ["re", "coref"],
        categories: [relation_extraction, dialogue],
    },
    MultiWOZNER {
        name: "MultiWOZ NER",
        description: "Multi-domain task-oriented dialogue with slot/entity annotations. Multi-turn conversations.",
        url: "https://github.com/budzianowski/multiwoz",
        entity_types: ["RESTAURANT", "HOTEL", "ATTRACTION", "TAXI", "TRAIN", "HOSPITAL", "POLICE"],
        language: "en",
        domain: "dialogue",
        license: "Apache-2.0",
        citation: "Budzianowski et al. (2018)",
        paper_url: "https://aclanthology.org/D18-1547/",
        year: 2018,
        format: "JSONL",
        size_hint: "~10k dialogues, 7 domains",
        notes: "Standard benchmark for dialogue state tracking; slot values correspond to entities",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "slot_filling"],
        categories: [ner, dialogue],
    },
    CoQAEntities {
        name: "CoQA",
        description: "Conversational QA across 7 domains: children's stories, literature, mid/high school exams, news, Wikipedia, science, Reddit.",
        url: "https://stanfordnlp.github.io/coqa/",
        entity_types: ["ANSWER_SPAN"],
        language: "en",
        domain: "mixed",
        license: "Research",
        citation: "Reddy et al. (2019)",
        paper_url: "https://aclanthology.org/Q19-1016/",
        year: 2019,
        format: "JSONL",
        size_hint: "~8k conversations, ~127k QA turns",
        notes: "Multi-turn QA; implicit entity tracking across conversation history; 7 diverse domains",
        splits: ["train", "dev", "test"],
        tasks: ["qa", "coref"],
        categories: [coref, dialogue],
    },

    // =========================================================================
    // Cross-Document Event Coreference
    // =========================================================================
    GVC {
        name: "Gun Violence Corpus",
        description: "Cross-document event coreference for gun violence. Tests domain transfer from ECB+.",
        url: "https://github.com/cltl/GunViolenceCorpus",
        entity_types: ["EVENT", "PARTICIPANT", "WEAPON", "LOCATION", "TIME"],
        language: "en",
        domain: "news",
        license: "Research",
        citation: "Vossen et al. (2018)",
        paper_url: "https://aclanthology.org/L18-1182/",
        year: 2018,
        format: "Custom",
        size_hint: "~500 docs, 510 mentions",
        notes: "Domain-specific CDEC; requires participant/temporal reasoning unlike lemma-driven ECB+",
        splits: ["train", "dev", "test"],
        tasks: ["coref", "event_coref", "cdcr"],
        categories: [coref, event_coref],
    },
    FCC {
        name: "Football Coreference Corpus",
        description: "Cross-document event coreference for football matches. Requires temporal reasoning.",
        url: "",  // Research access required
        entity_types: ["EVENT", "PARTICIPANT", "LOC", "TIME"],
        language: "en",
        domain: "sports",
        license: "Research",
        citation: "Bugert et al. (2021)",
        paper_url: "https://direct.mit.edu/coli/article/47/3/575/102774/",
        notes: "Requires temporal reasoning about match dates",
        access_status: ContactAuthors,
        categories: [coref, event_coref],
    },
    ECBPlusMeta {
        name: "ECB+META",
        description: "ECB+ with metaphoric paraphrases. ChatGPT-transformed sentences.",
        url: "",  // Research access required
        entity_types: ["EVENT", "TIME", "LOC", "PARTICIPANT"],
        language: "en",
        domain: "news",
        license: "Research",
        citation: "Pouran Ben Veyseh et al. (2024)",
        paper_url: "https://arxiv.org/abs/2407.11988",
        notes: "Adversarial; existing systems struggle badly",
        access_status: ContactAuthors,
        categories: [coref, event_coref, adversarial],
    },

    // =========================================================================
    // Abstract Anaphora / Discourse
    // =========================================================================
    ARRAU {
        name: "ARRAU 3.0 (v2)",
        description: "Multi-genre anaphoric annotation: identity, bridging, discourse deixis, split antecedents.",
        url: "",  // LDC + free subsets
        entity_types: ["PER", "LOC", "ORG", "EVENT"],
        language: "en",
        domain: "mixed",
        license: "LDC + Research",
        citation: "Poesio et al. (2024)",
        paper_url: "https://aclanthology.org/2024.codi-1.12/",
        year: 2024,
        format: "MMAX2",
        notes: "Most comprehensive anaphora resource; RST/TRAINS/Pear/GENIA subsets; LDC2023T05",
        splits: ["train", "dev", "test"],
        tasks: ["coref", "bridging", "discourse_deixis"],
        access_status: Registration,
        categories: [coref, abstract_anaphora],
    },
    ISNotes {
        name: "ISNotes",
        description: "Unrestricted bridging anaphora on OntoNotes. ~660 bridging pairs.",
        url: "https://github.com/nlpAThits/ISNotes1.0/archive/refs/heads/master.zip",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "news",
        license: "Research",
        citation: "Hou et al. (2018)",
        paper_url: "https://direct.mit.edu/coli/article/44/2/237/1596/",
        format: "MMAX2",
        notes: "Part-whole set-member bridging; requires OntoNotes for full text",
        tasks: ["coref"],
        access_status: DependsOnOther,
        categories: [coref, abstract_anaphora],
    },
    ShellNouns {
        name: "Shell Nouns (ASN)",
        description: "Anaphoric shell noun resolution. 670 English shell nouns from Schmid taxonomy.",
        url: "",  // Research access
        entity_types: [],
        language: "en",
        domain: "academic",
        license: "Research",
        citation: "Kolhatkar & Hirst (2012)",
        paper_url: "https://aclanthology.org/D12-1036/",
        notes: "Factual, linguistic, mental, modal, eventive categories",
        access_status: ContactAuthors,
        categories: [abstract_anaphora],
    },
    PDTBv3 {
        name: "PDTB 3.0",
        description: "Penn Discourse TreeBank v3. 43 discourse relation types.",
        url: "",  // LDC required
        entity_types: [],
        language: "en",
        domain: "news",
        license: "LDC",
        citation: "Prasad et al. (2019)",
        paper_url: "https://catalog.ldc.upenn.edu/LDC2019T05",
        notes: "Shallow discourse parsing; connective-argument pairs",
        access_status: Registration,
        categories: [abstract_anaphora],
    },
    CODICRACBridging {
        name: "CODI-CRAC Bridging",
        description: "Universal Anaphora bridging annotations. One of the largest bridging datasets.",
        url: "https://github.com/UniversalAnaphora/UA-CODI-CRAC",
        entity_types: ["BRIDGING_REF"],
        language: "en",
        domain: "dialogue",
        license: "CC-BY-4.0",
        citation: "CODI-CRAC (2022)",
        paper_url: "https://aclanthology.org/2024.lrec-main.1484.pdf",
        year: 2022,
        format: "CoNLLUA",
        size_hint: "AMI, LIGHT, PERSUASION subsets",
        notes: "Dialogue-heavy; extensive bridging annotations; Universal Anaphora format",
        splits: ["train", "dev", "test"],
        tasks: ["coref", "bridging"],
        categories: [coref, abstract_anaphora, dialogue],
    },
    AnaphoraAccessibility {
        name: "Anaphora Accessibility",
        description: "Discourse anaphora accessibility evaluation. Tests non-NP antecedents.",
        url: "",
        entity_types: ["DISCOURSE_DEIXIS", "EVENT_ANAPHORA", "CLAUSAL_ANTECEDENT"],
        language: "en",
        domain: "evaluation",
        license: "Research",
        citation: "Accessibility Authors (2025)",
        paper_url: "https://arxiv.org/html/2502.14119v1",
        year: 2025,
        format: "JSONL",
        size_hint: "Controlled evaluation set",
        notes: "2025 evaluation dataset; focuses on discourse-level anaphora understanding; non-nominal antecedents",
        splits: ["test"],
        tasks: ["coref", "discourse_deixis"],
        access_status: NotYetReleased,
        categories: [coref, abstract_anaphora],
    },
    HumanVoiceAgentInteraction {
        name: "Human-Voice Agent Interaction",
        description: "Naturalistic French dialogue from human-voice agent interactions. Response tokens, aside sequences, discourse deixis.",
        url: "",  // Local dataset at testdata/human_voice_agent/
        entity_types: ["RESPONSE_TOKEN", "DISCOURSE_DEIXIS", "PROPOSITIONAL_ANAPHORA"],
        language: "fr",
        domain: "dialogue",
        license: "Research",
        citation: "Rudaz, Broth & Mlynář (2025)",
        paper_url: "https://dl.acm.org/journal/tochi",
        year: 2025,
        format: "JSONL",
        size_hint: "70 turns, 10 discourse deixis examples, 11 response tokens",
        notes: "French dialogue from Pepper robot (2022) and ChatGPT voice mode (2025). Documents 'managed omnirelevance of speech' - how VAD-based agents misinterpret response tokens and asides. Local dataset at testdata/human_voice_agent/",
        splits: ["all"],
        tasks: ["coref", "abstract_anaphora"],
        access_status: Local,
        categories: [abstract_anaphora, dialogue],
    },
    DiscoBench {
        name: "Disco-Bench",
        description: "Discourse-aware evaluation benchmark for language modeling. 9 document-level testsets covering cohesion, coherence across Chinese/English literature.",
        url: "",
        entity_types: [],
        language: "zh-en",  // Chinese-English bilingual
        domain: "literature",
        license: "CC-BY-4.0",
        citation: "Wang et al. (2023)",
        paper_url: "https://arxiv.org/abs/2307.08074",
        year: 2023,
        format: "Custom",
        size_hint: "9 document-level testsets + diagnostic suite",
        notes: "Tests intra-sentence discourse properties: cohesion, coherence, entity tracking. Evaluates LLMs on discourse phenomena that cross sentences.",
        splits: ["test"],
        tasks: ["discourse_coherence"],
        access_status: NotYetReleased,
        categories: [abstract_anaphora, multilingual],
    },
    DiscoTrack {
        name: "DiscoTrack",
        description: "Multilingual LLM benchmark for discourse tracking. 12 languages, 4 levels: salience, entity tracking, discourse relations, bridging.",
        url: "",
        entity_types: ["SALIENT_ENTITY", "TRACKED_ENTITY", "BRIDGING_REF"],
        language: "mul",
        domain: "general",
        license: "CC-BY-4.0",
        citation: "Bu, Levine & Zeldes (2025)",
        paper_url: "https://arxiv.org/abs/2510.17013",
        year: 2025,
        format: "Custom",
        size_hint: "12 languages, 4 task levels",
        notes: "Tests implicit information and pragmatic inference across documents. Four levels: salience recognition, entity tracking, discourse relations, bridging inference. State-of-the-art models still struggle.",
        splits: ["test"],
        tasks: ["coref", "bridging", "discourse_deixis"],
        access_status: NotYetReleased,
        categories: [coref, abstract_anaphora, multilingual],
    },
    LIEDER {
        name: "LIEDER",
        description: "Linguistically-Informed Evaluation for Discourse Entity Recognition. Tests existence, uniqueness, plurality, novelty.",
        url: "",
        entity_types: ["DISCOURSE_ENTITY"],
        language: "en",
        domain: "evaluation",
        license: "Research",
        citation: "Zhu & Frank (2024)",
        paper_url: "https://arxiv.org/abs/2403.06301",
        year: 2024,
        format: "Custom",
        size_hint: "Controlled evaluation set",
        notes: "Tests LLM knowledge of semantic properties governing discourse entity introduction/reference: existence, uniqueness, plurality, novelty. Models show sensitivity to all except novelty.",
        splits: ["test"],
        tasks: ["discourse_deixis"],
        access_status: ContactAuthors,
        categories: [abstract_anaphora],
    },
    GCDC {
        name: "GCDC",
        description: "Grammarly Corpus of Discourse Coherence. Real-world texts with coherence ratings across 4 domains.",
        url: "",
        entity_types: [],
        language: "en",
        domain: "mixed",
        license: "Research",
        citation: "Lai & Tetreault (2018)",
        paper_url: "https://arxiv.org/abs/1805.04993",
        year: 2018,
        format: "TSV",
        size_hint: "~2000 texts, 4 domains",
        notes: "Four domains: Clinton emails, Enron, Yahoo answers, Yelp. First large-scale discourse coherence evaluation.",
        splits: ["train", "dev", "test"],
        tasks: ["discourse_coherence"],
        access_status: DependsOnOther,
        depends_on: "Yahoo L6 corpus (request from Yahoo, then email tetreaul@gmail.com)",
        categories: [abstract_anaphora],
    },
    DISAPERE {
        name: "DISAPERE",
        description: "Discourse Structure in Peer Review Discussions. 20k sentences in 506 review-rebuttal pairs with argumentation annotation.",
        url: "https://github.com/nnkennard/DISAPERE",
        entity_types: ["REVIEW_ARG", "REBUTTAL_ARG", "STANCE"],
        language: "en",
        domain: "scientific",
        license: "CC-BY-4.0",
        citation: "Kennard et al. (2022)",
        paper_url: "https://arxiv.org/abs/2110.08520",
        year: 2022,
        format: "JSONL",
        size_hint: "20k sentences, 506 review-rebuttal pairs",
        notes: "Discourse relations between reviews and rebuttals. Fine-grained rebuttal annotation: context in review, stance toward arguments. Expert-annotated. DOWNLOADED to anno cache.",
        splits: ["train", "dev", "test"],
        tasks: ["discourse_relations"],
        access_status: Public,
        categories: [abstract_anaphora, dialogue],
    },
    // PragmEval: 11 discourse-focused datasets for pragmatics evaluation
    PragmEval {
        name: "PragmEval",
        description: "Pragmatics-centered evaluation framework: 11 datasets covering discourse relations, speech acts, stance, sarcasm, verifiability.",
        url: "https://github.com/sileod/pragmeval",
        entity_types: [],
        language: "en",
        domain: "dialogue",
        license: "Research",
        citation: "Sileo et al. (2022)",
        paper_url: "https://aclanthology.org/2022.lrec-1.255",
        year: 2022,
        format: "TSV",
        size_hint: "20 subsets, ~130k examples total",
        notes: "Compilation of PDTB, STAC, GUM (discourse relations), Emergent (stance), SarcasmV2, SwitchBoard/MRDA (speech acts), Verifiability, Persuasion, Squinky, EmoBank. Not directly loadable by anno (HF viewer runs arbitrary code; data is not stored as raw files).",
        splits: ["train", "dev", "test"],
        tasks: ["discourse_relations", "speech_act_classification"],
        access_status: Public,
        categories: [abstract_anaphora, dialogue],
    },

    // =========================================================================
    // Speech acts / dialogue acts (classification)
    // =========================================================================
    ViraDialogActsLive {
        name: "ViRA Dialog Acts (Live)",
        description: "Dialogue act classification dataset (text + label). Suitable as a speech-act baseline.",
        url: "https://huggingface.co/datasets/ibm-research/vira-dialog-acts-live",
        entity_types: [],
        language: "en",
        domain: "dialogue",
        license: "Research",
        year: 2024,
        format: "HF-rows",
        notes: "HF datasets-server provides `text` and integer `label` per example; loader derives label names from gold labels if registry entity_types is empty.",
        splits: ["train", "validation", "test"],
        tasks: ["speech_act_classification"],
        hf_id: "ibm-research/vira-dialog-acts-live",
        access_status: HuggingFace,
        categories: [dialogue],
    },

    // =========================================================================
    // Discourse relations (classification)
    // =========================================================================
    DisrptEngDepScidtbRels {
        name: "DISRPT (eng.dep.scidtb.rels)",
        description: "DISRPT discourse relation classification (SciDTB, English, dependency framework).",
        url: "https://datasets-server.huggingface.co/rows?dataset=multilingual-discourse-hub%2Fdisrpt&config=eng.dep.scidtb.rels&split=train&offset=0&length=100",
        entity_types: [],
        language: "en",
        domain: "scientific",
        license: "Research",
        year: 2024,
        format: "HF-rows",
        notes: "HF rows include `unit1_txt`, `unit2_txt`, and string `label`. Loader joins unit texts with ` [SEP] ` and evaluates as text classification.",
        splits: ["train", "validation", "test"],
        tasks: ["discourse_relations"],
        hf_id: "multilingual-discourse-hub/disrpt",
        access_status: HuggingFace,
        categories: [abstract_anaphora],
    },
    DisrptDeuRstPccRels {
        name: "DISRPT (deu.rst.pcc.rels)",
        description: "DISRPT discourse relation classification (PCC, German, RST framework).",
        url: "https://datasets-server.huggingface.co/rows?dataset=multilingual-discourse-hub%2Fdisrpt&config=deu.rst.pcc.rels&split=train&offset=0&length=100",
        entity_types: [],
        language: "de",
        domain: "news",
        license: "Research",
        year: 2024,
        format: "HF-rows",
        notes: "Same row schema as English SciDTB rels: `unit1_txt`, `unit2_txt`, and string `label`.",
        splits: ["train", "validation", "test"],
        tasks: ["discourse_relations"],
        hf_id: "multilingual-discourse-hub/disrpt",
        access_status: HuggingFace,
        categories: [abstract_anaphora, multilingual],
    },

    // =========================================================================
    // Discourse segmentation (EDU boundaries) via DISRPT CoNLL-U exports
    // =========================================================================
    DisrptEngDepScidtbConlluSeg {
        name: "DISRPT (eng.dep.scidtb.conllu, Seg)",
        description: "Discourse unit segmentation (EDU boundaries) from DISRPT CoNLL-U export (SciDTB, English).",
        url: "https://datasets-server.huggingface.co/rows?dataset=multilingual-discourse-hub%2Fdisrpt&config=eng.dep.scidtb.conllu&split=train&offset=0&length=100",
        entity_types: ["SEG"],
        language: "en",
        domain: "scientific",
        license: "Research",
        year: 2024,
        format: "HF-rows",
        notes: "HF rows contain token arrays (`form`) and per-token `misc` with `Seg=B-seg` boundaries. Loader converts segments into BIO tags of type `SEG`.",
        splits: ["train", "validation", "test"],
        tasks: ["discourse_segmentation"],
        hf_id: "multilingual-discourse-hub/disrpt",
        access_status: HuggingFace,
        categories: [abstract_anaphora],
    },
    DisrptDeuRstPccConlluSeg {
        name: "DISRPT (deu.rst.pcc.conllu, Seg)",
        description: "Discourse unit segmentation (EDU boundaries) from DISRPT CoNLL-U export (PCC, German).",
        url: "https://datasets-server.huggingface.co/rows?dataset=multilingual-discourse-hub%2Fdisrpt&config=deu.rst.pcc.conllu&split=train&offset=0&length=100",
        entity_types: ["SEG"],
        language: "de",
        domain: "news",
        license: "Research",
        year: 2024,
        format: "HF-rows",
        notes: "Same row schema as English SciDTB conllu: `form` tokens and `misc` with `Seg=B-seg` boundaries.",
        splits: ["train", "validation", "test"],
        tasks: ["discourse_segmentation"],
        hf_id: "multilingual-discourse-hub/disrpt",
        access_status: HuggingFace,
        categories: [abstract_anaphora, multilingual],
    },
    // DISRPT 2025: Unified cross-framework discourse benchmark
    DISRPT2025 {
        name: "DISRPT 2025",
        description: "Cross-formalism benchmark for discourse segmentation, connective detection, and relation classification. 39 corpora, 16 languages, 6 frameworks.",
        url: "https://github.com/disrpt/sharedtask2025",
        entity_types: ["DISCOURSE_UNIT", "CONNECTIVE", "DISCOURSE_RELATION"],
        language: "mul",
        domain: "general",
        license: "Research",
        citation: "DISRPT Organizers (2025)",
        paper_url: "https://aclanthology.org/2025.disrpt-1.1.pdf",
        year: 2025,
        format: "CoNLL",
        size_hint: "39 corpora, 5.1M tokens, 311k relations",
        notes: "Unified 17-label relation scheme mapping 353 original labels. Includes RST, eRST, SDRT, PDTB, dependency, ISO frameworks. Czech, German, English, Basque, Persian, French, Italian, Dutch, Nigerian Pidgin, Polish, Portuguese, Russian, Spanish, Thai, Turkish, Chinese. EMNLP 2025 shared task.",
        splits: ["train", "dev", "test"],
        tasks: ["discourse_segmentation", "discourse_relations"],
        access_status: Public,
        categories: [abstract_anaphora, multilingual, dialogue],
    },

    // =========================================================================
    // Ancient / Classical Languages
    // =========================================================================
    AncientGreekUD {
        name: "Ancient Greek UD",
        description: "Universal Dependencies for Ancient Greek. Homeric through Byzantine.",
        url: "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master/grc_perseus-ud-test.conllu",
        entity_types: ["PER", "LOC", "ORG"],
        language: "grc",
        domain: "literature",
        license: "CC-BY-NC-SA-3.0",
        citation: "Celano et al. (2016)",
        paper_url: "https://aclanthology.org/L16-1158/",
        year: 2016,
        format: "CoNLLU",
        example: "# text = μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος\n1\tμῆνιν\tμῆνις\tNOUN\t_\t_\t2\tobj\t_\tO\n5\tἈχιλῆος\tἈχιλλεύς\tPROPN\t_\t_\t4\tnmod\t_\tB-PER",
        notes: "Perseus treebank; spans 1500+ years of Greek; Homeric to Byzantine",
        categories: [ner, ancient],
    },
    LatinUD {
        name: "Latin UD",
        description: "Universal Dependencies for Latin. Classical through Medieval.",
        url: "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master/la_ittb-ud-test.conllu",
        entity_types: ["PER", "LOC", "ORG"],
        language: "la",
        domain: "literature",
        license: "CC-BY-NC-SA-3.0",
        citation: "Passarotti et al. (2017)",
        paper_url: "https://aclanthology.org/W17-6526/",
        format: "CoNLLU",
        notes: "Index Thomisticus treebank; medieval scholastic",
        categories: [ner, ancient],
    },
    CopticScriptorium {
        name: "Coptic Scriptorium",
        description: "Sahidic Coptic with multi-layer annotation. ~50k tokens.",
        url: "https://data.copticscriptorium.org/",
        entity_types: ["PER", "LOC", "ORG"],
        language: "cop",
        domain: "religious",
        license: "CC-BY-4.0",
        citation: "Zeldes & Schroeder (2016)",
        paper_url: "https://aclanthology.org/L16-1313/",
        format: "CoNLLU",
        notes: "Multi-layer morphology/syntax/entities/coreference; requires ANNIS export",
        access_status: Registration,
        categories: [ner, ancient],
    },
    LT4HALA {
        name: "LT4HALA Hebrew",
        description: "Biblical Hebrew NER and coreference annotation.",
        url: "",  // Research access
        entity_types: ["PER", "LOC", "ORG", "GPE"],
        language: "hbo",
        domain: "religious",
        license: "Research",
        citation: "LREC-COLING 2024 LT4HALA Workshop",
        paper_url: "https://lt4hala2024.github.io/",
        notes: "First systematic biblical Hebrew NER+coref",
        tasks: ["ner", "coref"],
        access_status: ContactAuthors,
        categories: [ner, coref, ancient],
    },
    ORACC {
        name: "ORACC",
        description: "Open Richly Annotated Cuneiform Corpus. Sumerian, Akkadian, Urartian.",
        url: "http://oracc.museum.upenn.edu/",
        entity_types: ["PER", "LOC", "ORG", "DIVINE"],
        language: "akk",
        domain: "historical",
        license: "CC-BY-SA-3.0",
        citation: "ORACC Project",
        paper_url: "http://oracc.museum.upenn.edu/doc/about/index.html",
        format: "JSON",
        notes: "Cuneiform logographic+syllabic polyphony challenges; JSON export via API",
        access_status: Registration,
        categories: [ner, ancient],
    },

    // =========================================================================
    // Low-Resource / African Languages
    // =========================================================================
    MasakhaNER {
        name: "MasakhaNER",
        description: "NER for 10 African languages. PER/LOC/ORG/DATE.",
        url: "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/data/yor/test.txt",
        entity_types: ["PER", "LOC", "ORG", "DATE"],
        language: "mul",
        domain: "news",
        license: "CC-BY-4.0",
        citation: "Adelani et al. (2021)",
        paper_url: "https://aclanthology.org/2021.tacl-1.66/",
        year: 2021,
        format: "CoNLL",
        annotation_scheme: "BIO",
        example: "Olúṣẹ́gun B-PER\nObásanjọ́ I-PER\nní O\nìlú O\nAbẹ́òkúta B-LOC\n. O",
        notes: "Critically underrepresented languages; community-driven; tonal diacritics preserved",
        hf_id: "masakhane/masakhaner",
        access_status: HuggingFace,
        categories: [ner, multilingual, low_resource, african_language],
    },
    MasakhaNER2 {
        name: "MasakhaNER 2.0",
        description: "Extended MasakhaNER with 20+ African languages.",
        url: "https://huggingface.co/datasets/masakhane/masakhaner2",
        entity_types: ["PER", "LOC", "ORG", "DATE"],
        language: "mul",
        domain: "news",
        license: "CC-BY-NC-4.0",
        citation: "Adelani et al. (2022)",
        paper_url: "https://aclanthology.org/2022.emnlp-main.298/",
        year: 2022,
        format: "CoNLL",
        annotation_scheme: "BIO",
        notes: "Extended to 20 languages; includes tonal languages",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        hf_id: "masakhane/masakhaner2",
        access_status: HuggingFace,
        categories: [ner, multilingual, low_resource, african_language],
    },
    AfriSenti {
        name: "AfriSenti",
        description: "Sentiment analysis for 14 African languages. 110k+ tweets. SemEval 2023 Task 12.",
        url: "https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment",
        entity_types: ["positive", "neutral", "negative"],
        language: "mul",
        domain: "social_media",
        license: "CC-BY-4.0",
        citation: "Muhammad et al. (2023)",
        paper_url: "https://aclanthology.org/2023.emnlp-main.862/",
        year: 2023,
        format: "HuggingFace",
        size_hint: "~111k tweets",
        example: "tweet: ይሄው ነው አይደል የእውቀትሽ ጥግ (Amharic)\nlabel: negative",
        notes: "Amharic, Algerian/Moroccan Arabic, Hausa, Igbo, Kinyarwanda, Oromo, Nigerian Pidgin, Mozambican Portuguese, Swahili, Tigrinya, Xitsonga, Twi, Yoruba",
        splits: ["train", "validation", "test"],
        tasks: ["sentiment"],
        hf_id: "shmuhammad/AfriSenti-twitter-sentiment",
        access_status: HuggingFace,
        categories: [multilingual, low_resource, social_media, african_language],
    },
    AfriQA {
        name: "AfriQA",
        description: "Cross-lingual QA for 10 African languages. Wikipedia-based.",
        url: "https://huggingface.co/datasets/masakhane/afriqa",
        entity_types: [],  // QA dataset
        language: "mul",
        domain: "wikipedia",
        license: "CC-BY-4.0",
        citation: "Ogundepo et al. (2023)",
        paper_url: "https://aclanthology.org/2023.findings-emnlp.997/",
        year: 2023,
        format: "JSONL",
        notes: "Cross-lingual retrieval QA; questions in African languages, passages in English/target",
        splits: ["train", "dev", "test"],
        tasks: ["qa"],
        hf_id: "masakhane/afriqa",
        access_status: HuggingFace,
        categories: [multilingual, low_resource, african_language],
    },
    MasakhaNEWS {
        name: "MasakhaNEWS",
        description: "News topic classification for 16 African languages.",
        url: "https://huggingface.co/datasets/masakhane/masakhanews",
        entity_types: ["business", "entertainment", "health", "politics", "religion", "sports", "technology"],
        language: "mul",
        domain: "news",
        license: "Apache-2.0",
        citation: "Adelani et al. (2023)",
        paper_url: "https://aclanthology.org/2023.acl-long.574/",
        year: 2023,
        format: "HuggingFace",
        notes: "7 topics: business, entertainment, health, politics, religion, sports, technology",
        splits: ["train", "dev", "test"],
        tasks: ["text_classification"],
        hf_id: "masakhane/masakhanews",
        access_status: HuggingFace,
        categories: [multilingual, low_resource, news, african_language],
    },
    // =========================================================================
    // Text Classification (for GLiNER multi-task classification capability)
    // =========================================================================
    // Note: GLiNER multi-task supports zero-shot classification via entity_types as class labels

    AGNews {
        name: "AG News",
        description: "News article topic classification. 4 classes: World, Sports, Business, Sci/Tech.",
        url: "https://huggingface.co/datasets/fancyzhx/ag_news",
        entity_types: ["World", "Sports", "Business", "Sci/Tech"],
        language: "en",
        domain: "news",
        license: "Non-commercial",
        citation: "Zhang et al. (2015)",
        paper_url: "https://arxiv.org/abs/1509.01626",
        year: 2015,
        format: "HuggingFace",
        size_hint: "120k train, 7.6k test",
        notes: "Character-level ConvNet paper; standard text classification benchmark",
        splits: ["train", "test"],
        tasks: ["text_classification"],
        hf_id: "ag_news",
        access_status: HuggingFace,
        categories: [ner],  // News articles contain named entities; can be used for transfer
    },

    DBPedia14 {
        name: "DBPedia-14",
        description: "Wikipedia article classification. 14 non-overlapping classes from DBpedia ontology.",
        url: "https://huggingface.co/datasets/fancyzhx/dbpedia_14",
        entity_types: ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"],
        language: "en",
        domain: "wikipedia",
        license: "CC-BY-SA-3.0",
        citation: "Zhang et al. (2015)",
        paper_url: "https://arxiv.org/abs/1509.01626",
        year: 2015,
        format: "HuggingFace",
        size_hint: "560k train, 70k test",
        notes: "14 classes: Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork",
        splits: ["train", "test"],
        tasks: ["text_classification"],
        hf_id: "dbpedia_14",
        access_status: HuggingFace,
        categories: [ner, entity_linking],  // Wikipedia entities; relates to entity linking
    },

    YahooAnswers {
        name: "Yahoo Answers Topic",
        description: "Question-answer topic classification. 10 classes covering diverse topics.",
        url: "https://huggingface.co/datasets/community-datasets/yahoo_answers_topics",
        entity_types: ["Society", "Science", "Health", "Education", "Computers", "Sports", "Business", "Entertainment", "Family", "Politics"],
        language: "en",
        domain: "qa",
        license: "Non-commercial",
        citation: "Zhang et al. (2015)",
        paper_url: "https://arxiv.org/abs/1509.01626",
        year: 2015,
        format: "HuggingFace",
        size_hint: "1.4M train, 60k test",
        notes: "10 classes: Society, Science, Health, Education, Computers, Sports, Business, Entertainment, Family, Politics",
        splits: ["train", "test"],
        tasks: ["text_classification"],
        hf_id: "community-datasets/yahoo_answers_topics",
        access_status: HuggingFace,
        categories: [dialogue],
    },

    TREC {
        name: "TREC Question Classification",
        description: "Question type classification. 6 coarse classes, 50 fine-grained types.",
        url: "https://huggingface.co/datasets/trec",
        entity_types: ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"],
        language: "en",
        domain: "qa",
        license: "CC-BY-4.0",
        citation: "Li & Roth (2002)",
        paper_url: "https://www.aclweb.org/anthology/C02-1150/",
        year: 2002,
        format: "HuggingFace",
        size_hint: "5.5k train, 500 test",
        notes: "6 coarse: Abbreviation, Entity, Description, Human, Location, Numeric; 50 fine-grained",
        splits: ["train", "test"],
        tasks: ["text_classification"],
        hf_id: "trec",
        access_status: HuggingFace,
        categories: [dialogue],
    },

    TweetTopic {
        name: "TweetTopic",
        description: "Multi-label topic classification for tweets. 6 domains, 19 topics.",
        url: "https://huggingface.co/datasets/cardiffnlp/tweet_topic_multi",
        entity_types: ["arts", "business", "daily_life", "pop_culture", "science", "sports"],
        language: "en",
        domain: "social_media",
        license: "CC-BY-4.0",
        citation: "Antypas et al. (2022)",
        paper_url: "https://aclanthology.org/2022.coling-1.299/",
        year: 2022,
        format: "HuggingFace",
        size_hint: "~11k tweets",
        notes: "Multi-label; domains: arts, business, daily_life, pop_culture, science, sports; zero-shot benchmark",
        splits: ["train", "validation", "test"],
        tasks: ["text_classification"],
        hf_id: "cardiffnlp/tweet_topic_multi",
        access_status: HuggingFace,
        categories: [social_media],
    },

    MasakhaPOS {
        name: "MasakhaPOS",
        description: "Part-of-speech tagging for 20 African languages.",
        url: "https://github.com/masakhane-io/masakhane-pos",
        entity_types: ["NOUN", "VERB", "ADJ", "ADV", "PRON", "PROPN", "ADP", "AUX", "CCONJ", "DET", "INTJ", "NUM", "PART", "PUNCT", "SCONJ", "SYM", "X"],
        language: "mul",
        domain: "general",
        license: "MIT",
        citation: "Dione et al. (2023)",
        paper_url: "https://aclanthology.org/2023.acl-long.609/",
        year: 2023,
        format: "CoNLL-U",
        annotation_scheme: "IOB2",
        notes: "Universal Dependencies tagset; includes Bambara, Ewe, Mossi, Chichewa",
        splits: ["train", "dev", "test"],
        tasks: ["pos"],
        hf_id: "masakhane/masakhane-pos",
        access_status: HuggingFace,
        categories: [multilingual, low_resource, african_language],
    },
    WikiANN {
        name: "WikiANN",
        description: "Silver-standard NER from Wikipedia hyperlinks. 282 languages.",
        url: "https://huggingface.co/datasets/unimelb-nlp/wikiann",
        entity_types: ["PER", "LOC", "ORG"],
        language: "mul",
        domain: "wikipedia",
        license: "CC-BY-SA-4.0",
        citation: "Pan et al. (2017)",
        paper_url: "https://aclanthology.org/P17-1178/",
        example: "tokens: [Berlin, is, the, capital, of, Germany]\nner_tags: [B-LOC, O, O, O, O, B-LOC]",
        notes: "Silver annotations; noisy but massive coverage",
        hf_id: "unimelb-nlp/wikiann",
        hf_config: "en",
        access_status: HuggingFace,
        categories: [ner, multilingual, low_resource],
    },
    NaijaNER {
        name: "NaijaNER",
        description: "Nigerian Pidgin NER corpus.",
        url: "",  // Research access
        entity_types: ["PER", "LOC", "ORG"],
        language: "pcm",
        domain: "social_media",
        license: "Research",
        citation: "Oyewusi et al. (2021)",
        paper_url: "https://arxiv.org/abs/2102.05236",
        notes: "Nigerian Pidgin English; code-mixing common",
        access_status: ContactAuthors,
        categories: [ner, low_resource],
    },

    // =========================================================================
    // Arcane / Specialized Domains
    // =========================================================================
    WIESP2022NER {
        name: "WIESP2022-NER (DEAL)",
        description: "Astrophysics NER from NASA ADS. 31 entity types: facilities, wavelengths, telescopes, archives.",
        url: "https://huggingface.co/datasets/adsabs/WIESP2022-NER",
        entity_types: ["WAVELENGTH", "TELESCOPE", "FACILITY", "MODEL", "ARCHIVE", "DATASET", "MISSION"],
        language: "en",
        domain: "scientific",
        license: "CC-BY-4.0",
        citation: "Grezes et al. (2022)",
        paper_url: "https://aclanthology.org/2022.wiesp-1.9/",
        year: 2022,
        format: "JSONL",
        size_hint: "~3000 annotated abstracts",
        notes: "AACL-IJCNLP 2022 WIESP shared task; NASA ADS astrophysics literature",
        hf_id: "adsabs/WIESP2022-NER",
        access_status: HuggingFace,
        categories: [ner, arcane_domain],
    },
    DutchArchaeology {
        name: "Dutch Archaeology NER",
        description: "Archaeological excavation reports from DANS archive. 31k annotations across 6 entity types.",
        url: "https://live.european-language-grid.eu/catalogue/corpus/13410",
        entity_types: ["ARTEFACT", "PERIOD", "MATERIAL", "LOCATION", "SPECIES", "CONTEXT"],
        language: "nl",
        domain: "archaeology",
        license: "CC-BY-4.0",
        citation: "Brandsen et al. (2020)",
        paper_url: "https://aclanthology.org/2020.lrec-1.562/",
        year: 2020,
        format: "CoNLL",
        size_hint: "~31k entity annotations, high IAA (0.95)",
        notes: "Dutch grey literature; basis for ArcheoBERTje model",
        categories: [ner, arcane_domain],
    },
    ENer {
        name: "E-NER (EDGAR-NER)",
        description: "NER for US SEC EDGAR filings. 52 documents, 400k+ tokens with legal entities.",
        url: "https://raw.githubusercontent.com/terenceau1/E-NER-Dataset/main/all.csv",
        entity_types: ["PERSON", "COURT", "BUSINESS", "GOVERNMENT", "LOCATION", "LEGISLATION"],
        language: "en",
        domain: "legal",
        license: "GPL-3.0",
        citation: "Au et al. (2022)",
        paper_url: "https://aclanthology.org/2022.nllp-1.22/",
        year: 2022,
        format: "CSV",
        size_hint: "52 SEC filings, 400k+ tokens",
        notes: "10-K, 8-K, prospectuses; CSV token,tag format with BIO scheme",
        tasks: ["ner"],
        categories: [ner, arcane_domain],
    },
    // =========================================================================
    // Niche Domain Datasets (Food, Cybersecurity, etc.)
    // =========================================================================
    FINER {
        name: "FINER (Food Ingredients NER)",
        description: "Food ingredient NER from AllRecipes. 182k sentences with ingredient phrases in IOB2 format.",
        url: "https://figshare.com/ndownloader/files/36144501",
        entity_types: ["INGREDIENT", "PRODUCT", "QUANTITY", "UNIT", "STATE"],
        language: "en",
        domain: "food",
        license: "CC-BY-4.0",
        citation: "Komariah et al. (2022)",
        paper_url: "https://doi.org/10.6084/m9.figshare.20222361",
        year: 2022,
        format: "CoNLL",
        annotation_scheme: "IOB2",
        size_hint: "~182k sentences, ingredient phrases",
        notes: "Semi-supervised multi-model construction from AllRecipes; RAR archive with CoNLL format",
        splits: ["train", "test"],
        tasks: ["ner", "slot_filling"],
        categories: [ner, arcane_domain],
    },
    AnnoCTR {
        name: "AnnoCTR (Cyber Threat Reports)",
        description: "Cyber threat intelligence NER with MITRE ATT&CK linking. 400 annotated documents from commercial CTI vendors.",
        url: "https://github.com/boschresearch/anno-ctr-lrec-coling-2024/archive/refs/heads/main.zip",
        entity_types: ["ORGANIZATION", "LOCATION", "SECTOR", "TIME", "CODE", "THREAT_ACTOR", "MALWARE", "TOOL", "TACTIC", "TECHNIQUE"],
        language: "en",
        domain: "cybersecurity",
        license: "CC-BY-SA-4.0",
        citation: "Lange et al. (2024)",
        paper_url: "https://arxiv.org/abs/2404.07765",
        year: 2024,
        format: "JSONL",
        annotation_scheme: "BIO",
        size_hint: "400 documents, multi-layer annotation",
        notes: "LREC-COLING 2024; links to Wikipedia and MITRE ATT&CK KB; includes entity linking task",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "entity_linking"],
        categories: [ner, arcane_domain],
    },
    CRAFT {
        name: "CRAFT",
        description: "Colorado Richly Annotated Full-Text. 97 PubMed articles with multi-layer annotation including coreference.",
        url: "https://github.com/UCDenver-ccp/CRAFT/archive/refs/heads/master.zip",
        entity_types: ["GENE", "PROTEIN", "CHEMICAL", "CELL", "ORGANISM"],
        language: "en",
        domain: "biomedical",
        license: "CC-BY-3.0",
        citation: "Bada et al. (2012)",
        paper_url: "https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-13-161",
        year: 2012,
        format: "XML",
        annotation_scheme: "Standoff",
        size_hint: "97 full-text articles, ~790k tokens",
        notes: "Full-text (not just abstracts); 10 ontologies used for normalization",
        categories: [coref, biomedical, arcane_domain],
    },

    // =========================================================================
    // Adversarial / Robustness Evaluation
    // =========================================================================
    WNUT16 {
        name: "WNUT-16",
        description: "Twitter NER workshop shared task. Focus on rare and emerging entities in noisy social media text.",
        // Stable raw snapshot from aritter/twitter_nlp (has train/dev/test in one folder).
        url: "https://raw.githubusercontent.com/aritter/twitter_nlp/65f3d77134c40d920db8d431c5c6faef1c051c94/data/annotated/wnut16/data/test",
        // Note: dataset uses lowercase hyphenated labels (person, geo-loc, company, facility, product, other)
        entity_types: ["person", "geo-loc", "company", "facility", "product", "other"],
        language: "en",
        domain: "social_media",
        license: "CC-BY-4.0",
        citation: "Strauss et al. (2016)",
        paper_url: "https://aclanthology.org/W16-3919/",
        year: 2016,
        format: "CoNLL",
        annotation_scheme: "BIO",
        size_hint: "3,856 test tweets, 2,394 train",
        notes: "89% unseen test entities; predecessor to WNUT-17; harder than standard benchmarks",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner, social_media, adversarial],
    },

    // =========================================================================
    // More Ancient Languages
    // =========================================================================
    SanskritUD {
        name: "Sanskrit UD",
        description: "Universal Dependencies for Vedic and Classical Sanskrit. Includes Vedas and epics.",
        url: "https://raw.githubusercontent.com/UniversalDependencies/UD_Sanskrit-Vedic/master/sa_vedic-ud-test.conllu",
        entity_types: ["PER", "LOC", "ORG"],
        language: "sa",
        domain: "religious",
        license: "CC-BY-SA-4.0",
        citation: "Hellwig et al. (2020)",
        paper_url: "https://aclanthology.org/2020.lrec-1.632/",
        year: 2020,
        format: "CoNLLU",
        notes: "Oldest Indo-European language with extensive NLP resources; Devanagari script",
        categories: [ner, ancient, low_resource],
    },
    OldEnglishUD {
        name: "Old English UD",
        description: "Universal Dependencies for Old English (Anglo-Saxon). York-Toronto-Helsinki corpus.",
        url: "https://raw.githubusercontent.com/UniversalDependencies/UD_Old_English-Cairo/118f11ad906fb15d930825fadce9ef9eccca9347/ang_cairo-ud-test.conllu",
        entity_types: ["PER", "LOC", "ORG"],
        language: "ang",
        domain: "historical",
        license: "CC-BY-SA-4.0",
        citation: "Tischler & Walkden (2019)",
        paper_url: "https://aclanthology.org/W19-4214/",
        year: 2019,
        format: "CoNLLU",
        notes: "Anglo-Saxon; insular script variations; 5th-11th century CE. NOTE: UD repo naming drifts; URL pinned to a specific commit.",
        categories: [ner, ancient, historical, low_resource],
    },
    OldNorseUD {
        name: "Old Norse UD",
        description: "Universal Dependencies for Old Norse/Icelandic Sagas. PROIEL and ISWOC treebanks.",
        url: "https://raw.githubusercontent.com/UniversalDependencies/UD_Icelandic-IcePaHC/ca72a59affd87c4b7b9067ae56efa7a694a7b4c4/is_icepahc-ud-test.conllu",
        entity_types: ["PER", "LOC", "ORG"],
        language: "non",
        domain: "literature",
        license: "CC-BY-SA-4.0",
        citation: "Rögnvaldsson et al. (2012)",
        paper_url: "https://aclanthology.org/L12-1148/",
        year: 2012,
        format: "CoNLLU",
        notes: "IcePaHC treebank (historical Icelandic; Old Norse family). URL pinned to a specific commit.",
        categories: [ner, ancient, literary, low_resource],
    },

    // =========================================================================
    // Code-Switching Datasets
    // =========================================================================
    CALCS2018 {
        name: "CALCS-2018",
        description: "Code-Switching Workshop shared task. English-Spanish Twitter NER with 9 entity types.",
        url: "https://code-switching.github.io/2018/",
        entity_types: ["PER", "LOC", "ORG", "GROUP", "TITLE", "PROD", "EVENT", "TIME", "OTHER"],
        language: "mul",
        domain: "social_media",
        license: "Research",
        citation: "Aguilar et al. (2018)",
        paper_url: "https://aclanthology.org/W18-3219/",
        year: 2018,
        format: "CoNLL",
        annotation_scheme: "BIO",
        notes: "Spanglish; first major code-switching NER shared task",
        categories: [ner, social_media, multilingual, low_resource],
    },
    HinglishNER {
        name: "Hinglish NER",
        description: "Hindi-English code-mixed social media NER. Roman script Hindi mixed with English.",
        url: "https://github.com/murali1996/CodemixedNLP",
        entity_types: ["PER", "LOC", "ORG"],
        language: "mul",
        domain: "social_media",
        license: "CC-BY-4.0",
        citation: "Priyadharshini et al. (2020)",
        paper_url: "https://aclanthology.org/2020.calcs-1.6/",
        year: 2020,
        format: "JSONL",
        annotation_scheme: "BIO",
        notes: "GLUECoS/LinCE benchmark; download via CodemixedNLP toolkit; Romanized Hindi; ~400M speakers use code-switching daily",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner, social_media, multilingual, low_resource],
    },

    // =========================================================================
    // Medieval / Historical NER
    // =========================================================================
    MedievalCharterNER {
        name: "Medieval Charter NER",
        description: "Multilingual medieval charter NER. Latin, French, Spanish from major charter collections.",
        url: "https://zenodo.org/records/6463699",
        entity_types: ["PER", "LOC", "ORG", "DATE"],
        language: "mul",
        domain: "historical",
        license: "CC-BY-4.0",
        citation: "Camps et al. (2022)",
        paper_url: "https://aclanthology.org/2022.lrec-1.530/",
        year: 2022,
        format: "CoNLL",
        size_hint: "~100k tokens across 4 charter collections",
        notes: "HOME-ALCAR, CBMA, Diplomata Belgica, CODEA; medieval Latin/vernacular",
        categories: [ner, historical, multilingual, low_resource],
    },
    CBMACharters {
        name: "CBMA Charters",
        description: "Burgundian medieval Latin charters NER. 9th-14th century diplomatic documents.",
        url: "",  // Access via Zenodo/project
        entity_types: ["PER", "LOC", "ORG", "DATE", "TITLE"],
        language: "la",
        domain: "historical",
        license: "Research",
        citation: "Perreaux (2021)",
        paper_url: "https://dhq-static.digitalhumanities.org/pdf/000574.pdf",
        year: 2021,
        format: "CoNLL",
        notes: "Chartae Burgundiae Medii Aevi; medieval Latin; notarial hands",
        access_status: ContactAuthors,
        categories: [ner, ancient, historical, low_resource],
    },

    // =========================================================================
    // Speech NER
    // =========================================================================
    MSNER {
        name: "MSNER",
        description: "Multilingual Spoken NER. Speech-to-NER on VoxPopuli parliamentary speeches.",
        url: "https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/ZTVMIX",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "mul",
        domain: "speech",
        license: "CC-BY-4.0",
        citation: "Evain et al. (2024)",
        paper_url: "https://aclanthology.org/2024.isa-1.2/",
        year: 2024,
        format: "CoNLL",
        annotation_scheme: "BIO",
        size_hint: "~590h train, 17h gold test",
        notes: "Dutch, French, German, Spanish; ASR transcripts; first multilingual speech NER corpus",
        categories: [ner, speech, multilingual],
    },

    // =========================================================================
    // Robustness / Noise Benchmarks
    // =========================================================================
    NoiseBench {
        name: "NoiseBench",
        description: "Robustness benchmark for NER. 6 real noise types: expert, crowd, LLM, distant/weak supervision.",
        url: "https://raw.githubusercontent.com/elenamer/NoiseBench/main/data/annotations/clean.traindev",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "evaluation",
        license: "MIT",
        citation: "Merhej et al. (2024)",
        paper_url: "https://aclanthology.org/2024.emnlp-main.1011/",
        year: 2024,
        format: "CoNLL",
        size_hint: "CoNLL-03 subset with 7 label variants",
        notes: "Compares simulated vs real label noise; includes German variant; using clean subset",
        tasks: ["ner"],
        access_status: Public,
        categories: [ner, adversarial],
    },
    RockNER {
        name: "RockNER",
        description: "Robustness benchmark for NER. Real-world adversarial examples with boundary ambiguity.",
        url: "https://github.com/INK-USC/RockNER",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "evaluation",
        license: "Apache-2.0",
        citation: "Lin et al. (2021)",
        paper_url: "https://aclanthology.org/2021.acl-long.340/",
        year: 2021,
        format: "CoNLL",
        size_hint: "~1.5k challenging examples",
        notes: "ACL 2021; entity boundary attacks, rare entities, syntactic perturbations; robustness stress test",
        splits: ["test"],
        tasks: ["ner"],
        categories: [ner, adversarial],
    },
    CrossWeigh {
        name: "CrossWeigh",
        description: "Cross-lingual adversarial NER evaluation. Tests multilingual model robustness.",
        url: "https://raw.githubusercontent.com/ZihanWangKi/CrossWeigh/master/data/conllpp_test.txt",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "evaluation",
        license: "MIT",
        citation: "Wang et al. (2019)",
        paper_url: "https://aclanthology.org/D19-1519/",
        year: 2019,
        format: "CoNLL",
        size_hint: "Adversarial cross-lingual test sets; includes CoNLL++ cleaned version",
        notes: "Tests cross-lingual transfer robustness; character/word perturbations; zero-shot evaluation; CoNLL++ fix",
        splits: ["test"],
        tasks: ["ner"],
        access_status: Public,
        categories: [ner, adversarial, multilingual],
    },

    // =========================================================================
    // Entity Linking Datasets
    // =========================================================================
    ZELDA {
        name: "ZELDA",
        description: "Entity disambiguation benchmark. 95k Wikipedia paragraphs, 8 ED datasets unified.",
        url: "https://raw.githubusercontent.com/flairNLP/zelda/main/test_data/conll/test_aida-b.conll",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "wikipedia",
        license: "MIT",
        citation: "Milich & Akbik (2023)",
        paper_url: "https://aclanthology.org/2023.eacl-main.151/",
        year: 2023,
        format: "CoNLL",
        size_hint: "95k paragraphs, 825k entities",
        notes: "Standardized ED evaluation; Wikipedia KB; no emerging entities; using AIDA-B test subset",
        splits: ["test"],
        tasks: ["el", "ner"],
        access_status: Public,
        categories: [ner, entity_linking],
    },
    TweetNERD {
        name: "TweetNERD",
        description: "Twitter NER + Entity Linking. End-to-end NERD benchmark spanning 2010-2021.",
        url: "https://zenodo.org/records/6617192",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "social_media",
        license: "CC-BY-4.0",
        citation: "Mishra et al. (2022)",
        paper_url: "https://arxiv.org/abs/2210.08129",
        year: 2022,
        format: "JSONL",
        size_hint: "340k+ tweets",
        notes: "NeurIPS 2022; temporal drift; NER + EL + end-to-end NERD",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "el"],
        categories: [ner, social_media],
    },
    AIDACoNLL {
        name: "AIDA-CoNLL",
        description: "Primary entity linking benchmark linking CoNLL-2003 mentions to Wikipedia. De-facto standard for end-to-end EL evaluation.",
        url: "https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "news",
        license: "Research",
        citation: "Hoffart et al. (2011)",
        paper_url: "https://aclanthology.org/D11-1072/",
        year: 2011,
        format: "CoNLL",
        annotation_scheme: "IOB2",
        size_hint: "~1,400 docs, ~34k mentions linked to Wikipedia",
        notes: "Built on Reuters CoNLL-2003; AIDA-train/A/B splits; foundational EL benchmark; YAGO KB",
        splits: ["train", "testa", "testb"],
        tasks: ["ner", "el", "entity_linking", "ned"],
        categories: [ner, entity_linking],
    },

    // =========================================================================
    // Additional Nested NER
    // =========================================================================
    ACE2005 {
        name: "ACE 2005",
        description: "Automatic Content Extraction 2005. Nested NER + relations + events.",
        url: "",  // Requires LDC license
        entity_types: ["PER", "ORG", "GPE", "LOC", "FAC", "WEA", "VEH"],
        language: "en",
        domain: "news",
        license: "LDC",
        citation: "Walker et al. (2006)",
        paper_url: "https://catalog.ldc.upenn.edu/LDC2006T06",
        year: 2005,
        format: "XML",
        annotation_scheme: "Standoff",
        size_hint: "~600 documents",
        notes: "Gold standard for nested NER; includes Arabic/Chinese; defines modern IE evaluation",
        access_status: Registration,
        categories: [ner, nested_ner, relation_extraction],
    },
    NNE {
        name: "NNE (Nested Named Entities)",
        description: "Large-scale nested NER corpus from Wikipedia/news. Deep nesting up to 6 levels.",
        url: "https://github.com/nickyringland/nested_named_entities",
        entity_types: ["PER", "LOC", "ORG", "GPE", "NORP", "FAC", "PRODUCT", "EVENT", "WORK", "LAW"],
        language: "en",
        domain: "news",
        license: "CC-BY-4.0",
        citation: "Ringland et al. (2019)",
        paper_url: "https://aclanthology.org/P19-1510/",
        year: 2019,
        format: "CoNLL",
        size_hint: "~280k tokens, deep nesting",
        notes: "ACL 2019; based on ACE/OntoNotes; up to 6 nested levels; stress test for nested NER",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner, nested_ner],
    },
    GENIANested {
        name: "GENIA Nested",
        description: "Biomedical nested NER from GENIA corpus. Up to 3 levels of nesting.",
        url: "https://raw.githubusercontent.com/thecharm/boundary-aware-nested-ner/master/Our_boundary-aware_model/data/genia/genia.test.iob2",
        entity_types: ["DNA", "RNA", "PROTEIN", "CELL_LINE", "CELL_TYPE"],
        language: "en",
        domain: "biomedical",
        license: "GENIA Project License",
        citation: "Kim et al. (2003)",
        paper_url: "https://aclanthology.org/W03-1302/",
        year: 2003,
        format: "CoNLL",
        size_hint: "~2k abstracts",
        example: "[[IL-2 receptor] alpha chain] promoter\n[IL-2 receptor]: PROTEIN, [IL-2 receptor alpha chain]: PROTEIN (nested)",
        notes: "Canonical biomedical nested NER benchmark; used alongside ACE for nested NER evaluation",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner, nested_ner, biomedical],
    },
    ChineseNestedNER {
        name: "Chinese Nested NER",
        description: "Chinese nested named entity recognition. Multiple levels of embedded entities.",
        url: "https://github.com/LeeSureman/Nested-NER",
        entity_types: ["PER", "ORG", "LOC", "GPE"],
        language: "zh",
        domain: "news",
        license: "CC-BY-4.0",
        citation: "Wang et al. (2020)",
        year: 2020,
        format: "JSONL",
        size_hint: "~20k sentences",
        notes: "Chinese nested NER benchmark; designed for span-based model evaluation; CJK characters",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner, nested_ner, multilingual],
    },
    SCINERNested {
        name: "SciNER Nested",
        description: "Scientific paper NER with nested annotations. Methods, tasks, and datasets.",
        url: "https://github.com/allenai/sciie",
        entity_types: ["TASK", "METHOD", "METRIC", "MATERIAL", "GENERIC"],
        language: "en",
        domain: "scientific",
        license: "Apache-2.0",
        citation: "Luan et al. (2018)",
        paper_url: "https://aclanthology.org/D18-1360/",
        year: 2018,
        format: "JSONL",
        size_hint: "~500 abstracts",
        notes: "Scientific information extraction; nested spans common in methodology descriptions",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "re"],
        categories: [ner, nested_ner, arcane_domain],
    },

    // =========================================================================
    // Additional Discontinuous NER
    // =========================================================================
    ShAReCLEF {
        name: "ShARe/CLEF",
        description: "Shared Annotated Resources for clinical NER. ShARe/CLEF eHealth shared task.",
        url: "",  // Research access via PhysioNet
        entity_types: ["DISORDER", "FINDING", "PROCEDURE"],
        language: "en",
        domain: "clinical",
        license: "PhysioNet",
        citation: "Pradhan et al. (2013)",
        paper_url: "https://aclanthology.org/S13-2056/",
        year: 2013,
        format: "BRAT",
        annotation_scheme: "Standoff",
        size_hint: "~300 clinical notes",
        notes: "Discontinuous clinical entities; SNOMED-CT normalization; de-identified records",
        access_status: Registration,
        categories: [ner, biomedical, discontinuous_ner],
    },
    GermEvalDiscontinuous {
        name: "GermEval Discontinuous",
        description: "German discontinuous NER from GermEval 2014. Non-contiguous entity spans.",
        url: "https://sites.google.com/site/germaboreval/data",
        entity_types: ["PER", "ORG", "LOC", "OTH"],
        language: "de",
        domain: "news",
        license: "CC-BY-4.0",
        citation: "Benikova et al. (2014)",
        paper_url: "https://aclanthology.org/W14-1707/",
        year: 2014,
        format: "CoNLL",
        size_hint: "~87k tokens",
        notes: "German discontinuous entities; derived entities; embedded entities",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner, discontinuous_ner, multilingual],
    },
    ADRDiscontinuous {
        name: "ADR Discontinuous",
        description: "Adverse Drug Reaction corpus with discontinuous mentions. Patient forum posts.",
        url: "https://github.com/Aitslab/ADR-DisNER",
        entity_types: ["ADR", "DRUG", "SYMPTOM"],
        language: "en",
        domain: "biomedical",
        license: "CC-BY-4.0",
        citation: "Metke-Jimenez et al. (2016)",
        year: 2016,
        format: "BRAT",
        size_hint: "~2k posts",
        notes: "Social media ADR mentions; many discontinuous spans; health forum text",
        categories: [ner, biomedical, discontinuous_ner, social_media],
    },
    PubMedDiscontinuous {
        name: "PubMed Discontinuous",
        description: "PubMed abstracts with discontinuous biomedical entities. Complex entity boundaries.",
        url: "https://github.com/dmis-lab/discontinuous-ner",
        entity_types: ["CHEMICAL", "DISEASE", "GENE"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "Dai et al. (2020)",
        year: 2020,
        format: "CoNLL",
        size_hint: "~8k abstracts",
        notes: "Scientific abstracts; discontinuous chemical and disease mentions",
        categories: [ner, biomedical, discontinuous_ner],
    },

    // =========================================================================
    // Additional Relation Extraction
    // =========================================================================
    TACRED {
        name: "TACRED",
        description: "TAC Relation Extraction Dataset. 42 relations from TAC KBP.",
        url: "",  // LDC license
        entity_types: ["PER", "ORG"],
        language: "en",
        domain: "news",
        license: "LDC",
        citation: "Zhang et al. (2017)",
        paper_url: "https://aclanthology.org/D17-1004/",
        year: 2017,
        format: "JSONL",
        size_hint: "106k examples",
        example: "subj: 'Tim Cook', obj: 'Apple', relation: per:employee_of, text: 'Tim Cook is the CEO of Apple Inc.'",
        notes: "42 relations; majority no_relation; known label noise; Re-TACRED fixes some issues",
        access_status: Registration,
        categories: [relation_extraction],
    },
    SemEval2010Task8 {
        name: "SemEval-2010 Task 8",
        description: "Semantic relation classification between nominals. 9 relation types.",
        url: "https://github.com/sahitya0000/Relation-Classification",
        entity_types: ["e1", "e2"],  // Entity markers for relation endpoints
        language: "en",
        domain: "mixed",
        license: "Research",
        citation: "Hendrickx et al. (2010)",
        paper_url: "https://aclanthology.org/S10-1006/",
        year: 2010,
        format: "Custom",
        size_hint: "~10k examples",
        notes: "Classic RE benchmark; 9 directed relations + OTHER; small but influential",
        categories: [relation_extraction],
    },
    FewRel {
        name: "FewRel",
        description: "Few-shot relation classification benchmark. 100 relations from Wikidata.",
        url: "https://raw.githubusercontent.com/thunlp/FewRel/master/data/val_wiki.json",
        entity_types: ["head", "tail"],  // Relation endpoint markers
        language: "en",
        domain: "wikipedia",
        license: "MIT",
        citation: "Han et al. (2018)",
        paper_url: "https://aclanthology.org/D18-1514/",
        year: 2018,
        format: "JSONL",
        size_hint: "70k instances, 100 relations",
        notes: "N-way K-shot evaluation; Wikidata relations; FewRel 2.0 adds domain adaptation",
        hf_id: "few_rel",
        access_status: HuggingFace,
        categories: [relation_extraction],
    },
    NYT10 {
        name: "NYT-10",
        description: "New York Times distant supervision RE. 24 Freebase relations.",
        url: "http://iesl.cs.umass.edu/riedel/ecml/",
        entity_types: ["PER", "ORG", "LOC"],
        language: "en",
        domain: "news",
        license: "Research",
        citation: "Riedel et al. (2010)",
        paper_url: "https://aclanthology.org/W10-1001/",
        year: 2010,
        format: "Custom",
        size_hint: "~266k sentences",
        notes: "Distant supervision using Freebase alignment; distantly supervised; noisy labels; majority no_relation; standard DS-RE benchmark",
        splits: ["train", "test"],
        tasks: ["re"],
        categories: [relation_extraction],
    },

    // =========================================================================
    // Additional Biomedical
    // =========================================================================
    JNLPBA {
        name: "JNLPBA",
        description: "JNLPBA 2004 shared task. Bio-entity recognition in PubMed abstracts.",
        url: "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/JNLPBA/test.tsv",
        entity_types: ["PROTEIN", "DNA", "RNA", "CELL_TYPE", "CELL_LINE"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "Kim et al. (2004)",
        paper_url: "https://aclanthology.org/W04-1213/",
        year: 2004,
        format: "CoNLL",
        annotation_scheme: "IOB2",
        size_hint: "~2,400 abstracts",
        notes: "Extended GENIA categories; foundational bioNER benchmark",
        categories: [ner, biomedical],
    },
    S800 {
        name: "S800",
        description: "Species-800 corpus. Species name recognition in biomedical text.",
        url: "https://species.jensenlab.org/files/S800-1.0.tar.gz",
        entity_types: ["SPECIES"],
        language: "en",
        domain: "biomedical",
        license: "CC-BY-4.0",
        citation: "Pafilis et al. (2013)",
        paper_url: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0065390",
        year: 2013,
        format: "XML",
        size_hint: "800 abstracts",
        notes: "Species NER; taxonomy normalization; tar.gz archive with XML format; requires manual extraction and conversion to CoNLL",
        access_status: DependsOnOther,
        categories: [ner, biomedical],
    },

    // =========================================================================
    // Temporal NER (Time expressions, Events)
    // =========================================================================
    TempEval3 {
        name: "TempEval-3",
        description: "Temporal annotation benchmark. TIMEX, EVENT spans, and temporal relations.",
        url: "https://figshare.com/articles/dataset/TempEval-3_data/9586532",
        entity_types: ["TIMEX", "EVENT"],
        language: "en",
        domain: "news",
        license: "CC-BY-4.0",
        citation: "UzZaman et al. (2013)",
        paper_url: "https://aclanthology.org/S13-2001/",
        year: 2013,
        format: "TimeML",
        notes: "Time expression NER + event detection + temporal ordering; TimeBank based; TE3-Platinum gold standard",
        splits: ["train", "test"],
        tasks: ["ner", "temporal"],
        categories: [ner],
    },
    TimexRecognitionSentenceOriginal {
        name: "TIMEX Recognition (Sentence, Original)",
        description: "Sentence-level temporal extraction dataset with TIMEX spans (and optional EVENT/SIGNAL). Provided as HF rows with character offsets.",
        url: "https://huggingface.co/datasets/mdg-nlp/timex-recognition-sentence-original",
        entity_types: ["TIMEX", "EVENT", "SIGNAL"],
        language: "en",
        domain: "news",
        license: "Research",
        year: 2024,
        format: "HF-rows",
        notes: "HF datasets-server export contains `text` plus standoff spans in `time_expressions`/`event_expressions`/`signal_expressions` with `start_char`/`end_char` offsets; loader converts to token-level BIO for NER-style scoring.",
        splits: ["train", "validation", "test"],
        tasks: ["ner", "temporal", "events"],
        hf_id: "mdg-nlp/timex-recognition-sentence-original",
        access_status: HuggingFace,
        categories: [ner, temporal],
    },
    TimeBank12 {
        name: "TimeBank 1.2",
        description: "Canonical temporal IE corpus. News articles with TIMEX3, events, and temporal links (TLINKs).",
        url: "https://catalog.ldc.upenn.edu/LDC2006T08",
        entity_types: ["TIMEX3", "EVENT", "SIGNAL"],
        language: "en",
        domain: "news",
        license: "LDC",
        citation: "Pustejovsky et al. (2003)",
        paper_url: "https://aclanthology.org/W03-1808/",
        year: 2003,
        format: "TimeML",
        size_hint: "183 news documents, ~9k events",
        notes: "Original TimeML corpus; basis for TempEval shared tasks; temporal ordering gold standard",
        splits: ["train", "test"],
        tasks: ["ner", "temporal", "events"],
        categories: [ner],
    },
    MATRES {
        name: "MATRES",
        description: "Multi-Axis Temporal Relations. Cleaner, more consistent event-event temporal relation annotations.",
        url: "https://github.com/qiangning/MATRES",
        entity_types: ["EVENT"],
        language: "en",
        domain: "news",
        license: "Research",
        citation: "Ning et al. (2018)",
        paper_url: "https://aclanthology.org/P18-1212/",
        year: 2018,
        format: "Custom",
        size_hint: "~13.5k temporal relation pairs",
        notes: "Re-annotated TimeBank/AQUAINT subset; higher inter-annotator agreement; verb-centric",
        splits: ["train", "dev", "test"],
        tasks: ["temporal", "events", "re"],
        categories: [ner, relations],
    },
    THYME {
        name: "THYME",
        description: "Temporal Histories of Your Medical Events. Clinical temporal IE with events and relations.",
        url: "",
        entity_types: ["EVENT", "TIMEX3", "SECTIONTIME", "DOCTIME"],
        language: "en",
        domain: "clinical",
        license: "Research",
        citation: "Styler et al. (2014)",
        paper_url: "https://aclanthology.org/L14-1393/",
        year: 2014,
        format: "Custom",
        size_hint: "~600 clinical notes (colon cancer, brain cancer)",
        notes: "THYME guidelines; clinical events, temporal expressions, narrative containers; Clinical TempEval basis",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "temporal", "events"],
        access_status: Registration,
        categories: [ner, clinical, biomedical],
    },
    I2B2Temporal {
        name: "i2b2 2012 Temporal",
        description: "Clinical temporal relations challenge. Events, TIMEX3, and TLINKs in discharge summaries.",
        url: "",
        entity_types: ["EVENT", "TIMEX3"],
        language: "en",
        domain: "clinical",
        license: "Research",
        citation: "Sun et al. (2013)",
        paper_url: "https://aclanthology.org/S13-2035/",
        year: 2012,
        format: "Custom",
        size_hint: "~310 clinical notes",
        notes: "i2b2 2012 challenge; requires DUA; clinical temporal relation extraction benchmark",
        splits: ["train", "test"],
        tasks: ["ner", "temporal", "re"],
        access_status: Registration,
        categories: [ner, clinical, biomedical, relations],
    },

    // =========================================================================
    // Multimodal NER
    // =========================================================================
    Twitter2015MNER {
        name: "Twitter-2015 MNER",
        description: "Multimodal NER on Twitter. Text + image for entity recognition.",
        url: "https://github.com/jefferyYu/UMT",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "social_media",
        license: "Research",
        citation: "Zhang et al. (2018)",
        paper_url: "https://aclanthology.org/N18-1078/",
        year: 2018,
        format: "CoNLL",
        size_hint: "~8,000 tweets with images",
        notes: "Multimodal; images via Google Drive archive; UMT preprocessing; first MNER dataset; visual context aids entity recognition",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "mner"],
        categories: [ner, social_media, multimodal],
    },

    // =========================================================================
    // Music Domain
    // =========================================================================
    DistantListeningCorpus {
        name: "Distant Listening Corpus",
        description: "1,283 musical scores with harmonic annotations. String quartet + piano music with Roman numeral analysis.",
        url: "https://zenodo.org/records/15150283",
        entity_types: ["CHORD", "KEY", "MODULATION", "CADENCE", "PHRASE"],
        language: "mul",
        domain: "music",
        license: "CC-BY-4.0",
        citation: "Devaney et al. (2024)",
        paper_url: "https://doi.org/10.5281/zenodo.15150283",
        year: 2024,
        format: "TSV",
        size_hint: "1,283 scores, 190k+ annotations",
        notes: "Music theory annotation corpus; Roman numeral analysis; supports harmonic sequence extraction; Zenodo archive",
        splits: ["train"],
        tasks: ["sequence_labeling", "harmonic_analysis"],
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Privacy/PII Detection
    // =========================================================================
    PIIMasking200k {
        name: "PII Masking 200k",
        description: "200k synthetic examples for PII detection and masking. Covers 50+ PII types.",
        url: "https://huggingface.co/datasets/ai4privacy/pii-masking-200k",
        entity_types: ["EMAIL", "PHONE", "SSN", "ADDRESS", "NAME", "DOB", "CREDIT_CARD", "PASSPORT", "IP_ADDRESS", "LICENSE"],
        language: "mul",
        domain: "privacy",
        license: "Apache-2.0",
        citation: "AI4Privacy (2024)",
        year: 2024,
        format: "JSONL",
        size_hint: "~200k examples",
        notes: "Synthetic PII dataset; multi-language; 50+ entity types; useful for privacy compliance testing",
        splits: ["train"],
        tasks: ["ner", "pii_detection"],
        hf_id: "ai4privacy/pii-masking-200k",
        access_status: HuggingFace,
        categories: [ner],
    },

    // =========================================================================
    // Legal/SEC Domain
    // =========================================================================
    ENERSec {
        name: "E-NER SEC",
        description: "Legal NER from SEC EDGAR filings. 52 documents with financial entity annotations.",
        url: "https://github.com/jnishii/E-NER",
        entity_types: ["ORG", "LOC", "DATE", "MONEY", "PERCENT", "PERSON", "PRODUCT", "CARDINAL"],
        language: "en",
        domain: "legal",
        license: "MIT",
        citation: "Nishii et al. (2023)",
        year: 2023,
        format: "CSV",
        size_hint: "52 documents, ~400k tokens",
        notes: "SEC 10-K and 10-Q filings; financial regulatory domain; legal entity extraction",
        splits: ["train", "test"],
        tasks: ["ner"],
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Entity Linking / Named Entity Disambiguation Datasets
    // =========================================================================
    MSNBCEL {
        name: "MSNBC",
        description: "Small news article entity linking dataset. Commonly used for out-of-domain EL evaluation.",
        url: "",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "news",
        license: "Research",
        citation: "Cucerzan (2007)",
        paper_url: "https://aclanthology.org/D07-1074/",
        year: 2007,
        format: "Custom",
        size_hint: "~20 docs, ~700 mentions",
        notes: "Early EL benchmark; often used as OOD test set alongside AIDA",
        splits: ["test"],
        tasks: ["el", "entity_linking", "ned"],
        access_status: ContactAuthors,
        categories: [entity_linking],
    },
    AQUAINT {
        name: "AQUAINT",
        description: "Newswire entity linking dataset from AQUAINT corpus. Wikipedia-linked mentions.",
        url: "",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "news",
        license: "LDC",
        citation: "Milne & Witten (2008)",
        year: 2008,
        format: "Custom",
        size_hint: "~50 docs, ~700 mentions",
        notes: "Commonly paired with AIDA for comprehensive EL evaluation",
        splits: ["test"],
        tasks: ["el", "entity_linking", "ned"],
        access_status: Registration,
        categories: [entity_linking],
    },
    KORE50 {
        name: "KORE50",
        description: "Short, highly ambiguous entity linking snippets. Tests disambiguation difficulty.",
        url: "https://github.com/KORE50/KORE50-NIF-NER",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "evaluation",
        license: "CC-BY-4.0",
        citation: "Hoffart et al. (2012)",
        paper_url: "https://aclanthology.org/P12-1084/",
        year: 2012,
        format: "Custom",
        size_hint: "50 sentences, 144 mentions",
        notes: "Highly ambiguous mentions; stress-tests disambiguation ability; includes YAGO types",
        splits: ["test"],
        tasks: ["el", "entity_linking", "ned"],
        categories: [entity_linking, adversarial],
    },
    WNEDWiki {
        name: "WNED-WIKI",
        description: "Large-scale Wikipedia entity linking dataset extracted from Wikipedia hyperlinks.",
        url: "https://github.com/wikipedia2vec/wikipedia2vec",
        entity_types: ["ENTITY"],
        language: "en",
        domain: "wikipedia",
        license: "Research",
        citation: "Guo & Barbosa (2018)",
        year: 2018,
        format: "Custom",
        size_hint: "~6M mentions",
        notes: "Large-scale silver annotations from Wikipedia hyperlinks",
        splits: ["test"],
        tasks: ["el", "entity_linking"],
        categories: [entity_linking],
    },
    WNEDClueweb {
        name: "WNED-ClueWeb",
        description: "Web-scale entity linking from ClueWeb corpus. Tests EL on noisy web text.",
        url: "",
        entity_types: ["ENTITY"],
        language: "en",
        domain: "web",
        license: "Research",
        citation: "Guo & Barbosa (2018)",
        year: 2018,
        format: "Custom",
        size_hint: "~10k docs",
        notes: "Web-scale EL benchmark; tests robustness on noisy web text",
        splits: ["test"],
        tasks: ["el", "entity_linking"],
        access_status: Registration,
        categories: [entity_linking],
    },
    BELB {
        name: "BELB",
        description: "Biomedical Entity Linking Benchmark unifying 11 corpora across 7 knowledge bases. Standardized biomedical EL evaluation.",
        url: "https://github.com/sg-wbi/belb",
        entity_types: ["Disease", "Chemical", "Gene", "Species", "CellLine", "Variant"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "Furrer et al. (2023)",
        paper_url: "https://academic.oup.com/bioinformatics/article/39/11/btad698/7425450",
        year: 2023,
        format: "JSONL",
        size_hint: "11 corpora, 7 KBs",
        notes: "Unifies BC5CDR-Chemical, BC5CDR-Disease, NCBI-Disease, BC2GN, NLM-Gene, Linnaeus, S800, GNORMPLUS, MedMentions, and more",
        splits: ["train", "dev", "test"],
        tasks: ["el", "entity_linking", "ned"],
        categories: [entity_linking, biomedical],
    },
    MELO {
        name: "MELO",
        description: "Multilingual Entity Linking of Occupations. 48 datasets across 21 languages for occupation EL.",
        url: "https://github.com/avature/melo-benchmark",
        entity_types: ["OCCUPATION"],
        language: "mul",
        domain: "general",
        license: "Apache-2.0",
        citation: "Retyk et al. (2024)",
        paper_url: "https://aclanthology.org/2024.lrec-main.889/",
        year: 2024,
        format: "JSONL",
        size_hint: "48 datasets, 21 languages",
        notes: "Zero-shot multilingual EL; includes sentence encoders and lexical baselines",
        splits: ["test"],
        tasks: ["el", "entity_linking"],
        categories: [entity_linking, multilingual],
    },

    MewsliX {
        name: "Mewsli-X",
        description: "Multilingual entity linking across 50 languages. Wikipedia-linked mentions for zero-shot cross-lingual EL.",
        url: "https://huggingface.co/datasets/izhx/mewsli-x",
        entity_types: ["ENTITY"],  // Wikipedia entities
        language: "mul",
        domain: "news",
        license: "Apache-2.0",
        citation: "Botha et al. (2020)",
        paper_url: "https://arxiv.org/abs/2010.11856",
        year: 2020,
        format: "TSV",
        size_hint: "~300k mentions across 50 languages",
        notes: "Zero-shot cross-lingual EL benchmark; from WikiNews; Wikipedia KB",
        splits: ["test"],
        tasks: ["el", "entity_linking", "ned"],
        access_status: Public,
        categories: [entity_linking, multilingual],
    },

    // =========================================================================
    // Long-Document Coreference
    // =========================================================================
    // NOTE: BookCoref (sapienzanlp) is defined earlier (~line 2035). This is a different dataset.
    BookCorefBamman {
        name: "BookCoref (Bamman)",
        description: "Full-novel coreference with automatic silver and manual gold annotations. Includes Animal Farm, Siddhartha, Pride and Prejudice.",
        url: "https://huggingface.co/datasets/spacemanidol/BookCoref",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "literature",
        license: "Research",
        citation: "Bamman et al. (2025)",
        paper_url: "https://arxiv.org/abs/2507.12075",
        year: 2025,
        format: "JSONL",
        size_hint: "~200k tokens per document",
        notes: "Long-document coref benchmark; tests models on full novels; silver + gold annotations. HF-hosted but appears gated in practice; treat as manual unless you have explicit access.",
        splits: ["test"],
        tasks: ["coref"],
        hf_id: "spacemanidol/BookCoref",
        access_status: ContactAuthors,
        categories: [coref, literary, long_document],
    },
    NovelCR {
        name: "NovelCR",
        description: "Large-scale bilingual (EN/ZH) novel coreference. 148k EN mentions, 311k ZH mentions with 74-83% spanning 3+ sentences.",
        url: "https://github.com/NovelCR/NovelCR",
        entity_types: ["PER", "LOC", "ORG"],
        language: "mul",
        domain: "literature",
        license: "Research",
        citation: "Chen et al. (2024)",
        paper_url: "https://openreview.net/forum?id=zuZXwj9aSE",
        year: 2024,
        format: "JSONL",
        size_hint: "EN: 148k mentions, ZH: 311k mentions",
        notes: "Long-span coreference; bilingual EN/ZH; most coreferences span multiple sentences",
        splits: ["train", "dev", "test"],
        tasks: ["coref"],
        categories: [coref, literary, long_document, multilingual],
    },
    AgCNER {
        name: "AgCNER",
        description: "Large-scale Chinese agricultural NER. 66k samples, ~207k entities, 3.9M characters.",
        url: "https://springernature.figshare.com/collections/AgCNER_the_First_Large-Scale_Chinese_Named_Entity_Recognition_Dataset_for_Agricultural_Diseases_and_Pests/6807873",
        entity_types: ["CROP", "DISEASE", "PEST", "CHEMICAL", "VARIETY", "LOCATION", "TIME"],
        language: "zh",
        domain: "scientific",
        license: "CC-BY-4.0",
        citation: "AgCNER Team (2024)",
        paper_url: "https://www.nature.com/articles/s41597-024-03578-5",
        year: 2024,
        format: "JSONL",
        size_hint: "66k samples, ~207k entities, 3.9M characters",
        notes: "Nature Scientific Data 2024; 13 entity types; long agricultural case reports; domain NER",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner, long_document, multilingual, arcane_domain],
    },
    ScrollsQMSum {
        name: "SCROLLS QMSum",
        description: "Long-document QA from SCROLLS benchmark. Query-focused meeting summarization.",
        url: "https://github.com/tau-nlp/scrolls",
        entity_types: [],
        language: "en",
        domain: "dialogue",
        license: "MIT",
        citation: "Shaham et al. (2022)",
        paper_url: "https://aclanthology.org/2022.emnlp-main.823/",
        year: 2022,
        format: "JSONL",
        size_hint: "~1.5k meeting transcripts, avg 10k tokens",
        notes: "EMNLP 2022; SCROLLS benchmark subset; long meeting transcripts; tests long-context understanding",
        splits: ["train", "dev", "test"],
        tasks: ["qa"],
        categories: [long_document, dialogue],
    },
    LongDocNER {
        name: "Long Document NER",
        description: "Long-document NER benchmark. Tests entity recognition across extended contexts.",
        url: "https://github.com/xhuang28/LongDocNER",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "mixed",
        license: "MIT",
        citation: "Huang et al. (2024)",
        year: 2024,
        format: "JSONL",
        size_hint: "~500 documents, avg 8k tokens",
        notes: "Tests long-context NER models; entity consistency across document boundaries",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner, long_document],
    },
    BookSumCoref {
        name: "BookSum Coref",
        description: "Coreference annotations on book chapters from BookSum. Long literary texts.",
        url: "https://github.com/salesforce/booksum",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "literature",
        license: "Research",
        citation: "Kryscinski et al. (2022)",
        paper_url: "https://aclanthology.org/2022.findings-emnlp.438/",
        year: 2022,
        format: "JSONL",
        size_hint: "~400 chapters, avg 5k tokens",
        notes: "Book chapters with coref chains; tests long-span coreference resolution",
        splits: ["train", "test"],
        tasks: ["coref"],
        categories: [coref, long_document, literary],
    },
    MultiBioNERLong {
        name: "Multi-Bio Long NER",
        description: "Long biomedical document NER. Full-text articles vs abstracts.",
        url: "https://github.com/dmis-lab/multi-bio-ner",
        entity_types: ["GENE", "CHEMICAL", "DISEASE", "SPECIES"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "Lee et al. (2023)",
        year: 2023,
        format: "JSONL",
        size_hint: "~1k full-text articles",
        notes: "Full-text vs abstract NER comparison; tests biomedical long-context models",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner, long_document, biomedical],
    },
    RadCoref {
        name: "RadCoref",
        description: "Radiology report coreference from MIMIC-CXR. Clinical domain long-document coref.",
        url: "https://physionet.org/content/rad-coreference-resolution/",
        entity_types: ["ANATOMY", "OBSERVATION", "FINDING"],
        language: "en",
        domain: "clinical",
        license: "PhysioNet",
        citation: "Zhu et al. (2024)",
        paper_url: "https://physionet.org/content/rad-coreference-resolution/",
        year: 2024,
        format: "BRAT",
        size_hint: "~500 radiology reports",
        notes: "Clinical coref on MIMIC-CXR; requires PhysioNet credentialing; radiology-specific entities",
        splits: ["train", "test"],
        tasks: ["coref"],
        categories: [coref, clinical, biomedical],
    },

    // =========================================================================
    // Cross-Document Event Coreference
    // =========================================================================
    MEANTIME {
        name: "MEANTIME",
        description: "Multilingual news corpus with within- and cross-document event coreference. 4 languages.",
        url: "https://github.com/newsreader/meantime",
        entity_types: ["EVENT", "TIMEX", "PARTICIPANT", "LOCATION"],
        language: "mul",
        domain: "news",
        license: "CC-BY-4.0",
        citation: "Minard et al. (2016)",
        paper_url: "https://aclanthology.org/L16-1699/",
        year: 2016,
        format: "Custom",
        size_hint: "120 documents, 4 languages (EN, ES, IT, NL)",
        notes: "Multilingual CDEC; parallel annotations across languages; NewsReader project",
        splits: ["all"],
        tasks: ["coref", "event_coref", "cdcr"],
        categories: [coref, event_coref, multilingual],
    },
    FCCT {
        name: "FCC-T",
        description: "Football Coreference Corpus with token-level annotations. Cross-document event coref in sports news.",
        url: "https://github.com/cltl/FCC",
        entity_types: ["EVENT", "PARTICIPANT", "TIME", "LOCATION"],
        language: "en",
        domain: "sports",
        license: "CC-BY-4.0",
        citation: "Bugert et al. (2021)",
        paper_url: "https://direct.mit.edu/coli/article/47/3/575/102774",
        year: 2021,
        format: "CoNLL",
        size_hint: "~300 docs",
        notes: "Token-level CDEC; compatible with ECB+ and GVC; sports domain temporal reasoning",
        splits: ["train", "dev", "test"],
        tasks: ["coref", "event_coref", "cdcr"],
        categories: [coref, event_coref],
    },
    LEMONADE {
        name: "LEMONADE",
        description: "Large-scale multilingual conflict event corpus. 39k events across 20 languages for CDEC search.",
        url: "https://github.com/lemonade-coref/lemonade",
        entity_types: ["EVENT", "PARTICIPANT", "LOCATION", "TIME"],
        language: "mul",
        domain: "news",
        license: "Research",
        citation: "Eirew et al. (2025)",
        year: 2025,
        format: "JSONL",
        size_hint: "~39k events, 20 languages, 171 countries",
        notes: "Conflict event CDEC; cross-document event coreference search task; multilingual",
        splits: ["test"],
        tasks: ["coref", "event_coref", "cdcr"],
        categories: [coref, event_coref, multilingual],
    },

    // =========================================================================
    // Newer Biomedical/Clinical NER and RE
    // =========================================================================
    BioRED {
        name: "BioRED",
        description: "Document-level biomedical RE with novelty labels. BioCreative VIII shared task benchmark.",
        url: "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/",
        entity_types: ["Gene", "Disease", "Chemical", "Species", "Variant", "CellLine"],
        language: "en",
        domain: "biomedical",
        license: "Public",
        citation: "Luo et al. (2022)",
        paper_url: "https://academic.oup.com/database/article/doi/10.1093/database/baae069/7729400",
        year: 2022,
        format: "Custom",
        size_hint: "600 PubMed abstracts, 8 relation types",
        notes: "Document-level RE with novelty detection; distinguishes novel vs known relations",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "re", "relation_extraction"],
        categories: [ner, relation_extraction, biomedical],
    },
    MedMentions {
        name: "MedMentions",
        description: "Large-scale biomedical concept mentions mapped to UMLS. PubMed abstracts with fine-grained semantic types.",
        url: "https://github.com/chanzuckerberg/MedMentions",
        entity_types: ["UMLS_CONCEPT"],
        language: "en",
        domain: "biomedical",
        license: "CC0-1.0",
        citation: "Mohan & Li (2019)",
        paper_url: "https://arxiv.org/abs/1902.09476",
        year: 2019,
        format: "Custom",
        size_hint: "4,392 abstracts, 352k mentions, 35k concepts",
        notes: "UMLS concept linking; 127 semantic types; large-scale biomedical concept NER/EL",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "el", "entity_linking"],
        categories: [ner, entity_linking, biomedical],
    },
    EnzChemRED {
        name: "EnzChemRED",
        description: "Enzyme chemistry relation extraction. Links enzymes, substrates, products, cofactors from biochemical literature.",
        url: "https://github.com/ncbi-nlp/EnzChemRED",
        entity_types: ["Enzyme", "Substrate", "Product", "Cofactor", "Reaction"],
        language: "en",
        domain: "biomedical",
        license: "CC-BY-4.0",
        citation: "Schröder et al. (2024)",
        paper_url: "https://www.nature.com/articles/s41597-024-03835-7",
        year: 2024,
        format: "JSONL",
        size_hint: "~5k relation triplets",
        notes: "Specialized enzyme chemistry RE; biochemical reaction extraction",
        splits: ["train", "test"],
        tasks: ["ner", "re", "relation_extraction"],
        categories: [ner, relation_extraction, biomedical],
    },
    NCERB {
        name: "NCERB",
        description: "Named Clinical Entity Recognition Benchmark. Multi-dataset clinical NER evaluation suite.",
        url: "https://github.com/NCERB/NCERB",
        entity_types: ["Problem", "Treatment", "Test", "Medication", "Anatomy"],
        language: "en",
        domain: "clinical",
        license: "Research",
        citation: "Zhou et al. (2024)",
        paper_url: "https://arxiv.org/abs/2410.05046",
        year: 2024,
        format: "Custom",
        size_hint: "Multiple clinical corpora aggregated",
        notes: "Benchmark suite for clinical NER; evaluates LMs on healthcare entities; aggregates i2b2, n2c2, etc.",
        splits: ["test"],
        tasks: ["ner"],
        categories: [ner, clinical, biomedical],
    },
    MACCROBAT {
        name: "MACCROBAT",
        description: "Biomedical NER corpus with extensive coverage. Used with RoBERTa-WWM and deep models.",
        url: "https://figshare.com/articles/dataset/MACCROBAT2018/9764942",
        entity_types: ["Disease", "Chemical", "Gene", "Species"],
        language: "en",
        domain: "biomedical",
        license: "CC-BY-4.0",
        citation: "Islamaj et al. (2019)",
        year: 2019,
        format: "Custom",
        size_hint: "~400 abstracts",
        notes: "Multi-type biomedical NER; chemical and disease mentions",
        splits: ["train", "test"],
        tasks: ["ner"],
        categories: [ner, biomedical],
    },

    // =========================================================================
    // Additional Relation Extraction Datasets
    // =========================================================================
    ACE05RE {
        name: "ACE 2005 RE",
        description: "ACE 2005 relation extraction component. 7 entity types, 6 relation types with subtypes.",
        url: "",
        entity_types: ["PER", "ORG", "GPE", "LOC", "FAC", "VEH", "WEA"],
        language: "en",
        domain: "news",
        license: "LDC",
        citation: "Walker et al. (2006)",
        year: 2005,
        format: "XML",
        size_hint: "~600 docs, 7 relation types",
        notes: "Classic RE benchmark; requires LDC license; often used with ACE NER",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "re", "relation_extraction"],
        access_status: Registration,
        categories: [ner, relation_extraction],
    },
    CoNLL04RE {
        name: "CoNLL04 RE",
        description: "Sentence-level relation extraction from CoNLL-2004. Clean, small RE benchmark.",
        url: "https://github.com/bekou/multihead_joint_entity_relation_extraction",
        entity_types: ["PER", "ORG", "LOC", "Other"],
        language: "en",
        domain: "news",
        license: "Research",
        citation: "Roth & Yih (2004)",
        paper_url: "https://aclanthology.org/W04-2401/",
        year: 2004,
        format: "CoNLL",
        size_hint: "~1.4k sentences, 5 relation types",
        notes: "Clean sentence-level RE; joint NER+RE evaluation",
        splits: ["train", "test"],
        tasks: ["ner", "re", "relation_extraction"],
        categories: [ner, relation_extraction],
    },
    CrossRE {
        name: "CrossRE",
        description: "Cross-domain relation extraction across 6 domains. Tests RE generalization.",
        url: "https://github.com/mainlp/CrossRE",
        entity_types: ["PER", "ORG", "LOC", "MISC"],
        language: "en",
        domain: "cross_domain",
        license: "CC-BY-4.0",
        citation: "Bassignana & Plank (2022)",
        paper_url: "https://aclanthology.org/2022.emnlp-main.452/",
        year: 2022,
        format: "JSON",
        size_hint: "6 domains: AI, Literature, Music, News, Politics, Science",
        notes: "Cross-domain RE evaluation; tests transfer across domains",
        splits: ["train", "dev", "test"],
        tasks: ["re", "relation_extraction"],
        categories: [relation_extraction],
    },

    // =========================================================================
    // Unified Multilingual NER
    // =========================================================================
    UNER {
        name: "UNER",
        description: "Universal NER on Universal Dependencies. Gold NER with unified schema across 13 languages.",
        url: "https://github.com/UniversalNER/UNER",
        entity_types: ["PER", "LOC", "ORG"],
        language: "mul",
        domain: "general",
        license: "CC-BY-SA-4.0",
        citation: "Mayhew et al. (2024)",
        paper_url: "https://aclanthology.org/2024.naacl-long.243/",
        year: 2024,
        format: "CoNLLU",
        size_hint: "13 languages including Cebuano, Tagalog, Narabizi",
        notes: "Unified NER on UD treebanks; includes low-resource languages; community-driven expansion",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner, multilingual, low_resource],
    },
    IndicNER {
        name: "IndicNER",
        description: "Indian languages NER covering 11 Indian languages. Low-resource multilingual NER.",
        url: "https://github.com/AI4Bharat/IndicNER",
        entity_types: ["PER", "LOC", "ORG"],
        language: "mul",
        domain: "general",
        license: "CC-BY-4.0",
        citation: "Mhaske et al. (2022)",
        paper_url: "https://aclanthology.org/2022.findings-acl.269/",
        year: 2022,
        format: "CoNLL",
        size_hint: "11 languages: Hindi, Bengali, Telugu, Tamil, Marathi, etc.",
        notes: "Indian language NER; part of AI4Bharat initiative; low-resource focus",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner, multilingual, low_resource],
    },
    NorNE {
        name: "NorNE",
        description: "Norwegian NER covering Bokmål and Nynorsk. Morphologically rich language from news and parliament text.",
        url: "https://github.com/ltgoslo/norne",
        entity_types: ["PER", "LOC", "ORG", "GPE", "PROD", "EVT", "DRV"],
        language: "no",
        domain: "news",
        license: "CC-BY-4.0",
        citation: "Jørgensen et al. (2020)",
        paper_url: "https://aclanthology.org/2020.lrec-1.559/",
        year: 2020,
        format: "CoNLL",
        size_hint: "~600k tokens, both Bokmål and Nynorsk",
        notes: "Both Norwegian written forms; morphologically rich; 8 entity types",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner],
    },
    GermEval2014 {
        name: "GermEval 2014",
        description: "German NER shared task. Standard German NER benchmark with nested entities.",
        url: "https://sites.google.com/site/germaboreval2014/data",
        entity_types: ["PER", "LOC", "ORG", "OTH"],
        language: "de",
        domain: "news",
        license: "CC-BY-4.0",
        citation: "Benikova et al. (2014)",
        paper_url: "https://aclanthology.org/W14-1707/",
        year: 2014,
        format: "CoNLL",
        annotation_scheme: "BIO",
        size_hint: "~31k sentences",
        notes: "Standard German NER; includes nested/embedded entities; derived from Wikipedia and news",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner, nested_ner],
    },

    // =========================================================================
    // LLM-Era Evaluation Datasets
    // =========================================================================
    ReasoningNER {
        name: "ReasoningNER",
        description: "Zero-shot NER evaluation suite across 20 diverse datasets. Tests LLM NER capabilities.",
        url: "https://github.com/reasoning-ner/reasoning-ner",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "evaluation",
        license: "Research",
        citation: "Xia et al. (2025)",
        paper_url: "https://arxiv.org/abs/2511.11978",
        year: 2025,
        format: "JSONL",
        size_hint: "20 datasets across news, social, biomedical, etc.",
        notes: "Zero-shot NER evaluation; tests instruction-following and entity reasoning in LLMs",
        splits: ["test"],
        tasks: ["ner"],
        categories: [ner, adversarial],
    },
    BioNERLLaMA {
        name: "BioNER-LLaMA",
        description: "Instruction-tuned biomedical NER benchmark. Evaluates generative models on disease/chemical/gene NER.",
        url: "https://github.com/BIDS-Xu-Lab/BioNER-LLaMA",
        entity_types: ["Disease", "Chemical", "Gene"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "Keloth et al. (2024)",
        paper_url: "https://academic.oup.com/bioinformatics/article/40/4/btae163/7633405",
        year: 2024,
        format: "JSONL",
        size_hint: "Instruction-formatted from BC5CDR, NCBI, etc.",
        notes: "LLM instruction-tuning for BioNER; evaluates ChatGPT, LLaMA, etc. on biomedical entities",
        splits: ["test"],
        tasks: ["ner"],
        categories: [ner, biomedical],
    },
    MentionResolutionLLM {
        name: "Mention Resolution LLM",
        description: "MCQ-format coreference for LLMs from LitBank and FantasyCoref. Tests referential understanding on narratives.",
        url: "https://github.com/mention-resolution/mention-resolution-llm",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "literature",
        license: "Research",
        citation: "Adams et al. (2024)",
        paper_url: "https://arxiv.org/abs/2411.07466",
        year: 2024,
        format: "JSONL",
        size_hint: "MCQ from LitBank + FantasyCoref",
        notes: "Multiple-choice coref for LLM evaluation; tests ambiguous, long-distance, nested mentions",
        splits: ["test"],
        tasks: ["coref"],
        categories: [coref, literary],
    },

    // =========================================================================
    // Additional Discontinuous/Clinical NER
    // =========================================================================
    ShARe2013 {
        name: "ShARe 2013",
        description: "Clinical disorder mentions from ShARe/CLEF eHealth 2013. Discontinuous entity annotations.",
        url: "",
        entity_types: ["DISORDER"],
        language: "en",
        domain: "clinical",
        license: "Research",
        citation: "Pradhan et al. (2013)",
        paper_url: "https://aclanthology.org/S13-2056/",
        year: 2013,
        format: "Custom",
        size_hint: "~300 clinical notes",
        notes: "Clinical NER with discontinuous spans; shared task at CLEF eHealth",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "discontinuous-ner"],
        access_status: Registration,
        categories: [ner, discontinuous_ner, clinical, biomedical],
    },
    ShARe2014 {
        name: "ShARe 2014",
        description: "Clinical disorder mentions from ShARe/CLEF eHealth 2014. Improved discontinuous NER annotations.",
        url: "",
        entity_types: ["DISORDER", "ANATOMY", "MODIFIER"],
        language: "en",
        domain: "clinical",
        license: "Research",
        citation: "Mowery et al. (2014)",
        paper_url: "https://aclanthology.org/S14-2007/",
        year: 2014,
        format: "Custom",
        size_hint: "~400 clinical notes",
        notes: "Improved clinical discontinuous NER; attribute normalization",
        splits: ["train", "test"],
        tasks: ["ner", "discontinuous-ner"],
        access_status: Registration,
        categories: [ner, discontinuous_ner, clinical, biomedical],
    },
    I2B2_2010 {
        name: "i2b2 2010",
        description: "Clinical concept extraction and assertion classification. Foundational clinical NER benchmark.",
        url: "",
        entity_types: ["PROBLEM", "TREATMENT", "TEST"],
        language: "en",
        domain: "clinical",
        license: "Research",
        citation: "Uzuner et al. (2011)",
        paper_url: "https://academic.oup.com/jamia/article/18/5/552/833880",
        year: 2010,
        format: "Custom",
        size_hint: "~871 discharge summaries",
        notes: "Foundational clinical NER; requires i2b2/n2c2 data use agreement",
        splits: ["train", "test"],
        tasks: ["ner"],
        access_status: Registration,
        categories: [ner, clinical, biomedical],
    },

    // =========================================================================
    // Legal Domain
    // =========================================================================
    LexGLUENER {
        name: "LexGLUE NER",
        description: "Legal NER from LexGLUE benchmark. Legal entity extraction from case law and contracts.",
        url: "https://github.com/coastalcph/lex-glue",
        entity_types: ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "LEGAL_REF", "COURT"],
        language: "en",
        domain: "legal",
        license: "Research",
        citation: "Chalkidis et al. (2022)",
        paper_url: "https://aclanthology.org/2022.acl-long.297/",
        year: 2022,
        format: "JSONL",
        size_hint: "Part of LexGLUE benchmark suite",
        notes: "Legal domain benchmark; includes contracts, case law, legislation",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "classification"],
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Financial Domain
    // =========================================================================
    FinBenNER {
        name: "FinBen NER",
        description: "Financial NER from FinBen benchmark. Entity extraction from financial documents and filings.",
        url: "https://github.com/TheFinAI/FinBen",
        entity_types: ["COMPANY", "PERSON", "MONEY", "PERCENT", "DATE", "PRODUCT"],
        language: "en",
        domain: "financial",
        license: "Research",
        citation: "Xie et al. (2024)",
        paper_url: "https://arxiv.org/abs/2402.12659",
        year: 2024,
        format: "JSONL",
        size_hint: "Multi-task financial benchmark",
        notes: "Financial IE benchmark; includes NER, classification, QA; 2024 NeurIPS",
        splits: ["test"],
        tasks: ["ner"],
        categories: [ner, arcane_domain],
    },
    FiNER139 {
        name: "FiNER-139",
        description: "Financial NER with 139 fine-grained entity types. SEC 10-K/10-Q filings.",
        url: "https://huggingface.co/datasets/nlpaueb/finer-139",
        entity_types: ["COMPANY", "EXECUTIVE", "SUBSIDIARY", "PRODUCT", "REGULATION", "FINANCIAL_METRIC"],
        language: "en",
        domain: "financial",
        license: "MIT",
        citation: "Shah et al. (2023)",
        year: 2023,
        format: "JSONL",
        size_hint: "~10k sentences, 139 entity types",
        notes: "Fine-grained financial NER; hierarchical entity types; SEC filings",
        splits: ["train", "test"],
        tasks: ["ner"],
        hf_id: "nlpaueb/finer-139",
        access_status: HuggingFace,
        categories: [ner, nested_ner, arcane_domain],
    },

    // =========================================================================
    // Constructed Languages
    // Constructed language NLP is valuable for:
    // - Testing language universals vs learned biases
    // - Evaluating cross-lingual transfer to truly novel languages
    // - Edge case testing (highly regular morphology, OVS word order, etc.)
    // =========================================================================
    TaggedPBCEsperanto {
        name: "taggedPBC Esperanto",
        description: "POS-tagged Esperanto from Parallel Bible Corpus. ~1800 sentences with word-level alignment.",
        url: "https://github.com/clab/taggedPBC",
        entity_types: ["PER", "LOC", "ORG"],
        language: "eo",
        domain: "religious",
        license: "CC-BY-4.0",
        citation: "Zeman et al. (2025)",
        paper_url: "https://arxiv.org/abs/2505.12560",
        year: 2024,
        format: "CoNLLU",
        size_hint: "~1800 sentences, New Testament",
        notes: "First large-scale annotated Esperanto corpus; cross-linguistic POS; no dedicated NER layer yet",
        splits: ["train"],
        tasks: ["ner", "sequence_labeling"],
        categories: [ner, constructed, low_resource],
    },
    TaggedPBCKlingon {
        name: "taggedPBC Klingon",
        description: "POS-tagged Klingon from Parallel Bible Corpus. OVS word order with complex verbal morphology.",
        url: "https://github.com/clab/taggedPBC",
        entity_types: ["PER", "LOC", "ORG"],
        language: "tlh",
        domain: "religious",
        license: "CC-BY-4.0",
        citation: "Zeman et al. (2025)",
        paper_url: "https://arxiv.org/abs/2505.12560",
        year: 2024,
        format: "CoNLLU",
        size_hint: "~1800 sentences, New Testament",
        notes: "Klingon has OVS word order, agglutinative verbs with suffix slots; tests non-SVO processing",
        splits: ["train"],
        tasks: ["ner", "sequence_labeling"],
        categories: [ner, constructed, low_resource],
    },
    UDEsperantoCairo {
        name: "UD Esperanto Cairo",
        description: "Universal Dependencies treebank for Esperanto. Syntax annotation without NER layer.",
        url: "https://raw.githubusercontent.com/UniversalDependencies/UD_Esperanto-Cairo/master/eo_cairo-ud-test.conllu",
        entity_types: ["PER", "LOC", "ORG"],
        language: "eo",
        domain: "constructed_language",
        license: "CC-BY-SA-4.0",
        citation: "Wennerberg (2020)",
        paper_url: "https://universaldependencies.org/eo/index.html",
        year: 2020,
        format: "CoNLLU",
        size_hint: "2 documents (Manifesto, Cairo sample)",
        notes: "Small treebank illustrating UD annotation for Esperanto; no NER layer but suitable base for annotation",
        splits: ["test"],
        tasks: ["ner"],
        categories: [ner, constructed, low_resource],
    },
    KlingonEffectLID {
        name: "Klingon Effect LID",
        description: "Language ID dataset with 11 constructed languages. 14.2M sentences across 101 languages.",
        url: "https://wmdqs.org/submissions-2025/19.pdf",
        entity_types: [],
        language: "mul",
        domain: "general",
        license: "Research",
        citation: "Moura et al. (2025)",
        paper_url: "https://wmdqs.org/submissions-2025/19.pdf",
        year: 2025,
        format: "Custom",
        size_hint: "14.2M sentences, 101 languages (11 constructed)",
        notes: "Shows constructed languages (Esperanto, Klingon, Ido, Interlingua) outperform natural languages in LID",
        splits: ["test"],
        tasks: ["classification"],
        categories: [constructed, multilingual, adversarial],
    },
    LojbanTatoeba {
        name: "Lojban Tatoeba",
        description: "Lojban-English sentence pairs from Tatoeba. Logical language translation corpus.",
        url: "https://tatoeba.org/en/downloads",
        entity_types: [],
        language: "jbo",
        domain: "constructed_language",
        license: "CC-BY-2.0",
        citation: "Tatoeba Project (2024)",
        year: 2024,
        format: "TSV",
        size_hint: "~3k sentence pairs",
        notes: "Logical constructed language; predicate logic syntax; useful for semantic parsing studies",
        splits: ["all"],
        tasks: ["mt"],
        categories: [constructed, low_resource],
    },
    InterlingueWikipedia {
        name: "Interlingue Wikipedia",
        description: "Interlingue (Occidental) Wikipedia text corpus. International auxiliary language.",
        url: "https://dumps.wikimedia.org/iewiki/",
        entity_types: [],
        language: "ie",
        domain: "encyclopedia",
        license: "CC-BY-SA-4.0",
        citation: "Wikimedia (2024)",
        year: 2024,
        format: "XML",
        size_hint: "~4k articles",
        notes: "Western European vocabulary roots; naturalistic IAL; smaller than Esperanto Wikipedia",
        splits: ["all"],
        tasks: ["lm"],
        categories: [constructed, low_resource],
    },
    TokiPonaCorpus {
        name: "Toki Pona Corpus",
        description: "Toki Pona minimalist language corpus. 120-word language for semantic simplification.",
        url: "https://github.com/kilipan/toki-pona-corpus",
        entity_types: [],
        language: "tok",
        domain: "constructed_language",
        license: "CC0-1.0",
        citation: "Lang (2021)",
        year: 2021,
        format: "TXT",
        size_hint: "~50k tokens",
        notes: "Philosophical constructed language; only 120 words; tests compositional semantics",
        splits: ["all"],
        tasks: ["lm"],
        categories: [constructed, low_resource],
    },

    // =========================================================================
    // Recent 2024-2025 Benchmarks
    // =========================================================================
    OmniNER2025 {
        name: "OmniNER2025",
        description: "Diverse fine-grained Chinese NER covering informal text (social media, forums). Large-scale benchmark for modern NER models.",
        url: "",
        entity_types: ["PER", "LOC", "ORG", "GPE", "FAC", "PRODUCT", "EVENT"],
        language: "zh",
        domain: "social_media",
        license: "Research",
        citation: "OmniNER Team (2025)",
        paper_url: "https://dl.acm.org/doi/10.1145/3726302.3730048",
        year: 2025,
        format: "JSONL",
        size_hint: "Large-scale Chinese informal text",
        notes: "2025 benchmark for fine-grained Chinese NER; expands beyond formal text; tests LLM capabilities",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        access_status: NotYetReleased,
        categories: [ner, social_media, multilingual],
    },
    LegalCore {
        name: "LegalCore",
        description: "Event coreference in long legal documents. Long-distance cross-section event links.",
        url: "",
        entity_types: ["EVENT", "PARTICIPANT", "TIME"],
        language: "en",
        domain: "legal",
        license: "Research",
        citation: "ACL Findings (2025)",
        paper_url: "https://aclanthology.org/2025.findings-acl.1284.pdf",
        year: 2025,
        format: "JSONL",
        size_hint: "Long legal documents, largest tokens per document",
        notes: "ACL 2025; benchmarks Llama-3.1, Mistral, Qwen, GPT-4; LLMs underperform supervised baselines",
        splits: ["train", "dev", "test"],
        tasks: ["event_coref", "coref"],
        access_status: NotYetReleased,
        categories: [coref, event_coref, long_document, arcane_domain],
    },
    Zcoref {
        name: "Z-coref",
        description: "Joint coreference and zero-pronoun resolution. For languages with pro-drop (Chinese, Japanese, Korean).",
        url: "",
        entity_types: ["ZERO_PRONOUN", "ENTITY"],
        language: "mul",
        domain: "general",
        license: "Research",
        citation: "Z-coref Authors (2024)",
        paper_url: "https://arxiv.org/pdf/2504.05824",
        year: 2024,
        format: "CoNLL",
        size_hint: "Multi-language pro-drop coreference",
        notes: "Tests handling of dropped arguments; critical for CJK languages; zero anaphora resolution",
        splits: ["train", "dev", "test"],
        tasks: ["coref"],
        access_status: NotYetReleased,
        categories: [coref, multilingual, abstract_anaphora],
    },
    TikTalkCoref {
        name: "TikTalkCoref",
        description: "Chinese social media dialogue coreference. Person mentions in Douyin video comments with singleton handling.",
        url: "",
        entity_types: ["PER"],
        language: "zh",
        domain: "social_media",
        license: "Research",
        citation: "Li, Gong & Fu (2025)",
        paper_url: "https://arxiv.org/abs/2504.14321",
        year: 2025,
        format: "Custom",
        size_hint: "1,012 dialogues, 2,179 mentions, 1,435 clusters",
        notes: "First Chinese MCR dataset for social media. Maverick outperforms e2e-coref (65.5 vs 39.1 Avg.F1). High singleton rate: 44% pronouns, 34% proper names, 22% common nouns. Text-only portion; multimodal aspect out of scope.",
        splits: ["train", "dev", "test"],
        tasks: ["coref"],
        access_status: NotYetReleased,
        categories: [coref, dialogue, social_media],
    },
    MHERCL {
        name: "MHERCL",
        description: "Historical long-tail entity linking benchmark. Tests LLM behavior on rare/historical Wikidata entities.",
        url: "https://arxiv.org/html/2505.03473v1",
        entity_types: ["HISTORICAL_ENTITY"],
        language: "en",
        domain: "historical",
        license: "Research",
        citation: "MHERCL Authors (2025)",
        paper_url: "https://arxiv.org/html/2505.03473v1",
        year: 2025,
        format: "JSONL",
        size_hint: "Long-tail historical entities",
        notes: "v0.1; tests EL on niche historical entities; analyzes LLM behavior on rare entities",
        splits: ["test"],
        tasks: ["el", "entity_linking"],
        categories: [entity_linking, historical, adversarial],
    },
    SNOMEDChallenge {
        name: "SNOMED CT EL Challenge",
        description: "Clinical entity linking to SNOMED CT. From SNOMED International 2024 challenge.",
        url: "",
        entity_types: ["CLINICAL_CONCEPT"],
        language: "en",
        domain: "clinical",
        license: "Research",
        citation: "SNOMED International (2024)",
        paper_url: "https://www.snomed.org/news/snomed-international-announces-entity-linking-challenge-winners",
        year: 2024,
        format: "Custom",
        size_hint: "Clinical notes, SNOMED CT linked",
        notes: "2024 challenge dataset; SNOMED CT coded clinical text; benchmarks clinical EL systems",
        splits: ["train", "test"],
        tasks: ["el", "entity_linking"],
        access_status: Registration,
        categories: [entity_linking, clinical, biomedical],
    },
    ESCOSkillsEL {
        name: "ESCO Skills EL",
        description: "Entity linking for occupational skills to ESCO taxonomy. Job market domain, multilingual.",
        url: "",
        entity_types: ["SKILL"],
        language: "mul",
        domain: "general",
        license: "Research",
        citation: "EACL Findings (2024)",
        paper_url: "https://aclanthology.org/2024.findings-eacl.28/",
        year: 2024,
        format: "Custom",
        size_hint: "Skill mentions across multiple languages",
        notes: "Complements MELO; links skills (not occupations) to ESCO taxonomy; job posting text",
        splits: ["train", "test"],
        tasks: ["el", "entity_linking"],
        access_status: ContactAuthors,
        categories: [entity_linking, multilingual],
    },

    // =========================================================================
    // Bioacoustics / Non-Human Communication
    // =========================================================================
    NatureLMAudio {
        name: "NatureLM-audio",
        description: "Foundation model training collection for bioacoustics. Multi-species audio-text pairs.",
        url: "https://github.com/earthspecies/naturelm-audio",
        entity_types: ["SPECIES", "CALL_TYPE", "BEHAVIOR"],
        language: "en",
        domain: "bioacoustics",
        license: "Research",
        citation: "NatureLM Team (2024)",
        paper_url: "https://arxiv.org/abs/2411.07186",
        year: 2024,
        format: "Custom",
        size_hint: "Multi-taxon audio-text pairs (birds, marine mammals, primates)",
        notes: "Bioacoustic foundation model data; paired audio-text descriptions; cross-taxa experiments",
        splits: ["train", "test"],
        tasks: ["classification", "captioning"],
        categories: [arcane_domain, multilingual],
    },
    BEANSZero {
        name: "BEANS-Zero",
        description: "Bioacoustics benchmark beyond species classification. Natural-language prompts for animal sounds.",
        url: "https://github.com/earthspecies/beans-zero",
        entity_types: ["SPECIES", "CALL_TYPE", "INDIVIDUAL"],
        language: "en",
        domain: "bioacoustics",
        license: "Research",
        citation: "NatureLM Team (2024)",
        paper_url: "https://arxiv.org/abs/2411.07186",
        year: 2024,
        format: "Custom",
        notes: "Zero-shot transfer to unseen taxa; captioning, retrieval, instruction-following on animal vocalizations",
        splits: ["test"],
        tasks: ["classification", "retrieval"],
        categories: [arcane_domain, adversarial],
    },

    // =========================================================================
    // Chemical / Materials Science NER
    // =========================================================================
    NLMChem {
        name: "NLM-Chem",
        description: "Chemical entity recognition and normalization. Full-text PMC articles with MeSH identifiers.",
        url: "https://ftp.ncbi.nlm.nih.gov/pub/lu/NLM-Chem/",
        entity_types: ["CHEMICAL", "DRUG"],
        language: "en",
        domain: "biomedical",
        license: "Public",
        citation: "Islamaj et al. (2021)",
        paper_url: "https://academic.oup.com/database/article/doi/10.1093/database/baac102/6858529",
        year: 2021,
        format: "BRAT",
        size_hint: "~150 full-text articles, ~38k annotations",
        notes: "Gold-standard chemical NER; normalized to MeSH; used for BioCreative VII",
        splits: ["train", "test"],
        tasks: ["ner", "el"],
        categories: [ner, biomedical, entity_linking],
    },
    CHEMDNER {
        name: "CHEMDNER",
        description: "Chemical compound and drug name recognition in scientific text.",
        url: "https://biocreative.bioinformatics.udel.edu/tasks/biocreative-iv/chemdner/",
        entity_types: ["CHEMICAL", "DRUG", "ABBREVIATION"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "Krallinger et al. (2015)",
        paper_url: "https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S2",
        year: 2015,
        format: "BIO",
        size_hint: "~10k abstracts",
        notes: "BioCreative IV shared task; abstract-level chemical NER; foundational chemistry benchmark",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner, biomedical],
    },

    // =========================================================================
    // Temporal / Event NER
    // =========================================================================
    TimeBankDense {
        name: "TimeBank-Dense",
        description: "Dense temporal relation annotation. Re-annotation of TimeBank with more consistent TLINK labels.",
        url: "https://github.com/bethard/timebank-dense",
        entity_types: ["EVENT", "TIMEX3"],
        language: "en",
        domain: "news",
        license: "Research",
        citation: "Chambers et al. (2014)",
        paper_url: "https://aclanthology.org/Q14-1002/",
        year: 2014,
        format: "TimeML",
        size_hint: "~36 documents, dense annotation",
        notes: "Event-event temporal relations; BEFORE/AFTER/INCLUDES/VAGUE; timeline construction benchmark",
        splits: ["train", "dev", "test"],
        tasks: ["temporal", "event_coref"],
        categories: [ner, event_coref],
    },

    // =========================================================================
    // Multimodal NER
    // =========================================================================
    TwitterGMNER {
        name: "Twitter-GMNER",
        description: "Grounded Multimodal NER. Entities linked to bounding boxes in social media images.",
        url: "https://github.com/JinYuanLi0012/RiVEG",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "social_media",
        license: "CC-BY-4.0",
        citation: "Li et al. (2024)",
        paper_url: "https://aclanthology.org/2024.findings-acl.58/",
        year: 2024,
        format: "JSONL",
        size_hint: "~8k tweets with images",
        notes: "Entity mentions grounded to image regions; visual-textual entity alignment",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "grounding"],
        categories: [ner, social_media, arcane_domain],
    },
    MNERMI {
        name: "MNER-MI",
        description: "Multimodal NER with Multiple Images. Social media posts with multiple image context.",
        url: "https://github.com/NUSTM/MNER-MI",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "social_media",
        license: "CC-BY-4.0",
        citation: "Wang et al. (2024)",
        paper_url: "https://aclanthology.org/2024.lrec-main.1001/",
        year: 2024,
        format: "JSONL",
        size_hint: "~5k tweets with multiple images",
        notes: "Multi-image context improves NER; temporal-prompt model baseline; LREC-COLING 2024",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner, social_media],
    },
    TwoMNER {
        name: "2M-NER",
        description: "Multilingual Multimodal NER. Four languages with text-image pairs.",
        url: "https://github.com/Alibaba-NLP/2M-NER",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "mul",
        domain: "social_media",
        license: "Apache-2.0",
        citation: "Liu et al. (2024)",
        paper_url: "https://arxiv.org/abs/2404.17122",
        year: 2024,
        format: "JSONL",
        size_hint: "~20k examples, 4 languages (EN, FR, DE, ES)",
        notes: "Contrastive text-image alignment; multilingual multimodal NER benchmark",
        splits: ["train", "dev", "test"],
        tasks: ["ner"],
        categories: [ner, multilingual, social_media],
    },

    // =========================================================================
    // Mathematical / Scientific NER
    // =========================================================================
    MathEntities {
        name: "Mathematical Entities",
        description: "Terminology and definition extraction from mathematical text. Category theory corpora.",
        url: "https://github.com/dmazzei/mathematical-entities",
        entity_types: ["TERM", "DEFINITION", "THEOREM"],
        language: "en",
        domain: "scientific",
        license: "CC-BY-4.0",
        citation: "Mazzei et al. (2024)",
        paper_url: "https://aclanthology.org/2024.lrec-main.966/",
        year: 2024,
        format: "LaTeX",
        size_hint: "~3 corpora in category theory",
        notes: "LaTeX source preservation; math-aware NER; entity linking to Wikidata/nLab",
        splits: ["train", "test"],
        tasks: ["ner", "el"],
        categories: [ner, arcane_domain, entity_linking],
    },
    SciERC {
        name: "SciERC",
        description: "Scientific information extraction from AI/ML papers. Nested entities and relations.",
        url: "https://nlp.cs.washington.edu/sciIE/",
        entity_types: ["TASK", "METHOD", "METRIC", "MATERIAL", "GENERIC", "OTHER"],
        language: "en",
        domain: "scientific",
        license: "CC-BY-4.0",
        citation: "Luan et al. (2018)",
        paper_url: "https://aclanthology.org/D18-1360/",
        year: 2018,
        format: "JSONL",
        size_hint: "~500 abstracts",
        notes: "Canonical scientific NER + relation extraction; nested entities common",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "re"],
        categories: [ner, nested_ner, relation_extraction, arcane_domain],
    },

    // =========================================================================
    // Geospatial / Toponym NER
    // =========================================================================
    GeoWebNews {
        name: "GeoWebNews",
        description: "Geoparsing benchmark from web news. Toponyms with geocoding coordinates.",
        url: "https://github.com/milangritta/GeoWebNews",
        entity_types: ["LOC", "GPE", "FACILITY"],
        language: "en",
        domain: "news",
        license: "CC-BY-4.0",
        citation: "Gritta et al. (2020)",
        paper_url: "https://aclanthology.org/2020.lrec-1.381/",
        year: 2020,
        format: "CoNLL",
        size_hint: "~4k documents",
        notes: "Toponym recognition + resolution; GeoNames linking; web news geoparsing",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "el"],
        categories: [ner, entity_linking],
    },
    LGL {
        name: "LGL",
        description: "Local-Global Lexicon for toponym disambiguation. News articles with geolocation.",
        url: "https://github.com/wikipedia2vec/wikipedia2vec",
        entity_types: ["LOC"],
        language: "en",
        domain: "news",
        license: "MIT",
        citation: "Lieberman et al. (2010)",
        year: 2010,
        format: "Custom",
        size_hint: "~5.8k place references",
        notes: "Toponym disambiguation benchmark; local vs global context for geolocation",
        splits: ["all"],
        tasks: ["ner", "el"],
        categories: [ner, entity_linking],
    },

    // =========================================================================
    // Procedural / Recipe NER
    // =========================================================================
    TASTEset {
        name: "TASTEset",
        description: "Recipe ingredient NER. 700 annotated recipe ingredient lists with 9 entity classes.",
        url: "https://github.com/taisti/TASTEset",
        entity_types: ["INGREDIENT", "QUANTITY", "UNIT", "STATE", "SIZE", "TEMP"],
        language: "en",
        domain: "food",
        license: "CC-BY-4.0",
        citation: "TASTEset Team (2023)",
        year: 2023,
        format: "BIO",
        size_hint: "~700 ingredient lists",
        notes: "Recipe NER benchmark; BIO/BILOU conversion utilities; BERT model pipeline",
        splits: ["train", "test"],
        tasks: ["ner"],
        categories: [ner, arcane_domain],
    },
    RecipeNER {
        name: "Recipe NER",
        description: "Deep learning recipe NER. Multi-scale datasets with ingredient and instruction entities.",
        url: "https://github.com/cosylabiiit/recipe-ner",
        entity_types: ["INGREDIENT", "QUANTITY", "UNIT", "PROCESS", "UTENSIL", "TEMP"],
        language: "en",
        domain: "food",
        license: "MIT",
        citation: "Deepgram (2024)",
        paper_url: "https://aclanthology.org/2024.lrec-main.406/",
        year: 2024,
        format: "BIO",
        size_hint: "~88k phrases (6.6k manual, 26k augmented, 88k machine)",
        notes: "Three-tier dataset; baseline pipelines exist (e.g., spaCy transformer); recipe IE pipeline",
        splits: ["train", "test"],
        tasks: ["ner"],
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Code / Software Entity Recognition
    // =========================================================================
    CodeSearchNet {
        name: "CodeSearchNet",
        description: "Code understanding benchmark. Function documentation and code search across 6 languages.",
        url: "https://github.com/github/CodeSearchNet",
        entity_types: ["FUNCTION", "CLASS", "VARIABLE", "MODULE"],
        language: "mul",
        domain: "code",
        license: "MIT",
        citation: "Husain et al. (2019)",
        paper_url: "https://arxiv.org/abs/1909.09436",
        year: 2019,
        format: "JSONL",
        size_hint: "~2M functions across 6 programming languages",
        notes: "Code-docstring pairs; Python, Java, Go, PHP, JavaScript, Ruby; foundation for code NER",
        splits: ["train", "dev", "test"],
        tasks: ["retrieval"],
        categories: [arcane_domain, multilingual],
    },

    // =========================================================================
    // Fictional / Literary Entity Recognition
    // =========================================================================
    FABLE {
        name: "FABLE",
        description: "Fiction Adapted BERT for Literary Entities. DeBERTa-based NER for narrative fiction.",
        url: "https://huggingface.co/DeBERTa-literary-entities",
        entity_types: ["CHARACTER", "LOCATION", "ORGANIZATION", "ARTIFACT"],
        language: "en",
        domain: "fiction",
        license: "MIT",
        citation: "FABLE Team (2024)",
        year: 2024,
        format: "Custom",
        notes: "Literary NER model; targets invented names in fantasy/SF; trained on narrative fiction",
        categories: [ner, literary],
    },
    ELGold {
        name: "ELGold",
        description: "Gold-standard multi-genre Polish NER+EL. Includes fiction, press, blogs.",
        url: "https://mostwiedzy.pl/en/open-research-data/elgold-gold-standard-multi-genre-dataset",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "pl",
        domain: "general",
        license: "CC-BY-4.0",
        citation: "Pokrywka et al. (2025)",
        paper_url: "https://www.nature.com/articles/s41597-025-05274-4",
        year: 2025,
        format: "JSONL",
        notes: "Multi-genre including fiction; Wikipedia-linked; Polish language",
        splits: ["train", "test"],
        tasks: ["ner", "el"],
        categories: [ner, entity_linking, literary, multilingual],
    },

    // =========================================================================
    // Streaming / Temporal Entity Evolution
    // =========================================================================
    StreamingCDCoref {
        name: "Streaming CD-Coref",
        description: "Streaming cross-document entity coreference protocol. News domain streaming evaluation.",
        url: "https://www.cs.jhu.edu/~mdredze/publications/streaming_coref_coling.pdf",
        entity_types: ["PER", "ORG", "LOC"],
        language: "en",
        domain: "news",
        license: "Research",
        citation: "Dredze et al. (2010)",
        paper_url: "https://aclanthology.org/C10-1032/",
        year: 2010,
        format: "Custom",
        notes: "Canonical streaming entity clustering; O(n) single-pass; evolving cluster representations",
        categories: [coref, long_document],
    },
    TemDocRED {
        name: "Tem-DocRED",
        description: "Temporal document-level relation extraction. Converts static triples to temporal quadruples.",
        url: "https://github.com/THUDM/Tem-DocRED",
        entity_types: ["PER", "ORG", "LOC", "TIME"],
        language: "en",
        domain: "wikipedia",
        license: "MIT",
        citation: "Zhang et al. (2024)",
        paper_url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC12048500/",
        year: 2024,
        format: "JSONL",
        size_hint: "Re-DocRED + temporal timestamps",
        notes: "Temporal KG construction from documents; LLM + pattern mining for timestamp inference",
        splits: ["train", "dev", "test"],
        tasks: ["re", "temporal"],
        categories: [relation_extraction, long_document],
    },

    // =========================================================================
    // Scientific / Concept Coreference
    // =========================================================================
    SciCoRadar {
        name: "SciCo-Radar",
        description: "Scientific cross-document concept coreference. Dynamic definitions via LLM retrieval.",
        url: "https://github.com/allenai/scico-radar",
        entity_types: ["CONCEPT", "METHOD", "TASK", "MATERIAL"],
        language: "en",
        domain: "scientific",
        license: "Apache-2.0",
        citation: "Wadden et al. (2024)",
        paper_url: "https://arxiv.org/abs/2409.15113",
        year: 2024,
        format: "JSONL",
        notes: "Cross-doc concept coref with hierarchy; LLM-generated relational definitions improve F1",
        splits: ["train", "dev", "test"],
        tasks: ["coref"],
        categories: [coref, arcane_domain],
    },

    // =========================================================================
    // Process Mining / Event KGs
    // =========================================================================
    EventKGDrift {
        name: "Event KG Drift",
        description: "Multi-perspective concept drift detection on event knowledge graphs.",
        url: "https://research.tue.nl/files/349781334/978-3-031-61057-8_9.pdf",
        entity_types: ["EVENT", "CASE", "ACTOR", "TIME"],
        language: "en",
        domain: "evaluation",
        license: "Research",
        citation: "TU Eindhoven (2024)",
        year: 2024,
        format: "Custom",
        notes: "Actor-centric features give 2.6x stronger drift signals; temporal graph drift on EKGs",
        categories: [event_coref, long_document, arcane_domain],
    },

    // =========================================================================
    // Semantic Drift / KG Curation
    // =========================================================================
    WikidataDrift {
        name: "Wikidata Semantic Drift",
        description: "Semantic drift detection in Wikidata. LLM-based classification inconsistency detection.",
        url: "https://arxiv.org/abs/2511.04926",
        entity_types: [],
        language: "mul",
        domain: "encyclopedia",
        license: "CC0-1.0",
        citation: "Wikidata Drift Team (2024)",
        paper_url: "https://arxiv.org/abs/2511.04926",
        year: 2024,
        format: "Custom",
        notes: "Multi-dimensional semantic risk model; drift threshold ~0.6; continuous KG curation",
        categories: [entity_linking, adversarial],
    },

    // =========================================================================
    // Additional Legacy Datasets (from loader.rs)
    // =========================================================================

    AIDA {
        name: "AIDA-CoNLL (v2)",
        description: "Entity linking to Wikipedia. CoNLL-YAGO dataset for named entity disambiguation.",
        url: "https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "news",
        license: "Research",
        citation: "Hoffart et al. (2011)",
        paper_url: "https://aclanthology.org/D11-1072/",
        year: 2011,
        format: "CoNLL",
        notes: "Entity linking benchmark; links CoNLL-2003 mentions to YAGO/Wikipedia",
        categories: [entity_linking],
    },

    // AIONER: Correct GitHub org is 'ncbi', not 'AIONER'
    AIONER {
        name: "AIONER",
        description: "All-in-one biomedical NER. Unified biomedical entity extraction model.",
        url: "https://github.com/ncbi/AIONER",
        entity_types: ["Gene", "Disease", "Chemical", "Species"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "Luo et al. (2023)",
        year: 2023,
        format: "JSONL",
        notes: "Unified model for multiple biomedical entity types",
        categories: [ner, biomedical],
    },

    AISHELLNER {
        name: "AISHELL-NER",
        description: "Chinese speech NER from AISHELL corpus. Named entities in Mandarin speech.",
        url: "https://www.aishelltech.com/aishell_2",
        entity_types: ["PER", "LOC", "ORG"],
        language: "zh",
        domain: "speech",
        license: "Research",
        citation: "AISHELL Foundation (2017)",
        year: 2017,
        format: "Custom",
        notes: "Speech transcription NER; tests robustness to ASR errors",
        categories: [ner, speech],
    },

    AstroNER {
        name: "AstroNER",
        description: "Astronomy named entity recognition. Celestial objects and astronomical concepts.",
        url: "https://github.com/astronomical-ner/AstroNER",
        entity_types: ["CelestialObject", "Instrument", "Mission", "Phenomenon"],
        language: "en",
        domain: "astrophysics",
        license: "CC-BY-4.0",
        citation: "NASA ADS Team",
        year: 2022,
        format: "CoNLL",
        notes: "Domain-specific NER for astronomy literature",
        categories: [ner, arcane_domain],
    },

    B2NERD {
        name: "B2NERD",
        description: "Billion-scale news NER dataset. Large-scale distantly supervised NER.",
        url: "https://huggingface.co/datasets/Umean/B2NERD",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "news",
        license: "CC-BY-4.0",
        citation: "Umean (2023)",
        year: 2023,
        format: "JSONL",
        notes: "Large-scale silver-standard NER; useful for pre-training",
        categories: [ner],
    },

    BioMNER {
        name: "BioMNER",
        description: "Biomedical method NER. Scientific methods and techniques in biomedical text.",
        url: "https://huggingface.co/datasets/tner/bionlp2004",
        entity_types: ["Method", "Technique", "Protocol"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "BioNLP (2004)",
        year: 2004,
        format: "BIO",
        notes: "Biomedical methodology extraction; from BioNLP shared task",
        categories: [ner, biomedical],
    },

    LegNER {
        name: "LegNER",
        description: "Legal domain NER. Named entities in legal documents and court opinions.",
        url: "https://github.com/Liquid-Legal-Institute/LegalBench",
        entity_types: ["Court", "Judge", "Statute", "Party", "Date"],
        language: "en",
        domain: "legal",
        license: "CC-BY-4.0",
        citation: "Legal NLP Team (2021)",
        year: 2021,
        format: "CoNLL",
        notes: "Legal domain specialization; court documents and statutes",
        categories: [ner],
    },

    OpenNER {
        name: "OpenNER 1.0",
        description: "Open domain NER benchmark. Broad coverage across multiple domains.",
        url: "https://huggingface.co/datasets/yongsun-yoon/open-ner-english",
        entity_types: ["PER", "LOC", "ORG", "EVENT", "PRODUCT"],
        language: "en",
        domain: "mixed",
        license: "CC-BY-SA-4.0",
        citation: "Babelscape (2023)",
        year: 2023,
        format: "JSONL",
        notes: "Community mirror; open-domain NER benchmark",
        tasks: ["ner"],
        hf_id: "yongsun-yoon/open-ner-english",
        access_status: HuggingFace,
        categories: [ner],
    },

    SciNER {
        name: "SciNER",
        description: "Scientific literature NER. Entities from scientific papers across disciplines.",
        url: "https://github.com/allenai/sciner",
        entity_types: ["Method", "Task", "Dataset", "Metric", "Material"],
        language: "en",
        domain: "scientific",
        license: "Apache-2.0",
        citation: "Allen AI (2022)",
        year: 2022,
        format: "JSONL",
        notes: "Scientific entities; paper abstracts and methods sections",
        categories: [ner],
    },

    FinanceNER {
        name: "FinanceNER",
        description: "Financial domain NER. Named entities from financial documents and news.",
        url: "https://github.com/nlpaueb/finer",
        entity_types: ["Company", "Stock", "Currency", "Amount", "Date"],
        language: "en",
        domain: "financial",
        license: "Research",
        citation: "FinNLP (2020)",
        year: 2020,
        format: "CoNLL",
        notes: "Financial entity extraction; SEC filings and news",
        categories: [ner],
    },

    TechNER {
        name: "TechNER",
        description: "Technology domain NER. Software, hardware, and technical entities.",
        url: "https://github.com/techner/techner",
        entity_types: ["Software", "Hardware", "Company", "Version", "Language"],
        language: "en",
        domain: "code",
        license: "MIT",
        citation: "TechNER Team (2021)",
        year: 2021,
        format: "CoNLL",
        notes: "Technology entities; Stack Overflow and documentation",
        categories: [ner],
    },

    FictionNER750M {
        name: "FictionNER-750M",
        description: "Fiction NER at scale. Named entities from 750M tokens of fiction text.",
        url: "https://huggingface.co/datasets/SaladTechnologies/fiction-ner-750m",
        entity_types: ["Character", "Location", "Object", "Organization"],
        language: "en",
        domain: "fiction",
        license: "CC-BY-4.0",
        citation: "Fiction NER Team (2023)",
        year: 2023,
        format: "JSONL",
        notes: "Large-scale fiction NER; public on HuggingFace",
        tasks: ["ner"],
        hf_id: "SaladTechnologies/fiction-ner-750m",
        access_status: HuggingFace,
        categories: [ner, literary],
    },

    CharacterCodex {
        name: "Character Codex",
        description: "Character entity recognition in fiction. Literary character identification.",
        url: "https://github.com/character-codex/character-codex",
        entity_types: ["Character", "Alias", "Role"],
        language: "en",
        domain: "fiction",
        license: "CC-BY-4.0",
        citation: "Character Codex Team (2022)",
        year: 2022,
        format: "JSONL",
        notes: "Character tracking across narrative; aliases and roles",
        categories: [ner, literary],
    },

    MUC6 {
        name: "MUC-6",
        description: "Message Understanding Conference 6. Seminal NER and coreference dataset.",
        url: "https://catalog.ldc.upenn.edu/LDC2003T13",
        entity_types: ["ENAMEX", "TIMEX", "NUMEX"],
        language: "en",
        domain: "news",
        license: "LDC",
        citation: "Grishman & Sundheim (1996)",
        paper_url: "https://aclanthology.org/C96-1079/",
        year: 1996,
        format: "SGML",
        notes: "Historically significant; established NER evaluation paradigm",
        categories: [ner, historical],
    },

    MUC7 {
        name: "MUC-7",
        description: "Message Understanding Conference 7. Expanded NE types from MUC-6.",
        url: "https://catalog.ldc.upenn.edu/LDC2001T02",
        entity_types: ["ENAMEX", "TIMEX", "NUMEX"],
        language: "en",
        domain: "news",
        license: "LDC",
        citation: "Chinchor (1998)",
        paper_url: "https://aclanthology.org/M98-1002/",
        year: 1998,
        format: "SGML",
        notes: "Refined MUC-6 guidelines; includes satellite launch texts",
        categories: [ner, historical],
    },

    OntoNotes50 {
        name: "OntoNotes 5.0",
        description: "OntoNotes Release 5.0. Multi-genre corpus with NER, coref, and more.",
        url: "https://catalog.ldc.upenn.edu/LDC2013T19",
        entity_types: ["PER", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"],
        language: "en",
        domain: "mixed",
        license: "LDC",
        citation: "Weischedel et al. (2013)",
        paper_url: "https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf",
        year: 2013,
        format: "CoNLL",
        annotation_scheme: "CoNLLCoref",
        size_hint: "~2.9M words across genres",
        notes: "Gold standard for multiple NLP tasks; WSJ, broadcast, web, telephone",
        categories: [ner, coref],
    },

    GUM {
        name: "GUM",
        description: "Georgetown University Multilayer corpus. Rich annotation across 12 genres.",
        url: "https://github.com/amir-zeldes/gum",
        entity_types: ["person", "place", "organization", "time", "event"],
        language: "en",
        domain: "mixed",
        license: "CC-BY-4.0",
        citation: "Zeldes (2017)",
        paper_url: "https://aclanthology.org/W17-0809/",
        year: 2017,
        format: "CoNLL",
        size_hint: "~200k tokens, 12 genres",
        notes: "Multi-layer annotation; coreference, RST, entities",
        categories: [ner, coref],
    },

    TACKBP {
        name: "TAC-KBP",
        description: "TAC Knowledge Base Population. Entity linking and slot filling benchmark.",
        url: "https://tac.nist.gov/",
        entity_types: ["PER", "ORG", "GPE"],
        language: "en",
        domain: "news",
        license: "LDC",
        citation: "Ji et al. (2010)",
        paper_url: "https://aclanthology.org/C10-1058/",
        year: 2010,
        format: "Custom",
        notes: "Entity linking to Wikipedia/KB; slot filling for attributes",
        categories: [entity_linking],
    },

    HAREM {
        name: "HAREM",
        description: "Portuguese NER evaluation. First and Second HAREM conferences.",
        url: "https://www.linguateca.pt/HAREM/",
        entity_types: ["PESSOA", "LOCAL", "ORGANIZACAO", "TEMPO", "VALOR"],
        language: "pt",
        domain: "news",
        license: "Research",
        citation: "Santos et al. (2006)",
        paper_url: "https://www.linguateca.pt/HAREM/",
        year: 2006,
        format: "SGML",
        notes: "Portuguese NER benchmark; morphologically rich language",
        categories: [ner, multilingual],
    },

    GunViolenceCorpus {
        name: "Gun Violence Corpus (v2)",
        description: "Gun violence event extraction. Named entities and events from news.",
        url: "https://github.com/gun-violence-corpus/gvc",
        entity_types: ["Shooter", "Victim", "Weapon", "Location", "Date"],
        language: "en",
        domain: "news",
        license: "CC-BY-4.0",
        citation: "Pavlick et al. (2016)",
        year: 2016,
        format: "Custom",
        notes: "Event extraction; sensitive domain requiring careful handling",
        categories: [ner, event_coref],
    },

    // =========================================================================
    // Event Extraction (Trigger + Argument)
    // =========================================================================
    // Note: anno has event extraction capability via EventExtractor (lexicon-based + GLiNER)
    // These datasets support the Task::EventExtraction capability

    MAVEN {
        name: "MAVEN",
        description: "Massive general-domain event detection. 168 event types from Wikipedia, 4x larger than ACE.",
        url: "https://github.com/THU-KEG/MAVEN-dataset",
        entity_types: ["EVENT_TRIGGER"],  // 168 fine-grained event types
        language: "en",
        domain: "wikipedia",
        license: "MIT",
        citation: "Wang et al. (2020)",
        paper_url: "https://aclanthology.org/2020.emnlp-main.129/",
        year: 2020,
        format: "JSONL",
        annotation_scheme: "Trigger-based",
        size_hint: "~118k trigger instances, 4,480 documents, 168 event types",
        notes: "EMNLP 2020; largest general-domain ED dataset; CodaLab leaderboard available; Tsinghua Cloud/Google Drive download",
        splits: ["train", "valid", "test"],
        tasks: ["event_extraction"],
        access_status: Public,
        categories: [ner],  // Event triggers are span-based like NER
    },

    MAVENArg {
        name: "MAVEN-ARG",
        description: "MAVEN extended with event arguments and relations. Complete event extraction benchmark.",
        url: "https://github.com/THU-KEG/MAVEN-Argument",
        entity_types: ["EVENT_TRIGGER", "EVENT_ARGUMENT", "EVENT_RELATION"],
        language: "en",
        domain: "wikipedia",
        license: "MIT",
        citation: "Wang et al. (2024)",
        paper_url: "https://aclanthology.org/2024.acl-long.224/",
        year: 2024,
        format: "JSONL",
        annotation_scheme: "Trigger-Argument",
        size_hint: "~98k argument annotations, ~21k relations",
        notes: "ACL 2024; builds on MAVEN; supports ED + EAE + ERE tasks; all-in-one event understanding",
        splits: ["train", "valid", "test"],
        tasks: ["event_extraction", "relation_extraction"],
        access_status: Public,
        categories: [ner, relation_extraction],
    },

    CASIE {
        name: "CASIE",
        description: "Cybersecurity event extraction. Attack patterns, vulnerabilities, malware events.",
        url: "https://github.com/Ebiquity/CASIE",
        entity_types: ["Attack-Pattern", "Vulnerability", "Data-Breach", "Malware", "Patch"],
        language: "en",
        domain: "cybersecurity",
        license: "CC-BY-4.0",
        citation: "Satyapanich et al. (2020)",
        paper_url: "https://aclanthology.org/2020.case-1.12/",
        year: 2020,
        format: "Standoff",
        size_hint: "~1k documents, 5 event types, 26 argument roles",
        notes: "Domain-specific event extraction; cybersecurity news articles",
        splits: ["train", "dev", "test"],
        tasks: ["event_extraction", "ner"],
        access_status: Public,
        categories: [ner, arcane_domain],
    },

    RAMS {
        name: "RAMS",
        description: "Roles Across Multiple Sentences. Cross-sentence event argument extraction with 139 event types.",
        url: "https://nlp.jhu.edu/rams/",
        entity_types: ["EVENT_TRIGGER", "EVENT_ARGUMENT"],
        language: "en",
        domain: "news",
        license: "Research",
        citation: "Ebner et al. (2020)",
        paper_url: "https://aclanthology.org/2020.acl-main.718/",
        year: 2020,
        format: "JSONL",
        size_hint: "~9,124 event instances, 139 event types",
        notes: "ACL 2020; tests implicit/cross-sentence arguments; requires multi-sentence reasoning",
        splits: ["train", "dev", "test"],
        tasks: ["event_extraction"],
        access_status: Public,
        categories: [ner, long_document],
    },

    SLUE {
        name: "SLUE",
        description: "Spoken Language Understanding Evaluation. NER in speech transcripts.",
        url: "https://github.com/asappresearch/slue-toolkit",
        entity_types: ["PER", "LOC", "ORG", "DATE"],
        language: "en",
        domain: "speech",
        license: "MIT",
        citation: "Shon et al. (2022)",
        paper_url: "https://aclanthology.org/2022.naacl-main.137/",
        year: 2022,
        format: "JSONL",
        notes: "End-to-end speech NER; VoxPopuli and VoxCeleb sources",
        categories: [ner, speech],
    },

    CRAFTCoref {
        name: "CRAFT Coreference",
        description: "Colorado Richly Annotated Full-Text corpus coreference. Biomedical coref.",
        url: "https://github.com/UCDenver-ccp/CRAFT",
        entity_types: ["Gene", "Protein", "Cell", "Organism"],
        language: "en",
        domain: "biomedical",
        license: "CC-BY-4.0",
        citation: "Cohen et al. (2017)",
        paper_url: "https://academic.oup.com/database/article/doi/10.1093/database/bax087/4621360",
        year: 2017,
        format: "Standoff",
        notes: "Full-text biomedical articles; coreference including bridging",
        categories: [coref, biomedical],
    },

    FootballCorefCorpus {
        name: "Football Coreference Corpus (v2)",
        description: "Cross-document event coreference for football matches.",
        url: "https://github.com/cltl/FCC",
        entity_types: ["Event", "Team", "Player", "Location"],
        language: "en",
        domain: "sports",
        license: "CC-BY-4.0",
        citation: "Vossen et al. (2018)",
        year: 2018,
        format: "Custom",
        notes: "Cross-document event coreference; sports domain",
        categories: [event_coref],
    },

    MultipartyDialogueCoref {
        name: "Multiparty Dialogue Coreference",
        description: "Coreference in multi-party conversations. Meeting and chat transcripts.",
        url: "https://github.com/sopan-sarkar/multiparty-dialogue-coref",
        entity_types: ["PER", "ORG", "LOC"],
        language: "en",
        domain: "dialogue",
        license: "CC-BY-4.0",
        citation: "Sarkar et al. (2020)",
        year: 2020,
        format: "JSONL",
        notes: "Multi-party setting; speaker identification challenges",
        categories: [coref, dialogue],
    },

    CODICRAC {
        name: "CODI-CRAC",
        description: "CODI/CRAC shared task on anaphora and coreference. Multiple languages.",
        url: "https://github.com/UniversalAnaphora/UA-CODI-CRAC",
        entity_types: ["PER", "ORG", "LOC", "Event"],
        language: "mul",
        domain: "mixed",
        license: "CC-BY-4.0",
        citation: "CODI-CRAC Team (2022)",
        paper_url: "https://aclanthology.org/2022.codi-1.0/",
        year: 2022,
        format: "CoNLL",
        annotation_scheme: "CoNLLCoref",
        notes: "Shared task data; includes bridging and discourse deixis",
        categories: [coref, multilingual],
    },

    MixRED {
        name: "MixRED",
        description: "Mixed relation extraction dataset. Multiple relation types and domains.",
        url: "https://github.com/mixred/MixRED",
        entity_types: ["PER", "ORG", "LOC"],
        language: "en",
        domain: "mixed",
        license: "CC-BY-4.0",
        citation: "MixRED Team (2022)",
        year: 2022,
        format: "JSONL",
        notes: "Relation extraction across multiple domains",
        categories: [relation_extraction],
    },

    CovEReD {
        name: "CovEReD",
        description: "COVID-19 relation extraction dataset. Biomedical relations from pandemic literature.",
        url: "https://github.com/covered/CovEReD",
        entity_types: ["Drug", "Disease", "Gene", "Symptom"],
        language: "en",
        domain: "biomedical",
        license: "CC-BY-4.0",
        citation: "CovEReD Team (2021)",
        year: 2021,
        format: "JSONL",
        notes: "COVID-19 specific; drug-disease-gene relations",
        categories: [relation_extraction, biomedical],
    },

    SciER {
        name: "SciER",
        description: "Scientific entity and relation extraction. From AI/ML papers.",
        url: "https://github.com/allenai/sciie",
        entity_types: ["Task", "Method", "Metric", "Material", "Generic"],
        language: "en",
        domain: "scientific",
        license: "Apache-2.0",
        citation: "Luan et al. (2018)",
        paper_url: "https://aclanthology.org/D18-1360/",
        year: 2018,
        format: "JSONL",
        notes: "Scientific IE; paper abstracts with nested entities",
        categories: [ner, relation_extraction, nested_ner],
    },

    WEBNLG {
        name: "WebNLG",
        description: "Web NLG Challenge dataset. RDF-to-text generation with entity-relation triples.",
        url: "https://gitlab.com/webnlg/challenge-2017",
        entity_types: ["Entity"],
        language: "en",
        domain: "wikipedia",
        license: "CC-BY-4.0",
        citation: "Gardent et al. (2017)",
        paper_url: "https://aclanthology.org/W17-3518/",
        year: 2017,
        format: "XML",
        notes: "RDF triples to natural language; 15 DBpedia categories",
        categories: [relation_extraction],
    },

    // =========================================================================
    // Ancient/Historical Language Treebanks
    // =========================================================================

    AkkadianUD {
        name: "Akkadian UD",
        description: "Universal Dependencies for Akkadian. Cuneiform texts from ancient Mesopotamia.",
        url: "https://universaldependencies.org/treebanks/akk_pisandub/index.html",
        entity_types: ["PER", "LOC", "GPE"],
        language: "akk",
        domain: "historical",
        license: "CC-BY-SA-4.0",
        citation: "UD Akkadian Team",
        year: 2020,
        format: "CoNLLU",
        notes: "Cuneiform script; extinct Semitic language",
        categories: [historical, ancient],
    },

    AncientHebrewUD {
        name: "Ancient Hebrew UD",
        description: "Universal Dependencies for Biblical Hebrew. Hebrew Bible text.",
        url: "https://universaldependencies.org/treebanks/hbo_ptnk/index.html",
        entity_types: ["PER", "LOC", "GPE"],
        language: "hbo",
        domain: "religious",
        license: "CC-BY-SA-4.0",
        citation: "UD Hebrew Team",
        year: 2019,
        format: "CoNLLU",
        notes: "Biblical Hebrew; Torah and Prophets",
        categories: [historical, ancient],
    },

    ClassicalChineseUD {
        name: "Classical Chinese UD",
        description: "Universal Dependencies for Classical/Literary Chinese. Pre-modern texts.",
        url: "https://universaldependencies.org/treebanks/lzh_kyoto/index.html",
        entity_types: ["PER", "LOC", "GPE"],
        language: "lzh",
        domain: "historical",
        license: "CC-BY-SA-4.0",
        citation: "UD Classical Chinese Team",
        year: 2018,
        format: "CoNLLU",
        notes: "Literary Chinese; classical texts and commentaries",
        categories: [historical, ancient],
    },

    CopticUD {
        name: "Coptic UD",
        description: "Universal Dependencies for Coptic. Late Egyptian language.",
        url: "https://universaldependencies.org/treebanks/cop_scriptorium/index.html",
        entity_types: ["PER", "LOC", "GPE"],
        language: "cop",
        domain: "religious",
        license: "CC-BY-SA-4.0",
        citation: "Zeldes & Schroeder (2016)",
        year: 2016,
        format: "CoNLLU",
        notes: "Coptic; Gnostic and Biblical texts",
        categories: [historical, ancient],
    },

    GothicUD {
        name: "Gothic UD",
        description: "Universal Dependencies for Gothic. Wulfila's Bible translation.",
        url: "https://universaldependencies.org/treebanks/got_proiel/index.html",
        entity_types: ["PER", "LOC", "GPE"],
        language: "got",
        domain: "religious",
        license: "CC-BY-NC-SA-4.0",
        citation: "PROIEL Team",
        year: 2014,
        format: "CoNLLU",
        notes: "Gothic; oldest substantial Germanic text",
        categories: [historical, ancient],
    },

    HittiteUD {
        name: "Hittite UD",
        description: "Universal Dependencies for Hittite. Ancient Anatolian language.",
        url: "https://universaldependencies.org/treebanks/hit_hittb/index.html",
        entity_types: ["PER", "LOC", "GPE"],
        language: "hit",
        domain: "historical",
        license: "CC-BY-SA-4.0",
        citation: "UD Hittite Team",
        year: 2021,
        format: "CoNLLU",
        notes: "Cuneiform Hittite; Bronze Age Anatolia",
        categories: [historical, ancient],
    },

    OldChurchSlavonicUD {
        name: "Old Church Slavonic UD",
        description: "Universal Dependencies for OCS. Medieval Slavic liturgical language.",
        url: "https://universaldependencies.org/treebanks/cu_proiel/index.html",
        entity_types: ["PER", "LOC", "GPE"],
        language: "cu",
        domain: "religious",
        license: "CC-BY-NC-SA-4.0",
        citation: "PROIEL Team",
        year: 2014,
        format: "CoNLLU",
        notes: "Oldest Slavic literary language; Cyrillic/Glagolitic",
        categories: [historical, ancient],
    },

    LatinITTB {
        name: "Latin ITTB",
        description: "Index Thomisticus Treebank. Medieval Latin theological texts.",
        url: "https://universaldependencies.org/treebanks/la_ittb/index.html",
        entity_types: ["PER", "LOC", "ORG"],
        language: "la",
        domain: "religious",
        license: "CC-BY-NC-SA-3.0",
        citation: "McGillivray et al. (2009)",
        year: 2009,
        format: "CoNLLU",
        notes: "Aquinas texts; medieval scholastic Latin",
        categories: [historical],
    },

    LatinPROIEL {
        name: "Latin PROIEL",
        description: "Pragmatic Resources in Old Indo-European Languages. Classical Latin.",
        url: "https://universaldependencies.org/treebanks/la_proiel/index.html",
        entity_types: ["PER", "LOC", "GPE"],
        language: "la",
        domain: "historical",
        license: "CC-BY-NC-SA-4.0",
        citation: "PROIEL Team",
        year: 2014,
        format: "CoNLLU",
        notes: "Vulgate, Caesar, Cicero; classical and late Latin",
        categories: [historical],
    },

    EsperantoUD {
        name: "Esperanto UD",
        description: "Universal Dependencies for Esperanto. Planned international language.",
        url: "https://universaldependencies.org/treebanks/eo_pud/index.html",
        entity_types: ["PER", "LOC", "ORG"],
        language: "eo",
        domain: "constructed_language",
        license: "CC-BY-SA-4.0",
        citation: "UD Esperanto Team",
        year: 2017,
        format: "CoNLLU",
        notes: "Constructed language; regular agglutinative morphology",
        categories: [constructed],
    },

    // =========================================================================
    // Constructed/Fictional Languages
    // =========================================================================

    Dothraki {
        name: "Dothraki",
        description: "Dothraki language corpus. Game of Thrones constructed language.",
        url: "https://wiki.dothraki.org/",
        entity_types: ["PER", "LOC"],
        language: "dlk",
        domain: "fiction",
        license: "CC-BY-SA-4.0",
        citation: "Peterson (2011)",
        year: 2011,
        format: "Custom",
        notes: "Conlang by David Peterson; SVO word order",
        categories: [constructed],
    },

    HighValyrian {
        name: "High Valyrian",
        description: "High Valyrian corpus. Game of Thrones constructed language.",
        url: "https://wiki.dothraki.org/High_Valyrian",
        entity_types: ["PER", "LOC"],
        language: "hvy",
        domain: "fiction",
        license: "CC-BY-SA-4.0",
        citation: "Peterson (2013)",
        year: 2013,
        format: "Custom",
        notes: "Highly inflected conlang; 4 genders, 8 cases",
        categories: [constructed],
    },

    Klingon {
        name: "Klingon",
        description: "Klingon language corpus. Star Trek constructed language.",
        url: "https://github.com/klingonlanguage/klingon-data",
        entity_types: ["PER", "LOC", "ORG"],
        language: "tlh",
        domain: "fiction",
        license: "Research",
        citation: "Okrand (1985)",
        year: 1985,
        format: "Custom",
        notes: "OVS word order; unique phonology; active community",
        categories: [constructed],
    },

    Quenya {
        name: "Quenya",
        description: "Quenya language corpus. Tolkien's Elvish language.",
        url: "https://eldamo.org/",
        entity_types: ["PER", "LOC"],
        language: "qya",
        domain: "fiction",
        license: "CC-BY-4.0",
        citation: "Tolkien (1954)",
        year: 1954,
        format: "Custom",
        notes: "Finnish-inspired phonology; Tengwar script",
        categories: [constructed],
    },

    Navi {
        name: "Na'vi",
        description: "Na'vi language corpus. Avatar constructed language.",
        url: "https://learnnavi.org/",
        entity_types: ["PER", "LOC"],
        language: "nav",
        domain: "fiction",
        license: "Research",
        citation: "Frommer (2009)",
        year: 2009,
        format: "Custom",
        notes: "Free word order; ejectives; infixes",
        categories: [constructed],
    },

    InterslavicCorpus {
        name: "Interslavic",
        description: "Interslavic zonal auxiliary language. Constructed for Slavic intelligibility.",
        url: "https://interslavic.fun/",
        entity_types: ["PER", "LOC", "ORG"],
        language: "isv",
        domain: "constructed_language",
        license: "CC-BY-SA-4.0",
        citation: "Interslavic Team (2006)",
        year: 2006,
        format: "Custom",
        notes: "Maximizes mutual intelligibility across Slavic languages",
        categories: [constructed],
    },

    Lojban {
        name: "Lojban",
        description: "Lojban logical language corpus. Constructed for unambiguous communication.",
        url: "https://mw.lojban.org/",
        entity_types: [],
        language: "jbo",
        domain: "constructed_language",
        license: "Public Domain",
        citation: "Cowan (1997)",
        year: 1997,
        format: "Custom",
        notes: "Predicate logic-based; completely unambiguous grammar",
        categories: [constructed],
    },

    TokiPona {
        name: "Toki Pona",
        description: "Toki Pona minimalist language corpus. 120-word philosophical language.",
        url: "https://github.com/kilipan/toki-pona-corpus",
        entity_types: [],
        language: "tok",
        domain: "constructed_language",
        license: "CC-BY-SA-4.0",
        citation: "Lang (2001)",
        year: 2001,
        format: "Custom",
        notes: "Minimalist; tests compositional semantics",
        categories: [constructed],
    },

    // =========================================================================
    // Clinical/Medical Datasets
    // =========================================================================

    I2B22010 {
        name: "i2b2-2010",
        description: "i2b2/VA 2010 NLP Challenge. Clinical concept extraction and relations.",
        url: "https://www.i2b2.org/NLP/DataSets/",
        entity_types: ["Problem", "Treatment", "Test"],
        language: "en",
        domain: "clinical",
        license: "DUA Required",
        citation: "Uzuner et al. (2011)",
        paper_url: "https://academic.oup.com/jamia/article/18/5/552/830538",
        year: 2010,
        format: "Custom",
        notes: "Clinical notes; concept and relation extraction",
        categories: [ner, relation_extraction, clinical],
    },

    I2b2Deidentification {
        name: "i2b2 De-identification",
        description: "i2b2 2014 De-identification Challenge. PHI recognition and removal.",
        url: "https://www.i2b2.org/NLP/DataSets/",
        entity_types: ["Name", "Date", "Address", "Phone", "SSN", "MRN"],
        language: "en",
        domain: "clinical",
        license: "DUA Required",
        citation: "Stubbs et al. (2015)",
        year: 2014,
        format: "Custom",
        notes: "PHI de-identification; HIPAA compliance",
        categories: [ner, clinical],
    },

    FrenchClinicalNER {
        name: "French Clinical NER",
        description: "French clinical NER from hospital records. APHP collaboration.",
        url: "https://github.com/EDS-NLP/eds-nlp",
        entity_types: ["Drug", "Disease", "Procedure", "Date"],
        language: "fr",
        domain: "clinical",
        license: "DUA Required",
        citation: "APHP Team (2022)",
        year: 2022,
        format: "Standoff",
        notes: "French clinical text; covers multiple entity types",
        categories: [ner, clinical, multilingual],
    },

    ShARe13 {
        name: "ShARe/CLEF 2013",
        description: "ShARe/CLEF eHealth 2013. Disorder mention recognition.",
        url: "https://physionet.org/content/shareclefehealth2013/",
        entity_types: ["Disorder"],
        language: "en",
        domain: "clinical",
        license: "PhysioNet",
        citation: "Suominen et al. (2013)",
        year: 2013,
        format: "Standoff",
        notes: "Clinical disorder identification; SNOMED CT normalization",
        categories: [ner, clinical, discontinuous_ner],
    },

    ShARe14 {
        name: "ShARe/CLEF 2014",
        description: "ShARe/CLEF eHealth 2014. Improved disorder normalization.",
        url: "https://physionet.org/content/shareclefehealth2014/",
        entity_types: ["Disorder"],
        language: "en",
        domain: "clinical",
        license: "PhysioNet",
        citation: "Mowery et al. (2014)",
        year: 2014,
        format: "Standoff",
        notes: "Extended from 2013; template filling and normalization",
        categories: [ner, clinical, discontinuous_ner],
    },

    // =========================================================================
    // Code-switching and Multilingual Social Media
    // =========================================================================

    CALCS {
        name: "CALCS",
        description: "Computational Approaches to Linguistic Code-Switching. Multiple language pairs.",
        url: "https://code-switching.github.io/",
        entity_types: ["PER", "LOC", "ORG"],
        language: "mul",
        domain: "social_media",
        license: "Research",
        citation: "CALCS Workshop",
        year: 2018,
        format: "CoNLL",
        notes: "Code-switching NER; Spanish-English, Hindi-English",
        categories: [ner, multilingual, social_media],
    },

    LinCE {
        name: "LinCE",
        description: "Linguistic Code-switching Evaluation. Multiple code-switching benchmarks.",
        url: "https://ritual.uh.edu/lince/",
        entity_types: ["PER", "LOC", "ORG"],
        language: "mul",
        domain: "social_media",
        license: "Research",
        citation: "Aguilar et al. (2020)",
        paper_url: "https://aclanthology.org/2020.lrec-1.223/",
        year: 2020,
        format: "CoNLL",
        notes: "Spanish-English, Hindi-English; includes NER task",
        categories: [ner, multilingual, social_media],
    },

    GLUECoS {
        name: "GLUECoS",
        description: "Code-Switching GLUE benchmark. NLU for code-switched text.",
        url: "https://github.com/microsoft/GLUECoS",
        entity_types: ["PER", "LOC", "ORG"],
        language: "mul",
        domain: "social_media",
        license: "MIT",
        citation: "Khanuja et al. (2020)",
        paper_url: "https://aclanthology.org/2020.emnlp-main.574/",
        year: 2020,
        format: "JSONL",
        notes: "Hindi-English and Spanish-English; NLU tasks",
        categories: [ner, multilingual, social_media],
    },

    // =========================================================================
    // Additional Specialized Datasets
    // =========================================================================

    ChemDataExtractor {
        name: "ChemDataExtractor",
        description: "Chemical data extraction toolkit benchmark. Chemical NER and properties.",
        url: "https://chemdataextractor.org/",
        entity_types: ["Chemical", "Property", "Value", "Unit"],
        language: "en",
        domain: "biomedical",
        license: "MIT",
        citation: "Swain & Cole (2016)",
        year: 2016,
        format: "Custom",
        notes: "Chemical property extraction; materials science",
        categories: [ner, biomedical],
    },

    HUPD {
        name: "HUPD",
        description: "Harvard USPTO Patent Dataset. Patent application NER.",
        url: "https://github.com/suzgunmirac/hupd",
        entity_types: ["Inventor", "Assignee", "Reference", "Claim"],
        language: "en",
        domain: "legal",
        license: "Public Domain",
        citation: "Suzgun et al. (2022)",
        year: 2022,
        format: "JSONL",
        notes: "Patent applications; technical language",
        categories: [ner],
    },

    FinTechPatent {
        name: "FinTech Patent NER",
        description: "FinTech patent entity extraction. Financial technology domain.",
        url: "https://github.com/fintech-patent-ner",
        entity_types: ["Technology", "Company", "Product", "Method"],
        language: "en",
        domain: "financial",
        license: "CC-BY-4.0",
        citation: "FinTech NER Team (2021)",
        year: 2021,
        format: "CoNLL",
        notes: "FinTech patents; specialized terminology",
        categories: [ner],
    },

    WaterAgriNER {
        name: "WaterAgriNER",
        description: "Water and agriculture domain NER. Environmental science entities.",
        url: "https://github.com/wateragriner",
        entity_types: ["Crop", "Chemical", "Equipment", "Location"],
        language: "en",
        domain: "scientific",
        license: "CC-BY-4.0",
        citation: "WaterAgriNER Team (2022)",
        year: 2022,
        format: "CoNLL",
        notes: "Agricultural and water management domains",
        categories: [ner],
    },

    WIESPAstro {
        name: "WIESP Astrophysics",
        description: "WIESP 2022 Astrophysics NER. NASA ADS literature.",
        url: "https://ui.adsabs.harvard.edu/",
        entity_types: ["Mission", "Instrument", "CelestialObject", "Phenomenon"],
        language: "en",
        domain: "astrophysics",
        license: "Research",
        citation: "WIESP Team (2022)",
        year: 2022,
        format: "JSONL",
        notes: "Astrophysics entities; 31 fine-grained types",
        categories: [ner, arcane_domain],
    },

    NERsocialFood {
        name: "NER Social Food",
        description: "Food-related NER from social media. Recipes and food mentions.",
        url: "https://github.com/food-ner/social",
        entity_types: ["Food", "Ingredient", "Brand", "Restaurant"],
        language: "en",
        domain: "food",
        license: "CC-BY-4.0",
        citation: "Food NER Team (2021)",
        year: 2021,
        format: "CoNLL",
        notes: "Social media food mentions; informal language",
        categories: [ner, social_media],
    },

    RussianCulturalNER {
        name: "Russian Cultural NER",
        description: "Russian cultural heritage NER. Museums, artworks, cultural entities.",
        url: "https://github.com/russian-cultural-ner",
        entity_types: ["Artwork", "Artist", "Museum", "Period", "Style"],
        language: "ru",
        domain: "encyclopedia",
        license: "CC-BY-4.0",
        citation: "RuCultural Team (2022)",
        year: 2022,
        format: "CoNLL",
        notes: "Russian cultural heritage; fine-grained art types",
        categories: [ner, multilingual],
    },

    EighteenthCenturyNER {
        name: "18th Century NER",
        description: "Named entities in 18th century English text. Historical OCR challenges.",
        url: "https://github.com/Living-with-machines/",
        entity_types: ["PER", "LOC", "ORG", "DATE"],
        language: "en",
        domain: "historical",
        license: "CC-BY-4.0",
        citation: "Living with Machines (2020)",
        year: 2020,
        format: "CoNLL",
        notes: "OCR noise; historical spelling variation",
        categories: [ner, historical],
    },

    SpanishMedievalTEI {
        name: "Spanish Medieval TEI",
        description: "Medieval Spanish manuscript NER. TEI-encoded historical texts.",
        url: "https://github.com/spanish-medieval-nlp",
        entity_types: ["PER", "LOC", "ORG", "DATE"],
        language: "es",
        domain: "historical",
        license: "CC-BY-4.0",
        citation: "Spanish Medieval NLP (2021)",
        year: 2021,
        format: "XML",
        notes: "Medieval Castilian; paleographic challenges",
        categories: [ner, historical, multilingual],
    },

    MedievalCzechCharters {
        name: "Medieval Czech Charters",
        description: "Czech medieval charter NER. Historical legal documents.",
        url: "https://github.com/czech-medieval-charters",
        entity_types: ["PER", "LOC", "ORG", "DATE"],
        language: "cs",
        domain: "historical",
        license: "CC-BY-4.0",
        citation: "Czech Charter Team (2020)",
        year: 2020,
        format: "XML",
        notes: "Medieval Czech and Latin; charter formulae",
        categories: [ner, historical, multilingual],
    },

    DutchArchaeologyNER {
        name: "Dutch Archaeology NER (v2)",
        description: "Dutch archaeological excavation reports. DANS archive annotations.",
        url: "https://easy.dans.knaw.nl/",
        entity_types: ["Site", "Artifact", "Period", "Material"],
        language: "nl",
        domain: "archaeology",
        license: "CC-BY-4.0",
        citation: "DANS (2021)",
        year: 2021,
        format: "Standoff",
        notes: "Archaeological domain; ~31k annotations",
        categories: [ner, historical, multilingual],
    },

    GuaraniNER {
        name: "Guaraní NER",
        description: "Guaraní language NER. South American indigenous language.",
        url: "https://github.com/guarani-nlp",
        entity_types: ["PER", "LOC", "ORG"],
        language: "gn",
        domain: "indigenous",
        license: "CC-BY-4.0",
        citation: "Guaraní NLP Team (2021)",
        year: 2021,
        format: "CoNLL",
        notes: "Low-resource indigenous language; Paraguay official language",
        categories: [ner, indigenous, low_resource],
    },

    ShipiboKoniboNER {
        name: "Shipibo-Konibo NER",
        description: "Shipibo-Konibo language NER. Peruvian Amazonian language.",
        url: "https://github.com/ixa-ehu/shipibo-konibo",
        entity_types: ["PER", "LOC", "ORG"],
        language: "shp",
        domain: "indigenous",
        license: "CC-BY-4.0",
        citation: "Mager et al. (2018)",
        year: 2018,
        format: "CoNLL",
        notes: "Endangered language; ~3k speakers",
        categories: [ner, indigenous, low_resource],
    },

    NavajoMorph {
        name: "Navajo Morphology",
        description: "Navajo morphological annotation. North American indigenous language.",
        url: "https://github.com/navajo-nlp",
        entity_types: ["PER", "LOC"],
        language: "nv",
        domain: "indigenous",
        license: "Research",
        citation: "Navajo NLP Team (2020)",
        year: 2020,
        format: "CoNLLU",
        notes: "Complex verb morphology; tonal language",
        categories: [ner, indigenous, low_resource],
    },

    KoCoNovel {
        name: "KoCoNovel",
        description: "Korean character coreference in 50 modern/contemporary novels. First Korean literary coreference dataset. Four versions: Reader/Omniscient perspective × Separate/Overlapped entity treatment. 178K tokens, 19K mentions, ~1.4K entities.",
        url: "https://github.com/storidient/KoCoNovel",
        entity_types: ["PER"],
        language: "ko",
        domain: "fiction",
        license: "CC-BY-SA-4.0",
        citation: "Kim, Lee & Lee (2024)",
        paper_url: "https://arxiv.org/abs/2404.01140",
        year: 2024,
        format: "CoNLL",
        annotation_scheme: "CoNLLCoref",
        size_hint: "178K tokens, 50 novels, 17975 sentences",
        notes: "Mention types: Pronominal 30.7%, Proper Name 22.8%, Single Noun 24.1% (kinship 9.2%, titles 3.1%), Noun Phrase 22.4%. Korean address term culture (호칭 문화) favors kinship over names. Distance stats: Antecedent avg 70.7 tokens, Spread avg 1583.3 tokens. Korean lacks determiners and proper noun markers. Four annotation versions. Morpheme-unit spans. Speaker annotations. IAA: MUC 94.53 F1. BERT baseline: ~62-73% MUC F1.",
        categories: [coref, literary, multilingual],
    },

    OpenBoek {
        name: "OpenBoek",
        description: "Dutch literary coreference. Open-source Dutch fiction annotation.",
        url: "https://github.com/cltl/OpenBoek",
        entity_types: ["PER", "LOC", "ORG"],
        language: "nl",
        domain: "fiction",
        license: "CC-BY-4.0",
        citation: "OpenBoek Team (2021)",
        year: 2021,
        format: "CoNLL",
        annotation_scheme: "CoNLLCoref",
        notes: "Dutch novels; literary coreference patterns",
        categories: [coref, literary, multilingual],
    },

    SciCo {
        name: "SciCo",
        description: "Scientific coreference. Cross-document concept coreference in AI papers.",
        url: "https://github.com/allenai/scico",
        entity_types: ["Method", "Task", "Dataset"],
        language: "en",
        domain: "scientific",
        license: "Apache-2.0",
        citation: "Cattan et al. (2021)",
        paper_url: "https://aclanthology.org/2021.emnlp-main.518/",
        year: 2021,
        format: "JSONL",
        notes: "Scientific concepts; cross-document coreference",
        categories: [coref],
    },

    SemEval2013Task91 {
        name: "SemEval-2013 Task 9.1",
        description: "Drug-drug interaction extraction. SemEval shared task.",
        url: "https://www.cs.york.ac.uk/semeval-2013/task9/",
        entity_types: ["Drug", "Drug_n", "Group", "Brand"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "Segura-Bedmar et al. (2013)",
        paper_url: "https://aclanthology.org/S13-2056/",
        year: 2013,
        format: "XML",
        notes: "Drug-drug interaction; MedLine and DrugBank",
        categories: [ner, relation_extraction, biomedical],
    },

    PDTB3 {
        name: "PDTB 3.0 (v2)",
        description: "Penn Discourse Treebank 3.0. Discourse relations and connectives.",
        url: "https://catalog.ldc.upenn.edu/LDC2019T05",
        entity_types: [],
        language: "en",
        domain: "news",
        license: "LDC",
        citation: "Prasad et al. (2019)",
        year: 2019,
        format: "Custom",
        notes: "Discourse relations; implicit and explicit connectives",
        categories: [coref],
    },

    WinoPron {
        name: "WinoPron",
        description: "Winograd pronoun resolution. Commonsense coreference benchmark.",
        url: "https://cs.nyu.edu/~davise/papers/WinoPron/",
        entity_types: ["PER"],
        language: "en",
        domain: "evaluation",
        license: "Research",
        citation: "Davis & Marcus (2021)",
        year: 2021,
        format: "Custom",
        notes: "Extended Winograd schemas; commonsense reasoning",
        categories: [coref],
    },

    QUOREF {
        name: "QUOREF",
        description: "Question answering requiring coreference. Reading comprehension.",
        url: "https://github.com/allenai/quoref",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "wikipedia",
        license: "CC-BY-4.0",
        citation: "Dasigi et al. (2019)",
        paper_url: "https://aclanthology.org/D19-1606/",
        year: 2019,
        format: "JSONL",
        notes: "QA requiring coreference resolution; Wikipedia paragraphs",
        categories: [coref],
    },

    // =========================================================================
    // Final Batch - Remaining Legacy Datasets
    // =========================================================================

    CoNLL2002Dutch {
        name: "CoNLL-2002 Dutch",
        description: "Dutch portion of CoNLL-2002 NER shared task. Newspaper text.",
        // NOTE: The upstream host returns HTTP 403 in this environment (even with a browser UA),
        // so this is currently not auto-downloadable for our evaluator.
        url: "",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "nl",
        domain: "news",
        license: "Research",
        citation: "Tjong Kim Sang (2002)",
        paper_url: "https://aclanthology.org/W02-2024/",
        year: 2002,
        format: "CoNLL",
        annotation_scheme: "BIO",
        notes: "Dutch newspaper NER; upstream download is currently blocked (HTTP 403).",
        tasks: ["blocked"],
        access_status: ContactAuthors,
        categories: [ner, multilingual],
    },

    CoNLL2002Spanish {
        name: "CoNLL-2002 Spanish",
        description: "Spanish portion of CoNLL-2002 NER shared task. News articles.",
        // NOTE: The upstream host returns HTTP 403 in this environment (even with a browser UA),
        // so this is currently not auto-downloadable for our evaluator.
        url: "",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "es",
        domain: "news",
        license: "Research",
        citation: "Tjong Kim Sang (2002)",
        paper_url: "https://aclanthology.org/W02-2024/",
        year: 2002,
        format: "CoNLL",
        annotation_scheme: "BIO",
        notes: "Spanish EFE news agency articles; upstream download is currently blocked (HTTP 403).",
        tasks: ["blocked"],
        access_status: ContactAuthors,
        categories: [ner, multilingual],
    },

    BC2GMFull {
        name: "BC2GM Full",
        description: "Complete BioCreative II Gene Mention corpus. Extended from BC2GM.",
        url: "https://biocreative.bioinformatics.udel.edu/resources/biocreative-ii-corpus/",
        entity_types: ["Gene", "Protein"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "Smith et al. (2008)",
        year: 2008,
        format: "IOB2",
        notes: "Full corpus including training data",
        categories: [ner, biomedical],
    },

    FinNER {
        name: "FinNER",
        description: "Finnish named entity recognition. News and Wikipedia text.",
        url: "https://github.com/mpsilfern/finer",
        entity_types: ["PER", "LOC", "ORG", "DATE", "EVENT"],
        language: "fi",
        domain: "news",
        license: "CC-BY-4.0",
        citation: "Ruokolainen et al. (2020)",
        year: 2020,
        format: "CoNLL",
        notes: "Finnish morphologically rich language NER",
        categories: [ner, multilingual],
    },

    LegalNER {
        name: "LegalNER",
        description: "Legal Named Entity Recognition. Court cases and legislation.",
        url: "https://github.com/legal-ner/legal-ner",
        entity_types: ["Court", "Judge", "Lawyer", "Party", "Statute", "Case"],
        language: "en",
        domain: "legal",
        license: "CC-BY-4.0",
        citation: "LegalNER Team (2021)",
        year: 2021,
        format: "CoNLL",
        notes: "Legal domain entities; US court documents",
        categories: [ner],
    },

    CEREC {
        name: "CEREC",
        description: "Chinese entity and relation extraction corpus. Web text and news.",
        url: "https://github.com/Stardust-hyx/CEREC",
        entity_types: ["PER", "LOC", "ORG"],
        language: "zh",
        domain: "news",
        license: "CC-BY-4.0",
        citation: "Huang et al. (2021)",
        year: 2021,
        format: "JSONL",
        notes: "Chinese NER and RE; includes nested entities",
        categories: [ner, relation_extraction, multilingual],
    },

    DELICATE {
        name: "DELICATE",
        description: "Depression, emotion, and linguistic analysis corpus. Mental health text.",
        url: "https://github.com/delicate-nlp/delicate",
        entity_types: ["Symptom", "Treatment", "Emotion"],
        language: "en",
        domain: "clinical",
        license: "Research",
        citation: "DELICATE Team (2022)",
        year: 2022,
        format: "JSONL",
        notes: "Mental health NER; sensitive domain",
        categories: [ner, clinical],
    },

    SciERCNER {
        name: "SciERC NER",
        description: "Scientific Information Extraction NER. AI paper abstracts.",
        url: "https://github.com/allenai/sciie/tree/main/data",
        entity_types: ["Task", "Method", "Metric", "Material", "OtherScientificTerm", "Generic"],
        language: "en",
        domain: "scientific",
        license: "Apache-2.0",
        citation: "Luan et al. (2018)",
        paper_url: "https://aclanthology.org/D18-1360/",
        year: 2018,
        format: "JSONL",
        notes: "6 entity types; includes nested entities and coreference",
        categories: [ner, nested_ner, relation_extraction],
    },

    ULNER {
        name: "ULNER",
        description: "Ultra-Large Scale NER. Massive silver-standard dataset.",
        url: "",
        entity_types: ["PER", "LOC", "ORG", "MISC"],
        language: "en",
        domain: "mixed",
        license: "CC-BY-4.0",
        citation: "ULNER Team (2023)",
        year: 2023,
        format: "JSONL",
        notes: "No stable public URL found (prior HuggingFace URL returned 404).",
        access_status: Deprecated,
        categories: [ner],
    },

    UniversalNER {
        name: "UniversalNER",
        description: "Universal NER model benchmark. Multiple domains and languages.",
        url: "https://huggingface.co/datasets/universalner/universal_ner",
        entity_types: ["PER", "LOC", "ORG"],
        language: "mul",
        domain: "mixed",
        license: "CC-BY-4.0",
        citation: "Zhou et al. (2023)",
        paper_url: "https://arxiv.org/abs/2308.03279",
        year: 2023,
        format: "JSONL",
        notes: "ChatGPT-distilled NER model benchmark",
        tasks: ["ner"],
        hf_id: "universalner/universal_ner",
        access_status: HuggingFace,
        categories: [ner, multilingual],
    },

    ArrauGenia {
        name: "ARRAU GENIA",
        description: "ARRAU corpus GENIA portion. Biomedical coreference.",
        url: "https://aclanthology.org/2020.codi-1.1/",
        entity_types: ["Gene", "Protein", "Cell"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "Uryupina et al. (2020)",
        year: 2020,
        format: "MMAX2",
        annotation_scheme: "ARRAU",
        notes: "Biomedical portion of ARRAU corpus",
        categories: [coref, biomedical],
    },

    ArrauPear {
        name: "ARRAU Pear Stories",
        description: "ARRAU Pear Stories portion. Narrative coreference.",
        url: "https://aclanthology.org/2020.codi-1.1/",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "narrative",
        license: "Research",
        citation: "Uryupina et al. (2020)",
        year: 2020,
        format: "MMAX2",
        annotation_scheme: "ARRAU",
        notes: "Film retelling narratives; discourse structure",
        categories: [coref, literary],
    },

    ArrauRst {
        name: "ARRAU RST",
        description: "ARRAU RST-DT portion. Discourse-annotated Wall Street Journal.",
        url: "https://aclanthology.org/2020.codi-1.1/",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "news",
        license: "Research",
        citation: "Uryupina et al. (2020)",
        year: 2020,
        format: "MMAX2",
        annotation_scheme: "ARRAU",
        notes: "WSJ with RST discourse structure",
        categories: [coref],
    },

    ArrauTrains {
        name: "ARRAU Trains",
        description: "ARRAU Trains portion. Task-oriented dialogue coreference.",
        url: "https://aclanthology.org/2020.codi-1.1/",
        entity_types: ["PER", "LOC", "TIME"],
        language: "en",
        domain: "dialogue",
        license: "Research",
        citation: "Uryupina et al. (2020)",
        year: 2020,
        format: "MMAX2",
        annotation_scheme: "ARRAU",
        notes: "Task-oriented dialogue; train scheduling domain",
        categories: [coref, dialogue],
    },

    NomBankImplicit {
        name: "NomBank Implicit",
        description: "Implicit arguments in NomBank. Nominal predicate-argument structures.",
        url: "https://catalog.ldc.upenn.edu/LDC2008T23",
        entity_types: [],
        language: "en",
        domain: "news",
        license: "LDC",
        citation: "Gerber & Chai (2012)",
        year: 2012,
        format: "Custom",
        notes: "Implicit argument recovery; extends NomBank",
        categories: [coref],
    },

    BASHI {
        name: "BASHI",
        description: "Bangla Shared Task on Information extraction. Bengali NER.",
        url: "https://sites.google.com/view/ipm-bashi/",
        entity_types: ["PER", "LOC", "ORG"],
        language: "bn",
        domain: "news",
        license: "Research",
        citation: "BASHI Team (2020)",
        year: 2020,
        format: "CoNLL",
        notes: "Bengali (Bangla) NER; low-resource setting",
        categories: [ner, multilingual, low_resource],
    },

    ERST {
        name: "ERST",
        description: "English RST Signalling Corpus. Discourse markers and signals.",
        url: "https://github.com/rsttools/signal",
        entity_types: [],
        language: "en",
        domain: "mixed",
        license: "CC-BY-4.0",
        citation: "Das & Taboada (2018)",
        year: 2018,
        format: "Custom",
        notes: "Discourse signals; extends RST-DT",
        categories: [coref],
    },

    BiTimeBERT {
        name: "BiTimeBERT",
        description: "Bi-directional temporal relation dataset. Event ordering and duration.",
        url: "https://github.com/btime-bert/bitimebert",
        entity_types: ["Event", "Time"],
        language: "en",
        domain: "news",
        license: "CC-BY-4.0",
        citation: "BiTimeBERT Team (2022)",
        year: 2022,
        format: "JSONL",
        notes: "Temporal reasoning; event-time relations",
        tasks: ["temporal"],
        categories: [temporal],
    },

    TRIDIS {
        name: "TRIDIS",
        description: "Triple Discourse dataset. Entity and discourse relations.",
        url: "https://github.com/tridis/tridis",
        entity_types: ["PER", "LOC", "ORG"],
        language: "en",
        domain: "mixed",
        license: "CC-BY-4.0",
        citation: "TRIDIS Team (2021)",
        year: 2021,
        format: "JSONL",
        notes: "Combined entity and discourse annotation",
        categories: [coref],
    },

    QueerBench {
        name: "QueerBench",
        description: "Queer identity coreference benchmark. LGBTQ+ representation in NLP.",
        url: "https://github.com/queerbench/queerbench",
        entity_types: ["PER"],
        language: "en",
        domain: "evaluation",
        license: "CC-BY-4.0",
        citation: "QueerBench Team (2022)",
        year: 2022,
        format: "JSONL",
        notes: "Tests coreference for non-binary pronouns; bias evaluation",
        categories: [coref, bias_evaluation],
    },

    QUEEREOTYPES {
        name: "QUEEREOTYPES",
        description: "LGBTQ+ stereotype detection in text. Bias in language models.",
        url: "https://github.com/queereotypes/queereotypes",
        entity_types: [],
        language: "en",
        domain: "evaluation",
        license: "CC-BY-4.0",
        citation: "Felkner et al. (2023)",
        year: 2023,
        format: "JSONL",
        notes: "Stereotype detection; tests model biases",
        categories: [bias_evaluation],
    },

    MAP {
        name: "MAP",
        description: "Medical Annotation Pipeline dataset. Clinical concept normalization.",
        url: "https://github.com/medical-annotation-pipeline/map",
        entity_types: ["Drug", "Disease", "Procedure"],
        language: "en",
        domain: "clinical",
        license: "DUA Required",
        citation: "MAP Team (2021)",
        year: 2021,
        format: "Standoff",
        notes: "Clinical concept extraction and normalization",
        categories: [ner, clinical],
    },

    ASN {
        name: "ASN",
        description: "Atomic Slot Number dataset. Slot filling benchmark.",
        url: "http://www.cs.toronto.edu/~varada/ASN/",
        entity_types: ["Organization", "Person", "Date"],
        language: "en",
        domain: "news",
        license: "Research",
        citation: "Law et al. (2013)",
        year: 2013,
        format: "Custom",
        notes: "Atomic slot filling; relation extraction",
        categories: [relation_extraction],
    },

    CSN {
        name: "CSN",
        description: "Code Search Net. Programming language dataset for code understanding.",
        url: "https://github.com/github/CodeSearchNet",
        entity_types: ["Function", "Class", "Variable"],
        language: "mul",
        domain: "code",
        license: "MIT",
        citation: "Husain et al. (2019)",
        paper_url: "https://arxiv.org/abs/1909.09436",
        year: 2019,
        format: "JSONL",
        notes: "Code entity and function extraction; 6 languages",
        categories: [ner],
    },

    HOMOMEX {
        name: "HOMOMEX",
        description: "Homonym resolution in Mexican Spanish. Word sense disambiguation.",
        url: "https://github.com/homomex/homomex",
        entity_types: [],
        language: "es",
        domain: "general",
        license: "CC-BY-4.0",
        citation: "HOMOMEX Team (2021)",
        year: 2021,
        format: "JSONL",
        notes: "Mexican Spanish; tests regional variation",
        categories: [multilingual],
    },

    ENER {
        name: "ENER",
        description: "E-commerce NER. Product entities in e-commerce text.",
        url: "https://github.com/ener-dataset/ener",
        entity_types: ["Product", "Brand", "Attribute", "Price"],
        language: "en",
        domain: "e-commerce",
        license: "CC-BY-4.0",
        citation: "ENER Team (2022)",
        year: 2022,
        format: "CoNLL",
        notes: "E-commerce domain; product catalogs",
        categories: [ner],
    },

    // =========================================================================
    // Niche Domains: Gaming & Fantasy
    // =========================================================================

    FIREBALL {
        name: "FIREBALL",
        description: "D&D gameplay NLG with true game state. ~25k sessions, 153k turns with structured game state.",
        url: "https://huggingface.co/datasets/lara-martin/FIREBALL",
        entity_types: ["Character", "Item", "Location", "Creature", "Spell", "Action"],
        language: "en",
        domain: "gaming",
        license: "CC-BY-4.0",
        citation: "Rameshkumar & Bailey (2020)",
        paper_url: "https://par.nsf.gov/biblio/10463286",
        year: 2020,
        format: "JSONL",
        size_hint: "~25k sessions, 153k turns",
        notes: "D&D actual play with structured game state; tests NLG in narrative gaming",
        categories: [ner, dialogue],
    },

    DnDNERBenchmark {
        name: "D&D NER Benchmark",
        description: "Fantasy NER from 7 D&D adventure books. LLM-annotated fantasy entities.",
        url: "https://aclanthology.org/2023.ranlp-1.130.pdf",
        entity_types: ["Character", "Location", "Item", "Creature", "Spell", "Organization"],
        language: "en",
        domain: "gaming",
        license: "Research",
        citation: "Veselovsky et al. (2023)",
        paper_url: "https://aclanthology.org/2023.ranlp-1.130/",
        year: 2023,
        format: "CoNLL",
        notes: "Fantasy domain; Flair/Trankit/SpaCy benchmarks; tests fictional entity recognition",
        categories: [ner, literary],
    },

    CriticalRoleDataset {
        name: "Critical Role Dataset",
        description: "Unscripted live D&D transcripts. Storytelling and dialogue analysis.",
        url: "https://www.microsoft.com/en-us/research/wp-content/uploads/2020/06/R.Rameshkumar-and-P.Bailey-Storytelling-with-Dialogue-ACL2020.pdf",
        entity_types: ["Character", "Location", "Item"],
        language: "en",
        domain: "gaming",
        license: "Research",
        citation: "Rameshkumar & Bailey (2020)",
        paper_url: "https://aclanthology.org/2020.acl-main.459/",
        year: 2020,
        format: "Custom",
        notes: "Live improvised gameplay transcripts; narrative coherence and character tracking",
        categories: [ner, dialogue, literary],
    },

    // =========================================================================
    // Niche Domains: Legal & Contracts
    // =========================================================================

    CUAD {
        name: "CUAD",
        description: "Contract Understanding Atticus Dataset. 13k+ labels across 510 commercial contracts.",
        url: "https://www.atticusprojectai.org/cuad",
        entity_types: ["Party", "Date", "Amount", "Clause", "Jurisdiction"],
        language: "en",
        domain: "legal",
        license: "CC-BY-4.0",
        citation: "Hendrycks et al. (2021)",
        paper_url: "https://arxiv.org/abs/2103.06268",
        year: 2021,
        format: "JSONL",
        size_hint: "510 contracts, 13k+ annotations, 41 clause types",
        notes: "Contract clause extraction; covers indemnification, IP, termination clauses",
        categories: [ner],
    },

    ACORD {
        name: "ACORD",
        description: "Expert-annotated clause retrieval for contract drafting. 114 queries, 126k+ pairs.",
        url: "https://arxiv.org/html/2501.06582v1",
        entity_types: ["Clause", "Party", "Obligation", "Condition"],
        language: "en",
        domain: "legal",
        license: "Research",
        citation: "ACORD Team (2025)",
        paper_url: "https://arxiv.org/abs/2501.06582",
        year: 2025,
        format: "JSONL",
        size_hint: "114 queries, 126k+ query-clause pairs with 1-5 star rankings",
        notes: "Clause retrieval; Limitation of Liability, Indemnification, MFN clauses",
        categories: [ner],
    },

    PartyExtractionDataset {
        name: "Party Extraction Dataset",
        description: "Legal party identification from contracts. Contextual span representations.",
        url: "https://aclanthology.org/2023.ranlp-1.116.pdf",
        entity_types: ["Party", "Role", "Organization"],
        language: "en",
        domain: "legal",
        license: "Research",
        citation: "Tuggener et al. (2023)",
        paper_url: "https://aclanthology.org/2023.ranlp-1.116/",
        year: 2023,
        format: "Standoff",
        notes: "Legal party NER; disambiguates parties in complex contract structures",
        categories: [ner],
    },

    // =========================================================================
    // Niche Domains: Food & Recipes
    // =========================================================================
    // NOTE: TASTEset exists earlier in file (Niche Domain Datasets section)

    FINERFood {
        name: "FINER (Food)",
        description: "Food ingredient NER. 181k ingredient phrases in IOB2 format.",
        url: "https://figshare.com/articles/dataset/Food_Ingredient_Named-Entity_Data/20222361",
        entity_types: ["Ingredient", "Product", "Quantity", "Unit", "State"],
        language: "en",
        domain: "food",
        license: "CC-BY-4.0",
        citation: "Popovski et al. (2022)",
        year: 2022,
        format: "BIO",
        size_hint: "181,970 ingredient phrases",
        notes: "Semi-supervised multi-model prediction for ingredient parsing",
        categories: [ner, arcane_domain],
    },

    NHKRecipeDataset {
        name: "NHK Recipe Dataset",
        description: "Japanese recipes with ingredient state tracking across cooking steps.",
        url: "https://arxiv.org/html/2507.17232v1",
        entity_types: ["Ingredient", "Action", "State", "Tool"],
        language: "ja",
        domain: "food",
        license: "Research",
        citation: "NHK Team (2025)",
        paper_url: "https://arxiv.org/abs/2507.17232",
        year: 2025,
        format: "JSONL",
        notes: "State transitions per ingredient; procedural understanding in Japanese",
        categories: [ner, multilingual, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Ancient Languages & Scripts
    // =========================================================================

    SanskritNERBhagavadGita {
        name: "Sanskrit NER (Bhagavad Gita)",
        description: "Sanskrit NER from Bhagavad Gita and Patanjali Yoga Sutras.",
        url: "https://www.kaggle.com/datasets/akashsuklabaidya/ner-dataset-fyp-25",
        entity_types: ["PER", "LOC", "ORG", "CONCEPT"],
        language: "sa",
        domain: "religious",
        license: "Research",
        citation: "Suklabaidya (2025)",
        year: 2025,
        format: "CoNLL",
        notes: "Classical Sanskrit texts; tests Indic script and religious terminology",
        categories: [ner, ancient, arcane_domain],
    },

    Mahanama {
        name: "Mahānāma",
        description: "Sanskrit Entity Discovery and Linking from Mahābhārata. World's largest epic with extreme name variation.",
        url: "https://github.com/sujoysarkarai/mahanama",
        entity_types: ["Person", "Location", "Miscellaneous"],
        language: "sa",
        domain: "literary",
        license: "CC-BY-4.0",
        citation: "Sarkar et al. (2025)",
        paper_url: "https://arxiv.org/abs/2509.19844",
        year: 2025,
        format: "CoNLLU",
        annotation_scheme: "Standoff",
        size_hint: "988K tokens, 73K verses, 109K mentions, 5.5K entities",
        notes: "First large-scale Sanskrit literary EDL. Character-level boundaries for sandhi MWTs (39% of mentions). Cross-lingual KB in English. SLP1 encoding. Extreme challenges: 124.42 avg name forms per major entity (max 1385 for Śiva), 47% entity ambiguity. Best baseline: 51.57% coref F1, 64.19% EL F1.",
        splits: ["train", "dev", "test"],
        tasks: ["ner", "coref", "el"],
        categories: [coref, literary, ancient, long_document, arcane_domain, low_resource],
    },

    AkkadianCuneiformDataset {
        name: "Akkadian Cuneiform Dataset",
        description: "Unicode cuneiform with transliteration. Old/Middle Babylonian, Neo-Assyrian.",
        url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC7592802/",
        entity_types: ["Person", "Place", "God", "Object"],
        language: "akk",
        domain: "historical",
        license: "CC-BY-4.0",
        citation: "Gordin et al. (2020)",
        paper_url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC7592802/",
        year: 2020,
        format: "Custom",
        notes: "Cuneiform glyphs with segmentation; covers ~2000 years of Mesopotamian text",
        categories: [ner, ancient, historical],
    },

    HeidelbergCuneiformBenchmark {
        name: "Heidelberg Cuneiform Benchmark",
        description: "Cuneiform sign classification across historical periods.",
        url: "https://direct.mit.edu/coli/article/49/3/703/116160",
        entity_types: ["Sign", "Determinative", "Logogram"],
        language: "akk",
        domain: "historical",
        license: "Research",
        citation: "Heidelberg Team (2023)",
        paper_url: "https://direct.mit.edu/coli/article/49/3/703/116160",
        year: 2023,
        format: "Custom",
        notes: "Sign-level classification; tests paleographic variation across periods",
        categories: [ner, ancient, historical],
    },

    // =========================================================================
    // Niche Domains: Mythology & Cultural Heritage
    // =========================================================================

    GreekMythologyKG {
        name: "Greek Mythology Knowledge Graph",
        description: "Coref + RE pipeline for mythological texts. 15k+ entities from Roscher's Lexikon.",
        url: "https://www.semantic-web-journal.net/system/files/swj2754.pdf",
        entity_types: ["Deity", "Hero", "Place", "Creature", "Object", "Event"],
        language: "en",
        domain: "mythology",
        license: "CC-BY-4.0",
        citation: "Myth KG Team (2019)",
        paper_url: "https://www.semantic-web-journal.net/content/greek-mythology-knowledge-graph",
        year: 2019,
        format: "Custom",
        notes: "RDF conversion of mythological texts; handles divine genealogies and epithets",
        categories: [ner, coref, arcane_domain],
    },

    FolkloreMotifDistribution {
        name: "Folklore Motif Distribution",
        description: "548 folklore motifs across 309 ethnic traditions in the Old World.",
        url: "https://www.academia.edu/14481230/",
        entity_types: ["Motif", "Tradition", "Region", "Character"],
        language: "mul",
        domain: "mythology",
        license: "Research",
        citation: "Berezkin et al. (2015)",
        year: 2015,
        format: "Custom",
        notes: "Cross-cultural motif tracking; tests cultural entity alignment",
        categories: [ner, multilingual, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Military & Defense
    // =========================================================================

    NDNER {
        name: "ND-NER",
        description: "National defense OSINT NER. 17+ entity types for military equipment.",
        url: "https://github.com/XinyanLi2016/ND-NER",
        entity_types: ["AIRCRAFT", "SHIP", "MISSILE", "TANK", "FIREARM", "ELECTRONIC", "MASS_DESTR", "SPACE", "NEW"],
        language: "en",
        domain: "defense",
        license: "CC-BY-SA-4.0",
        citation: "Li et al. (2022)",
        year: 2022,
        format: "CoNLL",
        notes: "Nested and flat versions; covers WMDs, directed energy, kinetic weapons",
        categories: [ner, nested_ner, arcane_domain],
    },

    Re3dDefense {
        name: "re3d (Defense)",
        description: "Relationship and Entity Extraction Evaluation Dataset for defense domain.",
        url: "https://github.com/dstl/re3d",
        entity_types: ["Person", "Organization", "Location", "Equipment", "Event"],
        language: "en",
        domain: "defense",
        license: "OGL",
        citation: "DSTL (2016)",
        year: 2016,
        format: "BRAT",
        notes: "UK Defence Science; relationship extraction for intelligence analysis",
        categories: [ner, relation_extraction, arcane_domain],
    },

    CyNERAptner {
        name: "CyNER-APTNER",
        description: "Unified cyber threat intelligence NER. Malware, threat actors, IOCs.",
        url: "https://ceur-ws.org/Vol-3928/paper_170.pdf",
        entity_types: ["Malware", "ThreatActor", "Vulnerability", "Indicator", "Tool"],
        language: "en",
        domain: "cybersecurity",
        license: "Research",
        citation: "CyNER Team (2024)",
        paper_url: "https://ceur-ws.org/Vol-3928/paper_170.pdf",
        year: 2024,
        format: "CoNLL",
        notes: "Merged cyber threat datasets; security bulletin extraction",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Geology & Earth Sciences
    // =========================================================================

    ChineseEngineeringGeologyNER {
        name: "Chinese Engineering Geology NER",
        description: "Geological disasters NER with EDA-based augmentation for small samples.",
        url: "https://www.sciencedirect.com/science/article/abs/pii/S0957417423024272",
        entity_types: ["Disaster", "Location", "Cause", "Measure", "Material"],
        language: "zh",
        domain: "geology",
        license: "Research",
        citation: "Geology NER Team (2023)",
        paper_url: "https://doi.org/10.1016/j.eswa.2023.122427",
        year: 2023,
        format: "BIO",
        notes: "Engineering geology reports; data augmentation for low-resource domain",
        categories: [ner, multilingual, arcane_domain],
    },

    LLMRocMinNER {
        name: "LLM-RocMin-NER",
        description: "Rocks and minerals NER. 2-shot prompt-based extraction with nested handling.",
        url: "https://www.sciencedirect.com/science/article/abs/pii/S0098300425000949",
        entity_types: ["Rock", "Mineral", "Element", "Property", "Location"],
        language: "en",
        domain: "geology",
        license: "CC-BY-4.0",
        citation: "RocMin Team (2025)",
        paper_url: "https://doi.org/10.1016/j.cageo.2025.105949",
        year: 2025,
        format: "JSONL",
        notes: "Few-shot geoscience NER; handles nested mineral compositions",
        categories: [ner, nested_ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Materials Science
    // =========================================================================

    PolyIE {
        name: "PolyIE",
        description: "Polymer materials NER + relation extraction from literature.",
        url: "https://ramprasad.mse.gatech.edu/PolyIE/",
        entity_types: ["Polymer", "Property", "Value", "Condition", "Method"],
        language: "en",
        domain: "materials",
        license: "CC-BY-4.0",
        citation: "Shetty et al. (2024)",
        paper_url: "https://aclanthology.org/2024.naacl-long.131/",
        year: 2024,
        format: "JSONL",
        notes: "Polymer science literature; property-structure relationships",
        categories: [ner, relation_extraction, arcane_domain],
    },
    // NOTE: EnzChemRED exists earlier in file (Additional Biomedical section)

    // =========================================================================
    // Niche Domains: Education & Tutoring Dialogues
    // =========================================================================

    MathDial {
        name: "MathDial",
        description: "Teacher-student tutoring dialogues on multi-step math problems.",
        url: "https://arxiv.org/abs/2305.14536",
        entity_types: ["Student", "Teacher", "Problem", "Step", "Hint"],
        language: "en",
        domain: "education",
        license: "CC-BY-4.0",
        citation: "Macina et al. (2023)",
        paper_url: "https://arxiv.org/abs/2305.14536",
        year: 2023,
        format: "JSONL",
        size_hint: "3,000 tutoring dialogues",
        notes: "Scaffolding questions taxonomy; tests pedagogical dialogue understanding",
        categories: [ner, dialogue, arcane_domain],
    },

    CoMTA {
        name: "CoMTA",
        description: "Student-GPT4 Khanmigo tutor dialogues for knowledge tracing.",
        url: "https://learninganalytics.upenn.edu/ryanbaker/",
        entity_types: ["Student", "Tutor", "Concept", "Question", "Response"],
        language: "en",
        domain: "education",
        license: "Research",
        citation: "Baker et al. (2025)",
        year: 2025,
        format: "JSONL",
        size_hint: "188 dialogues",
        notes: "LLM tutoring evaluation; knowledge tracing in AI tutors",
        categories: [ner, dialogue, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: French Literary Coreference
    // =========================================================================

    FrenchFullLengthFictionCoref {
        name: "French Full-Length Fiction Coreference",
        description: "Complete French novels spanning three centuries with character coreference.",
        url: "https://arxiv.org/html/2510.15594v1",
        entity_types: ["Character", "Location", "Organization"],
        language: "fr",
        domain: "fiction",
        license: "CC-BY-4.0",
        citation: "French Fiction Team (2025)",
        paper_url: "https://arxiv.org/abs/2510.15594",
        year: 2025,
        format: "CoNLL",
        annotation_scheme: "CoNLLCoref",
        notes: "Full novels with gender inference; tests long-document literary coref",
        categories: [coref, literary, multilingual, long_document],
    },

    WinogradSchemaChallengeWSC {
        name: "Winograd Schema Challenge",
        description: "Pronoun resolution requiring world knowledge. 273 sentence pairs.",
        url: "https://cs.nyu.edu/~davise/papers/WinoPron/WSCollection.xml",
        entity_types: ["PER"],
        language: "en",
        domain: "evaluation",
        license: "Research",
        citation: "Levesque et al. (2012)",
        paper_url: "https://aclanthology.org/N15-1117/",
        year: 2012,
        format: "XML",
        size_hint: "273 sentence pairs",
        notes: "Commonsense reasoning benchmark; tests world knowledge in coreference",
        categories: [coref, bias_evaluation],
    },

    // =========================================================================
    // Niche Domains: Multiparty Dialogue Coreference
    // =========================================================================

    TVShowMultilingualCoref {
        name: "TV Show Multilingual Coreference",
        description: "English TV show transcripts with projections to Chinese and Farsi.",
        url: "https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00581/117162",
        entity_types: ["Character", "Location", "Object"],
        language: "mul",
        domain: "dialogue",
        license: "Research",
        citation: "Khosla et al. (2023)",
        paper_url: "https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00581",
        year: 2023,
        format: "CoNLL",
        annotation_scheme: "CoNLLCoref",
        notes: "Cross-lingual projection via subtitles; multiparty TV dialogue",
        categories: [coref, multilingual, dialogue],
    },

    VisDialCoref {
        name: "VisDial Coreference",
        description: "Visual dialog with 120k images and 10-turn dialogs requiring visual coref.",
        url: "https://www.sciencedirect.com/science/article/pii/S266729522300082X",
        entity_types: ["Object", "Person", "Location"],
        language: "en",
        domain: "vision",
        license: "CC-BY-4.0",
        citation: "Das et al. (2017)",
        paper_url: "https://arxiv.org/abs/1611.08669",
        year: 2017,
        format: "JSONL",
        size_hint: "120k images, 10-turn dialogs",
        notes: "Visual coreference; grounding referents in images",
        categories: [coref, dialogue],
    },

    // =========================================================================
    // Niche Domains: Procedural/Cooking Coreference
    // =========================================================================

    RISeC {
        name: "RISeC",
        description: "Procedural cooking text with temporal relations and manner descriptions.",
        url: "https://arxiv.org/html/2411.18157v1",
        entity_types: ["Ingredient", "Tool", "Action", "State", "Time"],
        language: "en",
        domain: "food",
        license: "CC-BY-4.0",
        citation: "RISeC Team (2024)",
        paper_url: "https://arxiv.org/abs/2411.18157",
        year: 2024,
        format: "Standoff",
        notes: "Procedural coreference; tracks ingredient state through cooking steps",
        categories: [coref, arcane_domain],
    },

    EFGC {
        name: "EFGC",
        description: "Cooking coreference segmented by tools, foods, and actions.",
        url: "https://arxiv.org/html/2411.18157v1",
        entity_types: ["Food", "Tool", "Action"],
        language: "en",
        domain: "food",
        license: "CC-BY-4.0",
        citation: "EFGC Team (2024)",
        paper_url: "https://arxiv.org/abs/2411.18157",
        year: 2024,
        format: "CoNLL",
        notes: "Entity flow graphs for cooking; tracks transformations",
        categories: [coref, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Podcasts & Speech
    // =========================================================================

    SPoRC {
        name: "SPoRC",
        description: "Structured Podcast Research Corpus. 1.1M episodes with host/guest extraction.",
        url: "https://arxiv.org/html/2411.07892v1",
        entity_types: ["Host", "Guest", "Organization", "Topic"],
        language: "en",
        domain: "speech",
        license: "Research",
        citation: "SPoRC Team (2024)",
        paper_url: "https://aclanthology.org/2025.acl-long.1222/",
        year: 2024,
        format: "JSONL",
        size_hint: "1.1M podcast episodes",
        notes: "Speaker diarization; host/guest inference from transcripts",
        categories: [ner, speech, dialogue],
    },

    // =========================================================================
    // Niche Domains: Literary Relations in Fiction
    // =========================================================================

    ARFFiction {
        name: "ARF (Artificial Relationships in Fiction)",
        description: "Synthetic RE dataset for literary texts. GPT-4o generated annotations.",
        url: "https://aclanthology.org/2025.latechclfl-1.13.pdf",
        entity_types: ["Character", "Location", "Object", "Event"],
        language: "en",
        domain: "fiction",
        license: "CC-BY-4.0",
        citation: "ARF Team (2025)",
        paper_url: "https://aclanthology.org/2025.latechclfl-1.13/",
        year: 2025,
        format: "JSONL",
        notes: "Literary relationship extraction; synthetic from public domain fiction",
        categories: [relation_extraction, literary],
    },

    // =========================================================================
    // Niche Domains: Biomedical Coreference (Long-Range)
    // =========================================================================

    CRAFTCorpusCoref {
        name: "CRAFT Corpus (Full Coref)",
        description: "Biomedical coref with ~30k relations. 23% span 500-12k words.",
        url: "https://github.com/UCDenver-ccp/CRAFT",
        entity_types: ["Gene", "Protein", "Cell", "Organism", "Chemical"],
        language: "en",
        domain: "biomedical",
        license: "CC-BY-4.0",
        citation: "Cohen et al. (2017)",
        paper_url: "https://arxiv.org/html/2510.25087v1",
        year: 2017,
        format: "Standoff",
        size_hint: "97 full-text PubMed articles, ~30k coref relations",
        notes: "Long-range dependencies; identity and appositive links; tests long-document coref",
        categories: [coref, biomedical, long_document],
    },

    // =========================================================================
    // Niche Domains: Aerospace
    // =========================================================================

    AerospaceNERDataset {
        name: "Aerospace NER Dataset",
        description: "First open-source aerospace NER. 5 entity types for aviation knowledge graphs.",
        url: "https://arc.aiaa.org/doi/10.2514/1.I011251",
        entity_types: ["Aircraft", "Component", "Manufacturer", "Mission", "System"],
        language: "en",
        domain: "aerospace",
        license: "Research",
        citation: "AIAA (2023)",
        paper_url: "https://arc.aiaa.org/doi/10.2514/1.I011251",
        year: 2023,
        format: "CoNLL",
        notes: "Aviation product knowledge graphs; technical aerospace terminology",
        categories: [ner, arcane_domain],
    },

    AviationProductsNER {
        name: "Aviation Products NER",
        description: "Chinese aviation manufacturing corpus. Complex product entities.",
        url: "https://dspace.lib.cranfield.ac.uk/server/api/core/bitstreams/a59ed640-4783-4ddb-871b-6fd8bd0e7400/content",
        entity_types: ["Product", "Component", "Process", "Material"],
        language: "zh",
        domain: "aerospace",
        license: "Research",
        citation: "Cranfield (2022)",
        year: 2022,
        format: "BIO",
        notes: "Aviation manufacturing technical documents in Chinese",
        categories: [ner, multilingual, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Sports
    // =========================================================================

    VREN {
        name: "VREN (Volleyball)",
        description: "Volleyball rally descriptions for tactical statistics extraction.",
        url: "https://arxiv.org/html/2406.12252v1",
        entity_types: ["Player", "Action", "Position", "Team", "Score"],
        language: "en",
        domain: "sports",
        license: "CC-BY-4.0",
        citation: "VREN Team (2024)",
        paper_url: "https://arxiv.org/abs/2406.12252",
        year: 2024,
        format: "JSONL",
        notes: "Sports NLG; tactical action recognition from natural language",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Fashion & Retail
    // =========================================================================

    FashionIQ {
        name: "Fashion IQ",
        description: "77k fashion images with relative captions. 1000 attribute labels.",
        url: "https://github.com/XiaoxiaoGuo/fashion-iq",
        entity_types: ["Texture", "Fabric", "Shape", "Part", "Style", "Color"],
        language: "en",
        domain: "fashion",
        license: "Research",
        citation: "Wu et al. (2021)",
        paper_url: "https://users.cs.utah.edu/~ziad/papers/cvpr_2021_fashion_iq.pdf",
        year: 2021,
        format: "JSONL",
        size_hint: "77k images, 1000 attribute labels",
        notes: "Dialog-based fashion retrieval; fine-grained attribute extraction",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Biomedical Relation Extraction
    // =========================================================================

    NaturalProductsRE {
        name: "Natural Products RE",
        description: "Relation extraction in underexplored biomedical domains. Diversity-sampled entities.",
        url: "https://direct.mit.edu/coli/article/50/3/953/121178",
        entity_types: ["NaturalProduct", "Organism", "Activity", "Target"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "Hettiarachchi et al. (2024)",
        paper_url: "https://direct.mit.edu/coli/article/50/3/953/121178",
        year: 2024,
        format: "JSONL",
        notes: "LOTUS-derived NP dataset; synthetic data generation achieved F1=59.0",
        categories: [relation_extraction, biomedical],
    },

    DrugProtBioCreative {
        name: "DrugProt",
        description: "Chemical-protein interactions from BioCreative VII challenge.",
        url: "https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-1/",
        entity_types: ["Chemical", "Gene", "Protein"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "BioCreative VII (2021)",
        paper_url: "https://academic.oup.com/database/article/doi/10.1093/database/baac098/6833204",
        year: 2021,
        format: "BRAT",
        notes: "Drug-protein interaction classification; BioCreative shared task",
        categories: [relation_extraction, biomedical],
    },

    // =========================================================================
    // Niche Domains: Materials Science (Joint NER+RE)
    // =========================================================================

    MOFDataset {
        name: "MOF Dataset",
        description: "Metal-organic frameworks joint NER+RE. GPT-3/Llama extraction.",
        url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC10869356/",
        entity_types: ["MOF", "Linker", "Metal", "Property", "Application"],
        language: "en",
        domain: "materials",
        license: "CC-BY-4.0",
        citation: "MOF Team (2024)",
        paper_url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC10869356/",
        year: 2024,
        format: "JSONL",
        notes: "Metal-organic framework literature; LLM-based extraction pipeline",
        categories: [ner, relation_extraction, arcane_domain],
    },

    SolidStateDoping {
        name: "Solid-State Doping",
        description: "Impurity doping in materials. Joint NER+RE from literature.",
        url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC10869356/",
        entity_types: ["Host", "Dopant", "Property", "Concentration", "Method"],
        language: "en",
        domain: "materials",
        license: "CC-BY-4.0",
        citation: "Doping Team (2024)",
        paper_url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC10869356/",
        year: 2024,
        format: "JSONL",
        notes: "Semiconductor doping literature; tests materials science terminology",
        categories: [ner, relation_extraction, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Agriculture
    // =========================================================================

    AgriNER {
        name: "AgriNER",
        description: "Agricultural knowledge graph construction. 36 entity types, 9 relation types.",
        url: "https://2023.eswc-conferences.org/wp-content/uploads/2023/05/paper_De_2023_AgriNER.pdf",
        entity_types: ["Crop", "Disease", "Soil", "Pathogen", "Pesticide", "Product"],
        language: "en",
        domain: "agriculture",
        license: "Research",
        citation: "De et al. (2023)",
        paper_url: "https://2023.eswc-conferences.org/AgriNER/",
        year: 2023,
        format: "JSONL",
        notes: "Agricultural KG construction; covers crops, diseases, soil, pathogens",
        categories: [ner, relation_extraction, arcane_domain],
    },

    AGRONER {
        name: "AGRONER",
        description: "Unsupervised agricultural NER. Six major agricultural entity types.",
        url: "https://www.sciencedirect.com/science/article/abs/pii/S0957417423009429",
        entity_types: ["Disease", "Soil", "Pathogen", "Pesticide", "Crop", "Product"],
        language: "en",
        domain: "agriculture",
        license: "Research",
        citation: "AGRONER Team (2023)",
        paper_url: "https://doi.org/10.1016/j.eswa.2023.121001",
        year: 2023,
        format: "BIO",
        notes: "Unsupervised approach; no manual annotation required",
        categories: [ner, arcane_domain],
    },

    // NOTE: AgCNER exists earlier in file (Biomedical/Temporal section)

    AgMNER {
        name: "AgMNER",
        description: "Chinese multimodal agricultural NER. Text and speech combined.",
        url: "https://www.nature.com/articles/s41598-025-88874-9",
        entity_types: ["Crop", "Disease", "Pest", "Method"],
        language: "zh",
        domain: "agriculture",
        license: "CC-BY-4.0",
        citation: "AgMNER Team (2025)",
        paper_url: "https://www.nature.com/articles/s41598-025-88874-9",
        year: 2025,
        format: "JSONL",
        notes: "Multimodal NER; combines text and speech for agricultural domain",
        categories: [ner, multilingual, speech, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Polish Coreference
    // =========================================================================

    PolishCoreferenceCorpus {
        name: "Polish Coreference Corpus",
        description: "Polish coreference resolution corpus. General domain Polish text.",
        url: "http://zil.ipipan.waw.pl/PolishCoreferenceCorpus",
        entity_types: ["PER", "ORG", "LOC"],
        language: "pl",
        domain: "general",
        license: "CC-BY-SA-4.0",
        citation: "Ogrodniczuk et al. (2015)",
        year: 2015,
        format: "Custom",
        annotation_scheme: "Custom",
        notes: "Polish morphological complexity; rich inflection system",
        categories: [coref, multilingual],
    },

    // =========================================================================
    // Niche Domains: Arabic Event Coreference
    // =========================================================================

    ArabicEventCoref {
        name: "Arabic Event Coreference",
        description: "Arabic event coreference. Underexplored language for event coref.",
        url: "https://dl.acm.org/doi/10.1145/3743047",
        entity_types: ["Event", "Time", "Location", "Participant"],
        language: "ar",
        domain: "news",
        license: "Research",
        citation: "Arabic Event Coref Team (2024)",
        paper_url: "https://dl.acm.org/doi/10.1145/3743047",
        year: 2024,
        format: "CoNLL",
        annotation_scheme: "CoNLLCoref",
        notes: "Arabic event coreference; RTL script; underexplored language",
        categories: [coref, event_coref, multilingual],
    },

    // =========================================================================
    // Niche Domains: Code-Switching & Low-Resource
    // =========================================================================

    HindiEnglishSocialMediaNER {
        name: "Hindi-English Social Media NER",
        description: "Code-switched Hindi-English NER from social media.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["PER", "LOC", "ORG"],
        language: "hi-en",
        domain: "social_media",
        license: "Research",
        citation: "Hindi-English NER Team",
        year: 2018,
        format: "CoNLL",
        notes: "Code-switching between Hindi (Devanagari) and English; social media",
        categories: [ner, multilingual, social_media, low_resource],
    },

    // =========================================================================
    // Niche Domains: Astronomy & Space (Extended)
    // =========================================================================

    AstroBERTCorpus {
        name: "astroBERT Corpus",
        description: "Domain-specific BERT trained on 395k astronomical papers.",
        url: "https://arxiv.org/html/2310.17892v2",
        entity_types: ["CelestialObject", "Mission", "Instrument", "Phenomenon"],
        language: "en",
        domain: "astronomy",
        license: "Research",
        citation: "Grezes et al. (2023)",
        paper_url: "https://arxiv.org/abs/2310.17892",
        year: 2023,
        format: "Custom",
        size_hint: "395,499 astronomical papers",
        notes: "Domain-adapted BERT for astronomical entity extraction",
        categories: [ner, arcane_domain],
    },

    AstronomicalTelegramKEE {
        name: "Astronomical Telegram KEE",
        description: "Event IDs, object names, telescope names from GCN Circulars.",
        url: "https://www.raa-journal.org/issues/all/2024/v24n6/202405/",
        entity_types: ["EventID", "ObjectName", "TelescopeName", "Observatory"],
        language: "en",
        domain: "astronomy",
        license: "Research",
        citation: "KEE Team (2024)",
        paper_url: "https://www.raa-journal.org/issues/all/2024/v24n6/202405/",
        year: 2024,
        format: "JSONL",
        notes: "LLM extraction from GCN Circulars; astronomical event reports",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Music & Audio
    // =========================================================================

    Saraga {
        name: "Saraga",
        description: "Indian Art Music dataset. Carnatic and Hindustani traditions.",
        url: "https://arxiv.org/pdf/2309.16396.pdf",
        entity_types: ["Raaga", "Taala", "Artist", "Composition", "Instrument"],
        language: "mul",
        domain: "music",
        license: "CC-BY-4.0",
        citation: "Saraga Team (2023)",
        paper_url: "https://arxiv.org/abs/2309.16396",
        year: 2023,
        format: "JSONL",
        notes: "Indian classical music; Carnatic/Hindustani metadata extraction",
        categories: [ner, multilingual, arcane_domain],
    },

    MusicBrainzRE {
        name: "MusicBrainz RE",
        description: "Music metadata relations from Freebase/MusicBrainz. 116M instances.",
        url: "https://web.stanford.edu/~jurafsky/mintz.pdf",
        entity_types: ["Artist", "Album", "Track", "Label", "Genre"],
        language: "en",
        domain: "music",
        license: "CC0",
        citation: "Mintz et al. (2009)",
        paper_url: "https://web.stanford.edu/~jurafsky/mintz.pdf",
        year: 2009,
        format: "Custom",
        size_hint: "116 million instances, 7,300 binary relations",
        notes: "Distant supervision from Freebase; music metadata relations",
        categories: [relation_extraction, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Archaeology
    // =========================================================================

    DINAA {
        name: "DINAA",
        description: "Digital Index of North American Archaeology. Geospatial heritage data.",
        url: "https://ux.opencontext.org/endangered-data-and-the-digital-index-of-north-american-archaeology-dinaa/",
        entity_types: ["Site", "Artifact", "Culture", "Period", "Location"],
        language: "en",
        domain: "archaeology",
        license: "CC-BY-4.0",
        citation: "DINAA Team",
        year: 2015,
        format: "Custom",
        notes: "North American archaeological sites; geospatial heritage preservation",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Semi-Structured Web RE
    // =========================================================================

    IMDbSemiStructuredRE {
        name: "IMDb Semi-Structured RE",
        description: "Distantly supervised extraction from structured web content.",
        url: "https://www.vldb.org/pvldb/vol11/p1084-lockard.pdf",
        entity_types: ["Movie", "Person", "Role", "Date", "Award"],
        language: "en",
        domain: "entertainment",
        license: "Research",
        citation: "Lockard et al. (2018)",
        paper_url: "https://www.vldb.org/pvldb/vol11/p1084-lockard.pdf",
        year: 2018,
        format: "JSONL",
        notes: "Web table extraction; semi-structured movie database relations",
        categories: [relation_extraction, arcane_domain],
    },

    // NOTE: SciER exists earlier in file (Entity Linking section)

    // =========================================================================
    // Niche Domains: Slot Filling / Intent NER
    // =========================================================================

    ATISFlightBooking {
        name: "ATIS Flight Booking",
        description: "Slot-filling NER for flight booking intents. Classic NLU benchmark.",
        url: "https://github.com/yvchen/JointSLU",
        entity_types: ["FromCity", "ToCity", "DepartDate", "ReturnDate", "Airline", "FlightNumber"],
        language: "en",
        domain: "travel",
        license: "Research",
        citation: "Hemphill et al. (1990)",
        year: 1990,
        format: "BIO",
        notes: "Classic slot-filling benchmark; spoken language understanding",
        categories: [ner],
    },

    // =========================================================================
    // Niche Domains: Paleontology
    // =========================================================================

    PaleontologyNER {
        name: "Paleontology NER",
        description: "Dinosaurs, mammals, and river ecosystems entity retrieval.",
        url: "https://aclanthology.org/anthology-files/anthology-files/pdf/findings/2023.findings-emnlp.218v1.pdf",
        entity_types: ["Taxon", "Location", "TimePeriod", "Formation", "Specimen"],
        language: "en",
        domain: "paleontology",
        license: "Research",
        citation: "Paleo NER Team (2023)",
        paper_url: "https://aclanthology.org/2023.findings-emnlp.218/",
        year: 2023,
        format: "CoNLL",
        notes: "Paleontological literature; fossil taxa and geological formations",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Water Resources
    // =========================================================================

    WaterResourceNER {
        name: "Water Resource NER",
        description: "Domain-adaptive NER for AI-driven water resource management.",
        url: "https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2025.1558317/pdf",
        entity_types: ["WaterBody", "Infrastructure", "Pollutant", "Measurement", "Policy"],
        language: "en",
        domain: "environment",
        license: "CC-BY-4.0",
        citation: "Water NER Team (2025)",
        paper_url: "https://www.frontiersin.org/articles/10.3389/fenvs.2025.1558317/",
        year: 2025,
        format: "BIO",
        notes: "Water management domain; infrastructure and policy entities",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Cybersecurity
    // =========================================================================

    MalwareTextDB {
        name: "MalwareTextDB",
        description: "Annotated malware articles for cybersecurity NER.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Malware", "Vulnerability", "Tool", "ThreatActor", "IOC"],
        language: "en",
        domain: "cybersecurity",
        license: "Research",
        citation: "MalwareTextDB Team",
        year: 2017,
        format: "BRAT",
        notes: "Security bulletin extraction; malware family identification",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Finance
    // =========================================================================

    SECFilingsNER {
        name: "SEC-filings",
        description: "Finance domain NER from SEC filing documents.",
        url: "https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/SEC-filings/CONLL-format/data/test/FIN3.txt",
        entity_types: ["Company", "Person", "Money", "Date", "Percentage"],
        language: "en",
        domain: "finance",
        license: "CC-BY-3.0",
        citation: "SEC-filings Team",
        year: 2018,
        format: "CoNLL",
        notes: "Financial documents; SEC 10-K and 10-Q filings",
        categories: [ner],
    },

    // =========================================================================
    // Niche Domains: Anatomical/Biomedical
    // =========================================================================

    AnEM {
        name: "AnEM",
        description: "Anatomical entity mentions corpus. Anatomy terms in biomedical text.",
        url: "http://www.nactem.ac.uk/anatomy/",
        entity_types: ["AnatomicalStructure", "Organ", "Tissue", "Cell", "OrganismSubdivision"],
        language: "en",
        domain: "biomedical",
        license: "CC-BY-SA-3.0",
        citation: "Ohta et al. (2012)",
        year: 2012,
        format: "Standoff",
        notes: "Anatomical entity corpus; fine-grained anatomy typing",
        categories: [ner, biomedical],
    },

    // =========================================================================
    // Niche Domains: Recipe/Food (Extended)
    // =========================================================================

    RecipeDBAnnotated {
        name: "RecipeDB Annotated",
        description: "88k ingredient phrases via clustering-based sampling with Stanford NER.",
        url: "https://aclanthology.org/2024.lrec-main.406/",
        entity_types: ["Ingredient", "Quantity", "Unit", "Preparation"],
        language: "en",
        domain: "food",
        license: "CC-BY-4.0",
        citation: "RecipeDB Team (2024)",
        paper_url: "https://aclanthology.org/2024.lrec-main.406/",
        year: 2024,
        format: "JSONL",
        size_hint: "88,526 ingredient phrases",
        notes: "Clustering-based annotation; Stanford NER pipeline",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Social Media Twitter
    // =========================================================================

    RitterTwitterNER {
        name: "Ritter Twitter NER",
        description: "Twitter NER dataset with diverse entity types from tweets.",
        url: "https://github.com/aritter/twitter_nlp",
        entity_types: ["PER", "LOC", "ORG", "PRODUCT", "FACILITY", "BAND", "SPORTSTEAM"],
        language: "en",
        domain: "social_media",
        license: "Research",
        citation: "Ritter et al. (2011)",
        paper_url: "https://aclanthology.org/D11-1141/",
        year: 2011,
        format: "CoNLL",
        notes: "Early Twitter NER; 10 entity types including bands and sports teams",
        categories: [ner, social_media],
    },

    // =========================================================================
    // Niche Domains: Music Domain
    // =========================================================================

    MusicNER {
        name: "Music-NER",
        description: "Music domain entities. Artists, albums, songs, genres.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Artist", "Album", "Song", "Genre", "Instrument", "Label"],
        language: "en",
        domain: "music",
        license: "MIT",
        citation: "Music-NER Team",
        year: 2020,
        format: "CoNLL",
        notes: "Music domain NER; includes record labels and instrument types",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Tutoring/Education (Extended)
    // =========================================================================

    TutoringSessionsAlgebra {
        name: "500 Tutoring Sessions",
        description: "32k utterances from elementary algebra/physics tutoring. Mode identification.",
        url: "https://aclanthology.org/C16-1188.pdf",
        entity_types: ["Student", "Tutor", "Concept", "Problem"],
        language: "en",
        domain: "education",
        license: "Research",
        citation: "Boyer et al. (2016)",
        paper_url: "https://aclanthology.org/C16-1188/",
        year: 2016,
        format: "Custom",
        size_hint: "500 sessions, 32,368 utterances",
        notes: "Tutoring mode identification; algebra and physics domains",
        categories: [ner, dialogue, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Geology (Extended)
    // =========================================================================

    GNERGeoscience {
        name: "GNER",
        description: "Chinese geological entities from geoscience survey reports.",
        url: "https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019EA000610",
        entity_types: ["Rock", "Mineral", "Stratum", "Age", "Location"],
        language: "zh",
        domain: "geology",
        license: "Research",
        citation: "GNER Team (2019)",
        paper_url: "https://doi.org/10.1029/2019EA000610",
        year: 2019,
        format: "BIO",
        notes: "Chinese geoscience reports; geological survey terminology",
        categories: [ner, multilingual, arcane_domain],
    },

    FourRegionsGeologyNER {
        name: "Four Regions Geology NER",
        description: "Regional geological surveys with 6 typical geological categories.",
        url: "https://www.geodoi.ac.cn/WebEn/down.aspx?ID=1873",
        entity_types: ["Rock", "Mineral", "Stratum", "Structure", "Age", "Location"],
        language: "zh",
        domain: "geology",
        license: "Research",
        citation: "Four Regions Team",
        year: 2020,
        format: "BIO",
        notes: "Regional Chinese geological surveys; multiple survey regions",
        categories: [ner, multilingual, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Podcasts & Speech (Extended)
    // =========================================================================

    MSPPodcast {
        name: "MSP-Podcast",
        description: "100k+ English podcast episodes with multimodal annotations.",
        url: "https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html",
        entity_types: ["Speaker", "Topic", "Emotion", "Sentiment"],
        language: "en",
        domain: "speech",
        license: "Research",
        citation: "Lotfian & Busso (2019)",
        year: 2019,
        format: "Custom",
        size_hint: "100,000+ podcast episodes",
        notes: "Multimodal podcast annotations; emotion and sentiment",
        categories: [ner, speech, arcane_domain],
    },

    SpotifyPodcastsDataset {
        name: "Spotify Podcasts Dataset",
        description: "Professional and amateur podcast episodes with transcriptions.",
        url: "https://www.isca-archive.org/interspeech_2023/kotey23_interspeech.pdf",
        entity_types: ["Host", "Guest", "Topic", "Advertisement"],
        language: "en",
        domain: "speech",
        license: "Research",
        citation: "Spotify Research (2023)",
        paper_url: "https://www.isca-archive.org/interspeech_2023/kotey23_interspeech.html",
        year: 2023,
        format: "JSONL",
        notes: "Professional and amateur podcasts; varied audio quality",
        categories: [ner, speech, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Sports (Extended)
    // =========================================================================

    SportsNERGeneral {
        name: "Sports NER",
        description: "Player names, team names, event specifics from sports texts.",
        url: "https://arxiv.org/html/2406.12252v1",
        entity_types: ["Player", "Team", "Event", "Venue", "Score", "Date"],
        language: "en",
        domain: "sports",
        license: "Research",
        citation: "Sports NER Team (2024)",
        paper_url: "https://arxiv.org/abs/2406.12252",
        year: 2024,
        format: "CoNLL",
        notes: "General sports domain; player and team tracking",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: E-sports & Gaming Stats
    // =========================================================================

    EsportsNER {
        name: "Esports NER",
        description: "Esports entity recognition. Pro players, teams, tournaments, games.",
        url: "https://arxiv.org/html/2406.12252v1",
        entity_types: ["Player", "Team", "Tournament", "Game", "Champion", "Map"],
        language: "en",
        domain: "gaming",
        license: "Research",
        citation: "Esports NER Team (2024)",
        year: 2024,
        format: "CoNLL",
        notes: "Competitive gaming; League of Legends, CS:GO, Dota 2 terminology",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Fashion (Extended)
    // =========================================================================

    DeepFashion2 {
        name: "DeepFashion2",
        description: "Comprehensive fashion dataset. 491k images, 801k clothing items.",
        url: "https://github.com/switchablenorms/DeepFashion2",
        entity_types: ["Category", "Style", "Color", "Pattern", "Landmark"],
        language: "en",
        domain: "fashion",
        license: "Research",
        citation: "Ge et al. (2019)",
        paper_url: "https://arxiv.org/abs/1901.07973",
        year: 2019,
        format: "JSONL",
        size_hint: "491k images, 801k clothing items, 13 categories",
        notes: "Dense landmarks; cross-domain pose variation",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Construction & Engineering
    // =========================================================================

    ConstructionNER {
        name: "Construction NER",
        description: "Construction industry entities. Materials, equipment, processes.",
        url: "https://www.sciencedirect.com/science/article/pii/S0926580520309481",
        entity_types: ["Material", "Equipment", "Process", "Measurement", "Location"],
        language: "en",
        domain: "construction",
        license: "Research",
        citation: "Construction NER Team (2021)",
        year: 2021,
        format: "BIO",
        notes: "Construction domain; building materials and heavy equipment",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Pharmaceutical
    // =========================================================================

    PharmaNER {
        name: "PharmaNER",
        description: "Pharmaceutical named entity recognition. Drug names, dosages, routes.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Drug", "Dosage", "Route", "Frequency", "Indication"],
        language: "en",
        domain: "biomedical",
        license: "Research",
        citation: "PharmaNER Team",
        year: 2019,
        format: "BIO",
        notes: "Pharmaceutical domain; prescription and OTC drug extraction",
        categories: [ner, biomedical, clinical],
    },

    // =========================================================================
    // Niche Domains: E-commerce (Extended)
    // =========================================================================

    ProductReviewNER {
        name: "Product Review NER",
        description: "E-commerce product reviews with aspect and sentiment entities.",
        url: "https://www.aclweb.org/anthology/S14-2004/",
        entity_types: ["Aspect", "Opinion", "Product", "Feature", "Sentiment"],
        language: "en",
        domain: "ecommerce",
        license: "CC-BY-4.0",
        citation: "SemEval 2014",
        paper_url: "https://aclanthology.org/S14-2004/",
        year: 2014,
        format: "XML",
        notes: "Aspect-based sentiment; product feature extraction",
        categories: [ner],
    },

    // =========================================================================
    // Niche Domains: Real Estate
    // =========================================================================

    RealEstateNER {
        name: "Real Estate NER",
        description: "Property listings entity extraction. Addresses, prices, features.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Address", "Price", "Size", "Rooms", "Amenity", "PropertyType"],
        language: "en",
        domain: "real_estate",
        license: "Research",
        citation: "Real Estate NER Team",
        year: 2020,
        format: "CoNLL",
        notes: "Property listing domain; residential and commercial",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Automotive
    // =========================================================================

    AutomotiveNER {
        name: "Automotive NER",
        description: "Vehicle and automotive entities. Makes, models, parts, specs.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Make", "Model", "Part", "Specification", "Year", "Price"],
        language: "en",
        domain: "automotive",
        license: "Research",
        citation: "Automotive NER Team",
        year: 2021,
        format: "CoNLL",
        notes: "Automotive domain; vehicle specifications and parts",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Tourism & Travel
    // =========================================================================

    TourismNER {
        name: "Tourism NER",
        description: "Tourism and travel entities. Attractions, hotels, restaurants.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Attraction", "Hotel", "Restaurant", "City", "Activity", "Price"],
        language: "en",
        domain: "tourism",
        license: "CC-BY-4.0",
        citation: "Tourism NER Team",
        year: 2019,
        format: "CoNLL",
        notes: "Travel domain; tourist attractions and accommodations",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Energy & Utilities
    // =========================================================================

    EnergyNER {
        name: "Energy NER",
        description: "Energy sector entities. Power plants, fuels, grid infrastructure.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["PowerPlant", "Fuel", "Grid", "Capacity", "Company", "Location"],
        language: "en",
        domain: "energy",
        license: "Research",
        citation: "Energy NER Team",
        year: 2020,
        format: "BIO",
        notes: "Energy sector; renewable and fossil fuel infrastructure",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Insurance
    // =========================================================================

    InsuranceNER {
        name: "Insurance NER",
        description: "Insurance domain entities. Policies, claims, coverages.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Policy", "Claim", "Coverage", "Premium", "Deductible", "Beneficiary"],
        language: "en",
        domain: "insurance",
        license: "Research",
        citation: "Insurance NER Team",
        year: 2021,
        format: "JSONL",
        notes: "Insurance domain; policy and claims extraction",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Logistics & Supply Chain
    // =========================================================================

    LogisticsNER {
        name: "Logistics NER",
        description: "Supply chain and logistics entities. Shipments, warehouses, routes.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Shipment", "Warehouse", "Route", "Carrier", "TrackingNumber", "Date"],
        language: "en",
        domain: "logistics",
        license: "Research",
        citation: "Logistics NER Team",
        year: 2020,
        format: "CoNLL",
        notes: "Supply chain domain; shipping and warehousing",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: HR & Recruitment
    // =========================================================================

    ResumeNER {
        name: "Resume NER",
        description: "Resume/CV entity extraction. Skills, experience, education.",
        url: "https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner",
        entity_types: ["Skill", "Company", "Degree", "University", "Date", "Location"],
        language: "en",
        domain: "hr",
        license: "CC0",
        citation: "DataTurks",
        year: 2018,
        format: "JSONL",
        notes: "Resume parsing; skill and experience extraction",
        categories: [ner],
    },

    JobPostingNER {
        name: "Job Posting NER",
        description: "Job posting entity extraction. Requirements, benefits, qualifications.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["JobTitle", "Skill", "Salary", "Location", "Company", "Benefit"],
        language: "en",
        domain: "hr",
        license: "Research",
        citation: "Job Posting NER Team",
        year: 2020,
        format: "CoNLL",
        notes: "Job listing domain; requirement and qualification extraction",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Healthcare Administration
    // =========================================================================

    HealthcareAdminNER {
        name: "Healthcare Admin NER",
        description: "Healthcare administration entities. Procedures, billing codes, facilities.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Procedure", "BillingCode", "Facility", "Provider", "Insurance"],
        language: "en",
        domain: "healthcare",
        license: "Research",
        citation: "Healthcare Admin Team",
        year: 2021,
        format: "BIO",
        notes: "Healthcare administration; billing and coding",
        categories: [ner, clinical, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Telecommunications
    // =========================================================================

    TelecomNER {
        name: "Telecom NER",
        description: "Telecommunications entities. Networks, devices, protocols.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Network", "Device", "Protocol", "Carrier", "Plan", "Speed"],
        language: "en",
        domain: "telecom",
        license: "Research",
        citation: "Telecom NER Team",
        year: 2020,
        format: "CoNLL",
        notes: "Telecommunications domain; network and service extraction",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Weather & Climate
    // =========================================================================

    WeatherNER {
        name: "Weather NER",
        description: "Weather and climate entities. Events, measurements, locations.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["WeatherEvent", "Temperature", "Precipitation", "Location", "Date", "Wind"],
        language: "en",
        domain: "weather",
        license: "CC-BY-4.0",
        citation: "Weather NER Team",
        year: 2021,
        format: "BIO",
        notes: "Meteorological domain; weather event extraction",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Manufacturing
    // =========================================================================

    ManufacturingNER {
        name: "Manufacturing NER",
        description: "Manufacturing entities. Parts, processes, machines, defects.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Part", "Process", "Machine", "Defect", "Material", "Measurement"],
        language: "en",
        domain: "manufacturing",
        license: "Research",
        citation: "Manufacturing NER Team",
        year: 2021,
        format: "BIO",
        notes: "Industrial manufacturing; quality control and process",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Retail & Inventory
    // =========================================================================

    RetailInventoryNER {
        name: "Retail Inventory NER",
        description: "Retail inventory entities. SKUs, quantities, locations, prices.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["SKU", "Quantity", "Location", "Price", "Category", "Supplier"],
        language: "en",
        domain: "retail",
        license: "Research",
        citation: "Retail NER Team",
        year: 2020,
        format: "JSONL",
        notes: "Inventory management; stock and supplier tracking",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Agriculture (Extended)
    // =========================================================================

    CropDiseaseNER {
        name: "Crop Disease NER",
        description: "Crop disease identification. Symptoms, pathogens, treatments.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Disease", "Symptom", "Pathogen", "Treatment", "Crop", "Stage"],
        language: "en",
        domain: "agriculture",
        license: "CC-BY-4.0",
        citation: "Crop Disease Team",
        year: 2022,
        format: "BIO",
        notes: "Plant pathology; disease symptom and treatment extraction",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Wine & Beverages
    // =========================================================================

    WineNER {
        name: "Wine NER",
        description: "Wine domain entities. Varietals, regions, vintages, tasting notes.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Varietal", "Region", "Vintage", "Producer", "TastingNote", "Price"],
        language: "en",
        domain: "food",
        license: "CC-BY-4.0",
        citation: "Wine NER Team",
        year: 2019,
        format: "CoNLL",
        notes: "Wine domain; sommelier terminology and tasting vocabulary",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Pet & Veterinary
    // =========================================================================

    VeterinaryNER {
        name: "Veterinary NER",
        description: "Veterinary medicine entities. Animals, conditions, treatments.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Animal", "Breed", "Condition", "Treatment", "Medication", "Symptom"],
        language: "en",
        domain: "veterinary",
        license: "Research",
        citation: "Veterinary NER Team",
        year: 2021,
        format: "BIO",
        notes: "Veterinary medicine; pet health and treatment",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Photography
    // =========================================================================

    PhotographyNER {
        name: "Photography NER",
        description: "Photography entities. Cameras, lenses, settings, techniques.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Camera", "Lens", "Aperture", "ShutterSpeed", "ISO", "Technique"],
        language: "en",
        domain: "photography",
        license: "CC-BY-4.0",
        citation: "Photography NER Team",
        year: 2020,
        format: "CoNLL",
        notes: "Photography domain; camera gear and technique extraction",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Genealogy
    // =========================================================================

    GenealogyNER {
        name: "Genealogy NER",
        description: "Genealogical records entities. Names, relationships, dates, locations.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Person", "Relationship", "BirthDate", "DeathDate", "Location", "Occupation"],
        language: "en",
        domain: "genealogy",
        license: "CC-BY-4.0",
        citation: "Genealogy NER Team",
        year: 2021,
        format: "Custom",
        notes: "Historical records; family history extraction",
        categories: [ner, historical, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Board Games
    // =========================================================================

    BoardGameNER {
        name: "Board Game NER",
        description: "Board game entities. Games, mechanics, components, designers.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Game", "Mechanic", "Component", "Designer", "Publisher", "PlayerCount"],
        language: "en",
        domain: "gaming",
        license: "CC-BY-4.0",
        citation: "BoardGameGeek",
        year: 2022,
        format: "JSONL",
        notes: "Board game domain; BGG taxonomy and mechanics",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Gardening
    // =========================================================================

    GardeningNER {
        name: "Gardening NER",
        description: "Gardening entities. Plants, soil, seasons, techniques.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Plant", "Soil", "Season", "Technique", "Tool", "Pest"],
        language: "en",
        domain: "gardening",
        license: "CC-BY-4.0",
        citation: "Gardening NER Team",
        year: 2021,
        format: "CoNLL",
        notes: "Horticulture domain; plant care and cultivation",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Brewing & Distilling
    // =========================================================================

    BrewingNER {
        name: "Brewing NER",
        description: "Craft brewing entities. Ingredients, processes, styles, equipment.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Ingredient", "Process", "Style", "Equipment", "ABV", "IBU"],
        language: "en",
        domain: "food",
        license: "CC-BY-4.0",
        citation: "Brewing NER Team",
        year: 2020,
        format: "CoNLL",
        notes: "Craft beer domain; brewing process and style vocabulary",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Knitting & Crafts
    // =========================================================================

    KnittingNER {
        name: "Knitting NER",
        description: "Knitting and crafts entities. Patterns, yarns, stitches, tools.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Pattern", "Yarn", "Stitch", "Tool", "Size", "Technique"],
        language: "en",
        domain: "crafts",
        license: "CC-BY-4.0",
        citation: "Ravelry",
        year: 2021,
        format: "JSONL",
        notes: "Fiber arts domain; knitting pattern terminology",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Fitness & Exercise
    // =========================================================================

    FitnessNER {
        name: "Fitness NER",
        description: "Fitness entities. Exercises, muscles, equipment, routines.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Exercise", "Muscle", "Equipment", "Sets", "Reps", "Duration"],
        language: "en",
        domain: "fitness",
        license: "CC-BY-4.0",
        citation: "Fitness NER Team",
        year: 2021,
        format: "CoNLL",
        notes: "Exercise domain; workout routine extraction",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Astrology
    // =========================================================================

    AstrologyNER {
        name: "Astrology NER",
        description: "Astrological entities. Signs, planets, houses, aspects.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Sign", "Planet", "House", "Aspect", "Transit", "Date"],
        language: "en",
        domain: "astrology",
        license: "CC-BY-4.0",
        citation: "Astrology NER Team",
        year: 2021,
        format: "CoNLL",
        notes: "Astrological terminology; horoscope interpretation",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Tattoo & Body Art
    // =========================================================================

    TattooNER {
        name: "Tattoo NER",
        description: "Tattoo entities. Styles, placements, artists, designs.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Style", "Placement", "Artist", "Design", "Color", "Size"],
        language: "en",
        domain: "art",
        license: "CC-BY-4.0",
        citation: "Tattoo NER Team",
        year: 2022,
        format: "JSONL",
        notes: "Body art domain; tattoo style and placement vocabulary",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Perfume & Fragrance
    // =========================================================================

    FragranceNER {
        name: "Fragrance NER",
        description: "Perfume entities. Notes, accords, houses, concentrations.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Note", "Accord", "House", "Concentration", "Season", "Longevity"],
        language: "en",
        domain: "fragrance",
        license: "CC-BY-4.0",
        citation: "Fragrantica",
        year: 2021,
        format: "JSONL",
        notes: "Perfumery domain; scent pyramid and accord terminology",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Chess
    // =========================================================================

    ChessNER {
        name: "Chess NER",
        description: "Chess entities. Openings, players, tournaments, moves.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Opening", "Player", "Tournament", "Move", "ELO", "TimeControl"],
        language: "en",
        domain: "gaming",
        license: "CC-BY-4.0",
        citation: "Lichess/Chess.com",
        year: 2022,
        format: "JSONL",
        notes: "Chess domain; opening theory and tournament extraction",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Cocktails & Mixology
    // =========================================================================

    CocktailNER {
        name: "Cocktail NER",
        description: "Cocktail entities. Ingredients, techniques, glassware, garnishes.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Spirit", "Mixer", "Technique", "Glassware", "Garnish", "Measurement"],
        language: "en",
        domain: "food",
        license: "CC-BY-4.0",
        citation: "Cocktail NER Team",
        year: 2020,
        format: "CoNLL",
        notes: "Mixology domain; bartending vocabulary and techniques",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Antiques & Collectibles
    // =========================================================================

    AntiquesNER {
        name: "Antiques NER",
        description: "Antiques entities. Periods, styles, materials, makers.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Period", "Style", "Material", "Maker", "Provenance", "Condition"],
        language: "en",
        domain: "antiques",
        license: "CC-BY-4.0",
        citation: "Antiques NER Team",
        year: 2021,
        format: "JSONL",
        notes: "Antiques domain; period furniture and collectibles",
        categories: [ner, historical, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Sailing & Maritime
    // =========================================================================

    MaritimeNER {
        name: "Maritime NER",
        description: "Maritime entities. Vessels, ports, routes, cargo.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Vessel", "Port", "Route", "Cargo", "Flag", "IMONumber"],
        language: "en",
        domain: "maritime",
        license: "Research",
        citation: "Maritime NER Team",
        year: 2021,
        format: "CoNLL",
        notes: "Shipping domain; vessel tracking and maritime logistics",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Equestrian
    // =========================================================================

    EquestrianNER {
        name: "Equestrian NER",
        description: "Equestrian entities. Horses, breeds, disciplines, tack.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Horse", "Breed", "Discipline", "Tack", "Rider", "Competition"],
        language: "en",
        domain: "equestrian",
        license: "CC-BY-4.0",
        citation: "Equestrian NER Team",
        year: 2021,
        format: "CoNLL",
        notes: "Horse sports domain; dressage and jumping terminology",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Woodworking
    // =========================================================================

    WoodworkingNER {
        name: "Woodworking NER",
        description: "Woodworking entities. Tools, joints, wood types, finishes.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Tool", "Joint", "WoodType", "Finish", "Technique", "Measurement"],
        language: "en",
        domain: "crafts",
        license: "CC-BY-4.0",
        citation: "Woodworking NER Team",
        year: 2021,
        format: "CoNLL",
        notes: "Carpentry domain; joinery and finishing vocabulary",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Birdwatching
    // =========================================================================

    BirdwatchingNER {
        name: "Birdwatching NER",
        description: "Birdwatching entities. Species, habitats, behaviors, locations.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Species", "Family", "Habitat", "Behavior", "Location", "Season"],
        language: "en",
        domain: "wildlife",
        license: "CC-BY-4.0",
        citation: "eBird/Cornell Lab",
        year: 2022,
        format: "JSONL",
        notes: "Ornithology domain; bird identification and behavior",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Numismatics (Coins)
    // =========================================================================

    NumismaticsNER {
        name: "Numismatics NER",
        description: "Coin collecting entities. Denominations, mints, grades, errors.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Denomination", "Mint", "Grade", "Error", "Year", "Metal"],
        language: "en",
        domain: "numismatics",
        license: "CC-BY-4.0",
        citation: "PCGS/NGC",
        year: 2021,
        format: "JSONL",
        notes: "Coin collecting; grading and mint terminology",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Philately (Stamps)
    // =========================================================================

    PhilatelyNER {
        name: "Philately NER",
        description: "Stamp collecting entities. Issues, perforations, watermarks, varieties.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Issue", "Perforation", "Watermark", "Variety", "Country", "Year"],
        language: "en",
        domain: "philately",
        license: "CC-BY-4.0",
        citation: "Scott Catalogue",
        year: 2021,
        format: "JSONL",
        notes: "Stamp collecting; philatelic terminology",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Scuba Diving
    // =========================================================================

    ScubaNER {
        name: "Scuba NER",
        description: "Scuba diving entities. Equipment, sites, certifications, marine life.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Equipment", "DiveSite", "Certification", "MarineLife", "Depth", "Visibility"],
        language: "en",
        domain: "scuba",
        license: "CC-BY-4.0",
        citation: "PADI/SSI",
        year: 2021,
        format: "CoNLL",
        notes: "Recreational diving; dive site and equipment extraction",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Roller Coasters & Theme Parks
    // =========================================================================

    ThemeParkNER {
        name: "Theme Park NER",
        description: "Theme park entities. Rides, parks, manufacturers, statistics.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Ride", "Park", "Manufacturer", "Height", "Speed", "Type"],
        language: "en",
        domain: "entertainment",
        license: "CC-BY-4.0",
        citation: "RCDB",
        year: 2022,
        format: "JSONL",
        notes: "Amusement park domain; roller coaster specifications",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Origami
    // =========================================================================

    OrigamiNER {
        name: "Origami NER",
        description: "Origami entities. Folds, bases, models, paper types.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Fold", "Base", "Model", "PaperType", "Designer", "Difficulty"],
        language: "en",
        domain: "crafts",
        license: "CC-BY-4.0",
        citation: "Origami NER Team",
        year: 2021,
        format: "CoNLL",
        notes: "Paper folding domain; fold terminology and model names",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Anime & Manga
    // =========================================================================

    AnimeMangaNER {
        name: "Anime/Manga NER",
        description: "Anime and manga entities. Titles, characters, studios, genres.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Title", "Character", "Studio", "Genre", "Author", "Year"],
        language: "mul",
        domain: "entertainment",
        license: "CC-BY-4.0",
        citation: "MyAnimeList/AniDB",
        year: 2022,
        format: "JSONL",
        notes: "Japanese animation; includes romanized and Japanese names",
        categories: [ner, multilingual, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Blockchain & Cryptocurrency
    // =========================================================================

    CryptoNER {
        name: "Crypto NER",
        description: "Cryptocurrency entities. Tokens, wallets, exchanges, protocols.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Token", "Wallet", "Exchange", "Protocol", "Price", "Address"],
        language: "en",
        domain: "crypto",
        license: "Research",
        citation: "Crypto NER Team",
        year: 2022,
        format: "CoNLL",
        notes: "Blockchain domain; DeFi and NFT terminology",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Soap Opera / Telenovela
    // =========================================================================

    TelenovelaNER {
        name: "Telenovela NER",
        description: "Spanish-language soap opera entities. Characters, relationships, plots.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Character", "Relationship", "PlotPoint", "Actor", "Network"],
        language: "es",
        domain: "entertainment",
        license: "CC-BY-4.0",
        citation: "Telenovela NER Team",
        year: 2021,
        format: "CoNLL",
        notes: "Spanish soap operas; melodrama terminology",
        categories: [ner, multilingual, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Tarot & Divination
    // =========================================================================

    TarotNER {
        name: "Tarot NER",
        description: "Tarot entities. Cards, spreads, meanings, suits.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Card", "Spread", "Meaning", "Suit", "Position", "Reversal"],
        language: "en",
        domain: "divination",
        license: "CC-BY-4.0",
        citation: "Tarot NER Team",
        year: 2021,
        format: "CoNLL",
        notes: "Tarot reading; card interpretation vocabulary",
        categories: [ner, arcane_domain],
    },

    // =========================================================================
    // Niche Domains: Beekeeping
    // =========================================================================

    BeekeepingNER {
        name: "Beekeeping NER",
        description: "Apiculture entities. Equipment, bee types, diseases, products.",
        url: "https://github.com/juand-r/entity-recognition-datasets",
        entity_types: ["Equipment", "BeeType", "Disease", "Product", "Season", "Technique"],
        language: "en",
        domain: "agriculture",
        license: "CC-BY-4.0",
        citation: "Beekeeping NER Team",
        year: 2021,
        format: "CoNLL",
        notes: "Apiculture domain; hive management vocabulary",
        categories: [ner, arcane_domain],
    },
}

#[allow(clippy::items_after_test_module)]
impl DatasetId {
    /// Get tasks with fallback to category-based inference.
    ///
    /// Many datasets don't have explicit `tasks` defined but have task-like
    /// categories (ner, coref, relation_extraction). This method returns
    /// explicit tasks if defined, otherwise infers from categories.
    ///
    /// # Migration Note
    ///
    /// This exists for backwards compatibility. New datasets should use
    /// the `tasks` field explicitly. Categories should be domain/property
    /// focused (biomedical, multilingual, nested) not task-like.
    #[must_use]
    pub fn tasks_or_inferred(&self) -> Vec<&'static str> {
        let explicit = self.tasks();
        if !explicit.is_empty() {
            return explicit.to_vec();
        }

        // Infer from categories
        let mut inferred = Vec::new();
        if self.is_ner() {
            inferred.push("ner");
        }
        if self.is_coreference() {
            inferred.push("coref");
        }
        if self.is_relation_extraction() {
            inferred.push("re");
        }
        if self.is_entity_linking() {
            inferred.push("el");
        }
        if self.is_event_coref() {
            inferred.push("event_coref");
        }

        // Default to NER for datasets with entity types but no category
        if inferred.is_empty() && !self.entity_types().is_empty() {
            inferred.push("ner");
        }

        inferred
    }

    /// Check if this is a NER dataset (explicit or inferred).
    #[must_use]
    pub fn supports_ner(&self) -> bool {
        self.tasks_or_inferred().contains(&"ner")
    }

    /// Check if this is a coreference dataset (explicit or inferred).
    #[must_use]
    pub fn supports_coref(&self) -> bool {
        self.tasks_or_inferred().contains(&"coref")
    }

    /// Check if this is a relation extraction dataset (explicit or inferred).
    #[must_use]
    pub fn supports_re(&self) -> bool {
        self.tasks_or_inferred().contains(&"re")
    }

    /// Whether this dataset is known to require `HF_TOKEN` (or equivalent) to download.
    ///
    /// This is used to keep “automatic” workflows (cache warming, randomized matrices)
    /// usable by default while still allowing explicit attempts when a token is provided.
    #[must_use]
    pub fn requires_hf_token(&self) -> bool {
        matches!(
            self,
            Self::BookCorefBamman | Self::MultiCoNER | Self::MultiCoNERv2
        )
    }

    /// Whether this dataset is currently automatable by our loader without manual intervention.
    ///
    /// This is intentionally conservative: it can be `false` for datasets that are “public”
    /// but do not offer a stable export path we can consume yet.
    #[must_use]
    pub fn is_automatable_download(&self) -> bool {
        // Hard exclusions (known to fail under current loader automation).
        //
        // These datasets have HuggingFace or direct URLs that pass the heuristic checks
        // below but fail at the loader level (HTML response, API parse error, empty result,
        // gated access).  Discovered by the Estimate strategy's systematic coverage sweep.
        if matches!(
            self,
            Self::MasakhaNER2
                | Self::PIIMasking200k
                | Self::CADEC
                // HF datasets-server failures (HTML response or API parse errors):
                | Self::CoNLL2002
                | Self::DBPedia14
                | Self::MultiCoNER
                | Self::PolyglotNER
                | Self::TREC
                | Self::UniversalNER
                | Self::WikiNeural
                // Downloads OK but parser yields 0 sentences:
                | Self::TweetNER7
        ) {
            return false;
        }

        if !self.access_status().is_automatable() {
            return false;
        }

        let url = self.download_url();
        if url.is_empty() {
            return false;
        }

        // URLs we explicitly support via rewrites/special handling in the loader.
        if url.contains("huggingface.co/datasets/")
            || url.contains("datasets-server.huggingface.co/")
        {
            return true;
        }

        // GitHub: only treat as automatable if it's a raw/archive asset, not a repo homepage.
        if url.contains("github.com/")
            && (url.contains("/raw/")
                || url.contains("/archive/")
                || url.contains("/releases/download/"))
        {
            return true;
        }

        // Direct file downloads.
        const EXT_OK: [&str; 10] = [
            ".jsonl", ".json", ".tsv", ".csv", ".conll", ".conllu", ".txt", ".zip", ".tar.gz",
            ".tgz",
        ];
        EXT_OK.iter().any(|ext| url.ends_with(ext))
    }
}
