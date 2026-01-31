//! Table-driven dataset metadata.
//!
//! This module provides a single source of truth for all dataset metadata,
//! replacing the 15+ separate `match` statements that previously existed.
//!
//! # Design
//!
//! Instead of:
//! ```rust,ignore
//! fn domain(&self) -> &'static str {
//!     match self {
//!         DatasetId::WikiGold => "news",
//!         DatasetId::BC5CDR => "biomedical",
//!         // ... 450+ more cases
//!     }
//! }
//! ```
//!
//! We use:
//! ```rust,ignore
//! fn domain(&self) -> &'static str {
//!     METADATA[self.index()].domain
//! }
//! ```
//!
//! This reduces ~18K lines to ~3K lines and makes adding datasets O(1) instead of O(n).

use bitflags::bitflags;

bitflags! {
    /// Dataset capability/category flags.
    ///
    /// Using bitflags allows O(1) category checks and easy combination queries.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DatasetFlags: u32 {
        /// Named Entity Recognition dataset
        const NER = 1 << 0;
        /// Coreference resolution dataset
        const COREFERENCE = 1 << 1;
        /// Intra-document coreference
        const INTRA_DOC_COREF = 1 << 2;
        /// Cross-document / inter-document coreference
        const INTER_DOC_COREF = 1 << 3;
        /// Temporal NER (dates, durations, temporal expressions)
        const TEMPORAL_NER = 1 << 4;
        /// Biomedical/clinical domain
        const BIOMEDICAL = 1 << 5;
        /// Social media domain (Twitter, Reddit, etc.)
        const SOCIAL_MEDIA = 1 << 6;
        /// Specialized domain (not general news)
        const SPECIALIZED_DOMAIN = 1 << 7;
        /// Relation extraction dataset
        const RELATION_EXTRACTION = 1 << 8;
        /// Historical texts
        const HISTORICAL = 1 << 9;
        /// Bias evaluation dataset
        const BIAS_EVALUATION = 1 << 10;
        /// Dialogue/conversational coreference
        const DIALOGUE_COREF = 1 << 11;
        /// Joint NER + Relation Extraction
        const JOINT_NER_RE = 1 << 12;
        /// Discontinuous/nested NER
        const DISCONTINUOUS_NER = 1 << 13;
        /// Few-shot learning dataset
        const FEW_SHOT = 1 << 14;
        /// Multilingual dataset
        const MULTILINGUAL = 1 << 15;
        /// Constructed/artificial language
        const CONSTRUCTED_LANGUAGE = 1 << 16;
        /// Code-switching dataset
        const CODE_SWITCHING = 1 << 17;
        /// African language dataset
        const AFRICAN_LANGUAGE = 1 << 18;
        /// Entity linking dataset
        const ENTITY_LINKING = 1 << 19;
        /// Event extraction dataset
        const EVENT_EXTRACTION = 1 << 20;
        /// Legal domain
        const LEGAL = 1 << 21;
        /// Financial domain
        const FINANCIAL = 1 << 22;
        /// Scientific domain
        const SCIENTIFIC = 1 << 23;
        /// Literary/fiction domain
        const LITERARY = 1 << 24;
        /// News domain
        const NEWS = 1 << 25;
        /// Low-resource language
        const LOW_RESOURCE = 1 << 26;
    }
}

impl Default for DatasetFlags {
    fn default() -> Self {
        Self::NER
    }
}

/// Complete metadata for a single dataset.
///
/// All fields are static references for zero-cost access.
#[derive(Debug, Clone, Copy)]
pub struct DatasetMetadata {
    /// Human-readable name
    pub name: &'static str,
    /// Short description (1-2 sentences)
    pub description: &'static str,
    /// Download URL (HuggingFace, GitHub, etc.)
    pub download_url: &'static str,
    /// Primary domain (news, biomedical, social-media, etc.)
    pub domain: &'static str,
    /// ISO 639-1/3 language code (en, de, zh, multilingual, etc.)
    pub language: &'static str,
    /// Entity types in this dataset
    pub entity_types: &'static [&'static str],
    /// Capability flags (replaces all is_* methods)
    pub flags: DatasetFlags,
    /// Academic citation (if available)
    pub citation: Option<&'static str>,
    /// License (MIT, CC-BY, etc.)
    pub license: Option<&'static str>,
    /// Publication year
    pub year: Option<u16>,
    /// Paper URL (arXiv, ACL Anthology, etc.)
    pub paper_url: Option<&'static str>,
}

impl DatasetMetadata {
    /// Create metadata with required fields only.
    #[must_use]
    pub const fn new(
        name: &'static str,
        description: &'static str,
        download_url: &'static str,
    ) -> Self {
        Self {
            name,
            description,
            download_url,
            domain: "general",
            language: "en",
            entity_types: &[],
            flags: DatasetFlags::NER,
            citation: None,
            license: None,
            year: None,
            paper_url: None,
        }
    }

    /// Builder: set domain.
    #[must_use]
    pub const fn domain(mut self, domain: &'static str) -> Self {
        self.domain = domain;
        self
    }

    /// Builder: set language.
    #[must_use]
    pub const fn language(mut self, language: &'static str) -> Self {
        self.language = language;
        self
    }

    /// Builder: set entity types.
    #[must_use]
    pub const fn entity_types(mut self, types: &'static [&'static str]) -> Self {
        self.entity_types = types;
        self
    }

    /// Builder: set flags.
    #[must_use]
    pub const fn flags(mut self, flags: DatasetFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Builder: set citation.
    #[must_use]
    pub const fn citation(mut self, citation: &'static str) -> Self {
        self.citation = Some(citation);
        self
    }

    /// Builder: set license.
    #[must_use]
    pub const fn license(mut self, license: &'static str) -> Self {
        self.license = Some(license);
        self
    }

    /// Builder: set year.
    #[must_use]
    pub const fn year(mut self, year: u16) -> Self {
        self.year = Some(year);
        self
    }

    /// Builder: set paper URL.
    #[must_use]
    pub const fn paper_url(mut self, url: &'static str) -> Self {
        self.paper_url = Some(url);
        self
    }

    // Flag query methods (all O(1))

    /// Returns `true` if this dataset supports named entity recognition.
    ///
    /// NER datasets provide entity span annotations with type labels.
    #[inline]
    pub const fn is_ner(&self) -> bool {
        self.flags.contains(DatasetFlags::NER)
    }

    /// Returns `true` if this dataset contains coreference annotations.
    ///
    /// Coreference datasets provide mention clusters or entity chains
    /// indicating which mentions refer to the same entity.
    #[inline]
    pub const fn is_coreference(&self) -> bool {
        self.flags.contains(DatasetFlags::COREFERENCE)
    }

    /// Returns `true` if this dataset contains within-document coreference.
    ///
    /// Intra-document coreference datasets link mentions within a single
    /// document, forming coreference chains or clusters.
    #[inline]
    pub const fn is_intra_doc_coref(&self) -> bool {
        self.flags.contains(DatasetFlags::INTRA_DOC_COREF)
    }

    /// Returns `true` if this dataset contains cross-document coreference.
    ///
    /// Inter-document coreference datasets link entities across multiple
    /// documents, enabling cross-document entity resolution.
    #[inline]
    pub const fn is_inter_doc_coref(&self) -> bool {
        self.flags.contains(DatasetFlags::INTER_DOC_COREF)
    }

    /// Returns `true` if this dataset contains temporal entity annotations.
    ///
    /// Temporal NER datasets include time expressions, events with temporal
    /// anchors, or entities with temporal attributes (e.g., historical entities).
    #[inline]
    pub const fn is_temporal_ner(&self) -> bool {
        self.flags.contains(DatasetFlags::TEMPORAL_NER)
    }

    /// Returns `true` if this dataset is from the biomedical domain.
    ///
    /// Biomedical datasets include medical, clinical, or life science text
    /// with specialized entity types (diseases, genes, proteins, chemicals).
    #[inline]
    pub const fn is_biomedical(&self) -> bool {
        self.flags.contains(DatasetFlags::BIOMEDICAL)
    }

    /// Returns `true` if this dataset contains social media text.
    ///
    /// Social media datasets include Twitter, Reddit, or other informal text
    /// with non-standard capitalization, abbreviations, and informal language.
    #[inline]
    pub const fn is_social_media(&self) -> bool {
        self.flags.contains(DatasetFlags::SOCIAL_MEDIA)
    }

    /// Returns `true` if this dataset is from a specialized domain.
    ///
    /// Specialized domain datasets include technical, legal, financial, or
    /// other domain-specific text requiring specialized entity recognition.
    #[inline]
    pub const fn is_specialized_domain(&self) -> bool {
        self.flags.contains(DatasetFlags::SPECIALIZED_DOMAIN)
    }

    /// Returns `true` if this dataset supports relation extraction.
    ///
    /// Relation extraction datasets provide entity-relation-entity triples,
    /// enabling evaluation of models that extract structured relationships.
    #[inline]
    pub const fn is_relation_extraction(&self) -> bool {
        self.flags.contains(DatasetFlags::RELATION_EXTRACTION)
    }

    /// Returns `true` if this dataset contains historical text.
    ///
    /// Historical datasets include ancient texts, historical documents, or
    /// diachronic corpora that test model robustness to language evolution.
    #[inline]
    pub const fn is_historical(&self) -> bool {
        self.flags.contains(DatasetFlags::HISTORICAL)
    }

    /// Returns `true` if this dataset is designed for bias evaluation.
    ///
    /// Bias evaluation datasets test for gender, demographic, or other biases
    /// in entity recognition and coreference resolution.
    #[inline]
    pub const fn is_bias_evaluation(&self) -> bool {
        self.flags.contains(DatasetFlags::BIAS_EVALUATION)
    }

    /// Returns `true` if this dataset contains dialogue coreference annotations.
    ///
    /// Dialogue coreference datasets include multi-party conversations, meetings,
    /// or interviews where coreference resolution must handle speaker turns and
    /// dialogue-specific phenomena (e.g., prosody, gestures).
    #[inline]
    pub const fn is_dialogue_coref(&self) -> bool {
        self.flags.contains(DatasetFlags::DIALOGUE_COREF)
    }

    /// Returns `true` if this dataset supports joint NER and relation extraction.
    ///
    /// Joint datasets provide both entity annotations and relation triples,
    /// enabling evaluation of models that perform both tasks simultaneously.
    #[inline]
    pub const fn is_joint_ner_re(&self) -> bool {
        self.flags.contains(DatasetFlags::JOINT_NER_RE)
    }

    /// Returns `true` if this dataset contains discontinuous entity annotations.
    ///
    /// Discontinuous entities span non-contiguous tokens (e.g., "left and right
    /// ventricle" where "ventricle" is split). Requires specialized evaluation
    /// metrics beyond standard span-based NER.
    #[inline]
    pub const fn is_discontinuous_ner(&self) -> bool {
        self.flags.contains(DatasetFlags::DISCONTINUOUS_NER)
    }

    /// Returns `true` if this dataset is designed for few-shot learning evaluation.
    ///
    /// Few-shot datasets typically have small training sets or are used to test
    /// zero-shot transfer from related domains.
    #[inline]
    pub const fn is_few_shot(&self) -> bool {
        self.flags.contains(DatasetFlags::FEW_SHOT)
    }

    /// Returns `true` if this dataset covers multiple languages.
    ///
    /// Multilingual datasets enable cross-lingual evaluation and testing of
    /// zero-shot transfer between languages.
    #[inline]
    pub const fn is_multilingual(&self) -> bool {
        self.flags.contains(DatasetFlags::MULTILINGUAL)
    }

    /// Returns `true` if this dataset contains constructed/artificial languages.
    ///
    /// Constructed languages (e.g., Esperanto, Klingon) test model generalization
    /// to languages with different structural properties than natural languages.
    #[inline]
    pub const fn is_constructed_language(&self) -> bool {
        self.flags.contains(DatasetFlags::CONSTRUCTED_LANGUAGE)
    }

    /// Returns `true` if this dataset contains code-switched text.
    ///
    /// Code-switching datasets include text where speakers mix multiple languages
    /// within the same utterance, requiring models to handle language boundaries.
    #[inline]
    pub const fn is_code_switching(&self) -> bool {
        self.flags.contains(DatasetFlags::CODE_SWITCHING)
    }

    /// Returns `true` if this dataset contains African languages.
    ///
    /// African language datasets are important for evaluating model performance
    /// on under-resourced languages and diverse linguistic structures.
    #[inline]
    pub const fn is_african_language(&self) -> bool {
        self.flags.contains(DatasetFlags::AFRICAN_LANGUAGE)
    }
}

// =============================================================================
// Standard entity type sets (reduces duplication)
// =============================================================================

/// CoNLL-style entity types (PER, LOC, ORG, MISC)
pub static CONLL_TYPES: &[&str] = &["PER", "LOC", "ORG", "MISC"];

/// OntoNotes entity types (18 types)
pub static ONTONOTES_TYPES: &[&str] = &[
    "PERSON",
    "NORP",
    "FAC",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "DATE",
    "TIME",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL",
];

/// Biomedical entity types
pub static BIO_TYPES: &[&str] = &["Chemical", "Disease", "Gene", "Species"];

/// ACE entity types
pub static ACE_TYPES: &[&str] = &["PER", "ORG", "GPE", "LOC", "FAC", "VEH", "WEA"];

// =============================================================================
// Static metadata table (proof of concept - first 20 datasets)
// =============================================================================
//
// This table replaces the 15+ match statements in loader.rs.
// Full migration will happen incrementally.

/// Metadata for WikiGold dataset
pub static WIKIGOLD: DatasetMetadata = DatasetMetadata::new(
    "WikiGold",
    "Wikipedia-based NER (PER, LOC, ORG, MISC)",
    "https://huggingface.co/datasets/wikigold",
)
.domain("news")
.language("en")
.entity_types(CONLL_TYPES)
.flags(DatasetFlags::NER.union(DatasetFlags::NEWS))
.year(2009);

/// Metadata for WNUT-17 dataset
pub static WNUT17: DatasetMetadata = DatasetMetadata::new(
    "WNUT-17",
    "Social media NER (emerging entities)",
    "https://huggingface.co/datasets/wnut_17",
)
.domain("social-media")
.language("en")
.entity_types(&[
    "person",
    "location",
    "corporation",
    "product",
    "creative-work",
    "group",
])
.flags(DatasetFlags::NER.union(DatasetFlags::SOCIAL_MEDIA))
.year(2017);

/// Metadata for BC5CDR dataset
pub static BC5CDR: DatasetMetadata = DatasetMetadata::new(
    "BC5CDR",
    "Biomedical NER (chemicals, diseases)",
    "https://huggingface.co/datasets/bc5cdr",
)
.domain("biomedical")
.language("en")
.entity_types(&["Chemical", "Disease"])
.flags(
    DatasetFlags::NER
        .union(DatasetFlags::BIOMEDICAL)
        .union(DatasetFlags::SPECIALIZED_DOMAIN),
)
.year(2015);

/// Metadata for GAP coreference dataset
pub static GAP: DatasetMetadata = DatasetMetadata::new(
    "GAP",
    "Gendered Ambiguous Pronouns",
    "https://huggingface.co/datasets/gap",
)
.domain("coreference")
.language("en")
.entity_types(&["Pronoun", "Name"])
.flags(
    DatasetFlags::COREFERENCE
        .union(DatasetFlags::INTRA_DOC_COREF)
        .union(DatasetFlags::BIAS_EVALUATION),
)
.year(2018);

/// Metadata for OntoNotes 5.0 coreference dataset
pub static ONTONOTES_COREF: DatasetMetadata = DatasetMetadata::new(
    "OntoNotes 5.0 (Coreference)",
    "Standard coreference benchmark",
    "https://catalog.ldc.upenn.edu/LDC2013T19",
)
.domain("coreference")
.language("en")
.entity_types(ONTONOTES_TYPES)
.flags(
    DatasetFlags::COREFERENCE
        .union(DatasetFlags::INTRA_DOC_COREF)
        .union(DatasetFlags::NER),
)
.year(2013);

/// Metadata for ECB+ cross-document coreference dataset
pub static ECBPLUS: DatasetMetadata = DatasetMetadata::new(
    "ECB+",
    "Event Coreference Bank Plus",
    "http://www.newsreader-project.eu/results/data/the-ecb-corpus/",
)
.domain("coreference")
.language("en")
.entity_types(&["Event", "Entity"])
.flags(
    DatasetFlags::COREFERENCE
        .union(DatasetFlags::INTER_DOC_COREF)
        .union(DatasetFlags::EVENT_EXTRACTION),
)
.year(2014);

/// Metadata for MultiNERD multilingual dataset
pub static MULTINERD: DatasetMetadata = DatasetMetadata::new(
    "MultiNERD",
    "Multilingual NER (10 languages)",
    "https://huggingface.co/datasets/Babelscape/multinerd",
)
.domain("multilingual")
.language("multilingual")
.entity_types(&[
    "PER", "LOC", "ORG", "ANIM", "BIO", "CEL", "DIS", "EVE", "FOOD", "INST", "MEDIA", "MYTH",
    "PLANT", "TIME", "VEHI",
])
.flags(DatasetFlags::NER.union(DatasetFlags::MULTILINGUAL))
.year(2022);

/// Metadata for FewNERD dataset
pub static FEWNERD: DatasetMetadata = DatasetMetadata::new(
    "FewNERD",
    "Few-shot NER with fine-grained types",
    "https://huggingface.co/datasets/DFKI-SLT/few-nerd",
)
.domain("general")
.language("en")
.entity_types(&[
    "person",
    "location",
    "organization",
    "building",
    "art",
    "product",
    "event",
    "other",
])
.flags(DatasetFlags::NER.union(DatasetFlags::FEW_SHOT))
.year(2021);

/// Metadata for MasakhaNER African languages dataset
pub static MASAKHANER: DatasetMetadata = DatasetMetadata::new(
    "MasakhaNER",
    "NER for African languages",
    "https://huggingface.co/datasets/masakhaner",
)
.domain("low-resource")
.language("multilingual")
.entity_types(CONLL_TYPES)
.flags(
    DatasetFlags::NER
        .union(DatasetFlags::MULTILINGUAL)
        .union(DatasetFlags::AFRICAN_LANGUAGE)
        .union(DatasetFlags::LOW_RESOURCE),
)
.year(2021);

/// Metadata for GENIA biomedical dataset
pub static GENIA: DatasetMetadata = DatasetMetadata::new(
    "GENIA",
    "Biomedical NER (genes, proteins)",
    "http://www.geniaproject.org/",
)
.domain("biomedical")
.language("en")
.entity_types(&["DNA", "RNA", "protein", "cell_line", "cell_type"])
.flags(
    DatasetFlags::NER
        .union(DatasetFlags::BIOMEDICAL)
        .union(DatasetFlags::SPECIALIZED_DOMAIN),
)
.year(2003);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flags_operations() {
        let flags = DatasetFlags::NER | DatasetFlags::BIOMEDICAL | DatasetFlags::SPECIALIZED_DOMAIN;
        assert!(flags.contains(DatasetFlags::NER));
        assert!(flags.contains(DatasetFlags::BIOMEDICAL));
        assert!(!flags.contains(DatasetFlags::SOCIAL_MEDIA));
    }

    #[test]
    fn test_metadata_builder() {
        let meta = DatasetMetadata::new("Test", "A test dataset", "https://example.com")
            .domain("biomedical")
            .language("en")
            .entity_types(&["Disease", "Drug"])
            .flags(DatasetFlags::NER | DatasetFlags::BIOMEDICAL)
            .year(2023);

        assert_eq!(meta.name, "Test");
        assert_eq!(meta.domain, "biomedical");
        assert!(meta.is_biomedical());
        assert!(!meta.is_social_media());
        assert_eq!(meta.year, Some(2023));
    }

    #[test]
    fn test_const_construction() {
        // Verify const construction works (for static tables)
        const META: DatasetMetadata = DatasetMetadata::new("Const", "Desc", "url")
            .domain("test")
            .language("en");

        assert_eq!(META.name, "Const");
        assert_eq!(META.domain, "test");
    }

    #[test]
    fn test_static_wikigold() {
        assert_eq!(WIKIGOLD.name, "WikiGold");
        assert_eq!(WIKIGOLD.domain, "news");
        assert!(WIKIGOLD.is_ner());
        assert!(!WIKIGOLD.is_coreference());
    }

    #[test]
    fn test_static_bc5cdr() {
        assert!(BC5CDR.is_biomedical());
        assert!(BC5CDR.is_specialized_domain());
        assert_eq!(BC5CDR.entity_types.len(), 2);
    }

    #[test]
    fn test_static_gap() {
        assert!(GAP.is_coreference());
        assert!(GAP.is_intra_doc_coref());
        assert!(GAP.is_bias_evaluation());
        assert!(!GAP.is_ner());
    }

    #[test]
    fn test_static_ecbplus() {
        assert!(ECBPLUS.is_inter_doc_coref());
        assert!(!ECBPLUS.is_intra_doc_coref());
    }

    #[test]
    fn test_static_masakhaner() {
        assert!(MASAKHANER.is_african_language());
        assert!(MASAKHANER.is_multilingual());
    }
}
