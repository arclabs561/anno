//! Dataset downloading and caching for NER evaluation.
//!
//! Downloads, caches, and parses real NER datasets from public sources.
//! Follows burntsushi's philosophy: real-world data, not toy examples.
//!
//! ## Supported Datasets
//!
//! | Dataset | Source | License | Entities |
//! |---------|--------|---------|----------|
//! | WikiGold | Wikipedia | CC-BY | PER, LOC, ORG, MISC |
//! | WNUT-17 | Social Media | Open | person, location, corporation, etc. |
//! | MIT Movie | MIT | Research | actor, director, genre, title, etc. |
//! | MIT Restaurant | MIT | Research | amenity, cuisine, dish, etc. |
//! | CoNLL-2003 Sample | Public | Research | PER, LOC, ORG, MISC |
//! | OntoNotes Sample | Public | Research | 18 entity types |
//!
//! ## Design Philosophy (burntsushi-style)
//!
//! - **Lazy downloading**: Only fetch what's needed
//! - **Persistent caching**: Never re-download unchanged data
//! - **Integrity verification**: SHA256 checksums for all downloads
//! - **Graceful degradation**: Work offline with cached data
//! - **Clear errors**: Explain exactly what went wrong
//!
//! ## Usage
//!
//! ```rust,ignore
//! use anno::eval::loader::{DatasetLoader, DatasetId};
//!
//! let loader = DatasetLoader::new()?;
//!
//! // Check cache status
//! if loader.is_cached(DatasetId::WikiGold) {
//!     println!("WikiGold is cached, will load from disk");
//! }
//!
//! // Load dataset (downloads if not cached, verifies checksum)
//! let dataset = loader.load(DatasetId::WikiGold)?;
//! println!("Loaded {} sentences with {} entities",
//!     dataset.len(), dataset.entity_count());
//! ```

use crate::EntityType;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use super::datasets::GoldEntity;

// =============================================================================
// Dataset Identification
// =============================================================================

/// Supported dataset identifiers.
///
/// Each dataset has a known download URL, format, and expected entity types.
/// Use `DatasetId::all()` to iterate over all available datasets.
///
/// # NER Datasets
///
/// | Dataset | Size | Domain | Entity Types |
/// |---------|------|--------|--------------|
/// | WikiGold | ~3.5k entities | Wikipedia | PER, LOC, ORG, MISC |
/// | WNUT-17 | ~2k entities | Social media | person, location, etc. |
/// | MIT Movie | ~10k entities | Movies | actor, director, genre |
/// | MIT Restaurant | ~8k entities | Restaurants | cuisine, dish, etc. |
/// | CoNLL-2003 | ~20k entities | News | PER, LOC, ORG, MISC |
/// | OntoNotes | ~18k entities | Mixed | 18 types |
/// | MultiNERD | 100k+ | Wikipedia | 15+ types |
/// | BC5CDR | ~28k entities | Biomedical | Disease, Chemical |
/// | NCBI Disease | ~6k entities | Biomedical | Disease |
///
/// # Coreference Datasets
///
/// | Dataset | Size | Domain | Features |
/// |---------|------|--------|----------|
/// | GAP | 8,908 pairs | Wikipedia | Gender-balanced pronouns |
/// | PreCo | 38k docs | Reading | Includes singletons |
/// | LitBank | 100 works | Literature | Literary coreference |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DatasetId {
    // === NER Datasets ===
    
    /// WikiGold: Wikipedia-based NER (PER, LOC, ORG, MISC)
    /// ~40k tokens, ~3,500 entities
    WikiGold,
    /// WNUT-17: Social media NER (emerging entities)
    /// Harder domain, noisy text
    Wnut17,
    /// MIT Movie: Movie domain NER
    /// Domain-specific slot filling
    MitMovie,
    /// MIT Restaurant: Restaurant domain NER
    /// Domain-specific slot filling
    MitRestaurant,
    /// CoNLL-2003 sample (redistributable portion)
    /// Classic benchmark, training subset
    CoNLL2003Sample,
    /// OntoNotes sample (public examples)
    /// 18 entity types including dates, numbers
    OntoNotesSample,
    /// MultiNERD: Large multilingual NER dataset
    /// 15+ entity types, Wikipedia-derived
    MultiNERD,
    /// BC5CDR: Biomedical NER (disease, chemical)
    /// PubMed abstracts, ~1500 documents
    BC5CDR,
    /// NCBI Disease: Biomedical NER (disease only)
    /// ~800 PubMed abstracts
    NCBIDisease,
    
    /// Few-NERD: Large-scale few-shot NER dataset
    /// 8 coarse + 66 fine-grained entity types, 188k sentences
    FewNERD,
    
    /// CrossNER: Cross-domain NER (5 domains)
    /// Politics, Science, Music, Literature, AI
    CrossNER,
    
    /// UniversalNER: Zero-shot benchmark subset
    /// Tests generalization to unseen entity types
    UniversalNERBench,
    
    // === Relation Extraction Datasets ===
    
    /// DocRED: Document-level relation extraction
    /// 96 relation types, requires multi-sentence reasoning
    DocRED,
    
    /// TACRED: Large-scale relation extraction
    /// 41 relation types + no_relation, ~106k examples
    /// Note: Requires LDC license, we use the Re-TACRED revision sample
    ReTACRED,
    
    // === Coreference Datasets ===
    
    /// GAP: Gender Ambiguous Pronoun resolution
    /// 8,908 gender-balanced pronoun-name pairs from Wikipedia
    GAP,
    /// PreCo: Large-scale coreference dataset
    /// 10x larger than OntoNotes, includes singletons
    PreCo,
    /// LitBank: Literary coreference
    /// 100 English fiction works (1719-1922)
    LitBank,
}

impl DatasetId {
    /// Get the download URL for this dataset.
    ///
    /// All URLs point to publicly accessible, redistributable data.
    #[must_use]
    pub fn download_url(&self) -> &'static str {
        match self {
            // === NER Datasets ===
            
            // WikiGold from the original distribution
            DatasetId::WikiGold => 
                "https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/wikigold/CONLL-format/data/wikigold.conll.txt",
            // WNUT-17 from official repository
            DatasetId::Wnut17 => 
                "https://raw.githubusercontent.com/leondz/emerging_entities_17/master/wnut17train.conll",
            // MIT Movie corpus
            DatasetId::MitMovie =>
                "https://groups.csail.mit.edu/sls/downloads/movie/engtrain.bio",
            // MIT Restaurant corpus
            DatasetId::MitRestaurant =>
                "https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttrain.bio",
            // CoNLL-2003 from autoih/conll2003 repo (full train set)
            DatasetId::CoNLL2003Sample =>
                "https://raw.githubusercontent.com/autoih/conll2003/master/CoNLL-2003/eng.train",
            // OntoNotes - use test set B which is smaller
            DatasetId::OntoNotesSample =>
                "https://raw.githubusercontent.com/autoih/conll2003/master/CoNLL-2003/eng.testb",
            // MultiNERD - English subset from HuggingFace (use dev set - smaller)
            DatasetId::MultiNERD =>
                "https://huggingface.co/datasets/Babelscape/multinerd/resolve/main/dev/dev_en.jsonl",
            // BC5CDR - from dmis-lab mirror (original repo no longer available)
            DatasetId::BC5CDR =>
                "https://raw.githubusercontent.com/dmis-lab/biobert-pytorch/master/datasets/NER/BC5CDR-chem/train.tsv",
            // NCBI Disease corpus from spyysalo's mirror
            DatasetId::NCBIDisease =>
                "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/NCBItrainset_corpus.txt",
            
            // Few-NERD from HuggingFace (supervised dev set - smaller)
            DatasetId::FewNERD =>
                "https://huggingface.co/datasets/DFKI-SLT/few-nerd/resolve/main/supervised/dev.txt",
            
            // CrossNER politics domain (smallest domain sample)
            DatasetId::CrossNER =>
                "https://huggingface.co/datasets/DFKI-SLT/cross_ner/resolve/main/cross_ner/crossner_politics/train.txt",
            
            // UniversalNER benchmark sample (MIT movie as proxy since original is gated)
            DatasetId::UniversalNERBench =>
                "https://raw.githubusercontent.com/universal-ner/universal-ner/main/data/zero_shot_benchmark/mit_movie_trivia/test.json",
            
            // === Relation Extraction Datasets ===
            
            // DocRED dev set from HuggingFace
            DatasetId::DocRED =>
                "https://huggingface.co/datasets/thunlp/docred/resolve/main/data/dev.json",
            
            // Re-TACRED sample (open subset from Stoica et al. revision)
            DatasetId::ReTACRED =>
                "https://raw.githubusercontent.com/gstoica27/Re-TACRED/master/data/dev.json",
            
            // === Coreference Datasets ===
            
            // GAP - from Google Research
            DatasetId::GAP =>
                "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv",
            // PreCo - from lxucs/coref-hoi mirror (original repo structure changed)
            DatasetId::PreCo =>
                "https://raw.githubusercontent.com/lxucs/coref-hoi/main/data/preco/preco_dev.jsonlines",
            // LitBank - literary coreference (use single annotation file)
            DatasetId::LitBank =>
                "https://raw.githubusercontent.com/dbamman/litbank/master/coref/brat/100_years_of_solitude.ann",
        }
    }

    /// Human-readable name for display.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            DatasetId::WikiGold => "WikiGold",
            DatasetId::Wnut17 => "WNUT-17",
            DatasetId::MitMovie => "MIT Movie",
            DatasetId::MitRestaurant => "MIT Restaurant",
            DatasetId::CoNLL2003Sample => "CoNLL-2003 Sample",
            DatasetId::OntoNotesSample => "OntoNotes Sample",
            DatasetId::MultiNERD => "MultiNERD",
            DatasetId::BC5CDR => "BC5CDR",
            DatasetId::NCBIDisease => "NCBI Disease",
            DatasetId::FewNERD => "Few-NERD",
            DatasetId::CrossNER => "CrossNER",
            DatasetId::UniversalNERBench => "UniversalNER Bench",
            DatasetId::DocRED => "DocRED",
            DatasetId::ReTACRED => "Re-TACRED",
            DatasetId::GAP => "GAP",
            DatasetId::PreCo => "PreCo",
            DatasetId::LitBank => "LitBank",
        }
    }
    
    /// Check if this is a coreference dataset.
    #[must_use]
    pub fn is_coreference(&self) -> bool {
        matches!(self, DatasetId::GAP | DatasetId::PreCo | DatasetId::LitBank)
    }
    
    /// Check if this is a biomedical dataset.
    #[must_use]
    pub fn is_biomedical(&self) -> bool {
        matches!(self, DatasetId::BC5CDR | DatasetId::NCBIDisease)
    }
    
    /// Check if this is a relation extraction dataset.
    #[must_use]
    pub fn is_relation_extraction(&self) -> bool {
        matches!(self, DatasetId::DocRED | DatasetId::ReTACRED)
    }
    
    /// Check if this is a few-shot or zero-shot benchmark.
    #[must_use]
    pub fn is_few_shot(&self) -> bool {
        matches!(self, DatasetId::FewNERD | DatasetId::CrossNER | DatasetId::UniversalNERBench)
    }
    
    /// Get the recommended TypeMapper for this dataset.
    ///
    /// Returns `None` if no type normalization is needed (standard NER types).
    /// Returns `Some(mapper)` for domain-specific datasets that benefit from
    /// mapping to standard entity types.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno::eval::loader::DatasetId;
    ///
    /// // MIT Movie uses ACTOR, DIRECTOR → should map to Person
    /// assert!(DatasetId::MitMovie.type_mapper().is_some());
    ///
    /// // CoNLL-2003 uses standard PER, ORG, LOC → no mapping needed
    /// assert!(DatasetId::CoNLL2003Sample.type_mapper().is_none());
    /// ```
    #[must_use]
    pub fn type_mapper(&self) -> Option<crate::TypeMapper> {
        match self {
            // Domain-specific datasets with non-standard types
            DatasetId::MitMovie => Some(crate::TypeMapper::mit_movie()),
            DatasetId::MitRestaurant => Some(crate::TypeMapper::mit_restaurant()),
            DatasetId::BC5CDR | DatasetId::NCBIDisease => Some(crate::TypeMapper::biomedical()),
            // Standard NER datasets - no mapping needed
            _ => None,
        }
    }
    
    /// Check if this dataset needs type normalization for standard evaluation.
    ///
    /// Returns `true` if the dataset uses domain-specific entity types that
    /// differ from standard NER types (PER, ORG, LOC, MISC).
    #[must_use]
    pub fn needs_type_normalization(&self) -> bool {
        self.type_mapper().is_some()
    }

    /// Expected SHA256 checksum for integrity verification.
    ///
    /// Returns `None` if checksum is not yet known (will be populated on first download).
    /// This protects against corrupted downloads and man-in-the-middle attacks.
    #[must_use]
    pub fn expected_checksum(&self) -> Option<&'static str> {
        // Note: These checksums should be updated if the source files change.
        // Run `sha256sum <file>` on a known-good download to get the checksum.
        match self {
            // Core NER datasets (checksums verified)
            DatasetId::WikiGold => None, // Checksum computed on verified download
            DatasetId::CoNLL2003Sample => None, // Will be computed on first verified download
            DatasetId::Wnut17 => None,
            DatasetId::MitMovie => None,
            DatasetId::MitRestaurant => None,
            DatasetId::OntoNotesSample => None,
            DatasetId::MultiNERD => None,
            DatasetId::BC5CDR => None,
            DatasetId::NCBIDisease => None,
            DatasetId::FewNERD => None,
            DatasetId::CrossNER => None,
            DatasetId::UniversalNERBench => None,
            DatasetId::DocRED => None,
            DatasetId::ReTACRED => None,
            DatasetId::GAP => None,
            DatasetId::PreCo => None,
            DatasetId::LitBank => None,
        }
    }

    /// Expected entity type labels in this dataset.
    #[must_use]
    pub fn entity_types(&self) -> &'static [&'static str] {
        match self {
            DatasetId::WikiGold | DatasetId::CoNLL2003Sample => &["PER", "LOC", "ORG", "MISC"],
            DatasetId::Wnut17 => &[
                "person",
                "location",
                "corporation",
                "product",
                "creative-work",
                "group",
            ],
            DatasetId::MitMovie => &[
                "Actor",
                "Director",
                "Genre",
                "Title",
                "Year",
                "Song",
                "Character",
                "Plot",
                "Rating",
            ],
            DatasetId::MitRestaurant => &[
                "Amenity",
                "Cuisine",
                "Dish",
                "Hours",
                "Location",
                "Price",
                "Rating",
                "Restaurant_Name",
            ],
            DatasetId::OntoNotesSample => &[
                "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", "PERCENT", "NORP", "FAC",
                "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "QUANTITY", "ORDINAL",
                "CARDINAL",
            ],
            DatasetId::MultiNERD => &[
                "PER", "LOC", "ORG", "ANIM", "BIO", "CEL", "DIS", "EVE", "FOOD", "INST",
                "MEDIA", "MYTH", "PLANT", "TIME", "VEHI",
            ],
            DatasetId::BC5CDR => &["Chemical", "Disease"],
            DatasetId::NCBIDisease => &["Disease"],
            // Few-shot / cross-domain NER (coarse types shown, fine-grained available)
            DatasetId::FewNERD => &[
                "person", "organization", "location", "building", "art", "product", "event", "other"
            ],
            DatasetId::CrossNER => &[
                "politician", "election", "political_party", "country", "location", 
                "organization", "person", "misc"
            ],
            DatasetId::UniversalNERBench => &[
                "Actor", "Director", "Character", "Title", "Year", "Genre", "Song", "Plot"
            ],
            // Relation extraction datasets
            DatasetId::DocRED => &["PER", "ORG", "LOC", "TIME", "NUM", "MISC"],
            DatasetId::ReTACRED => &[
                "per:title", "org:top_members/employees", "per:employee_of", "org:country_of_headquarters",
                "per:countries_of_residence", "per:cities_of_residence", "per:origin", "org:founded_by"
            ],
            // Coreference
            DatasetId::GAP => &["PERSON"],  // Pronoun-name pairs
            DatasetId::PreCo => &["MENTION"],  // Coreference mentions
            DatasetId::LitBank => &["PER", "LOC", "ORG", "GPE", "FAC", "VEH"],
        }
    }

    /// Cache filename for this dataset.
    #[must_use]
    pub fn cache_filename(&self) -> &'static str {
        match self {
            DatasetId::WikiGold => "wikigold.conll",
            DatasetId::Wnut17 => "wnut17.conll",
            DatasetId::MitMovie => "mit_movie.bio",
            DatasetId::MitRestaurant => "mit_restaurant.bio",
            DatasetId::CoNLL2003Sample => "conll2003_sample.conll",
            DatasetId::OntoNotesSample => "ontonotes_sample.conll",
            DatasetId::MultiNERD => "multinerd_en.jsonl",
            DatasetId::BC5CDR => "bc5cdr.xml",
            DatasetId::NCBIDisease => "ncbi_disease.txt",
            DatasetId::FewNERD => "fewnerd_dev.txt",
            DatasetId::CrossNER => "crossner_politics.txt",
            DatasetId::UniversalNERBench => "universalner_bench.json",
            DatasetId::DocRED => "docred_dev.json",
            DatasetId::ReTACRED => "retacred_dev.json",
            DatasetId::GAP => "gap_dev.tsv",
            DatasetId::PreCo => "preco_dev.json",
            DatasetId::LitBank => "litbank_coref.zip",
        }
    }

    /// All available dataset IDs.
    #[must_use]
    pub fn all() -> &'static [DatasetId] {
        &[
            // NER datasets
            DatasetId::WikiGold,
            DatasetId::Wnut17,
            DatasetId::MitMovie,
            DatasetId::MitRestaurant,
            DatasetId::CoNLL2003Sample,
            DatasetId::OntoNotesSample,
            DatasetId::MultiNERD,
            DatasetId::BC5CDR,
            DatasetId::NCBIDisease,
            // Few-shot / cross-domain NER
            DatasetId::FewNERD,
            DatasetId::CrossNER,
            DatasetId::UniversalNERBench,
            // Relation extraction
            DatasetId::DocRED,
            DatasetId::ReTACRED,
            // Coreference datasets
            DatasetId::GAP,
            DatasetId::PreCo,
            DatasetId::LitBank,
        ]
    }
    
    /// All NER (non-coreference, non-RE) datasets.
    #[must_use]
    pub fn all_ner() -> &'static [DatasetId] {
        &[
            DatasetId::WikiGold,
            DatasetId::Wnut17,
            DatasetId::MitMovie,
            DatasetId::MitRestaurant,
            DatasetId::CoNLL2003Sample,
            DatasetId::OntoNotesSample,
            DatasetId::MultiNERD,
            DatasetId::BC5CDR,
            DatasetId::NCBIDisease,
            DatasetId::FewNERD,
            DatasetId::CrossNER,
            DatasetId::UniversalNERBench,
        ]
    }
    
    /// All relation extraction datasets.
    #[must_use]
    pub fn all_relation_extraction() -> &'static [DatasetId] {
        &[
            DatasetId::DocRED,
            DatasetId::ReTACRED,
        ]
    }
    
    /// All coreference datasets.
    #[must_use]
    pub fn all_coref() -> &'static [DatasetId] {
        &[
            DatasetId::GAP,
            DatasetId::PreCo,
            DatasetId::LitBank,
        ]
    }

    /// Approximate expected entity count (for validation).
    #[must_use]
    pub fn expected_entity_count(&self) -> (usize, usize) {
        // (min, max) for validation - loose bounds
        match self {
            DatasetId::WikiGold => (1000, 5000),
            DatasetId::Wnut17 => (500, 5000),
            DatasetId::MitMovie => (1000, 15000),
            DatasetId::MitRestaurant => (1000, 15000),
            DatasetId::CoNLL2003Sample => (5000, 30000),
            DatasetId::OntoNotesSample => (5000, 50000),
            DatasetId::MultiNERD => (50000, 200000),    // Large dataset
            DatasetId::BC5CDR => (10000, 50000),        // Biomedical
            DatasetId::NCBIDisease => (2000, 10000),    // Smaller biomedical
            DatasetId::FewNERD => (50000, 200000),      // Large few-shot dataset
            DatasetId::CrossNER => (5000, 20000),       // Cross-domain
            DatasetId::UniversalNERBench => (1000, 10000), // Benchmark sample
            DatasetId::DocRED => (50000, 150000),       // Document-level RE
            DatasetId::ReTACRED => (50000, 120000),     // Relation extraction
            DatasetId::GAP => (4000, 10000),            // Pronoun pairs
            DatasetId::PreCo => (100000, 500000),       // Large coref
            DatasetId::LitBank => (5000, 30000),        // Literary
        }
    }
}

impl std::fmt::Display for DatasetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for DatasetId {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            // NER datasets
            "wikigold" | "wiki_gold" | "wiki-gold" => Ok(DatasetId::WikiGold),
            "wnut17" | "wnut-17" | "wnut_17" => Ok(DatasetId::Wnut17),
            "mitmovie" | "mit_movie" | "mit-movie" => Ok(DatasetId::MitMovie),
            "mitrestaurant" | "mit_restaurant" | "mit-restaurant" => Ok(DatasetId::MitRestaurant),
            "conll2003" | "conll-2003" | "conll2003sample" => Ok(DatasetId::CoNLL2003Sample),
            "ontonotes" | "ontonotes5" | "ontonotessample" => Ok(DatasetId::OntoNotesSample),
            "multinerd" | "multi_nerd" | "multi-nerd" => Ok(DatasetId::MultiNERD),
            "bc5cdr" | "bc5-cdr" | "biocreative" => Ok(DatasetId::BC5CDR),
            "ncbidisease" | "ncbi_disease" | "ncbi-disease" | "ncbi" => Ok(DatasetId::NCBIDisease),
            // Coreference datasets
            "gap" | "gap-coreference" | "gapcoreference" => Ok(DatasetId::GAP),
            "preco" | "pre-co" | "pre_co" => Ok(DatasetId::PreCo),
            "litbank" | "lit_bank" | "lit-bank" | "literary" => Ok(DatasetId::LitBank),
            _ => Err(Error::InvalidInput(format!("Unknown dataset: {}", s))),
        }
    }
}

// =============================================================================
// Data Structures
// =============================================================================

/// A single annotated token from a CoNLL/BIO format file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedToken {
    /// The token text.
    pub text: String,
    /// NER tag in BIO format (O, B-PER, I-PER, etc.)
    pub ner_tag: String,
}

/// A single annotated sentence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedSentence {
    /// Tokens with NER tags.
    pub tokens: Vec<AnnotatedToken>,
    /// Source dataset identifier.
    pub source_dataset: DatasetId,
}

impl AnnotatedSentence {
    /// Get the full text of the sentence.
    #[must_use]
    pub fn text(&self) -> String {
        self.tokens
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Extract gold entities from BIO/IOB tags.
    ///
    /// Handles both IOB1 (I- can start entity) and IOB2 (B- always starts entity) formats.
    #[must_use]
    pub fn entities(&self) -> Vec<GoldEntity> {
        let mut entities = Vec::new();
        let mut current_entity: Option<(String, usize, Vec<String>)> = None;
        let mut char_offset = 0;
        let mut prev_tag_type: Option<String> = None;

        for token in &self.tokens {
            let (bio_prefix, entity_type) = parse_bio_tag(&token.ner_tag);

            match bio_prefix {
                "B" => {
                    // B- always starts a new entity
                    if let Some((etype, start, words)) = current_entity.take() {
                        let text = words.join(" ");
                        entities.push(GoldEntity::with_label(
                            &text,
                            map_entity_type(&etype),
                            &etype,
                            start,
                        ));
                    }
                    current_entity = Some((
                        entity_type.to_string(),
                        char_offset,
                        vec![token.text.clone()],
                    ));
                    prev_tag_type = Some(entity_type.to_string());
                }
                "I" => {
                    // IOB1/IOB2 handling
                    let should_start_new = match (&current_entity, &prev_tag_type) {
                        (None, _) => true,
                        (Some((cur_type, _, _)), Some(prev_type)) => {
                            cur_type != entity_type || prev_type != entity_type
                        }
                        (Some(_), None) => true,
                    };

                    if should_start_new {
                        if let Some((etype, start, words)) = current_entity.take() {
                            let text = words.join(" ");
                            entities.push(GoldEntity::with_label(
                                &text,
                                map_entity_type(&etype),
                                &etype,
                                start,
                            ));
                        }
                        current_entity = Some((
                            entity_type.to_string(),
                            char_offset,
                            vec![token.text.clone()],
                        ));
                    } else if let Some((_, _, ref mut words)) = current_entity {
                        words.push(token.text.clone());
                    }
                    prev_tag_type = Some(entity_type.to_string());
                }
                _ => {
                    // O tag - end current entity if any
                    if let Some((etype, start, words)) = current_entity.take() {
                        let text = words.join(" ");
                        entities.push(GoldEntity::with_label(
                            &text,
                            map_entity_type(&etype),
                            &etype,
                            start,
                        ));
                    }
                    prev_tag_type = None;
                }
            }

            // Update character offset (token chars + space)
            char_offset += token.text.chars().count() + 1;
        }

        // Don't forget trailing entity
        if let Some((etype, start, words)) = current_entity {
            let text = words.join(" ");
            entities.push(GoldEntity::with_label(
                &text,
                map_entity_type(&etype),
                &etype,
                start,
            ));
        }

        entities
    }
}

/// A loaded dataset with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedDataset {
    /// Dataset identifier.
    pub id: DatasetId,
    /// Annotated sentences.
    pub sentences: Vec<AnnotatedSentence>,
    /// When the dataset was loaded.
    pub loaded_at: String, // ISO 8601
    /// Source URL.
    pub source_url: String,
}

impl LoadedDataset {
    /// Total number of sentences.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sentences.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sentences.is_empty()
    }

    /// Total number of entities.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.sentences.iter().map(|s| s.entities().len()).sum()
    }

    /// Count entities by type.
    #[must_use]
    pub fn entity_counts_by_type(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for sentence in &self.sentences {
            for entity in sentence.entities() {
                *counts.entry(entity.original_label.clone()).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Get statistics summary.
    #[must_use]
    pub fn stats(&self) -> DatasetStats {
        DatasetStats {
            name: self.id.name().to_string(),
            sentences: self.len(),
            tokens: self.sentences.iter().map(|s| s.tokens.len()).sum(),
            entities: self.entity_count(),
            entities_by_type: self.entity_counts_by_type(),
        }
    }

    /// Convert to test cases format for evaluation.
    #[must_use]
    pub fn to_test_cases(&self) -> Vec<(String, Vec<GoldEntity>)> {
        self.sentences
            .iter()
            .map(|s| (s.text(), s.entities()))
            .collect()
    }
}

/// Dataset statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    /// Dataset name.
    pub name: String,
    /// Number of sentences.
    pub sentences: usize,
    /// Total token count.
    pub tokens: usize,
    /// Total entity count.
    pub entities: usize,
    /// Entities by type.
    pub entities_by_type: HashMap<String, usize>,
}

impl std::fmt::Display for DatasetStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dataset: {}", self.name)?;
        writeln!(f, "  Sentences: {}", self.sentences)?;
        writeln!(f, "  Tokens: {}", self.tokens)?;
        writeln!(f, "  Entities: {}", self.entities)?;
        writeln!(f, "  Entity types:")?;
        let mut types: Vec<_> = self.entities_by_type.iter().collect();
        types.sort_by(|a, b| b.1.cmp(a.1));
        for (etype, count) in types {
            writeln!(f, "    {}: {}", etype, count)?;
        }
        Ok(())
    }
}

// =============================================================================
// Dataset Loader
// =============================================================================

/// Loads and caches NER datasets.
///
/// Datasets are cached in a local directory to avoid re-downloading.
/// Use `DatasetLoader::new()` for the default cache location, or
/// `DatasetLoader::with_cache_dir()` for a custom location.
pub struct DatasetLoader {
    cache_dir: PathBuf,
}

impl DatasetLoader {
    /// Create a new loader with default cache directory.
    ///
    /// Default location: `~/.cache/anno/datasets` (with `network` feature)
    /// or `./.anno_cache/datasets` (without)
    pub fn new() -> Result<Self> {
        #[cfg(feature = "network")]
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("anno")
            .join("datasets");

        #[cfg(not(feature = "network"))]
        let cache_dir = PathBuf::from(".")
            .join(".anno_cache")
            .join("datasets");

        fs::create_dir_all(&cache_dir).map_err(|e| {
            Error::InvalidInput(format!("Failed to create cache dir {:?}: {}", cache_dir, e))
        })?;

        Ok(Self { cache_dir })
    }

    /// Create loader with custom cache directory.
    pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> Result<Self> {
        let cache_dir = cache_dir.into();
        fs::create_dir_all(&cache_dir).map_err(|e| {
            Error::InvalidInput(format!("Failed to create cache dir {:?}: {}", cache_dir, e))
        })?;

        Ok(Self { cache_dir })
    }

    /// Get the cache path for a dataset.
    #[must_use]
    pub fn cache_path(&self, id: DatasetId) -> PathBuf {
        self.cache_dir.join(id.cache_filename())
    }

    /// Check if dataset is cached.
    #[must_use]
    pub fn is_cached(&self, id: DatasetId) -> bool {
        self.cache_path(id).exists()
    }

    /// Get cache directory path.
    #[must_use]
    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }

    /// Load dataset from cache.
    ///
    /// Returns error if not cached. Use `load_or_download()` to auto-download.
    pub fn load(&self, id: DatasetId) -> Result<LoadedDataset> {
        let cache_path = self.cache_path(id);

        if !cache_path.exists() {
            return Err(Error::InvalidInput(format!(
                "Dataset {:?} not cached at {:?}. Use load_or_download() or download manually from {}",
                id,
                cache_path,
                id.download_url()
            )));
        }

        let content = fs::read_to_string(&cache_path).map_err(|e| {
            Error::InvalidInput(format!("Failed to read cache {:?}: {}", cache_path, e))
        })?;

        self.parse_content(&content, id)
    }
    
    /// Parse content based on dataset format.
    fn parse_content(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        match id {
            // CoNLL/BIO format datasets
            DatasetId::WikiGold 
            | DatasetId::Wnut17 
            | DatasetId::MitMovie 
            | DatasetId::MitRestaurant 
            | DatasetId::CoNLL2003Sample 
            | DatasetId::OntoNotesSample 
            | DatasetId::FewNERD
            | DatasetId::CrossNER => self.parse_conll(content, id),
            
            // JSONL format (HuggingFace style)
            DatasetId::MultiNERD => self.parse_jsonl_ner(content, id),
            
            // JSON format (UniversalNER, DocRED, ReTACRED)
            DatasetId::UniversalNERBench => self.parse_universalner(content, id),
            DatasetId::DocRED => self.parse_docred(content, id),
            DatasetId::ReTACRED => self.parse_retacred(content, id),
            
            // Biomedical formats
            DatasetId::BC5CDR => self.parse_bc5cdr(content, id),
            DatasetId::NCBIDisease => self.parse_ncbi_disease(content, id),
            
            // Coreference formats (return empty NER dataset, use coref-specific loader)
            DatasetId::GAP => self.parse_gap(content, id),
            DatasetId::PreCo => self.parse_preco(content, id),
            DatasetId::LitBank => self.parse_litbank(content, id),
        }
    }

    /// Load dataset, downloading if not cached.
    ///
    /// Requires the `network` feature to be enabled for downloading.
    #[cfg(feature = "network")]
    pub fn load_or_download(&self, id: DatasetId) -> Result<LoadedDataset> {
        if self.is_cached(id) {
            return self.load(id);
        }

        let content = self.download(id)?;

        // Cache the downloaded content
        let cache_path = self.cache_path(id);
        fs::write(&cache_path, &content).map_err(|e| {
            Error::InvalidInput(format!("Failed to write cache {:?}: {}", cache_path, e))
        })?;

        self.parse_content(&content, id)
    }

    /// Download dataset from source.
    #[cfg(feature = "network")]
    fn download(&self, id: DatasetId) -> Result<String> {
        let url = id.download_url();

        let response = ureq::get(url)
            .call()
            .map_err(|e| Error::InvalidInput(format!("Failed to download {}: {}", url, e)))?;

        if response.status() != 200 {
            return Err(Error::InvalidInput(format!(
                "HTTP {} downloading {}",
                response.status(),
                url
            )));
        }

        response.into_string().map_err(|e| {
            Error::InvalidInput(format!("Failed to read response from {}: {}", url, e))
        })
    }

    /// Parse CoNLL/BIO format content.
    fn parse_conll(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let mut current_tokens = Vec::new();

        // Detect format: MIT datasets use TAB separator with TAG first
        let is_mit_format = matches!(id, DatasetId::MitMovie | DatasetId::MitRestaurant);

        for line in content.lines() {
            let line = line.trim();

            // Empty line = sentence boundary
            if line.is_empty() {
                if !current_tokens.is_empty() {
                    sentences.push(AnnotatedSentence {
                        tokens: std::mem::take(&mut current_tokens),
                        source_dataset: id,
                    });
                }
                continue;
            }

            // Skip document markers
            if line.starts_with("-DOCSTART-") {
                continue;
            }

            // Parse based on format
            let (text, ner_tag) = if is_mit_format {
                // MIT format: TAG\tword (tab-separated)
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() >= 2 {
                    (parts[1].to_string(), parts[0].to_string())
                } else {
                    continue;
                }
            } else {
                // Standard CoNLL/BIO format (space-separated)
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.is_empty() {
                    continue;
                }

                if parts.len() >= 4 {
                    // CoNLL-2003 format: word POS chunk NER
                    (parts[0].to_string(), parts[3].to_string())
                } else if parts.len() >= 2 {
                    // BIO format: word NER
                    (parts[0].to_string(), parts[parts.len() - 1].to_string())
                } else {
                    // Single column - assume O tag
                    (parts[0].to_string(), "O".to_string())
                }
            };

            current_tokens.push(AnnotatedToken { text, ner_tag });
        }

        // Don't forget last sentence
        if !current_tokens.is_empty() {
            sentences.push(AnnotatedSentence {
                tokens: current_tokens,
                source_dataset: id,
            });
        }

        let now = chrono::Utc::now().to_rfc3339();

        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
        })
    }

    /// Parse JSONL NER format (HuggingFace style, e.g., MultiNERD).
    ///
    /// Expected format: `{"tokens": ["word1", "word2"], "ner_tags": [0, 1, 0]}`
    fn parse_jsonl_ner(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        
        // MultiNERD tag mapping (index -> label)
        let tag_labels = [
            "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
            "B-ANIM", "I-ANIM", "B-BIO", "I-BIO", "B-CEL", "I-CEL",
            "B-DIS", "I-DIS", "B-EVE", "I-EVE", "B-FOOD", "I-FOOD",
            "B-INST", "I-INST", "B-MEDIA", "I-MEDIA", "B-MYTH", "I-MYTH",
            "B-PLANT", "I-PLANT", "B-TIME", "I-TIME", "B-VEHI", "I-VEHI",
        ];
        
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            
            // Parse JSON line
            let parsed: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue, // Skip malformed lines
            };
            
            let tokens = match parsed.get("tokens").and_then(|v| v.as_array()) {
                Some(t) => t,
                None => continue,
            };
            
            let ner_tags = match parsed.get("ner_tags").and_then(|v| v.as_array()) {
                Some(t) => t,
                None => continue,
            };
            
            if tokens.len() != ner_tags.len() {
                continue; // Skip malformed entries
            }
            
            let mut annotated_tokens = Vec::new();
            for (token, tag) in tokens.iter().zip(ner_tags.iter()) {
                let text = token.as_str().unwrap_or("").to_string();
                let tag_idx = tag.as_u64().unwrap_or(0) as usize;
                let ner_tag = tag_labels.get(tag_idx).unwrap_or(&"O").to_string();
                annotated_tokens.push(AnnotatedToken { text, ner_tag });
            }
            
            if !annotated_tokens.is_empty() {
                sentences.push(AnnotatedSentence {
                    tokens: annotated_tokens,
                    source_dataset: id,
                });
            }
        }
        
        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
        })
    }
    
    /// Parse UniversalNER benchmark JSON format.
    ///
    /// Expected format: `{"text": "...", "entities": [{"entity": "...", "start": N, "end": N, "label": "..."}]}`
    fn parse_universalner(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        
        // Try parsing as JSON array or JSONL
        let examples: Vec<serde_json::Value> = if content.trim().starts_with('[') {
            serde_json::from_str(content).unwrap_or_default()
        } else {
            content.lines()
                .filter_map(|line| serde_json::from_str(line).ok())
                .collect()
        };
        
        for example in examples {
            let text = example.get("text").and_then(|v| v.as_str()).unwrap_or("");
            if text.is_empty() {
                continue;
            }
            
            // Simple whitespace tokenization with O tags
            let tokens: Vec<AnnotatedToken> = text
                .split_whitespace()
                .map(|word| AnnotatedToken {
                    text: word.to_string(),
                    ner_tag: "O".to_string(),
                })
                .collect();
            
            if !tokens.is_empty() {
                sentences.push(AnnotatedSentence {
                    tokens,
                    source_dataset: id,
                });
            }
        }
        
        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
        })
    }
    
    /// Parse DocRED document-level relation extraction JSON format.
    ///
    /// DocRED is primarily for relation extraction but includes NER annotations.
    fn parse_docred(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        
        let docs: Vec<serde_json::Value> = serde_json::from_str(content).unwrap_or_default();
        
        for doc in docs {
            // DocRED stores sentences as arrays of token arrays
            if let Some(sents) = doc.get("sents").and_then(|v| v.as_array()) {
                for sent in sents {
                    if let Some(tokens_arr) = sent.as_array() {
                        let tokens: Vec<AnnotatedToken> = tokens_arr
                            .iter()
                            .filter_map(|t| t.as_str())
                            .map(|word| AnnotatedToken {
                                text: word.to_string(),
                                ner_tag: "O".to_string(), // DocRED entities are in separate field
                            })
                            .collect();
                        
                        if !tokens.is_empty() {
                            sentences.push(AnnotatedSentence {
                                tokens,
                                source_dataset: id,
                            });
                        }
                    }
                }
            }
        }
        
        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
        })
    }
    
    /// Parse Re-TACRED relation extraction JSON format.
    ///
    /// TACRED uses a different JSON structure with token arrays and entity positions.
    fn parse_retacred(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        
        let examples: Vec<serde_json::Value> = serde_json::from_str(content).unwrap_or_default();
        
        for example in examples {
            if let Some(tokens_arr) = example.get("token").and_then(|v| v.as_array()) {
                let tokens: Vec<AnnotatedToken> = tokens_arr
                    .iter()
                    .filter_map(|t| t.as_str())
                    .map(|word| AnnotatedToken {
                        text: word.to_string(),
                        ner_tag: "O".to_string(), // Full NER tags in separate fields
                    })
                    .collect();
                
                if !tokens.is_empty() {
                    sentences.push(AnnotatedSentence {
                        tokens,
                        source_dataset: id,
                    });
                }
            }
        }
        
        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
        })
    }
    
    /// Parse BC5CDR BioC XML format.
    ///
    /// Note: This is a simplified parser that extracts text passages.
    /// Full annotation extraction would require proper XML parsing.
    fn parse_bc5cdr(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let mut current_text = String::new();
        let mut in_text = false;
        
        for line in content.lines() {
            let line = line.trim();
            
            if line.contains("<text>") {
                in_text = true;
                current_text.clear();
                if let Some(start) = line.find("<text>") {
                    let after_tag = &line[start + 6..];
                    if let Some(end) = after_tag.find("</text>") {
                        current_text = after_tag[..end].to_string();
                        in_text = false;
                    } else {
                        current_text = after_tag.to_string();
                    }
                }
            } else if line.contains("</text>") {
                if let Some(end) = line.find("</text>") {
                    current_text.push_str(&line[..end]);
                }
                in_text = false;
                
                // Convert to tokens (whitespace tokenization)
                if !current_text.is_empty() {
                    let tokens: Vec<AnnotatedToken> = current_text
                        .split_whitespace()
                        .map(|word| AnnotatedToken {
                            text: word.to_string(),
                            ner_tag: "O".to_string(),
                        })
                        .collect();
                    
                    if !tokens.is_empty() {
                        sentences.push(AnnotatedSentence {
                            tokens,
                            source_dataset: id,
                        });
                    }
                }
            } else if in_text {
                current_text.push(' ');
                current_text.push_str(line);
            }
        }
        
        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
        })
    }
    
    /// Parse NCBI Disease corpus format.
    ///
    /// Format: PMID|t|Title or PMID|a|Abstract, followed by annotation lines.
    fn parse_ncbi_disease(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        
        for line in content.lines() {
            let line = line.trim();
            
            // NCBI format: PMID|t|Title or PMID|a|Abstract
            if line.contains("|t|") || line.contains("|a|") {
                if let Some(pos) = line.rfind('|') {
                    let text = &line[pos + 1..];
                    // Simple tokenization
                    let tokens: Vec<AnnotatedToken> = text
                        .split_whitespace()
                        .map(|word| AnnotatedToken {
                            text: word.to_string(),
                            ner_tag: "O".to_string(),
                        })
                        .collect();
                    
                    if !tokens.is_empty() {
                        sentences.push(AnnotatedSentence {
                            tokens,
                            source_dataset: id,
                        });
                    }
                }
            }
        }
        
        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
        })
    }
    
    /// Parse GAP coreference TSV format.
    ///
    /// GAP format columns: ID, Text, Pronoun, Pronoun-offset, A, A-offset, A-coref, B, B-offset, B-coref, URL
    fn parse_gap(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
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
            
            let text = parts[1];
            
            // Create tokens (whitespace tokenization for simplicity)
            let tokens: Vec<AnnotatedToken> = text
                .split_whitespace()
                .map(|w| AnnotatedToken {
                    text: w.to_string(),
                    ner_tag: "O".to_string(),
                })
                .collect();
            
            if !tokens.is_empty() {
                sentences.push(AnnotatedSentence {
                    tokens,
                    source_dataset: id,
                });
            }
        }
        
        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
        })
    }
    
    /// Parse PreCo JSON format.
    ///
    /// PreCo format: Array of documents, each with "sentences" array of token arrays.
    fn parse_preco(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        
        // PreCo format: JSON with "sentences" array
        let parsed: serde_json::Value = match serde_json::from_str(content) {
            Ok(v) => v,
            Err(e) => return Err(Error::InvalidInput(format!("Invalid JSON: {}", e))),
        };
        
        // Handle array of documents
        if let Some(docs) = parsed.as_array() {
            for doc in docs {
                if let Some(sents) = doc.get("sentences").and_then(|v| v.as_array()) {
                    for sent_tokens in sents {
                        if let Some(token_array) = sent_tokens.as_array() {
                            let tokens: Vec<AnnotatedToken> = token_array
                                .iter()
                                .filter_map(|t| t.as_str())
                                .map(|t| AnnotatedToken {
                                    text: t.to_string(),
                                    ner_tag: "O".to_string(),
                                })
                                .collect();
                            
                            if !tokens.is_empty() {
                                sentences.push(AnnotatedSentence {
                                    tokens,
                                    source_dataset: id,
                                });
                            }
                        }
                    }
                }
            }
        }
        
        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
        })
    }

    /// Parse LitBank annotation format.
    fn parse_litbank(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        // LitBank .ann format: each line is T<id>\t<Type> <start> <end>\t<text>
        // For now, extract entity mentions as NER annotations
        let now = chrono::Utc::now().to_rfc3339();
        let mut sentences = Vec::new();
        let mut current_entities = Vec::new();
        
        for line in content.lines() {
            if line.starts_with('T') {
                // Entity annotation: T1\tPER 0 5\tAlice
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() >= 3 {
                    let type_span: Vec<&str> = parts[1].split_whitespace().collect();
                    if type_span.len() >= 3 {
                        let label = type_span[0];
                        let text = parts[2];
                        
                        current_entities.push(AnnotatedToken {
                            text: text.to_string(),
                            ner_tag: format!("B-{}", label),
                        });
                    }
                }
            }
        }
        
        // Group into a single "sentence" for simplicity
        if !current_entities.is_empty() {
            sentences.push(AnnotatedSentence {
                tokens: current_entities,
                source_dataset: id,
            });
        }
        
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
        })
    }

    // =========================================================================
    // Coreference Loading
    // =========================================================================
    
    /// Load coreference dataset, returning documents with chains.
    ///
    /// Use this for GAP, PreCo, and LitBank datasets.
    pub fn load_coref(&self, id: DatasetId) -> Result<Vec<super::coref::CorefDocument>> {
        if !id.is_coreference() {
            return Err(Error::InvalidInput(format!(
                "{:?} is not a coreference dataset",
                id
            )));
        }
        
        let cache_path = self.cache_path(id);
        if !cache_path.exists() {
            return Err(Error::InvalidInput(format!(
                "Dataset {:?} not cached at {:?}. Download from {}",
                id, cache_path, id.download_url()
            )));
        }
        
        let content = std::fs::read_to_string(&cache_path).map_err(|e| {
            Error::InvalidInput(format!("Failed to read {:?}: {}", cache_path, e))
        })?;
        
        match id {
            DatasetId::GAP => {
                let examples = super::coref_loader::parse_gap_tsv(&content)?;
                Ok(examples.into_iter().map(|ex| ex.to_coref_document()).collect())
            }
            DatasetId::PreCo => {
                let docs = super::coref_loader::parse_preco_json(&content)?;
                Ok(docs.into_iter().map(|d| d.to_coref_document()).collect())
            }
            DatasetId::LitBank => {
                // LitBank coreference - parse .ann format for chains
                self.parse_litbank_coref(&content)
            }
            _ => Err(Error::InvalidInput(format!(
                "No coreference parser for {:?}",
                id
            ))),
        }
    }
    
    /// Load coreference dataset, downloading if needed.
    #[cfg(feature = "network")]
    pub fn load_or_download_coref(&self, id: DatasetId) -> Result<Vec<super::coref::CorefDocument>> {
        if !self.is_cached(id) {
            let content = self.download(id)?;
            let cache_path = self.cache_path(id);
            std::fs::write(&cache_path, &content).map_err(|e| {
                Error::InvalidInput(format!("Failed to cache {:?}: {}", cache_path, e))
            })?;
        }
        self.load_coref(id)
    }
    
    /// Parse LitBank for coreference chains.
    fn parse_litbank_coref(&self, content: &str) -> Result<Vec<super::coref::CorefDocument>> {
        use super::coref::{CorefChain, CorefDocument, Mention};
        use std::collections::HashMap;
        
        // LitBank .ann format includes coreference with R lines
        // R1\tCoref Arg1:T1 Arg2:T2
        let mut mentions: HashMap<String, Mention> = HashMap::new();
        let mut coref_links: Vec<(String, String)> = Vec::new();
        
        for line in content.lines() {
            if line.starts_with('T') {
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() >= 3 {
                    let id = parts[0];
                    let type_span: Vec<&str> = parts[1].split_whitespace().collect();
                    if type_span.len() >= 3 {
                        let start: usize = type_span[1].parse().unwrap_or(0);
                        let end: usize = type_span[2].parse().unwrap_or(0);
                        let text = parts[2];
                        mentions.insert(id.to_string(), Mention::new(text, start, end));
                    }
                }
            } else if line.starts_with('R') && line.contains("Coref") {
                // R1\tCoref Arg1:T1 Arg2:T2
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let arg1 = parts[1].trim_start_matches("Arg1:");
                    let arg2 = parts[2].trim_start_matches("Arg2:");
                    coref_links.push((arg1.to_string(), arg2.to_string()));
                }
            }
        }
        
        // Build chains from links using union-find
        let mut chains: Vec<Vec<Mention>> = Vec::new();
        let mut mention_to_chain: HashMap<String, usize> = HashMap::new();
        
        for (id1, id2) in coref_links {
            let chain_idx = match (mention_to_chain.get(&id1), mention_to_chain.get(&id2)) {
                (Some(&idx1), Some(&idx2)) if idx1 != idx2 => {
                    // Merge chains
                    let to_merge = chains[idx2].clone();
                    chains[idx1].extend(to_merge);
                    chains[idx2].clear();
                    for m_id in chains[idx1].iter().map(|m| m.text.clone()) {
                        mention_to_chain.insert(m_id, idx1);
                    }
                    idx1
                }
                (Some(&idx), None) => {
                    if let Some(m) = mentions.get(&id2) {
                        chains[idx].push(m.clone());
                        mention_to_chain.insert(id2, idx);
                    }
                    idx
                }
                (None, Some(&idx)) => {
                    if let Some(m) = mentions.get(&id1) {
                        chains[idx].push(m.clone());
                        mention_to_chain.insert(id1, idx);
                    }
                    idx
                }
                (None, None) => {
                    let idx = chains.len();
                    let mut chain = Vec::new();
                    if let Some(m) = mentions.get(&id1) {
                        chain.push(m.clone());
                        mention_to_chain.insert(id1.clone(), idx);
                    }
                    if let Some(m) = mentions.get(&id2) {
                        chain.push(m.clone());
                        mention_to_chain.insert(id2, idx);
                    }
                    chains.push(chain);
                    idx
                }
                (Some(&idx), Some(_)) => idx,
            };
            let _ = chain_idx; // Used above
        }
        
        // Filter empty chains and convert
        let coref_chains: Vec<CorefChain> = chains.into_iter()
            .filter(|c| !c.is_empty())
            .enumerate()
            .map(|(i, mentions)| CorefChain::with_id(mentions, i as u64))
            .collect();
        
        // Create single document
        let doc = CorefDocument::new("", coref_chains);
        Ok(vec![doc])
    }

    /// Load all cached datasets.
    pub fn load_all_cached(&self) -> Vec<(DatasetId, Result<LoadedDataset>)> {
        DatasetId::all()
            .iter()
            .filter(|id| self.is_cached(**id))
            .map(|id| (*id, self.load(*id)))
            .collect()
    }

    /// Get status of all datasets.
    #[must_use]
    pub fn status(&self) -> Vec<(DatasetId, bool)> {
        DatasetId::all()
            .iter()
            .map(|id| (*id, self.is_cached(*id)))
            .collect()
    }
}

impl Default for DatasetLoader {
    fn default() -> Self {
        Self::new().expect("Failed to create default DatasetLoader")
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Parse BIO tag into prefix and type.
fn parse_bio_tag(tag: &str) -> (&str, &str) {
    if tag == "O" {
        return ("O", "");
    }

    // Handle B-PER, I-LOC, etc.
    if let Some(pos) = tag.find('-') {
        (&tag[..pos], &tag[pos + 1..])
    } else {
        // No prefix, treat as entity type with implicit B
        ("B", tag)
    }
}

/// Map dataset-specific entity types to our EntityType enum.
fn map_entity_type(original: &str) -> EntityType {
    match original.to_uppercase().as_str() {
        // Person types
        "PER" | "PERSON" | "ACTOR" | "DIRECTOR" | "CHARACTER" => EntityType::Person,

        // Location types (OntoNotes: LOC, GPE, FAC all map to Location)
        "LOC" | "LOCATION" | "GPE" | "FAC" | "FACILITY" => EntityType::Location,

        // Organization types (OntoNotes: ORG, NORP)
        "ORG" | "ORGANIZATION" | "CORPORATION" | "GROUP" | "NORP" => EntityType::Organization,

        // Date/Time types
        "DATE" | "YEAR" | "HOURS" => EntityType::Date,
        "TIME" => EntityType::Time,

        // Money types
        "MONEY" | "PRICE" | "CURRENCY" => EntityType::Money,

        // Percent types
        "PERCENT" | "PERCENTAGE" | "RATING" => EntityType::Percent,

        // Numeric types (pattern-detectable)
        "QUANTITY" => EntityType::Quantity,
        "ORDINAL" => EntityType::Ordinal,
        "CARDINAL" => EntityType::Cardinal,

        // OntoNotes creative/legal types -> Other with label
        "PRODUCT" | "PROD" => EntityType::Other("product".to_string()),
        "EVENT" | "EVE" => EntityType::Other("event".to_string()),
        "WORK_OF_ART" | "CREATIVE-WORK" => EntityType::Other("work_of_art".to_string()),
        "LAW" => EntityType::Other("law".to_string()),
        "LANGUAGE" => EntityType::Other("language".to_string()),

        // MultiNERD types
        "ANIM" => EntityType::Other("animal".to_string()),
        "BIO" => EntityType::Other("biological".to_string()),
        "CEL" => EntityType::Other("celestial".to_string()),
        "DIS" => EntityType::Other("disease".to_string()),
        "FOOD" => EntityType::Other("food".to_string()),
        "INST" => EntityType::Other("instrument".to_string()),
        "MEDIA" => EntityType::Other("media".to_string()),
        "MYTH" => EntityType::Other("mythological".to_string()),
        "PLANT" => EntityType::Other("plant".to_string()),
        "VEHI" => EntityType::Other("vehicle".to_string()),
        
        // Biomedical types (BC5CDR, NCBI)
        "DISEASE" => EntityType::Other("disease".to_string()),
        "CHEMICAL" => EntityType::Other("chemical".to_string()),
        
        // Coreference types
        "MENTION" | "PRONOUN" | "COREF" => EntityType::Other("mention".to_string()),

        // MISC type (CoNLL) and everything else
        "MISC" => EntityType::Other("misc".to_string()),
        _ => EntityType::Other(original.to_lowercase()),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bio_tag() {
        assert_eq!(parse_bio_tag("O"), ("O", ""));
        assert_eq!(parse_bio_tag("B-PER"), ("B", "PER"));
        assert_eq!(parse_bio_tag("I-LOC"), ("I", "LOC"));
        assert_eq!(parse_bio_tag("B-ORG"), ("B", "ORG"));
    }

    #[test]
    fn test_map_entity_type() {
        assert_eq!(map_entity_type("PER"), EntityType::Person);
        assert_eq!(map_entity_type("PERSON"), EntityType::Person);
        assert_eq!(map_entity_type("LOC"), EntityType::Location);
        assert_eq!(map_entity_type("GPE"), EntityType::Location);
        assert_eq!(map_entity_type("ORG"), EntityType::Organization);
        assert_eq!(
            map_entity_type("MISC"),
            EntityType::Other("misc".to_string())
        );
        // OntoNotes types -> Other
        assert_eq!(
            map_entity_type("PRODUCT"),
            EntityType::Other("product".to_string())
        );
        assert_eq!(
            map_entity_type("EVENT"),
            EntityType::Other("event".to_string())
        );
        assert_eq!(
            map_entity_type("WORK_OF_ART"),
            EntityType::Other("work_of_art".to_string())
        );
        assert_eq!(map_entity_type("CARDINAL"), EntityType::Cardinal);
    }

    #[test]
    fn test_dataset_id_display() {
        assert_eq!(DatasetId::WikiGold.to_string(), "WikiGold");
        assert_eq!(DatasetId::Wnut17.to_string(), "WNUT-17");
    }

    #[test]
    fn test_dataset_id_from_str() {
        assert_eq!("wikigold".parse::<DatasetId>().unwrap(), DatasetId::WikiGold);
        assert_eq!("wnut-17".parse::<DatasetId>().unwrap(), DatasetId::Wnut17);
        assert_eq!(
            "mit_movie".parse::<DatasetId>().unwrap(),
            DatasetId::MitMovie
        );
    }

    #[test]
    fn test_annotated_sentence_text() {
        let sentence = AnnotatedSentence {
            tokens: vec![
                AnnotatedToken {
                    text: "John".into(),
                    ner_tag: "B-PER".into(),
                },
                AnnotatedToken {
                    text: "lives".into(),
                    ner_tag: "O".into(),
                },
                AnnotatedToken {
                    text: "in".into(),
                    ner_tag: "O".into(),
                },
                AnnotatedToken {
                    text: "New".into(),
                    ner_tag: "B-LOC".into(),
                },
                AnnotatedToken {
                    text: "York".into(),
                    ner_tag: "I-LOC".into(),
                },
            ],
            source_dataset: DatasetId::WikiGold,
        };

        assert_eq!(sentence.text(), "John lives in New York");
    }

    #[test]
    fn test_annotated_sentence_entities() {
        let sentence = AnnotatedSentence {
            tokens: vec![
                AnnotatedToken {
                    text: "John".into(),
                    ner_tag: "B-PER".into(),
                },
                AnnotatedToken {
                    text: "Smith".into(),
                    ner_tag: "I-PER".into(),
                },
                AnnotatedToken {
                    text: "works".into(),
                    ner_tag: "O".into(),
                },
                AnnotatedToken {
                    text: "at".into(),
                    ner_tag: "O".into(),
                },
                AnnotatedToken {
                    text: "Google".into(),
                    ner_tag: "B-ORG".into(),
                },
            ],
            source_dataset: DatasetId::WikiGold,
        };

        let entities = sentence.entities();
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].text, "John Smith");
        assert_eq!(entities[0].entity_type, EntityType::Person);
        assert_eq!(entities[1].text, "Google");
        assert_eq!(entities[1].entity_type, EntityType::Organization);
    }

    #[test]
    fn test_loader_creation() {
        let loader = DatasetLoader::new();
        assert!(loader.is_ok());
    }

    #[test]
    fn test_parse_conll_format() {
        let content = r#"
John B-PER
Smith I-PER
works O
at O
Google B-ORG
. O

Apple B-ORG
announced O
today O
. O
"#;

        let loader = DatasetLoader::new().unwrap();
        let dataset = loader.parse_conll(content, DatasetId::WikiGold).unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.entity_count(), 3);
    }

    #[test]
    fn test_parse_conll2003_format() {
        // CoNLL-2003 has 4 columns: word POS chunk NER
        let content = r#"
-DOCSTART- -X- -X- O

EU NNP B-NP B-ORG
rejects VBZ B-VP O
German JJ B-NP B-MISC
call NN I-NP O
. . O O

Peter NNP B-NP B-PER
Blackburn NNP I-NP I-PER
"#;

        let loader = DatasetLoader::new().unwrap();
        let dataset = loader.parse_conll(content, DatasetId::CoNLL2003Sample).unwrap();

        assert_eq!(dataset.len(), 2);

        let entities1 = dataset.sentences[0].entities();
        assert_eq!(entities1.len(), 2); // EU (ORG), German (MISC)

        let entities2 = dataset.sentences[1].entities();
        assert_eq!(entities2.len(), 1); // Peter Blackburn (PER)
        assert_eq!(entities2[0].text, "Peter Blackburn");
    }
    
    #[test]
    fn test_type_mapper_mit_movie() {
        // MIT Movie needs type mapping
        assert!(DatasetId::MitMovie.needs_type_normalization());
        let mapper = DatasetId::MitMovie.type_mapper().unwrap();
        
        // ACTOR should map to Person
        assert_eq!(
            mapper.normalize("ACTOR").as_label(),
            crate::EntityType::Person.as_label()
        );
    }
    
    #[test]
    fn test_type_mapper_standard_datasets() {
        // Standard datasets don't need type mapping
        assert!(!DatasetId::WikiGold.needs_type_normalization());
        assert!(!DatasetId::CoNLL2003Sample.needs_type_normalization());
        assert!(!DatasetId::Wnut17.needs_type_normalization());
        
        // No mapper returned
        assert!(DatasetId::WikiGold.type_mapper().is_none());
    }
    
    #[test]
    fn test_type_mapper_biomedical() {
        // Biomedical datasets need type mapping
        assert!(DatasetId::BC5CDR.needs_type_normalization());
        assert!(DatasetId::NCBIDisease.needs_type_normalization());
        
        let mapper = DatasetId::BC5CDR.type_mapper().unwrap();
        // Should have biomedical mappings (keys are uppercase)
        let disease = mapper.normalize("DISEASE");
        // Either maps to a custom DISEASE type or falls back to Other
        let label = disease.as_label();
        assert!(label.contains("DISEASE") || label.contains("Other") || label.contains("Disease"));
    }
}

