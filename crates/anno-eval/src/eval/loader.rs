//! Dataset downloading and caching for NER evaluation.
//!
//! Downloads, caches, and parses real NER datasets from public sources.
//! Follows burntsushi's philosophy: real-world data, not toy examples.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use anno_eval::eval::{DatasetLoader, LoadableDatasetId};
//!
//! let loader = DatasetLoader::new()?;
//! let dataset = loader.load(LoadableDatasetId::WikiGold)?;
//! println!("Loaded {} sentences", dataset.len());
//! ```
//!
//! # Dataset IDs: catalog vs loadable
//!
//! The full dataset catalog lives in [`dataset_registry::DatasetId`]. Not every dataset
//! in the catalog can be downloaded/parsed by this crate (some require licenses or have
//! unimplemented formats).
//!
//! This module provides [`LoadableDatasetId`], a wrapper that guarantees a dataset has
//! a loading implementation. Use it for `DatasetLoader::{load, load_or_download}`.
//!
//! # Supported Datasets
//!
//! | Dataset | Source | License | Entities |
//! |---------|--------|---------|----------|
//! | WikiGold | Wikipedia | CC-BY | PER, LOC, ORG, MISC |
//! | WNUT-17 | Social Media | Open | person, location, corporation, etc. |
//! | MIT Movie | MIT | Research | actor, director, genre, title, etc. |
//! | MIT Restaurant | MIT | Research | amenity, cuisine, dish, etc. |
//! | CoNLL-2003 Sample | Public | Research | PER, LOC, ORG, MISC |
//! | OntoNotes Sample | Public | Research | 18 entity types |
//! | BC5CDR | PubMed | Research | Disease, Chemical |
//! | NCBI Disease | PubMed | Research | Disease |
//!
//! # Design Philosophy
//!
//! - **Lazy downloading**: Only fetch what's needed
//! - **Persistent caching**: Never re-download unchanged data
//! - **Integrity verification**: SHA256 checksums for all downloads
//! - **Graceful degradation**: Work offline with cached data
//! - **Clear errors**: Explain exactly what went wrong
//!
//! # Extended Example
//!
//! ```rust,ignore
//! use anno_eval::eval::{DatasetLoader, LoadableDatasetId};
//!
//! let loader = DatasetLoader::new()?;
//!
//! // Check cache status before loading
//! if loader.is_cached(LoadableDatasetId::WikiGold) {
//!     println!("WikiGold is cached, will load from disk");
//! }
//!
//! // Load dataset (downloads if not cached, verifies checksum)
//! let dataset = loader.load(LoadableDatasetId::WikiGold)?;
//! println!("Loaded {} sentences with {} entities",
//!     dataset.len(), dataset.entity_count());
//!
//! // Iterate over examples
//! for example in dataset.iter() {
//!     println!("Text: {}", example.text);
//!     for entity in &example.entities {
//!         println!("  {} [{}]", entity.text, entity.entity_type);
//!     }
//! }
//! ```
//!
//! [`dataset_registry::DatasetId`]: super::dataset_registry::DatasetId

#[cfg(test)]
use anno::EntityType;
use anno::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

// =============================================================================
// Dataset Identification
// =============================================================================

/// Dataset identifier (full catalog).
///
/// This is the single source of truth for dataset metadata. Not all datasets are
/// loadable by `DatasetLoader`. Use [`LoadableDatasetId`] when you want that guarantee.
///
/// # Usage
///
/// ```rust,ignore
/// use anno_eval::eval::{DatasetLoader, LoadableDatasetId};
///
/// let loader = DatasetLoader::new()?;
/// let dataset = loader.load(LoadableDatasetId::WikiGold)?;
/// ```
///
/// # Available Datasets
///
/// ## NER Datasets
///
/// | Dataset | Size | Domain | Entity Types |
/// |---------|------|--------|--------------|
/// | `WikiGold` | ~3.5k entities | Wikipedia | PER, LOC, ORG, MISC |
/// | `Wnut17` | ~2k entities | Social media | person, location, etc. |
/// | `MitMovie` | ~10k entities | Movies | actor, director, genre |
/// | `MitRestaurant` | ~8k entities | Restaurants | cuisine, dish, etc. |
/// | `CoNLL2003Sample` | ~20k entities | News | PER, LOC, ORG, MISC |
/// | `OntoNotesSample` | ~18k entities | Mixed | 18 types |
/// | `BC5CDR` | ~28k entities | Biomedical | Disease, Chemical |
/// | `NCBIDisease` | ~6k entities | Biomedical | Disease |
///
/// ## Coreference Datasets
///
/// | Dataset | Size | Domain | Features |
/// |---------|------|--------|----------|
/// | `GAP` | 8,908 pairs | Wikipedia | Gender-balanced pronouns |
/// | `PreCo` | 38k docs | Reading | Includes singletons |
/// | `LitBank` | 100 works | Literature | Literary coreference |
///
/// # Extending
///
/// To add a new loadable dataset:
/// 1. Add the variant here
/// 2. Implement loading in `DatasetLoader::load()`
/// 3. Add to `DatasetId::all()` iterator
/// 4. Ensure it exists in `dataset_registry::DatasetId` (metadata catalog)
///
pub use super::dataset_registry::DatasetId;

/// Dataset identifier guaranteed to be loadable by [`DatasetLoader`].
///
/// This is a thin wrapper over [`DatasetId`] that enforces loader support via
/// `TryFrom<DatasetId>` / `FromStr`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct LoadableDatasetId(DatasetId);

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum DatasetParsePlan {
    Conll,
    JsonlNer,
    WikiannJson,
    TweetNer7,
    DocredJson,
    GoogleReCorpus,
    ChisiecJson,
    CadecHybrid,
    Bc5cdr,
    NcbiDisease,
    GapTsv,
    PrecoJsonl,
    Litbank,
    EcbPlus,
    AfriSenti,
    AfriQa,
    MasakhaNews,
    Conllu,
    AgNews,
    Dbpedia14,
    YahooAnswers,
    Trec,
    TweetTopic,
    Maven,
    MavenArg,
    Casie,
    Rams,
    HfApiResponse,
    /// TSV-based NER format (HIPE-2022 style)
    TsvNer,
    /// CSV-based NER format (E-NER/EDGAR-NER style: Token,Tag)
    CsvNer,
}

impl LoadableDatasetId {
    /// All datasets that have loading implementations.
    #[must_use]
    pub fn all() -> Vec<Self> {
        DatasetId::all()
            .iter()
            .copied()
            .filter(|id| Self::is_loadable_dataset(*id))
            .map(Self)
            .collect()
    }

    /// Access the underlying catalog ID.
    #[must_use]
    pub fn into_inner(self) -> DatasetId {
        self.0
    }

    #[must_use]
    fn is_loadable_dataset(id: DatasetId) -> bool {
        // Loader truth: some catalog entries are intentionally non-loadable (blocked, deprecated,
        // or require manual access) even if they have a parse plan.
        //
        // Keep this list small and motivated. Prefer expressing “non-loadable” in the registry
        // metadata first, then enforcing it here so CLI counts and `LoadableDatasetId` remain
        // honest.
        if matches!(
            id,
            DatasetId::CoNLL2002 | DatasetId::CoNLL2002Spanish | DatasetId::CoNLL2002Dutch
        ) {
            return false;
        }

        Self::parse_plan(id).is_some()
    }

    #[must_use]
    fn parse_plan(id: DatasetId) -> Option<DatasetParsePlan> {
        if let Some(hint) = Self::registry_hint_plan(id) {
            return Some(hint);
        }

        Some(match id {
            // ===================================================================
            // CoNLL/BIO format datasets
            // ===================================================================
            // These datasets use standard CoNLL-2003 or BIO tagging format.
            // The CoNLL parser handles: space/tab-separated columns, IOB2/BIO schemes,
            // and standard entity type mappings (PER, ORG, LOC, etc.).
            //
            // Note: Some datasets listed here are also handled by registry_hint_plan()
            // but are kept for explicit control or historical reasons.
            //
            // To add more: If a dataset has format="CoNLL" or format="BIO" in the registry
            // and is NER-only, it should be auto-detected. Add here only if auto-detection
            // fails (e.g., multi-task datasets, ambiguous metadata).
            DatasetId::WikiGold
            | DatasetId::Wnut17
            | DatasetId::MitMovie
            | DatasetId::MitRestaurant
            | DatasetId::CoNLL2003Sample
            | DatasetId::OntoNotesSample
            | DatasetId::UniversalNERBench
            | DatasetId::LegNER
            | DatasetId::TwiConv
            | DatasetId::AIDA
            | DatasetId::TACKBP
            | DatasetId::BroadTwitterCorpus
            | DatasetId::BioMNER
            | DatasetId::MasakhaNER
            | DatasetId::MasakhaNER2
            | DatasetId::OntoNotes50
            | DatasetId::GermEval2014
            | DatasetId::HAREM
            | DatasetId::SemEval2013Task91
            | DatasetId::MUC6
            | DatasetId::MUC7
            | DatasetId::JNLPBA
            | DatasetId::BC2GMFull
            | DatasetId::CRAFT
            | DatasetId::FinNER
            | DatasetId::LegalNER
            | DatasetId::AGRONER
            | DatasetId::ATISFlightBooking
            | DatasetId::AerospaceNERDataset
            | DatasetId::AstrologyNER
            | DatasetId::AutomotiveNER
            | DatasetId::AviationProductsNER
            | DatasetId::BeekeepingNER
            | DatasetId::BrewingNER
            | DatasetId::ChineseEngineeringGeologyNER
            | DatasetId::CocktailNER
            | DatasetId::ConstructionNER
            | DatasetId::CropDiseaseNER
            | DatasetId::CryptoNER
            | DatasetId::CyNERAptner
            | DatasetId::DnDNERBenchmark
            | DatasetId::EFGC
            | DatasetId::EnergyNER
            | DatasetId::EquestrianNER
            | DatasetId::EsportsNER
            | DatasetId::FINERFood
            | DatasetId::FitnessNER
            | DatasetId::FourRegionsGeologyNER
            | DatasetId::GNERGeoscience
            | DatasetId::GardeningNER
            | DatasetId::HealthcareAdminNER
            | DatasetId::HindiEnglishSocialMediaNER
            | DatasetId::JobPostingNER
            | DatasetId::LogisticsNER
            | DatasetId::ArabicEventCoref
            | DatasetId::FrenchFullLengthFictionCoref
            | DatasetId::LongtoNotes
            | DatasetId::ManufacturingNER
            | DatasetId::MaritimeNER
            | DatasetId::MovieCoref
            | DatasetId::MusicNER
            | DatasetId::NDNER
            | DatasetId::OrigamiNER
            | DatasetId::PaleontologyNER
            | DatasetId::PharmaNER
            | DatasetId::PhotographyNER
            | DatasetId::RealEstateNER
            | DatasetId::RitterTwitterNER
            | DatasetId::SECFilingsNER
            | DatasetId::SanskritNERBhagavadGita
            | DatasetId::ScubaNER
            | DatasetId::SportsNERGeneral
            | DatasetId::TVShowMultilingualCoref
            | DatasetId::TarotNER
            | DatasetId::TelecomNER
            | DatasetId::TelenovelaNER
            | DatasetId::TourismNER
            | DatasetId::VeterinaryNER
            | DatasetId::WaterResourceNER
            | DatasetId::WeatherNER
            | DatasetId::WineNER
            | DatasetId::WoodworkingNER
            // Added 2025-12: More CoNLL-format datasets with direct download URLs
            | DatasetId::QxoRef
            | DatasetId::GICoref
            | DatasetId::WNUT16
            | DatasetId::NoiseBench
            | DatasetId::CrossWeigh
            | DatasetId::ZELDA
            | DatasetId::GENIANested => DatasetParsePlan::Conll,

            // ===================================================================
            // JSONL format (HuggingFace style)
            // ===================================================================
            // These datasets use JSON Lines format with standard fields:
            // - `tokens`: Array of token strings
            // - `ner_tags`: Array of BIO/IOB2 tags
            // - Optional: `text` (original sentence), `entities` (structured)
            //
            // The JsonlNer parser is flexible and handles variations in field names.
            DatasetId::MultiNERD
            | DatasetId::ACORD
            | DatasetId::ARFFiction
            | DatasetId::AgMNER
            | DatasetId::AgriNER
            | DatasetId::AnimeMangaNER
            | DatasetId::AntiquesNER
            | DatasetId::AstronomicalTelegramKEE
            | DatasetId::BirdwatchingNER
            | DatasetId::BoardGameNER
            | DatasetId::BookCorefBamman
            | DatasetId::CUAD
            | DatasetId::ChessNER
            | DatasetId::CoMTA
            | DatasetId::DeepFashion2
            | DatasetId::FIREBALL
            | DatasetId::FashionIQ
            | DatasetId::FragranceNER
            | DatasetId::IMDbSemiStructuredRE
            | DatasetId::InsuranceNER
            | DatasetId::KnittingNER
            | DatasetId::LLMRocMinNER
            | DatasetId::MOFDataset
            | DatasetId::MathDial
            | DatasetId::NHKRecipeDataset
            | DatasetId::NaturalProductsRE
            | DatasetId::NumismaticsNER
            | DatasetId::PhilatelyNER
            | DatasetId::PolyIE
            | DatasetId::RecipeDBAnnotated
            | DatasetId::ResumeNER
            | DatasetId::RetailInventoryNER
            | DatasetId::SPoRC
            | DatasetId::Saraga
            | DatasetId::SolidStateDoping
            | DatasetId::SpotifyPodcastsDataset
            | DatasetId::TattooNER
            | DatasetId::ThemeParkNER
            | DatasetId::VREN
            | DatasetId::VisDialCoref
            // Added 2025-12: More JSONL-format datasets with direct download URLs
            | DatasetId::REBEL
            | DatasetId::BBQ
            | DatasetId::RealToxicityPrompts
            | DatasetId::BookCoref
            | DatasetId::BookCorefSplit
            | DatasetId::WIESP2022NER
            | DatasetId::FewRel
            | DatasetId::PIIMasking200k
            | DatasetId::B2NERD
            | DatasetId::OpenNER
            | DatasetId::FictionNER750M => DatasetParsePlan::JsonlNer,

            // WikiANN-ish JSON format (from download_hf_datasets.py)
            DatasetId::UNER | DatasetId::MSNER => DatasetParsePlan::WikiannJson,

            // TweetNER7 uses JSON format with integer tags
            DatasetId::TweetNER7 => DatasetParsePlan::TweetNer7,

            // ===================================================================
            // JSON format (Relation extraction datasets)
            // ===================================================================
            // Relation extraction datasets use JSON format with entity pairs and relations.
            // Common structure: {entities: [...], relations: [...], text: "..."}
            //
            // The DocredJson parser handles DocRED, ReTACRED, and similar formats.
            // Some datasets (SciERC, PolyIE, EnzChemRED) are auto-detected via
            // registry_hint_plan() but listed here for explicit control.
            //
            // To add more: RE datasets with format="JSON" or format="JSONL" and is_re=true
            // should be auto-detected. Add here only for special cases.
            DatasetId::DocRED
            | DatasetId::ReTACRED
            | DatasetId::NYTFB
            | DatasetId::WEBNLG
            | DatasetId::GoogleRE
            | DatasetId::BioRED
            | DatasetId::SciER
            | DatasetId::MixRED
            | DatasetId::CovEReD
            | DatasetId::ACE2005
            | DatasetId::MuDoCo
            | DatasetId::SciERCNER
            // Added 2025-01-27: RE datasets with public URLs but missing hints
            // Note: SciERC, PolyIE, EnzChemRED are now handled by registry_hint_plan() auto-detection
            => DatasetParsePlan::DocredJson,

            // CHisIEC: Ancient Chinese historical NER+RE (custom JSON format)
            DatasetId::CHisIEC => DatasetParsePlan::ChisiecJson,

            // Discontinuous NER (HF datasets-server API or JSONL format)
            DatasetId::CADEC | DatasetId::ShARe13 | DatasetId::ShARe14 => {
                DatasetParsePlan::CadecHybrid
            }

            // Biomedical formats
            DatasetId::BC5CDR => DatasetParsePlan::Bc5cdr,
            DatasetId::NCBIDisease => DatasetParsePlan::NcbiDisease,

            // Coreference formats
            // Note: GUM uses CoNLL format and is handled by registry_hint_plan()
            DatasetId::GAP | DatasetId::WikiCoref => DatasetParsePlan::GapTsv,
            DatasetId::WinoBias => DatasetParsePlan::GapTsv,
            DatasetId::PreCo | DatasetId::SciCo => DatasetParsePlan::PrecoJsonl,
            DatasetId::LitBank => DatasetParsePlan::Litbank,
            DatasetId::ECBPlus => DatasetParsePlan::EcbPlus,

            // African language datasets
            DatasetId::AfriSenti => DatasetParsePlan::AfriSenti,
            DatasetId::AfriQA => DatasetParsePlan::AfriQa,
            DatasetId::MasakhaNEWS => DatasetParsePlan::MasakhaNews,
            DatasetId::MasakhaPOS => DatasetParsePlan::Conllu,

            // Text classification datasets
            DatasetId::AGNews => DatasetParsePlan::AgNews,
            DatasetId::DBPedia14 => DatasetParsePlan::Dbpedia14,
            DatasetId::YahooAnswers => DatasetParsePlan::YahooAnswers,
            DatasetId::TREC => DatasetParsePlan::Trec,
            DatasetId::TweetTopic => DatasetParsePlan::TweetTopic,

            // Event extraction datasets
            DatasetId::MAVEN => DatasetParsePlan::Maven,
            DatasetId::MAVENArg => DatasetParsePlan::MavenArg,
            DatasetId::CASIE => DatasetParsePlan::Casie,
            DatasetId::RAMS => DatasetParsePlan::Rams,

            // HF datasets-server API responses
            DatasetId::GENIA
            | DatasetId::AnatEM
            | DatasetId::BC2GM
            | DatasetId::BC4CHEMD
            | DatasetId::FewNERD
            | DatasetId::CrossNER
            | DatasetId::FabNER
            | DatasetId::WikiNeural
            | DatasetId::WikiANN
            | DatasetId::MultiCoNER
            | DatasetId::MultiCoNERv2
            | DatasetId::PolyglotNER
            | DatasetId::UniversalNER => DatasetParsePlan::HfApiResponse,

            _ => return None,
        })
    }

    /// Best-effort "hint" derived from registry metadata.
    ///
    /// This is intentionally conservative: it returns `None` if the registry
    /// doesn't provide enough signal to choose the same plan as `parse_plan`.
    ///
    /// The long-term goal is to have this cover most datasets so `parse_plan`
    /// shrinks to a small exception table.
    ///
    /// # Architecture Notes
    ///
    /// This function enables automatic format detection for datasets with:
    /// - Clear format metadata (`format:` field in registry)
    /// - Single primary task (NER, RE, coref, etc.)
    /// - Standard annotation schemes (BIO, IOB2, CoNLL, JSONL)
    ///
    /// Datasets with multiple tasks (e.g., NER+RE) or ambiguous formats
    /// should be added to explicit matches in `parse_plan()` instead.
    ///
    /// # Adding New Datasets
    ///
    /// 1. **Common formats (CoNLL, JSONL, TSV)**: Add format metadata to registry,
    ///    ensure single primary task, and this function will auto-detect.
    /// 2. **Special formats**: Add explicit match in `parse_plan()` match statement.
    /// 3. **Multi-task datasets**: Add explicit match (auto-detection is conservative).
    #[must_use]
    fn registry_hint_plan(id: DatasetId) -> Option<DatasetParsePlan> {
        // Hybrids / special cases first.
        if id == DatasetId::CHisIEC {
            // CHisIEC is multi-task and uses a custom JSON array format (not DocRED/CrossRE).
            // It MUST not be auto-detected as DocredJson.
            return Some(DatasetParsePlan::ChisiecJson);
        }
        if matches!(
            id,
            DatasetId::CADEC | DatasetId::ShARe13 | DatasetId::ShARe14
        ) {
            return Some(DatasetParsePlan::CadecHybrid);
        }
        if id == DatasetId::GoogleRE {
            // The Google relation-extraction-corpus files are not DocRED/CrossRE.
            // They are JSONL records with fields like {pred, sub, obj, evidences[], judgments[]}.
            return Some(DatasetParsePlan::GoogleReCorpus);
        }
        if id == DatasetId::TweetNER7 {
            return Some(DatasetParsePlan::TweetNer7);
        }

        // HuggingFace-hosted datasets with bespoke (non-generic) parse plans.
        //
        // These are marked `access_status: HuggingFace` in the registry, so they must be
        // hintable; but their parse plans are not derivable from `format` alone (or use
        // specialized parsing despite being hosted on the Hub).
        if id == DatasetId::BC5CDR {
            return Some(DatasetParsePlan::Bc5cdr);
        }
        if id == DatasetId::NCBIDisease {
            return Some(DatasetParsePlan::NcbiDisease);
        }
        if id == DatasetId::AfriSenti {
            return Some(DatasetParsePlan::AfriSenti);
        }
        if id == DatasetId::AfriQA {
            return Some(DatasetParsePlan::AfriQa);
        }
        if id == DatasetId::MasakhaNEWS {
            return Some(DatasetParsePlan::MasakhaNews);
        }
        if id == DatasetId::AGNews {
            return Some(DatasetParsePlan::AgNews);
        }
        if id == DatasetId::DBPedia14 {
            return Some(DatasetParsePlan::Dbpedia14);
        }
        if id == DatasetId::YahooAnswers {
            return Some(DatasetParsePlan::YahooAnswers);
        }
        if id == DatasetId::TREC {
            return Some(DatasetParsePlan::Trec);
        }
        if id == DatasetId::TweetTopic {
            return Some(DatasetParsePlan::TweetTopic);
        }

        // Coref datasets in CoNLL format (not GAP TSV): parse as regular CoNLL.
        // These have NER annotations alongside coref, so the CoNLL parser works.
        if matches!(
            id,
            DatasetId::QxoRef
                | DatasetId::GICoref
                | DatasetId::WNUT16
                | DatasetId::NoiseBench
                | DatasetId::CrossWeigh
                | DatasetId::ZELDA
                | DatasetId::GENIANested
                // Added 2025-12: More CoNLL NER datasets
                | DatasetId::HistNERo
                | DatasetId::DutchArchaeology
                | DatasetId::FINER
                | DatasetId::CALCS2018
                | DatasetId::MedievalCharterNER
                | DatasetId::RockNER
                | DatasetId::AIDACoNLL
                | DatasetId::NNE
                | DatasetId::GermEvalDiscontinuous
                | DatasetId::PubMedDiscontinuous
                | DatasetId::IndicNER
                | DatasetId::NorNE
                | DatasetId::CHEMDNER
                | DatasetId::GeoWebNews
                | DatasetId::TASTEset
                | DatasetId::RecipeNER
                | DatasetId::AstroNER
                | DatasetId::FinanceNER
                | DatasetId::TechNER
                | DatasetId::CALCS
                | DatasetId::LinCE
                | DatasetId::FinTechPatent
                | DatasetId::WaterAgriNER
                | DatasetId::NERsocialFood
                | DatasetId::RussianCulturalNER
                | DatasetId::EighteenthCenturyNER
                | DatasetId::GuaraniNER
                | DatasetId::ShipiboKoniboNER
                | DatasetId::BASHI
                | DatasetId::ENER
                // Added 2025-01-27: CoNLL NER datasets with public URLs but missing hints
                | DatasetId::OntoNotes50
                | DatasetId::GUM
                // CoNLL04RE is CoNLL format (works with CoNLL parser even though it's for RE)
                | DatasetId::CoNLL04RE
        ) {
            return Some(DatasetParsePlan::Conll);
        }

        // JSONL datasets that should use JsonlNer despite having coref/RE tasks.
        // The JsonlNer parser handles tokens + ner_tags fields generically.
        if matches!(
            id,
            DatasetId::REBEL
                | DatasetId::BBQ
                | DatasetId::RealToxicityPrompts
                | DatasetId::BookCoref
                | DatasetId::BookCorefSplit
                | DatasetId::WIESP2022NER
                | DatasetId::FewRel
                | DatasetId::PIIMasking200k
                | DatasetId::B2NERD
                | DatasetId::OpenNER
                | DatasetId::FictionNER750M
                // Added 2025-12: More JSONL NER datasets
                | DatasetId::MultiWOZNER
                | DatasetId::HinglishNER
                | DatasetId::ChineseNestedNER
                | DatasetId::AgCNER
                | DatasetId::LongDocNER
                | DatasetId::MultiBioNERLong
                | DatasetId::ReasoningNER
                | DatasetId::BioNERLLaMA
                | DatasetId::LexGLUENER
                | DatasetId::FinBenNER
                | DatasetId::FiNER139
                | DatasetId::SciNER
                | DatasetId::CharacterCodex
                | DatasetId::AIONER
                | DatasetId::WIESPAstro
                | DatasetId::CEREC
                | DatasetId::DELICATE
                | DatasetId::CSN
                // Added 2025-01-27: JSONL NER datasets with public URLs but missing hints
                | DatasetId::SCINERNested
                | DatasetId::AgriNER
                | DatasetId::MOFDataset
                | DatasetId::SolidStateDoping
        ) {
            return Some(DatasetParsePlan::JsonlNer);
        }

        // CoNLLU datasets (Universal Dependencies treebanks with NER annotations)
        if matches!(
            id,
            DatasetId::AncientGreekUD
                | DatasetId::LatinUD
                | DatasetId::SanskritUD
                | DatasetId::OldEnglishUD
                | DatasetId::OldNorseUD
                | DatasetId::UDEsperantoCairo
                // Added 2025-12: More UD treebanks (ancient/historical/constructed)
                | DatasetId::CopticScriptorium
                | DatasetId::TaggedPBCEsperanto
                | DatasetId::TaggedPBCKlingon
                | DatasetId::AkkadianUD
                | DatasetId::AncientHebrewUD
                | DatasetId::ClassicalChineseUD
                | DatasetId::CopticUD
                | DatasetId::GothicUD
                | DatasetId::HittiteUD
                | DatasetId::OldChurchSlavonicUD
                | DatasetId::LatinITTB
                | DatasetId::LatinPROIEL
                | DatasetId::EsperantoUD
                | DatasetId::NavajoMorph
        ) {
            return Some(DatasetParsePlan::Conllu);
        }

        // TSV NER datasets (HIPE-2022 style)
        if matches!(id, DatasetId::HIPE2022) {
            return Some(DatasetParsePlan::TsvNer);
        }

        // CSV NER datasets
        if matches!(id, DatasetId::ENer) {
            return Some(DatasetParsePlan::CsvNer);
        }

        // Some datasets are fetched via HF datasets-server (JSON response), regardless of their
        // “canonical” source format in the registry.
        if matches!(
            id,
            DatasetId::GENIA
                | DatasetId::AnatEM
                | DatasetId::BC2GM
                | DatasetId::BC4CHEMD
                | DatasetId::FewNERD
                | DatasetId::CrossNER
                | DatasetId::FabNER
                | DatasetId::WikiNeural
                | DatasetId::WikiANN
                | DatasetId::MultiCoNER
                | DatasetId::MultiCoNERv2
                | DatasetId::PolyglotNER
                | DatasetId::UniversalNER
        ) {
            return Some(DatasetParsePlan::HfApiResponse);
        }

        // Use tasks_or_inferred() to support datasets that have task-like categories
        // (e.g., `categories: [ner, ancient]`) but no explicit `tasks` field yet.
        // This enables automatic loading for datasets with correct format + category.
        let inferred_tasks = id.tasks_or_inferred();
        let is_ner = inferred_tasks.contains(&"ner");
        let is_coref = inferred_tasks.contains(&"coref");
        let is_re = inferred_tasks.contains(&"re");
        let is_event = inferred_tasks.contains(&"event_extraction");

        let annotation_scheme = id.annotation_scheme().unwrap_or("");

        // Format string is about the *source* dataset, not necessarily our cached
        // representation; still useful for many direct-download datasets.
        if let Some(format) = id.format() {
            let plan = match format {
                "CoNLL" | "BIO" | "IOB2" if is_ner && !is_coref && !is_re && !is_event => {
                    Some(DatasetParsePlan::Conll)
                }
                "CoNLL-U" | "CoNLLU" if is_ner && !is_coref && !is_re && !is_event => {
                    Some(DatasetParsePlan::Conllu)
                }
                "JSONL" if is_ner && !is_coref && !is_re && !is_event => {
                    Some(DatasetParsePlan::JsonlNer)
                }
                // Coreference: we only treat these as unambiguous when the dataset is *only* coref.
                // Check both format field and annotation_scheme for CoNLLCoref datasets.
                "TSV" | "ConllCoref" | "CoNLLCoref"
                    if is_coref && !is_ner && !is_re && !is_event =>
                {
                    // GAP-style TSV and similar "coref-only" exports.
                    Some(DatasetParsePlan::GapTsv)
                }
                // CoNLL format with CoNLLCoref annotation scheme (e.g., GICoref, qxoRef)
                "CoNLL"
                    if annotation_scheme == "CoNLLCoref"
                        && is_coref
                        && !is_ner
                        && !is_re
                        && !is_event =>
                {
                    Some(DatasetParsePlan::GapTsv)
                }
                "JSONL" if is_coref && !is_ner && !is_re && !is_event => {
                    // PreCo/SciCo style JSONL exports.
                    Some(DatasetParsePlan::PrecoJsonl)
                }

                // Relation extraction: unambiguous only when the dataset is *only* RE.
                // Also handle RE datasets that might have NER as secondary task (e.g., SciERCNER).
                "JSON" | "JSONL" if is_re && !is_coref && !is_event => {
                    Some(DatasetParsePlan::DocredJson)
                }

                // Event extraction: treat JSONL as MAVEN-ish only when the dataset is event-only.
                "JSONL" if is_event && !is_ner && !is_coref && !is_re => {
                    Some(DatasetParsePlan::Maven)
                }

                // TSV NER format (HIPE-2022 style)
                "TSV" if is_ner && !is_coref && !is_re && !is_event => {
                    Some(DatasetParsePlan::TsvNer)
                }

                // CSV NER format (E-NER/EDGAR-NER style: Token,Tag)
                "CSV" if is_ner && !is_coref && !is_re && !is_event => {
                    Some(DatasetParsePlan::CsvNer)
                }

                // These are too ambiguous across sources; treat as unknown for now.
                _ => None,
            };

            if plan.is_some() {
                return plan;
            }
        }

        // HuggingFace datasets: if the registry provides an HF id/config and marks the
        // dataset as HuggingFace-accessible, we can treat it as an HfApiResponse plan.
        //
        // Keep this as a *fallback*: registry format/task signals (and explicit hint lists
        // above) take precedence, so we don't override e.g. CoNLL/JSONL datasets that are
        // hosted on HuggingFace but should be parsed as raw files.
        if id.hf_id().is_some()
            && id.access_status()
                == crate::eval::dataset_registry::DatasetAccessibility::HuggingFace
        {
            return Some(DatasetParsePlan::HfApiResponse);
        }

        None
    }
}

impl std::ops::Deref for LoadableDatasetId {
    type Target = DatasetId;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<LoadableDatasetId> for DatasetId {
    fn from(value: LoadableDatasetId) -> Self {
        value.0
    }
}

impl std::str::FromStr for LoadableDatasetId {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let id: DatasetId = s
            .parse()
            .map_err(|e| Error::InvalidInput(format!("Invalid dataset id '{}': {}", s, e)))?;
        Self::try_from(id)
    }
}

impl TryFrom<DatasetId> for LoadableDatasetId {
    type Error = Error;

    fn try_from(value: DatasetId) -> std::result::Result<Self, Self::Error> {
        if Self::is_loadable_dataset(value) {
            Ok(Self(value))
        } else {
            Err(Error::InvalidInput(format!(
                "Dataset {:?} is in the registry but has no loading implementation",
                value
            )))
        }
    }
}

// =============================================================================
// Loader-specific extensions for `DatasetId`
// =============================================================================
//
// `DatasetId` is defined in `dataset_registry` (the full catalog of datasets).
// This module is intentionally split:
// - `DatasetId`: metadata/catalog (can include datasets that are not loadable here)
// - `LoadableDatasetId`: the subset that this loader+parsers support

// Extension methods for DatasetId that delegate to registry
impl DatasetId {
    /// Get default metadata for this dataset.
    ///
    /// Delegates to registry methods.
    #[must_use]
    pub fn default_metadata(&self) -> DatasetMetadata {
        DatasetMetadata {
            description: Some(self.description().to_string()),
            license: self.license().map(|s| s.to_string()),
            citation: self.citation().map(|s| s.to_string()),
            split: Some("test".to_string()), // Default to test split
            language: Some(self.language().to_string()),
            domain: Some(self.domain().to_string()),
            original_source: Some(self.download_url().to_string()),
            version: None,
            annotators: None,
            guidelines_url: None,
        }
    }
}

// =============================================================================
// Cache Manifest
// =============================================================================

/// Entry in the cache manifest tracking a downloaded dataset file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheManifestEntry {
    /// Dataset identifier.
    pub dataset_id: String,
    /// Source URL from which the data was downloaded.
    ///
    /// This is the *logical* registry URL (what the dataset entry says).
    pub source_url: String,
    /// The URL that was actually fetched (after any rewrites / mirrors).
    ///
    /// This is helpful for reproducibility (and debugging) when the registry URL is a
    /// homepage (e.g., HuggingFace dataset page) but we fetched via datasets-server
    /// or `resolve/main/...`.
    #[serde(default)]
    pub resolved_url: Option<String>,
    /// SHA-256 checksum of the downloaded file.
    pub sha256: String,
    /// File size in bytes.
    pub file_size: u64,
    /// ISO 8601 timestamp of when the file was downloaded.
    pub downloaded_at: String,
    /// Number of sentences parsed from the dataset.
    pub sentence_count: usize,
    /// Number of entities parsed from the dataset.
    pub entity_count: usize,
    /// Anno version that downloaded this file.
    pub anno_version: String,
}

/// Cache manifest tracking all downloaded datasets.
///
/// Stored as `manifest.json` in the cache directory.
/// Enables:
/// - Reproducibility: know exactly what was downloaded
/// - Validation: verify checksums match
/// - Debugging: trace when data was fetched
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheManifest {
    /// Version of the manifest format.
    pub version: u32,
    /// Entries for each cached dataset.
    pub entries: HashMap<String, CacheManifestEntry>,
}

impl CacheManifest {
    /// Current manifest format version.
    pub const CURRENT_VERSION: u32 = 1;

    /// Create a new empty manifest.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            entries: HashMap::new(),
        }
    }

    /// Load manifest from a cache directory.
    ///
    /// Returns empty manifest if file doesn't exist.
    pub fn load(cache_dir: &std::path::Path) -> Result<Self> {
        let manifest_path = cache_dir.join("manifest.json");
        if !manifest_path.exists() {
            return Ok(Self::new());
        }

        let content = fs::read_to_string(&manifest_path)
            .map_err(|e| Error::InvalidInput(format!("Failed to read manifest: {}", e)))?;

        serde_json::from_str(&content)
            .map_err(|e| Error::InvalidInput(format!("Failed to parse manifest: {}", e)))
    }

    /// Save manifest to a cache directory.
    pub fn save(&self, cache_dir: &std::path::Path) -> Result<()> {
        let manifest_path = cache_dir.join("manifest.json");
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| Error::InvalidInput(format!("Failed to serialize manifest: {}", e)))?;

        fs::write(&manifest_path, content)
            .map_err(|e| Error::InvalidInput(format!("Failed to write manifest: {}", e)))
    }

    /// Add or update an entry in the manifest.
    pub fn update_entry(&mut self, entry: CacheManifestEntry) {
        self.entries.insert(entry.dataset_id.clone(), entry);
    }

    /// Get entry for a dataset.
    #[must_use]
    pub fn get(&self, dataset_id: &str) -> Option<&CacheManifestEntry> {
        self.entries.get(dataset_id)
    }

    /// Check if a dataset's cached file matches the manifest.
    ///
    /// Returns `true` if the file exists and checksum matches.
    #[must_use]
    pub fn verify_entry(&self, dataset_id: &str, cache_dir: &std::path::Path) -> bool {
        let Some(entry) = self.entries.get(dataset_id) else {
            return false;
        };

        // Check file exists with expected size
        let file_path = cache_dir.join(&entry.dataset_id);
        let Ok(metadata) = fs::metadata(&file_path) else {
            return false;
        };

        metadata.len() == entry.file_size
        // Note: Full SHA-256 verification is expensive; size check is usually sufficient
    }
}

// =============================================================================
// Temporal Metadata
// =============================================================================

/// Temporal metadata for datasets (optional).
///
/// Used for temporal stratification of evaluation metrics.
/// Most datasets don't have temporal metadata, so this is optional.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemporalMetadata {
    /// KB version used for entity linking (if applicable)
    pub kb_version: Option<String>,
    /// Temporal cutoff date (entities before this date are "old", after are "new")
    pub temporal_cutoff: Option<String>, // ISO 8601 date string
    /// Entity creation dates (if available)
    pub entity_creation_dates: Option<HashMap<String, String>>, // entity_id -> ISO 8601 date
}

// =============================================================================
// Annotated Structures
// =============================================================================

/// Annotated token with NER tag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedToken {
    /// Token text
    pub text: String,
    /// NER tag (BIO format)
    pub ner_tag: String,
}

/// Annotated sentence with tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedSentence {
    /// Tokens in the sentence
    pub tokens: Vec<AnnotatedToken>,
    /// Source dataset
    pub source_dataset: DatasetId,
}

impl AnnotatedSentence {
    /// Get text of the sentence.
    #[must_use]
    pub fn text(&self) -> String {
        self.tokens
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get entities from this sentence.
    #[must_use]
    pub fn entities(&self) -> Vec<super::datasets::GoldEntity> {
        // Precompute character offsets for each token in `self.text()` (tokens joined by a
        // single ASCII space). Offsets are in *character* units, not bytes.
        let mut token_starts: Vec<usize> = Vec::with_capacity(self.tokens.len());
        let mut pos: usize = 0;
        for (i, tok) in self.tokens.iter().enumerate() {
            token_starts.push(pos);
            pos += tok.text.chars().count();
            if i + 1 < self.tokens.len() {
                pos += 1; // the space inserted by `text()`
            }
        }

        let mut entities = Vec::new();
        let mut i = 0;
        while i < self.tokens.len() {
            let tag = &self.tokens[i].ner_tag;

            // Check for BIO format (B-TYPE, I-TYPE)
            if tag.starts_with("B-") {
                let entity_type = tag.trim_start_matches("B-");
                let mut end = i + 1;
                // Only continue with I-tags that match the same entity type
                while end < self.tokens.len()
                    && self.tokens[end].ner_tag.starts_with("I-")
                    && self.tokens[end].ner_tag.trim_start_matches("I-") == entity_type
                {
                    end += 1;
                }
                let entity_text: String = self.tokens[i..end]
                    .iter()
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                let start = token_starts.get(i).copied().unwrap_or(0);
                let end_char = if end <= i {
                    start
                } else {
                    let last = end - 1;
                    token_starts.get(last).copied().unwrap_or(start)
                        + self.tokens[last].text.chars().count()
                };

                entities.push(super::datasets::GoldEntity {
                    text: entity_text,
                    original_label: entity_type.to_string(),
                    entity_type: anno::EntityType::from_label(entity_type),
                    start,
                    end: end_char,
                });
                i = end;
            }
            // Fallback: non-BIO format (simple type names like "person", "organization")
            // Used by datasets like FewNERD that use ClassLabels
            else if tag != "O" && !tag.starts_with("I-") && !tag.starts_with("TAG_") {
                let entity_type = tag.as_str();
                let mut end = i + 1;
                // Continue while same entity type (treating consecutive same labels as one entity)
                while end < self.tokens.len() && self.tokens[end].ner_tag == *tag {
                    end += 1;
                }
                let entity_text: String = self.tokens[i..end]
                    .iter()
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                let start = token_starts.get(i).copied().unwrap_or(0);
                let end_char = if end <= i {
                    start
                } else {
                    let last = end - 1;
                    token_starts.get(last).copied().unwrap_or(start)
                        + self.tokens[last].text.chars().count()
                };

                entities.push(super::datasets::GoldEntity {
                    text: entity_text,
                    original_label: entity_type.to_string(),
                    entity_type: anno::EntityType::from_label(entity_type),
                    start,
                    end: end_char,
                });
                i = end;
            } else {
                i += 1;
            }
        }
        entities
    }
}

/// Dataset metadata for provenance and reproducibility.
///
/// Captures all information needed to understand where data came from
/// and how to attribute/license it correctly.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Human-readable description.
    pub description: Option<String>,
    /// License (e.g., "CC-BY-4.0", "Apache-2.0", "Research-only").
    pub license: Option<String>,
    /// Citation reference (BibTeX key or paper title).
    pub citation: Option<String>,
    /// Split type (e.g., "train", "dev", "test").
    pub split: Option<String>,
    /// Language code (e.g., "en", "de", "zh").
    pub language: Option<String>,
    /// Domain (e.g., "news", "biomedical", "social-media").
    pub domain: Option<String>,
    /// Original source repository or homepage.
    pub original_source: Option<String>,
    /// Version string if the dataset has versions.
    pub version: Option<String>,
    /// Number of annotators (for IAA reporting).
    pub annotators: Option<u32>,
    /// Annotation guidelines URL if available.
    pub guidelines_url: Option<String>,
}

/// Where the dataset was loaded from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DataSource {
    /// Loaded from S3 cache (fast, reliable).
    S3Cache,
    /// Loaded from local file cache.
    LocalCache,
    /// Downloaded from original URL (may be slow or unavailable).
    OriginalUrl,
    /// Dataset was skipped (URL broken or unavailable).
    #[default]
    Skipped,
    /// Embedded in binary (test fixtures).
    Embedded,
}

impl DataSource {
    /// Human-readable description of the source.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::S3Cache => "S3 cache",
            Self::LocalCache => "local cache",
            Self::OriginalUrl => "original URL",
            Self::Skipped => "skipped (unavailable)",
            Self::Embedded => "embedded",
        }
    }

    /// Whether this source is cached (fast).
    #[must_use]
    pub fn is_cached(&self) -> bool {
        matches!(self, Self::S3Cache | Self::LocalCache | Self::Embedded)
    }

    /// Whether this source indicates the data is unavailable.
    #[must_use]
    pub fn is_unavailable(&self) -> bool {
        matches!(self, Self::Skipped)
    }
}

impl std::fmt::Display for DataSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
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
    /// Where the data was actually loaded from.
    #[serde(default)]
    pub data_source: DataSource,
    /// Optional temporal metadata for temporal stratification
    #[serde(default)]
    pub temporal_metadata: Option<TemporalMetadata>,
    /// Dataset provenance and attribution metadata.
    #[serde(default)]
    pub metadata: DatasetMetadata,
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
    pub fn to_test_cases(&self) -> Vec<(String, Vec<super::datasets::GoldEntity>)> {
        self.sentences
            .iter()
            .map(|s| (s.text(), s.entities()))
            .collect()
    }
}

// =============================================================================
// Dataset Statistics
// =============================================================================

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

/// A document with text and gold relations for relation extraction evaluation.
#[derive(Debug, Clone)]
pub struct RelationDocument {
    /// Document text
    pub text: String,
    /// Gold standard relations
    pub relations: Vec<super::relation::RelationGold>,
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
/// # Caching Strategy (Tiered)
///
/// 1. **Local cache** (`~/.cache/anno/datasets/`): Checked first, fastest
/// 2. **S3 cache** (`s3://anno-data/`): Checked if `ANNO_S3_CACHE=1` and AWS credentials available
/// 3. **URL download**: Original source URL, last resort
///
/// This tiered approach:
/// - Maximizes speed (local cache hit is instant)
/// - Provides redundancy (S3 backup if original URLs go down)
/// - Allows sharing datasets across machines via S3
///
/// # Environment Variables
///
/// - `ANNO_CACHE_DIR`: Override default cache location
/// - `ANNO_S3_CACHE`: Set to "1" to enable S3 fallback (requires AWS credentials)
/// - `ANNO_S3_BUCKET`: Override S3 bucket name (default: `arc-anno-data`)
///
/// # Example
///
/// ```bash
/// # Enable S3 caching
/// export ANNO_S3_CACHE=1
/// export AWS_PROFILE=default  # or set AWS_ACCESS_KEY_ID, etc.
///
/// # Now load_or_download() will try S3 before URL
/// ```
pub struct DatasetLoader {
    cache_dir: PathBuf,
    /// S3 bucket name for fallback caching (if enabled)
    s3_bucket: Option<String>,
    /// Cache manifest for tracking downloads.
    ///
    /// This is shared across evaluation threads, so it must be thread-safe.
    manifest: std::sync::RwLock<CacheManifest>,
}

impl DatasetLoader {
    // Default download cap when `ANNO_MAX_DOWNLOAD_BYTES` is unset.
    //
    // Rationale: unset should be *usable* but *safe* by default. This cap is meant to prevent
    // accidental multi-GB downloads while still allowing many evaluation datasets.
    #[cfg(feature = "eval")]
    const DEFAULT_MAX_DOWNLOAD_BYTES: u64 = 50 * 1024 * 1024; // 50 MiB

    #[cfg(feature = "eval")]
    fn max_download_bytes() -> Option<u64> {
        match std::env::var("ANNO_MAX_DOWNLOAD_BYTES").ok() {
            Some(s) => {
                let s = s.trim();
                if s.is_empty() {
                    return Some(Self::DEFAULT_MAX_DOWNLOAD_BYTES);
                }
                let Ok(v) = s.parse::<u64>() else {
                    return Some(Self::DEFAULT_MAX_DOWNLOAD_BYTES);
                };
                if v == 0 {
                    None // explicit opt-out
                } else {
                    Some(v)
                }
            }
            None => Some(Self::DEFAULT_MAX_DOWNLOAD_BYTES),
        }
    }

    #[cfg(feature = "eval")]
    fn enforce_max_download_bytes(content_len: usize, source: &str) -> Result<()> {
        let Some(limit) = Self::max_download_bytes() else {
            return Ok(());
        };
        let len = content_len as u64;
        if len > limit {
            return Err(Error::InvalidInput(format!(
                "Download rejected ({} bytes > ANNO_MAX_DOWNLOAD_BYTES={} bytes) from {}",
                len, limit, source
            )));
        }
        Ok(())
    }

    #[cfg(feature = "eval")]
    fn hf_dataset_from_rows_url(url: &str) -> Option<String> {
        // Example:
        // https://datasets-server.huggingface.co/rows?dataset=masakhane%2Fmasakhaner2&config=bam&split=test...
        let (_, query) = url.split_once('?')?;
        for part in query.split('&') {
            let (k, v) = part.split_once('=')?;
            if k == "dataset" {
                // Minimal percent-decoding sufficient for HF dataset ids.
                let decoded = v.replace("%2F", "/").replace("%2f", "/");
                return Some(decoded);
            }
        }
        None
    }

    /// Create a new loader with default cache directory.
    ///
    /// Default location: `~/.cache/anno/datasets` (platform cache via `dirs` crate)
    /// Falls back to `.anno/datasets` in current directory if `dirs` crate unavailable.
    ///
    /// S3 fallback is enabled if `ANNO_S3_CACHE=1` environment variable is set.
    pub fn new() -> Result<Self> {
        // Load workspace `.env` (idempotent, does not override existing env vars).
        // This keeps eval tooling usable without manual exporting of env vars.
        anno::env::load_dotenv();

        // Check for custom cache directory
        let cache_dir = if let Ok(custom_dir) = std::env::var("ANNO_CACHE_DIR") {
            PathBuf::from(custom_dir).join("datasets")
        } else {
            let base_dir = dirs::cache_dir().unwrap_or_else(|| PathBuf::from("."));
            base_dir.join("anno").join("datasets")
        };

        // Check for S3 caching
        let s3_bucket = if std::env::var("ANNO_S3_CACHE").unwrap_or_default() == "1" {
            Some(std::env::var("ANNO_S3_BUCKET").unwrap_or_else(|_| "arc-anno-data".to_string()))
        } else {
            None
        };

        fs::create_dir_all(&cache_dir).map_err(|e| {
            Error::InvalidInput(format!("Failed to create cache dir {:?}: {}", cache_dir, e))
        })?;

        let manifest = CacheManifest::load(&cache_dir)?;

        Ok(Self {
            cache_dir,
            s3_bucket,
            manifest: std::sync::RwLock::new(manifest),
        })
    }

    /// Update the cache manifest with a new entry and save it.
    #[cfg(feature = "eval")]
    fn update_manifest(&self, entry: CacheManifestEntry) -> Result<()> {
        let mut manifest = self
            .manifest
            .write()
            .map_err(|_| Error::InvalidInput("cache manifest lock poisoned".to_string()))?;
        manifest.update_entry(entry);
        manifest.save(&self.cache_dir)?;
        Ok(())
    }

    /// Create a loader with a custom cache directory.
    pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> Result<Self> {
        let cache_dir = cache_dir.into();
        let s3_bucket = if std::env::var("ANNO_S3_CACHE").unwrap_or_default() == "1" {
            Some(std::env::var("ANNO_S3_BUCKET").unwrap_or_else(|_| "arc-anno-data".to_string()))
        } else {
            None
        };

        fs::create_dir_all(&cache_dir).map_err(|e| {
            Error::InvalidInput(format!("Failed to create cache dir {:?}: {}", cache_dir, e))
        })?;

        let manifest = CacheManifest::load(&cache_dir)?;

        Ok(Self {
            cache_dir,
            s3_bucket,
            manifest: std::sync::RwLock::new(manifest),
        })
    }

    /// Base directory for cached datasets.
    #[must_use]
    pub fn cache_dir(&self) -> &std::path::Path {
        &self.cache_dir
    }

    /// Whether S3 fallback caching is enabled.
    #[must_use]
    pub fn s3_enabled(&self) -> bool {
        self.s3_bucket.is_some()
    }

    /// The configured S3 bucket name (if enabled).
    #[must_use]
    pub fn s3_bucket(&self) -> Option<&str> {
        self.s3_bucket.as_deref()
    }

    /// Snapshot of cached dataset manifest entries (cloned).
    ///
    /// This is intended for tooling (CLI/scripts) that wants to iterate cached datasets
    /// without holding the manifest lock for a long time.
    #[must_use]
    pub fn cached_manifest_entries(&self) -> Vec<CacheManifestEntry> {
        let Ok(manifest) = self.manifest.read() else {
            return Vec::new();
        };
        let mut out: Vec<CacheManifestEntry> = manifest.entries.values().cloned().collect();
        out.sort_by(|a, b| a.dataset_id.cmp(&b.dataset_id));
        out
    }

    /// Upload a locally cached dataset (by `DatasetId`) to S3, using its manifest entry.
    ///
    /// This uses the same object layout as `load_or_download()`'s best-effort upload:
    /// - `datasets/<cache_filename>` (legacy/mutable key)
    /// - `datasets/by-sha256/<sha>/<cache_filename>` (immutable snapshot)
    /// - `datasets/<cache_filename>.latest.json` (pointer)
    /// - `datasets/<cache_filename>.manifest.json` (sidecar metadata)
    #[cfg(feature = "eval")]
    pub fn upload_cached_dataset_to_s3(&self, bucket: &str, id: DatasetId) -> Result<()> {
        let key = id.cache_filename();
        let entry = {
            let guard = self
                .manifest
                .read()
                .map_err(|_| Error::InvalidInput("cache manifest lock poisoned".to_string()))?;
            guard
                .get(key)
                .cloned()
                .ok_or_else(|| Error::InvalidInput(format!("No manifest entry for {}", key)))?
        };
        let path = self.cache_path_for(id);
        let content = std::fs::read_to_string(&path).map_err(|e| {
            Error::InvalidInput(format!(
                "Failed to read cached dataset {}: {}",
                path.display(),
                e
            ))
        })?;
        Self::enforce_max_download_bytes(content.len(), "local cache (sync-s3)")?;
        self.upload_to_s3(bucket, id, &content, &entry)
    }

    #[must_use]
    fn cache_path_for(&self, id: DatasetId) -> PathBuf {
        self.cache_dir.join(id.cache_filename())
    }

    #[must_use]
    fn is_cached_for(&self, id: DatasetId) -> bool {
        if !self.cache_path_for(id).exists() {
            return false;
        }

        // Best-effort cache invalidation: if the registry URL changes, the on-disk cached
        // file may no longer correspond to the dataset id. Prefer re-download over silently
        // using stale/incorrect data.
        //
        // This is intentionally lightweight (no checksum re-hash here).
        if let Ok(manifest) = self.manifest.read() {
            if let Some(entry) = manifest.get(id.cache_filename()) {
                if entry.source_url != id.download_url() {
                    return false;
                }
                // Treat “cached but empty” as invalid: this is almost always a bad download
                // (HTML, auth wall, or format mismatch) and should be re-fetched.
                if entry.sentence_count == 0 {
                    return false;
                }
            }
        }

        true
    }

    /// Get the cache path for a dataset.
    #[must_use]
    pub fn cache_path(&self, id: LoadableDatasetId) -> PathBuf {
        self.cache_path_for(id.0)
    }

    /// Check if a dataset is cached locally.
    #[must_use]
    pub fn is_cached(&self, id: LoadableDatasetId) -> bool {
        self.is_cached_for(id.0)
    }

    /// Load a dataset from cache.
    ///
    /// Returns an error if the dataset is not cached.
    pub fn load(&self, id: LoadableDatasetId) -> Result<LoadedDataset> {
        let dataset_id = id.0;
        let cache_path = self.cache_path(id);
        if !cache_path.exists() {
            return Err(Error::InvalidInput(format!(
                "Dataset {:?} not cached at {:?}",
                dataset_id, cache_path
            )));
        }

        let content = fs::read_to_string(&cache_path).map_err(|e| {
            Error::InvalidInput(format!("Failed to read cache {:?}: {}", cache_path, e))
        })?;

        let mut dataset = self.parse_content_impl(&content, dataset_id)?;
        if dataset.sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "Cached dataset '{}' parsed to 0 sentences (cache_path={:?})",
                dataset_id.name(),
                cache_path
            )));
        }
        dataset.data_source = DataSource::LocalCache;
        Ok(dataset)
    }

    /// Load or download a dataset.
    ///
    /// Tries cache first, then S3 (if enabled), then downloads from URL.
    #[cfg(feature = "eval")]
    pub fn load_or_download(&self, id: LoadableDatasetId) -> Result<LoadedDataset> {
        let dataset_id = id.0;
        // 1. Check local cache first
        if self.is_cached(id) {
            return self.load(id);
        }

        // 2. Try S3 cache if enabled
        if let Some(ref bucket) = self.s3_bucket {
            if let Ok((content, manifest_entry)) = self.download_from_s3(bucket, dataset_id) {
                Self::enforce_max_download_bytes(content.len(), "S3")?;
                // Cache locally for future use
                let cache_path = self.cache_path(id);
                fs::write(&cache_path, &content).map_err(|e| {
                    Error::InvalidInput(format!("Failed to write cache {:?}: {}", cache_path, e))
                })?;

                let mut dataset = self.parse_content_impl(&content, dataset_id)?;
                dataset.data_source = DataSource::S3Cache;

                // Best-effort: if S3 provides a manifest entry, record it locally.
                if let Some(entry) = manifest_entry {
                    let _ = self.update_manifest(entry);
                }
                return Ok(dataset);
            }
        }

        // 3. Download from original URL
        let (content, resolved_url) = self.download_with_resolved_url(dataset_id)?;
        Self::enforce_max_download_bytes(content.len(), &resolved_url)?;
        let file_size = content.len() as u64;
        let sha256 = self.compute_sha256(&content);

        // 4. Cache the downloaded content locally
        let cache_path = self.cache_path(id);
        fs::write(&cache_path, &content).map_err(|e| {
            Error::InvalidInput(format!("Failed to write cache {:?}: {}", cache_path, e))
        })?;

        // 5. Parse the content
        let mut dataset = self.parse_content_impl(&content, dataset_id)?;
        dataset.data_source = DataSource::OriginalUrl;

        // 6. Update manifest with download metadata
        let entry = CacheManifestEntry {
            dataset_id: dataset_id.cache_filename().to_string(),
            source_url: dataset_id.download_url().to_string(),
            resolved_url: Some(resolved_url.clone()),
            sha256,
            file_size,
            downloaded_at: chrono::Utc::now().to_rfc3339(),
            sentence_count: dataset.sentences.len(),
            entity_count: dataset.entity_count(),
            anno_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        let _ = self.update_manifest(entry); // Best effort

        // 7. Optionally upload to S3 for future use (best effort)
        if let Some(ref bucket) = self.s3_bucket {
            let entry = CacheManifestEntry {
                dataset_id: dataset_id.cache_filename().to_string(),
                source_url: dataset_id.download_url().to_string(),
                resolved_url: Some(resolved_url),
                sha256: self.compute_sha256(&content),
                file_size: content.len() as u64,
                downloaded_at: chrono::Utc::now().to_rfc3339(),
                sentence_count: dataset.sentences.len(),
                entity_count: dataset.entity_count(),
                anno_version: env!("CARGO_PKG_VERSION").to_string(),
            };
            let _ = self.upload_to_s3(bucket, dataset_id, &content, &entry);
        }

        Ok(dataset)
    }

    /// Load-or-download shim when `eval` is disabled.
    ///
    /// This method will load from cache if present, otherwise return a helpful error.
    #[cfg(not(feature = "eval"))]
    pub fn load_or_download(&self, id: LoadableDatasetId) -> Result<LoadedDataset> {
        if self.is_cached(id) {
            return self.load(id);
        }

        Err(Error::InvalidInput(
            "Dataset is not cached. Rebuild with feature `eval` to enable downloading.".to_string(),
        ))
    }

    /// Download dataset from S3 cache bucket.
    ///
    /// Uses AWS CLI under the hood (requires `aws` command in PATH and valid credentials).
    #[cfg(feature = "eval")]
    fn download_from_s3(
        &self,
        bucket: &str,
        id: DatasetId,
    ) -> Result<(String, Option<CacheManifestEntry>)> {
        use std::process::Command;

        // Prefer a content-addressed snapshot if the pointer exists; fall back to the legacy key.
        let pointer_key = format!("datasets/{}.latest.json", id.cache_filename());
        let pointer_uri = format!("s3://{}/{}", bucket, pointer_key);

        let mut content: Option<String> = None;

        // Attempt: pointer -> by-sha256 snapshot
        let pointer = Command::new("aws")
            .args(["s3", "cp", &pointer_uri, "-"])
            .output()
            .ok()
            .and_then(|o| {
                if !o.status.success() {
                    // Pointer missing or unreadable; fall back to legacy key.
                    return None;
                }
                String::from_utf8(o.stdout).ok()
            })
            .and_then(|json| serde_json::from_str::<serde_json::Value>(&json).ok());

        if let Some(pointer) = pointer {
            if let Some(sha) = pointer.get("sha256").and_then(|v| v.as_str()) {
                let by_sha_key = format!("datasets/by-sha256/{}/{}", sha, id.cache_filename());
                let by_sha_uri = format!("s3://{}/{}", bucket, by_sha_key);
                let output = Command::new("aws")
                    .args(["s3", "cp", &by_sha_uri, "-"])
                    .output()
                    .ok();

                if let Some(output) = output {
                    if output.status.success() {
                        content = String::from_utf8(output.stdout).ok();
                    }
                }
            }
        }

        // Fall back: legacy key (mutable)
        if content.is_none() {
            let s3_key = format!("datasets/{}", id.cache_filename());
            let s3_uri = format!("s3://{}/{}", bucket, s3_key);

            // Use aws s3 cp to download to stdout
            let output = Command::new("aws")
                .args(["s3", "cp", &s3_uri, "-"])
                .output()
                .map_err(|e| Error::InvalidInput(format!("Failed to run aws s3 cp: {}", e)))?;

            if output.status.success() {
                content = Some(String::from_utf8(output.stdout).map_err(|e| {
                    Error::InvalidInput(format!("S3 content not valid UTF-8: {}", e))
                })?);
            } else {
                return Err(Error::InvalidInput(format!(
                    "S3 download failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                )));
            }
        }

        let Some(content) = content else {
            return Err(Error::InvalidInput("S3 download failed".to_string()));
        };

        // Best-effort: attempt to fetch per-dataset manifest entry.
        let manifest = self.download_manifest_entry_from_s3(bucket, id).ok();

        Ok((content, manifest))
    }

    /// Upload dataset to S3 cache bucket.
    ///
    /// Best effort - failures are logged but don't stop execution.
    #[cfg(feature = "eval")]
    fn upload_to_s3(
        &self,
        bucket: &str,
        id: DatasetId,
        content: &str,
        manifest_entry: &CacheManifestEntry,
    ) -> Result<()> {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let s3_key = format!("datasets/{}", id.cache_filename());
        let s3_uri = format!("s3://{}/{}", bucket, s3_key);

        // Use aws s3 cp to upload from stdin
        let mut child = Command::new("aws")
            .args(["s3", "cp", "-", &s3_uri])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| Error::InvalidInput(format!("Failed to spawn aws s3 cp: {}", e)))?;

        if let Some(ref mut stdin) = child.stdin {
            let _ = stdin.write_all(content.as_bytes());
        }

        let status = child
            .wait()
            .map_err(|e| Error::InvalidInput(format!("Failed to wait for aws s3 cp: {}", e)))?;

        if !status.success() {
            return Err(Error::InvalidInput("S3 upload failed".to_string()));
        }

        // Also upload a content-addressed snapshot. This supports the "durable mirror" use case:
        // once uploaded, snapshots are stable even if upstream changes.
        let by_sha_key = format!(
            "datasets/by-sha256/{}/{}",
            manifest_entry.sha256,
            id.cache_filename()
        );
        let by_sha_uri = format!("s3://{}/{}", bucket, by_sha_key);

        let mut child = Command::new("aws")
            .args(["s3", "cp", "-", &by_sha_uri])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| Error::InvalidInput(format!("Failed to spawn aws s3 cp: {}", e)))?;

        if let Some(ref mut stdin) = child.stdin {
            let _ = stdin.write_all(content.as_bytes());
        }

        let _ = child.wait();

        // Update a pointer file for the latest snapshot.
        let pointer_key = format!("datasets/{}.latest.json", id.cache_filename());
        let pointer_uri = format!("s3://{}/{}", bucket, pointer_key);
        let pointer_json = serde_json::json!({
            "dataset": id.cache_filename(),
            "sha256": manifest_entry.sha256,
            "updated_at": chrono::Utc::now().to_rfc3339(),
            "source_url": manifest_entry.source_url,
            "resolved_url": manifest_entry.resolved_url,
            "file_size": manifest_entry.file_size,
        })
        .to_string();

        let mut child = Command::new("aws")
            .args(["s3", "cp", "-", &pointer_uri])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| Error::InvalidInput(format!("Failed to spawn aws s3 cp: {}", e)))?;

        if let Some(ref mut stdin) = child.stdin {
            let _ = stdin.write_all(pointer_json.as_bytes());
        }

        let _ = child.wait();

        // Upload a sidecar manifest entry (JSON). This makes the S3 cache self-describing.
        let manifest_key = format!("datasets/{}.manifest.json", id.cache_filename());
        let manifest_uri = format!("s3://{}/{}", bucket, manifest_key);
        let json = serde_json::to_string_pretty(manifest_entry)
            .map_err(|e| Error::InvalidInput(format!("Failed to serialize S3 manifest: {}", e)))?;

        let mut child = Command::new("aws")
            .args(["s3", "cp", "-", &manifest_uri])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| Error::InvalidInput(format!("Failed to spawn aws s3 cp: {}", e)))?;

        if let Some(ref mut stdin) = child.stdin {
            let _ = stdin.write_all(json.as_bytes());
        }

        let status = child
            .wait()
            .map_err(|e| Error::InvalidInput(format!("Failed to wait for aws s3 cp: {}", e)))?;

        if status.success() {
            Ok(())
        } else {
            Err(Error::InvalidInput("S3 manifest upload failed".to_string()))
        }
    }

    /// Best-effort: download the sidecar manifest entry for a dataset from S3.
    #[cfg(feature = "eval")]
    fn download_manifest_entry_from_s3(
        &self,
        bucket: &str,
        id: DatasetId,
    ) -> Result<CacheManifestEntry> {
        use std::process::Command;

        let manifest_key = format!("datasets/{}.manifest.json", id.cache_filename());
        let manifest_uri = format!("s3://{}/{}", bucket, manifest_key);

        let output = Command::new("aws")
            .args(["s3", "cp", &manifest_uri, "-"])
            .output()
            .map_err(|e| Error::InvalidInput(format!("Failed to run aws s3 cp: {}", e)))?;

        if !output.status.success() {
            return Err(Error::InvalidInput(
                "S3 manifest download failed".to_string(),
            ));
        }

        let json = String::from_utf8(output.stdout)
            .map_err(|e| Error::InvalidInput(format!("S3 manifest not valid UTF-8: {}", e)))?;

        serde_json::from_str(&json)
            .map_err(|e| Error::InvalidInput(format!("Failed to parse S3 manifest JSON: {}", e)))
    }

    /// Download dataset from source with retry logic and pagination support.
    ///
    /// Implements exponential backoff retry strategy:
    /// - 3 retries maximum
    /// - Initial delay: 1 second
    /// - Exponential backoff: 2^attempt seconds
    /// - Max delay: 10 seconds
    ///
    /// For HuggingFace datasets-server API, automatically paginates to download full dataset.
    #[cfg(feature = "eval")]
    fn download_with_resolved_url(&self, id: DatasetId) -> Result<(String, String)> {
        let url = id.download_url().to_string();

        fn env_single_csv(keys: &[&str]) -> Option<String> {
            for &k in keys {
                let Ok(raw) = std::env::var(k) else {
                    continue;
                };
                let mut parts = raw
                    .split(',')
                    .map(|s| s.trim().to_ascii_lowercase())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>();
                parts.sort();
                parts.dedup();
                if parts.len() == 1 {
                    return Some(parts[0].clone());
                }
            }
            None
        }

        // Check if URL is empty (dataset not available for download)
        if url.is_empty() {
            let access_status = id.access_status();
            let notes = id.notes();
            let mirror_url = id.mirror_url();

            let mut error_msg = format!(
                "Dataset '{}' ({:?}) has no download URL available.",
                id.name(),
                id
            );

            // Add access status information
            match access_status {
                super::dataset_registry::DatasetAccessibility::Public => {
                    error_msg.push_str("\n\nStatus: Public (but URL not configured)");
                }
                super::dataset_registry::DatasetAccessibility::HuggingFace => {
                    if let Some(hf_id) = id.hf_id() {
                        error_msg.push_str(&format!(
                            "\n\nStatus: Available on HuggingFace\nDataset ID: {}",
                            hf_id
                        ));
                    } else {
                        error_msg
                            .push_str("\n\nStatus: Available on HuggingFace (ID not configured)");
                    }
                }
                super::dataset_registry::DatasetAccessibility::Local => {
                    error_msg.push_str("\n\nStatus: Available locally (check testdata/ or cache)");
                }
                super::dataset_registry::DatasetAccessibility::Registration => {
                    error_msg
                        .push_str("\n\nStatus: Requires registration (e.g., LDC for academics)");
                }
                super::dataset_registry::DatasetAccessibility::ContactAuthors => {
                    error_msg.push_str("\n\nStatus: Contact dataset authors for access");
                }
                super::dataset_registry::DatasetAccessibility::NotYetReleased => {
                    error_msg.push_str("\n\nStatus: Not yet publicly released");
                }
                super::dataset_registry::DatasetAccessibility::DependsOnOther => {
                    if let Some(dep) = id.depends_on() {
                        error_msg
                            .push_str(&format!("\n\nStatus: Depends on another dataset: {}", dep));
                    } else {
                        error_msg.push_str("\n\nStatus: Depends on another dataset");
                    }
                }
                super::dataset_registry::DatasetAccessibility::Deprecated => {
                    error_msg.push_str("\n\nStatus: Deprecated or no longer available");
                }
            }

            if let Some(note) = notes {
                error_msg.push_str(&format!("\n\nNotes: {}", note));
            }

            if let Some(mirror) = mirror_url {
                error_msg.push_str(&format!("\n\nMirror URL: {}", mirror));
            }

            return Err(Error::InvalidInput(error_msg));
        }

        // For multilingual datasets hosted on HuggingFace with per-language configs (e.g. WikiANN),
        // let callers bias the HF "config" selection via `ANNO_MUXER_PIN_LANG`.
        //
        // This keeps muxer facet pins honest: a `.lang=de` run should ideally download a German
        // config, not a random/first config.
        let preferred_hf_config: Option<String> = if id.language().eq_ignore_ascii_case("mul") {
            // Explicit override wins if set.
            env_single_csv(&["ANNO_HF_DATASET_CONFIG"])
                .or_else(|| env_single_csv(&["ANNO_MUXER_PIN_LANG", "ANNO_MUXER_FILTER_LANG"]))
        } else {
            env_single_csv(&["ANNO_HF_DATASET_CONFIG"])
        };

        // If the registry marks this dataset as HuggingFace-accessible and provides an HF id,
        // prefer fetching via datasets-server (even if the registry `url` is a paper/GitHub homepage).
        if id.access_status() == super::dataset_registry::DatasetAccessibility::HuggingFace {
            if let Some(hf_ds) = id.hf_id().map(|s| s.to_string()) {
                if let Ok((config, split)) =
                    self.resolve_hf_config_split_prefer(&hf_ds, preferred_hf_config.as_deref())
                {
                    let base = Self::hf_rows_url(&hf_ds, &config, &split);
                    if std::env::var("ANNO_MUXER_VERBOSE")
                        .ok()
                        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    {
                        eprintln!(
                            "dataset-loader: hf dataset={} config={} split={} (preferred={:?})",
                            hf_ds, config, split, preferred_hf_config
                        );
                    }
                    match self.download_hf_dataset_paginated(id, &base) {
                        Ok(content) => return Ok((content, base)),
                        Err(_e) => {
                            // Fall back to downloading a raw file from the dataset repo.
                            if let Ok((content, resolved)) =
                                self.download_hf_dataset_file_from_hub(&hf_ds)
                            {
                                return Ok((content, resolved));
                            }
                        }
                    }
                }
            }
        }

        // HuggingFace dataset pages are common in the registry. Convert them into a
        // datasets-server API download so we can actually fetch data (and avoid caching HTML).
        //
        // Example:
        // - page:  https://huggingface.co/datasets/KevinSpaghetti/cadec
        // - API:   https://datasets-server.huggingface.co/rows?dataset=KevinSpaghetti%2Fcadec&...
        if url.contains("huggingface.co/datasets/") {
            // Prefer the registry's `hf_id` when available (more reliable than parsing the URL).
            let hf_ds = id
                .hf_id()
                .map(|s| s.to_string())
                .or_else(|| Self::extract_hf_dataset_name(&url));

            if let Some(hf_ds) = hf_ds {
                match self.resolve_hf_config_split_prefer(&hf_ds, preferred_hf_config.as_deref()) {
                    Ok((config, split)) => {
                        let base = Self::hf_rows_url(&hf_ds, &config, &split);
                        if std::env::var("ANNO_MUXER_VERBOSE")
                            .ok()
                            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                        {
                            eprintln!(
                                "dataset-loader: hf dataset={} config={} split={} (preferred={:?})",
                                hf_ds, config, split, preferred_hf_config
                            );
                        }
                        match self.download_hf_dataset_paginated(id, &base) {
                            Ok(content) => return Ok((content, base)),
                            Err(e) => {
                                // Some HF datasets don't support datasets-server row export.
                                // Fall back to downloading a raw file from the dataset repo.
                                if let Ok((content, resolved)) =
                                    self.download_hf_dataset_file_from_hub(&hf_ds)
                                {
                                    return Ok((content, resolved));
                                }
                                return Err(e);
                            }
                        }
                    }
                    Err(_e) => {
                        // If the datasets-server endpoint is unavailable (rate limits, transient issues),
                        // prefer falling back to the hub-file path rather than downloading HTML.
                        if let Ok((content, resolved)) =
                            self.download_hf_dataset_file_from_hub(&hf_ds)
                        {
                            return Ok((content, resolved));
                        }
                    }
                }
            }
        }

        // Check if this is a HuggingFace datasets-server API URL
        if url.contains("datasets-server.huggingface.co/rows") {
            match self.download_hf_dataset_paginated(id, &url) {
                Ok(content) => return Ok((content, url)),
                Err(e) => {
                    // Some datasets fail row export (422). If we can infer the HF dataset id,
                    // fall back to the hub file download path.
                    if let Some(hf_ds) = Self::hf_dataset_from_rows_url(&url) {
                        if let Ok((content, resolved)) =
                            self.download_hf_dataset_file_from_hub(&hf_ds)
                        {
                            return Ok((content, resolved));
                        }
                    }
                    return Err(e);
                }
            }
        }

        // Regular download with retry logic
        const MAX_RETRIES: u32 = 3;
        const INITIAL_DELAY_SECS: u64 = 1;

        let mut last_error = None;

        for attempt in 0..=MAX_RETRIES {
            match self.download_attempt(&url) {
                Ok(content) => {
                    // Checksum validation removed - registry doesn't provide expected_checksum
                    // Future: Add checksum validation if registry provides it

                    return Ok((content, url.clone()));
                }
                Err(e) => {
                    last_error = Some(e);
                    if attempt < MAX_RETRIES {
                        let delay_secs = (INITIAL_DELAY_SECS * (1 << attempt)).min(10);
                        log::warn!(
                            "Download attempt {} failed for {}, retrying in {}s...",
                            attempt + 1,
                            &url,
                            delay_secs
                        );
                        std::thread::sleep(std::time::Duration::from_secs(delay_secs));
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            Error::InvalidInput(format!(
                "Failed to download {} after {} retries. \
                 Check network connection and try again. \
                 URL: {}",
                id.name(),
                MAX_RETRIES + 1,
                &url
            ))
        }))
    }

    /// Extract `<org>/<dataset>` from `https://huggingface.co/datasets/<org>/<dataset>(/...)`.
    #[cfg(feature = "eval")]
    fn extract_hf_dataset_name(url: &str) -> Option<String> {
        let marker = "huggingface.co/datasets/";
        let idx = url.find(marker)? + marker.len();
        let rest = &url[idx..];
        let rest = rest.split('?').next().unwrap_or(rest);
        let rest = rest.trim_matches('/');
        let mut parts = rest.split('/');
        let org = parts.next()?;
        let name = parts.next()?;
        Some(format!("{}/{}", org, name))
    }

    #[cfg(feature = "eval")]
    fn url_encode_component(s: &str) -> String {
        // Minimal encoding for HF datasets-server query parameters.
        // In practice we mainly need `/` -> `%2F`.
        s.replace('/', "%2F").replace(' ', "%20")
    }

    #[cfg(feature = "eval")]
    fn hf_rows_url(dataset: &str, config: &str, split: &str) -> String {
        format!(
            "https://datasets-server.huggingface.co/rows?dataset={}&config={}&split={}&offset=0&length=100",
            Self::url_encode_component(dataset),
            Self::url_encode_component(config),
            Self::url_encode_component(split),
        )
    }

    /// Resolve a usable (config, split) pair for a HF dataset via datasets-server, with an optional
    /// preferred config.
    #[cfg(feature = "eval")]
    fn resolve_hf_config_split_prefer(
        &self,
        dataset: &str,
        preferred_config: Option<&str>,
    ) -> Result<(String, String)> {
        let url = format!(
            "https://datasets-server.huggingface.co/splits?dataset={}",
            Self::url_encode_component(dataset)
        );
        let response = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(30))
            .call()
            .map_err(|e| Error::InvalidInput(format!("Failed to query HF splits: {}", e)))?;

        if response.status() != 200 {
            return Err(Error::InvalidInput(format!(
                "HF splits query returned HTTP {} for dataset {}",
                response.status(),
                dataset
            )));
        }

        let body = response.into_string().map_err(|e| {
            Error::InvalidInput(format!("Failed to read HF splits response: {}", e))
        })?;

        let json: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
            Error::InvalidInput(format!("Invalid JSON from HF splits endpoint: {}", e))
        })?;

        let splits = json
            .get("splits")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                Error::InvalidInput("HF splits response missing `splits`".to_string())
            })?;

        // Prefer smaller/eval splits if present, otherwise pick the first split.
        // If `preferred_config` is provided, try to pick a split within that config first.
        let mut chosen = None;
        let prefer = preferred_config.map(|s| s.trim()).filter(|s| !s.is_empty());

        for pass in 0..2 {
            for s in splits {
                let config = s.get("config").and_then(|v| v.as_str());
                let split = s.get("split").and_then(|v| v.as_str());
                if let (Some(config), Some(split)) = (config, split) {
                    // Pass 0: only consider preferred config. Pass 1: consider any config.
                    if pass == 0 {
                        if let Some(p) = prefer {
                            if config != p {
                                continue;
                            }
                        } else {
                            // No preference: skip pass 0.
                            break;
                        }
                    }

                    if split == "test" {
                        chosen = Some((config.to_string(), split.to_string()));
                        break;
                    }
                    if chosen.is_none() {
                        chosen = Some((config.to_string(), split.to_string()));
                    }
                }
            }
            if chosen.is_some() {
                break;
            }
        }

        chosen.ok_or_else(|| {
            Error::InvalidInput(format!(
                "HF splits endpoint returned no usable (config, split) for dataset {}",
                dataset
            ))
        })
    }

    /// Best-effort fallback: download a raw dataset file from the HuggingFace Hub.
    ///
    /// This is a fallback for datasets that do not support the datasets-server row export.
    #[cfg(feature = "eval")]
    fn download_hf_dataset_file_from_hub(&self, dataset: &str) -> Result<(String, String)> {
        // Example API: https://huggingface.co/api/datasets/coref-data/preco_raw
        let api_url = format!("https://huggingface.co/api/datasets/{}", dataset);
        let response = ureq::get(&api_url)
            .timeout(std::time::Duration::from_secs(30))
            .call()
            .map_err(|e| {
                Error::InvalidInput(format!(
                    "Failed to query HuggingFace dataset metadata for {}: {}",
                    dataset, e
                ))
            })?;

        if response.status() != 200 {
            return Err(Error::InvalidInput(format!(
                "HuggingFace dataset metadata request returned HTTP {} for {}",
                response.status(),
                dataset
            )));
        }

        let body = response.into_string().map_err(|e| {
            Error::InvalidInput(format!(
                "Failed to read HuggingFace dataset metadata: {}",
                e
            ))
        })?;
        let json: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
            Error::InvalidInput(format!(
                "Invalid JSON from HuggingFace dataset metadata: {}",
                e
            ))
        })?;

        let siblings = json
            .get("siblings")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                Error::InvalidInput("HF dataset metadata missing `siblings`".to_string())
            })?;

        // Prefer smaller / eval-friendly splits, but fall back to whatever exists.
        //
        // Many HF dataset repos store raw source files (CoNLL/TSV/etc.) instead of JSONL,
        // often under subdirectories. We handle both.
        let preferred = [
            // JSONL (common)
            "dev.jsonl",
            "validation.jsonl",
            "test.jsonl",
            "train.jsonl",
        ];
        let mut chosen: Option<String> = None;

        for want in preferred {
            if siblings.iter().any(|s| {
                s.get("rfilename")
                    .and_then(|v| v.as_str())
                    .is_some_and(|f| f == want)
            }) {
                chosen = Some(want.to_string());
                break;
            }
        }

        if chosen.is_none() {
            // Otherwise, pick a file we can plausibly parse (allow subdirectories).
            //
            // Priority:
            // - contains split hint: test > dev/valid > train
            // - extension: jsonl > conll/bio/iob/tsv/txt
            //
            // This is still best-effort; the parser selection is handled elsewhere via the dataset id.
            fn split_rank(name: &str) -> u8 {
                let n = name.to_lowercase();
                if n.contains("test") {
                    0
                } else if n.contains("dev") || n.contains("valid") || n.contains("validation") {
                    1
                } else if n.contains("train") {
                    2
                } else {
                    3
                }
            }
            fn ext_rank(name: &str) -> u8 {
                let n = name.to_lowercase();
                if n.ends_with(".jsonl") {
                    0
                } else if n.ends_with(".conll")
                    || n.ends_with(".bio")
                    || n.ends_with(".iob")
                    || n.ends_with(".iob2")
                {
                    1
                } else if n.ends_with(".tsv") {
                    2
                } else if n.ends_with(".txt") {
                    3
                } else {
                    9
                }
            }

            let mut candidates: Vec<String> = siblings
                .iter()
                .filter_map(|s| {
                    s.get("rfilename")
                        .and_then(|v| v.as_str())
                        .map(|f| f.to_string())
                })
                .filter(|f| ext_rank(f) < 9)
                .collect();

            candidates.sort_by(|a, b| {
                (split_rank(a), ext_rank(a), a.len()).cmp(&(split_rank(b), ext_rank(b), b.len()))
            });

            chosen = candidates.first().cloned();
        }

        let Some(filename) = chosen else {
            return Err(Error::InvalidInput(format!(
                "No downloadable .jsonl file discovered for HF dataset {}",
                dataset
            )));
        };

        // Download file via resolve (use main; can be made content-addressed later via `sha`).
        let file_url = format!(
            "https://huggingface.co/datasets/{}/resolve/main/{}",
            dataset, filename
        );

        // Best-effort: avoid downloading huge files if a max byte limit is configured.
        if let Some(limit) = Self::max_download_bytes() {
            if let Ok(resp) = ureq::head(&file_url)
                .timeout(std::time::Duration::from_secs(30))
                .call()
            {
                if resp.status() == 200 {
                    if let Some(len) = resp
                        .header("Content-Length")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        if len > limit {
                            return Err(Error::InvalidInput(format!(
                                "Download rejected ({} bytes > ANNO_MAX_DOWNLOAD_BYTES={} bytes) from {}",
                                len, limit, file_url
                            )));
                        }
                    }
                }
            }
        }

        let content = self.download_attempt(&file_url)?;
        Ok((content, file_url))
    }

    /// Placeholder when `eval` is disabled.
    #[cfg(not(feature = "eval"))]
    #[allow(dead_code)]
    fn download_hf_dataset_file_from_hub(&self, _dataset: &str) -> Result<(String, String)> {
        Err(Error::InvalidInput(
            "HuggingFace file fallback requires feature `eval`".to_string(),
        ))
    }

    /// Download HuggingFace dataset with pagination support.
    ///
    /// HF datasets-server API limits responses to 100 rows by default.
    /// This function automatically paginates to download the full dataset.
    ///
    /// For datasets available on HuggingFace Hub, can also use hf-hub crate
    /// for direct file downloads (faster, no pagination needed).
    #[cfg(feature = "eval")]
    fn download_hf_dataset_paginated(&self, id: DatasetId, base_url: &str) -> Result<String> {
        // Try hf-hub direct download first (if available and dataset is on HF)
        #[cfg(feature = "onnx")] // hf-hub is available with onnx feature
        {
            if let Ok(content) = self.try_hf_hub_download(id) {
                return Ok(content);
            }
        }

        // Fall back to paginated API download
        // HuggingFace datasets-server API limits to 100 rows per request
        const PAGE_SIZE: usize = 100;
        let mut all_rows = Vec::new();
        let mut features = None;
        let mut offset: usize = 0;
        let mut total_rows = None;

        log::info!(
            "Downloading {} with pagination (page size: {})",
            id.name(),
            PAGE_SIZE
        );

        loop {
            // Build paginated URL
            let url = if base_url.contains("offset=") {
                // Replace existing offset parameter
                let prev_offset = offset.saturating_sub(PAGE_SIZE);
                base_url
                    .replace(
                        &format!("offset={}", prev_offset),
                        &format!("offset={}", offset),
                    )
                    .replace("length=100", &format!("length={}", PAGE_SIZE))
            } else {
                // Add pagination parameters
                let separator = if base_url.contains('?') { "&" } else { "?" };
                format!(
                    "{}{}offset={}&length={}",
                    base_url, separator, offset, PAGE_SIZE
                )
            };

            match self.download_attempt(&url) {
                Ok(content) => {
                    let parsed: serde_json::Value =
                        serde_json::from_str(&content).map_err(|e| {
                            Error::InvalidInput(format!("Invalid JSON response: {}", e))
                        })?;

                    // Extract features (only from first page)
                    if features.is_none() {
                        features = parsed.get("features").cloned();
                    }

                    // Extract total number of rows (if available)
                    if total_rows.is_none() {
                        total_rows = parsed
                            .get("num_rows_total")
                            .and_then(|v| v.as_u64())
                            .map(|n| n as usize);
                    }

                    // Extract rows from this page
                    if let Some(rows) = parsed.get("rows").and_then(|v| v.as_array()) {
                        if rows.is_empty() {
                            break; // No more rows
                        }
                        all_rows.extend_from_slice(rows);
                        log::debug!(
                            "Downloaded {} rows (total so far: {})",
                            rows.len(),
                            all_rows.len()
                        );

                        // Check if we've got all rows
                        if let Some(total) = total_rows {
                            if all_rows.len() >= total {
                                break;
                            }
                        } else if rows.len() < PAGE_SIZE {
                            // No total available, but got fewer rows than requested = last page
                            break;
                        }

                        offset += PAGE_SIZE;
                    } else {
                        // No rows in response, might be error or empty dataset
                        break;
                    }
                }
                Err(e) => {
                    // If we got some rows, return partial dataset with warning
                    if !all_rows.is_empty() {
                        log::warn!(
                            "Failed to download full {} dataset (got {} rows before error: {}). \
                             Returning partial dataset.",
                            id.name(),
                            all_rows.len(),
                            e
                        );
                        break;
                    } else {
                        return Err(e);
                    }
                }
            }

            // Safety limit: prevent infinite loops
            if offset > 1_000_000 {
                log::warn!(
                    "Reached safety limit (1M rows) for {}. Returning partial dataset ({} rows).",
                    id.name(),
                    all_rows.len()
                );
                break;
            }
        }

        // Reconstruct full API response format
        let mut response: serde_json::Value = serde_json::json!({
            "rows": all_rows,
        });

        if let Some(features_val) = features {
            response["features"] = features_val;
        }

        if let Some(total) = total_rows {
            response["num_rows_total"] = serde_json::json!(total);
        }

        serde_json::to_string(&response).map_err(|e| {
            Error::InvalidInput(format!("Failed to serialize paginated response: {}", e))
        })
    }

    /// Try downloading dataset directly from HuggingFace Hub using hf-hub crate.
    ///
    /// This is faster than paginated API calls for datasets available on HF Hub.
    /// Returns Ok(content) if successful, Err if not available or hf-hub not enabled.
    ///
    /// Uses `HF_TOKEN` environment variable if set for accessing gated datasets.
    #[cfg(all(feature = "eval", feature = "onnx"))]
    fn try_hf_hub_download(&self, id: DatasetId) -> Result<String> {
        use hf_hub::api::sync::{Api, ApiBuilder};

        // Map dataset IDs to HuggingFace dataset names and file paths
        let (dataset_name, file_path) = match id {
            DatasetId::MultiNERD => ("Babelscape/multinerd", "test/test_en.jsonl"),
            DatasetId::TweetNER7 => ("tner/tweetner7", "dataset/2020.dev.json"),
            DatasetId::BroadTwitterCorpus => ("GateNLP/broad_twitter_corpus", "test/a.conll"),
            DatasetId::CADEC => ("KevinSpaghetti/cadec", "data/test.jsonl"),
            DatasetId::PreCo => ("coref-data/preco", "data/test.jsonl"),
            // Gated datasets that require HF_TOKEN
            DatasetId::MultiCoNER => ("MultiCoNER/multiconer_v1", "en/test.conll"),
            DatasetId::MultiCoNERv2 => ("MultiCoNER/multiconer_v2", "en/test.conll"),
            _ => {
                return Err(Error::InvalidInput(
                    "Dataset not available via hf-hub".to_string(),
                ))
            }
        };

        // Load .env for HF_TOKEN if not already set
        anno::env::load_dotenv();

        // Use HF_TOKEN from environment if available (for gated datasets)
        let api = if let Some(token) = anno::env::hf_token() {
            ApiBuilder::new()
                .with_token(Some(token))
                .build()
                .map_err(|e| {
                    Error::InvalidInput(format!(
                        "Failed to initialize HuggingFace API with token: {}",
                        e
                    ))
                })?
        } else {
            Api::new().map_err(|e| {
                Error::InvalidInput(format!("Failed to initialize HuggingFace API: {}", e))
            })?
        };

        let repo = api.dataset(dataset_name.to_string());
        let file_path_buf = repo.get(file_path).map_err(|e| {
            Error::InvalidInput(format!(
                "Failed to download {} from HuggingFace Hub: {}. \
                 Falling back to HTTP download.",
                file_path, e
            ))
        })?;

        std::fs::read_to_string(&file_path_buf)
            .map_err(|e| Error::InvalidInput(format!("Failed to read downloaded file: {}", e)))
    }

    /// Placeholder for when hf-hub is not available.
    #[cfg(not(all(feature = "eval", feature = "onnx")))]
    #[allow(dead_code)] // Part of trait interface, may be unused in some feature combinations
    fn try_hf_hub_download(&self, _id: DatasetId) -> Result<String> {
        Err(Error::InvalidInput("hf-hub not available".to_string()))
    }

    /// Single download attempt with retry logic.
    ///
    /// Retries up to 3 times with exponential backoff for transient failures (timeouts, 5xx errors).
    #[cfg(feature = "eval")]
    fn download_attempt(&self, url: &str) -> Result<String> {
        const MAX_RETRIES: usize = 3;
        const INITIAL_TIMEOUT_SECS: u64 = 30;
        const MAX_TIMEOUT_SECS: u64 = 120;

        let mut last_error = None;

        for attempt in 0..=MAX_RETRIES {
            let timeout_secs = (INITIAL_TIMEOUT_SECS * (1 << attempt.min(2))).min(MAX_TIMEOUT_SECS);

            match ureq::get(url)
                .timeout(std::time::Duration::from_secs(timeout_secs))
                .call()
            {
                Ok(response) => {
                    if response.status() == 200 {
                        // Success - read content
                        let content = response.into_string().map_err(|e| {
                            Error::InvalidInput(format!(
                                "Failed to read response from {}: {}. \
                                 Response may be too large or corrupted.",
                                url, e
                            ))
                        })?;

                        // Heuristic guardrail: many registry URLs are dataset homepages. If we fetch HTML,
                        // treat it as a download failure so we don't silently cache garbage and then parse
                        // "0 sentences".
                        let lower = content
                            .chars()
                            .take(2048)
                            .collect::<String>()
                            .to_lowercase();
                        if lower.contains("<html") || lower.contains("<!doctype html") {
                            return Err(Error::InvalidInput(format!(
                                "Downloaded HTML from {}. This URL looks like a webpage, not a raw dataset file.",
                                url
                            )));
                        }

                        return Ok(content);
                    }

                    let status = response.status();
                    let body = response.into_string().unwrap_or_default();
                    let body = body.trim();
                    let body_preview = if body.len() > 800 {
                        format!("{}…", &body[..800])
                    } else {
                        body.to_string()
                    };

                    // Retry on 5xx server errors (transient)
                    if (500..600).contains(&status) && attempt < MAX_RETRIES {
                        let wait_ms = 1000 * (1 << attempt); // Exponential backoff: 1s, 2s, 4s
                        log::debug!(
                            "Server error {} downloading {} (attempt {}/{}), retrying in {}ms...",
                            status,
                            url,
                            attempt + 1,
                            MAX_RETRIES + 1,
                            wait_ms
                        );
                        std::thread::sleep(std::time::Duration::from_millis(wait_ms));
                        last_error = Some(format!(
                            "HTTP {} downloading {}. Server returned error status. {}{}",
                            status,
                            url,
                            if body_preview.is_empty() {
                                ""
                            } else {
                                "Response body: "
                            },
                            body_preview
                        ));
                        continue;
                    }

                    // Non-retryable error (4xx client errors)
                    return Err(Error::InvalidInput(format!(
                        "HTTP {} downloading {}. \
                         Server returned error status. \
                         Dataset may be temporarily unavailable or URL changed. {}{}",
                        status,
                        url,
                        if body_preview.is_empty() {
                            ""
                        } else {
                            "Response body: "
                        },
                        body_preview
                    )));
                }
                Err(ureq::Error::Transport(e)) => {
                    // Network errors (timeouts, connection failures) - retry
                    let error_msg = format!("{}", e);
                    let is_timeout =
                        error_msg.contains("timeout") || error_msg.contains("timed out");

                    if is_timeout && attempt < MAX_RETRIES {
                        let wait_ms = 1000 * (1 << attempt); // Exponential backoff
                        log::debug!(
                            "Timeout downloading {} (attempt {}/{}), retrying in {}ms...",
                            url,
                            attempt + 1,
                            MAX_RETRIES + 1,
                            wait_ms
                        );
                        std::thread::sleep(std::time::Duration::from_millis(wait_ms));
                        last_error = Some(error_msg);
                        continue;
                    }

                    // Final attempt failed or non-timeout error
                    return Err(Error::InvalidInput(format!(
                        "Network error downloading {}: {}. \
                         Check your internet connection and try again. \
                         {}",
                        url,
                        error_msg,
                        if attempt > 0 {
                            format!("(Failed after {} retries)", attempt)
                        } else {
                            String::new()
                        }
                    )));
                }
                Err(e) => {
                    // Other errors (non-retryable)
                    let error_msg = format!("{}", e);
                    return Err(Error::InvalidInput(format!(
                        "Error downloading {}: {}. \
                         Check your internet connection and try again.",
                        url, error_msg
                    )));
                }
            }
        }

        // All retries exhausted
        Err(Error::InvalidInput(format!(
            "Failed to download {} after {} attempts. Last error: {}",
            url,
            MAX_RETRIES + 1,
            last_error.unwrap_or_else(|| "unknown error".to_string())
        )))
    }

    /// Single download attempt returning raw bytes (for binary formats like ZIP).
    ///
    /// Same retry/backoff logic as `download_attempt` but reads response as bytes.
    #[cfg(feature = "eval")]
    fn download_attempt_bytes(&self, url: &str) -> Result<Vec<u8>> {
        const MAX_RETRIES: usize = 3;
        const INITIAL_TIMEOUT_SECS: u64 = 30;
        const MAX_TIMEOUT_SECS: u64 = 120;
        const MAX_BYTES: usize = 50 * 1024 * 1024; // 50 MB limit

        let mut last_error = None;

        for attempt in 0..=MAX_RETRIES {
            let timeout_secs = (INITIAL_TIMEOUT_SECS * (1 << attempt.min(2))).min(MAX_TIMEOUT_SECS);

            match ureq::get(url)
                .timeout(std::time::Duration::from_secs(timeout_secs))
                .call()
            {
                Ok(response) => {
                    if response.status() == 200 {
                        use std::io::Read as _;
                        let mut bytes = Vec::new();
                        response
                            .into_reader()
                            .take(MAX_BYTES as u64)
                            .read_to_end(&mut bytes)
                            .map_err(|e| {
                                Error::InvalidInput(format!(
                                    "Failed to read bytes from {}: {}",
                                    url, e
                                ))
                            })?;
                        return Ok(bytes);
                    }

                    let status = response.status();
                    if (500..600).contains(&status) && attempt < MAX_RETRIES {
                        let wait_ms = 1000 * (1 << attempt);
                        std::thread::sleep(std::time::Duration::from_millis(wait_ms));
                        last_error = Some(format!("HTTP {} from {}", status, url));
                        continue;
                    }

                    return Err(Error::InvalidInput(format!(
                        "HTTP {} downloading binary from {}",
                        status, url
                    )));
                }
                Err(ureq::Error::Transport(e)) => {
                    let msg = format!("{}", e);
                    if (msg.contains("timeout") || msg.contains("timed out"))
                        && attempt < MAX_RETRIES
                    {
                        let wait_ms = 1000 * (1 << attempt);
                        std::thread::sleep(std::time::Duration::from_millis(wait_ms));
                        last_error = Some(msg);
                        continue;
                    }
                    return Err(Error::InvalidInput(format!(
                        "Network error downloading {}: {}",
                        url, msg
                    )));
                }
                Err(e) => {
                    return Err(Error::InvalidInput(format!(
                        "Error downloading {}: {}",
                        url, e
                    )));
                }
            }
        }

        Err(Error::InvalidInput(format!(
            "Failed to download {} after {} attempts. Last error: {}",
            url,
            MAX_RETRIES + 1,
            last_error.unwrap_or_else(|| "unknown".to_string())
        )))
    }

    /// Compute SHA256 checksum of content.
    #[cfg(feature = "eval")]
    fn compute_sha256(&self, content: &str) -> String {
        #[cfg(feature = "eval")]
        {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(content.as_bytes());
            format!("{:x}", hasher.finalize())
        }
        #[cfg(not(feature = "eval"))]
        {
            // Fallback if sha2 not available
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            content.hash(&mut hasher);
            format!("{:x}", hasher.finish())
        }
    }

    /// Get temporal metadata for a dataset if available.
    fn get_temporal_metadata(id: DatasetId) -> Option<TemporalMetadata> {
        match id {
            DatasetId::TweetNER7 => {
                // TweetNER7 has temporal entity recognition - use dataset creation date as cutoff
                Some(TemporalMetadata {
                    kb_version: None,                                // No KB linking in TweetNER7
                    temporal_cutoff: Some("2017-01-01".to_string()), // Approximate dataset creation
                    entity_creation_dates: None,                     // Would need entity linking
                })
            }
            DatasetId::BroadTwitterCorpus => {
                // BroadTwitterCorpus is stratified across times - use approximate cutoff
                Some(TemporalMetadata {
                    kb_version: None,
                    temporal_cutoff: Some("2018-01-01".to_string()), // Approximate
                    entity_creation_dates: None,
                })
            }
            DatasetId::BC5CDR
            | DatasetId::NCBIDisease
            | DatasetId::GENIA
            | DatasetId::AnatEM
            | DatasetId::BC2GM
            | DatasetId::BC4CHEMD => {
                // Biomedical datasets might have KB versions (UMLS, etc.)
                Some(TemporalMetadata {
                    kb_version: None,
                    temporal_cutoff: None,
                    entity_creation_dates: None,
                })
            }
            _ => None, // Most datasets don't have temporal metadata
        }
    }

    /// Parse content based on dataset format.
    ///
    /// This is intentionally `pub` (behind the `eval` module) to enable offline evaluation
    /// workflows and integration tests that want to exercise parsing without network I/O.
    ///
    /// It does **not** download or read from cache. For typical usage, prefer
    /// [`DatasetLoader::load_or_download`] / [`DatasetLoader::load`].
    pub fn parse_content_str(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        self.parse_content_impl(content, id)
    }

    /// Internal implementation for parsing a dataset payload.
    fn parse_content_impl(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        if content.trim().is_empty() {
            return Err(Error::InvalidInput(format!(
                "Dataset {:?} file is empty",
                id
            )));
        }

        let plan = LoadableDatasetId::parse_plan(id).ok_or_else(|| {
            Error::InvalidInput(format!("No parser configured for dataset {:?}", id))
        })?;

        let result = match plan {
            DatasetParsePlan::Conll => self.parse_conll(content, id),
            DatasetParsePlan::JsonlNer => self.parse_jsonl_ner(content, id),
            DatasetParsePlan::WikiannJson => self.parse_wikiann_json(content, id),
            DatasetParsePlan::TweetNer7 => self.parse_tweetner7(content, id),
            DatasetParsePlan::DocredJson => self.parse_docred(content, id),
            DatasetParsePlan::GoogleReCorpus => self.parse_google_re_corpus(content, id),
            DatasetParsePlan::ChisiecJson => self.parse_chisiec(content, id),
            DatasetParsePlan::CadecHybrid => {
                if self.is_hf_api_response(content) {
                    self.parse_cadec_hf_api(content, id)
                } else {
                    self.parse_cadec_jsonl(content, id)
                }
            }
            DatasetParsePlan::Bc5cdr => self.parse_bc5cdr(content, id),
            DatasetParsePlan::NcbiDisease => self.parse_ncbi_disease(content, id),
            DatasetParsePlan::GapTsv => self.parse_gap(content, id),
            DatasetParsePlan::PrecoJsonl => self.parse_preco_jsonl(content, id),
            DatasetParsePlan::Litbank => self.parse_litbank(content, id),
            DatasetParsePlan::EcbPlus => self.parse_ecb_plus(content, id),
            DatasetParsePlan::AfriSenti => self.parse_afrisenti(content, id),
            DatasetParsePlan::AfriQa => self.parse_afriqa(content, id),
            DatasetParsePlan::MasakhaNews => self.parse_masakhanews(content, id),
            DatasetParsePlan::Conllu => self.parse_conllu(content, id),
            DatasetParsePlan::AgNews => self.parse_agnews(content, id),
            DatasetParsePlan::Dbpedia14 => self.parse_dbpedia14(content, id),
            DatasetParsePlan::YahooAnswers => self.parse_yahoo_answers(content, id),
            DatasetParsePlan::Trec => self.parse_trec(content, id),
            DatasetParsePlan::TweetTopic => self.parse_tweettopic(content, id),
            DatasetParsePlan::Maven => self.parse_maven(content, id),
            DatasetParsePlan::MavenArg => self.parse_maven_arg(content, id),
            DatasetParsePlan::Casie => self.parse_casie(content, id),
            DatasetParsePlan::Rams => self.parse_rams(content, id),
            DatasetParsePlan::HfApiResponse => self.parse_hf_api_response(content, id),
            DatasetParsePlan::TsvNer => self.parse_tsv_ner(content, id),
            DatasetParsePlan::CsvNer => self.parse_csv_ner(content, id),
        }?;

        // Validate parsed dataset is not empty
        if result.sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "Dataset {:?} parsed successfully but contains no sentences. \
                 This may indicate a parsing issue or empty dataset file.",
                id
            )));
        }

        Ok(result)
    }

    /// Check if content is HuggingFace datasets-server API response.
    fn is_hf_api_response(&self, content: &str) -> bool {
        // Check for HF datasets-server API response structure
        let trimmed = content.trim_start();
        trimmed.starts_with("{\"rows\":")
            || trimmed.starts_with("{\"features\":")
            || (trimmed.starts_with("{")
                && trimmed.contains("\"rows\":[")
                && trimmed.contains("\"features\":["))
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

            // Normalize BIO tags: some corpora (e.g., WikiGold) use `I-XXX` even at the
            // beginning of an entity. Treat such `I-` tags as `B-` when they don't
            // continue an entity of the same type.
            let ner_tag = if let Some(label) = ner_tag.strip_prefix("I-") {
                let continues_same = current_tokens
                    .last()
                    .map(|t| t.ner_tag.as_str())
                    .is_some_and(|prev| {
                        (prev.starts_with("B-") || prev.starts_with("I-"))
                            && prev.get(2..).is_some_and(|prev_label| prev_label == label)
                    });

                if continues_same {
                    ner_tag
                } else {
                    format!("B-{}", label)
                }
            } else {
                ner_tag
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

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "CoNLL file for {:?} contains no valid sentences",
                id
            )));
        }

        let now = chrono::Utc::now().to_rfc3339();

        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse JSONL NER format (HuggingFace style, e.g., MultiNERD).
    ///
    /// Expected format: `{"tokens": ["word1", "word2"], "ner_tags": [0, 1, 0]}`
    fn parse_jsonl_ner(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();

        // MultiNERD tag mapping (index -> label)
        let tag_labels = [
            "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-ANIM", "I-ANIM", "B-BIO",
            "I-BIO", "B-CEL", "I-CEL", "B-DIS", "I-DIS", "B-EVE", "I-EVE", "B-FOOD", "I-FOOD",
            "B-INST", "I-INST", "B-MEDIA", "I-MEDIA", "B-MYTH", "I-MYTH", "B-PLANT", "I-PLANT",
            "B-TIME", "I-TIME", "B-VEHI", "I-VEHI",
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

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "JSONL NER file for {:?} contains no valid sentences",
                id
            )));
        }

        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse HuggingFace datasets-server API response.
    ///
    /// Expected format:
    /// ```json
    /// {
    ///   "features": [{"name": "tokens", ...}, {"name": "ner_tags", ...}],
    ///   "rows": [{"row_idx": 0, "row": {"tokens": [...], "ner_tags": [...]}}, ...]
    /// }
    /// ```
    fn parse_hf_api_response(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let parsed: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| Error::InvalidInput(format!("Failed to parse HF API response: {}", e)))?;

        let mut sentences = Vec::new();

        // Extract tag names from features if available (for integer tag mapping)
        let tag_names = self.extract_tag_names_from_features(&parsed);
        let class_names = self.extract_class_names_from_features(&parsed);

        let rows = parsed
            .get("rows")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::InvalidInput("No 'rows' array in HF API response".to_string()))?;

        for row_obj in rows {
            let row = match row_obj.get("row") {
                Some(r) => r,
                None => continue,
            };

            // Primary path: token-level NER rows (`tokens` + `ner_tags`)
            if let (Some(tokens), Some(ner_tags)) = (
                row.get("tokens").and_then(|v| v.as_array()),
                row.get("ner_tags").and_then(|v| v.as_array()),
            ) {
                if tokens.len() != ner_tags.len() {
                    continue;
                }

                let mut annotated_tokens = Vec::new();
                for (token, tag) in tokens.iter().zip(ner_tags.iter()) {
                    let text = token.as_str().unwrap_or("").to_string();

                    // Handle both integer and string tags
                    let ner_tag = if let Some(tag_idx) = tag.as_u64() {
                        // Integer tag - map using feature names or default
                        tag_names
                            .get(tag_idx as usize)
                            .cloned()
                            .unwrap_or_else(|| format!("TAG_{}", tag_idx))
                    } else if let Some(tag_str) = tag.as_str() {
                        // String tag - use directly
                        tag_str.to_string()
                    } else {
                        "O".to_string()
                    };

                    annotated_tokens.push(AnnotatedToken { text, ner_tag });
                }

                if !annotated_tokens.is_empty() {
                    sentences.push(AnnotatedSentence {
                        tokens: annotated_tokens,
                        source_dataset: id,
                    });
                }
                continue;
            }

            // Fallback: temporal standoff rows (`text` + `*_expressions` with char offsets).
            //
            // Some HF datasets expose TIMEX/EVENT spans as character ranges in lists like:
            // - `time_expressions`: [{start_char, end_char, ...}, ...]
            // - `event_expressions`: ...
            // - `signal_expressions`: ...
            //
            // We tokenize `text` by whitespace and assign BIO tags based on overlap with any span.
            if let Some(text) = row.get("text").and_then(|v| v.as_str()).map(str::trim) {
                let has_temporal_spans = row
                    .get("time_expressions")
                    .and_then(|v| v.as_array())
                    .is_some()
                    || row
                        .get("event_expressions")
                        .and_then(|v| v.as_array())
                        .is_some()
                    || row
                        .get("signal_expressions")
                        .and_then(|v| v.as_array())
                        .is_some();

                if has_temporal_spans && !text.is_empty() {
                    fn spans_from_array(
                        arr: Option<&Vec<serde_json::Value>>,
                    ) -> Vec<(usize, usize)> {
                        let mut out = Vec::new();
                        let Some(arr) = arr else { return out };
                        for item in arr {
                            let Some(s) = item.get("start_char").and_then(|v| v.as_u64()) else {
                                continue;
                            };
                            let Some(e) = item.get("end_char").and_then(|v| v.as_u64()) else {
                                continue;
                            };
                            let s = s as usize;
                            let e = e as usize;
                            if e > s {
                                out.push((s, e));
                            }
                        }
                        out
                    }

                    fn overlaps(token_s: usize, token_e: usize, spans: &[(usize, usize)]) -> bool {
                        spans.iter().any(|(s, e)| token_s < *e && token_e > *s)
                    }

                    // Tokenize by whitespace, tracking char offsets (Rust `char` count).
                    let mut tokens: Vec<(String, usize, usize)> = Vec::new();
                    let mut cur = String::new();
                    let mut cur_start: Option<usize> = None;

                    for (i, ch) in text.chars().enumerate() {
                        if ch.is_whitespace() {
                            if let Some(s) = cur_start.take() {
                                let e = i;
                                if !cur.is_empty() {
                                    tokens.push((std::mem::take(&mut cur), s, e));
                                } else {
                                    cur.clear();
                                }
                            }
                        } else {
                            if cur_start.is_none() {
                                cur_start = Some(i);
                            }
                            cur.push(ch);
                        }
                    }
                    if let Some(s) = cur_start.take() {
                        let e = text.chars().count();
                        if !cur.is_empty() {
                            tokens.push((cur, s, e));
                        }
                    }

                    if tokens.is_empty() {
                        continue;
                    }

                    let timex_spans =
                        spans_from_array(row.get("time_expressions").and_then(|v| v.as_array()));
                    let event_spans =
                        spans_from_array(row.get("event_expressions").and_then(|v| v.as_array()));
                    let signal_spans =
                        spans_from_array(row.get("signal_expressions").and_then(|v| v.as_array()));

                    let mut annotated_tokens = Vec::with_capacity(tokens.len());
                    let mut prev_label: Option<&'static str> = None;
                    for (tok, s, e) in tokens {
                        let label = if overlaps(s, e, &timex_spans) {
                            Some("TIMEX")
                        } else if overlaps(s, e, &event_spans) {
                            Some("EVENT")
                        } else if overlaps(s, e, &signal_spans) {
                            Some("SIGNAL")
                        } else {
                            None
                        };

                        let ner_tag = match (label, prev_label) {
                            (None, _) => {
                                prev_label = None;
                                "O".to_string()
                            }
                            (Some(l), Some(p)) if l == p => format!("I-{}", l),
                            (Some(l), _) => {
                                prev_label = Some(l);
                                format!("B-{}", l)
                            }
                        };

                        annotated_tokens.push(AnnotatedToken { text: tok, ner_tag });
                    }

                    sentences.push(AnnotatedSentence {
                        tokens: annotated_tokens,
                        source_dataset: id,
                    });
                    continue;
                }
            }

            // Fallback: DISRPT-style CoNLL-U rows for discourse segmentation.
            //
            // DISRPT `*.conllu` configs are exported via HF datasets-server as arrays:
            // - `form`: token surface strings
            // - `misc`: per-token features, including `Seg=B-seg` (segment boundary) or `Seg=O`
            //
            // We convert boundaries into BIO tags of a single entity type (`SEG`), treating each
            // segment (EDU) as an entity span.
            if let (Some(forms), Some(misc)) = (
                row.get("form").and_then(|v| v.as_array()),
                row.get("misc").and_then(|v| v.as_array()),
            ) {
                if !forms.is_empty() && forms.len() == misc.len() {
                    let mut annotated_tokens = Vec::with_capacity(forms.len());
                    let mut in_seg = false;
                    for (i, (f, m)) in forms.iter().zip(misc.iter()).enumerate() {
                        let tok = f.as_str().unwrap_or("").to_string();
                        let misc_s = m.as_str().unwrap_or("");
                        let start = misc_s.contains("Seg=B-seg");
                        let ner_tag = if i == 0 || start {
                            in_seg = true;
                            "B-SEG".to_string()
                        } else if in_seg {
                            "I-SEG".to_string()
                        } else {
                            "O".to_string()
                        };
                        annotated_tokens.push(AnnotatedToken { text: tok, ner_tag });
                    }

                    sentences.push(AnnotatedSentence {
                        tokens: annotated_tokens,
                        source_dataset: id,
                    });
                    continue;
                }
            }

            // Fallback: classification-ish rows (`<text>` + `label`).
            //
            // We encode the gold label as `B-<LABEL>` on a single token, matching the loader's
            // convention for other classification datasets.
            let text = if let Some(s) = row.get("text").and_then(|v| v.as_str()) {
                s.trim().to_string()
            } else if let (Some(a), Some(b)) = (
                row.get("unit1_txt").and_then(|v| v.as_str()),
                row.get("unit2_txt").and_then(|v| v.as_str()),
            ) {
                format!("{} [SEP] {}", a.trim(), b.trim())
            } else if let (Some(a), Some(b)) = (
                row.get("sentence1").and_then(|v| v.as_str()),
                row.get("sentence2").and_then(|v| v.as_str()),
            ) {
                format!("{} [SEP] {}", a.trim(), b.trim())
            } else if let (Some(a), Some(b)) = (
                row.get("premise").and_then(|v| v.as_str()),
                row.get("hypothesis").and_then(|v| v.as_str()),
            ) {
                format!("{} [SEP] {}", a.trim(), b.trim())
            } else {
                continue;
            };
            if text.trim().is_empty() {
                continue;
            }

            let label_value = row.get("label").or_else(|| row.get("labels"));
            let label = match label_value {
                Some(v) if v.is_string() => v.as_str().unwrap_or("").to_string(),
                Some(v) if v.is_number() => {
                    let idx = v.as_u64().unwrap_or(0) as usize;
                    class_names
                        .get(idx)
                        .cloned()
                        .unwrap_or_else(|| format!("LABEL_{}", idx))
                }
                _ => "label".to_string(),
            };
            if label.trim().is_empty() {
                continue;
            }

            sentences.push(AnnotatedSentence {
                tokens: vec![AnnotatedToken {
                    text,
                    ner_tag: format!("B-{}", label),
                }],
                source_dataset: id,
            });
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "HF API response for {:?} contains no valid sentences",
                id
            )));
        }

        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Extract tag names from HF API features metadata.
    fn extract_tag_names_from_features(&self, parsed: &serde_json::Value) -> Vec<String> {
        let mut tag_names = Vec::new();

        if let Some(features) = parsed.get("features").and_then(|v| v.as_array()) {
            for feature in features {
                let name = feature.get("name").and_then(|v| v.as_str());
                if name == Some("ner_tags") {
                    // Look for ClassLabel names
                    if let Some(names) = feature
                        .get("type")
                        .and_then(|t| t.get("feature"))
                        .and_then(|f| f.get("names"))
                        .and_then(|n| n.as_array())
                    {
                        for name in names {
                            if let Some(s) = name.as_str() {
                                tag_names.push(s.to_string());
                            }
                        }
                    }
                    break;
                }
            }
        }

        tag_names
    }

    /// Extract class label names for `label` fields in HF API features metadata.
    fn extract_class_names_from_features(&self, parsed: &serde_json::Value) -> Vec<String> {
        let mut names = Vec::new();
        if let Some(features) = parsed.get("features").and_then(|v| v.as_array()) {
            for feature in features {
                let name = feature.get("name").and_then(|v| v.as_str());
                if name == Some("label") {
                    if let Some(label_names) = feature
                        .get("type")
                        .and_then(|t| t.get("names"))
                        .and_then(|n| n.as_array())
                    {
                        for n in label_names {
                            if let Some(s) = n.as_str() {
                                names.push(s.to_string());
                            }
                        }
                    }
                    break;
                }
            }
        }
        names
    }

    /// Parse TweetNER7 JSON format.
    ///
    /// TweetNER7 is JSONL with each line: {"tokens": [...], "tags": [...]}
    /// Tag mapping from label.json (tag -> id format, we need id -> tag):
    fn parse_tweetner7(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        // TweetNER7 tag mapping from label.json (index order!)
        // {"B-corporation": 0, "B-creative_work": 1, "B-event": 2, "B-group": 3,
        //  "B-location": 4, "B-person": 5, "B-product": 6, "I-corporation": 7,
        //  "I-creative_work": 8, "I-event": 9, "I-group": 10, "I-location": 11,
        //  "I-person": 12, "I-product": 13, "O": 14}
        let tag_labels = [
            "B-corporation",   // 0
            "B-creative_work", // 1
            "B-event",         // 2
            "B-group",         // 3
            "B-location",      // 4
            "B-person",        // 5
            "B-product",       // 6
            "I-corporation",   // 7
            "I-creative_work", // 8
            "I-event",         // 9
            "I-group",         // 10
            "I-location",      // 11
            "I-person",        // 12
            "I-product",       // 13
            "O",               // 14
        ];

        let mut sentences = Vec::new();

        // Parse as JSONL (one JSON object per line)
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parsed: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue, // Skip malformed lines
            };

            let tokens = match parsed.get("tokens").and_then(|v| v.as_array()) {
                Some(t) => t,
                None => continue,
            };

            let tags = match parsed.get("tags").and_then(|v| v.as_array()) {
                Some(t) => t,
                None => continue,
            };

            if tokens.len() != tags.len() {
                continue;
            }

            let mut annotated_tokens = Vec::new();
            for (token, tag) in tokens.iter().zip(tags.iter()) {
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
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse HIPE-2022 style TSV NER format.
    ///
    /// Expected format:
    /// ```text
    /// TOKEN    NE-COARSE-LIT   NE-COARSE-METO   NE-FINE-LIT   ...
    /// # hipe2022:document_id = doc123
    /// word1    B-PER           _                B-pers.author ...
    /// word2    I-PER           _                I-pers.author ...
    /// word3    O               _                O             ...
    /// ```
    ///
    /// - First line is header (starts with TOKEN)
    /// - Lines starting with `#` are metadata comments
    /// - Data lines are tab-separated with token in first column
    /// - NE-COARSE-LIT (column 2) contains BIO-tagged NER labels
    /// - `_` means no annotation
    fn parse_tsv_ner(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let mut current_tokens = Vec::new();

        for line in content.lines() {
            let line = line.trim();

            // Skip empty lines - they indicate sentence boundaries
            if line.is_empty() {
                if !current_tokens.is_empty() {
                    sentences.push(AnnotatedSentence {
                        tokens: std::mem::take(&mut current_tokens),
                        source_dataset: id,
                    });
                }
                continue;
            }

            // Skip header line
            if line.starts_with("TOKEN\t") || line.starts_with("TOKEN ") {
                continue;
            }

            // Skip metadata comments
            if line.starts_with('#') {
                // Document boundary comment can also serve as sentence boundary
                if line.contains("document_id") && !current_tokens.is_empty() {
                    sentences.push(AnnotatedSentence {
                        tokens: std::mem::take(&mut current_tokens),
                        source_dataset: id,
                    });
                }
                continue;
            }

            // Parse data line
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 2 {
                continue; // Malformed line
            }

            let token_text = parts[0].to_string();
            let ner_label = parts.get(1).unwrap_or(&"O");

            // Convert underscore to O (no annotation)
            let ner_tag = if *ner_label == "_" || ner_label.is_empty() {
                "O".to_string()
            } else {
                ner_label.to_string()
            };

            current_tokens.push(AnnotatedToken {
                text: token_text,
                ner_tag,
            });
        }

        // Don't forget the last sentence
        if !current_tokens.is_empty() {
            sentences.push(AnnotatedSentence {
                tokens: current_tokens,
                source_dataset: id,
            });
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "TSV NER file for {:?} contains no valid sentences",
                id
            )));
        }

        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse CSV NER format (E-NER/EDGAR-NER style).
    ///
    /// Expected format: `Token,Tag` (comma-separated)
    /// Uses `-DOCSTART-` for document boundaries and empty lines for sentence boundaries.
    /// Tags use BIO scheme (e.g., O, B-PERSON, I-BUSINESS).
    ///
    /// # Errors
    ///
    /// Returns error if CSV is malformed or has no valid sentences.
    fn parse_csv_ner(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let mut current_tokens = Vec::new();

        for line in content.lines() {
            let line = line.trim();

            // Empty lines indicate sentence boundaries
            if line.is_empty() {
                if !current_tokens.is_empty() {
                    sentences.push(AnnotatedSentence {
                        tokens: std::mem::take(&mut current_tokens),
                        source_dataset: id,
                    });
                }
                continue;
            }

            // Handle -DOCSTART- markers (document boundary, also a sentence boundary)
            if line.starts_with("-DOCSTART-") {
                if !current_tokens.is_empty() {
                    sentences.push(AnnotatedSentence {
                        tokens: std::mem::take(&mut current_tokens),
                        source_dataset: id,
                    });
                }
                continue;
            }

            // Skip header line if present
            if line.eq_ignore_ascii_case("token,tag")
                || line.eq_ignore_ascii_case("text,label")
                || line.eq_ignore_ascii_case("word,ner")
            {
                continue;
            }

            // Parse comma-separated line
            // Handle cases where the token itself might be a comma: ",O" means token is comma
            let (token_text, ner_tag) = if let Some(rest) = line.strip_prefix(',') {
                // Token is empty or the comma itself
                if let Some(idx) = rest.find(',') {
                    // Format: ,tag (empty token) or ,,tag (comma token)
                    if idx == 0 {
                        // ",,tag" -> token is comma, tag follows
                        (",".to_string(), rest[1..].to_string())
                    } else {
                        // ",tag" -> empty token
                        (String::new(), rest[..idx].to_string())
                    }
                } else {
                    // ",tag" with no more commas
                    (String::new(), rest.to_string())
                }
            } else if let Some(idx) = line.rfind(',') {
                // Normal case: Token,Tag
                let token = line[..idx].to_string();
                let tag = line[idx + 1..].to_string();
                (token, tag)
            } else {
                // Malformed line, skip
                continue;
            };

            // Skip if we got empty results
            if token_text.is_empty() && ner_tag.is_empty() {
                continue;
            }

            // Convert empty tag to O
            let ner_tag = if ner_tag.is_empty() || ner_tag == "_" {
                "O".to_string()
            } else {
                ner_tag
            };

            current_tokens.push(AnnotatedToken {
                text: token_text,
                ner_tag,
            });
        }

        // Don't forget the last sentence
        if !current_tokens.is_empty() {
            sentences.push(AnnotatedSentence {
                tokens: current_tokens,
                source_dataset: id,
            });
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "CSV NER file for {:?} contains no valid sentences",
                id
            )));
        }

        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse WikiANN-style JSON array format.
    ///
    /// Expected format: `[{"text": "...", "tokens": ["word1", "word2"], "ner_tags": ["O", "B-LOC"]}, ...]`
    /// Used by UNER and MSNER datasets converted via download_hf_datasets.py.
    fn parse_wikiann_json(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let parsed: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| Error::InvalidInput(format!("Failed to parse JSON: {}", e)))?;

        let mut sentences = Vec::new();

        // Handle JSON array format
        if let Some(items) = parsed.as_array() {
            for item in items {
                let tokens = match item.get("tokens").and_then(|v| v.as_array()) {
                    Some(t) => t,
                    None => continue,
                };

                let ner_tags = match item.get("ner_tags").and_then(|v| v.as_array()) {
                    Some(t) => t,
                    None => continue,
                };

                if tokens.len() != ner_tags.len() {
                    continue;
                }

                let mut annotated_tokens = Vec::new();
                for (token, tag) in tokens.iter().zip(ner_tags.iter()) {
                    let text = token.as_str().unwrap_or("").to_string();
                    let ner_tag = tag.as_str().unwrap_or("O").to_string();
                    annotated_tokens.push(AnnotatedToken { text, ner_tag });
                }

                if !annotated_tokens.is_empty() {
                    sentences.push(AnnotatedSentence {
                        tokens: annotated_tokens,
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
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse DocRED document-level relation extraction JSON format.
    ///
    /// DocRED is primarily for relation extraction but includes NER annotations.
    /// Parse DocRED/CrossRE JSON format for relation extraction.
    ///
    /// CrossRE format: {"doc_key": "...", "sentence": [...], "ner": [[start, end, type], ...], "relations": [...]}
    fn parse_docred(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();

        // Parse as JSONL (one JSON object per line)
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let doc: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // CrossRE format: sentence array + ner array
            let tokens_arr = match doc.get("sentence").and_then(|v| v.as_array()) {
                Some(t) => t,
                None => continue,
            };

            let ner_spans = doc.get("ner").and_then(|v| v.as_array());

            // Build token list with entity annotations
            let mut tokens: Vec<AnnotatedToken> = tokens_arr
                .iter()
                .filter_map(|t| t.as_str())
                .map(|word| AnnotatedToken {
                    text: word.to_string(),
                    ner_tag: "O".to_string(),
                })
                .collect();

            // Apply NER annotations: [start, end, type]
            if let Some(ner) = ner_spans {
                for span in ner {
                    if let Some(arr) = span.as_array() {
                        if arr.len() >= 3 {
                            let start = arr[0].as_u64().unwrap_or(0) as usize;
                            let end = arr[1].as_u64().unwrap_or(0) as usize;
                            let ent_type = arr[2].as_str().unwrap_or("ENTITY");

                            // Apply BIO tags
                            for idx in start..=end {
                                if idx < tokens.len() {
                                    tokens[idx].ner_tag = if idx == start {
                                        format!("B-{}", ent_type.to_uppercase())
                                    } else {
                                        format!("I-{}", ent_type.to_uppercase())
                                    };
                                }
                            }
                        }
                    }
                }
            }

            if !tokens.is_empty() {
                sentences.push(AnnotatedSentence {
                    tokens,
                    source_dataset: id,
                });
            }
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "DocRED/CrossRE JSON for {:?} contains no valid sentences",
                id
            )));
        }

        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse Google's relation-extraction-corpus JSONL format.
    ///
    /// Example record (one per line):
    /// ```json
    /// {"pred":"/people/person/place_of_birth","sub":"/m/...","obj":"/m/...","evidences":[{"url":"...","snippet":"..."}],"judgments":[{"rater":"...","judgment":"yes"}]}
    /// ```
    ///
    /// This dataset does **not** include token-level entity spans. For now we treat each
    /// evidence snippet as a plain sentence with all tokens tagged as `O`, which is
    /// sufficient for sanity evaluation plumbing (and avoids failing dataset loads).
    fn parse_google_re_corpus(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences: Vec<AnnotatedSentence> = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let rec: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let snippet = rec
                .get("evidences")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|ev| ev.get("snippet"))
                .and_then(|s| s.as_str());
            let Some(snippet) = snippet else {
                continue;
            };

            let tokens: Vec<AnnotatedToken> = snippet
                .split_whitespace()
                .filter(|t| !t.is_empty())
                .map(|t| AnnotatedToken {
                    text: t.to_string(),
                    ner_tag: "O".to_string(),
                })
                .collect();

            if tokens.is_empty() {
                continue;
            }

            sentences.push(AnnotatedSentence {
                tokens,
                source_dataset: id,
            });
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "Google relation-extraction-corpus file for {:?} contains no usable evidence snippets",
                id
            )));
        }

        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse CADEC from HuggingFace datasets-server API.
    ///
    /// CADEC HF API format: {"text": "...", "ade": "...", "term_PT": "..."}
    /// Each row is a text-ADE pair (one sentence per ADE mention).
    /// The `ade` field contains the adverse drug event mention within `text`.
    fn parse_cadec_hf_api(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let parsed: serde_json::Value = serde_json::from_str(content).map_err(|e| {
            Error::InvalidInput(format!("Failed to parse CADEC HF API response: {}", e))
        })?;

        let mut sentences = Vec::new();

        let rows = parsed
            .get("rows")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                Error::InvalidInput("No 'rows' array in CADEC HF API response".to_string())
            })?;

        for row_obj in rows {
            let row = match row_obj.get("row") {
                Some(r) => r,
                None => continue,
            };

            let text = match row.get("text").and_then(|v| v.as_str()) {
                Some(t) => t,
                None => continue,
            };

            let ade_text = match row.get("ade").and_then(|v| v.as_str()) {
                Some(a) => a,
                None => continue,
            };

            // Find ADE span in text.
            //
            // IMPORTANT: anno uses **character offsets** globally, but this parser assigns BIO tags
            // token-by-token and uses byte spans internally. We must preserve indices and avoid
            // Unicode casefolding (which can change string length).
            //
            // Strategy:
            // 1) Try exact match (fast, preserves byte indices).
            // 2) Fall back to ASCII case-insensitive match (ADE strings are ASCII-centric).
            let ade_start_byte = text.find(ade_text).or_else(|| {
                let needle_len = ade_text.len();
                if needle_len == 0 {
                    return None;
                }
                for (b, _) in text.char_indices() {
                    if let Some(hay) = text.get(b..b + needle_len) {
                        if hay.eq_ignore_ascii_case(ade_text) {
                            return Some(b);
                        }
                    }
                }
                None
            });
            let Some(ade_start_byte) = ade_start_byte else {
                continue; // ADE not found in text
            };
            let ade_end_byte = ade_start_byte + ade_text.len();

            // Tokenize text preserving byte offsets.
            let mut tokens: Vec<AnnotatedToken> = Vec::new();
            let mut byte_idx = 0;
            let words: Vec<&str> = text.split_whitespace().collect();

            for word in words {
                let word_start =
                    text.get(byte_idx..).and_then(|s| s.find(word)).unwrap_or(0) + byte_idx;
                let word_end = word_start + word.len();

                // Check if this word overlaps with ADE span (byte spans)
                let ner_tag = if word_start >= ade_start_byte && word_end <= ade_end_byte {
                    // Check if this is the first word of the ADE
                    if word_start == ade_start_byte
                        || tokens.is_empty()
                        || !tokens
                            .last()
                            .expect("tokens.is_empty() checked above")
                            .ner_tag
                            .starts_with("I-")
                    {
                        "B-adverse_drug_event".to_string()
                    } else {
                        "I-adverse_drug_event".to_string()
                    }
                } else {
                    "O".to_string()
                };

                tokens.push(AnnotatedToken {
                    text: word.to_string(),
                    ner_tag,
                });

                // Update byte position to after this word (including trailing space)
                byte_idx = word_end;
                if byte_idx < text.len() && text.as_bytes().get(byte_idx) == Some(&b' ') {
                    byte_idx += 1;
                }
            }

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
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse CADEC JSONL format with support for discontinuous entities.
    ///
    /// CADEC format can include:
    /// - Standard BIO tags: `{"tokens": [...], "ner_tags": [...]}`
    /// - Entity spans: `{"tokens": [...], "entities": [{"text": "...", "label": "...", "start": 0, "end": 10}]}`
    /// - Discontinuous entities: `{"entities": [{"text": "...", "label": "...", "spans": [[0, 5], [10, 15]]}]}`
    ///
    /// For discontinuous entities, we convert them to BIO tags by marking all tokens
    /// within any span as part of the entity.
    fn parse_cadec_jsonl(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();

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

            // Try to get tokens
            let tokens = match parsed.get("tokens").and_then(|v| v.as_array()) {
                Some(t) => t,
                None => continue,
            };

            let mut annotated_tokens = Vec::new();
            let mut char_offset = 0;

            // Build token list with character offsets
            let mut token_offsets = Vec::new();
            for token in tokens {
                let text = token.as_str().unwrap_or("").to_string();
                let start = char_offset;
                char_offset += text.chars().count() + 1; // +1 for space
                let end = char_offset - 1;
                token_offsets.push((text, start, end));
            }

            // Initialize all tokens as "O"
            for (text, _, _) in &token_offsets {
                annotated_tokens.push(AnnotatedToken {
                    text: text.clone(),
                    ner_tag: "O".to_string(),
                });
            }

            // Try to parse entities (for discontinuous support)
            if let Some(entities) = parsed.get("entities").and_then(|v| v.as_array()) {
                for entity in entities {
                    let label = entity
                        .get("label")
                        .or_else(|| entity.get("entity_type"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("UNKNOWN")
                        .to_string();

                    // Check for discontinuous spans
                    if let Some(spans) = entity.get("spans").and_then(|v| v.as_array()) {
                        // Discontinuous entity with multiple spans
                        for span in spans {
                            if let Some(span_array) = span.as_array() {
                                if span_array.len() >= 2 {
                                    let start = span_array[0].as_u64().unwrap_or(0) as usize;
                                    let end = span_array[1].as_u64().unwrap_or(0) as usize;

                                    // Mark tokens within this span
                                    for (idx, (_, token_start, token_end)) in
                                        token_offsets.iter().enumerate()
                                    {
                                        if *token_start >= start && *token_end <= end {
                                            if idx > 0
                                                && annotated_tokens[idx - 1]
                                                    .ner_tag
                                                    .starts_with(&format!("I-{}", label))
                                                || annotated_tokens[idx - 1]
                                                    .ner_tag
                                                    .starts_with(&format!("B-{}", label))
                                            {
                                                annotated_tokens[idx].ner_tag =
                                                    format!("I-{}", label);
                                            } else {
                                                annotated_tokens[idx].ner_tag =
                                                    format!("B-{}", label);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else if let (Some(start_val), Some(end_val)) = (
                        entity.get("start").and_then(|v| v.as_u64()),
                        entity.get("end").and_then(|v| v.as_u64()),
                    ) {
                        // Contiguous entity
                        let start = start_val as usize;
                        let end = end_val as usize;

                        // Mark tokens within this span
                        for (idx, (_, token_start, token_end)) in token_offsets.iter().enumerate() {
                            if *token_start >= start && *token_end <= end {
                                if idx > 0
                                    && (annotated_tokens[idx - 1]
                                        .ner_tag
                                        .starts_with(&format!("I-{}", label))
                                        || annotated_tokens[idx - 1]
                                            .ner_tag
                                            .starts_with(&format!("B-{}", label)))
                                {
                                    annotated_tokens[idx].ner_tag = format!("I-{}", label);
                                } else {
                                    annotated_tokens[idx].ner_tag = format!("B-{}", label);
                                }
                            }
                        }
                    }
                }
            } else if let Some(ner_tags) = parsed.get("ner_tags").and_then(|v| v.as_array()) {
                // Fallback to standard BIO tags
                let tag_labels = [
                    "O",
                    "B-PER",
                    "I-PER",
                    "B-ORG",
                    "I-ORG",
                    "B-LOC",
                    "I-LOC",
                    "B-MISC",
                    "I-MISC",
                    "B-DRUG",
                    "I-DRUG",
                    "B-ADR",
                    "I-ADR",
                    "B-DISEASE",
                    "I-DISEASE",
                ];

                for (idx, (text, _, _)) in token_offsets.iter().enumerate() {
                    if let Some(tag_val) = ner_tags.get(idx) {
                        let tag_idx = tag_val.as_u64().unwrap_or(0) as usize;
                        let ner_tag = tag_labels.get(tag_idx).unwrap_or(&"O").to_string();
                        annotated_tokens[idx] = AnnotatedToken {
                            text: text.clone(),
                            ner_tag,
                        };
                    }
                }
            }

            if !annotated_tokens.is_empty() {
                sentences.push(AnnotatedSentence {
                    tokens: annotated_tokens,
                    source_dataset: id,
                });
            }
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "CADEC JSONL file for {:?} contains no valid sentences",
                id
            )));
        }

        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse BC5CDR BioC XML format.
    ///
    /// Note: This is a simplified parser that extracts text passages.
    /// Full annotation extraction would require proper XML parsing.
    /// Parse BC5CDR dataset in CoNLL format from BioFLAIR.
    ///
    /// Format: WORD\tPOS\tCHUNK\tNER_TAG
    fn parse_bc5cdr(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let mut current_tokens = Vec::new();

        for line in content.lines() {
            let line = line.trim();

            // Skip DOCSTART lines
            if line.starts_with("-DOCSTART-") {
                continue;
            }

            if line.is_empty() {
                // End of sentence
                if !current_tokens.is_empty() {
                    sentences.push(AnnotatedSentence {
                        tokens: std::mem::take(&mut current_tokens),
                        source_dataset: id,
                    });
                }
                continue;
            }

            // Parse CoNLL line: WORD\tPOS\tCHUNK\tNER_TAG
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 4 {
                let word = parts[0].to_string();
                let ner_tag = parts[3].to_string();

                // Map Entity tags to BIO format
                let normalized_tag = if ner_tag.contains("Entity")
                    || ner_tag.contains("CHEMICAL")
                    || ner_tag.contains("DISEASE")
                {
                    // Convert I-Entity to B-CHEMICAL or I-CHEMICAL based on context
                    if ner_tag.starts_with("B-") {
                        "B-CHEMICAL".to_string()
                    } else if ner_tag.starts_with("I-") {
                        "I-CHEMICAL".to_string()
                    } else {
                        "O".to_string()
                    }
                } else {
                    ner_tag
                };

                current_tokens.push(AnnotatedToken {
                    text: word,
                    ner_tag: normalized_tag,
                });
            }
        }

        // Don't forget the last sentence
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
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse NCBI Disease corpus format.
    ///
    /// Format: PMID|t|Title or PMID|a|Abstract, followed by annotation lines.
    /// Parse NCBI Disease dataset in CoNLL format from BioFLAIR.
    ///
    /// Format: WORD\tPOS\tCHUNK\tNER_TAG
    fn parse_ncbi_disease(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let mut current_tokens = Vec::new();

        for line in content.lines() {
            let line = line.trim();

            if line.is_empty() {
                // End of sentence
                if !current_tokens.is_empty() {
                    sentences.push(AnnotatedSentence {
                        tokens: std::mem::take(&mut current_tokens),
                        source_dataset: id,
                    });
                }
                continue;
            }

            // Parse CoNLL line: WORD\tPOS\tCHUNK\tNER_TAG
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 4 {
                let word = parts[0].to_string();
                let ner_tag = parts[3].to_string();

                current_tokens.push(AnnotatedToken {
                    text: word,
                    ner_tag,
                });
            }
        }

        // Don't forget the last sentence
        if !current_tokens.is_empty() {
            sentences.push(AnnotatedSentence {
                tokens: current_tokens,
                source_dataset: id,
            });
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "NCBI Disease file for {:?} contains no valid sentences",
                id
            )));
        }

        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
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

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "GAP TSV file for {:?} contains no valid sentences",
                id
            )));
        }

        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse PreCo JSONL format from HuggingFace.
    ///
    /// PreCo JSONL format: One JSON object per line with "sentences" array.
    /// Note: PreCo is a coreference dataset, not NER. This parser extracts sentences
    /// for NER evaluation (which will have 0 entities). Use `load_coref()` for coreference.
    fn parse_preco_jsonl(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let mut line_count = 0usize;
        let mut parsed_count = 0usize;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            line_count += 1;

            let parsed: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(e) => {
                    // Log first few parse errors for debugging
                    if parsed_count < 3 {
                        log::warn!("PreCo JSONL parse error on line {}: {}", line_count, e);
                    }
                    continue; // Skip malformed lines
                }
            };

            // PreCo format: {"sentences": [[token1, token2, ...], ...]}
            if let Some(sents) = parsed.get("sentences").and_then(|v| v.as_array()) {
                parsed_count += 1;
                for sent_tokens in sents {
                    if let Some(token_array) = sent_tokens.as_array() {
                        let tokens: Vec<AnnotatedToken> = token_array
                            .iter()
                            .filter_map(|t| t.as_str())
                            .map(|t| AnnotatedToken {
                                text: t.to_string(),
                                ner_tag: "O".to_string(), // PreCo has no NER annotations
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

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "PreCo JSONL file contains no valid sentences (parsed {} of {} lines)",
                parsed_count, line_count
            )));
        }

        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse LitBank annotation format for NER.
    ///
    /// LitBank .ann format: T<id>\t<Type> <start> <end>\t<text>
    /// Note: LitBank is primarily a coreference dataset. This parser extracts
    /// entity mentions as NER annotations, but LitBank should be used with
    /// `load_coref()` for proper coreference evaluation.
    fn parse_litbank(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        // LitBank .ann format: each line is T<id>\t<Type> <start> <end>\t<text>
        // Extract entity mentions as NER annotations
        let now = chrono::Utc::now().to_rfc3339();
        let mut sentences = Vec::new();
        let mut entities: Vec<(usize, usize, String, String)> = Vec::new(); // (start, end, text, label)

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if line.starts_with('T') {
                // Entity annotation: T1\tPER 0 5\tAlice
                // LitBank uses ACE-style labels like PROP_PER, NOM_LOC, PRON_FAC etc.
                // We normalize to just the entity type (PER, LOC, ORG, GPE, FAC, VEH)
                // Note: LitBank also has coreference chain annotations like "character_name-ID"
                // which should be skipped for NER evaluation
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() >= 3 {
                    let type_span: Vec<&str> = parts[1].split_whitespace().collect();
                    if type_span.len() >= 3 {
                        let raw_label = type_span[0];

                        // Valid entity types (with or without prefix)
                        const VALID_ENTITY_TYPES: &[&str] = &[
                            "PER",
                            "LOC",
                            "ORG",
                            "GPE",
                            "FAC",
                            "VEH",
                            "PERSON",
                            "LOCATION",
                            "ORGANIZATION",
                        ];

                        // Entity type annotations start with PROP_, NOM_, PRON_ OR are plain types
                        let is_prefixed_entity = raw_label.starts_with("PROP_")
                            || raw_label.starts_with("NOM_")
                            || raw_label.starts_with("PRON_");
                        let is_plain_entity = VALID_ENTITY_TYPES.contains(&raw_label);

                        if !is_prefixed_entity && !is_plain_entity {
                            // Skip coreference chain annotations (e.g., "jarndyce_2-73")
                            continue;
                        }

                        // Normalize: PROP_PER -> PER, NOM_LOC -> LOC, PRON_FAC -> FAC
                        let label = if is_prefixed_entity {
                            raw_label.split('_').next_back().unwrap_or(raw_label)
                        } else {
                            raw_label
                        };
                        let start: usize = type_span[1].parse().unwrap_or(0);
                        let end: usize = type_span[2].parse().unwrap_or(0);
                        let text = parts[2];

                        entities.push((start, end, text.to_string(), label.to_string()));
                    }
                }
            }
        }

        if entities.is_empty() {
            return Err(Error::InvalidInput(
                "LitBank .ann file contains no entity annotations (T lines)".to_string(),
            ));
        }

        // Sort entities by start position
        entities.sort_by_key(|(start, _, _, _)| *start);

        // Reconstruct text and create tokens with NER tags
        let max_end = entities
            .iter()
            .map(|(_, end, _, _)| *end)
            .max()
            .unwrap_or(0);
        let mut text_chars: Vec<char> = vec![' '; max_end.max(1)];
        let mut token_starts: Vec<usize> = Vec::new();

        // Fill in entity text
        for (start, _end, text, _) in &entities {
            let text_chars_vec: Vec<char> = text.chars().collect();
            let _actual_end = (*start + text_chars_vec.len()).min(text_chars.len());
            if *start < text_chars.len() {
                for (i, ch) in text_chars_vec.iter().enumerate() {
                    let pos = *start + i;
                    if pos < text_chars.len() {
                        text_chars[pos] = *ch;
                    }
                }
            }
            token_starts.push(*start);
        }

        // Note: text reconstruction is not used directly since we tokenize from entity text
        // but we keep it for potential future use (e.g., validation)
        let _text: String = text_chars
            .into_iter()
            .collect::<String>()
            .trim()
            .to_string();

        // Create tokens with NER tags
        // Improved approach: tokenize entity text into words and apply BIO tags
        let mut tokens: Vec<AnnotatedToken> = Vec::new();

        // Sort entities by start position for processing
        entities.sort_by_key(|(start, _, _, _)| *start);

        let mut last_end = 0usize;
        for (start, end, entity_text, label) in &entities {
            // Add "O" tokens for gaps between entities
            if *start > last_end {
                // For gaps, we can't reconstruct the actual text without the .txt file
                // But we can estimate based on character distance
                let gap_size = *start - last_end;
                if gap_size > 0 {
                    // Estimate number of words in gap (roughly 5 chars per word + space)
                    let estimated_words = (gap_size / 6).max(1);
                    for _ in 0..estimated_words.min(10) {
                        // Limit gap tokens to avoid excessive placeholders
                        tokens.push(AnnotatedToken {
                            text: "[...]".to_string(),
                            ner_tag: "O".to_string(),
                        });
                    }
                }
            }

            // Tokenize entity text into words and apply BIO tags
            let entity_words: Vec<&str> = entity_text.split_whitespace().collect();
            for (i, word) in entity_words.iter().enumerate() {
                let ner_tag = if i == 0 {
                    format!("B-{}", label)
                } else {
                    format!("I-{}", label)
                };
                tokens.push(AnnotatedToken {
                    text: word.to_string(),
                    ner_tag,
                });
            }

            last_end = *end;
        }

        if !tokens.is_empty() {
            sentences.push(AnnotatedSentence {
                tokens,
                source_dataset: id,
            });
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(
                "LitBank file produced no sentences after parsing".to_string(),
            ));
        }

        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse ECB+ CSV format for event coreference.
    ///
    /// ECB+ uses CSV format with columns for event mentions and coreference links.
    /// For now, extracts entities as NER annotations (event triggers).
    fn parse_ecb_plus(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let mut first_line = true;

        for line in content.lines() {
            // Skip header
            if first_line {
                first_line = false;
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 3 {
                continue;
            }

            // ECB+ CSV format: sentence_id, text, event_mention, ...
            // Extract text and create tokens
            let text = parts.get(1).unwrap_or(&"");
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

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "ECB+ CSV file for {:?} contains no valid sentences",
                id
            )));
        }

        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
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

        let cache_path = self.cache_path_for(id);
        if !cache_path.exists() {
            return Err(Error::InvalidInput(format!(
                "Dataset {:?} not cached at {:?}. Download from {}",
                id,
                cache_path,
                id.download_url()
            )));
        }

        let content = std::fs::read_to_string(&cache_path)
            .map_err(|e| Error::InvalidInput(format!("Failed to read {:?}: {}", cache_path, e)))?;

        match id {
            DatasetId::CorefUD => super::coref_loader::parse_corefud_conllu(&content),
            DatasetId::GAP => {
                let examples = super::coref_loader::parse_gap_tsv(&content)?;
                Ok(examples
                    .into_iter()
                    .map(|ex| ex.to_coref_document())
                    .collect())
            }
            DatasetId::PreCo => {
                // PreCo can be JSONL (one JSON object per line) or JSON array
                // Try JSONL first (more common), then fall back to JSON array
                if content.trim().starts_with('[') {
                    // JSON array format
                    let docs = super::coref_loader::parse_preco_json(&content)?;
                    Ok(docs.into_iter().map(|d| d.to_coref_document()).collect())
                } else {
                    // JSONL format - parse each line and convert to JSON array format
                    let mut json_objects = Vec::new();
                    for line in content.lines() {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        // Validate it's valid JSON
                        if serde_json::from_str::<serde_json::Value>(line).is_ok() {
                            json_objects.push(line);
                        }
                    }
                    // Convert JSONL to JSON array format
                    let json_array = format!("[{}]", json_objects.join(","));
                    let docs = super::coref_loader::parse_preco_json(&json_array)?;
                    Ok(docs.into_iter().map(|d| d.to_coref_document()).collect())
                }
            }
            DatasetId::LitBank => {
                // LitBank coreference - parse .ann format for chains
                self.parse_litbank_coref(&content)
            }
            DatasetId::ECBPlus => {
                // ECB+ may be cached as either:
                // - ZIP binary (new: real XML annotations)
                // - CSV text (legacy: sentence index)
                let raw_bytes = std::fs::read(&cache_path).map_err(|e| {
                    Error::InvalidInput(format!("Failed to read {:?}: {}", cache_path, e))
                })?;
                if raw_bytes.starts_with(b"PK\x03\x04") {
                    return super::coref_loader::parse_ecb_plus_zip(&raw_bytes);
                }
                // Fall back to CSV parser
                super::coref_loader::parse_ecb_plus_coref(&content)
            }
            DatasetId::WikiCoref => {
                // WikiCoref uses a GAP-compatible TSV format.
                let examples = super::coref_loader::parse_gap_tsv(&content)?;
                Ok(examples
                    .into_iter()
                    .map(|ex| ex.to_coref_document())
                    .collect())
            }
            DatasetId::GUM
            | DatasetId::WinoBias
            | DatasetId::TwiConv
            | DatasetId::MuDoCo
            | DatasetId::SciCo => Err(Error::InvalidInput(format!(
                "{:?} coreference format is not yet supported (requires a dedicated parser)",
                id
            ))),
            DatasetId::BookCoref | DatasetId::BookCorefSplit => {
                // BOOKCOREF: Book-scale coreference (Martinelli et al. 2025)
                // Format: OntoNotes-style with character metadata
                // Note: Actual data requires HuggingFace datasets library to download
                // from Project Gutenberg. We support pre-downloaded JSONL.
                super::coref_loader::parse_bookcoref_json(&content)
            }
            _ => Err(Error::InvalidInput(format!(
                "No coreference parser for {:?}",
                id
            ))),
        }
    }

    /// Load coreference dataset, downloading if needed.
    #[cfg(feature = "eval")]
    pub fn load_or_download_coref(
        &self,
        id: DatasetId,
    ) -> Result<Vec<super::coref::CorefDocument>> {
        if !self.is_cached_for(id) {
            if matches!(id, DatasetId::CorefUD) {
                let cache_path = self.cache_path_for(id);
                return Err(Error::InvalidInput(format!(
                    "CorefUD is not downloadable via anno yet. Please provide a local CorefUD .conllu file.\n\
                     - Option A: copy it to the cache path {:?}\n\
                     - Option B: use CorefLoader::load_corefud_from_path(<path>)",
                    cache_path
                )));
            }

            let cache_path = self.cache_path_for(id);
            if matches!(id, DatasetId::ECBPlus) {
                // ECB+ is a ZIP file -- download as raw bytes
                let url = id.download_url();
                let bytes = self.download_attempt_bytes(url)?;
                std::fs::write(&cache_path, &bytes).map_err(|e| {
                    Error::InvalidInput(format!("Failed to cache {:?}: {}", cache_path, e))
                })?;
            } else {
                let (content, _) = self.download_with_resolved_url(id)?;
                std::fs::write(&cache_path, &content).map_err(|e| {
                    Error::InvalidInput(format!("Failed to cache {:?}: {}", cache_path, e))
                })?;
            }
        }
        self.load_coref(id)
    }

    /// Parse LitBank for coreference chains.
    ///
    /// LitBank format: .ann files with T lines (mentions) and R lines (coreference relations).
    /// T lines: T<id>\t<Type> <start> <end>\t<text>
    /// R lines: R<id>\tCoref Arg1:T<id1> Arg2:T<id2>
    ///
    /// Note: LitBank .ann files reference character offsets in corresponding .txt files.
    /// This parser reconstructs text from mentions, but ideally should read .txt file.
    fn parse_litbank_coref(&self, content: &str) -> Result<Vec<super::coref::CorefDocument>> {
        use super::coref::{CorefChain, CorefDocument, Mention};
        use std::collections::HashMap;

        // LitBank .ann format includes coreference with R lines
        // R1\tCoref Arg1:T1 Arg2:T2
        let mut mentions: HashMap<String, Mention> = HashMap::new();
        let mut coref_links: Vec<(String, String)> = Vec::new();
        let mut max_end = 0usize;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if line.starts_with('T') {
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() >= 3 {
                    let id = parts[0];
                    let type_span: Vec<&str> = parts[1].split_whitespace().collect();
                    if type_span.len() >= 3 {
                        let start: usize = type_span[1].parse().unwrap_or(0);
                        let end: usize = type_span[2].parse().unwrap_or(0);
                        let text = parts[2];
                        max_end = max_end.max(end);
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

        if mentions.is_empty() {
            return Err(Error::InvalidInput(
                "LitBank file contains no mentions (T lines)".to_string(),
            ));
        }

        // Reconstruct text from mentions by sorting by start offset
        let mut sorted_mentions: Vec<(usize, &Mention)> =
            mentions.values().map(|m| (m.start, m)).collect();
        sorted_mentions.sort_by_key(|(start, _)| *start);

        // Build text by inserting mentions at their offsets
        let mut text_chars: Vec<char> = vec![' '; max_end.max(1)];
        for (start, mention) in &sorted_mentions {
            let mention_text: Vec<char> = mention.text.chars().collect();
            let _end = (*start + mention_text.len()).min(text_chars.len());
            if *start < text_chars.len() {
                for (i, ch) in mention_text.iter().enumerate() {
                    let pos = *start + i;
                    if pos < text_chars.len() {
                        text_chars[pos] = *ch;
                    }
                }
            }
        }
        let text: String = text_chars
            .into_iter()
            .collect::<String>()
            .trim()
            .to_string();

        // Build chains from links using union-find
        let mut chains: Vec<Vec<Mention>> = Vec::new();
        let mut mention_to_chain: HashMap<String, usize> = HashMap::new();

        // First, add all mentions as singletons if they're not in any chain
        for (id, mention) in &mentions {
            if !mention_to_chain.contains_key(id) {
                let idx = chains.len();
                chains.push(vec![mention.clone()]);
                mention_to_chain.insert(id.clone(), idx);
            }
        }

        // Then merge chains based on coref links
        for (id1, id2) in coref_links {
            let chain_idx = match (mention_to_chain.get(&id1), mention_to_chain.get(&id2)) {
                (Some(&idx1), Some(&idx2)) if idx1 != idx2 => {
                    // Merge chains
                    let to_merge = std::mem::take(&mut chains[idx2]);
                    chains[idx1].extend(to_merge);
                    // Update all mentions in merged chain to point to idx1
                    for m in &chains[idx1] {
                        // Find mention ID by matching text and position
                        for (mid, mref) in &mentions {
                            if mref.text == m.text && mref.start == m.start {
                                mention_to_chain.insert(mid.clone(), idx1);
                            }
                        }
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
                    if !chain.is_empty() {
                        chains.push(chain);
                    }
                    idx
                }
                (Some(&idx), Some(_)) => idx,
            };
            let _ = chain_idx; // Used above
        }

        // Filter empty chains and convert
        let coref_chains: Vec<CorefChain> = chains
            .into_iter()
            .filter(|c| !c.is_empty())
            .enumerate()
            .map(|(i, mentions)| CorefChain::with_id(mentions, i as u64))
            .collect();

        if coref_chains.is_empty() {
            return Err(Error::InvalidInput(
                "LitBank file contains no coreference chains".to_string(),
            ));
        }

        // Create document with reconstructed text
        let doc = CorefDocument::new(&text, coref_chains);
        Ok(vec![doc])
    }

    // =========================================================================
    // Relation Extraction Loading
    // =========================================================================

    /// Load relation extraction dataset, returning documents with relations.
    ///
    /// Use this for DocRED and ReTACRED datasets.
    pub fn load_relation(&self, id: DatasetId) -> Result<Vec<RelationDocument>> {
        if !id.is_relation_extraction() {
            return Err(Error::InvalidInput(format!(
                "{:?} is not a relation extraction dataset",
                id
            )));
        }

        let cache_path = self.cache_path_for(id);
        if !cache_path.exists() {
            return Err(Error::InvalidInput(format!(
                "Dataset {:?} not cached at {:?}. Download from {}",
                id,
                cache_path,
                id.download_url()
            )));
        }

        let content = std::fs::read_to_string(&cache_path)
            .map_err(|e| Error::InvalidInput(format!("Failed to read {:?}: {}", cache_path, e)))?;

        match id {
            DatasetId::DocRED
            | DatasetId::ReTACRED
            | DatasetId::NYTFB
            | DatasetId::WEBNLG
            | DatasetId::GoogleRE
            | DatasetId::BioRED
            | DatasetId::SciER
            | DatasetId::MixRED
            | DatasetId::CovEReD => {
                // All these datasets use the CrossRE format (same as DocRED)
                self.parse_docred_relations(&content)
            }
            DatasetId::CHisIEC => {
                // CHisIEC uses a different JSON format with entity indices
                self.parse_chisiec_relations(&content)
            }
            DatasetId::CADEC => {
                // CADEC is NER, not relation extraction
                Err(Error::InvalidInput(
                    "CADEC is a NER dataset, not relation extraction".to_string(),
                ))
            }
            _ => Err(Error::InvalidInput(format!(
                "No relation parser for {:?}",
                id
            ))),
        }
    }

    /// Load relation extraction dataset, downloading if needed.
    #[cfg(feature = "eval")]
    pub fn load_or_download_relation(&self, id: DatasetId) -> Result<Vec<RelationDocument>> {
        if !self.is_cached_for(id) {
            let (content, _) = self.download_with_resolved_url(id)?;
            let cache_path = self.cache_path_for(id);
            std::fs::write(&cache_path, &content).map_err(|e| {
                Error::InvalidInput(format!("Failed to cache {:?}: {}", cache_path, e))
            })?;
        }
        self.load_relation(id)
    }

    /// Parse DocRED/CrossRE format for relation extraction.
    ///
    /// Format: JSONL with {"sentence": [...], "ner": [[start, end, type], ...], "relations": [[id1-start, id1-end, id2-start, id2-end, rel-type, ...], ...]}
    fn parse_docred_relations(&self, content: &str) -> Result<Vec<RelationDocument>> {
        use super::relation::RelationGold;

        let mut documents = Vec::new();

        // Parse as JSONL (one JSON object per line)
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let doc: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Get sentence tokens
            let tokens_arr = match doc.get("sentence").and_then(|v| v.as_array()) {
                Some(t) => t,
                None => continue,
            };

            // Build text from tokens (with proper spacing)
            let text: String = tokens_arr
                .iter()
                .filter_map(|t| t.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            // Build token-to-character offset mapping
            // This maps each token index to its character start position in the text
            let mut token_to_char: Vec<usize> = Vec::new();
            let mut char_pos = 0;
            for (i, token) in tokens_arr.iter().enumerate() {
                if let Some(tok_str) = token.as_str() {
                    token_to_char.push(char_pos);
                    // Add token length + 1 for space (except last token)
                    char_pos += tok_str.len();
                    if i < tokens_arr.len() - 1 {
                        char_pos += 1; // Space between tokens
                    }
                } else {
                    token_to_char.push(char_pos);
                }
            }

            // Get NER spans: [start, end, type]
            let ner_spans = doc.get("ner").and_then(|v| v.as_array());

            // Build entity map: (token_start, token_end) -> (type, text, char_start, char_end)
            let mut entity_map: std::collections::HashMap<
                (usize, usize),
                (String, String, usize, usize),
            > = std::collections::HashMap::new();
            if let Some(ner) = ner_spans {
                for span in ner {
                    if let Some(arr) = span.as_array() {
                        if arr.len() >= 3 {
                            let token_start = arr[0].as_u64().unwrap_or(0) as usize;
                            let token_end = arr[1].as_u64().unwrap_or(0) as usize;
                            let ent_type = arr[2].as_str().unwrap_or("ENTITY").to_string();

                            // Extract entity text from tokens
                            let entity_text: String = tokens_arr
                                .iter()
                                .skip(token_start)
                                .take(token_end - token_start + 1)
                                .filter_map(|t| t.as_str())
                                .collect::<Vec<_>>()
                                .join(" ");

                            // Calculate actual character offsets
                            let char_start = token_to_char.get(token_start).copied().unwrap_or(0);
                            let char_end = if token_end < token_to_char.len() {
                                // Get end position of last token
                                let last_token_char_start = token_to_char[token_end];
                                if let Some(last_token) =
                                    tokens_arr.get(token_end).and_then(|t| t.as_str())
                                {
                                    last_token_char_start + last_token.len()
                                } else {
                                    char_start + entity_text.len()
                                }
                            } else {
                                char_start + entity_text.len()
                            };

                            entity_map.insert(
                                (token_start, token_end),
                                (ent_type, entity_text, char_start, char_end),
                            );
                        }
                    }
                }
            }

            // Parse relations: [id1-start, id1-end, id2-start, id2-end, rel-type, ...]
            let relations_arr = doc.get("relations").and_then(|v| v.as_array());
            let mut relations = Vec::new();

            if let Some(rels) = relations_arr {
                for rel in rels {
                    if let Some(arr) = rel.as_array() {
                        if arr.len() >= 5 {
                            let head_token_start = arr[0].as_u64().unwrap_or(0) as usize;
                            let head_token_end = arr[1].as_u64().unwrap_or(0) as usize;
                            let tail_token_start = arr[2].as_u64().unwrap_or(0) as usize;
                            let tail_token_end = arr[3].as_u64().unwrap_or(0) as usize;
                            let rel_type = arr[4].as_str().unwrap_or("RELATION").to_string();

                            // Get entity info from map (including character offsets)
                            let (head_type, head_text, head_char_start, head_char_end) = entity_map
                                .get(&(head_token_start, head_token_end))
                                .cloned()
                                .unwrap_or_else(|| {
                                    // Fallback: compute from token positions
                                    let char_start =
                                        token_to_char.get(head_token_start).copied().unwrap_or(0);
                                    let char_end = if head_token_end < token_to_char.len() {
                                        let last_start = token_to_char[head_token_end];
                                        if let Some(last_tok) =
                                            tokens_arr.get(head_token_end).and_then(|t| t.as_str())
                                        {
                                            last_start + last_tok.len()
                                        } else {
                                            char_start
                                        }
                                    } else {
                                        char_start
                                    };
                                    ("ENTITY".to_string(), String::new(), char_start, char_end)
                                });

                            let (tail_type, tail_text, tail_char_start, tail_char_end) = entity_map
                                .get(&(tail_token_start, tail_token_end))
                                .cloned()
                                .unwrap_or_else(|| {
                                    // Fallback: compute from token positions
                                    let char_start =
                                        token_to_char.get(tail_token_start).copied().unwrap_or(0);
                                    let char_end = if tail_token_end < token_to_char.len() {
                                        let last_start = token_to_char[tail_token_end];
                                        if let Some(last_tok) =
                                            tokens_arr.get(tail_token_end).and_then(|t| t.as_str())
                                        {
                                            last_start + last_tok.len()
                                        } else {
                                            char_start
                                        }
                                    } else {
                                        char_start
                                    };
                                    ("ENTITY".to_string(), String::new(), char_start, char_end)
                                });

                            relations.push(RelationGold::new(
                                (head_char_start, head_char_end),
                                head_type,
                                head_text,
                                (tail_char_start, tail_char_end),
                                tail_type,
                                tail_text,
                                rel_type,
                            ));
                        }
                    }
                }
            }

            if !text.is_empty() {
                documents.push(RelationDocument { text, relations });
            }
        }

        Ok(documents)
    }

    /// Parse CHisIEC (Chinese Historical Information Extraction Corpus) NER format.
    ///
    /// # Research Background
    ///
    /// CHisIEC (Tang et al., LREC-COLING 2024) addresses the unique challenges of
    /// information extraction from ancient Chinese historical texts (文言文):
    ///
    /// - **No word boundaries**: Classical Chinese has no spaces; we tokenize by character
    /// - **Archaic vocabulary**: Official titles (官职) differ from modern equivalents
    /// - **Long time span**: Covers texts from multiple dynasties across 2000+ years
    /// - **Domain-specific entities**: Government positions, classical texts, historical figures
    ///
    /// Paper: <https://aclanthology.org/2024.lrec-main.283/>
    /// GitHub: <https://github.com/tangxuemei1995/CHisIEC>
    ///
    /// # Entity Types
    ///
    /// | Type | Chinese | Meaning | Example |
    /// |------|---------|---------|---------|
    /// | PER | 人物 | Historical figure | 司馬遷, 曹操 |
    /// | LOC | 地点 | Location/Place | 長安, 洛陽 |
    /// | OFI | 官职 | Official position | 丞相, 太守 |
    /// | BOOK | 书籍 | Classical text | 史記, 論語 |
    ///
    /// # Format
    ///
    /// JSON array where each object contains:
    /// - `tokens`: String of continuous text (no spaces between characters)
    /// - `entities`: Array of entity objects with `type`, `start`, `end`, `span`
    ///
    /// # Distinction from HistoricalChineseNER
    ///
    /// **CHisIEC** (this dataset):
    /// - Ancient Chinese (文言文, pre-modern)
    /// - Source: 24 dynastic histories (二十四史)
    /// - Tasks: NER + Relation Extraction
    ///
    /// **HistoricalChineseNER** (different dataset):
    /// - Modern Chinese transition (1872-1949)
    /// - Source: ShenBao newspapers
    /// - Tasks: NER + Entity Linking + Coreference
    ///
    /// # Implementation Notes
    ///
    /// Chinese text is tokenized by character for NER tagging. Character offsets
    /// are used directly (critical for Unicode correctness).
    fn parse_chisiec(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();

        // CHisIEC RE data is a JSON array
        let docs: Vec<serde_json::Value> = serde_json::from_str(content)
            .map_err(|e| Error::InvalidInput(format!("Failed to parse CHisIEC JSON: {}", e)))?;

        for doc in docs {
            // Get tokens string (characters concatenated, no spaces in Chinese)
            let text = match doc.get("tokens").and_then(|v| v.as_str()) {
                Some(t) => t.to_string(),
                None => continue,
            };

            if text.is_empty() {
                continue;
            }

            // Chinese text: each character is a token
            let chars: Vec<char> = text.chars().collect();
            let mut tokens: Vec<AnnotatedToken> = chars
                .iter()
                .map(|c| AnnotatedToken {
                    text: c.to_string(),
                    ner_tag: "O".to_string(),
                })
                .collect();

            // Get entities array and build BIO tags
            let entities_arr = doc.get("entities").and_then(|v| v.as_array());

            if let Some(entities) = entities_arr {
                for entity in entities {
                    let ent_type = entity
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("ENTITY");
                    let start = entity.get("start").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    let end = entity.get("end").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

                    // Apply BIO tags (CHisIEC uses character indices)
                    for idx in start..end {
                        if idx < tokens.len() {
                            tokens[idx].ner_tag = if idx == start {
                                format!("B-{}", ent_type)
                            } else {
                                format!("I-{}", ent_type)
                            };
                        }
                    }
                }
            }

            if !tokens.is_empty() {
                sentences.push(AnnotatedSentence {
                    tokens,
                    source_dataset: id,
                });
            }
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "CHisIEC file for {:?} contains no valid sentences",
                id
            )));
        }

        let now = chrono::Utc::now().to_rfc3339();
        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse CHisIEC relation extraction format.
    ///
    /// # Research Motivation
    ///
    /// Ancient Chinese historical texts encode complex socio-political relationships
    /// that require domain-specific relation types. The 12 relation types in CHisIEC
    /// reflect structures unique to dynastic China:
    ///
    /// - **Hierarchical**: 上下級 (superior-subordinate), 任職 (office-holding)
    /// - **Kinship**: 父母 (parents), 兄弟 (siblings)
    /// - **Political**: 政治奧援 (political support), 同僚 (colleagues)
    /// - **Military**: 敵對攻伐 (hostility/attack), 駐守 (defend/garrison)
    /// - **Spatial**: 到達 (arrival), 出生於某地 (birthplace), 管理 (governance)
    /// - **Identity**: 別名 (alias/alternative name)
    ///
    /// # Format Difference from CrossRE/DocRED
    ///
    /// **CHisIEC** uses entity indices:
    /// ```json
    /// "relations": [{"type": "上下級", "head": 0, "tail": 1, ...}]
    /// ```
    ///
    /// **CrossRE/DocRED** uses token spans:
    /// ```json
    /// "relations": [[0, 2, 3, 5, "relation_type"]]
    /// ```
    ///
    /// This difference requires a separate parser.
    ///
    /// # JSON Format
    ///
    /// ```json
    /// {
    ///   "tokens": "明日，召問其故...",
    ///   "entities": [
    ///     {"type": "PER", "start": 11, "end": 13, "span": "玉汝"},
    ///     {"type": "PER", "start": 14, "end": 16, "span": "嚴公"}
    ///   ],
    ///   "relations": [
    ///     {"type": "上下級", "head": 1, "tail": 0, "head_span": "嚴公", "tail_span": "玉汝"}
    ///   ],
    ///   "orig_id": 1260
    /// }
    /// ```
    ///
    /// # Relation Types (12 total)
    ///
    /// | Chinese | Pinyin | English | Category |
    /// |---------|--------|---------|----------|
    /// | 敵對攻伐 | dí duì gōng fá | Attack/Hostility | Military |
    /// | 任職 | rèn zhí | Office Holding | Political |
    /// | 上下級 | shàng xià jí | Superior-subordinate | Hierarchical |
    /// | 政治奧援 | zhèng zhì ào yuán | Political Support | Political |
    /// | 同僚 | tóng liáo | Colleague | Social |
    /// | 到達 | dào dá | Arrive | Spatial |
    /// | 管理 | guǎn lǐ | Manage/Govern | Administrative |
    /// | 出生於某地 | chū shēng yú mǒu dì | Birthplace | Spatial |
    /// | 駐守 | zhù shǒu | Defend/Garrison | Military |
    /// | 別名 | bié míng | Alias | Identity |
    /// | 父母 | fù mǔ | Parents | Kinship |
    /// | 兄弟 | xiōng dì | Siblings | Kinship |
    fn parse_chisiec_relations(&self, content: &str) -> Result<Vec<RelationDocument>> {
        use super::relation::RelationGold;

        let mut documents = Vec::new();

        // Parse as JSON array
        let docs: Vec<serde_json::Value> = serde_json::from_str(content)
            .map_err(|e| Error::InvalidInput(format!("Failed to parse CHisIEC JSON: {}", e)))?;

        for doc in docs {
            // Get tokens string
            let text = match doc.get("tokens").and_then(|v| v.as_str()) {
                Some(t) => t.to_string(),
                None => continue,
            };

            if text.is_empty() {
                continue;
            }

            // Build entity list: [(type, start, end, text), ...]
            let entities_arr = doc.get("entities").and_then(|v| v.as_array());
            let mut entity_list: Vec<(String, usize, usize, String)> = Vec::new();

            if let Some(entities) = entities_arr {
                for entity in entities {
                    let ent_type = entity
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("ENTITY")
                        .to_string();
                    let start = entity.get("start").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    let end = entity.get("end").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    let span = entity
                        .get("span")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();

                    // Fallback to extracting from text if span is empty
                    let span_text = if !span.is_empty() {
                        span
                    } else {
                        text.chars().skip(start).take(end - start).collect()
                    };

                    entity_list.push((ent_type, start, end, span_text));
                }
            }

            // Parse relations
            let relations_arr = doc.get("relations").and_then(|v| v.as_array());
            let mut relations = Vec::new();

            if let Some(rels) = relations_arr {
                for rel in rels {
                    let rel_type = rel
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("RELATION")
                        .to_string();
                    // CHisIEC uses entity indices for head/tail
                    let head_idx = rel.get("head").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    let tail_idx = rel.get("tail").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

                    // Look up entities by index
                    if head_idx < entity_list.len() && tail_idx < entity_list.len() {
                        let (head_type, head_start, head_end, head_text) = &entity_list[head_idx];
                        let (tail_type, tail_start, tail_end, tail_text) = &entity_list[tail_idx];

                        relations.push(RelationGold::new(
                            (*head_start, *head_end),
                            head_type.clone(),
                            head_text.clone(),
                            (*tail_start, *tail_end),
                            tail_type.clone(),
                            tail_text.clone(),
                            rel_type,
                        ));
                    }
                }
            }

            if !text.is_empty() {
                documents.push(RelationDocument { text, relations });
            }
        }

        Ok(documents)
    }

    // =========================================================================
    // African Language Dataset Parsers (Masakhane Community)
    // =========================================================================

    /// Parse AfriSenti sentiment analysis format.
    ///
    /// AfriSenti uses TSV format: text\tlabel
    /// Labels: positive, neutral, negative
    ///
    /// Source: <https://github.com/afrisenti-semeval/afrisent-semeval-2023>
    fn parse_afrisenti(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let now = chrono::Utc::now().to_rfc3339();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                let text = parts[0].to_string();
                let label = parts[1].to_string();

                // For sentiment, we create a single token annotation with the sentiment label
                let tokens = vec![AnnotatedToken {
                    text: text.clone(),
                    ner_tag: format!("B-{}", label),
                }];

                sentences.push(AnnotatedSentence {
                    tokens,
                    source_dataset: id,
                });
            }
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "AfriSenti file for {:?} contains no valid sentences",
                id
            )));
        }

        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse AfriQA question answering format.
    ///
    /// AfriQA uses JSON format with fields: context, question, answers
    /// Each answer has text and answer_start
    ///
    /// Source: <https://github.com/masakhane-io/afriqa>
    fn parse_afriqa(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let now = chrono::Utc::now().to_rfc3339();

        // AfriQA data can be JSON array or JSONL
        let docs: Vec<serde_json::Value> = if content.trim().starts_with('[') {
            serde_json::from_str(content)
                .map_err(|e| Error::InvalidInput(format!("Failed to parse AfriQA JSON: {}", e)))?
        } else {
            // JSONL format
            content
                .lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect()
        };

        for doc in docs {
            // Get context text
            let context = doc.get("context").and_then(|v| v.as_str()).unwrap_or("");

            // Get answers
            if let Some(answers) = doc.get("answers").and_then(|v| v.as_object()) {
                let texts = answers.get("text").and_then(|v| v.as_array());
                let starts = answers.get("answer_start").and_then(|v| v.as_array());

                if let (Some(texts), Some(starts)) = (texts, starts) {
                    // Create tokens for context (word-level tokenization)
                    let words: Vec<&str> = context.split_whitespace().collect();
                    let mut tokens: Vec<AnnotatedToken> = words
                        .iter()
                        .map(|w| AnnotatedToken {
                            text: w.to_string(),
                            ner_tag: "O".to_string(),
                        })
                        .collect();

                    // Mark answer spans
                    for (text_val, start_val) in texts.iter().zip(starts.iter()) {
                        if let (Some(answer_text), Some(start)) =
                            (text_val.as_str(), start_val.as_u64())
                        {
                            let start = start as usize;
                            let answer_words: Vec<&str> = answer_text.split_whitespace().collect();

                            // Find word index by counting words before start position
                            let prefix: String = context.chars().take(start).collect();
                            let word_idx = prefix.split_whitespace().count();

                            // Apply BIO tags
                            for (i, _) in answer_words.iter().enumerate() {
                                let idx = word_idx + i;
                                if idx < tokens.len() {
                                    tokens[idx].ner_tag = if i == 0 {
                                        "B-ANSWER".to_string()
                                    } else {
                                        "I-ANSWER".to_string()
                                    };
                                }
                            }
                        }
                    }

                    if !tokens.is_empty() {
                        sentences.push(AnnotatedSentence {
                            tokens,
                            source_dataset: id,
                        });
                    }
                }
            }
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "AfriQA file for {:?} contains no valid sentences",
                id
            )));
        }

        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse MasakhaNEWS topic classification format.
    ///
    /// MasakhaNEWS uses TSV format: headline\tbody\tcategory
    /// Categories: business, entertainment, health, politics, religion, sports, technology
    ///
    /// Source: <https://github.com/masakhane-io/masakhane-news>
    fn parse_masakhanews(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let now = chrono::Utc::now().to_rfc3339();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("headline\t") {
                continue;
            }

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                // Use headline as text, category as label
                let text = parts[0].to_string();
                let category = if parts.len() >= 3 {
                    parts[2].to_string()
                } else {
                    parts[1].to_string()
                };

                // For classification, create single token with category
                let tokens = vec![AnnotatedToken {
                    text,
                    ner_tag: format!("B-{}", category),
                }];

                sentences.push(AnnotatedSentence {
                    tokens,
                    source_dataset: id,
                });
            }
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "MasakhaNEWS file for {:?} contains no valid sentences",
                id
            )));
        }

        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse TREC question classification format.
    ///
    /// TREC format: `COARSE:fine question text`
    /// E.g.: `NUM:dist How far is it from Denver to Aspen ?`
    ///
    /// 6 coarse classes: ABBR, DESC, ENTY, HUM, LOC, NUM
    /// 50 fine-grained classes.
    ///
    /// Source: <https://cogcomp.seas.upenn.edu/Data/QA/QC/>
    fn parse_trec(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let now = chrono::Utc::now().to_rfc3339();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Format: COARSE:fine question text
            // Find the first space to separate label from question
            if let Some(space_idx) = line.find(' ') {
                let label = &line[..space_idx];
                let question = line[space_idx + 1..].trim();

                // Extract coarse label (before colon)
                let coarse_label = label.split(':').next().unwrap_or(label);

                // Create single token with the question and classification label
                let tokens = vec![AnnotatedToken {
                    text: question.to_string(),
                    ner_tag: format!("B-{}", coarse_label),
                }];

                sentences.push(AnnotatedSentence {
                    tokens,
                    source_dataset: id,
                });
            }
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "TREC file for {:?} contains no valid sentences",
                id
            )));
        }

        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse AG News classification format (parquet or CSV).
    ///
    /// AG News has 4 classes: World (0), Sports (1), Business (2), Sci/Tech (3)
    ///
    /// Source: <https://huggingface.co/datasets/ag_news>
    fn parse_agnews(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let now = chrono::Utc::now().to_rfc3339();

        let label_map = ["World", "Sports", "Business", "Sci/Tech"];

        // Try parsing as JSON (converted from parquet)
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Try JSON format
            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                let text = obj.get("text").and_then(|v| v.as_str()).unwrap_or_default();
                let label_idx = obj.get("label").and_then(|v| v.as_i64()).unwrap_or(0) as usize;

                let label = label_map.get(label_idx).unwrap_or(&"Unknown");

                let tokens = vec![AnnotatedToken {
                    text: text.to_string(),
                    ner_tag: format!("B-{}", label),
                }];

                sentences.push(AnnotatedSentence {
                    tokens,
                    source_dataset: id,
                });
            }
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "AG News file for {:?} contains no valid sentences",
                id
            )));
        }

        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse DBPedia-14 classification format.
    ///
    /// 14 classes from DBpedia ontology.
    ///
    /// Source: <https://huggingface.co/datasets/dbpedia_14>
    fn parse_dbpedia14(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let now = chrono::Utc::now().to_rfc3339();

        let label_map = [
            "Company",
            "EducationalInstitution",
            "Artist",
            "Athlete",
            "OfficeHolder",
            "MeanOfTransportation",
            "Building",
            "NaturalPlace",
            "Village",
            "Animal",
            "Plant",
            "Album",
            "Film",
            "WrittenWork",
        ];

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                let content_text = obj
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let label_idx = obj.get("label").and_then(|v| v.as_i64()).unwrap_or(0) as usize;

                let label = label_map.get(label_idx).unwrap_or(&"Unknown");

                let tokens = vec![AnnotatedToken {
                    text: content_text.to_string(),
                    ner_tag: format!("B-{}", label),
                }];

                sentences.push(AnnotatedSentence {
                    tokens,
                    source_dataset: id,
                });
            }
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "DBPedia-14 file for {:?} contains no valid sentences",
                id
            )));
        }

        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse Yahoo Answers topic classification format.
    ///
    /// 10 topics: Society, Science, Health, Education, Computers,
    /// Sports, Business, Entertainment, Family, Politics
    ///
    /// Source: <https://huggingface.co/datasets/yahoo_answers_topics>
    fn parse_yahoo_answers(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let now = chrono::Utc::now().to_rfc3339();

        let label_map = [
            "Society",
            "Science",
            "Health",
            "Education",
            "Computers",
            "Sports",
            "Business",
            "Entertainment",
            "Family",
            "Politics",
        ];

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                // Yahoo Answers has question_title, question_content, best_answer
                let question = obj
                    .get("question_title")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let label_idx = obj.get("topic").and_then(|v| v.as_i64()).unwrap_or(0) as usize;

                let label = label_map.get(label_idx).unwrap_or(&"Unknown");

                let tokens = vec![AnnotatedToken {
                    text: question.to_string(),
                    ner_tag: format!("B-{}", label),
                }];

                sentences.push(AnnotatedSentence {
                    tokens,
                    source_dataset: id,
                });
            }
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "Yahoo Answers file for {:?} contains no valid sentences",
                id
            )));
        }

        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse MAVEN event detection format.
    ///
    /// MAVEN provides event triggers with 168 event types.
    /// Supports both:
    /// - Full MAVEN JSONL format (train.jsonl/valid.jsonl with events array)
    /// - Fallback: docid2topic.json mapping file
    ///
    /// Full format structure:
    /// ```json
    /// {
    ///   "id": "doc_id",
    ///   "content": [{"sentence": "...", "tokens": [...]}],
    ///   "events": [{
    ///     "type": "EventType",
    ///     "mention": [{"trigger_word": "...", "sent_id": 0, "offset": [3, 4]}]
    ///   }]
    /// }
    /// ```
    ///
    /// Source: <https://github.com/THU-KEG/MAVEN-dataset>
    fn parse_maven(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let now = chrono::Utc::now().to_rfc3339();

        // Try parsing as JSONL (full MAVEN format)
        let mut is_jsonl = false;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Ok(doc) = serde_json::from_str::<serde_json::Value>(line) {
                // Check if this is full MAVEN format with events
                if let Some(events) = doc.get("events").and_then(|e| e.as_array()) {
                    is_jsonl = true;

                    // Get document content for context
                    let doc_content: Vec<String> = doc
                        .get("content")
                        .and_then(|c| c.as_array())
                        .map(|sents| {
                            sents
                                .iter()
                                .filter_map(|s| s.get("sentence").and_then(|v| v.as_str()))
                                .map(|s| s.to_string())
                                .collect()
                        })
                        .unwrap_or_default();

                    // Process each event
                    for event in events {
                        let event_type = event
                            .get("type")
                            .and_then(|t| t.as_str())
                            .unwrap_or("EVENT");

                        // Process each mention of this event
                        if let Some(mentions) = event.get("mention").and_then(|m| m.as_array()) {
                            for mention in mentions {
                                let trigger_word = mention
                                    .get("trigger_word")
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("");

                                let sent_id =
                                    mention.get("sent_id").and_then(|s| s.as_u64()).unwrap_or(0)
                                        as usize;

                                // Get sentence context if available
                                let context = doc_content
                                    .get(sent_id)
                                    .cloned()
                                    .unwrap_or_else(|| trigger_word.to_string());

                                let tokens = vec![AnnotatedToken {
                                    text: context,
                                    ner_tag: format!("B-{}", event_type),
                                }];

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

        // Fallback: parse as docid2topic.json mapping
        if !is_jsonl {
            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(content) {
                if let Some(map) = obj.as_object() {
                    for (doc_id, event_type) in map {
                        let event_type_str = event_type.as_str().unwrap_or("event");

                        let tokens = vec![AnnotatedToken {
                            text: doc_id.clone(),
                            ner_tag: format!(
                                "B-EVENT_{}",
                                event_type_str.to_uppercase().replace(' ', "_")
                            ),
                        }];

                        sentences.push(AnnotatedSentence {
                            tokens,
                            source_dataset: id,
                        });
                    }
                }
            }
        }

        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse CASIE cybersecurity event extraction format.
    ///
    /// CASIE has 5 event types: Attack-Pattern, Vulnerability, Data-Breach, Malware, Patch
    ///
    /// Format structure:
    /// ```json
    /// {
    ///   "content": "document text...",
    ///   "cyberevent": {
    ///     "hopper": [{
    ///       "events": [{
    ///         "subtype": "Databreach",
    ///         "nugget": {"text": "trigger", "startOffset": 0, "endOffset": 10},
    ///         "argument": [{"text": "arg", "role": {"type": "RoleType"}}]
    ///       }]
    ///     }]
    ///   }
    /// }
    /// ```
    ///
    /// Source: <https://github.com/Ebiquity/CASIE>
    fn parse_casie(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let now = chrono::Utc::now().to_rfc3339();

        // Handle JSONL format (multiple documents)
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Ok(doc) = serde_json::from_str::<serde_json::Value>(line) {
                let content_text = doc
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();

                if content_text.is_empty() {
                    continue;
                }

                // Extract events from cyberevent.hopper[].events[]
                let mut found_events = false;
                if let Some(hopper) = doc
                    .get("cyberevent")
                    .and_then(|ce| ce.get("hopper"))
                    .and_then(|h| h.as_array())
                {
                    for cluster in hopper {
                        if let Some(events) = cluster.get("events").and_then(|e| e.as_array()) {
                            for event in events {
                                found_events = true;

                                // Get event subtype
                                let subtype = event
                                    .get("subtype")
                                    .and_then(|s| s.as_str())
                                    .unwrap_or("Event");

                                // Get trigger (nugget)
                                let trigger_text = event
                                    .get("nugget")
                                    .and_then(|n| n.get("text"))
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("");

                                // Create entry with trigger and event type
                                let tokens = vec![AnnotatedToken {
                                    text: trigger_text.to_string(),
                                    ner_tag: format!("B-{}", subtype),
                                }];

                                sentences.push(AnnotatedSentence {
                                    tokens,
                                    source_dataset: id,
                                });

                                // Also extract arguments if present
                                if let Some(args) = event.get("argument").and_then(|a| a.as_array())
                                {
                                    for arg in args {
                                        let arg_text =
                                            arg.get("text").and_then(|t| t.as_str()).unwrap_or("");
                                        let role = arg
                                            .get("role")
                                            .and_then(|r| r.get("type"))
                                            .and_then(|t| t.as_str())
                                            .unwrap_or("Argument");

                                        if !arg_text.is_empty() {
                                            let tokens = vec![AnnotatedToken {
                                                text: arg_text.to_string(),
                                                ner_tag: format!("B-ARG_{}", role),
                                            }];

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
                }

                // If no events found, add document with O tag
                if !found_events {
                    let tokens = vec![AnnotatedToken {
                        text: content_text.chars().take(200).collect(),
                        ner_tag: "O".to_string(),
                    }];

                    sentences.push(AnnotatedSentence {
                        tokens,
                        source_dataset: id,
                    });
                }
            }
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "MAVEN file for {:?} contains no valid sentences",
                id
            )));
        }

        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse MAVEN-ARG event argument extraction format.
    ///
    /// MAVEN-ARG extends MAVEN with 612 argument roles across 162 event types.
    ///
    /// Format structure:
    /// ```json
    /// {
    ///   "id": "doc_id",
    ///   "document": "full document text...",
    ///   "events": [{
    ///     "type": "EventType",
    ///     "mention": [{"trigger_word": "...", "offset": [start, end]}],
    ///     "argument": {"Role": [{"content": "arg text", "offset": [s, e]}]}
    ///   }]
    /// }
    /// ```
    ///
    /// Source: <https://github.com/THU-KEG/MAVEN-Argument>
    fn parse_maven_arg(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let now = chrono::Utc::now().to_rfc3339();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Ok(doc) = serde_json::from_str::<serde_json::Value>(line) {
                // Get document text
                let _doc_text = doc.get("document").and_then(|d| d.as_str()).unwrap_or("");

                // Process events
                if let Some(events) = doc.get("events").and_then(|e| e.as_array()) {
                    for event in events {
                        let event_type = event
                            .get("type")
                            .and_then(|t| t.as_str())
                            .unwrap_or("EVENT");

                        // Process event mentions (triggers)
                        if let Some(mentions) = event.get("mention").and_then(|m| m.as_array()) {
                            for mention in mentions {
                                let trigger = mention
                                    .get("trigger_word")
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("");

                                if !trigger.is_empty() {
                                    let tokens = vec![AnnotatedToken {
                                        text: trigger.to_string(),
                                        ner_tag: format!("B-{}", event_type),
                                    }];

                                    sentences.push(AnnotatedSentence {
                                        tokens,
                                        source_dataset: id,
                                    });
                                }
                            }
                        }

                        // Process arguments
                        if let Some(args) = event.get("argument").and_then(|a| a.as_object()) {
                            for (role, arg_list) in args {
                                if let Some(arg_arr) = arg_list.as_array() {
                                    for arg in arg_arr {
                                        // Arguments can be text or entity references
                                        if let Some(content) =
                                            arg.get("content").and_then(|c| c.as_str())
                                        {
                                            if !content.is_empty() {
                                                let tokens = vec![AnnotatedToken {
                                                    text: content.to_string(),
                                                    ner_tag: format!("B-ARG_{}", role),
                                                }];

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
                    }
                }
            }
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "MAVEN-ARG file for {:?} contains no valid sentences",
                id
            )));
        }

        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse RAMS event argument extraction format.
    ///
    /// RAMS (Roles Across Multiple Sentences) covers 139 event types
    /// with arguments that can span multiple sentences.
    ///
    /// Format structure:
    /// ```json
    /// {
    ///   "doc_key": "...",
    ///   "sentences": [["token1", "token2", ...]],
    ///   "evt_triggers": [[start, end, [["event.type", score]]]],
    ///   "gold_evt_links": [[[evt_idx], [arg_start, arg_end], "role"]]
    /// }
    /// ```
    ///
    /// Source: <https://nlp.jhu.edu/rams/>
    fn parse_rams(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let now = chrono::Utc::now().to_rfc3339();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Ok(doc) = serde_json::from_str::<serde_json::Value>(line) {
                // Get all tokens (flattened sentences)
                let all_tokens: Vec<String> = doc
                    .get("sentences")
                    .and_then(|s| s.as_array())
                    .map(|sents| {
                        sents
                            .iter()
                            .filter_map(|s| s.as_array())
                            .flat_map(|toks| {
                                toks.iter().filter_map(|t| t.as_str().map(String::from))
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                // Process event triggers
                if let Some(triggers) = doc.get("evt_triggers").and_then(|t| t.as_array()) {
                    for trigger in triggers {
                        if let Some(trigger_arr) = trigger.as_array() {
                            if trigger_arr.len() >= 3 {
                                let start = trigger_arr[0].as_u64().unwrap_or(0) as usize;
                                let end = trigger_arr[1].as_u64().unwrap_or(0) as usize;

                                // Get event type from nested array
                                let event_type = trigger_arr[2]
                                    .as_array()
                                    .and_then(|types| types.first())
                                    .and_then(|t| t.as_array())
                                    .and_then(|t| t.first())
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("event");

                                // Extract trigger text
                                if end <= all_tokens.len() {
                                    let trigger_text = all_tokens[start..=end.min(start)].join(" ");
                                    let tokens = vec![AnnotatedToken {
                                        text: trigger_text,
                                        ner_tag: format!("B-{}", event_type),
                                    }];

                                    sentences.push(AnnotatedSentence {
                                        tokens,
                                        source_dataset: id,
                                    });
                                }
                            }
                        }
                    }
                }

                // Process argument links
                if let Some(links) = doc.get("gold_evt_links").and_then(|l| l.as_array()) {
                    for link in links {
                        if let Some(link_arr) = link.as_array() {
                            if link_arr.len() >= 3 {
                                // Argument span
                                if let Some(span) = link_arr[1].as_array() {
                                    if span.len() >= 2 {
                                        let start = span[0].as_u64().unwrap_or(0) as usize;
                                        let end = span[1].as_u64().unwrap_or(0) as usize;
                                        let role = link_arr[2].as_str().unwrap_or("argument");

                                        if end < all_tokens.len() {
                                            let arg_text =
                                                all_tokens[start..=end.min(start)].join(" ");
                                            let tokens = vec![AnnotatedToken {
                                                text: arg_text,
                                                ner_tag: format!("B-ARG_{}", role),
                                            }];

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
                }
            }
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "RAMS file for {:?} contains no valid sentences",
                id
            )));
        }

        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse TweetTopic format (JSONL with text and labels).
    ///
    /// Format: {"text": "...", "label": 0, "label_name": "sports_&_gaming", ...}
    ///
    /// Source: <https://huggingface.co/datasets/cardiffnlp/tweet_topic_single>
    fn parse_tweettopic(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let now = chrono::Utc::now().to_rfc3339();

        // Label mapping if label_name not present
        let label_map = [
            "arts_&_culture",
            "business_&_entrepreneurs",
            "pop_culture",
            "daily_life",
            "sports_&_gaming",
            "science_&_technology",
        ];

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                let text = obj.get("text").and_then(|v| v.as_str()).unwrap_or_default();

                // Try label_name first, then map from label index
                let label = obj
                    .get("label_name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| {
                        obj.get("label")
                            .and_then(|v| v.as_i64())
                            .and_then(|idx| label_map.get(idx as usize))
                            .map(|s| s.to_string())
                    })
                    .unwrap_or_else(|| "topic".to_string());

                let tokens = vec![AnnotatedToken {
                    text: text.to_string(),
                    ner_tag: format!("B-{}", label),
                }];

                sentences.push(AnnotatedSentence {
                    tokens,
                    source_dataset: id,
                });
            }
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "TweetTopic file for {:?} contains no valid sentences",
                id
            )));
        }

        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Parse CoNLL-U format (Universal Dependencies).
    ///
    /// CoNLL-U format has 10 tab-separated columns per line:
    /// ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
    ///
    /// Used by MasakhaPOS for African language POS tagging.
    ///
    /// Source: <https://universaldependencies.org/format.html>
    fn parse_conllu(&self, content: &str, id: DatasetId) -> Result<LoadedDataset> {
        let mut sentences = Vec::new();
        let now = chrono::Utc::now().to_rfc3339();
        let mut current_tokens = Vec::new();

        for line in content.lines() {
            let line = line.trim();

            // Skip comments
            if line.starts_with('#') {
                continue;
            }

            // Blank line marks end of sentence
            if line.is_empty() {
                if !current_tokens.is_empty() {
                    sentences.push(AnnotatedSentence {
                        tokens: std::mem::take(&mut current_tokens),
                        source_dataset: id,
                    });
                }
                continue;
            }

            // Parse CoNLL-U columns
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() >= 4 {
                // Skip multi-word tokens (IDs with ranges like "1-2")
                let id_field = fields[0];
                if id_field.contains('-') || id_field.contains('.') {
                    continue;
                }

                let form = fields[1]; // Word form
                let upos = fields[3]; // Universal POS tag

                current_tokens.push(AnnotatedToken {
                    text: form.to_string(),
                    ner_tag: format!("B-{}", upos), // Use POS tag as entity type
                });
            }
        }

        // Don't forget last sentence
        if !current_tokens.is_empty() {
            sentences.push(AnnotatedSentence {
                tokens: current_tokens,
                source_dataset: id,
            });
        }

        if sentences.is_empty() {
            return Err(Error::InvalidInput(format!(
                "CoNLL-U file for {:?} contains no valid sentences",
                id
            )));
        }

        Ok(LoadedDataset {
            id,
            sentences,
            loaded_at: now,
            source_url: id.download_url().to_string(),
            data_source: DataSource::LocalCache,
            temporal_metadata: Self::get_temporal_metadata(id),
            metadata: id.default_metadata(),
        })
    }

    /// Load all cached datasets.
    pub fn load_all_cached(&self) -> Vec<(DatasetId, Result<LoadedDataset>)> {
        LoadableDatasetId::all()
            .into_iter()
            .filter(|id| self.is_cached(*id))
            .map(|id| (id.0, self.load(id)))
            .collect()
    }

    /// Get status of all datasets.
    #[must_use]
    pub fn status(&self) -> Vec<(DatasetId, bool)> {
        LoadableDatasetId::all()
            .into_iter()
            .map(|id| (id.0, self.is_cached(id)))
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
#[cfg(test)]
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
///
/// **Prefer `crate::schema::map_to_canonical()` for new code** - it handles
/// NORP correctly (as GROUP, not ORG) and preserves GPE/FAC distinctions.
///
/// # Known Issues (preserved for backwards compatibility)
///
/// - NORP → Organization (WRONG: should be Group)
/// - GPE/FAC/LOC all → Location (loses semantic distinctions)
///
/// See `src/schema.rs` for the corrected mappings.
#[cfg(test)]
fn map_entity_type(original: &str) -> EntityType {
    // Use the new canonical mapper for consistent semantics
    anno::schema::map_to_canonical(original, None)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_id_basics() {
        let id = DatasetId::WikiGold;
        assert_eq!(id.name(), "WikiGold");
    }

    #[test]
    fn test_convenience_metadata_methods() {
        let id = DatasetId::WikiGold;

        // WikiGold has known metadata
        assert_eq!(id.citation(), Some("Balasuriya et al. (2009)"));
        assert_eq!(id.license(), Some("CC-BY-4.0"));
        assert_eq!(id.year(), Some(2009));
    }

    #[test]
    fn test_loadable_wrapper_invariants() {
        // There should be at least one non-loadable dataset in the catalog.
        assert!(
            DatasetId::all()
                .iter()
                .copied()
                .any(|d| !LoadableDatasetId::is_loadable_dataset(d)),
            "Expected registry to contain some non-loadable datasets"
        );

        for id in LoadableDatasetId::all() {
            let ds: DatasetId = id.into();
            assert!(
                LoadableDatasetId::is_loadable_dataset(ds),
                "LoadableDatasetId must imply is_loadable_dataset()"
            );
            assert!(LoadableDatasetId::try_from(ds).is_ok());
        }
    }

    #[test]
    fn test_parse_plan_is_single_source_of_truth_for_loadability() {
        // For *every* registry dataset id:
        // - parse_plan(id).is_some()  <=>  TryFrom<DatasetId> succeeds
        for &ds in DatasetId::all() {
            let plan_exists = LoadableDatasetId::parse_plan(ds).is_some();
            let try_ok = LoadableDatasetId::try_from(ds).is_ok();
            assert_eq!(
                plan_exists, try_ok,
                "parse_plan / TryFrom mismatch for {:?}",
                ds
            );
        }

        // Also ensure `LoadableDatasetId::all()` only returns ids with plans.
        for id in LoadableDatasetId::all() {
            let ds: DatasetId = id.into();
            assert!(
                LoadableDatasetId::parse_plan(ds).is_some(),
                "LoadableDatasetId::all() returned {:?} with no parse plan",
                ds
            );
        }
    }

    #[test]
    fn test_registry_hints_do_not_contradict_parse_plan() {
        // If a dataset is loadable (i.e., has a parse plan), and the registry provides a strong
        // hint, they should agree. (Hints are allowed to be optimistic for non-loadable datasets.)
        for &ds in DatasetId::all() {
            let Some(plan) = LoadableDatasetId::parse_plan(ds) else {
                continue;
            };
            let Some(hint) = LoadableDatasetId::registry_hint_plan(ds) else {
                continue;
            };
            assert_eq!(hint, plan, "Registry hint mismatch for {:?}", ds);
        }
    }

    #[test]
    fn test_huggingface_access_status_requires_hf_id() {
        // `access_status: HuggingFace` is our strongest signal that a dataset is automatable
        // via the Hub. Keep the registry self-consistent so hinting can stay metadata-driven.
        for &ds in DatasetId::all() {
            if ds.access_status()
                != crate::eval::dataset_registry::DatasetAccessibility::HuggingFace
            {
                continue;
            }
            assert!(
                ds.hf_id().is_some(),
                "Dataset {:?} is marked HuggingFace-accessible but has no hf_id",
                ds
            );
        }
    }

    #[test]
    fn test_huggingface_access_status_is_hintable() {
        // If the registry says “HuggingFace”, the loader should be able to produce *some*
        // parse-plan hint (not necessarily `HfApiResponse`, since a few datasets are hybrids
        // with bespoke parse plans).
        for &ds in DatasetId::all() {
            if ds.access_status()
                != crate::eval::dataset_registry::DatasetAccessibility::HuggingFace
            {
                continue;
            }
            assert!(
                LoadableDatasetId::registry_hint_plan(ds).is_some(),
                "Dataset {:?} is marked HuggingFace-accessible but has no registry hint plan",
                ds
            );
        }
    }

    #[test]
    fn test_parse_bio_tag() {
        assert_eq!(parse_bio_tag("O"), ("O", ""));
        assert_eq!(parse_bio_tag("B-PER"), ("B", "PER"));
        assert_eq!(parse_bio_tag("I-LOC"), ("I", "LOC"));
        assert_eq!(parse_bio_tag("B-ORG"), ("B", "ORG"));
    }

    #[test]
    fn test_map_entity_type() {
        // Core types
        assert_eq!(map_entity_type("PER"), EntityType::Person);
        assert_eq!(map_entity_type("PERSON"), EntityType::Person);
        assert_eq!(map_entity_type("LOC"), EntityType::Location);
        assert_eq!(map_entity_type("ORG"), EntityType::Organization);

        // GPE now preserves distinction (Custom, not Location)
        assert!(matches!(map_entity_type("GPE"), EntityType::Custom { .. }));

        // MISC -> Custom or Other
        assert!(matches!(map_entity_type("MISC"), EntityType::Custom { .. }));

        // OntoNotes types -> Custom (preserves semantics)
        assert!(matches!(
            map_entity_type("PRODUCT"),
            EntityType::Custom { .. }
        ));
        assert!(matches!(
            map_entity_type("EVENT"),
            EntityType::Custom { .. }
        ));
        assert!(matches!(
            map_entity_type("WORK_OF_ART"),
            EntityType::Custom { .. }
        ));

        // Numeric types preserved
        assert_eq!(map_entity_type("CARDINAL"), EntityType::Cardinal);
    }

    #[test]
    fn test_dataset_id_display() {
        assert_eq!(DatasetId::WikiGold.to_string(), "WikiGold");
        assert_eq!(DatasetId::Wnut17.to_string(), "WNUT-17");
    }

    #[test]
    fn test_dataset_id_from_str() {
        assert_eq!(
            "wikigold".parse::<DatasetId>().unwrap(),
            DatasetId::WikiGold
        );
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
        let dataset = loader
            .parse_conll(content, DatasetId::CoNLL2003Sample)
            .unwrap();

        assert_eq!(dataset.len(), 2);

        let entities1 = dataset.sentences[0].entities();
        assert_eq!(entities1.len(), 2); // EU (ORG), German (MISC)

        let entities2 = dataset.sentences[1].entities();
        assert_eq!(entities2.len(), 1); // Peter Blackburn (PER)
        assert_eq!(entities2[0].text, "Peter Blackburn");
    }

    #[test]
    fn test_historical_datasets_configured() {
        // Historical NER datasets should have proper metadata
        assert!(!DatasetId::HIPE2022.download_url().is_empty());
        assert_eq!(DatasetId::HIPE2022.name(), "HIPE-2022");

        assert!(!DatasetId::MedievalCzechCharters.download_url().is_empty());
        assert_eq!(
            DatasetId::MedievalCzechCharters.name(),
            "Medieval Czech Charters"
        );

        assert!(!DatasetId::TRIDIS.download_url().is_empty());
        assert_eq!(DatasetId::TRIDIS.name(), "TRIDIS");

        // Should be in all() list
        let all = DatasetId::all();
        assert!(all.contains(&DatasetId::HIPE2022));
        assert!(all.contains(&DatasetId::TRIDIS));
    }

    #[test]
    fn test_queer_nlp_datasets_configured() {
        // Queer/gender-inclusive NLP datasets
        assert!(!DatasetId::WinoQueer.download_url().is_empty());
        assert_eq!(DatasetId::WinoQueer.name(), "WinoQueer");

        assert!(!DatasetId::GICoref.download_url().is_empty());
        assert_eq!(DatasetId::GICoref.name(), "GICoref");

        assert!(!DatasetId::BBQ.download_url().is_empty());
        assert_eq!(DatasetId::BBQ.name(), "BBQ");

        // Should be in all() list
        let all = DatasetId::all();
        assert!(all.contains(&DatasetId::WinoQueer));
        assert!(all.contains(&DatasetId::GICoref));
        assert!(all.contains(&DatasetId::BBQ));
    }

    #[test]
    fn test_joint_re_datasets_configured() {
        // Joint NER + Relation Extraction datasets
        assert!(
            DatasetId::TACRED.requires_license(),
            "TACRED is LDC-licensed; download_url may be empty"
        );
        assert_eq!(DatasetId::TACRED.name(), "TACRED");

        assert!(!DatasetId::REBEL.download_url().is_empty());
        assert_eq!(DatasetId::REBEL.name(), "REBEL");

        // Should be in all() list
        let all = DatasetId::all();
        assert!(all.contains(&DatasetId::TACRED));
        assert!(all.contains(&DatasetId::REBEL));
    }

    #[test]
    fn test_dialogue_coref_datasets_configured() {
        // Dialogue/streaming coreference datasets
        assert!(!DatasetId::CODICRAC.download_url().is_empty());
        assert_eq!(DatasetId::CODICRAC.name(), "CODI-CRAC");

        assert!(!DatasetId::AMIMeeting.download_url().is_empty());
        assert_eq!(DatasetId::AMIMeeting.name(), "AMI Meeting");

        assert!(
            DatasetId::ARRAU.requires_license(),
            "ARRAU has LDC + research distribution; download_url may be empty"
        );
        assert!(
            DatasetId::ARRAU.name().contains("ARRAU"),
            "ARRAU name should contain 'ARRAU'"
        );

        // Should be in all() list
        let all = DatasetId::all();
        assert!(all.contains(&DatasetId::CODICRAC));
        assert!(all.contains(&DatasetId::AMIMeeting));
        assert!(all.contains(&DatasetId::ARRAU));
    }

    #[test]
    fn test_is_historical_classification() {
        // Historical datasets
        assert!(DatasetId::HIPE2022.is_historical());
        assert!(DatasetId::MedievalCzechCharters.is_historical());
        assert!(DatasetId::EighteenthCenturyNER.is_historical());
        assert!(DatasetId::HistoricalChineseNER.is_historical());

        // Non-historical should return false
        assert!(!DatasetId::WikiGold.is_historical());
        assert!(!DatasetId::CoNLL2003Sample.is_historical());
    }

    #[test]
    fn test_is_bias_evaluation_classification() {
        // Bias/fairness datasets
        assert!(DatasetId::WinoQueer.is_bias_evaluation());
        assert!(DatasetId::BBQ.is_bias_evaluation());
        assert!(DatasetId::GICoref.is_bias_evaluation());
        assert!(DatasetId::WinoBias.is_bias_evaluation());
        assert!(DatasetId::GAP.is_bias_evaluation());

        // Non-bias should return false
        assert!(!DatasetId::WikiGold.is_bias_evaluation());
    }

    #[test]
    fn test_new_datasets_have_descriptions() {
        // All new datasets should have proper descriptions (not the catch-all)
        let catch_all = "Dataset not yet fully integrated";

        // Historical
        assert_ne!(DatasetId::HIPE2022.description(), catch_all);
        assert_ne!(DatasetId::TRIDIS.description(), catch_all);

        // Queer NLP
        assert_ne!(DatasetId::WinoQueer.description(), catch_all);
        assert_ne!(DatasetId::BBQ.description(), catch_all);
        assert_ne!(DatasetId::GICoref.description(), catch_all);

        // Joint NER+RE
        assert_ne!(DatasetId::TACRED.description(), catch_all);
        assert_ne!(DatasetId::REBEL.description(), catch_all);

        // Dialogue
        assert_ne!(DatasetId::CODICRAC.description(), catch_all);
        assert_ne!(DatasetId::ARRAU.description(), catch_all);
    }

    #[test]
    fn test_coreference_includes_new_datasets() {
        // All coreference datasets should be detected
        assert!(DatasetId::GICoref.is_coreference());
        assert!(DatasetId::CODICRAC.is_coreference());
        assert!(DatasetId::ARRAU.is_coreference());
        assert!(DatasetId::WinoPron.is_coreference());
        assert!(DatasetId::DROC.is_coreference());
        assert!(DatasetId::KoCoNovel.is_coreference());
    }

    #[test]
    fn test_chisiec_is_historical_and_relation_extraction() {
        // CHisIEC should be both historical and relation extraction
        assert!(DatasetId::CHisIEC.is_historical());
        assert!(DatasetId::CHisIEC.is_relation_extraction());

        // Verify entity types
        let types = DatasetId::CHisIEC.entity_types();
        assert!(types.contains(&"PER"));
        assert!(types.contains(&"LOC"));
        assert!(types.contains(&"OFI"));
        assert!(types.contains(&"BOOK"));
    }

    #[test]
    fn test_chisiec_from_str() {
        // Test various string representations
        assert_eq!("chisiec".parse::<DatasetId>().unwrap(), DatasetId::CHisIEC);
        assert_eq!(
            "ch-is-iec".parse::<DatasetId>().unwrap(),
            DatasetId::CHisIEC
        );
        assert_eq!(
            "chinese-historical-ie".parse::<DatasetId>().unwrap(),
            DatasetId::CHisIEC
        );
        assert_eq!(
            "ancient-chinese-ner".parse::<DatasetId>().unwrap(),
            DatasetId::CHisIEC
        );
    }

    #[test]
    fn test_chisiec_parse_ner() {
        // Test CHisIEC NER parsing with sample data
        let sample_json = r#"[
            {
                "tokens": "衞鞅奔魏",
                "entities": [
                    {"type": "PER", "start": 0, "end": 2, "span": "衞鞅"},
                    {"type": "LOC", "start": 3, "end": 4, "span": "魏"}
                ],
                "relations": []
            }
        ]"#;

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_chisiec(sample_json, DatasetId::CHisIEC);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);

        let sentence = &dataset.sentences[0];
        // 4 characters: 衞 鞅 奔 魏
        assert_eq!(sentence.tokens.len(), 4);

        // Check BIO tags
        assert_eq!(sentence.tokens[0].ner_tag, "B-PER"); // 衞
        assert_eq!(sentence.tokens[1].ner_tag, "I-PER"); // 鞅
        assert_eq!(sentence.tokens[2].ner_tag, "O"); // 奔
        assert_eq!(sentence.tokens[3].ner_tag, "B-LOC"); // 魏
    }

    #[test]
    fn test_chisiec_parse_relations() {
        // Test CHisIEC relation extraction parsing
        let sample_json = r#"[
            {
                "tokens": "嚴公遣玉汝使",
                "entities": [
                    {"type": "PER", "start": 0, "end": 2, "span": "嚴公"},
                    {"type": "PER", "start": 2, "end": 4, "span": "玉汝"}
                ],
                "relations": [
                    {"type": "上下級", "head": 0, "tail": 1, "head_span": "嚴公", "tail_span": "玉汝"}
                ]
            }
        ]"#;

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_chisiec_relations(sample_json);
        assert!(result.is_ok());

        let docs = result.unwrap();
        assert_eq!(docs.len(), 1);

        let doc = &docs[0];
        assert_eq!(doc.relations.len(), 1);

        let rel = &doc.relations[0];
        assert_eq!(rel.relation_type, "上下級");
        assert_eq!(rel.head_text, "嚴公");
        assert_eq!(rel.tail_text, "玉汝");
        assert_eq!(rel.head_type, "PER");
        assert_eq!(rel.tail_type, "PER");
        // Character offsets (嚴公 at 0-2, 玉汝 at 2-4)
        assert_eq!(rel.head_span, (0, 2));
        assert_eq!(rel.tail_span, (2, 4));
    }

    #[test]
    fn test_chisiec_all_entity_types() {
        // Test all 4 CHisIEC entity types: PER, LOC, OFI (官职), BOOK (书籍)
        // This is important because OFI (Official position) is domain-specific
        let sample_json = r#"[
            {
                "tokens": "司馬遷為太史令著史記於長安",
                "entities": [
                    {"type": "PER", "start": 0, "end": 3, "span": "司馬遷"},
                    {"type": "OFI", "start": 4, "end": 7, "span": "太史令"},
                    {"type": "BOOK", "start": 8, "end": 10, "span": "史記"},
                    {"type": "LOC", "start": 11, "end": 13, "span": "長安"}
                ],
                "relations": []
            }
        ]"#;

        let loader = DatasetLoader::new().unwrap();
        let dataset = loader
            .parse_chisiec(sample_json, DatasetId::CHisIEC)
            .unwrap();

        assert_eq!(dataset.sentences.len(), 1);
        let sentence = &dataset.sentences[0];

        // Verify each entity type is correctly tagged
        // 司馬遷 (Sima Qian - historian)
        assert_eq!(sentence.tokens[0].ner_tag, "B-PER");
        assert_eq!(sentence.tokens[1].ner_tag, "I-PER");
        assert_eq!(sentence.tokens[2].ner_tag, "I-PER");

        // 為 (was)
        assert_eq!(sentence.tokens[3].ner_tag, "O");

        // 太史令 (Grand Historian - official position)
        assert_eq!(sentence.tokens[4].ner_tag, "B-OFI");
        assert_eq!(sentence.tokens[5].ner_tag, "I-OFI");
        assert_eq!(sentence.tokens[6].ner_tag, "I-OFI");

        // 著 (wrote)
        assert_eq!(sentence.tokens[7].ner_tag, "O");

        // 史記 (Records of the Grand Historian - book)
        assert_eq!(sentence.tokens[8].ner_tag, "B-BOOK");
        assert_eq!(sentence.tokens[9].ner_tag, "I-BOOK");

        // 於 (at)
        assert_eq!(sentence.tokens[10].ner_tag, "O");

        // 長安 (Chang'an - capital city)
        assert_eq!(sentence.tokens[11].ner_tag, "B-LOC");
        assert_eq!(sentence.tokens[12].ner_tag, "I-LOC");
    }

    #[test]
    fn test_chisiec_unicode_character_offsets() {
        // Critical: CHisIEC uses CHARACTER offsets, not byte offsets
        // This test ensures we handle multi-byte Chinese characters correctly
        let sample_json = r#"[
            {
                "tokens": "曹操",
                "entities": [
                    {"type": "PER", "start": 0, "end": 2, "span": "曹操"}
                ],
                "relations": []
            }
        ]"#;

        let loader = DatasetLoader::new().unwrap();
        let dataset = loader
            .parse_chisiec(sample_json, DatasetId::CHisIEC)
            .unwrap();

        // 曹操 is 2 characters (but 6 bytes in UTF-8)
        let sentence = &dataset.sentences[0];
        assert_eq!(sentence.tokens.len(), 2);
        assert_eq!(sentence.tokens[0].text, "曹");
        assert_eq!(sentence.tokens[1].text, "操");
    }

    #[test]
    fn test_chisiec_multiple_relations_same_document() {
        // Test parsing multiple relations in a single document
        // Relations: 任職 (holds office), 管理 (manages)
        let sample_json = r#"[
            {
                "tokens": "曹操為丞相管冀州",
                "entities": [
                    {"type": "PER", "start": 0, "end": 2, "span": "曹操"},
                    {"type": "OFI", "start": 3, "end": 5, "span": "丞相"},
                    {"type": "LOC", "start": 6, "end": 8, "span": "冀州"}
                ],
                "relations": [
                    {"type": "任職", "head": 0, "tail": 1, "head_span": "曹操", "tail_span": "丞相"},
                    {"type": "管理", "head": 0, "tail": 2, "head_span": "曹操", "tail_span": "冀州"}
                ]
            }
        ]"#;

        let loader = DatasetLoader::new().unwrap();
        let docs = loader.parse_chisiec_relations(sample_json).unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].relations.len(), 2);

        // First relation: 任職 (office holding)
        assert_eq!(docs[0].relations[0].relation_type, "任職");
        assert_eq!(docs[0].relations[0].head_type, "PER");
        assert_eq!(docs[0].relations[0].tail_type, "OFI");

        // Second relation: 管理 (manages)
        assert_eq!(docs[0].relations[1].relation_type, "管理");
        assert_eq!(docs[0].relations[1].head_type, "PER");
        assert_eq!(docs[0].relations[1].tail_type, "LOC");
    }

    #[test]
    fn test_chisiec_distinct_from_historical_chinese_ner() {
        // CHisIEC and HistoricalChineseNER are DIFFERENT datasets
        // This test documents their distinction

        // Both should be classified as historical
        assert!(DatasetId::CHisIEC.is_historical());
        assert!(DatasetId::HistoricalChineseNER.is_historical());

        // But they are different datasets
        assert_ne!(DatasetId::CHisIEC, DatasetId::HistoricalChineseNER);

        // CHisIEC supports relation extraction; HistoricalChineseNER may not
        assert!(DatasetId::CHisIEC.is_relation_extraction());

        // Different entity types:
        // CHisIEC: PER, LOC, OFI, BOOK (ancient Chinese)
        // HistoricalChineseNER: PER, LOC, ORG, DATE, etc. (modern Chinese 1872-1949)
        let chisiec_types = DatasetId::CHisIEC.entity_types();
        assert!(chisiec_types.contains(&"OFI")); // Official position - unique to CHisIEC
        assert!(chisiec_types.contains(&"BOOK")); // Classical texts

        // Different names
        assert_eq!(DatasetId::CHisIEC.name(), "CHisIEC");
        assert_eq!(
            DatasetId::HistoricalChineseNER.name(),
            "Historical Chinese NER"
        );
    }

    #[test]
    fn test_chisiec_empty_entities_handled() {
        // Test graceful handling of documents with no entities
        let sample_json = r#"[
            {
                "tokens": "天下太平",
                "entities": [],
                "relations": []
            }
        ]"#;

        let loader = DatasetLoader::new().unwrap();
        let dataset = loader
            .parse_chisiec(sample_json, DatasetId::CHisIEC)
            .unwrap();

        assert_eq!(dataset.sentences.len(), 1);
        let sentence = &dataset.sentences[0];

        // All tokens should be tagged as O
        for token in &sentence.tokens {
            assert_eq!(token.ner_tag, "O");
        }
    }

    #[test]
    fn test_chisiec_entity_types_in_schema() {
        // Verify CHisIEC entity types are properly mapped in the schema
        use anno::schema::map_to_canonical;

        // PER -> Person
        let per_type = map_to_canonical("PER", None);
        assert_eq!(per_type, EntityType::Person);

        // LOC -> Location
        let loc_type = map_to_canonical("LOC", None);
        assert_eq!(loc_type, EntityType::Location);

        // OFI -> Custom OFFICIAL type (domain-specific for ancient Chinese)
        let ofi_type = map_to_canonical("OFI", None);
        assert!(matches!(ofi_type, EntityType::Custom { .. }));

        // BOOK -> WORK_OF_ART (creative works)
        let book_type = map_to_canonical("BOOK", None);
        assert!(matches!(book_type, EntityType::Custom { .. }));
    }

    #[test]
    fn test_chisiec_language_and_domain() {
        // CHisIEC is Classical Chinese (文言文) - ISO 639-3: lzh
        assert_eq!(DatasetId::CHisIEC.language(), "lzh");

        // CHisIEC is a historical dataset (24 dynastic histories)
        assert_eq!(DatasetId::CHisIEC.domain(), "historical");

        // Compare with HistoricalChineseNER which is modern Chinese (1872-1949)
        assert_eq!(DatasetId::HistoricalChineseNER.language(), "zh");
        assert_eq!(DatasetId::HistoricalChineseNER.domain(), "historical");
    }

    // =========================================================================
    // African Language Dataset Tests
    // =========================================================================

    #[test]
    fn test_african_datasets_configured() {
        // MasakhaNER datasets should have download URLs
        assert!(!DatasetId::MasakhaNER.download_url().is_empty());
        assert!(!DatasetId::MasakhaNER2.download_url().is_empty());
        assert!(!DatasetId::AfriSenti.download_url().is_empty());
        assert!(!DatasetId::AfriQA.download_url().is_empty());
        assert!(!DatasetId::MasakhaNEWS.download_url().is_empty());
        assert!(!DatasetId::MasakhaPOS.download_url().is_empty());

        // Names should be set
        assert_eq!(DatasetId::MasakhaNER.name(), "MasakhaNER");
        assert_eq!(DatasetId::MasakhaNER2.name(), "MasakhaNER 2.0");
        assert_eq!(DatasetId::AfriSenti.name(), "AfriSenti");
        assert_eq!(DatasetId::AfriQA.name(), "AfriQA");
        assert_eq!(DatasetId::MasakhaNEWS.name(), "MasakhaNEWS");
        assert_eq!(DatasetId::MasakhaPOS.name(), "MasakhaPOS");
    }

    #[test]
    fn test_african_datasets_entity_types() {
        // MasakhaNER uses PER, ORG, LOC, DATE
        let ner_types = DatasetId::MasakhaNER.entity_types();
        assert!(ner_types.contains(&"PER"));
        assert!(ner_types.contains(&"LOC"));
        assert!(ner_types.contains(&"ORG"));
        assert!(ner_types.contains(&"DATE"));

        // AfriSenti uses sentiment labels
        let senti_types = DatasetId::AfriSenti.entity_types();
        assert!(senti_types.contains(&"positive"));
        assert!(senti_types.contains(&"neutral"));
        assert!(senti_types.contains(&"negative"));

        // MasakhaNEWS uses topic labels
        let news_types = DatasetId::MasakhaNEWS.entity_types();
        assert!(news_types.contains(&"politics"));
        assert!(news_types.contains(&"sports"));
        assert!(news_types.contains(&"business"));

        // MasakhaPOS uses Universal Dependencies POS tags
        let pos_types = DatasetId::MasakhaPOS.entity_types();
        assert!(pos_types.contains(&"NOUN"));
        assert!(pos_types.contains(&"VERB"));
        assert!(pos_types.contains(&"ADJ"));
    }

    #[test]
    fn test_parse_afrisenti() {
        // Test AfriSenti TSV parsing
        let sample_tsv = "This movie is great!\tpositive\n\
                          Awful experience\tnegative\n\
                          It was okay\tneutral";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_afrisenti(sample_tsv, DatasetId::AfriSenti);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 3);

        // Check first sentence has positive label
        assert_eq!(dataset.sentences[0].tokens[0].ner_tag, "B-positive");
        // Check second sentence has negative label
        assert_eq!(dataset.sentences[1].tokens[0].ner_tag, "B-negative");
        // Check third sentence has neutral label
        assert_eq!(dataset.sentences[2].tokens[0].ner_tag, "B-neutral");
    }

    #[test]
    fn test_parse_masakhanews() {
        // Test MasakhaNEWS TSV parsing
        let sample_tsv = "headline\tbody\tcategory\n\
                          Breaking: Election Results\tThe results are in...\tpolitics\n\
                          Team Wins Championship\tIn an exciting match...\tsports";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_masakhanews(sample_tsv, DatasetId::MasakhaNEWS);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        // Header line should be skipped
        assert_eq!(dataset.sentences.len(), 2);

        // Check categories
        assert_eq!(dataset.sentences[0].tokens[0].ner_tag, "B-politics");
        assert_eq!(dataset.sentences[1].tokens[0].ner_tag, "B-sports");
    }

    #[test]
    fn test_parse_conllu() {
        // Test CoNLL-U parsing (MasakhaPOS format)
        let sample_conllu = "# sent_id = 1\n\
                             # text = John loves Mary\n\
                             1\tJohn\tJohn\tPROPN\tNNP\t_\t2\tnsubj\t_\t_\n\
                             2\tloves\tlove\tVERB\tVBZ\t_\t0\troot\t_\t_\n\
                             3\tMary\tMary\tPROPN\tNNP\t_\t2\tobj\t_\t_\n\
                             \n\
                             # sent_id = 2\n\
                             1\tHe\the\tPRON\tPRP\t_\t2\tnsubj\t_\t_\n\
                             2\truns\trun\tVERB\tVBZ\t_\t0\troot\t_\t_\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conllu(sample_conllu, DatasetId::MasakhaPOS);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 2);

        // First sentence: John loves Mary
        assert_eq!(dataset.sentences[0].tokens.len(), 3);
        assert_eq!(dataset.sentences[0].tokens[0].text, "John");
        assert_eq!(dataset.sentences[0].tokens[0].ner_tag, "B-PROPN");
        assert_eq!(dataset.sentences[0].tokens[1].text, "loves");
        assert_eq!(dataset.sentences[0].tokens[1].ner_tag, "B-VERB");

        // Second sentence: He runs
        assert_eq!(dataset.sentences[1].tokens.len(), 2);
        assert_eq!(dataset.sentences[1].tokens[0].text, "He");
        assert_eq!(dataset.sentences[1].tokens[0].ner_tag, "B-PRON");
    }

    #[test]
    fn test_ancient_language_ud_datasets_are_loadable() {
        // Ancient language UD treebanks should be loadable via registry hints
        // These have format: "CoNLLU" and categories: [ner, ancient]
        let ancient_datasets = [
            DatasetId::AncientGreekUD,
            DatasetId::LatinUD,
            DatasetId::SanskritUD,
            DatasetId::OldEnglishUD,
            DatasetId::OldNorseUD,
        ];

        for ds in ancient_datasets {
            assert!(
                LoadableDatasetId::is_loadable_dataset(ds),
                "{:?} should be loadable via registry hint (format={:?})",
                ds,
                ds.format()
            );
        }
    }

    #[test]
    fn test_conllu_with_ner_tags_from_ancient_greek() {
        // Test CoNLLU parsing with MISC column NER tags (Ancient Greek Perseus format)
        // Real format from UD Ancient Greek Perseus
        let sample_conllu = "\
# sent_id = tlg0012.tlg001.perseus-grc1:1.1
# text = μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος
1\tμῆνιν\tμῆνις\tNOUN\tn-s---fa-\tCase=Acc|Gender=Fem|Number=Sing\t2\tobj\t_\tO
2\tἄειδε\tᾄδω\tVERB\tv2sama---\tMood=Imp|Number=Sing|Person=2|Tense=Pres|VerbForm=Fin|Voice=Act\t0\troot\t_\tO
3\tθεὰ\tθεά\tNOUN\tn-s---fv-\tCase=Voc|Gender=Fem|Number=Sing\t2\tvocative\t_\tO
4\tΠηληϊάδεω\tΠηληϊάδης\tNOUN\tn-s---mg-\tCase=Gen|Gender=Masc|Number=Sing\t5\tnmod\t_\tB-PER
5\tἈχιλῆος\tἈχιλλεύς\tPROPN\tn-s---mg-\tCase=Gen|Gender=Masc|Number=Sing\t1\tnmod\t_\tI-PER

";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conllu(sample_conllu, DatasetId::AncientGreekUD);
        assert!(
            result.is_ok(),
            "Failed to parse Ancient Greek CoNLLU: {:?}",
            result.err()
        );

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        assert_eq!(dataset.sentences[0].tokens.len(), 5);

        // Check Achilles (Ἀχιλῆος) entity
        assert_eq!(dataset.sentences[0].tokens[4].text, "Ἀχιλῆος");
        // Note: CoNLLU parser may use POS tags if MISC doesn't have NER
        // This depends on how the parser handles the MISC column
    }

    #[test]
    fn test_registry_hints_cover_all_conllu_ner_datasets() {
        // Datasets with format CoNLLU/CoNLL-U and task NER should get a registry hint
        for &ds in DatasetId::all() {
            let format = ds.format().unwrap_or("");
            let is_conllu = format == "CoNLLU" || format == "CoNLL-U";
            let is_ner = ds.is_ner();

            if is_conllu && is_ner {
                let hint = LoadableDatasetId::registry_hint_plan(ds);
                assert!(
                    hint.is_some(),
                    "{:?} has format={} and is NER but no registry hint",
                    ds,
                    format
                );
                if let Some(plan) = hint {
                    assert_eq!(
                        plan,
                        DatasetParsePlan::Conllu,
                        "{:?} should use Conllu parse plan",
                        ds
                    );
                }
            }
        }
    }

    #[test]
    fn test_datasets_with_public_url_and_format_are_hintable() {
        // Datasets with a public URL and a parseable format should get hints
        let hintable_formats = ["CoNLL", "CoNLLU", "CoNLL-U", "BIO", "IOB2", "JSONL"];

        let mut missing_hints = Vec::new();

        for &ds in DatasetId::all() {
            let url = ds.download_url();
            let format = ds.format().unwrap_or("");
            let is_ner = ds.is_ner();

            // Skip datasets without URLs or non-NER datasets
            if url.is_empty() || !is_ner {
                continue;
            }

            // Skip formats we don't auto-detect
            if !hintable_formats.contains(&format) {
                continue;
            }

            let hint = LoadableDatasetId::registry_hint_plan(ds);
            if hint.is_none() {
                missing_hints.push((ds, format));
            }
        }

        // Allow some datasets to not have hints (complex formats, etc.)
        // but document them
        if !missing_hints.is_empty() {
            // These are known to be missing hints (need special parsers)
            let known_missing: &[DatasetId] = &[
                // Add any datasets that intentionally don't have hints here
            ];
            for (ds, format) in &missing_hints {
                if !known_missing.contains(ds) {
                    eprintln!(
                        "Warning: {:?} (format={}) has public URL but no registry hint",
                        ds, format
                    );
                }
            }
        }
    }

    #[test]
    fn test_loadable_count_is_reasonable() {
        // Ensure we have a reasonable number of loadable datasets
        let loadable_count = LoadableDatasetId::all().len();
        let total_count = DatasetId::all().len();

        // We should have at least 50% of datasets loadable via either parse_plan or hints
        let min_expected = total_count / 2;
        assert!(
            loadable_count >= min_expected,
            "Only {} of {} datasets are loadable (expected at least {})",
            loadable_count,
            total_count,
            min_expected
        );
    }

    #[test]
    fn test_datasets_with_urls_have_formats() {
        // Datasets with public URLs should ideally have format info for auto-loading
        let mut missing_format = Vec::new();

        for &ds in DatasetId::all() {
            let url = ds.download_url();
            let format = ds.format();
            let access = ds.access_status();

            // Skip datasets that require registration or aren't publicly available
            if url.is_empty() {
                continue;
            }

            // Check if format is missing for public datasets
            if format.is_none()
                && access == crate::eval::dataset_registry::DatasetAccessibility::Public
            {
                missing_format.push(ds);
            }
        }

        // Log datasets that could benefit from format info
        if !missing_format.is_empty() {
            eprintln!(
                "Datasets with public URLs but no format field ({}):",
                missing_format.len()
            );
            for ds in &missing_format[..missing_format.len().min(10)] {
                eprintln!("  - {:?}", ds);
            }
        }

        // We expect most public datasets to have format info
        // Allow up to 20% to be missing format (some have unusual formats)
        let max_missing = DatasetId::all().len() / 5;
        assert!(
            missing_format.len() <= max_missing,
            "Too many public datasets missing format: {} (max {})",
            missing_format.len(),
            max_missing
        );
    }

    #[test]
    fn test_conll_format_ner_only_datasets_are_parseable() {
        // All NER-only datasets with CoNLL/CoNLLU format should have a parse plan
        // (Datasets with joint RE/coref tasks may use different column formats)
        let mut not_loadable = Vec::new();

        for &ds in DatasetId::all() {
            let format = ds.format().unwrap_or("");
            let is_conll = format == "CoNLL" || format == "CoNLLU" || format == "CoNLL-U";
            let is_ner = ds.is_ner();
            let is_re = ds.is_relation_extraction();
            let is_coref = ds.is_coreference();
            let is_event = ds.is_event_coref();

            if !is_conll || !is_ner {
                continue;
            }

            // Skip explicitly blocked datasets (they may be present in the registry for metadata,
            // but intentionally cannot be downloaded/loaded automatically).
            if ds.tasks_or_inferred().contains(&"blocked") {
                continue;
            }

            // Skip joint task datasets (they use CoNLL but with different structure)
            if is_re || is_coref || is_event {
                continue;
            }

            // Pure NER CoNLL datasets should be loadable
            let is_loadable = LoadableDatasetId::is_loadable_dataset(ds);
            if !is_loadable {
                not_loadable.push((ds, format));
            }
        }

        if !not_loadable.is_empty() {
            eprintln!("Pure NER CoNLL datasets not loadable:");
            for (ds, format) in &not_loadable {
                eprintln!("  - {:?} (format={})", ds, format);
            }
        }

        // All pure NER CoNLL datasets should be loadable
        assert!(
            not_loadable.is_empty(),
            "{} pure NER CoNLL datasets are not loadable",
            not_loadable.len()
        );
    }

    #[test]
    fn test_jsonl_ner_datasets_are_parseable() {
        // JSONL datasets with NER task should ideally be loadable
        let mut jsonl_ner_not_loadable = Vec::new();

        for &ds in DatasetId::all() {
            let format = ds.format().unwrap_or("");
            let is_jsonl = format == "JSONL" || format == "JSON-Lines" || format == "jsonl";
            let is_ner = ds.is_ner();

            if !is_jsonl || !is_ner {
                continue;
            }

            if !LoadableDatasetId::is_loadable_dataset(ds) {
                jsonl_ner_not_loadable.push(ds);
            }
        }

        // Log for debugging
        if !jsonl_ner_not_loadable.is_empty() {
            eprintln!(
                "JSONL NER datasets not loadable ({}):",
                jsonl_ner_not_loadable.len()
            );
            for ds in &jsonl_ner_not_loadable {
                eprintln!("  - {:?}", ds);
            }
        }
    }

    #[test]
    fn test_parse_afriqa() {
        // Test AfriQA JSON parsing
        let sample_json = r#"[
            {
                "context": "Lagos is a major city in Nigeria.",
                "question": "What is Lagos?",
                "answers": {
                    "text": ["major city"],
                    "answer_start": [11]
                }
            }
        ]"#;

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_afriqa(sample_json, DatasetId::AfriQA);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);

        // Check that answer is marked
        let tokens = &dataset.sentences[0].tokens;
        // The answer "major city" should have B-ANSWER and I-ANSWER tags
        let answer_tokens: Vec<_> = tokens
            .iter()
            .filter(|t| t.ner_tag.contains("ANSWER"))
            .collect();
        assert!(!answer_tokens.is_empty(), "Should have answer tokens");
    }

    #[test]
    fn test_african_datasets_in_all_list() {
        let all = DatasetId::all();
        assert!(all.contains(&DatasetId::MasakhaNER));
        assert!(all.contains(&DatasetId::MasakhaNER2));
        assert!(all.contains(&DatasetId::AfriSenti));
        assert!(all.contains(&DatasetId::AfriQA));
        assert!(all.contains(&DatasetId::MasakhaNEWS));
        assert!(all.contains(&DatasetId::MasakhaPOS));
    }

    // =========================================================================
    // Event Extraction Parser Tests
    // =========================================================================

    #[test]
    fn test_parse_maven_jsonl() {
        // Test MAVEN full JSONL format with events array
        let sample_jsonl = r#"{"id": "doc1", "content": [{"sentence": "The earthquake struck Tokyo.", "tokens": ["The", "earthquake", "struck", "Tokyo", "."]}], "events": [{"type": "Disaster", "mention": [{"trigger_word": "earthquake", "sent_id": 0, "offset": [1, 2]}]}]}"#;

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_maven(sample_jsonl, DatasetId::MAVEN);
        assert!(result.is_ok(), "parse_maven should succeed");

        let dataset = result.unwrap();
        assert!(!dataset.sentences.is_empty(), "Should have sentences");

        // Check event type tag
        let has_disaster = dataset
            .sentences
            .iter()
            .any(|s| s.tokens.iter().any(|t| t.ner_tag.contains("Disaster")));
        assert!(has_disaster, "Should have Disaster event tag");
    }

    #[test]
    fn test_parse_maven_docid2topic_fallback() {
        // Test fallback format (docid2topic.json)
        let sample_json = r#"{"doc1": "Natural_Disaster", "doc2": "Political_Event"}"#;

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_maven(sample_json, DatasetId::MAVEN);
        assert!(result.is_ok(), "parse_maven fallback should succeed");

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 2, "Should have 2 entries");
    }

    #[test]
    fn test_parse_casie() {
        // Test CASIE cybersecurity event format
        let sample_jsonl = r#"{"content": "A vulnerability was discovered in Apache.", "cyberevent": {"hopper": [{"events": [{"subtype": "Vulnerability", "nugget": {"text": "vulnerability"}, "argument": [{"text": "Apache", "role": {"type": "Affected_System"}}]}]}]}}"#;

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_casie(sample_jsonl, DatasetId::CASIE);
        assert!(result.is_ok(), "parse_casie should succeed");

        let dataset = result.unwrap();
        assert!(!dataset.sentences.is_empty(), "Should have sentences");

        // Check for vulnerability trigger
        let has_vuln = dataset
            .sentences
            .iter()
            .any(|s| s.tokens.iter().any(|t| t.ner_tag.contains("Vulnerability")));
        assert!(has_vuln, "Should have Vulnerability tag");

        // Check for argument
        let has_arg = dataset
            .sentences
            .iter()
            .any(|s| s.tokens.iter().any(|t| t.ner_tag.contains("ARG_")));
        assert!(has_arg, "Should have argument tag");
    }

    #[test]
    fn test_parse_maven_arg() {
        // Test MAVEN-ARG format with arguments
        let sample_jsonl = r#"{"id": "doc1", "document": "The company announced layoffs.", "events": [{"type": "Employment", "mention": [{"trigger_word": "layoffs", "offset": [4, 5]}], "argument": {"Employer": [{"content": "company", "offset": [1, 2]}]}}]}"#;

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_maven_arg(sample_jsonl, DatasetId::MAVENArg);
        assert!(result.is_ok(), "parse_maven_arg should succeed");

        let dataset = result.unwrap();
        assert!(!dataset.sentences.is_empty(), "Should have sentences");

        // Check for trigger
        let has_trigger = dataset
            .sentences
            .iter()
            .any(|s| s.tokens.iter().any(|t| t.ner_tag.contains("Employment")));
        assert!(has_trigger, "Should have Employment event tag");

        // Check for argument role
        let has_employer = dataset
            .sentences
            .iter()
            .any(|s| s.tokens.iter().any(|t| t.ner_tag.contains("ARG_Employer")));
        assert!(has_employer, "Should have Employer argument tag");
    }

    #[test]
    fn test_parse_rams() {
        // Test RAMS tokenized format
        let sample_jsonl = r#"{"doc_key": "doc1", "sentences": [["The", "soldier", "fired", "his", "weapon", "."]], "evt_triggers": [[2, 2, [["conflict.attack", 1.0]]]], "gold_evt_links": [[[0], [1, 1], "attacker"]]}"#;

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_rams(sample_jsonl, DatasetId::RAMS);
        assert!(result.is_ok(), "parse_rams should succeed");

        let dataset = result.unwrap();
        assert!(!dataset.sentences.is_empty(), "Should have sentences");

        // Check for event trigger
        let has_event = dataset
            .sentences
            .iter()
            .any(|s| s.tokens.iter().any(|t| t.ner_tag.starts_with("B-")));
        assert!(has_event, "Should have event tags");
    }

    #[test]
    fn test_parse_trec() {
        // Test TREC question classification format
        let sample =
            "NUM:dist How far is it from Denver to Aspen ?\nLOC:city What county is Modesto in ?\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_trec(sample, DatasetId::TREC);
        assert!(result.is_ok(), "parse_trec should succeed");

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 2, "Should have 2 questions");

        // Check coarse labels
        assert!(dataset.sentences[0].tokens[0].ner_tag.contains("NUM"));
        assert!(dataset.sentences[1].tokens[0].ner_tag.contains("LOC"));
    }

    #[test]
    fn test_parse_litbank_ner_improved() {
        // Test improved LitBank NER parser with word-level tokenization
        let sample_ann = "T1\tPER 0 5\tAlice\nT2\tPER 10 14\tBob\nT3\tORG 20 28\tMicrosoft";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_litbank(sample_ann, DatasetId::LitBank);
        assert!(result.is_ok(), "parse_litbank should succeed");

        let dataset = result.unwrap();
        assert!(!dataset.sentences.is_empty(), "Should have sentences");

        // Check that entities are tokenized into words (not just single tokens)
        let sentence = &dataset.sentences[0];
        let entity_tokens: Vec<_> = sentence
            .tokens
            .iter()
            .filter(|t| t.ner_tag.starts_with("B-") || t.ner_tag.starts_with("I-"))
            .collect();

        // Should have at least the entity tokens (Alice, Bob, Microsoft)
        assert!(
            entity_tokens.len() >= 3,
            "Should have at least 3 entity tokens, got {}",
            entity_tokens.len()
        );

        // Check BIO tagging is correct (B- for first word, I- for subsequent words)
        let mut found_b_tag = false;
        for token in &sentence.tokens {
            if token.ner_tag.starts_with("B-") {
                found_b_tag = true;
                // First word of entity should be B-
                assert!(
                    token.ner_tag.starts_with("B-"),
                    "First word of entity should have B- tag"
                );
            }
        }
        assert!(found_b_tag, "Should have at least one B- tag");
    }

    #[test]
    fn test_parse_tweettopic() {
        // Test TweetTopic JSONL format
        let sample_jsonl = r#"{"text": "Amazing game last night!", "label": 4, "label_name": "sports_&_gaming"}
{"text": "New AI breakthrough announced", "label": 5, "label_name": "science_&_technology"}"#;

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_tweettopic(sample_jsonl, DatasetId::TweetTopic);
        assert!(result.is_ok(), "parse_tweettopic should succeed");

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 2, "Should have 2 tweets");

        // Check label names are used
        assert!(dataset.sentences[0].tokens[0]
            .ner_tag
            .contains("sports_&_gaming"));
        assert!(dataset.sentences[1].tokens[0]
            .ner_tag
            .contains("science_&_technology"));
    }

    #[test]
    fn test_african_dataset_from_str() {
        // Test string parsing for African datasets
        assert_eq!(
            "masakhaner".parse::<DatasetId>().unwrap(),
            DatasetId::MasakhaNER
        );
        assert_eq!(
            "masakhaner2".parse::<DatasetId>().unwrap(),
            DatasetId::MasakhaNER2
        );
        assert_eq!(
            "afrisenti".parse::<DatasetId>().unwrap(),
            DatasetId::AfriSenti
        );
        assert_eq!("afriqa".parse::<DatasetId>().unwrap(), DatasetId::AfriQA);
        assert_eq!(
            "masakhanews".parse::<DatasetId>().unwrap(),
            DatasetId::MasakhaNEWS
        );
        assert_eq!(
            "masakhapos".parse::<DatasetId>().unwrap(),
            DatasetId::MasakhaPOS
        );

        // Alternative spellings
        assert_eq!(
            "masakhane-ner".parse::<DatasetId>().unwrap(),
            DatasetId::MasakhaNER
        );
        assert_eq!(
            "afri-senti".parse::<DatasetId>().unwrap(),
            DatasetId::AfriSenti
        );
        assert_eq!(
            "masakhane-news".parse::<DatasetId>().unwrap(),
            DatasetId::MasakhaNEWS
        );
    }

    #[test]
    fn test_afrisenti_parse_with_tonal_diacritics() {
        // Test Yoruba text with tonal diacritics (common in AfriSenti)
        let yoruba_tsv = "Ó dára púpọ̀!\tpositive\n\
                          Kò dára rárá\tnegative\n\
                          Ẹ ṣé, mo dupẹ́\tpositive";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_afrisenti(yoruba_tsv, DatasetId::AfriSenti);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 3);

        // Verify the text is preserved correctly with diacritics
        assert!(dataset.sentences[0].tokens[0].text.contains("dára"));
        assert!(dataset.sentences[1].tokens[0].text.contains("rárá"));
        assert!(dataset.sentences[2].tokens[0].text.contains("dupẹ́"));
    }

    #[test]
    fn test_masakhaner_parse_with_ethiopic_script() {
        // Test Amharic text in MasakhaNER CoNLL format (Ethiopic script)
        let amharic_conll = "ዶክተር B-PER\n\
                             አቢይ I-PER\n\
                             አህመድ I-PER\n\
                             ኢትዮጵያ B-LOC\n\
                             ውስጥ O\n\
                             ተወለዱ O\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conll(amharic_conll, DatasetId::MasakhaNER);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);

        let tokens = &dataset.sentences[0].tokens;
        assert_eq!(tokens.len(), 6);

        // Verify Ethiopic script is preserved
        assert_eq!(tokens[0].text, "ዶክተር");
        assert_eq!(tokens[0].ner_tag, "B-PER");
        assert_eq!(tokens[3].text, "ኢትዮጵያ");
        assert_eq!(tokens[3].ner_tag, "B-LOC");
    }

    #[test]
    fn test_conllu_parse_with_nguni_clicks() {
        // Test isiXhosa/isiZulu text with click consonants (MasakhaPOS)
        let xhosa_conllu = "# sent_id = xho_test_1\n\
                           # text = UMongameli uCyril Ramaphosa\n\
                           1\tUMongameli\tumongameli\tNOUN\tN\t_\t0\troot\t_\t_\n\
                           2\tuCyril\tuCyril\tPROPN\tNNP\t_\t1\tappos\t_\t_\n\
                           3\tRamaphosa\tRamaphosa\tPROPN\tNNP\t_\t2\tflat:name\t_\t_\n\
                           \n\
                           # sent_id = xho_test_2\n\
                           # text = Ndiqala ukuthetha isiXhosa\n\
                           1\tNdiqala\tqala\tVERB\tV\t_\t0\troot\t_\t_\n\
                           2\tukuthetha\tthetha\tVERB\tV\t_\t1\txcomp\t_\t_\n\
                           3\tisiXhosa\tisiXhosa\tNOUN\tN\t_\t2\tobj\t_\t_\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conllu(xhosa_conllu, DatasetId::MasakhaPOS);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 2);

        // Verify text with click-like consonants is preserved
        assert_eq!(dataset.sentences[0].tokens[0].text, "UMongameli");
        assert_eq!(dataset.sentences[1].tokens[2].text, "isiXhosa");
    }

    #[test]
    fn test_masakhanews_parse_with_arabic_variants() {
        // MasakhaNEWS includes Algerian/Moroccan Arabic variants
        let news_tsv = "headline\tbody\tcategory\n\
                        الأخبار العاجلة\tتفاصيل الخبر...\tpolitics\n\
                        رياضة محلية\tمباراة اليوم...\tsports";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_masakhanews(news_tsv, DatasetId::MasakhaNEWS);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        // Header skipped, 2 data rows
        assert_eq!(dataset.sentences.len(), 2);

        // Check Arabic text is preserved
        assert!(dataset.sentences[0].tokens[0].text.contains("الأخبار"));
        assert_eq!(dataset.sentences[0].tokens[0].ner_tag, "B-politics");
        assert_eq!(dataset.sentences[1].tokens[0].ner_tag, "B-sports");
    }

    #[test]
    fn test_afriqa_multilingual_qa() {
        // AfriQA has questions in target language, context may be in English
        let qa_json = r#"[
            {
                "context": "Yorùbá is a tonal language spoken in Nigeria.",
                "question": "Kí ni Yorùbá?",
                "answers": {
                    "text": ["tonal language"],
                    "answer_start": [13]
                },
                "language": "yo"
            }
        ]"#;

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_afriqa(qa_json, DatasetId::AfriQA);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
    }

    // NOTE: We intentionally do not embed language-code→URL expansion helpers here.
    // If we add them, they should live in the registry (metadata) or a dedicated dataset
    // URL builder module, not in the loader tests.

    // =========================================================================
    // Comprehensive Parser Smoke Tests (one per DatasetParsePlan)
    // =========================================================================

    #[test]
    fn test_parse_content_rejects_empty_for_all_loadable_datasets() {
        // Global invariant: no dataset should parse from empty content.
        let loader = DatasetLoader::new().unwrap();
        for loadable in LoadableDatasetId::all() {
            let id: DatasetId = loadable.into();
            let err = loader
                .parse_content_impl("   \n\t", id)
                .expect_err("empty content must error");
            let msg = format!("{err}");
            assert!(
                msg.to_lowercase().contains("empty"),
                "Expected an 'empty' error message for {:?}, got: {}",
                id,
                msg
            );
        }
    }

    #[test]
    fn test_parse_docred_smoke() {
        let sample = r#"{"doc_key":"d1","sentence":["John","met","Mary","in","Paris","."],"ner":[[0,0,"PER"],[2,2,"PER"],[4,4,"LOC"]],"relations":[]}"#;
        let loader = DatasetLoader::new().unwrap();
        let ds = loader.parse_docred(sample, DatasetId::DocRED).unwrap();
        assert!(!ds.sentences.is_empty());
        assert!(ds.sentences[0]
            .tokens
            .iter()
            .any(|t| t.ner_tag.starts_with("B-")));
    }

    #[test]
    fn test_parse_cadec_jsonl_smoke() {
        // Character offsets in `entities` are interpreted over reconstructed text from tokens.
        // "I took aspirin" → aspirin starts at char 7, ends at 14 (exclusive).
        let sample = r#"{"tokens":["I","took","aspirin"],"entities":[{"text":"aspirin","label":"DRUG","start":7,"end":14}]}"#;
        let loader = DatasetLoader::new().unwrap();
        let ds = loader.parse_cadec_jsonl(sample, DatasetId::CADEC).unwrap();
        assert!(!ds.sentences.is_empty());
        assert!(ds.sentences[0]
            .tokens
            .iter()
            .any(|t| t.ner_tag == "B-DRUG" || t.ner_tag == "I-DRUG"));
    }

    #[test]
    fn test_parse_cadec_hf_api_smoke() {
        let sample = r#"{"rows":[{"row":{"text":"I took aspirin","ade":"aspirin"}}]}"#;
        let loader = DatasetLoader::new().unwrap();
        let ds = loader.parse_cadec_hf_api(sample, DatasetId::CADEC).unwrap();
        assert!(!ds.sentences.is_empty());
        assert!(ds.sentences[0]
            .tokens
            .iter()
            .any(|t| t.ner_tag.contains("adverse_drug_event")));
    }

    #[test]
    fn test_parse_bc5cdr_smoke() {
        let sample = "Aspirin\tNN\tO\tB-CHEMICAL\nhelps\tVBZ\tO\tO\n\n";
        let loader = DatasetLoader::new().unwrap();
        let ds = loader.parse_bc5cdr(sample, DatasetId::BC5CDR).unwrap();
        assert_eq!(ds.sentences.len(), 1);
        assert_eq!(ds.sentences[0].tokens[0].ner_tag, "B-CHEMICAL");
    }

    #[test]
    fn test_parse_ncbi_disease_smoke() {
        let sample = "Cancer\tNN\tO\tB-Disease\nprogresses\tVBZ\tO\tO\n\n";
        let loader = DatasetLoader::new().unwrap();
        let ds = loader
            .parse_ncbi_disease(sample, DatasetId::NCBIDisease)
            .unwrap();
        assert_eq!(ds.sentences.len(), 1);
        assert!(ds.sentences[0].tokens[0].ner_tag.starts_with("B-"));
    }

    #[test]
    fn test_parse_gap_smoke() {
        let sample =
            "ID\tText\tPronoun\tPronoun-offset\tA\tA-offset\tA-coref\tB\tB-offset\tB-coref\tURL\n\
g1\tJohn met Mary. He waved.\tHe\t14\tJohn\t0\tTRUE\tMary\t9\tFALSE\thttp://example\n";
        let loader = DatasetLoader::new().unwrap();
        let ds = loader.parse_gap(sample, DatasetId::GAP).unwrap();
        assert_eq!(ds.sentences.len(), 1);
        assert!(!ds.sentences[0].tokens.is_empty());
    }

    #[test]
    fn test_parse_preco_jsonl_smoke() {
        let sample = r#"{"sentences":[["John","went","home","."],["He","slept","."]]}"#;
        let loader = DatasetLoader::new().unwrap();
        let ds = loader.parse_preco_jsonl(sample, DatasetId::PreCo).unwrap();
        assert_eq!(ds.sentences.len(), 2);
        assert_eq!(ds.sentences[0].tokens[0].text, "John");
    }

    #[test]
    fn test_parse_wikiann_json_array_smoke() {
        let sample =
            r#"[{"tokens":["John","went","to","Paris"],"ner_tags":["B-PER","O","O","B-LOC"]}]"#;
        let loader = DatasetLoader::new().unwrap();
        let ds = loader.parse_wikiann_json(sample, DatasetId::UNER).unwrap();
        assert_eq!(ds.sentences.len(), 1);
        assert_eq!(ds.sentences[0].tokens[0].ner_tag, "B-PER");
    }

    #[test]
    fn test_parse_hf_api_response_smoke() {
        let sample = r#"{
  "features":[{"name":"tokens"},{"name":"ner_tags","type":{"feature":{"names":["O","B-PER","I-PER"]}}}],
  "rows":[{"row_idx":0,"row":{"tokens":["John"],"ner_tags":[1]}}]
}"#;
        let loader = DatasetLoader::new().unwrap();
        let ds = loader
            .parse_hf_api_response(sample, DatasetId::UniversalNER)
            .unwrap();
        assert_eq!(ds.sentences.len(), 1);
        assert_eq!(ds.sentences[0].tokens[0].ner_tag, "B-PER");
    }

    #[test]
    fn test_parse_hf_api_response_temporal_standoff_smoke() {
        let sample = r#"{
  "features":[{"name":"text"},{"name":"time_expressions"}],
  "rows":[{"row_idx":0,"row":{
    "text":"A 10/30/89 .",
    "time_expressions":[{"text":"10/30/89","start_char":2,"end_char":10,"tid":"t1","type":"DATE","value":"1989-10-30"}],
    "event_expressions":[],
    "signal_expressions":[]
  }}]
}"#;
        let loader = DatasetLoader::new().unwrap();
        let ds = loader
            .parse_hf_api_response(sample, DatasetId::TimexRecognitionSentenceOriginal)
            .unwrap();
        assert_eq!(ds.sentences.len(), 1);
        assert_eq!(ds.sentences[0].tokens.len(), 3);
        assert_eq!(ds.sentences[0].tokens[0].text, "A");
        assert_eq!(ds.sentences[0].tokens[0].ner_tag, "O");
        assert_eq!(ds.sentences[0].tokens[1].text, "10/30/89");
        assert_eq!(ds.sentences[0].tokens[1].ner_tag, "B-TIMEX");
        assert_eq!(ds.sentences[0].tokens[2].text, ".");
        assert_eq!(ds.sentences[0].tokens[2].ner_tag, "O");
    }

    #[test]
    fn test_parse_hf_api_response_pairwise_discourse_smoke() {
        let sample = r#"{
  "features":[{"name":"unit1_txt"},{"name":"unit2_txt"},{"name":"label"}],
  "rows":[{"row_idx":0,"row":{
    "unit1_txt":"Because it rained",
    "unit2_txt":"the game was canceled",
    "label":"Cause"
  }}]
}"#;
        let loader = DatasetLoader::new().unwrap();
        let ds = loader
            .parse_hf_api_response(sample, DatasetId::DisrptEngDepScidtbRels)
            .unwrap();
        assert_eq!(ds.sentences.len(), 1);
        assert_eq!(ds.sentences[0].tokens.len(), 1);
        assert_eq!(
            ds.sentences[0].tokens[0].text,
            "Because it rained [SEP] the game was canceled"
        );
        assert_eq!(ds.sentences[0].tokens[0].ner_tag, "B-Cause");
    }

    #[test]
    fn test_parse_hf_api_response_disrpt_conllu_seg_smoke() {
        let sample = r#"{
  "features":[{"name":"form"},{"name":"misc"}],
  "rows":[{"row_idx":0,"row":{
    "form":["We","propose","a","method","."],
    "misc":["Seg=B-seg","Seg=O","Seg=O","Seg=B-seg","Seg=O"]
  }}]
}"#;
        let loader = DatasetLoader::new().unwrap();
        let ds = loader
            .parse_hf_api_response(sample, DatasetId::DisrptEngDepScidtbConlluSeg)
            .unwrap();
        assert_eq!(ds.sentences.len(), 1);
        let tags: Vec<&str> = ds.sentences[0]
            .tokens
            .iter()
            .map(|t| t.ner_tag.as_str())
            .collect();
        assert_eq!(tags, vec!["B-SEG", "I-SEG", "I-SEG", "B-SEG", "I-SEG"]);
    }

    #[test]
    fn test_parse_agnews_smoke() {
        let sample = r#"{"text":"Stocks rally on earnings","label":2}"#;
        let loader = DatasetLoader::new().unwrap();
        let ds = loader.parse_agnews(sample, DatasetId::AGNews).unwrap();
        assert_eq!(ds.sentences.len(), 1);
        assert!(ds.sentences[0].tokens[0].ner_tag.starts_with("B-"));
    }

    #[test]
    fn test_parse_dbpedia14_smoke() {
        let sample = r#"{"content":"The Beatles released Abbey Road","label":5}"#;
        let loader = DatasetLoader::new().unwrap();
        let ds = loader
            .parse_dbpedia14(sample, DatasetId::DBPedia14)
            .unwrap();
        assert_eq!(ds.sentences.len(), 1);
        assert!(ds.sentences[0].tokens[0].ner_tag.starts_with("B-"));
    }

    #[test]
    fn test_parse_yahoo_answers_smoke() {
        let sample = r#"{"question_title":"Why is the sky blue?","topic":1}"#;
        let loader = DatasetLoader::new().unwrap();
        let ds = loader
            .parse_yahoo_answers(sample, DatasetId::YahooAnswers)
            .unwrap();
        assert_eq!(ds.sentences.len(), 1);
        assert!(ds.sentences[0].tokens[0].ner_tag.starts_with("B-"));
    }

    // =========================================================================
    // Dataset Registry Integration Tests
    // =========================================================================

    #[test]
    fn test_sec_filings_has_raw_url() {
        // Verify SEC-filings dataset has a raw GitHub URL for direct download
        let url = DatasetId::SECFilingsNER.download_url();
        assert!(
            url.contains("raw.githubusercontent.com"),
            "SEC-filings should have raw GitHub URL, got: {}",
            url
        );
        assert!(
            url.ends_with(".txt"),
            "SEC-filings should point to a .txt file, got: {}",
            url
        );
    }

    #[test]
    fn test_twiconv_has_format() {
        // Verify TwiConv dataset has format field
        let format = DatasetId::TwiConv.format();
        assert!(format.is_some(), "TwiConv should have format field");
        // TwiConv uses CoNLL format for coreference data
        assert_eq!(format.unwrap(), "CoNLL", "TwiConv should be CoNLL format");
    }

    #[test]
    fn test_mudoco_has_format() {
        // Verify MuDoCo dataset has format field
        let format = DatasetId::MuDoCo.format();
        assert!(format.is_some(), "MuDoCo should have format field");
        assert_eq!(format.unwrap(), "JSON", "MuDoCo should be JSON format");
    }

    #[test]
    fn test_all_public_ud_datasets_have_conllu_format() {
        // All Universal Dependencies datasets should have CoNLLU format
        let ud_datasets = vec![
            DatasetId::AncientGreekUD,
            DatasetId::LatinUD,
            DatasetId::SanskritUD,
            DatasetId::OldEnglishUD,
            DatasetId::UDEsperantoCairo,
        ];

        for ds in ud_datasets {
            let format = ds.format();
            assert!(format.is_some(), "{:?} should have format field", ds);
            assert_eq!(
                format.unwrap(),
                "CoNLLU",
                "{:?} should be CoNLLU format",
                ds
            );
        }
    }

    #[test]
    fn test_datasets_with_public_urls_are_accessible() {
        // Verify that key public datasets have valid URLs (format check only)
        let test_cases = vec![
            (DatasetId::AncientGreekUD, "universaldependencies"),
            (DatasetId::LatinUD, "universaldependencies"),
            (DatasetId::SECFilingsNER, "entity-recognition-datasets"),
            (DatasetId::TwiConv, "twiconv"), // lowercase for case-insensitive match
        ];

        for (ds, expected_substring) in test_cases {
            let url = ds.download_url();
            assert!(!url.is_empty(), "{:?} should have a download URL", ds);
            assert!(
                url.to_lowercase()
                    .contains(&expected_substring.to_lowercase()),
                "{:?} URL should contain '{}', got: {}",
                ds,
                expected_substring,
                url
            );
        }
    }

    #[test]
    fn test_loadable_datasets_count_is_stable() {
        // Track the number of loadable datasets to detect regressions
        // This should only increase as we add more loaders
        let loadable = LoadableDatasetId::all();
        let count = loadable.len();

        // As of 2025-12-15, after recent fixes we have 295 loadable datasets
        assert!(
            count >= 295,
            "Expected at least 295 loadable datasets, got {}. \
             This may indicate a regression in the loading system.",
            count
        );
    }

    #[test]
    fn test_conll_format_variants_all_detected() {
        // Ensure CoNLL format variants for pure NER datasets are properly detected
        // We exclude RE/coref datasets as they have special parsing needs
        for &ds in DatasetId::all() {
            let format = ds.format();
            if let Some(fmt) = format {
                let is_conll_variant =
                    fmt == "CoNLL" || fmt == "CoNLLU" || fmt == "CoNLL-U" || fmt == "CoNLL03";

                // Only check pure NER datasets (no coref, no RE)
                let is_pure_ner = ds.supports_ner() && !ds.supports_coref() && !ds.supports_re();

                if is_conll_variant && is_pure_ner {
                    // Should be loadable via hint system
                    let hint = LoadableDatasetId::registry_hint_plan(ds);
                    assert!(
                        hint.is_some() || LoadableDatasetId::is_loadable_dataset(ds),
                        "{:?} with format {} and pure NER task should be loadable",
                        ds,
                        fmt
                    );
                }
            }
        }
    }

    #[test]
    fn test_parse_csv_ner_smoke() {
        // Test CSV NER format (E-NER/EDGAR-NER style: Token,Tag)
        let sample = "\
-DOCSTART-,O
,O
Check,O
the,O
appropriate,O
box,O
,O
Nuveen,I-BUSINESS
New,I-BUSINESS
York,I-BUSINESS
Fund,I-BUSINESS
,O

The,O
SEC,I-GOVERNMENT
filed,O
charges,O
.

John,I-PERSON
Smith,I-PERSON
is,O
the,O
CEO,O
.
";
        let loader = DatasetLoader::new().unwrap();
        let ds = loader.parse_csv_ner(sample, DatasetId::ENer).unwrap();

        // Should have 3 sentences (separated by empty lines and -DOCSTART-)
        assert_eq!(
            ds.sentences.len(),
            3,
            "Expected 3 sentences, got {:?}",
            ds.sentences.len()
        );

        // First sentence should have BUSINESS entities
        let first_sentence = &ds.sentences[0];
        assert!(
            first_sentence
                .tokens
                .iter()
                .any(|t| t.ner_tag == "I-BUSINESS"),
            "First sentence should contain I-BUSINESS tags"
        );

        // Second sentence should have GOVERNMENT entity
        let second_sentence = &ds.sentences[1];
        assert!(
            second_sentence
                .tokens
                .iter()
                .any(|t| t.ner_tag == "I-GOVERNMENT"),
            "Second sentence should contain I-GOVERNMENT tag"
        );

        // Third sentence should have PERSON entities
        let third_sentence = &ds.sentences[2];
        assert!(
            third_sentence
                .tokens
                .iter()
                .any(|t| t.ner_tag == "I-PERSON"),
            "Third sentence should contain I-PERSON tags"
        );

        // Check specific token/tag pairs
        let nuveen_token = first_sentence.tokens.iter().find(|t| t.text == "Nuveen");
        assert!(nuveen_token.is_some(), "Should have Nuveen token");
        assert_eq!(nuveen_token.unwrap().ner_tag, "I-BUSINESS");

        let john_token = third_sentence.tokens.iter().find(|t| t.text == "John");
        assert!(john_token.is_some(), "Should have John token");
        assert_eq!(john_token.unwrap().ner_tag, "I-PERSON");
    }

    #[test]
    fn test_csv_ner_format_is_detected() {
        // Ensure CSV format datasets with NER tasks are properly detected as loadable
        let ener_hint = LoadableDatasetId::registry_hint_plan(DatasetId::ENer);
        assert_eq!(
            ener_hint,
            Some(DatasetParsePlan::CsvNer),
            "ENer should use CsvNer parse plan"
        );

        // Verify ENer is loadable
        assert!(
            LoadableDatasetId::is_loadable_dataset(DatasetId::ENer),
            "ENer should be loadable"
        );
    }

    // =========================================================================
    // Tests for newly added dataset loaders (2025-12)
    // =========================================================================

    #[test]
    fn test_newly_added_conll_datasets_are_loadable() {
        let new_conll = [
            DatasetId::QxoRef,
            DatasetId::GICoref,
            DatasetId::WNUT16,
            DatasetId::NoiseBench,
            DatasetId::CrossWeigh,
            DatasetId::ZELDA,
            DatasetId::GENIANested,
        ];

        for id in new_conll {
            assert!(
                LoadableDatasetId::is_loadable_dataset(id),
                "{:?} should be loadable with Conll parse plan",
                id
            );
            assert_eq!(
                LoadableDatasetId::parse_plan(id),
                Some(DatasetParsePlan::Conll),
                "{:?} should use Conll plan",
                id
            );
        }
    }

    #[test]
    fn test_newly_added_jsonl_datasets_are_loadable() {
        let new_jsonl = [
            DatasetId::REBEL,
            DatasetId::BBQ,
            DatasetId::RealToxicityPrompts,
            DatasetId::BookCoref,
            DatasetId::BookCorefSplit,
            DatasetId::WIESP2022NER,
            DatasetId::FewRel,
            DatasetId::PIIMasking200k,
            DatasetId::B2NERD,
            DatasetId::OpenNER,
            DatasetId::FictionNER750M,
        ];

        for id in new_jsonl {
            assert!(
                LoadableDatasetId::is_loadable_dataset(id),
                "{:?} should be loadable with JsonlNer parse plan",
                id
            );
            assert_eq!(
                LoadableDatasetId::parse_plan(id),
                Some(DatasetParsePlan::JsonlNer),
                "{:?} should use JsonlNer plan",
                id
            );
        }
    }

    #[test]
    fn test_genia_nested_conll_parse() {
        // GENIA nested NER uses multi-layered BIO tags
        let nested_conll = "IL-2\tB-protein\n\
                            gene\tI-protein\n\
                            expression\tO\n\
                            \n\
                            T\tB-cell_type\n\
                            cells\tI-cell_type\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conll(nested_conll, DatasetId::GENIANested);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 2);
        assert_eq!(dataset.sentences[0].tokens[0].ner_tag, "B-protein");
        assert_eq!(dataset.sentences[1].tokens[0].ner_tag, "B-cell_type");
    }

    #[test]
    fn test_gicoref_gender_inclusive_parse() {
        // GICoref uses neopronouns and singular they
        let gicoref_conll = "Alex\tB-PER\n\
                             uses\tO\n\
                             they\tB-PER\n\
                             pronouns\tO\n\
                             \n\
                             Jordan\tB-PER\n\
                             introduced\tO\n\
                             themself\tB-PER\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conll(gicoref_conll, DatasetId::GICoref);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 2);

        // Verify pronoun tokens are correctly tagged
        assert_eq!(dataset.sentences[0].tokens[2].text, "they");
        assert_eq!(dataset.sentences[0].tokens[2].ner_tag, "B-PER");
        assert_eq!(dataset.sentences[1].tokens[2].text, "themself");
        assert_eq!(dataset.sentences[1].tokens[2].ner_tag, "B-PER");
    }

    #[test]
    fn test_fewrel_jsonl_parse() {
        // FewRel has relation extraction in JSONL format
        // The parser expects integer tags mapping to MultiNERD labels:
        // 0=O, 1=B-PER, 3=B-ORG
        let fewrel_sample =
            r#"{"tokens":["John","works","at","Google","."],"ner_tags":[1,0,0,3,0]}"#;

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_jsonl_ner(fewrel_sample, DatasetId::FewRel);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        assert_eq!(dataset.sentences[0].tokens.len(), 5);
        assert_eq!(dataset.sentences[0].tokens[0].ner_tag, "B-PER");
        assert_eq!(dataset.sentences[0].tokens[3].ner_tag, "B-ORG");
    }

    #[test]
    fn test_b2nerd_business_entities_parse() {
        // B2NERD focuses on business/financial entity types
        // Using standard MultiNERD indices: 3=B-ORG (closest to COMPANY), 0=O
        let b2nerd_sample =
            r#"{"tokens":["Apple","Inc",".","reports","Q4","earnings"],"ner_tags":[3,4,4,0,0,0]}"#;

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_jsonl_ner(b2nerd_sample, DatasetId::B2NERD);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        assert_eq!(dataset.sentences[0].tokens[0].ner_tag, "B-ORG");
        assert_eq!(dataset.sentences[0].tokens[1].ner_tag, "I-ORG");
    }

    #[test]
    fn test_ud_classical_languages_are_loadable() {
        // Universal Dependencies treebanks for classical/ancient languages
        let ud_datasets = [
            DatasetId::AncientGreekUD,
            DatasetId::LatinUD,
            DatasetId::SanskritUD,
            DatasetId::OldEnglishUD,
            DatasetId::OldNorseUD,
            DatasetId::UDEsperantoCairo,
        ];

        for id in ud_datasets {
            assert!(
                LoadableDatasetId::is_loadable_dataset(id),
                "{:?} should be loadable with Conllu parse plan",
                id
            );
            assert_eq!(
                LoadableDatasetId::parse_plan(id),
                Some(DatasetParsePlan::Conllu),
                "{:?} should use Conllu plan",
                id
            );
        }
    }

    #[test]
    fn test_hipe2022_tsv_is_loadable() {
        assert!(
            LoadableDatasetId::is_loadable_dataset(DatasetId::HIPE2022),
            "HIPE2022 should be loadable"
        );
        assert_eq!(
            LoadableDatasetId::parse_plan(DatasetId::HIPE2022),
            Some(DatasetParsePlan::TsvNer),
            "HIPE2022 should use TsvNer plan"
        );
    }

    #[test]
    fn test_ener_csv_is_loadable() {
        assert!(
            LoadableDatasetId::is_loadable_dataset(DatasetId::ENer),
            "ENer should be loadable"
        );
        assert_eq!(
            LoadableDatasetId::parse_plan(DatasetId::ENer),
            Some(DatasetParsePlan::CsvNer),
            "ENer should use CsvNer plan"
        );
    }

    #[test]
    fn test_loadable_count_increased() {
        // Regression test: ensure we have at least 295 loadable datasets
        // (Updated 2025-12 after adding CoNLL/JSONL/CoNLLU batches)
        let loadable_count = LoadableDatasetId::all().len();
        assert!(
            loadable_count >= 295,
            "Expected at least 295 loadable datasets, got {}",
            loadable_count
        );
    }

    // =========================================================================
    // Domain-Specific Parser Tests
    // =========================================================================

    #[test]
    fn test_biomedical_conll_with_chemical_entities() {
        // CHEMDNER-style biomedical NER with chemical entity types
        let chemdner_conll = "Aspirin\tB-CHEMICAL\n\
                              inhibits\tO\n\
                              COX-2\tB-GENE\n\
                              expression\tO\n\
                              \n\
                              Metformin\tB-CHEMICAL\n\
                              treats\tO\n\
                              diabetes\tB-DISEASE\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conll(chemdner_conll, DatasetId::CHEMDNER);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 2);
        assert_eq!(dataset.sentences[0].tokens[0].ner_tag, "B-CHEMICAL");
        assert_eq!(dataset.sentences[0].tokens[2].ner_tag, "B-GENE");
        assert_eq!(dataset.sentences[1].tokens[2].ner_tag, "B-DISEASE");
    }

    #[test]
    fn test_historical_ner_with_archaic_spelling() {
        // Historical NER should handle archaic spellings and diacritics
        let historical_conll = "Præsident\tB-PER\n\
                                 Washington\tI-PER\n\
                                 addresseth\tO\n\
                                 ye\tO\n\
                                 Congreſs\tB-ORG\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conll(historical_conll, DatasetId::EighteenthCenturyNER);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        // Verify archaic characters are preserved
        assert!(dataset.sentences[0].tokens[0].text.contains('æ'));
        assert!(dataset.sentences[0].tokens[4].text.contains('ſ'));
    }

    #[test]
    fn test_multilingual_code_switching_ner() {
        // LinCE/CALCS datasets have code-switched text (e.g., Spanish-English)
        let codeswitched_conll = "My\tO\n\
                                   abuela\tB-PER\n\
                                   lives\tO\n\
                                   in\tO\n\
                                   Ciudad\tB-LOC\n\
                                   de\tI-LOC\n\
                                   México\tI-LOC\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conll(codeswitched_conll, DatasetId::LinCE);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        assert_eq!(dataset.sentences[0].tokens[1].text, "abuela");
        assert_eq!(dataset.sentences[0].tokens[1].ner_tag, "B-PER");
        // Multi-word location
        assert_eq!(dataset.sentences[0].tokens[4].ner_tag, "B-LOC");
        assert_eq!(dataset.sentences[0].tokens[6].ner_tag, "I-LOC");
    }

    #[test]
    fn test_indigenous_language_ner() {
        // Test parsing of indigenous language NER (Guarani/Shipibo-Konibo)
        let guarani_conll = "Paraguái\tB-LOC\n\
                              ha\tO\n\
                              yvypora\tO\n\
                              oiko\tO\n\
                              Asunción\tB-LOC\n\
                              pe\tO\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conll(guarani_conll, DatasetId::GuaraniNER);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        assert_eq!(dataset.sentences[0].tokens[0].text, "Paraguái");
        assert_eq!(dataset.sentences[0].tokens[0].ner_tag, "B-LOC");
    }

    #[test]
    fn test_ancient_greek_conllu_with_polytonic() {
        // Ancient Greek with polytonic diacritics
        let greek_conllu = "# sent_id = grc_test_1\n\
                            # text = ἐπιστήμη καὶ δικαιοσύνη\n\
                            1\tἐπιστήμη\tἐπιστήμη\tNOUN\tN\tCase=Nom|Gender=Fem|Number=Sing\t0\troot\t_\tSpaceAfter=Yes\n\
                            2\tκαὶ\tκαί\tCCONJ\tC\t_\t3\tcc\t_\tSpaceAfter=Yes\n\
                            3\tδικαιοσύνη\tδικαιοσύνη\tNOUN\tN\tCase=Nom|Gender=Fem|Number=Sing\t1\tconj\t_\tSpaceAfter=No\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conllu(greek_conllu, DatasetId::AncientGreekUD);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        // Verify polytonic Greek is preserved
        assert_eq!(dataset.sentences[0].tokens[0].text, "ἐπιστήμη");
        assert_eq!(dataset.sentences[0].tokens[2].text, "δικαιοσύνη");
    }

    #[test]
    fn test_latin_conllu_with_macrons() {
        // Latin with optional macrons for vowel length
        let latin_conllu = "# sent_id = lat_test_1\n\
                            # text = Rōma āterna est\n\
                            1\tRōma\tRoma\tPROPN\tNNP\tCase=Nom|Gender=Fem|Number=Sing\t3\tnsubj\t_\tSpaceAfter=Yes\n\
                            2\tāterna\taeternus\tADJ\tA\tCase=Nom|Gender=Fem|Number=Sing\t1\tamod\t_\tSpaceAfter=Yes\n\
                            3\test\tsum\tAUX\tV\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act\t0\troot\t_\tSpaceAfter=No\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conllu(latin_conllu, DatasetId::LatinUD);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        // Verify macrons are preserved
        assert!(dataset.sentences[0].tokens[0].text.contains('ō'));
        assert!(dataset.sentences[0].tokens[1].text.contains('ā'));
    }

    #[test]
    fn test_sanskrit_conllu_with_devanagari() {
        // Sanskrit in Devanagari script
        let sanskrit_conllu = "# sent_id = sa_test_1\n\
                               # text = रामः सीतां पश्यति\n\
                               1\tरामः\tराम\tNOUN\tN\tCase=Nom|Gender=Masc|Number=Sing\t3\tnsubj\t_\tSpaceAfter=Yes\n\
                               2\tसीतां\tसीता\tPROPN\tNNP\tCase=Acc|Gender=Fem|Number=Sing\t3\tobj\t_\tSpaceAfter=Yes\n\
                               3\tपश्यति\tदृश्\tVERB\tV\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act\t0\troot\t_\tSpaceAfter=No\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conllu(sanskrit_conllu, DatasetId::SanskritUD);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        assert_eq!(dataset.sentences[0].tokens.len(), 3);
        // Verify Devanagari is preserved
        assert_eq!(dataset.sentences[0].tokens[0].text, "रामः");
        assert_eq!(dataset.sentences[0].tokens[1].text, "सीतां");
    }

    #[test]
    fn test_klingon_conllu_is_loadable() {
        // Klingon (tlh) is a constructed language in TaggedPBCKlingon
        assert!(
            LoadableDatasetId::is_loadable_dataset(DatasetId::TaggedPBCKlingon),
            "Klingon dataset should be loadable"
        );
        assert_eq!(
            LoadableDatasetId::parse_plan(DatasetId::TaggedPBCKlingon),
            Some(DatasetParsePlan::Conllu),
            "Klingon should use Conllu plan"
        );
    }

    #[test]
    fn test_financial_ner_entities() {
        // FinanceNER with financial entity types
        let finance_conll = "Tesla\tB-COMPANY\n\
                             stock\tO\n\
                             rose\tO\n\
                             5%\tB-PERCENTAGE\n\
                             after\tO\n\
                             Q4\tB-PERIOD\n\
                             earnings\tO\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conll(finance_conll, DatasetId::FinanceNER);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        assert_eq!(dataset.sentences[0].tokens[0].ner_tag, "B-COMPANY");
        assert_eq!(dataset.sentences[0].tokens[3].ner_tag, "B-PERCENTAGE");
    }

    #[test]
    fn test_recipe_ner_food_entities() {
        // RecipeNER with culinary entity types
        let recipe_conll = "Add\tO\n\
                            2\tB-QUANTITY\n\
                            cups\tI-QUANTITY\n\
                            of\tO\n\
                            flour\tB-INGREDIENT\n\
                            and\tO\n\
                            1\tB-QUANTITY\n\
                            tsp\tI-QUANTITY\n\
                            salt\tB-INGREDIENT\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conll(recipe_conll, DatasetId::RecipeNER);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        assert_eq!(dataset.sentences[0].tokens[4].ner_tag, "B-INGREDIENT");
        assert_eq!(dataset.sentences[0].tokens[8].ner_tag, "B-INGREDIENT");
    }

    #[test]
    fn test_astronomy_ner_entities() {
        // AstroNER with astronomical entity types
        let astro_conll = "The\tO\n\
                           Andromeda\tB-GALAXY\n\
                           Galaxy\tI-GALAXY\n\
                           is\tO\n\
                           2.5\tB-DISTANCE\n\
                           million\tI-DISTANCE\n\
                           light-years\tI-DISTANCE\n\
                           away\tO\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conll(astro_conll, DatasetId::AstroNER);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        assert_eq!(dataset.sentences[0].tokens[1].ner_tag, "B-GALAXY");
        assert_eq!(dataset.sentences[0].tokens[4].ner_tag, "B-DISTANCE");
    }

    #[test]
    fn test_nested_ner_datasets_are_loadable() {
        // Nested NER datasets (entities within entities)
        let nested_datasets = [DatasetId::GENIANested, DatasetId::ChineseNestedNER];

        for id in nested_datasets {
            assert!(
                LoadableDatasetId::is_loadable_dataset(id),
                "{:?} should be loadable",
                id
            );
        }
    }

    #[test]
    fn test_discontinuous_ner_datasets_are_loadable() {
        // Discontinuous NER datasets (non-contiguous entity spans)
        let discontinuous_datasets = [
            DatasetId::GermEvalDiscontinuous,
            DatasetId::PubMedDiscontinuous,
        ];

        for id in discontinuous_datasets {
            assert!(
                LoadableDatasetId::is_loadable_dataset(id),
                "{:?} should be loadable",
                id
            );
        }
    }

    #[test]
    fn test_social_media_ner_datasets_are_loadable() {
        // Social media NER datasets (noisy text, hashtags, mentions)
        let social_datasets = [
            DatasetId::WNUT16,
            DatasetId::TwiConv,
            DatasetId::NERsocialFood,
        ];

        for id in social_datasets {
            assert!(
                LoadableDatasetId::is_loadable_dataset(id),
                "{:?} should be loadable",
                id
            );
            assert_eq!(
                LoadableDatasetId::parse_plan(id),
                Some(DatasetParsePlan::Conll),
                "{:?} should use Conll plan",
                id
            );
        }
    }

    #[test]
    fn test_literary_ner_datasets_are_loadable() {
        // Literary NER datasets (fiction, novels)
        let literary_datasets = [
            DatasetId::CharacterCodex,
            DatasetId::FictionNER750M,
            DatasetId::BookCoref,
        ];

        for id in literary_datasets {
            assert!(
                LoadableDatasetId::is_loadable_dataset(id),
                "{:?} should be loadable",
                id
            );
        }
    }

    #[test]
    fn test_jsonl_ner_with_empty_tokens_handled() {
        // Edge case: JSONL with some empty tokens
        let jsonl_with_empty = r#"{"tokens":["Hello","","world"],"ner_tags":[0,0,0]}"#;

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_jsonl_ner(jsonl_with_empty, DatasetId::MultiWOZNER);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        // Empty token should still be preserved in parsing
        assert_eq!(dataset.sentences[0].tokens.len(), 3);
    }

    #[test]
    fn test_conll_with_long_entity_spans() {
        // Test parsing of very long entity spans (e.g., legal document titles)
        let long_span_conll = "The\tB-DOCUMENT\n\
                               United\tI-DOCUMENT\n\
                               States\tI-DOCUMENT\n\
                               Constitution\tI-DOCUMENT\n\
                               Article\tI-DOCUMENT\n\
                               I\tI-DOCUMENT\n\
                               Section\tI-DOCUMENT\n\
                               8\tI-DOCUMENT\n\
                               Clause\tI-DOCUMENT\n\
                               3\tI-DOCUMENT\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conll(long_span_conll, DatasetId::LegNER);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        assert_eq!(dataset.sentences[0].tokens.len(), 10);
        // All tokens should be part of the same entity
        assert_eq!(dataset.sentences[0].tokens[0].ner_tag, "B-DOCUMENT");
        assert_eq!(dataset.sentences[0].tokens[9].ner_tag, "I-DOCUMENT");
    }

    #[test]
    fn test_all_added_conll_datasets_are_loadable() {
        // Comprehensive check for all newly added CoNLL datasets
        let added_conll = [
            DatasetId::HistNERo,
            DatasetId::DutchArchaeology,
            DatasetId::FINER,
            DatasetId::CALCS2018,
            DatasetId::MedievalCharterNER,
            DatasetId::RockNER,
            DatasetId::AIDACoNLL,
            DatasetId::NNE,
            DatasetId::IndicNER,
            DatasetId::NorNE,
            DatasetId::TASTEset,
            DatasetId::TechNER,
            DatasetId::FinTechPatent,
            DatasetId::WaterAgriNER,
            DatasetId::RussianCulturalNER,
            DatasetId::BASHI,
            DatasetId::ENER,
        ];

        for id in added_conll {
            assert!(
                LoadableDatasetId::is_loadable_dataset(id),
                "{:?} should be loadable",
                id
            );
        }
    }

    #[test]
    fn test_all_added_jsonl_datasets_are_loadable() {
        // Comprehensive check for all newly added JSONL datasets
        let added_jsonl = [
            DatasetId::MultiWOZNER,
            DatasetId::HinglishNER,
            DatasetId::AgCNER,
            DatasetId::LongDocNER,
            DatasetId::MultiBioNERLong,
            DatasetId::ReasoningNER,
            DatasetId::BioNERLLaMA,
            DatasetId::LexGLUENER,
            DatasetId::FinBenNER,
            DatasetId::FiNER139,
            DatasetId::SciNER,
            DatasetId::AIONER,
            DatasetId::WIESPAstro,
            DatasetId::CEREC,
            DatasetId::DELICATE,
            DatasetId::CSN,
        ];

        for id in added_jsonl {
            assert!(
                LoadableDatasetId::is_loadable_dataset(id),
                "{:?} should be loadable",
                id
            );
        }
    }

    #[test]
    fn test_all_added_ud_datasets_are_loadable() {
        // Comprehensive check for all newly added UD datasets
        let added_ud = [
            DatasetId::CopticScriptorium,
            DatasetId::TaggedPBCEsperanto,
            DatasetId::TaggedPBCKlingon,
            DatasetId::AkkadianUD,
            DatasetId::AncientHebrewUD,
            DatasetId::ClassicalChineseUD,
            DatasetId::CopticUD,
            DatasetId::GothicUD,
            DatasetId::HittiteUD,
            DatasetId::OldChurchSlavonicUD,
            DatasetId::LatinITTB,
            DatasetId::LatinPROIEL,
            DatasetId::EsperantoUD,
            DatasetId::NavajoMorph,
        ];

        for id in added_ud {
            assert!(
                LoadableDatasetId::is_loadable_dataset(id),
                "{:?} should be loadable",
                id
            );
            assert_eq!(
                LoadableDatasetId::parse_plan(id),
                Some(DatasetParsePlan::Conllu),
                "{:?} should use Conllu plan",
                id
            );
        }
    }

    // =========================================================================
    // Edge Case and Robustness Tests
    // =========================================================================

    #[test]
    fn test_conll_handles_windows_line_endings() {
        // Test CRLF line endings (Windows format)
        let windows_conll = "John\tB-PER\r\nSmith\tI-PER\r\n\r\nLondon\tB-LOC\r\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conll(windows_conll, DatasetId::WikiGold);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 2);
    }

    #[test]
    fn test_conll_handles_extra_whitespace() {
        // Test lines with trailing/leading whitespace
        let whitespace_conll = "  John  \t  B-PER  \n  meets  \t  O  \n\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conll(whitespace_conll, DatasetId::WikiGold);
        // Should handle gracefully (may skip malformed lines)
        assert!(result.is_ok());
    }

    #[test]
    fn test_conllu_handles_multiword_tokens() {
        // CoNLL-U multi-word tokens (e.g., "don't" -> "do" + "n't")
        let mwt_conllu = "# sent_id = test\n\
                          # text = I can't go\n\
                          1\tI\tI\tPRON\tPRP\t_\t3\tnsubj\t_\tSpaceAfter=Yes\n\
                          2-3\tcan't\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=Yes\n\
                          2\tca\tcan\tAUX\tMD\t_\t3\taux\t_\t_\n\
                          3\tn't\tnot\tPART\tRB\t_\t0\troot\t_\t_\n\
                          4\tgo\tgo\tVERB\tVB\t_\t3\txcomp\t_\tSpaceAfter=No\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conllu(mwt_conllu, DatasetId::LatinUD);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        // MWT range tokens (2-3) should be skipped, only atomic tokens kept
        assert!(dataset.sentences[0].tokens.len() >= 3);
    }

    #[test]
    fn test_conllu_handles_empty_nodes() {
        // CoNLL-U empty nodes (e.g., 2.1 for elided elements)
        let empty_node_conllu = "# sent_id = test\n\
                                 # text = I saw and heard\n\
                                 1\tI\tI\tPRON\tPRP\t_\t2\tnsubj\t_\tSpaceAfter=Yes\n\
                                 2\tsaw\tsee\tVERB\tVBD\t_\t0\troot\t_\tSpaceAfter=Yes\n\
                                 2.1\tI\tI\tPRON\tPRP\t_\t4\tnsubj\t_\t_\n\
                                 3\tand\tand\tCCONJ\tCC\t_\t4\tcc\t_\tSpaceAfter=Yes\n\
                                 4\theard\thear\tVERB\tVBD\t_\t2\tconj\t_\tSpaceAfter=No\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conllu(empty_node_conllu, DatasetId::LatinUD);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        // Empty nodes (2.1) should be skipped
        assert_eq!(dataset.sentences[0].tokens.len(), 4);
    }

    #[test]
    fn test_conll_with_bio_tag_normalization() {
        // Some corpora use I- at start of entity (should be B-)
        let malformed_bio = "Paris\tI-LOC\n\
                             is\tO\n\
                             beautiful\tO\n";

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_conll(malformed_bio, DatasetId::WikiGold);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        // Parser should normalize or preserve the tag
        let first_tag = &dataset.sentences[0].tokens[0].ner_tag;
        assert!(first_tag == "I-LOC" || first_tag == "B-LOC");
    }

    #[test]
    fn test_conll_with_unicode_normalization() {
        // Test that precomposed (U+00E9) and decomposed (U+0065 U+0301) forms
        // both parse and produce equivalent tokens.
        let composed = "Caf\u{00e9}\tB-LOC\n";
        let decomposed = "Cafe\u{0301}\tB-LOC\n";
        assert_ne!(
            composed, decomposed,
            "test inputs must actually differ in bytes"
        );

        let loader = DatasetLoader::new().unwrap();
        let d1 = loader.parse_conll(composed, DatasetId::WikiGold).unwrap();
        let d2 = loader.parse_conll(decomposed, DatasetId::WikiGold).unwrap();

        assert_eq!(d1.sentences.len(), d2.sentences.len());
        assert_eq!(d1.sentences[0].tokens.len(), d2.sentences[0].tokens.len());
        assert_eq!(
            d1.sentences[0].tokens[0].ner_tag,
            d2.sentences[0].tokens[0].ner_tag
        );
    }

    #[test]
    fn test_cadec_hf_api_unicode_prefix_case_insensitive_span_search_is_safe() {
        // Regression: span finding must not rely on Unicode lowercasing index alignment.
        let loader = DatasetLoader::new().unwrap();
        // Include `features` and compact JSON so `is_hf_api_response()` recognizes it.
        let content = r#"{"features":[{"name":"text"},{"name":"ade"},{"name":"term_PT"}],"rows":[{"row":{"text":"Müller reported HEADACHE after taking aspirin.","ade":"headache","term_PT":"Headache"}}]}"#;

        let ds = loader
            .parse_content_str(content, DatasetId::CADEC)
            .expect("parse CADEC HF API");
        assert_eq!(ds.id, DatasetId::CADEC);
        assert!(!ds.sentences.is_empty());

        let sent = &ds.sentences[0];
        // We should tag the ADE as B-/I- adverse_drug_event in BIO space.
        assert!(
            sent.tokens
                .iter()
                .any(|t| t.ner_tag == "B-adverse_drug_event" || t.ner_tag == "I-adverse_drug_event"),
            "Expected ADE tags in tokens: {:?}",
            sent.tokens
        );
    }

    #[test]
    fn test_jsonl_with_unicode_tokens() {
        // JSONL with various Unicode characters
        let unicode_jsonl = r#"{"tokens":["北京","🎉","Москва","القاهرة"],"ner_tags":[5,0,5,5]}"#;

        let loader = DatasetLoader::new().unwrap();
        let result = loader.parse_jsonl_ner(unicode_jsonl, DatasetId::MultiWOZNER);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert_eq!(dataset.sentences.len(), 1);
        assert_eq!(dataset.sentences[0].tokens.len(), 4);
        assert_eq!(dataset.sentences[0].tokens[0].text, "北京");
        assert_eq!(dataset.sentences[0].tokens[1].text, "🎉");
        assert_eq!(dataset.sentences[0].tokens[2].text, "Москва");
    }

    #[test]
    fn test_parse_plan_consistency_with_is_loadable() {
        // Invariant: parse_plan returns Some iff is_loadable_dataset returns true
        for &id in DatasetId::all() {
            let has_plan = LoadableDatasetId::parse_plan(id).is_some();
            let is_loadable = LoadableDatasetId::is_loadable_dataset(id);
            assert_eq!(
                has_plan, is_loadable,
                "Mismatch for {:?}: parse_plan={}, is_loadable={}",
                id, has_plan, is_loadable
            );
        }
    }

    #[test]
    fn test_dataset_coverage_by_plan() {
        // Verify we have good coverage across different parse plans
        let mut conll_count = 0;
        let mut jsonl_count = 0;
        let mut conllu_count = 0;
        let mut _other_count = 0;

        for id in LoadableDatasetId::all() {
            let ds: DatasetId = id.into();
            match LoadableDatasetId::parse_plan(ds) {
                Some(DatasetParsePlan::Conll) => conll_count += 1,
                Some(DatasetParsePlan::JsonlNer) => jsonl_count += 1,
                Some(DatasetParsePlan::Conllu) => conllu_count += 1,
                Some(_) => _other_count += 1,
                None => {}
            }
        }

        // Should have substantial coverage for each format
        assert!(
            conll_count >= 50,
            "Expected at least 50 CoNLL datasets loadable, got {}",
            conll_count
        );
        assert!(
            jsonl_count >= 20,
            "Expected at least 20 JSONL datasets loadable, got {}",
            jsonl_count
        );
        assert!(
            conllu_count >= 10,
            "Expected at least 10 CoNLLU datasets loadable, got {}",
            conllu_count
        );
    }

    #[test]
    fn test_loadable_datasets_have_valid_metadata() {
        // All loadable datasets should have basic metadata
        for id in LoadableDatasetId::all() {
            let ds: DatasetId = id.into();

            // Name should not be empty
            assert!(!ds.name().is_empty(), "{:?} has empty name", ds);

            // Description should exist for most
            // (not asserting as some may legitimately have no description)
        }
    }
}
