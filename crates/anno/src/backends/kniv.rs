//! English Kniv v7.1 inference with a persistent local Python/ONNX worker.
use super::inference::{ExtractionWithRelations, RelationExtractor, RelationTriple};
use super::python_worker::PythonWorker;
use crate::{Entity, EntityType, Error, ExtractionMethod, Language, Model, Provenance, Result};
use serde::{Deserialize, Serialize};
use std::ffi::OsStr;
use std::path::Path;

const VERSION: &str = "kniv-v7-large-en@44e6d50774a22ada4fd825e1c48663afb5fd7dc0";
const TAGS: &[&str] = &[
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

/// Word-level BIO prediction grounded in half-open source character offsets.
#[derive(Debug, Serialize, Deserialize)]
pub struct KnivMention {
    /// Upstream type label, before Anno normalization.
    pub label: String,
    /// Half-open character offsets in the original document.
    pub span: [usize; 2],
    /// Geometric mean of the decoded tags' token softmax scores.
    pub confidence: f64,
}

/// Kniv fact endpoint with every predicted occurrence.
#[derive(Debug, Serialize, Deserialize)]
pub struct KnivNode {
    /// Upstream type label, before Anno normalization.
    pub label: String,
    /// Every occurrence of this node as a half-open character range.
    pub mentions: Vec<[usize; 2]>,
}

/// A directed edge between local nodes. The raw margin compares the relation with the null label.
#[derive(Debug, Serialize, Deserialize)]
pub struct KnivFact {
    /// Index of the head in this window’s nodes.
    pub head: usize,
    /// Requested relation label selected by the model.
    pub relation: String,
    /// Index of the tail in this window’s nodes.
    pub tail: usize,
    /// Calibrated margin above the null label, rounded by the upstream decoder.
    pub margin: f64,
    /// Upstream polarity, such as positive or negative.
    pub polarity: String,
    /// Upstream modality, such as asserted, planned or hypothetical.
    pub modality: String,
    /// Optional node index for the attributed holder.
    pub holder: Option<usize>,
    /// Optional node index for the temporal qualifier.
    pub time: Option<usize>,
}

/// Predictions for one disjoint source window. All ranges are document-global.
#[derive(Debug, Serialize, Deserialize)]
pub struct KnivWindow {
    /// Half-open character offsets in the original document.
    pub span: [usize; 2],
    /// Fixed-label linguistic named entities.
    pub ner: Vec<KnivMention>,
    /// Entity groups produced by the separate relation endpoint head.
    pub nodes: Vec<KnivNode>,
    /// All model-produced facts, including qualified and negated facts.
    pub facts: Vec<KnivFact>,
}

/// Window-local node identities and facts, with document-global occurrence offsets.
#[derive(Debug, Serialize, Deserialize)]
pub struct KnivDocument {
    /// Scored windows in document order.
    pub windows: Vec<KnivWindow>,
}

/// Local English extractor. Each instance retains its ONNX sessions across calls.
pub struct Kniv {
    worker: PythonWorker,
}

impl Kniv {
    /// Start a worker for the pinned English v7.1 release directory.
    pub fn new(python: impl AsRef<OsStr>, directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory
            .as_ref()
            .canonicalize()
            .map_err(|e| Error::model_init(format!("Kniv model directory: {e}")))?;
        Ok(Self {
            worker: PythonWorker::start(
                python.as_ref(),
                include_str!("kniv_worker.py"),
                vec![directory.into_os_string()],
                "Kniv",
            )?,
        })
    }

    /// Read the model location from KNIV_MODEL_DIR and optional interpreter from KNIV_PYTHON.
    pub fn from_env() -> Result<Self> {
        let directory = std::env::var_os("KNIV_MODEL_DIR")
            .ok_or_else(|| Error::model_init("set KNIV_MODEL_DIR to the Kniv release directory"))?;
        Self::new(
            std::env::var_os("KNIV_PYTHON").unwrap_or_else(|| "python3".into()),
            directory,
        )
    }

    /// Run fixed-label NER and retain the full attributed graph for the chosen relation names.
    pub fn analyze(&self, text: &str, relation_types: &[&str]) -> Result<KnivDocument> {
        self.analyze_with_schema(text, &[], relation_types)
    }

    /// Supply custom node labels to the span head. Fixed BIO NER remains available in each window.
    pub fn analyze_with_schema(
        &self,
        text: &str,
        entity_types: &[&str],
        relation_types: &[&str],
    ) -> Result<KnivDocument> {
        if entity_types
            .iter()
            .chain(relation_types)
            .any(|s| s.trim().is_empty())
        {
            return Err(Error::invalid_input("empty Kniv schema label"));
        }
        let doc: KnivDocument = self.worker.request(&serde_json::json!({
            "text": text, "entity_types": entity_types, "relation_types": relation_types,
        }))?;
        doc.validate(text.chars().count())?;
        Ok(doc)
    }
}

impl KnivDocument {
    fn validate(&self, length: usize) -> Result<()> {
        let mut prior_end = 0;
        for w in &self.windows {
            let [lo, hi] = w.span;
            let valid = |[start, end]: [usize; 2]| lo <= start && start < end && end <= hi;
            if !(prior_end <= lo && lo < hi && hi <= length)
                || w.ner
                    .iter()
                    .any(|m| !valid(m.span) || !(0.0..=1.0).contains(&m.confidence))
                || w.nodes
                    .iter()
                    .any(|n| n.mentions.is_empty() || !n.mentions.iter().copied().all(valid))
                || w.facts.iter().any(|f| {
                    [Some(f.head), Some(f.tail), f.holder, f.time]
                        .into_iter()
                        .flatten()
                        .any(|i| i >= w.nodes.len())
                        || !f.margin.is_finite()
                })
            {
                return Err(Error::inference(
                    "Kniv returned invalid source offsets, scores or node indices",
                ));
            }
            prior_end = hi;
        }
        Ok(())
    }

    fn project(&self, text: &str, threshold: f32) -> ExtractionWithRelations {
        let boundaries = crate::offset::build_char_to_byte_map(text);
        let mut out = ExtractionWithRelations::default();
        for window in &self.windows {
            let mut representatives = Vec::with_capacity(window.nodes.len());
            for node in &window.nodes {
                representatives.push(out.entities.len());
                for &[start, end] in &node.mentions {
                    // Upstream supplies cluster membership without a mention probability.
                    let mut entity = Entity::new(
                        &text[boundaries[start]..boundaries[end]],
                        EntityType::from_label(&node.label),
                        start,
                        end,
                        0.0,
                    );
                    entity.provenance = Some(origin());
                    out.entities.push(entity);
                }
            }
            for fact in &window.facts {
                // Generic triples cannot carry negation or modality. The document API retains both.
                let confidence = 1.0 / (1.0 + (-fact.margin).exp());
                if fact.polarity == "positive"
                    && fact.modality == "asserted"
                    && confidence >= f64::from(threshold)
                {
                    out.relations.push(RelationTriple {
                        head_idx: representatives[fact.head],
                        tail_idx: representatives[fact.tail],
                        relation_type: fact.relation.clone(),
                        confidence: confidence.into(),
                    });
                }
            }
        }
        out
    }
}

fn origin() -> Provenance {
    Provenance::from_producer("kniv", ExtractionMethod::Neural).with_version(VERSION)
}

impl crate::sealed::Sealed for Kniv {}

impl Model for Kniv {
    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        let doc = self.analyze(text, &[])?;
        let offsets = crate::offset::build_char_to_byte_map(text);
        Ok(doc
            .windows
            .into_iter()
            .flat_map(|w| w.ner)
            .map(|m| {
                let [start, end] = m.span;
                let mut entity = Entity::new(
                    &text[offsets[start]..offsets[end]],
                    EntityType::from_label(&m.label),
                    start,
                    end,
                    m.confidence,
                );
                entity.provenance =
                    Some(Provenance::ml("kniv", m.confidence).with_version(VERSION));
                entity
            })
            .collect())
    }
    fn name(&self) -> &'static str {
        "kniv"
    }
    fn version(&self) -> String {
        VERSION.into()
    }
    fn description(&self) -> &'static str {
        "English Kniv v7.1 INT8 extraction"
    }
    fn is_available(&self) -> bool {
        self.worker.is_available()
    }
    fn supported_types(&self) -> Vec<EntityType> {
        let mut types = Vec::new();
        for name in TAGS {
            let ty = EntityType::from_label(name);
            if !types.contains(&ty) {
                types.push(ty);
            }
        }
        types
    }
    fn as_relation_extractor(&self) -> Option<&dyn RelationExtractor> {
        Some(self)
    }
}

impl RelationExtractor for Kniv {
    fn extract_with_relations(
        &self,
        text: &str,
        entity_types: &[&str],
        relation_types: &[&str],
        threshold: f32,
    ) -> Result<ExtractionWithRelations> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(Error::invalid_input(
                "Kniv relation threshold must be in [0, 1]",
            ));
        }
        Ok(self
            .analyze_with_schema(text, entity_types, relation_types)?
            .project(text, threshold))
    }
    fn extract_with_relation_provenance(
        &self,
        text: &str,
        entity_types: &[&str],
        relation_types: &[&str],
        threshold: f32,
    ) -> Result<(ExtractionWithRelations, Option<Provenance>)> {
        Ok((
            self.extract_with_relations(text, entity_types, relation_types, threshold)?,
            Some(origin()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_facts_use_global_unicode_offsets_and_keep_qualifiers() {
        let doc: KnivDocument = serde_json::from_value(serde_json::json!({"windows":[
            {"span":[0,1],"ner":[],"nodes":[],"facts":[]},
            {"span":[2,10],"ner":[], "nodes":[{"label":"PER","mentions":[[2,5],[6,7]]},{"label":"LOC","mentions":[[8,10]]}],
            "facts":[{"head":0,"tail":1,"margin":2,"relation":"visited","polarity":"positive","modality":"asserted"},
                     {"head":1,"tail":0,"margin":2,"relation":"hired","polarity":"negative","modality":"asserted"},
                     {"head":1,"tail":0,"margin":2,"relation":"hosts","polarity":"positive","modality":"planned"}]}
        ]})).unwrap();
        let text = "📄 Éva E 東京";
        doc.validate(text.chars().count()).unwrap();
        let out = doc.project(text, 0.5);
        assert_eq!(
            out.entities
                .iter()
                .map(|e| e.text.as_str())
                .collect::<Vec<_>>(),
            ["Éva", "E", "東京"]
        );
        assert_eq!(out.relations.len(), 1);
        assert_eq!(
            (out.relations[0].head_idx, out.relations[0].tail_idx),
            (0, 2)
        );
        assert!(doc.project(text, 0.99).relations.is_empty());
    }

    #[test]
    #[ignore = "requires the pinned local Kniv model and KNIV_PYTHON"]
    fn published_checkpoint_finds_entities_and_place_of_birth() {
        let model = Kniv::from_env().unwrap();
        let text = "Ada Lovelace was born in London.";
        let mentions = model.extract_entities(text, None).unwrap();
        assert!(mentions
            .iter()
            .any(|e| e.text == "Ada Lovelace" && e.start() == 0 && e.end() == 12));
        let out = model
            .extract_with_relations(text, &[], &["place of birth"], 0.55)
            .unwrap();
        assert!(out
            .relations
            .iter()
            .any(|r| r.relation_type == "place of birth"
                && out.entities[r.head_idx].text == "Ada Lovelace"
                && out.entities[r.tail_idx].text == "London"));
    }
}
