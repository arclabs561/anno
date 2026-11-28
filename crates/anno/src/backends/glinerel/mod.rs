//! SauerkrautLM-GLiNEREL-72m zero-shot entity and directed relation extraction.
//! The Python worker loads the original safetensors once and runs FP32 on CPU.
use super::inference::{ExtractionWithRelations, RelationExtractor, RelationTriple, ZeroShotNER};
use super::python_worker::PythonWorker;
use crate::{
    Confidence, Entity, EntityType, Error, ExtractionMethod, Language, Model, Provenance, Result,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::path::Path;

const NAME: &str = "glinerel-72m";
const DEFAULT_LABELS: &[&str] = &["person", "organization", "location"];

/// A persistent `glinerel-fast` CPU worker. See the GLiNEREL setup guide.
pub struct GLiNEREL {
    worker: PythonWorker,
}

impl crate::sealed::Sealed for GLiNEREL {}

impl GLiNEREL {
    /// Reads `ANNO_GLINEREL_PYTHON` and `ANNO_GLINEREL_MODEL_DIR`.
    /// By default Python loads the pinned Hub snapshot or its cached copy.
    pub fn from_env() -> Result<Self> {
        let python =
            std::env::var_os("ANNO_GLINEREL_PYTHON").unwrap_or_else(|| OsString::from("python3"));
        let checkpoint = std::env::var_os("ANNO_GLINEREL_MODEL_DIR");
        Self::new(python, checkpoint.as_deref().map(Path::new))
    }

    /// Load a prepared checkpoint directory, or the pinned Hub revision when omitted.
    pub fn new(python: impl AsRef<OsStr>, checkpoint: Option<&Path>) -> Result<Self> {
        Ok(Self {
            worker: PythonWorker::start(
                python.as_ref(),
                include_str!("worker.py"),
                vec![checkpoint
                    .map(Path::as_os_str)
                    .unwrap_or_default()
                    .to_owned()],
                NAME,
            )?,
        })
    }

    fn predict(
        &self,
        text: &str,
        entities: &[&str],
        relations: &[&str],
        threshold: f32,
    ) -> Result<ExtractionWithRelations> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(Error::invalid_input(
                "GLiNEREL threshold must be between zero and one",
            ));
        }
        if text.trim().is_empty() || (entities.is_empty() && relations.is_empty()) {
            return Ok(ExtractionWithRelations::default());
        }
        let output: Prediction = self.worker.request(&serde_json::json!({
            "text": text, "entities": entities, "relations": relations, "threshold": threshold,
        }))?;
        output.align(text)
    }
}

impl Model for GLiNEREL {
    fn name(&self) -> &'static str {
        NAME
    }
    fn description(&self) -> &'static str {
        "SauerkrautLM-GLiNEREL-72m joint entity and relation extraction on CPU"
    }
    fn supported_types(&self) -> Vec<EntityType> {
        DEFAULT_LABELS
            .iter()
            .map(|label| EntityType::from_label(label))
            .collect()
    }
    fn is_available(&self) -> bool {
        self.worker.is_available()
    }
    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        self.extract_with_types(text, DEFAULT_LABELS, 0.5)
    }
    fn as_zero_shot(&self) -> Option<&dyn ZeroShotNER> {
        Some(self)
    }
    fn as_relation_extractor(&self) -> Option<&dyn RelationExtractor> {
        Some(self)
    }
}

impl ZeroShotNER for GLiNEREL {
    fn default_types(&self) -> &[&'static str] {
        DEFAULT_LABELS
    }

    fn extract_with_types(
        &self,
        text: &str,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        Ok(self.predict(text, labels, &[], threshold)?.entities)
    }
    fn extract_with_descriptions(
        &self,
        text: &str,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        self.extract_with_types(text, labels, threshold)
    }
}

impl RelationExtractor for GLiNEREL {
    fn extract_with_relations(
        &self,
        text: &str,
        entities: &[&str],
        relations: &[&str],
        threshold: f32,
    ) -> Result<ExtractionWithRelations> {
        self.predict(text, entities, relations, threshold)
    }
    fn extract_with_relation_provenance(
        &self,
        text: &str,
        entities: &[&str],
        relations: &[&str],
        threshold: f32,
    ) -> Result<(ExtractionWithRelations, Option<Provenance>)> {
        let result = self.predict(text, entities, relations, threshold)?;
        Ok((
            result,
            Some(Provenance::from_producer(NAME, ExtractionMethod::Neural)),
        ))
    }
}

#[derive(Deserialize)]
struct Span {
    start: usize,
    end: usize,
    text: String,
}

impl Span {
    fn validate(&self, source: &str, offsets: &[usize]) -> Result<()> {
        if self.start >= self.end
            || self.end >= offsets.len()
            || source[offsets[self.start]..offsets[self.end]] != self.text
        {
            return Err(Error::inference(
                "GLiNEREL returned an invalid Unicode source span",
            ));
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct Detection {
    #[serde(flatten)]
    span: Span,
    label: String,
    score: f64,
}
#[derive(Deserialize)]
struct Edge {
    head: Span,
    tail: Span,
    label: String,
    score: f64,
}
#[derive(Deserialize)]
struct Prediction {
    entities: Vec<Detection>,
    relations: Vec<Edge>,
}

impl Prediction {
    fn align(self, text: &str) -> Result<ExtractionWithRelations> {
        let offsets = crate::offset::build_char_to_byte_map(text);
        let mut out = ExtractionWithRelations::default();
        let mut indices: HashMap<(usize, usize), usize> = HashMap::new();
        for det in self.entities {
            det.span.validate(text, &offsets)?;
            let key = (det.span.start, det.span.end);
            if indices
                .get(&key)
                .is_none_or(|&i| out.entities[i].confidence < det.score)
            {
                indices.insert(key, out.entities.len());
            }
            out.entities.push(Entity::with_provenance(
                det.span.text,
                EntityType::from_label(&det.label),
                key.0,
                key.1,
                det.score,
                Provenance::ml(NAME, det.score),
            ));
        }
        for edge in self.relations {
            // Relation heads may propose arguments absent from the NER output.
            // Retain their occurrences without guessing a semantic type. The worker
            // returns edges in descending confidence, so the first edge is strongest.
            let mut endpoint = |span: Span| -> Result<usize> {
                span.validate(text, &offsets)?;
                let key = (span.start, span.end);
                Ok(*indices.entry(key).or_insert_with(|| {
                    let index = out.entities.len();
                    out.entities.push(Entity::with_provenance(
                        span.text,
                        EntityType::from_label("RELATION_ARGUMENT"),
                        key.0,
                        key.1,
                        edge.score,
                        Provenance::from_producer(
                            "glinerel-72m:relation-argument",
                            ExtractionMethod::Neural,
                        ),
                    ));
                    index
                }))
            };
            let head_idx = endpoint(edge.head)?;
            let tail_idx = endpoint(edge.tail)?;
            out.relations.push(RelationTriple {
                head_idx,
                tail_idx,
                relation_type: edge.label,
                confidence: Confidence::new(edge.score),
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relations_keep_source_occurrences_and_untyped_arguments() {
        let predict = serde_json::json!({
            "entities": [
                {"label":"person", "text":"Ana", "start":2, "end":5, "score":0.96},
                {"label":"person", "text":"Ana", "start":7, "end":10, "score":0.94}
            ],
            "relations": [
                {"label":"from","head":{"text":"Ana","start":7,"end":10},
                 "tail":{"text":"Lima","start":12,"end":16},"score":0.74},
                {"label":"from","head":{"text":"Ana","start":2,"end":5},
                 "tail":{"text":"Lima","start":12,"end":16},"score":0.65}
            ]
        });
        let output: Prediction = serde_json::from_value(predict).unwrap();
        let result = output.align("É Ana, Ana: Lima").unwrap();
        assert_eq!(result.entities.len(), 3);
        assert_eq!(
            result.entities[2].entity_type.as_label(),
            "RELATION_ARGUMENT"
        );
        assert_eq!(result.relations[0].head_idx, 1);
        assert_eq!(result.relations[1].head_idx, 0);
        assert_eq!(result.relations[0].tail_idx, result.relations[1].tail_idx);
        assert_eq!(result.relations[0].tail_idx, 2);
    }
}
