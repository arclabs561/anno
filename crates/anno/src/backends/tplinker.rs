//! TPLinker: Single-stage Joint Entity-Relation Extraction
//!
//! TPLinker uses a handshaking tagging scheme for joint entity-relation extraction.
//! It models entity boundaries and relations simultaneously using a unified tagging matrix.
//!
//! # Implementation Status
//!
//! This module defines the TPLinker *shape* but **does not implement** the TPLinker model.
//! It exists to keep the interface and wiring points explicit without returning fake outputs.
//!
//! If you need relation extraction today, treat it as out-of-scope for `anno`'s primary surface.
//!
//! # Research
//!
//! - **Paper**: [TPLinker: Single-stage Joint Extraction](https://aclanthology.org/2020.coling-main.138/)
//! - **Architecture**: Handshaking matrix where each cell (i,j) encodes:
//!   - Entity boundaries (SH2OH, OH2SH, ST2OT, OT2ST)
//!   - Relations (handshaking between entity pairs)
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::backends::tplinker::TPLinker;
//!
//! let extractor = TPLinker::new()?;
//! let result = extractor.extract_with_relations(
//!     "Steve Jobs founded Apple in 1976.",
//!     &["person", "organization"],
//!     &["founded", "works_for"],
//!     0.5
//! )?;
//!
//! for entity in &result.entities {
//!     println!("Entity: {} ({})", entity.text, entity.entity_type);
//! }
//!
//! for relation in &result.relations {
//!     let head = &result.entities[relation.head_idx];
//!     let tail = &result.entities[relation.tail_idx];
//!     println!("Relation: {} --[{}]--> {}", head.text, relation.relation_type, tail.text);
//! }
//! ```

use crate::backends::inference::{
    extract_relation_triples, ExtractionWithRelations, RelationExtractionConfig, RelationExtractor,
    SemanticRegistry,
};
use crate::{Entity, EntityType, Model, Result};
use std::borrow::Cow;
use std::collections::HashSet;

/// TPLinker backend for joint entity-relation extraction.
///
/// Uses handshaking matrix to simultaneously extract entities and relations.
#[derive(Debug)]
pub struct TPLinker {
    /// Confidence threshold for entity extraction
    #[allow(dead_code)]
    entity_threshold: f32,
    /// Confidence threshold for relation extraction
    #[allow(dead_code)]
    relation_threshold: f32,
}

impl TPLinker {
    /// Create a new TPLinker instance.
    pub fn new() -> Result<Self> {
        Err(crate::Error::model_init(
            "TPLinker is not implemented in this crate (reserved name; no placeholder outputs)",
        ))
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(entity_threshold: f32, relation_threshold: f32) -> Self {
        Self {
            entity_threshold,
            relation_threshold,
        }
    }

    /// Reserved decoder entrypoint (not implemented).
    ///
    /// A full TPLinker implementation would:
    /// 1. Run ONNX model to get handshaking matrix predictions
    /// 2. Decode entity boundaries from SH2OH/OH2SH tags
    /// 3. Decode relations from handshaking between entity pairs
    #[allow(dead_code)] // Placeholder helper; kept for future TPLinker ONNX decoding work.
    fn extract_with_handshaking(
        &self,
        text: &str,
        entity_types: &[&str],
        relation_types: &[&str],
        threshold: f32,
    ) -> Result<ExtractionWithRelations> {
        // Interpret the call-site `threshold` as the *relation* threshold.
        // Entity extraction should remain governed by `self.entity_threshold`, otherwise
        // relation-eval runs with `threshold=0.5` can accidentally wipe out almost all
        // heuristic entities and produce zero relations.
        let rel_threshold = if threshold > 0.0 {
            threshold
        } else {
            self.relation_threshold
        };
        let ent_threshold = self.entity_threshold;

        // Placeholder: Use HeuristicNER for entity extraction
        // This properly handles multi-word entity names
        let heuristic = crate::HeuristicNER::new();
        let mut entities = heuristic.extract_entities(text, None)?;

        // Respect the requested entity schema when possible.
        // Note: Some relation datasets provide rich, dataset-specific entity type labels
        // (e.g. "programlang", "academicjournal"). Those are not representable in our
        // `EntityType` enum, so filtering via `EntityType::from_label` would collapse them
        // (typically to `Misc`) and accidentally drop all HeuristicNER entities.
        //
        // We only apply filtering when the requested schema looks like it targets the
        // canonical types we can actually emit.
        if !entity_types.is_empty() {
            let requested: Vec<String> = entity_types.iter().map(|s| s.to_lowercase()).collect();
            let looks_supported = requested.iter().all(|t| {
                matches!(
                    t.as_str(),
                    "person"
                        | "per"
                        | "organization"
                        | "organisation"
                        | "org"
                        | "location"
                        | "loc"
                        | "date"
                        | "time"
                        | "money"
                        | "misc"
                )
            });
            if looks_supported {
                let allowed: HashSet<EntityType> = entity_types
                    .iter()
                    .map(|s| EntityType::from_label(s))
                    .collect();
                entities.retain(|e| allowed.contains(&e.entity_type));
            }
        }

        // Apply the *entity* threshold to entity confidences.
        entities.retain(|e| e.confidence >= f64::from(ent_threshold));

        // Add provenance to indicate heuristic baseline (not a neural TPLinker).
        for entity in &mut entities {
            entity.provenance = Some(crate::Provenance {
                source: Cow::Borrowed("tplinker"),
                method: crate::ExtractionMethod::Heuristic,
                pattern: None,
                raw_confidence: Some(entity.confidence),
                model_version: Some(Cow::Borrowed("heuristic")),
                timestamp: None,
            });
        }

        // Extract relations: heuristic trigger-based extraction implemented in `inference.rs`.
        //
        // This is deliberately conservative: we only emit relations when we match a known trigger
        // pattern *and* the relation type is present in `relation_types`. We do not "guess" a
        // relation type just because two entities are nearby.
        let registry = {
            let mut builder = SemanticRegistry::builder();
            for rel in relation_types {
                // Description is a best-effort placeholder; only the slug is used by the
                // heuristic matcher today.
                builder = builder.add_relation(rel, rel);
            }
            builder.build_placeholder(1)
        };

        let rel_config = RelationExtractionConfig {
            threshold: rel_threshold,
            max_span_distance: 120,
            extract_triggers: false,
        };

        let relations = extract_relation_triples(&entities, text, &registry, &rel_config);

        Ok(ExtractionWithRelations {
            entities,
            relations,
        })
    }
}

impl Model for TPLinker {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        let _ = text;
        Err(crate::Error::inference(
            "TPLinker is not implemented (no placeholder extraction)",
        ))
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
        ]
    }

    fn is_available(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "tplinker"
    }

    fn description(&self) -> &'static str {
        "TPLinker joint entity-relation extraction (not implemented)"
    }
}

impl RelationExtractor for TPLinker {
    fn extract_with_relations(
        &self,
        text: &str,
        entity_types: &[&str],
        relation_types: &[&str],
        threshold: f32,
    ) -> Result<ExtractionWithRelations> {
        let _ = (text, entity_types, relation_types, threshold);
        Err(crate::Error::inference(
            "TPLinker is not implemented (no placeholder extraction)",
        ))
    }
}

// Make TPLinker implement BatchCapable and StreamingCapable for consistency
impl crate::BatchCapable for TPLinker {
    fn extract_entities_batch(
        &self,
        texts: &[&str],
        _language: Option<&str>,
    ) -> Result<Vec<Vec<Entity>>> {
        texts
            .iter()
            .map(|text| self.extract_entities(text, None))
            .collect()
    }
}

impl crate::StreamingCapable for TPLinker {
    fn extract_entities_streaming(&self, chunk: &str, offset: usize) -> Result<Vec<Entity>> {
        let mut entities = self.extract_entities(chunk, None)?;
        for entity in &mut entities {
            entity.start += offset;
            entity.end += offset;
        }
        Ok(entities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tplinker_creation() {
        let err = TPLinker::new().unwrap_err().to_string();
        assert!(
            err.to_lowercase().contains("not implemented"),
            "expected not-implemented error; got: {err}"
        );
    }

    #[test]
    fn test_tplinker_entity_extraction() {
        let tplinker = TPLinker::with_thresholds(0.5, 0.5);
        let err = tplinker
            .extract_entities("Steve Jobs founded Apple.", None)
            .unwrap_err()
            .to_string();
        assert!(
            err.to_lowercase().contains("not implemented"),
            "expected not-implemented error; got: {err}"
        );
    }

    #[test]
    fn test_tplinker_relation_extraction() {
        let tplinker = TPLinker::with_thresholds(0.5, 0.5);
        let err = tplinker
            .extract_with_relations(
                "Steve Jobs founded Apple in 1976.",
                &["person", "organization"],
                &["founded"],
                0.5,
            )
            .unwrap_err()
            .to_string();
        assert!(
            err.to_lowercase().contains("not implemented"),
            "expected not-implemented error; got: {err}"
        );
    }

    #[test]
    fn test_tplinker_unicode_offsets_invariants() {
        // Diverse scripts + emoji (multi-byte). Offsets must be character-based and valid.
        // Since TPLinker is not implemented, this is an error-path test.
        let tplinker = TPLinker::with_thresholds(0.5, 0.5);
        let text = "Dr. 田中 met François Müller in 東京. 🎉";
        let err = tplinker
            .extract_with_relations(
                text,
                &["person", "location", "organization"],
                &["works_for", "located_in", "founded"],
                0.0,
            )
            .unwrap_err()
            .to_string();
        assert!(
            err.to_lowercase().contains("not implemented"),
            "expected not-implemented error; got: {err}"
        );
    }
}
