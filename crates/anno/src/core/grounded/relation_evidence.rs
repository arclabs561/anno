//! Persist the level at which a relation was observed in the source edition.

use super::html_source::source_sha256;
use super::{
    GroundedDocument, GroundedRelation, Location, Provenance, RelationProjectionError, SignalId,
    Track,
};
use crate::core::entity::Relation;
use crate::{Confidence, TypeLabel};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A model's original relation and the source signals it used.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMentionRelation {
    /// Signal selected as the directed head.
    pub head_signal_id: SignalId,
    /// Signal selected as the directed tail.
    pub tail_signal_id: SignalId,
    /// Original extraction, including mention spans, trigger and score.
    pub relation: Relation,
    /// Producer of the relation decision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
}

/// A source-declared member of a concept. The snapshot detects signal ID reuse.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConceptMember {
    /// Signal selected by the source annotation.
    pub signal_id: SignalId,
    /// Complete location including all segments.
    pub location: Location,
    /// Signal text at the time of annotation.
    pub surface: String,
    /// Semantic type at the time of annotation.
    pub label: TypeLabel,
}

/// Concept identifier in the relation producer's annotation namespace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConceptEndpoint {
    /// Original identifier; equal-looking concepts can have distinct IDs.
    pub concept_id: String,
    /// Full membership supplied by the source; absent when the source did not specify it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub members: Option<Vec<ConceptMember>>,
}

/// Directed annotation made about concepts. A concept edge can have no mention-pair witness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptRelation {
    /// Source concept.
    pub head: ConceptEndpoint,
    /// Target concept.
    pub tail: ConceptEndpoint,
    /// Predicate in the producer's schema.
    pub relation_type: String,
    /// Score if one was supplied; a gold annotation need not have a score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<Confidence>,
    /// Producer or annotation source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
}

/// Current track projections of the two source-declared concepts.
#[derive(Debug)]
pub struct ConceptRelationProjection<'a> {
    /// Original relation and the member evidence supplied for each endpoint.
    pub relation: &'a ConceptRelation,
    /// Set only when all and only the declared head members constitute one track.
    pub head_track: Option<&'a Track>,
    /// Set only when all and only the declared tail members constitute one track.
    pub tail_track: Option<&'a Track>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct RelationEvidence {
    pub(super) document_id: String,
    text_sha256: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    mentions: Vec<StoredMentionRelation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    concepts: Vec<ConceptRelation>,
}

impl GroundedDocument {
    fn evidence_mut(&mut self) -> &mut RelationEvidence {
        self.relation_evidence
            .get_or_insert_with(|| RelationEvidence {
                document_id: self.id.clone(),
                text_sha256: source_sha256(&self.text),
                mentions: Vec::new(),
                concepts: Vec::new(),
            })
    }

    /// Record a relation after grounding its two endpoints unambiguously.
    pub fn record_relation(
        &mut self,
        relation: Relation,
        provenance: Option<Provenance>,
    ) -> Result<(), RelationProjectionError> {
        let view = self.project_relation(&relation, provenance.as_ref())?;
        let (head_signal_id, tail_signal_id) = (view.head.signal.id, view.tail.signal.id);
        self.evidence_mut().mentions.push(StoredMentionRelation {
            head_signal_id,
            tail_signal_id,
            relation,
            provenance,
        });
        Ok(())
    }

    /// Record a directed relation with the exact signals selected by the producer.
    /// A failed validation leaves the evidence collection unchanged.
    pub fn record_relation_by_signal_ids(
        &mut self,
        relation: Relation,
        head: SignalId,
        tail: SignalId,
        provenance: Option<Provenance>,
    ) -> Result<(), RelationProjectionError> {
        self.project_relation_by_signal_ids(&relation, head, tail, provenance.as_ref())?;
        self.evidence_mut().mentions.push(StoredMentionRelation {
            head_signal_id: head,
            tail_signal_id: tail,
            relation,
            provenance,
        });
        Ok(())
    }

    /// Original directed mention relations, in the order they were recorded.
    #[must_use]
    pub fn recorded_relations(&self) -> &[StoredMentionRelation] {
        self.relation_evidence.as_ref().map_or(&[], |r| &r.mentions)
    }

    /// Ground each saved relation against current track memberships.
    pub fn project_recorded_relations(
        &self,
    ) -> impl Iterator<Item = Result<GroundedRelation<'_>, RelationProjectionError>> {
        self.recorded_relations().iter().map(|saved| {
            self.project_relation_by_signal_ids(
                &saved.relation,
                saved.head_signal_id,
                saved.tail_signal_id,
                saved.provenance.as_ref(),
            )
        })
    }

    /// Capture the complete member set when supplied, or just the source concept ID.
    pub fn concept_endpoint(
        &self,
        concept_id: impl Into<String>,
        members: Option<&[SignalId]>,
    ) -> Result<ConceptEndpoint, String> {
        let mut endpoint = ConceptEndpoint {
            concept_id: concept_id.into(),
            members: None,
        };
        if let Some(ids) = members {
            let snapshots = ids
                .iter()
                .map(|&id| {
                    let s = self
                        .get_signal(id)
                        .ok_or_else(|| format!("unknown concept member signal {id}"))?;
                    Ok(ConceptMember {
                        signal_id: id,
                        location: s.location.clone(),
                        surface: s.surface.clone(),
                        label: s.label.clone(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            endpoint.members = Some(snapshots);
        }
        self.validate_concept_endpoint(&endpoint)?;
        Ok(endpoint)
    }

    fn validate_concept_endpoint(&self, endpoint: &ConceptEndpoint) -> Result<(), String> {
        if endpoint.concept_id.trim().is_empty() {
            return Err("a concept endpoint needs its source identifier".into());
        }
        let mut seen = HashSet::new();
        for member in endpoint.members.iter().flatten() {
            if !seen.insert(member.signal_id) {
                return Err(format!(
                    "duplicate member {} in concept {}",
                    member.signal_id, endpoint.concept_id
                ));
            }
            let signal = self
                .get_signal(member.signal_id)
                .ok_or_else(|| format!("unknown concept member signal {}", member.signal_id))?;
            if signal.location != member.location
                || signal.surface != member.surface
                || signal.label != member.label
            {
                return Err(format!(
                    "signal {} no longer matches concept {}",
                    member.signal_id, endpoint.concept_id
                ));
            }
            if let Some(error) = signal.validate_against(self.text()) {
                return Err(error.to_string());
            }
        }
        Ok(())
    }

    /// Record a directed concept-level annotation without generating mention pairs.
    pub fn record_concept_relation(&mut self, relation: ConceptRelation) -> Result<(), String> {
        self.validate_concept_endpoint(&relation.head)?;
        self.validate_concept_endpoint(&relation.tail)?;
        self.evidence_mut().concepts.push(relation);
        Ok(())
    }

    /// Source concept relations, including any supplied member evidence.
    #[must_use]
    pub fn recorded_concept_relations(&self) -> &[ConceptRelation] {
        self.relation_evidence.as_ref().map_or(&[], |r| &r.concepts)
    }

    fn exact_concept_track(&self, endpoint: &ConceptEndpoint) -> Option<&Track> {
        let members = endpoint.members.as_ref()?;
        let track = self.track_for_signal(members.first()?.signal_id)?;
        (track.signals.len() == members.len()
            && members.iter().all(|member| {
                self.track_for_signal(member.signal_id)
                    .is_some_and(|t| t.id == track.id)
            }))
        .then_some(track)
    }

    /// Project only complete, exact memberships. Unspecified, untracked, split
    /// and merged member sets keep their original concept endpoints.
    pub fn project_recorded_concept_relations(
        &self,
    ) -> impl Iterator<Item = Result<ConceptRelationProjection<'_>, String>> {
        self.recorded_concept_relations().iter().map(|relation| {
            self.validate_concept_endpoint(&relation.head)?;
            self.validate_concept_endpoint(&relation.tail)?;
            Ok(ConceptRelationProjection {
                relation,
                head_track: self.exact_concept_track(&relation.head),
                tail_track: self.exact_concept_track(&relation.tail),
            })
        })
    }

    pub(super) fn validate_relation_evidence(&self) -> Result<(), String> {
        if let Some(evidence) = &self.relation_evidence {
            if evidence.document_id != self.id || evidence.text_sha256 != source_sha256(&self.text)
            {
                return Err(
                    "relation evidence belongs to a different document or source text edition"
                        .into(),
                );
            }
            for (index, edge) in self.project_recorded_relations().enumerate() {
                edge.map_err(|error| format!("mention relation {index}: {error}"))?;
            }
            for (index, edge) in self.project_recorded_concept_relations().enumerate() {
                edge.map_err(|error| format!("concept relation {index}: {error}"))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::grounded::{Corpus, DeliveryOwnership, DocumentDelivery};
    use crate::{Entity, EntityType};

    fn saved() -> GroundedDocument {
        let mentions = [
            Entity::new("Zoë", EntityType::Person, 0, 3, 0.99),
            Entity::new("She", EntityType::Person, 5, 8, 0.88),
            Entity::new("Acme", EntityType::Organization, 14, 18, 0.77),
        ];
        let mut doc = GroundedDocument::from_entities("source", "Zoë. She sold Acme.", &mentions);
        let mut relation = Relation::with_trigger(
            mentions[1].clone(),
            mentions[2].clone(),
            "sold",
            9,
            13,
            0.83,
        );
        relation.head.provenance = Some(Provenance::ml("mention model", 0.88));
        doc.record_relation(relation, Some(Provenance::ml("relation model", 0.83)))
            .unwrap();
        doc
    }

    #[test]
    fn relation_observation_survives_regrouping_serialization_and_delivery() {
        let mut doc = saved();
        let zoe = doc.track_for_signal(doc.signals()[0].id).unwrap().id;
        let she = doc.track_for_signal(doc.signals()[1].id).unwrap().id;
        let original = serde_json::to_value(doc.recorded_relations()).unwrap();
        let new_track = doc.merge_tracks(&[zoe, she]).unwrap();
        assert_ne!(she, new_track);
        let restored: GroundedDocument =
            serde_json::from_slice(&serde_json::to_vec(&doc).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(restored.recorded_relations()).unwrap(),
            original
        );
        let view = restored
            .project_recorded_relations()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(view.head.signal.surface, "She");
        assert_eq!(view.head.track.unwrap().id, new_track);
        assert_eq!(view.relation.trigger_span, Some((9, 13)));
        assert_eq!(view.provenance.unwrap().source, "relation model");

        let request = DocumentDelivery {
            issuer: "test".into(),
            delivery_id: "delivery".into(),
            source_revision: 1,
            ownership: DeliveryOwnership::Independent,
        };
        let mut corpus = Corpus::new();
        let input = serde_json::to_vec(&restored).unwrap();
        let receipt = corpus
            .deliver_document(&request, &input)
            .unwrap()
            .receipt()
            .clone();
        let delivered = corpus.get_document(&receipt.document_id).unwrap();
        assert_ne!(delivered.id(), restored.id());
        let copy: GroundedDocument =
            serde_json::from_slice(&serde_json::to_vec(delivered).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(copy.recorded_relations()).unwrap(),
            original
        );
    }

    #[test]
    fn rejects_stale_edition_dangling_reference_and_changed_signal() {
        let mut doc = saved();
        let wire = serde_json::to_value(&doc).unwrap();
        for (key, replacement) in [
            ("id", serde_json::json!("another document")),
            // Changes outside either endpoint still identify a different edition.
            ("text", serde_json::json!("Zoë. She sold Acme!")),
        ] {
            let mut changed = wire.clone();
            changed[key] = replacement;
            assert!(serde_json::from_value::<GroundedDocument>(changed).is_err());
        }
        let mut missing = wire.clone();
        missing["relation_evidence"]["mentions"][0]["tail_signal_id"] = 999.into();
        assert!(serde_json::from_value::<GroundedDocument>(missing).is_err());
        let mut legacy = wire;
        legacy.as_object_mut().unwrap().remove("relation_evidence");
        let old = serde_json::from_value::<GroundedDocument>(legacy).unwrap();
        assert!(old.recorded_relations().is_empty());
        assert!(serde_json::to_value(old)
            .unwrap()
            .get("relation_evidence")
            .is_none());

        doc.signals_mut()[1].label = TypeLabel::from("ORG");
        assert!(!doc.validate_invariants().is_empty());
        assert!(doc.project_recorded_relations().next().unwrap().is_err());
        assert!(serde_json::to_vec(&doc).is_err());
    }

    #[test]
    fn preserves_source_concepts_when_memberships_are_unknown_split_or_merged() {
        let mut doc = saved();
        let [a, b, c] = doc.signals() else {
            panic!("three mentions")
        };
        let (a, b, c) = (a.id, b.id, c.id);
        assert!(doc.concept_endpoint("duplicate", Some(&[a, a])).is_err());
        let buyer = doc.concept_endpoint("speaker-12", Some(&[a, b])).unwrap();
        let company = doc.concept_endpoint("org-4", Some(&[c])).unwrap();
        doc.record_concept_relation(ConceptRelation {
            head: buyer,
            tail: company.clone(),
            relation_type: "owns".into(),
            confidence: None,
            provenance: None,
        })
        .unwrap();
        let unspecified = doc
            .concept_endpoint("nontextual-source-concept", None)
            .unwrap();
        let declared_empty = doc
            .concept_endpoint("no-listed-mentions", Some(&[]))
            .unwrap();
        doc.record_concept_relation(ConceptRelation {
            head: unspecified,
            tail: declared_empty,
            relation_type: "mentions".into(),
            confidence: None,
            provenance: None,
        })
        .unwrap();
        let projects = |d: &GroundedDocument| {
            let view = d
                .project_recorded_concept_relations()
                .next()
                .unwrap()
                .unwrap();
            (view.head_track.map(|t| t.id), view.tail_track.map(|t| t.id))
        };
        let company_track = doc.track_for_signal(c).unwrap().id;
        assert_eq!(projects(&doc), (None, Some(company_track)));
        let joined = doc
            .merge_tracks(&[
                doc.track_for_signal(a).unwrap().id,
                doc.track_for_signal(b).unwrap().id,
            ])
            .unwrap();
        assert_eq!(projects(&doc), (Some(joined), Some(company_track)));
        doc.merge_tracks(&[joined, company_track]).unwrap();
        let restored: GroundedDocument =
            serde_json::from_slice(&serde_json::to_vec(&doc).unwrap()).unwrap();
        assert_eq!(projects(&restored), (None, None));
        assert!(restored.recorded_concept_relations()[1]
            .head
            .members
            .is_none());
        assert_eq!(
            restored.recorded_concept_relations()[1]
                .tail
                .members
                .as_deref(),
            Some([].as_slice())
        );
        assert_eq!(restored.recorded_concept_relations().len(), 2);
        assert_eq!(restored.recorded_relations().len(), 1);
        let mut wire = serde_json::to_value(restored).unwrap();
        wire["relation_evidence"]["concepts"][0]["head"]["members"][0]["signal_id"] = 999.into();
        assert!(serde_json::from_value::<GroundedDocument>(wire).is_err());
    }
}
