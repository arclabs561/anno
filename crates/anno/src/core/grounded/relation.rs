//! Resolve the endpoints of a directed extraction against one grounded document.

use super::{Entity, GroundedDocument, Provenance, Signal, SignalId, SignalValidationError, Track};
use crate::core::entity::Relation;

/// A mention supporting the relation and its current document-local membership.
#[derive(Debug)]
pub struct RelationEndpoint<'a> {
    /// Exact source signal, including its own extraction provenance.
    pub signal: &'a Signal,
    /// The track containing this signal, when one has been assigned.
    pub track: Option<&'a Track>,
}

/// A read-only projection onto a particular document and its current tracks.
///
/// The immutable document borrow prevents cached membership from surviving a
/// regrouping. Project the original extraction again after a membership change.
#[derive(Debug)]
pub struct GroundedRelation<'a> {
    /// Source document whose signals and tracks the endpoints reference.
    pub document: &'a GroundedDocument,
    /// Original directed relation, including its trigger and extraction score.
    pub relation: &'a Relation,
    /// Producer of the relation decision, when supplied by the extraction run.
    pub provenance: Option<&'a Provenance>,
    /// Source endpoint of the directed relation.
    pub head: RelationEndpoint<'a>,
    /// Target endpoint of the directed relation.
    pub tail: RelationEndpoint<'a>,
}

/// Why a flat extraction could not be grounded unambiguously in the document.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RelationProjectionError {
    /// The head or tail has no signal with matching surface, type, and location.
    #[error("no matching signal for the relation {0}")]
    MissingEndpoint(&'static str),
    /// Multiple detector outputs agree on the endpoint description.
    #[error("multiple signals match the relation {0}")]
    AmbiguousEndpoint(&'static str),
    /// The supplied document-local signal ID is missing or does not match the endpoint.
    #[error("signal {id} does not match the relation {role}")]
    EndpointIdMismatch {
        /// Directed endpoint being grounded.
        role: &'static str,
        /// ID supplied for this document.
        id: SignalId,
    },
    /// The endpoint does not cover its stated surface in this source.
    #[error(transparent)]
    InvalidSignal(#[from] SignalValidationError),
    /// Trigger offsets must describe a nonempty character range inside the text.
    #[error("relation trigger [{start}, {end}) is outside the source or empty")]
    InvalidTrigger {
        /// Inclusive start character offset.
        start: usize,
        /// Exclusive end character offset.
        end: usize,
    },
}

impl GroundedDocument {
    /// Ground a flat relation emitted for this document's text.
    ///
    /// Both endpoints must uniquely match their extracted surface, type and
    /// complete text location, including discontinuous segments. An endpoint
    /// may still be untracked. Callers supply provenance for the relation
    /// decision independently of the mention detectors.
    ///
    /// # Errors
    ///
    /// Returns an error for missing or ambiguous endpoints, invalid source
    /// signals, or an out-of-bounds trigger.
    pub fn project_relation<'a>(
        &'a self,
        relation: &'a Relation,
        provenance: Option<&'a Provenance>,
    ) -> Result<GroundedRelation<'a>, RelationProjectionError> {
        self.project_relation_with_ids(relation, None, provenance)
    }

    /// Ground a relation using the producer’s document-local signal IDs.
    ///
    /// Source mentions can coincide even when their identities differ. This
    /// path checks the IDs against the full endpoint geometry, label and
    /// surface. It preserves a known assignment without resolving a tie by
    /// signal order. The IDs must come from this edition of the document.
    ///
    /// # Errors
    ///
    /// Returns an error when either ID fails to identify its endpoint, the
    /// underlying source signal is invalid, or the trigger is out of bounds.
    pub fn project_relation_by_signal_ids<'a>(
        &'a self,
        relation: &'a Relation,
        head: SignalId,
        tail: SignalId,
        provenance: Option<&'a Provenance>,
    ) -> Result<GroundedRelation<'a>, RelationProjectionError> {
        self.project_relation_with_ids(relation, Some((head, tail)), provenance)
    }

    fn project_relation_with_ids<'a>(
        &'a self,
        relation: &'a Relation,
        ids: Option<(SignalId, SignalId)>,
        provenance: Option<&'a Provenance>,
    ) -> Result<GroundedRelation<'a>, RelationProjectionError> {
        if let Some((start, end)) = relation.trigger_span {
            if start >= end || end > self.text().chars().count() {
                return Err(RelationProjectionError::InvalidTrigger { start, end });
            }
        }
        let endpoint = |entity: &Entity, role, id| {
            let target = Self::signal_from_entity(entity);
            if target.text_offsets() != Some((entity.start(), entity.end())) {
                return Err(RelationProjectionError::MissingEndpoint(role));
            }
            let matches = |signal: &&Signal| {
                signal.surface == target.surface
                    && signal.location == target.location
                    && signal.label == target.label
            };
            let signal = if let Some(id) = id {
                self.get_signal(id)
                    .filter(matches)
                    .ok_or(RelationProjectionError::EndpointIdMismatch { role, id })?
            } else {
                let mut matching = self.signals().iter().filter(matches);
                let first = matching
                    .next()
                    .ok_or(RelationProjectionError::MissingEndpoint(role))?;
                if matching.next().is_some() {
                    return Err(RelationProjectionError::AmbiguousEndpoint(role));
                }
                first
            };
            if let Some(err) = signal.validate_against(self.text()) {
                return Err(RelationProjectionError::InvalidSignal(err));
            }
            Ok(RelationEndpoint {
                signal,
                track: self.track_for_signal(signal.id),
            })
        };
        Ok(GroundedRelation {
            document: self,
            relation,
            provenance,
            head: endpoint(&relation.head, "head", ids.map(|(head, _)| head))?,
            tail: endpoint(&relation.tail, "tail", ids.map(|(_, tail)| tail))?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EntityType;

    #[test]
    fn retains_directed_mentions_and_independent_relation_provenance_after_grouping() {
        let entities = [
            Entity::new("Zoë", EntityType::Person, 0, 3, 0.9),
            Entity::new("She", EntityType::Person, 5, 8, 0.9),
            Entity::new("Acme", EntityType::Organization, 14, 18, 0.9),
        ];
        let mut doc =
            GroundedDocument::from_entity_signals("doc", "Zoë. She sold Acme.", &entities);
        let relation =
            Relation::with_trigger(entities[1].clone(), entities[2].clone(), "sold", 9, 13, 0.8);
        let producer = Provenance::default();
        let first = doc.project_relation(&relation, Some(&producer)).unwrap();
        assert!(first.head.track.is_none());
        assert!(first.tail.track.is_none());
        assert!(std::ptr::eq(first.provenance.unwrap(), &producer));
        let she = first.head.signal.id;
        let zoe = doc.signals()[0].id;
        let group = doc.create_track_from_signals("Zoë", &[zoe, she]).unwrap();

        let next = doc.project_relation(&relation, None).unwrap();
        assert_eq!(next.head.signal.id, she);
        assert_eq!(next.head.track.unwrap().id, group);
        assert_eq!(next.tail.signal.surface, "Acme");
        assert!(next.tail.track.is_none());
        assert!(next.provenance.is_none());
        assert!(std::ptr::eq(next.relation, &relation));
    }

    #[test]
    fn retains_segment_coverage_and_current_membership_after_document_round_trip() {
        // Both heads have the same label, bounding offsets and surface. The
        // segmented mention uses a legacy bounding surface accepted by Anno.
        let contiguous = Entity::new("A and B", EntityType::Person, 0, 7, 0.8);
        let mut segmented = contiguous.clone();
        segmented.set_discontinuous_span(crate::DiscontinuousSpan::new(vec![0..1, 6..7]));
        let org = Entity::new("C", EntityType::Organization, 12, 13, 0.7);
        let doc = GroundedDocument::from_entities(
            "d",
            "A and B met C",
            &[contiguous, segmented.clone(), org.clone()],
        );
        let saved: GroundedDocument =
            serde_json::from_value(serde_json::to_value(doc).unwrap()).unwrap();
        let relation = Relation::new(segmented, org, "met", 0.9);
        let result = saved.project_relation(&relation, None).unwrap();
        assert_eq!(result.head.signal.id, saved.signals()[1].id);
        assert_eq!(result.tail.signal.id, saved.signals()[2].id);
        assert_eq!(
            result.head.track.unwrap().signals()[0].signal_id,
            result.head.signal.id
        );
    }

    #[test]
    fn supplied_ids_preserve_distinct_concepts_with_coincident_mentions() {
        let place = Entity::new("A", EntityType::Location, 0, 1, 0.9);
        let other = Entity::new("B", EntityType::Location, 2, 3, 0.9);
        let doc =
            GroundedDocument::from_entities("d", "A B", &[place.clone(), place.clone(), other]);
        let relation = Relation::new(place.clone(), place, "located in", 0.9);
        let [a, b, c] = doc.signals() else {
            panic!("expected three mentions")
        };
        assert!(matches!(
            doc.project_relation(&relation, None),
            Err(RelationProjectionError::AmbiguousEndpoint("head"))
        ));
        let resolved = doc
            .project_relation_by_signal_ids(&relation, b.id, a.id, None)
            .unwrap();
        assert_eq!(
            (resolved.head.signal.id, resolved.tail.signal.id),
            (b.id, a.id)
        );
        assert_ne!(
            resolved.head.track.unwrap().id,
            resolved.tail.track.unwrap().id
        );
        for wrong in [c.id, SignalId::new(99)] {
            assert!(
                matches!(doc.project_relation_by_signal_ids(&relation, b.id, wrong, None),
                Err(RelationProjectionError::EndpointIdMismatch { role: "tail", id }) if id == wrong)
            );
        }
    }

    #[test]
    fn rejects_ambiguous_or_missing_mentions_and_invalid_triggers() {
        let e = Entity::new("A", EntityType::Person, 0, 1, 0.9);
        let mut doc = GroundedDocument::from_entity_signals("d", "A", std::slice::from_ref(&e));
        let mut relation = Relation::new(e.clone(), e.clone(), "refers to", 0.5);
        relation.tail.text = "B".into();
        assert!(matches!(
            doc.project_relation(&relation, None),
            Err(RelationProjectionError::MissingEndpoint("tail"))
        ));
        relation.tail = e.clone();
        doc.add_signal(Signal::from(&e));
        assert!(matches!(
            doc.project_relation(&relation, None),
            Err(RelationProjectionError::AmbiguousEndpoint("head"))
        ));
        relation.trigger_span = Some((0, 2));
        assert!(matches!(
            doc.project_relation(&relation, None),
            Err(RelationProjectionError::InvalidTrigger { .. })
        ));
    }
}
