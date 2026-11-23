//! Checked conversion from flat extraction results to a grounded document.

use super::super::types::CanonicalId;
use super::{Entity, EntityType, GroundedDocument};
use std::collections::HashMap;

/// Evidence in a flat entity that cannot be retained by grounded grouping.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum EntityProjectionError {
    /// The grounded signal stores a label whose flat representation changes the type.
    #[error("entity {index} has an entity type that cannot round-trip through its signal label")]
    LabelMapping {
        /// Zero-based index in the input slice.
        index: usize,
    },
    /// The grounded signal cannot represent this mention's linguistic type.
    #[error("entity {index} has a mention type that cannot be stored on a grounded signal")]
    MentionType {
        /// Zero-based index in the input slice.
        index: usize,
    },
    /// The grounded location cannot retain the separate visual span.
    #[error("entity {index} has a visual span that cannot round-trip through a grounded location")]
    VisualSpan {
        /// Zero-based index in the input slice.
        index: usize,
    },
    /// Two entities assigned to one track carry different populated KB IDs.
    #[error(
        "entities {first_index} and {index} in canonical group {group} have conflicting KB IDs"
    )]
    ConflictingLinks {
        /// Document-local canonical ID supplied by the caller.
        group: CanonicalId,
        /// Index of the first member with a KB ID.
        first_index: usize,
        /// Index of the first member with a different KB ID.
        index: usize,
    },
}

impl GroundedDocument {
    /// Group entities as in [`Self::from_entities`], checking for projection losses first.
    ///
    /// A missing KB ID and a populated ID in the same group are compatible.
    /// The track receives the populated ID. Independent singleton mentions may
    /// have different links.
    ///
    /// As with [`Self::from_entities`], track IDs are allocated in first-seen order.
    /// This check does not validate source offsets or preserve input canonical ID numbers.
    ///
    /// # Errors
    ///
    /// Returns the first entity with a mention type, visual span, lossy type label,
    /// or KB link that conflicts with an earlier member of its canonical group.
    pub fn try_from_entities(
        id: impl Into<String>,
        text: impl Into<String>,
        entities: &[Entity],
    ) -> Result<Self, EntityProjectionError> {
        let mut links = HashMap::new();
        for (index, entity) in entities.iter().enumerate() {
            if entity.mention_type.is_some() {
                return Err(EntityProjectionError::MentionType { index });
            }
            if entity.visual_span.is_some() {
                return Err(EntityProjectionError::VisualSpan { index });
            }
            if EntityType::from_label(entity.entity_type.as_label()) != entity.entity_type {
                return Err(EntityProjectionError::LabelMapping { index });
            }
            if let (Some(group), Some(kb_id)) = (entity.canonical_id, entity.kb_id.as_deref()) {
                let (first_index, first_link) = *links.entry(group).or_insert((index, kb_id));
                if first_link != kb_id {
                    return Err(EntityProjectionError::ConflictingLinks {
                        group,
                        first_index,
                        index,
                    });
                }
            }
        }
        Ok(Self::from_entities(id, text, entities))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EntityCategory, MentionType};

    fn entity(kb: Option<&str>, group: Option<CanonicalId>) -> Entity {
        let mut e = Entity::new("A", EntityType::Person, 0, 1, 0.123456789);
        e.kb_id = kb.map(str::to_owned);
        e.canonical_id = group;
        e
    }

    #[test]
    fn checked_grouping_reports_both_conflicting_mentions() {
        let group = CanonicalId::new(42);
        let entities = [None, Some("Q1"), Some("Q1"), Some("Q2")].map(|kb| entity(kb, Some(group)));
        assert_eq!(
            GroundedDocument::try_from_entities("d", "A", &entities).unwrap_err(),
            EntityProjectionError::ConflictingLinks {
                group,
                first_index: 1,
                index: 3
            },
        );
        // Legacy callers still receive the first link.
        assert_eq!(
            GroundedDocument::from_entities("d", "A", &entities).to_entities()[3]
                .kb_id
                .as_deref(),
            Some("Q1")
        );
    }

    #[test]
    fn checked_grouping_preserves_scores_and_tolerates_missing_links() {
        let group = Some(CanonicalId::new(42));
        let entities = [
            entity(None, group),
            entity(Some("Q1"), group),
            entity(Some("Q2"), None),
        ];
        let doc = GroundedDocument::try_from_entities("d", "A", &entities).unwrap();
        let wire = serde_json::to_string(&doc).unwrap();
        let saved: GroundedDocument = serde_json::from_str(&wire).unwrap();
        let flat = saved.to_entities();
        assert_eq!(flat[0].confidence, entities[0].confidence);
        assert_eq!(flat[0].kb_id.as_deref(), Some("Q1"));
        assert_eq!(flat[0].canonical_id, flat[1].canonical_id);
        assert_ne!(flat[0].canonical_id, flat[2].canonical_id);
        assert_eq!(flat[2].kb_id.as_deref(), Some("Q2"));
    }

    #[test]
    fn checked_grouping_reports_unrepresentable_metadata() {
        let mut e = entity(None, None);
        e.entity_type = EntityType::custom("DISEASE", EntityCategory::Agent);
        assert_eq!(
            GroundedDocument::try_from_entities("d", "A", std::slice::from_ref(&e)).unwrap_err(),
            EntityProjectionError::LabelMapping { index: 0 }
        );
        e.entity_type = EntityType::Person;
        e.visual_span = Some(crate::Span::BoundingBox {
            x: 0.0,
            y: 0.0,
            width: 0.5,
            height: 1.0,
            page: Some(1),
        });
        assert_eq!(
            GroundedDocument::try_from_entities("d", "A", std::slice::from_ref(&e)).unwrap_err(),
            EntityProjectionError::VisualSpan { index: 0 }
        );
        e.visual_span = None;
        e.mention_type = Some(MentionType::Pronominal);
        assert_eq!(
            GroundedDocument::try_from_entities("d", "A", &[e]).unwrap_err(),
            EntityProjectionError::MentionType { index: 0 }
        );
    }
}
