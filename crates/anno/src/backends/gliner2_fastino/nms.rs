//! Greedy NMS for span-level entities. Adapted from
//! SemplificaAI/gliner2-rs `extract_standard` lines ~870-885.

use crate::Entity;

/// Sort entities by confidence descending and drop overlapping ones.
/// `flat_ner = true`: any token-overlap drops the lower-scored entity
/// regardless of label. `flat_ner = false`: only same-label overlaps drop.
pub(crate) fn greedy_nms(mut candidates: Vec<Entity>, flat_ner: bool) -> Vec<Entity> {
    candidates.sort_by(|a, b| {
        let ac: f32 = a.confidence.into();
        let bc: f32 = b.confidence.into();
        bc.partial_cmp(&ac).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut selected: Vec<Entity> = Vec::with_capacity(candidates.len());
    for c in candidates {
        let overlaps = selected.iter().any(|s| {
            let span_overlap = !(c.end() <= s.start() || c.start() >= s.end());
            span_overlap && (flat_ner || s.entity_type == c.entity_type)
        });
        if !overlaps {
            selected.push(c);
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EntityType;

    fn ent(text: &str, ty: EntityType, start: usize, end: usize, score: f32) -> Entity {
        Entity::new(text, ty, start, end, score)
    }

    #[test]
    fn nms_keeps_higher_score_drops_overlap_same_label() {
        let cands = vec![
            ent("Acme", EntityType::Organization, 0, 4, 0.8),
            ent("Acme Corp", EntityType::Organization, 0, 9, 0.95),
        ];
        let kept = greedy_nms(cands, false);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].text, "Acme Corp");
    }

    #[test]
    fn nms_flat_ner_drops_overlap_across_labels() {
        let cands = vec![
            ent("Acme", EntityType::Organization, 0, 4, 0.6),
            ent("Acme", EntityType::Person, 0, 4, 0.95),
        ];
        let kept = greedy_nms(cands, true);
        assert_eq!(kept.len(), 1);
        assert!(matches!(kept[0].entity_type, EntityType::Person));
    }

    #[test]
    fn nms_keeps_disjoint_spans() {
        let cands = vec![
            ent("Acme", EntityType::Organization, 0, 4, 0.9),
            ent("Paris", EntityType::Location, 13, 18, 0.85),
        ];
        let kept = greedy_nms(cands, false);
        assert_eq!(kept.len(), 2);
    }
}
