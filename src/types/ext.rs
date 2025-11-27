//! Extension traits for entity collections.

use crate::Entity;
use std::collections::HashMap;

/// Extension methods for slices of entities.
///
/// This trait adds useful operations to `[Entity]` and `Vec<Entity>`
/// without requiring you to wrap them in a newtype.
///
/// # Example
///
/// ```rust
/// use anno::{Entity, EntityType};
/// use anno::types::EntitySliceExt;
///
/// let entities = vec![
///     Entity::new("John", EntityType::Person, 0, 4, 0.9),
///     Entity::new("$100", EntityType::Money, 10, 14, 0.95),
///     Entity::new("Paris", EntityType::Location, 20, 25, 0.7),
/// ];
///
/// // Filter by confidence
/// let high_conf: Vec<_> = entities.above_confidence(0.8).collect();
/// assert_eq!(high_conf.len(), 2);
///
/// // Check for overlaps
/// assert!(!entities.has_overlaps());
/// ```
pub trait EntitySliceExt {
    /// Filter entities by minimum confidence threshold.
    fn above_confidence(&self, min: f64) -> impl Iterator<Item = &Entity>;

    /// Filter entities by type.
    fn of_type(&self, ty: &crate::EntityType) -> impl Iterator<Item = &Entity>;

    /// Check if any entities overlap with each other.
    fn has_overlaps(&self) -> bool;

    /// Find all overlapping pairs of entities.
    fn overlapping_pairs(&self) -> Vec<(&Entity, &Entity)>;

    /// Get entities sorted by confidence (descending).
    fn sorted_by_confidence(&self) -> Vec<&Entity>;

    /// Get entities sorted by position (ascending).
    fn sorted_by_position(&self) -> Vec<&Entity>;

    /// Get the entity with highest confidence.
    fn highest_confidence(&self) -> Option<&Entity>;

    /// Calculate average confidence across all entities.
    fn mean_confidence(&self) -> Option<f64>;

    /// Group entities by type.
    fn group_by_type(&self) -> HashMap<String, Vec<&Entity>>;

    /// Check if a position falls within any entity span.
    fn contains_position(&self, pos: usize) -> bool;

    /// Get entity at a specific position (if any).
    fn at_position(&self, pos: usize) -> Option<&Entity>;

    /// Filter to only named entities (Person, Org, Location).
    fn named_only(&self) -> impl Iterator<Item = &Entity>;

    /// Filter to only structured entities (Date, Money, Email, etc.).
    fn structured_only(&self) -> impl Iterator<Item = &Entity>;
}

impl EntitySliceExt for [Entity] {
    fn above_confidence(&self, min: f64) -> impl Iterator<Item = &Entity> {
        self.iter().filter(move |e| e.confidence >= min)
    }

    fn of_type(&self, ty: &crate::EntityType) -> impl Iterator<Item = &Entity> {
        let ty = ty.clone();
        self.iter().filter(move |e| e.entity_type == ty)
    }

    fn has_overlaps(&self) -> bool {
        for i in 0..self.len() {
            for j in (i + 1)..self.len() {
                if self[i].overlaps(&self[j]) {
                    return true;
                }
            }
        }
        false
    }

    fn overlapping_pairs(&self) -> Vec<(&Entity, &Entity)> {
        let mut pairs = Vec::new();
        for i in 0..self.len() {
            for j in (i + 1)..self.len() {
                if self[i].overlaps(&self[j]) {
                    pairs.push((&self[i], &self[j]));
                }
            }
        }
        pairs
    }

    fn sorted_by_confidence(&self) -> Vec<&Entity> {
        let mut sorted: Vec<_> = self.iter().collect();
        sorted.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    fn sorted_by_position(&self) -> Vec<&Entity> {
        let mut sorted: Vec<_> = self.iter().collect();
        sorted.sort_by_key(|e| (e.start, e.end));
        sorted
    }

    fn highest_confidence(&self) -> Option<&Entity> {
        self.iter().max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn mean_confidence(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        let sum: f64 = self.iter().map(|e| e.confidence).sum();
        Some(sum / self.len() as f64)
    }

    fn group_by_type(&self) -> HashMap<String, Vec<&Entity>> {
        let mut groups: HashMap<String, Vec<&Entity>> = HashMap::new();
        for entity in self {
            groups
                .entry(entity.entity_type.as_label().to_string())
                .or_default()
                .push(entity);
        }
        groups
    }

    fn contains_position(&self, pos: usize) -> bool {
        self.iter().any(|e| pos >= e.start && pos < e.end)
    }

    fn at_position(&self, pos: usize) -> Option<&Entity> {
        self.iter().find(|e| pos >= e.start && pos < e.end)
    }

    fn named_only(&self) -> impl Iterator<Item = &Entity> {
        self.iter().filter(|e| e.is_named())
    }

    fn structured_only(&self) -> impl Iterator<Item = &Entity> {
        self.iter().filter(|e| e.is_structured())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EntityType;

    fn sample_entities() -> Vec<Entity> {
        vec![
            Entity::new("John", EntityType::Person, 0, 4, 0.9),
            Entity::new("$100", EntityType::Money, 10, 14, 0.95),
            Entity::new("Paris", EntityType::Location, 20, 25, 0.7),
            Entity::new("2024", EntityType::Date, 30, 34, 0.85),
        ]
    }

    #[test]
    fn above_confidence_filters() {
        let entities = sample_entities();
        let high: Vec<_> = entities.above_confidence(0.85).collect();
        assert_eq!(high.len(), 3);
    }

    #[test]
    fn of_type_filters() {
        let entities = sample_entities();
        let people: Vec<_> = entities.of_type(&EntityType::Person).collect();
        assert_eq!(people.len(), 1);
        assert_eq!(people[0].text, "John");
    }

    #[test]
    fn has_overlaps_detects() {
        let entities = sample_entities();
        assert!(!entities.has_overlaps());

        let overlapping = vec![
            Entity::new("New York", EntityType::Location, 0, 8, 0.9),
            Entity::new("York", EntityType::Location, 4, 8, 0.8),
        ];
        assert!(overlapping.has_overlaps());
    }

    #[test]
    fn sorted_by_confidence_descending() {
        let entities = sample_entities();
        let sorted = entities.sorted_by_confidence();
        assert_eq!(sorted[0].text, "$100");
        assert_eq!(sorted[1].text, "John");
    }

    #[test]
    fn sorted_by_position_ascending() {
        let mut entities = sample_entities();
        entities.reverse();
        let sorted = entities.sorted_by_position();
        assert_eq!(sorted[0].text, "John");
        assert_eq!(sorted[1].text, "$100");
    }

    #[test]
    fn highest_confidence_finds_max() {
        let entities = sample_entities();
        let highest = entities.highest_confidence().unwrap();
        assert_eq!(highest.text, "$100");
    }

    #[test]
    fn mean_confidence_calculates() {
        let entities = sample_entities();
        let mean = entities.mean_confidence().unwrap();
        assert!((mean - 0.85).abs() < 1e-10);
    }

    #[test]
    fn group_by_type_groups() {
        let entities = sample_entities();
        let groups = entities.group_by_type();
        assert_eq!(groups.get("PER").map(|v| v.len()), Some(1));
        assert_eq!(groups.get("MONEY").map(|v| v.len()), Some(1));
    }

    #[test]
    fn position_queries() {
        let entities = sample_entities();
        assert!(entities.contains_position(2));
        assert!(!entities.contains_position(5));
        assert_eq!(entities.at_position(12).unwrap().text, "$100");
    }

    #[test]
    fn named_and_structured_filters() {
        let entities = sample_entities();
        let named: Vec<_> = entities.named_only().collect();
        assert_eq!(named.len(), 2);
        let structured: Vec<_> = entities.structured_only().collect();
        assert_eq!(structured.len(), 2);
    }

    #[test]
    fn empty_slice_handles_gracefully() {
        let entities: Vec<Entity> = vec![];
        assert!(!entities.has_overlaps());
        assert!(entities.highest_confidence().is_none());
        assert!(entities.mean_confidence().is_none());
    }
}

