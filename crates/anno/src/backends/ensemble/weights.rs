//! Backend weighting, type weights, and candidate/span-key helpers.

use super::*;

// =============================================================================
// Backend Weights
// =============================================================================

/// Reliability weight for a backend (0.0 to 1.0).
///
/// Higher weight = more trusted when resolving conflicts.
#[derive(Debug, Clone, Copy)]
pub struct BackendWeight {
    /// Overall reliability of this backend
    pub overall: f64,
    /// Type-specific weights (optional overrides)
    pub per_type: Option<TypeWeights>,
}

impl Default for BackendWeight {
    fn default() -> Self {
        Self {
            overall: 0.5,
            per_type: None,
        }
    }
}

/// Type-specific reliability weights.
///
/// Different backends may have different accuracy profiles for different entity types.
/// These weights adjust confidence scores based on the entity type being extracted.
#[derive(Debug, Clone, Copy, Default)]
pub struct TypeWeights {
    /// Weight multiplier for Person entities
    pub person: f64,
    /// Weight multiplier for Organization entities
    pub organization: f64,
    /// Weight multiplier for Location entities
    pub location: f64,
    /// Weight multiplier for Date entities
    pub date: f64,
    /// Weight multiplier for Money entities
    pub money: f64,
    /// Weight multiplier for other/misc entity types
    pub other: f64,
}

impl TypeWeights {
    pub(super) fn get(&self, entity_type: &EntityType) -> f64 {
        match entity_type {
            EntityType::Person => self.person,
            EntityType::Organization => self.organization,
            EntityType::Location => self.location,
            EntityType::Date => self.date,
            EntityType::Money => self.money,
            _ => self.other,
        }
    }
}

/// Default weights based on empirical observations.
pub(super) fn default_backend_weights() -> HashMap<&'static str, BackendWeight> {
    let mut weights = HashMap::new();

    // Pattern backends: very high precision when they fire
    weights.insert(
        "regex",
        BackendWeight {
            overall: 0.98,
            per_type: Some(TypeWeights {
                date: 0.99,
                money: 0.99,
                person: 0.50, // Pattern doesn't do NER
                organization: 0.50,
                location: 0.50,
                other: 0.95, // URLs, emails, etc.
            }),
        },
    );

    // GLiNER: good ML-based NER
    weights.insert(
        "gliner",
        BackendWeight {
            overall: 0.85,
            per_type: Some(TypeWeights {
                person: 0.90,
                organization: 0.85,
                location: 0.80,
                date: 0.75,
                money: 0.70,
                other: 0.75,
            }),
        },
    );
    weights.insert(
        "GLiNER-ONNX",
        BackendWeight {
            overall: 0.85,
            per_type: Some(TypeWeights {
                person: 0.90,
                organization: 0.85,
                location: 0.80,
                date: 0.75,
                money: 0.70,
                other: 0.75,
            }),
        },
    );

    // GLiNER Candle (stable ID + backend name() alias)
    weights.insert(
        "gliner-candle",
        BackendWeight {
            overall: 0.85,
            per_type: None,
        },
    );
    weights.insert(
        "GLiNER-Candle",
        BackendWeight {
            overall: 0.85,
            per_type: None,
        },
    );

    // BERT NER (stable ID + backend name() alias)
    weights.insert(
        "bert-ner-onnx",
        BackendWeight {
            overall: 0.80,
            per_type: None,
        },
    );
    weights.insert(
        "bert-onnx",
        BackendWeight {
            overall: 0.80,
            per_type: None,
        },
    );

    // Heuristic: reasonable but noisy
    weights.insert(
        "heuristic",
        BackendWeight {
            overall: 0.60,
            per_type: Some(TypeWeights {
                person: 0.65,       // Title + Name pattern works well
                organization: 0.70, // "Inc", "Corp" patterns
                location: 0.55,     // Context-dependent
                date: 0.40,         // Better to use pattern
                money: 0.40,
                other: 0.50,
            }),
        },
    );

    weights
}

// =============================================================================
// Candidate Entity (with source tracking)
// =============================================================================

/// An entity candidate from a specific backend.
#[derive(Debug, Clone)]
pub(super) struct Candidate {
    pub(super) entity: Entity,
    pub(super) source: String,
    pub(super) backend_weight: f64,
}

// =============================================================================
// Span Key (for grouping overlapping entities)
// =============================================================================

/// Key for grouping entities by span.
///
/// Two entities are considered "same span" if they significantly overlap.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct SpanKey {
    pub(super) start: usize,
    pub(super) end: usize,
}

impl SpanKey {
    pub(super) fn from_entity(e: &Entity) -> Self {
        Self {
            start: e.start(),
            end: e.end(),
        }
    }

    /// Check if two spans overlap significantly (>50% of smaller span).
    pub(super) fn overlaps(&self, other: &SpanKey) -> bool {
        let overlap_start = self.start.max(other.start);
        let overlap_end = self.end.min(other.end);

        if overlap_start >= overlap_end {
            return false;
        }

        let overlap = overlap_end - overlap_start;
        let smaller_span = (self.end - self.start).min(other.end - other.start);

        // Overlap if >50% of smaller span is covered
        (overlap as f64 / smaller_span as f64) > 0.5
    }
}

// =============================================================================
// EnsembleNER
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // BackendWeight defaults
    // =========================================================================

    #[test]
    fn backend_weight_default_is_neutral() {
        let w = BackendWeight::default();
        assert!(
            (w.overall - 0.5).abs() < f64::EPSILON,
            "default overall should be 0.5"
        );
        assert!(w.per_type.is_none(), "default per_type should be None");
    }

    // =========================================================================
    // TypeWeights::get dispatches correctly
    // =========================================================================

    #[test]
    fn type_weights_get_returns_matching_field() {
        let tw = TypeWeights {
            person: 0.1,
            organization: 0.2,
            location: 0.3,
            date: 0.4,
            money: 0.5,
            other: 0.6,
        };

        assert!((tw.get(&EntityType::Person) - 0.1).abs() < f64::EPSILON);
        assert!((tw.get(&EntityType::Organization) - 0.2).abs() < f64::EPSILON);
        assert!((tw.get(&EntityType::Location) - 0.3).abs() < f64::EPSILON);
        assert!((tw.get(&EntityType::Date) - 0.4).abs() < f64::EPSILON);
        assert!((tw.get(&EntityType::Money) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn type_weights_get_falls_back_to_other() {
        let tw = TypeWeights {
            other: 0.77,
            ..TypeWeights::default()
        };

        // EntityType variants not explicitly matched should return `other`.
        assert!(
            (tw.get(&EntityType::Email) - 0.77).abs() < f64::EPSILON,
            "Email should fall back to `other`"
        );
        assert!(
            (tw.get(&EntityType::Percent) - 0.77).abs() < f64::EPSILON,
            "Percent should fall back to `other`"
        );
    }

    // =========================================================================
    // default_backend_weights coverage
    // =========================================================================

    #[test]
    fn default_weights_contain_all_known_backends() {
        let w = default_backend_weights();
        let expected = [
            "regex",
            "gliner",
            "GLiNER-ONNX",
            "gliner-candle",
            "bert-ner-onnx",
            "heuristic",
        ];
        for name in expected {
            assert!(
                w.contains_key(name),
                "missing default weight for backend '{}'",
                name
            );
        }
    }

    #[test]
    fn default_weights_are_in_unit_range() {
        let weights = default_backend_weights();
        for (name, bw) in &weights {
            assert!(
                (0.0..=1.0).contains(&bw.overall),
                "overall weight for '{}' out of range: {}",
                name,
                bw.overall
            );
            if let Some(ref tw) = bw.per_type {
                for (label, val) in [
                    ("person", tw.person),
                    ("organization", tw.organization),
                    ("location", tw.location),
                    ("date", tw.date),
                    ("money", tw.money),
                    ("other", tw.other),
                ] {
                    assert!(
                        (0.0..=1.0).contains(&val),
                        "type weight '{}' for '{}' out of range: {}",
                        label,
                        name,
                        val
                    );
                }
            }
        }
    }

    // =========================================================================
    // SpanKey basics
    // =========================================================================

    #[test]
    fn span_key_from_entity_round_trips() {
        let e = Entity::new("hello", EntityType::Person, 3, 8, 0.9);
        let sk = SpanKey::from_entity(&e);
        assert_eq!(sk.start, 3);
        assert_eq!(sk.end, 8);
    }

    #[test]
    fn span_key_no_overlap_when_disjoint() {
        let a = SpanKey { start: 0, end: 5 };
        let b = SpanKey { start: 10, end: 15 };
        assert!(!a.overlaps(&b));
        assert!(!b.overlaps(&a));
    }

    #[test]
    fn span_key_overlap_threshold_boundary() {
        // Exactly 50% overlap should NOT count (strict >0.5).
        // a=[0,10), b=[5,15): overlap=[5,10)=5, smaller=10, ratio=0.5 -> false
        let a = SpanKey { start: 0, end: 10 };
        let b = SpanKey { start: 5, end: 15 };
        assert!(
            !a.overlaps(&b),
            "exactly 50% overlap should be below the >0.5 threshold"
        );
    }

    #[test]
    fn span_key_overlap_just_above_threshold() {
        // a=[0,10), b=[4,14): overlap=[4,10)=6, smaller=10, ratio=0.6 -> true
        let a = SpanKey { start: 0, end: 10 };
        let b = SpanKey { start: 4, end: 14 };
        assert!(
            a.overlaps(&b),
            "60% overlap should be above the >0.5 threshold"
        );
        assert!(b.overlaps(&a), "overlap should be symmetric");
    }

    // =========================================================================
    // Candidate construction (smoke test)
    // =========================================================================

    #[test]
    fn candidate_holds_source_and_weight() {
        let e = Entity::new("ACME", EntityType::Organization, 0, 4, 0.85);
        let c = Candidate {
            entity: e.clone(),
            source: "test-backend".to_string(),
            backend_weight: 0.75,
        };
        assert_eq!(c.source, "test-backend");
        assert!((c.backend_weight - 0.75).abs() < f64::EPSILON);
        assert_eq!(c.entity.text, "ACME");
    }
}
