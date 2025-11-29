//! Shared strategy generators for property-based testing.

use anno::schema::CanonicalType;
use anno::{Entity, EntityCategory, EntityType};
use proptest::prelude::*;

/// Generate a random EntityType.
pub fn entity_type_strategy() -> impl Strategy<Value = EntityType> {
    prop_oneof![
        Just(EntityType::Person),
        Just(EntityType::Organization),
        Just(EntityType::Location),
        Just(EntityType::Date),
        Just(EntityType::Time),
        Just(EntityType::Money),
        Just(EntityType::Percent),
        Just(EntityType::Email),
        Just(EntityType::Url),
        Just(EntityType::Phone),
        // Custom types
        "[A-Z_]{1,20}".prop_map(|s| EntityType::custom(&s, EntityCategory::Misc)),
    ]
}

/// Generate a random Entity with valid constraints.
pub fn entity_strategy() -> impl Strategy<Value = Entity> {
    (
        ".{1,100}", // text (non-empty)
        entity_type_strategy(),
        0.0f64..1.0f64, // confidence
    )
        .prop_flat_map(|(text, entity_type, confidence)| {
            let text_len = text.chars().count();
            (
                Just(text),
                Just(entity_type),
                0usize..text_len,
                1usize..=text_len,
                Just(confidence),
            )
                .prop_filter_map(
                    "start must be < end",
                    |(text, entity_type, start, end, confidence)| {
                        if start < end {
                            Some(Entity::new(text, entity_type, start, end, confidence))
                        } else {
                            None
                        }
                    },
                )
        })
}

/// Generate a random CanonicalType.
pub fn canonical_type_strategy() -> impl Strategy<Value = CanonicalType> {
    prop::sample::select(vec![
        CanonicalType::Person,
        CanonicalType::Organization,
        CanonicalType::Location,
        CanonicalType::GeopoliticalEntity,
        CanonicalType::NaturalLocation,
        CanonicalType::Facility,
        CanonicalType::Date,
        CanonicalType::Time,
        CanonicalType::Money,
        CanonicalType::Percent,
        CanonicalType::Quantity,
        CanonicalType::Cardinal,
        CanonicalType::Ordinal,
        CanonicalType::Group,
        CanonicalType::CreativeWork,
        CanonicalType::Product,
        CanonicalType::Event,
        CanonicalType::Law,
        CanonicalType::Language,
        CanonicalType::Disease,
        CanonicalType::Chemical,
        CanonicalType::Gene,
        CanonicalType::Drug,
        CanonicalType::Animal,
        CanonicalType::Plant,
        CanonicalType::Food,
        CanonicalType::Misc,
    ])
}

/// Generate a random string with Unicode characters.
pub fn unicode_text_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex(".{0,500}").unwrap()
}

/// Generate a random ASCII string.
pub fn ascii_text_strategy() -> impl Strategy<Value = String> {
    "[ -~]{0,500}"
}
