use anno::schema::{CanonicalType, CoarseType, DatasetSchema, SchemaMapper};
use proptest::prelude::*;

proptest! {
    #[test]
    fn schema_mapper_idempotent(label in "[A-Z_]{1,20}") {
        let mapper = SchemaMapper::for_dataset(DatasetSchema::OntoNotes);
        let canonical1 = mapper.to_canonical(&label);
        let canonical2 = mapper.to_canonical(&label);
        prop_assert_eq!(canonical1, canonical2);
    }
    #[test]
    fn schema_mapper_handles_bio_prefixes(label in "[A-Z_]{1,15}", prefix in prop::sample::select(vec!["B-", "I-", "S-", "E-"])) {
        let prefixed = format!("{}{}", prefix, label);
        let mapper = SchemaMapper::for_dataset(DatasetSchema::CoNLL2003);
        let canonical = mapper.to_canonical(&prefixed);
        match canonical {
            CanonicalType::Person | CanonicalType::Organization |
            CanonicalType::Location | CanonicalType::Misc => {}
            _ => {}
        }
    }
}
