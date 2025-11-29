use anno::schema::{DatasetSchema, SchemaMapper};
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_minimal(label in "[A-Z_]{1,20}") {
        let mapper = SchemaMapper::for_dataset(DatasetSchema::OntoNotes);
        let _ = mapper.to_canonical(&label);
    }
}
