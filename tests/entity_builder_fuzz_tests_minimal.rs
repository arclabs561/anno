use anno::{Entity, EntityBuilder, EntityType};
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_minimal(text in ".{1,10}") {
        let entity = EntityBuilder::new(&text, EntityType::Person)
            .span(0, text.len())
            .confidence(0.9)
            .build();
        prop_assert_eq!(entity.text, text);
    }
}
