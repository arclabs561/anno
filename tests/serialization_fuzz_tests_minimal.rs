use anno::{Entity, EntityType};
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_minimal(text in ".{1,10}") {
        let entity = Entity::new(&text, EntityType::Person, 0, text.len(), 0.9);
        let json = serde_json::to_string(&entity).unwrap();
        let _: Entity = serde_json::from_str(&json).unwrap();
    }
}
