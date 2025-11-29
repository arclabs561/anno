use anno::grounded::{GroundedDocument, Location, Signal};
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_minimal(text in ".{1,10}") {
        let mut doc = GroundedDocument::new("doc1", &text);
        let _ = doc.add_signal(Signal::new(0, Location::text(0, text.len()), &text, "Type", 0.9));
    }
}
