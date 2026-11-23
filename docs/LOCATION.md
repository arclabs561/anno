# Location and visual spans

Anno’s grounded locations use half-open Unicode scalar offsets in a specific
source text. `Location::Text` covers one range. `Location::Discontinuous`
covers multiple ranges. For OCR or layout annotations, `Entity::visual_span`
can carry a bounding box or a hybrid span containing text and geometry.

To migrate `Location::from(span)` or `span.into()`, choose the operation that
matches the caller’s requirements.

```rust
use anno::{Location, Span};

let text = Span::text(2, 5);
assert_eq!(Location::try_from(&text).unwrap(), Location::text(2, 5));

let bbox = Span::bbox(0.1, 0.1, 0.3, 0.2);
assert!(Location::try_from(&bbox).is_err());
assert_eq!(Location::project_text(&bbox), None);

let hybrid = Span::Hybrid { start: 2, end: 5, bbox: Box::new(bbox) };
assert!(Location::try_from(&hybrid).is_err());
assert_eq!(Location::project_text(&hybrid), Some(Location::text(2, 5)));
```

`GroundedDocument::try_from_entities` reports visual metadata it cannot preserve.
For an entity with a contiguous span, the compatibility projection
`from_entity_signals` uses the visual span’s text coordinates when available
and otherwise keeps the entity’s text offsets. Keep the visual
span alongside the entity when downstream consumers need geometry.
