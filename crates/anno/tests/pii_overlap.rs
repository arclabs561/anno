use anno::pii::{self, PiiEntity};

const TEXT: &str = "éAliceSmithJones!";

fn entity(start: usize, end: usize) -> PiiEntity {
    PiiEntity {
        text: TEXT.chars().skip(start).take(end - start).collect(),
        pii_type: "PERSON".into(),
        start,
        end,
        risk_level: "HIGH".into(),
    }
}

fn overlaps() -> Vec<PiiEntity> {
    vec![entity(1, 11), entity(6, 16), entity(1, 11)]
}

#[test]
fn mask_covers_the_union_of_overlapping_character_spans() {
    assert_eq!(pii::mask(TEXT, &overlaps(), '*'), "é***************!");
}

#[test]
fn replacement_receives_the_complete_merged_source_text_once() {
    let mut calls = Vec::new();
    let result = pii::replace(TEXT, &overlaps(), |entity| {
        calls.push((entity.start, entity.end, entity.text.clone()));
        "[HIDDEN]".into()
    });
    assert_eq!(result, "é[HIDDEN]!");
    assert_eq!(calls, vec![(1, 16, "AliceSmithJones".into())]);
}

#[test]
fn fingerprint_covers_and_hashes_the_complete_union() {
    assert_eq!(
        pii::fingerprint(TEXT, &overlaps()),
        pii::fingerprint(TEXT, &[entity(1, 16)])
    );
}

#[test]
fn pseudonymize_replaces_the_union_once_and_maps_the_complete_source() {
    let (result, mapping) = pii::pseudonymize(TEXT, &overlaps());
    assert_eq!(result, "éJohn Smith!");
    assert_eq!(mapping.len(), 1);
    assert_eq!(
        mapping.get("AliceSmithJones").map(String::as_str),
        Some("John Smith")
    );
}

#[test]
fn transitive_and_contained_overlaps_merge_independently_of_input_order() {
    let mut entities = vec![entity(1, 7), entity(6, 12), entity(11, 16), entity(2, 4)];
    for _ in 0..2 {
        assert_eq!(pii::mask(TEXT, &entities, '*'), "é***************!");
        let (result, mapping) = pii::pseudonymize(TEXT, &entities);
        assert_eq!(result, "éJohn Smith!");
        assert_eq!(mapping.len(), 1);
        assert!(mapping.contains_key("AliceSmithJones"));
        entities.reverse();
    }
}

#[test]
fn non_overlapping_entities_preserve_caller_supplied_text() {
    let mut supplied = entity(1, 6);
    supplied.text = "normalized name".into();
    let entities = [supplied];
    assert_eq!(
        pii::replace(TEXT, &entities, |e| e.text.clone()),
        "énormalized nameSmithJones!"
    );
    let (result, mapping) = pii::pseudonymize(TEXT, &entities);
    assert_eq!(result, "éJohn SmithSmithJones!");
    assert_eq!(
        mapping.get("normalized name").map(String::as_str),
        Some("John Smith")
    );
}

#[test]
fn touching_spans_remain_separate() {
    let mut calls = Vec::new();
    let result = pii::replace(TEXT, &[entity(1, 6), entity(6, 16)], |e| {
        calls.push(e.text.clone());
        "[HIDDEN]".into()
    });
    assert_eq!(result, "é[HIDDEN][HIDDEN]!");
    assert_eq!(calls, vec!["SmithJones", "Alice"]);
}
