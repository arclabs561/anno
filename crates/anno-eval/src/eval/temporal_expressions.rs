//! Exact TIMEX span evaluation on the original sentence and its stand-off annotations.
use anno::{Error, Model, Result};
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap};

pub(super) const TEST_SHA256: &str =
    "58cd74b68da63ff8acbb70e8acd0302f4edea6151a2f73787aa57799a57e93d3";
pub(super) const LABEL: &str = "time expression";

#[derive(Deserialize)]
pub(crate) struct TimexSentence {
    text: String,
    // A present null means this row has no TIMEX gold. Missing fields indicate a wrong schema.
    #[serde(deserialize_with = "Option::deserialize")]
    time_expressions: Option<Vec<Expression>>,
}

#[derive(Deserialize)]
struct Expression {
    text: String,
    start_char: usize,
    end_char: usize,
}

fn matches_source(source: &[char], start: usize, end: usize, surface: &str) -> bool {
    start < end
        && source
            .get(start..end)
            .is_some_and(|s| s.iter().copied().eq(surface.chars()))
}

impl TimexSentence {
    fn gold_spans(&self, source: &[char]) -> Result<BTreeSet<(usize, usize)>> {
        let mut spans = BTreeSet::new();
        for expression in self.time_expressions.iter().flatten() {
            let (start, end) = (expression.start_char, expression.end_char);
            if !matches_source(source, start, end, &expression.text) {
                return Err(Error::InvalidInput(format!(
                    "TIMEX surface does not match source at [{start}, {end})"
                )));
            }
            spans.insert((start, end));
        }
        Ok(spans)
    }
}

pub(super) fn score(
    sentences: &[TimexSentence],
    model: &dyn Model,
) -> Result<(HashMap<String, f64>, usize)> {
    let mut raw_annotations = 0;
    let mut gold_count = 0;
    let mut predicted_count = 0;
    let mut correct = 0;
    let mut overlapping_unmatched = 0;
    let mut gold_empty_sentences = 0;
    let mut gold_empty_with_predictions = 0;
    for sentence in sentences {
        let source: Vec<char> = sentence.text.chars().collect();
        let gold = sentence.gold_spans(&source)?;
        raw_annotations += sentence.time_expressions.as_ref().map_or(0, Vec::len);
        let entities = match model.as_zero_shot() {
            Some(backend) => backend.extract_with_types(&sentence.text, &[LABEL], 0.5)?,
            None => model.extract_entities(&sentence.text, None)?,
        };
        // The pattern backend distinguishes DATE from TIME; the corpus scores both as TIMEX.
        let mut predicted = BTreeSet::new();
        for entity in entities {
            if !matches!(
                entity.entity_type.as_label().to_ascii_lowercase().as_str(),
                "date" | "time" | LABEL | "timex"
            ) {
                continue;
            }
            let (start, end) = (entity.start(), entity.end());
            if entity.is_discontinuous() || !matches_source(&source, start, end, &entity.text) {
                return Err(Error::Inference(format!(
                    "TIMEX prediction must match a contiguous source span at [{start}, {end})"
                )));
            }
            predicted.insert((start, end));
        }
        // Both differences iterate in start order. Count each unmatched prediction
        // once, even if it intersects several missed or nested annotations.
        let mut missed = gold.difference(&predicted).peekable();
        for &(start, end) in predicted.difference(&gold) {
            while missed.peek().is_some_and(|span| span.1 <= start) {
                missed.next();
            }
            if missed.peek().is_some_and(|span| span.0 < end) {
                overlapping_unmatched += 1;
            }
        }
        if gold.is_empty() {
            gold_empty_sentences += 1;
            gold_empty_with_predictions += usize::from(!predicted.is_empty());
        }
        correct += predicted.intersection(&gold).count();
        predicted_count += predicted.len();
        gold_count += gold.len();
    }
    let ratio = |n: usize, d: usize| if d == 0 { 0.0 } else { n as f64 / d as f64 };
    let metrics = HashMap::from([
        (
            "f1".into(),
            ratio(2 * correct, predicted_count + gold_count),
        ),
        ("precision".into(), ratio(correct, predicted_count)),
        ("recall".into(), ratio(correct, gold_count)),
        (
            "unmatched_predictions_overlapping_missed_gold".into(),
            overlapping_unmatched as f64,
        ),
        ("gold_empty_sentences".into(), gold_empty_sentences as f64),
        (
            "gold_empty_sentences_with_predictions".into(),
            gold_empty_with_predictions as f64,
        ),
        ("gold_annotations".into(), raw_annotations as f64),
        ("gold_spans".into(), gold_count as f64),
        ("predicted_spans".into(), predicted_count as f64),
        ("matched_spans".into(), correct as f64),
    ]);
    Ok((metrics, gold_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anno::{AnyModel, Entity, EntityType};

    #[test]
    fn counts_exact_source_spans_per_sentence_including_nested_and_negative_cases() {
        let rows: Vec<TimexSentence> = serde_json::from_value(serde_json::json!([
            {"text":"🦊 this year", "time_expressions":[
                {"text":"this year","start_char":2,"end_char":11},
                {"text":"this year","start_char":2,"end_char":11},
                {"text":"year","start_char":7,"end_char":11}]},
            {"text":"🦊 next year", "time_expressions":null,
             "event_expressions":[{"text":"year","start_char":7,"end_char":11}]}
        ]))
        .unwrap();
        let model = AnyModel::new("source-test", "", vec![], |_, _| {
            Ok(vec![
                Entity::new("year", EntityType::Date, 7, 11, 1.0),
                Entity::new("year", EntityType::Time, 7, 11, 1.0),
                Entity::new("🦊", EntityType::Person, 0, 1, 1.0),
            ])
        });
        let (metrics, n) = score(&rows, &model).unwrap();
        assert_eq!(n, 2);
        assert_eq!(metrics["gold_annotations"], 3.0);
        assert_eq!(metrics["predicted_spans"], 2.0);
        assert_eq!(metrics["matched_spans"], 1.0);
        assert_eq!(metrics["f1"], 0.5);
        assert_eq!(metrics["gold_empty_sentences"], 1.0);
        assert_eq!(metrics["gold_empty_sentences_with_predictions"], 1.0);
        assert_eq!(
            metrics["unmatched_predictions_overlapping_missed_gold"],
            0.0
        );
    }

    #[test]
    fn counts_overlapping_unmatched_predictions_once_and_excludes_touching_spans() {
        let rows: Vec<TimexSentence> = serde_json::from_value(serde_json::json!([{
            "text":"a b c d", "time_expressions":[
                {"text":"a b c","start_char":0,"end_char":5},
                {"text":"b c","start_char":2,"end_char":5},
                {"text":"d","start_char":6,"end_char":7}]
        }]))
        .unwrap();
        let model = AnyModel::new("boundary-test", "", vec![], |_, _| {
            Ok(vec![
                Entity::new("a b", EntityType::Date, 0, 3, 1.0),
                Entity::new("c", EntityType::Time, 4, 5, 1.0),
                Entity::new(" d", EntityType::Date, 5, 7, 1.0),
                Entity::new("d", EntityType::Date, 6, 7, 1.0),
            ])
        });
        let (metrics, _) = score(&rows, &model).unwrap();
        assert_eq!(metrics["matched_spans"], 1.0);
        assert_eq!(
            metrics["unmatched_predictions_overlapping_missed_gold"],
            2.0
        );
    }

    #[test]
    fn rejects_stale_or_discontinuous_model_evidence() {
        let rows: Vec<TimexSentence> = serde_json::from_str(r#"[{"text":"🦊 year","time_expressions":[{"text":"year","start_char":2,"end_char":6}]}]"#).unwrap();
        let stale = Entity::new("moon", EntityType::Date, 2, 6, 1.0);
        let mut discontinuous = Entity::new("year", EntityType::Time, 2, 6, 1.0);
        discontinuous.set_discontinuous_span(anno::DiscontinuousSpan::new(vec![2..3, 5..6]));
        for invalid in [stale, discontinuous] {
            let model = AnyModel::new("stale-test", "", vec![], move |_, _| {
                Ok(vec![invalid.clone()])
            });
            assert!(score(&rows, &model)
                .unwrap_err()
                .to_string()
                .contains("contiguous source span"));
        }
    }

    #[test]
    fn rejects_invalid_source_evidence() {
        for row in [
            serde_json::json!({"text":"🦊 year", "time_expressions":[{"text":"year","start_char":0,"end_char":4}]}),
            serde_json::json!({"text":"🦊 year", "time_expressions":[{"text":"year","start_char":2,"end_char":22}]}),
        ] {
            let row: TimexSentence = serde_json::from_value(row).unwrap();
            assert!(row
                .gold_spans(&row.text.chars().collect::<Vec<_>>())
                .is_err());
        }
        assert!(serde_json::from_value::<TimexSentence>(
            serde_json::json!({"text":"missing gold"})
        )
        .is_err());
    }
}
