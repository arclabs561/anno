//! Small inputs that expose the score conventions used by the task evaluator.
use anno_eval::eval::coref::{CorefChain, Mention};
use anno_eval::eval::coref_metrics::{compute_chain_length_stratified, CorefEvaluation};
use anno_eval::eval::loader::{AnnotatedSentence, AnnotatedToken, DatasetId};
use anno_eval::eval::ner_metrics::evaluate_entities;
use anno_eval::{Entity, EntityType};

#[test]
fn missing_singleton_is_visible_in_the_singleton_metric() {
    let pair = CorefChain::new(vec![Mention::new("Jo", 0, 2), Mention::new("she", 9, 12)]);
    let single = CorefChain::new(vec![Mention::new("Lee", 5, 8)]);
    let gold = [pair, single];
    for (predicted, expected) in [(&gold[..], 1.0), (&gold[..1], 0.0)] {
        let stratified = compute_chain_length_stratified(predicted, &gold);
        assert_eq!(stratified.singleton_count, 1);
        assert_eq!(stratified.singleton_f1, expected);
        println!(
            "singleton F1 {expected}, CoNLL {}",
            CorefEvaluation::compute(predicted, &gold).conll_f1
        );
    }
}

#[test]
fn strict_scoring_counts_duplicate_emissions() {
    let person = Entity::new("Jo", EntityType::Person, 0, 2, 1.0);
    let gold = [person.clone()];
    let repeated = [person.clone(), person];
    assert_eq!(evaluate_entities(&gold, &gold).summary().strict_f1, 1.0);
    let score = evaluate_entities(&gold, &repeated).summary();
    assert_eq!((score.strict_precision, score.strict_recall), (0.5, 1.0));
    assert!((score.strict_f1 - 2.0 / 3.0).abs() < 1e-9);
}

#[test]
fn bio_sentence_support_depends_on_the_opening_tag() {
    for (first_tag, expected) in [("B-PER", 1), ("I-PER", 0)] {
        let sentence = AnnotatedSentence {
            source_dataset: DatasetId::CoNLL2003Sample,
            tokens: [("Jo", first_tag), ("Lee", "I-PER")]
                .into_iter()
                .map(|(word, tag)| AnnotatedToken {
                    text: word.into(),
                    ner_tag: tag.into(),
                })
                .collect(),
        };
        assert_eq!(sentence.entities().len(), expected);
    }
}
