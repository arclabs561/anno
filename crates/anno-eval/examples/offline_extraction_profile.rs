//! Compare the built-in offline extraction backends on one labeled fixture.
//!
//! The profile reports strict, micro-averaged span-and-label precision, recall,
//! and F1. It also reports the first corpus pass separately from repeated warm
//! passes. Warm per-document quantiles time only successful extraction calls,
//! excluding gold construction and scoring. Model construction is timed separately;
//! a backend may perform additional initialization on its first extraction.
//!
//! ```sh
//! cargo run -p anno-eval --example offline_extraction_profile
//! cargo run -p anno-eval --example offline_extraction_profile -- wikigold.conll
//! ```
//!
//! The fixture is deliberately small and synthetic. It is useful for a
//! deterministic smoke measurement, not for a production-quality claim.

use anno_eval::eval::dataset_registry::DatasetId;
use anno_eval::eval::loader::DatasetLoader;
use anno_eval::eval::ner_metrics::{evaluate_entities, NerEvalResults};
use anno_eval::eval::synthetic::{news_dataset, AnnotatedExample};
use anno_eval::eval::{entity_type_matches, GoldEntity};
use anno_eval::{Entity, HeuristicNER, Model, RegexNER, Result, StackedNER};
use std::time::{Duration, Instant};

const WARM_PASSES: usize = 10;

struct Profile {
    name: &'static str,
    metrics: NerEvalResults,
    construction: Duration,
    warm_calls: Vec<Duration>,
    first_pass: Duration,
    warm_pass_average: Duration,
}

fn evaluate(
    examples: &[AnnotatedExample],
    mut extract: impl FnMut(&str) -> Result<Vec<Entity>>,
) -> Result<NerEvalResults> {
    let mut results = NerEvalResults::new();

    for example in examples {
        let gold = example
            .entities
            .iter()
            .map(|entity| {
                Entity::new(
                    entity.text.clone(),
                    entity.entity_type.clone(),
                    entity.start,
                    entity.end,
                    1.0,
                )
            })
            .collect::<Vec<_>>();
        let predicted = extract(&example.text)?;
        results.merge(&evaluate_entities(&gold, &predicted));
    }

    Ok(results)
}

fn profile<M: Model>(
    name: &'static str,
    make_model: impl FnOnce() -> M,
    examples: &[AnnotatedExample],
) -> Result<Profile> {
    let first_start = Instant::now();
    let model = make_model();
    let construction = first_start.elapsed();
    let metrics = evaluate(examples, |text| model.extract_entities(text, None))?;
    let first_pass = first_start.elapsed();

    let mut warm_calls = Vec::with_capacity(WARM_PASSES * examples.len());
    let warm_start = Instant::now();
    for _ in 0..WARM_PASSES {
        // The result is deliberately discarded: the first-pass scores above are
        // the measured quality output, while this loop measures warmed calls.
        evaluate(examples, |text| {
            let start = Instant::now();
            let entities = model.extract_entities(text, None)?;
            warm_calls.push(start.elapsed());
            Ok(entities)
        })?;
    }
    let warm_pass_average = warm_start.elapsed() / WARM_PASSES as u32;

    warm_calls.sort_unstable();

    Ok(Profile {
        name,
        metrics,
        construction,
        warm_calls,
        first_pass,
        warm_pass_average,
    })
}

/// Nearest-rank percentile on a nonempty sorted sample, with p in 1..=100.
fn percentile(sorted: &[Duration], p: usize) -> Duration {
    sorted[(sorted.len() * p).div_ceil(100) - 1]
}

fn offline_stack() -> StackedNER {
    StackedNER::builder()
        .layer(RegexNER::new())
        .layer(HeuristicNER::new())
        .build()
}

fn contains_gold(entities: &[Entity], gold: &GoldEntity) -> bool {
    entities.iter().any(|e| {
        e.start() == gold.start
            && e.end() == gold.end
            && entity_type_matches(&e.entity_type, &gold.entity_type)
    })
}

fn report_retention(examples: &[AnnotatedExample]) -> Result<()> {
    let regex = RegexNER::new();
    let heuristic = HeuristicNER::new();
    let stack = offline_stack();
    let (mut retained, mut lost, mut gained, mut missed) = (0, 0, 0, 0);
    let mut surface_mismatches = 0;
    for example in examples {
        let mut proposed = regex.extract_entities(&example.text, None)?;
        proposed.extend(heuristic.extract_entities(&example.text, None)?);
        let output = stack
            .extract_entities_with_report(
                &example.text,
                None,
                anno::StackedExtractionPolicy::Strict,
            )
            .map_err(|e| e.into_error())?;
        for gold in &example.entities {
            match (
                contains_gold(&proposed, gold),
                contains_gold(output.entities(), gold),
            ) {
                (true, true) => retained += 1,
                (true, false) => lost += 1,
                (false, true) => gained += 1,
                (false, false) => missed += 1,
            }
        }
        surface_mismatches += output
            .entities()
            .iter()
            .filter(|e| e.extract_text(&example.text) != e.text)
            .count();
    }
    println!("gold spans: retained={retained} lost_in_stack={lost} gained_in_stack={gained} missed_by_both={missed}");
    println!("output spans with source-text mismatches: {surface_mismatches}");
    Ok(())
}

fn main() -> Result<()> {
    let (fixture, examples) = if let Some(path) = std::env::args_os().nth(1) {
        let content = std::fs::read_to_string(path)?;
        let dataset = DatasetLoader::new()?.parse_content_str(&content, DatasetId::WikiGold)?;
        (
            "WikiGold",
            dataset
                .sentences
                .into_iter()
                .map(|s| AnnotatedExample::new(s.text(), s.entities()))
                .collect(),
        )
    } else {
        ("anno_eval::synthetic::news_dataset", news_dataset())
    };
    let gold_count = examples
        .iter()
        .map(|example| example.entities.len())
        .sum::<usize>();

    if gold_count == 0 {
        return Err(anno::Error::InvalidInput(
            "no labeled spans to evaluate".into(),
        ));
    }

    println!(
        "fixture={fixture} examples={} gold_spans={} match=strict-exact-span-and-label",
        examples.len(),
        gold_count
    );
    println!("warm_passes={WARM_PASSES}");
    println!("| backend | P | R | F1 | first pass | warm pass average | constructor | warm calls | extract p50 | extract p95 |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");

    let profiles = [
        profile("regex", RegexNER::new, &examples)?,
        profile("heuristic", HeuristicNER::new, &examples)?,
        profile("stacked", offline_stack, &examples)?,
    ];

    for profile in profiles {
        let strict = &profile.metrics.strict;
        println!(
            "| {} | {:.1}% | {:.1}% | {:.1}% | {:?} | {:?} | {:?} | {} | {:?} | {:?} |",
            profile.name,
            strict.precision_exact() * 100.0,
            strict.recall_exact() * 100.0,
            strict.f1_exact() * 100.0,
            profile.first_pass,
            profile.warm_pass_average,
            profile.construction,
            profile.warm_calls.len(),
            percentile(&profile.warm_calls, 50),
            percentile(&profile.warm_calls, 95),
        );
    }

    report_retention(&examples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_percentiles_use_observed_samples() {
        let values: Vec<_> = (1..=21).map(Duration::from_nanos).collect();
        assert_eq!(percentile(&values, 50), Duration::from_nanos(11));
        assert_eq!(percentile(&values, 95), Duration::from_nanos(20));
        assert_eq!(percentile(&values[..1], 95), Duration::from_nanos(1));
    }

    #[test]
    fn warm_calls_cover_every_document_and_keep_first_pass_scores() {
        let examples = news_dataset();
        let model = HeuristicNER::new();
        let expected = evaluate(&examples, |text| model.extract_entities(text, None)).unwrap();
        let measured = profile("heuristic", HeuristicNER::new, &examples).unwrap();
        assert_eq!(measured.warm_calls.len(), WARM_PASSES * examples.len());
        assert_eq!(
            serde_json::to_value(measured.metrics.strict).unwrap(),
            serde_json::to_value(expected.strict).unwrap()
        );
    }
}
