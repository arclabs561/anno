//! Fixed-window CUAD passage retrieval with lexir and rankops.
//!
//! Supply one category and pipe the official test split into this example:
//! `unzip -p data.zip test.json | cargo run --locked -q -p anno-eval --example cuad_retrieval -- 'Non-Compete'`
//! Get the release from <https://github.com/TheAtticusProject/cuad/tree/67faa0e6023b04fcaae6cc09497ab00e5d63a2a2>.
//! Hit@3 counts each full gold answer once, including missed answers in the denominator.
//! Returned-on-negative counts unanswerable questions to which the retriever offered a passage.
use anno::offset::SpanConverter;
use anno_eval::eval::cuad;
use lexir::bm25::{Bm25Params, InvertedIndex};
use regex::Regex;
use std::collections::BTreeSet;
use std::io;
use std::ops::Range;
use std::sync::LazyLock;

static TERMS: LazyLock<Regex> = LazyLock::new(|| Regex::new("[a-z]+").unwrap());
static WORDS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\S+").unwrap());
const STOP: &[&str] = &[
    "a", "an", "the", "this", "that", "of", "in", "for", "to", "and", "or", "any", "which", "what",
    "with", "by", "be", "on", "is",
];

fn terms(s: &str) -> Vec<String> {
    TERMS
        .find_iter(&s.to_lowercase())
        .map(|m| m.as_str().to_owned())
        .collect()
}

fn windows(text: &str) -> Vec<Range<usize>> {
    let words: Vec<_> = WORDS.find_iter(text).collect();
    let mut out = Vec::new();
    for start in (0..words.len()).step_by(48) {
        let end = (start + 96).min(words.len());
        out.push(words[start].start()..words[end - 1].end());
        if end == words.len() {
            break;
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let category = std::env::args()
        .nth(1)
        .ok_or("expected a CUAD category, e.g. 'Non-Compete'")?;
    let docs = cuad::read(io::stdin().lock())?;
    println!("{}", evaluate(&docs, &category)?);
    Ok(())
}

fn evaluate(
    docs: &[cuad::Contract],
    category: &str,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let mut ranks = Vec::new();
    let mut questions = 0;
    let mut unanswerable = 0;
    let mut returned_on_negative = 0;
    for doc in docs {
        for p in &doc.paragraphs {
            let matching = p
                .qas
                .iter()
                .filter(|q| q.id.rsplit_once("__").is_some_and(|(_, c)| c == category));
            let passages = windows(&p.context);
            let mut index = InvertedIndex::new();
            for (i, span) in passages.iter().enumerate() {
                index.add_document(u32::try_from(i)?, &terms(&p.context[span.clone()]));
            }
            let offsets = SpanConverter::new(&p.context);
            for q in matching {
                questions += 1;
                let detail = q
                    .question
                    .split_once("Details: ")
                    .map_or(q.question.as_str(), |(_, d)| d);
                let query: BTreeSet<_> = terms(&format!("{category} {detail}"))
                    .into_iter()
                    .filter(|s| !STOP.contains(&s.as_str()))
                    .collect();
                let query: Vec<_> = query.into_iter().collect();
                let selected = if passages.is_empty() || query.is_empty() {
                    Vec::new()
                } else {
                    index.retrieve(&query, 3.min(passages.len()), Bm25Params::default())?
                };
                if q.is_impossible {
                    unanswerable += 1;
                    returned_on_negative += usize::from(!selected.is_empty());
                }
                for answer in &q.answers {
                    let start = offsets.char_to_byte(answer.answer_start);
                    let end = start + answer.text.len();
                    let hit = selected.iter().position(|(i, _)| {
                        let window = &passages[*i as usize];
                        window.start <= start && end <= window.end
                    });
                    ranks.push(hit.map_or(0, |rank| rank + 1));
                }
            }
        }
    }
    if questions == 0 {
        return Err(format!("no CUAD question for category {category:?}").into());
    }
    Ok(serde_json::json!({
        "category": category, "contracts": docs.len(), "questions": questions, "answers": ranks.len(),
        "full_answers_in_top3": ranks.iter().filter(|rank| **rank > 0).count(),
        "unanswerable": unanswerable, "returned_on_negative": returned_on_negative,
        "hit_at_3": (!ranks.is_empty()).then(|| rankops::metrics::hits_at_k(&ranks, 3)),
        "mrr_at_3": (!ranks.is_empty()).then(|| rankops::metrics::mrr(&ranks)),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn partial_passages_are_misses_and_negative_suggestions_are_visible() {
        let long_clause = "law ".repeat(97).trim_end().to_string();
        let context = format!("é {long_clause}");
        let data = json!({"data":[{"title":"fixture","paragraphs":[
            {"context":context,"qas":[{"id":"a__Law","question":"law","is_impossible":false,
            "answers":[{"text":"law","answer_start":2},{"text":long_clause,"answer_start":2}]}]},
            {"context":"ask a law clerk","qas":[{"id":"b__Law","question":"law","is_impossible":true,"answers":[]}]}
        ]}]});
        let docs = cuad::read(data.to_string().as_bytes()).unwrap();
        let report = evaluate(&docs, "Law").unwrap();
        assert_eq!(report["answers"], 2);
        assert_eq!(report["full_answers_in_top3"], 1);
        assert_eq!(report["hit_at_3"], 0.5);
        assert_eq!(report["returned_on_negative"], 1);
        assert!(evaluate(&docs, "Not present").is_err());
    }
}
