//! Fixed-window CUAD passage retrieval with lexir and rankops.
//!
//! Supply one category and pipe the official test split into this example:
//! `unzip -p data.zip test.json | cargo run --locked -q -p anno-eval --example cuad_retrieval -- 'Non-Compete'`
//! Get the release from <https://github.com/TheAtticusProject/cuad/tree/67faa0e6023b04fcaae6cc09497ab00e5d63a2a2>.
//! Hit@3 counts each full gold answer once, including missed answers in the denominator.
//! Compare BM25 with a literal category lookup that takes matching windows in document order.
//! Both methods can suggest up to three windows per question. The output counts the actual
//! number of suggested windows, including overlapping windows, and how many contain no full
//! annotated answer. Returned-on-negative counts explicitly unanswerable questions that get
//! at least one suggestion. A contract without the question is excluded from both methods.
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

const REVIEW_LIMIT: usize = 3;

#[derive(Default)]
struct Results {
    ranks: Vec<usize>,
    passages_for_review: usize,
    passages_without_full_answer: usize,
    passages_on_negative: usize,
    returned_on_negative: usize,
}

impl Results {
    fn score(&mut self, selected: &[usize], passages: &[Range<usize>], answers: &[Range<usize>]) {
        let mut supported = vec![false; selected.len()];
        for answer in answers {
            let mut first_hit = 0;
            for (slot, &i) in selected.iter().enumerate() {
                if passages[i].start <= answer.start && answer.end <= passages[i].end {
                    supported[slot] = true;
                    if first_hit == 0 {
                        first_hit = slot + 1;
                    }
                }
            }
            self.ranks.push(first_hit);
        }
        self.passages_for_review += selected.len();
        self.passages_without_full_answer += supported.iter().filter(|&&hit| !hit).count();
        if answers.is_empty() {
            self.passages_on_negative += selected.len();
            self.returned_on_negative += usize::from(!selected.is_empty());
        }
    }

    fn summary(&self) -> serde_json::Value {
        serde_json::json!({
            "answers": self.ranks.len(),
            "full_answers_in_top3": self.ranks.iter().filter(|&&rank| rank > 0).count(),
            "returned_on_negative": self.returned_on_negative,
            "passages_for_review": self.passages_for_review,
            "passages_without_full_answer": self.passages_without_full_answer,
            "passages_on_negative": self.passages_on_negative,
            "hit_at_3": (!self.ranks.is_empty()).then(|| rankops::metrics::hits_at_k(&self.ranks, REVIEW_LIMIT)),
            "mrr_at_3": (!self.ranks.is_empty()).then(|| rankops::metrics::mrr(&self.ranks)),
        })
    }
}

fn evaluate(
    docs: &[cuad::Contract],
    category: &str,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let mut bm25 = Results::default();
    let mut phrase = Results::default();
    let category_terms = terms(category);
    let mut questions = 0;
    let mut unanswerable = 0;
    let mut contracts_with_question = 0;
    for doc in docs {
        let mut found_category = false;
        for p in &doc.paragraphs {
            let mut matching = p
                .qas
                .iter()
                .filter(|q| q.id.rsplit_once("__").is_some_and(|(_, c)| c == category))
                .peekable();
            if matching.peek().is_none() {
                continue;
            }
            found_category = true;
            let passages = windows(&p.context);
            let mut index = InvertedIndex::new();
            let mut phrase_hits = Vec::new();
            for (i, span) in passages.iter().enumerate() {
                let tokens = terms(&p.context[span.clone()]);
                if phrase_hits.len() < REVIEW_LIMIT
                    && !category_terms.is_empty()
                    && tokens
                        .windows(category_terms.len())
                        .any(|w| w == category_terms)
                {
                    phrase_hits.push(i);
                }
                index.add_document(u32::try_from(i)?, &tokens);
            }
            let offsets = SpanConverter::new(&p.context);
            for q in matching {
                questions += 1;
                unanswerable += usize::from(q.is_impossible);
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
                    index
                        .retrieve(
                            &query,
                            REVIEW_LIMIT.min(passages.len()),
                            Bm25Params::default(),
                        )?
                        .into_iter()
                        .map(|(i, _)| i as usize)
                        .collect()
                };
                let answers: Vec<_> = q
                    .answers
                    .iter()
                    .map(|answer| {
                        let start = offsets.char_to_byte(answer.answer_start);
                        start..start + answer.text.len()
                    })
                    .collect();
                bm25.score(&selected, &passages, &answers);
                phrase.score(&phrase_hits, &passages, &answers);
            }
        }
        contracts_with_question += usize::from(found_category);
    }
    if questions == 0 {
        return Err(format!("no CUAD question for category {category:?}").into());
    }
    // Preserve the top-level BM25 result fields used by earlier replays.
    let mut report = bm25.summary();
    report["category"] = category.into();
    report["contracts"] = docs.len().into();
    report["contracts_without_question"] = (docs.len() - contracts_with_question).into();
    report["questions"] = questions.into();
    report["unanswerable"] = unanswerable.into();
    report["max_passages_per_question"] = REVIEW_LIMIT.into();
    report["category_phrase_baseline"] = phrase.summary();
    Ok(report)
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
        assert_eq!(report["passages_for_review"], 3);
        assert_eq!(report["passages_without_full_answer"], 2);
        assert_eq!(report["passages_on_negative"], 1);
        assert_eq!(
            report["category_phrase_baseline"]["full_answers_in_top3"],
            1
        );
        assert!(evaluate(&docs, "Not present").is_err());
    }

    #[test]
    fn phrase_lookup_can_abstain_and_missing_questions_are_excluded() {
        let data = json!({"data":[
            {"title":"positive","paragraphs":[{"context":"É legal remedy","qas":[
                {"id":"a__Governing Law","question":"Which legal remedy?","is_impossible":false,
                "answers":[{"text":"legal remedy","answer_start":2}]}]}]},
            {"title":"negative","paragraphs":[{"context":"GOVERNING-LAW heading","qas":[
                {"id":"b__Governing Law","question":"Which legal remedy?","is_impossible":true,"answers":[]}]}]},
            {"title":"unasked","paragraphs":[{"context":"Governing law","qas":[]}]}
        ]});
        let docs = cuad::read(data.to_string().as_bytes()).unwrap();
        let report = evaluate(&docs, "Governing Law").unwrap();
        assert_eq!(report["questions"], 2);
        assert_eq!(report["unanswerable"], 1);
        assert_eq!(report["contracts_without_question"], 1);
        assert_eq!(report["full_answers_in_top3"], 1);
        let phrase = &report["category_phrase_baseline"];
        assert_eq!(phrase["full_answers_in_top3"], 0);
        assert_eq!(phrase["passages_for_review"], 1);
        assert_eq!(phrase["passages_without_full_answer"], 1);
        assert_eq!(phrase["returned_on_negative"], 1);
    }
}
