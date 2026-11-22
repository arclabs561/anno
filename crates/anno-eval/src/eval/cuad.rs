//! Read official CUAD question/answer files without flattening their overlapping clauses into BIO tags.
use crate::{Error, Result};
use anno::offset::SpanConverter;
use serde::Deserialize;
use std::io::Read;

/// A contract with its original paragraphs.
#[derive(Debug, Deserialize)]
pub struct Contract {
    /// Original CUAD contract title.
    pub title: String,
    /// The context for each set of questions.
    pub paragraphs: Vec<Paragraph>,
}

/// Questions share one immutable source string.
#[derive(Debug, Deserialize)]
pub struct Paragraph {
    /// Original contract text.
    pub context: String,
    /// Questions, including those without answers.
    pub qas: Vec<Question>,
}

/// An annotated question about one contract.
#[derive(Debug, Deserialize)]
pub struct Question {
    /// Source question ID. The suffix after the last double underscore is its category.
    pub id: String,
    /// Full published prompt.
    pub question: String,
    /// Expert-marked source spans.
    pub answers: Vec<Answer>,
    /// Whether this contract has no answer to this question.
    pub is_impossible: bool,
}

/// Source span whose start is measured in Unicode scalar values.
#[derive(Debug, Deserialize)]
pub struct Answer {
    /// Verbatim answer from the paragraph.
    pub text: String,
    /// Zero-based character index in the paragraph.
    pub answer_start: usize,
}

/// Read a CUAD release member such as `test.json`, validating all answer spans.
///
/// Questions stay separate even when they annotate the same piece of source text.
pub fn read(reader: impl Read) -> Result<Vec<Contract>> {
    #[derive(Deserialize)]
    struct Release {
        data: Vec<Contract>,
    }
    let release: Release = serde_json::from_reader(reader)
        .map_err(|e| Error::InvalidInput(format!("invalid CUAD JSON: {e}")))?;
    for doc in &release.data {
        for p in &doc.paragraphs {
            let offsets = SpanConverter::new(&p.context);
            let chars = p.context.chars().count();
            for q in &p.qas {
                if q.is_impossible != q.answers.is_empty() {
                    return Err(Error::InvalidInput(format!(
                        "CUAD question {} has contradictory answerability",
                        q.id
                    )));
                }
                for a in &q.answers {
                    let end = a.answer_start.checked_add(a.text.chars().count());
                    let valid = end
                        .filter(|&end| end <= chars && end > a.answer_start)
                        .is_some_and(|end| {
                            p.context
                                [offsets.char_to_byte(a.answer_start)..offsets.char_to_byte(end)]
                                == a.text
                        });
                    if !valid {
                        return Err(Error::InvalidInput(format!(
                            "CUAD question {}: answer does not match source at character {}",
                            q.id, a.answer_start
                        )));
                    }
                }
            }
        }
    }
    Ok(release.data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn repeated_text_uses_supplied_character_offsets_and_preserves_absence() {
        let mut data = json!({"data":[{"title":"fixture","paragraphs":[{"context":"法 é law law","qas":[
            {"id":"doc__Rule","question":"Which law?","answers":[{"text":"law","answer_start":8}],"is_impossible":false},
            {"id":"doc__End","question":"When?","answers":[],"is_impossible":true}
        ]}]}]});
        let docs = read(data.to_string().as_bytes()).unwrap();
        assert_eq!(docs[0].paragraphs[0].qas[0].answers[0].answer_start, 8);
        assert!(docs[0].paragraphs[0].qas[1].is_impossible);
        let answer = &mut data["data"][0]["paragraphs"][0]["qas"][0]["answers"][0];
        answer["answer_start"] = json!(7);
        assert!(read(data.to_string().as_bytes()).is_err());
        data["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"] =
            json!(usize::MAX);
        assert!(read(data.to_string().as_bytes()).is_err());
    }
}
