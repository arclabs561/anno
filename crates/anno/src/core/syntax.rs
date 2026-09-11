//! Parser-neutral constituency syntax trees.
//!
//! A [`ParseDocument`] stores source text, globally offset tokens, and one
//! preorder, projective constituency tree per sentence. It deliberately
//! contains no parser trait or model-specific fields: adapters can import a
//! parser's output without making its runtime a dependency of `anno`.

use super::{types::CharSpan, Error, Result};
use serde::{de, Deserialize, Deserializer, Serialize};

const MAX_TREE_DEPTH: usize = 512;

/// A validated constituency parse for one source document.
///
/// Token offsets count Unicode scalar values in [`Self::text`]. Constituents
/// use sentence-local, half-open token offsets. JSON serialization has the
/// stable shape `{ text, sentences }` described by this module's child types.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ParseDocument {
    text: String,
    sentences: Vec<SentenceParse>,
}

/// One sentence's tokens and preorder constituency nodes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SentenceParse {
    tokens: Vec<ParseToken>,
    constituents: Vec<Constituent>,
}

/// One source-grounded token in a constituency parse.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ParseToken {
    text: String,
    start: usize,
    end: usize,
    tag: Option<String>,
}

/// A labeled constituent over a sentence-local half-open token interval.
///
/// `labels` collapses unary nodes in outer-to-inner order. Empty labels are
/// meaningful and are retained for parser output that has an unlabeled root or
/// leaf.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Constituent {
    start: usize,
    end: usize,
    labels: Vec<String>,
}

#[derive(Deserialize)]
struct RawDocument {
    text: String,
    sentences: Vec<RawSentence>,
}

#[derive(Deserialize)]
struct RawSentence {
    tokens: Vec<RawToken>,
    constituents: Vec<RawConstituent>,
}

#[derive(Deserialize)]
struct RawToken {
    text: String,
    start: usize,
    end: usize,
    tag: Option<String>,
}

#[derive(Deserialize)]
struct RawConstituent {
    start: usize,
    end: usize,
    labels: Vec<String>,
}

impl ParseDocument {
    /// Decode and validate a parse document from its interchange JSON.
    ///
    /// The returned error describes malformed JSON or a violated source/tree
    /// invariant. Validation also runs for ordinary serde deserialization.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|error| Error::parse(error.to_string()))
    }

    /// Return the complete, unmodified source text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Return sentence parses in source order.
    #[must_use]
    pub fn sentences(&self) -> &[SentenceParse] {
        &self.sentences
    }
}

impl SentenceParse {
    /// Return source-grounded tokens in sentence order.
    #[must_use]
    pub fn tokens(&self) -> &[ParseToken] {
        &self.tokens
    }

    /// Return constituents in preorder.
    #[must_use]
    pub fn constituents(&self) -> &[Constituent] {
        &self.constituents
    }

    /// Render the syntax tree in bracketed notation.
    ///
    /// Original source token text is used, with Penn-style escapes for bracket
    /// characters. A token tag becomes its preterminal when present; a token
    /// without a tag is rendered as a bare escaped leaf. Thus this rendering
    /// can differ from a model's normalized token stream.
    #[must_use]
    pub fn to_bracketed(&self) -> String {
        render_node(self, 0).0
    }
}

impl ParseToken {
    /// Return the exact source token text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Return the inclusive global Unicode-character start offset.
    #[must_use]
    pub const fn start(&self) -> usize {
        self.start
    }

    /// Return the exclusive global Unicode-character end offset.
    #[must_use]
    pub const fn end(&self) -> usize {
        self.end
    }

    /// Return this token's global Unicode-character span in the source text.
    #[must_use]
    pub const fn span(&self) -> CharSpan {
        CharSpan::new(self.start, self.end)
    }

    /// Return the optional POS tag supplied by the parser.
    #[must_use]
    pub fn tag(&self) -> Option<&str> {
        self.tag.as_deref()
    }
}

impl Constituent {
    /// Return the inclusive sentence-local token start index.
    #[must_use]
    pub const fn start(&self) -> usize {
        self.start
    }

    /// Return the exclusive sentence-local token end index.
    #[must_use]
    pub const fn end(&self) -> usize {
        self.end
    }

    /// Return this constituent's sentence-local half-open token range.
    #[must_use]
    pub fn span(&self) -> std::ops::Range<usize> {
        self.start..self.end
    }

    /// Return unary-collapsed labels in outer-to-inner order.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }
}

impl<'de> Deserialize<'de> for ParseDocument {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawDocument::deserialize(deserializer)?;
        Self::try_from(raw).map_err(de::Error::custom)
    }
}

impl TryFrom<RawDocument> for ParseDocument {
    type Error = Error;

    fn try_from(raw: RawDocument) -> Result<Self> {
        let character_boundaries = character_boundaries(&raw.text);
        let text_len = character_boundaries.len() - 1;
        let mut previous_end = 0;
        let mut sentences = Vec::with_capacity(raw.sentences.len());

        for (sentence_index, raw_sentence) in raw.sentences.into_iter().enumerate() {
            if raw_sentence.tokens.is_empty() {
                return Err(Error::invalid_input(format!(
                    "sentence {sentence_index} has no tokens"
                )));
            }
            let mut tokens = Vec::with_capacity(raw_sentence.tokens.len());
            for (token_index, token) in raw_sentence.tokens.into_iter().enumerate() {
                validate_token(
                    &raw.text,
                    &character_boundaries,
                    text_len,
                    previous_end,
                    sentence_index,
                    token_index,
                    &token,
                )?;
                previous_end = token.end;
                tokens.push(ParseToken {
                    text: token.text,
                    start: token.start,
                    end: token.end,
                    tag: token.tag,
                });
            }
            let constituents =
                validate_tree(sentence_index, tokens.len(), raw_sentence.constituents)?;
            sentences.push(SentenceParse {
                tokens,
                constituents,
            });
        }

        if !chars_between(&raw.text, &character_boundaries, previous_end, text_len)
            .is_some_and(|gap| gap.chars().all(char::is_whitespace))
        {
            return Err(Error::invalid_input(
                "non-whitespace source text is not covered by tokens",
            ));
        }
        Ok(Self {
            text: raw.text,
            sentences,
        })
    }
}

fn validate_token(
    text: &str,
    character_boundaries: &[usize],
    text_len: usize,
    previous_end: usize,
    sentence_index: usize,
    token_index: usize,
    token: &RawToken,
) -> Result<()> {
    if token.start > token.end {
        return Err(Error::invalid_input(format!(
            "sentence {sentence_index} token {token_index} has reversed offsets"
        )));
    }
    if token.start == token.end || token.end > text_len {
        return Err(Error::invalid_input(format!(
            "sentence {sentence_index} token {token_index} has an empty or out-of-bounds span"
        )));
    }
    if token.start < previous_end {
        return Err(Error::invalid_input(format!(
            "sentence {sentence_index} token {token_index} overlaps an earlier token"
        )));
    }
    let Some(source) = chars_between(text, character_boundaries, token.start, token.end) else {
        return Err(Error::invalid_input(format!(
            "sentence {sentence_index} token {token_index} has invalid character offsets"
        )));
    };
    if source != token.text {
        return Err(Error::invalid_input(format!(
            "sentence {sentence_index} token {token_index} does not match its source span"
        )));
    }
    if token.text.chars().any(char::is_whitespace) {
        return Err(Error::invalid_input(format!(
            "sentence {sentence_index} token {token_index} contains whitespace"
        )));
    }
    if !chars_between(text, character_boundaries, previous_end, token.start)
        .is_some_and(|gap| gap.chars().all(char::is_whitespace))
    {
        return Err(Error::invalid_input(format!(
            "sentence {sentence_index} token {token_index} skips non-whitespace source text"
        )));
    }
    if let Some(tag) = &token.tag {
        validate_symbol(tag, "token tag")?;
    }
    Ok(())
}

fn validate_tree(
    sentence_index: usize,
    token_count: usize,
    raw_nodes: Vec<RawConstituent>,
) -> Result<Vec<Constituent>> {
    if raw_nodes.is_empty() {
        return Err(Error::invalid_input(format!(
            "sentence {sentence_index} has no root constituent"
        )));
    }
    let mut nodes = Vec::with_capacity(raw_nodes.len());
    let mut stack: Vec<(usize, usize, usize, usize)> = Vec::new();
    // node index, start, end, next child start

    for (node_index, node) in raw_nodes.into_iter().enumerate() {
        if node.start >= node.end || node.end > token_count {
            return Err(Error::invalid_input(format!(
                "sentence {sentence_index} constituent {node_index} has an invalid token span"
            )));
        }
        for label in &node.labels {
            validate_symbol(label, "constituent label")?;
        }
        if node_index == 0 {
            if node.start != 0 || node.end != token_count {
                return Err(Error::invalid_input(format!(
                    "sentence {sentence_index} root does not cover every token"
                )));
            }
        } else {
            while let Some((_, start, end, next_child_start)) = stack.last().copied() {
                if node.start < end {
                    break;
                }
                if next_child_start != end && end - start != 1 {
                    return Err(Error::invalid_input(format!(
                        "sentence {sentence_index} constituent children do not partition their parent"
                    )));
                }
                stack.pop();
            }
            let Some((_, parent_start, parent_end, parent_next_child_start)) = stack.last_mut()
            else {
                return Err(Error::invalid_input(format!(
                    "sentence {sentence_index} has a constituent outside its root"
                )));
            };
            if node.end > *parent_end {
                return Err(Error::invalid_input(format!(
                    "sentence {sentence_index} has crossing constituents"
                )));
            }
            if node.start == *parent_start && node.end == *parent_end {
                return Err(Error::invalid_input(format!(
                    "sentence {sentence_index} has duplicate constituent spans"
                )));
            }
            if node.start != *parent_next_child_start {
                return Err(Error::invalid_input(format!(
                    "sentence {sentence_index} constituent children do not partition their parent"
                )));
            }
            *parent_next_child_start = node.end;
        }

        nodes.push(Constituent {
            start: node.start,
            end: node.end,
            labels: node.labels,
        });
        if stack.len() == MAX_TREE_DEPTH {
            return Err(Error::invalid_input(format!(
                "sentence {sentence_index} exceeds the maximum tree depth"
            )));
        }
        stack.push((node_index, node.start, node.end, node.start));
    }

    while let Some((_, start, end, next_child_start)) = stack.pop() {
        if next_child_start != end && end - start != 1 {
            return Err(Error::invalid_input(format!(
                "sentence {sentence_index} constituent children do not partition their parent"
            )));
        }
    }

    // A singleton is a leaf only when it has no child. A non-singleton must
    // have children; the partition checks above establish all singleton leaves.
    for (index, node) in nodes.iter().enumerate() {
        let has_child = nodes
            .get(index + 1)
            .is_some_and(|next| next.start >= node.start && next.end <= node.end);
        if node.end - node.start == 1 && has_child {
            return Err(Error::invalid_input(format!(
                "sentence {sentence_index} has duplicate singleton constituents"
            )));
        }
        if node.end - node.start > 1 && !has_child {
            return Err(Error::invalid_input(format!(
                "sentence {sentence_index} is missing a leaf constituent"
            )));
        }
    }
    Ok(nodes)
}

fn validate_symbol(symbol: &str, kind: &str) -> Result<()> {
    if symbol.is_empty()
        || symbol
            .chars()
            .any(|c| c.is_whitespace() || c.is_control() || matches!(c, '(' | ')'))
    {
        return Err(Error::invalid_input(format!(
            "{kind} must be non-empty and contain no whitespace, controls or parentheses"
        )));
    }
    Ok(())
}

fn character_boundaries(text: &str) -> Vec<usize> {
    let mut boundaries = text
        .char_indices()
        .map(|(offset, _)| offset)
        .collect::<Vec<_>>();
    boundaries.push(text.len());
    boundaries
}

fn chars_between<'a>(
    text: &'a str,
    character_boundaries: &[usize],
    start: usize,
    end: usize,
) -> Option<&'a str> {
    if start > end {
        return None;
    }
    let start_byte = *character_boundaries.get(start)?;
    let end_byte = *character_boundaries.get(end)?;
    text.get(start_byte..end_byte)
}

fn render_node(sentence: &SentenceParse, index: usize) -> (String, usize) {
    let node = &sentence.constituents[index];
    let base = if node.end - node.start == 1 {
        let token = &sentence.tokens[node.start];
        let leaf = escape_token(token.text());
        match token.tag() {
            Some(tag) => format!("({tag} {leaf})"),
            None => leaf,
        }
    } else {
        let mut child_index = index + 1;
        let mut children = Vec::new();
        while child_index < sentence.constituents.len()
            && sentence.constituents[child_index].start < node.end
        {
            let (child, next_index) = render_node(sentence, child_index);
            children.push(child);
            child_index = next_index;
        }
        let joined = children.join(" ");
        let rendered = if node.labels.is_empty() {
            format!("({joined})")
        } else {
            wrap_labels(joined, &node.labels)
        };
        return (rendered, child_index);
    };
    (wrap_labels(base, &node.labels), index + 1)
}

fn wrap_labels(mut rendered: String, labels: &[String]) -> String {
    for label in labels.iter().rev() {
        rendered = format!("({label} {rendered})");
    }
    rendered
}

fn escape_token(token: &str) -> String {
    token
        .chars()
        .map(|character| match character {
            '(' => "-LRB-".to_owned(),
            ')' => "-RRB-".to_owned(),
            '[' => "-LSB-".to_owned(),
            ']' => "-RSB-".to_owned(),
            '{' => "-LCB-".to_owned(),
            '}' => "-RCB-".to_owned(),
            c if c.is_control() => c.escape_default().to_string(),
            _ => character.to_string(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_json() -> &'static str {
        r#"{
          "text":"é (works).",
          "sentences":[{
            "tokens":[
              {"text":"é","start":0,"end":1,"tag":"NNP"},
              {"text":"(","start":2,"end":3,"tag":null},
              {"text":"works","start":3,"end":8,"tag":"VBZ"},
              {"text":")","start":8,"end":9,"tag":null},
              {"text":".","start":9,"end":10,"tag":"."}
            ],
            "constituents":[
              {"start":0,"end":5,"labels":["S","VP"]},
              {"start":0,"end":1,"labels":["NP"]},
              {"start":1,"end":5,"labels":[]},
              {"start":1,"end":2,"labels":[]},
              {"start":2,"end":3,"labels":[]},
              {"start":3,"end":4,"labels":[]},
              {"start":4,"end":5,"labels":[]}
            ]
          }]
        }"#
    }

    #[test]
    fn accepts_unicode_offsets_and_renders_tree() {
        let document = ParseDocument::from_json(valid_json()).unwrap();
        let sentence = &document.sentences()[0];
        assert_eq!(sentence.tokens()[0].end(), 1);
        assert_eq!(
            sentence.to_bracketed(),
            "(S (VP (NP (NNP é)) (-LRB- (VBZ works) -RRB- (. .))))"
        );
        assert_eq!(
            serde_json::to_value(&document).unwrap(),
            serde_json::from_str::<serde_json::Value>(valid_json()).unwrap()
        );
    }

    #[test]
    fn rejects_reversed_unicode_offsets_before_normalization() {
        let json = r#"{"text":"é","sentences":[{"tokens":[{"text":"é","start":1,"end":0,"tag":null}],"constituents":[{"start":0,"end":1,"labels":[]}]}]}"#;
        assert!(ParseDocument::from_json(json)
            .unwrap_err()
            .to_string()
            .contains("reversed"));
    }

    #[test]
    fn rejects_crossing_duplicate_and_missing_leaf_trees() {
        let crossing = r#"{"text":"a b c","sentences":[{"tokens":[{"text":"a","start":0,"end":1,"tag":null},{"text":"b","start":2,"end":3,"tag":null},{"text":"c","start":4,"end":5,"tag":null}],"constituents":[{"start":0,"end":3,"labels":["S"]},{"start":0,"end":2,"labels":["X"]},{"start":0,"end":1,"labels":[]},{"start":1,"end":3,"labels":["Y"]}]}]}"#;
        assert!(ParseDocument::from_json(crossing).is_err());
        let duplicate = r#"{"text":"a","sentences":[{"tokens":[{"text":"a","start":0,"end":1,"tag":null}],"constituents":[{"start":0,"end":1,"labels":[]},{"start":0,"end":1,"labels":[]}]}]}"#;
        assert!(ParseDocument::from_json(duplicate).is_err());
        let missing_leaf = r#"{"text":"a b","sentences":[{"tokens":[{"text":"a","start":0,"end":1,"tag":null},{"text":"b","start":2,"end":3,"tag":null}],"constituents":[{"start":0,"end":2,"labels":[]}]}]}"#;
        assert!(ParseDocument::from_json(missing_leaf).is_err());
        let incomplete_partition = r#"{"text":"a b","sentences":[{"tokens":[{"text":"a","start":0,"end":1,"tag":null},{"text":"b","start":2,"end":3,"tag":null}],"constituents":[{"start":0,"end":2,"labels":[]},{"start":0,"end":1,"labels":[]}]}]}"#;
        assert!(ParseDocument::from_json(incomplete_partition).is_err());
    }

    #[test]
    fn serde_cannot_forge_invalid_document() {
        let forged = r#"{"text":"a","sentences":[{"tokens":[{"text":"a","start":0,"end":1,"tag":"bad tag"}],"constituents":[{"start":0,"end":1,"labels":[]}]}]}"#;
        assert!(serde_json::from_str::<ParseDocument>(forged).is_err());
    }

    #[test]
    fn bracket_output_cannot_emit_terminal_controls() {
        let input = serde_json::json!({
            "text": "\u{1b}test\u{7}",
            "sentences": [{
                "tokens": [{"text": "\u{1b}test\u{7}", "start": 0, "end": 6, "tag": null}],
                "constituents": [{"start": 0, "end": 1, "labels": []}]
            }]
        });
        let document: ParseDocument = serde_json::from_value(input.clone()).unwrap();
        let rendered = document.sentences()[0].to_bracketed();
        assert!(!rendered.chars().any(char::is_control));
        assert_eq!(document.text(), "\u{1b}test\u{7}");
        let mut bad_tag = input.clone();
        bad_tag["sentences"][0]["tokens"][0]["tag"] = "NN\u{1b}".into();
        assert!(serde_json::from_value::<ParseDocument>(bad_tag).is_err());
        let mut bad_label = input;
        bad_label["sentences"][0]["constituents"][0]["labels"] = serde_json::json!(["S\u{1b}"]);
        assert!(serde_json::from_value::<ParseDocument>(bad_label).is_err());
    }

    #[test]
    fn accepts_real_benepar_contract_output() {
        // Snapshot of testdata/constituency/benepar_en3.json, kept source-contained
        // so the published crate's unit tests do not require repository-only files.
        let fixture = r#"{"text": "Mira builds small robots. She smiles.", "sentences": [{"tokens": [{"text": "Mira", "start": 0, "end": 4, "tag": "NNP"}, {"text": "builds", "start": 5, "end": 11, "tag": "VBZ"}, {"text": "small", "start": 12, "end": 17, "tag": "JJ"}, {"text": "robots", "start": 18, "end": 24, "tag": "NNS"}, {"text": ".", "start": 24, "end": 25, "tag": "."}], "constituents": [{"start": 0, "end": 5, "labels": ["TOP", "S"]}, {"start": 0, "end": 1, "labels": ["NP"]}, {"start": 1, "end": 4, "labels": ["VP"]}, {"start": 1, "end": 2, "labels": []}, {"start": 2, "end": 4, "labels": ["NP"]}, {"start": 2, "end": 3, "labels": []}, {"start": 3, "end": 4, "labels": []}, {"start": 4, "end": 5, "labels": []}]}, {"tokens": [{"text": "She", "start": 26, "end": 29, "tag": "PRP"}, {"text": "smiles", "start": 30, "end": 36, "tag": "VBZ"}, {"text": ".", "start": 36, "end": 37, "tag": "."}], "constituents": [{"start": 0, "end": 3, "labels": ["TOP", "S"]}, {"start": 0, "end": 1, "labels": ["NP"]}, {"start": 1, "end": 2, "labels": ["VP"]}, {"start": 2, "end": 3, "labels": []}]}]}"#;
        let document = ParseDocument::from_json(fixture).unwrap();
        assert_eq!(document.sentences().len(), 2);
        assert_eq!(document.sentences()[1].tokens()[0].span().start.get(), 26);
        assert_eq!(
            document.sentences()[0].to_bracketed(),
            "(TOP (S (NP (NNP Mira)) (VP (VBZ builds) (NP (JJ small) (NNS robots))) (. .)))"
        );
    }
}
