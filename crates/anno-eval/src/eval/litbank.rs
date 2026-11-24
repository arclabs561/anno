//! LitBank's paired BRAT source text and contiguous text-bound annotations.
use super::coref::{CorefChain, CorefDocument, Mention};
use super::loader::AnnotatedToken;
use anno::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

#[derive(Deserialize, Serialize)]
pub(super) struct SourcePair {
    pub text: String,
    pub annotations: String,
}

impl SourcePair {
    pub fn encode(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| Error::InvalidInput(e.to_string()))
    }

    pub fn decode(content: &str) -> Result<Self> {
        serde_json::from_str(content).map_err(|e| Error::InvalidInput(format!(
            "LitBank requires a JSON object with text and annotations from the matching BRAT .txt and .ann files: {e}"
        )))
    }
}

/// Older cache files hold just the .ann file. Accept them only with their matching .txt.
pub(super) fn cached_content<'a>(
    content: &'a str,
    path: &Path,
) -> Result<std::borrow::Cow<'a, str>> {
    if content.trim_start().starts_with('{') {
        return Ok(content.into());
    }
    SourcePair::from_annotations(path, content.to_owned())?
        .encode()
        .map(Into::into)
}

impl SourcePair {
    pub fn from_annotations(path: &Path, annotations: String) -> Result<Self> {
        let source_path = path.with_extension("txt");
        let text = std::fs::read_to_string(&source_path).map_err(|e| {
            Error::InvalidInput(format!(
                "LitBank annotations at {} require the matching BRAT source at {}: {e}",
                path.display(),
                source_path.display()
            ))
        })?;
        Ok(Self { text, annotations })
    }
}

struct Annotation<'a> {
    id: &'a str,
    label: &'a str,
    start: usize,
    end: usize,
    byte_start: usize,
    byte_end: usize,
}

impl SourcePair {
    fn annotations(&self) -> Result<Vec<Annotation<'_>>> {
        let offsets = anno::offset::build_char_to_byte_map(&self.text);
        let mut records = Vec::new();
        let mut ids = BTreeSet::new();
        for line in self
            .annotations
            .lines()
            .filter(|line| line.starts_with('T'))
        {
            let malformed = || {
                Error::InvalidInput(format!(
                    "Invalid contiguous LitBank text-bound annotation: {line}"
                ))
            };
            let mut fields = line.splitn(3, '\t');
            let id = fields.next().ok_or_else(malformed)?;
            let mut spec = fields.next().ok_or_else(malformed)?.split_whitespace();
            let surface = fields.next().ok_or_else(malformed)?;
            let label = spec.next().ok_or_else(malformed)?;
            let start: usize = spec
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(malformed)?;
            let end: usize = spec
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(malformed)?;
            if spec.next().is_some() || start >= end || !ids.insert(id) {
                return Err(malformed());
            }
            let byte_start = *offsets.get(start).ok_or_else(malformed)?;
            let byte_end = *offsets.get(end).ok_or_else(malformed)?;
            if &self.text[byte_start..byte_end] != surface {
                return Err(Error::InvalidInput(format!(
                    "LitBank {id} surface differs from the source at character span {start}..{end}"
                )));
            }
            records.push(Annotation {
                id,
                label,
                start,
                end,
                byte_start,
                byte_end,
            });
        }
        if records.is_empty() {
            return Err(Error::InvalidInput(
                "LitBank contains no text-bound annotations".into(),
            ));
        }
        Ok(records)
    }

    pub fn coref(&self) -> Result<CorefDocument> {
        let records = self.annotations()?;
        // Use spans as nodes: the type and identity layers can name the same mention twice.
        let mut graph = BTreeMap::new();
        let mut labels = HashMap::new();
        let mut spans = HashMap::new();
        let mut surfaces = HashMap::new();
        for r in &records {
            let span = (r.start, r.end);
            spans.insert(r.id, span);
            surfaces.insert(span, &self.text[r.byte_start..r.byte_end]);
            graph.entry(span).or_insert_with(Vec::new);
            if r.label.rsplit_once('-').is_some_and(|(name, n)| {
                !name.is_empty() && !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())
            }) {
                if let Some(previous) = labels.insert(r.label, span) {
                    graph.entry(span).or_default().push(previous);
                    graph.entry(previous).or_default().push(span);
                }
            }
        }
        // Some BRAT exports use explicit Coref relations. LitBank's appos and cop are separate tasks.
        for line in self
            .annotations
            .lines()
            .filter(|line| line.starts_with('R'))
        {
            let mut parts = line.split_whitespace();
            if parts.nth(1) != Some("Coref") {
                continue;
            }
            let endpoints = ["Arg1:", "Arg2:"].map(|prefix| {
                parts
                    .next()?
                    .strip_prefix(prefix)
                    .and_then(|id| spans.get(id).copied())
            });
            let [Some(a), Some(b)] = endpoints else {
                return Err(Error::InvalidInput(format!(
                    "Invalid LitBank Coref endpoints: {line}"
                )));
            };
            graph.entry(a).or_default().push(b);
            graph.entry(b).or_default().push(a);
        }
        let mut chains = Vec::new();
        let mut seen = BTreeSet::new();
        for &first in graph.keys() {
            if !seen.insert(first) {
                continue;
            }
            let mut pending = vec![first];
            let mut mentions = Vec::new();
            while let Some(span) = pending.pop() {
                mentions.push(Mention::new(surfaces[&span], span.0, span.1));
                pending.extend(graph[&span].iter().copied().filter(|s| seen.insert(*s)));
            }
            mentions.sort_by_key(|m| (m.start, m.end));
            chains.push(CorefChain::with_id(mentions, chains.len() as u64));
        }
        let mut document = CorefDocument::new(&self.text, chains);
        document.includes_singletons = true;
        Ok(document)
    }

    pub fn ner_tokens(&self) -> Result<Vec<AnnotatedToken>> {
        let mut spans = Vec::new();
        for r in self.annotations()? {
            let label = r
                .label
                .strip_prefix("PROP_")
                .or_else(|| r.label.strip_prefix("NOM_"))
                .or_else(|| r.label.strip_prefix("PRON_"))
                .unwrap_or(r.label);
            if matches!(
                label,
                "PER"
                    | "LOC"
                    | "ORG"
                    | "GPE"
                    | "FAC"
                    | "VEH"
                    | "PERSON"
                    | "LOCATION"
                    | "ORGANIZATION"
            ) {
                spans.push((r.byte_start, r.byte_end, label));
            }
        }
        if spans.is_empty() {
            return Err(Error::InvalidInput(
                "LitBank contains no NER annotations".into(),
            ));
        }
        // Flat BIO cannot express nesting. Prefer the leftmost outer mention consistently.
        spans.sort_by_key(|&(start, end, label)| (start, std::cmp::Reverse(end), label));
        let mut cursor = 0;
        let mut tokens = Vec::new();
        fn outside(s: &str) -> impl Iterator<Item = AnnotatedToken> + '_ {
            s.split_whitespace().map(|w| AnnotatedToken {
                text: w.into(),
                ner_tag: "O".into(),
            })
        }
        for (start, end, label) in spans {
            if start < cursor {
                continue;
            }
            tokens.extend(outside(&self.text[cursor..start]));
            for (i, w) in self.text[start..end].split_whitespace().enumerate() {
                tokens.push(AnnotatedToken {
                    text: w.into(),
                    ner_tag: format!("{}-{label}", if i == 0 { "B" } else { "I" }),
                });
            }
            cursor = end;
        }
        tokens.extend(outside(&self.text[cursor..]));
        Ok(tokens)
    }
}
