// Adapted from SemplificaAI/gliner2-rs (Apache-2.0):
// https://github.com/SemplificaAI/gliner2-rs/blob/main/rust_component/src/processor.rs
// Original: Copyright 2026 Dario Finardi, Semplifica s.r.l.
//
// Modifications: char offsets (anno convention) instead of token offsets;
// integration with anno::Entity / anno::backends::inference traits;
// removal of Relations and Classifications schema arms (NER-only Phase 1).
// Error type translated from anyhow::Result to backend-local Error.
#![allow(missing_docs)] // implementation internals; public API is on GLiNER2Fastino in mod.rs

use crate::backends::gliner2_fastino::errors::Error;
use regex::Regex;
use std::collections::HashMap;
use tokenizers::Tokenizer;

pub const P_TOKEN: &str = "[P]";
pub const E_TOKEN: &str = "[E]";
pub const C_TOKEN: &str = "[C]";
pub const L_TOKEN: &str = "[L]";
pub const R_TOKEN: &str = "[R]";
pub const SEP_STRUCT: &str = "[SEP_STRUCT]";
pub const SEP_TEXT: &str = "[SEP_TEXT]";
/// Phase 1.5: per-label description token. Emitted as
/// `[E] <label> [DESCRIPTION] <description>` when callers use
/// `SchemaTask::EntitiesDescribed`. Mirrors upstream
/// `SemplificaAI/gliner2-rs/rust_component/src/processor.rs::DESC_TOKEN`.
pub const DESC_TOKEN: &str = "[DESCRIPTION]";

/// Integer IDs for the seven fastino special tokens, resolved at load time
/// from the tokenizer's vocabulary. Never hardcoded.
#[derive(Debug, Clone)]
#[allow(dead_code)] // several token IDs reserved for Phase 2 structure/relation prompts
pub struct SpecialTokenIds {
    pub p: u32,
    pub e: u32,
    pub c: u32,
    pub l: u32,
    pub r: u32,
    pub sep_struct: u32,
    pub sep_text: u32,
}

impl SpecialTokenIds {
    pub fn resolve(tokenizer: &Tokenizer) -> Result<Self, Error> {
        let lookup = |tok: &'static str| -> Result<u32, Error> {
            tokenizer
                .token_to_id(tok)
                .ok_or(Error::SpecialTokenMissing { token: tok })
        };
        Ok(Self {
            p: lookup(P_TOKEN)?,
            e: lookup(E_TOKEN)?,
            c: lookup(C_TOKEN)?,
            l: lookup(L_TOKEN)?,
            r: lookup(R_TOKEN)?,
            sep_struct: lookup(SEP_STRUCT)?,
            sep_text: lookup(SEP_TEXT)?,
        })
    }
}

/// Word-level splitter mirroring upstream `gliner2-rs::WhitespaceTokenSplitter`.
/// Recognizes URLs, emails, @mentions, hyphenated words, and falls through to
/// any non-whitespace single char. Returns byte offsets (NOT char offsets) —
/// the decoder converts to char offsets via `crate::offset::bytes_to_chars`.
#[derive(Clone, Debug)]
pub struct WhitespaceTokenSplitter {
    re: Regex,
}

impl WhitespaceTokenSplitter {
    pub fn new() -> Result<Self, Error> {
        let re = Regex::new(
            r"(?xi)
            (?:https?://[^\s]+|www\.[^\s]+)
            |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
            |@[a-z0-9_]+
            |\w+(?:[-_]\w+)*
            |\S
        ",
        )
        .map_err(|e| Error::Tokenizer(format!("regex: {e}")))?;
        Ok(Self { re })
    }

    /// Split into words. Returns borrowed slices into `text`.
    #[allow(dead_code)] // pipeline uses split_with_offsets; split kept for parity/tests
    pub fn split<'a>(&self, text: &'a str) -> Vec<&'a str> {
        self.re.find_iter(text).map(|m| m.as_str()).collect()
    }

    /// Split into `(word, byte_start, byte_end)` triples.
    pub fn split_with_offsets<'a>(&self, text: &'a str) -> Vec<(&'a str, usize, usize)> {
        self.re
            .find_iter(text)
            .map(|m| (m.as_str(), m.start(), m.end()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// SchemaTask, TaskMapping, ProcessedRecord, SchemaTransformer
// Ported from SemplificaAI/gliner2-rs processor.rs (Apache-2.0).
// Phase 3: Entities and Classifications arms are implemented.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum SchemaTask {
    Entities(Vec<String>),
    /// Phase 1.5: entities with per-label descriptions for accuracy boost.
    /// Each tuple is (label, description). Emits
    /// `[E] <label> [DESCRIPTION] <description>` per pair in the prompt.
    EntitiesDescribed(Vec<(String, String)>),
    /// Phase 3: classification task. (task_name, labels). Uses [L] tokens.
    Classifications(String, Vec<String>),
    /// Phase 2: structured-data extraction. (task_name, [(field_name,
    /// field_type)]). Uses [C] tokens per field. The model treats each
    /// field as an attribute that may appear 0..MAX_COUNT times in the
    /// text; the scorer's `MAX_COUNT` axis decodes to per-instance
    /// occurrences (see `super::pipeline::decode_structure`).
    Structures(String, Vec<(String, super::schema::FieldType)>),
    // TODO(Phase 2.5): port Relations arm from upstream when a workload
    // requests it. Phase 2 ships Structures only.
    #[allow(dead_code)]
    Relations(String, Vec<String>),
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // task_name, task_type reserved for Phase 2 (structure extraction).
pub struct TaskMapping {
    pub task_name: String,
    pub task_type: String,
    pub labels: Vec<String>,
    pub prompt_tok_idx: usize,
    pub field_tok_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // tasks, text_start, text_end reserved for Phase 2.
pub struct ProcessedRecord {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub tasks: Vec<TaskMapping>,
    pub text_start: usize,
    pub text_end: usize,
    pub word_to_token_maps: Vec<(usize, usize)>,
    pub word_to_char_maps: Vec<(usize, usize)>,
}

pub struct SchemaTransformer {
    tokenizer: Tokenizer,
    word_splitter: WhitespaceTokenSplitter,
}

impl SchemaTransformer {
    pub fn new(tokenizer: Tokenizer) -> Result<Self, Error> {
        Ok(Self {
            tokenizer,
            word_splitter: WhitespaceTokenSplitter::new()?,
        })
    }

    pub fn transform(
        &self,
        text: &str,
        schema_tasks: &[SchemaTask],
    ) -> Result<ProcessedRecord, Error> {
        let words_with_offsets = self.word_splitter.split_with_offsets(text);

        let mut combined_tokens: Vec<&str> = Vec::new();
        let mut task_mappings_temp = Vec::new();

        for (i, task) in schema_tasks.iter().enumerate() {
            let mut field_indices = Vec::new();
            let mut labels = Vec::new();

            match task {
                SchemaTask::Entities(entity_labels) => {
                    combined_tokens.push("(");
                    let prompt_idx = combined_tokens.len();
                    combined_tokens.push(P_TOKEN);
                    combined_tokens.push("entities");
                    combined_tokens.push("(");

                    for label in entity_labels {
                        combined_tokens.push(E_TOKEN);
                        field_indices.push(combined_tokens.len());
                        combined_tokens.push(label.as_str());
                        labels.push(label.clone());
                    }
                    combined_tokens.push(")");
                    combined_tokens.push(")");

                    task_mappings_temp.push((
                        "entities".to_string(),
                        "entities".to_string(),
                        labels,
                        prompt_idx,
                        field_indices,
                    ));
                }
                SchemaTask::EntitiesDescribed(labeled) => {
                    combined_tokens.push("(");
                    let prompt_idx = combined_tokens.len();
                    combined_tokens.push(P_TOKEN);
                    combined_tokens.push("entities");
                    combined_tokens.push("(");

                    for (label, description) in labeled {
                        combined_tokens.push(E_TOKEN);
                        field_indices.push(combined_tokens.len());
                        combined_tokens.push(label.as_str());
                        combined_tokens.push(DESC_TOKEN);
                        combined_tokens.push(description.as_str());
                        labels.push(label.clone());
                    }
                    combined_tokens.push(")");
                    combined_tokens.push(")");

                    task_mappings_temp.push((
                        "entities".to_string(),
                        "entities".to_string(),
                        labels,
                        prompt_idx,
                        field_indices,
                    ));
                }
                SchemaTask::Classifications(task_name, cls_labels) => {
                    combined_tokens.push("(");
                    let prompt_idx = combined_tokens.len();
                    combined_tokens.push(P_TOKEN);
                    combined_tokens.push(task_name.as_str());
                    combined_tokens.push("(");
                    for label in cls_labels {
                        combined_tokens.push(L_TOKEN);
                        field_indices.push(combined_tokens.len());
                        combined_tokens.push(label.as_str());
                        labels.push(label.clone());
                    }
                    combined_tokens.push(")");
                    combined_tokens.push(")");
                    task_mappings_temp.push((
                        task_name.clone(),
                        "classifications".to_string(),
                        labels,
                        prompt_idx,
                        field_indices,
                    ));
                }
                SchemaTask::Structures(task_name, fields) => {
                    combined_tokens.push("(");
                    let prompt_idx = combined_tokens.len();
                    combined_tokens.push(P_TOKEN);
                    combined_tokens.push(task_name.as_str());
                    combined_tokens.push("(");
                    for (field_name, _ftype) in fields {
                        combined_tokens.push(C_TOKEN);
                        field_indices.push(combined_tokens.len());
                        combined_tokens.push(field_name.as_str());
                        labels.push(field_name.clone());
                    }
                    combined_tokens.push(")");
                    combined_tokens.push(")");

                    task_mappings_temp.push((
                        task_name.clone(),
                        "structures".to_string(),
                        labels,
                        prompt_idx,
                        field_indices,
                    ));
                }
                SchemaTask::Relations(..) => {}
            }

            if i < schema_tasks.len() - 1 {
                combined_tokens.push(SEP_STRUCT);
            }
        }

        combined_tokens.push(SEP_TEXT);
        let text_start_idx = combined_tokens.len();

        let mut word_to_char_maps = Vec::new();
        for (w, start_char, end_char) in &words_with_offsets {
            combined_tokens.push(*w);
            word_to_char_maps.push((*start_char, *end_char));
        }
        let text_end_idx = combined_tokens.len();

        let mut final_input_ids = Vec::new();
        let mut final_attention_mask = Vec::new();
        let mut word_to_token_maps = Vec::new();

        let mut combined_to_final_map: HashMap<usize, usize> = HashMap::new();

        // Encode [CLS] at start.
        let cls_id = self
            .tokenizer
            .encode("[CLS]", false)
            .map_err(|e| Error::Tokenizer(format!("encode [CLS]: {e}")))?
            .get_ids()[0] as i64;
        final_input_ids.push(cls_id);
        final_attention_mask.push(1);
        let mut current_subword_idx = 1;

        for (i, token) in combined_tokens.iter().enumerate() {
            combined_to_final_map.insert(i, current_subword_idx);

            let encoding = self
                .tokenizer
                .encode(*token, false)
                .map_err(|e| Error::Tokenizer(format!("encode '{token}': {e}")))?;

            let ids = encoding.get_ids();
            let start_sub = current_subword_idx;
            let end_sub = current_subword_idx + ids.len();

            for &id in ids {
                final_input_ids.push(id as i64);
                final_attention_mask.push(1);
                current_subword_idx += 1;
            }

            if i >= text_start_idx && i < text_end_idx {
                word_to_token_maps.push((start_sub, end_sub));
            }
        }

        let sep_id = self
            .tokenizer
            .encode("[SEP]", false)
            .map_err(|e| Error::Tokenizer(format!("encode [SEP]: {e}")))?
            .get_ids()[0] as i64;
        final_input_ids.push(sep_id);
        final_attention_mask.push(1);

        let text_real_start = word_to_token_maps.first().map(|v| v.0).unwrap_or(0);
        let text_real_end = word_to_token_maps.last().map(|v| v.1).unwrap_or(0);

        let mut tasks = Vec::new();
        for (task_name, task_type, labels, prompt_idx, field_indices) in task_mappings_temp {
            let real_prompt_idx = *combined_to_final_map.get(&prompt_idx).unwrap();
            let real_field_indices: Vec<usize> = field_indices
                .iter()
                .map(|idx| *combined_to_final_map.get(idx).unwrap())
                .collect();

            tasks.push(TaskMapping {
                task_name,
                task_type,
                labels,
                prompt_tok_idx: real_prompt_idx,
                field_tok_indices: real_field_indices,
            });
        }

        Ok(ProcessedRecord {
            input_ids: final_input_ids,
            attention_mask: final_attention_mask,
            tasks,
            text_start: text_real_start,
            text_end: text_real_end,
            word_to_token_maps,
            word_to_char_maps,
        })
    }
}

#[cfg(test)]
mod transformer_tests {
    use super::*;
    use tokenizers::Tokenizer;

    fn stub() -> Tokenizer {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/gliner2_fastino/stub_tokenizer.json");
        Tokenizer::from_file(path).unwrap()
    }

    #[test]
    fn entities_arm_assembles_expected_prompt_shape() {
        let tok = stub();
        let xfm = SchemaTransformer::new(tok).expect("transformer build");

        // Single Entities task, two labels, simple text.
        let labels: Vec<String> = vec!["person".into(), "organization".into()];
        let task = SchemaTask::Entities(labels);
        let rec = xfm.transform("Acme Corp in Paris .", &[task]).unwrap();

        let ids = &rec.input_ids;

        // Stub tokenizer ids: [CLS]=17, [P]=2, [E]=3, [SEP_TEXT]=8, [SEP]=18,
        //                     person=9, organization=10, Acme=12, Corp=13,
        //                     Paris=14, in=15, .=16, (=19, )=20, entities=21
        // Structural invariants (independent of exact subword splits):
        //   1. Begins with [CLS] (id 17)
        //   2. Contains [P] (id 2) somewhere
        //   3. Contains exactly 2 [E] markers (id 3)
        //   4. Contains [SEP_TEXT] (id 8) before the text tokens
        //   5. Ends with [SEP] (id 18)
        //   6. The labels (ids 9, 10) appear in the prompt area (BEFORE [SEP_TEXT])

        assert_eq!(ids[0], 17, "prompt must start with [CLS], got ids={ids:?}");
        assert_eq!(
            *ids.last().unwrap(),
            18,
            "prompt must end with [SEP], got ids={ids:?}"
        );
        assert!(ids.contains(&2), "missing [P], got {ids:?}");
        let e_count = ids.iter().filter(|&&i| i == 3).count();
        assert_eq!(
            e_count, 2,
            "expected 2 [E] markers, got {e_count} in {ids:?}"
        );
        let sep_pos = ids
            .iter()
            .position(|&i| i == 8)
            .expect("missing [SEP_TEXT]");
        // Acme (12) appears AFTER [SEP_TEXT]
        assert!(
            ids[sep_pos + 1..].contains(&12),
            "Acme not after SEP_TEXT, ids={ids:?}"
        );
        // Labels person (9), organization (10) appear BEFORE [SEP_TEXT]
        assert!(
            ids[..sep_pos].contains(&9),
            "person label not before SEP_TEXT"
        );
        assert!(
            ids[..sep_pos].contains(&10),
            "organization label not before SEP_TEXT"
        );

        // attention_mask matches input_ids length and is all-ones
        assert_eq!(rec.attention_mask.len(), rec.input_ids.len());
        assert!(rec.attention_mask.iter().all(|&m| m == 1));

        // word_to_char_maps records ["Acme", "Corp", "in", "Paris", "."] = 5 words.
        // The splitter regex treats "." as a single non-word match, so word count = 5.
        assert_eq!(rec.word_to_char_maps.len(), 5);
        // First word "Acme" starts at byte 0, ends at byte 4
        assert_eq!(rec.word_to_char_maps[0], (0, 4));
    }

    #[test]
    fn entities_described_arm_emits_desc_tokens() {
        let tok = stub();
        let xfm = SchemaTransformer::new(tok).expect("transformer build");

        let labels: Vec<(String, String)> = vec![
            ("person".into(), "a human being".into()),
            ("organization".into(), "a company or institution".into()),
        ];
        let task = SchemaTask::EntitiesDescribed(labels);
        let rec = xfm.transform("Acme Corp in Paris .", &[task]).unwrap();
        let ids = &rec.input_ids;

        // Should contain 2 [E] markers (one per label) AND 2 [DESCRIPTION] markers.
        let e_count = ids.iter().filter(|&&i| i == 3).count();
        let desc_count = ids.iter().filter(|&&i| i == 22).count();
        assert_eq!(
            e_count, 2,
            "expected 2 [E] markers, got {e_count} in {ids:?}"
        );
        assert_eq!(
            desc_count, 2,
            "expected 2 [DESCRIPTION] markers, got {desc_count} in {ids:?}"
        );
        // [SEP_TEXT] still present (id 8); text words after it.
        assert!(ids.contains(&8), "missing [SEP_TEXT] in {ids:?}");
    }

    #[test]
    fn empty_labels_still_returns_well_formed_record() {
        let tok = stub();
        let xfm = SchemaTransformer::new(tok).unwrap();
        let task = SchemaTask::Entities(vec![]);
        let rec = xfm.transform("Acme Corp", &[task]).unwrap();
        let ids = &rec.input_ids;

        // No [E] markers when labels is empty
        let e_count = ids.iter().filter(|&&i| i == 3).count();
        assert_eq!(e_count, 0);
        // Still wrapped in [CLS] / [SEP]
        assert_eq!(ids[0], 17);
        assert_eq!(*ids.last().unwrap(), 18);
        // [SEP_TEXT] still present
        assert!(ids.contains(&8));
    }

    #[test]
    fn structures_arm_emits_c_tokens_per_field() {
        use crate::backends::gliner2_fastino::schema::FieldType;
        let tok = stub();
        let xfm = SchemaTransformer::new(tok).expect("transformer build");

        let fields: Vec<(String, FieldType)> = vec![
            ("name".into(), FieldType::String),
            ("price".into(), FieldType::String),
        ];
        let task = SchemaTask::Structures("product".into(), fields);
        let rec = xfm.transform("Acme Corp Paris .", &[task]).unwrap();
        let ids = &rec.input_ids;

        // [C] = id 4. Two fields → exactly 2 [C] markers.
        let c_count = ids.iter().filter(|&&i| i == 4).count();
        assert_eq!(
            c_count, 2,
            "expected 2 [C] markers, got {c_count} in {ids:?}"
        );

        // [P] = id 2 — exactly one (single task).
        let p_count = ids.iter().filter(|&&i| i == 2).count();
        assert_eq!(p_count, 1, "expected 1 [P] marker, got {p_count}");

        // [SEP_TEXT] = id 8 — present.
        assert!(ids.contains(&8), "missing [SEP_TEXT]");

        // No [E] (entity, id 3) and no [L] (label, id 5) — this is a structure prompt.
        assert!(
            !ids.contains(&3),
            "unexpected [E] in structures prompt: {ids:?}"
        );
        assert!(
            !ids.contains(&5),
            "unexpected [L] in structures prompt: {ids:?}"
        );
    }

    #[test]
    fn structures_task_mapping_records_field_count() {
        use crate::backends::gliner2_fastino::schema::FieldType;
        let tok = stub();
        let xfm = SchemaTransformer::new(tok).expect("transformer build");
        let fields: Vec<(String, FieldType)> = vec![
            ("name".into(), FieldType::String),
            ("price".into(), FieldType::String),
            ("vendor".into(), FieldType::String),
        ];
        let task = SchemaTask::Structures("invoice".into(), fields);
        let rec = xfm
            .transform("Acme Corp paid 100 dollars to Globex.", &[task])
            .unwrap();

        // One task mapping, 3 labels, task_type = "structures".
        assert_eq!(rec.tasks.len(), 1);
        let t = &rec.tasks[0];
        assert_eq!(t.task_name, "invoice");
        assert_eq!(t.task_type, "structures");
        assert_eq!(t.labels, vec!["name", "price", "vendor"]);
        assert_eq!(t.field_tok_indices.len(), 3);
    }

    #[test]
    fn mixed_entities_and_structures_separated_by_sep_struct() {
        use crate::backends::gliner2_fastino::schema::FieldType;
        let tok = stub();
        let xfm = SchemaTransformer::new(tok).expect("transformer build");

        let entities_task = SchemaTask::Entities(vec!["person".into()]);
        let struct_task =
            SchemaTask::Structures("invoice".into(), vec![("vendor".into(), FieldType::String)]);
        let rec = xfm
            .transform("Acme Corp paid Globex.", &[entities_task, struct_task])
            .unwrap();
        let ids = &rec.input_ids;

        // [SEP_STRUCT] = id 7, must appear between the two tasks.
        let sep_struct_pos = ids.iter().position(|&i| i == 7);
        assert!(
            sep_struct_pos.is_some(),
            "expected [SEP_STRUCT] between tasks, got {ids:?}"
        );

        // [E] = id 3 from Entities task — exactly one.
        let e_count = ids.iter().filter(|&&i| i == 3).count();
        assert_eq!(e_count, 1, "expected 1 [E] marker (one entity label)");

        // [C] = id 4 from Structures task — exactly one.
        let c_count = ids.iter().filter(|&&i| i == 4).count();
        assert_eq!(c_count, 1, "expected 1 [C] marker (one field)");

        // Two task mappings recorded.
        assert_eq!(rec.tasks.len(), 2);
        assert_eq!(rec.tasks[0].task_type, "entities");
        assert_eq!(rec.tasks[1].task_type, "structures");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stub_path() -> std::path::PathBuf {
        // CARGO_MANIFEST_DIR points at crates/anno/. The fixture lives at the
        // workspace root's testdata/. Walk up two levels.
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/gliner2_fastino/stub_tokenizer.json")
    }

    fn stub_tokenizer() -> Tokenizer {
        Tokenizer::from_file(stub_path()).expect("stub fixture missing or invalid")
    }

    #[test]
    fn resolve_special_tokens_from_stub_fixture() {
        let tok = stub_tokenizer();
        let ids = SpecialTokenIds::resolve(&tok).unwrap();
        assert_eq!(ids.p, 2);
        assert_eq!(ids.e, 3);
        assert_eq!(ids.c, 4);
        assert_eq!(ids.l, 5);
        assert_eq!(ids.r, 6);
        assert_eq!(ids.sep_struct, 7);
        assert_eq!(ids.sep_text, 8);
    }

    #[test]
    fn missing_special_token_returns_typed_error() {
        // Build a tokenizer.json missing [SEP_TEXT]
        let mut content = std::fs::read_to_string(stub_path()).unwrap();
        content = content.replace("\"[SEP_TEXT]\"", "\"[NOT_THE_TOKEN]\"");
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &content).unwrap();
        let tok = Tokenizer::from_file(tmp.path()).unwrap();

        let err = SpecialTokenIds::resolve(&tok).unwrap_err();
        match err {
            Error::SpecialTokenMissing { token } => assert_eq!(token, SEP_TEXT),
            other => panic!("expected SpecialTokenMissing, got {other:?}"),
        }
    }

    #[test]
    fn whitespace_splitter_basic() {
        let s = WhitespaceTokenSplitter::new().expect("regex compile");
        let words: Vec<&str> = s.split("Acme Corp signed in Paris.");
        assert_eq!(words, vec!["Acme", "Corp", "signed", "in", "Paris", "."]);
    }

    #[test]
    fn whitespace_splitter_offsets_are_byte_offsets() {
        let s = WhitespaceTokenSplitter::new().unwrap();
        let pairs = s.split_with_offsets("ab cd");
        assert_eq!(pairs, vec![("ab", 0, 2), ("cd", 3, 5)]);
    }

    #[test]
    fn whitespace_splitter_unicode_offsets() {
        let s = WhitespaceTokenSplitter::new().unwrap();
        let text = "田中 Paris";
        let pairs = s.split_with_offsets(text);
        // "田中" is 6 bytes; " " is 1 byte; "Paris" starts at byte 7.
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "田中");
        assert_eq!(pairs[1].0, "Paris");
        assert_eq!(pairs[1].1, 7);
    }
}
