//! Source-aligned, pre-tokenized FCoref input. The downstream use case is
//! Alexander Lyon's Aurelius language lane (arlyon/aurelius@67b7797).
use std::collections::BTreeMap;

use super::clustering::{clusters_from_groups, UnionFind};
use super::{CorefCluster, Error, FCoref, Result, SpanConverter};

/// A source mention proposed by the model, including candidates without an antecedent.
#[derive(Debug, Clone)]
pub struct FCorefMention {
    /// Verbatim slice of the supplied document.
    pub text: String,
    /// Inclusive Unicode scalar offset.
    pub character_start: usize,
    /// Exclusive Unicode scalar offset.
    pub character_end: usize,
    /// Maximum uncalibrated logit among candidates for this source span.
    pub raw_mention_score: f32,
}

/// Word-aligned mentions and clusters with at least two distinct source spans.
#[derive(Debug, Clone, Default)]
pub struct FCorefAnalysis {
    /// Unique source spans in document order, including singletons.
    pub mentions: Vec<FCorefMention>,
    /// Model clusters projected onto the source. A coarse input range may
    /// occur in more than one cluster if it covers distinct model mentions.
    pub clusters: Vec<CorefCluster>,
}

impl FCoref {
    /// Analyze text using caller-supplied word boundaries. Ranges are nonempty,
    /// ordered, disjoint UTF-8 byte intervals. Gaps may contain whitespace.
    /// The encoded document, including special tokens, must fit the configured document budget.
    /// As in FastCoref, the candidate count is the non-padding token count times
    /// `top_lambda`, rounded down. Short inputs can produce no mentions.
    /// Candidate endpoints map to the full words containing them. Candidates
    /// with the same source span share one mention score. Their antecedent links
    /// are resolved before projection so unrelated clusters stay separate.
    pub fn analyze_words(&self, text: &str, words: &[(usize, usize)]) -> Result<FCorefAnalysis> {
        let surfaces = word_surfaces(text, words)?;
        if surfaces.is_empty() {
            return Ok(FCorefAnalysis::default());
        }
        let encoding = super::encode_input(&self.tokenizer, surfaces, &self.config)?;
        let max_candidates =
            candidate_budget(encoding.get_attention_mask(), self.config.top_lambda);
        if max_candidates == 0 {
            return Ok(FCorefAnalysis::default());
        }
        let (top, antecedents) = self.score_encoding(&encoding, max_candidates)?;
        let candidates = top
            .top_k_starts
            .into_iter()
            .zip(top.top_k_ends)
            .zip(top.top_k_logits)
            .map(|((start, end), score)| (start, end, score));
        Ok(project_candidates(
            text,
            words,
            encoding.get_word_ids(),
            candidates,
            &antecedents,
        ))
    }
}

fn candidate_budget(attention_mask: &[u32], top_lambda: f32) -> usize {
    let active_tokens = attention_mask.iter().filter(|&&mask| mask != 0).count();
    (active_tokens as f32 * top_lambda) as usize
}

fn word_surfaces<'a>(text: &'a str, words: &[(usize, usize)]) -> Result<Vec<&'a str>> {
    let mut surfaces = Vec::with_capacity(words.len());
    let mut previous = 0;
    for &(start, end) in words {
        let surface = text
            .get(start..end)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| Error::Parse("expected a nonempty UTF-8 word span".into()))?;
        if !text
            .get(previous..start)
            .is_some_and(|gap| gap.chars().all(char::is_whitespace))
        {
            return Err(Error::Parse(
                "overlapping words or omitted source content".into(),
            ));
        }
        surfaces.push(surface);
        previous = end;
    }
    if !text[previous..].chars().all(char::is_whitespace) {
        return Err(Error::Parse("word spans omit source suffix".into()));
    }
    Ok(surfaces)
}

fn project_candidates(
    text: &str,
    words: &[(usize, usize)],
    word_ids: &[Option<u32>],
    candidates: impl IntoIterator<Item = (usize, usize, f32)>,
    antecedents: &[usize],
) -> FCorefAnalysis {
    let mut scores = BTreeMap::<(usize, usize), f32>::new();
    let candidate_ranges: Vec<_> = candidates
        .into_iter()
        .map(|(start, end, score)| {
            let first = word_ids.get(start)?.as_ref()?;
            let last = word_ids.get(end)?.as_ref()?;
            let range = (words.get(*first as usize)?.0, words.get(*last as usize)?.1);
            if text.get(range.0..range.1)?.trim().is_empty() {
                return None;
            }
            scores
                .entry(range)
                .and_modify(|old| *old = old.max(score))
                .or_insert(score);
            Some(range)
        })
        .collect();

    let converter = SpanConverter::new(text);
    let mut mentions = Vec::with_capacity(scores.len());
    let mut indices = BTreeMap::new();
    for ((start, end), score) in scores {
        indices.insert((start, end), mentions.len());
        let surface = &text[start..end];
        let character_start = converter.byte_to_char(start);
        let character_end = converter.byte_to_char(end);
        mentions.push(FCorefMention {
            text: surface.into(),
            character_start,
            character_end,
            raw_mention_score: score,
        });
    }

    // FastCoref clusters model candidates before aligning them to the source.
    // A coarse input span can cover candidates that refer to different people.
    let mut groups = UnionFind::new(candidate_ranges.len());
    for (i, &antecedent) in antecedents.iter().take(candidate_ranges.len()).enumerate() {
        if antecedent < candidate_ranges.len() {
            groups.union(i, antecedent);
        }
    }
    let projected = groups.into_groups().map(|group| {
        let mut members: Vec<_> = group
            .into_iter()
            .filter_map(|i| candidate_ranges[i].map(|range| indices[&range]))
            .collect();
        members.sort_unstable();
        members.dedup();
        members
    });
    let clusters = clusters_from_groups(projected, |i| {
        let m = &mentions[i];
        (m.text.as_str(), (m.character_start, m.character_end))
    });
    FCorefAnalysis { mentions, clusters }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_does_not_create_word_candidates() {
        // A single word plus two special tokens, padded to twelve positions.
        assert_eq!(
            candidate_budget(&[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.25),
            0
        );
        assert_eq!(candidate_budget(&[1; 13], 0.25), 3);
    }

    #[test]
    #[ignore = "requires exported biu-nlp/f-coref model artifacts"]
    fn word_candidates_match_pretrained_fastcoref() {
        let model = FCoref::from_path(concat!(env!("CARGO_MANIFEST_DIR"), "/fcoref_onnx")).unwrap();
        // Three encoded tokens * 0.25 yields no candidates.
        assert!(model
            .analyze_words("Alice", &[(0, 5)])
            .unwrap()
            .mentions
            .is_empty());
        // Thirteen encoded tokens yield three candidates. Rounding up adds
        // Émilie and changes the cluster count on this downstream fixture.
        let text = "Émilie reviewed the proposal. She approved it.";
        let words = [
            (0, 7),
            (8, 16),
            (17, 20),
            (21, 29),
            (29, 30),
            (31, 34),
            (35, 43),
            (44, 46),
            (46, 47),
        ];
        let result = model.analyze_words(text, &words).unwrap();
        let mentions: Vec<_> = result
            .mentions
            .iter()
            .map(|m| (m.text.as_str(), m.character_start, m.character_end))
            .collect();
        assert_eq!(
            mentions,
            [("the proposal", 16, 28), ("She", 30, 33), ("it", 43, 45)]
        );
        assert_eq!(result.clusters.len(), 1);
        assert_eq!(result.clusters[0].spans, [(16, 28), (43, 45)]);
    }

    #[test]
    #[ignore = "requires exported biu-nlp/f-coref model artifacts"]
    fn coarse_word_span_preserves_pretrained_clusters() {
        let model = FCoref::from_path(concat!(env!("CARGO_MANIFEST_DIR"), "/fcoref_onnx")).unwrap();
        let text = concat!(
            "Alice and Bob met. Alice smiled at Bob. She thanked him. He waved at her. ",
            "The stone pavement stretched for miles into the distance."
        );
        let words: Vec<_> = regex::Regex::new(r"Alice and Bob|\w+|[^\w\s]")
            .unwrap()
            .find_iter(text)
            .map(|m| (m.start(), m.end()))
            .collect();
        let result = model.analyze_words(text, &words).unwrap();
        // FastCoref clusters Alice and Bob separately before mapping their
        // first mentions to the same caller-supplied span.
        assert_eq!(result.clusters.len(), 2);
        assert_eq!(
            result.clusters[0].spans,
            [(0, 13), (19, 24), (40, 43), (69, 72)]
        );
        assert_eq!(result.clusters[1].spans, [(0, 13), (35, 38), (57, 59)]);
    }

    #[test]
    fn validates_utf8_boundaries_order_and_coverage() {
        assert_eq!(
            word_surfaces("É\u{2003}x!", &[(0, 2), (5, 6), (6, 7)]).unwrap(),
            ["É", "x", "!"]
        );
        for words in [
            vec![(1, 2)],
            vec![(0, 0)],
            vec![(0, 2), (0, 3)],
            vec![(0, 2)],
            vec![(0, 4)],
            vec![(2, 3)],
        ] {
            assert!(word_surfaces("Éx", &words).is_err());
        }
        assert!(word_surfaces(" \t", &[]).unwrap().is_empty());
    }

    #[test]
    fn projects_byte_level_prefixes_to_full_words() {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::processors::roberta::RobertaProcessing;
        use tokenizers::Tokenizer;

        let model = WordLevel::builder()
            .vocab([("<unk>".to_owned(), 99)].into_iter().collect())
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
        tokenizer.with_post_processor(Some(RobertaProcessing::default().add_prefix_space(false)));
        let text = "Alice, she";
        let words = [(0, 5), (5, 6), (7, 10)];
        let encoding = tokenizer
            .encode(word_surfaces(text, &words).unwrap(), true)
            .unwrap();
        let candidates = [(1, 1, 1.), (2, 2, 1.), (3, 3, 1.)];
        let result = project_candidates(
            text,
            &words,
            encoding.get_word_ids(),
            candidates,
            &[0, 1, 2],
        );
        assert_eq!(
            result
                .mentions
                .iter()
                .map(|m| m.text.as_str())
                .collect::<Vec<_>>(),
            ["Alice", ",", "she"]
        );
    }

    #[test]
    fn projects_word_ids_to_original_source_and_preserves_links_and_singletons() {
        let text = "Zoë\tE\u{301}milie, she";
        let words = [(0, 4), (5, 13), (13, 14), (15, 18)];
        assert_eq!(
            word_surfaces(text, &words).unwrap(),
            ["Zoë", "E\u{301}milie", ",", "she"]
        );
        let ids = [
            None,
            Some(0),
            Some(0),
            Some(1),
            Some(1),
            Some(2),
            Some(3),
            None,
        ];
        let result = project_candidates(
            text,
            &words,
            &ids,
            [
                (0, 1, 99.), // special start
                (1, 1, 2.),
                (1, 2, 3.), // same source word, distinct subtokens
                (3, 4, 1.),
                (5, 5, -2.),
                (6, 6, 0.5),
                (6, 7, 99.), // special end
            ],
            &[0, 1, 1, 2, 4, 3, 1],
        );
        assert_eq!(
            result
                .mentions
                .iter()
                .map(|m| (
                    m.text.as_str(),
                    m.character_start,
                    m.character_end,
                    m.raw_mention_score
                ))
                .collect::<Vec<_>>(),
            [
                ("Zoë", 0, 3, 3.),
                ("E\u{301}milie", 4, 11, 1.),
                (",", 11, 12, -2.),
                ("she", 13, 16, 0.5)
            ]
        );
        assert_eq!(result.clusters.len(), 1);
        assert_eq!(result.clusters[0].mentions, ["Zoë", "E\u{301}milie", "she"]);
    }

    #[test]
    fn a_shared_word_span_does_not_join_distinct_model_clusters() {
        let text = "Alice/Bob she he";
        let result = project_candidates(
            text,
            &[(0, 9), (10, 13), (14, 16)],
            &[Some(0), Some(0), Some(0), Some(1), Some(2)],
            [(0, 0, 2.), (2, 2, 7.), (3, 3, 1.), (4, 4, 1.)],
            &[0, 1, 0, 1],
        );
        assert_eq!(result.mentions.len(), 3);
        assert_eq!(result.mentions[0].raw_mention_score, 7.);
        assert_eq!(result.clusters.len(), 2);
        assert_eq!(result.clusters[0].mentions, ["Alice/Bob", "she"]);
        assert_eq!(result.clusters[1].mentions, ["Alice/Bob", "he"]);
    }

    #[test]
    fn two_linked_candidates_in_one_word_project_to_a_singleton() {
        let result = project_candidates(
            "Alice/Bob",
            &[(0, 9)],
            &[Some(0), Some(0)],
            [(0, 0, 1.), (1, 1, 4.)],
            &[0, 0],
        );
        assert_eq!(result.mentions.len(), 1);
        assert_eq!(result.mentions[0].raw_mention_score, 4.);
        assert!(result.clusters.is_empty());
    }
}
