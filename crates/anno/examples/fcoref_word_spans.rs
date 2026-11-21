//! Preserve the original document when a caller supplies its own word boundaries.
//!
//! Pass a local FCoref export as the sole argument. The printed offsets are
//! half-open. A mention score is an uncalibrated model logit.
use anno::{backends::coref::FCoref, SpanConverter};
use regex::Regex;
use std::error::Error;

const TEXT: &str = "Émilie\nDurand reviewed the proposal. Émilie Durand said she approved it.";

fn words(text: &str) -> Result<Vec<(usize, usize)>, regex::Error> {
    // A caller with a tokenizer can pass its ranges directly. Include punctuation.
    Ok(Regex::new(r"\w+|[^\w\s]")?
        .find_iter(text)
        .map(|word| (word.start(), word.end()))
        .collect())
}

fn main() -> Result<(), Box<dyn Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: fcoref_word_spans MODEL_DIR")?;
    let model = FCoref::from_path(&model_dir)?;
    let analysis = model.analyze_words(TEXT, &words(TEXT)?)?;
    let offsets = SpanConverter::new(TEXT);
    println!("chars\tbytes\traw logit\tmention");
    for mention in analysis.mentions {
        let start = offsets.char_to_byte(mention.character_start);
        let end = offsets.char_to_byte(mention.character_end);
        assert_eq!(&TEXT[start..end], mention.text);
        println!(
            "{}..{}\t{}..{}\t{:.2}\t{:?}",
            mention.character_start,
            mention.character_end,
            start,
            end,
            mention.raw_mention_score,
            mention.text
        );
    }
    for (i, cluster) in analysis.clusters.iter().enumerate() {
        println!("cluster {} (char spans): {:?}", i + 1, cluster.spans);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_ranges_survive_combining_marks_emoji_and_whitespace() {
        let text = "🦀 e\u{301}\tZoë\nZoë.";
        let ranges = words(text).unwrap();
        assert_eq!(ranges, [(0, 4), (5, 8), (9, 13), (14, 18), (18, 19)]);
        let offsets = SpanConverter::new(text);
        let chars: Vec<_> = ranges
            .iter()
            .map(|&(start, end)| (offsets.byte_to_char(start), offsets.byte_to_char(end)))
            .collect();
        assert_eq!(chars, [(0, 1), (2, 4), (5, 8), (9, 12), (12, 13)]);
    }

    #[test]
    #[ignore = "requires a local FP32 biu-nlp/f-coref export in FCOREF_MODEL_DIR"]
    fn agrees_with_fastcoref_on_original_source() -> Result<(), Box<dyn Error>> {
        let model = FCoref::from_path(&std::env::var("FCOREF_MODEL_DIR")?)?;
        let analysis = model.analyze_words(TEXT, &words(TEXT)?)?;
        // PyTorch reference at ae224139, checkpoint e91dfff1, CPU FP32.
        let reference = [
            (0, 13, 0, 14, "Émilie\nDurand", -2.97135),
            (23, 35, 24, 36, "the proposal", -1.56330),
            (37, 50, 38, 52, "Émilie Durand", 2.34402),
            (56, 59, 58, 61, "she", -1.60377),
            (69, 71, 71, 73, "it", 0.49459),
        ];
        assert_eq!(analysis.mentions.len(), reference.len());
        let converter = SpanConverter::new(TEXT);
        for (mention, (start, end, byte_start, byte_end, surface, score)) in
            analysis.mentions.iter().zip(reference)
        {
            assert_eq!(
                (mention.character_start, mention.character_end),
                (start, end)
            );
            assert_eq!(
                (converter.char_to_byte(start), converter.char_to_byte(end)),
                (byte_start, byte_end)
            );
            assert_eq!(&TEXT[byte_start..byte_end], surface);
            assert_eq!(mention.text, surface);
            assert!(
                (mention.raw_mention_score - score).abs() < 0.01,
                "{mention:?}"
            );
        }
        let mut clusters: Vec<_> = analysis.clusters.into_iter().map(|c| c.spans).collect();
        for members in &mut clusters {
            members.sort_unstable();
        }
        clusters.sort_unstable();
        assert_eq!(
            clusters,
            [vec![(0, 13), (37, 50), (56, 59)], vec![(23, 35), (69, 71)]]
        );
        Ok(())
    }
}
