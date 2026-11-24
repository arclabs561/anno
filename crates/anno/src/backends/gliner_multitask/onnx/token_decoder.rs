//! GLiNER token-level output is shaped [start/end/inside, 1, words, classes].
//! This implements the public flat, single-label predict_entities defaults of GLiNER v0.2.16.
//! <https://github.com/urchade/GLiNER/blob/a865555af53920321c4593f7c091dfc9059e9028/gliner/decoding/decoder.py#L61-L94>

use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap};

#[derive(Debug, PartialEq)]
pub(super) struct WordSpan {
    pub start: usize,
    pub end: usize, // inclusive
    pub class: usize,
    pub score: f32,
}

// All candidate scores are finite probabilities. Match the reference's stable
// torch.where enumeration order (start, class, end) on equal confidence.
impl Eq for WordSpan {}
impl Ord for WordSpan {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.start.cmp(&self.start))
            .then_with(|| other.class.cmp(&self.class))
            .then_with(|| other.end.cmp(&self.end))
    }
}
impl PartialOrd for WordSpan {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct Channels<'a> {
    start: &'a [f32],
    end: &'a [f32],
    inside: &'a [f32],
    stride: usize,
    classes: usize,
    threshold: f32,
}

impl Channels<'_> {
    /// Best candidate from this start in the remaining free interval.
    fn best(&self, start: usize, limit: usize) -> Option<WordSpan> {
        let mut best = None;
        for class in 0..self.classes {
            if self.start[start * self.stride + class] > self.threshold {
                // The checkpoint scores spans with a float32 inside-probability mean.
                let mut sum = 0.0_f32;
                for end in start..limit {
                    let i = end * self.stride + class;
                    let inside = self.inside[i];
                    if inside.is_nan() || inside < self.threshold {
                        break;
                    }
                    sum += inside;
                    if self.end[i] > self.threshold {
                        let candidate = WordSpan {
                            start,
                            end,
                            class,
                            score: sum / (end - start + 1) as f32,
                        };
                        if best.as_ref().is_none_or(|prior| candidate > *prior) {
                            best = Some(candidate);
                        }
                    }
                }
            }
        }
        best
    }
}

/// `words` and `classes` are the tensor dimensions, including any padding.
pub(super) fn decode(
    logits: &[f32],
    words: usize,
    classes: usize,
    input_words: usize,
    input_classes: usize,
    threshold: f32,
) -> Vec<WordSpan> {
    let plane_size = words * classes;
    let probabilities: Vec<_> = logits.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    let (starts, tail) = probabilities.split_at(plane_size);
    let (ends, inside) = tail.split_at(plane_size);
    let channels = Channels {
        start: starts,
        end: ends,
        inside,
        stride: classes,
        classes: classes.min(input_classes),
        threshold,
    };

    // The best span for each start suffices to find the global best. When a
    // higher-scoring accepted span blocks an end, retry that start up to the
    // first occupied word. Every replacement ranks below the popped candidate.
    // This bounds candidate storage by the word count even at low thresholds.
    let n = words.min(input_words);
    let mut queue: BinaryHeap<_> = (0..n).filter_map(|start| channels.best(start, n)).collect();
    let mut occupied = BTreeSet::new();
    let mut spans = Vec::new();
    while let Some(span) = queue.pop() {
        if let Some(&blocker) = occupied.range(span.start..=span.end).next() {
            if blocker > span.start {
                queue.extend(channels.best(span.start, blocker));
            }
        } else {
            occupied.extend(span.start..=span.end);
            spans.push(span);
        }
    }
    spans.sort_unstable_by_key(|span| span.start);
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_channels_require_both_endpoints_and_every_inside_word() {
        // Start at 0 with end at 2. A missing end at 3 or an inside gap at 4
        // invalidates the other candidates. The interior at exactly 0.5 is allowed.
        let logits = [
            4., -4., -4., 4., -4., // start
            -4., -4., 4., -4., 4., // end
            2., 0., 2., 2., -2., // inside
        ];
        let spans = decode(&logits, 5, 1, 5, 1, 0.5);
        assert_eq!(spans.len(), 1);
        assert_eq!((spans[0].start, spans[0].end), (0, 2));
        assert!((spans[0].score - 0.753_864_7).abs() < 1e-6);
    }

    #[test]
    fn endpoint_at_threshold_is_rejected() {
        assert!(decode(&[0., 1., 1.], 1, 1, 1, 1, 0.5).is_empty());
        assert!(decode(&[1., 0., 1.], 1, 1, 1, 1, 0.5).is_empty());
    }

    #[test]
    fn flat_conflicts_use_mean_inside_then_stable_start_class_end_order() {
        // Class 0 proposes [0,1], class 1 proposes [1,1] and [2,2].
        // The stronger latter class wins the overlap, then touching word
        // intervals [1,1] and [2,2] are both emitted in source order.
        let logits = [
            2., -2., -2., 2., -2., 2., // start
            -2., -2., 2., 2., -2., 2., // end
            1., -2., 1., 3., -2., 3., // inside
        ];
        let spans = decode(&logits, 3, 2, 3, 2, 0.5);
        let indices: Vec<_> = spans.iter().map(|s| (s.start, s.end, s.class)).collect();
        assert_eq!(indices, [(1, 1, 1), (2, 2, 1)]);
        // At equal confidence, class 0 wins identical spans, and [0,0]
        // wins against [0,1] (the first same-class eligible end).
        let ties = decode(&[2.; 12], 2, 2, 2, 2, 0.5);
        let indices: Vec<_> = ties.iter().map(|s| (s.start, s.end, s.class)).collect();
        assert_eq!(indices, [(0, 0, 0), (1, 1, 0)]);
    }

    #[test]
    fn float32_inside_mean_selects_the_upstream_three_word_span() {
        // GLiNER v0.2.16 on float32 logits ranks the three-word mean one ULP
        // above each singleton. Widening the sum turns that difference into a tie.
        let spans = decode(&[2.; 9], 3, 1, 3, 1, 0.5);
        assert_eq!(spans.len(), 1);
        assert_eq!((spans[0].start, spans[0].end, spans[0].class), (0, 2, 0));
        assert!((spans[0].score - 0.880_797_1).abs() < 1e-6);
    }

    #[test]
    fn tensor_strides_include_padded_words_and_classes() {
        let mut logits = [-3.; 18]; // 3 planes x 3 words x 2 classes
        for offset in [0, 6, 12] {
            logits[offset + 2] = 3.; // second word, class 0
            logits[offset + 1] = 3.; // padded class
            logits[offset + 4] = 3.; // padded word
        }
        let spans = decode(&logits, 3, 2, 2, 1, 0.5);
        assert_eq!(spans.len(), 1);
        assert_eq!((spans[0].start, spans[0].end, spans[0].class), (1, 1, 0));
    }

    #[test]
    fn bounded_greedy_matches_exhaustive_flat_reference_on_synthetic_logits() {
        // Enumerate every reference candidate before sorting. Tiny dense cases
        // exercise ties, crossing/nested spans, retries and multiple labels.
        let (words, labels) = (7, 3);
        let mut seed: u32 = 123456;
        for _ in 0..500 {
            let logits: Vec<_> = (0..3 * words * labels)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 29) as i32 - 2) as f32
                })
                .collect();
            let stride = words * labels;
            let mut reference = vec![];
            for start in 0..words {
                for class in 0..labels {
                    for end in start..words {
                        // For threshold 0.5, logits > 0 select endpoints
                        // and logits >= 0 cover the whole interval.
                        if logits[start * labels + class] <= 0.
                            || logits[stride + end * labels + class] <= 0.
                        {
                            continue;
                        }
                        let inside = (start..=end).map(|i| logits[2 * stride + i * labels + class]);
                        if inside.clone().any(|x| x < 0.) {
                            continue;
                        }
                        let sum: f32 = inside.map(|x| 1.0 / (1.0 + (-x).exp())).sum();
                        reference.push(WordSpan {
                            start,
                            end,
                            class,
                            score: sum / (end - start + 1) as f32,
                        });
                    }
                }
            }
            reference.sort_by(|a, b| b.score.total_cmp(&a.score));
            let mut flat: Vec<WordSpan> = vec![];
            for span in reference {
                if flat
                    .iter()
                    .all(|s| s.end < span.start || span.end < s.start)
                {
                    flat.push(span);
                }
            }
            flat.sort_by_key(|s| s.start);
            assert_eq!(decode(&logits, words, labels, words, labels, 0.5), flat);
        }
    }
}
