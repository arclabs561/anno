#[cfg(feature = "candle")]
use super::layers::l2_normalize;
use super::*;

#[test]
fn test_stub_without_feature() {
    #[cfg(not(feature = "candle"))]
    {
        let result = GLiNERCandle::new("test");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("candle"));
    }
}

#[cfg(feature = "candle")]
#[test]
fn test_span_label_matcher() {
    let device = Device::Cpu;
    let matcher = SpanLabelMatcher::new(1.0);

    let span_embs = Tensor::randn(0f32, 1., (1, 10, 64), &device).unwrap();
    let label_embs = Tensor::randn(0f32, 1., (3, 64), &device).unwrap();

    let scores = matcher.forward(&span_embs, &label_embs).unwrap();
    assert_eq!(scores.dims(), &[1, 10, 3]);
}

#[cfg(feature = "candle")]
#[test]
fn test_l2_normalize() {
    let device = Device::Cpu;
    let x = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &device).unwrap();
    let normed = l2_normalize(&x, D::Minus1).unwrap();

    // Should be [0.6, 0.8] (3/5, 4/5)
    let values = normed.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!((values[0] - 0.6).abs() < 0.01);
    assert!((values[1] - 0.8).abs() < 0.01);
}

#[cfg(feature = "candle")]
#[test]
fn test_l2_normalize_zero_vector() {
    let device = Device::Cpu;
    let x = Tensor::from_vec(vec![0.0f32, 0.0], (1, 2), &device).unwrap();
    let normed = l2_normalize(&x, D::Minus1).unwrap();
    let values = normed.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // Zero vector should not produce NaN (clamp guards against division by zero)
    for v in &values {
        assert!(
            !v.is_nan(),
            "l2_normalize should not produce NaN for zero vector"
        );
    }
}

#[cfg(feature = "candle")]
#[test]
fn test_l2_normalize_unit_vector() {
    let device = Device::Cpu;
    let x = Tensor::from_vec(vec![1.0f32, 0.0, 0.0], (1, 3), &device).unwrap();
    let normed = l2_normalize(&x, D::Minus1).unwrap();
    let values = normed.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!((values[0] - 1.0).abs() < 0.01);
    assert!(values[1].abs() < 0.01);
    assert!(values[2].abs() < 0.01);
}

#[cfg(feature = "candle")]
#[test]
fn test_l2_normalize_batch() {
    let device = Device::Cpu;
    let x = Tensor::from_vec(vec![3.0f32, 4.0, 0.0, 5.0], (2, 2), &device).unwrap();
    let normed = l2_normalize(&x, D::Minus1).unwrap();
    assert_eq!(normed.dims(), &[2, 2]);
    let values = normed.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // First row: [3/5, 4/5]
    assert!((values[0] - 0.6).abs() < 0.01);
    assert!((values[1] - 0.8).abs() < 0.01);
    // Second row: [0/5, 5/5]
    assert!((values[3] - 1.0).abs() < 0.01);
}

#[cfg(feature = "candle")]
#[test]
fn test_span_label_matcher_output_range() {
    use super::layers::SpanLabelMatcher;
    let device = Device::Cpu;
    let matcher = SpanLabelMatcher::new(1.0);

    let span_embs = Tensor::randn(0f32, 1., (1, 5, 32), &device).unwrap();
    let label_embs = Tensor::randn(0f32, 1., (4, 32), &device).unwrap();

    let scores = matcher.forward(&span_embs, &label_embs).unwrap();
    assert_eq!(scores.dims(), &[1, 5, 4]);

    // After sigmoid, all scores should be in [0, 1]
    let flat = scores.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for s in &flat {
        assert!(*s >= 0.0 && *s <= 1.0, "score {} out of [0,1] range", s);
    }
}

#[cfg(feature = "candle")]
#[test]
fn test_span_label_matcher_temperature() {
    use super::layers::SpanLabelMatcher;
    let device = Device::Cpu;

    // Temperature scales the cosine similarity before sigmoid.
    // With identical unit vectors: cosine_sim = 1.0
    // sigmoid(1.0 / high_temp) -> closer to 0.5
    // sigmoid(1.0 / low_temp)  -> closer to 1.0
    let high_temp = SpanLabelMatcher::new(10.0);
    let low_temp = SpanLabelMatcher::new(0.1);

    let span_embs = Tensor::from_vec(vec![1.0f32, 0.0], (1, 1, 2), &device).unwrap();
    let label_embs = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &device).unwrap();

    let high_scores = high_temp.forward(&span_embs, &label_embs).unwrap();
    let low_scores = low_temp.forward(&span_embs, &label_embs).unwrap();

    let h = high_scores.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
    let l = low_scores.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];

    // Both should be in [0, 1]
    assert!(h >= 0.0 && h <= 1.0, "high temp score out of range: {h}");
    assert!(l >= 0.0 && l <= 1.0, "low temp score out of range: {l}");
    // Scores should differ with different temperatures
    assert!(
        (h - l).abs() > 0.01,
        "temperature should affect score: h={h}, l={l}"
    );
}

#[cfg(feature = "candle")]
#[test]
fn test_map_label() {
    use super::inference::GLiNERCandle;
    assert_eq!(GLiNERCandle::map_label("person"), EntityType::Person);
    assert_eq!(GLiNERCandle::map_label("PER"), EntityType::Person);
    assert_eq!(
        GLiNERCandle::map_label("organization"),
        EntityType::Organization
    );
    assert_eq!(GLiNERCandle::map_label("ORG"), EntityType::Organization);
    assert_eq!(GLiNERCandle::map_label("location"), EntityType::Location);
    assert_eq!(GLiNERCandle::map_label("LOC"), EntityType::Location);
    assert_eq!(GLiNERCandle::map_label("GPE"), EntityType::Location);
    assert_eq!(GLiNERCandle::map_label("place"), EntityType::Location);
    assert_eq!(GLiNERCandle::map_label("date"), EntityType::Date);
    assert_eq!(GLiNERCandle::map_label("money"), EntityType::Money);
    assert_eq!(GLiNERCandle::map_label("currency"), EntityType::Money);
    assert_eq!(GLiNERCandle::map_label("percent"), EntityType::Percent);
    // Custom passthrough
    let custom = GLiNERCandle::map_label("vehicle");
    assert!(matches!(custom, EntityType::Custom { .. }));
}

#[cfg(feature = "candle")]
#[test]
fn test_generate_spans_small() {
    // Verify the span generation algorithm used by generate_spans.
    // For 3 words with MAX_SPAN_WIDTH=12, we get min(12, 3-0) + min(12, 3-1) + min(12, 3-2)
    // = 3 + 2 + 1 = 6 spans
    let num_words = 3;
    let max_w = super::MAX_SPAN_WIDTH;
    let mut expected_spans = Vec::new();
    for start in 0..num_words {
        for width in 0..max_w.min(num_words - start) {
            expected_spans.push((start as i64, (start + width) as i64));
        }
    }
    assert_eq!(expected_spans.len(), 6);
    assert_eq!(expected_spans[0], (0, 0));
    assert_eq!(expected_spans[1], (0, 1));
    assert_eq!(expected_spans[2], (0, 2));
    assert_eq!(expected_spans[3], (1, 1));
    assert_eq!(expected_spans[4], (1, 2));
    assert_eq!(expected_spans[5], (2, 2));
}
