/// Return a maximum-cardinality assignment from predictions to gold entries.
///
/// The augmenting-path algorithm is small and sufficient for document-level
/// entity lists. It makes match counts independent of the input ordering.
pub(super) fn maximum_cardinality_matching<F>(
    prediction_count: usize,
    gold_count: usize,
    compatible: F,
) -> Vec<Option<usize>>
where
    F: Fn(usize, usize) -> bool,
{
    let candidates: Vec<Vec<usize>> = (0..prediction_count)
        .map(|p_idx| {
            (0..gold_count)
                .filter(|&g_idx| compatible(p_idx, g_idx))
                .collect()
        })
        .collect();
    let mut gold_to_prediction = vec![None; gold_count];

    for p_idx in 0..prediction_count {
        let mut visited_gold = vec![false; gold_count];
        augment_matching(
            p_idx,
            &candidates,
            &mut gold_to_prediction,
            &mut visited_gold,
        );
    }

    let mut prediction_to_gold = vec![None; prediction_count];
    for (g_idx, p_idx) in gold_to_prediction.into_iter().enumerate() {
        if let Some(p_idx) = p_idx {
            prediction_to_gold[p_idx] = Some(g_idx);
        }
    }
    prediction_to_gold
}

fn augment_matching(
    p_idx: usize,
    candidates: &[Vec<usize>],
    gold_to_prediction: &mut [Option<usize>],
    visited_gold: &mut [bool],
) -> bool {
    for &g_idx in &candidates[p_idx] {
        if visited_gold[g_idx] {
            continue;
        }
        visited_gold[g_idx] = true;
        let matched_prediction = gold_to_prediction[g_idx];
        if matched_prediction.is_none()
            || augment_matching(
                matched_prediction.expect("checked is_none above"),
                candidates,
                gold_to_prediction,
                visited_gold,
            )
        {
            gold_to_prediction[g_idx] = Some(p_idx);
            return true;
        }
    }
    false
}
