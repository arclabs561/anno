//! Example demonstrating the expanded bias evaluation datasets.
//!
//! Run: cargo run --example bias_dataset_sizes --features eval-bias

use anno::eval::demographic_bias::create_diverse_name_dataset;
use anno::eval::gender_bias::create_winobias_templates;
use anno::eval::length_bias::create_length_varied_dataset;
use anno::eval::temporal_bias::create_temporal_name_dataset;
use std::collections::HashMap;

fn main() {
    println!("=== Bias Evaluation Dataset Sizes ===\n");

    // Gender Bias
    let gender_templates = create_winobias_templates();
    println!("Gender Bias (WinoBias-style):");
    println!("  Total examples: {}", gender_templates.len());

    let pro_count = gender_templates
        .iter()
        .filter(|e| {
            matches!(
                e.stereotype_type,
                anno::eval::gender_bias::StereotypeType::ProStereotypical
            )
        })
        .count();
    let anti_count = gender_templates
        .iter()
        .filter(|e| {
            matches!(
                e.stereotype_type,
                anno::eval::gender_bias::StereotypeType::AntiStereotypical
            )
        })
        .count();
    let neutral_count = gender_templates
        .iter()
        .filter(|e| {
            matches!(
                e.stereotype_type,
                anno::eval::gender_bias::StereotypeType::Neutral
            )
        })
        .count();

    println!("  Pro-stereotypical: {}", pro_count);
    println!("  Anti-stereotypical: {}", anti_count);
    println!("  Neutral: {}", neutral_count);

    let unique_occupations: std::collections::HashSet<_> =
        gender_templates.iter().map(|e| &e.occupation).collect();
    println!("  Unique occupations: {}\n", unique_occupations.len());

    // Demographic Bias
    let demographic_names = create_diverse_name_dataset();
    println!("Demographic Bias (Names):");
    println!("  Total names: {}", demographic_names.len());

    let mut by_ethnicity: HashMap<String, usize> = HashMap::new();
    let mut by_script: HashMap<String, usize> = HashMap::new();
    let mut by_gender: HashMap<String, usize> = HashMap::new();

    for name in &demographic_names {
        *by_ethnicity
            .entry(format!("{:?}", name.ethnicity))
            .or_insert(0) += 1;
        *by_script.entry(format!("{:?}", name.script)).or_insert(0) += 1;
        if let Some(gender) = name.gender {
            *by_gender.entry(format!("{:?}", gender)).or_insert(0) += 1;
        }
    }

    println!("  By ethnicity:");
    let mut eth_vec: Vec<_> = by_ethnicity.iter().collect();
    eth_vec.sort_by_key(|(_, &count)| std::cmp::Reverse(count));
    for (eth, count) in eth_vec {
        println!("    {}: {}", eth, count);
    }

    println!("  By script:");
    let mut script_vec: Vec<_> = by_script.iter().collect();
    script_vec.sort_by_key(|(_, &count)| std::cmp::Reverse(count));
    for (script, count) in script_vec {
        println!("    {}: {}", script, count);
    }

    println!("  By gender:");
    for (gender, count) in &by_gender {
        println!("    {}: {}", gender, count);
    }
    println!();

    // Temporal Bias
    let temporal_names = create_temporal_name_dataset();
    println!("Temporal Bias (Names by Decade):");
    println!("  Total names: {}", temporal_names.len());

    let mut by_decade: HashMap<String, usize> = HashMap::new();
    for name in &temporal_names {
        *by_decade
            .entry(format!("{:?}", name.peak_decade))
            .or_insert(0) += 1;
    }

    let mut decade_vec: Vec<_> = by_decade.iter().collect();
    decade_vec.sort();
    for (decade, count) in decade_vec {
        println!("    {}: {}", decade, count);
    }
    println!();

    // Length Bias
    let length_examples = create_length_varied_dataset();
    println!("Length Bias (Entity Length Examples):");
    println!("  Total examples: {}", length_examples.len());

    let mut by_type: HashMap<String, usize> = HashMap::new();
    let mut by_bucket: HashMap<String, usize> = HashMap::new();

    for example in &length_examples {
        *by_type
            .entry(format!("{:?}", example.entity_type))
            .or_insert(0) += 1;
        *by_bucket
            .entry(format!("{:?}", example.char_bucket))
            .or_insert(0) += 1;
    }

    println!("  By entity type:");
    for (typ, count) in &by_type {
        println!("    {}: {}", typ, count);
    }

    println!("  By length bucket:");
    for (bucket, count) in &by_bucket {
        println!("    {}: {}", bucket, count);
    }

    println!("\n=== Summary ===");
    println!("All datasets have been significantly expanded:");
    println!(
        "  - Gender Bias: 30 → {} examples ({}x)",
        gender_templates.len(),
        gender_templates.len() / 30
    );
    println!(
        "  - Demographic: ~100 → {} names ({}x)",
        demographic_names.len(),
        demographic_names.len() / 100
    );
    println!(
        "  - Temporal: ~70 → {} names ({}x)",
        temporal_names.len(),
        temporal_names.len() / 70
    );
    println!(
        "  - Length: ~30 → {} examples ({}x)",
        length_examples.len(),
        length_examples.len() / 30
    );
}
