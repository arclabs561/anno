//! Coreference resolution metrics benchmark.
//!
//! Demonstrates all coreference evaluation metrics:
//! - MUC (link-based)
//! - B³ (mention-based)
//! - CEAF (entity-based)
//! - LEA (link-aware entity-based)
//! - BLANC (rand index)
//! - CoNLL F1 (average of MUC, B³, CEAFe)
//!
//! Run with: `cargo run --example coref`

use anno::eval::coref::{CorefChain, CorefDocument, Mention, MentionType};
use anno::eval::coref_loader::{adversarial_coref_examples, synthetic_coref_dataset};
use anno::eval::coref_metrics::{AggregateCorefEvaluation, CorefEvaluation, SignificanceTest};

fn main() {
    println!("=== Coreference Resolution Metrics Benchmark ===\n");

    // Example 1: Perfect match
    println!("--- Example 1: Perfect Match ---");
    let gold = vec![
        CorefChain::new(vec![
            Mention::new("John Smith", 0, 10),
            Mention::new("he", 20, 22),
            Mention::new("the CEO", 40, 47),
        ]),
        CorefChain::new(vec![
            Mention::new("Mary Johnson", 50, 62),
            Mention::new("she", 70, 73),
        ]),
    ];
    let pred = gold.clone();
    print_evaluation("Perfect match", &pred, &gold);

    // Example 2: Under-clustering (too many clusters)
    println!("\n--- Example 2: Under-clustering ---");
    println!("Gold: [[John, he, the CEO], [Mary, she]]");
    println!("Pred: [[John], [he], [the CEO], [Mary, she]]");
    let pred_under = vec![
        CorefChain::new(vec![Mention::new("John Smith", 0, 10)]),
        CorefChain::new(vec![Mention::new("he", 20, 22)]),
        CorefChain::new(vec![Mention::new("the CEO", 40, 47)]),
        CorefChain::new(vec![
            Mention::new("Mary Johnson", 50, 62),
            Mention::new("she", 70, 73),
        ]),
    ];
    print_evaluation("Under-clustering", &pred_under, &gold);

    // Example 3: Over-clustering (too few clusters)
    println!("\n--- Example 3: Over-clustering ---");
    println!("Gold: [[John, he, the CEO], [Mary, she]]");
    println!("Pred: [[John, he, the CEO, Mary, she]]");
    let pred_over = vec![CorefChain::new(vec![
        Mention::new("John Smith", 0, 10),
        Mention::new("he", 20, 22),
        Mention::new("the CEO", 40, 47),
        Mention::new("Mary Johnson", 50, 62),
        Mention::new("she", 70, 73),
    ])];
    print_evaluation("Over-clustering", &pred_over, &gold);

    // Example 4: Partial overlap
    println!("\n--- Example 4: Partial Overlap ---");
    println!("Gold: [[John, he, the CEO], [Mary, she]]");
    println!("Pred: [[John, he], [the CEO, Mary], [she]]");
    let pred_partial = vec![
        CorefChain::new(vec![
            Mention::new("John Smith", 0, 10),
            Mention::new("he", 20, 22),
        ]),
        CorefChain::new(vec![
            Mention::new("the CEO", 40, 47),
            Mention::new("Mary Johnson", 50, 62),
        ]),
        CorefChain::new(vec![Mention::new("she", 70, 73)]),
    ];
    print_evaluation("Partial overlap", &pred_partial, &gold);

    // Example 5: Singletons only
    println!("\n--- Example 5: All Singletons (Gold and Pred) ---");
    let gold_singletons = vec![
        CorefChain::new(vec![Mention::new("entity1", 0, 7)]),
        CorefChain::new(vec![Mention::new("entity2", 10, 17)]),
        CorefChain::new(vec![Mention::new("entity3", 20, 27)]),
    ];
    let pred_singletons = gold_singletons.clone();
    print_evaluation("All singletons", &pred_singletons, &gold_singletons);

    // Example 6: Complex realistic example
    println!("\n--- Example 6: Realistic Document ---");
    let doc_text = "Barack Obama was born in Hawaii. He served as the 44th President. \
                    Obama left office in 2017. Michelle Obama is his wife.";
    println!("Text: {}\n", doc_text);

    let gold_realistic = vec![
        CorefChain::new(vec![
            Mention::with_type("Barack Obama", 0, 12, MentionType::Proper),
            Mention::with_type("He", 33, 35, MentionType::Pronoun),
            Mention::with_type("Obama", 67, 72, MentionType::Proper),
            Mention::with_type("his", 112, 115, MentionType::Pronoun),
        ]),
        CorefChain::new(vec![
            Mention::with_type("Michelle Obama", 97, 111, MentionType::Proper),
        ]),
    ];

    // System 1: Correctly links all Obama mentions but misses 'his'
    let pred_realistic_1 = vec![
        CorefChain::new(vec![
            Mention::new("Barack Obama", 0, 12),
            Mention::new("He", 33, 35),
            Mention::new("Obama", 67, 72),
        ]),
        CorefChain::new(vec![Mention::new("Michelle Obama", 97, 111)]),
        CorefChain::new(vec![Mention::new("his", 112, 115)]), // Missed link
    ];

    // System 2: Links 'his' to Michelle instead of Barack
    let pred_realistic_2 = vec![
        CorefChain::new(vec![
            Mention::new("Barack Obama", 0, 12),
            Mention::new("He", 33, 35),
            Mention::new("Obama", 67, 72),
        ]),
        CorefChain::new(vec![
            Mention::new("Michelle Obama", 97, 111),
            Mention::new("his", 112, 115), // Wrong link
        ]),
    ];

    println!("Gold: [[Barack Obama, He, Obama, his], [Michelle Obama]]");
    println!("System 1: [[Barack Obama, He, Obama], [Michelle Obama], [his]] (missed link)");
    println!("System 2: [[Barack Obama, He, Obama], [Michelle Obama, his]] (wrong link)\n");

    println!("System 1 (missed link to 'his'):");
    print_evaluation("System 1", &pred_realistic_1, &gold_realistic);

    println!("\nSystem 2 (wrong link 'his' -> Michelle):");
    print_evaluation("System 2", &pred_realistic_2, &gold_realistic);

    // Compare metrics behavior
    println!("\n=== Metric Comparison Summary ===\n");
    println!("Key observations:");
    println!("1. MUC ignores singletons - good for comparing systems but inflates scores");
    println!("2. B³ gives credit for partial overlap - can be inflated by singletons");
    println!("3. CEAF aligns clusters optimally - entity-focused view");
    println!("4. LEA combines link accuracy with entity importance - balanced");
    println!("5. BLANC rewards correct non-coreference decisions - best discriminative power");
    println!("6. CoNLL F1 = avg(MUC, B³, CEAFe) - official shared task metric");

    // Using CorefDocument
    println!("\n=== Using CorefDocument ===\n");
    let doc = CorefDocument::new(
        "John went to the store. He bought milk for his family.",
        vec![CorefChain::new(vec![
            Mention::new("John", 0, 4),
            Mention::new("He", 24, 26),
            Mention::new("his", 43, 46),
        ])],
    );
    println!("Document: \"{}\"", doc.text);
    println!("Mentions: {}", doc.mention_count());
    println!("Chains: {}", doc.chain_count());
    println!("Non-singleton chains: {}", doc.non_singleton_count());
    
    // Show mention index
    let index = doc.mention_to_chain_index();
    println!("Mention -> Chain mapping: {:?}", index);

    // === Adversarial Examples ===
    println!("\n=== Adversarial Examples ===\n");
    println!("Testing edge cases that stress-test coreference metrics:\n");

    for (gold, pred, scenario) in adversarial_coref_examples() {
        println!("Scenario: {}", scenario);
        println!("  Gold chains: {}", gold.chain_count());
        println!("  Pred chains: {}", pred.chain_count());
        let eval = CorefEvaluation::compute(&pred.chains, &gold.chains);
        println!(
            "  Scores: MUC={:.1}%, B³={:.1}%, CEAFe={:.1}%, BLANC={:.1}%, CoNLL={:.1}%",
            eval.muc.f1 * 100.0,
            eval.b_cubed.f1 * 100.0,
            eval.ceaf_e.f1 * 100.0,
            eval.blanc.f1 * 100.0,
            eval.conll_f1 * 100.0
        );
        println!();
    }

    // === Synthetic Dataset Statistics ===
    println!("=== Synthetic Dataset ===\n");
    let synthetic = synthetic_coref_dataset(10);
    let total_mentions: usize = synthetic.iter().map(|d| d.mention_count()).sum();
    let total_chains: usize = synthetic.iter().map(|d| d.chain_count()).sum();
    let non_singletons: usize = synthetic.iter().map(|d| d.non_singleton_count()).sum();

    println!("Generated {} documents:", synthetic.len());
    println!("  Total mentions: {}", total_mentions);
    println!("  Total chains: {}", total_chains);
    println!("  Non-singleton chains: {}", non_singletons);
    println!("  Avg mentions/doc: {:.1}", total_mentions as f64 / synthetic.len() as f64);
    println!("  Avg chains/doc: {:.1}", total_chains as f64 / synthetic.len() as f64);

    // Perfect match on synthetic
    println!("\nPerfect match on synthetic data:");
    let mut total_conll = 0.0;
    for doc in &synthetic {
        let eval = CorefEvaluation::compute(&doc.chains, &doc.chains);
        total_conll += eval.conll_f1;
    }
    println!("  Avg CoNLL F1: {:.1}%", (total_conll / synthetic.len() as f64) * 100.0);

    // === Aggregate Evaluation Demo ===
    println!("\n=== Aggregate Evaluation with Confidence Intervals ===\n");
    
    // Create pairs of (predicted, gold) for aggregate evaluation
    let doc_pairs: Vec<(&[CorefChain], &[CorefChain])> = synthetic
        .iter()
        .map(|doc| (doc.chains.as_slice(), doc.chains.as_slice()))
        .collect();
    
    let aggregate = AggregateCorefEvaluation::compute(&doc_pairs);
    println!("{}", aggregate);
    
    // Analysis methods
    let eval = CorefEvaluation::compute(&pred_realistic_1, &gold_realistic);
    println!("System 1 analysis:");
    println!("  Average F1 across all metrics: {:.1}%", eval.average_f1() * 100.0);
    println!("  F1 std dev across metrics: {:.1}%", eval.f1_std_dev() * 100.0);
    println!("  Over-clustering: {}", eval.is_over_clustering());
    println!("  Under-clustering: {}", eval.is_under_clustering());
    println!("  Summary: {}", eval.summary_line());

    // === Statistical Significance Testing ===
    println!("\n=== Statistical Significance Testing ===\n");
    
    // Simulate two systems evaluated on multiple documents
    // System A: slightly better coreference
    // System B: baseline
    let system_a_scores = vec![0.85, 0.82, 0.88, 0.79, 0.84, 0.86, 0.81, 0.87, 0.83, 0.85];
    let system_b_scores = vec![0.78, 0.76, 0.82, 0.74, 0.79, 0.80, 0.75, 0.81, 0.77, 0.79];
    
    let test = SignificanceTest::paired_t_test(&system_a_scores, &system_b_scores);
    println!("Comparing System A vs System B:");
    println!("{}", test);
    
    // Example with no significant difference
    let system_c_scores = vec![0.80, 0.82, 0.79, 0.81, 0.80, 0.83, 0.78, 0.82, 0.81, 0.80];
    let system_d_scores = vec![0.81, 0.80, 0.82, 0.79, 0.81, 0.80, 0.82, 0.79, 0.80, 0.81];
    
    let test2 = SignificanceTest::paired_t_test(&system_c_scores, &system_d_scores);
    println!("Comparing System C vs System D (similar performance):");
    println!("{}", test2);
}

fn print_evaluation(name: &str, pred: &[CorefChain], gold: &[CorefChain]) {
    let eval = CorefEvaluation::compute(pred, gold);

    println!("\n{}", name);
    println!("{:-<60}", "");
    println!(
        "MUC:     P={:5.1}%  R={:5.1}%  F1={:5.1}%",
        eval.muc.precision * 100.0,
        eval.muc.recall * 100.0,
        eval.muc.f1 * 100.0
    );
    println!(
        "B³:      P={:5.1}%  R={:5.1}%  F1={:5.1}%",
        eval.b_cubed.precision * 100.0,
        eval.b_cubed.recall * 100.0,
        eval.b_cubed.f1 * 100.0
    );
    println!(
        "CEAFe:   P={:5.1}%  R={:5.1}%  F1={:5.1}%",
        eval.ceaf_e.precision * 100.0,
        eval.ceaf_e.recall * 100.0,
        eval.ceaf_e.f1 * 100.0
    );
    println!(
        "CEAFm:   P={:5.1}%  R={:5.1}%  F1={:5.1}%",
        eval.ceaf_m.precision * 100.0,
        eval.ceaf_m.recall * 100.0,
        eval.ceaf_m.f1 * 100.0
    );
    println!(
        "LEA:     P={:5.1}%  R={:5.1}%  F1={:5.1}%",
        eval.lea.precision * 100.0,
        eval.lea.recall * 100.0,
        eval.lea.f1 * 100.0
    );
    println!(
        "BLANC:   P={:5.1}%  R={:5.1}%  F1={:5.1}%",
        eval.blanc.precision * 100.0,
        eval.blanc.recall * 100.0,
        eval.blanc.f1 * 100.0
    );
    println!("CoNLL:   F1={:5.1}%", eval.conll_f1 * 100.0);
}

