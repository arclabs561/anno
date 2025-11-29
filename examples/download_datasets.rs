//! Download all NER datasets for offline use.
//!
//! This example pre-downloads evaluation datasets so they're cached
//! for offline use. Run once with network access, then use offline.
//!
//! # Usage
//!
//! ```bash
//! # Download all datasets
//! cargo run --example download_datasets --features eval-advanced
//!
//! # Download only NER datasets
//! cargo run --example download_datasets --features eval-advanced -- --ner-only
//! ```
//!
//! # Datasets Downloaded
//!
//! | Category | Count | Examples |
//! |----------|-------|----------|
//! | NER | 20+ | WikiGold, WNUT-17, CoNLL-2003, BC5CDR, etc. |
//! | Coreference | 3 | GAP, PreCo, LitBank |
//! | Relation Extraction | 2 | DocRED, Re-TACRED |
//!
//! # Cache Location
//!
//! Datasets are cached in: `~/.cache/anno/datasets/`
//!
//! # Lazy Loading
//!
//! If you don't run this example, datasets are downloaded automatically on first use.
//! This example is useful for:
//! - Pre-warming cache for offline deployment
//! - Checking which datasets are available
//! - Ensuring download works before production

use std::time::Instant;

#[cfg(feature = "eval-advanced")]
use anno::eval::loader::{DatasetId, DatasetLoader};

fn main() {
    println!("=== Anno Dataset Downloader ===\n");

    #[cfg(not(feature = "eval-advanced"))]
    {
        println!("This example requires --features eval-advanced");
        println!("Run with: cargo run --example download_datasets --features eval-advanced");
        return;
    }

    #[cfg(feature = "eval-advanced")]
    {
        let loader = match DatasetLoader::new() {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Failed to create DatasetLoader: {}", e);
                return;
            }
        };

        let start = Instant::now();
        let mut downloaded = 0;
        let mut failed = 0;
        let mut skipped = 0;

        // Check command line args for filtering
        let args: Vec<String> = std::env::args().collect();
        let ner_only = args.iter().any(|a| a == "--ner-only" || a == "-n");

        let datasets: &[DatasetId] = if ner_only {
            println!("--- NER Datasets Only ---\n");
            DatasetId::all_ner()
        } else {
            println!("--- All Datasets ---\n");
            DatasetId::all()
        };

        for dataset_id in datasets {
            print!("  {}... ", dataset_id.name());
            let ds_start = Instant::now();

            if loader.is_cached(*dataset_id) {
                println!("cached (skipped)");
                skipped += 1;
                continue;
            }

            match loader.load_or_download(*dataset_id) {
                Ok(dataset) => {
                    let elapsed = ds_start.elapsed();
                    let entity_count = dataset.entity_count();
                    println!(
                        "OK ({:.1}s, {} entities)",
                        elapsed.as_secs_f64(),
                        entity_count
                    );
                    downloaded += 1;
                }
                Err(e) => {
                    eprintln!("FAILED: {}", e);
                    failed += 1;
                }
            }
        }

        let elapsed = start.elapsed();
        println!("\n=== Summary ===");
        println!("Downloaded: {}", downloaded);
        println!("Cached (skipped): {}", skipped);
        println!("Failed: {}", failed);
        println!("Time: {:.1}s", elapsed.as_secs_f64());

        if downloaded > 0 {
            println!("\nDatasets cached in: {:?}", loader.cache_dir());
            println!("You can now use anno offline!");
        }

        if failed > 0 {
            println!("\nNote: Some datasets may require special access or have changed URLs.");
            println!("      Check the error messages above for details.");
        }
    }
}
