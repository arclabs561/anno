//! Extract text from downloaded datasets and save as individual files for crossdoc testing
//!
//! Usage: cargo run --bin extract_dataset_texts --features eval-advanced

#[cfg(feature = "eval-advanced")]
use anno::eval::loader::{DatasetId, DatasetLoader};
use std::fs;
use std::path::Path;

fn main() {
    #[cfg(not(feature = "eval-advanced"))]
    {
        eprintln!("This requires --features eval-advanced");
        return;
    }

    #[cfg(feature = "eval-advanced")]
    {
        let loader = DatasetLoader::new().expect("Failed to create loader");
        let output_dir = Path::new("hack/real_data/datasets");
        fs::create_dir_all(output_dir).expect("Failed to create output dir");

        // Use a few diverse datasets
        let datasets = vec![
            DatasetId::WikiGold,
            DatasetId::CoNLL2003Sample,
            DatasetId::Wnut17,
        ];

        for dataset_id in datasets {
            println!("Processing {}...", dataset_id.name());
            
            match loader.load(dataset_id) {
                Ok(dataset) => {
                    let dataset_dir = output_dir.join(dataset_id.name().to_lowercase());
                    fs::create_dir_all(&dataset_dir).expect("Failed to create dataset dir");

                    for (idx, example) in dataset.examples.iter().take(20).enumerate() {
                        let filename = format!("{}_{:03}.txt", dataset_id.name().to_lowercase(), idx);
                        let filepath = dataset_dir.join(&filename);
                        fs::write(&filepath, &example.text).expect("Failed to write file");
                    }
                    println!("  Extracted {} examples to {:?}", 
                        dataset.examples.len().min(20), dataset_dir);
                }
                Err(e) => {
                    eprintln!("  Failed to load {}: {}", dataset_id.name(), e);
                }
            }
        }
    }
}

