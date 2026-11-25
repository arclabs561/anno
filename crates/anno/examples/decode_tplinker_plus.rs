//! Read a sparse TPLinkerPlus prediction JSON from stdin and emit upstream-format entities and edges.
use anno::backends::tplinker_plus::PlusPrediction;
use std::io;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prediction: PlusPrediction = serde_json::from_reader(io::stdin().lock())?;
    let decoded = prediction.decode()?;
    serde_json::to_writer(io::stdout().lock(), &decoded)?;
    Ok(())
}
