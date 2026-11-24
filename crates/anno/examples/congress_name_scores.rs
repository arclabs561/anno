// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compare committee pilot pairing scores with anno's native name scorer.
use std::error::Error;
use std::io::{self, BufRead};

fn main() -> Result<(), Box<dyn Error>> {
    let mut count = 0;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let mut fields = line.split('\t');
        let printed = fields.next().ok_or("missing printed name")?;
        let catalog = fields.next().ok_or("missing catalog name")?;
        let expected: f32 = fields.next().ok_or("missing expected score")?.parse()?;
        if fields.next().is_some() {
            return Err("unexpected fields in pair".into());
        }
        let native = anno::coalesce::resolver::string_similarity(printed, catalog);
        if native != expected {
            return Err(format!(
                "{printed:?} vs {catalog:?}: native {native}, expected {expected}"
            )
            .into());
        }
        count += 1;
    }
    if count == 0 {
        return Err("no pairs were supplied".into());
    }
    println!("Native Jaccard agrees on all {count} scored name pairs.");
    Ok(())
}
