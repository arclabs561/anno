//! Rank reviewed journeys by historical confirmed-failure rate using Muxer.
use muxer::{Extract, MabConfig, Objective, Summary};
use serde::Deserialize;
use std::{collections::BTreeMap, error::Error, io};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Arm {
    name: String,
    measured: u64,
    failures: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arms: Vec<Arm> = serde_json::from_reader(io::stdin().lock())?;
    let mut order = Vec::new();
    let mut summaries = BTreeMap::new();
    for arm in arms {
        if arm.failures > arm.measured || summaries.contains_key(&arm.name) {
            return Err("invalid journey observations".into());
        }
        order.push(arm.name.clone());
        summaries.insert(
            arm.name,
            Summary {
                calls: arm.measured,
                // A confirmed contract violation is a hit for this discovery objective.
                ok: arm.failures,
                ..Default::default()
            },
        );
    }
    if order.is_empty() {
        return Err("empty journey panel".into());
    }
    let config = MabConfig {
        objectives: vec![Objective::maximize(Extract::OkRateUcb, 1.0)],
        ..Default::default()
    };
    println!("{}", muxer::select_mab(&order, &summaries, config).chosen);
    Ok(())
}
