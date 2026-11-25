//! Extract a Maverick MES model's tensor dictionary without executing Python checkpoint metadata.
use candle_core::{pickle::PthTensors, safetensors, Result};
use std::{collections::HashMap, env};

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [source, output] = args.as_slice() else {
        candle_core::bail!("usage: maverick_export <weights.ckpt> <model.safetensors>");
    };
    let archive = PthTensors::new(source, Some("state_dict"))?;
    let mut tensors = HashMap::new();
    for name in archive.tensor_infos().keys() {
        let Some(parameter) = name.strip_prefix("model.") else {
            candle_core::bail!("unexpected non-model state_dict parameter {name}");
        };
        let tensor = archive.get(name)?.expect("indexed tensor");
        tensors.insert(parameter, tensor.contiguous()?);
    }
    if tensors.is_empty() {
        candle_core::bail!("missing Maverick state_dict tensors");
    }
    safetensors::save(&tensors, output)?;
    println!("Wrote {} Maverick tensors.", tensors.len());
    Ok(())
}
