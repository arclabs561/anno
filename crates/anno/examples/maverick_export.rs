//! Extract the parameter tensors of a verified Maverick MES checkpoint.
use candle_core::pickle::PthTensors;
use safetensors::tensor::{Metadata, TensorInfo, View};
use std::{
    collections::HashMap,
    env, fs,
    io::{BufWriter, Write},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [source, output] = args.as_slice() else {
        return Err("usage: maverick_export <weights.ckpt> <model.safetensors>".into());
    };
    let archive = PthTensors::new(source, Some("state_dict"))?;
    let mut names: Vec<_> = archive.tensor_infos().keys().collect();
    if names.is_empty() {
        return Err("missing Maverick state_dict tensors".into());
    }
    names.sort();

    // Build the validated safetensors header from the index, then stream one
    // tensor at a time. The source archive contains about 2 GB of parameters.
    let mut offset = 0usize;
    let mut header = Vec::with_capacity(names.len());
    for &name in &names {
        let parameter = name
            .strip_prefix("model.")
            .ok_or("unexpected non-model parameter")?;
        let entry = &archive.tensor_infos()[name];
        let shape = entry.layout.dims().to_vec();
        let bytes = shape
            .iter()
            .try_fold(entry.dtype.size_in_bytes(), |n, &d| n.checked_mul(d))
            .ok_or("tensor size overflow")?;
        let end = offset.checked_add(bytes).ok_or("model size overflow")?;
        header.push((
            parameter.to_owned(),
            TensorInfo {
                shape,
                dtype: entry.dtype.into(),
                data_offsets: (offset, end),
            },
        ));
        offset = end;
    }
    let family = HashMap::from([(
        "anno_architecture".into(),
        "maverick-mes-deberta-v3-large-v1".into(),
    )]);
    let mut header = serde_json::to_vec(&Metadata::new(Some(family), header)?)?;
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut out = BufWriter::new(fs::File::create(output)?);
    out.write_all(&(header.len() as u64).to_le_bytes())?;
    out.write_all(&header)?;
    for name in &names {
        let tensor = archive
            .get(name)?
            .ok_or("missing indexed model parameter")?;
        out.write_all(&tensor.contiguous()?.data())?;
    }
    out.flush()?;
    eprintln!("Wrote {} Maverick tensors.", names.len());
    Ok(())
}
