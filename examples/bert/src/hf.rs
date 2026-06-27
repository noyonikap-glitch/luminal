use half::{bf16, f16};
use hf_hub::api::sync::Api;
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};


/// Index file structure for sharded safetensors models
#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Stored tensor data.
struct StoredTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// Downloads model files from HuggingFace and returns the cache directory path.
//bert-base-uncased stored as a single shard model, sharded model download code commented out for now
pub fn download_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    // Download tokenizer
    let tokenizer_path = repo.get("tokenizer.json")?;
    let model_dir = tokenizer_path.parent().unwrap().to_path_buf();
    // Try to download single shard model first
    /*
    if repo.get("model.safetensors").is_ok() {
        return Ok(model_dir);
    }*/
    match repo.get("model.safetensors") {
        Ok(_) => return Ok(model_dir),
        Err(e) => eprintln!("model.safetensors failed: {e}"),
    }
    let index_path = repo.get("model.safetensors.index.json")?;
    // Parse index to find shard files
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;
    let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();
    for shard_file in &shard_files {
        repo.get(shard_file)?;
    }
    Ok(model_dir)
}

/// Convert tensor data to f32 vec
fn tensor_to_f32(tensor: &safetensors::tensor::TensorView) -> Vec<f32> {
    let dtype = tensor.dtype();
    let data = tensor.data();

    match dtype {
        Dtype::F32 => bytemuck::cast_slice::<u8, f32>(data).to_vec(),
        Dtype::F16 => {
            let f16_slice: &[f16] = bytemuck::cast_slice(data);
            f16_slice.iter().map(|x| x.to_f32()).collect()
        }
        Dtype::BF16 => {
            let bf16_slice: &[bf16] = bytemuck::cast_slice(data);
            bf16_slice.iter().map(|x| x.to_f32()).collect()
        }
        other => {
            panic!("Unsupported dtype for conversion: {other:?}");
        }
    }
}

// Referred to Gemma example for this function.
// no embedding scale, no +1.0, no bf16 split for bert-base-uncased
pub fn combine_safetensors_to_fp32(
    model_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output = model_dir.join("model_combined_fp32.safetensors");
    // Skip if we already converted on a previous run
    if output.exists() {
        return Ok(output);
    }

    // Open the raw checkpoint file and memory-map it for efficient access
    // (avoids loading the whole file into heap memory at once)
    let shard_path = model_dir.join("model.safetensors");
    let file = File::open(&shard_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    // Parse the safetensors format — this reads the header but not the tensor data yet
    let st = SafeTensors::deserialize(&mmap)?;

    // Convert every tensor to f32 and store it with its name and shape
    // HuggingFace checkpoints can be f16/bf16/f32; we normalize everything to f32
    // so Luminal's runtime can load them without dtype handling
    let mut all_tensors: HashMap<String, StoredTensor> = HashMap::new();
    for name in st.names() {
        let tensor = st.tensor(name)?;
        all_tensors.insert(name.to_string(), StoredTensor {
            shape: tensor.shape().to_vec(),
            data: tensor_to_f32(&tensor),
        });
    }

    // Build TensorView objects — safetensors serialize() needs views over raw bytes,
    // not Vec<f32>, so we cast back to &[u8] here (no copy, just a reinterpretation)
    let tensor_views: HashMap<String, TensorView<'_>> = all_tensors
        .iter()
        .map(|(name, stored)| {
            let bytes: &[u8] = bytemuck::cast_slice(&stored.data);
            let view = TensorView::new(Dtype::F32, stored.shape.clone(), bytes).unwrap();
            (name.clone(), view)
        })
        .collect();

    // Serialize all tensors into the safetensors binary format and write to disk
    // This combined file is what Luminal's runtime.load_safetensors() will read
    let serialized = safetensors::serialize(&tensor_views, None)?;
    let mut file = File::create(&output)?;
    file.write_all(&serialized)?;

    Ok(output)
}

pub struct PreparedModel {
    pub model_dir: PathBuf,
    pub weight_file: PathBuf,
}

pub fn prepare_hf_model(repo_id: &str) -> Result<PreparedModel, Box<dyn std::error::Error>> {
    let model_dir = download_hf_model(repo_id)?;
    let weight_file = combine_safetensors_to_fp32(&model_dir)?;
    Ok(PreparedModel { model_dir, weight_file })
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::{bf16, f16};

    fn f32_view(values: &[f32]) -> TensorView<'_> {
        let bytes: &[u8] = bytemuck::cast_slice(values);
        TensorView::new(Dtype::F32, vec![values.len()], bytes).unwrap()
    }
    #[test]
    #[ignore]  // run explicitly with: cargo test -- --ignored
    fn test_download_bert() {
    let model_dir = download_hf_model("bert-base-uncased").unwrap();
    assert!(model_dir.join("tokenizer.json").exists());
    assert!(model_dir.join("model.safetensors").exists());
    }

    #[test]
    fn tensor_to_f32_passthrough_f32() {
        let values = [1.0f32, 2.0, 3.5];
        assert_eq!(tensor_to_f32(&f32_view(&values)), values.to_vec());
    }

    #[test]
    fn tensor_to_f32_converts_f16() {
        let f16_values = [f16::from_f32(1.0), f16::from_f32(-2.5)];
        let bytes: &[u8] = bytemuck::cast_slice(&f16_values);
        let view = TensorView::new(Dtype::F16, vec![2], bytes).unwrap();
        let out = tensor_to_f32(&view);
        assert!((out[0] - 1.0).abs() < 1e-3);
        assert!((out[1] - (-2.5)).abs() < 1e-2);
    }

    #[test]
    fn tensor_to_f32_converts_bf16() {
        let bf16_values = [bf16::from_f32(4.0), bf16::from_f32(0.125)];
        let bytes: &[u8] = bytemuck::cast_slice(&bf16_values);
        let view = TensorView::new(Dtype::BF16, vec![2], bytes).unwrap();
        let out = tensor_to_f32(&view);
        assert!((out[0] - 4.0).abs() < 1e-2);
        assert!((out[1] - 0.125).abs() < 1e-2);
    }

    #[test]
    #[should_panic(expected = "Unsupported dtype for conversion")]
    fn tensor_to_f32_rejects_unsupported_dtype() {
        let _ = tensor_to_f32(&TensorView::new(Dtype::U8, vec![1], &[0u8]).unwrap());
    }
}
