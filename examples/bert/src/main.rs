mod hf;
mod model;

fn main() {
    //download model, verbose for testing
    let model_dir = hf::download_hf_model("google-bert/bert-base-uncased")
        .expect("download failed");
    println!("model_dir: {}", model_dir.display());
    println!("tokenizer: {}", model_dir.join("tokenizer.json").display());
    println!("weights: {}", model_dir.join("model.safetensors").display());
}