mod hf;
mod model;

use hf::prepare_hf_model;
use luminal::prelude::*;
use luminal::dtype::DType;
use model::{Bert, BertConfig};
use tokenizers::Tokenizer;

#[cfg(feature = "metal")]
use luminal_metal::MetalRuntime as Runtime;
#[cfg(feature = "cuda")]
use luminal_cuda_lite::runtime::CudaRuntime as Runtime;

const REPO_ID: &str = "bert-base-uncased";
const PROMPT: &str = "The cat sat on the mat.";

fn main() {
    let config = BertConfig::default();

    // 1. download weights + tokenizer
    println!("Preparing model...");
    let prepared = prepare_hf_model(REPO_ID).expect("failed to prepare model");

    // 2. tokenize
    let tokenizer = Tokenizer::from_file(
        prepared.model_dir.join("tokenizer.json")
    ).unwrap();
    let encoding = tokenizer.encode(PROMPT, true).unwrap();
    let seq_len = encoding.get_ids().len();
    let input_ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
    let position_ids: Vec<i32> = (0..seq_len as i32).collect();
    let token_type_ids: Vec<i32> = vec![0; seq_len];

    println!("Tokens: {:?}", encoding.get_tokens());

    // 3. build graph
    let mut cx = Graph::new();
    let input_ids_t = cx.named_tensor("input_ids", seq_len).as_dtype(DType::Int);
    let position_ids_t = cx.named_tensor("position_ids", seq_len).as_dtype(DType::Int);
    let token_type_ids_t = cx.named_tensor("token_type_ids", seq_len).as_dtype(DType::Int);

    let bert = Bert::init(&mut cx, config);
    let (sequence_out, pooled_out) = bert.forward(
        input_ids_t, position_ids_t, token_type_ids_t, config,
    );
    let sequence_out = sequence_out.output();
    let pooled_out = pooled_out.output();

    // 4. initialize runtime
    #[cfg(feature = "cuda")]
    let ctx = luminal_cuda_lite::cudarc::driver::CudaContext::new(0).unwrap();
    #[cfg(feature = "cuda")]
    let stream = ctx.default_stream();

    #[cfg(feature = "metal")]
    let mut rt = Runtime::initialize(());
    #[cfg(feature = "cuda")]
    let mut rt = Runtime::initialize(stream);

    // 5. build search space + load weights + compile
    cx.build_search_space::<Runtime>(CompileOptions::default());
    rt.load_safetensors(&cx, prepared.weight_file.to_str().unwrap());

    rt.set_data(input_ids_t, input_ids.clone());
    rt.set_data(position_ids_t, position_ids.clone());
    rt.set_data(token_type_ids_t, token_type_ids.clone());

    println!("Compiling...");
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(50));

    // 6. execute
    rt.execute(&cx.dyn_map);

    // 7. print results
    let pooled = rt.get_f32(pooled_out);
    let sequence = rt.get_f32(sequence_out);

    println!("[CLS] pooled embedding (first 8 values): {:?}", &pooled[..8]);
    println!("sequence[0] first 8 values: {:?}", &sequence[..8]);
}