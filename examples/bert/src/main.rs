#[cfg(feature = "cuda")]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod hf;
mod model;

use hf::{prepare_hf_model, verify_weight_coverage};
use luminal::dtype::DType;
use luminal::prelude::*;
use model::{Bert, BertConfig};
use tokenizers::Tokenizer;

#[cfg(feature = "cuda")]
use luminal_cuda_lite::{
    cudarc::driver::CudaContext,
    runtime::CudaRuntime,
};
#[cfg(feature = "cuda")]
use luminal_tracing::*;
#[cfg(feature = "metal")]
use luminal_metal::MetalRuntime;
#[cfg(feature = "cuda")]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const REPO_ID: &str = "google-bert/bert-base-uncased";
const PROMPT: &str = "The cat sat on the mat.";
const SEARCH_MEMORY_MIB: usize = 2048;
const DEFAULT_SEARCH_GRAPHS: usize = 500;

fn search_graph_limit() -> usize {
    std::env::var("BERT_SEARCH_GRAPHS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_SEARCH_GRAPHS)
}

fn print_debug_hints() {
    println!("Debug hints (set before `cargo run`):");
    println!("  RUST_BACKTRACE=1              — full panic backtraces");
    println!("  LUMINAL_SEARCH_DUMP_LAST_LLIR=1 — on CUDA, writes the last profiled");
    println!("                               candidate to /tmp/luminal_search_last_candidate_llir.txt");
    println!("  LLIR_DUMP_DIR=/tmp/llir      — dump selected LLIR after successful search");
    println!("  BERT_SEARCH_GRAPHS=10        — lower search limit for faster failure loops");
    println!("Search prints the first few `initial-genome filter reject` lines to stderr.");
}

fn main() {
    #[cfg(feature = "cuda")]
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(luminal_filter())
        .init();

    let config = BertConfig::default();
    let search_graphs = search_graph_limit();

    println!("Preparing model...");
    let prepared = prepare_hf_model(REPO_ID).expect("failed to prepare model");
    println!("Weights: {}", prepared.weight_file.display());

    let tokenizer =
        Tokenizer::from_file(prepared.model_dir.join("tokenizer.json")).unwrap();
    let encoding = tokenizer.encode(PROMPT, true).unwrap();
    let seq_len = encoding.get_ids().len();
    let input_ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
    let position_ids: Vec<i32> = (0..seq_len as i32).collect();
    let token_type_ids = vec![0i32; seq_len];

    println!("Tokens: {:?}", encoding.get_tokens());
    println!("seq_len: {seq_len}");

    let mut cx = Graph::new();
    let input_ids_t = cx.named_tensor("input_ids", seq_len).as_dtype(DType::Int);
    let position_ids_t = cx.named_tensor("position_ids", seq_len).as_dtype(DType::Int);
    let token_type_ids_t = cx.named_tensor("token_type_ids", seq_len).as_dtype(DType::Int);

    let bert = Bert::init(&mut cx, config);
    let (sequence_out, pooled_out) =
        bert.forward(input_ids_t, position_ids_t, token_type_ids_t, config);
    let sequence_out = sequence_out.output();
    let pooled_out = pooled_out.output();

    verify_weight_coverage(&cx, &prepared.weight_file).expect("weight name mismatch");

    print_debug_hints();

    println!("Building E-Graph...");
    #[cfg(feature = "metal")]
    cx.build_search_space::<luminal_metal::MetalRuntime>(CompileOptions::default());
    #[cfg(feature = "cuda")]
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    println!("Loading weights...");
    #[cfg(feature = "metal")]
    let mut rt = MetalRuntime::initialize(());
    #[cfg(feature = "cuda")]
    let ctx = CudaContext::new(0).unwrap();
    #[cfg(feature = "cuda")]
    let stream = ctx.default_stream();
    #[cfg(feature = "cuda")]
    let mut rt = CudaRuntime::initialize(stream).with_max_memory_mib(SEARCH_MEMORY_MIB);

    rt.load_safetensors(&cx, prepared.weight_file.to_str().unwrap());

    // Dummy inputs for search profiling (same shapes as the real run).
    rt.set_data(input_ids_t, vec![1i32; seq_len]);
    rt.set_data(position_ids_t, position_ids.clone());
    rt.set_data(token_type_ids_t, vec![0i32; seq_len]);

    println!("Compiling (search_graph_limit={search_graphs})...");
    rt = cx.search(
        rt,
        CompileOptions::default()
            .search_graph_limit(search_graphs)
            .search_log(true),
    );

    #[cfg(feature = "cuda")]
    rt.release_pooled_memory();

    rt.set_data(input_ids_t, input_ids);
    rt.set_data(position_ids_t, position_ids);
    rt.set_data(token_type_ids_t, token_type_ids);

    rt.execute(&cx.dyn_map);

    let pooled = rt.get_f32(pooled_out);
    let sequence = rt.get_f32(sequence_out);

    println!(
        "[CLS] pooled embedding (first 8 values): {:?}",
        &pooled[..8.min(pooled.len())]
    );
    println!(
        "sequence row 0 (first 8 values): {:?}",
        &sequence[..8.min(sequence.len())]
    );
}
