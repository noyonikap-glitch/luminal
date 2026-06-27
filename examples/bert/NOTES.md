### Implementing BERT following the examples for models

Followed the loading, model and inference structure from llama and gemma examples.

## hf.rs

`download_hf_model` : fetches the model checkpoint and tokenizer from HuggingFace Hub into the local cache. bert-base-uncased is a single-shard model so the download is one file; the sharded fallback path exists for larger models.
`combine_safetensors_to_fp32`: converts the checkpoint to a single FP32 safetensors file. HuggingFace checkpoints can be stored in f16 or bf16 to save bandwidth; Luminal's runtime expects f32, so we normalize on load. The function is idempotent — if the combined file already exists it returns early, so repeated runs don't re-convert.
Memory mapping (MmapOptions) is used instead of fs::read so the OS handles paging the file in as needed rather than loading 420MB into heap memory at once.
`prepare_hf_model` is the public entry point that ties both steps together. main.rs will call this and get back model_dir (for tokenizer.json) and weight_file (for runtime.load_safetensors()).

## [model.rs](http://model.rs)



struct BertConfig: takes in hyperparameters and initializes 

