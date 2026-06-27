# BERT

Work in progress.

Run through Luminal's CUDA backend:

```bash
cargo run --release -p bert --features cuda
```

Run through Luminal's Metal backend on Apple targets:

```bash
cargo run --release -p bert --features metal
```

Download-only smoke test (no backend feature):

```bash
cargo run -p bert
```
