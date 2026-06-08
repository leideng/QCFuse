# Dataset Preprocessing

This directory builds the six JSONL files used by QCFuse:

- LongBench: `musique.jsonl`, `2wikimqa.jsonl`, `hotpotqa.jsonl`
- RULER: `ruler_mv.jsonl`, `ruler_mq.jsonl`, `ruler_vt.jsonl`

## Environment

```bash
pip install transformers langchain-text-splitters sentence-transformers numpy
```

For RULER preprocessing, follow the official
[NVIDIA/RULER](https://github.com/NVIDIA/RULER) installation instructions.

## LongBench

Put the official LongBench files here:

```text
data/raw_longbench/
  musique.jsonl
  2wikimqa.jsonl
  hotpotqa.jsonl
```

Build the QCFuse-format files:

```bash
python3 data/build_longbench_data.py \
  --input_dir data/raw_longbench \
  --output_dir data/final_data \
  --tokenizer_path models/qwen3-8b \
  --embedding_model models/bge-m3 \
  --chunk_size 512 \
  --chunk_overlap 50 \
  --context_topk 20 \
  --max_samples 200
```

## RULER

Clone the official RULER repository:

```bash
git clone https://github.com/NVIDIA/RULER.git third_party/RULER
```

Build the full files:

```bash
python3 data/build_ruler_data.py \
  --ruler_dir third_party/RULER \
  --raw_dir data/ruler_raw \
  --output_dir data/final_data \
  --tokenizer_path models/qwen3-8b \
  --num_samples 200 \
  --chunk_size 512 \
  --target_num_chunks 20 \
  --ruler_max_seq_length 11264
```

RULER outputs are trimmed to 20 chunks per sample. The script also writes
metadata files under `data/final_data`.

Use `data/final_data` as `--data_dir` in the QCFuse runner.
