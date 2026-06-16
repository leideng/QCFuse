python3 data/build_longbench_data.py \
  --input_dir data/raw_longbench \
  --output_dir data/final_data \
  --tokenizer_path /public/models/Qwen3-8B \
  --embedding_model /public/models/embedding/bge-m3 \
  --chunk_size 512 \
  --chunk_overlap 50 \
  --context_topk 20 \
  --max_samples 200
