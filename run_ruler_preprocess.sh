python3 data/build_ruler_data.py \
  --ruler_dir /data/ldeng/code/kvbridge/QCFuse/third_party/RULER \
  --raw_dir /data/ldeng/code/kvbridge/QCFuse/data/ruler_raw \
  --output_dir /data/ldeng/code/kvbridge/QCFuse/data/final_data \
  --tokenizer_path /public/models/Qwen3-8B \
  --num_samples 200 \
  --chunk_size 512 \
  --target_num_chunks 20 \
  --ruler_max_seq_length 11264
